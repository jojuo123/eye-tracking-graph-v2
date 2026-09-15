"""Encodes each fixation as an embedding built from a full chest X-ray image plus that
fixation's `(x, y)` location, for permutation-learning tasks where a model must recover a
scrambled fixation sequence's original order (e.g.
`dataloaders.permuted_h5_dataset.PermutedFixationH5Dataset`).

Strategy: a shared `ResNetEncoder` backbone produces a multi-scale feature pyramid from the
full image (once per sample, not once per fixation -- unlike `modules.encoders`, which encodes
one already-cropped patch per element). At each of a selected subset of pyramid levels
(`selected_levels`), that level's features are fused into a running per-fixation embedding,
one level at a time, in the order `selected_levels` lists them. Two fusion strategies are
available via `fusion_mode`:
  - `"cross_attention"` (default): the running embedding cross-attends to that level's spatial
    feature map as keys/values, so each fixation can learn to weigh different image regions
    (nearby ones, via the grid position encoding added to the keys -- see `_grid_coordinates`).
  - `"concat"`: the level's feature map is bilinearly sampled (`grid_sample`) at the fixation's
    own coordinate directly, no attention -- cheaper, and exactly what "look up the feature at
    my location" means when you don't need the model to *learn* where to look.

Permutation invariance: a fixation's output embedding depends only on the image and *its own*
coordinate. There is no self-attention or other operation across fixations anywhere in this
module (in either fusion mode), and the coordinate encoding (`modules.coordinate_encoding`) is
a function of the coordinate's value, not its index in the input sequence. So permuting the
input fixation list permutes the output list identically -- the encoder is
permutation-equivariant, and carries no information about the very order a downstream model is
trained to recover. (Contrast this with `modules.transformer.TransformerEncoder`, whose
`PositionalEncoding` is keyed by sequence index -- appropriate for tasks where element order is
meaningful input, but exactly the leak a permutation-recovery task's *encoder* must avoid. A
downstream self-attention stage over these per-fixation embeddings, if you add one, should skip
that positional encoding for the same reason -- `SinkhornPatchSorter` shares this caveat.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.coordinate_encoding import build_coordinate_encoding
from modules.cross_attention import CrossAttentionBlock
from modules.resnet import ResNetEncoder


def _grid_coordinates(height, width, device):
    """`(height * width, 2)` grid of `(x, y)` coordinates in `[0, 1]`, row-major to match
    `tensor.flatten(2)`'s ordering of a `(C, H, W)` feature map into `(H * W, C)` tokens."""
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, height, device=device),
        torch.linspace(0, 1, width, device=device),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (H * W, 2)


def _sample_at_coordinates(feature_map, coordinates):
    """Bilinearly samples `feature_map` (`(B, C, H, W)`) at each of `coordinates`
    (`(B, N, 2)`, `(x, y)` in `[0, 1]`), returning `(B, N, C)`. Each fixation is sampled
    independently of every other, so this is trivially permutation-equivariant."""
    grid = coordinates * 2 - 1  # grid_sample expects [-1, 1], not [0, 1]
    grid = grid.unsqueeze(2)  # (B, N, 1, 2) -- one "row" of width 1 per fixation
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=False)  # (B, C, N, 1)
    return sampled.squeeze(-1).transpose(1, 2)  # (B, N, C)


class ResNetCrossAttentionFixationEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embed_dim=128,
        resnet_base_channels=64,
        resnet_layers=(2, 2, 2, 2),
        selected_levels=(1, 2, 3),
        coord_encoding_cfg=None,
        fusion_mode="cross_attention",
        residual_cross_attention=True,
        n_heads=4,
        d_ff=None,
        dropout=0.1,
    ):
        """
        Args:
            in_channels: image channel count (1 for the grayscale chest X-rays this pipeline
                serves).
            embed_dim: dimension of the per-fixation embedding, the projected image tokens,
                and the coordinate encodings -- all must match for cross-attention.
            resnet_base_channels, resnet_layers: forwarded to `ResNetEncoder`. Its pyramid has
                `1 + len(resnet_layers)` levels: the stem (index 0), then one per stage
                (index `i` = stage `i - 1`'s output, downsampled 2x further each stage after
                the first).
            selected_levels: which pyramid level indices to fuse in, and in what order -- e.g.
                `(1, 2, 3)` fuses stage 0's (highest resolution, least abstract) features
                first, then stage 1's, then stage 2's.
            coord_encoding_cfg: config dict for `build_coordinate_encoding`
                (`modules/coordinate_encoding.py`), e.g. `{"type": "fourier",
                "num_frequencies": 64, "sigma": 10.0}` or `{"type": "sinusoidal"}`.
                `embed_dim` is always forced onto it. Defaults to `{"type": "sinusoidal"}`.
                When `fusion_mode == "cross_attention"`, a separate instance (independent
                weights, when learnable) built from the same config also encodes the image
                feature maps' grid positions.
            fusion_mode: `"cross_attention"` (default) refines the running embedding against
                each selected level's full spatial feature map via `CrossAttentionBlock`, one
                level at a time. `"concat"` instead bilinearly samples each selected level's
                feature map directly at the fixation's own coordinate (`grid_sample`, no
                attention), concatenates those samples with the coordinate encoding, and
                projects the result to `embed_dim` with a single linear layer -- cheaper, and
                appropriate when you don't need the model to learn *where* to look (it already
                knows: the fixation's own location).
            residual_cross_attention: only used when `fusion_mode == "cross_attention"`.
                If True (default), each level's cross-attention output is added to the
                embedding coming into it (`CrossAttentionBlock`'s usual residual, so
                information can flow past a level unchanged if attention doesn't need it). If
                False, each level's output *replaces* the incoming embedding instead --
                useful to ablate whether carrying state across levels helps.
            n_heads, d_ff, dropout: forwarded to each level's `CrossAttentionBlock` (ignored
                when `fusion_mode == "concat"`).
        """
        super().__init__()
        if fusion_mode not in ("cross_attention", "concat"):
            raise ValueError(f"fusion_mode must be 'cross_attention' or 'concat', got {fusion_mode!r}")
        self.fusion_mode = fusion_mode

        self.backbone = ResNetEncoder(
            in_channels, resnet_base_channels, resnet_layers, return_intermediate=True
        )

        num_levels = 1 + len(resnet_layers)
        if any(level < 0 or level >= num_levels for level in selected_levels):
            raise ValueError(f"selected_levels must be in [0, {num_levels}), got {selected_levels}")
        self.selected_levels = list(selected_levels)

        level_channels = [resnet_base_channels] + list(self.backbone.out_channels)
        self.level_projections = nn.ModuleList(
            [nn.Conv2d(level_channels[level], embed_dim, kernel_size=1) for level in self.selected_levels]
        )

        coord_encoding_cfg = dict(coord_encoding_cfg or {"type": "sinusoidal"})
        coord_encoding_cfg["embed_dim"] = embed_dim
        self.coord_encoder = build_coordinate_encoding(coord_encoding_cfg)

        if fusion_mode == "cross_attention":
            # A separate module (independent weights, if learnable) for image-token grid
            # positions -- a fixation's coordinate and an image token's grid location are
            # different kinds of thing and shouldn't be forced to share parameters just
            # because they use the same encoding scheme.
            self.grid_encoder = build_coordinate_encoding(dict(coord_encoding_cfg))
            self.cross_attn_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        embed_dim, n_heads, d_ff=d_ff, dropout=dropout, residual=residual_cross_attention
                    )
                    for _ in self.selected_levels
                ]
            )
        else:  # "concat"
            self.concat_proj = nn.Linear(embed_dim * (1 + len(self.selected_levels)), embed_dim)

    def forward(self, image, coordinates):
        """
        Args:
            image: `(B, C, H, W)` full image.
            coordinates: `(B, N, 2)` fixation `(x, y)` coordinates, normalized to `[0, 1]`
                (`SingleH5Dataset`'s `*_norm` fixation columns) -- the same coordinate space
                the image feature maps are keyed by below. Padded rows (from
                `SingleH5Dataset.collate_fn`) are encoded like any other coordinate; mask them
                out downstream with `fixations_mask`, same as for the raw fixation tensor.

        Returns:
            `(B, N, embed_dim)` per-fixation embeddings.
        """
        pyramid = self.backbone(image)
        tokens = self.coord_encoder(coordinates)  # (B, N, embed_dim)

        if self.fusion_mode == "cross_attention":
            for level, proj, cross_attn in zip(self.selected_levels, self.level_projections, self.cross_attn_blocks):
                level_feat = proj(pyramid[level])  # (B, embed_dim, H_l, W_l)
                _, _, height, width = level_feat.shape
                level_tokens = level_feat.flatten(2).transpose(1, 2)  # (B, H_l * W_l, embed_dim)
                grid = self.grid_encoder(_grid_coordinates(height, width, level_feat.device))
                level_tokens = level_tokens + grid.unsqueeze(0)
                tokens = cross_attn(tokens, level_tokens)
        else:  # "concat"
            sampled = [tokens]
            for level, proj in zip(self.selected_levels, self.level_projections):
                level_feat = proj(pyramid[level])  # (B, embed_dim, H_l, W_l)
                sampled.append(_sample_at_coordinates(level_feat, coordinates))  # (B, N, embed_dim)
            tokens = self.concat_proj(torch.cat(sampled, dim=-1))

        return tokens
