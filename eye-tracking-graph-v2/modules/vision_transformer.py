"""Vision Transformer (Dosovitskiy et al., 2020), supporting 2D images or 3D
volumes/video via `spatial_dims` (patch embedding generalizes to Conv3d).
"""

import torch
import torch.nn as nn

from modules.nd import get_nd_layers
from modules.transformer import TransformerEncoderBlock


class PatchEmbedding(nn.Module):
    """Splits an image/volume into non-overlapping patches and linearly embeds
    each one via a single strided conv (`kernel_size == stride == patch_size`).
    """

    def __init__(self, in_channels, embed_dim, patch_size, spatial_dims=2):
        super().__init__()
        Conv, _, _, _, _ = get_nd_layers(spatial_dims)
        self.proj = Conv(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                   # (B, embed_dim, *spatial_patches)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)


class VisionTransformer(nn.Module):
    """Standard ViT: patch embed -> prepend `[CLS]` token -> add a learned
    positional embedding -> a stack of `TransformerEncoderBlock`s -> LayerNorm
    + classification head on the `[CLS]` token's final representation.

    Works for 2D images (`spatial_dims=2`, `image_size`/`patch_size` an int or
    `(H, W)`) or 3D volumes/video (`spatial_dims=3`, an int or `(D, H, W)`).
    `image_size` must be divisible by `patch_size` in every dimension.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        num_classes,
        embed_dim=768,
        depth=12,
        n_heads=12,
        d_ff=None,
        dropout=0.1,
        spatial_dims=2,
    ):
        super().__init__()
        image_size = _to_tuple(image_size, spatial_dims)
        patch_size = _to_tuple(patch_size, spatial_dims)
        num_patches = 1
        for size, patch in zip(image_size, patch_size):
            assert size % patch == 0, f"image_size {image_size} must be divisible by patch_size {patch_size}"
            num_patches *= size // patch

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size, spatial_dims=spatial_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, n_heads, d_ff, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x[:, 0]))


def _to_tuple(value, n):
    if isinstance(value, (tuple, list)):
        assert len(value) == n, f"expected {n} values, got {value}"
        return tuple(value)
    return tuple([value] * n)
