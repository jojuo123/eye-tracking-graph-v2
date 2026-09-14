"""Sequence-reordering model built on the Gumbel-Sinkhorn operator (see
`modules/sinkhorn.py`).

Input: a sequence of `N` `(2D coordinate, image patch)` pairs presented in
some arbitrary/shuffled order (e.g. jigsaw-style shuffled image patches, or
eye-tracking fixations -- a coordinate plus the image patch around it -- in
scrambled temporal order). The model predicts the permutation that recovers
their correct/canonical order.

Architecture: encode each patch with a swappable image encoder (see
`modules/encoders.py` -- a small CNN, a ResNet, a ViT-style patch+attention
encoder, or any custom `ENCODERS`-registered backbone), encode each coordinate
with an MLP, sum them into one token per sequence element, run a Transformer
encoder over the sequence (self-attention lets every element's prediction
depend on all the others -- necessary since "correct order" is a joint, not
per-element, property), then score every (input slot, candidate target
position) pair with a bilinear head to get an `(N, N)` matrix `log_alpha`.
`GumbelSinkhorn` turns `log_alpha` into a soft, doubly-stochastic permutation
matrix for training (via a Sinkhorn/NLL loss against the ground-truth
permutation) and `hungarian_matching` turns it into a hard permutation for
inference/metrics.
"""

import torch
import torch.nn as nn

from models import MODELS
from models.base_model import BaseModel
from modules.encoders import build_encoder
from modules.mlp import MLP
from modules.sinkhorn import GumbelSinkhorn, hungarian_matching
from modules.transformer import TransformerEncoder
from trainers.losses import build_loss
from trainers.metrics import MetricCollection


@MODELS.register("sinkhorn_patch_sorter")
class SinkhornPatchSorter(BaseModel):
    """Batch contract (see module docstring for the task):
      - "patches": `(B, N, C, H, W)` float tensor, image patches in their
        current (e.g. shuffled) order.
      - "coords": `(B, N, coord_dim)` float tensor, the 2D coordinate
        associated with each patch, same order as "patches".
      - "perm" (optional): `(B, N)` long tensor. `perm[b, i]` is the target
        position that the element currently at slot `i` belongs to. Present
        during training/validation, absent at pure inference time.

    `encoder_cfg` selects the patch encoder via the `ENCODERS` registry
    (`modules/encoders.py`), e.g. `{"type": "resnet", "layers": [2, 2, 2, 2]}`
    or `{"type": "vit", "sub_patch_size": 4}` -- defaults to a lightweight
    `simple_cnn`. `in_channels` and `out_dim` are always forced onto it (the
    latter to `embed_dim`, so patch features can be added to coordinate
    features token-wise); any other kwarg is forwarded as-is.
    """

    def __init__(
        self,
        in_channels=3,
        coord_dim=2,
        embed_dim=128,
        encoder_cfg=None,
        n_heads=4,
        n_layers=4,
        d_ff=None,
        dropout=0.1,
        sinkhorn_cfg=None,
        loss_cfg=None,
        metrics_cfg=None,
    ):
        super().__init__()
        encoder_cfg = dict(encoder_cfg or {"type": "simple_cnn", "channels": (32, 64, 128)})
        encoder_cfg["in_channels"] = in_channels
        encoder_cfg["out_dim"] = embed_dim
        self.patch_encoder = build_encoder(encoder_cfg)
        self.coord_encoder = MLP(coord_dim, [embed_dim], embed_dim, activation="gelu")
        self.encoder = TransformerEncoder(embed_dim, n_heads, n_layers, d_ff=d_ff, dropout=dropout)

        self.row_proj = nn.Linear(embed_dim, embed_dim)
        self.col_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

        self.sinkhorn = GumbelSinkhorn(**(sinkhorn_cfg or {}))
        self.loss_fn = build_loss(loss_cfg or {"type": "nll"})
        self.metrics = MetricCollection(metrics_cfg or [{"type": "permutation_accuracy"}, {"type": "exact_match"}])

    def _encode(self, patches, coords):
        batch_size, n, c, h, w = patches.shape
        patch_feats = self.patch_encoder(patches.reshape(batch_size * n, c, h, w)).reshape(batch_size, n, -1)
        coord_feats = self.coord_encoder(coords)
        tokens = patch_feats + coord_feats
        return self.encoder(tokens)  # (B, N, embed_dim)

    def _log_alpha(self, tokens):
        row = self.row_proj(tokens)
        col = self.col_proj(tokens)
        return torch.matmul(row, col.transpose(-1, -2)) * self.scale  # (B, N, N)

    def forward(self, batch):
        patches, coords = batch["patches"], batch["coords"]
        perm = batch.get("perm")

        tokens = self._encode(patches, coords)
        log_alpha = self._log_alpha(tokens)
        hard_perm = hungarian_matching(log_alpha.detach())

        output = {"outputs": hard_perm, "log_alpha": log_alpha}
        if perm is not None:
            batch_size, n, _ = log_alpha.shape
            soft_perm = self.sinkhorn(log_alpha)  # (B * n_samples, N, N), doubly-stochastic
            n_samples = soft_perm.size(0) // batch_size
            perm_target = perm.repeat_interleave(n_samples, dim=0) if n_samples > 1 else perm

            log_probs = torch.log(soft_perm.clamp_min(1e-20))
            loss = self.loss_fn(log_probs.reshape(-1, n), perm_target.reshape(-1))

            output["loss"] = loss
            output["losses"] = {"nll_loss": loss.detach()}
            output["metrics"] = self.compute_metrics(hard_perm, perm)
        return output

    def compute_metrics(self, outputs, targets):
        return self.metrics(outputs, targets)
