"""Permutation-recovery model for REFLACX fixation sequences: given a chest X-ray image and
a scrambled fixation sequence (`dataloaders.permuted_h5_dataset.PermutedFixationH5Dataset`),
predicts the permutation that recovers the sequence's original (temporal) order.

Architecture:
  1. `modules.multiscale_fixation_encoder.ResNetCrossAttentionFixationEncoder` embeds each
     fixation from the full image plus its own `(x, y)` location -- permutation-equivariant
     by construction (see that module's docstring): a fixation's embedding never depends on
     any other fixation's identity or position in the sequence.
  2. An optional `modules.transformer.TransformerEncoder` (with `use_positional_encoding=
     False`, for the same reason) lets fixations exchange context via self-attention --
     still permutation-equivariant, since nothing here is keyed by sequence index either.
  3. A bilinear head scores every (input slot, candidate target position) pair into an
     `(N, N)` matrix `log_alpha` (same scheme as `models.sinkhorn_sort_model.
     SinkhornPatchSorter`). `modules.sinkhorn.GumbelSinkhorn` turns it into a soft,
     doubly-stochastic matrix for training; `hungarian_matching` turns it into a hard
     permutation for inference/metrics. Both are mask-aware, so variable-length, padded
     fixation sequences (`fixations_mask`) are handled correctly throughout.

Loss: any combination of `"permutation"`, `"doubly_stochastic_cross_entropy"`,
`"permutation_reconstruction"`, and `"permutation_wasserstein"` (all in
`trainers.losses.permutation_losses`), each independently weighted via `loss_cfg` -- see
`FixationPermutationSorter.__init__`.

Metrics (always computed when targets are present): `"permutation_accuracy"` (hard
permutation vs. ground truth) and `"fixation_sequence_mse"` (the sequence reconstructed via
the hard permutation vs. the true, unshuffled one).
"""

import torch
import torch.nn as nn

from models import MODELS
from models.base_model import BaseModel
from modules.multiscale_fixation_encoder import ResNetCrossAttentionFixationEncoder
from modules.sinkhorn import GumbelSinkhorn, hungarian_matching
from modules.transformer import TransformerEncoder
from trainers.losses import build_loss, permutation_to_matrix
from trainers.metrics import METRICS

_LOSS_TYPES = (
    "permutation",
    "doubly_stochastic_cross_entropy",
    "permutation_reconstruction",
    "permutation_wasserstein",
)
_METRIC_TYPES = ("permutation_accuracy", "fixation_sequence_mse")


@MODELS.register("fixation_permutation_sorter")
class FixationPermutationSorter(BaseModel):
    """Batch contract (see module docstring for the task):
      - "image": `(B, C, H, W)` float tensor.
      - "fixations": `(B, N, F)` float tensor -- only `coordinate_indices` are used by this
        model; other columns (duration, etc.), if present, are ignored.
      - "fixations_mask" (optional): `(B, N)` bool, True at valid (non-padded) positions.
      - "permutation" (optional): `(B, N)` long tensor, the target column for each row (see
        `modules.sinkhorn`'s docstring). Present during training/validation, absent at pure
        inference time -- when absent, only `"outputs"`/`"log_alpha"` are returned.
      - "inverse_permutation" (optional): `(B, N)` long tensor, required only by the
        `"permutation_reconstruction"` loss and the `"fixation_sequence_mse"` metric.
      - "soft_permutation" (optional): `(B, N, N)` float tensor (see
        `data.reflacx.reflacx.compute_soft_ground_truth`) -- if present, used as the
        `"doubly_stochastic_cross_entropy"` loss's target instead of the hard one-hot
        permutation matrix built from `"permutation"`.
    """

    def __init__(
        self,
        coordinate_indices=(0, 1),
        embed_dim=128,
        encoder_cfg=None,
        context_n_layers=2,
        context_n_heads=4,
        context_d_ff=None,
        context_dropout=0.1,
        sinkhorn_cfg=None,
        loss_cfg=None,
        metrics_cfg=None,
    ):
        """
        Args:
            coordinate_indices: `(x_index, y_index)` into `fixations`' last dimension. Must
                match whatever `fixation_columns` the dataset was configured with -- e.g. if
                a training config sets `fixation_columns: [x_position_norm,
                y_position_norm]`, `(0, 1)` (the default) is correct.
            embed_dim: shared dimension for the fixation encoder, the context encoder, and
                the bilinear scoring head. Forced onto `encoder_cfg["embed_dim"]`.
            encoder_cfg: kwargs for `ResNetCrossAttentionFixationEncoder` (e.g.
                `in_channels`, `resnet_layers`, `selected_levels`, `coord_encoding_cfg`,
                `fusion_mode`). `embed_dim` is always forced onto it; see that class for
                every other option and its default.
            context_n_layers: number of self-attention layers mixing context across
                fixations after encoding (0 disables this stage entirely, and the bilinear
                head scores the per-fixation embeddings directly, as in
                `SinkhornPatchSorter`).
            context_n_heads, context_d_ff, context_dropout: forwarded to the context
                `TransformerEncoder`, if `context_n_layers > 0`.
            sinkhorn_cfg: kwargs for `GumbelSinkhorn` (`tau`, `n_iters`, `noise_factor`,
                `n_samples`).
            loss_cfg: list of `{"type": <one of _LOSS_TYPES>, "weight": <float, default
                1.0>, "name": <optional, default: type>, **loss-specific kwargs}` dicts --
                any single one of `"permutation"`, `"doubly_stochastic_cross_entropy"`,
                `"permutation_reconstruction"`, `"permutation_wasserstein"`, or a weighted
                combination of several (`"name"` disambiguates using the same type twice,
                e.g. two `"permutation_wasserstein"` terms with different `ground_metric`).
                Defaults to a single `{"type": "permutation", "weight": 1.0}`.
            metrics_cfg: which of `_METRIC_TYPES` (`"permutation_accuracy"`,
                `"fixation_sequence_mse"`) to compute, as a list of names. Defaults to both.
        """
        super().__init__()
        encoder_cfg = dict(encoder_cfg or {})
        encoder_cfg["embed_dim"] = embed_dim
        self.encoder = ResNetCrossAttentionFixationEncoder(**encoder_cfg)

        self.coordinate_indices = tuple(coordinate_indices)

        self.context_encoder = None
        if context_n_layers > 0:
            self.context_encoder = TransformerEncoder(
                embed_dim,
                context_n_heads,
                context_n_layers,
                d_ff=context_d_ff,
                dropout=context_dropout,
                use_positional_encoding=False,
            )

        self.row_proj = nn.Linear(embed_dim, embed_dim)
        self.col_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

        self.sinkhorn = GumbelSinkhorn(**(sinkhorn_cfg or {}))

        loss_cfg = loss_cfg or [{"type": "permutation", "weight": 1.0}]
        self.loss_terms = []  # list of (name, weight, loss_type), modules live in loss_modules
        self.loss_modules = nn.ModuleDict()
        for term_cfg in loss_cfg:
            term_cfg = dict(term_cfg)
            loss_type = term_cfg.pop("type")
            if loss_type not in _LOSS_TYPES:
                raise ValueError(f"loss_cfg entry type must be one of {_LOSS_TYPES}, got {loss_type!r}")
            weight = term_cfg.pop("weight", 1.0)
            name = term_cfg.pop("name", None) or loss_type
            if name in self.loss_modules:
                raise ValueError(f"duplicate loss name {name!r} -- pass a distinct 'name' for each occurrence")
            self.loss_modules[name] = build_loss({"type": loss_type, **term_cfg})
            self.loss_terms.append((name, weight, loss_type))

        metrics_cfg = list(metrics_cfg) if metrics_cfg is not None else list(_METRIC_TYPES)
        unknown = set(metrics_cfg) - set(_METRIC_TYPES)
        if unknown:
            raise ValueError(f"metrics_cfg entries must be one of {_METRIC_TYPES}, got unknown: {sorted(unknown)}")
        self.enabled_metrics = set(metrics_cfg)
        self.permutation_accuracy_metric = METRICS.build({"type": "permutation_accuracy"})
        self.fixation_sequence_mse_metric = METRICS.build({"type": "fixation_sequence_mse"})

    def _encode(self, image, coords, mask):
        tokens = self.encoder(image, coords)  # (B, N, embed_dim), permutation-equivariant
        if self.context_encoder is not None:
            key_padding_mask = ~mask if mask is not None else None
            tokens = self.context_encoder(tokens, key_padding_mask=key_padding_mask)
        return tokens

    def _log_alpha(self, tokens):
        row = self.row_proj(tokens)
        col = self.col_proj(tokens)
        return torch.matmul(row, col.transpose(-1, -2)) * self.scale  # (B, N, N)

    @staticmethod
    def _reconstruct(matrix, source_sequence):
        """`reconstructed[b, j] = sum_i matrix[b, i, j] * source_sequence[b, i]` -- recovers
        the original sequence from `source_sequence` under a (soft or hard) permutation
        `matrix`. See `trainers.losses.permutation_losses.PermutationReconstructionLoss`."""
        return torch.einsum("bij,bif->bjf", matrix, source_sequence)

    @staticmethod
    def _gather_by_inverse(sequence, inverse_permutation):
        safe_inverse = inverse_permutation.clamp(min=0)
        index = safe_inverse.unsqueeze(-1).expand(-1, -1, sequence.shape[-1])
        return torch.gather(sequence, 1, index)

    def forward(self, batch):
        image = batch["image"]
        # torch.cat of single-column slices (rather than fancy-indexing with a list) keeps
        # this robust to non-contiguous coordinate_indices, e.g. (0, 3).
        coords = torch.cat([batch["fixations"][..., i : i + 1] for i in self.coordinate_indices], dim=-1)
        mask = batch.get("fixations_mask")

        tokens = self._encode(image, coords, mask)
        log_alpha = self._log_alpha(tokens)
        hard_perm = hungarian_matching(log_alpha.detach(), mask=mask)

        output = {"outputs": hard_perm, "log_alpha": log_alpha}

        perm = batch.get("permutation")
        if perm is not None:
            soft_perm = self.sinkhorn(log_alpha, mask=mask)
            loss, losses = self._compute_losses(
                soft_perm, coords, perm, batch.get("inverse_permutation"), mask, batch.get("soft_permutation")
            )
            output["loss"] = loss
            output["losses"] = losses
            output["metrics"] = self.compute_metrics(hard_perm, perm, coords, batch.get("inverse_permutation"), mask)

        return output

    def _compute_losses(self, soft_perm, coords, perm, inverse_permutation, mask, soft_permutation_target):
        batch_size = perm.shape[0]
        n_samples = soft_perm.shape[0] // batch_size

        def tile(x):
            return x if (x is None or n_samples == 1) else x.repeat_interleave(n_samples, dim=0)

        tiled_perm, tiled_mask, tiled_coords = tile(perm), tile(mask), tile(coords)

        total = soft_perm.new_zeros(())
        breakdown = {}
        for name, weight, loss_type in self.loss_terms:
            loss_fn = self.loss_modules[name]
            if loss_type in ("permutation", "permutation_wasserstein"):
                value = loss_fn(soft_perm, tiled_perm, mask=tiled_mask)
            elif loss_type == "doubly_stochastic_cross_entropy":
                if soft_permutation_target is not None:
                    target_matrix = tile(soft_permutation_target)
                else:
                    target_matrix = permutation_to_matrix(tiled_perm, mask=tiled_mask)
                value = loss_fn(soft_perm, target_matrix, mask=tiled_mask)
            elif loss_type == "permutation_reconstruction":
                assert inverse_permutation is not None, (
                    "'permutation_reconstruction' requires 'inverse_permutation' in the batch "
                    "(use PermutedFixationH5Dataset)"
                )
                target_sequence = self._gather_by_inverse(tiled_coords, tile(inverse_permutation))
                value = loss_fn(soft_perm, tiled_coords, target_sequence, mask=tiled_mask)
            total = total + weight * value
            breakdown[name] = value.detach()
        return total, breakdown

    def compute_metrics(self, hard_perm, perm, coords, inverse_permutation, mask):
        metrics = {}
        if "permutation_accuracy" in self.enabled_metrics:
            metrics["permutation_accuracy"] = self.permutation_accuracy_metric(hard_perm, perm, mask=mask)
        if "fixation_sequence_mse" in self.enabled_metrics and inverse_permutation is not None:
            hard_matrix = permutation_to_matrix(hard_perm, mask=mask)
            reconstructed = self._reconstruct(hard_matrix, coords)
            target_sequence = self._gather_by_inverse(coords, inverse_permutation)
            metrics["fixation_sequence_mse"] = self.fixation_sequence_mse_metric(reconstructed, target_sequence, mask=mask)
        return metrics
