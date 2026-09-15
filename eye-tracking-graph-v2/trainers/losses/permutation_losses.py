"""Losses for permutation-learning tasks: recovering a scrambled fixation sequence's
original order from a predicted, possibly-soft, permutation matrix (see
`modules.sinkhorn.GumbelSinkhorn` and `dataloaders.permuted_h5_dataset.PermutedFixationH5Dataset`).

All four losses here take an optional `mask` -- `(B, N)` bool, True at valid (non-padded)
rows, e.g. a batch's `fixations_mask` -- and are self-contained about it: unlike
`modules.sinkhorn.apply_padding_mask` (which forces a *matrix's* padded block into an
identity, a device for keeping Sinkhorn/Hungarian numerically well-posed), these losses
simply exclude every padded row from whatever they average over, and never assume the
matrix passed in has already been through `apply_padding_mask` -- pass `mask` regardless.
"""

import torch
import torch.nn as nn

from trainers.losses.losses import LOSSES


def _valid_mean(values, mask):
    """Mean of `values` (any shape) over `mask` (same shape, True = keep; None to average
    over everything). `clamp_min` on the denominator avoids a NaN (instead returning ~0)
    on the edge case of a batch with zero valid entries, with no data-dependent branching
    (so no CPU/GPU sync)."""
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1e-8)
    return (values * mask).sum() / denom


def permutation_to_matrix(perm, num_positions=None, mask=None):
    """Builds the `(B, N, N)` hard permutation matrix a `(B, N)` integer `perm` implies
    (`matrix[b, i, perm[b, i]] = 1`), for use as the target in `PermutationLoss` or
    `DoublyStochasticCrossEntropyLoss`.

    Padded rows (`mask[b, i]` False, or `perm[b, i] < 0` if `mask` isn't given -- matching
    `PermutedFixationH5Dataset.collate_fn`'s `-1` padding) get an identity row instead
    (`matrix[b, i, i] = 1`), since a real one-hot row can't be built from a `-1` target;
    this mirrors `modules.sinkhorn.apply_padding_mask`'s convention for the padded block.
    """
    batch_size, n = perm.shape
    num_positions = num_positions or n
    if mask is None:
        mask = perm >= 0
    identity_targets = torch.arange(n, device=perm.device).expand(batch_size, n)
    safe_perm = torch.where(mask, perm, identity_targets).clamp(min=0, max=num_positions - 1)
    matrix = torch.zeros(batch_size, n, num_positions, device=perm.device, dtype=torch.float32)
    matrix.scatter_(-1, safe_perm.unsqueeze(-1), 1.0)
    return matrix


@LOSSES.register("permutation_reconstruction")
class PermutationReconstructionLoss(nn.Module):
    """Applies the predicted (soft or hard) permutation to the shuffled input sequence and
    penalizes its distance (MSE, by default) from the true, unshuffled sequence -- the
    "reconstruction" loss for Gumbel-Sinkhorn-style permutation learning (Mena, Belanger,
    Linderman & Snoek, ICLR 2018): given `perm[b, i]` = the target position slot `i`'s
    element belongs to (see `modules.sinkhorn`'s docstring), the predicted doubly-stochastic
    matrix `P` reconstructs position `j` of the original sequence as
    `sum_i P[b, i, j] * source[b, i]`, i.e. `reconstructed = P^T @ source`.
    """

    def __init__(self, reduction="mse"):
        super().__init__()
        if reduction not in ("mse", "l1"):
            raise ValueError(f"reduction must be 'mse' or 'l1', got {reduction!r}")
        self.reduction = reduction

    def forward(self, soft_perm, source_sequence, target_sequence, mask=None):
        """
        Args:
            soft_perm: `(B, N, N)` doubly-stochastic matrix (e.g. `GumbelSinkhorn`'s output).
            source_sequence: `(B, N, F)`, the shuffled sequence `soft_perm` was predicted for.
            target_sequence: `(B, N, F)`, the true, unshuffled sequence.
            mask: `(B, N)` bool, True at valid (non-padded) positions of `target_sequence`.
        """
        reconstructed = torch.einsum("bij,bif->bjf", soft_perm, source_sequence)
        if self.reduction == "mse":
            error = (reconstructed - target_sequence).pow(2).mean(dim=-1)
        else:
            error = (reconstructed - target_sequence).abs().mean(dim=-1)
        return _valid_mean(error, mask)


@LOSSES.register("permutation_wasserstein")
class PermutationWassersteinLoss(nn.Module):
    """Wasserstein/earth-mover loss between each row of the predicted doubly-stochastic
    matrix and the delta distribution the ground-truth permutation places on its target
    column (Frogner et al., NeurIPS 2015, "Learning with a Wasserstein Loss"): since the
    target is a delta at `perm[b, i]`, the exact Wasserstein distance between it and
    predicted row `P[b, i, :]` under ground metric `M` -- no Sinkhorn/OT solve needed --
    is simply `sum_j P[b, i, j] * M(j, perm[b, i])`. Unlike cross-entropy, which only cares
    whether the correct column got the most mass, this gives partial credit that shrinks
    smoothly as predicted mass moves closer (in column *index*) to the true target --
    appropriate since a fixation's neighboring sequence positions are usually genuinely
    more plausible mistakes than a far-away one.
    """

    def __init__(self, ground_metric="squared"):
        super().__init__()
        if ground_metric not in ("squared", "abs"):
            raise ValueError(f"ground_metric must be 'squared' or 'abs', got {ground_metric!r}")
        self.ground_metric = ground_metric

    def forward(self, soft_perm, target_perm, mask=None):
        """
        Args:
            soft_perm: `(B, N, N)` doubly-stochastic matrix.
            target_perm: `(B, N)` integer target column per row (see `modules.sinkhorn`'s
                docstring); padded rows may hold any value (e.g. `-1`) since `mask` excludes
                them from both the distance computation and the final average.
            mask: `(B, N)` bool, True at valid (non-padded) rows.
        """
        n = soft_perm.shape[-1]
        safe_target = target_perm.clamp(min=0, max=n - 1)
        positions = torch.arange(n, device=soft_perm.device).view(1, 1, n)
        distance = positions - safe_target.unsqueeze(-1)  # (B, N, N)
        cost = distance.pow(2) if self.ground_metric == "squared" else distance.abs()
        per_row = (soft_perm * cost).sum(dim=-1)  # (B, N)
        return _valid_mean(per_row, mask)


@LOSSES.register("permutation")
class PermutationLoss(nn.Module):
    """The "permutation loss" of Wang, Yan & Yang, "Learning Combinatorial Embedding
    Networks for Deep Graph Matching", ICCV 2019, Sec. 3.6 ("Permutation Cross-Entropy
    Loss"), Eq. (17):

        L_perm = -sum_{i, j} ( S^gt_ij * log(S_ij) + (1 - S^gt_ij) * log(1 - S_ij) )

    i.e. *element-wise* binary cross-entropy between the predicted doubly-stochastic
    matrix `S` and the target matrix `S^gt` (typically the hard one-hot permutation matrix
    from `permutation_to_matrix`, but the paper's derivation only needs `S^gt` to be
    non-negative -- REFLACX's soft ground truth, `data.reflacx.reflacx.
    compute_soft_ground_truth`, also works). This is *not* the same as `PermutationLoss`
    treating rows as categorical distributions (that's `DoublyStochasticCrossEntropyLoss`,
    below): the negative term `(1 - S^gt_ij) * log(1 - S_ij)` additionally pushes every
    *non*-target entry of a row towards 0, not just the target entry towards 1.

    Following the reference implementation (github.com/Thinklab-SJTU/ThinkMatch), each
    row's BCE terms are *summed* over columns, and that per-row sum is then averaged over
    valid rows (via `_valid_mean`) -- not averaged over columns too -- so the loss scales
    with how spread out a row's incorrect mass is, not just whether the argmax is right.
    """

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, predicted, target, mask=None):
        """
        Args:
            predicted: `(B, N, N)` doubly-stochastic matrix (`S`).
            target: `(B, N, N)` target matrix (`S^gt`), e.g.
                `permutation_to_matrix(perm, mask=mask)`.
            mask: `(B, N)` bool, True at valid (non-padded) rows -- also used to exclude
                padded *columns* from each valid row's sum, since this loss (unlike the
                others in this module) touches every entry of the matrix, not just the
                target column per row.
        """
        # Clamp each log's argument separately (not `predicted` itself, e.g. via
        # `.clamp(eps, 1 - eps)`): in float32, `1 - eps` for a small `eps` can round back to
        # exactly `1.0`, making `log(1 - predicted)` = `log(0)` = `-inf` at a target-1 entry
        # -- multiplied by that entry's `(1 - target) == 0` weight, IEEE float arithmetic
        # gives `0 * -inf = NaN`, not the mathematically-intended 0.
        positive_term = target * torch.log(predicted.clamp_min(self.eps))
        negative_term = (1 - target) * torch.log((1 - predicted).clamp_min(self.eps))
        bce = -(positive_term + negative_term)
        if mask is not None:
            valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, N, N)
            bce = bce * valid_pair.to(bce.dtype)
        per_row = bce.sum(dim=-1)  # (B, N)
        return _valid_mean(per_row, mask)


@LOSSES.register("doubly_stochastic_cross_entropy")
class DoublyStochasticCrossEntropyLoss(nn.Module):
    """Element-wise *categorical* cross-entropy between a target doubly-stochastic matrix
    (typically the hard, one-hot permutation matrix from `permutation_to_matrix`, but any
    valid row-stochastic target works, e.g. label-smoothed targets) and the predicted one:
    per row `i`, `-sum_j target[b, i, j] * log(predicted[b, i, j])`.

    Unlike `PermutationLoss` (Wang, Yan & Yang, ICCV 2019's element-wise *binary*
    cross-entropy, which also penalizes non-target entries via a `(1 - target) * log(1 -
    predicted)` term), this only scores the target entries -- the more standard
    "cross-entropy against a soft label" loss, and *not* the same value even when `target`
    is a hard one-hot matrix.
    """

    def __init__(self, eps=1e-20):
        super().__init__()
        self.eps = eps

    def forward(self, predicted, target, mask=None):
        """
        Args:
            predicted: `(B, N, N)` doubly-stochastic matrix.
            target: `(B, N, N)` target distribution matrix, row-stochastic (each row sums
                to 1); e.g. `permutation_to_matrix(perm, mask=mask)`.
            mask: `(B, N)` bool, True at valid (non-padded) rows.
        """
        log_probs = torch.log(predicted.clamp_min(self.eps))
        per_row = -(target * log_probs).sum(dim=-1)  # (B, N)
        return _valid_mean(per_row, mask)
