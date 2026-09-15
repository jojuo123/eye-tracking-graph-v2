"""Metrics comparing two eye-tracking scanpaths (fixation sequences), e.g. a reconstructed
sequence (shuffled input re-ordered by a predicted permutation) against the true, unshuffled
one -- for validating/testing a permutation-learning model
(`dataloaders.permuted_h5_dataset.PermutedFixationH5Dataset`,
`modules.sinkhorn.GumbelSinkhorn`), where `PermutationAccuracy`/`ExactMatch`
(`trainers/metrics/custom_metrics.py`) only look at position indices, not the actual fixation
data those positions hold.
"""

import math

import torch

from trainers.metrics.metrics import METRICS


@METRICS.register("fixation_sequence_mse")
class FixationSequenceMSE:
    """Average pairwise MSE between corresponding fixations of a predicted and a target
    sequence (e.g. a reconstructed scanpath vs. the true, unshuffled one).

    `outputs`/`targets`: `(B, N, F)` float tensors, any per-fixation feature layout (e.g.
    just `(x, y)`, or `(x, y, duration)` -- averaged over the feature dimension first, then
    over valid sequence positions). Pass `mask` (`(B, N)` bool, True at valid/non-padded
    positions) to exclude padded fixations from the average.
    """

    @torch.no_grad()
    def __call__(self, outputs, targets, mask=None):
        error = (outputs - targets).pow(2).mean(dim=-1)  # (B, N)
        if mask is None:
            return error.mean()
        mask = mask.to(error.dtype)
        return (error * mask).sum() / mask.sum().clamp_min(1e-8)


def _saccade_vectors(xy):
    """`(n, 2)` fixation positions -> `(n - 1, 2)` consecutive saccade vectors."""
    return xy[1:] - xy[:-1]


def _align(cost):
    """Lowest-cost monotonic alignment path from `(0, 0)` to `(m - 1, n - 1)` through an
    `(m, n)` cost matrix, moving right/down/diagonally at each step -- the reference
    MultiMatch algorithm finds this via Dijkstra's shortest path over the same grid graph;
    this dynamic-programming formulation (equivalent, since the grid is a DAG) is simpler
    to implement correctly. Returns the path as a list of `(i, j)` index pairs, one per
    aligned (vector_a, vector_b) pair.
    """
    m, n = cost.shape
    dp = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            best_prev = 0.0
            if i > 0 and j > 0:
                best_prev = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            elif i > 0:
                best_prev = dp[i - 1][j]
            elif j > 0:
                best_prev = dp[i][j - 1]
            dp[i][j] = cost[i, j].item() + best_prev

    path = [(m - 1, n - 1)]
    i, j = m - 1, n - 1
    while (i, j) != (0, 0):
        candidates = []
        if i > 0:
            candidates.append((dp[i - 1][j], i - 1, j))
        if j > 0:
            candidates.append((dp[i][j - 1], i, j - 1))
        if i > 0 and j > 0:
            candidates.append((dp[i - 1][j - 1], i - 1, j - 1))
        _, i, j = min(candidates, key=lambda c: c[0])
        path.append((i, j))
    path.reverse()
    return path


def _multimatch_single(fixations_a, fixations_b, screen_diagonal):
    """MultiMatch's 5 similarity dimensions (1 = identical, 0 = maximally different)
    between two single scanpaths. `fixations_a`/`fixations_b`: `(n, 3)` CPU tensors,
    columns `(x, y, duration)`. Returns `None` if either scanpath is too short (< 2
    fixations) to form a single saccade vector.
    """
    if fixations_a.shape[0] < 2 or fixations_b.shape[0] < 2:
        return None

    vectors_a = _saccade_vectors(fixations_a[:, :2])
    vectors_b = _saccade_vectors(fixations_b[:, :2])
    # "Vector difference" cost drives the alignment itself, per the reference algorithm.
    cost = torch.linalg.norm(vectors_a.unsqueeze(1) - vectors_b.unsqueeze(0), dim=-1)  # (m, n)
    path = _align(cost)

    lengths_a, lengths_b = torch.linalg.norm(vectors_a, dim=-1), torch.linalg.norm(vectors_b, dim=-1)
    angles_a = torch.atan2(vectors_a[:, 1], vectors_a[:, 0])
    angles_b = torch.atan2(vectors_b[:, 1], vectors_b[:, 0])

    vector_d, length_d, direction_d, position_d, duration_d = [], [], [], [], []
    for i, j in path:
        vector_d.append(cost[i, j].item())
        length_d.append(abs(lengths_a[i].item() - lengths_b[j].item()))
        angle_diff = abs(angles_a[i].item() - angles_b[j].item())
        direction_d.append(min(angle_diff, 2 * math.pi - angle_diff))  # smaller of the two arcs
        position_d.append(torch.linalg.norm(fixations_a[i, :2] - fixations_b[j, :2]).item())
        duration_a, duration_b = fixations_a[i, 2].item(), fixations_b[j, 2].item()
        duration_d.append(abs(duration_a - duration_b) / max(duration_a, duration_b, 1e-8))

    def similarity(distances, normalizer):
        return max(0.0, 1.0 - (sum(distances) / len(distances)) / normalizer)

    return {
        "vector": similarity(vector_d, screen_diagonal),
        "length": similarity(length_d, screen_diagonal),
        "direction": similarity(direction_d, math.pi),
        "position": similarity(position_d, screen_diagonal),
        "duration": similarity(duration_d, 1.0),
    }


@METRICS.register("multimatch")
class MultiMatch:
    """MultiMatch scanpath similarity (Dewhurst et al., 2012, "It depends on how you look
    at it: Scanpath comparison in multiple dimensions with MultiMatch, a vector-based
    approach"): represents each scanpath as its sequence of saccade vectors, finds the
    lowest-cost monotonic alignment between the two scanpaths' vectors, then reports 5
    normalized similarity scores (1 = identical) over that alignment: vector (shape),
    length, direction, position, and duration.

    Note: this implementation skips the original algorithm's optional scanpath
    "simplification" pass (iteratively merging locally-insignificant consecutive
    saccades) -- it compares the raw fixation sequences directly.

    Unlike every other metric in this module, `__call__` returns a *dict* of the 5
    sub-scores plus their mean under `"multimatch"` (the same pattern
    `trainers.losses.custom_losses.WeightedMultiLoss` uses for its breakdown), since
    MultiMatch is inherently multi-dimensional -- there is no single canonical scalar.

    Args:
        screen_diagonal: normalizer for the spatial dimensions (vector, length,
            position), i.e. the maximum possible distance in the coordinate space
            `outputs`/`targets` use. Defaults to `sqrt(2)`, the diagonal of the unit
            square -- correct if fixation coordinates are normalized to `[0, 1]` (e.g.
            `SingleH5Dataset`'s `*_norm` columns).

    `outputs`/`targets`: `(B, N, 3)` float tensors, columns `(x, y, duration)`. `mask`:
    `(B, N)` bool, True at valid (non-padded) fixations. Samples with fewer than 2 valid
    fixations are skipped (too short to form a saccade vector) and don't contribute to
    the average.
    """

    def __init__(self, screen_diagonal=2**0.5):
        self.screen_diagonal = screen_diagonal
        self.dimensions = ("vector", "length", "direction", "position", "duration")

    @torch.no_grad()
    def __call__(self, outputs, targets, mask=None):
        totals = {dim: 0.0 for dim in self.dimensions}
        count = 0
        for b in range(outputs.shape[0]):
            n = int(mask[b].sum().item()) if mask is not None else outputs.shape[1]
            result = _multimatch_single(outputs[b, :n].cpu(), targets[b, :n].cpu(), self.screen_diagonal)
            if result is None:
                continue
            for dim in self.dimensions:
                totals[dim] += result[dim]
            count += 1

        if count == 0:
            scores = {dim: torch.zeros(()) for dim in self.dimensions}
        else:
            scores = {dim: torch.tensor(total / count) for dim, total in totals.items()}
        scores["multimatch"] = torch.stack(list(scores.values())).mean()
        return scores
