"""The shortlisted scanpath-order metrics, every one normalized to a `[0, 1]`
*dissimilarity* (0 = the predicted order reproduces the ground truth exactly, 1 = the
metric's worst possible value), so they can share one axis and one reading direction.
Used by both `run_damage_evaluation.py` (synthetic damage) and
`run_model_evaluation.py` (a trained model's predicted orderings).

    kendall_tau           -- pairwise order disagreements / n(n-1)/2
    windowed_kendall_tau  -- same, ignoring pairs within a DTW-style warping window of
                              `w * n` GT positions; / number of pairs outside the window
    spearman_footrule     -- total rank displacement / floor(n^2 / 2)
    dtw                   -- DTW cost / (n + m - 1) / sqrt(2): mean per-step cost along
                              the warping path, as a fraction of the unit square's
                              diagonal (the largest possible point-to-point distance)
    lcss                  -- 1 - LCSS length / min(n, m) (already in [0, 1])
    edr                   -- EDR / max(n, m) (already in [0, 1])
    scanmatch             -- 1 - ScanMatch similarity
    mm_vector, mm_length, mm_direction, mm_position, multimatch
                          -- 1 - each MultiMatch similarity (and 1 - their mean)

Every coordinate-based metric assumes `[0, 1]`-normalized `(x, y)` positions -- the
DTW bound and the `eps`/grid defaults are only meaningful on that scale.
"""

import numpy as np

from permutation_distances import (
    kendall_tau_distance,
    spearman_footrule_distance,
    windowed_kendall_max,
    windowed_kendall_tau_distance,
)
from scanpath_similarity import multimatch, scanmatch
from trajectory_distances import dynamic_time_warping, edr_distance, lcss_distance

ORDER_METRICS = ["kendall_tau", "windowed_kendall_tau", "spearman_footrule"]
TRAJECTORY_METRICS = ["dtw", "lcss", "edr", "scanmatch"]
MULTIMATCH_METRICS = ["mm_vector", "mm_length", "mm_direction", "mm_position", "multimatch"]

METRIC_GROUPS = [
    ("Order-based (permutation)", ORDER_METRICS),
    ("Trajectory / alignment", TRAJECTORY_METRICS),
    ("MultiMatch (1 - similarity)", MULTIMATCH_METRICS),
]
ALL_METRICS = [m for _, group in METRIC_GROUPS for m in group]

_UNIT_SQUARE_DIAGONAL = 2**0.5


def evaluate_pair(coords, perm, window_fracs=(0.1,), eps_values=(0.05,), grid_sizes=(5,)):
    """`coords`: `(n, 2)` ground-truth scanpath, in its true temporal order. `perm`: a
    permutation of `0..n-1` -- the compared scanpath is `coords[perm]` (a synthetically
    damaged copy, or a model's predicted ordering of the same fixations).

    Returns `{metric_name: {param_label: value}}` over `ALL_METRICS`, every value in
    `[0, 1]`, 0 = identical order. Metrics with no free parameter get a single
    `"default"` entry; `windowed_kendall_tau` gets one entry per `window_fracs` value
    (window = `round(w * n)` GT positions), `lcss`/`edr` one per `eps_values`, and
    `scanmatch` one per `grid_sizes` (an `n x n` AOI grid). MultiMatch dimensions are
    NaN for scanpaths too short (< 2 fixations) to have a saccade."""
    n = coords.shape[0]
    identity = np.arange(n)
    damaged = coords[perm]

    max_kendall = n * (n - 1) / 2
    max_footrule = (n * n) // 2

    values = {
        "kendall_tau": {
            "default": kendall_tau_distance(identity, perm) / max_kendall if max_kendall > 0 else 0.0
        },
        "spearman_footrule": {
            "default": spearman_footrule_distance(identity, perm) / max_footrule if max_footrule > 0 else 0.0
        },
        "dtw": {"default": dynamic_time_warping(coords, damaged, normalize=True) / _UNIT_SQUARE_DIAGONAL},
    }

    values["windowed_kendall_tau"] = {}
    for w in window_fracs:
        window = int(round(w * n))
        denom = windowed_kendall_max(n, window)
        d = windowed_kendall_tau_distance(identity, perm, window=window)
        values["windowed_kendall_tau"][f"w={w:g}"] = d / denom if denom > 0 else 0.0

    values["lcss"] = {f"eps={eps:g}": lcss_distance(coords, damaged, eps=eps) for eps in eps_values}
    values["edr"] = {f"eps={eps:g}": edr_distance(coords, damaged, eps=eps, normalize=True) for eps in eps_values}
    values["scanmatch"] = {
        f"grid={g}x{g}": 1.0 - scanmatch(coords, damaged, grid_shape=(g, g)) for g in grid_sizes
    }

    mm = multimatch(coords, damaged)
    for dim in ("vector", "length", "direction", "position", "multimatch"):
        key = dim if dim == "multimatch" else f"mm_{dim}"
        values[key] = {"default": 1.0 - mm[dim] if mm[dim] is not None else float("nan")}

    return values


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 1.0, size=(30, 2))
    n = coords.shape[0]
    kwargs = dict(window_fracs=(0.0, 0.1, 0.2), eps_values=(0.025, 0.05, 0.1), grid_sizes=(3, 5, 8))

    same = evaluate_pair(coords, np.arange(n), **kwargs)
    for metric, by_param in same.items():
        for label, v in by_param.items():
            assert abs(v) < 1e-9, f"{metric} ({label}) on identical order: expected 0, got {v}"
    print("Identity permutation: every metric is 0.")

    for name, perm in [("reversed", np.arange(n)[::-1]), ("shuffled", rng.permutation(n))]:
        vals = evaluate_pair(coords, perm, **kwargs)
        for metric, by_param in vals.items():
            for label, v in by_param.items():
                assert 0.0 <= v <= 1.0, f"{metric} ({label}) out of [0, 1]: {v}"
        print(f"{name}: " + ", ".join(f"{m}={next(iter(p.values())):.3f}" for m, p in vals.items()))
        if name == "reversed":
            assert vals["kendall_tau"]["default"] == 1.0
            assert all(abs(v - 1.0) < 1e-12 for v in vals["windowed_kendall_tau"].values())
    print("All values within [0, 1]; reversal maxes out (windowed) Kendall tau.")
