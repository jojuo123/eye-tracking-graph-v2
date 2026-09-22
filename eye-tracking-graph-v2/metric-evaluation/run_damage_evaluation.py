"""Sensitivity analysis: for every synthetic damage type in `perturbations.py`, at a
range of severity levels, damage real REFLACX scanpaths and measure how much every
metric in `permutation_distances.py` / `scanpath_similarity.py` /
`trajectory_distances.py` moves -- producing one figure per damage type, each a grid
of small multiples (one subplot per metric family, since the families live on
different scales) plotting metric value vs. severity level.

This is purely a diagnostic/exploratory script (which metrics are sensitive to which
kind of scanpath-order error, and which are blind to it) -- it doesn't change any of
the three metric modules.

Requires `h5py` and `matplotlib`, unlike the rest of this directory (which is
numpy-only): both are optional extras for this one script, not full project
dependencies.

Usage: `python3 run_damage_evaluation.py` (uses this repo's local
`reflacx_data.h5` smoke-test file, 12 samples, by default -- see `--h5-path` to
point at a larger H5 file, e.g. once ERDA is mounted).
"""

import argparse
import json
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perturbations import NEEDS_COORDS, PERTURBATIONS
from permutation_distances import (
    cayley_distance,
    hamming_distance,
    kendall_tau_distance,
    spearman_footrule_distance,
    ulam_distance,
)
from scanpath_similarity import cross_recurrence_analysis, multimatch, scanmatch, string_edit_distance
from trajectory_distances import (
    discrete_frechet_distance,
    dynamic_time_warping,
    edr_distance,
    erp_distance,
    hausdorff_distance,
    lcss_distance,
)

# ---------------------------------------------------------------------------
# Metric families -- grouped by scale, since a single axis can't honestly show a
# [0, 1]-bounded score next to a raw spatial distance (see dataviz "one axis" rule).
# ---------------------------------------------------------------------------

PERM_METRICS = ["hamming", "kendall_tau", "spearman_footrule", "cayley", "ulam"]
EDIT_METRICS = ["string_edit_distance", "scanmatch", "edr"]
MULTIMATCH_DIMS = ["vector", "length", "direction", "position", "multimatch"]
CRQA_METRICS = ["rec", "det", "lam", "corm"]
TRAJ_BOUNDED = ["dtw", "lcss"]
TRAJ_SPATIAL = ["discrete_frechet", "hausdorff", "erp"]

METRIC_GROUPS = [
    ("Permutation distances (fraction of max)", PERM_METRICS),
    ("Edit-based similarity (0-1)", EDIT_METRICS),
    ("MultiMatch dimensions (similarity, 0-1)", MULTIMATCH_DIMS),
    ("Cross-recurrence analysis (%)", CRQA_METRICS),
    ("Trajectory distances, bounded (0-1)", TRAJ_BOUNDED),
    ("Trajectory distances, spatial units", TRAJ_SPATIAL),
]
ALL_METRICS = [m for _, group in METRIC_GROUPS for m in group]


def evaluate_pair(coords, perm, crqa_radius=0.05):
    """`coords`: `(n, 2)` original scanpath. `perm`: a permutation of `0..n-1` (see
    `perturbations.py`) -- the damaged scanpath is `coords[perm]`. Returns a flat
    `{metric_name: value}` dict covering every metric in `ALL_METRICS`. Count-based
    permutation distances are normalized by their maximum possible value (so they're
    comparable across scanpaths of different length); `dtw`/`string_edit_distance`/
    `edr` use those functions' own `normalize=True`; `erp` is divided by `n` (average
    per-point cost) to bring it onto roughly the same scale as `discrete_frechet`/
    `hausdorff` (otherwise it grows with `n` even for a fixed *fraction* of the path
    disturbed, since more points means more accumulated edit cost)."""
    n = coords.shape[0]
    identity = np.arange(n)
    damaged = coords[perm]

    max_kendall = n * (n - 1) / 2
    max_footrule = (n * n) // 2

    values = {
        "hamming": hamming_distance(identity, perm) / n,
        "kendall_tau": (kendall_tau_distance(identity, perm) / max_kendall) if max_kendall > 0 else 0.0,
        "spearman_footrule": (spearman_footrule_distance(identity, perm) / max_footrule) if max_footrule > 0 else 0.0,
        "cayley": (cayley_distance(identity, perm) / (n - 1)) if n > 1 else 0.0,
        "ulam": (ulam_distance(identity, perm) / (n - 1)) if n > 1 else 0.0,
        "string_edit_distance": string_edit_distance(coords, damaged, normalize=True),
        "scanmatch": scanmatch(coords, damaged),
        "edr": edr_distance(coords, damaged, normalize=True),
        "dtw": dynamic_time_warping(coords, damaged, normalize=True),
        "lcss": lcss_distance(coords, damaged),
        "discrete_frechet": discrete_frechet_distance(coords, damaged),
        "hausdorff": hausdorff_distance(coords, damaged),
        "erp": erp_distance(coords, damaged) / n if n > 0 else 0.0,
    }

    mm = multimatch(coords, damaged)
    for dim in MULTIMATCH_DIMS:
        values[dim] = mm[dim] if mm[dim] is not None else float("nan")

    crqa = cross_recurrence_analysis(coords, damaged, radius=crqa_radius)
    for key in CRQA_METRICS:
        values[key] = crqa[key] if crqa[key] is not None else float("nan")

    return values


def load_local_reflacx_sequences(
    h5_path, split="TRAIN", dataset_name="reflacx", x_col="x_position_norm", y_col="y_position_norm", max_fixations=100
):
    """Loads every sample's fixation `(x, y)` sequence from the REFLACX H5 file (same
    layout/columns `visualize_fixation_order_heatmap.py` reads), normalized `[0, 1]`.
    `max_fixations`: sequences longer than this are truncated to their first
    `max_fixations` fixations, to bound the runtime of this script's O(n^2)-ish
    metrics (several samples run into the hundreds of fixations)."""
    sequences = {}
    with h5py.File(h5_path, "r") as f:
        group = f[split][dataset_name]
        for sample_id in sorted(group.keys()):
            fx = group[sample_id]["fixations"]
            xy = np.stack([fx[x_col][()], fx[y_col][()]], axis=-1).astype(np.float64)
            xy = np.clip(xy, 0.0, 1.0)
            if max_fixations is not None and xy.shape[0] > max_fixations:
                xy = xy[:max_fixations]
            sequences[sample_id] = xy
    return sequences


def run_experiment(sequences, levels, n_repeats, seed=0, crqa_radius=0.05):
    """For every (perturbation type, severity level), draws `n_repeats` random
    damage realizations per sample (this is the "good amount of artificial
    orderings per sample" the averaging relies on -- a single realization at a
    given level is a noisy draw, e.g. which adjacent pairs `local_swaps` happens to
    pick; several repeats per sample, pooled with all other samples, is what turns
    that into a stable curve) and evaluates every metric, comparing each damaged
    scanpath back to its own original.

    Returns `{perturbation_name: {metric_name: [(level, mean, std, n_samples), ...]}}`.
    `swap_nearby_locations` silently skips (sample, repeat) draws where the sample
    has no eligible near-revisit pair at the given `crqa_radius`-scaled search radius
    (`perturbations.swap_nearby_locations` returns the identity permutation in that
    case) -- counting those as "zero damage" would understate the metrics' true
    sensitivity by diluting the average with samples the perturbation couldn't even
    apply to.
    """
    rng = np.random.default_rng(seed)
    results = {name: {m: [] for m in ALL_METRICS} for name in PERTURBATIONS}

    for pert_name, pert_fn in PERTURBATIONS.items():
        needs_coords = pert_name in NEEDS_COORDS
        for level in levels:
            per_metric_samples = {m: [] for m in ALL_METRICS}
            for coords in sequences.values():
                n = coords.shape[0]
                for _ in range(n_repeats):
                    perm = pert_fn(n, level, rng, coords=coords if needs_coords else None)
                    if pert_name == "swap_nearby_locations" and level > 0 and np.array_equal(perm, np.arange(n)):
                        continue
                    vals = evaluate_pair(coords, perm, crqa_radius=crqa_radius)
                    for m in ALL_METRICS:
                        v = vals[m]
                        if not (isinstance(v, float) and np.isnan(v)):
                            per_metric_samples[m].append(v)
            for m in ALL_METRICS:
                samples = per_metric_samples[m]
                mean = float(np.mean(samples)) if samples else float("nan")
                std = float(np.std(samples)) if samples else float("nan")
                results[pert_name][m].append([level, mean, std, len(samples)])
    return results


# ---------------------------------------------------------------------------
# Plotting -- validated categorical palette (dataviz skill, references/palette.md):
# fixed hue order, never cycled/reassigned per-chart; each metric group uses a fixed
# prefix of the 8-slot order (so within any subplot, only adjacent-in-the-validated-
# order pairs appear together).
# ---------------------------------------------------------------------------

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
GRID, AXIS_LINE = "#e1e0d9", "#c3c2b7"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#898781"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)


def plot_perturbation(pert_name, pert_results, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.patch.set_facecolor(PAGE)
    for ax, (group_title, metric_names) in zip(axes.ravel(), METRIC_GROUPS):
        _style_axes(ax)
        for color, metric in zip(PALETTE, metric_names):
            series = pert_results[metric]
            xs = [row[0] for row in series]
            ys = [row[1] for row in series]
            stds = [row[2] for row in series]
            ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=4.5, label=metric)
            lo = [y - s for y, s in zip(ys, stds)]
            hi = [y + s for y, s in zip(ys, stds)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.set_title(group_title, fontsize=9.5, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("severity level", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_SECONDARY, loc="best")
    fig.suptitle(f"Damage type: {pert_name}", fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, f"{pert_name}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--h5-path", default=os.path.join(os.path.dirname(__file__), "..", "..", "reflacx_data.h5")
    )
    parser.add_argument("--split", default="TRAIN")
    parser.add_argument("--dataset-name", default="reflacx")
    parser.add_argument("--max-fixations", type=int, default=100)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--n-repeats", type=int, default=5, help="random damage realizations per sample per level")
    parser.add_argument("--crqa-radius", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "work_dir", "metric_damage_curves"),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = load_local_reflacx_sequences(
        args.h5_path, args.split, args.dataset_name, max_fixations=args.max_fixations
    )
    lengths = {sid: xy.shape[0] for sid, xy in sequences.items()}
    print(f"Loaded {len(sequences)} scanpaths from {args.h5_path}: lengths {lengths}")

    results = run_experiment(
        sequences, args.levels, args.n_repeats, seed=args.seed, crqa_radius=args.crqa_radius
    )

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    for pert_name, pert_results in results.items():
        out_path = plot_perturbation(pert_name, pert_results, args.output_dir)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
