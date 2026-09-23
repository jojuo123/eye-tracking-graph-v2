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


def evaluate_pair(
    coords,
    perm,
    eps_values=(0.05,),
    crqa_radius_values=(0.05,),
    grid_sizes=(5,),
    erp_gap_points=((0.0, 0.0),),
):
    """`coords`: `(n, 2)` original scanpath. `perm`: a permutation of `0..n-1` (see
    `perturbations.py`) -- the damaged scanpath is `coords[perm]`. Returns
    `{metric_name: {param_label: value}}` covering every metric in `ALL_METRICS`:
    metrics with no free parameter get a single `"default"` entry; metrics that take
    one (`lcss`/`edr`'s `eps`, the CRQA family's `radius`, `string_edit_distance`/
    `scanmatch`'s AOI `grid_shape`, `erp`'s `gap_point`) get one entry per value in
    the corresponding `*_values`/`grid_sizes`/`erp_gap_points` argument, so callers
    can see both severity-sensitivity and parameter-sensitivity at once. Count-based
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
        "hamming": {"default": hamming_distance(identity, perm) / n},
        "kendall_tau": {
            "default": (kendall_tau_distance(identity, perm) / max_kendall) if max_kendall > 0 else 0.0
        },
        "spearman_footrule": {
            "default": (spearman_footrule_distance(identity, perm) / max_footrule) if max_footrule > 0 else 0.0
        },
        "cayley": {"default": (cayley_distance(identity, perm) / (n - 1)) if n > 1 else 0.0},
        "ulam": {"default": (ulam_distance(identity, perm) / (n - 1)) if n > 1 else 0.0},
        "dtw": {"default": dynamic_time_warping(coords, damaged, normalize=True)},
        "discrete_frechet": {"default": discrete_frechet_distance(coords, damaged)},
        "hausdorff": {"default": hausdorff_distance(coords, damaged)},
    }

    values["string_edit_distance"] = {
        f"grid={g}x{g}": string_edit_distance(coords, damaged, grid_shape=(g, g), normalize=True)
        for g in grid_sizes
    }
    values["scanmatch"] = {f"grid={g}x{g}": scanmatch(coords, damaged, grid_shape=(g, g)) for g in grid_sizes}
    values["edr"] = {f"eps={eps:g}": edr_distance(coords, damaged, eps=eps, normalize=True) for eps in eps_values}
    values["lcss"] = {f"eps={eps:g}": lcss_distance(coords, damaged, eps=eps) for eps in eps_values}
    values["erp"] = {
        f"gap=({gp[0]:g},{gp[1]:g})": (erp_distance(coords, damaged, gap_point=gp) / n if n > 0 else 0.0)
        for gp in erp_gap_points
    }

    mm = multimatch(coords, damaged)
    for dim in MULTIMATCH_DIMS:
        values[dim] = {"default": mm[dim] if mm[dim] is not None else float("nan")}

    for radius in crqa_radius_values:
        crqa = cross_recurrence_analysis(coords, damaged, radius=radius)
        label = f"radius={radius:g}"
        for key in CRQA_METRICS:
            values.setdefault(key, {})[label] = crqa[key] if crqa[key] is not None else float("nan")

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


def run_experiment(
    sequences,
    levels,
    n_repeats,
    seed=0,
    eps_values=(0.05,),
    crqa_radius_values=(0.05,),
    grid_sizes=(5,),
    erp_gap_points=((0.0, 0.0),),
):
    """For every (perturbation type, severity level), draws `n_repeats` random
    damage realizations per sample (this is the "good amount of artificial
    orderings per sample" the averaging relies on -- a single realization at a
    given level is a noisy draw, e.g. which adjacent pairs `local_swaps` happens to
    pick; several repeats per sample, pooled with all other samples, is what turns
    that into a stable curve) and evaluates every metric -- at every swept parameter
    value, for metrics that take one (see `evaluate_pair`) -- comparing each damaged
    scanpath back to its own original.

    Returns
    `{perturbation_name: {metric_name: {param_label: [(level, mean, std, n_samples), ...]}}}`.
    `swap_nearby_locations` silently skips (sample, repeat) draws where the sample
    has no eligible near-revisit pair at that draw's search radius
    (`perturbations.swap_nearby_locations` returns the identity permutation in that
    case) -- counting those as "zero damage" would understate the metrics' true
    sensitivity by diluting the average with samples the perturbation couldn't even
    apply to.
    """
    rng = np.random.default_rng(seed)
    results = {name: {m: {} for m in ALL_METRICS} for name in PERTURBATIONS}

    for pert_name, pert_fn in PERTURBATIONS.items():
        needs_coords = pert_name in NEEDS_COORDS
        for level in levels:
            per_metric_param_samples = {m: {} for m in ALL_METRICS}
            for coords in sequences.values():
                n = coords.shape[0]
                for _ in range(n_repeats):
                    perm = pert_fn(n, level, rng, coords=coords if needs_coords else None)
                    if pert_name == "swap_nearby_locations" and level > 0 and np.array_equal(perm, np.arange(n)):
                        continue
                    vals = evaluate_pair(
                        coords,
                        perm,
                        eps_values=eps_values,
                        crqa_radius_values=crqa_radius_values,
                        grid_sizes=grid_sizes,
                        erp_gap_points=erp_gap_points,
                    )
                    for m in ALL_METRICS:
                        for label, v in vals[m].items():
                            if not (isinstance(v, float) and np.isnan(v)):
                                per_metric_param_samples[m].setdefault(label, []).append(v)
            for m in ALL_METRICS:
                for label, samples in per_metric_param_samples[m].items():
                    mean = float(np.mean(samples)) if samples else float("nan")
                    std = float(np.std(samples)) if samples else float("nan")
                    results[pert_name][m].setdefault(label, []).append([level, mean, std, len(samples)])
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

# One metric = one color (fixed across its param sweep, so the eye groups by metric
# first); one parameter value = one line style, in sweep order, so the eye can then
# read severity within a metric.
LINESTYLES = ["-", "--", ":", "-."]


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
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    fig.patch.set_facecolor(PAGE)
    for ax, (group_title, metric_names) in zip(axes.ravel(), METRIC_GROUPS):
        _style_axes(ax)
        for color, metric in zip(PALETTE, metric_names):
            param_series = pert_results[metric]  # {param_label: [(level, mean, std, n), ...]}
            single_default = list(param_series.keys()) == ["default"]
            for i, (label, series) in enumerate(param_series.items()):
                xs = [row[0] for row in series]
                ys = [row[1] for row in series]
                stds = [row[2] for row in series]
                linestyle = LINESTYLES[i % len(LINESTYLES)]
                legend_label = metric if single_default else f"{metric} ({label})"
                ax.plot(
                    xs, ys, color=color, linewidth=1.6, marker="o", markersize=3.5,
                    linestyle=linestyle, label=legend_label,
                )
                lo = [y - s for y, s in zip(ys, stds)]
                hi = [y + s for y, s in zip(ys, stds)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.set_title(group_title, fontsize=9.5, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("severity level", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(
            fontsize=6, frameon=False, labelcolor=TEXT_SECONDARY,
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
        )
    fig.suptitle(f"Damage type: {pert_name}", fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, f"{pert_name}.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
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
    parser.add_argument(
        "--crqa-radius-values", type=float, nargs="+", default=[0.03, 0.05, 0.08],
        help="cross_recurrence_analysis's `radius`, swept independently of severity",
    )
    parser.add_argument(
        "--eps-values", type=float, nargs="+", default=[0.025, 0.05, 0.1],
        help="lcss/edr's spatial match threshold `eps`, swept independently of severity",
    )
    parser.add_argument(
        "--grid-sizes", type=int, nargs="+", default=[3, 5, 8],
        help="string_edit_distance/scanmatch's AOI grid resolution (n -> an n x n grid)",
    )
    parser.add_argument(
        "--erp-gap-points", type=float, nargs="+", default=[0.0, 0.0, 0.5, 0.5],
        help="erp's reference gap point(s), as flat x,y pairs, e.g. `0 0 0.5 0.5` for two points",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "work_dir", "metric_damage_curves"),
    )
    args = parser.parse_args()

    if len(args.erp_gap_points) % 2 != 0:
        parser.error("--erp-gap-points must be a flat list of x,y pairs (even count)")
    erp_gap_points = [
        (args.erp_gap_points[i], args.erp_gap_points[i + 1]) for i in range(0, len(args.erp_gap_points), 2)
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = load_local_reflacx_sequences(
        args.h5_path, args.split, args.dataset_name, max_fixations=args.max_fixations
    )
    lengths = {sid: xy.shape[0] for sid, xy in sequences.items()}
    print(f"Loaded {len(sequences)} scanpaths from {args.h5_path}: lengths {lengths}")

    results = run_experiment(
        sequences,
        args.levels,
        args.n_repeats,
        seed=args.seed,
        eps_values=args.eps_values,
        crqa_radius_values=args.crqa_radius_values,
        grid_sizes=args.grid_sizes,
        erp_gap_points=erp_gap_points,
    )

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    for pert_name, pert_results in results.items():
        out_path = plot_perturbation(pert_name, pert_results, args.output_dir)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
