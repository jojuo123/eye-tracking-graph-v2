"""Sensitivity analysis: for every synthetic damage type in `perturbations.py`, at a
range of severity levels, damage real REFLACX scanpaths and measure how much each
shortlisted metric in `normalized_metrics.py` moves -- producing one figure per damage
type, one subplot per metric family, plotting metric value vs. severity level. Every
metric is a `[0, 1]` dissimilarity there, so all subplots share one y-axis.

This is purely a diagnostic/exploratory script (which metrics are sensitive to which
kind of scanpath-order error, and which are blind to it) -- it doesn't change any of
the metric modules.

Requires `h5py` and `matplotlib`, unlike the rest of this directory (which is
numpy-only): both are optional extras for this one script, not full project
dependencies.

Usage: `python3 run_damage_evaluation.py` (uses this repo's local
`reflacx_data.h5` smoke-test file, 12 samples, by default -- see `--h5-path` to
point at a larger H5 file, e.g. once ERDA is mounted).
"""

import argparse
import json
import multiprocessing
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from normalized_metrics import ALL_METRICS, METRIC_GROUPS, evaluate_pair
from perturbations import NEEDS_COORDS, PERTURBATIONS


def load_local_reflacx_sequences(
    h5_path,
    split="TRAIN",
    dataset_name="reflacx",
    x_col="x_position_norm",
    y_col="y_position_norm",
    max_fixations=100,
    max_samples=None,
    seed=0,
    coord_scale=1.0,
):
    """Loads every sample's fixation `(x, y)` sequence from the REFLACX H5 file (same
    layout/columns `visualize_fixation_order_heatmap.py` reads), normalized `[0, 1]`.
    `max_fixations`: sequences longer than this are truncated to their first
    `max_fixations` fixations, to bound the runtime of this script's O(n^2)-ish
    metrics (several samples run into the hundreds of fixations). `max_samples`: if set,
    loads only a `seed`-fixed random subset of that many samples (the full ERDA splits
    have thousands, far more than this pure-Python sweep needs for stable means).
    `coord_scale`: coordinates are divided by this before clipping, for H5 files that
    store `*_norm` columns pre-multiplied (e.g. 20 for `normalized_data_softGT_scaled20.h5`)
    -- the metrics' bounds and `eps`/grid defaults all assume `[0, 1]`."""
    sequences = {}
    with h5py.File(h5_path, "r") as f:
        group = f[split][dataset_name]
        sample_ids = sorted(group.keys())
        if max_samples is not None and len(sample_ids) > max_samples:
            rng = np.random.default_rng(seed)
            sample_ids = sorted(rng.choice(sample_ids, size=max_samples, replace=False).tolist())
        for sample_id in sample_ids:
            fx = group[sample_id]["fixations"]
            xy = np.stack([fx[x_col][()], fx[y_col][()]], axis=-1).astype(np.float64) / coord_scale
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
    window_fracs=(0.1,),
    eps_values=(0.05,),
    grid_sizes=(5,),
    workers=1,
):
    """For every (perturbation type, severity level), draws `n_repeats` random
    damage realizations per sample (this is the "good amount of artificial
    orderings per sample" the averaging relies on -- a single realization at a
    given level is a noisy draw, e.g. which adjacent pairs `local_swaps` happens to
    pick; several repeats per sample, pooled with all other samples, is what turns
    that into a stable curve) and evaluates every metric -- at every swept parameter
    value, for metrics that take one (see `normalized_metrics.evaluate_pair`) -- comparing each damaged
    scanpath back to its own original.

    Returns
    `{perturbation_name: {metric_name: {param_label: [(level, mean, std, n_samples), ...]}}}`.
    `swap_nearby_locations` silently skips (sample, repeat) draws where the sample
    has no eligible near-revisit pair at that draw's search radius
    (`perturbations.swap_nearby_locations` returns the identity permutation in that
    case) -- counting those as "zero damage" would understate the metrics' true
    sensitivity by diluting the average with samples the perturbation couldn't even
    apply to.

    `workers`: number of processes the per-sample work is spread over.
    """
    tasks = [
        (i, coords, levels, n_repeats, seed, window_fracs, eps_values, grid_sizes)
        for i, coords in enumerate(sequences.values())
    ]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            per_sample = pool.map(_damage_one_sample, tasks, chunksize=1)
    else:
        per_sample = [_damage_one_sample(t) for t in tasks]

    results = {name: {m: {} for m in ALL_METRICS} for name in PERTURBATIONS}
    for pert_name in PERTURBATIONS:
        for li, level in enumerate(levels):
            for m in ALL_METRICS:
                pooled = {}
                for sample_vals in per_sample:
                    for label, vs in sample_vals[pert_name][li][m].items():
                        pooled.setdefault(label, []).extend(vs)
                for label, samples in pooled.items():
                    mean = float(np.mean(samples)) if samples else float("nan")
                    std = float(np.std(samples)) if samples else float("nan")
                    results[pert_name][m].setdefault(label, []).append([level, mean, std, len(samples)])
    return results


def _damage_one_sample(task):
    """All (perturbation, level, repeat) draws for one sample -- the unit of work
    `run_experiment` farms out to worker processes. Each sample gets its own RNG
    stream (`[seed, sample_index]`), so results don't depend on `workers`. Returns
    `{perturbation_name: [ {metric: {param_label: [values...]}} per level ]}`."""
    sample_index, coords, levels, n_repeats, seed, window_fracs, eps_values, grid_sizes = task
    rng = np.random.default_rng([seed, sample_index])
    n = coords.shape[0]
    out = {}
    for pert_name, pert_fn in PERTURBATIONS.items():
        needs_coords = pert_name in NEEDS_COORDS
        out[pert_name] = []
        for level in levels:
            per_metric = {m: {} for m in ALL_METRICS}
            for _ in range(n_repeats):
                perm = pert_fn(n, level, rng, coords=coords if needs_coords else None)
                if pert_name == "swap_nearby_locations" and level > 0 and np.array_equal(perm, np.arange(n)):
                    continue
                vals = evaluate_pair(
                    coords, perm, window_fracs=window_fracs, eps_values=eps_values, grid_sizes=grid_sizes
                )
                for m in ALL_METRICS:
                    for label, v in vals[m].items():
                        if not (isinstance(v, float) and np.isnan(v)):
                            per_metric[m].setdefault(label, []).append(v)
            out[pert_name].append(per_metric)
    return out


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
    fig, axes = plt.subplots(1, len(METRIC_GROUPS), figsize=(19, 4.8), sharey=True)
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
        if ax is axes.ravel()[0]:
            ax.set_ylabel("dissimilarity (0 = identical order)", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
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
    parser.add_argument(
        "--max-fixations", type=int, default=100, help="truncate longer sequences to this many (0 = no truncation)"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="random subset of this many samples")
    parser.add_argument(
        "--coord-scale", type=float, default=1.0,
        help="divide stored coordinates by this to get [0, 1] (20 for *_scaled20.h5 files)",
    )
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--n-repeats", type=int, default=5, help="random damage realizations per sample per level")
    parser.add_argument(
        "--window-fracs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2],
        help="windowed_kendall_tau's warping window, as a fraction of n (0 = plain Kendall tau)",
    )
    parser.add_argument(
        "--eps-values", type=float, nargs="+", default=[0.025, 0.05, 0.1],
        help="lcss/edr's spatial match threshold `eps`, swept independently of severity",
    )
    parser.add_argument(
        "--grid-sizes", type=int, nargs="+", default=[3, 5, 8],
        help="scanmatch's AOI grid resolution (n -> an n x n grid)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="processes to spread samples over")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "work_dir", "metric_damage_curves"),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = load_local_reflacx_sequences(
        args.h5_path, args.split, args.dataset_name, max_fixations=args.max_fixations or None,
        max_samples=args.max_samples, seed=args.seed, coord_scale=args.coord_scale,
    )
    lengths = np.array([xy.shape[0] for xy in sequences.values()])
    print(f"Loaded {len(sequences)} scanpaths from {args.h5_path}:{args.split} "
          f"(length median {np.median(lengths):g}, min {lengths.min()}, max {lengths.max()})")

    results = run_experiment(
        sequences,
        args.levels,
        args.n_repeats,
        seed=args.seed,
        window_fracs=args.window_fracs,
        eps_values=args.eps_values,
        grid_sizes=args.grid_sizes,
        workers=args.workers,
    )

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "samples.json"), "w") as f:
        json.dump({"h5_path": args.h5_path, "split": args.split, "max_fixations": args.max_fixations,
                   "sample_ids": list(sequences.keys())}, f, indent=2)

    for pert_name, pert_results in results.items():
        out_path = plot_perturbation(pert_name, pert_results, args.output_dir)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
