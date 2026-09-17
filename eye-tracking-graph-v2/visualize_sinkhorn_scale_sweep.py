"""Visualizes how `data.reflacx.reflacx.compute_soft_ground_truth`'s Sinkhorn-normalized
matrix changes purely as a function of `soft_ground_truth.scale`
(`configs/fixation_permutation_sorter/reflacx_data.yaml`), holding `tau` fixed.

`load_fixations` bakes `scale` into a sample's fixation coordinates *before* the pairwise
distance map is computed (see `visualize_permutation.py`'s and `visualize_soft_gt_scale.py`'s
docstrings), and Euclidean distance is homogeneous of degree 1, so scaling coordinates by `s`
scales every pairwise distance by exactly `s` too -- `distance(coords * s) == s * distance
(coords)`. That means the entire effect of `scale` on the Sinkhorn recipe (`-distance / tau`)
collapses to a single knob: it's equivalent to leaving coordinates alone and instead dividing
`tau` by `s`. At `s=1` (or small `s`), `[0, 1]`-normalized pairwise distances are tiny, so
`-distance / tau` barely varies between fixation pairs and Sinkhorn normalizes to something
close to uniform (every entry near `1/n`, i.e. a "soft" target that barely distinguishes
fixations at all). As `s` grows, the score spread widens and Sinkhorn sharpens smoothly
towards the hard identity permutation. Neither extreme is a bug -- `scale` is precisely the
knob for choosing where on that spectrum the soft ground truth sits.

This is a read-only diagnostic (like `visualize_soft_gt_scale.py`): for up to `--num-samples`
samples per split with `n_fixations < --max-length`, it recovers each sample's true `[0, 1]`
fixation coordinates (dividing the H5's already-scaled columns back out by
`data.reflacx.reflacx.SOFT_GROUND_TRUTH_SCALE`) and, from there, sweeps freely over candidate
`scale` values independent of whatever is currently baked into the H5 file. Each sample gets
one figure with:
  1. a row of Sinkhorn heatmaps at a handful of representative `--scales`;
  2. a summary curve -- diagonal mean and mean largest off-diagonal entry vs. `scale`, over a
     much finer `--sweep-min`/`--sweep-max`/`--sweep-points` log-spaced range -- with the
     currently configured `soft_ground_truth.scale` marked, so the heatmap row's snapshots can
     be located on the full transition curve.

    cd eye-tracking-graph-v2
    python visualize_sinkhorn_scale_sweep.py --max-length 30 --num-samples 5
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.reflacx import reflacx
from dataloaders import build_dataset
from visualize_permutation import annotate_heatmap, pairwise_distance_map, sinkhorn_normalize
from visualize_soft_gt_scale import select_sample_ids

SPLITS = ("TRAIN", "VAL", "TEST")
COLUMNS = tuple(reflacx.SOFT_GROUND_TRUTH_COLUMNS)
DEFAULT_TAU = reflacx.SOFT_GROUND_TRUTH_TAU
DEFAULT_SINKHORN_ITERS = reflacx.SOFT_GROUND_TRUTH_N_ITERS
# What `load_fixations` already baked into the H5 file's stored coordinates -- divided back
# out below to recover true [0, 1] coordinates as the sweep's starting point.
DEFAULT_STORED_SCALE = reflacx.SOFT_GROUND_TRUTH_SCALE
# Representative checkpoints for the heatmap row, chosen to bracket the configured default
# (`DEFAULT_STORED_SCALE`, included exactly) on both sides.
DEFAULT_SCALES = (1.0, 5.0, 20.0, 60.0, DEFAULT_STORED_SCALE, 600.0, 2000.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-length", type=int, default=20, help="Only consider samples with n_fixations < this")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples to visualize per split")
    parser.add_argument("--splits", type=str, nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Sinkhorn temperature, held fixed across the scale sweep")
    parser.add_argument("--sinkhorn-iters", type=int, default=DEFAULT_SINKHORN_ITERS)
    parser.add_argument(
        "--stored-scale",
        type=float,
        default=DEFAULT_STORED_SCALE,
        help="Factor already baked into the H5 file's stored coordinates -- divided back out to recover "
        "true [0, 1] coordinates, which is what this script actually sweeps `--scales` over.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="Scale values to render as individual Sinkhorn heatmaps",
    )
    parser.add_argument("--sweep-min", type=float, default=1.0, help="Smallest scale in the summary curve's log-spaced sweep")
    parser.add_argument("--sweep-max", type=float, default=3000.0, help="Largest scale in the summary curve's log-spaced sweep")
    parser.add_argument("--sweep-points", type=int, default=60, help="Number of scale values in the summary curve's sweep")
    parser.add_argument(
        "--seed", type=int, default=0, help="Selects which samples are visualized when a split has more than --num-samples eligible ones"
    )
    parser.add_argument("--output-dir", type=str, default="work_dir/sinkhorn_scale_sweep_visualizations")
    return parser.parse_args()


def sinkhorn_at_scale(distance_true, scale, tau, n_iters):
    """`distance_true`: `(n, n)` pairwise distance over true `[0, 1]` coordinates.
    Euclidean distance is homogeneous of degree 1, so `distance(coords * scale) == scale *
    distance(coords)` -- multiplying the distance map directly is equivalent to (and much
    cheaper than) re-deriving it from rescaled coordinates every time, exactly matching
    `data.reflacx.reflacx.load_fixations`/`compute_soft_ground_truth`'s actual recipe of
    scaling coordinates before computing distances."""
    return sinkhorn_normalize(distance_true * scale, tau, n_iters)


def summary_curve(distance_true, scales, tau, n_iters):
    """For each `scale` in `scales`, returns `(diag_mean, mean_max_offdiag)` of the resulting
    Sinkhorn matrix: `diag_mean` (avg. probability a fixation is matched to itself) rises
    from `~1/n` towards `1` as `scale` grows, while `mean_max_offdiag` (avg., per row, of the
    largest probability assigned to any *other* fixation) falls the opposite way -- together
    they trace the uniform-to-hard-permutation transition `scale` controls."""
    diag_means = np.empty(len(scales))
    mean_max_offdiags = np.empty(len(scales))
    n = distance_true.shape[0]
    eye = np.eye(n, dtype=bool)
    for i, scale in enumerate(scales):
        matrix = sinkhorn_at_scale(distance_true, scale, tau, n_iters)
        diag_means[i] = np.diagonal(matrix).mean()
        off_diag = np.where(eye, -np.inf, matrix)
        mean_max_offdiags[i] = off_diag.max(axis=1).mean()
    return diag_means, mean_max_offdiags


def visualize_sample(dataset, sample_index, sample_id, tau, n_iters, stored_scale, scales, sweep_scales, out_path):
    item = dataset[sample_index]
    coords_scaled = item["fixations"].numpy()  # (n, 2) = COLUMNS, as stored in the H5 file
    coords_true = coords_scaled / stored_scale  # recovers true [0, 1] normalized coordinates
    distance_true = pairwise_distance_map(coords_true)
    n = coords_true.shape[0]

    fig = plt.figure(figsize=(4 * len(scales), 9))
    grid = fig.add_gridspec(2, len(scales), height_ratios=[1, 0.8])

    for col, scale in enumerate(scales):
        ax = fig.add_subplot(grid[0, col])
        matrix = sinkhorn_at_scale(distance_true, scale, tau, n_iters)
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=matrix.max())
        marker = " (configured)" if np.isclose(scale, stored_scale) else ""
        ax.set_title(f"scale={scale:g}{marker}\ndiag mean={np.diagonal(matrix).mean():.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        annotate_heatmap(ax, matrix, im)

    diag_means, mean_max_offdiags = summary_curve(distance_true, sweep_scales, tau, n_iters)
    ax_curve = fig.add_subplot(grid[1, :])
    ax_curve.plot(sweep_scales, diag_means, marker="o", markersize=3, label="diagonal mean (self-match probability)")
    ax_curve.plot(sweep_scales, mean_max_offdiags, marker="o", markersize=3, label="mean largest off-diagonal entry")
    ax_curve.axhline(1.0 / n, color="gray", linestyle=":", linewidth=1, label=f"uniform (1/n = {1.0 / n:.3f})")
    ax_curve.axvline(stored_scale, color="red", linestyle="--", linewidth=1, label=f"configured scale = {stored_scale:g}")
    for scale in scales:
        ax_curve.axvline(scale, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("scale (log scale)")
    ax_curve.set_ylabel("probability")
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.set_title(f"Sinkhorn transition vs. scale (tau={tau:g}, n_iters={n_iters})")
    ax_curve.legend(loc="center right", fontsize=8)

    fig.suptitle(f"{sample_id}  (n={n} fixations)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    sweep_scales = np.logspace(np.log10(args.sweep_min), np.log10(args.sweep_max), args.sweep_points)

    for split in args.splits:
        sample_ids = select_sample_ids(
            reflacx.H5_FILE, split, reflacx.DATASET_NAME, args.max_length, args.num_samples, args.seed
        )
        if not sample_ids:
            print(f"No samples with n_fixations < {args.max_length} found in split {split!r}, skipping")
            continue

        dataset_kwargs = reflacx.dataset_kwargs(split, fixation_columns=list(COLUMNS), load_soft_permutation=False)
        dataset = build_dataset({"type": "single_h5", **dataset_kwargs})
        id_to_index = {sid: i for i, sid in enumerate(dataset.sample_ids)}
        try:
            for sample_id in sample_ids:
                out_path = os.path.join(args.output_dir, split.lower(), f"{sample_id}.png")
                visualize_sample(
                    dataset,
                    id_to_index[sample_id],
                    sample_id,
                    args.tau,
                    args.sinkhorn_iters,
                    args.stored_scale,
                    args.scales,
                    sweep_scales,
                    out_path,
                )
                print(f"Wrote {out_path}")
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
