"""Visualizes how `data.reflacx.reflacx.compute_soft_ground_truth`'s Sinkhorn-normalized
matrix changes as a function of `soft_ground_truth.tau`
(`configs/fixation_permutation_sorter/reflacx_data.yaml`).

`compute_soft_ground_truth` scores every fixation pair `(i, j)` by `-distance[i, j] / tau`
before running Sinkhorn. Over `[0, 1]`-normalized coordinates, pairwise distances are small
(at most `sqrt(2)`), so a `tau` of that same order barely perturbs the score away from
uniform, and Sinkhorn normalizes to something close to `1/n` everywhere -- a "soft" target
that barely distinguishes fixations at all. Shrinking `tau` widens the score spread and
sharpens Sinkhorn smoothly towards the hard identity permutation. (There used to be a
separate `scale` knob that multiplied the distance map instead of dividing `tau` -- since
`-distance / tau == -(distance * s) / (tau * s)` for any `s > 0`, that was exactly redundant
with dividing `tau` by `s`, so it's been removed in favor of just setting `tau` directly.)

This is a read-only diagnostic (like `visualize_soft_gt_scale.py`): for up to `--num-samples`
samples per split with `n_fixations < --max-length`, it computes the pairwise distance map
over each sample's `[0, 1]`-normalized fixation coordinates once, then sweeps `tau` over it.
Each sample gets one figure with:
  1. a row of Sinkhorn heatmaps at a handful of representative `--taus`;
  2. a summary curve -- diagonal mean and mean largest off-diagonal entry vs. `tau` (log
     x-axis), over a much finer `--sweep-min`/`--sweep-max`/`--sweep-points` range -- with the
     currently configured `soft_ground_truth.tau` marked, so the heatmap row's snapshots can
     be located on the full transition curve.

    cd eye-tracking-graph-v2
    python visualize_sinkhorn_tau_sweep.py --max-length 30 --num-samples 5
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
# Representative checkpoints for the heatmap row, chosen to bracket the configured default
# (`DEFAULT_TAU`, included exactly) on both sides -- ascending, since smaller tau sharpens.
DEFAULT_TAUS = tuple(sorted((0.0002, 0.001, 0.02, 0.1, 0.5, 2.0, DEFAULT_TAU)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-length", type=int, default=20, help="Only consider samples with n_fixations < this")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples to visualize per split")
    parser.add_argument("--splits", type=str, nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--sinkhorn-iters", type=int, default=DEFAULT_SINKHORN_ITERS)
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=list(DEFAULT_TAUS),
        help="Tau values to render as individual Sinkhorn heatmaps",
    )
    parser.add_argument("--sweep-min", type=float, default=1e-4, help="Smallest tau in the summary curve's log-spaced sweep")
    parser.add_argument("--sweep-max", type=float, default=3.0, help="Largest tau in the summary curve's log-spaced sweep")
    parser.add_argument("--sweep-points", type=int, default=60, help="Number of tau values in the summary curve's sweep")
    parser.add_argument(
        "--seed", type=int, default=0, help="Selects which samples are visualized when a split has more than --num-samples eligible ones"
    )
    parser.add_argument("--output-dir", type=str, default="work_dir/sinkhorn_tau_sweep_visualizations")
    return parser.parse_args()


def summary_curve(distance, taus, n_iters):
    """For each `tau` in `taus`, returns `(diag_mean, mean_max_offdiag)` of the resulting
    Sinkhorn matrix: `diag_mean` (avg. probability a fixation is matched to itself) rises
    from `~1/n` towards `1` as `tau` shrinks, while `mean_max_offdiag` (avg., per row, of the
    largest probability assigned to any *other* fixation) falls the opposite way -- together
    they trace the uniform-to-hard-permutation transition `tau` controls."""
    diag_means = np.empty(len(taus))
    mean_max_offdiags = np.empty(len(taus))
    n = distance.shape[0]
    eye = np.eye(n, dtype=bool)
    for i, tau in enumerate(taus):
        matrix = sinkhorn_normalize(distance, tau, n_iters)
        diag_means[i] = np.diagonal(matrix).mean()
        off_diag = np.where(eye, -np.inf, matrix)
        mean_max_offdiags[i] = off_diag.max(axis=1).mean()
    return diag_means, mean_max_offdiags


def visualize_sample(dataset, sample_index, sample_id, n_iters, taus, sweep_taus, out_path):
    item = dataset[sample_index]
    coords = item["fixations"].numpy()  # (n, 2) = COLUMNS, [0, 1]-normalized, canonical order
    distance = pairwise_distance_map(coords)
    n = coords.shape[0]

    fig = plt.figure(figsize=(4 * len(taus), 9))
    grid = fig.add_gridspec(2, len(taus), height_ratios=[1, 0.8])

    for col, tau in enumerate(taus):
        ax = fig.add_subplot(grid[0, col])
        matrix = sinkhorn_normalize(distance, tau, n_iters)
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=matrix.max())
        marker = " (configured)" if np.isclose(tau, DEFAULT_TAU) else ""
        ax.set_title(f"tau={tau:g}{marker}\ndiag mean={np.diagonal(matrix).mean():.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        annotate_heatmap(ax, matrix, im)

    diag_means, mean_max_offdiags = summary_curve(distance, sweep_taus, n_iters)
    ax_curve = fig.add_subplot(grid[1, :])
    ax_curve.plot(sweep_taus, diag_means, marker="o", markersize=3, label="diagonal mean (self-match probability)")
    ax_curve.plot(sweep_taus, mean_max_offdiags, marker="o", markersize=3, label="mean largest off-diagonal entry")
    ax_curve.axhline(1.0 / n, color="gray", linestyle=":", linewidth=1, label=f"uniform (1/n = {1.0 / n:.3f})")
    ax_curve.axvline(DEFAULT_TAU, color="red", linestyle="--", linewidth=1, label=f"configured tau = {DEFAULT_TAU:g}")
    for tau in taus:
        ax_curve.axvline(tau, color="black", linestyle=":", linewidth=0.5, alpha=0.5)
    ax_curve.set_xscale("log")
    ax_curve.invert_xaxis()  # smaller tau (sharper) on the right, matching the heatmap row's left-to-right sharpening
    ax_curve.set_xlabel("tau (log scale, decreasing)")
    ax_curve.set_ylabel("probability")
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.set_title(f"Sinkhorn transition vs. tau (n_iters={n_iters})")
    ax_curve.legend(loc="center left", fontsize=8)

    fig.suptitle(f"{sample_id}  (n={n} fixations)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    sweep_taus = np.logspace(np.log10(args.sweep_min), np.log10(args.sweep_max), args.sweep_points)

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
                    args.sinkhorn_iters,
                    args.taus,
                    sweep_taus,
                    out_path,
                )
                print(f"Wrote {out_path}")
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
