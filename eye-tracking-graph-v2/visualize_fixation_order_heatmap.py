"""Checks whether fixation *order* carries spatial structure on its own, independent of any
particular image: for a fixed index i, does the i-th fixation across the dataset's samples
cluster in a consistent region (e.g. everyone starts top-left, or index 3 tends to land near
the center), or is it scattered like the rest of the sequence?

For each i = 0, 1, 2, ..., this collects every sample's i-th fixation (`x_position_norm`,
y_position_norm`) -- skipping samples whose sequence has fewer than i+1 fixations -- and
renders a Gaussian-smoothed 2D density heatmap of those points in normalized [0, 1] image
coordinates, with the contributing points overlaid as markers. One PNG is written per index.

Reads the `*_norm` fixation columns straight out of the H5 file written by
`data.reflacx.reflacx.preprocess(..., to_h5=True)` -- no torch/model needed. A handful of
REFLACX fixations fall slightly outside [0, 1] (raw gaze noise); these are clipped into bounds
purely for plotting, matching `coordinate_bounds=(0, 1)` elsewhere in this repo.

Each heatmap is normalized independently (divided by its own peak density) so the *shape* of
the i-th distribution is comparable across indices regardless of how many samples contributed
(that count shrinks as i grows, since not every sample's sequence is that long -- it's reported
in the title and a companion `sample_counts.png` bar chart).

    cd eye-tracking-graph-v2
    python visualize_fixation_order_heatmap.py --max-index 40 --min-samples 3
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_H5_PATH = os.path.join(os.path.dirname(__file__), "..", "reflacx_data.h5")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5-path", type=str, default=DEFAULT_H5_PATH)
    parser.add_argument("--split", type=str, default="TRAIN")
    parser.add_argument("--dataset-name", type=str, default="reflacx")
    parser.add_argument(
        "--coordinate-columns", type=str, nargs=2, default=("x_position_norm", "y_position_norm"), metavar=("X_COL", "Y_COL")
    )
    parser.add_argument(
        "--max-index", type=int, default=None, help="Highest i to plot (default: longest sequence's last index)"
    )
    parser.add_argument(
        "--min-samples", type=int, default=3, help="Skip indices i where fewer than this many samples have a fixation"
    )
    parser.add_argument("--grid-size", type=int, default=224, help="Heatmap resolution (grid_size x grid_size)")
    parser.add_argument("--sigma", type=float, default=6.0, help="Gaussian blur stddev, in grid cells")
    parser.add_argument("--output-dir", type=str, default="work_dir/fixation_order_heatmaps")
    return parser.parse_args()


def load_sequences(h5_path, split, dataset_name, coordinate_columns):
    """Returns `{sample_id: (n, 2) array}` of (x, y) in `coordinate_columns`' order, clipped
    to [0, 1] (a few REFLACX fixations fall just outside it -- raw gaze noise)."""
    x_col, y_col = coordinate_columns
    sequences = {}
    n_clipped = 0
    with h5py.File(h5_path, "r") as f:
        group = f[split][dataset_name]
        for sample_id in sorted(group.keys()):
            fixations = group[sample_id]["fixations"]
            xy = np.stack([fixations[x_col][()], fixations[y_col][()]], axis=-1).astype(np.float32)
            n_clipped += int(np.count_nonzero((xy < 0.0) | (xy > 1.0)))
            sequences[sample_id] = np.clip(xy, 0.0, 1.0)
    if n_clipped:
        print(f"Clipped {n_clipped} out-of-[0,1] coordinate value(s) into bounds for plotting")
    return sequences


def density_heatmap(points, grid_size, sigma):
    """`points`: `(n, 2)` array of (x, y) in `[0, 1]` -> `(grid_size, grid_size)` Gaussian-
    smoothed density, normalized so its peak is 1.0. `points` splat one count each into the
    nearest grid cell before blurring -- the standard "eye-tracking heatmap" recipe."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    cols = np.clip((points[:, 0] * (grid_size - 1)).round().astype(int), 0, grid_size - 1)
    rows = np.clip((points[:, 1] * (grid_size - 1)).round().astype(int), 0, grid_size - 1)
    np.add.at(grid, (rows, cols), 1.0)
    grid = gaussian_filter(grid, sigma=sigma)
    peak = grid.max()
    if peak > 0:
        grid = grid / peak
    return grid


def plot_index(i, points, sample_ids, grid_size, sigma, out_path):
    heatmap = density_heatmap(points, grid_size, sigma)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap, cmap="viridis", vmin=0, vmax=1, extent=(0, 1, 1, 0), origin="upper")
    ax.scatter(points[:, 0], points[:, 1], s=18, c="white", edgecolors="black", linewidths=0.6, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("x_position_norm")
    ax.set_ylabel("y_position_norm")
    ax.set_title(f"Fixation index i={i}  (n={len(sample_ids)} samples, normalized to peak=1)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="density (normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_counts(counts_by_index, out_path):
    indices = sorted(counts_by_index)
    counts = [counts_by_index[i] for i in indices]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(indices, counts, color="#3B6FA0", width=1.0)
    ax.set_xlabel("fixation index i")
    ax.set_ylabel("# samples with a fixation at index i")
    ax.set_title("Samples contributing to each per-index heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    sequences = load_sequences(args.h5_path, args.split, args.dataset_name, tuple(args.coordinate_columns))
    if not sequences:
        raise SystemExit(f"No samples found at {args.h5_path}:{args.split}/{args.dataset_name}")

    lengths = {sid: xy.shape[0] for sid, xy in sequences.items()}
    max_index = args.max_index if args.max_index is not None else max(lengths.values()) - 1
    print(f"{len(sequences)} samples, sequence lengths {min(lengths.values())}-{max(lengths.values())}, plotting i=0..{max_index}")

    os.makedirs(args.output_dir, exist_ok=True)
    counts_by_index = {}
    n_written = 0
    for i in range(max_index + 1):
        contributing = [sid for sid, xy in sequences.items() if xy.shape[0] > i]
        counts_by_index[i] = len(contributing)
        if len(contributing) < args.min_samples:
            continue
        points = np.stack([sequences[sid][i] for sid in contributing], axis=0)
        out_path = os.path.join(args.output_dir, f"fixation_{i:03d}.png")
        plot_index(i, points, contributing, args.grid_size, args.sigma, out_path)
        n_written += 1

    counts_path = os.path.join(args.output_dir, "sample_counts.png")
    plot_sample_counts(counts_by_index, counts_path)

    print(f"Wrote {n_written} heatmap(s) to {args.output_dir} (skipped indices with < {args.min_samples} samples)")
    print(f"Wrote {counts_path}")


if __name__ == "__main__":
    main()
