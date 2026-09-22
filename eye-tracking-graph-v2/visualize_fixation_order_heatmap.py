"""Checks whether fixation *order* carries spatial structure on its own, independent of any
particular image: for a fixed index i, does the i-th fixation across the dataset's samples
cluster in a consistent region (e.g. everyone starts top-left, or index 3 tends to land near
the center), or is it scattered like the rest of the sequence?

Grouping by raw fixation index (`--bin-mode index`, the default) is somewhat arbitrary: samples
scan at different speeds, so "the 5th fixation" can mean half a second into a fast scan or three
seconds into a slow one -- any structure it finds conflates *order* with *elapsed time*. Two
alternative groupings disentangle that:

  - `--bin-mode timestamp`: groups fixations that fall in the same wall-clock time window
    (e.g. every fixation between t=2s and t=3s across all samples), regardless of what index
    they were within their own sequence.
  - `--bin-mode percent`: like `timestamp`, but each sample's time axis is first rescaled to
    [0, 100]% of its own total viewing duration, so a 5-second scan and a 20-second scan both
    contribute to e.g. the "50-60% of viewing time" bin instead of being misaligned by raw
    seconds.

For each bin (an index, a time window, or a percent-of-duration window), this collects every
contributing fixation's (`x_position_norm`, `y_position_norm`) and renders a Gaussian-smoothed
2D density heatmap of those points in normalized [0, 1] image coordinates, with the contributing
points overlaid as markers. One PNG is written per bin.

Reads the `*_norm` fixation columns (and, for `timestamp`/`percent`, the raw timestamp columns)
straight out of the H5 file written by `data.reflacx.reflacx.preprocess(..., to_h5=True)` -- no
torch/model needed. A handful of REFLACX fixations fall slightly outside [0, 1] (raw gaze
noise); these are clipped into bounds purely for plotting, matching `coordinate_bounds=(0, 1)`
elsewhere in this repo.

Each heatmap is normalized independently (divided by its own peak density) so the *shape* of
each bin's distribution is comparable regardless of how many samples/fixations contributed
(that count varies across bins -- it's reported in the title and a companion
`sample_counts.png` bar chart).

    cd eye-tracking-graph-v2
    python visualize_fixation_order_heatmap.py --bin-mode index --max-index 40 --min-samples 3
    python visualize_fixation_order_heatmap.py --bin-mode timestamp --time-bin-width 1.0
    python visualize_fixation_order_heatmap.py --bin-mode percent --n-percent-bins 10
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
        "--bin-mode",
        type=str,
        choices=["index", "timestamp", "percent"],
        default="index",
        help="How to group fixations across samples before plotting: 'index' groups the i-th "
        "fixation of every sequence together (order-based, ignores elapsed time); 'timestamp' "
        "groups fixations falling in the same wall-clock time window; 'percent' groups "
        "fixations by what percentage of each sample's own total viewing time they fall into.",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="timestamp_start_fixation",
        help="Column giving each fixation's time position, used by --bin-mode timestamp/percent",
    )
    parser.add_argument(
        "--duration-column",
        type=str,
        default="timestamp_end_fixation",
        help="Column whose last value (per sample) gives that sample's total viewing duration, "
        "used to rescale --timestamp-column into 0-100%% for --bin-mode percent",
    )
    parser.add_argument(
        "--max-index", type=int, default=None, help="Highest i to plot (default: longest sequence's last index) (--bin-mode index)"
    )
    parser.add_argument(
        "--time-bin-width", type=float, default=1.0, help="Width of each time bin, in seconds (--bin-mode timestamp)"
    )
    parser.add_argument(
        "--max-time", type=float, default=None, help="Highest timestamp to plot, in seconds (default: longest sample's last "
        "timestamp) (--bin-mode timestamp)"
    )
    parser.add_argument(
        "--n-percent-bins", type=int, default=10, help="Number of equal-width bins spanning 0-100%% of viewing time (--bin-mode percent)"
    )
    parser.add_argument(
        "--min-samples", type=int, default=3, help="Skip bins where fewer than this many samples contribute a fixation"
    )
    parser.add_argument("--grid-size", type=int, default=224, help="Heatmap resolution (grid_size x grid_size)")
    parser.add_argument("--sigma", type=float, default=6.0, help="Gaussian blur stddev, in grid cells")
    parser.add_argument("--output-dir", type=str, default="work_dir/fixation_order_heatmaps")
    return parser.parse_args()


def load_fixation_columns(h5_path, split, dataset_name, columns):
    """Returns `{sample_id: {col: (n,) array}}` for each `col` in `columns`, one H5 open."""
    data = {}
    with h5py.File(h5_path, "r") as f:
        group = f[split][dataset_name]
        for sample_id in sorted(group.keys()):
            fixations = group[sample_id]["fixations"]
            data[sample_id] = {col: fixations[col][()] for col in columns}
    return data


def build_coordinate_sequences(raw, x_col, y_col):
    """`raw`: `load_fixation_columns` output -> `{sample_id: (n, 2) array}` of (x, y), clipped
    to [0, 1] (a few REFLACX fixations fall just outside it -- raw gaze noise)."""
    sequences = {}
    n_clipped = 0
    for sample_id, cols in raw.items():
        xy = np.stack([cols[x_col], cols[y_col]], axis=-1).astype(np.float32)
        n_clipped += int(np.count_nonzero((xy < 0.0) | (xy > 1.0)))
        sequences[sample_id] = np.clip(xy, 0.0, 1.0)
    if n_clipped:
        print(f"Clipped {n_clipped} out-of-[0,1] coordinate value(s) into bounds for plotting")
    return sequences


def to_percent_positions(timestamps, durations):
    """`timestamps`/`durations`: `{sample_id: (n,) array}` / `{sample_id: scalar}` ->
    `{sample_id: (n,) array}` of each fixation's timestamp rescaled to [0, 100]% of that
    sample's own duration. Samples with a non-positive duration are dropped (can't rescale)."""
    positions = {}
    for sample_id, ts in timestamps.items():
        duration = durations.get(sample_id, 0.0)
        if duration <= 0 or len(ts) == 0:
            continue
        positions[sample_id] = np.clip(ts.astype(np.float64) / duration * 100.0, 0.0, 100.0)
    return positions


def bin_by_value(value_by_sample, xy_sequences, bin_edges):
    """`value_by_sample`: `{sample_id: (n,) array}` of a per-fixation scalar (e.g. timestamp or
    percent-of-duration), aligned row-for-row with `xy_sequences[sample_id]`. `bin_edges`:
    increasing 1D array of `B + 1` edges. Returns a length-`B` list of `(points, sample_ids)`,
    one per bin, where a fixation falls in bin b when `bin_edges[b] <= value < bin_edges[b+1]`
    (values outside `[bin_edges[0], bin_edges[-1]]` are clipped into the first/last bin)."""
    n_bins = len(bin_edges) - 1
    bin_points = [[] for _ in range(n_bins)]
    bin_samples = [set() for _ in range(n_bins)]
    for sample_id, values in value_by_sample.items():
        if len(values) == 0:
            continue
        xy = xy_sequences[sample_id]
        bin_idx = np.clip(np.searchsorted(bin_edges, values, side="right") - 1, 0, n_bins - 1)
        for b in np.unique(bin_idx):
            mask = bin_idx == b
            bin_points[int(b)].append(xy[mask])
            bin_samples[int(b)].add(sample_id)
    return [
        (np.concatenate(points, axis=0) if points else np.empty((0, 2), dtype=np.float32), samples)
        for points, samples in zip(bin_points, bin_samples)
    ]


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


def plot_bin(points, title, grid_size, sigma, out_path):
    heatmap = density_heatmap(points, grid_size, sigma)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap, cmap="viridis", vmin=0, vmax=1, extent=(0, 1, 1, 0), origin="upper")
    ax.scatter(points[:, 0], points[:, 1], s=18, c="white", edgecolors="black", linewidths=0.6, alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("x_position_norm")
    ax.set_ylabel("y_position_norm")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="density (normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_counts(x_values, counts, xlabel, title, out_path, bar_width=1.0):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x_values, counts, color="#3B6FA0", width=bar_width, align="edge")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# samples contributing")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_index_mode(args, sequences):
    lengths = {sid: xy.shape[0] for sid, xy in sequences.items()}
    max_index = args.max_index if args.max_index is not None else max(lengths.values()) - 1
    print(f"{len(sequences)} samples, sequence lengths {min(lengths.values())}-{max(lengths.values())}, plotting i=0..{max_index}")

    counts_by_index = {}
    n_written = 0
    for i in range(max_index + 1):
        contributing = [sid for sid, xy in sequences.items() if xy.shape[0] > i]
        counts_by_index[i] = len(contributing)
        if len(contributing) < args.min_samples:
            continue
        points = np.stack([sequences[sid][i] for sid in contributing], axis=0)
        title = f"Fixation index i={i}  (n={len(contributing)} samples, normalized to peak=1)"
        out_path = os.path.join(args.output_dir, f"fixation_{i:03d}.png")
        plot_bin(points, title, args.grid_size, args.sigma, out_path)
        n_written += 1

    indices = sorted(counts_by_index)
    counts_path = os.path.join(args.output_dir, "sample_counts.png")
    plot_counts(
        [i - 0.5 for i in indices],
        [counts_by_index[i] for i in indices],
        "fixation index i",
        "Samples contributing to each per-index heatmap",
        counts_path,
        bar_width=1.0,
    )

    print(f"Wrote {n_written} heatmap(s) to {args.output_dir} (skipped indices with < {args.min_samples} samples)")
    print(f"Wrote {counts_path}")


def run_binned_time_mode(args, sequences, value_by_sample, bin_edges, filename_prefix, format_range, xlabel, counts_title):
    """Shared plotting loop for `timestamp` and `percent` modes, which differ only in what
    `value_by_sample` holds and how bin edges are labeled (`format_range`)."""
    bins = bin_by_value(value_by_sample, sequences, bin_edges)

    n_written = 0
    count_x, counts = [], []
    for b, (points, samples) in enumerate(bins):
        count_x.append(bin_edges[b])
        counts.append(len(samples))
        if len(samples) < args.min_samples:
            continue
        range_label = format_range(bin_edges[b], bin_edges[b + 1])
        title = f"{range_label}  (n={len(samples)} samples, {len(points)} fixations, normalized to peak=1)"
        out_path = os.path.join(args.output_dir, f"{filename_prefix}_{b:03d}.png")
        plot_bin(points, title, args.grid_size, args.sigma, out_path)
        n_written += 1

    counts_path = os.path.join(args.output_dir, "sample_counts.png")
    bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
    plot_counts(count_x, counts, xlabel, counts_title, counts_path, bar_width=bin_width)

    print(f"Wrote {n_written} heatmap(s) to {args.output_dir} (skipped bins with < {args.min_samples} samples)")
    print(f"Wrote {counts_path}")


def run_timestamp_mode(args, sequences, timestamps):
    max_time = args.max_time if args.max_time is not None else max(ts.max() for ts in timestamps.values() if len(ts))
    n_bins = max(1, int(np.ceil(max_time / args.time_bin_width)))
    bin_edges = np.arange(n_bins + 1) * args.time_bin_width
    print(f"{len(sequences)} samples, {args.timestamp_column} spans 0-{max_time:.1f}s, plotting {n_bins} bin(s) of width {args.time_bin_width}s")

    run_binned_time_mode(
        args,
        sequences,
        timestamps,
        bin_edges,
        filename_prefix="fixation_time",
        format_range=lambda lo, hi: f"t ∈ [{lo:.1f}, {hi:.1f})s",
        xlabel="time bin start (s)",
        counts_title="Samples contributing to each time-bin heatmap",
    )


def run_percent_mode(args, sequences, timestamps, durations):
    percent_positions = to_percent_positions(timestamps, durations)
    n_dropped = len(timestamps) - len(percent_positions)
    if n_dropped:
        print(f"Dropped {n_dropped} sample(s) with a non-positive total duration (can't rescale to percent)")
    if not percent_positions:
        raise SystemExit("No samples had a usable total duration for --bin-mode percent")

    bin_edges = np.linspace(0.0, 100.0, args.n_percent_bins + 1)
    print(f"{len(percent_positions)} samples, plotting {args.n_percent_bins} bin(s) of viewing-time percentage")

    run_binned_time_mode(
        args,
        sequences,
        percent_positions,
        bin_edges,
        filename_prefix="fixation_percent",
        format_range=lambda lo, hi: f"{lo:.0f}-{hi:.0f}% of viewing time",
        xlabel="% of viewing time (bin start)",
        counts_title="Samples contributing to each viewing-time-percent heatmap",
    )


def main():
    args = parse_args()
    x_col, y_col = args.coordinate_columns

    columns = [x_col, y_col]
    if args.bin_mode in ("timestamp", "percent"):
        columns.append(args.timestamp_column)
    if args.bin_mode == "percent":
        columns.append(args.duration_column)

    raw = load_fixation_columns(args.h5_path, args.split, args.dataset_name, columns)
    if not raw:
        raise SystemExit(f"No samples found at {args.h5_path}:{args.split}/{args.dataset_name}")

    sequences = build_coordinate_sequences(raw, x_col, y_col)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.bin_mode == "index":
        run_index_mode(args, sequences)
    elif args.bin_mode == "timestamp":
        timestamps = {sid: cols[args.timestamp_column].astype(np.float64) for sid, cols in raw.items()}
        run_timestamp_mode(args, sequences, timestamps)
    else:
        timestamps = {sid: cols[args.timestamp_column].astype(np.float64) for sid, cols in raw.items()}
        durations = {
            sid: float(cols[args.duration_column][-1]) if len(cols[args.duration_column]) else 0.0 for sid, cols in raw.items()
        }
        run_percent_mode(args, sequences, timestamps, durations)


if __name__ == "__main__":
    main()
