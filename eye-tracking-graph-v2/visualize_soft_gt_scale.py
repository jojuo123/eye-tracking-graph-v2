"""Compares the soft ground-truth Sinkhorn matrix (`data.reflacx.reflacx.compute_soft_ground_truth`)
when the pairwise fixation distance map is computed in normalized `[0, 1]` coordinate space --
what `compute_soft_ground_truth` actually uses, via `configs/fixation_permutation_sorter/
reflacx_data.yaml`'s `soft_ground_truth.columns: [x_position_norm, y_position_norm]` -- versus
pixel ("image") space, i.e. those same coordinates scaled up to the resized image's actual
`(width, height)` (`image.resize` in that same config, `224x224` by default).

At `tau=1.0`, normalized-space squared distances are tiny (coordinates live in `[0, 1]`, so a
typical pairwise MSE is of order `1e-2`), so `-distance / tau` barely varies between pairs and
Sinkhorn normalizes to something close to uniform (every entry near `1/n`) -- this is the
"entries are all suspiciously close together" behavior in question. Scaling the same
coordinates up to pixel space multiplies distances by roughly `width * height` (~50000 for
224x224), which at the *same* `tau=1.0` swings the opposite way: `-distance / tau` becomes
hugely negative for anything but the diagonal, and Sinkhorn collapses to (numerically) the hard
identity permutation instead. Neither extreme is a Sinkhorn bug -- `tau` just has to be scaled
to match whatever units the distance map is in (there is no separate `scale` knob any more --
lowering `tau` and multiplying the distance map by a fixed factor have identical effect, since
`-distance / tau == -(distance * s) / (tau * s)` for any `s > 0`, so `tau` alone controls it).
This script visualizes all three for a few samples per split: the normalized-space matrix, the
raw pixel-space matrix at the same `tau`, and a pixel-space matrix whose `tau` is auto-rescaled
by the empirical ratio between the two distance maps (so its spread is comparable to the
normalized-space one).

This is a read-only diagnostic: it only reads the H5 file written by
`data.reflacx.reflacx.preprocess(..., to_h5=True)` and does not need a trained model/checkpoint
(unlike `visualize_permutation.py`, which also predicts and shows a model's permutation). See
also `visualize_sinkhorn_tau_sweep.py`, which sweeps a much finer range of that effective
factor to show the full uniform-to-hard-permutation transition.

    cd eye-tracking-graph-v2
    python visualize_soft_gt_scale.py --max-length 30 --num-samples 5
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.reflacx import reflacx
from dataloaders import build_dataset
from visualize_permutation import annotate_heatmap, draw_path, pairwise_distance_map, sinkhorn_normalize, to_image_hw, to_pixels

SPLITS = ("TRAIN", "VAL", "TEST")
COLUMNS = tuple(reflacx.SOFT_GROUND_TRUTH_COLUMNS)  # e.g. ("x_position_norm", "y_position_norm")
DEFAULT_TAU = reflacx.SOFT_GROUND_TRUTH_TAU
DEFAULT_SINKHORN_ITERS = reflacx.SOFT_GROUND_TRUTH_N_ITERS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-length", type=int, default=20, help="Only consider samples with n_fixations < this")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples to visualize per split")
    parser.add_argument("--splits", type=str, nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help="Sinkhorn temperature, applied as-is to both the normalized- and pixel-space distance maps "
        "(the whole point being to show how differently the same tau behaves in each)",
    )
    parser.add_argument("--sinkhorn-iters", type=int, default=DEFAULT_SINKHORN_ITERS)
    parser.add_argument(
        "--seed", type=int, default=0, help="Selects which samples are visualized when a split has more than --num-samples eligible ones"
    )
    parser.add_argument("--output-dir", type=str, default="work_dir/soft_gt_scale_visualizations")
    return parser.parse_args()


def select_sample_ids(h5_path, split, dataset_name, max_length, num_samples, seed):
    """Same approach as `visualize_permutation.select_sample_ids`: reads only each sample's
    fixation-sequence length out of the H5 file (not the fixations/image themselves), then
    samples up to `num_samples` of the eligible (short enough) ones."""
    with h5py.File(h5_path, "r") as f:
        group = f[split][dataset_name]
        sample_ids = sorted(group.keys())
        lengths = {}
        for sample_id in sample_ids:
            fixations_group = group[sample_id]["fixations"]
            first_column = next(iter(fixations_group.keys()))
            lengths[sample_id] = fixations_group[first_column].shape[0]

    eligible = sorted(sid for sid in sample_ids if lengths[sid] < max_length)
    if len(eligible) > num_samples:
        rng = np.random.default_rng(seed)
        eligible = sorted(rng.choice(eligible, size=num_samples, replace=False).tolist())
    return eligible


def visualize_sample(dataset, sample_index, sample_id, tau, n_iters, out_path):
    item = dataset[sample_index]
    image = item["image"]  # (C, H, W)
    coords_norm = item["fixations"].numpy()  # (n, 2) = COLUMNS, in original (canonical) order

    height, width = image.shape[-2], image.shape[-1]
    coords_px = to_pixels(coords_norm, width, height)

    distance_norm = pairwise_distance_map(coords_norm)
    distance_px = pairwise_distance_map(coords_px)

    sinkhorn_norm_space = sinkhorn_normalize(distance_norm, tau, n_iters)
    sinkhorn_px_raw = sinkhorn_normalize(distance_px, tau, n_iters)

    # Rescale tau by the empirical ratio between the two distance maps' magnitudes, so the
    # pixel-space matrix has spread comparable to the normalized-space one instead of collapsing
    # to (numerically) a hard identity permutation at the same nominal tau.
    scale = distance_px.mean() / max(distance_norm.mean(), 1e-12)
    tau_auto = tau * scale
    sinkhorn_px_scaled = sinkhorn_normalize(distance_px, tau_auto, n_iters)

    image_hw = to_image_hw(image)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    axes[0, 0].imshow(image_hw, cmap="gray")
    draw_path(axes[0, 0], coords_px, "lime")
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"Fixation path (n={coords_norm.shape[0]})")

    im01 = axes[0, 1].imshow(distance_norm, cmap="viridis")
    axes[0, 1].set_title(f"Normalized-space distance (mean={distance_norm.mean():.2e})")
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.046)

    im02 = axes[0, 2].imshow(sinkhorn_norm_space, cmap="viridis", vmin=0, vmax=sinkhorn_norm_space.max())
    axes[0, 2].set_title(f"Sinkhorn, normalized space, tau={tau:g}\n(diag mean={np.diagonal(sinkhorn_norm_space).mean():.3f})")
    fig.colorbar(im02, ax=axes[0, 2], fraction=0.046)
    annotate_heatmap(axes[0, 2], sinkhorn_norm_space, im02)

    im10 = axes[1, 0].imshow(distance_px, cmap="viridis")
    axes[1, 0].set_title(f"Pixel-space distance (mean={distance_px.mean():.2e}, x{scale:.0f} vs. normalized)")
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.046)

    im11 = axes[1, 1].imshow(sinkhorn_px_raw, cmap="viridis", vmin=0, vmax=sinkhorn_px_raw.max())
    axes[1, 1].set_title(f"Sinkhorn, pixel space, tau={tau:g}\n(diag mean={np.diagonal(sinkhorn_px_raw).mean():.3f})")
    fig.colorbar(im11, ax=axes[1, 1], fraction=0.046)
    annotate_heatmap(axes[1, 1], sinkhorn_px_raw, im11)

    im12 = axes[1, 2].imshow(sinkhorn_px_scaled, cmap="viridis", vmin=0, vmax=sinkhorn_px_scaled.max())
    axes[1, 2].set_title(f"Sinkhorn, pixel space, tau={tau_auto:.1f} (auto-scaled)\n(diag mean={np.diagonal(sinkhorn_px_scaled).mean():.3f})")
    fig.colorbar(im12, ax=axes[1, 2], fraction=0.046)
    annotate_heatmap(axes[1, 2], sinkhorn_px_scaled, im12)

    fig.suptitle(sample_id)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

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
                visualize_sample(dataset, id_to_index[sample_id], sample_id, args.tau, args.sinkhorn_iters, out_path)
                print(f"Wrote {out_path}")
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
