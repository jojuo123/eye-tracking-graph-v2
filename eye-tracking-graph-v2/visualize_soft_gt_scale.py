"""Compares the soft ground-truth Sinkhorn matrix (`data.reflacx.reflacx.compute_soft_ground_truth`)
computed from the fixation coordinates as actually stored in the H5 file -- i.e. with
`soft_ground_truth.scale` (`configs/fixation_permutation_sorter/reflacx_data.yaml`) already
baked into them by `data.reflacx.reflacx.load_fixations` at preprocessing time -- against the
same matrix computed from the true, un-scaled `[0, 1]`-normalized coordinates
(`soft_ground_truth.columns: [x_position_norm, y_position_norm]` before scaling).

At `tau=1.0`, true normalized-space squared distances are tiny (coordinates live in `[0, 1]`,
so a typical pairwise distance is of order `1e-1`), so `-distance / tau` barely varies between
pairs and Sinkhorn normalizes to something close to uniform (every entry near `1/n`) -- this is
the "entries are all suspiciously close together" behavior `soft_ground_truth.scale` exists to
fix. Baking in `scale` (`224.0` by default -- see that config) multiplies distances by roughly
that much, which at the *same* `tau=1.0` sharpens `-distance / tau` well away from uniform,
without needing to also rescale `tau` or switch `columns` to pixel space. Neither extreme is a
Sinkhorn bug -- `tau` just has to be scaled to match whatever units the distance map is in, and
`scale` is exactly how the H5 file's fixations now pick those units. This script visualizes,
for a few samples per split: the true-normalized-space matrix (what training would see if
`scale=1`), and the as-stored (actually-scaled) matrix -- what the model was really supervised
on.

This is a read-only diagnostic: it only reads the H5 file written by
`data.reflacx.reflacx.preprocess(..., to_h5=True)` and does not need a trained model/checkpoint
(unlike `visualize_permutation.py`, which also predicts and shows a model's permutation).

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
# What `load_fixations` already multiplied `COLUMNS` by before this script's dataset ever
# reads them back out of the H5 file -- divided back out below to recover true [0, 1]
# coordinates for comparison and for correct image-pixel overlay.
DEFAULT_SCALE = reflacx.SOFT_GROUND_TRUTH_SCALE


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
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Factor `soft_ground_truth.scale` baked into the H5 file's stored fixation coordinates "
        "(data.reflacx.reflacx.load_fixations) -- divided back out to recover true [0, 1] coordinates "
        "for the 'true normalized-space' panels and the image overlay.",
    )
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


def visualize_sample(dataset, sample_index, sample_id, tau, n_iters, scale, out_path):
    item = dataset[sample_index]
    image = item["image"]  # (C, H, W)
    # As stored in the H5 file -- already scaled by `soft_ground_truth.scale` (`load_fixations`
    # bakes it in at preprocessing time), in original (canonical) order.
    coords_scaled = item["fixations"].numpy()  # (n, 2) = COLUMNS
    coords_true = coords_scaled / scale  # recovers true [0, 1] normalized coordinates

    height, width = image.shape[-2], image.shape[-1]
    coords_px = to_pixels(coords_true, width, height)

    distance_true = pairwise_distance_map(coords_true)
    distance_scaled = pairwise_distance_map(coords_scaled)  # what compute_soft_ground_truth now sees
    distance_px = pairwise_distance_map(coords_px)

    sinkhorn_true = sinkhorn_normalize(distance_true, tau, n_iters)
    sinkhorn_scaled = sinkhorn_normalize(distance_scaled, tau, n_iters)  # the actual training target
    sinkhorn_px = sinkhorn_normalize(distance_px, tau, n_iters)

    image_hw = to_image_hw(image)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    axes[0, 0].imshow(image_hw, cmap="gray")
    draw_path(axes[0, 0], coords_px, "lime")
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"Fixation path (n={coords_true.shape[0]})")

    im01 = axes[0, 1].imshow(distance_true, cmap="viridis")
    axes[0, 1].set_title(f"True normalized-space distance (mean={distance_true.mean():.2e})")
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.046)

    im02 = axes[0, 2].imshow(sinkhorn_true, cmap="viridis", vmin=0, vmax=sinkhorn_true.max())
    axes[0, 2].set_title(f"Sinkhorn, true normalized space, tau={tau:g}\n(diag mean={np.diagonal(sinkhorn_true).mean():.3f})")
    fig.colorbar(im02, ax=axes[0, 2], fraction=0.046)
    annotate_heatmap(axes[0, 2], sinkhorn_true, im02)

    im10 = axes[1, 0].imshow(distance_scaled, cmap="viridis")
    axes[1, 0].set_title(f"As-stored distance (scale={scale:g}, mean={distance_scaled.mean():.2e})")
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.046)

    im11 = axes[1, 1].imshow(sinkhorn_scaled, cmap="viridis", vmin=0, vmax=sinkhorn_scaled.max())
    axes[1, 1].set_title(f"Sinkhorn, as-stored (actual training target), tau={tau:g}\n(diag mean={np.diagonal(sinkhorn_scaled).mean():.3f})")
    fig.colorbar(im11, ax=axes[1, 1], fraction=0.046)
    annotate_heatmap(axes[1, 1], sinkhorn_scaled, im11)

    im12 = axes[1, 2].imshow(sinkhorn_px, cmap="viridis", vmin=0, vmax=sinkhorn_px.max())
    axes[1, 2].set_title(f"Sinkhorn, raw pixel space, tau={tau:g}\n(diag mean={np.diagonal(sinkhorn_px).mean():.3f})")
    fig.colorbar(im12, ax=axes[1, 2], fraction=0.046)
    annotate_heatmap(axes[1, 2], sinkhorn_px, im12)

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
                visualize_sample(dataset, id_to_index[sample_id], sample_id, args.tau, args.sinkhorn_iters, args.scale, out_path)
                print(f"Wrote {out_path}")
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
