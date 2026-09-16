"""Visualizes `models.fixation_permutation_sorter.FixationPermutationSorter`'s predicted
fixation order against ground truth.

For up to `--num-samples` (default 5) samples from each of train/val/test whose fixation
sequence is shorter than `--max-length`, this:
  1. computes the pairwise (Euclidean) distance map between fixations in their original,
     canonical order;
  2. Sinkhorn-normalizes that distance map into a doubly-stochastic matrix (same recipe as
     `data.reflacx.reflacx.compute_soft_ground_truth`, but computed directly here so this
     script works regardless of whether the H5 file has a precomputed `soft_permutation`);
  3. shuffles the fixation sequence (`dataloaders.permuted_h5_dataset.PermutedFixationH5Dataset`,
     with a fixed seed so re-running reproduces the same figures) and runs it through a
     trained model to predict the permutation that recovers the original order;
  4. saves one figure per sample with five panels: the distance map, the normalized
     (Sinkhorn) matrix, the image overlaid with the true fixation path, the image overlaid
     with the predicted path, and the image overlaid with both.

    cd eye-tracking-graph-v2
    python visualize_permutation.py \
        --config configs/fixation_permutation_sorter/train.yaml \
        --checkpoint work_dir/fixation_permutation_sorter/checkpoints/checkpoint_best.pt \
        --max-length 30
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloaders import build_dataset
from models import build_model
from modules.sinkhorn import sinkhorn_norm
from utils.checkpoint import CheckpointManager
from utils.config import load_config

SPLITS = ("train", "val", "test")

# Matches `configs/fixation_permutation_sorter/reflacx_data.yaml`'s `soft_ground_truth`
# defaults, so the visualized normalized matrix is comparable to the one a model trained
# with `load_soft_permutation: true` was actually supervised on.
DEFAULT_TAU = 1.0
DEFAULT_SINKHORN_ITERS = 20
DEFAULT_SCALE = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/fixation_permutation_sorter/train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint (.pt)")
    parser.add_argument("--max-length", type=int, default=20, help="Only consider samples with n_fixations < this")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples to visualize per split")
    parser.add_argument("--splits", type=str, nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Sinkhorn temperature for the distance map")
    parser.add_argument("--sinkhorn-iters", type=int, default=DEFAULT_SINKHORN_ITERS)
    parser.add_argument(
        "--scale", type=float, default=DEFAULT_SCALE, help="Multiplies the distance map before Sinkhorn (-distance * scale / tau)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Selects which samples/shuffle are visualized")
    parser.add_argument("--output-dir", type=str, default="work_dir/permutation_visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def select_sample_ids(h5_path, split_name, dataset_name, max_length, num_samples, seed):
    """Picks up to `num_samples` sample ids with `n_fixations < max_length` straight out of
    the H5 file (only reading each fixation group's shape, not its data), so this doesn't
    need a full `Dataset` built first just to filter by length."""
    with h5py.File(h5_path, "r") as f:
        if split_name not in f or dataset_name not in f[split_name]:
            return []
        group = f[split_name][dataset_name]
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


def gather_by_inverse(sequence, inverse_permutation):
    """`sequence`: `(n, f)` array in shuffled order; `inverse_permutation`: `(n,)` array such
    that `result[j] = sequence[inverse_permutation[j]]` -- i.e. recovers the order `sequence`
    would need to be in for slot `j` to hold the fixation that belongs there. Single-sample
    (no batch dim) counterpart of `FixationPermutationSorter._gather_by_inverse`."""
    return sequence[inverse_permutation]


def pairwise_distance_map(coords):
    """`(n, 2) -> (n, n)` pairwise Euclidean distance -- same formula as
    `data.reflacx.reflacx.compute_soft_ground_truth`."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def sinkhorn_normalize(distance, tau, n_iters, scale=DEFAULT_SCALE):
    """Same recipe as `data.reflacx.reflacx.compute_soft_ground_truth`'s `"sinkhorn"` method:
    `scale` multiplies the distance map before it's turned into a Sinkhorn score
    (`-distance * scale / tau`)."""
    log_alpha = torch.from_numpy(-distance * scale / tau).float().unsqueeze(0)
    return sinkhorn_norm(log_alpha, n_iters=n_iters)[0].numpy()


def to_pixels(coords_norm, width, height):
    """`coords_norm`: `(n, 2)` array of (x, y) in `[0, 1]` -> pixel coordinates in an
    `(height, width)` image."""
    pixels = coords_norm.copy()
    pixels[:, 0] *= width
    pixels[:, 1] *= height
    return pixels


def annotate_heatmap(ax, matrix, im, fmt="{:.2f}", max_n_for_text=25):
    """Writes each cell's value as text on top of `ax.imshow(matrix)` (`im`), colored for
    contrast against that cell's color. Skipped above `max_n_for_text` per side, since text
    for every cell of a large matrix is unreadable rather than informative.

    Existing at `tau=1.0` (`DEFAULT_TAU`) with `[0, 1]`-normalized coordinates, the
    Sinkhorn-normalized matrix's entries are legitimately all close to `1/n` (see
    `compute_soft_ground_truth`'s doubling of the score range never exceeding 1) -- e.g. for
    `n=20`, even the diagonal only reaches ~0.05-0.07, nowhere near 1. That's real
    doubly-stochastic behavior (every row still sums to 1), not a bug, but it looks like
    "everything is 0" on a fixed `vmin=0, vmax=1` color scale -- these annotations make the
    actual numbers visible regardless of how flat the color scale is."""
    n = matrix.shape[0]
    if n > max_n_for_text:
        return
    norm = im.norm
    cmap = im.cmap
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            # Choose black/white text based on the cell color's luminance so the number
            # stays legible whether the cell is near the colormap's light or dark end.
            r, g, b, _ = cmap(norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "black" if luminance > 0.5 else "white"
            ax.text(j, i, fmt.format(value), ha="center", va="center", color=text_color, fontsize=6)


def draw_path(ax, coords_px, color, linestyle="-", marker_size=30, label=None):
    xs, ys = coords_px[:, 0], coords_px[:, 1]
    ax.plot(xs, ys, linestyle, color=color, linewidth=1.5, alpha=0.85, label=label)
    ax.scatter(xs, ys, c=color, s=marker_size, zorder=3, edgecolors="white", linewidths=0.5)
    ax.scatter(xs[0], ys[0], c=color, s=marker_size * 3, marker="^", zorder=4, edgecolors="black")  # start
    ax.scatter(xs[-1], ys[-1], c=color, s=marker_size * 3, marker="s", zorder=4, edgecolors="black")  # end


def to_image_hw(image):
    """`(C, H, W)` torch tensor -> `(H, W)` (grayscale) or `(H, W, C)` numpy array for imshow."""
    image = image.numpy()
    if image.shape[0] == 1:
        return image[0]
    return np.transpose(image, (1, 2, 0))


@torch.no_grad()
def predict_permutation(model, image, fixations, device):
    batch = {
        "image": image.unsqueeze(0).to(device),
        "fixations": fixations.unsqueeze(0).to(device),
        "fixations_mask": torch.ones(1, fixations.shape[0], dtype=torch.bool, device=device),
    }
    was_training = model.training
    model.eval()
    output = model(batch)
    if was_training:
        model.train()
    return output["outputs"][0].cpu().numpy()


def visualize_sample(
    model, dataset, sample_index, sample_id, coordinate_indices, tau, n_iters, scale, device, out_path
):
    item = dataset[sample_index]
    image = item["image"]  # (C, H, W)
    shuffled_fixations = item["fixations"]  # (n, F), shuffled order
    inverse_permutation = item["inverse_permutation"].numpy()
    true_permutation = item["permutation"].numpy()

    shuffled_coords = shuffled_fixations[:, coordinate_indices].numpy()
    original_coords = gather_by_inverse(shuffled_coords, inverse_permutation)  # canonical order

    distance = pairwise_distance_map(original_coords)
    normalized = sinkhorn_normalize(distance, tau, n_iters, scale)

    predicted_permutation = predict_permutation(model, image, shuffled_fixations, device)
    predicted_inverse = np.argsort(predicted_permutation)
    predicted_coords = gather_by_inverse(shuffled_coords, predicted_inverse)

    height, width = image.shape[-2], image.shape[-1]
    true_px = to_pixels(original_coords, width, height)
    pred_px = to_pixels(predicted_coords, width, height)
    image_hw = to_image_hw(image)

    accuracy = float((predicted_permutation == true_permutation).mean())

    fig, axes = plt.subplots(1, 5, figsize=(27, 5.2))

    im0 = axes[0].imshow(distance, cmap="viridis")
    axes[0].set_title("Pairwise distance map")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # Auto-ranged (not a fixed vmin=0/vmax=1 scale): at `tau=1.0` over [0, 1]-normalized
    # coordinates, entries are legitimately all close to 1/n (see `annotate_heatmap`'s
    # docstring) -- a fixed 0-1 scale would crush that real structure to "looks all black".
    im1 = axes[1].imshow(normalized, cmap="viridis", vmin=0, vmax=normalized.max())
    axes[1].set_title(f"Sinkhorn-normalized (diag mean={np.diagonal(normalized).mean():.3f})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    annotate_heatmap(axes[1], normalized, im1)

    axes[2].imshow(image_hw, cmap="gray")
    draw_path(axes[2], true_px, "lime")
    axes[2].axis("off")
    axes[2].set_title("Ground-truth order")

    axes[3].imshow(image_hw, cmap="gray")
    draw_path(axes[3], pred_px, "red")
    axes[3].axis("off")
    axes[3].set_title("Predicted order")

    axes[4].imshow(image_hw, cmap="gray")
    draw_path(axes[4], true_px, "lime", linestyle="-", label="ground truth")
    draw_path(axes[4], pred_px, "red", linestyle="--", label="predicted")
    axes[4].axis("off")
    axes[4].legend(loc="lower right", fontsize=8, framealpha=0.6)
    axes[4].set_title("Ground truth vs. predicted")

    fig.suptitle(f"{sample_id}  (n={shuffled_fixations.shape[0]}, permutation_accuracy={accuracy:.2f})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    model = build_model(cfg["model"]).to(device)
    state = CheckpointManager.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    coordinate_indices = list(cfg["model"].get("coordinate_indices", (0, 1)))

    for split in args.splits:
        dataset_cfg = cfg["data"].get(f"{split}_dataset")
        if dataset_cfg is None:
            print(f"No '{split}_dataset' in config, skipping split {split!r}")
            continue
        dataset_cfg = dict(dataset_cfg)
        # Force a fixed shuffle regardless of the training config's own deterministic
        # setting, so re-running this script reproduces the same figures.
        dataset_cfg["deterministic"] = True
        dataset_cfg["seed"] = args.seed
        # The distance map/normalized matrix are computed directly from coordinates below,
        # so a precomputed `soft_permutation` isn't needed -- don't require the H5 file to
        # have one.
        dataset_cfg["load_soft_permutation"] = False

        sample_ids = select_sample_ids(
            dataset_cfg["h5_path"],
            dataset_cfg["split"],
            dataset_cfg.get("dataset_name", "reflacx"),
            args.max_length,
            args.num_samples,
            args.seed,
        )
        if not sample_ids:
            print(f"No samples with n_fixations < {args.max_length} found in split {split!r}, skipping")
            continue

        dataset = build_dataset(dataset_cfg)
        id_to_index = {sid: i for i, sid in enumerate(dataset.sample_ids)}
        try:
            for sample_id in sample_ids:
                out_path = os.path.join(args.output_dir, split, f"{sample_id}.png")
                visualize_sample(
                    model,
                    dataset,
                    id_to_index[sample_id],
                    sample_id,
                    coordinate_indices,
                    args.tau,
                    args.sinkhorn_iters,
                    args.scale,
                    device,
                    out_path,
                )
                print(f"Wrote {out_path}")
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
