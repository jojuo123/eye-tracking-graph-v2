"""Scores a trained `FixationPermutationSorter`'s predicted fixation orderings with the
shortlisted `[0, 1]` metrics in `normalized_metrics.py`, side by side with two
reference orderings of the very same fixations:

    random         -- uniformly random order (chance level; mean over `--n-random`
                      draws per sample)
    top_to_bottom  -- fixations sorted by y, then x: a model-free "reading order"
                      heuristic, i.e. what a model gets for free from layout alone

For each sample, the fixation sequence is shuffled exactly as
`dataloaders.permuted_h5_dataset.PermutedFixationH5Dataset` does at eval time
(`deterministic=True`, `--seed`; samples are read straight from the H5 file by
`iter_samples` rather than through that class, so this also works on older H5 layouts
such as the repo's local smoke-test file), passed through the model, and the predicted
ordering is compared back to the true temporal order. A metric "sees" the model if
the model's scores sit clearly below `random` (paired Wilcoxon signed-rank test over
samples, one-sided), and is informative beyond layout if they also sit below
`top_to_bottom`.

Writes `per_sample.csv`, `summary.json` and `model_vs_baselines.png` to
`--output-dir`.

    cd eye-tracking-graph-v2/metric-evaluation
    python run_model_evaluation.py \\
        --checkpoint ../work_dir/loss_ablation/doubly_stochastic_cross_entropy_soft/checkpoints/checkpoint_best.pt

Needs torch/h5py/scipy/matplotlib (the main project's dependencies), unlike the
numpy-only metric modules in this directory.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import zlib

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from models import build_model  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.config import load_config  # noqa: E402

from normalized_metrics import ALL_METRICS, METRIC_GROUPS, evaluate_pair  # noqa: E402
from run_damage_evaluation import (  # noqa: E402
    AXIS_LINE,
    PAGE,
    PALETTE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _default_worker_count,
    _style_axes,
)

METHODS = ["model", "top_to_bottom", "random"]


@torch.no_grad()
def predict_order(model, item, device):
    """Runs one `PermutedFixationH5Dataset` item through `model` and returns the
    predicted ordering as a permutation of the *ground-truth* indices: `order[k]` = the
    GT index of the fixation the model places at position `k` (so `coords_gt[order]`
    is the predicted scanpath, and `order` is the identity iff the model is exact)."""
    batch = {
        "image": item["image"].unsqueeze(0).to(device),
        "fixations": item["fixations"].unsqueeze(0).to(device),
        "fixations_mask": torch.ones(1, item["fixations"].shape[0], dtype=torch.bool, device=device),
    }
    predicted = model(batch)["outputs"][0].cpu().numpy()  # predicted `permutation`
    shuffle = item["permutation"].numpy()  # shuffled[i] = gt[shuffle[i]]
    # The model's hard permutation follows the same convention as `item["permutation"]`,
    # so its inverse gathers the shuffled sequence back into predicted order (see
    # `visualize_permutation.gather_by_inverse`); composing with `shuffle` expresses
    # that in GT indices.
    return shuffle[np.argsort(predicted)]


def iter_samples(h5_path, split, dataset_name, fixation_columns, seed, max_samples=None, sample_seed=0):
    """Yields one `PermutedFixationH5Dataset`-style item per sample (`id`, `image`,
    shuffled `fixations`, `n_fixations`, `permutation`, `inverse_permutation`) -- same
    image scaling (`/ 255`) and same per-sample shuffle (`_rng_for`'s crc32-seeded RNG)
    as that class with `deterministic=True`, but reading only the `image`/`fixations`
    groups, so it doesn't depend on the H5 file's metadata layout. `max_samples`: if
    set, only a random subset of that many samples, drawn with `sample_seed` exactly as
    `run_damage_evaluation.load_local_reflacx_sequences` draws it -- so the same
    `--max-samples`/seed in both scripts selects the same samples."""
    with h5py.File(h5_path, "r") as f:
        group = f[split][dataset_name]
        sample_ids = sorted(group.keys())
        if max_samples is not None and len(sample_ids) > max_samples:
            rng = np.random.default_rng(sample_seed)
            sample_ids = sorted(rng.choice(sample_ids, size=max_samples, replace=False).tolist())
        for sample_id in sample_ids:
            image = torch.from_numpy(group[sample_id]["image"][()].astype(np.float32) / 255.0)
            fx = group[sample_id]["fixations"]
            fixations = np.stack([fx[col][()] for col in fixation_columns], axis=-1).astype(np.float32)
            n = fixations.shape[0]
            rng = np.random.default_rng((zlib.crc32(sample_id.encode("utf-8")) ^ seed) & 0xFFFFFFFF)
            permutation = rng.permutation(n)
            yield {
                "id": sample_id,
                "image": image,
                "fixations": torch.from_numpy(fixations[permutation]),
                "n_fixations": n,
                "permutation": torch.from_numpy(permutation.astype(np.int64)),
                "inverse_permutation": torch.from_numpy(np.argsort(permutation).astype(np.int64)),
            }


def top_to_bottom_order(coords):
    return np.lexsort((coords[:, 0], coords[:, 1]))


def collect(model, samples, coordinate_indices, device, n_random, seed, max_fixations, metric_kwargs,
            coord_scale=1.0, workers=1):
    """Returns `rows`: one dict per (sample, method, metric, param) with its value
    (for `random`, the mean over `n_random` draws). The model sees the coordinates as
    stored; the metrics see them divided by `coord_scale`, since their bounds and
    `eps`/grid defaults assume `[0, 1]`.

    Two phases: the model predicts every sample's order in this process (one forward
    pass each, cheap), then the per-sample metric scoring -- the expensive, pure-Python
    part -- runs on `workers` processes. Each sample's random baseline draws from its
    own generator (`[seed, sample_index]`), so results don't depend on `workers`."""
    tasks = []
    for item in samples:
        n = item["n_fixations"]
        if max_fixations is not None and n > max_fixations:
            print(f"  skip {item['id']} (n={n} > --max-fixations {max_fixations})")
            continue
        shuffled = item["fixations"][:, coordinate_indices].numpy().astype(np.float64)
        coords = shuffled[item["inverse_permutation"].numpy()] / coord_scale  # true temporal order
        orders = {"model": predict_order(model, item, device), "top_to_bottom": top_to_bottom_order(coords)}
        tasks.append((len(tasks), item["id"], coords, orders, n_random, seed, metric_kwargs))
    print(f"Predicted orders for {len(tasks)} samples; scoring on {workers} worker(s)")

    if workers > 1:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            per_sample = list(pool.imap(_score_sample, tasks, chunksize=1))
    else:
        per_sample = [_score_sample(t) for t in tasks]

    rows = []
    for sample_rows, line in per_sample:
        print(line)
        rows.extend(sample_rows)
    return rows


def _score_sample(task):
    """Scores one sample's model and top-to-bottom orders, plus `n_random` random
    orders (averaged), against its true order. Runs in a worker process under
    `--workers > 1`. Returns `(rows, progress_line)`."""
    sample_index, sample_id, coords, orders, n_random, seed, metric_kwargs = task
    n = coords.shape[0]
    rng = np.random.default_rng([seed, sample_index])
    per_method = {name: evaluate_pair(coords, order, **metric_kwargs) for name, order in orders.items()}

    random_vals = [evaluate_pair(coords, rng.permutation(n), **metric_kwargs) for _ in range(n_random)]
    per_method["random"] = {
        m: {label: float(np.nanmean([rv[m][label] for rv in random_vals])) for label in random_vals[0][m]}
        for m in ALL_METRICS
    }

    exact = float(np.mean(orders["model"] == np.arange(n)))
    line = (f"  {sample_id}: n={n}, model exact-position accuracy={exact:.3f}, "
            f"kendall model/random = {per_method['model']['kendall_tau']['default']:.3f}/"
            f"{per_method['random']['kendall_tau']['default']:.3f}")

    rows = []
    for method, vals in per_method.items():
        for m in ALL_METRICS:
            for label, v in vals[m].items():
                rows.append({"sample": sample_id, "n": n, "method": method, "metric": m, "param": label,
                             "value": float(v)})
    return rows, line


def summarize(rows):
    """`{metric: {param: {...}}}` with per-method mean/std, the model's score as a
    fraction of the random baseline's, and one-sided paired Wilcoxon p-values for
    model < random and model < top_to_bottom."""
    table = {}
    for r in rows:
        table.setdefault(r["metric"], {}).setdefault(r["param"], {}).setdefault(r["method"], {})[r["sample"]] = r["value"]

    summary = {}
    for metric in ALL_METRICS:
        for param, by_method in table.get(metric, {}).items():
            samples = sorted(by_method["model"])
            vecs = {meth: np.array([by_method[meth][s] for s in samples]) for meth in METHODS}
            entry = {meth: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))} for meth, v in vecs.items()}
            rand_mean = entry["random"]["mean"]
            entry["model_over_random"] = entry["model"]["mean"] / rand_mean if rand_mean > 0 else float("nan")
            for ref in ("random", "top_to_bottom"):
                diff = vecs["model"] - vecs[ref]
                diff = diff[~np.isnan(diff)]
                if diff.size == 0 or np.all(diff == 0):
                    p = float("nan")
                else:
                    p = float(wilcoxon(diff, alternative="less").pvalue)
                entry[f"p_model_lt_{ref}"] = p
                entry[f"frac_samples_model_lt_{ref}"] = float(np.mean(diff < 0)) if diff.size else float("nan")
            entry["n_samples"] = len(samples)
            summary.setdefault(metric, {})[param] = entry
    return summary, table


def plot(summary, table, out_path, title):
    """One row per (metric, param), x = `[0, 1]` dissimilarity: per-sample values as
    faint dots, method mean as a solid marker -- so both the gap between methods and
    how consistent it is across samples are visible."""
    labels, keys = [], []
    for group_title, metrics in METRIC_GROUPS:
        for m in metrics:
            for param in summary.get(m, {}):
                labels.append(m if param == "default" else f"{m} ({param})")
                keys.append((m, param))

    colors = {"model": PALETTE[0], "top_to_bottom": PALETTE[3], "random": PALETTE[1]}
    offsets = {"model": -0.22, "top_to_bottom": 0.0, "random": 0.22}

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(keys) + 1.6))
    fig.patch.set_facecolor(PAGE)
    _style_axes(ax)
    for row, (m, param) in enumerate(keys):
        for meth in METHODS:
            vals = np.array(list(table[m][param][meth].values()))
            y = row + offsets[meth]
            ax.scatter(vals, np.full_like(vals, y), s=9, color=colors[meth], alpha=0.25, linewidths=0, zorder=2)
            ax.scatter([summary[m][param][meth]["mean"]], [y], s=42, color=colors[meth], zorder=3,
                       edgecolors="white", linewidths=0.8, label=meth if row == 0 else None)
        p = summary[m][param]["p_model_lt_random"]
        if not np.isnan(p) and p < 0.05:
            ax.text(1.015, row, "*" if p >= 0.01 else "**", va="center", fontsize=9, color=TEXT_SECONDARY,
                    transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=7.5, color=TEXT_SECONDARY)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("dissimilarity to ground-truth order (0 = identical)", fontsize=8, color=TEXT_SECONDARY)
    ax.axvline(0, color=AXIS_LINE, linewidth=0.8)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=TEXT_SECONDARY, loc="lower right")
    ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
    fig.text(0.01, 0.005, "* / ** : model < random, one-sided paired Wilcoxon p < 0.05 / 0.01",
             fontsize=7, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "fixation_permutation_sorter",
                                                         "train.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5-path", default=os.path.join(PROJECT_ROOT, "..", "reflacx_data.h5"))
    parser.add_argument("--split", default="TRAIN", help="H5 split to evaluate (the local smoke file only has TRAIN)")
    parser.add_argument("--seed", type=int, default=0, help="per-sample input shuffle seed, as at eval time")
    parser.add_argument("--n-random", type=int, default=20, help="random-order draws per sample for the baseline")
    parser.add_argument("--max-fixations", type=int, default=None, help="skip samples longer than this")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="random subset of this many samples (same draw as run_damage_evaluation.py)")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="seed for the --max-samples draw (match run_damage_evaluation.py's --seed)")
    parser.add_argument("--coord-scale", type=float, default=1.0,
                        help="divide stored coordinates by this for the metrics (20 for *_scaled20.h5 files)")
    parser.add_argument("--window-fracs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--eps-values", type=float, nargs="+", default=[0.025, 0.05, 0.1])
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes for the metric scoring (default 1); 0 = every core this job has")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "work_dir", "metric_model_evaluation"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg["model"]).to(device)
    state = CheckpointManager.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    coordinate_indices = list(cfg["model"].get("coordinate_indices", (0, 1)))

    dataset_cfg = cfg["data"]["test_dataset"]
    samples = iter_samples(args.h5_path, args.split, dataset_cfg.get("dataset_name", "reflacx"),
                           dataset_cfg["fixation_columns"], args.seed, args.max_samples, args.sample_seed)
    print(f"Evaluating {args.checkpoint} (epoch {state.get('epoch')}) on {args.h5_path}:{args.split}")

    metric_kwargs = dict(window_fracs=args.window_fracs, eps_values=args.eps_values, grid_sizes=args.grid_sizes)
    workers = args.workers if args.workers > 0 else _default_worker_count()
    rows = collect(model, samples, coordinate_indices, device, args.n_random, args.seed, args.max_fixations,
                   metric_kwargs, args.coord_scale, workers)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "per_sample.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary, table = summarize(rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "h5_path": args.h5_path, "split": args.split,
                   "metrics": summary}, f, indent=2)

    print(f"\n{'metric':<34s} {'model':>7s} {'t2b':>7s} {'random':>7s} {'mod/rnd':>8s} {'p<rnd':>8s} {'p<t2b':>8s}")
    for metric, by_param in summary.items():
        for param, e in by_param.items():
            name = metric if param == "default" else f"{metric} ({param})"
            print(f"{name:<34s} {e['model']['mean']:7.3f} {e['top_to_bottom']['mean']:7.3f} "
                  f"{e['random']['mean']:7.3f} {e['model_over_random']:8.3f} "
                  f"{e['p_model_lt_random']:8.4f} {e['p_model_lt_top_to_bottom']:8.4f}")

    ckpt_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint))))
    out_png = os.path.join(args.output_dir, "model_vs_baselines.png")
    plot(summary, table, out_png,
         f"{ckpt_name}: predicted vs. GT fixation order ({summary['kendall_tau']['default']['n_samples']} samples, "
         f"{args.split})")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
