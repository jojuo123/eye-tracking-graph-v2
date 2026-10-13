"""Overlays a trained model's scores (from `run_model_evaluation.py`) on the synthetic
damage curves (from `run_damage_evaluation.py`) for the same samples: one figure per
damage type, one small panel per metric (and parameter value), showing the damage
curve (mean +/- 1 std over samples and repeats vs. severity) with the model's mean
score drawn as a horizontal line -- so the model's predicted orderings can be read off
as "about as wrong as damage type X at severity s". The random-order baseline is drawn
too, as a fainter dotted horizontal line, to mark chance level.

Where the model line crosses the damage curve, the panel is annotated with the
(linearly interpolated) equivalent severity.

    python plot_damage_vs_model.py \\
        --damage-dir ../work_dir/metric_damage_curves_scaled20_test \\
        --model-dir ../work_dir/metric_model_evaluation_scaled20_test

Both directories should come from the same H5 file/split and the same samples
(`--max-samples`/seed, `--max-fixations 0` on the damage side) -- checked against the
damage run's `samples.json` when present.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from normalized_metrics import METRIC_GROUPS
from run_damage_evaluation import (
    AXIS_LINE, DIRECTION_ARROW, DIRECTION_TEXT, HIGHER, LOWER, METRIC_DIRECTION, PAGE, PALETTE, TEXT_MUTED,
    TEXT_PRIMARY, TEXT_SECONDARY, ZERO, _style_axes,
)

N_COLS = 6
MODEL_COLOR = TEXT_PRIMARY


def panel_keys(summary):
    """`[(group_index, metric, param), ...]` in `METRIC_GROUPS` order."""
    keys = []
    for gi, (_, metrics) in enumerate(METRIC_GROUPS):
        for m in metrics:
            for param in summary.get(m, {}):
                keys.append((gi, m, param))
    return keys


def equivalent_severity(xs, ys, target):
    """First severity at which the (piecewise-linear) damage curve reaches `target`,
    or None if it never does."""
    for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return None


def plot_perturbation(pert_name, pert_results, summary, out_path, subtitle):
    keys = panel_keys(summary)
    n_rows = int(np.ceil(len(keys) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(3.1 * N_COLS, 2.5 * n_rows))
    fig.patch.set_facecolor(PAGE)
    axes = axes.ravel()

    for ax, (gi, metric, param) in zip(axes, keys):
        _style_axes(ax)
        color = PALETTE[gi]
        series = pert_results[metric][param]
        xs = np.array([r[0] for r in series])
        ys = np.array([r[1] for r in series])
        sd = np.array([r[2] for r in series])
        model = summary[metric][param]["model"]["mean"]
        random = summary[metric][param]["random"]["mean"]

        ax.fill_between(xs, ys - sd, ys + sd, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=3, label="damage curve")
        ax.axhline(model, color=MODEL_COLOR, linewidth=1.5, label="model", zorder=3)
        # Drawn over the model line: the two often coincide, and this keeps chance visible.
        ax.axhline(random, color="#b0afa8", linewidth=1.6, linestyle=(0, (1.5, 1.5)), label="random order",
                   zorder=4)

        sev = equivalent_severity(xs, ys, model)
        note = f"model {model:.3f}" + (f"  (≈ severity {sev:.2f})" if sev is not None else "")
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=6.5, va="top", color=TEXT_SECONDARY)

        top = max(np.nanmax(ys + sd), model, random)
        ax.set_ylim(0, top * 1.18 if top > 0 else 1)
        ax.set_xlim(-0.02, 1.02)
        title = metric if param == "default" else f"{metric} ({param})"
        ax.set_title(f"{DIRECTION_ARROW[METRIC_DIRECTION[metric]]} {title}", fontsize=8, color=TEXT_PRIMARY,
                     loc="left")
        ax.tick_params(labelsize=6.5)
    for ax in axes[len(keys):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, frameon=False, labelcolor=TEXT_SECONDARY)
    fig.suptitle(f"Damage type: {pert_name}", fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
    directions = {METRIC_DIRECTION[metric] for _, metric, _ in keys}
    legend = "   ".join(f"{DIRECTION_ARROW[d]} = {DIRECTION_TEXT[d]}" for d in (HIGHER, LOWER, ZERO) if d in directions)
    fig.text(0.01, 0.955, subtitle + "  ·  x: severity level, y: dissimilarity (0 = identical order; "
             "each panel has its own y-scale)  ·  " + legend, fontsize=8, color=TEXT_SECONDARY, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--damage-dir", required=True, help="run_damage_evaluation.py --output-dir")
    parser.add_argument("--model-dir", required=True, help="run_model_evaluation.py --output-dir")
    parser.add_argument("--output-dir", default=None, help="default: <model-dir>/damage_vs_model")
    args = parser.parse_args()

    with open(os.path.join(args.damage_dir, "results.json")) as f:
        damage = json.load(f)
    with open(os.path.join(args.model_dir, "summary.json")) as f:
        model_summary = json.load(f)
    summary = model_summary["metrics"]

    samples_path = os.path.join(args.damage_dir, "samples.json")
    if os.path.exists(samples_path):
        with open(samples_path) as f:
            damage_ids = set(json.load(f)["sample_ids"])
        with open(os.path.join(args.model_dir, "per_sample.csv")) as f:
            model_ids = {row["sample"] for row in csv.DictReader(f)}
        if damage_ids != model_ids:
            print(f"WARNING: damage run has {len(damage_ids)} samples, model run {len(model_ids)}, "
                  f"{len(damage_ids & model_ids)} in common -- curves and model lines aren't on the same samples")

    out_dir = args.output_dir or os.path.join(args.model_dir, "damage_vs_model")
    os.makedirs(out_dir, exist_ok=True)
    n = next(iter(next(iter(summary.values())).values()))["n_samples"]
    subtitle = f"{os.path.basename(model_summary['h5_path'])}:{model_summary['split']}, {n} samples"
    for pert_name, pert_results in damage.items():
        out_path = os.path.join(out_dir, f"{pert_name}.png")
        plot_perturbation(pert_name, pert_results, summary, out_path, subtitle)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
