"""Loss-ablation experiment runner for `models.fixation_permutation_sorter.
FixationPermutationSorter`: trains and tests one model per loss configuration --

    1. only the permutation loss (Wang, Yan & Yang, ICCV 2019), against the hard target
    2. only the doubly-stochastic cross-entropy loss, against the soft ground-truth target
    3. only the Wasserstein loss
    4. only the reconstruction loss
    5. a uniform-weight (1.0 each) combination of all four

-- and appends every run's final test-set metrics as one row to a single CSV.

Each experiment reuses `configs/fixation_permutation_sorter/{data,model,metrics,trainer}.yaml`
as-is (via `train.yaml`'s `includes:`, see `utils/config.py`); only `loss_cfg` and `work_dir`
differ per run, defined in `EXPERIMENTS` below rather than as five near-duplicate config
folders. `data.yaml` already sets `load_soft_permutation: true`, so both the hard and soft
targets are available in every batch regardless of which experiment is running -- experiments
1 and 2 use `"target": "hard"`/`"soft"` (see `FixationPermutationSorter.__init__`) to pin down
which one is used rather than relying on the batch's contents alone.

    python run_loss_ablation.py
    python run_loss_ablation.py --h5-path /path/to/reflacx_data.h5 --num-epochs 5 --device cpu
"""

import argparse
import csv
import os

from trainers import build_trainer
from utils.config import Config, load_config

CONFIG_DIR = os.path.join("configs", "fixation_permutation_sorter")

EXPERIMENTS = [
    {
        "name": "permutation_hard",
        "loss_cfg": [{"type": "permutation", "weight": 1.0, "target": "hard"}],
    },
    {
        "name": "doubly_stochastic_cross_entropy_soft",
        "loss_cfg": [{"type": "doubly_stochastic_cross_entropy", "weight": 1.0, "target": "soft"}],
    },
    {
        "name": "permutation_wasserstein",
        "loss_cfg": [{"type": "permutation_wasserstein", "weight": 1.0}],
    },
    {
        "name": "permutation_reconstruction",
        "loss_cfg": [{"type": "permutation_reconstruction", "weight": 1.0}],
    },
    {
        "name": "combined_uniform",
        "loss_cfg": [
            {"type": "permutation", "weight": 1.0, "target": "hard"},
            {"type": "doubly_stochastic_cross_entropy", "weight": 1.0, "target": "soft"},
            {"type": "permutation_wasserstein", "weight": 0.1},
            {"type": "permutation_reconstruction", "weight": 0.5},
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5-path", type=str, default=None, help="Override every experiment's dataset h5_path")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override every experiment's trainer.num_epochs")
    parser.add_argument("--device", type=str, default=None, help="Override every experiment's device")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("work_dir", "loss_ablation_results.csv"),
        help="CSV path the combined results are written to",
    )
    return parser.parse_args()


def build_experiment_config(experiment, args):
    """Loads `train.yaml` fresh (so each experiment gets its own copy, safe to mutate) and
    overrides just this experiment's `loss_cfg`/`work_dir`, plus any CLI overrides."""
    cfg = dict(load_config(os.path.join(CONFIG_DIR, "train.yaml")))
    cfg["loss_cfg"] = experiment["loss_cfg"]
    cfg["work_dir"] = os.path.join("work_dir", "loss_ablation_scale20", experiment["name"])

    if args.h5_path:
        cfg["data"] = dict(cfg["data"])
        for key in ("train_dataset", "val_dataset", "test_dataset"):
            if key in cfg["data"]:
                cfg["data"][key] = dict(cfg["data"][key])
                cfg["data"][key]["h5_path"] = args.h5_path
    if args.num_epochs:
        cfg["trainer"] = dict(cfg["trainer"])
        cfg["trainer"]["num_epochs"] = args.num_epochs
    if args.device:
        cfg["device"] = args.device

    return Config(cfg)


def run_experiment(experiment, args):
    print(f"\n=== Running experiment: {experiment['name']} ===")
    cfg = build_experiment_config(experiment, args)

    trainer = build_trainer(cfg)
    trainer.build()
    trainer.train()
    test_metrics = trainer.test()

    return {"experiment": experiment["name"], **test_metrics}


def write_csv(rows, path):
    # Different experiments enable different loss terms, so their "losses" breakdown keys
    # differ (e.g. only "permutation_wasserstein" for experiment 3) -- take the union of
    # every row's keys as the CSV's columns, leaving a blank cell wherever a row lacks one.
    fieldnames = ["experiment"] + sorted({key for row in rows for key in row if key != "experiment"})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = [run_experiment(experiment, args) for experiment in EXPERIMENTS]
    write_csv(rows, args.output)
    print(f"\nWrote results for {len(rows)} experiments to {args.output}")


if __name__ == "__main__":
    main()
