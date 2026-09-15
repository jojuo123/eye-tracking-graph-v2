"""Entry point: build a trainer from a config file and run training, validation, and testing.

Training already validates after every epoch (`BaseTrainer._validate_and_checkpoint`,
against `cfg["data"]["val_dataset"]`) and checkpoints the best epoch by `monitor_metric`.
This script additionally runs a final pass over `cfg["data"]["test_dataset"]` (if the config
defines one) once training finishes, via `BaseTrainer.test()`.

    cd eye-tracking-graph-v2
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml --resume work_dir/example_run/checkpoints/checkpoint_best.pt

    # Skip the post-training test pass:
    python main.py --config configs/default.yaml --skip-test

    # Only test (no training) -- loads the best checkpoint by default:
    python main.py --config configs/default.yaml --test-only
    python main.py --config configs/default.yaml --test-only --resume path/to/checkpoint.pt
"""

import argparse

from trainers import build_trainer
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch template entry point")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a YAML config file")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from, or 'best' for the tracked best checkpoint. "
        "Training defaults to starting from scratch if omitted; --test-only defaults to 'best'.",
    )
    parser.add_argument("--test-only", action="store_true", help="Skip training; only run the final test pass")
    parser.add_argument("--skip-test", action="store_true", help="Skip the test pass after training")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    trainer = build_trainer(cfg)
    trainer.build()

    if args.test_only:
        loaded = trainer.resume(args.resume or "best")
        if not loaded:
            raise SystemExit(
                "--test-only requires an existing checkpoint, but none was found in "
                f"{trainer.ckpt_manager.save_dir}. Pass --resume <path> to point at one explicitly."
            )
    elif args.resume:
        trainer.resume(args.resume)

    if not args.test_only:
        trainer.train()  # validates against cfg["data"]["val_dataset"] after every epoch

    if not args.skip_test:
        trainer.test()  # evaluates cfg["data"]["test_dataset"], if the config defines one


if __name__ == "__main__":
    main()
