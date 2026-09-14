"""Entry point: build a trainer from a config file and run training.

    cd eye-tracking-graph-v2
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml --resume work_dir/example_run/checkpoints/checkpoint_best.pt
"""

import argparse

from trainers import build_trainer
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch template entry point")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from (default: latest)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    trainer = build_trainer(cfg)
    trainer.build()

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
