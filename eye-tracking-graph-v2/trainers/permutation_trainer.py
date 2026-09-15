"""Trainer for permutation-learning models (e.g. `models.fixation_permutation_sorter.
FixationPermutationSorter`) whose datasets need a custom `collate_fn` to batch
variable-length, padded sequences (`SingleH5Dataset.collate_fn`,
`PermutedFixationH5Dataset.collate_fn`) -- a YAML config can only select a dataset by its
registered `type`, not hand `DataLoader` a Python callable, so `BaseTrainer`'s generic
`build()` (which never passes `collate_fn`) would silently fall back to the default collate
and crash on the first ragged batch. This overrides `build()` just enough to construct each
dataset first and forward its own `collate_fn` (falling back to the default when a dataset
doesn't define one, e.g. the synthetic example datasets) -- everything else, including
`ExampleTrainer`'s optional gradient clipping, is inherited unchanged.

It also lets a top-level `loss_cfg:`/`metrics_cfg:` in the full experiment config (rather
than nested inside `model:`) reach the model's constructor -- so an experiment folder can
keep "what loss(es)/metrics to use" in their own `losses.yaml`/`metrics.yaml` files, merged
in via `utils.config.load_config`'s `includes:`, separate from `model.yaml`'s architecture
settings, without the two fighting over the same key: an explicit `model.loss_cfg` (or
`metrics_cfg`) still wins if both are given.
"""

import os

from dataloaders import build_dataloader, build_dataset
from models import build_model
from trainers.evaluator import Evaluator
from trainers.example_trainer import ExampleTrainer
from trainers.inferrer import Inferrer
from trainers.optimizers.optimizers import build_optimizer
from trainers.schedulers.schedulers import build_scheduler
from utils.checkpoint import CheckpointManager


class PermutationTrainer(ExampleTrainer):
    def build(self):
        cfg = self.cfg

        model_cfg = dict(cfg["model"])
        if "loss_cfg" in cfg:
            model_cfg.setdefault("loss_cfg", cfg["loss_cfg"])
        if "metrics_cfg" in cfg:
            model_cfg.setdefault("metrics_cfg", cfg["metrics_cfg"])

        self.model = build_model(model_cfg).to(self.device)
        self.logger.info(f"Built model '{model_cfg['type']}' with {self.model.num_parameters:,} trainable params")

        train_dataset = build_dataset(cfg["data"]["train_dataset"])
        self.train_loader = build_dataloader(
            cfg["data"]["train_loader"], dataset=train_dataset, collate_fn=getattr(train_dataset, "collate_fn", None)
        )

        self.val_loader = None
        if "val_dataset" in cfg["data"]:
            val_dataset = build_dataset(cfg["data"]["val_dataset"])
            self.val_loader = build_dataloader(
                cfg["data"]["val_loader"], dataset=val_dataset, collate_fn=getattr(val_dataset, "collate_fn", None)
            )

        self.optimizer = build_optimizer(cfg["optimizer"], self.model.parameters())
        self.scheduler = build_scheduler(cfg.get("scheduler"), self.optimizer)

        self.ckpt_manager = CheckpointManager(
            save_dir=os.path.join(self.work_dir, "checkpoints"),
            max_to_keep=cfg.get("max_checkpoints", 5),
            metric_name=cfg.get("monitor_metric"),
            mode=cfg.get("monitor_mode", "min"),
        )

        if self.val_loader is not None:
            self.evaluator = Evaluator(self.model, self.val_loader, device=self.device)
        self.inferrer = Inferrer(self.model, device=self.device)

        self._built = True
        return self
