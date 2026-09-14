"""BaseTrainer: wires together model, data, optimizer, scheduler, checkpointing,
evaluation and inference from a single config dict.

Usage:
    trainer = BaseTrainer(cfg)   # or any subclass, or `build_trainer(cfg)`
    trainer.build()
    trainer.train()
"""

import os
import time

import torch

from dataloaders import build_dataloader
from models import build_model
from trainers.evaluator import Evaluator
from trainers.inferrer import Inferrer
from trainers.optimizers.optimizers import build_optimizer
from trainers.schedulers.schedulers import build_scheduler
from utils.checkpoint import CheckpointManager
from utils.logger import get_logger
from utils.meter import MetricTracker
from utils.seed import set_seed
from utils.tensor import infer_batch_size, to_device


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.work_dir = cfg.get("work_dir", "./work_dir")
        os.makedirs(self.work_dir, exist_ok=True)

        self.logger = get_logger("trainer", log_file=os.path.join(self.work_dir, "train.log"))
        set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", False))

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.evaluator = None
        self.inferrer = None
        self.ckpt_manager = None

        self.epoch = 0
        self.global_step = 0
        self._built = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def build(self):
        """Construct model, dataloaders, optimizer, scheduler, checkpoint manager,
        evaluator and inferrer from `self.cfg`. Must be called before `train()`."""
        cfg = self.cfg

        self.model = build_model(cfg["model"]).to(self.device)
        self.logger.info(f"Built model '{cfg['model']['type']}' with {self.model.num_parameters:,} trainable params")

        self.train_loader = build_dataloader(cfg["data"]["train_loader"], dataset_cfg=cfg["data"]["train_dataset"])
        self.val_loader = None
        if "val_dataset" in cfg["data"]:
            self.val_loader = build_dataloader(cfg["data"]["val_loader"], dataset_cfg=cfg["data"]["val_dataset"])

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

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self):
        assert self._built, "Call trainer.build() before trainer.train()"
        num_epochs = self.cfg["trainer"].get("num_epochs", 10)
        log_interval = self.cfg["trainer"].get("log_interval", 50)

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            self._train_one_epoch(log_interval)
            self._step_scheduler_post_epoch()
            self._validate_and_checkpoint(epoch)

        self.logger.info("Training finished.")

    def _train_one_epoch(self, log_interval):
        self.model.train()
        tracker = MetricTracker()
        start = time.time()

        for i, batch in enumerate(self.train_loader):
            batch = to_device(batch, self.device)
            output = self.model(batch)
            loss = output["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self._optimizer_step()
            self.global_step += 1

            batch_size = infer_batch_size(batch)
            log_dict = {"loss": loss.item()}
            log_dict.update({k: v.item() for k, v in output.get("losses", {}).items()})
            log_dict.update({k: v.item() for k, v in output.get("metrics", {}).items()})
            tracker.update(log_dict, n=batch_size)

            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start
                self.logger.info(
                    f"[Epoch {self.epoch}][{i + 1}/{len(self.train_loader)}] {tracker} | {elapsed:.1f}s"
                )

    def _optimizer_step(self):
        """Extension point: override to inject e.g. gradient clipping (see
        `trainers/example_trainer.py`) before the actual optimizer step."""
        self.optimizer.step()

    def _step_scheduler_post_epoch(self):
        if self.scheduler is None or isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return  # ReduceLROnPlateau is stepped in `_validate_and_checkpoint` with a metric value
        self.scheduler.step()

    def _validate_and_checkpoint(self, epoch):
        metric_value = None
        if self.evaluator is not None:
            val_metrics = self.evaluator.evaluate()
            self.logger.info(f"[Epoch {epoch}] val: " + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

            monitor = self.cfg.get("monitor_metric")
            if monitor and monitor in val_metrics:
                metric_value = val_metrics[monitor]

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metric_value if metric_value is not None else 0.0)

        self.ckpt_manager.save(step=epoch, state=self.state_dict(), metric_value=metric_value, tag=f"epoch{epoch}")

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def state_dict(self):
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def load_state_dict(self, state):
        self.epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])

    def resume(self, path=None):
        """Resume from an explicit checkpoint path, or the latest one if not given."""
        path = path or self.ckpt_manager.latest()
        if path is None:
            self.logger.info("No checkpoint found, starting from scratch.")
            return
        self.logger.info(f"Resuming from checkpoint: {path}")
        state = CheckpointManager.load(path, map_location=self.device)
        self.load_state_dict(state)
