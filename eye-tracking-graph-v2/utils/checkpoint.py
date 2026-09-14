"""Checkpoint (weight file) management: saving, pruning old checkpoints, tracking the best one."""

import glob
import os

import torch


class CheckpointManager:
    """Saves training state dicts to disk, keeps only the `max_to_keep` most recent
    ones, and separately tracks/keeps a `checkpoint_best.pt` based on a monitored metric.
    """

    def __init__(self, save_dir, max_to_keep=5, metric_name=None, mode="min"):
        assert mode in ("min", "max")
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self.metric_name = metric_name
        self.mode = mode
        os.makedirs(save_dir, exist_ok=True)

        self._saved = []
        self._best_value = None

    def save(self, step, state, metric_value=None, tag=None):
        """Persist `state` (any picklable dict, typically from `trainer.state_dict()`) to disk."""
        fname = f"checkpoint_{tag if tag is not None else step}.pt"
        path = os.path.join(self.save_dir, fname)
        torch.save(state, path)
        self._saved.append(path)
        self._prune()

        if metric_value is not None:
            is_best = self._best_value is None or (
                metric_value < self._best_value if self.mode == "min" else metric_value > self._best_value
            )
            if is_best:
                self._best_value = metric_value
                torch.save(state, os.path.join(self.save_dir, "checkpoint_best.pt"))
        return path

    def _prune(self):
        if self.max_to_keep is None:
            return
        while len(self._saved) > self.max_to_keep:
            old_path = self._saved.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

    def latest(self):
        """Path to the most recently saved (non-"best") checkpoint, or None."""
        ckpts = glob.glob(os.path.join(self.save_dir, "checkpoint_*.pt"))
        ckpts = [c for c in ckpts if os.path.basename(c) != "checkpoint_best.pt"]
        if not ckpts:
            return None
        return max(ckpts, key=os.path.getmtime)

    def best(self):
        path = os.path.join(self.save_dir, "checkpoint_best.pt")
        return path if os.path.exists(path) else None

    @staticmethod
    def load(path, map_location="cpu"):
        return torch.load(path, map_location=map_location)
