"""Scalar tracking helpers for logging losses/metrics during training and evaluation."""

from collections import defaultdict


class AverageMeter:
    """Tracks a running average of a single scalar (e.g. one loss or metric)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class MetricTracker:
    """Tracks a dict of named `AverageMeter`s, e.g. {"loss": ..., "accuracy": ...}."""

    def __init__(self):
        self._meters = defaultdict(AverageMeter)

    def update(self, metrics: dict, n=1):
        for k, v in metrics.items():
            self._meters[k].update(float(v), n)

    def reset(self):
        for m in self._meters.values():
            m.reset()

    def averages(self) -> dict:
        return {k: m.avg for k, m in self._meters.items()}

    def __str__(self):
        return " | ".join(f"{k}: {m.avg:.4f}" for k, m in self._meters.items())
