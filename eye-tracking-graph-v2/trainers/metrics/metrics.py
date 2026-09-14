"""Registry of standard metrics, mirroring the pattern in `trainers/losses/losses.py`.

Each metric is a small callable class: `metric(outputs, targets) -> scalar tensor`
(detached, ready for logging). Unlike losses, these never need gradients.
"""

import torch

from utils.registry import Registry

METRICS = Registry("metrics")


@METRICS.register("accuracy")
class Accuracy:
    """Multiclass classification accuracy: `argmax(outputs) == targets`."""

    @torch.no_grad()
    def __call__(self, outputs, targets):
        preds = outputs.argmax(dim=-1)
        return (preds == targets).float().mean()


@METRICS.register("precision")
class Precision:
    """Precision per class, averaged. `average="macro"` (default) returns the
    unweighted mean over classes; `average="none"` returns one score per class."""

    def __init__(self, num_classes, average="macro", eps=1e-8):
        self.num_classes = num_classes
        self.average = average
        self.eps = eps

    @torch.no_grad()
    def __call__(self, outputs, targets):
        preds = outputs.argmax(dim=-1)
        scores = torch.zeros(self.num_classes, device=outputs.device)
        for c in range(self.num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fp = ((preds == c) & (targets != c)).sum().float()
            scores[c] = tp / (tp + fp + self.eps)
        return scores.mean() if self.average == "macro" else scores


@METRICS.register("recall")
class Recall:
    """Recall per class, averaged. `average="macro"` (default) returns the
    unweighted mean over classes; `average="none"` returns one score per class."""

    def __init__(self, num_classes, average="macro", eps=1e-8):
        self.num_classes = num_classes
        self.average = average
        self.eps = eps

    @torch.no_grad()
    def __call__(self, outputs, targets):
        preds = outputs.argmax(dim=-1)
        scores = torch.zeros(self.num_classes, device=outputs.device)
        for c in range(self.num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fn = ((preds != c) & (targets == c)).sum().float()
            scores[c] = tp / (tp + fn + self.eps)
        return scores.mean() if self.average == "macro" else scores


@METRICS.register("f1")
class F1Score:
    """Macro-averaged F1 = harmonic mean of per-class precision and recall."""

    def __init__(self, num_classes, average="macro", eps=1e-8):
        self.precision = Precision(num_classes, average="none", eps=eps)
        self.recall = Recall(num_classes, average="none", eps=eps)
        self.average = average
        self.eps = eps

    @torch.no_grad()
    def __call__(self, outputs, targets):
        p = self.precision(outputs, targets)
        r = self.recall(outputs, targets)
        f1 = 2 * p * r / (p + r + self.eps)
        return f1.mean() if self.average == "macro" else f1


@METRICS.register("mae")
class MeanAbsoluteError:
    """Mean absolute error, for regression tasks."""

    @torch.no_grad()
    def __call__(self, outputs, targets):
        return (outputs - targets).abs().mean()


@METRICS.register("mse")
class MeanSquaredErrorMetric:
    """Mean squared error, for regression tasks (as a metric, not a loss)."""

    @torch.no_grad()
    def __call__(self, outputs, targets):
        return torch.mean((outputs - targets) ** 2)
