"""Base class contract for all models in this template."""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Every model composes building blocks from `modules/` and owns its own
    forward pass, loss computation and metrics.

    Contract: `forward(batch)` receives a batch (typically a dict, e.g. produced
    by a dataset in `dataloaders/`) and must return a dict with, depending on
    whether targets are available:
      - "outputs": raw predictions/logits (always present; used at inference time)
      - "loss":    scalar total loss the trainer will call `.backward()` on
      - "losses":  dict of individual (detached) loss components, for logging
      - "metrics": dict of scalar (detached) metrics (accuracy, F1, ...), for logging

    Keeping loss/metric computation inside the model (rather than the trainer)
    means the trainer loop stays generic across very different tasks.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch):
        ...

    def compute_metrics(self, outputs, targets):
        """Override to compute task-specific metrics. Must return a plain dict
        of detached scalar tensors or floats."""
        return {}

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
