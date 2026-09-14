"""Evaluator: runs a model over a dataloader in eval mode and aggregates losses/metrics.

Relies on the model contract from `models.base_model.BaseModel`: `forward(batch)`
returns a dict optionally containing "losses" and "metrics" (both plain dicts of
scalars) whenever targets are present in the batch.
"""

import torch

from utils.meter import MetricTracker
from utils.tensor import infer_batch_size, to_device


class Evaluator:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Run one full pass over `dataloader`, returning averaged {name: value} metrics."""
        was_training = self.model.training
        self.model.eval()
        tracker = MetricTracker()

        for batch in self.dataloader:
            batch = to_device(batch, self.device)
            output = self.model(batch)
            batch_size = infer_batch_size(batch)

            for key in ("losses", "metrics"):
                if key in output:
                    tracker.update({k: v.item() if torch.is_tensor(v) else v for k, v in output[key].items()},
                                    n=batch_size)

        if was_training:
            self.model.train()
        return tracker.averages()
