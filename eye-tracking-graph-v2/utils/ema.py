"""Exponential Moving Average of model weights — an example of an "algorithm" helper
that doesn't belong to any single model/trainer but is broadly reusable."""

import copy

import torch


class ModelEMA:
    """Maintains a shadow copy of a model whose weights are an exponential moving
    average of the online model's weights. Often improves eval stability.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        for batch in loader:
            ...
            optimizer.step()
            ema.update(model)
        # use ema.ema for evaluation/inference
    """

    def __init__(self, model, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.ema.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

        ema_buffers = dict(self.ema.named_buffers())
        for name, b in model.named_buffers():
            ema_buffers[name].copy_(b)

    def state_dict(self):
        return self.ema.state_dict()
