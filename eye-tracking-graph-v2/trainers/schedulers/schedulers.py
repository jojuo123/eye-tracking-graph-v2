"""Registry of LR schedulers available to trainers, built from config."""

import torch.optim.lr_scheduler as lr_scheduler

from utils.registry import Registry

SCHEDULERS = Registry("schedulers")

for _name, _cls in [
    ("step_lr", lr_scheduler.StepLR),
    ("multi_step", lr_scheduler.MultiStepLR),
    ("cosine", lr_scheduler.CosineAnnealingLR),
    ("exponential", lr_scheduler.ExponentialLR),
    ("reduce_on_plateau", lr_scheduler.ReduceLROnPlateau),
]:
    SCHEDULERS.register(_name)(_cls)


def build_scheduler(cfg, optimizer):
    """Build an LR scheduler from a config dict, e.g. {"type": "cosine", "T_max": 10}.
    Returns None if `cfg` is None (no scheduler configured)."""
    if cfg is None:
        return None
    cfg = dict(cfg)
    cfg["optimizer"] = optimizer
    return SCHEDULERS.build(cfg)
