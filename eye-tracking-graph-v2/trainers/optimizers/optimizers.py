"""Registry of optimizers available to trainers, built from config."""

import torch.optim as optim

from utils.registry import Registry

OPTIMIZERS = Registry("optimizers")

for _name, _cls in [
    ("sgd", optim.SGD),
    ("adam", optim.Adam),
    ("adamw", optim.AdamW),
    ("rmsprop", optim.RMSprop),
    ("adagrad", optim.Adagrad),
]:
    OPTIMIZERS.register(_name)(_cls)


def build_optimizer(cfg, params):
    """Build an optimizer from a config dict, e.g. {"type": "adamw", "lr": 1e-3}."""
    cfg = dict(cfg)
    cfg["params"] = params
    return OPTIMIZERS.build(cfg)
