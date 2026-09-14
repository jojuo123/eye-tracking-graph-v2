"""Registry of loss functions available to trainers/models.

Many models (see `models/example_model.py`) compute their loss inline with
`torch.nn.functional`, which is perfectly fine for simple cases. This registry
exists for the common alternative pattern: a model returns raw "outputs" and
the trainer/model builds a configurable loss (e.g. to sweep loss types from
a config file without touching model code).
"""

import torch.nn as nn

from utils.registry import Registry

LOSSES = Registry("losses")

LOSSES.register("cross_entropy")(nn.CrossEntropyLoss)
LOSSES.register("mse")(nn.MSELoss)
LOSSES.register("l1")(nn.L1Loss)
LOSSES.register("smooth_l1")(nn.SmoothL1Loss)
LOSSES.register("bce_with_logits")(nn.BCEWithLogitsLoss)
LOSSES.register("nll")(nn.NLLLoss)
LOSSES.register("kl_div")(nn.KLDivLoss)


def build_loss(cfg):
    """Build a loss module from a config dict, e.g. {"type": "cross_entropy", "label_smoothing": 0.1}."""
    return LOSSES.build(cfg)
