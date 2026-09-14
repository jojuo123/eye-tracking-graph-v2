"""Custom loss examples, registered into the same `LOSSES` registry as the
built-in wrappers in `losses.py`. This is the pattern to copy for your own
custom loss: subclass `nn.Module`, implement `forward`, decorate with
`@LOSSES.register("name")`.
"""

import torch.nn as nn
import torch.nn.functional as F

from trainers.losses.losses import LOSSES


@LOSSES.register("focal")
class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification (Lin et al., 2017, "Focal Loss
    for Dense Object Detection").

    Down-weights easy, well-classified examples by `(1 - p_t) ** gamma` so
    training focuses on hard/misclassified ones. Useful for class-imbalanced
    classification. Reduces to (optionally class-weighted) cross-entropy when
    `gamma=0`.
    """

    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = target_log_probs.exp()

        focal_term = (1 - target_probs).pow(self.gamma)
        loss = -focal_term * target_log_probs

        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@LOSSES.register("weighted_multi")
class WeightedMultiLoss(nn.Module):
    """Combines several named, registered losses into one weighted total —
    the pattern for multi-term objectives (e.g. a VAE's
    `recon_loss + beta * kl_loss`, or blending cross-entropy with an
    auxiliary/regularizing loss).

    All sub-losses are called with the same `forward(*args, **kwargs)`, so
    this fits terms that share the same inputs (e.g. combining `cross_entropy`
    and `focal` for the same `(logits, targets)` classification pair).

    Returns a dict `{"loss": total, <name>: individual_value, ...}` so every
    term can be logged separately, e.g. in a model:

        result = self.loss_fn(logits, y)
        output["loss"] = result.pop("loss")
        output["losses"] = result

    Example config:
        loss_cfg:
          type: weighted_multi
          terms:
            ce: {type: cross_entropy, weight: 1.0}
            focal: {type: focal, weight: 0.5, gamma: 2.0}
    """

    def __init__(self, terms: dict):
        super().__init__()
        self.weights = {}
        self.losses = nn.ModuleDict()
        for name, term_cfg in terms.items():
            term_cfg = dict(term_cfg)
            self.weights[name] = term_cfg.pop("weight", 1.0)
            self.losses[name] = LOSSES.build(term_cfg)

    def forward(self, *args, **kwargs):
        total = 0.0
        breakdown = {}
        for name, loss_fn in self.losses.items():
            value = loss_fn(*args, **kwargs)
            breakdown[name] = value.detach()
            total = total + self.weights[name] * value
        breakdown["loss"] = total
        return breakdown
