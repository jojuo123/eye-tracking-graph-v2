"""Example model: a small MLP classifier, wired up to the `models.MODELS` registry.

This is the pattern to copy for a new model:
  1. Subclass `BaseModel`.
  2. Build the architecture out of `modules/` building blocks.
  3. In `forward`, return a dict with "outputs" and, if targets are present,
     "loss" / "losses" / "metrics".
  4. Register it with `@MODELS.register("name")` so it can be built from a config.

The loss is built via `loss_cfg` through the `LOSSES` registry (see
`trainers/losses/`) instead of being hardcoded, so swapping in a custom loss
-- e.g. `FocalLoss` from `trainers/losses/custom_losses.py` -- is a config
change, not a code change. See `configs/focal_loss_example.yaml`.

Likewise, `metrics_cfg` builds a `MetricCollection` through the `METRICS`
registry (see `trainers/metrics/`), defaulting to plain accuracy. See
`configs/custom_metrics_example.yaml` for logging precision/recall/f1/top-k
accuracy alongside it, again with no code change.
"""

from models import MODELS
from models.base_model import BaseModel
from modules.mlp import MLP
from trainers.losses import build_loss
from trainers.metrics import MetricCollection


@MODELS.register("mlp_classifier")
class MLPClassifier(BaseModel):
    def __init__(self, in_dim, hidden_dims, num_classes, dropout=0.1, loss_cfg=None, metrics_cfg=None):
        super().__init__()
        self.backbone = MLP(in_dim, hidden_dims, num_classes, dropout=dropout)
        self.loss_fn = build_loss(loss_cfg or {"type": "cross_entropy"})
        self.metrics = MetricCollection(metrics_cfg or [{"type": "accuracy"}])

    def forward(self, batch):
        x, y = batch["x"], batch.get("y")
        logits = self.backbone(x)

        output = {"outputs": logits}
        if y is not None:
            loss = self.loss_fn(logits, y)
            output["loss"] = loss
            output["losses"] = {"cls_loss": loss.detach()}
            output["metrics"] = self.compute_metrics(logits, y)
        return output

    def compute_metrics(self, outputs, targets):
        return self.metrics(outputs, targets)
