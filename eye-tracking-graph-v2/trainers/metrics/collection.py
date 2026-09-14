"""Build and evaluate several registered metrics together from a config list."""

from trainers.metrics.metrics import METRICS


def build_metrics(cfgs):
    """Build a dict of `{name: metric_instance}` from a list of config dicts.

    Each cfg's `type` selects the registered metric; an optional `name` key
    aliases it (useful for reusing the same metric type with different kwargs,
    e.g. `topk_accuracy` with `k=1` and `k=5` as "top1_accuracy"/"top5_accuracy").
    """
    metrics = {}
    for cfg in cfgs:
        cfg = dict(cfg)
        name = cfg.pop("name", None) or cfg["type"]
        metrics[name] = METRICS.build(cfg)
    return metrics


class MetricCollection:
    """Evaluates a list of registered metrics against the same `(outputs, targets)`
    pair, returning a `{name: value}` dict. Pass a list of metric config dicts,
    e.g. `[{"type": "accuracy"}, {"type": "topk_accuracy", "k": 3, "name": "top3_acc"}]`.
    """

    def __init__(self, cfgs):
        self.metrics = build_metrics(cfgs)

    def __call__(self, outputs, targets):
        return {name: metric(outputs, targets) for name, metric in self.metrics.items()}
