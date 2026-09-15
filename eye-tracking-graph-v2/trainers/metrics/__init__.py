from trainers.metrics.metrics import METRICS
from trainers.metrics.collection import build_metrics, MetricCollection

# Import custom metric modules so their @METRICS.register(...) decorators run.
from trainers.metrics import custom_metrics  # noqa: E402,F401
from trainers.metrics import scanpath_metrics  # noqa: E402,F401

__all__ = ["METRICS", "build_metrics", "MetricCollection"]
