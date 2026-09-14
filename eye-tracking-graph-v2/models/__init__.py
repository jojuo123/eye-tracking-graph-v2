from utils.registry import Registry

MODELS = Registry("models")


def build_model(cfg):
    """Build a model from a config dict, e.g. {"type": "mlp_classifier", "in_dim": 32, ...}."""
    return MODELS.build(cfg)


# Import model modules so their @MODELS.register(...) decorators run.
from models import example_model  # noqa: E402,F401
from models import sinkhorn_sort_model  # noqa: E402,F401

__all__ = ["MODELS", "build_model"]
