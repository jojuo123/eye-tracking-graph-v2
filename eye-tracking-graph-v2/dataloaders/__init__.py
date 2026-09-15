from torch.utils.data import DataLoader

from utils.registry import Registry

DATASETS = Registry("datasets")


def build_dataset(cfg):
    """Build a dataset from a config dict, e.g. {"type": "synthetic_classification", ...}."""
    return DATASETS.build(cfg)


def build_dataloader(loader_cfg, dataset_cfg=None, dataset=None, **loader_kwargs):
    """Build a `torch.utils.data.DataLoader`.

    Either pass an already-constructed `dataset`, or a `dataset_cfg` to build
    one via the `DATASETS` registry. `loader_cfg` holds DataLoader kwargs
    (batch_size, shuffle, num_workers, ...).
    """
    if dataset is None:
        assert dataset_cfg is not None, "Provide either `dataset` or `dataset_cfg`"
        dataset = build_dataset(dataset_cfg)
    cfg = dict(loader_cfg or {})
    cfg.update(loader_kwargs)
    return DataLoader(dataset, **cfg)


# Import dataset modules so their @DATASETS.register(...) decorators run.
from dataloaders import example_dataset  # noqa: E402,F401
from dataloaders import patch_sequence_dataset  # noqa: E402,F401
from dataloaders import reflacx_h5_dataset  # noqa: E402,F401

__all__ = ["DATASETS", "build_dataset", "build_dataloader"]
