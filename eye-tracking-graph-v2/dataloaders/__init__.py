from torch.utils.data import DataLoader, WeightedRandomSampler

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

    If `loader_cfg` (or `loader_kwargs`) sets `weighted_sampling: true`, the dataset must
    expose a `sample_weights` sequence (e.g. `MultiH5Dataset`, whose weights encode each
    sub-dataset's `drawing_rates`) -- a `WeightedRandomSampler` built from it replaces
    `shuffle` for that loader.
    """
    if dataset is None:
        assert dataset_cfg is not None, "Provide either `dataset` or `dataset_cfg`"
        dataset = build_dataset(dataset_cfg)
    cfg = dict(loader_cfg or {})
    cfg.update(loader_kwargs)
    if cfg.pop("weighted_sampling", False):
        weights = dataset.sample_weights
        cfg["sampler"] = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        cfg.pop("shuffle", None)  # mutually exclusive with an explicit sampler
    return DataLoader(dataset, **cfg)


# Import dataset modules so their @DATASETS.register(...) decorators run.
from dataloaders import example_dataset  # noqa: E402,F401
from dataloaders import patch_sequence_dataset  # noqa: E402,F401
from dataloaders import single_h5_dataset  # noqa: E402,F401
from dataloaders import permuted_h5_dataset  # noqa: E402,F401

__all__ = ["DATASETS", "build_dataset", "build_dataloader"]
