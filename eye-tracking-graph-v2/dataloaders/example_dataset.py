"""Example dataset: synthetic, linearly-clustered classification data.

Useful for smoke-testing the whole pipeline (model/trainer/losses/metrics)
without depending on any external data. Replace with a real dataset by
following the same pattern: subclass `BaseDataset` and register it.
"""

import torch

from dataloaders import DATASETS
from dataloaders.base_dataset import BaseDataset


@DATASETS.register("synthetic_classification")
class SyntheticClassificationDataset(BaseDataset):
    def __init__(self, num_samples=1000, in_dim=32, num_classes=10, seed=0, centers_seed=0):
        # Class centers are drawn from `centers_seed`, independent of `seed`,
        # so train/val/test splits (typically constructed with different
        # `seed`s so they don't share exact samples) still share the same
        # underlying class concept -- otherwise there is nothing to
        # generalize to and validation accuracy would be near chance.
        centers_g = torch.Generator().manual_seed(centers_seed)
        centers = torch.randn(num_classes, in_dim, generator=centers_g) * 3

        g = torch.Generator().manual_seed(seed)
        # Assign labels uniformly at random first, then place each point as
        # noise around its class's center -- this guarantees balanced classes.
        # (Generating x ~ N(0, I) and labeling by nearest random center instead
        # degenerates in high dimensions: one center ends up nearest to almost
        # every point, making the "dataset" nearly single-class.)
        self.y = torch.randint(0, num_classes, (num_samples,), generator=g)
        noise = torch.randn(num_samples, in_dim, generator=g)
        self.x = centers[self.y] + noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}
