"""Base dataset contract for all datasets in this template."""

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Subclasses implement `__len__` and `__getitem__`. Each item should be a
    dict whose keys match what the target model's `forward(batch)` expects
    (e.g. {"x": ..., "y": ...}) — the default `DataLoader` collate function
    will batch dict items into a dict of stacked tensors automatically."""

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, idx):
        ...
