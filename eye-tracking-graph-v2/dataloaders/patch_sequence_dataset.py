"""Synthetic dataset for the `sinkhorn_patch_sorter` model (see
`models/sinkhorn_sort_model.py`): sequences of `(2D coordinate, image patch)`
pairs whose *canonical* order is known, presented in a random shuffled order.

Each canonical sequence element `i` gets a 2D coordinate placed along a random
walk (so nearby positions tend to be adjacent in the true order, like a
scanpath) and a patch whose pixel intensity encodes `i / (seq_len - 1)` plus
noise (so content alone is also informative, like the classic "sort N numbers"
Gumbel-Sinkhorn benchmark generalized to images). A random permutation is then
applied jointly to coordinates and patches; recovering it is the task.
"""

import torch

from dataloaders import DATASETS
from dataloaders.base_dataset import BaseDataset


@DATASETS.register("synthetic_patch_sequence")
class SyntheticPatchSequenceDataset(BaseDataset):
    def __init__(
        self,
        num_samples=1000,
        seq_len=6,
        patch_size=8,
        in_channels=3,
        coord_step_scale=1.0,
        noise_std=0.05,
        seed=0,
    ):
        g = torch.Generator().manual_seed(seed)

        # Canonical coordinates: a random walk, so element `i` and `i + 1` tend
        # to be spatially close -- position alone is a useful (not fully
        # sufficient) cue for order, same as content.
        steps = torch.randn(num_samples, seq_len, 2, generator=g) * coord_step_scale
        coords_canon = steps.cumsum(dim=1)

        # Canonical patches: solid intensity `i / (seq_len - 1)` plus noise.
        levels = torch.linspace(0, 1, seq_len).view(1, seq_len, 1, 1, 1)
        patches_canon = levels.expand(num_samples, seq_len, in_channels, patch_size, patch_size).clone()
        patches_canon += torch.randn(num_samples, seq_len, in_channels, patch_size, patch_size, generator=g) * noise_std

        # A random permutation per sample: `perm[s, i]` is the canonical index
        # that ends up at shuffled slot `i`, so `perm` is exactly the label the
        # model must recover (see `SinkhornPatchSorter`'s batch contract).
        perm = torch.argsort(torch.rand(num_samples, seq_len, generator=g), dim=1)

        batch_idx = torch.arange(num_samples).unsqueeze(1)
        self.coords = coords_canon[batch_idx, perm]
        self.patches = patches_canon[batch_idx, perm]
        self.perm = perm

    def __len__(self):
        return self.perm.size(0)

    def __getitem__(self, idx):
        return {"coords": self.coords[idx], "patches": self.patches[idx], "perm": self.perm[idx]}
