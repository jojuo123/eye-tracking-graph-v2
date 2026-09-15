"""Dataset for permutation-learning tasks (e.g. the sinkhorn-sort model in
`configs/sinkhorn_sort_example.yaml`): shuffles each sample's fixation sequence and exposes
the permutation needed to recover the original order, so a model can be trained/evaluated on
"unshuffle this sequence" rather than baking shuffling into `SingleH5Dataset` itself, which
other tasks still need to read fixations in their original (temporal) order.

When `load_soft_permutation=True` (see `SingleH5Dataset`), the loaded `soft_permutation`
matrix is row-permuted the same way `fixations` is, so it stays a valid target for the
*shuffled* sequence: row `i` of the shuffled matrix is the original matrix's row
`permutation[i]`, since shuffled slot `i` holds original fixation `permutation[i]` -- the
same correspondence `permutation` itself encodes for `trainers.losses.permutation_losses`.
"""

import zlib

import numpy as np
import torch

from dataloaders import DATASETS
from dataloaders.single_h5_dataset import SingleH5Dataset


@DATASETS.register("permuted_h5")
class PermutedFixationH5Dataset(SingleH5Dataset):
    """Shuffles each sample's fixation sequence and returns the permutation used, alongside
    its inverse, so a model can be trained to recover the original fixation order from the
    shuffled one.

    By default (`deterministic=False`) a fresh random permutation is drawn on every access --
    since a Dataset's `__getitem__` is called anew each epoch, this means the model sees a
    different shuffle of the same sample every epoch, which is what you want for training.
    Set `deterministic=True` (typically for val/test) to instead derive a fixed permutation
    per `sample_id` (seeded by `seed` plus a hash of the id), so evaluation always shuffles
    a given sample the same way and metrics stay comparable across epochs/runs.
    """

    def __init__(self, *args, deterministic=False, seed=0, **kwargs):
        """
        Args:
            deterministic: if True, derive each sample's permutation from a fixed seed
                (`seed` + a hash of its `sample_id`) instead of drawing a new one every
                access. Use this for val/test splits.
            seed: base seed combined with the sample_id hash when `deterministic=True`.
                Ignored otherwise.
            *args, **kwargs: forwarded to `SingleH5Dataset`.
        """
        super().__init__(*args, **kwargs)
        self.deterministic = deterministic
        self.seed = seed

    def _rng_for(self, sample_id):
        if not self.deterministic:
            return np.random.default_rng()
        sample_seed = (zlib.crc32(sample_id.encode("utf-8")) ^ self.seed) & 0xFFFFFFFF
        return np.random.default_rng(sample_seed)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        n = item["n_fixations"]

        rng = self._rng_for(item["id"])
        permutation = rng.permutation(n)
        inverse_permutation = np.argsort(permutation)

        item["fixations"] = item["fixations"][permutation]
        item["permutation"] = torch.from_numpy(permutation.astype(np.int64))
        item["inverse_permutation"] = torch.from_numpy(inverse_permutation.astype(np.int64))

        if "soft_permutation" in item:
            item["soft_permutation"] = item["soft_permutation"][permutation]

        return item

    @staticmethod
    def collate_fn(batch):
        """Extends `SingleH5Dataset.collate_fn` with the batch's `permutation` and
        `inverse_permutation`, padded to the batch's longest sequence with -1 (an invalid
        index, since real values only ever range over `[0, n_fixations)`) at positions the
        `fixations_mask` already marks as padding."""
        collated = SingleH5Dataset.collate_fn(batch)
        max_len = collated["fixations"].shape[1]

        permutation = torch.full((len(batch), max_len), -1, dtype=torch.long)
        inverse_permutation = torch.full((len(batch), max_len), -1, dtype=torch.long)
        for i, item in enumerate(batch):
            n = item["n_fixations"]
            permutation[i, :n] = item["permutation"]
            inverse_permutation[i, :n] = item["inverse_permutation"]

        collated["permutation"] = permutation
        collated["inverse_permutation"] = inverse_permutation
        return collated
