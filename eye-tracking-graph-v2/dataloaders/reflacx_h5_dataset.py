"""PyTorch dataset reading REFLACX eye-tracking data out of the H5 file produced by
`data/reflacx/reflacx.py::preprocess(..., to_h5=True)`.

Per sample, that file stores:
    {split}/{dataset_name}/{sample_id}/image        -- (C, H, W) uint8
    {split}/{dataset_name}/{sample_id}/metadata/*    -- one scalar dataset per metadata field
    {split}/{dataset_name}/{sample_id}/fixations/*   -- one (N_fixations,) dataset per fixation field

REFLACX's three annotation phases use different finding columns (phase 2, for example, has
"Abnormal mediastinal contour" where phase 1 has "Airway wall thickening"), so metadata is
returned as a plain dict rather than a fixed-size vector -- there is no single column layout
that fits every sample. Fixation sequences also vary in length per sample, so batches need
`ReflacxH5Dataset.collate_fn` (which pads them) instead of the default collate.
"""

import h5py
import numpy as np
import torch

from dataloaders import DATASETS
from dataloaders.base_dataset import BaseDataset


def _decode_scalar(dset):
    """Each metadata field is written (via `write_dataframe`) as a length-1 dataset; unwrap
    it to a plain Python value, decoding the UTF-8 bytes h5py returns for string columns."""
    value = dset[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


@DATASETS.register("reflacx_h5")
class ReflacxH5Dataset(BaseDataset):
    def __init__(
        self,
        h5_path,
        split,
        dataset_name="reflacx",
        fixation_columns=None,
        image_dtype=torch.float32,
        normalize_image=True,
    ):
        """
        Args:
            h5_path: path to the H5 file written by `reflacx.preprocess(..., to_h5=True)`.
            split: "TRAIN", "VAL", or "TEST" (the top-level group `preprocess` wrote to).
            dataset_name: sub-group name written by `preprocess` (`DATASET_NAME`, default "reflacx").
            fixation_columns: which `fixations.csv` columns to stack into the returned fixation
                tensor, and in what order. Defaults to every column present for the first
                sample, sorted -- consistent across samples since they all share the same
                `fixations.csv` schema (unlike metadata, which varies by phase).
            image_dtype: dtype of the returned image tensor.
            normalize_image: divide the (uint8) image by 255 before casting to `image_dtype`.
        """
        self.h5_path = h5_path
        self.split = split
        self.dataset_name = dataset_name
        self.image_dtype = image_dtype
        self.normalize_image = normalize_image
        self._h5file = None  # opened lazily, see `_file` -- h5py handles aren't fork-safe

        with h5py.File(h5_path, "r") as f:
            group = f[split][dataset_name]
            self.sample_ids = sorted(group.keys())
            if fixation_columns is None:
                fixation_columns = sorted(group[self.sample_ids[0]]["fixations"].keys())
        self.fixation_columns = list(fixation_columns)

    @property
    def _file(self):
        # Each DataLoader worker process calls this on its own first access, so every
        # process gets its own handle instead of sharing one across a fork/spawn boundary.
        if self._h5file is None:
            self._h5file = h5py.File(self.h5_path, "r")
        return self._h5file

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        group = self._file[self.split][self.dataset_name][sample_id]

        image = group["image"][()]
        image = torch.from_numpy(image.astype(np.float32) if self.normalize_image else image)
        if self.normalize_image:
            image = image / 255.0
        image = image.to(self.image_dtype)

        fixations_group = group["fixations"]
        fixations = np.stack([fixations_group[col][()] for col in self.fixation_columns], axis=-1)
        fixations = torch.from_numpy(fixations.astype(np.float32))

        metadata = {key: _decode_scalar(dset) for key, dset in group["metadata"].items()}

        return {
            "id": sample_id,
            "image": image,
            "fixations": fixations,
            "n_fixations": fixations.shape[0],
            "metadata": metadata,
        }

    def close(self):
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def __del__(self):
        self.close()

    @staticmethod
    def collate_fn(batch):
        """Batches items from `ReflacxH5Dataset`, padding fixation sequences to the batch's
        longest one (with a boolean mask marking real vs. padded steps) since the default
        `DataLoader` collate can't stack ragged sequences or heterogeneous metadata dicts."""
        ids = [item["id"] for item in batch]
        images = torch.stack([item["image"] for item in batch])
        metadata = [item["metadata"] for item in batch]

        lengths = torch.tensor([item["n_fixations"] for item in batch], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(batch) else 0
        num_features = batch[0]["fixations"].shape[-1] if batch else 0

        fixations = torch.zeros(len(batch), max_len, num_features, dtype=torch.float32)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            n = item["n_fixations"]
            fixations[i, :n] = item["fixations"]
            mask[i, :n] = True

        return {
            "id": ids,
            "image": images,
            "fixations": fixations,
            "fixations_mask": mask,
            "n_fixations": lengths,
            "metadata": metadata,
        }
