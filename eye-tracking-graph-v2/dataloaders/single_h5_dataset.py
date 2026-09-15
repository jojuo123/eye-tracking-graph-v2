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
`SingleH5Dataset.collate_fn` (which pads them) instead of the default collate.

Eye-tracker coordinates aren't guaranteed to land inside the image; `out_of_bounds` (with
`coordinate_columns`/`coordinate_bounds`) optionally clips or drops fixations that fall
outside it -- see `SingleH5Dataset.__init__`.

`load_soft_permutation=True` additionally loads the `soft_permutation` matrix
`reflacx.compute_soft_ground_truth` writes alongside `fixations` (if the H5 file has one) --
see `SingleH5Dataset.__init__`.

`MultiH5Dataset` (below) combines several such H5-backed datasets -- e.g. REFLACX plus another
eye-tracking dataset in its own H5 file -- into one, with a per-dataset `drawing_rates` weight
controlling how often each one's samples are drawn relative to the others.
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


@DATASETS.register("single_h5")
class SingleH5Dataset(BaseDataset):
    def __init__(
        self,
        h5_path,
        split,
        dataset_name="reflacx",
        fixation_columns=None,
        image_dtype=torch.float32,
        normalize_image=True,
        coordinate_columns=None,
        coordinate_bounds=(0.0, 1.0),
        out_of_bounds="ignore",
        load_soft_permutation=False,
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
            coordinate_columns: `(x_column, y_column)` naming which two entries of
                `fixation_columns` hold pixel coordinates, e.g. `("x_position_norm",
                "y_position_norm")` -- REFLACX's eye-tracker output isn't guaranteed to land
                inside the image (raw gaze noise/calibration error), so out-of-bounds
                fixations need to be found before they can be handled via `out_of_bounds`.
                Required (and otherwise ignored) whenever `out_of_bounds` is not `"ignore"`.
            coordinate_bounds: either one `(min, max)` pair applied to both axes, or
                `((x_min, x_max), (y_min, y_max))` for separate axis bounds. Defaults to
                `(0.0, 1.0)`, matching `reflacx.py`'s normalized `*_norm` fixation columns;
                pass e.g. `((0, 223), (0, 223))` if using pixel columns in the resized
                224x224 image's coordinate space instead (raw `x_position`/`y_position` are
                in the *original*, pre-resize image's pixel space -- don't bound them against
                the resized image's shape).
            out_of_bounds: how to handle a fixation whose `coordinate_columns` fall outside
                `coordinate_bounds`: `"ignore"` (default, no check performed), `"clip"` (clamp
                the coordinates to the nearest in-bounds value, keeping the fixation), or
                `"remove"` (drop the fixation from the sequence entirely, shortening it).
            load_soft_permutation: if True, also loads the `(n, n)` `soft_permutation`
                matrix `reflacx.compute_soft_ground_truth` wrote for this sample (raises a
                clear `KeyError` if the H5 file doesn't have one). When `out_of_bounds ==
                "remove"` drops fixations, the matrix's rows/columns are subset to match --
                it always stays aligned with whatever `fixations` ends up being returned.
        """
        self.h5_path = h5_path
        self.split = split
        self.dataset_name = dataset_name
        self.image_dtype = image_dtype
        self.normalize_image = normalize_image
        self.load_soft_permutation = load_soft_permutation
        self._h5file = None  # opened lazily, see `_file` -- h5py handles aren't fork-safe

        if out_of_bounds not in ("ignore", "clip", "remove"):
            raise ValueError(f"out_of_bounds must be 'ignore', 'clip', or 'remove', got {out_of_bounds!r}")
        if out_of_bounds != "ignore" and coordinate_columns is None:
            raise ValueError("coordinate_columns is required when out_of_bounds is not 'ignore'")
        self.out_of_bounds = out_of_bounds
        self.coordinate_columns = coordinate_columns

        with h5py.File(h5_path, "r") as f:
            group = f[split][dataset_name]
            self.sample_ids = sorted(group.keys())
            if fixation_columns is None:
                fixation_columns = sorted(group[self.sample_ids[0]]["fixations"].keys())
        self.fixation_columns = list(fixation_columns)

        if coordinate_columns is not None:
            x_col, y_col = coordinate_columns
            self._coordinate_indices = (self.fixation_columns.index(x_col), self.fixation_columns.index(y_col))
            if np.isscalar(coordinate_bounds[0]):
                x_bounds = y_bounds = coordinate_bounds
            else:
                x_bounds, y_bounds = coordinate_bounds
            self._coordinate_bounds = (tuple(x_bounds), tuple(y_bounds))

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

        keep_mask = None
        if self.out_of_bounds != "ignore":
            fixations, keep_mask = self._handle_out_of_bounds(fixations)

        metadata = {key: _decode_scalar(dset) for key, dset in group["metadata"].items()}

        item = {
            "id": sample_id,
            "image": image,
            "fixations": fixations,
            "n_fixations": fixations.shape[0],
            "metadata": metadata,
        }

        if self.load_soft_permutation:
            soft_permutation = torch.from_numpy(group["soft_permutation"][()].astype(np.float32))
            if keep_mask is not None:
                soft_permutation = soft_permutation[keep_mask][:, keep_mask]
            item["soft_permutation"] = soft_permutation

        return item

    def _handle_out_of_bounds(self, fixations):
        """Clips or drops fixations whose `coordinate_columns` fall outside `coordinate_bounds`,
        per `out_of_bounds`. Called from `__getitem__` once `out_of_bounds != "ignore"`.

        Returns `(fixations, keep_mask)`: `keep_mask` is the `(n,)` bool tensor of which
        original fixations survived (True = kept), or `None` when `out_of_bounds == "clip"`
        (nothing is ever dropped, so there's nothing to subset a `soft_permutation` matrix
        by) -- see `__getitem__`.
        """
        x_idx, y_idx = self._coordinate_indices
        (x_min, x_max), (y_min, y_max) = self._coordinate_bounds

        if self.out_of_bounds == "clip":
            fixations[:, x_idx] = fixations[:, x_idx].clamp(x_min, x_max)
            fixations[:, y_idx] = fixations[:, y_idx].clamp(y_min, y_max)
            return fixations, None

        in_bounds = (
            (fixations[:, x_idx] >= x_min)
            & (fixations[:, x_idx] <= x_max)
            & (fixations[:, y_idx] >= y_min)
            & (fixations[:, y_idx] <= y_max)
        )
        return fixations[in_bounds], in_bounds

    def close(self):
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def __del__(self):
        self.close()

    @staticmethod
    def collate_fn(batch):
        """Batches items from `SingleH5Dataset`, padding fixation sequences to the batch's
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

        result = {
            "id": ids,
            "image": images,
            "fixations": fixations,
            "fixations_mask": mask,
            "n_fixations": lengths,
            "metadata": metadata,
        }

        if "soft_permutation" in batch[0]:
            soft_permutation = torch.zeros(len(batch), max_len, max_len, dtype=torch.float32)
            for i, item in enumerate(batch):
                n = item["n_fixations"]
                soft_permutation[i, :n, :n] = item["soft_permutation"]
            result["soft_permutation"] = soft_permutation

        return result


@DATASETS.register("multi_h5")
class MultiH5Dataset(BaseDataset):
    """Combines several `SingleH5Dataset`s (or datasets from other H5-backed sources sharing
    the same item/collate contract) into a single dataset, e.g. to train on REFLACX alongside
    another eye-tracking dataset stored in its own H5 file.

    Indexing this dataset visits every underlying sample exactly once per epoch, in
    concatenation order -- combining datasets on its own doesn't change how much any one of
    them is seen. To actually control how often each dataset's samples are drawn, pair this
    with `sample_weights` and a `torch.utils.data.WeightedRandomSampler` (see `dataloaders.
    build_dataloader`, which does this automatically when asked to via `weighted_sampling`).
    """

    def __init__(self, datasets, drawing_rates=None):
        """
        Args:
            datasets: list of kwarg dicts, each passed to `SingleH5Dataset(**kwargs)` to build
                one sub-dataset. Every entry may point at a different h5_path, split, or
                dataset_name, so datasets living in separate files (or separate groups of the
                same file) can be combined.
            drawing_rates: per-dataset resampling weight controlling how often each dataset's
                samples should be drawn relative to the others, e.g. `[1.0, 0.5]` draws from
                the first dataset roughly twice as often as the second regardless of their
                relative sizes. Defaults to `1.0` (equal weight) for every dataset.
        """
        self.datasets = [SingleH5Dataset(**cfg) for cfg in datasets]
        if not self.datasets:
            raise ValueError("`datasets` must contain at least one dataset config")

        if drawing_rates is None:
            drawing_rates = [1.0] * len(self.datasets)
        if len(drawing_rates) != len(self.datasets):
            raise ValueError("`drawing_rates` must have exactly one entry per dataset in `datasets`")
        self.drawing_rates = list(drawing_rates)

        fixation_columns = self.datasets[0].fixation_columns
        for dataset in self.datasets[1:]:
            if dataset.fixation_columns != fixation_columns:
                raise ValueError(
                    "All datasets combined in a MultiH5Dataset must share the same "
                    f"fixation_columns to be batched together, got {fixation_columns} "
                    f"and {dataset.fixation_columns}"
                )
        self.fixation_columns = fixation_columns

        self._offsets = np.cumsum([0] + [len(d) for d in self.datasets])

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local_idx = idx - int(self._offsets[dataset_idx])
        return self.datasets[dataset_idx][local_idx]

    @property
    def sample_weights(self):
        """One weight per sample, in this dataset's index order, for use with
        `torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=...)`. Each
        dataset's `drawing_rate` is split evenly across its own samples, so a dataset's total
        share of draws depends only on its `drawing_rate` and not on how many samples it has."""
        weights = []
        for dataset, rate in zip(self.datasets, self.drawing_rates):
            weights.extend([rate / len(dataset)] * len(dataset))
        return weights

    def close(self):
        for dataset in self.datasets:
            dataset.close()

    def __del__(self):
        self.close()

    collate_fn = staticmethod(SingleH5Dataset.collate_fn)
