"""HDF5 writing helpers: create/write a dataset under an (optionally nested) group."""

from contextlib import contextmanager

import h5py
import numpy as np
import torch
import pandas as pd

DATASETS = ['TRAIN', 'VAL', 'TEST']


@contextmanager
def open_h5(file, mode="a"):
    """Yield an open `h5py.File`/`h5py.Group` for `file`.

    If `file` is already an open `h5py.File`/`h5py.Group`, it is passed
    through unchanged and left open (the caller owns its lifetime) -- this
    lets a file be opened once and reused across several write calls
    instead of reopening it each time. If `file` is a path, it is opened
    with `mode` and closed on exit.
    """
    if isinstance(file, (h5py.File, h5py.Group)):
        yield file
    else:
        with h5py.File(file, mode) as f:
            yield f


def _stringify_object_array(values):
    """Convert an object-dtype array's elements to `str` (HDF5 has no native object dtype)."""
    return np.array([str(v) for v in values.ravel()], dtype=object).reshape(values.shape)


def get_or_create_group(h5file, group):
    """Navigate to (creating as needed) the group at `group`, returning the `h5py.Group`.

    `group` may be a "/"-joined string, a list/tuple of nested group names
    (outermost first), or None/"" to mean the file root.
    """
    if not group:
        return h5file
    names = group.split("/") if isinstance(group, str) else list(group)
    names = [n for n in names if n]

    node = h5file
    for name in names:
        node = node.require_group(name)
    return node


def write_h5(file, group, name, data, mode="a", attrs=None, **dataset_kwargs):
    """Write `data` as dataset `name` under `group` in the HDF5 file `file`.

    `file` is either a path (opened with `mode` and closed afterward) or an
    already-open `h5py.File`/`h5py.Group` (reused as-is, left open --
    pass one in to avoid reopening the file on every call). `group` may be
    a "/"-joined string or a list of nested group names (missing groups
    are created); an existing dataset at the same location is overwritten.
    `attrs`, if given, is a dict set as attributes on the resulting
    dataset. Object-dtype data (e.g. a mixed-type row) is stored as
    variable-length UTF-8, one string per element. Returns the dataset's
    full path within the file.
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)

    kwargs = dict(dataset_kwargs)
    if data.dtype == object:
        data = _stringify_object_array(data)
        kwargs.setdefault("dtype", h5py.string_dtype(encoding="utf-8"))

    with open_h5(file, mode) as f:
        target = get_or_create_group(f, group)
        if name in target:
            del target[name]
        dset = target.create_dataset(name, data=data, **kwargs)
        if attrs:
            for key, value in attrs.items():
                dset.attrs[key] = value
        return dset.name


def write_dataframe(file, group, df, mode="a", columns=None, column_attrs=None, **dataset_kwargs):
    """Write each column of `df` as its own dataset (named after the column) under `group`.

    `file` is either a path (opened with `mode` and closed afterward) or an
    already-open `h5py.File`/`h5py.Group` (reused as-is, left open --
    pass one in to avoid reopening the file on every call). `columns`
    restricts which columns are written (default: all of `df`).
    `column_attrs`, if given, maps column name -> dict of attributes set on
    that column's dataset. Object/string columns are stored as
    variable-length UTF-8. Returns the group's full path within the file.
    """
    columns = df.columns if columns is None else columns
    column_attrs = column_attrs or {}

    with open_h5(file, mode) as f:
        target = get_or_create_group(f, group)
        for col in columns:
            values = df[col].to_numpy()
            kwargs = dict(dataset_kwargs)
            if values.dtype == object:
                values = _stringify_object_array(values)
                kwargs.setdefault("dtype", h5py.string_dtype(encoding="utf-8"))

            if col in target:
                del target[col]
            dset = target.create_dataset(col, data=values, **kwargs)
            for key, value in column_attrs.get(col, {}).items():
                dset.attrs[key] = value
        return target.name

def h5_tree(val, pre='', out=""):
    length = len(val)
    for key, val in val.items():
        length -= 1
        if length == 0:  # the last item
            if type(val) == h5py._hl.group.Group:
                out += pre + '└── ' + key + "\n"
                out = h5_tree(val, pre+'    ', out)
            else:
                out += pre + '└── ' + key + f' {val.shape}\n'
        else:
            if type(val) == h5py._hl.group.Group:
                out += pre + '├── ' + key + "\n"
                out = h5_tree(val, pre+'│   ', out)
            else:
                out += pre + '├── ' + key + f' {val.shape}\n'
    return out