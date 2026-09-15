import os
import sys
from contextlib import nullcontext

sys.path.insert(0, os.getcwd())

import cv2
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from modules.sinkhorn import sinkhorn_norm
from utils.config import load_config
from utils.h5 import *

# All REFLACX preprocessing constants -- and the dataloader-side out-of-bounds/soft-ground-
# truth settings that must agree with them, see `configs/reflacx_data.yaml`'s header comment
# -- live in one shared YAML file, loaded once here. `dataset_kwargs` (below) exposes the
# dataloader-relevant half of it for `dataloaders.single_h5_dataset.SingleH5Dataset`.
_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "fixation_permutation_sorter", "reflacx_data.yaml"
)
REFLACX_CONFIG = load_config(_CONFIG_PATH)

METADATA = list(REFLACX_CONFIG.paths.metadata)
REMOVED_COLUMNS = list(REFLACX_CONFIG.removed_metadata_columns)
SPLIT_MAP = lambda x: 'TRAIN' if x == 'train' else ('VAL' if x == 'validate' else 'TEST')

FIXATIONS_ROOT = REFLACX_CONFIG.paths.fixations_root
METADATA_ROOT = REFLACX_CONFIG.paths.metadata_root
IMAGE_ROOT = REFLACX_CONFIG.paths.image_root
H5_FILE = REFLACX_CONFIG.paths.h5_path

RESIZE = tuple(REFLACX_CONFIG.image.resize)
NORMALIZE = REFLACX_CONFIG.fixations.normalize
DATASET_NAME = REFLACX_CONFIG.dataset_name

# Out-of-bounds handling: mirrors `dataloaders.single_h5_dataset.SingleH5Dataset`'s
# constructor args of the same name -- see `compute_soft_ground_truth`.
COORDINATE_COLUMNS = tuple(REFLACX_CONFIG.fixations.coordinate_columns)
COORDINATE_BOUNDS = tuple(REFLACX_CONFIG.fixations.coordinate_bounds)
OUT_OF_BOUNDS = REFLACX_CONFIG.fixations.out_of_bounds

# Soft ground-truth doubly-stochastic matrix (see `compute_soft_ground_truth`), written
# alongside `fixations` for permutation-learning tasks that want a smoother target than the
# hard identity permutation -- load it via `SingleH5Dataset(load_soft_permutation=True)`,
# or build matching kwargs with `dataset_kwargs`.
SOFT_GROUND_TRUTH = REFLACX_CONFIG.soft_ground_truth.enabled
SOFT_GROUND_TRUTH_METHOD = REFLACX_CONFIG.soft_ground_truth.method
SOFT_GROUND_TRUTH_COLUMNS = list(REFLACX_CONFIG.soft_ground_truth.columns)
SOFT_GROUND_TRUTH_TAU = REFLACX_CONFIG.soft_ground_truth.tau
SOFT_GROUND_TRUTH_N_ITERS = REFLACX_CONFIG.soft_ground_truth.n_iters


def _expand_bounds(bounds):
    """`(min, max)` (applied to both axes) or `((x_min, x_max), (y_min, y_max))` ->
    `((x_min, x_max), (y_min, y_max))` -- mirrors `SingleH5Dataset.__init__`'s handling of
    `coordinate_bounds`, so both sides interpret the config the same way."""
    if np.isscalar(bounds[0]):
        return tuple(bounds), tuple(bounds)
    x_bounds, y_bounds = bounds
    return tuple(x_bounds), tuple(y_bounds)

def refine_metadata_fields(df):
    def get_image_path(row):
        p = row.split('/')
        p[-1] = p[-1].replace('.dcm', '.jpg')
        patient = p[5:]
        return '/'.join(patient)
        
    df = df[df['eye_tracking_data_discarded'] == False] #discard rows with discarded eye tracking data
    df['split'] = df['split'].apply(SPLIT_MAP)
    df['image'] = df['image'].apply(get_image_path)

    return df

def load_image(image_path, resize=None, channel_first=True):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if resize:
            image = cv2.resize(image, resize)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if channel_first:
            image = np.transpose(image, (2, 0, 1))
        return image
    except Exception as e:
        print('Error loading image:', image_path)
        print(e)
        return np.zeros((1, resize[1], resize[0]), dtype=np.uint8) if channel_first else np.zeros((resize[1], resize[0], 1), dtype=np.uint8)
    
def load_fixations(fixation_path, normalize=True, image_size=None):
    # image_size = (X, Y) = (width, height)
    fixations = pd.read_csv(fixation_path)
    if normalize and image_size is not None:
        fixations['x_position_norm'] = fixations['x_position'] / image_size[0]
        fixations['y_position_norm'] = fixations['y_position'] / image_size[1]
    return fixations

def compute_soft_ground_truth(
    fixations,
    columns=SOFT_GROUND_TRUTH_COLUMNS,
    method=SOFT_GROUND_TRUTH_METHOD,
    tau=SOFT_GROUND_TRUTH_TAU,
    n_iters=SOFT_GROUND_TRUTH_N_ITERS,
    out_of_bounds=OUT_OF_BOUNDS,
    coordinate_columns=COORDINATE_COLUMNS,
    coordinate_bounds=COORDINATE_BOUNDS,
):
    """Builds a soft "ground truth" doubly-stochastic `(n, n)` matrix for one sample's
    fixation sequence, in its original (canonical) order, for use as a target in
    `trainers.losses.permutation_losses.DoublyStochasticCrossEntropyLoss` instead of the
    hard identity permutation matrix.

    Two fixations that are nearly identical (e.g. two closely-spaced fixations on the same
    region) are, for the purposes of *recovering scanpath order*, nearly interchangeable --
    a hard 0/1 target penalizes a model exactly as much for confusing two such fixations as
    for a wildly wrong prediction, which mis-states the task. This instead scores every pair
    of fixations `(i, j)` by their squared difference (mean over `columns`, i.e. an MSE) and
    turns the resulting `(n, n)` distance matrix into a doubly-stochastic matrix, so a
    prediction that spreads probability across genuinely similar fixations is barely
    penalized.

    Args:
        fixations: a sample's `fixations.csv` DataFrame (as returned by `load_fixations`),
            in its original order -- this is computed before any shuffling, which happens
            later at dataset-loading time (`dataloaders.permuted_h5_dataset`).
        columns: which columns to compare fixations on. Defaults to the normalized position
            columns `load_fixations` adds, so distances are comparable across samples with
            different image sizes; requires `normalize=True` when those were loaded.
        method: `"sinkhorn"` (default) treats `-distance / tau` as a compatibility score and
            runs `modules.sinkhorn.sinkhorn_norm` on it -- a smooth matrix that's close to
            identity but bleeds probability mass towards near-duplicate fixations.
            `"linear_sum_assignment"` instead solves the assignment problem exactly (via
            `scipy`), giving a hard 0/1 matrix. Since comparing a fixation to itself is
            always at least as good a match as comparing it to anything else
            (`distance[i, i] == 0` is the smallest possible value in row/column `i`), this
            always recovers the identity permutation for genuinely distinct fixations --
            provided mainly for comparison/debugging, not because it's expected to differ
            from a plain hard target in practice.
        tau, n_iters: forwarded to `sinkhorn_norm` (ignored for `"linear_sum_assignment"`).
        out_of_bounds, coordinate_columns, coordinate_bounds: mirror
            `dataloaders.single_h5_dataset.SingleH5Dataset`'s constructor args of the same
            name (default: `configs/reflacx_data.yaml`'s `fixations.*`, so both sides agree
            unless explicitly told otherwise) -- distances must be computed on whatever
            coordinates the dataloader will actually return, or this matrix would silently
            disagree with what a model sees:
              - `"ignore"`/`"remove"`: no value transform is needed here. `"remove"`'s
                dropped fixations are excluded post-hoc from the loaded matrix by
                `SingleH5Dataset.__getitem__` itself -- subsetting rows/cols of a
                pairwise-distance matrix computed over a superset gives exactly the same
                submatrix as computing it over just the subset -- so the *stored* matrix
                must stay full-size, matching the untouched `fixations` group.
              - `"clip"`: the dataloader permanently clamps coordinates before a model ever
                sees them, so distances must be computed on the same clamped values here,
                not the raw ones -- otherwise an out-of-bounds fixation's distance to its
                neighbors would be systematically wrong relative to what's actually observed.
    """
    missing = [col for col in columns if col not in fixations.columns]
    if missing:
        raise ValueError(f"compute_soft_ground_truth: columns {missing} not found in fixations (normalize=False?)")

    if out_of_bounds == 'clip':
        fixations = fixations.copy()
        x_col, y_col = coordinate_columns
        (x_min, x_max), (y_min, y_max) = _expand_bounds(coordinate_bounds)
        fixations[x_col] = fixations[x_col].clip(x_min, x_max)
        fixations[y_col] = fixations[y_col].clip(y_min, y_max)
    elif out_of_bounds not in ('ignore', 'remove'):
        raise ValueError(f"out_of_bounds must be 'ignore', 'clip', or 'remove', got {out_of_bounds!r}")

    values = fixations[list(columns)].to_numpy(dtype=np.float64)
    n = len(values)
    distance = np.mean((values[:, None, :] - values[None, :, :]) ** 2, axis=-1)  # (n, n) pairwise MSE

    if method == 'sinkhorn':
        log_alpha = torch.from_numpy(-distance / tau).float().unsqueeze(0)
        soft_gt = sinkhorn_norm(log_alpha, n_iters=n_iters)[0].numpy()
    elif method == 'linear_sum_assignment':
        row_idx, col_idx = linear_sum_assignment(distance)
        soft_gt = np.zeros((n, n))
        soft_gt[row_idx, col_idx] = 1.0
    else:
        raise ValueError(f"method must be 'sinkhorn' or 'linear_sum_assignment', got {method!r}")

    return soft_gt.astype(np.float32)


def dataset_kwargs(split, config=REFLACX_CONFIG, **overrides):
    """Builds the kwarg dict for `dataloaders.single_h5_dataset.SingleH5Dataset` (or
    `dataloaders.permuted_h5_dataset.PermutedFixationH5Dataset`) that matches whatever this
    module's shared `configs/reflacx_data.yaml` says preprocessing used, e.g.:

        SingleH5Dataset(**reflacx.dataset_kwargs("TRAIN"))

    `overrides` replaces any field, e.g. `dataset_kwargs("TRAIN", load_soft_permutation=True)`
    -- but overriding `out_of_bounds`/`coordinate_columns`/`coordinate_bounds` this way
    re-introduces exactly the preprocessing/dataloader mismatch this shared config exists to
    prevent, unless `compute_soft_ground_truth`/`preprocess` was also run with the same
    override.
    """
    kwargs = dict(
        h5_path=config.paths.h5_path,
        split=split,
        dataset_name=config.dataset_name,
        fixation_columns=config.fixations.fixation_columns,
        image_dtype=getattr(torch, config.image.image_dtype),
        normalize_image=config.image.normalize_image,
        coordinate_columns=tuple(config.fixations.coordinate_columns),
        coordinate_bounds=tuple(config.fixations.coordinate_bounds),
        out_of_bounds=config.fixations.out_of_bounds,
        load_soft_permutation=config.load_soft_permutation,
    )
    kwargs.update(overrides)
    return kwargs


def get_field(df, resize, normalize):
    ret_dict = {}
    for index, row in df.iterrows():
        image_path = os.path.join(IMAGE_ROOT, row['image'])
        image = load_image(image_path, resize, channel_first=True)
        
        id_ = row['id']
        split = row['split']
        group = f"{split}/{DATASET_NAME}/{id_}"
        
        metadata = row.drop(REMOVED_COLUMNS)
        image_size = (row['image_size_x'], row['image_size_y'])
        fixations_path = os.path.join(FIXATIONS_ROOT, id_, 'fixations.csv')
        fixations = load_fixations(fixations_path, normalize, image_size)
        
        yield group, metadata, image, fixations, split

def preprocess(
    metadata_paths=METADATA,
    resize=RESIZE,
    normalize=NORMALIZE,
    to_h5=False,
    h5_path=None,
    n_first=None,
    soft_ground_truth=SOFT_GROUND_TRUTH,
    soft_ground_truth_method=SOFT_GROUND_TRUTH_METHOD,
    soft_ground_truth_columns=SOFT_GROUND_TRUTH_COLUMNS,
    soft_ground_truth_tau=SOFT_GROUND_TRUTH_TAU,
    soft_ground_truth_n_iters=SOFT_GROUND_TRUTH_N_ITERS,
    out_of_bounds=OUT_OF_BOUNDS,
    coordinate_columns=COORDINATE_COLUMNS,
    coordinate_bounds=COORDINATE_BOUNDS,
):
    # write_dataframe(h5_path, group, df, mode='a', columns=df.columns, column_attrs=None)
    dfs = []
    count = 0
    count_stat = {(split.upper(), str(phase)): 0 for split in ['train', 'val', 'test'] for phase in range(1, 4)}
    with open_h5(h5_path, mode='w') if to_h5 else nullcontext(None) as h5file:
        for path in metadata_paths:
            df = pd.read_csv(os.path.join(METADATA_ROOT, path))
            df = refine_metadata_fields(df)

            for group, metadata, image, fixations, split in get_field(df, resize, normalize):
                phase = path.split('_')[-1].split('.')[0]
                metadata['phase'] = phase # add phase to metadata

                if to_h5:
                    metadata['N_fixations'] = len(fixations.index)
                    # Build a single-row DataFrame (rather than metadata.to_numpy()) so each
                    # column keeps its own native dtype -- a mixed-type Series would otherwise
                    # collapse to a generic object array, and per-column HDF5 datasets (like
                    # fixations below) let a reader recover column names without relying on
                    # h5py attribute order, which is alphabetical rather than insertion order.
                    metadata_df = pd.DataFrame({col: [val] for col, val in metadata.items()})
                    write_dataframe(h5file, group+'/metadata', metadata_df, columns=metadata_df.columns)
                    write_h5(h5file, group, 'image', image)
                    write_dataframe(h5file, group+'/fixations', fixations, columns=fixations.columns)
                    if soft_ground_truth:
                        soft_gt = compute_soft_ground_truth(
                            fixations,
                            columns=soft_ground_truth_columns,
                            method=soft_ground_truth_method,
                            tau=soft_ground_truth_tau,
                            n_iters=soft_ground_truth_n_iters,
                            out_of_bounds=out_of_bounds,
                            coordinate_columns=coordinate_columns,
                            coordinate_bounds=coordinate_bounds,
                        )
                        write_h5(h5file, group, 'soft_permutation', soft_gt)
                else:
                    dfs.append((group, metadata, image, fixations, split)) #smoke test, return the data instead of writing to h5
                count_stat[(split, phase)] += 1
                count += 1
                if n_first is not None and count >= n_first:
                    break

    return dfs, count_stat

if __name__ == "__main__":
    reflacx_data = preprocess(n_first=None, to_h5=True, h5_path=H5_FILE) #test
    print(reflacx_data[1])
    
    # with open_h5("reflacx_data.h5", mode='r') as h5file:
    #     print(h5_tree(h5file))
    
    