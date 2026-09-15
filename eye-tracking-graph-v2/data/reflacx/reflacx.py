import os
import sys
from contextlib import nullcontext

sys.path.insert(0, os.getcwd())

import cv2
import pandas as pd

from utils.h5 import *

METADATA = [f"metadata_phase_{i}.csv" for i in range(1, 4)]
REMOVED_COLUMNS = ['id', 'split', 'eye_tracking_data_discarded', 'image', 'dicom_id', 'subject_id']
SPLIT_MAP = lambda x: 'TRAIN' if x == 'train' else ('VAL' if x == 'validate' else 'TEST')
FIXATIONS_ROOT = 'erda2/eye-tracking/physionet.org/files/reflacx-xray-localization/1.0.0/main_data'
METADATA_ROOT = 'erda2/eye-tracking/physionet.org/files/reflacx-xray-localization/1.0.0/main_data'
IMAGE_ROOT = 'erda2/eye-tracking/mimic-cxr-jpg/files'

# FIXATIONS_ROOT = '/home/extra/eye-tracking/reflacx/main_data'
# METADATA_ROOT = '/home/extra/eye-tracking/reflacx/main_data'
# IMAGE_ROOT = '/home/extra/eye-tracking/mimic-cxr-jpg/files'
H5_FILE = 'erda2/eye-tracking/normalized_data.h5'

RESIZE = (224, 224)
NORMALIZE = True
DATASET_NAME = 'reflacx'

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

def preprocess(metadata_paths=METADATA, resize=RESIZE, normalize=NORMALIZE, to_h5=False, h5_path=None, n_first=None):
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
    
    