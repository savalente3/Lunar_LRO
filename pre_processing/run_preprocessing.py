#!/usr/bin/env python
# coding: utf-8
"""
Full single-tile patch extraction — data_pre_processing.ipynb, unchanged
logic, with only two differences:
  1. PATCHES_DIR points at a LOCAL path (was /Volumes/WIN10, a macOS SSD)
  2. notebook-only plotting/inspection cells removed so it runs headless

Everything else — maskGeneration, cos(lat) correction, the +/-100 px offset,
background patches, the 3-way longitude split — is hers, byte-for-byte.

Run:  nohup python run_preprocessing.py > log_prep.txt 2>&1 &
      tail -f log_prep.txt
"""

import sys, os
sys.path.append('../data_extraction')

import numpy as np
import cv2
from skimage.draw import circle_perimeter
from tqdm import tqdm

from LRO_data_class import getFilteredLabels, LunarDataset, percentileNormalise


# CHANGED: local path instead of /Volumes/WIN10/lunar_patches
PATCHES_DIR = 'patches'

# tqdm tuned for a nohup log: update at most every 30s, ascii, one stream
TQDM_KW = dict(mininterval=30.0, ascii=True, file=sys.stdout, ncols=80)


print('Loading DEM, WAC, labels (downloads rasters + catalogue)...', flush=True)
ds = LunarDataset()

dataDEM = ds.DEMLunarData
dataWAC = ds.regionalLunarData

filteredLabels = getFilteredLabels()
print(f'  filteredLabels: {filteredLabels.shape}', flush=True)


# Crop full DEM (60S-60N, 0-360E) to tile bounds (0-60N, 90-180E)
dataDEM = dataDEM[0:60*128, 90*128:180*128]
print(f'  cropped DEM: {dataDEM.shape}', flush=True)


# ------------------------------------------------------------------ mask
def maskGeneration(patch_wac_col, patch_wac_row, filteredLabels, wac_col, wac_row, cos_lat):
    mask = np.zeros((256, 256), dtype=np.uint8)
    half_col = 128 / cos_lat
    in_patch = (
        (wac_col >= patch_wac_col - half_col) & (wac_col < patch_wac_col + half_col) &
        (wac_row >= patch_wac_row - 128) & (wac_row < patch_wac_row + 128)
    )
    for i in filteredLabels[in_patch].index:
        rel_col = int(128 + (wac_col[i] - patch_wac_col) * cos_lat)
        rel_row = int(128 + (wac_row[i] - patch_wac_row))
        radius = int((filteredLabels.loc[i, 'DIAM_CIRC_IMG'] / 2) / 0.1)
        if radius < 1:
            continue
        rr, cc = circle_perimeter(rel_row, rel_col, radius, shape=(256, 256))
        mask[rr, cc] = 1
    return mask


# ------------------------------------------------------------ crater patches
wac_col = (filteredLabels['LON_CIRC_IMG'] - 90) * (dataWAC.shape[1] / 90)
wac_row = (60 - filteredLabels['LAT_CIRC_IMG']) * (dataWAC.shape[0] / 60)

dem_half = int(128 * dataDEM.shape[1] / dataWAC.shape[1])

stored_indices = []
patch_lons = []
wac_batch, dem_batch, mask_batch = [], [], []
batch_num = 0
batch_size = 1000

os.makedirs(PATCHES_DIR, exist_ok=True)
np.random.seed(42)

print('Extracting crater patches...', flush=True)
for i in tqdm(range(len(filteredLabels)), desc='crater patches', **TQDM_KW):
    wac_center_col = int(round(wac_col.iloc[i])) + np.random.randint(-100, 101)
    wac_center_row = int(round(wac_row.iloc[i])) + np.random.randint(-100, 101)

    lat = 60 - wac_center_row / (dataWAC.shape[0] / 60)
    cos_lat = np.cos(np.radians(lat))
    wac_half_col = int(round(128 / cos_lat))
    dem_half_col = int(round(dem_half / cos_lat))

    dem_center_col = int(round(wac_center_col * dataDEM.shape[1] / dataWAC.shape[1]))
    dem_center_row = int(round(wac_center_row * dataDEM.shape[0] / dataWAC.shape[0]))

    if (wac_center_row - 128 < 0 or wac_center_row + 128 > dataWAC.shape[0] or
        wac_center_col - wac_half_col < 0 or wac_center_col + wac_half_col > dataWAC.shape[1] or
        dem_center_row - dem_half < 0 or dem_center_row + dem_half > dataDEM.shape[0] or
        dem_center_col - dem_half_col < 0 or dem_center_col + dem_half_col > dataDEM.shape[1]):
        continue

    wac_patch = dataWAC[wac_center_row-128:wac_center_row+128, wac_center_col-wac_half_col:wac_center_col+wac_half_col]
    dem_patch = dataDEM[dem_center_row-dem_half:dem_center_row+dem_half, dem_center_col-dem_half_col:dem_center_col+dem_half_col]
    mask_patch = maskGeneration(wac_center_col, wac_center_row, filteredLabels, wac_col, wac_row, cos_lat)

    if wac_patch.shape != (256, 2*wac_half_col) or dem_patch.shape != (dem_half*2, dem_half_col*2):
        continue

    wac_patch = cv2.resize(wac_patch, (256, 256))
    dem_patch = cv2.resize(dem_patch, (256, 256))

    wac_batch.append(wac_patch)
    dem_batch.append(dem_patch)
    mask_batch.append(mask_patch)
    stored_indices.append(i)
    patch_lons.append(90 + wac_center_col / (dataWAC.shape[1] / 90))

    if len(wac_batch) == batch_size:
        np.savez_compressed(os.path.join(PATCHES_DIR, f'X_wac_{batch_num}'), np.array(wac_batch))
        np.savez_compressed(os.path.join(PATCHES_DIR, f'X_dem_{batch_num}'), np.array(dem_batch))
        np.savez_compressed(os.path.join(PATCHES_DIR, f'X_mask_{batch_num}'), np.array(mask_batch))
        wac_batch, dem_batch, mask_batch = [], [], []
        batch_num += 1
        if batch_num % 10 == 0:
            print(f'  {batch_num} batches ({batch_num*1000} patches)', flush=True)


# --------------------------------------------------------- background patches
n_background = len(stored_indices) // 4
print(f'Extracting {n_background} background patches...', flush=True)

for i in tqdm(range(n_background), desc='background patches', **TQDM_KW):
    wac_rand_row = np.random.randint(128, dataWAC.shape[0] - 128)
    wac_rand_col = np.random.randint(128, dataWAC.shape[1] - 128)

    lat = 60 - wac_rand_row / (dataWAC.shape[0] / 60)
    cos_lat = np.cos(np.radians(lat))
    wac_half_col = int(round(128 / cos_lat))
    dem_half_col = int(round(dem_half / cos_lat))

    dem_rand_row = int(round(wac_rand_row * dataDEM.shape[0] / dataWAC.shape[0]))
    dem_rand_col = int(round(wac_rand_col * dataDEM.shape[1] / dataWAC.shape[1]))

    if (wac_rand_col - wac_half_col < 0 or wac_rand_col + wac_half_col > dataWAC.shape[1] or
        dem_rand_row - dem_half < 0 or dem_rand_row + dem_half > dataDEM.shape[0] or
        dem_rand_col - dem_half_col < 0 or dem_rand_col + dem_half_col > dataDEM.shape[1]):
        continue

    wac_patch = dataWAC[wac_rand_row-128:wac_rand_row+128, wac_rand_col-wac_half_col:wac_rand_col+wac_half_col]
    dem_patch = dataDEM[dem_rand_row-dem_half:dem_rand_row+dem_half, dem_rand_col-dem_half_col:dem_rand_col+dem_half_col]

    wac_patch = cv2.resize(wac_patch, (256, 256))
    dem_patch = cv2.resize(dem_patch, (256, 256))
    mask_patch = maskGeneration(wac_rand_col, wac_rand_row, filteredLabels, wac_col, wac_row, cos_lat)

    if wac_patch.shape != (256, 256) or dem_patch.shape != (256, 256):
        continue

    wac_batch.append(wac_patch)
    dem_batch.append(dem_patch)
    mask_batch.append(mask_patch)
    stored_indices.append(-1)
    patch_lons.append(90 + wac_rand_col / (dataWAC.shape[1] / 90))

    if len(wac_batch) == batch_size:
        np.savez_compressed(os.path.join(PATCHES_DIR, f'X_wac_{batch_num}'), np.array(wac_batch))
        np.savez_compressed(os.path.join(PATCHES_DIR, f'X_dem_{batch_num}'), np.array(dem_batch))
        np.savez_compressed(os.path.join(PATCHES_DIR, f'X_mask_{batch_num}'), np.array(mask_batch))
        wac_batch, dem_batch, mask_batch = [], [], []
        batch_num += 1


# ------------------------------------------------------------- final flush
if wac_batch:
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_wac_{batch_num}'), np.array(wac_batch))
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_dem_{batch_num}'), np.array(dem_batch))
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_mask_{batch_num}'), np.array(mask_batch))

filteredLabels['wac_col'] = wac_col.values
filteredLabels['wac_row'] = wac_row.values

kept_labels = filteredLabels.iloc[stored_indices].reset_index(drop=True)
kept_labels['patch_lon'] = patch_lons


# --------------------------------------------------- background NaN labels
background_positions = [i for i, idx in enumerate(stored_indices) if idx == -1]
label_cols = kept_labels.columns != 'patch_lon'
kept_labels.loc[background_positions, label_cols] = np.nan
kept_labels.to_csv(os.path.join(PATCHES_DIR, 'kept_labels.csv'), index=False)


# --------------------------------------------------- region split by longitude
patch_lon = kept_labels['patch_lon'].values
margin = 0.5
train_idx = np.where(patch_lon < 150 - margin)[0]
val_idx   = np.where((patch_lon >= 150 + margin) & (patch_lon < 165 - margin))[0]
test_idx  = np.where(patch_lon >= 165 + margin)[0]

total = len(patch_lon)
dropped = total - len(train_idx) - len(val_idx) - len(test_idx)
print(f'Train: {len(train_idx)} ({len(train_idx)/total*100:.1f}%)', flush=True)
print(f'Val:   {len(val_idx)} ({len(val_idx)/total*100:.1f}%)', flush=True)
print(f'Test:  {len(test_idx)} ({len(test_idx)/total*100:.1f}%)', flush=True)
print(f'Dropped at boundaries: {dropped}', flush=True)

np.save(os.path.join(PATCHES_DIR, 'train_idx.npy'), train_idx)
np.save(os.path.join(PATCHES_DIR, 'val_idx.npy'), val_idx)
np.save(os.path.join(PATCHES_DIR, 'test_idx.npy'), test_idx)

print(f'Done. {batch_num+1} batch files + splits written to {PATCHES_DIR}', flush=True)