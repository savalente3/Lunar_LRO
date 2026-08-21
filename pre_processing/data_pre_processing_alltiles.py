#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.append('../data_extraction')

import numpy as np
import cv2
from skimage.draw import circle_perimeter

from LRO_data_class import getRegionalLunarData, getDEMLunarData, getFilteredLabels, percentileNormalise
import pandas as pd

# external SSD - separate path from the single-tile patches dir,
PATCHES_DIR = '/Volumes/WIN10/lunar_patches_alltiles'
FILTERED_LABELS_PATH = '../data_preparation/filtered_labels_alltiles.csv'


# In[2]:


TILES = [
    {'name': 'WAC_GLOBAL_E300N0450_100M', 'lat_min':   0, 'lat_max':  60, 'lon_min':   0, 'lon_max':  90},
    {'name': 'WAC_GLOBAL_E300N1350_100M', 'lat_min':   0, 'lat_max':  60, 'lon_min':  90, 'lon_max': 180},
    {'name': 'WAC_GLOBAL_E300N2250_100M', 'lat_min':   0, 'lat_max':  60, 'lon_min': 180, 'lon_max': 270},
    {'name': 'WAC_GLOBAL_E300N3150_100M', 'lat_min':   0, 'lat_max':  60, 'lon_min': 270, 'lon_max': 360},
    {'name': 'WAC_GLOBAL_E300S0450_100M', 'lat_min': -60, 'lat_max':   0, 'lon_min':   0, 'lon_max':  90},
    {'name': 'WAC_GLOBAL_E300S1350_100M', 'lat_min': -60, 'lat_max':   0, 'lon_min':  90, 'lon_max': 180},
    {'name': 'WAC_GLOBAL_E300S2250_100M', 'lat_min': -60, 'lat_max':   0, 'lon_min': 180, 'lon_max': 270},
    {'name': 'WAC_GLOBAL_E300S3150_100M', 'lat_min': -60, 'lat_max':   0, 'lon_min': 270, 'lon_max': 360},
]


# ## Mask generation

# In[3]:


def maskGeneration(patch_wac_col, patch_wac_row, tileLabels, wac_col, wac_row, cos_lat):
    mask = np.zeros((256, 256), dtype=np.uint8)

    # patch covers 128 px N-S but 128/cos(lat) px E-W in the original tile
    half_col = 128 / cos_lat

    in_patch = (
        (wac_col >= patch_wac_col - half_col) & (wac_col < patch_wac_col + half_col) &
        (wac_row >= patch_wac_row - 128) & (wac_row < patch_wac_row + 128)
    )

    for i in tileLabels[in_patch].index:
        rel_col = int(128 + (wac_col[i] - patch_wac_col) * cos_lat)
        rel_row = int(128 + (wac_row[i] - patch_wac_row))
        radius = int((tileLabels.loc[i, 'DIAM_CIRC_IMG'] / 2) / 0.1)

        if radius < 1: continue
        rr, cc = circle_perimeter(rel_row, rel_col, radius, shape=(256, 256))
        mask[rr, cc] = 1

    return mask


# ## Patch extraction

# In[4]:


 # 60S-60N, 0-360E - loaded once, cropped per tile below
dataDEM_full = getDEMLunarData()
filteredLabels_all = getFilteredLabels(FILTERED_LABELS_PATH)

os.makedirs(PATCHES_DIR, exist_ok=True)
np.random.seed(42)

batch_num = 0
batch_size = 1000

wac_batch, dem_batch, mask_batch = [], [], []
kept_rows = []

for tile in TILES:
    print(f"\n--- {tile['name']} ---")

    dataWAC = getRegionalLunarData(tile['name'])

    row_start = int(round((60 - tile['lat_max']) * 128))
    row_end   = int(round((60 - tile['lat_min']) * 128))
    col_start = int(round(tile['lon_min'] * 128))
    col_end   = int(round(tile['lon_max'] * 128))
    dataDEM = dataDEM_full[row_start:row_end, col_start:col_end]

    tileLabels = filteredLabels_all[
        (filteredLabels_all['LAT_CIRC_IMG'] >= tile['lat_min']) & (filteredLabels_all['LAT_CIRC_IMG'] < tile['lat_max']) &
        (filteredLabels_all['LON_CIRC_IMG'] >= tile['lon_min']) & (filteredLabels_all['LON_CIRC_IMG'] < tile['lon_max'])
    ].reset_index(drop=True)

    wac_col = (tileLabels['LON_CIRC_IMG'] - tile['lon_min']) * (dataWAC.shape[1] / (tile['lon_max'] - tile['lon_min']))
    wac_row = (tile['lat_max'] - tileLabels['LAT_CIRC_IMG']) * (dataWAC.shape[0] / (tile['lat_max'] - tile['lat_min']))

    dem_half = int(128 * dataDEM.shape[1] / dataWAC.shape[1])

    tile_crater_count = 0

    # craters
    for i in range(len(tileLabels)):
        wac_center_col = int(round(wac_col.iloc[i])) + np.random.randint(-100, 101)
        wac_center_row = int(round(wac_row.iloc[i])) + np.random.randint(-100, 101)

        lat = tile['lat_max'] - wac_center_row / (dataWAC.shape[0] / (tile['lat_max'] - tile['lat_min']))
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
        mask_patch = maskGeneration(wac_center_col, wac_center_row, tileLabels, wac_col, wac_row, cos_lat)

        if wac_patch.shape != (256, 2*wac_half_col) or dem_patch.shape != (dem_half*2, dem_half_col*2):
            continue

        wac_patch = cv2.resize(wac_patch, (256, 256))
        dem_patch = cv2.resize(dem_patch, (256, 256))

        wac_batch.append(wac_patch)
        dem_batch.append(dem_patch)
        mask_batch.append(mask_patch)

        row = tileLabels.loc[i].to_dict()
        row['tile'] = tile['name']
        row['wac_col'] = wac_col.iloc[i]
        row['wac_row'] = wac_row.iloc[i]
        row['patch_lon'] = tile['lon_min'] + wac_center_col / (dataWAC.shape[1] / (tile['lon_max'] - tile['lon_min']))
        row['patch_lat'] = lat
        kept_rows.append(row)
        tile_crater_count += 1

        if len(wac_batch) == batch_size:
            np.savez_compressed(os.path.join(PATCHES_DIR, f'X_wac_{batch_num}'), np.array(wac_batch))
            np.savez_compressed(os.path.join(PATCHES_DIR, f'X_dem_{batch_num}'), np.array(dem_batch))
            np.savez_compressed(os.path.join(PATCHES_DIR, f'X_mask_{batch_num}'), np.array(mask_batch))
            wac_batch, dem_batch, mask_batch = [], [], []
            batch_num += 1

    print(f"{tile['name']}: {tile_crater_count} crater patches kept")

    # background (25% of this tile's crater patches, same tile's data)
    n_background = tile_crater_count // 4

    for _ in range(n_background):
        wac_rand_row = np.random.randint(128, dataWAC.shape[0] - 128)
        wac_rand_col = np.random.randint(128, dataWAC.shape[1] - 128)

        lat = tile['lat_max'] - wac_rand_row / (dataWAC.shape[0] / (tile['lat_max'] - tile['lat_min']))
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

        mask_patch = maskGeneration(wac_rand_col, wac_rand_row, tileLabels, wac_col, wac_row, cos_lat)

        if wac_patch.shape != (256, 256) or dem_patch.shape != (256, 256):
            continue

        wac_batch.append(wac_patch)
        dem_batch.append(dem_patch)
        mask_batch.append(mask_patch)

        kept_rows.append({
            'tile': tile['name'],
            'patch_lon': tile['lon_min'] + wac_rand_col / (dataWAC.shape[1] / (tile['lon_max'] - tile['lon_min'])),
            'patch_lat': lat,
        })

        if len(wac_batch) == batch_size:
            np.savez_compressed(os.path.join(PATCHES_DIR, f'X_wac_{batch_num}'), np.array(wac_batch))
            np.savez_compressed(os.path.join(PATCHES_DIR, f'X_dem_{batch_num}'), np.array(dem_batch))
            np.savez_compressed(os.path.join(PATCHES_DIR, f'X_mask_{batch_num}'), np.array(mask_batch))
            wac_batch, dem_batch, mask_batch = [], [], []
            batch_num += 1

    print(f"{tile['name']}: {n_background} background patches attempted")

if wac_batch:
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_wac_{batch_num}'), np.array(wac_batch))
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_dem_{batch_num}'), np.array(dem_batch))
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_mask_{batch_num}'), np.array(mask_batch))


# In[ ]:


kept_labels = pd.DataFrame(kept_rows)
kept_labels.to_csv(os.path.join(PATCHES_DIR, 'kept_labels.csv'), index=False)

print(f"{len(kept_labels)} total patches kept ({kept_labels['CRATER_ID'].notna().sum()} crater, "
      f"{kept_labels['CRATER_ID'].isna().sum()} background)")
print(f"{batch_num + 1} batch files written to {PATCHES_DIR}")

