#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.append('../data_extraction')

import matplotlib.pyplot as plt
import numpy as np

import cv2
from skimage.draw import circle_perimeter

from sklearn.model_selection import train_test_split

from LRO_data_class import getFilteredLabels, LunarDataset, percentileNormalise

# external SSD - keep the ~90GB+ patch set off the internal disk and
PATCHES_DIR = '/Volumes/WIN10/lunar_patches'


# In[2]:


ds = LunarDataset()

dataDEM = ds.DEMLunarData
dataWAC = ds.regionalLunarData

filteredLabels = getFilteredLabels()


# In[3]:


filteredLabels.head()


# In[4]:


# Crop full DEM (60S–60N, 0–360°E) to tile bounds (0–60°N, 90–180°E)
# resolution = 128
# row 0 = 60°N
# row 60*128 = 0°N. 
# Col 90*128 = 90°E, 
# col 180*128 = 180°E

dataDEM = dataDEM[0:60*128, 90*128:180*128]
print(dataDEM.shape)


# ## Patch extraction 

# In[5]:


# Mask Generation
def maskGeneration(patch_wac_col, patch_wac_row, filteredLabels, wac_col, wac_row, cos_lat):
    mask = np.zeros((256, 256), dtype=np.uint8)

    # patch covers 128 px N-S but 128/cos(lat) px E-W in the original tile
    half_col = 128 / cos_lat

    # find all craters within patch bounds in one step
    in_patch = (
        (wac_col >= patch_wac_col - half_col) & (wac_col < patch_wac_col + half_col) &
        (wac_row >= patch_wac_row - 128) & (wac_row < patch_wac_row + 128)
    )

    for i in filteredLabels[in_patch].index:
        # column offsets shrink by cos(lat) when the wide window is resized
        # back to 256 - craters end up circular in patch space
        rel_col = int(128 + (wac_col[i] - patch_wac_col) * cos_lat)
        rel_row = int(128 + (wac_row[i] - patch_wac_row))
        radius = int((filteredLabels.loc[i, 'DIAM_CIRC_IMG'] / 2) / 0.1)

        # ring mask (1px rim), not filled disk - rings stay distinct when
        # craters overlap (Silburt et al. 2019). skip degenerate radii
        if radius < 1: continue
        rr, cc = circle_perimeter(rel_row, rel_col, radius, shape=(256, 256))
        mask[rr, cc] = 1

    return mask


# In[6]:


# [source]:  Silburt et al. (2019) - code

# Converting crater lat/lon to WAC pixel coordinates.
# LON_CIRC_IMG - 90: shifts longitude. 90degE = 0 (tile starts at 90degE, not 0deg)
# 90 is total degrees of longitude for the tile: pixels per degree for WAC
wac_col = (filteredLabels['LON_CIRC_IMG'] - 90) * (dataWAC.shape[1] / 90)
wac_row = (60 - filteredLabels['LAT_CIRC_IMG']) * (dataWAC.shape[0] / 60)

dem_half = int(128 * dataDEM.shape[1] / dataWAC.shape[1])

stored_indices = []
patch_lons = []

wac_batch = []
dem_batch = []
mask_batch = []

batch_num = 0
batch_size = 1000

os.makedirs(PATCHES_DIR, exist_ok=True)

np.random.seed(42)

for i in range(len(filteredLabels)):

    wac_center_col = int(round(wac_col.iloc[i]))
    wac_center_row = int(round(wac_row.iloc[i]))

    wac_center_col += np.random.randint(-100, 101)
    wac_center_row += np.random.randint(-100, 101)

    # plate carree stretches the image E-W by 1/cos(lat) -> extract a wider window E-W (128/cos(lat) px) and resize back to 256,
    # patch: <1deg of latitude -> cos(lat) ~constant inside it
    lat = 60 - wac_center_row / (dataWAC.shape[0] / 60)
    cos_lat = np.cos(np.radians(lat))
    wac_half_col = int(round(128 / cos_lat))
    dem_half_col = int(round(dem_half / cos_lat))

    # Scale WAC pixel coordinates to DEM pixel
    dem_center_col = int(round(wac_center_col * dataDEM.shape[1] / dataWAC.shape[1]))
    dem_center_row = int(round(wac_center_row * dataDEM.shape[0] / dataWAC.shape[0]))

    # prevent patches in borders
    if (wac_center_row - 128 < 0 or wac_center_row + 128 > dataWAC.shape[0] or
        wac_center_col - wac_half_col < 0 or wac_center_col + wac_half_col > dataWAC.shape[1] or
        dem_center_row - dem_half < 0 or dem_center_row + dem_half > dataDEM.shape[0] or
        dem_center_col - dem_half_col < 0 or dem_center_col + dem_half_col > dataDEM.shape[1]): continue

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


        wac_batch = []
        dem_batch = []
        mask_batch = []
        batch_num += 1


# In[7]:


# Background patches for variability and diversity. 
# Better for model to distinguish between crater and non crater

wac_arr = dataWAC
dem_arr = dataDEM

# 25% of crater patches
n_background = len(stored_indices) // 4

for i in range(n_background):
    wac_rand_row = np.random.randint(128, dataWAC.shape[0] - 128)
    wac_rand_col = np.random.randint(128, dataWAC.shape[1] - 128)

    # E-W stretch correction as the crater patches
    lat = 60 - wac_rand_row / (dataWAC.shape[0] / 60)
    cos_lat = np.cos(np.radians(lat))
    wac_half_col = int(round(128 / cos_lat))
    dem_half_col = int(round(dem_half / cos_lat))


    dem_rand_row = int(round(wac_rand_row * dataDEM.shape[0] / dataWAC.shape[0]))
    dem_rand_col = int(round(wac_rand_col * dataDEM.shape[1] / dataWAC.shape[1]))

    # skip if any window crosses the border
    if (wac_rand_col - wac_half_col < 0 or wac_rand_col + wac_half_col > dataWAC.shape[1] or
        dem_rand_row - dem_half < 0 or dem_rand_row + dem_half > dataDEM.shape[0] or
        dem_rand_col - dem_half_col < 0 or dem_rand_col + dem_half_col > dataDEM.shape[1]):
        continue

    wac_patch = wac_arr[wac_rand_row-128:wac_rand_row+128, wac_rand_col-wac_half_col:wac_rand_col+wac_half_col]
    dem_patch = dem_arr[dem_rand_row-dem_half:dem_rand_row+dem_half, dem_rand_col-dem_half_col:dem_rand_col+dem_half_col]

    wac_patch = cv2.resize(wac_patch, (256, 256))
    dem_patch = cv2.resize(dem_patch, (256, 256))

    # mask
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

        wac_batch = []
        dem_batch = []
        mask_batch = []
        batch_num += 1


# In[8]:


if wac_batch:
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_wac_{batch_num}'), np.array(wac_batch))
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_dem_{batch_num}'), np.array(dem_batch))
    np.savez_compressed(os.path.join(PATCHES_DIR, f'X_mask_{batch_num}'), np.array(mask_batch))


filteredLabels['wac_col'] = wac_col.values
filteredLabels['wac_row'] = wac_row.values

kept_labels = filteredLabels.iloc[stored_indices].reset_index(drop=True)
kept_labels['patch_lon'] = patch_lons


# In[9]:


background_positions = []

for i, idx in enumerate(stored_indices):

    if (idx == -1):
        background_positions.append(i)

# Background patches have no crater label
# set to NaN to avoid misslabeling - but keep patch_lon for the split
label_cols = kept_labels.columns != 'patch_lon'
kept_labels.loc[background_positions, label_cols] = np.nan

kept_labels.to_csv(os.path.join(PATCHES_DIR, 'kept_labels.csv'), index=False)

batch_num_check = background_positions[0] // 1000

wac_check = np.load(os.path.join(PATCHES_DIR, f'X_wac_{batch_num_check}.npz'))['arr_0']
dem_check = np.load(os.path.join(PATCHES_DIR, f'X_dem_{batch_num_check}.npz'))['arr_0']
mask_check = np.load(os.path.join(PATCHES_DIR, f'X_mask_{batch_num_check}.npz'))['arr_0']

print(f'WAC batch shape: {wac_check.shape}')
print(f'DEM batch shape: {dem_check.shape}')
print(f'Mask batch shape: {mask_check.shape}')


# In[10]:


# checking the wac and dem aligh
patch_idx = 100

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(wac_check[patch_idx], cmap='gray')
axes[0].axhline(128, color='r', linewidth=0.8, alpha=0.7)
axes[0].axvline(128, color='r', linewidth=0.8, alpha=0.7)
axes[0].set_title(f"WAC center value: {wac_check[patch_idx][128, 128]:.2f}")

axes[1].imshow(dem_check[patch_idx], cmap='terrain')
axes[1].axhline(128, color='r', linewidth=0.8, alpha=0.7)
axes[1].axvline(128, color='r', linewidth=0.8, alpha=0.7)
axes[1].set_title(f"DEM center value: {dem_check[patch_idx][128, 128]:.2f}")

axes[2].imshow(mask_check[patch_idx])
axes[2].axhline(128, color='r', linewidth=0.8, alpha=0.7)
axes[2].axvline(128, color='r', linewidth=0.8, alpha=0.7)
axes[2].set_title(f"DEM center value: {mask_check[patch_idx][128, 128]:.2f}")


plt.show()


# ## Normalisation and loss choices
# 
# The normalisation:`LRO_data_class.getNormalisedBatch()` -> per-patch percentile, wac/dem
# 
# Patches analysis to justify:
# - **why percentile normalisation** — raw WAC and DEM have very different, per-patch-variable ranges
# - **why the loss must be weighted** — crater rim pixels are a tiny minority
# 

# In[11]:


sample_batch = 50
wac_raw  = np.load(os.path.join(PATCHES_DIR, f'X_wac_{sample_batch}.npz'))['arr_0']
dem_raw  = np.load(os.path.join(PATCHES_DIR, f'X_dem_{sample_batch}.npz'))['arr_0']
mask_raw = np.load(os.path.join(PATCHES_DIR, f'X_mask_{sample_batch}.npz'))['arr_0']


# In[12]:


fig, axes = plt.subplots(2, 2, figsize=(11, 7))
j = 0

# WAC - float32 reflectance, NOT 8-bit DN
axes[0, 0].hist(wac_raw[j].flatten(), bins=100)
axes[0, 0].set_title(f'WAC raw - reflectance [{wac_raw[j].min():.3f}, {wac_raw[j].max():.3f}]')
axes[0, 1].hist(percentileNormalise(wac_raw[j]).flatten(), bins=100)
axes[0, 1].set_title('WAC after percentile normalisation')

# DEM - elevation in km
axes[1, 0].hist(dem_raw[j].flatten(), bins=100)
axes[1, 0].set_title(f'DEM raw - km [{dem_raw[j].min():.2f}, {dem_raw[j].max():.2f}]')
axes[1, 1].hist(percentileNormalise(dem_raw[j]).flatten(), bins=100)
axes[1, 1].set_title('DEM after percentile normalisation')

for ax in axes.flat:
    ax.set_ylabel('pixel count')

plt.tight_layout()
plt.show()

# spikes at 0 and 1 -> clipped 1st/99th percentile tails
print('WAC raw range across batch: [%.4f, %.4f]' % (wac_raw.min(), wac_raw.max()))
print('DEM raw range across batch: [%.2f, %.2f] km' % (dem_raw.min(), dem_raw.max()))


# In[13]:


# Crater vs background pixels
acc0 = (mask_raw == 0).sum()
acc1 = (mask_raw == 1).sum()

ratio = acc0 / acc1
frac  = acc1 / (acc0 + acc1)

plt.bar(['background (0)', 'crater rim (1)'], [acc0, acc1])
plt.title(f'Class imbalance - {ratio:.0f} : 1  (crater pixels {frac*100:.2f}%)')
plt.ylabel('Pixel count')
plt.show()

print(f'background : crater = {ratio:.1f} : 1')
print('-> loss must compensate: weighted BCE (DeepMoon) or Dice')


# In[14]:


# Region split by longitude
# Neighbouring patches overlap
# random split puts the same terrain in train AND test -> spatial leakage -> inflated test scores.
# Whole bands instead: train 90-150E / val 150-165E / test 165-180E
# Patches within 0.5 deg of a boundary are dropped.

patch_lon = kept_labels['patch_lon'].values
margin = 0.5

train_idx = np.where(patch_lon < 150 - margin)[0]
val_idx   = np.where((patch_lon >= 150 + margin) & (patch_lon < 165 - margin))[0]
test_idx  = np.where(patch_lon >= 165 + margin)[0]

total = len(patch_lon)
dropped = total - len(train_idx) - len(val_idx) - len(test_idx)
print(f'Train: {len(train_idx)} patches ({len(train_idx)/total*100:.1f}%)')
print(f'Val:   {len(val_idx)} patches ({len(val_idx)/total*100:.1f}%)')
print(f'Test:  {len(test_idx)} patches ({len(test_idx)/total*100:.1f}%)')
print(f'Dropped at band boundaries: {dropped}')


# In[15]:


np.save(os.path.join(PATCHES_DIR, 'train_idx.npy'), train_idx)
np.save(os.path.join(PATCHES_DIR, 'val_idx.npy'), val_idx)
np.save(os.path.join(PATCHES_DIR, 'test_idx.npy'), test_idx)

