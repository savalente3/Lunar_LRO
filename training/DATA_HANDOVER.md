# Data Handover — Lunar Crater Detection

This document explains the pre-processed dataset produced for the multi-modal lunar crater detection model.

---

## Overview

The pipeline produces 256×256 patch triplets (WAC image, DEM elevation, binary mask) extracted from:

- **WAC tile E300N1350** — LRO WAC Global Mosaic, 0–60°N, 90–180°E, 100m/pixel
- **SLDEM2015** — Hybrid LOLA + Kaguya DEM, 128 px/degree, cropped to tile bounds
- **Robbins (2019)** — Crater catalogue, filtered to small craters < 10km diameter in tile bounds

Total patches: ~215,000 (crater-centred + 25% background)

---

## File Structure

```
pre_processing/patches/
    X_wac_0.npz         # raw WAC batches (1000 patches each)
    X_dem_0.npz         # raw DEM batches
    X_mask_0.npz        # binary mask batches
    X_wac_norm_0.npz    # normalised WAC batches
    X_dem_norm_0.npz    # normalised DEM batches
    kept_labels.csv     # crater metadata per patch (NaN for background)

pre_processing/dataset_splits/
    train_idx.npy       # training patch indices (~70%)
    val_idx.npy         # validation patch indices (~15%)
    test_idx.npy        # test patch indices (~15%)
```

---

## Patch Format

Each `.npz` file contains a batch of 1000 patches (last batch may be smaller):

```python
batch = np.load('patches/X_wac_norm_0.npz')['arr_0']
# shape: (1000, 256, 256)
```

---

## Loading Data for Training

```python
import numpy as np

# load split indices
train_idx = np.load('pre_processing/dataset_splits/train_idx.npy')
val_idx   = np.load('pre_processing/dataset_splits/val_idx.npy')
test_idx  = np.load('pre_processing/dataset_splits/test_idx.npy')

# retrieve a specific patch by index
def load_patch(idx):
    batch_num = idx // 1000
    patch_pos = idx % 1000

    wac  = np.load(f'pre_processing/patches/X_wac_norm_{batch_num}.npz')['arr_0'][patch_pos]
    dem  = np.load(f'pre_processing/patches/X_dem_norm_{batch_num}.npz')['arr_0'][patch_pos]
    mask = np.load(f'pre_processing/patches/X_mask_{batch_num}.npz')['arr_0'][patch_pos]

    return wac, dem, mask
```

---

## Model Inputs / Outputs

| | Shape | dtype | Range |
|---|---|---|---|
| WAC patch | (256, 256) | float32 | [0, 1] |
| DEM patch | (256, 256) | float32 | [0, 1] |
| Mask (target) | (256, 256) | uint8 | {0, 1} |

- **WAC** normalised by dividing by 255 (standard 8-bit)
- **DEM** normalised per-patch using 1st–99th percentile clipping then min-max scaling
- **Mask** is binary: 1 = crater pixel, 0 = background

---

## Class Imbalance

Background pixels (~24M) significantly outnumber crater pixels (~4M), roughly 6:1. Handle with weighted binary cross-entropy or Dice loss. DeepMoon (Silburt et al. 2019) uses weighted binary cross-entropy.

---

## Notes

- DEM patches cover the same geographic footprint as WAC patches — `cv2.resize` upsamples from ~108×108 to 256×256
- Background patches are randomly sampled with `np.random.seed(42)` — reproducible
- `kept_labels.csv` rows are NaN for background patches to avoid mislabelling
- Split indices are fixed with `random_state=42` — do not regenerate
- Raw (unnormalised) patches are preserved in `X_wac_*.npz` / `X_dem_*.npz`

---

## Reference

Silburt et al. (2019) — DeepMoon: A deep learning approach to crater counting. *Icarus*, 317, 27–38.
