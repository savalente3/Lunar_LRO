# Data Handover — Lunar Crater Detection

This document explains the pre-processed dataset produced for the multi-modal lunar crater detection model.

*Last updated: 2026-07-20, after full pipeline regeneration.*

---

## Overview

The pipeline produces 256×256 patch triplets (WAC image, DEM elevation, binary ring mask) extracted from:

- **WAC tile E300N1350** — LRO WAC Global Mosaic, 0–60°N, 90–180°E, 100 m/pixel
- **SLDEM2015** — Hybrid LOLA + Kaguya DEM, 128 px/degree, cropped to tile bounds
- **Robbins (2019)** — Crater catalogue, filtered to < 10 km diameter, in tile bounds, **`ARC_IMG > 0.5`** → 172,774 craters

**Total patches: 211,031** (crater-centred + 25% background), in **212 batch files**.

---

## File Structure

```
pre_processing/patches/
    X_wac_{0..211}.npz    # WAC batches (1000 patches each, last is partial)
    X_dem_{0..211}.npz    # DEM batches
    X_mask_{0..211}.npz   # binary ring mask batches
    kept_labels.csv       # crater metadata per patch (NaN for background) + patch_lon
    train_idx.npy         # 136,108 patches (66.1%)
    val_idx.npy           #  34,701 patches (16.9%)
    test_idx.npy          #  35,057 patches (17.0%)
```

⚠️ Patches on disk are **raw**. No normalised files are saved — normalisation happens on load (see below).

---

## Raw patch format

| | Shape | dtype | Raw range | Units |
|---|---|---|---|---|
| WAC | (1000, 256, 256) | float32 | ~[0, 0.4] | reflectance (I/F) |
| DEM | (1000, 256, 256) | float32 | ~[−5, +7] | **kilometres** |
| Mask | (1000, 256, 256) | uint8 | {0, 1} | 1 = crater rim |

> **WAC is NOT 8-bit DN.** The PDS BDR product reads as float32 reflectance — verified across all 8 WAC tiles (float32 dtype, negative minima, maxima 0.29–0.57). Dividing by 255 is wrong and destroys the signal.

---

## Loading data for training

```python
import sys; sys.path.append('../data_extraction')
from LRO_data_class import getSplitIndices, getNormalisedBatch, getAugmentedBatch

train_idx, val_idx, test_idx = getSplitIndices()

# training — normalised + randomly augmented (flips, 90° rotations)
wac, dem, mask = getAugmentedBatch(batch_num)

# validation / test — normalised only
wac, dem, mask = getNormalisedBatch(batch_num)
```

Do **not** instantiate `LunarDataset()` for training — it downloads the WAC tile, the full SLDEM and the catalogue (~3.3 GB) which training does not need.

### Normalisation (on load, not saved)

Both channels use **per-patch percentile normalisation** (1st–99th percentile clip → min-max to [0, 1]) via `percentileNormalise()`:

- **WAC** — per-patch because illumination varies across the tile; this neutralises sun-angle differences between patches, directly addressing the main weakness of optical crater detection
- **DEM** — per-patch because elevation range varies with terrain; clipping removes spikes

Verified output: both channels [0.000, 1.000], mean ≈ 0.44 (WAC) / 0.49 (DEM), no NaNs.

### Model inputs / outputs

| | Shape | dtype | Range |
|---|---|---|---|
| WAC patch (input) | (256, 256) | float32 | [0, 1] |
| DEM patch (input) | (256, 256) | float32 | [0, 1] |
| Mask (target) | (256, 256) | uint8 | {0, 1} |

---

## Class imbalance

Ring masks (1-px rims, DeepMoon style) give a crater-pixel fraction of **~2.6%** → **background : crater ≈ 37 : 1**.

Handle with weighted binary cross-entropy (DeepMoon's choice) or Dice loss. Decision pending at model phase.

---

## Splits — region-based, NOT random

Split by **patch centre longitude**, in whole bands:

| Split | Longitude band | Patches | % |
|---|---|---|---|
| Train | 90–150°E | 136,108 | 66.1 |
| Val | 150–165°E | 34,701 | 16.9 |
| Test | 165–180°E | 35,057 | 17.0 |

Patches within 0.5° of a boundary are dropped (5,165 total). Verified: train spans 90.4–149.5°E, val 150.5–164.5°E, test 165.5–179.6°E — no overlap.

> A random split would place heavily-overlapping patches of the same terrain in both train and test (spatial leakage) and inflate test scores. Same reasoning as Silburt et al.'s geographically exclusive split.

---

## Notes

- **Plate Carrée correction:** patches are cut with an E–W half-width of `128/cos(lat)` px and resized to 256², so craters are circular and the scale is ~100 m/px on both axes at every latitude. Masks use matching `cos(lat)`-scaled column offsets.
- **Random offset:** patch centres are offset from the crater by up to ±100 px per axis, so craters do not always sit at pixel (128, 128).
- DEM patches cover the same geographic footprint as WAC patches — `cv2.resize` upsamples to 256×256.
- Background patches sampled with `np.random.seed(42)` — reproducible.
- `kept_labels.csv` has NaN label columns for background rows, but `patch_lon` is always populated (needed for the split).
- Patches total ~91 GB (float32). Converting WAC to uint16 / DEM to float16 would cut this ~4× if I/O becomes a bottleneck.

---

## Reference

Silburt et al. (2019) — Lunar crater identification via deep learning. *Icarus*, 317, 27–38.
