"""
One-time conversion — .npz batches -> normalised, memory-mapped .npy arrays.

The training bottleneck is decompressing a 262 MB .npz and re-running percentile
normalisation EVERY epoch. This does both ONCE:

  - reads every .npz batch in order
  - applies per-patch 1-99 percentile normalisation (the same step that used to
    happen on every load)
  - writes three flat .npy files that memory-map cleanly:
        dem_all.npy   float16   normalised DEM     ~28 GB
        wac_all.npy   float16   normalised WAC     ~28 GB
        mask_all.npy  uint8     ring mask {0,1}    ~14 GB

After this, training reads only the patches it needs per step (one seek, no
decompress) -> GPU-bound instead of I/O-bound. Your existing train/val/test
idx files stay valid: patch ordering is preserved (batch b, position p ->
global index b*1000 + p).

Run ONCE from training/ (or wherever, adjusting paths):
    python convert_to_memmap.py

Out: <PATCHES_DIR>/dem_all.npy, wac_all.npy, mask_all.npy, meta.json
"""

import os
import json
import glob
import numpy as np


PATCHES_DIR = '../pre_processing/patches'
OUT_DIR = '../pre_processing/patches'   # same dir; big files, 1.4 TB is fine
FILE_SIZE = 1000                         # patches per .npz batch


def percentileNormalise(patch, low=1, high=99):
    p_low, p_high = np.percentile(patch, [low, high])
    return ((np.clip(patch, p_low, p_high) - p_low) / (p_high - p_low + 1e-8))


def count_patches():
    """Total patches = sum of batch sizes. Last batch may be partial."""
    files = sorted(glob.glob(os.path.join(PATCHES_DIR, 'X_mask_*.npz')),
                   key=lambda f: int(f.split('_')[-1].split('.')[0]))
    n_batches = len(files)
    # peek last batch for its true size
    last = np.load(files[-1])['arr_0']
    total = (n_batches - 1) * FILE_SIZE + len(last)
    return total, n_batches


def main():
    total, n_batches = count_patches()
    print(f'{total} patches across {n_batches} batches', flush=True)

    # pre-allocate memory-mapped output files (created on disk, not in RAM)
    dem_mm = np.lib.format.open_memmap(
        os.path.join(OUT_DIR, 'dem_all.npy'), mode='w+',
        dtype=np.float16, shape=(total, 256, 256))
    wac_mm = np.lib.format.open_memmap(
        os.path.join(OUT_DIR, 'wac_all.npy'), mode='w+',
        dtype=np.float16, shape=(total, 256, 256))
    mask_mm = np.lib.format.open_memmap(
        os.path.join(OUT_DIR, 'mask_all.npy'), mode='w+',
        dtype=np.uint8, shape=(total, 256, 256))

    pos = 0
    for b in range(n_batches):
        wac = np.load(os.path.join(PATCHES_DIR, f'X_wac_{b}.npz'))['arr_0']
        dem = np.load(os.path.join(PATCHES_DIR, f'X_dem_{b}.npz'))['arr_0']
        mask = np.load(os.path.join(PATCHES_DIR, f'X_mask_{b}.npz'))['arr_0']
        n = len(wac)

        for j in range(n):
            wac_mm[pos + j] = percentileNormalise(wac[j]).astype(np.float16)
            dem_mm[pos + j] = percentileNormalise(dem[j]).astype(np.float16)
            mask_mm[pos + j] = mask[j].astype(np.uint8)

        pos += n
        if (b + 1) % 10 == 0:
            print(f'  {b+1}/{n_batches} batches ({pos} patches)', flush=True)

    dem_mm.flush(); wac_mm.flush(); mask_mm.flush()

    with open(os.path.join(OUT_DIR, 'meta.json'), 'w') as f:
        json.dump({'total': int(total), 'shape': [256, 256],
                   'dem': 'float16', 'wac': 'float16', 'mask': 'uint8'}, f)

    print(f'Done. {total} patches -> dem_all.npy, wac_all.npy, mask_all.npy', flush=True)


if __name__ == '__main__':
    main()