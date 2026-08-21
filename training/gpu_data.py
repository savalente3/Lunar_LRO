"""
Data loading for the GPU training run.

Deliberately does NOT import LRO_data_class: that module imports rasterio and
kagglehub at module level, which training does not need. The two functions
actually required (percentileNormalise, getNormalisedBatch) are pure numpy and
are reproduced here, matching DATA_HANDOVER.md exactly.

Patches on disk are RAW. Normalisation happens on load:
per-patch 1-99 percentile clip -> min-max to [0, 1].

MODE:
  'dem'  -> (256, 256, 1)  DEM only
  'dual' -> (256, 256, 2)  channel 0 = WAC, channel 1 = DEM
"""

import os
import numpy as np


PATCHES_DIR = '../pre_processing/patches'


def percentileNormalise(patch, low=1, high=99):
    """Clip to percentile range, rescale to [0, 1]. Matches LRO_data_class."""
    p_low, p_high = np.percentile(patch, [low, high])
    return (np.clip(patch, p_low, p_high) - p_low) / (p_high - p_low + 1e-8)


def getSplitIndices(patches_dir=PATCHES_DIR):
    train_idx = np.load(os.path.join(patches_dir, 'train_idx.npy'))
    val_idx = np.load(os.path.join(patches_dir, 'val_idx.npy'))
    test_idx = np.load(os.path.join(patches_dir, 'test_idx.npy'))
    return train_idx, val_idx, test_idx


def getNormalisedBatch(batch_num, patches_dir=PATCHES_DIR):
    wac = np.load(os.path.join(patches_dir, f'X_wac_{batch_num}.npz'))['arr_0']
    dem = np.load(os.path.join(patches_dir, f'X_dem_{batch_num}.npz'))['arr_0']
    mask = np.load(os.path.join(patches_dir, f'X_mask_{batch_num}.npz'))['arr_0']

    norm_wac = np.zeros_like(wac, dtype=np.float32)
    norm_dem = np.zeros_like(dem, dtype=np.float32)
    for j in range(len(wac)):
        norm_wac[j] = percentileNormalise(wac[j])
        norm_dem[j] = percentileNormalise(dem[j])

    return norm_wac, norm_dem, mask


def load_subset(mode='dem', n_train_batches=20, n_val_batches=5,
                patches_dir=PATCHES_DIR, max_train=None, max_val=None,
                verbose=True):
    """Load a capped subset of the train/val splits into memory."""
    train_idx, val_idx, _ = getSplitIndices(patches_dir)

    def collect(indices, max_batches, cap):
        batch_ids = np.unique(indices // 1000)[:max_batches]
        X_parts, Y_parts = [], []
        total = 0

        for b in batch_ids:
            wac, dem, mask = getNormalisedBatch(int(b), patches_dir)
            local = indices[(indices >= b * 1000) & (indices < (b + 1) * 1000)] - b * 1000
            local = local[local < len(mask)]
            if len(local) == 0:
                continue

            if mode == 'dem':
                x = dem[local][..., None]
            else:
                x = np.stack([wac[local], dem[local]], axis=-1)

            X_parts.append(x.astype(np.float32))
            Y_parts.append(mask[local][..., None].astype(np.float32))
            total += len(local)

            del wac, dem, mask, x

            if verbose:
                print(f'    batch {int(b)}: +{len(local)} (total {total})', flush=True)
            if cap and total >= cap:
                break

        X = np.concatenate(X_parts)
        Y = np.concatenate(Y_parts)
        if cap:
            X, Y = X[:cap], Y[:cap]
        return X, Y

    X_tr, Y_tr = collect(train_idx, n_train_batches, max_train)
    X_val, Y_val = collect(val_idx, n_val_batches, max_val)
    return X_tr, Y_tr, X_val, Y_val


# ---------------------------------------------------------------------------
# STREAMING GENERATOR — for the full tile (90 GB won't fit in RAM)
# ---------------------------------------------------------------------------

import keras


class PatchSequence(keras.utils.PyDataset):
    """Streams patches from the .npz batch files, one file cached at a time.

    Shuffles at the file level and within files, so consecutive minibatches
    aren't all the same terrain. mode: 'dem' -> (B,256,256,1),
    'dual' -> (B,256,256,2) [wac, dem]. y is (B,256,256,1).
    """

    def __init__(self, indices, mode='dem', batch_size=16, shuffle=True,
                 patches_dir=PATCHES_DIR, **kw):
        super().__init__(**kw)
        self.indices = np.asarray(indices)
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.patches_dir = patches_dir
        self._cache_file = None
        self._cache = None
        self._arrange()

    def _arrange(self):
        by_file = {}
        for idx in self.indices:
            by_file.setdefault(idx // 1000, []).append(idx % 1000)
        files = list(by_file.keys())
        if self.shuffle:
            np.random.shuffle(files)
        order = []
        for f in files:
            pos = by_file[f]
            if self.shuffle:
                np.random.shuffle(pos)
            for p in pos:
                order.append((f, p))
        self._order = order

    def _get_file(self, f):
        if self._cache_file != f:
            self._cache = getNormalisedBatch(int(f), self.patches_dir)
            self._cache_file = f
        return self._cache

    def __len__(self):
        return int(np.ceil(len(self._order) / self.batch_size))

    def __getitem__(self, i):
        items = self._order[i * self.batch_size:(i + 1) * self.batch_size]
        # group by file to minimise reloads within a batch
        items = sorted(items)
        ch = 1 if self.mode == 'dem' else 2
        X = np.empty((len(items), 256, 256, ch), dtype=np.float32)
        Y = np.empty((len(items), 256, 256, 1), dtype=np.float32)
        for j, (f, p) in enumerate(items):
            wac, dem, mask = self._get_file(f)
            if self.mode == 'dem':
                X[j, :, :, 0] = dem[p]
            else:
                X[j, :, :, 0] = wac[p]
                X[j, :, :, 1] = dem[p]
            Y[j, :, :, 0] = mask[p]
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            self._arrange()


# ---------------------------------------------------------------------------
# MEMMAP GENERATOR — reads pre-normalised memory-mapped .npy (fast path)
# ---------------------------------------------------------------------------

class MemmapSequence(keras.utils.PyDataset):
    """Streams patches from memory-mapped .npy arrays produced by
    convert_to_memmap.py.

    Reads CONTIGUOUS slices from the memmap (fast sequential disk access) and
    shuffles at the batch-block level rather than per patch. This keeps disk
    reads sequential — the big win over scattered fancy-indexing — while still
    giving good shuffling. Assumes patch ordering in the memmap is already
    spatially mixed (crater patches then background, across the tile), which it
    is; block shuffling on top is plenty for training.

    Requires dem_all.npy / wac_all.npy / mask_all.npy in patches_dir.
    mode: 'dem' -> (B,256,256,1), 'dual' -> (B,256,256,2) [wac, dem].
    """

    def __init__(self, indices, mode='dem', batch_size=16, shuffle=True,
                 patches_dir=PATCHES_DIR, **kw):
        super().__init__(**kw)
        self.indices = np.sort(np.asarray(indices))  # sorted -> contiguous runs
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dem = np.load(os.path.join(patches_dir, 'dem_all.npy'), mmap_mode='r')
        self.mask = np.load(os.path.join(patches_dir, 'mask_all.npy'), mmap_mode='r')
        if mode == 'dual':
            self.wac = np.load(os.path.join(patches_dir, 'wac_all.npy'), mmap_mode='r')
        else:
            self.wac = None
        # split the sorted index list into batch-sized blocks; shuffle block order
        self._blocks = [self.indices[k:k + batch_size]
                        for k in range(0, len(self.indices), batch_size)]
        self._block_order = np.arange(len(self._blocks))
        if self.shuffle:
            np.random.shuffle(self._block_order)

    def __len__(self):
        return len(self._blocks)

    def __getitem__(self, i):
        idx = self._blocks[self._block_order[i]]  # already sorted -> near-contiguous
        ch = 1 if self.mode == 'dem' else 2
        X = np.empty((len(idx), 256, 256, ch), dtype=np.float32)
        Y = np.empty((len(idx), 256, 256, 1), dtype=np.float32)
        # contiguous slice when the block is a run; fancy-index otherwise (still sorted)
        lo, hi = idx[0], idx[-1] + 1
        if hi - lo == len(idx):
            dem = np.asarray(self.dem[lo:hi], dtype=np.float32)
            mask = np.asarray(self.mask[lo:hi], dtype=np.float32)
            wac = np.asarray(self.wac[lo:hi], dtype=np.float32) if self.mode == 'dual' else None
        else:
            dem = np.asarray(self.dem[idx], dtype=np.float32)
            mask = np.asarray(self.mask[idx], dtype=np.float32)
            wac = np.asarray(self.wac[idx], dtype=np.float32) if self.mode == 'dual' else None
        if self.mode == 'dem':
            X[..., 0] = dem
        else:
            X[..., 0] = wac
            X[..., 1] = dem
        Y[..., 0] = mask
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._block_order)