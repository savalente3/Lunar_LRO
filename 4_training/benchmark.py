"""
Throwaway benchmark - answers one question: is the 435 ms/step the loader or the GPU?

Times each part separately, on the real memmaps, in about 5 minutes:
  GPU only     - synthetic batch already in RAM, no loading at all
  loader only  - reads patches, no model
  full         - both, as training actually runs

If GPU-only is already ~435 ms, the loader is irrelevant and the levers are batch
size / less data / smaller model. If loader-only dominates, the loader is worth fixing.

Run from 4_training/:  python -u benchmark.py
"""

import os
import sys
import time

sys.path.append('../1_data_extraction')

import numpy as np
import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout
from keras.regularizers import l2

from LRO_data_class import getSplitIndices, augment

PATCHES_DIR = '../3_pre_processing/lunar_patches_alltiles'
DIM = 256
N_BATCHES = 30          # per measurement, after warmup
WARMUP = 5

print(tf.config.list_physical_devices('GPU'), flush=True)


# --------------------------------------------------------------------------- model
def build(in_ch, n_filters=32, FL=3, init='he_normal', lmbda=1e-6, dropout=0.15):
    def conv(x, f):
        for _ in range(2):
            x = Conv2D(f, FL, activation='relu', kernel_initializer=init,
                       kernel_regularizer=l2(lmbda), padding='same')(x)
        return x

    inp = keras.Input(shape=(DIM, DIM, in_ch))
    a1 = conv(inp, n_filters)
    a2 = conv(MaxPooling2D((2, 2), strides=(2, 2))(a1), n_filters * 2)
    a3 = conv(MaxPooling2D((2, 2), strides=(2, 2))(a2), n_filters * 4)
    a4 = conv(MaxPooling2D((2, 2), strides=(2, 2))(a3), n_filters * 8)
    u = conv(MaxPooling2D((2, 2), strides=(2, 2))(a4), n_filters * 16)

    for skip, f in [(a4, 8), (a3, 4), (a2, 2), (a1, 1)]:
        u = Conv2DTranspose(n_filters * f, kernel_size=2, strides=2, padding='same')(u)
        u = Dropout(dropout)(Concatenate()([u, skip]))
        u = conv(u, n_filters * f)

    out = Conv2D(1, 1, activation='sigmoid')(u)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.BinaryFocalCrossentropy(
                      apply_class_balancing=True, alpha=0.75, gamma=2.0))
    return model


# --------------------------------------------------------------------------- loaders
class PerPatch:
    """the 435 ms/step version - one memmap read per patch"""

    def __init__(self, indices, batch_size, augment_data=True):
        self.bs = batch_size
        self.augment_data = augment_data
        self.rng = np.random.default_rng(42)
        self.wac = np.load(os.path.join(PATCHES_DIR, 'wac_all.npy'), mmap_mode='r')
        self.dem = np.load(os.path.join(PATCHES_DIR, 'dem_all.npy'), mmap_mode='r')
        self.mask = np.load(os.path.join(PATCHES_DIR, 'mask_all.npy'), mmap_mode='r')
        by_file = {}
        for i in indices:
            by_file.setdefault(int(i // 1000), []).append(int(i % 1000))
        self.order = []
        for f in self.rng.permutation(sorted(by_file)):
            for p in self.rng.permutation(np.array(by_file[int(f)])):
                self.order.append((int(f), int(p)))

    def __getitem__(self, i):
        items = self.order[i * self.bs:(i + 1) * self.bs]
        X = np.zeros((len(items), DIM, DIM, 2), np.float32)
        y = np.zeros((len(items), DIM, DIM, 1), np.float32)
        for j, (fn, p) in enumerate(items):
            k = fn * 1000 + p
            w = np.asarray(self.wac[k], np.float32)
            d = np.asarray(self.dem[k], np.float32)
            m = np.asarray(self.mask[k], np.float32)
            if self.augment_data:
                w, d, m = augment(w, d, m, self.rng)
            X[j, :, :, 0], X[j, :, :, 1], y[j, :, :, 0] = w, d, m
        return X, y


class Slab(PerPatch):
    """this morning's version - one 1000-patch read, served from RAM"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cache = {}

    def load(self, fn):
        if self.cache.get('file') == fn:
            return
        lo, hi = fn * 1000, fn * 1000 + 1000
        self.cache['wac'] = np.asarray(self.wac[lo:hi], np.float32)
        self.cache['dem'] = np.asarray(self.dem[lo:hi], np.float32)
        self.cache['mask'] = np.asarray(self.mask[lo:hi], np.float32)
        self.cache['file'] = fn

    def __getitem__(self, i):
        items = self.order[i * self.bs:(i + 1) * self.bs]
        X = np.zeros((len(items), DIM, DIM, 2), np.float32)
        y = np.zeros((len(items), DIM, DIM, 1), np.float32)
        for j, (fn, p) in enumerate(items):
            self.load(fn)
            w, d, m = self.cache['wac'][p], self.cache['dem'][p], self.cache['mask'][p]
            if self.augment_data:
                w, d, m = augment(w, d, m, self.rng)
            X[j, :, :, 0], X[j, :, :, 1], y[j, :, :, 0] = w, d, m
        return X, y


# --------------------------------------------------------------------------- timing
def time_it(fn, n=N_BATCHES, warmup=WARMUP):
    for i in range(warmup):
        fn(i)
    t0 = time.perf_counter()
    for i in range(warmup, warmup + n):
        fn(i)
    return (time.perf_counter() - t0) / n * 1000        # ms per batch


def main():
    train_idx, _, _ = getSplitIndices(PATCHES_DIR)
    print(f'train {len(train_idx):,}\n', flush=True)

    results = {}

    # ---- GPU only: same batch reused, already on the host, no loading ----
    for bs in (8, 16, 32):
        model = build(2)
        X = np.random.rand(bs, DIM, DIM, 2).astype(np.float32)
        y = (np.random.rand(bs, DIM, DIM, 1) > 0.98).astype(np.float32)
        ms = time_it(lambda i: model.train_on_batch(X, y))
        results[f'GPU only            bs={bs}'] = ms
        print(f'  GPU only            bs={bs:<3} {ms:8.1f} ms/step', flush=True)
        del model
        keras.backend.clear_session()

    # ---- loader only: no model at all ----
    for name, cls in (('per-patch', PerPatch), ('slab', Slab)):
        for bs in (8, 32):
            ld = cls(train_idx, bs)
            ms = time_it(lambda i: ld[i])
            results[f'loader {name:<12} bs={bs}'] = ms
            print(f'  loader {name:<12} bs={bs:<3} {ms:8.1f} ms/step', flush=True)
            del ld

    # ---- full pipeline ----
    for name, cls in (('per-patch', PerPatch), ('slab', Slab)):
        model = build(2)
        ld = cls(train_idx, 8)
        ms = time_it(lambda i: model.train_on_batch(*ld[i]))
        results[f'FULL   {name:<12} bs=8'] = ms
        print(f'  FULL   {name:<12} bs=8   {ms:8.1f} ms/step', flush=True)
        del model, ld
        keras.backend.clear_session()

    # ---- verdict ----
    gpu8 = results['GPU only            bs=8']
    ld8 = results['loader per-patch    bs=8']
    print('\n' + '=' * 62)
    print(f'  GPU alone      {gpu8:8.1f} ms')
    print(f'  loader alone   {ld8:8.1f} ms')
    if gpu8 > ld8 * 2:
        print('\n  -> GPU BOUND. The loader is not the problem. Levers are')
        print('     bigger batch, fewer patches, or a smaller model.')
    elif ld8 > gpu8 * 2:
        print('\n  -> LOADER BOUND. Worth fixing the loader.')
    else:
        print('\n  -> BALANCED. Both matter; overlap them (more workers).')

    print('\n  epoch estimates at bs=8, 122,776 steps:')
    for k, v in results.items():
        if k.startswith('FULL'):
            print(f'    {k}: {v*122776/1000/3600:.1f} h/epoch')
    print('=' * 62, flush=True)


if __name__ == '__main__':
    main()
