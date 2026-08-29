"""
Streams patches from the memory-mapped .npy arrays built by convert_to_memmap.py.

Same file/position ordering as the .npz loader it replaced - only the read changed.
"""

import os
import numpy as np
import keras

from LRO_data_class import augment


class MemmapPatchSequence(keras.utils.PyDataset):

    def __init__(self, indices, patches_dir, params, augment_data=True, **kwargs):
        super().__init__(**kwargs)

        self.augment_data = augment_data
        self.rng = np.random.default_rng(params['seed'])

        self.dim = params['dim']
        self.batch_size = params['batch_size']
        self.channels = params['channels']
        self.input_channels = params['input_channels']

        self.wac = np.load(os.path.join(patches_dir, 'wac_all.npy'), mmap_mode='r')
        self.dem = np.load(os.path.join(patches_dir, 'dem_all.npy'), mmap_mode='r')
        self.mask = np.load(os.path.join(patches_dir, 'mask_all.npy'), mmap_mode='r')

        self.by_file = {}

        for i in indices:
            self.by_file.setdefault(int(i // 1000), []).append(int(i % 1000))

        self.buildOrder()

    def buildOrder(self):

        order = []

        for f in self.rng.permutation(sorted(self.by_file)):

            positions = np.array(self.by_file[int(f)])

            if self.augment_data:
                positions = self.rng.permutation(positions)

            for p in positions:
                order.append((int(f), int(p)))

        self.order = order

    def __len__(self):
        return len(self.order) // self.batch_size

    def __getitem__(self, i):

        items = self.order[i * self.batch_size:(i + 1) * self.batch_size]

        X = np.zeros((len(items), self.dim, self.dim, self.input_channels), np.float32)
        y = np.zeros((len(items), self.dim, self.dim, 1), np.float32)

        for j, (file_num, position) in enumerate(items):

            patch_idx = file_num * 1000 + position

            wac_patch = np.asarray(self.wac[patch_idx], np.float32)
            dem_patch = np.asarray(self.dem[patch_idx], np.float32)
            mask_patch = np.asarray(self.mask[patch_idx], np.float32)

            if self.augment_data:
                wac_patch, dem_patch, mask_patch = augment(wac_patch, dem_patch, mask_patch, self.rng)

            match self.channels:

                case 'both':
                    X[j, :, :, 0] = wac_patch
                    X[j, :, :, 1] = dem_patch

                case 'wac':
                    X[j, :, :, 0] = wac_patch

                case 'dem':
                    X[j, :, :, 0] = dem_patch

                case _:
                    raise ValueError(f"channels must be 'both', 'wac' or 'dem', got {self.channels!r}")

            y[j, :, :, 0] = mask_patch

        return X, y

    def on_epoch_end(self):

        if self.augment_data:
            self.buildOrder()
