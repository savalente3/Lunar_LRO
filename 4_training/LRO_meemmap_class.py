# LRO_meemmap_class
# writes the .npz patch batches out as flat memory-mapped arrays, and serves
# keras batches from them during training.
# parameters:
#         none, each function and the class take their own
# outputs:
#         countPatches, buildMemmaps and the MemmapPatchSequence class

import os
import json
import glob
import numpy as np
import keras

from LRO_data_class import augment, percentileNormalise


# [source]: N. Khedkar (project partner) - training/convert_to_memmap.py
# countPatches
# total patches across the .npz batches. every batch holds file_size except the
# last, which is opened to read its real length.
# parameters:
#         patches_dir: directory holding the .npz batches
#         file_size: patches per .npz batch
# outputs:
#         total patch count, and the number of batches
def countPatches(patches_dir, file_size):

    files = sorted(glob.glob(os.path.join(patches_dir, 'X_mask_*.npz')), key=lambda f: int(f.split('_')[-1].split('.')[0]))

    last = np.load(files[-1])['arr_0']

    return (len(files) - 1) * file_size + len(last), len(files)


# [source]: N. Khedkar (project partner) - training/convert_to_memmap.py
# buildMemmaps
# writes the .npz batches out as three flat memory-mapped arrays, normalising
# once here instead of on every epoch. run once, after that a patch costs one
# seek and no decompression. patch order is preserved, so the existing split
# index files stay valid.
# parameters:
#         patches_dir: directory holding the .npz batches, also written to
#         file_size: patches per .npz batch, default 1000
# outputs:
#         wac_all.npy and dem_all.npy float16, mask_all.npy uint8, and meta.json
def buildMemmaps(patches_dir, file_size=1000):

    total, n_batches = countPatches(patches_dir, file_size)
    print(f'{total} patches across {n_batches} batches', flush=True)

    wac_all = np.lib.format.open_memmap(os.path.join(patches_dir, 'wac_all.npy'), mode='w+', dtype=np.float16, shape=(total, 256, 256))
    dem_all = np.lib.format.open_memmap(os.path.join(patches_dir, 'dem_all.npy'), mode='w+', dtype=np.float16, shape=(total, 256, 256))
    mask_all = np.lib.format.open_memmap(os.path.join(patches_dir, 'mask_all.npy'), mode='w+', dtype=np.uint8, shape=(total, 256, 256))

    position = 0

    for batch in range(n_batches):

        wac = np.load(os.path.join(patches_dir, f'X_wac_{batch}.npz'))['arr_0']
        dem = np.load(os.path.join(patches_dir, f'X_dem_{batch}.npz'))['arr_0']
        mask = np.load(os.path.join(patches_dir, f'X_mask_{batch}.npz'))['arr_0']

        for j in range(len(wac)):
            wac_all[position + j] = percentileNormalise(wac[j]).astype(np.float16)
            dem_all[position + j] = percentileNormalise(dem[j]).astype(np.float16)
            mask_all[position + j] = mask[j].astype(np.uint8)

        position += len(wac)

        if (batch + 1) % 10 == 0:
            print(f'  {batch + 1}/{n_batches} batches ({position} patches)', flush=True)

    wac_all.flush()
    dem_all.flush()
    mask_all.flush()

    with open(os.path.join(patches_dir, 'meta.json'), 'w') as f:
        json.dump({'total': int(total), 'shape': [256, 256], 'wac': 'float16', 'dem': 'float16', 'mask': 'uint8'}, f)

    print(f'done. {total} patches -> wac_all.npy, dem_all.npy, mask_all.npy', flush=True)


# [source]: N. Khedkar (project partner) - training/convert_to_memmap.py
# MemmapPatchSequence
# feeds keras batches straight off the memory-mapped patch arrays, so each
# patch costs one seek and no decompression.
class MemmapPatchSequence(keras.utils.PyDataset):

    # __init__
    # opens the memmaps and groups the wanted indices by the file they live in.
    # parameters:
    #         indices: patch indices this sequence serves
    #         patches_dir: directory holding wac_all.npy, dem_all.npy, mask_all.npy
    #         params: training params, read for dim, batch_size, channels, seed
    #         augment_data: apply flips and rotations, default True
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

    # buildOrder
    # shuffles the file order, and the positions inside each file, so a batch
    # is not all the same terrain.
    def buildOrder(self):

        order = []

        for f in self.rng.permutation(sorted(self.by_file)):

            positions = np.array(self.by_file[int(f)])

            if self.augment_data:
                positions = self.rng.permutation(positions)

            for p in positions:
                order.append((int(f), int(p)))

        self.order = order

    # __len__
    # outputs:
    #         int, number of full batches per epoch
    def __len__(self):
        return len(self.order) // self.batch_size

    # __getitem__
    # builds one batch, augmenting it when augment_data is set.
    # parameters:
    #         i: batch index
    # outputs:
    #         X (batch_size, dim, dim, input_channels) float32, y (batch_size, dim, dim, 1) float32
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

    # on_epoch_end
    # reshuffles the order between epochs when augmenting.
    def on_epoch_end(self):

        if self.augment_data:
            self.buildOrder()
