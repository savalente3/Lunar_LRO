# Model V1
#
# Training only - metrics and figures are in `5_evaluation/evaluation.ipynb`.

import sys
sys.path.append('../1_data_extraction')

import os
import subprocess
import numpy as np
import mlflow
import keras
import tensorflow as tf

import mlflow.keras
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout
from keras.regularizers import l2

from LRO_data_class import getSplitIndices, getNormalisedBatch, percentileNormalise, augment

DATASET = 'alltiles'

if DATASET == 'single':
    PATCHES_DIR = '../3_pre_processing/lunar_patches'
else:
    PATCHES_DIR = '../3_pre_processing/lunar_patches_alltiles'

print(tf.config.list_physical_devices('GPU'))


train_idx, val_idx, test_idx = getSplitIndices(PATCHES_DIR)
print(f'train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}')

# sanity check on one file
wac, dem, mask = getNormalisedBatch(batch_num=0, patches_dir=PATCHES_DIR)

print(f'wac  {wac.shape}  {wac.dtype}   [{wac.min():.3f}, {wac.max():.3f}]')
print(f'dem  {dem.shape}  {dem.dtype}   [{dem.min():.3f}, {dem.max():.3f}]')
print(f'mask {mask.shape}  {mask.dtype}   crater pixels {mask.mean()*100:.2f}%')


# Hyperparameters

# all hyperparameters in one place — change here and MLflow logs them automatically

# one place to change the seed - generators and weight init both read it
SEED = 42

# set_random_seed covers python/numpy/tf seeds but NOT cuDNN kernel choice.
# enable_op_determinism makes GPU ops deterministic, at a speed cost.
keras.utils.set_random_seed(SEED)
# disabled 2026-08-21: 4s/step against ~150ms/step for the same shape (notes 14.4). grappler
# failed on stateless_dropout, the kernel this forces. correlated, not confirmed - re-time it
# tf.config.experimental.enable_op_determinism()

params = {
    'dim': 256,
    'channels': 'both',                 # 'both' | 'wac' | 'dem'
    'input_channels': 2,                # 2 for both, 1 for ablations
    'n_filters': 32,                    # DeepMoon used 112 (paper 2.3)
    'FL': 3,                            # kernel size
    'init': 'he_normal',
    'lmbda': 1e-6,                      # L2. NB paper 2.7 says 1e-5, repo says 1e-6 - sources disagree
    'dropout': 0.15,
    'learning_rate': 0.0001,
    'batch_size': 8,
    'epochs': 20,
    'loss': 'binary_focal_crossentropy',
    'focal_alpha': 0.75,                # weight on class 1, the rim. rare at 37:1 so it takes the lrager share
    'focal_gamma': 2.0,
    'focal_class_balancing': True,
    'model': 'U-Net-v1',
    'seed': SEED,                        # same batch order + augmentation across all runs
    'patience': 3,                      # epochs without improvement before stopping
    'queue': 64,                        # batches buffered ahead of the GPU
    'train_subsample': 250000,          # patches drawn from the train split. None = all
}


# Data
#
# Streams from memory-mapped `.npy` arrays (`../training/convert_to_memmap.py`, run once - only its paths were changed). Replaces the old `.npz`-per-epoch reload.

# patches are read from memory-mapped .npy built once by ../training/convert_to_memmap.py
# (only its paths changed). percentileNormalise still runs on every patch - it happens in
# that conversion instead of on every epoch, so the loader must NOT normalise again.

# one-off, takes hours. skipped on every later run once the three .npy files exist
if not os.path.exists(os.path.join(PATCHES_DIR, 'wac_all.npy')):
    print('memmaps not found - running convert_to_memmap.py first', flush=True)
    subprocess.run([sys.executable, 'convert_to_memmap.py'], cwd='../training', check=True)


class MemmapPatchSequence(keras.utils.PyDataset):

    def __init__(self, indices, patches_dir, seed, augment_data=True, **kwargs):
        super().__init__(**kwargs)

        self.patches_dir = patches_dir
        self.augment_data = augment_data
        self.rng = np.random.default_rng(seed)

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
        return len(self.order) // params['batch_size']

    def __getitem__(self, i):

        items = self.order[i * params['batch_size']:(i + 1) * params['batch_size']]

        X = np.zeros((len(items), params['dim'], params['dim'], params['input_channels']), np.float32)
        y = np.zeros((len(items), params['dim'], params['dim'], 1), np.float32)

        for j, (file_num, position) in enumerate(items):

            patch_idx = file_num * 1000 + position

            wac_patch = np.asarray(self.wac[patch_idx], np.float32)
            dem_patch = np.asarray(self.dem[patch_idx], np.float32)
            mask_patch = np.asarray(self.mask[patch_idx], np.float32)

            if self.augment_data:
                wac_patch, dem_patch, mask_patch = augment(wac_patch, dem_patch, mask_patch, self.rng)

            if params['channels'] == 'both':
                X[j, :, :, 0] = wac_patch
                X[j, :, :, 1] = dem_patch
            elif params['channels'] == 'wac':
                X[j, :, :, 0] = wac_patch
            else:
                X[j, :, :, 0] = dem_patch

            y[j, :, :, 0] = mask_patch

        return X, y

    def on_epoch_end(self):

        if self.augment_data:
            self.buildOrder()


# fewer patches per epoch. drawn across the whole split, so every tile stays
# represented - neighbouring patches overlap heavily so the last 700k add little
if params['train_subsample']:
    train_idx = np.sort(np.random.default_rng(params['seed']).choice(
        train_idx, params['train_subsample'], replace=False))
    print(f'train subsampled to {len(train_idx)}')


# workers=1 keeps one background loader - more threads would thrash the file cache.
# the queue buffers batches so a file change does not stall the GPU
train_seq = MemmapPatchSequence(train_idx, PATCHES_DIR, params['seed'], augment_data=True, workers=1, max_queue_size=params['queue'])
val_seq = MemmapPatchSequence(val_idx, PATCHES_DIR, params['seed'], augment_data=False)

print(f'{len(train_seq)} train steps, {len(val_seq)} val steps per epoch')


# Model Architecture

img_input = keras.Input(shape=(params['dim'], params['dim'], params['input_channels']))

# Encoder1
a1 = Conv2D(
    params['n_filters'], 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(img_input)

a1 = Conv2D(
    params['n_filters'], 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(a1)

a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

# Encoder2
a2 = Conv2D(
    params['n_filters'] * 2, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(a1P)

a2 = Conv2D(
    params['n_filters'] * 2, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(a2)

a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

# Encoder3
a3 = Conv2D(
    params['n_filters'] * 4, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(a2P)

a3 = Conv2D(
    params['n_filters'] * 4, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(a3)

a3P = MaxPooling2D((2, 2), strides=(2, 2))(a3)

# Encoder4
a4 = Conv2D(
    params['n_filters'] * 8, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(a3P)

a4 = Conv2D(
    params['n_filters'] * 8, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(a4)

a4P = MaxPooling2D((2, 2), strides=(2, 2))(a4)

u = Conv2D(
    params['n_filters'] * 16, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(a4P)

u = Conv2D(
    params['n_filters'] * 16, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(u)

# Decoder1
d1CT = Conv2DTranspose(params['n_filters']*8, kernel_size=2, strides=2, padding='same')(u)
d1c = Concatenate()([d1CT, a4])
x1 = Dropout(params['dropout'])(d1c)

d1 = Conv2D(
    params['n_filters'] * 8, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(x1)

d1 = Conv2D(
    params['n_filters'] * 8, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(d1)

# Decoder2
d2CT = Conv2DTranspose(params['n_filters']*4, kernel_size=2, strides=2, padding='same')(d1)
d2c = Concatenate()([d2CT, a3])
x2 = Dropout(params['dropout'])(d2c)

d2 = Conv2D(
    params['n_filters'] * 4, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(x2)

d2 = Conv2D(
    params['n_filters'] * 4, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(d2)

# Deocder3
d3CT = Conv2DTranspose(params['n_filters']*2, kernel_size=2, strides=2, padding='same')(d2)
d3c = Concatenate()([d3CT, a2])
x3 = Dropout(params['dropout'])(d3c)

d3 = Conv2D(
    params['n_filters'] * 2, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(x3)

d3 = Conv2D(
    params['n_filters'] * 2, 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(d3)

# Decoder4
d4CT = Conv2DTranspose(params['n_filters'], kernel_size=2, strides=2, padding='same')(d3)
d4c = Concatenate()([d4CT, a1])
x4 = Dropout(params['dropout'])(d4c)

d4 = Conv2D(
    params['n_filters'], 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(x4)

d4 = Conv2D(
    params['n_filters'], 
    params['FL'], 
    activation='relu', 
    kernel_initializer=params['init'], 
    kernel_regularizer=l2(params['lmbda']), 
    padding='same'
)(d4)

# Output layer
output = Conv2D(1, 1, activation='sigmoid')(d4)
model = keras.Model(img_input, output)
model.summary()


# loss is constructed from params so the two can't drift apart
# MLflow logs the params
loss_fn = keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=params['focal_class_balancing'],
    alpha=params['focal_alpha'],
    gamma=params['focal_gamma'],
)

model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']), loss=loss_fn)

# EarlyStopping: stop once val_losos stops improving for `patience` epochs
# CSVLogger writes each epoch as it finishes - a crash keeps the curve
os.makedirs('checkpoints', exist_ok=True)

run_name = f"{params['model']}_{params['channels']}_{params['n_filters']}f_s{params['seed']}"

# one file per run, overwritten each time val_loss improves - it always holds the best weights
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True, verbose=1, min_delta=1e-4),
    keras.callbacks.ModelCheckpoint(
        f'checkpoints/{run_name}.keras',
        monitor='val_loss', save_best_only=True, verbose=1),
    keras.callbacks.CSVLogger(f'checkpoints/history_{run_name}.csv'),
]


# Training

# MLflow tracks every training run — hiperparameters, metrics, and the model itself
# mlflow ui for visualising


# [source]: https://mlflow.org/docs/latest/python_api/mlflow.keras.html
# [example source]: https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py

mlflow.set_experiment('lunar-crater-detection')

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=params['epochs'],
        callbacks=callbacks,
    )

    for epoch, (tl, vl) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        mlflow.log_metric('train_loss', tl, step=epoch)
        mlflow.log_metric('val_loss', vl, step=epoch)

    mlflow.keras.log_model(model, 'model')
