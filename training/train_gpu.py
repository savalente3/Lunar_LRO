"""
Train one model on the FULL tile — DEM-only or dual-channel (WAC+DEM).

Streams patches from disk (90 GB won't fit in RAM). Trains one mode per run so
you can compare them afterwards.

Usage:
    python train_gpu.py dem     # DEM-only  -> model_dem.keras
    python train_gpu.py dual    # WAC+DEM   -> model_dual.keras

Out: model_<mode>.keras, history_<mode>.json
"""

import os
import sys
import json

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import keras
from keras import layers, ops

from gpu_data import MemmapSequence, getSplitIndices


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-3
BASE_FILTERS = 32
DEPTH = 4
POS_WEIGHT = 37.0        # matches full-tile ring-mask imbalance (~37:1)
SEED = 42
PATCHES_DIR = '../pre_processing/patches'

# cap training patches for a time-boxed run; set to None for the whole split
MAX_TRAIN = None
MAX_VAL = None

keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def build_unet(in_channels, base=BASE_FILTERS, depth=DEPTH):
    inp = keras.Input((256, 256, in_channels))
    skips, x = [], inp
    for d in range(depth):
        x = conv_block(x, base * 2 ** d)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, base * 2 ** depth)
    for d in reversed(range(depth)):
        x = layers.Conv2DTranspose(base * 2 ** d, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skips[d]])
        x = conv_block(x, base * 2 ** d)
    out = layers.Conv2D(1, 1, activation='sigmoid')(x)
    return keras.Model(inp, out, name=f'unet_{in_channels}ch')


def dice_coef(y_true, y_pred, smooth=1.0):
    yt = ops.reshape(y_true, (-1,))
    yp = ops.reshape(y_pred, (-1,))
    inter = ops.sum(yt * yp)
    return (2 * inter + smooth) / (ops.sum(yt) + ops.sum(yp) + smooth)


def combined_loss(pos_weight=POS_WEIGHT):
    def loss(y_true, y_pred):
        yp = ops.clip(y_pred, keras.config.epsilon(), 1 - keras.config.epsilon())
        wbce = -(pos_weight * y_true * ops.log(yp) + (1 - y_true) * ops.log(1 - yp))
        return ops.mean(wbce) + (1 - dice_coef(y_true, y_pred))
    return loss


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('dem', 'dual') else 'dem'
    print(f'=== training {mode} ===', flush=True)

    train_idx, val_idx, _ = getSplitIndices(PATCHES_DIR)
    if MAX_TRAIN:
        train_idx = train_idx[:MAX_TRAIN]
    if MAX_VAL:
        val_idx = val_idx[:MAX_VAL]
    print(f'train {len(train_idx)}, val {len(val_idx)}', flush=True)

    train_gen = MemmapSequence(train_idx, mode=mode, batch_size=BATCH_SIZE,
                               shuffle=True, patches_dir=PATCHES_DIR)
    val_gen = MemmapSequence(val_idx, mode=mode, batch_size=BATCH_SIZE,
                             shuffle=False, patches_dir=PATCHES_DIR)

    in_ch = 1 if mode == 'dem' else 2
    model = build_unet(in_ch)
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss=combined_loss(), metrics=[dice_coef])

    cbs = [
        keras.callbacks.ModelCheckpoint(
            f'model_{mode}.keras', monitor='val_dice_coef',
            mode='max', save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coef', mode='max', patience=5,
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=EPOCHS, callbacks=cbs, verbose=1)

    with open(f'history_{mode}.json', 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    print(f'Done. Saved model_{mode}.keras + history_{mode}.json', flush=True)


if __name__ == '__main__':
    main()