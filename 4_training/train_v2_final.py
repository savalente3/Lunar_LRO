#!/usr/bin/env python
# coding: utf-8
"""
Final Model V2 training run.

Trains the attention-gated Model V2 at the same budget as the baseline models
(10% of the training split, 15 epochs) so the result is directly comparable to
model_v1 (Sofia Valente) and the DeepMoon baseline (Silburt et al. 2019). This
script produces the model that is evaluated and reported.

The architecture and its rationale are documented in model_v2: a depth-3 U-Net
with a dilated bottleneck and attention gates on every skip connection. The
three mechanisms each target the small-crater recall limitation of the baseline
models. Depth 3 keeps small craters resolvable rather than pooling them to
sub-pixel. The dilated bottleneck widens the receptive field for context without
that extra pooling (Yu & Koltun 2016, arXiv:1511.07122). The attention gates let
the decoder localise the small, sparse crater targets and suppress irrelevant
terrain, a mechanism shown to raise sensitivity to small structures at
negligible cost (Oktay et al. 2018, arXiv:1804.03999; Schlemper et al. 2019).

The loss is the focal Tversky loss (alpha 0.3, beta 0.7, gamma 1.333), defined
in losses_v2. The Tversky loss penalises false negatives more heavily than false
positives, biasing the model toward recall (Salehi, Erdogmus & Gholipour 2017,
arXiv:1706.05721), and the focal exponent concentrates learning on the small,
hard craters where recall is weakest. Pairing this loss with attention gates for
small-target segmentation under class imbalance follows Abraham & Khan (2019,
arXiv:1810.07842), whose problem structure - small targets, high imbalance -
matches sub-2 km crater detection under the roughly 37:1 rim-to-background ratio.

The input channel set is chosen by CHANNELS below: 'both' for WAC and DEM fusion
or 'wac' for optical only. Run once per channel set to compare the two
modalities under the same architecture and loss. The DEM resolution follows the
project-wide DEM_PPD setting, so the run reads the patches for whichever
resolution the pipeline is configured to.

Uses the shared data loader and split utilities (LRO_meemmap_class and
LRO_data_class (Sofia Valente)), the Model V2 architecture (model_v2) and the
recall-oriented losses (losses_v2). MLflow logs parameters and per-epoch metrics
only, with no artifact calls, so the run completes without depending on a
writable artifact store.

Usage:
    python train_v2_final.py
"""

import sys
sys.path.append('../1_data_extraction')

import os
import json
import numpy as np
import mlflow
import keras
from keras import ops
import tensorflow as tf

from LRO_data_class import getSplitIndices, patchesDirName
from LRO_meemmap_class import MemmapPatchSequence
from model_v2 import buildModel
from losses_v2 import build_loss


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# input modality: 'both' = WAC + DEM fusion, 'wac' = optical only.
# run once per channel set (typically one per GPU) to compare the modalities.
CHANNELS = 'both'

DATASET = 'alltiles'
# patches directory follows the project-wide DEM_PPD resolution setting, so a
# 256 ppd run reads its own patches and cannot silently use the 128 ppd set
PATCHES_DIR = '../3_pre_processing/' + patchesDirName(DATASET)
CKPT_DIR = 'checkpoints'

SEED = 42

params = {
    'dataset': DATASET,
    'dim': 256,
    'channels': CHANNELS,
    'input_channels': 2 if CHANNELS == 'both' else 1,
    'n_filters': 32,
    'FL': 3,
    'init': 'he_normal',
    'lmbda': 1e-6,
    'dropout': 0.15,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'epochs': 15,
    'seed': SEED,
    'patience': 5,
    'queue': 64,
    'training_sample_percentage': 10,
    'model': 'U-Net-v2-attention',
    # loss: focal Tversky, recall-oriented (see losses_v2)
    'loss': 'focal_tversky',
    'tversky_alpha': 0.3,
    'tversky_beta': 0.7,
    'tversky_gamma': 1.333,
}


# ---------------------------------------------------------------------------
# training-time metrics on the rim class.
#
# Validation loss is dominated by the background and is a weak proxy for rim
# detection, so the run is monitored on rim overlap (Dice) and a soft,
# threshold-free recall. val_dice_coef is used for early stopping and
# checkpointing so the saved model is the best at rim detection.
# ---------------------------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1.0):
    yt = ops.reshape(y_true, (-1,))
    yp = ops.reshape(y_pred, (-1,))
    inter = ops.sum(yt * yp)
    return (2 * inter + smooth) / (ops.sum(yt) + ops.sum(yp) + smooth)


def soft_recall(y_true, y_pred, smooth=1.0):
    yt = ops.reshape(y_true, (-1,))
    yp = ops.reshape(y_pred, (-1,))
    tp = ops.sum(yt * yp)
    return (tp + smooth) / (ops.sum(yt) + smooth)


class LiveMLflow(keras.callbacks.Callback):
    """Logs each epoch's metrics to MLflow as they complete, so the dashboard
    curves update during training rather than only at the end."""
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        for k, v in logs.items():
            try:
                mlflow.log_metric(k, float(v), step=epoch)
            except Exception:
                pass


def main():
    print(tf.config.list_physical_devices('GPU'), flush=True)
    keras.utils.set_random_seed(params['seed'])

    train_idx, val_idx, _ = getSplitIndices(PATCHES_DIR)
    pct = params['training_sample_percentage']
    n_train = int(len(train_idx) * pct / 100)
    n_val = int(len(val_idx) * pct / 100)
    rng = np.random.default_rng(params['seed'])
    train_idx = np.sort(rng.choice(train_idx, n_train, replace=False))
    val_idx = np.sort(rng.choice(val_idx, n_val, replace=False))
    print(f'{n_train} train / {n_val} val ({pct}%), '
          f'channels={params["channels"]}, patches={PATCHES_DIR}', flush=True)

    train_seq = MemmapPatchSequence(train_idx, PATCHES_DIR, params,
                                    augment_data=True, workers=1,
                                    max_queue_size=params['queue'])
    val_seq = MemmapPatchSequence(val_idx, PATCHES_DIR, params,
                                  augment_data=False)

    model = buildModel(params)
    model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                  loss=build_loss(params),
                  metrics=[dice_coef, soft_recall])
    model.summary()

    os.makedirs(CKPT_DIR, exist_ok=True)

    # run_name records the model, loss, channel set and resolution, so the
    # checkpoint and its later evaluation refer to the same configuration
    res_tag = patchesDirName(DATASET).replace('lunar_patches_alltiles', '').lstrip('_') or '128ppd'
    run_name = (f"{params['model']}_{params['loss']}_{params['channels']}_"
                f"{params['n_filters']}f_s{params['seed']}_{pct}pct_{res_tag}")
    print('run_name:', run_name, flush=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_dice_coef', mode='max',
                                      patience=params['patience'],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(f'{CKPT_DIR}/{run_name}.keras',
                                        monitor='val_dice_coef', mode='max',
                                        save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(f'{CKPT_DIR}/history_{run_name}.csv'),
        LiveMLflow(),
    ]

    with open(f'{CKPT_DIR}/{run_name}_params.json', 'w') as f:
        json.dump(params, f, indent=2)

    mlflow.set_tracking_uri('mlruns')
    mlflow.set_experiment('lunar-crater-detection')
    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.log_params(params)
        except Exception:
            pass
        history = model.fit(train_seq, validation_data=val_seq,
                            epochs=params['epochs'], callbacks=callbacks)
        try:
            mlflow.log_metric('best_val_dice',
                              float(max(history.history['val_dice_coef'])))
            mlflow.log_metric('best_val_soft_recall',
                              float(max(history.history['val_soft_recall'])))
        except Exception:
            pass

    print(f'\n=== {run_name} done ===', flush=True)


if __name__ == '__main__':
    main()