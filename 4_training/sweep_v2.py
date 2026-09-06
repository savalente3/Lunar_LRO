#!/usr/bin/env python
# coding: utf-8
"""
Experiment sweep for Model V2 small-crater architectures and losses.

Runs a sequence of Model V2 configurations, each logged to MLflow per epoch so
the dashboard curves update during training, not only on completion. Every run
early-stops independently and writes its own checkpoint, per-epoch history and
MLflow run, so results do not overwrite one another.

RATIONALE
---------
The baseline model_v1 (Sofia Valente) is recall-limited: per-band crater recall
is 0.62 / 0.59 / 0.46 (1-2 / 2-5 / 5-10 km) against precision 0.79 / 0.94 /
0.96. Model V2 aims to raise recall without collapsing precision. Two
independent levers are examined: the architecture (whether a
resolution-preserving network detects more craters) and the loss (whether an
explicitly recall-oriented Tversky loss detects more). The sweep is staged -
architecture first at a fixed loss, then loss on the best architecture - so both
questions are answered without evaluating the full grid.

Validation loss alone is a weak discriminator here: it is dominated by the
background class and barely reflects rim detection. Each run therefore also
tracks a Dice coefficient and a soft recall on the rim class during training, so
the configurations can be separated on quantities that reflect the actual
objective rather than on background-dominated loss. These are training-time
proxies; the definitive per-band crater metrics come from the evaluation
pipeline run on the saved checkpoints.

RUNS
----
Stage 1 - architecture screen (loss held at the baseline focal cross-entropy so
that only the network varies):
  model_v2A  depth 3, no dilation, no attention.
             A plain depth-3 U-Net. Control condition that isolates the effect
             of the mechanisms added in the other runs.
  model_v2B  depth 3, dilated bottleneck, no attention.
             Adds dilated convolutions (rates 1, 2, 4) to widen the receptive
             field without additional pooling, testing whether preserved
             resolution recovers missed craters.
  model_v2C  depth 4, dilated bottleneck, no attention.
             A deeper encoder with dilation, testing whether additional context
             from an extra pooling stage helps or harms small-crater recall.
  model_v2D  depth 3, dilated bottleneck, attention gates.
             Adds attention on the skip connections, testing whether emphasising
             crater-like features surfaces the faint craters that are otherwise
             missed.

Stage 2 - loss and recall tuning (architecture fixed to the Stage 1 winner,
set via STAGE1_WINNER):
  model_v2E  Tversky loss (alpha 0.3, beta 0.7).
             Recall-oriented: false negatives are penalised more than false
             positives.
  model_v2F  Tversky loss (alpha 0.2, beta 0.8).
             A stronger recall weighting, probing how far recall can be raised
             before precision falls below the baseline band values.
  model_v2G  Focal Tversky loss (alpha 0.3, beta 0.7, gamma 1.333).
             Adds a focal exponent so hard, low-overlap craters dominate the
             gradient.
  model_v2H  Binary focal cross-entropy (control).
             The winning architecture under the baseline loss, isolating the
             loss effect from the architecture effect.

Each run uses a modest data subset and epoch budget: the sweep ranks
configurations rather than producing final models; the selected configuration
is retrained on the full training fraction for the final comparison.

Each run is wrapped so that a failure in one run is logged and the sweep
continues to the next, rather than aborting the whole batch. MLflow is used for
params and per-epoch metrics only; no MLflow artifact calls are made, so the
sweep does not depend on a writable artifact store and runs unattended.

DEPENDENCIES
------------
Uses the shared data loader and split utilities of this project
(LRO_meemmap_class and LRO_data_class (Sofia Valente)), the Model V2
architecture (model_v2) and the recall-oriented losses (losses_v2).

Usage:
    python sweep_v2.py

Set STAGE to 1 or 2 (and STAGE1_WINNER for Stage 2) before running. All runs
log to the MLflow experiment 'lunar-crater-detection'.
"""

import sys
sys.path.append('../1_data_extraction')

import os
import json
import traceback
import numpy as np
import mlflow
import keras
from keras import ops
import tensorflow as tf

from LRO_data_class import getSplitIndices
from LRO_meemmap_class import MemmapPatchSequence
from model_v2 import buildModel
from losses_v2 import build_loss


# ---------------------------------------------------------------------------
# sweep configuration
# ---------------------------------------------------------------------------

STAGE = 2                      # 1 = architecture screen, 2 = loss tuning
STAGE1_WINNER = dict(          # architecture selected after Stage 1
    v2_depth=3, v2_dilation=True, v2_attention=False,
)

DATASET = 'alltiles'
PATCHES_DIR = ('../3_pre_processing/lunar_patches' if DATASET == 'single'
               else '../3_pre_processing/lunar_patches_alltiles')

CKPT_DIR = 'checkpoints'

# screening budget. 5% subset and 10 epochs give a clearer separation between
# configurations than a smaller/shorter screen, while remaining much cheaper
# than the full training fraction used for the final confirmation run.
SAMPLE_PCT = 5
EPOCHS = 10
SEED = 42

BASE = {
    'dataset': DATASET, 'dim': 256, 'channels': 'both', 'input_channels': 2,
    'n_filters': 32, 'FL': 3, 'init': 'he_normal', 'lmbda': 1e-6,
    'dropout': 0.15, 'learning_rate': 1e-4, 'batch_size': 8, 'epochs': EPOCHS,
    'seed': SEED, 'patience': 5, 'queue': 64,
    'training_sample_percentage': SAMPLE_PCT, 'model': 'U-Net-v2',
    'loss': 'binary_focal_crossentropy', 'focal_alpha': 0.75,
    'focal_gamma': 2.0, 'focal_class_balancing': True,
}


def stage1_runs():
    return {
        'model_v2A': dict(v2_depth=3, v2_dilation=False, v2_attention=False),
        'model_v2B': dict(v2_depth=3, v2_dilation=True,  v2_attention=False),
        'model_v2C': dict(v2_depth=4, v2_dilation=True,  v2_attention=False),
        'model_v2D': dict(v2_depth=3, v2_dilation=True,  v2_attention=True),
    }


def stage2_runs():
    w = STAGE1_WINNER
    return {
        'model_v2E': dict(**w, loss='tversky',
                          tversky_alpha=0.3, tversky_beta=0.7),
        'model_v2F': dict(**w, loss='tversky',
                          tversky_alpha=0.2, tversky_beta=0.8),
        'model_v2G': dict(**w, loss='focal_tversky',
                          tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=1.333),
        'model_v2H': dict(**w, loss='binary_focal_crossentropy'),
    }


# ---------------------------------------------------------------------------
# training-time metrics on the rim class.
#
# Validation loss is dominated by the background and is a weak proxy for rim
# detection. Dice and a soft recall on the rim class track the objective more
# directly and give a stronger signal for ranking configurations. These are
# monitored per epoch (val_dice, val_soft_recall) and, because val_soft_recall
# reflects the recall-limited baseline's weakness, val_dice is also used as the
# early-stopping and checkpoint criterion.
# ---------------------------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1.0):
    yt = ops.reshape(y_true, (-1,))
    yp = ops.reshape(y_pred, (-1,))
    inter = ops.sum(yt * yp)
    return (2 * inter + smooth) / (ops.sum(yt) + ops.sum(yp) + smooth)


def soft_recall(y_true, y_pred, smooth=1.0):
    # soft (threshold-free) recall on the rim class: sum(p on true rim) / sum(true rim)
    yt = ops.reshape(y_true, (-1,))
    yp = ops.reshape(y_pred, (-1,))
    tp = ops.sum(yt * yp)
    return (tp + smooth) / (ops.sum(yt) + smooth)


# ---------------------------------------------------------------------------
# per-epoch MLflow logging, so the dashboard curves update during training.
# metrics are logged to the tracking store (not the artifact store), so this
# does not touch the artifact directory and cannot hit an artifact-path error.
# ---------------------------------------------------------------------------

class LiveMLflow(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        for k, v in logs.items():
            try:
                mlflow.log_metric(k, float(v), step=epoch)
            except Exception:
                # never let a logging hiccup interrupt training overnight
                pass


# ---------------------------------------------------------------------------
# a single training run
#
# The model and its per-epoch history are written to CKPT_DIR on disk by the
# ModelCheckpoint and CSVLogger callbacks - that is the authoritative record and
# it OVERWRITES any earlier checkpoint of the same name. MLflow is used only for
# params and per-epoch metrics, both of which write to the tracking store. No
# MLflow artifact calls are made (log_model / log_artifact), because the store's
# artifact root is not writable in this environment; avoiding them keeps every
# run crash-free for unattended overnight running.
# ---------------------------------------------------------------------------

def run_one(run_name, overrides, train_idx, val_idx):
    params = {**BASE, **overrides}

    keras.utils.set_random_seed(params['seed'])

    train_seq = MemmapPatchSequence(train_idx, PATCHES_DIR, params,
                                    augment_data=True, workers=1,
                                    max_queue_size=params['queue'])
    val_seq = MemmapPatchSequence(val_idx, PATCHES_DIR, params,
                                  augment_data=False)

    model = buildModel(params)
    model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                  loss=build_loss(params),
                  metrics=[dice_coef, soft_recall])

    os.makedirs(CKPT_DIR, exist_ok=True)
    # checkpoint / early-stop on val_dice (rim overlap), not val_loss, because
    # the objective is rim detection, not background-dominated loss.
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
        # params + per-epoch metrics only; no artifact writes (see note above)
        try:
            mlflow.log_params(params)
        except Exception:
            pass
        history = model.fit(train_seq, validation_data=val_seq,
                            epochs=params['epochs'], callbacks=callbacks)
        # log the best val_dice as a summary metric for easy run ranking
        try:
            mlflow.log_metric('best_val_dice',
                              float(max(history.history['val_dice_coef'])))
            mlflow.log_metric('best_val_soft_recall',
                              float(max(history.history['val_soft_recall'])))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print(tf.config.list_physical_devices('GPU'), flush=True)

    train_idx, val_idx, _ = getSplitIndices(PATCHES_DIR)
    n_train = int(len(train_idx) * SAMPLE_PCT / 100)
    n_val = int(len(val_idx) * SAMPLE_PCT / 100)
    rng = np.random.default_rng(SEED)
    train_idx = np.sort(rng.choice(train_idx, n_train, replace=False))
    val_idx = np.sort(rng.choice(val_idx, n_val, replace=False))
    print(f'sweep on {n_train} train / {n_val} val ({SAMPLE_PCT}%)', flush=True)

    runs = stage1_runs() if STAGE == 1 else stage2_runs()
    print(f'STAGE {STAGE}: {list(runs)}', flush=True)

    for run_name, overrides in runs.items():
        # NOTE: runs are NOT skipped - each overwrites any earlier checkpoint of
        # the same name, so the whole sweep is re-run cleanly at the new budget.
        print(f'\n>>> starting {run_name}: {overrides}', flush=True)
        try:
            run_one(run_name, overrides, train_idx, val_idx)
            print(f'=== {run_name} done ===', flush=True)
        except Exception:
            # one run failing must not stop the overnight batch - log and move on
            print(f'!!! {run_name} FAILED, continuing to next run:', flush=True)
            traceback.print_exc()

    print('\nsweep complete.', flush=True)


if __name__ == '__main__':
    main()