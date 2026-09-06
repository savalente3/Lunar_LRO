# [source]: N. Khedkar (project partner) - 4_training/sweep_v2.py

# sweep_v2
# runs the v2 experiments one after another, each logged to mlflow per epoch and
# each writing its own checkpoint and history, so nothing overwrites anything
# else. staged on purpose: stage 1 varies only the architecture at the baseline
# loss, stage 2 varies only the loss on the architecture stage 1 picked, so both
# questions are answered without running the full grid.
# set 'stage' and, for stage 2, 'stage1_winner' in params below.
# parameters:
#         stage: 1 architecture screen | 2 loss tuning
#         stage1_winner: the v2 settings stage 1 selected
#         training_sample_percentage: % of each split, the screen runs small
# outputs:
#         checkpoints/<run_name>.keras, the best weights by val dice
#         checkpoints/history_<run_name>.csv, per epoch metrics
#         checkpoints/<run_name>_params.json, the params used
#         one mlflow run per configuration under 'lunar-crater-detection'


import sys
sys.path.append('../1_data_extraction')

import os
import json
import traceback
import numpy as np
import mlflow
import keras
import tensorflow as tf

from keras import ops

from LRO_data_class import getSplitIndices
from LRO_meemmap_class import MemmapPatchSequence
from model_v2 import buildModel
from losses_v2 import buildLoss


# only change: 'stage' and 'stage1_winner'
params = {
    'stage': 2,                         # 1 architecture screen | 2 loss tuning
    'dataset': 'alltiles',              # 'single' | 'alltiles'
    'dim': 256,
    'channels': 'both',                 # 'both' | 'wac' | 'dem'
    'n_filters': 32,
    'FL': 3,
    'init': 'he_normal',
    'lmbda': 1e-6,
    'dropout': 0.15,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'epochs': 10,
    'loss': 'binary_focal_crossentropy',
    'focal_alpha': 0.75,
    'focal_gamma': 2.0,
    'focal_class_balancing': True,
    'model': 'model_v2',
    'seed': 42,
    'patience': 5,
    'queue': 64,
    'training_sample_percentage': 5,    # % of each split, the screen runs small
}

# the architecture stage 1 selected, carried into every stage 2 run
stage1_winner = {
    'v2_depth': 3,
    'v2_dilation': True,
    'v2_attention': False,
}

# everything below follows from params
PATCHES_DIR = '../3_pre_processing/lunar_patches_alltiles'
CHECKPOINT_DIR = 'checkpoints'

params['input_channels'] = 2

if params['channels'] != 'both':
    params['input_channels'] = 1

# stage 1 holds the loss fixed and varies only the network
stage1_runs = {
    'model_v2A': {'v2_depth': 3, 'v2_dilation': False, 'v2_attention': False},
    'model_v2B': {'v2_depth': 3, 'v2_dilation': True, 'v2_attention': False},
    'model_v2C': {'v2_depth': 4, 'v2_dilation': True, 'v2_attention': False},
    'model_v2D': {'v2_depth': 3, 'v2_dilation': True, 'v2_attention': True},
}

# stage 2 holds the network fixed and varies only the loss
stage2_runs = {
    'model_v2E': {**stage1_winner, 'loss': 'tversky', 'tversky_alpha': 0.3, 'tversky_beta': 0.7},
    'model_v2F': {**stage1_winner, 'loss': 'tversky', 'tversky_alpha': 0.2, 'tversky_beta': 0.8},
    'model_v2G': {**stage1_winner, 'loss': 'focal_tversky', 'tversky_alpha': 0.3, 'tversky_beta': 0.7, 'tversky_gamma': 1.333},
    'model_v2H': {**stage1_winner, 'loss': 'binary_focal_crossentropy'},
}

if params['stage'] == 1:
    runs = stage1_runs
else:
    runs = stage2_runs


# diceCoef
# overlap between the predicted and true rim. val_loss is dominated by the
# background and barely moves with rim detection, so the runs are ranked and
# early stopped on this instead.
# parameters:
#         y_true: true mask
#         y_pred: predicted probabilities
#         smooth: added to both sides so an empty patch does not divide by zero
# outputs:
#         scalar tensor between 0 and 1
def diceCoef(y_true, y_pred, smooth=1.0):

    y_true = ops.reshape(y_true, (-1,))
    y_pred = ops.reshape(y_pred, (-1,))

    intersection = ops.sum(y_true * y_pred)

    return (2 * intersection + smooth) / (ops.sum(y_true) + ops.sum(y_pred) + smooth)


# softRecall
# recall on the rim class without thresholding, so it tracks the weakness the v2
# work is aimed at while training is still running.
# parameters:
#         y_true: true mask
#         y_pred: predicted probabilities
#         smooth: added to both sides so an empty patch does not divide by zero
# outputs:
#         scalar tensor between 0 and 1
def softRecall(y_true, y_pred, smooth=1.0):

    y_true = ops.reshape(y_true, (-1,))
    y_pred = ops.reshape(y_pred, (-1,))

    tp = ops.sum(y_true * y_pred)

    return (tp + smooth) / (ops.sum(y_true) + smooth)


# LiveMLflow
# logs every metric as the epoch ends, so the mlflow curves move during training
# rather than only once the run finishes.
class LiveMLflow(keras.callbacks.Callback):

    # on_epoch_end
    # logs one epoch's metrics.
    # parameters:
    #         epoch: epoch index
    #         logs: the metrics keras collected for the epoch
    def on_epoch_end(self, epoch, logs=None):

        for name, value in logs.items():
            mlflow.log_metric(name, float(value), step=epoch)


# runOne
# trains one configuration and logs it.
# parameters:
#         run_name: name for the checkpoint, history and mlflow run
#         overrides: the params this run changes
#         train_idx: training patch indices
#         val_idx: validation patch indices
def runOne(run_name, overrides, train_idx, val_idx):

    run_params = {**params, **overrides}

    keras.utils.set_random_seed(run_params['seed'])

    train_seq = MemmapPatchSequence(
        train_idx,
        PATCHES_DIR,
        run_params,
        augment_data=True,
        workers=1,
        max_queue_size=run_params['queue']
    )

    val_seq = MemmapPatchSequence(
        val_idx,
        PATCHES_DIR,
        run_params,
        augment_data=False
    )

    model = buildModel(run_params)

    model.compile(
        optimizer=keras.optimizers.Adam(run_params['learning_rate']),
        loss=buildLoss(run_params),
        metrics=[diceCoef, softRecall]
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_diceCoef',
            mode='max',
            patience=run_params['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'{CHECKPOINT_DIR}/{run_name}.keras',
            monitor='val_diceCoef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(f'{CHECKPOINT_DIR}/history_{run_name}.csv'),
        LiveMLflow(),
    ]

    with open(f'{CHECKPOINT_DIR}/{run_name}_params.json', 'w') as f:
        json.dump(run_params, f, indent=2)

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(run_params)

        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=run_params['epochs'],
            callbacks=callbacks,
        )

        mlflow.log_metric('best_val_dice', float(max(history.history['val_diceCoef'])))
        mlflow.log_metric('best_val_soft_recall', float(max(history.history['val_softRecall'])))


print(tf.config.list_physical_devices('GPU'))


train_idx, val_idx, test_idx = getSplitIndices(PATCHES_DIR)

n_train = int(len(train_idx) * params['training_sample_percentage'] / 100)
n_val = int(len(val_idx) * params['training_sample_percentage'] / 100)

train_idx = np.sort(np.random.default_rng(params['seed']).choice(train_idx, n_train, replace=False))
val_idx = np.sort(np.random.default_rng(params['seed']).choice(val_idx, n_val, replace=False))

print(f"sweep on {n_train} train / {n_val} val ({params['training_sample_percentage']}%)")
print(f"stage {params['stage']}: {list(runs)}")


# [source]: https://mlflow.org/docs/latest/python_api/mlflow.keras.html
# [example source]: https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py

mlflow.set_tracking_uri('mlruns')
mlflow.set_experiment('lunar-crater-detection')

for run_name, overrides in runs.items():

    print(f'\nstarting {run_name}: {overrides}', flush=True)

    # one configuration failing must not stop the rest of an overnight sweep
    try:
        runOne(run_name, overrides, train_idx, val_idx)
        print(f'{run_name} done', flush=True)
    except Exception:
        print(f'{run_name} failed, continuing to the next run:', flush=True)
        traceback.print_exc()

print('\nsweep complete.')
