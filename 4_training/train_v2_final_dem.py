# [source]: N. Khedkar (project partner) - 4_training/train_v2_final_dem.py

# train_v2_final_dem
# the same run as train_dilated_U_net.py with the input channel fixed to dem.
# parameters:
#         none, the channel is fixed below
# outputs:
#         checkpoints/<run_name>.keras, the best weights by val_dice_coef
#         checkpoints/history_<run_name>.csv, per epoch metrics
#         checkpoints/<run_name>_params.json, the params used
#         an mlflow run under 'lunar-crater-detection'

import sys
sys.path.append('../1_data_extraction')

import os
import json
import numpy as np
import mlflow
import keras
from keras import ops
import tensorflow as tf

from LRO_data_class import getSplitIndices
from LRO_meemmap_class import MemmapPatchSequence
from dilated_U_net import buildModel
from losses import buildLoss


# input channels for this copy
CHANNELS = 'dem'

DATASET = 'alltiles'
# the 256 ppd all tiles patches written by data_pre_processing_alltiles.ipynb
PATCHES_DIR = '../3_pre_processing/lunar_patches_alltiles'
RES_TAG = '256ppd'
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
    'model': 'dilated_U_net',
    'loss': 'focal_tversky',
    'tversky_alpha': 0.3,
    'tversky_beta': 0.7,
    'tversky_gamma': 1.333,
}


# dice_coef
# overlap between the predicted and true rim. val_loss is dominated by the
# background, so early stopping and checkpointing follow this instead.
# parameters:
#         y_true: true mask
#         y_pred: predicted probabilities
#         smooth: added to both sides so an empty patch does not divide by zero
# outputs:
#         scalar tensor between 0 and 1
def dice_coef(y_true, y_pred, smooth=1.0):
    yt = ops.reshape(y_true, (-1,))
    yp = ops.reshape(y_pred, (-1,))
    inter = ops.sum(yt * yp)
    return (2 * inter + smooth) / (ops.sum(yt) + ops.sum(yp) + smooth)


# soft_recall
# recall on the rim class without thresholding, so it can be followed while
# training is still running.
# parameters:
#         y_true: true mask
#         y_pred: predicted probabilities
#         smooth: added to both sides so an empty patch does not divide by zero
# outputs:
#         scalar tensor between 0 and 1
def soft_recall(y_true, y_pred, smooth=1.0):
    yt = ops.reshape(y_true, (-1,))
    yp = ops.reshape(y_pred, (-1,))
    tp = ops.sum(yt * yp)
    return (tp + smooth) / (ops.sum(yt) + smooth)


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
        if not logs:
            return
        for k, v in logs.items():
            try:
                mlflow.log_metric(k, float(v), step=epoch)
            except Exception:
                pass


# main
# subsamples the splits, builds and compiles the model, then trains it and logs
# the run.
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
                  loss=buildLoss(params),
                  metrics=[dice_coef, soft_recall])
    model.summary()

    os.makedirs(CKPT_DIR, exist_ok=True)

    # the run name records model, loss, channels and resolution, so a checkpoint
    # and its evaluation refer to the same configuration
    run_name = (f"{params['model']}_{params['loss']}_{params['channels']}_"
                f"{params['n_filters']}f_s{params['seed']}_{pct}pct_{RES_TAG}")
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
