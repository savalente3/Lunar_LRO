# train
# trains one model on the memory-mapped patches and logs the run to mlflow.
# set 'model' and 'channels' in params below, everything else follows from them.
# parameters:
#         dataset: 'single' | 'alltiles'
#         model: any model file in this folder
#         channels: 'both' | 'wac' | 'dem'
#         loss: 'binary_focal_crossentropy' or any keras loss name
#         training_sample_percentage: % of each split, None uses the whole split
# outputs:
#         checkpoints/<run_name>.keras, the best weights by val_loss
#         checkpoints/history_<run_name>.csv, per epoch train and val loss
#         checkpoints/<run_name>_params.json, the params used
#         an mlflow run under 'lunar-crater-detection'

import sys
sys.path.append('../1_data_extraction')

import os
import json
import importlib
import subprocess
import numpy as np
import mlflow
import keras
import tensorflow as tf

import mlflow.keras

from LRO_data_class import getSplitIndices
from LRO_meemmap_class import MemmapPatchSequence


# only change: 'model' and 'channels'
params = {
    'dataset': 'alltiles',              # 'single' | 'alltiles'
    'dim': 256,
    'channels': 'dem',                  # 'both' | 'wac' | 'dem'
    'n_filters': 32,
    'FL': 3,
    'init': 'he_normal',
    'lmbda': 1e-6,
    'dropout': 0.15,
    'learning_rate': 0.0001,
    'batch_size': 8,
    'epochs': 15,
    'loss': 'binary_focal_crossentropy',  # or any keras loss name
    'focal_alpha': 0.75,
    'focal_gamma': 2.0,
    'focal_class_balancing': True,
    'model': 'U_Net_v1',                # any model file in this folder
    'seed': 42,
    'patience': 5,
    'queue': 64,
    'training_sample_percentage': 10,   # % of each split
}

# Parameter controls
if params['dataset'] == 'single':
    PATCHES_DIR = '../3_pre_processing/lunar_patches'
else:
    PATCHES_DIR = '../3_pre_processing/lunar_patches_alltiles'


if params['channels'] == 'both':
    params['input_channels'] = 2
else:
    params['input_channels'] = 1

if params['model'] == 'baseline':
    params['n_filters'] = 112

if params['loss'] == 'binary_focal_crossentropy':
    loss_fn = keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=params['focal_class_balancing'],
        alpha=params['focal_alpha'],
        gamma=params['focal_gamma'],
    )
else:
    loss_fn = params['loss']



run_name = f"{params['model']}_{params['channels']}_{params['n_filters']}f_s{params['seed']}_{params['training_sample_percentage']}pct"
buildModel = importlib.import_module(params['model']).buildModel

keras.utils.set_random_seed(params['seed'])

print(tf.config.list_physical_devices('GPU'))


if not os.path.exists(os.path.join(PATCHES_DIR, 'wac_all.npy')):
    print('memmaps not found - running convert_to_memmap.py', flush=True)
    subprocess.run([sys.executable, 'convert_to_memmap.py'], cwd='../training', check=True)


train_idx, val_idx, test_idx = getSplitIndices(PATCHES_DIR)
print(f'train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}')


if params['training_sample_percentage']:
    n_train = int(len(train_idx) * params['training_sample_percentage'] / 100)
    n_val = int(len(val_idx) * params['training_sample_percentage'] / 100)

    train_idx = np.sort(np.random.default_rng(params['seed']).choice(train_idx, n_train, replace=False))
    val_idx = np.sort(np.random.default_rng(params['seed']).choice(val_idx, n_val, replace=False))

    params['train_patches'] = n_train
    params['val_patches'] = n_val

    print(f"subsampled to {n_train} train / {n_val} val ({params['training_sample_percentage']}%)")


train_seq = MemmapPatchSequence(
    train_idx,
    PATCHES_DIR,
    params,
    augment_data=True,
    workers=1,
    max_queue_size=params['queue']
)

val_seq = MemmapPatchSequence(
    val_idx,
    PATCHES_DIR,
    params,
    augment_data=False
)

print(f'{len(train_seq)} train steps, {len(val_seq)} val steps per epoch')

X, y = train_seq[0]
print(f'X {X.shape} {X.dtype}  [{X.min():.3f}, {X.max():.3f}]')
print(f'y {y.shape} {y.dtype}  crater pixels {y.mean()*100:.2f}%')


model = buildModel(params)
model.summary()


model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']), loss=loss_fn)

os.makedirs('checkpoints', exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['patience'],
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        f'checkpoints/{run_name}.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.CSVLogger(f'checkpoints/history_{run_name}.csv'),
]

with open(f'checkpoints/{run_name}_params.json', 'w') as f:
    json.dump(params, f, indent=2)


# [source]: https://mlflow.org/docs/latest/python_api/mlflow.keras.html
# [example source]: https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py

mlflow.set_tracking_uri('mlruns')
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

    model = keras.models.load_model(f'checkpoints/{run_name}.keras')
    mlflow.keras.log_model(model, 'model')
