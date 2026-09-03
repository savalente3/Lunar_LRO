#!/usr/bin/env python
# coding: utf-8

# # Training
# 
# Training only - metrics and figures are in `5_evaluation/evaluation.ipynb`.

# In[ ]:


import sys
sys.path.append('../1_data_extraction')

import os
import json
import numpy as np
import mlflow
import keras
import tensorflow as tf

import mlflow.keras
from LRO_data_class import getSplitIndices, patchesDirName, resolutionTag
from LRO_meemmap_class import MemmapPatchSequence, buildMemmaps, memmapsExist

DATASET = 'alltiles'

# resolution-tagged, so a 256 ppd run reads its own patches and cannot silently
# train on (or overwrite) the 128 ppd set
PATCHES_DIR = os.path.join('../3_pre_processing', patchesDirName(DATASET))

print(tf.config.list_physical_devices('GPU'))


# In[ ]:


train_idx, val_idx, test_idx = getSplitIndices(PATCHES_DIR)
print(f'train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}')


# ## Hyperparameters

# In[ ]:


SEED = 42

keras.utils.set_random_seed(SEED)

params = {
    'dataset': DATASET,
    'dim': 256,
    'channels': 'dem',                 # 'both' | 'wac' | 'dem'
    'input_channels': 1,                # 2 for both, 1 for ablations
    'n_filters': 32,                    # v1's n filters -> baseline overrides this
    'FL': 3,                            # kernel size
    'init': 'he_normal',
    'lmbda': 1e-6,                      # L2. NB paper 2.7 says 1e-5, repo says 1e-6 - sources disagree
    'dropout': 0.15,
    'learning_rate': 0.0001,
    'batch_size': 8,
    'epochs': 15,
    'loss': 'binary_focal_crossentropy',
    'focal_alpha': 0.75,                # weight on class 1, the rim. rare at 37:1 so it takes the lrager share
    'focal_gamma': 2.0,
    'focal_class_balancing': True,
    'model': 'U-Net-v1',                # 'U-Net-v1' | 'DeepMoon-baseline'. selects the model file
    'seed': SEED,                        # same batch order + augmentation across all runs
    'patience': 5,                       # epochs without improvement before stopping
    'queue': 64,                         # batches buffered ahead of the GPU
    'training_sample_percentage': 10,     # % of each split to use. None = the whole split
}

# 1 channel for the ablations, 2 for fusion - derived so it cannot drift from channels
if params['channels'] != 'both':
    params['input_channels'] = 1

# DeepMoon uses 112 filters (paper 2.3)
if params['model'] == 'DeepMoon-baseline':
    params['n_filters'] = 112


# ## Data
# 
# Streams from memory-mapped `.npy` arrays (built by `LRO_meemmap_class.buildMemmaps`, run once). Replaces the old `.npz`-per-epoch reload.

# In[ ]:


# built in-process by the memmap module, so training depends on no external script
if not memmapsExist(PATCHES_DIR):
    print('memmaps not found - building them first', flush=True)
    buildMemmaps(PATCHES_DIR)


# fewer patches per epoch. drawn across the whole split, so every tile stays
# represented - neighbouring patches overlap heavily so the rest add little
if params['training_sample_percentage']:
    n_train = int(len(train_idx) * params['training_sample_percentage'] / 100)
    n_val = int(len(val_idx) * params['training_sample_percentage'] / 100)

    train_idx = np.sort(np.random.default_rng(params['seed']).choice(train_idx, n_train, replace=False))
    val_idx = np.sort(np.random.default_rng(params['seed']).choice(val_idx, n_val, replace=False))

    # resolved counts go into params so MLflow records what was actually used
    params['train_patches'] = n_train
    params['val_patches'] = n_val

    print(f"subsampled to {n_train} train / {n_val} val ({params['training_sample_percentage']}%)")


# workers=1 keeps one background loader
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


# ## Model Architecture

# In[ ]:


# architectures live in their own files -> the pipeline, loss and seed are shared
# and only the network differs between runs. notes 9

if params['model'] == 'DeepMoon-baseline':
    from model_baseline import buildModel
else:
    from model_v1 import buildModel

model = buildModel(params)
model.summary()


# In[ ]:


# loss is built from params
# MLflow logs the params
loss_fn = keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=params['focal_class_balancing'],
    alpha=params['focal_alpha'],
    gamma=params['focal_gamma'],
)

model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']), loss=loss_fn)

# EarlyStopping: stop once val_loss stops improving for `patience` epochs.
# min_delta stays at the Keras default of 0 - at a focal loss of ~0.008 the real
# epoch-to-epoch gain near convergence is 3-9e-5, so 1e-4 scored it as noise and
# stopped runs early. notes 24.5
# CSVLogger writes each epoch as it finishes
os.makedirs('checkpoints', exist_ok=True)

if params['training_sample_percentage']:
    sample_tag = f"{params['training_sample_percentage']}pct"
else:
    sample_tag = 'all'
run_name = f"{params['model']}_{params['channels']}_{params['n_filters']}f_s{params['seed']}_{sample_tag}{resolutionTag()}"

# Always holds the best weights
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

# saved next to the checkpoint -> evaluation reads what this run was actually trained with
with open(f'checkpoints/{run_name}_params.json', 'w') as f:
    json.dump(params, f, indent=2)


# ## Training

# In[ ]:


# MLflow tracks every training run — hiperparameters, metrics, and the model itself
# mlflow ui for visualising


# [source]: https://mlflow.org/docs/latest/python_api/mlflow.keras.html
# [example source]: https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py

# explicit, matching evaluation.ipynb - without it MLflow writes ./mlruns relative to
# wherever the process was launched, which is how a second empty store appeared at the repo root
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

    # the checkpoint holds the lowest val_loss - log that, so the MLflow model and
    # the checkpoint on disk are the same network. notes 24.4
    model = keras.models.load_model(f'checkpoints/{run_name}.keras')
    mlflow.keras.log_model(model, 'model')

