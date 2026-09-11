#!/usr/bin/env python
# coding: utf-8

# # Evaluation

# In[ ]:


import sys
sys.path.append('../1_data_extraction')

import os
import json
import numpy as np
import pandas as pd
import mlflow
import keras
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from crater_extraction import template_match_t, match_coords, filter_to_detectable, filter_edge_craters, truth_coords_for_patch, matchesExcludedCrater
from LRO_data_class import getSplitIndices, percentileNormalise, getLunarRobbinsLabels


# In[ ]:


# only change: 'model'
channels = ['wac', 'dem', 'both']

params = {
    'dataset': 'alltiles',              # 'single' | 'alltiles'
    'patch_source': 'memmap',           # 'memmap' | 'npz'
    'model': 'U_Net_v1',                # any model file in 4_training
    'n_filters': 32,
    'seed': 42,
    'training_sample_percentage': 10,   # % of each split
    'n_sweep': 200,
    'n_eval': 2000,
}


# In[ ]:


PATCHES_DIR = '../3_pre_processing/lunar_patches_alltiles'

if params['dataset'] == 'single':
    LABELS_CSV = '../2_data_preparation/filtered_labels.csv'
else:
    LABELS_CSV = '../2_data_preparation/filtered_labels_alltiles.csv'

CHECKPOINT_DIR = '../4_training/checkpoints'
RESULTS_ROOT = os.path.join('results', params['model'])

kept_labels = pd.read_csv(os.path.join(PATCHES_DIR, 'kept_labels.csv'), low_memory=False)
filtered_labels = pd.read_csv(LABELS_CSV)

train_idx, val_idx, test_idx = getSplitIndices(PATCHES_DIR)

mlflow.set_tracking_uri('../4_training/mlruns')
mlflow.set_experiment('lunar-crater-detection')


# In[ ]:


if 'tile' not in kept_labels.columns:
    kept_labels['tile'] = 'single'

tile_craters = {}
tile_large = {}

large_craters = getLunarRobbinsLabels()
large_craters = large_craters[large_craters['DIAM_CIRC_IMG'] >= 10]

for tile_name in kept_labels['tile'].dropna().unique():

    tile_rows = kept_labels[kept_labels['tile'] == tile_name]
    crater_rows = tile_rows.dropna(subset=['LON_CIRC_IMG', 'wac_col'])

    col_map = np.polyfit(crater_rows['LON_CIRC_IMG'], crater_rows['wac_col'], 1)
    row_map = np.polyfit(crater_rows['LAT_CIRC_IMG'], crater_rows['wac_row'], 1)

    tile_craters[tile_name] = (np.polyval(col_map, filtered_labels['LON_CIRC_IMG'].values), np.polyval(row_map, filtered_labels['LAT_CIRC_IMG'].values), filtered_labels['DIAM_CIRC_IMG'].values)
    tile_large[tile_name] = (np.polyval(col_map, large_craters['LON_CIRC_IMG'].values), np.polyval(row_map, large_craters['LAT_CIRC_IMG'].values), large_craters['DIAM_CIRC_IMG'].values)


print(f'{len(large_craters)} catalogue craters >= 10 km held back for false-positive exclusion')


# In[ ]:


# safeDivide
# divides, returning 0 when the denominator is 0.
# parameters:
#         numerator: number
#         denominator: number
# outputs:
#         float
def safeDivide(numerator, denominator):
    return numerator / denominator if denominator else 0


# stackOrEmpty
# stacks a list of arrays, returning a correctly shaped empty array when the
# list holds nothing.
# parameters:
#         parts: list of arrays
#         columns: column count for the empty case
# outputs:
#         array (n, columns)
def stackOrEmpty(parts, columns):
    return np.vstack(parts) if parts else np.empty((0, columns))


# [source]: N. Khedkar (project partner) - 5_evaluation/evaluation_memmap.py
# convert_to_memmap.py already applied the same percentileNormalise before
# writing these, so the memmap path must not normalise again. it also indexes
# flat, since the three arrays hold every patch rather than 1000 per file.

if params['patch_source'] == 'memmap':
    wac_all = np.load(os.path.join(PATCHES_DIR, 'wac_all.npy'), mmap_mode='r')
    dem_all = np.load(os.path.join(PATCHES_DIR, 'dem_all.npy'), mmap_mode='r')
    mask_all = np.load(os.path.join(PATCHES_DIR, 'mask_all.npy'), mmap_mode='r')


loaded = {}


# loadPatchFile
# loads the .npz batch holding a patch, and keeps it so consecutive patches
# from the same file are not reloaded.
# parameters:
#         patch_idx: global patch index
def loadPatchFile(patch_idx):
    file_num = int(patch_idx // 1000)

    if loaded.get('file') != file_num:
        loaded['wac'] = np.load(os.path.join(PATCHES_DIR, f'X_wac_{file_num}.npz'))['arr_0']
        loaded['dem'] = np.load(os.path.join(PATCHES_DIR, f'X_dem_{file_num}.npz'))['arr_0']
        loaded['mask'] = np.load(os.path.join(PATCHES_DIR, f'X_mask_{file_num}.npz'))['arr_0']
        loaded['file'] = file_num


# patchWac
# the normalised WAC image for one patch.
# parameters:
#         patch_idx: global patch index
# outputs:
#         array (256, 256) float
def patchWac(patch_idx):

    if params['patch_source'] == 'memmap':
        return np.asarray(wac_all[patch_idx], np.float32)

    loadPatchFile(patch_idx)

    return percentileNormalise(loaded['wac'][patch_idx % 1000])


# patchDem
# the normalised DEM image for one patch.
# parameters:
#         patch_idx: global patch index
# outputs:
#         array (256, 256) float
def patchDem(patch_idx):

    if params['patch_source'] == 'memmap':
        return np.asarray(dem_all[patch_idx], np.float32)

    loadPatchFile(patch_idx)

    return percentileNormalise(loaded['dem'][patch_idx % 1000])


# patchInput
# builds the model input for one patch, normalised and stacked to match the
# channel being evaluated.
# parameters:
#         patch_idx: global patch index
#         channel: 'both' | 'wac' | 'dem'
# outputs:
#         array (256, 256, 1) or (256, 256, 2) float
def patchInput(patch_idx, channel):

    wac_patch = patchWac(patch_idx)
    dem_patch = patchDem(patch_idx)

    if channel == 'both':
        return np.stack([wac_patch, dem_patch], axis=-1)
    elif channel == 'wac':
        return wac_patch[..., None]
    else:
        return dem_patch[..., None]


# patchMask
# the stored ring mask for one patch.
# parameters:
#         patch_idx: global patch index
# outputs:
#         array (256, 256) uint8
def patchMask(patch_idx):

    if params['patch_source'] == 'memmap':
        return np.asarray(mask_all[patch_idx])

    loadPatchFile(patch_idx)

    return loaded['mask'][patch_idx % 1000]


# In[ ]:


# patchTruth
# the catalogue craters for one patch, in patch pixel coordinates, filtered to
# those template matching could actually find.
# parameters:
#         patch_idx: global patch index
# outputs:
#         array (n, 3), each row x, y, radius in px
def patchTruth(patch_idx):
    row = kept_labels.iloc[patch_idx]

    tile_wac_col, tile_wac_row, tile_diameters = tile_craters[row['tile']]
    truth = truth_coords_for_patch(row['center_col'], row['center_row'], row['patch_lat'], tile_wac_col, tile_wac_row, tile_diameters)

    return filter_edge_craters(filter_to_detectable(truth))


# patchLarge
# craters of 10 km or more near one patch, used to excuse detections that fall
# outside the label set.
# parameters:
#         patch_idx: global patch index
# outputs:
#         array (n, 3), each row x, y, radius in px
def patchLarge(patch_idx):
    row = kept_labels.iloc[patch_idx]

    large_col, large_row, large_diameters = tile_large[row['tile']]

    return truth_coords_for_patch(row['center_col'], row['center_row'], row['patch_lat'], large_col, large_row, large_diameters, margin=600)


# ## Threshold sweep
# 
# Swept on validation, applied once to test.

# In[ ]:


thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]


# sweepThresholds
# runs the model over validation patches at each threshold and picks the one
# with the best crater F1.
# parameters:
#         model: loaded keras model
#         channel: 'both' | 'wac' | 'dem'
# outputs:
#         best_threshold float, and a dataframe of threshold, precision, recall, f1
def sweepThresholds(model, channel):

    sweep_rng = np.random.default_rng(params['seed'])
    sweep_idx = np.sort(sweep_rng.choice(val_idx, size=params['n_sweep'], replace=False))

    predictions = []
    truths = []
    masks = []
    larges = []

    for patch_idx in sweep_idx:
        predictions.append(model.predict(patchInput(patch_idx, channel)[None, ...], verbose=0)[0, :, :, 0])
        truths.append(patchTruth(patch_idx))
        masks.append(patchMask(patch_idx) > 0)
        larges.append(patchLarge(patch_idx))

    rows = []

    for threshold in thresholds:

        swept_match = 0
        swept_detected = 0
        swept_truth = 0

        for prediction, truth, true_rim, large in zip(predictions, truths, masks, larges):

            detections = filter_edge_craters(template_match_t(prediction.copy(), target_thresh=threshold))
            match_count, detection_count, truth_count, _, false_positives, _ = match_coords(truth, detections)

            excluded = matchesExcludedCrater(false_positives, large).sum()

            swept_match += match_count
            swept_detected += detection_count - excluded
            swept_truth += truth_count

        precision = safeDivide(swept_match, swept_detected)
        recall = safeDivide(swept_match, swept_truth)

        rows.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': safeDivide(2 * precision * recall, precision + recall),
        })

    sweep_table = pd.DataFrame(rows)
    best_threshold = sweep_table.loc[sweep_table['f1'].idxmax(), 'threshold']

    return best_threshold, sweep_table


# ## Crater and pixel metrics

# In[ ]:


# evaluateChannel
# runs the model over the test patches at the chosen threshold and collects
# crater level, pixel level and per patch results.
# parameters:
#         model: loaded keras model
#         channel: 'both' | 'wac' | 'dem'
#         best_threshold: threshold chosen on validation
# outputs:
#         headline dict of metrics, and an arrays dict holding matched (n, 6),
#         false_positives (n, 3), excluded (n, 3), truth_radii (n,),
#         per_patch dataframe and pixel counts
def evaluateChannel(model, channel, best_threshold):

    rng = np.random.default_rng(params['seed'])
    eval_idx = np.sort(rng.choice(test_idx, size=params['n_eval'], replace=False))

    total_match = 0
    total_detected = 0
    total_truth = 0

    pixel_tp = 0
    pixel_fp = 0
    pixel_fn = 0
    pixel_tn = 0

    matched_all = []
    false_positives_all = []
    excluded_all = []
    truth_radii_all = []
    per_patch_rows = []

    for patch_idx in eval_idx:

        prediction = model.predict(patchInput(patch_idx, channel)[None, ...], verbose=0)[0, :, :, 0]
        detections = filter_edge_craters(template_match_t(prediction.copy(), target_thresh=best_threshold))
        truth = np.asarray(patchTruth(patch_idx)).reshape(-1, 3)

        predicted_rim = prediction >= best_threshold
        true_rim = patchMask(patch_idx) > 0
        both = (predicted_rim & true_rim).sum()

        pixel_tp += both
        pixel_fp += predicted_rim.sum() - both
        pixel_fn += true_rim.sum() - both
        pixel_tn += predicted_rim.size - predicted_rim.sum() - true_rim.sum() + both

        match_count, detection_count, truth_count, matched_pairs, false_positives, _ = match_coords(truth, detections)

        matched_pairs = np.asarray(matched_pairs).reshape(-1, 6)
        false_positives = np.asarray(false_positives).reshape(-1, 3)

        explained = matchesExcludedCrater(false_positives, patchLarge(patch_idx))
        detection_count -= int(explained.sum())

        excluded_all.append(false_positives[explained])
        false_positives = false_positives[~explained]

        matched_all.append(matched_pairs)
        false_positives_all.append(false_positives)
        truth_radii_all.append(truth[:, 2])

        total_match += match_count
        total_detected += detection_count
        total_truth += truth_count

        per_patch_rows.append({
            'patch_idx': int(patch_idx),
            'tp': int(match_count),
            'detected': int(detection_count),
            'truth': int(truth_count),
            'tp_small': int((matched_pairs[:, 5] < 15).sum()),
            'truth_small': int((truth[:, 2] < 15).sum()),
            'unmatched': int(detection_count - match_count),
        })

    precision = safeDivide(total_match, total_detected)
    recall = safeDivide(total_match, total_truth)

    pixel_precision = safeDivide(pixel_tp, pixel_tp + pixel_fp)
    pixel_recall = safeDivide(pixel_tp, pixel_tp + pixel_fn)

    arrays = {
        'matched': stackOrEmpty(matched_all, 6),
        'false_positives': stackOrEmpty(false_positives_all, 3),
        'excluded': stackOrEmpty(excluded_all, 3),
        'truth_radii': np.concatenate(truth_radii_all),
        'per_patch': pd.DataFrame(per_patch_rows),
        'pixels': {'tp': int(pixel_tp), 'fp': int(pixel_fp), 'fn': int(pixel_fn), 'tn': int(pixel_tn)},
    }

    headline = {
        'channel': channel,
        'best_threshold': float(best_threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(safeDivide(2 * precision * recall, precision + recall)),
        'tp': int(total_match),
        'detected': int(total_detected),
        'truth': int(total_truth),
        'excluded': int(len(arrays['excluded'])),
        'pixel_precision': float(pixel_precision),
        'pixel_recall': float(pixel_recall),
        'pixel_dice': float(safeDivide(2 * pixel_precision * pixel_recall, pixel_precision + pixel_recall)),
    }

    return headline, arrays


# ## Per-patch metrics
# 
# DeepMoon reports means across images, so these are the comparable figures.

# In[ ]:


# perPatchStats
# turns the per patch counts into means and standard deviations, the estimator
# DeepMoon reports.
# parameters:
#         per_patch: dataframe of per patch counts
# outputs:
#         dict of mean and sd for recall, recall_small, precision, new_crater_pct
def perPatchStats(per_patch):

    per_patch['recall'] = per_patch['tp'] / per_patch['truth'].replace(0, np.nan)
    per_patch['recall_small'] = per_patch['tp_small'] / per_patch['truth_small'].replace(0, np.nan)
    per_patch['precision'] = per_patch['tp'] / per_patch['detected'].replace(0, np.nan)
    per_patch['new_crater_pct'] = per_patch['unmatched'] / (per_patch['unmatched'] + per_patch['truth']).replace(0, np.nan)

    stats = {}

    for name in ['recall', 'recall_small', 'precision', 'new_crater_pct']:
        values = per_patch[name].dropna()
        stats[name] = {'mean': float(values.mean()), 'sd': float(values.std(ddof=1))}

    return stats


# ## Run every channel

# In[ ]:


results = {}

for channel in channels:

    print(f'evaluating {channel}', flush=True)

    run_name = f"{params['model']}_{channel}_{params['n_filters']}f_s{params['seed']}_{params['training_sample_percentage']}pct"
    checkpoint = os.path.join(CHECKPOINT_DIR, f'{run_name}.keras')
    results_dir = os.path.join(RESULTS_ROOT, channel)

    os.makedirs(results_dir, exist_ok=True)

    model = keras.models.load_model(checkpoint)

    best_threshold, sweep_table = sweepThresholds(model, channel)
    headline, arrays = evaluateChannel(model, channel, best_threshold)

    headline['per_patch'] = perPatchStats(arrays['per_patch'])

    sweep_table.to_csv(os.path.join(results_dir, 'sweep.csv'), index=False)
    arrays['per_patch'].to_csv(os.path.join(results_dir, 'per_patch.csv'), index=False)

    with open(os.path.join(results_dir, 'headline.json'), 'w') as handle:
        json.dump(headline, handle, indent=2)

    results[channel] = {'headline': headline, 'arrays': arrays, 'sweep': sweep_table}

    print(f"  threshold {best_threshold}   P {headline['precision']:.3f}   R {headline['recall']:.3f}   F1 {headline['f1']:.3f}")

    with mlflow.start_run(run_name=f'eval-{run_name}'):
        mlflow.log_params({'channel': channel, 'checkpoint': checkpoint, 'target_thresh': best_threshold})
        mlflow.log_metrics({k: v for k, v in headline.items() if isinstance(v, (int, float))})


# In[ ]:


summary = pd.DataFrame([results[c]['headline'] for c in channels])

display(summary[['channel', 'best_threshold', 'precision', 'recall', 'f1', 'tp', 'detected', 'truth']].round(3))


# ## Channel comparison

# In[ ]:


channel_colours = {'wac': 'tab:blue', 'dem': 'tab:orange', 'both': 'tab:green'}
channel_labels = {'wac': 'WAC only', 'dem': 'DEM only', 'both': 'WAC + DEM'}

deepmoon = {'recall': 0.57, 'precision': 0.80, 'f1': 0.666}

metrics = ['recall', 'precision', 'f1']

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(metrics))
width = 0.2

for n, channel in enumerate(channels):

    values = [results[channel]['headline'][m] for m in metrics]

    bars = ax.bar(x + n * width, values, width, color=channel_colours[channel], label=channel_labels[channel])
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

bars = ax.bar(x + len(channels) * width, [deepmoon[m] for m in metrics], width, color='grey', label='DeepMoon post-CNN')
ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

ax.set_xticks(x + 1.5 * width, ['recall', 'precision', 'F1'])
ax.set_ylabel('score')
ax.set_ylim(0, 1)
ax.legend()
ax.set_title('Crater-level metrics by input channel')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'comparison.png'), dpi=200)
plt.show()


# ## Precision-recall against DeepMoon

# In[ ]:


fig, ax = plt.subplots(figsize=(7, 6))

for channel in channels:

    sweep = results[channel]['sweep'].sort_values('recall')
    headline = results[channel]['headline']
    colour = channel_colours[channel]

    ax.plot(sweep['recall'], sweep['precision'], marker='o', markersize=4, color=colour, linewidth=1.4, label=channel_labels[channel])

    ax.scatter([headline['recall']], [headline['precision']], s=110, color=colour, edgecolor='black', zorder=5, linewidth=0.9)

    ax.annotate(f"F1 {headline['f1']:.3f}", (headline['recall'], headline['precision']),
                textcoords='offset points', xytext=(8, -12), fontsize=8, color=colour)

ax.errorbar([0.57], [0.80], xerr=[0.20], yerr=[0.15], fmt='o', markersize=9,
            color='black', ecolor='grey', elinewidth=1.2, capsize=4, zorder=6,
            label='DeepMoon post-CNN')

ax.set_xlabel('recall')
ax.set_ylabel('precision')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'precision_recall.png'), dpi=200)
plt.show()


# ## Precision and recall by crater diameter

# In[ ]:


bin_edges = [5, 10, 25, 50]
bin_labels = ['1-2 km', '2-5 km', '5-10 km']

fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4), sharey=True)

for ax, channel in zip(axes, channels):

    matched = results[channel]['arrays']['matched']
    truth_radii = results[channel]['arrays']['truth_radii']
    fp_radii = results[channel]['arrays']['false_positives'][:, 2]

    bin_recall = []
    bin_precision = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):

        truth_in_bin = ((truth_radii >= lower) & (truth_radii < upper)).sum()
        matched_in_bin = ((matched[:, 5] >= lower) & (matched[:, 5] < upper)).sum()
        bin_recall.append(safeDivide(matched_in_bin, truth_in_bin))

        det_in_bin = ((matched[:, 2] >= lower) & (matched[:, 2] < upper)).sum()
        fp_in_bin = ((fp_radii >= lower) & (fp_radii < upper)).sum()
        bin_precision.append(safeDivide(det_in_bin, det_in_bin + fp_in_bin))

    x = np.arange(len(bin_labels))

    ax.bar(x - 0.2, bin_recall, 0.4, label='recall', color='tab:blue')
    ax.bar(x + 0.2, bin_precision, 0.4, label='precision', color='tab:orange')

    ax.set_xticks(x, bin_labels)
    ax.set_ylim(0, 1)
    ax.set_title(channel_labels[channel])

axes[0].set_ylabel('score')
axes[0].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'diameter_bins.png'), dpi=200)
plt.show()


# ## Threshold sweep curves

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for channel in channels:

    sweep = results[channel]['sweep']
    colour = channel_colours[channel]

    axes[0].plot(sweep['threshold'], sweep['f1'], marker='o', color=colour, label=f'{channel_labels[channel]} F1')
    axes[0].plot(sweep['threshold'], sweep['precision'], marker='^', linestyle='--', color=colour, alpha=0.6, label=f'{channel_labels[channel]} P')
    axes[0].plot(sweep['threshold'], sweep['recall'], marker='v', linestyle=':', color=colour, alpha=0.6, label=f'{channel_labels[channel]} R')

    axes[1].plot(sweep['recall'], sweep['precision'], marker='o', color=colour, label=channel_labels[channel])

axes[0].set_xlabel('target_thresh')
axes[0].set_ylabel('score')
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=7, ncol=3)

axes[1].set_xlabel('recall')
axes[1].set_ylabel('precision')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'sweep.png'), dpi=200)
plt.show()


# ## Size-frequency distribution

# In[ ]:


diameter_bins = np.logspace(np.log10(1), np.log10(10), 15)

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(results[channels[0]]['arrays']['truth_radii'] * 2 * 0.1, bins=diameter_bins,
        histtype='step', color='black', label='Robbins (in patch)')

for channel in channels:

    arrays = results[channel]['arrays']
    detected = np.concatenate([arrays['matched'][:, 2], arrays['false_positives'][:, 2]]) * 2 * 0.1

    ax.hist(detected, bins=diameter_bins, histtype='step', color=channel_colours[channel], label=channel_labels[channel])

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('diameter (km)')
ax.set_ylabel('count')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'size_frequency.png'), dpi=200)
plt.show()


# ## Position and radius error

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for channel in channels:

    matched = results[channel]['arrays']['matched']
    mean_radius = (matched[:, 2] + matched[:, 5]) / 2

    errors = [
        abs(matched[:, 0] - matched[:, 3]) / mean_radius,
        abs(matched[:, 1] - matched[:, 4]) / mean_radius,
        abs(matched[:, 2] - matched[:, 5]) / mean_radius,
    ]

    for ax, values, name in zip(axes, errors, ['x', 'y', 'radius']):
        ax.hist(values, bins=40, histtype='step', color=channel_colours[channel], label=f'{channel_labels[channel]} (med {np.median(values):.3f})')
        ax.axvline(np.median(values), color=channel_colours[channel], linestyle='--', alpha=0.6)
        ax.set_xlabel(f'{name} fractional error')

for ax in axes:
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'errors.png'), dpi=200)
plt.show()


# ## False positives by radius

# In[ ]:


radius_bins = np.arange(5, 52, 1)

fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4), sharey=True)

for ax, channel in zip(axes, channels):

    arrays = results[channel]['arrays']

    ax.hist(arrays['false_positives'][:, 2], bins=radius_bins, histtype='step', label='false positives')
    ax.hist(arrays['excluded'][:, 2], bins=radius_bins, histtype='step', label='excluded >= 10 km')
    ax.hist(arrays['truth_radii'], bins=radius_bins, histtype='step', label='truth')

    ax.set_yscale('log')
    ax.set_xlabel('radius (px)')
    ax.set_title(channel_labels[channel])

axes[0].set_ylabel('count')
axes[0].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'false_positives.png'), dpi=200)
plt.show()


# ## Pixel confusion

# In[ ]:


fig, axes = plt.subplots(1, len(channels), figsize=(4.5 * len(channels), 4))

for ax, channel in zip(axes, channels):

    pixels = results[channel]['arrays']['pixels']

    confusion = np.array([[pixels['tn'], pixels['fp']], [pixels['fn'], pixels['tp']]], dtype=float)
    normalised = confusion / confusion.sum(axis=1, keepdims=True)

    ax.imshow(normalised, cmap='Blues', vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            if normalised[i, j] > 0.5:
                colour = 'white'
            else:
                colour = 'black'

            ax.text(j, i, f'{normalised[i, j]:.3f}\n{int(confusion[i, j]):,}', ha='center', va='center', color=colour)

    ax.set_xticks([0, 1], ['pred background', 'pred rim'])
    ax.set_yticks([0, 1], ['true background', 'true rim'])
    ax.set_title(channel_labels[channel])

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'confusion.png'), dpi=200)
plt.show()


# ## Loss curves

# In[ ]:


fig, ax = plt.subplots(figsize=(7, 5))

for channel in channels:

    run_name = f"{params['model']}_{channel}_{params['n_filters']}f_s{params['seed']}_{params['training_sample_percentage']}pct"
    curves = pd.read_csv(os.path.join(CHECKPOINT_DIR, f'history_{run_name}.csv'))

    colour = channel_colours[channel]
    best_epoch = int(curves['val_loss'].idxmin())

    ax.plot(curves['loss'], color=colour, label=f'{channel_labels[channel]} train')
    ax.plot(curves['val_loss'], color=colour, linestyle='--', label=f'{channel_labels[channel]} val')
    ax.axvline(best_epoch, color=colour, linestyle=':', alpha=0.5)

ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_ROOT, 'loss_curves.png'), dpi=200)
plt.show()


# ## Labelled craters
# 
# Circles are drawn in patch pixel coordinates, the same frame the model works in.
# 
# `patchTruth` turns catalogue craters into that frame: a per-tile linear fit maps Robbins
# lon/lat to tile pixels, `truth_coords_for_patch` subtracts the patch origin and applies the
# cos(lat) correction, giving (x, y, radius) in pixels. `template_match_t` returns detections
# in the same frame, and `match_coords` pairs the two. Nothing is read off the image itself.
# 
# Diameter shown is `2 * radius * 0.1` km, since 1 px = 100 m.

# In[ ]:


deep_dive = 'both'

deep_dive_name = f"{params['model']}_{deep_dive}_{params['n_filters']}f_s{params['seed']}_{params['training_sample_percentage']}pct"
deep_dive_model = keras.models.load_model(os.path.join(CHECKPOINT_DIR, f'{deep_dive_name}.keras'))

deep_dive_threshold = results[deep_dive]['headline']['best_threshold']
deep_dive_dir = os.path.join(RESULTS_ROOT, deep_dive)


# classifyPatch
# splits one patch's craters into matched, missed and extra, plus the ones
# excluded for being 10 km or more.
# parameters:
#         patch_idx: global patch index
# outputs:
#         prediction (256, 256), matched (n, 6), missed (n, 3),
#         false_positives (n, 3), excluded (n, 3)
def classifyPatch(patch_idx):

    prediction = deep_dive_model.predict(patchInput(patch_idx, deep_dive)[None, ...], verbose=0)[0, :, :, 0]

    detections = filter_edge_craters(template_match_t(prediction.copy(), target_thresh=deep_dive_threshold))
    truth = patchTruth(patch_idx)

    _, _, _, matched_pairs, false_positives, _ = match_coords(truth, detections)

    matched = np.asarray(matched_pairs).reshape(-1, 6)
    false_positives = np.asarray(false_positives).reshape(-1, 3)

    explained = matchesExcludedCrater(false_positives, patchLarge(patch_idx))
    excluded = false_positives[explained]
    false_positives = false_positives[~explained]

    claimed = {tuple(np.round(row, 4)) for row in matched[:, 3:6]}
    missed = np.array([row for row in truth if tuple(np.round(row, 4)) not in claimed]).reshape(-1, 3)

    return prediction, matched, missed, false_positives, excluded


# drawLabelled
# draws a numbered circle per crater, annotated with its diameter in km.
# parameters:
#         ax: matplotlib axis
#         craters: array (n, 3) of x, y, radius in px
#         colour: circle colour
#         tag: label prefix, for example 'TP'
#         style: line style, default '-'
def drawLabelled(ax, craters, colour, tag, style='-'):

    for n, crater in enumerate(craters, 1):
        x, y, radius = crater[0], crater[1], crater[2]

        ax.add_patch(plt.Circle((x, y), radius, fill=False, color=colour, linewidth=1.4, linestyle=style))

        ax.annotate(f'{tag}{n}  {2 * radius * 0.1:.1f}km', (x, y - radius),
                    textcoords='offset points', xytext=(0, 4), ha='center',
                    fontsize=6.5, color=colour,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground='black')])


# labelledFigure
# draws one patch with its craters labelled by outcome, and saves it.
# parameters:
#         patch_idx: global patch index
#         save_as: destination png path
def labelledFigure(patch_idx, save_as):

    prediction, matched, missed, false_positives, excluded = classifyPatch(patch_idx)

    wac_patch = patchWac(patch_idx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axes:
        ax.imshow(wac_patch, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title('WAC')

    axes[1].imshow(np.ma.masked_where(prediction < deep_dive_threshold, prediction),
                   cmap='autumn', alpha=0.7, vmin=deep_dive_threshold, vmax=1)
    axes[1].set_title(f'predicted rim (>= {deep_dive_threshold})')

    for ax in axes:
        drawLabelled(ax, matched, 'lime', 'TP')
        drawLabelled(ax, missed, 'red', 'FN', style='--')
        drawLabelled(ax, false_positives, 'deepskyblue', 'FP', style=':')

    handles = [
        Line2D([], [], color='lime', label=f'matched ({len(matched)})'),
        Line2D([], [], color='red', linestyle='--', label=f'missed ({len(missed)})'),
        Line2D([], [], color='deepskyblue', linestyle=':', label=f'extra ({len(false_positives)})'),
    ]

    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f'{deep_dive} - patch {patch_idx} - {len(matched)}/{len(matched) + len(missed)} craters recovered')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_as, dpi=200, bbox_inches='tight')
    plt.show()


show_patches = sorted(test_idx[:60], key=lambda i: len(patchTruth(int(i))), reverse=True)[:3]

for n, patch_idx in enumerate(show_patches, 1):
    labelledFigure(int(patch_idx), os.path.join(deep_dive_dir, f'labelled_{n}.png'))

