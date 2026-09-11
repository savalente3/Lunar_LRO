# evaluate_model
# evaluates one trained checkpoint on the held out test split, sweeping the
# detection threshold on validation and applying it once to test. metric
# definitions and the results layout follow evaluation.ipynb, and every model in
# the report is evaluated through this one script so the figures and tables are
# drawn from identical numbers.
# parameters:
#         argv[1]: baseline | deep_U_net | dilated_U_net
#         argv[2]: both | wac | dem
#         argv[3]: checkpoint path, optional for the runs listed in DEFAULT_RUNS
# outputs:
#         results/<model>/<channel>/{sweep.csv, per_patch.csv, headline.json,
#         arrays.npz, labelled_*.png}

import sys
sys.path.append('../1_data_extraction')

import os
import json
import numpy as np
import pandas as pd
import mlflow
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from crater_extraction import template_match_t, match_coords, filter_to_detectable, filter_edge_craters, truth_coords_for_patch, matchesExcludedCrater
from LRO_data_class import getSplitIndices, percentileNormalise, getLunarRobbinsLabels


# checkpoint name templates for the runs this project reports, so the common
# case needs no path on the command line. anything not listed here is passed as
# argv[3]. {channel} is substituted.
DEFAULT_RUNS = {
    'baseline': 'baseline_{channel}_s42_10pct',
    'deep_U_net': 'deep_U_net_{channel}_s42_10pct',
    'dilated_U_net': 'dilated_U_net_focal_tversky_{channel}_32f_s42_10pct_256ppd',
}

# the reported checkpoints were trained before the models were renamed, so they
# carry the old run names. these are tried when the new name is not on disk.
LEGACY_RUNS = {
    'baseline': 'model_deepmoon_{channel}_s42_10pct',
    'deep_U_net': 'U_Net_v1_{channel}_s42_10pct',
    'dilated_U_net': 'U-Net-v2-attention_focal_tversky_{channel}_32f_s42_10pct_256ppd',
}

MODEL = sys.argv[1]
CHANNEL = sys.argv[2]

if len(sys.argv) > 3:
    CHECKPOINT = sys.argv[3]
    RUN_NAME = os.path.splitext(os.path.basename(CHECKPOINT))[0]
else:
    RUN_NAME = DEFAULT_RUNS[MODEL].format(channel=CHANNEL)
    CHECKPOINT = os.path.join('../4_training/checkpoints', f'{RUN_NAME}.keras')
    legacy_name = LEGACY_RUNS[MODEL].format(channel=CHANNEL)
    legacy_path = os.path.join('../4_training/checkpoints', f'{legacy_name}.keras')
    if not os.path.exists(CHECKPOINT) and os.path.exists(legacy_path):
        RUN_NAME, CHECKPOINT = legacy_name, legacy_path

PATCHES_DIR = '../3_pre_processing/lunar_patches_alltiles'
LABELS_CSV = '../2_data_preparation/filtered_labels_alltiles.csv'
CHECKPOINT_DIR = '../4_training/checkpoints'

RESULTS_ROOT = os.path.join('results', MODEL)
RESULTS_DIR = os.path.join(RESULTS_ROOT, CHANNEL)

params = {
    'dataset': 'alltiles',
    'patch_source': 'memmap',
    'model': MODEL,
    'seed': 42,
    'training_sample_percentage': 10,
    'n_sweep': 200,
    'n_eval': 2000,
}

os.makedirs(RESULTS_DIR, exist_ok=True)

kept_labels = pd.read_csv(os.path.join(PATCHES_DIR, 'kept_labels.csv'), low_memory=False)
filtered_labels = pd.read_csv(LABELS_CSV)

train_idx, val_idx, test_idx = getSplitIndices(PATCHES_DIR)

mlflow.set_tracking_uri('../4_training/mlruns')
mlflow.set_experiment('lunar-crater-detection')


# per tile linear fit from lon/lat to tile pixels, applied to the filtered labels
# and to the catalogue craters of 10 km or more
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


# buildMemmaps in LRO_meemmap_class.py already applied percentileNormalise when
# writing these, so the memmap path does not normalise again. it indexes flat,
# since the three arrays hold every patch rather than 1000 per file.
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
# outputs:
#         none, the batch is held in loaded
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
#         channel: both | wac | dem
# outputs:
#         array (256, 256, 1) or (256, 256, 2) float
def patchInput(patch_idx, channel):

    if channel == 'both':
        return np.stack([patchWac(patch_idx), patchDem(patch_idx)], axis=-1)
    elif channel == 'wac':
        return patchWac(patch_idx)[..., None]
    else:
        return patchDem(patch_idx)[..., None]


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


# [source]: Mukhoti et al. (2020) - focal losses change the spread of predicted probabilities
# threshold grid shared by every model. the focal Tversky runs put their rim
# probabilities in a different range from the cross entropy runs, so the grid
# steps by 0.01 below 0.05 and runs up to 0.90, keeping every optimum inside it.
thresholds = [0.01, 0.02, 0.03, 0.04] + [round(0.05 * n, 2) for n in range(1, 19)]


# sweepThresholds
# runs the model over validation patches at each threshold and picks the one
# with the best crater F1.
# parameters:
#         model: loaded keras model
#         channel: both | wac | dem
# outputs:
#         best_threshold float, a dataframe of threshold, precision, recall,
#         f1, and whether the chosen threshold sits on the grid boundary
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

        for prediction, truth, _, large in zip(predictions, truths, masks, larges):

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

    # an optimum on either end of the grid means the search was cut short, so it
    # is flagged in the results
    at_edge = best_threshold in (thresholds[0], thresholds[-1])

    if at_edge:
        print(f'warning: best threshold {best_threshold} is at the edge of the sweep grid', flush=True)

    return best_threshold, sweep_table, at_edge


# evaluateChannel
# runs the model over the test patches at the chosen threshold and collects
# crater level, pixel level and per patch results.
# parameters:
#         model: loaded keras model
#         channel: both | wac | dem
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

    for n, patch_idx in enumerate(eval_idx, 1):

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

        if n % 250 == 0:
            print(f'  {n}/{len(eval_idx)} patches', flush=True)

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
        'model': MODEL,
        'run_name': RUN_NAME,
        'channel': channel,
        'best_threshold': float(best_threshold),
        'threshold_at_grid_edge': bool(threshold_at_grid_edge),
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
        'pixel_iou': float(safeDivide(pixel_tp, pixel_tp + pixel_fp + pixel_fn)),
    }

    return headline, arrays


# [source]: Silburt et al. (2019) - per image mean and standard deviation of precision and recall
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


# perBandStats
# precision and recall in the 1-2, 2-5 and 5-10 km diameter bands. radii are in
# pixels at 100 m/px.
# parameters:
#         arrays: the arrays dict from evaluateChannel
# outputs:
#         dict of bins, recall, precision
def perBandStats(arrays):

    bin_edges = [5, 10, 25, 50]
    bin_labels = ['1-2 km', '2-5 km', '5-10 km']

    matched = arrays['matched']
    truth_radii = arrays['truth_radii']
    fp_radii = arrays['false_positives'][:, 2]

    recall = []
    precision = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):

        truth_in_bin = ((truth_radii >= lower) & (truth_radii < upper)).sum()
        matched_in_bin = ((matched[:, 5] >= lower) & (matched[:, 5] < upper)).sum()
        recall.append(float(safeDivide(matched_in_bin, truth_in_bin)))

        det_in_bin = ((matched[:, 2] >= lower) & (matched[:, 2] < upper)).sum()
        fp_in_bin = ((fp_radii >= lower) & (fp_radii < upper)).sum()
        precision.append(float(safeDivide(det_in_bin, det_in_bin + fp_in_bin)))

    return {'bins': bin_labels, 'recall': recall, 'precision': precision}


# compile=False skips rebuilding the loss, since focal Tversky is a custom loss
# that is not registered on load and inference does not need it
print(f'loading {CHECKPOINT}', flush=True)
model = keras.models.load_model(CHECKPOINT, compile=False)

print('sweeping thresholds on validation', flush=True)
best_threshold, sweep_table, threshold_at_grid_edge = sweepThresholds(model, CHANNEL)

print(f'evaluating test at threshold {best_threshold}', flush=True)
headline, arrays = evaluateChannel(model, CHANNEL, best_threshold)

headline['per_patch'] = perPatchStats(arrays['per_patch'])
headline['per_band'] = perBandStats(arrays)

sweep_table.to_csv(os.path.join(RESULTS_DIR, 'sweep.csv'), index=False)
arrays['per_patch'].to_csv(os.path.join(RESULTS_DIR, 'per_patch.csv'), index=False)

with open(os.path.join(RESULTS_DIR, 'headline.json'), 'w') as handle:
    json.dump(headline, handle, indent=2)

# the raw match arrays are kept so compare_models.py can redraw the figures
# without re-running inference
np.savez_compressed(
    os.path.join(RESULTS_DIR, 'arrays.npz'),
    matched=arrays['matched'],
    false_positives=arrays['false_positives'],
    excluded=arrays['excluded'],
    truth_radii=arrays['truth_radii'],
    pixels=np.array([arrays['pixels'][k] for k in ['tp', 'fp', 'fn', 'tn']]),
)

print(f"threshold {best_threshold}   P {headline['precision']:.3f}   R {headline['recall']:.3f}   F1 {headline['f1']:.3f}", flush=True)

with mlflow.start_run(run_name=f'eval-{RUN_NAME}'):
    mlflow.log_params({'channel': CHANNEL, 'checkpoint': CHECKPOINT, 'target_thresh': best_threshold})
    mlflow.log_metrics({k: v for k, v in headline.items() if isinstance(v, (int, float))})


# labelled crater figures, drawn in patch pixel coordinates, the frame both the
# catalogue craters and the detections are in. diameters are 2 * radius * 0.1 km,
# since 1 px = 100 m.

# classifyPatch
# splits one patch's craters into matched, missed and extra, plus the ones
# excluded for being 10 km or more.
# parameters:
#         patch_idx: global patch index
# outputs:
#         prediction (256, 256), matched (n, 6), missed (n, 3),
#         false_positives (n, 3), excluded (n, 3)
def classifyPatch(patch_idx):

    prediction = model.predict(patchInput(patch_idx, CHANNEL)[None, ...], verbose=0)[0, :, :, 0]

    detections = filter_edge_craters(template_match_t(prediction.copy(), target_thresh=best_threshold))
    truth = patchTruth(patch_idx)

    _, _, _, matched_pairs, false_positives, _ = match_coords(truth, detections)

    matched_pairs = np.asarray(matched_pairs).reshape(-1, 6)
    false_positives = np.asarray(false_positives).reshape(-1, 3)

    explained = matchesExcludedCrater(false_positives, patchLarge(patch_idx))
    excluded = false_positives[explained]
    false_positives = false_positives[~explained]

    claimed = {tuple(np.round(row, 4)) for row in matched_pairs[:, 3:6]}
    missed = np.array([row for row in truth if tuple(np.round(row, 4)) not in claimed]).reshape(-1, 3)

    return prediction, matched_pairs, missed, false_positives, excluded


# drawLabelled
# draws a numbered circle per crater, annotated with its diameter in km.
# parameters:
#         ax: matplotlib axis
#         craters: array (n, 3) of x, y, radius in px
#         colour: circle colour
#         tag: label prefix, for example TP
#         style: line style, default solid
# outputs:
#         none, the circles are drawn on ax
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
# outputs:
#         none, the figure is saved to save_as
def labelledFigure(patch_idx, save_as):

    prediction, matched_pairs, missed, false_positives, excluded = classifyPatch(patch_idx)

    wac_patch = patchWac(patch_idx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axes:
        ax.imshow(wac_patch, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title('WAC')

    axes[1].imshow(np.ma.masked_where(prediction < best_threshold, prediction),
                   cmap='autumn', alpha=0.7, vmin=best_threshold, vmax=1)
    axes[1].set_title(f'predicted rim (>= {best_threshold})')

    for ax in axes:
        drawLabelled(ax, matched_pairs, 'lime', 'TP')
        drawLabelled(ax, missed, 'red', 'FN', style='--')
        drawLabelled(ax, false_positives, 'deepskyblue', 'FP', style=':')

    handles = [
        Line2D([], [], color='lime', label=f'matched ({len(matched_pairs)})'),
        Line2D([], [], color='red', linestyle='--', label=f'missed ({len(missed)})'),
        Line2D([], [], color='deepskyblue', linestyle=':', label=f'extra ({len(false_positives)})'),
    ]

    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f'{CHANNEL} - patch {patch_idx} - {len(matched_pairs)}/{len(matched_pairs) + len(missed)} craters recovered')

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_as, dpi=200, bbox_inches='tight')
    plt.close(fig)


# the three densest of the first 60 test patches
show_patches = sorted(test_idx[:60], key=lambda i: len(patchTruth(int(i))), reverse=True)[:3]

for n, patch_idx in enumerate(show_patches, 1):
    labelledFigure(int(patch_idx), os.path.join(RESULTS_DIR, f'labelled_{n}.png'))

print(f'results written to {RESULTS_DIR}', flush=True)