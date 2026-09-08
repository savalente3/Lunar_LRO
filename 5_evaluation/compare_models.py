#!/usr/bin/env python
# coding: utf-8

# compare_models
# redraws the shared report figures across a set of evaluated runs, reading the
# per run artefacts evaluate_model.py wrote rather than re-running inference.
# figure names, axes and styling follow evaluation.ipynb (Sofia Valente). every
# run must carry the complete artefact set, so a figure either covers every
# model in the comparison or is not drawn at all.
# parameters:
#         argv[1:]: series to include, each model_dir:channel, for example
#                   model_v2att:both U_Net_v1:both deepmoon-paper:both.
#                   with no arguments every channel of every model under
#                   results/ that carries a headline.json is included.
# outputs:
#         results/comparison/{comparison, precision_recall, diameter_bins,
#         diameter_recall, sweep, size_frequency, errors, false_positives,
#         confusion,
#         loss_curves}.png and summary.csv

import sys
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_ROOT = 'results'
OUTPUT_DIR = os.path.join(RESULTS_ROOT, 'comparison')
CHECKPOINT_DIR = '../4_training/checkpoints'

# [ref]: Silburt et al. (2019), Lunar Crater Identification via Deep Learning,
# Icarus 317, arXiv:1803.02192. the published post-cnn operating point, a
# literature reference drawn in grey. the same architecture retrained on this
# project's data is an evaluated series like any other, so the two appear side
# by side and the difference between them is attributable to the data.
DEEPMOON_PUBLISHED = {'recall': 0.57, 'precision': 0.80, 'f1': 0.666, 'recall_sd': 0.20, 'precision_sd': 0.15}
DEEPMOON_LABEL = 'DeepMoon, Silburt et al. (2019)'

CHANNEL_LABELS = {'wac': 'WAC only', 'dem': 'DEM only', 'both': 'WAC + DEM', 'wac+dem': 'WAC + DEM'}

MODEL_LABELS = {
    'model_v2att': 'U-Net v2 attention',
    'model_deepmoon': 'DeepMoon architecture',
    'deepmoon-paper': 'DeepMoon architecture',
    'U_Net_v1': 'U-Net v1',
}

PALETTE = ['tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:red', 'tab:brown', 'tab:pink']

os.makedirs(OUTPUT_DIR, exist_ok=True)


# discoverSeries
# finds every evaluated run under results/, or just the ones named on the
# command line.
# parameters:
#         argv: list of model_dir:channel strings, possibly empty
# outputs:
#         list of (model_dir, channel) pairs
def discoverSeries(argv):

    if argv:
        return [tuple(item.split(':', 1)) for item in argv]

    found = []

    for path in sorted(glob.glob(os.path.join(RESULTS_ROOT, '*', '*', 'headline.json'))):
        channel = os.path.basename(os.path.dirname(path))
        model = os.path.basename(os.path.dirname(os.path.dirname(path)))
        found.append((model, channel))

    return found


# loadSeries
# reads one run's headline, sweep table and saved match arrays.
# parameters:
#         model: results subdirectory for the model
#         channel: results subdirectory for the channel
# outputs:
#         dict of label, colour placeholder, headline, sweep, arrays, or None
#         when the run has not been evaluated
def loadSeries(model, channel):

    directory = os.path.join(RESULTS_ROOT, model, channel)

    required = ['headline.json', 'sweep.csv', 'per_patch.csv', 'arrays.npz']
    missing = [name for name in required if not os.path.exists(os.path.join(directory, name))]

    if missing:
        sys.exit(f"{directory}: missing {', '.join(missing)}. re-run evaluate_model.py for this run.")

    with open(os.path.join(directory, 'headline.json')) as handle:
        headline = json.load(handle)

    if 'per_band' not in headline:
        sys.exit(f'{directory}: headline.json has no per_band block, so it predates evaluate_model.py. re-run it.')

    label = MODEL_LABELS.get(model, model) + ', ' + CHANNEL_LABELS.get(channel, channel)

    # the resolution is part of the identity of a run, since the same
    # architecture trained on 128 and 256 ppd data is two separate results.
    if headline.get('ppd'):
        label += f", {headline['ppd']} ppd"

    series = {
        'model': model,
        'channel': channel,
        'label': label,
        'headline': headline,
        'sweep': pd.read_csv(os.path.join(directory, 'sweep.csv')),
        'arrays': np.load(os.path.join(directory, 'arrays.npz')),
    }

    return series


series_list = [loadSeries(model, channel) for model, channel in discoverSeries(sys.argv[1:])]

if not series_list:
    sys.exit('no evaluated runs found under results/')

for n, series in enumerate(series_list):
    series['colour'] = PALETTE[n % len(PALETTE)]


# saveFigure
# writes a figure to the comparison directory and closes it.
# parameters:
#         fig: matplotlib figure
#         name: file name within the comparison directory
def saveFigure(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=200, bbox_inches='tight')
    plt.close(fig)


rows = []

for series in series_list:

    headline = series['headline']
    per_band = headline['per_band']

    row = {
        'model': series['model'],
        'channel': series['channel'],
        'ppd': headline.get('ppd'),
        'best_threshold': headline['best_threshold'],
        'precision': headline['precision'],
        'recall': headline['recall'],
        'f1': headline['f1'],
        'pixel_dice': headline.get('pixel_dice'),
        'pixel_iou': headline.get('pixel_iou'),
        'tp': headline['tp'],
        'detected': headline['detected'],
        'truth': headline['truth'],
    }

    # the per band figures are the small crater result, so they belong in the
    # table the report quotes rather than only in a figure.
    for n, band in enumerate(per_band['bins']):
        row[f'recall_{band}'] = per_band['recall'][n]
        row[f'precision_{band}'] = per_band['precision'][n]

    rows.append(row)

summary = pd.DataFrame(rows)

summary.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)


# crater level metrics against the deepmoon operating point

metrics = ['recall', 'precision', 'f1']

fig, ax = plt.subplots(figsize=(2.6 + 1.6 * len(series_list), 5))

x = np.arange(len(metrics))
width = 0.8 / (len(series_list) + 1)

for n, series in enumerate(series_list):

    bars = ax.bar(x + n * width, [series['headline'][m] for m in metrics], width, color=series['colour'], label=series['label'])
    ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

bars = ax.bar(x + len(series_list) * width, [DEEPMOON_PUBLISHED[m] for m in metrics], width, color='grey', label=DEEPMOON_LABEL)
ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

ax.set_xticks(x + width * len(series_list) / 2, ['recall', 'precision', 'F1'])
ax.set_ylabel('score')
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.set_title('Crater-level metrics')

saveFigure(fig, 'comparison.png')


# precision against recall, with each run's chosen operating point marked

fig, ax = plt.subplots(figsize=(7, 6))

for series in series_list:

    sweep = series['sweep'].sort_values('recall')
    headline = series['headline']

    ax.plot(sweep['recall'], sweep['precision'], marker='o', markersize=4, color=series['colour'], linewidth=1.4, label=series['label'])
    ax.scatter([headline['recall']], [headline['precision']], s=110, color=series['colour'], edgecolor='black', zorder=5, linewidth=0.9)
    ax.annotate(f"F1 {headline['f1']:.3f}", (headline['recall'], headline['precision']), textcoords='offset points', xytext=(8, -12), fontsize=8, color=series['colour'])

ax.errorbar([DEEPMOON_PUBLISHED['recall']], [DEEPMOON_PUBLISHED['precision']], xerr=[DEEPMOON_PUBLISHED['recall_sd']], yerr=[DEEPMOON_PUBLISHED['precision_sd']],
            fmt='o', markersize=9, color='black', ecolor='grey', elinewidth=1.2, capsize=4, zorder=6, label=DEEPMOON_LABEL)

ax.set_xlabel('recall')
ax.set_ylabel('precision')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
ax.legend(loc='lower left', fontsize=8)

saveFigure(fig, 'precision_recall.png')


# precision and recall by crater diameter, with the model_v1 baseline marked

bin_labels = series_list[0]['headline']['per_band']['bins']
x = np.arange(len(bin_labels))

fig, axes = plt.subplots(1, len(series_list), figsize=(4.6 * len(series_list), 4.2), sharey=True, squeeze=False)

for ax, series in zip(axes[0], series_list):

    per_band = series['headline']['per_band']

    bars = ax.bar(x - 0.2, per_band['recall'], 0.4, label='recall', color='tab:blue')
    ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)

    bars = ax.bar(x + 0.2, per_band['precision'], 0.4, label='precision', color='tab:orange')
    ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)

    ax.set_xticks(x, bin_labels)
    ax.set_ylim(0, 1)
    ax.set_title(series['label'], fontsize=10)

axes[0][0].set_ylabel('score')
axes[0][0].legend(fontsize=8)

saveFigure(fig, 'diameter_bins.png')


# per band recall on shared axes, the comparison the small crater work is for

fig, ax = plt.subplots(figsize=(2.6 + 1.4 * len(series_list), 4.5))

width = 0.8 / len(series_list)

for n, series in enumerate(series_list):

    bars = ax.bar(x + n * width, series['headline']['per_band']['recall'], width, color=series['colour'], label=series['label'])
    ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)

ax.set_xticks(x + width * (len(series_list) - 1) / 2, bin_labels)
ax.set_ylabel('recall')
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.set_title('Crater recall by diameter band')

saveFigure(fig, 'diameter_recall.png')


# threshold sweep curves

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for series in series_list:

    sweep = series['sweep']
    colour = series['colour']

    axes[0].plot(sweep['threshold'], sweep['f1'], marker='o', color=colour, label=f"{series['label']} F1")
    axes[0].plot(sweep['threshold'], sweep['precision'], marker='^', linestyle='--', color=colour, alpha=0.6)
    axes[0].plot(sweep['threshold'], sweep['recall'], marker='v', linestyle=':', color=colour, alpha=0.6)

    axes[1].plot(sweep['recall'], sweep['precision'], marker='o', color=colour, label=series['label'])

axes[0].set_xlabel('target_thresh')
axes[0].set_ylabel('score')
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=7)

axes[1].set_xlabel('recall')
axes[1].set_ylabel('precision')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8)

saveFigure(fig, 'sweep.png')


# size-frequency distribution

diameter_bins = np.logspace(np.log10(1), np.log10(10), 15)

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(series_list[0]['arrays']['truth_radii'] * 2 * 0.1, bins=diameter_bins, histtype='step', color='black', label='Robbins (in patch)')

for series in series_list:

    arrays = series['arrays']
    detected = np.concatenate([arrays['matched'][:, 2], arrays['false_positives'][:, 2]]) * 2 * 0.1

    ax.hist(detected, bins=diameter_bins, histtype='step', color=series['colour'], label=series['label'])

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('diameter (km)')
ax.set_ylabel('count')
ax.legend(fontsize=8)

saveFigure(fig, 'size_frequency.png')


# position and radius error

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for series in series_list:

    matched = series['arrays']['matched']
    mean_radius = (matched[:, 2] + matched[:, 5]) / 2

    errors = [
        abs(matched[:, 0] - matched[:, 3]) / mean_radius,
        abs(matched[:, 1] - matched[:, 4]) / mean_radius,
        abs(matched[:, 2] - matched[:, 5]) / mean_radius,
    ]

    for ax, values, name in zip(axes, errors, ['x', 'y', 'radius']):
        ax.hist(values, bins=40, histtype='step', color=series['colour'], label=f"{series['label']} (med {np.median(values):.3f})")
        ax.axvline(np.median(values), color=series['colour'], linestyle='--', alpha=0.6)
        ax.set_xlabel(f'{name} fractional error')

for ax in axes:
    ax.legend(fontsize=7)

saveFigure(fig, 'errors.png')


# false positives by radius

radius_bins = np.arange(5, 52, 1)

fig, axes = plt.subplots(1, len(series_list), figsize=(4.6 * len(series_list), 4), sharey=True, squeeze=False)

for ax, series in zip(axes[0], series_list):

    arrays = series['arrays']

    ax.hist(arrays['false_positives'][:, 2], bins=radius_bins, histtype='step', label='false positives')
    ax.hist(arrays['excluded'][:, 2], bins=radius_bins, histtype='step', label='excluded >= 10 km')
    ax.hist(arrays['truth_radii'], bins=radius_bins, histtype='step', label='truth')

    ax.set_yscale('log')
    ax.set_xlabel('radius (px)')
    ax.set_title(series['label'], fontsize=10)

axes[0][0].set_ylabel('count')
axes[0][0].legend(fontsize=8)

saveFigure(fig, 'false_positives.png')


# pixel confusion, row normalised

fig, axes = plt.subplots(1, len(series_list), figsize=(4.2 * len(series_list), 4), squeeze=False)

for ax, series in zip(axes[0], series_list):

    tp, fp, fn, tn = [int(v) for v in series['arrays']['pixels']]

    confusion = np.array([[tn, fp], [fn, tp]], dtype=float)
    normalised = confusion / confusion.sum(axis=1, keepdims=True)

    ax.imshow(normalised, cmap='Blues', vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            colour = 'white' if normalised[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{normalised[i, j]:.3f}\n{int(confusion[i, j]):,}', ha='center', va='center', color=colour, fontsize=8)

    ax.set_xticks([0, 1], ['pred background', 'pred rim'], fontsize=8)
    ax.set_yticks([0, 1], ['true background', 'true rim'], fontsize=8)
    ax.set_title(series['label'], fontsize=10)

saveFigure(fig, 'confusion.png')


# training curves, for the runs whose history file is present

curve_series = []

for series in series_list:

    path = os.path.join(CHECKPOINT_DIR, f"history_{series['headline']['run_name']}.csv")

    if os.path.exists(path):
        curve_series.append((series, pd.read_csv(path)))

# the curves figure is drawn only when every run's history is present, so it
# never shows a subset of the comparison.
if len(curve_series) == len(series_list):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for series, curves in curve_series:

        colour = series['colour']

        axes[0].plot(curves['loss'], color=colour, label=f"{series['label']} train")
        axes[0].plot(curves['val_loss'], color=colour, linestyle='--', label=f"{series['label']} val")

        # selection monitored val_dice_coef rather than val_loss, so the chosen
        # epoch is read from that column where the run recorded it.
        if 'val_dice_coef' in curves.columns:
            axes[1].plot(curves['val_dice_coef'], color=colour, label=f"{series['label']} val_dice")
            axes[1].axvline(int(curves['val_dice_coef'].idxmax()), color=colour, linestyle=':', alpha=0.6)

    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('val_dice_coef')
    axes[1].legend(fontsize=7)

    saveFigure(fig, 'loss_curves.png')