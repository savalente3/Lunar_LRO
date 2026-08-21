#!/usr/bin/env python
"""
Generate filtered_labels.csv for tile E300N1350 — her data_merge.ipynb logic.
Filters the Robbins catalogue to: < 10 km diameter, ARC_IMG > 0.5,
within tile bounds (0-60N, 90-180E). Fast, no raster downloads.

Run from pre_processing/:  python make_labels.py
"""
import sys, os
sys.path.append('../data_extraction')

from LRO_data_class import getLunarRobbinsLabels

OUT = '../data_preparation/filtered_labels.csv'

print('Loading Robbins catalogue via Kaggle...', flush=True)
labels = getLunarRobbinsLabels()
print(f'  full catalogue: {labels.shape}', flush=True)

smallLabelCraters = labels[
    (labels['DIAM_CIRC_IMG'] < 10) &
    (labels['LAT_CIRC_IMG'] >= 0) &
    (labels['LAT_CIRC_IMG'] <= 60) &
    (labels['LON_CIRC_IMG'] >= 90) &
    (labels['LON_CIRC_IMG'] <= 180) &
    (labels['ARC_IMG'] > 0.5)
]

os.makedirs('../data_preparation', exist_ok=True)
smallLabelCraters.to_csv(OUT, index=False)
print(f'  wrote {len(smallLabelCraters)} craters to {OUT}', flush=True)