"""
Master evaluation — everything for one trained model in one call.

Loads the saved model, computes pixel-level segmentation metrics, saves a
precision-recall curve, confusion matrix, and overlay figures, then runs
crater-level post-processing (circle detection + matching to ground truth).

Usage:
    python run_all.py --mode dem     # evaluates model_dem.keras
    python run_all.py --mode dual    # evaluates model_dual.keras
    python run_all.py --mode dual --threshold 0.5 --nbatches 30

Out (per mode):
    metrics_<mode>.json      pixel + crater-level metrics
    overlay_<mode>.png       DEM / truth / prediction / overlay panels
    pr_curve_<mode>.png      precision-recall curve
    confusion_<mode>.png     pixel confusion matrix

Compare the two metrics_<mode>.json files to read DEM-only vs WAC+DEM.
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

import sys
import json
import argparse

import numpy as np
import keras

from gpu_data import MemmapSequence, getSplitIndices
import evaluate as ev
import postprocess as pp


PATCHES_DIR = '../pre_processing/patches'
BATCH_SIZE = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dem', 'dual'], required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--nbatches', type=int, default=30,
                    help='validation batches to score (30*16 ~ 480 patches)')
    ap.add_argument('--patches_dir', default=PATCHES_DIR)
    args = ap.parse_args()

    model_path = f'model_{args.mode}.keras'
    if not os.path.exists(model_path):
        sys.exit(f'{model_path} not found — train it first with '
                 f'`python train_gpu.py {args.mode}`')

    print(f'Loading {model_path}...', flush=True)
    model = keras.models.load_model(model_path, compile=False)

    _, val_idx, _ = getSplitIndices(args.patches_dir)
    val_gen = MemmapSequence(val_idx, mode=args.mode, batch_size=BATCH_SIZE,
                            shuffle=False, patches_dir=args.patches_dir)

    # ---- pixel-level metrics ----
    print('Pixel metrics...', flush=True)
    probs, gt = ev.collect_predictions(model, val_gen, n_batches=args.nbatches)
    print(f'  prob range {probs.min():.3f}-{probs.max():.3f}, '
          f'mean {probs.mean():.4f} (crater frac {gt.mean():.4f})', flush=True)
    pix = ev.pixel_metrics(probs, gt, threshold=args.threshold)
    print(f'  pixel: {pix}', flush=True)

    # ---- figures ----
    print('Saving figures...', flush=True)
    f_overlay = ev.save_overlays(model, val_gen, args.mode, threshold=args.threshold)
    f_pr = ev.save_pr_curve(probs, gt, args.mode)
    f_cm = ev.save_confusion(pix, args.mode)

    # ---- crater-level post-processing ----
    print('Crater-level post-processing (Hough + matching)...', flush=True)
    crater = pp.crater_level_metrics(model, val_gen,
                                     n_batches=min(args.nbatches, 15),
                                     threshold=args.threshold)
    print(f'  crater: {crater}', flush=True)

    # ---- write everything ----
    out = {
        'mode': args.mode,
        'threshold': args.threshold,
        'pixel': pix,
        'crater_level': crater,
        'figures': {'overlay': f_overlay, 'pr_curve': f_pr, 'confusion': f_cm},
    }
    with open(f'metrics_{args.mode}.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f'\nDone. Wrote metrics_{args.mode}.json + 3 figures.', flush=True)
    print('SUMMARY', flush=True)
    print(f"  pixel   F1={pix['pixel_f1']}  IoU={pix['iou']}  "
          f"P={pix['pixel_precision']}  R={pix['pixel_recall']}", flush=True)
    print(f"  crater  F1={crater['crater_f1']}  "
          f"P={crater['crater_precision']}  R={crater['crater_recall']}", flush=True)


if __name__ == '__main__':
    main()