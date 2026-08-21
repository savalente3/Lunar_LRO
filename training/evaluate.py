"""
Evaluation + figures for a trained model.

Computes pixel-level segmentation metrics (precision, recall, F1, IoU),
a precision-recall curve, a confusion matrix, and saves overlay prediction
figures. Imported by run_all.py.
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collect_predictions(model, val_gen, n_batches=30):
    """Gather predicted probabilities and ground truth over val batches."""
    P, Y = [], []
    n = min(n_batches, len(val_gen))
    for i in range(n):
        x, y = val_gen[i]
        P.append(model.predict(x, verbose=0)[..., 0])
        Y.append(y[..., 0])
    return np.concatenate(P), np.concatenate(Y)


def pixel_metrics(probs, gt, threshold=0.5):
    p = probs > threshold
    y = gt > 0.5
    tp = float((p & y).sum()); fp = float((p & ~y).sum())
    fn = float((~p & y).sum()); tn = float((~p & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    u = tp + fp + fn
    return {
        'pixel_precision': round(prec, 4),
        'pixel_recall': round(rec, 4),
        'pixel_f1': round(f1, 4),
        'iou': round(tp / u, 4) if u else 0.0,
        'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'threshold': threshold,
    }


def save_pr_curve(probs, gt, mode, out='pr_curve'):
    ts = np.linspace(0.05, 0.95, 19)
    precs, recs = [], []
    y = gt > 0.5
    for t in ts:
        p = probs > t
        tp = float((p & y).sum()); fp = float((p & ~y).sum()); fn = float((~p & y).sum())
        precs.append(tp / (tp + fp) if tp + fp else 0.0)
        recs.append(tp / (tp + fn) if tp + fn else 0.0)
    plt.figure(figsize=(6, 5))
    plt.plot(recs, precs, 'o-')
    for t, r, pr in zip(ts, recs, precs):
        if abs((t * 20) % 2) < 0.5:
            plt.annotate(f'{t:.2f}', (r, pr), fontsize=7)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall — {mode}')
    plt.grid(alpha=0.3); plt.tight_layout()
    fn = f'{out}_{mode}.png'
    plt.savefig(fn, dpi=130); plt.close()
    return fn


def save_confusion(metrics, mode, out='confusion'):
    c = metrics['confusion']
    mat = np.array([[c['tn'], c['fp']], [c['fn'], c['tp']]])
    # normalise by row for readability
    norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for (r, cc), v in np.ndenumerate(mat):
        plt.text(cc, r, f'{int(v):,}\n({norm[r,cc]*100:.1f}%)',
                 ha='center', va='center', fontsize=9)
    plt.xticks([0, 1], ['pred bg', 'pred rim'])
    plt.yticks([0, 1], ['true bg', 'true rim'])
    plt.title(f'Confusion (pixels) — {mode}')
    plt.tight_layout()
    fn = f'{out}_{mode}.png'
    plt.savefig(fn, dpi=130); plt.close()
    return fn


def save_overlays(model, val_gen, mode, n=4, threshold=0.5, out='overlay'):
    # find crater-bearing patches
    X, Y = val_gen[0]
    for i in range(1, min(6, len(val_gen))):
        x, y = val_gen[i]
        X = np.concatenate([X, x]); Y = np.concatenate([Y, y])
    counts = Y.reshape(len(Y), -1).sum(1)
    pick = np.argsort(counts)[-n:]
    preds = model.predict(X[pick], verbose=0)
    dem_ch = 0 if X.shape[-1] == 1 else 1

    fig, axes = plt.subplots(len(pick), 4, figsize=(15, 3.6 * len(pick)))
    if len(pick) == 1:
        axes = axes[None, :]
    for i, k in enumerate(pick):
        dem = X[k, ..., dem_ch]
        gt = Y[k, ..., 0] > 0.5
        pr = preds[i, ..., 0] > threshold
        axes[i, 0].imshow(dem, cmap='gray'); axes[i, 0].set_title('DEM input')
        axes[i, 1].imshow(gt, cmap='gray'); axes[i, 1].set_title('Ground truth')
        axes[i, 2].imshow(pr, cmap='gray'); axes[i, 2].set_title(f'Prediction (t={threshold})')
        axes[i, 3].imshow(dem, cmap='gray')
        a = np.zeros((*gt.shape, 4)); a[gt] = [0, 1, 1, 1]
        b = np.zeros((*pr.shape, 4)); b[pr] = [1, 0, 0, 1]
        axes[i, 3].imshow(a); axes[i, 3].imshow(b, alpha=0.6)
        axes[i, 3].set_title('Overlay (cyan=truth, red=pred)')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    fn = f'{out}_{mode}.png'
    plt.savefig(fn, dpi=130, bbox_inches='tight'); plt.close()
    return fn