"""
Post-processing — turn predicted rim masks into crater-level detections.

A segmentation mask is not a crater list. This extracts circles from the
predicted mask (Hough transform on the thresholded rings), matches them to the
ground-truth mask's craters with a position + radius tolerance, and reports
crater-level precision / recall / F1 — the scientifically meaningful metric,
and far more robust to a few pixels of registration offset than pixel F1.

Imported by run_all.py; not usually run alone.
"""

import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.measure import label, regionprops


def circles_from_mask(mask, min_radius=4, max_radius=60, threshold=0.5):
    """Detect circles in a binary/probability mask via Hough transform.

    Returns list of (row, col, radius).
    """
    binary = mask > threshold
    if binary.sum() == 0:
        return []
    edges = canny(binary.astype(float), sigma=1.0)
    radii = np.arange(min_radius, max_radius, 2)
    if len(radii) == 0:
        return []
    hspace = hough_circle(edges, radii)
    accums, cx, cy, rad = hough_circle_peaks(
        hspace, radii, total_num_peaks=40, min_xdistance=6, min_ydistance=6,
        threshold=0.3 * hspace.max() if hspace.max() > 0 else None)
    return list(zip(cy, cx, rad))  # (row, col, radius)


def circles_from_labels(mask, min_area=6):
    """Ground-truth craters from the ring mask via connected components.

    Ring masks give annuli; fit a circle to each component's bounding box.
    Returns list of (row, col, radius).
    """
    lbl = label(mask > 0.5)
    out = []
    for r in regionprops(lbl):
        if r.area < min_area:
            continue
        minr, minc, maxr, maxc = r.bbox
        cr = (minr + maxr) / 2
        cc = (minc + maxc) / 2
        rad = max(maxr - minr, maxc - minc) / 2
        out.append((cr, cc, rad))
    return out


def match(pred_circles, gt_circles, pos_tol=8, rad_tol=0.5):
    """Greedy match predicted to GT craters.

    A match requires centre distance <= pos_tol px AND radius within rad_tol
    fraction. Returns (tp, fp, fn).
    """
    matched_gt = set()
    tp = 0
    for pr, pc, prad in pred_circles:
        best = None
        best_d = 1e9
        for k, (gr, gc, grad) in enumerate(gt_circles):
            if k in matched_gt:
                continue
            d = np.hypot(pr - gr, pc - gc)
            if d <= pos_tol and abs(prad - grad) <= rad_tol * max(grad, 1):
                if d < best_d:
                    best_d, best = d, k
        if best is not None:
            matched_gt.add(best)
            tp += 1
    fp = len(pred_circles) - tp
    fn = len(gt_circles) - len(matched_gt)
    return tp, fp, fn


def crater_level_metrics(model, val_gen, n_batches=30, threshold=0.5):
    """Run detection + matching over a sample of validation batches."""
    TP = FP = FN = 0
    n = min(n_batches, len(val_gen))
    for i in range(n):
        X, Y = val_gen[i]
        preds = model.predict(X, verbose=0)
        for j in range(len(X)):
            pred_c = circles_from_mask(preds[j, ..., 0], threshold=threshold)
            gt_c = circles_from_labels(Y[j, ..., 0])
            tp, fp, fn = match(pred_c, gt_c)
            TP += tp; FP += fp; FN += fn

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        'crater_precision': round(prec, 4),
        'crater_recall': round(rec, 4),
        'crater_f1': round(f1, 4),
        'tp': TP, 'fp': FP, 'fn': FN,
        'n_patches_scored': n * val_gen.batch_size,
    }