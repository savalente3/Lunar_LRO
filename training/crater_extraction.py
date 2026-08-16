# Crater extraction and matching for evaluation.

# Shared by every model run - the baseline and the proposed model must use the
# identical extraction, or metric differences cannot be attributed to the model.

# maxrad=50, not DeepMoon's 40: at a fixed 100 m/px, r=40 caps detection at 8 km
# and misses the top of the 1-10 km range. See global_project_notes.md 13.5


import numpy as np
import cv2
from skimage.feature import match_template


# [source]: Silburt et al. (2019) - utils/template_match_target.py
# Turns the U-Net's fuzzy probability mask into a list of (x, y, radius) craters.

def template_match_t(target, minrad=5, maxrad=50, longlat_thresh2=1.8, rad_thresh=1.0, template_thresh=0.5, target_thresh=0.1):

    # ring thickness of the stamp
    rw = 2

    # fuzzy prediction: crisp binary rings
    target[target >= target_thresh] = 1
    target[target < target_thresh] = 0

    radii = np.arange(minrad, maxrad + 1, 1, dtype=int)
    coords = []
    corr = []

    # slide a ring stamp of every size over the image and keeps good matches
    for r in radii:
        n = 2 * (r + rw + 1)
        template = np.zeros((n, n))
        cv2.circle(template, (r + rw + 1, r + rw + 1), r, 1, rw)

        result = match_template(target, template, pad_input=True)

        # score > 0.5
        index_r = np.where(result > template_thresh)
        coords_r = np.asarray(list(zip(*index_r)))
        corr_r = np.asarray(result[index_r])

        if len(coords_r) > 0:
            
            for c in coords_r:
                # (row,col) -> (x,y)
                coords.append([c[1], c[0], r])

            for l in corr_r:
                corr.append(np.abs(l))

    # one crater matches several stamps - keep the best, drop the rest
    coords, corr = np.asarray(coords), np.asarray(corr)
    i, N = 0, len(coords)
    while i < N:
        Long, Lat, Rad = coords.T
        lo, la, r = coords[i]
        minr = np.minimum(r, Rad)

        # both tests divided by radius - "close" scales with crater size
        dL = ((Long - lo)**2 + (Lat - la)**2) / minr**2
        dR = abs(Rad - r) / minr
        index = (dR < rad_thresh) & (dL < longlat_thresh2)

        if len(np.where(index == True)[0]) > 1:
            coords_i = coords[np.where(index == True)]
            corr_i = corr[np.where(index == True)]
            coords[i] = coords_i[corr_i == np.max(corr_i)][0]
            index[i] = False
            coords = coords[np.where(index == False)]
        N, i = len(coords), i + 1

    return coords


def match_coords(ground_truth, crater_detections, longlat_thresh=1.8, rad_thresh=1.0):
    gt = np.asarray(ground_truth).copy()
    n_gt, n_det = len(gt), len(crater_detections)
    n_match = 0

    for x, y, r in crater_detections:
        
        if len(gt) == 0:
            break
        
        X, Y, R = gt.T
        minr = np.minimum(r, R)

        dL = ((X - x)**2 + (Y - y)**2) / minr**2
        dR = abs(R - r) / minr

        index = (dR < rad_thresh) & (dL < longlat_thresh)
        n_match += min(1, index.sum())
        
        gt = gt[np.where(index == False)]
    
    return n_match, n_det, n_gt
