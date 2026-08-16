# Crater extraction and matching
# shared by every run - differences here would look like model differences
# maxrad=50 not DeepMoon's 40: at 100 m/px, r=40 caps at 8 km. notes 13.5


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


# Drops craters outside the detectable radius range.
# masks hold craters below minrad - no template can match them
# counting those as misses penalises the model for what it can't find
# same idea as DeepMoon's rmv_oor_csvs flag

def filter_to_detectable(coords, minrad=5, maxrad=50):

    coords = np.asarray(coords)

    if len(coords) == 0:
        return coords

    in_range = (coords[:, 2] >= minrad) & (coords[:, 2] <= maxrad)

    return coords[np.where(in_range == True)]


# Matches detections against ground truth.
# returns full pairs (det x,y,r then truth x,y,r) -> diameter bins, error plots
# plus detections that matched nothing -> false positive inspection

def match_coords(ground_truth, crater_detections, longlat_thresh=1.8, rad_thresh=1.0):

    remaining_truth = np.asarray(ground_truth).copy()

    truth_count = len(remaining_truth)
    detection_count = len(crater_detections)
    match_count = 0

    matched_pairs = []
    false_positives = []

    for det_x, det_y, det_radius in crater_detections:

        if len(remaining_truth) == 0:
            false_positives.append([det_x, det_y, det_radius])
            continue

        truth_x, truth_y, truth_radius = remaining_truth.T
        smaller_radius = np.minimum(det_radius, truth_radius)

        # both tests divided by radius - "close" scales with crater size
        dist_ratio = ((truth_x - det_x)**2 + (truth_y - det_y)**2) / smaller_radius**2
        radius_ratio = abs(truth_radius - det_radius) / smaller_radius

        is_match = (radius_ratio < rad_thresh) & (dist_ratio < longlat_thresh)

        if is_match.sum() > 0:
            # first match in array order, as DeepMoon does - not the closest
            hit = remaining_truth[np.where(is_match == True)][0]
            matched_pairs.append([det_x, det_y, det_radius, hit[0], hit[1], hit[2]])
        else:
            false_positives.append([det_x, det_y, det_radius])

        match_count += min(1, is_match.sum())

        # a matched truth crater cannot be claimed twice
        remaining_truth = remaining_truth[np.where(is_match == False)]

    return match_count, detection_count, truth_count, np.asarray(matched_pairs), np.asarray(false_positives)
