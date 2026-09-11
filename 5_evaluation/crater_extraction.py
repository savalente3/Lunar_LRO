# crater_extraction
# turns a predicted rim map into a crater list and matches it against the
# catalogue. shared by every run, so a difference here would look like a model
# difference. the largest template radius is 50 px rather than DeepMoon's 40,
# since at 100 m/px a 40 px radius stops at 8 km.
# parameters:
#         none, each function takes its own
# outputs:
#         template_match_t, filter_to_detectable, match_coords,
#         truth_coords_for_patch, filter_edge_craters, matchesExcludedCrater


import numpy as np
import cv2
from skimage.feature import match_template


# [source]: Silburt et al. (2019) - utils/template_match_target.py, https://github.com/silburt/DeepMoon
# template_match_t
# turns the model's probability map into a crater list by sliding a ring
# template of every radius over it and keeping the good matches, then dropping
# duplicates that describe the same crater.
# parameters:
#         target: array (256, 256), predicted rim probabilities
#         minrad: smallest radius searched in px, default 5
#         maxrad: largest radius searched in px, default 50
#         longlat_thresh2: squared centre distance tolerance, default 1.8
#         rad_thresh: radius tolerance, default 1.0
#         template_thresh: correlation needed to accept a match, default 0.5
#         target_thresh: probability above which a pixel counts as rim, default 0.1
# outputs:
#         array (n, 3), each row x, y, radius in px
def template_match_t(target, minrad=5, maxrad=50, longlat_thresh2=1.8, rad_thresh=1.0, template_thresh=0.5, target_thresh=0.1):

    # ring thickness of the stamp
    rw = 2

    # binarise the probability map at target_thresh
    target[target >= target_thresh] = 1
    target[target < target_thresh] = 0

    radii = np.arange(minrad, maxrad + 1, 1, dtype=int)
    coords = []
    corr = []

    # slide a ring stamp of every radius over the map and keep the good matches
    for r in radii:
        n = 2 * (r + rw + 1)
        template = np.zeros((n, n))
        cv2.circle(template, (r + rw + 1, r + rw + 1), r, 1, rw)

        result = match_template(target, template, pad_input=True)

        # positions whose correlation beats template_thresh
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
            best = np.max(corr_i)
            coords[i] = coords_i[corr_i == best][0]
            corr[i] = best
            index[i] = False
            coords = coords[np.where(index == False)]
            # drop from corr as well as coords. DeepMoon drops from coords only,
            # which leaves corr pointing at the wrong craters
            corr = corr[np.where(index == False)]
        N, i = len(coords), i + 1

    return coords


# filter_to_detectable
# drops craters outside the radius range template matching can find, so they
# are not counted as misses the model never had a chance at. same idea as
# DeepMoon's rmv_oor_csvs flag.
# parameters:
#         coords: array (n, 3) of x, y, radius
#         minrad: smallest detectable radius in px, default 5
#         maxrad: largest detectable radius in px, default 50
# outputs:
#         array (m, 3), the craters inside the range
def filter_to_detectable(coords, minrad=5, maxrad=50):

    coords = np.asarray(coords)

    if len(coords) == 0:
        return coords

    in_range = (coords[:, 2] >= minrad) & (coords[:, 2] <= maxrad)

    return coords[np.where(in_range == True)]


# [source]: Silburt et al. (2019) - utils/template_match_target.py, https://github.com/silburt/DeepMoon
# match_coords
# pairs detections with catalogue craters. both tolerances are divided by the
# crater radius, so 'close' scales with crater size. each detection takes only
# the closest catalogue crater within tolerance, so the others stay matchable.
# parameters:
#         ground_truth: array (n, 3) of catalogue craters
#         crater_detections: array (m, 3) of detections
#         longlat_thresh: squared centre distance tolerance, default 1.8
#         rad_thresh: radius tolerance, default 1.0
# outputs:
#         match_count, detection_count, truth_count,
#         matched_pairs (k, 6) as detection x,y,r then truth x,y,r,
#         false_positives (j, 3), and multi_match_count, how often one detection
#         fell within tolerance of several catalogue craters (DeepMoon's frac_dupes)
def match_coords(ground_truth, crater_detections, longlat_thresh=1.8, rad_thresh=1.0):

    remaining_truth = np.asarray(ground_truth).copy()

    truth_count = len(remaining_truth)
    detection_count = len(crater_detections)
    match_count = 0

    matched_pairs = []
    false_positives = []
    multi_match_count = 0

    for det_x, det_y, det_radius in crater_detections:

        if len(remaining_truth) == 0:
            false_positives.append([det_x, det_y, det_radius])
            continue

        truth_x, truth_y, truth_radius = remaining_truth.T

        # clip at 1 px - a zero radius would divide by zero below
        smaller_radius = np.maximum(np.minimum(det_radius, truth_radius), 1)

        # both tests divided by radius - "close" scales with crater size
        dist_ratio = ((truth_x - det_x)**2 + (truth_y - det_y)**2) / smaller_radius**2
        radius_ratio = abs(truth_radius - det_radius) / smaller_radius

        is_match = (radius_ratio < rad_thresh) & (dist_ratio < longlat_thresh)

        if is_match.sum() > 0:
            # closest match, not the first in array order
            candidates = np.where(is_match)[0]
            best = candidates[np.argmin(dist_ratio[candidates])]

            hit = remaining_truth[best]
            matched_pairs.append([det_x, det_y, det_radius, hit[0], hit[1], hit[2]])

            # one detection fell within tolerance of several truth craters
            if is_match.sum() > 1:
                multi_match_count += 1

            remaining_truth = np.delete(remaining_truth, best, axis=0)
        else:
            false_positives.append([det_x, det_y, det_radius])

        match_count += min(1, is_match.sum())

    return (match_count, detection_count, truth_count, np.asarray(matched_pairs), np.asarray(false_positives), multi_match_count)


# truth_coords_for_patch
# converts catalogue craters from tile pixels into one patch's pixel frame,
# applying the cos(lat) correction for the E-W stretch.
# parameters:
#         center_col: patch centre column in tile pixels
#         center_row: patch centre row in tile pixels
#         patch_lat: patch latitude in degrees
#         wac_col: array of crater columns in tile pixels
#         wac_row: array of crater rows in tile pixels
#         diameters: array of crater diameters in km
#         margin: px of slack outside the patch, default 0
# outputs:
#         array (n, 3), each row x, y, radius in patch px
def truth_coords_for_patch(center_col, center_row, patch_lat, wac_col, wac_row, diameters, margin=0):

    cos_lat = np.cos(np.radians(patch_lat))

    # patch covers 128 px N-S but 128/cos(lat) px E-W in the original tile
    # margin catches craters centred outside the patch whose rim still crosses it
    half_col = 128 / cos_lat + margin

    in_patch = (
        (wac_col >= center_col - half_col) & (wac_col < center_col + half_col) &
        (wac_row >= center_row - 128 - margin) & (wac_row < center_row + 128 + margin)
    )

    if in_patch.sum() == 0:
        return np.empty((0, 3))

    # column offsets shrink by cos(lat) when the wide window is resized back to 256
    rel_col = 128 + (wac_col[in_patch] - center_col) * cos_lat
    rel_row = 128 + (wac_row[in_patch] - center_row)
    radius = (diameters[in_patch] / 2) / 0.1

    return np.column_stack([rel_col, rel_row, radius])


# filter_edge_craters
# drops craters sitting too close to the patch edge to be measured properly.
# parameters:
#         coords: array (n, 3) of x, y, radius
#         dim: patch size in px, default 256
#         cutrad: fraction of the radius allowed outside the patch, default 0.8
# outputs:
#         array (m, 3), the craters far enough inside
def filter_edge_craters(coords, dim=256, cutrad=0.8):

    coords = np.asarray(coords)

    if len(coords) == 0:
        return coords

    x, y, radius = coords.T

    inside = (
        (x + cutrad * radius <= dim) & (x - cutrad * radius > 0) &
        (y + cutrad * radius <= dim) & (y - cutrad * radius > 0)
    )

    return coords[np.where(inside == True)]


# matchesExcludedCrater
# flags detections that sit on a crater of 10 km or more. the crater is real but
# outside the < 10 km label set, so the detection is excluded rather than
# counted as a false positive.
# parameters:
#         detections: array (n, 3) of x, y, radius
#         large_craters: array (m, 3) of craters >= 10 km in patch px
# outputs:
#         bool array (n,), True where a detection is explained by a large crater
def matchesExcludedCrater(detections, large_craters):

    detections = np.asarray(detections)

    if len(detections) == 0 or len(large_craters) == 0:
        return np.zeros(len(detections), dtype=bool)

    det_x, det_y, det_radius = detections.T
    large_x, large_y, large_radius = np.asarray(large_craters).T

    distance = np.sqrt((det_x[:, None] - large_x[None, :])**2 +
                       (det_y[:, None] - large_y[None, :])**2)

    concentric = distance <= det_radius[:, None]
    on_rim = np.abs(distance - large_radius[None, :]) <= det_radius[:, None]

    return (concentric | on_rim).any(axis=1)
