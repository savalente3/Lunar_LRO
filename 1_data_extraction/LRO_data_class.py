import os
import requests
import rasterio
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from skimage.draw import circle_perimeter


# LunarDataset
# loads the WAC tile, the DEM, the Robbins catalogue and the filtered labels,
# and holds them together so the notebooks can reach one object.
class LunarDataset:
 
    # __init__
    # loads every dataset on construction.
    def __init__(self):
        self.labels = None
        self.regionalLunarData = None
        self.DEMLunarData = None
        self.mergedData = None

        self.loadRegionalLunarImages()
        self.loadLunarLabels()
        self.loadDEMLunarData()
        self.loadFilteredLabels()
    
    # loadRegionalLunarImages
    # loads the default WAC tile into self.regionalLunarData.
    def loadRegionalLunarImages(self):
        self.regionalLunarData = getRegionalLunarData()

    # loadDEMLunarData
    # loads the SLDEM2015 memmap into self.DEMLunarData.
    def loadDEMLunarData(self):
        self.DEMLunarData = getDEMLunarData()
    
    # loadLunarLabels
    # loads the Robbins catalogue into self.labels.
    def loadLunarLabels(self):
        self.labels = getLunarRobbinsLabels()
    
    # loadFilteredLabels
    # loads the filtered crater subset into self.mergedData.
    def loadFilteredLabels(self):
        self.mergedData = getFilteredLabels()

    # rebuildMasks
    # redraws the stored masks using the catalogue already held on the object.
    # parameters:
    #         patches_dir: directory holding the patch files
    # outputs:
    #         float, the fraction of pixels that are rim
    def rebuildMasks(self, patches_dir, **kwargs):
        return rebuildMasks(patches_dir, catalogue=self.labels, **kwargs)

    # saveFiles
    # writes the WAC array and the catalogue to disk.
    # parameters:
    #         output_dir: destination directory, default 'data'
    def saveFiles(self, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "RegionalLunarData.npy"), self.regionalLunarData)
        self.labels.to_csv(os.path.join(output_dir, "LunarLabels.csv"))
 
 
# getRegionalLunarData
# downloads one WAC 100 m/px global tile the first time it is asked for and
# reads it from disk on every later call.
# parameters:
#         tile: tile name, default 'WAC_GLOBAL_E300N1350_100M'
#         data_dir: where the .IMG is cached, default '../1_data_extraction/data'
# outputs:
#         array (18194, 27291) float32, reflectance
def getRegionalLunarData(tile='WAC_GLOBAL_E300N1350_100M', data_dir='../1_data_extraction/data'):
    path = os.path.join(data_dir, f'{tile}.IMG')

    if not os.path.exists(path):
        url = f'https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/{tile}.IMG'

        os.makedirs(data_dir, exist_ok=True)

        response = requests.get(url, stream=True)

        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1 << 20):
                file.write(chunk)

    with rasterio.open(path) as src:
         data = src.read(1)

    return data
 
 
# getLunarRobbinsLabels
# loads the Robbins (2019) lunar crater catalogue from kaggle.
# parameters:
#         file_path: csv inside the kaggle dataset, default the 2018 database
# outputs:
#         dataframe, one row per crater, 21 columns
def getLunarRobbinsLabels(file_path="lunar_crater_database_robbins_2018.csv"):
    return pd.DataFrame(kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "sujaykapadnis/moon-crater-database-v1-robbins",
        file_path,
    ))

# getDEMLunarData
# downloads SLDEM2015 at 256 ppd (118 m/px) once, then memory-maps it so the
# 11 GB stays on disk and only the touched pages are read.
# parameters:
#         data_dir: where the .IMG is cached, default '../1_data_extraction/data'
# outputs:
#         memmap (30720, 92160) float32, elevation in km
def getDEMLunarData(data_dir='../1_data_extraction/data'):
    path = os.path.join(data_dir, 'SLDEM2015_256_60S_60N_000_360_FLOAT.IMG')

    if not os.path.exists(path):
        url = 'http://imbrium.mit.edu/DATA/SLDEM2015/GLOBAL/FLOAT_IMG/SLDEM2015_256_60S_60N_000_360_FLOAT.IMG'

        os.makedirs(data_dir, exist_ok=True)

        response = requests.get(url, allow_redirects=True, stream=True)

        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1 << 20):
                file.write(chunk)

    return np.memmap(path, dtype=np.float32, mode='r', shape=(30720, 92160))
 
 
# getFilteredLabels
# reads the crater subset written by the data_merge notebook.
# parameters:
#         path: csv written by data_merge, default '../2_data_preparation/filtered_labels.csv'
# outputs:
#         dataframe, or None if the file does not exist
def getFilteredLabels(path='../2_data_preparation/filtered_labels.csv'):
    if not os.path.exists(path):
        print(f'{path} not found. Run 2_data_preparation/data_merge_alltiles.ipynb first.')
        return None
    
    return pd.read_csv(path)

# getSplitIndices
# loads the train, validation and test patch indices written by pre-processing.
# parameters:
#         splits: patches directory holding the .npy index files
# outputs:
#         three int arrays: train_idx, val_idx, test_idx
def getSplitIndices(splits='../3_pre_processing/lunar_patches'):
    
    train_idx = np.load(os.path.join(splits, 'train_idx.npy'))
    val_idx   = np.load(os.path.join(splits, 'val_idx.npy'))
    test_idx  = np.load(os.path.join(splits, 'test_idx.npy'))
    
    return train_idx, val_idx, test_idx

# augment
# applies a random flip and 90 degree rotation to a patch and its mask.
# craters are rotationally symmetric so every orientation is still valid.
# parameters:
#         wac: array (256, 256), optical patch
#         dem: array (256, 256), elevation patch
#         mask: array (256, 256), ring mask
#         rng: numpy generator for reproducibility, default np.random
# outputs:
#         three arrays (256, 256), the augmented wac, dem and mask
def augment(wac, dem, mask, rng=None):
    if rng is None:
        rng = np.random

    if rng.random() > 0.5:
        wac  = np.fliplr(wac).copy()
        dem  = np.fliplr(dem).copy()
        mask = np.fliplr(mask).copy()

    if rng.random() > 0.5:
        wac  = np.flipud(wac).copy()
        dem  = np.flipud(dem).copy()
        mask = np.flipud(mask).copy()

    k = rng.integers(0, 4) if hasattr(rng, 'integers') else rng.randint(0, 4)
    if k > 0:
        wac  = np.rot90(wac, k).copy()
        dem  = np.rot90(dem, k).copy()
        mask = np.rot90(mask, k).copy()

    return wac, dem, mask


# percentileNormalise
# clips a patch to its percentile range and rescales it to [0, 1], so a few
# extreme pixels do not set the scale for the whole patch.
# parameters:
#         patch: 2D array, one WAC or DEM patch
#         low: lower percentile, default 1
#         high: upper percentile, default 99
# outputs:
#         array (256, 256) float, values in [0, 1]
def percentileNormalise(patch, low=1, high=99):
    p_low, p_high = np.percentile(patch, [low, high])
    return (np.clip(patch, p_low, p_high) - p_low) / (p_high - p_low + 1e-8)


# maskGeneration
# draws a 1 px ring for every catalogue crater whose centre falls inside the
# patch. rings rather than filled disks so overlapping craters stay separable.
# parameters:
#         patch_wac_col: patch centre column in tile pixels
#         patch_wac_row: patch centre row in tile pixels
#         wac_col: array of crater columns in tile pixels
#         wac_row: array of crater rows in tile pixels
#         diameters: array of crater diameters in km
#         cos_lat: cosine of the patch latitude, corrects the E-W stretch
# outputs:
#         array (256, 256) uint8, 1 on a rim pixel and 0 elsewhere
def maskGeneration(patch_wac_col, patch_wac_row, wac_col, wac_row, diameters, cos_lat):
    wac_col = np.asarray(wac_col)
    wac_row = np.asarray(wac_row)
    diameters = np.asarray(diameters)

    mask = np.zeros((256, 256), dtype=np.uint8)

    half_col = 128 / cos_lat

    in_patch = (
        (wac_col >= patch_wac_col - half_col) & (wac_col < patch_wac_col + half_col) &
        (wac_row >= patch_wac_row - 128) & (wac_row < patch_wac_row + 128)
    )

    for i in np.where(in_patch)[0]:

        rel_col = int(128 + (wac_col[i] - patch_wac_col) * cos_lat)
        rel_row = int(128 + (wac_row[i] - patch_wac_row))
        radius = int((diameters[i] / 2) / 0.1)

        if radius < 1:
            continue

        rr, cc = circle_perimeter(rel_row, rel_col, radius, shape=(256, 256))
        mask[rr, cc] = 1

    return mask


# fitTileMap
# fits lon/lat to tile pixel coordinates for one tile, using the craters that
# kept_labels already carries, then applies that fit to the catalogue.
# parameters:
#         kept_labels: dataframe of stored patches
#         tile_name: which tile to fit
#         catalogue: Robbins dataframe to map
#         margin: degrees of slack around the tile, default 2.0
# outputs:
#         three arrays: crater columns, rows and diameters near the tile
def fitTileMap(kept_labels, tile_name, catalogue, margin=2.0):
    rows = kept_labels[kept_labels['tile'] == tile_name].dropna(subset=['LON_CIRC_IMG', 'wac_col'])

    col_map = np.polyfit(rows['LON_CIRC_IMG'], rows['wac_col'], 1)
    row_map = np.polyfit(rows['LAT_CIRC_IMG'], rows['wac_row'], 1)

    near = catalogue[
        catalogue['LON_CIRC_IMG'].between(
            rows['LON_CIRC_IMG'].min() - margin,
            rows['LON_CIRC_IMG'].max() + margin
        ) &
        catalogue['LAT_CIRC_IMG'].between(
            rows['LAT_CIRC_IMG'].min() - margin,
            rows['LAT_CIRC_IMG'].max() + margin
        )
    ]

    return (
        np.polyval(col_map, near['LON_CIRC_IMG'].values),
        np.polyval(row_map, near['LAT_CIRC_IMG'].values),
        near['DIAM_CIRC_IMG'].values
    )


# rebuildMasks
# redraws every stored mask from the catalogue without re-extracting patches,
# and rewrites the .npz files and mask_all.npy in place.
# parameters:
#         patches_dir: directory holding the patch files
#         catalogue: Robbins dataframe, default loads it here
#         arc_min: ARC_IMG filter, default 0.5
#         max_diameter: km cap on crater size, default None
#         file_size: patches per .npz, default 1000
#         verbose: print progress, default True
# outputs:
#         float, the fraction of pixels that are rim
def rebuildMasks(patches_dir, catalogue=None, arc_min=0.5, max_diameter=None, file_size=1000, verbose=True):
    kept = pd.read_csv(os.path.join(patches_dir, 'kept_labels.csv'), low_memory=False)

    if catalogue is None:
        catalogue = getLunarRobbinsLabels()

    catalogue = catalogue[catalogue['ARC_IMG'] > arc_min]

    if max_diameter is not None:
        catalogue = catalogue[catalogue['DIAM_CIRC_IMG'] < max_diameter]

    if verbose:
        cut = 'no size cut' if max_diameter is None else f'DIAM < {max_diameter} km'
        print(f'{len(kept)} patches, {len(catalogue)} catalogue craters ({cut})', flush=True)

    tile_map = {}

    for tile in kept['tile'].dropna().unique():
        tile_map[tile] = fitTileMap(kept, tile, catalogue)

    mask_all = np.lib.format.open_memmap(os.path.join(patches_dir, 'mask_all.npy'), mode='w+', dtype=np.uint8, shape=(len(kept), 256, 256))

    n_files = (len(kept) + file_size - 1) // file_size
    positives = 0

    for f in range(n_files):

        block = kept.iloc[f * file_size:(f + 1) * file_size]
        masks = np.zeros((len(block), 256, 256), dtype=np.uint8)

        for j, (_, row) in enumerate(block.iterrows()):
            wac_col, wac_row, diameters = tile_map[row['tile']]
            cos_lat = np.cos(np.radians(row['patch_lat']))

            masks[j] = maskGeneration(row['center_col'], row['center_row'], wac_col, wac_row, diameters, cos_lat)

        positives += int(masks.sum())

        np.savez_compressed(os.path.join(patches_dir, f'X_mask_{f}'), masks)
        mask_all[f * file_size:f * file_size + len(block)] = masks

        if verbose:
            print(f'  {f + 1}/{n_files} files', flush=True)

    mask_all.flush()

    fraction = positives / (len(kept) * 256 * 256)

    if verbose:
        print(f'rim pixels {positives:,} ({fraction * 100:.2f}%)')

    return fraction


# getNormalisedBatch
# loads one .npz batch and percentile normalises the wac and dem patches.
# parameters:
#         batch_num: which .npz batch to load
#         patches_dir: directory holding the patch files
# outputs:
#         three arrays (1000, 256, 256): normalised wac, normalised dem, mask
def getNormalisedBatch(batch_num, patches_dir='../3_pre_processing/lunar_patches'):

    wac  = np.load(os.path.join(patches_dir, f'X_wac_{batch_num}.npz'))['arr_0']
    dem  = np.load(os.path.join(patches_dir, f'X_dem_{batch_num}.npz'))['arr_0']
    mask = np.load(os.path.join(patches_dir, f'X_mask_{batch_num}.npz'))['arr_0']

    # both are float32 with variable per-patch range:
    #   WAC - reflectance (I/F), tile range ~[0, 0.4], varies with illumination
    #   DEM - elevation in km

    # per-patch percentile normalisation
    norm_wac = np.zeros_like(wac, dtype=np.float32)
    norm_dem = np.zeros_like(dem, dtype=np.float32)

    for j in range(len(wac)):
        norm_wac[j] = percentileNormalise(wac[j])
        norm_dem[j] = percentileNormalise(dem[j])

    return norm_wac, norm_dem, mask


# getAugmentedBatch
# loads one normalised batch and augments every patch in it.
# parameters:
#         batch_num: which .npz batch to load
#         patches_dir: directory holding the patch files
#         rng: numpy generator for reproducibility, default np.random
# outputs:
#         three arrays (1000, 256, 256): wac, dem, mask
def getAugmentedBatch(batch_num, patches_dir='../3_pre_processing/lunar_patches', rng=None):
    wac, dem, mask = getNormalisedBatch(batch_num, patches_dir)

    for j in range(len(wac)):
        wac[j], dem[j], mask[j] = augment(wac[j], dem[j], mask[j], rng)

    return wac, dem, mask


# stepsPerEpoch
# how many full batches one pass over the indices produces.
# parameters:
#         indices: array of patch indices
#         batch_size: patches per batch, default 8
# outputs:
#         int, number of batches
def stepsPerEpoch(indices, batch_size=8):
    return len(indices) // batch_size


# patchGenerator
# yields training batches forever, shuffling the file order and the positions
# inside each file so one batch is not all the same terrain.
# parameters:
#         indices: patch indices to draw from
#         batch_size: patches per batch, default 8
#         channels: 'both' | 'wac' | 'dem', default 'both'
#         augment_data: apply flips and rotations, default True
#         patches_dir: directory holding the patch files
#         rng: numpy generator for reproducibility, default np.random
#         file_size: patches per .npz, default 1000
# outputs:
#         X (batch_size, 256, 256, channels) float32, y (batch_size, 256, 256, 1) float32
def patchGenerator(indices, batch_size=8, channels='both', augment_data=True, patches_dir='../3_pre_processing/lunar_patches', rng=None, file_size=1000):
    if channels not in ('both', 'wac', 'dem'):
        raise ValueError(f"channels must be 'both', 'wac' or 'dem', got {channels!r}")

    if rng is None:
        rng = np.random

    by_file = {}
    for idx in indices:
        by_file.setdefault(idx // file_size, []).append(idx % file_size)
    file_nums = np.array(sorted(by_file))

    buf_wac, buf_dem, buf_mask = [], [], []

    while True:
        for f in rng.permutation(file_nums):
            wac, dem, mask = getNormalisedBatch(int(f), patches_dir)

            positions = np.array(by_file[int(f)])
            for p in rng.permutation(positions):
                w, d, m = wac[p], dem[p], mask[p]

                if augment_data:
                    w, d, m = augment(w, d, m, rng)

                buf_wac.append(w)
                buf_dem.append(d)
                buf_mask.append(m)

                if len(buf_wac) == batch_size:
                    w_arr = np.asarray(buf_wac, dtype=np.float32)
                    d_arr = np.asarray(buf_dem, dtype=np.float32)
                    m_arr = np.asarray(buf_mask, dtype=np.float32)

                    if channels == 'both':
                        X = np.stack([w_arr, d_arr], axis=-1)
                    elif channels == 'wac':
                        X = w_arr[..., None]
                    else:
                        X = d_arr[..., None]

                    y = m_arr[..., None]

                    buf_wac, buf_dem, buf_mask = [], [], []
                    yield X, y