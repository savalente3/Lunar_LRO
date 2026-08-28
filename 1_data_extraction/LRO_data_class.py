import os
import io
import requests
import rasterio
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from skimage.draw import circle_perimeter


class LunarDataset:
 
    def __init__(self):
        self.labels = None
        self.regionalLunarData = None
        self.DEMLunarData = None
        self.mergedData = None

        self.loadRegionalLunarImages()
        self.loadLunarLabels()
        self.loadDEMLunarData()
        self.loadFilteredLabels()
    
    def loadRegionalLunarImages(self):
        self.regionalLunarData = getRegionalLunarData()

    def loadDEMLunarData(self):
        self.DEMLunarData = getDEMLunarData()
    
    def loadLunarLabels(self):
        self.labels = getLunarRobbinsLabels()
    
    def loadFilteredLabels(self):
        self.mergedData = getFilteredLabels()

    def rebuildMasks(self, patches_dir, **kwargs):
        return rebuildMasks(patches_dir, catalogue=self.labels, **kwargs)

    def saveFiles(self, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "RegionalLunarData.npy"), self.regionalLunarData)
        self.labels.to_csv(os.path.join(output_dir, "LunarLabels.csv"))
 
 
def getRegionalLunarData(tile='WAC_GLOBAL_E300N1350_100M'):
    url = f'https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/{tile}.IMG'
    
    response = requests.get(url)
    
    with rasterio.open(io.BytesIO(response.content)) as src:
         data = src.read(1)

    return data
 
 
def getLunarRobbinsLabels(file_path="lunar_crater_database_robbins_2018.csv"):
    return pd.DataFrame(kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "sujaykapadnis/moon-crater-database-v1-robbins",
        file_path,
    ))

def getDEMLunarData():
    # Raw binary float array — no format header, rasterio cannot detect it. 
    url = 'http://imbrium.mit.edu/DATA/SLDEM2015/GLOBAL/FLOAT_IMG/SLDEM2015_128_60S_60N_000_360_FLOAT.IMG'
    
    response = requests.get(url, allow_redirects=True)
    data = np.frombuffer(response.content, dtype=np.float32).reshape(15360, 46080)

    return data
 
 
def getFilteredLabels(path='../2_data_preparation/filtered_labels.csv'):
    if not os.path.exists(path):
        print(f'filtered_labels.csv not found. Run smallLabelCraters.to_csv() in data_merge.ipynb first.')
        return None
    
    return pd.read_csv(path)

def getSplitIndices(splits='../3_pre_processing/lunar_patches'):
    
    train_idx = np.load(os.path.join(splits, 'train_idx.npy'))
    val_idx   = np.load(os.path.join(splits, 'val_idx.npy'))
    test_idx  = np.load(os.path.join(splits, 'test_idx.npy'))
    
    return train_idx, val_idx, test_idx

def augment(wac, dem, mask, rng=None):
    # craters are rotationally symmetric, so flips/rotations are always valid.
    # pass rng=np.random.default_rng(seed) to make a run reproducible - ablation
    # runs must share the same augmentation or it becomes a confound.
    if rng is None:
        rng = np.random

    # random horizontal flip
    if rng.random() > 0.5:
        wac  = np.fliplr(wac).copy()
        dem  = np.fliplr(dem).copy()
        mask = np.fliplr(mask).copy()

    # random vertical flip
    if rng.random() > 0.5:
        wac  = np.flipud(wac).copy()
        dem  = np.flipud(dem).copy()
        mask = np.flipud(mask).copy()

    # random 90 degree rotation (k=1,2,3 -> 90,180,270; k=0 -> none)
    k = rng.integers(0, 4) if hasattr(rng, 'integers') else rng.randint(0, 4)
    if k > 0:
        wac  = np.rot90(wac, k).copy()
        dem  = np.rot90(dem, k).copy()
        mask = np.rot90(mask, k).copy()

    return wac, dem, mask


def percentileNormalise(patch, low=1, high=99):
    """Clip to percentile range, rescale to [0, 1]. Robust to outliers."""
    p_low, p_high = np.percentile(patch, [low, high])
    return (np.clip(patch, p_low, p_high) - p_low) / (p_high - p_low + 1e-8)


def maskGeneration(patch_wac_col, patch_wac_row, wac_col, wac_row, diameters, cos_lat):
    """Ring mask for one patch - every catalogue crater whose centre falls inside it.

    wac_col / wac_row / diameters run over the same crater set, in tile pixel
    coordinates. Rings (1 px rim), not filled disks - they stay distinct when
    craters overlap (Silburt et al. 2019).
    """
    wac_col = np.asarray(wac_col)
    wac_row = np.asarray(wac_row)
    diameters = np.asarray(diameters)

    mask = np.zeros((256, 256), dtype=np.uint8)

    # patch covers 128 px N-S but 128/cos(lat) px E-W in the original tile
    half_col = 128 / cos_lat

    in_patch = (
        (wac_col >= patch_wac_col - half_col) & (wac_col < patch_wac_col + half_col) &
        (wac_row >= patch_wac_row - 128) & (wac_row < patch_wac_row + 128)
    )

    for i in np.where(in_patch)[0]:

        # column offsest shrink by cos(lat) when the wide window resizes
        rel_col = int(128 + (wac_col[i] - patch_wac_col) * cos_lat)
        rel_row = int(128 + (wac_row[i] - patch_wac_row))
        radius = int((diameters[i] / 2) / 0.1)

        if radius < 1:
            continue

        rr, cc = circle_perimeter(rel_row, rel_col, radius, shape=(256, 256))
        mask[rr, cc] = 1

    return mask


def fitTileMap(kept_labels, tile_name, catalogue, margin=2.0):
    """lon/lat -> tile pixel coords, fitted from the craters kept_labels already carries.

    Each tile counts pixels from its own corner, so the map is fitted per tile rather than
    hardcoded. Returns (wac_col, wac_row, diameters) for the catalogue craters near it -
    anything further than `margin` degrees out cannot reach a patch.
    """
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


def rebuildMasks(patches_dir, catalogue=None, arc_min=0.5, max_diameter=None, file_size=1000, verbose=True):
    """Redraw every mask in a patches directory from catalogue.

    catalogue    : Robbins dataframe. None -> loaded here
    arc_min      : ARC_IMG, the same filter the patches were built with

    Rewrites X_mask_{n}.npz and mask_all.npy in place. Returns the rim-pixel fraction.
    """
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


def getAugmentedBatch(batch_num, patches_dir='../3_pre_processing/lunar_patches', rng=None):
    wac, dem, mask = getNormalisedBatch(batch_num, patches_dir)

    for j in range(len(wac)):
        wac[j], dem[j], mask[j] = augment(wac[j], dem[j], mask[j], rng)

    return wac, dem, mask


def stepsPerEpoch(indices, batch_size=8):
    return len(indices) // batch_size


def patchGenerator(indices, batch_size=8, channels='both', augment_data=True, patches_dir='../3_pre_processing/lunar_patches', rng=None, file_size=1000):
    """


    channels: 'both' -> X (B,256,256,2) [wac, dem]   <- fusion model
              'wac'  -> X (B,256,256,1)              <- ablation
              'dem'  -> X (B,256,256,1)              <- baseline
    y is always (B,256,256,1) float32.


    """
    if channels not in ('both', 'wac', 'dem'):
        raise ValueError(f"channels must be 'both', 'wac' or 'dem', got {channels!r}")

    if rng is None:
        rng = np.random

    # group global indices by file
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