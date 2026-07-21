import os
import io
import requests
import rasterio
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np


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

    def getNormalisedBatch(self, batch_num, **kwargs):
        return getNormalisedBatch(batch_num, **kwargs)

    def getAugmentedBatch(self, batch_num, **kwargs):
        return getAugmentedBatch(batch_num, **kwargs)

    def patchGenerator(self, indices, **kwargs):
        return patchGenerator(indices, **kwargs)

    @staticmethod
    def augment(wac, dem, mask, rng=None):
        return augment(wac, dem, mask, rng)

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
 
 
def getFilteredLabels(path='../data_preparation/filtered_labels.csv'):
    if not os.path.exists(path):
        print(f'filtered_labels.csv not found. Run smallLabelCraters.to_csv() in data_merge.ipynb first.')
        return None
    
    return pd.read_csv(path)

def getSplitIndices(splits='../pre_processing/patches'):
    
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


def getNormalisedBatch(batch_num, patches_dir='../pre_processing/patches'):
    if not os.path.exists(os.path.join(patches_dir, f'X_wac_{batch_num}.npz')):
        raise FileNotFoundError(
            f'batch {batch_num} not found in {patches_dir} '
            f'(valid range 0-211; paths are relative to the notebook directory)')

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


def getAugmentedBatch(batch_num, patches_dir='../pre_processing/patches', rng=None):
    wac, dem, mask = getNormalisedBatch(batch_num, patches_dir)

    for j in range(len(wac)):
        wac[j], dem[j], mask[j] = augment(wac[j], dem[j], mask[j], rng)

    return wac, dem, mask


def stepsPerEpoch(indices, batch_size=8):
    """Number of training batches in one pass over `indices`."""
    return len(indices) // batch_size


def patchGenerator(indices, batch_size=8, channels='both', augment_data=True,
                   patches_dir='../pre_processing/patches', rng=None,
                   file_size=1000):
    """Yield (X, y) training batches from the .npz patch files, forever.

    Patches are stored 1000 per file, so global patch index i lives at
    position i % file_size in file i // file_size. This walks the files in a
    shuffled order, shuffles the wanted positions inside each one, and emits
    fixed-size batches - so consecutive batches are not all the same terrain.

    channels: 'both' -> X (B,256,256,2) [wac, dem]   <- fusion model
              'wac'  -> X (B,256,256,1)              <- ablation
              'dem'  -> X (B,256,256,1)              <- baseline
    y is always (B,256,256,1) float32.

    Infinite by design: Keras fit() pulls batches until steps_per_epoch is hit,
    so pass steps_per_epoch=stepsPerEpoch(indices, batch_size).
    Set augment_data=False for validation and test.
    """
    if channels not in ('both', 'wac', 'dem'):
        raise ValueError(f"channels must be 'both', 'wac' or 'dem', got {channels!r}")

    if rng is None:
        rng = np.random

    # group global indices by the file they live in
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