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
        self.globalLunarData = None
        self.regionalLunarData = None
        self.DEMLunarData = None
        self.mergedData = None
 
        self.loadLunarImages()
        self.loadRegionalLunarImages()
        self.loadLunarLabels()
        self.loadDEMLunarData()
        self.loadFilteredLabels()
    
    # might not be necessary - no analysis for now
    def loadLunarImages(self):
        self.globalLunarData = getGlobalLunarData()
    
    def loadRegionalLunarImages(self):
        self.regionalLunarData = getRegionalLunarData()

    def loadDEMLunarData(self):
        self.DEMLunarData = getDEMLunarData()
    
    def loadLunarLabels(self):
        self.labels = getLunarRobbinsLabels()
    
    def loadFilteredLabels(self):
        self.mergedData = getFilteredLabels()

    def getNormalisedBatch(self, batch_num):
        return getNormalisedBatch(batch_num)

    def getAugmentedBatch(self, batch_num):
        return getAugmentedBatch(batch_num)

    @staticmethod
    def augment(wac, dem, mask):
        # random horizontal flip
        if np.random.rand() > 0.5:
            wac  = np.fliplr(wac)
            dem  = np.fliplr(dem)
            mask = np.fliplr(mask)

        # random vertical flip
        if np.random.rand() > 0.5:
            wac  = np.flipud(wac)
            dem  = np.flipud(dem)
            mask = np.flipud(mask)

        # random 90° rotation (k=1,2,3 → 90°,180°,270°; k=0 → no rotation)
        k = np.random.randint(0, 4)
        if k > 0:
            wac  = np.rot90(wac, k)
            dem  = np.rot90(dem, k)
            mask = np.rot90(mask, k)

        return wac, dem, mask

    def saveFiles(self, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        
        self.globalLunarData.to_csv(os.path.join(output_dir, "WholeLunarData.csv"))
        self.regionalLunarData.to_csv(os.path.join(output_dir, "RegionalLunarData.csv"))
        self.labels.to_csv(os.path.join(output_dir, "LunarLabels.csv"))
 
 
def getGlobalLunarData():
    url = "https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/WAC_GLOBAL_O000N0000_004P.IMG"
    response = requests.get(url)
    with rasterio.open(io.BytesIO(response.content)) as src:
        data = src.read(1)
    return pd.DataFrame(data)
 
 
def getRegionalLunarData(tile='WAC_GLOBAL_E300N1350_100M'):
    url = f'https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/{tile}.IMG'
    
    response = requests.get(url)
    
    with rasterio.open(io.BytesIO(response.content)) as src:
         data = src.read(1)

    return pd.DataFrame(data)
 
 
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

    return pd.DataFrame(data)
 
 
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

def getNormalisedBatch(batch_num, patches_dir='../pre_processing/patches'):
    wac  = np.load(os.path.join(patches_dir, f'X_wac_{batch_num}.npz'))['arr_0']
    dem  = np.load(os.path.join(patches_dir, f'X_dem_{batch_num}.npz'))['arr_0']
    mask = np.load(os.path.join(patches_dir, f'X_mask_{batch_num}.npz'))['arr_0']

    # WAC: 8-bit uint8 -> float32 [0, 1]
    norm_wac = wac.astype(np.float32) / 255.0

    # DEM: percentile-based min-max per patch -> float32 [0, 1]
    # per-patch because elevation range varies across the tile
    norm_dem = np.zeros_like(dem, dtype=np.float32)
    for j in range(len(dem)):
        p1, p99 = np.percentile(dem[j], [1, 99])
        norm_dem[j] = (np.clip(dem[j], p1, p99) - p1) / (p99 - p1 + 1e-8)

    return norm_wac, norm_dem, mask


def getAugmentedBatch(batch_num, patches_dir='../pre_processing/patches'):
    wac, dem, mask = getNormalisedBatch(batch_num, patches_dir)

    # apply random augmentation to each patch independently
    for j in range(len(wac)):
        wac[j], dem[j], mask[j] = LunarDataset.augment(wac[j], dem[j], mask[j])

    return wac, dem, mask