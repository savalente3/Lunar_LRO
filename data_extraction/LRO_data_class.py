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