import os
import io
import requests
import rasterio
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


class LunarDataset:
 
    def __init__(self):
        self.labels = None
        self.globalLunarData = None
        self.regionalLunarData = None
        self.mergedData = None
 
        self.loadLunarImages()

        # To heavy to load on init. Need manual invoking
        self.loadRegionalLunarImages()
        
        self.loadLunarLabels()
        self.generateCraterMasks()
 
    def loadLunarImages(self):
        self.globalLunarData = getGlobalLunarData()
    
    def loadRegionalLunarImages(self):
        self.regionalLunarData = getRegionalLunarData()
    
    def loadLunarLabels(self):
        self.labels = getLunarRobbinsLabels()
    
    def generateCraterMasks(self):
        self.mergedData = getGeneratedCraterMasks(self.globalLunarData, self.labels)
 
    def saveFiles(self, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        
        self.globalLunarData.to_csv(os.path.join(output_dir, "WholeLunarData.csv"))
        self.regionalLunarData.to_csv(os.path.join(output_dir, "RegionalLunarData.csv"))
        self.labels.to_csv(os.path.join(output_dir, "LunarLabels.csv"))
        # self.mergedData.to_csv(os.path.join(output_dir, "LunarDataAndLabels.csv"))
 
 
def getGlobalLunarData():
    url = "https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/WAC_GLOBAL_O000N0000_004P.IMG"
    response = requests.get(url)
    with rasterio.open(io.BytesIO(response.content)) as src:
        data = src.read(1)
    return pd.DataFrame(data)
 
 
def getRegionalLunarData(tile='WAC_GLOBAL_E300S0450_100M'):
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
 
 
def getGeneratedCraterMasks(data, labels):
    return ''