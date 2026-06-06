# LRO Crater Classifier — Setup Guide

## Project Structure

```
Lunar_LRO/
├── data_extraction/
│   ├── data/
│   │   ├── LunarData.csv              # Saved WAC image metadata
│   │   └── LunarLabels.csv            # Saved Robbins catalogue
│   ├── LRO_data_class.ipynb           # LunarDataset class
│   ├── Lunar_LRO_images.ipynb         # Fetch WAC imagery from LROC PDS
│   └── Lunar_Robbins_labels.ipynb     # Fetch Robbins catalogue from Kaggle
├── training/
│   └── training.ipynb                 # Model training + MLflow logging
├── environment.yml                    # Conda environment
└── README.md                          # This file
```

---

## 1. Conda Environment

All dependencies are managed via `environment.yml`. To set up the environment:

```bash
conda env create -f environment.yml
conda activate lunar_lro
```

When new packages are installed, update the file to keep it in sync:

```bash
conda env export > environment.yml
```

Key packages installed:
- `pandas` — data wrangling
- `rasterio` — reading PDS `.IMG` files from LROC
- `kagglehub` — fetching the Robbins catalogue
- `keras` / `tensorflow` — model building and training
- `mlflow` — experiment tracking
- `matplotlib` — visualisation

---

## 2. Kaggle API Key (Robbins Crater Catalogue)

An API token can be generated at [kaggle.com/settings](https://kaggle.com/settings) → **API Tokens → Generate New Token**

```bash
mkdir -p ~/.kaggle
echo 'YOUR_KAGGLE_API_TOKEN' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

The Robbins catalogue is fetched using `kagglehub`:

```python
import kagglehub
kagglehub.dataset_load("robbins-crater-catalogue")
```

Note: use `dataset_load()` not `load_dataset()` — the latter is deprecated.

---

## 3. NASA WAC Imagery (LROC PDS)

WAC imagery is downloaded directly from the LROC PDS server — no API key required. Files are loaded into memory using `rasterio` without saving to disk:

```python
import requests, rasterio, io

url = "https://pds.lroc.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/BDR/WAC_GLOBAL/WAC_GLOBAL_O000N0000_004P.IMG"

response = requests.get(url)
with rasterio.open(io.BytesIO(response.content)) as src:
    data = src.read(1)   # numpy array of pixel brightness values
    transform = src.transform  # used for coordinate conversion in mask generation
```

Available resolutions (global mosaic `O000N0000`):

| File suffix | Resolution | File size |
|---|---|---|
| `004P` | lowest | 828 KB |
| `008P` | | 3.2 MB |
| `016P` | | 13 MB |
| `032P` | | 51 MB |
| `064P` | | 205 MB |
| `100M` | highest | 4.5 GB |

Start with `004P` for development, scale up for training.

---

## 4. MLflow Experiment Tracking

MLflow logs hyperparameters and results for every training run. To view results:

```bash
mlflow ui
```

Then open `localhost:5000` in your browser. Training runs are logged in `training/training.ipynb`.
