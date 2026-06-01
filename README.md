# LRO Crater Classifier — Setup Guide

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

---

## 2. Kaggle API Key (Robbins Crater Catalogue)

An API token can be generated at [kaggle.com/settings](https://kaggle.com/settings) → **API Tokens → Generate New Token**

```bash
mkdir -p ~/.kaggle
echo 'YOUR_KAGGLE_API_TOKEN' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

The Jupyter notebook will use this key automatically once stored.

---

## 3. NASA Earthdata Token (WAC Lunar Imagery)

A token can be generated at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) → **My Profile → Generate Token**

```bash
echo 'export NASA_EARTHDATA_TOKEN=YOUR_TOKEN' >> ~/.zshrc
source ~/.zshrc
```

The token is used in API requests as follows:

```python
headers = {"Authorization": f"Bearer {os.environ['NASA_EARTHDATA_TOKEN']}"}
```