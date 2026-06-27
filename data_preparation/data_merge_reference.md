# Data Merge Reference

## Tile: E300N1350
- **Longitude:** 90°E → 180°E
- **Latitude:** 0° → 60°N
- **Small craters in tile:** 209,872

---

## Alignment Steps

### 1. Filter Robbins Labels
Keep craters where:
- `LAT_CIRC_IMG`: 0 → 60 (tile lat bounds)
- `LON_CIRC_IMG`: 90 → 180 (tile lon bounds)
- `DIAM_CIRC_IMG` < 10 (small craters only)

**Source:** Robbins (2019) — *"A New Global Database of Lunar Impact Craters"*  
Column names `LAT_CIRC_IMG`, `LON_CIRC_IMG`, `DIAM_CIRC_IMG` defined there.

### 2. Crop the DEM
SLDEM2015 covers 60S–60N (15360 rows) × 0–360°E (46080 cols)  
Resolution: 128 px/degree

Row/col index formula:
- `row = (60 - lat) * 128` → lat 60N = row 0, lat 0 = row 7680
- `col = lon * 128` → lon 90 = col 11520, lon 180 = col 23040
- Slice: `dem[0:7680, 11520:23040]`

**Source:** Barker et al. (2016) — *"A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera"*

### 3. Load WAC Tile
`getRegionalLunarData('WAC_GLOBAL_E300N1350_100M')`  
Already covers exact tile bounds — no cropping needed.

**Source:** Robinson et al. (2010) — *"Lunar Reconnaissance Orbiter Camera (LROC) Instrument Overview"*

---

## Verification
Overlay Robbins crater centres on WAC tile image to confirm spatial alignment before generating masks.
