# archive

Superseded files, kept for the record. Nothing in the main pipeline imports or
runs them, and they are not needed to reproduce the results in the report.
Paths inside them are relative to their original folders, so they will not run
from here without editing.

| File | Original location | Superseded by |
|---|---|---|
| `data_merge.ipynb` | `2_data_preparation/` | `data_merge_alltiles.ipynb` (single tile E300N1350 → all 8 tiles). Imports from the old `../data_extraction` folder name. |
| `data_pre_processing.ipynb` | `3_pre_processing/` | `data_pre_processing_alltiles.ipynb` (single tile → all 8 tiles). Imports from the old `../data_extraction` folder name. |
| `train_v2_final_dem.py`, `train_v2_final_wac.py` | `4_training/` | `4_training/train_dilated_U_net.py <channel>` (was `train_v2_final.py`). The three copies differed only in the `CHANNELS` line, and imported `build_loss` and `patchesDirName`, which no longer exist. |
| `evaluation.py` | `5_evaluation/` | `evaluate_model.py`. An older export of `evaluation.ipynb` that uses the notebook-only `display()`. |
| `run_eval.py` | `5_evaluation/` | `evaluate_model.py`. Re-ran `evaluation.ipynb` end to end. |
