# [source]: N. Khedkar (project partner) - 4_training/combine_csvs.py

# combine_csvs
# gathers every per run history csv in checkpoints into one table, tagged by run,
# so the per run files can be dropped and the combined table plus mlflow hold the
# record.
# parameters:
#         none, it reads whatever history csv files are in checkpoints
# outputs:
#         checkpoints/all_histories.csv, one row per epoch per run


import os
import glob
import pandas as pd


CHECKPOINT_DIR = 'checkpoints'
OUT_CSV = 'checkpoints/all_histories.csv'


parts = []

for path in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'history_*.csv'))):

    run = os.path.basename(path)[len('history_'):-len('.csv')]

    history = pd.read_csv(path)
    history.insert(0, 'run', run)

    parts.append(history)


combined = pd.concat(parts, ignore_index=True)
combined.to_csv(OUT_CSV, index=False)

print(f'combined {len(parts)} runs -> {OUT_CSV} ({len(combined)} rows)')
print(combined['run'].value_counts())
