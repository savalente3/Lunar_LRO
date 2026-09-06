"""
Combine per-run training history CSVs into one table.

Reads every history_*.csv in the checkpoints directory, tags each row with the
run it came from, and writes a single combined CSV. The per-run files can then
be removed and the individual params JSONs ignored - the combined table and
MLflow together hold the record.
"""

import glob
import os
import pandas as pd

CKPT_DIR = 'checkpoints'
OUT = 'checkpoints/all_histories.csv'

parts = []
for path in sorted(glob.glob(os.path.join(CKPT_DIR, 'history_*.csv'))):
    run = os.path.basename(path)[len('history_'):-len('.csv')]
    df = pd.read_csv(path)
    df.insert(0, 'run', run)
    parts.append(df)

combined = pd.concat(parts, ignore_index=True)
combined.to_csv(OUT, index=False)
print(f'combined {len(parts)} runs -> {OUT} ({len(combined)} rows)')
print(combined['run'].value_counts())