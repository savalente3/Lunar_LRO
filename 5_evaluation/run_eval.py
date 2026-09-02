#!/usr/bin/env python
"""Run evaluation.ipynb once per input channel.

    python run_eval.py              # wac, dem, both
    python run_eval.py both wac     # just those, in that order

Each channel runs in its own interpreter, so nothing leaks between runs - Keras
holds a loaded model and a populated graph, and the notebook caches patch files
in a module-level dict, both of which would otherwise carry across channels.

The channel reaches the notebook through EVAL_CHANNELS, which cell 2 reads. Each
run writes sweep.csv, per_patch.csv and headline.json into
results/<model>/<channel>/, and the final cell redraws results/<model>/comparison_pr.png
from whichever channels have completed.

Figures are written to disk rather than displayed - the runner forces a
non-interactive matplotlib backend so an unattended run cannot block.
"""

import os
import subprocess
import sys

CHANNELS = ('wac', 'dem', 'both')
NOTEBOOK = 'evaluation.ipynb'

# executes the notebook's code cells in order, in this fresh interpreter
RUNNER = r'''
import json, sys, traceback

import matplotlib
matplotlib.use('Agg')          # never open a window; savefig still works
import matplotlib.pyplot as plt

notebook_path = sys.argv[1]

with open(notebook_path) as handle:
    notebook = json.load(handle)

# the notebook is written for IPython, which supplies display() as a builtin
namespace = {'__name__': '__main__', 'display': lambda *args, **kwargs: [print(a) for a in args]}

cells = [(i, ''.join(c['source'])) for i, c in enumerate(notebook['cells']) if c['cell_type'] == 'code']

for index, source in cells:

    if not source.strip():
        continue

    print(f'\n----- cell {index} -----', flush=True)

    try:
        exec(compile(source, f'<cell {index}>', 'exec'), namespace)
    except Exception:
        print(f'\nFAILED in cell {index}:', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    plt.close('all')           # cells build figures then show(); free them
'''


def main():
    requested = sys.argv[1:] or list(CHANNELS)

    unknown = [c for c in requested if c not in CHANNELS]

    if unknown:
        sys.exit(f'unknown channel(s) {unknown}; expected any of {list(CHANNELS)}')

    here = os.path.dirname(os.path.abspath(__file__))
    notebook = os.path.join(here, NOTEBOOK)

    if not os.path.exists(notebook):
        sys.exit(f'{notebook} not found')

    failures = []

    for channel in requested:

        print(f'\n{"=" * 62}\n  {channel}\n{"=" * 62}', flush=True)

        environment = dict(os.environ)
        environment['EVAL_CHANNELS'] = channel
        environment['MPLBACKEND'] = 'Agg'

        result = subprocess.run(
            [sys.executable, '-c', RUNNER, notebook],
            cwd=here,               # notebook paths are relative to 5_evaluation/
            env=environment,
        )

        if result.returncode != 0:
            failures.append(channel)
            print(f'\n*** {channel} failed (exit {result.returncode}) ***', flush=True)

    print(f'\n{"=" * 62}')

    if failures:
        print(f'failed: {failures}')
        sys.exit(1)

    print(f'completed: {requested}')


if __name__ == '__main__':
    main()
