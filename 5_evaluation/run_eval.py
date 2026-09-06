import os
import sys

import nbformat
from nbclient import NotebookClient


channels = sys.argv[1:] or ['wac', 'dem', 'both']

here = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(here, 'evaluation.ipynb')


for channel in channels:

    print(f'evaluating {channel}', flush=True)

    os.environ['EVAL_CHANNELS'] = channel
    os.environ['MPLBACKEND'] = 'Agg'

    notebook = nbformat.read(notebook_path, as_version=4)

    client = NotebookClient(notebook, timeout=None, kernel_name='python3', resources={'metadata': {'path': here}})
    client.execute()

print(f'done: {channels}')
