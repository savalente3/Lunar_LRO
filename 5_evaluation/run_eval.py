# run_eval
# executes evaluation.ipynb start to finish and writes the outputs back into it,
# so the notebook holds its own figures and tables after the run.
# parameters:
#         none, everything is set in evaluation.ipynb
# outputs:
#         evaluation.ipynb with outputs, and the csv, json and png files the
#         notebook saves under results/<model>/

import os

import nbformat
from nbclient import NotebookClient


here = os.path.dirname(os.path.abspath(__file__))

os.environ['MPLBACKEND'] = 'Agg'

notebook = nbformat.read(os.path.join(here, 'evaluation.ipynb'), as_version=4)

client = NotebookClient(notebook, timeout=None, kernel_name='python3', resources={'metadata': {'path': here}})
client.execute()

nbformat.write(notebook, os.path.join(here, 'evaluation.ipynb'))
