"""
make_notebooks.py -- build and execute the EN and RO notebooks of one case.

Usage (from repo/cases):  python common/make_notebooks.py C1_crisis_2024 [--no-exec]

Each case folder holds ``nb_spec.py`` with
    TITLE = (title_en, title_ro)
    CELLS = [("md", text_en, text_ro), ("code", source), ...]
The code is identical in both notebooks; only the markdown and the LANG switch differ
(the RO notebook shows the Romanian charts). Both notebooks write the same outputs
(figures in both languages, numbers, tables), so running either one is enough.
"""

import importlib.util
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

SETUP = """import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join("..", "common"))   # shared helpers: cases/common/case_utils.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import case_utils as cu
LANG = "{lang}"            # "en" or "ro": language of the charts shown below (both are saved)
cu.set_lang(LANG)
T = lambda en, ro: cu.T(LANG, en, ro)
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)
%matplotlib inline"""


def build(case, execute=True):
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.join(here, "..", case)
    spec = importlib.util.spec_from_file_location("nb_spec", os.path.join(cdir, "nb_spec.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for lang, tag in (("en", "EN"), ("ro", "RO")):
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        title = mod.TITLE[0] if lang == "en" else mod.TITLE[1]
        sub = ("Master - Neural Networks and Deep Learning (NN) - Prof. Daniel Traian Pele\n\n"
               "Bucharest University of Economic Studies - Romanian Academy, Institute for Economic Forecasting"
               if lang == "en" else
               "Master - Rețele neuronale și deep learning (NN) - Prof. Daniel Traian Pele\n\n"
               "Academia de Studii Economice din București - Academia Română, Institutul de Prognoză Economică")
        cells = [nbformat.v4.new_markdown_cell(f"# {title}\n\n{sub}"),
                 nbformat.v4.new_code_cell(SETUP.format(lang=lang))]
        for c in mod.CELLS:
            if c[0] == "md":
                cells.append(nbformat.v4.new_markdown_cell(c[1] if lang == "en" else c[2]))
            else:
                cells.append(nbformat.v4.new_code_cell(c[1].strip("\n")))
        nb.cells = cells
        path = os.path.join(cdir, f"{case.split('_')[0]}_notebook_{tag}.ipynb")
        if execute:
            ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
            ep.preprocess(nb, {"metadata": {"path": cdir}})
        nbformat.write(nb, path)
        print("wrote", os.path.relpath(path, os.path.join(here, "..")))


if __name__ == "__main__":
    build(sys.argv[1], execute="--no-exec" not in sys.argv)
