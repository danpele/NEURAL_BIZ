"""
nn_quantlet.py -- one Quantlet per chart of the NN decks (same layout as the MFM and SFM courses)
=====================================================================================================
Quantlets/Lk/NN_Lk_<chart>/
    Metainfo.txt            QuantLet metadata (name, description, keywords, data, output)
    NN_Lk_<chart>.ipynb   self-contained notebook, executed: loads the public data, computes, draws the chart
    <chart>.png             the chart it draws

Rules (see Quantlets/README.md):
* the notebook runs on its own, locally or on Google Colab: no import of the lecture modules, only the code needed
  for that chart; data are read from the course repository (raw GitHub URL), or from data/clean on a local copy;
* the numbers printed by the notebook are those on the slide;
* a frame of a deck links the Quantlets of its charts automatically (lectures/common/nn_build.py, quantlet_name).

Usage in lectures/Lk/make_quantlets_Lk.py:
    from nn_quantlet import Quantlet
    q = Quantlet(k=2, chart="naive_mae", description="...", keywords="...", datafile="...")
    q.md("# ...").code("...")
    q.write(execute=True)
"""
import datetime
import json
import os
import re

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
QL_ROOT = os.path.join(ROOT, "Quantlets")
RAW = "https://raw.githubusercontent.com/danpele/NEURAL_BIZ/main/"

# first code cell of every Quantlet: where the data are
DATA_CELL = (
    "import os\n"
    "# course data: the local copy of the repository if there is one, else the public repository on GitHub\n"
    "_LOCAL = os.path.join('..', '..', '..', 'data', 'clean')\n"
    "DATA = _LOCAL + os.sep if os.path.isdir(_LOCAL) else '" + RAW + "data/clean/'\n"
    "REPO = os.path.join('..', '..', '..') + os.sep if os.path.isdir(_LOCAL) else '" + RAW + "'"
)


# second code cell of every Quantlet: the course chart style and the show() helper, inlined (no import of course modules)
STYLE_CELL = r"""import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
# course chart style: transparent background, no gridlines, legend at the bottom outside the axes, four-digit years
plt.rcParams.update({
    'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.transparent': True,
    'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#222222', 'axes.labelcolor': '#222222', 'xtick.color': '#222222', 'ytick.color': '#222222',
    'text.color': '#222222', 'font.size': 11, 'axes.labelsize': 11, 'legend.fontsize': 10.5, 'legend.frameon': False,
    'lines.linewidth': 1.4, 'font.family': 'DejaVu Sans'})
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
PAL = {'data': '#2a78d6', 'model': '#eb6834', 'benchmark': '#1baf7a', 'alt': '#eda100',
       'train': '#2a78d6', 'valid': '#eb6834', 'test': '#6250d6', 'positive': '#e34948', 'negative': '#2a78d6'}


def legend_bottom(fig, ncol=4):
    h, l_ = [], []
    for ax in fig.axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l_:
                h.append(hh); l_.append(ll)
    if h:
        fig.legend(h, l_, loc='lower center', ncol=min(ncol, len(h)), bbox_to_anchor=(0.5, 0.0), frameon=False)


def show(key, label, value):
    # print a number of the slide; [key] = its name in lectures/Lk/Lk_numbers.json
    print(f'{label} = {value}   [{key}]')"""

LINE = re.compile(r"^(.*) = (.*?)   \[([\w.\-]+)\]$")


def check_numbers(q, numbers_json, tol=1e-9):
    """Compare every number printed with show() by the executed Quantlet notebook with the lecture numbers JSON.
    Returns (n_checked, list_of_mismatches)."""
    N = json.load(open(numbers_json))
    nb = nbformat.read(os.path.join(q.folder, q.name + ".ipynb"), as_version=4)
    out = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
                  if o.get("output_type") == "stream")
    bad, n = [], 0
    for line in out.splitlines():
        m = LINE.match(line.rstrip())
        if not m:
            continue
        val, key = m.group(2), m.group(3)
        ref = N
        for part in key.split("."):
            ref = ref.get(part) if isinstance(ref, dict) else None
        n += 1
        if ref is None:
            bad.append(f"{key}: not in {os.path.basename(numbers_json)} (notebook {val})")
            continue
        try:
            ok = abs(float(val) - float(ref)) <= tol * max(1.0, abs(float(ref)))
        except (TypeError, ValueError):
            ok = str(val) == str(ref)
        if not ok:
            bad.append(f"{key}: notebook {val} vs json {ref}")
    return n, bad


def _today():
    d = datetime.date.today()
    return d.strftime("%A, %d %B %Y")


class Quantlet:
    def __init__(self, k, chart, description, keywords, datafile, title=None, submitted=None):
        self.k, self.chart = k, chart
        self.name = f"NN_L{k}_{chart}"
        self.folder = os.path.join(QL_ROOT, f"L{k}", self.name)
        self.description, self.keywords, self.datafile = description, keywords, datafile
        self.title = title or description
        self.submitted = submitted or _today()
        self.cells = []

    def md(self, text):
        self.cells.append(nbformat.v4.new_markdown_cell(text))
        return self

    def code(self, text):
        self.cells.append(nbformat.v4.new_code_cell(text))
        return self

    def metainfo(self):
        q = lambda s: s.replace("'", "’")
        return (f"Name of QuantLet: '{self.name}'\n\n"
                "Published in: 'Neural Networks and Deep Learning (NN)'\n\n"
                f"Description: '{q(self.description)}'\n\n"
                f"Keywords: '{q(self.keywords)}'\n\n"
                "Author: 'Daniel Traian Pele'\n\n"
                f"Submitted: '{self.submitted}'\n\n"
                f"Datafile: '{q(self.datafile)}'\n\n"
                f"Output: '{self.chart}.png'\n")

    def write(self, execute=True, timeout=1800):
        os.makedirs(self.folder, exist_ok=True)
        nb = nbformat.v4.new_notebook()
        head = [nbformat.v4.new_markdown_cell(
                    f"# {self.name}\n\n**{self.title}**\n\n"
                    "Neural Networks and Deep Learning (NN)" + (f", Lecture {self.k}" if self.k else "")
                    + " · Daniel Traian Pele, Bucharest University of Economic Studies\n\n"
                    "[Open in Colab](https://colab.research.google.com/github/danpele/NEURAL_BIZ/blob/main/"
                    + os.path.relpath(self.folder, ROOT).replace(os.sep, "/") + f"/{self.name}.ipynb)"),
                nbformat.v4.new_code_cell(DATA_CELL), nbformat.v4.new_code_cell(STYLE_CELL)]
        nb.cells = head + self.cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if execute:
            ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(nb, {"metadata": {"path": self.folder}})
        with open(os.path.join(self.folder, self.name + ".ipynb"), "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        with open(os.path.join(self.folder, "Metainfo.txt"), "w", encoding="utf-8") as f:
            f.write(self.metainfo())
        if execute and not os.path.isfile(os.path.join(self.folder, self.chart + ".png")):
            raise RuntimeError(f"{self.name}: the notebook did not save {self.chart}.png")
        print("Quantlet", self.name, "written" + (" and executed" if execute else ""))
        return self
