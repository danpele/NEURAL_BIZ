"""
nn_notebook.py -- student notebooks of the NN lectures and seminars (EN + RO), written and executed by script.

Every lecture k has two notebook pairs in notebooks/:
    Lk_notebook.ipynb,  Lk_notebook_ro.ipynb     lecture notebook: reproduces the numbers and charts of the slides;
                                                 its section numbers (§) are the "Experiment yourself" pointers
    Lk_seminar.ipynb,   Lk_seminar_ro.ipynb      seminar notebook for students: tasks + setup, empty cells for the answers
    Lk_seminar_solutions.ipynb (+ _ro)           the same with the solutions, executed (git-ignored, instructor only)

Usage in lectures/Lk/make_notebook_Lk.py:
    from nn_notebook import Notebook
    def cells(nb):                       # nb.T(en, ro) picks the language
        nb.md(nb.T("## §1 Data", "## §1 Datele"))
        nb.code("df = pd.read_csv(DATA + 'german_credit.csv')\\ndf.head()")
        nb.task(nb.T("Fit a logistic regression.", "Estimați o regresie logistică."))   # seminar only
        nb.solution("...code...")         # kept in *_solutions, replaced by an empty cell in the student notebook
    Notebook.build_all(k=1, title=("Deep learning for business", "Deep learning pentru afaceri"), cells=cells,
                       kind="lecture")    # or kind="seminar"

Rules: the notebook runs from a clone of the repository (data in ../data/clean) and on Google Colab (first cell clones
the repository); same code in both languages, only the text differs; executed end to end before it is committed;
real data only; a fixed seed is set in the first code cell.
"""
import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from colab_cell import add_colab_cell  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NBDIR = os.path.join(ROOT, "notebooks")
COLAB = "https://colab.research.google.com/github/danpele/NEURAL_BIZ/blob/main/notebooks/"

HEAD = {
    "en": ("# Neural Networks and Deep Learning — {kind} {k}: {title}\n"
           "Prof. dr. Daniel Traian Pele — Bucharest University of Economic Studies (ASE); "
           "Romanian Academy, Institute for Economic Forecasting (IPE)\n\n"
           "[Open in Colab]({colab})\n\n{intro}"),
    "ro": ("# Rețele neuronale și deep learning — {kind} {k}: {title}\n"
           "Prof. dr. Daniel Traian Pele — Academia de Studii Economice din București (ASE); "
           "Academia Română, Institutul de Prognoză Economică (IPE)\n\n"
           "[Deschide în Colab]({colab})\n\n{intro}"),
}
KIND = {"lecture": {"en": "Lecture", "ro": "Cursul"}, "seminar": {"en": "Seminar", "ro": "Seminarul"}}
INTRO = {
    "lecture": {"en": "This notebook reproduces the numbers and charts of the lecture from the real data in `../data/clean` "
                      "(see `../data/SOURCES.md`). Section numbers (§) match the *Experiment yourself* pointers on the slides.",
                "ro": "Acest notebook reproduce cifrele și graficele cursului pe datele reale din `../data/clean` "
                      "(vezi `../data/SOURCES.md`). Numerele secțiunilor (§) corespund indicațiilor *Exercițiu practic* de pe slide-uri."},
    "seminar": {"en": "Seminar tasks on real data from `../data/clean` (see `../data/SOURCES.md`). Write your code in the empty "
                      "cells under each task; keep the seed fixed so that your numbers can be compared.",
                "ro": "Sarcinile seminarului, pe datele reale din `../data/clean` (vezi `../data/SOURCES.md`). Scrieți codul în "
                      "celulele goale de sub fiecare sarcină; păstrați valoarea fixă a generatorului aleator, ca rezultatele să "
                      "poată fi comparate."},
}
SETUP = ("import os, random, warnings\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
         "warnings.filterwarnings('ignore')\n"
         "DATA = '../data/clean/'\nSEED = 20261108\nrandom.seed(SEED); np.random.seed(SEED)\n"
         "try:\n    import torch\n    torch.manual_seed(SEED)\n"
         "    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')\n"
         "except ImportError:\n    DEVICE = 'cpu'\n"
         "plt.rcParams.update({'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.grid': False,\n"
         "                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})\n"
         "print('device:', DEVICE)")
TODO = {"en": "# your code here", "ro": "# codul vostru aici"}


class Notebook:
    def __init__(self, k, title, lang="en", kind="lecture", solutions=True):
        self.k, self.title, self.lang, self.kind, self.with_sol = k, title, lang, kind, solutions
        self.cells = []

    def T(self, en, ro):
        return en if self.lang == "en" else ro

    def md(self, text):
        self.cells.append(nbformat.v4.new_markdown_cell(text))
        return self

    def code(self, src):
        self.cells.append(nbformat.v4.new_code_cell(src))
        return self

    def task(self, text):
        """A seminar task (markdown, numbered automatically)."""
        n = sum(1 for c in self.cells if c.metadata.get("nn_task")) + 1
        c = nbformat.v4.new_markdown_cell(f"**{self.T('Task', 'Sarcina')} {n}.** {text}")
        c.metadata["nn_task"] = True
        self.cells.append(c)
        return self

    def solution(self, src):
        """Solution code: kept (and executed) in the *_solutions notebook, an empty cell in the student notebook."""
        if self.with_sol:
            c = nbformat.v4.new_code_cell(src)
        else:
            c = nbformat.v4.new_code_cell(TODO[self.lang])
        c.metadata["nn_solution"] = True
        self.cells.append(c)
        return self

    def filename(self):
        base = f"L{self.k}_{'notebook' if self.kind == 'lecture' else 'seminar'}"
        sfx = "_ro" if self.lang == "ro" else ""
        if self.kind == "seminar" and self.with_sol:
            return f"{base}_solutions{sfx}.ipynb"
        return f"{base}{sfx}.ipynb"

    def notebook(self):
        nb = nbformat.v4.new_notebook()
        public = self.filename().replace("_solutions", "")
        head = HEAD[self.lang].format(kind=KIND[self.kind][self.lang], k=self.k, title=self.title,
                                      colab=COLAB + public, intro=INTRO[self.kind][self.lang])
        nb.cells = [nbformat.v4.new_markdown_cell(head), nbformat.v4.new_code_cell(SETUP)] + self.cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        return add_colab_cell(nb, self.lang, "notebooks")

    def write(self, execute=True, timeout=3600):
        os.makedirs(NBDIR, exist_ok=True)
        nb = self.notebook()
        run = execute and (self.kind == "lecture" or self.with_sol)
        if run:
            ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(nb, {"metadata": {"path": NBDIR}})
            errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
            if errs:
                raise RuntimeError(f"{self.filename()}: {errs[0].get('ename')}: {errs[0].get('evalue')}")
        path = os.path.join(NBDIR, self.filename())
        nbformat.write(nb, path)
        print("notebook", os.path.relpath(path, ROOT), "written" + (" and executed" if run else " (student version, not executed)"))
        return path

    @classmethod
    def build_all(cls, k, title, cells, kind="lecture", langs=("en", "ro"), execute=True):
        """Write (and execute) the notebooks of lecture k in both languages; for a seminar, the solutions notebook
        (executed) and the student notebook (solutions replaced by empty cells)."""
        paths = []
        for lang in langs:
            variants = (True, False) if kind == "seminar" else (True,)
            for sol in variants:
                nb = cls(k, title[0] if lang == "en" else title[1], lang, kind, solutions=sol)
                cells(nb)
                paths.append(nb.write(execute=execute))
        return paths
