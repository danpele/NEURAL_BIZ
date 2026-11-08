"""build_briefs.py -- write the EN/RO wrappers of each case brief and compile them with pdflatex.

Usage (from repo/cases):  python common/build_briefs.py [C1_crisis_2024 ...]   (no argument = all cases)
The notebook of the case must have been run first (it writes numbers_<lang>.tex, tab_*.tex, fig/)."""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WRAP = r"""\documentclass[11pt]{article}
\newif\ifRO \RO%s
\input{../common/case_preamble}
\input{%s_brief}
"""


def build(case):
    cdir = os.path.join(ROOT, case)
    tag = case.split("_")[0]
    for lang, flag in (("EN", "false"), ("RO", "true")):
        name = f"{tag}_case_{lang}"
        with open(os.path.join(cdir, name + ".tex"), "w", encoding="utf-8") as f:
            f.write(WRAP % (flag, tag))
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                               cwd=cdir, capture_output=True, text=True)
        log = open(os.path.join(cdir, name + ".log"), encoding="latin-1").read()
        bad = [l for l in log.splitlines() if l.startswith("!") or "Overfull" in l or "undefined" in l.lower()]
        pages = [l for l in log.splitlines() if "Output written" in l]
        print(case, lang, "rc", r.returncode, pages[-1] if pages else "NO PDF", "| issues:", len(bad))
        for l in bad[:8]:
            print("   ", l)
        for ext in ("aux", "log", "out"):
            p = os.path.join(cdir, f"{name}.{ext}")
            if os.path.exists(p) and r.returncode == 0 and not bad:
                os.remove(p)


if __name__ == "__main__":
    cases = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "C*_*")))
    for c in cases:
        build(c)
