"""
case_utils.py -- shared helpers for the NN research case studies (cases/Cx_*/).

* Chart style = course standard (lectures/common/nn_plot.py): transparent background,
  no gridlines, legend at the bottom outside the axes, one fixed colour per entity.
* Every chart is drawn twice, once with English and once with Romanian labels:
  write ``def fig_x(lang): ...`` and call ``save_both(fig_x, "name")``.
* Numbers used in the case brief are written by the notebook to ``numbers.json`` and to
  ``numbers_en.tex`` / ``numbers_ro.tex`` (LaTeX macros, decimal point / decimal comma).
* Tables are written to ``tab_<name>_en.tex`` / ``tab_<name>_ro.tex`` with the best value
  of each column in bold.

All paths are relative to this file, so the notebooks run from any working directory
as long as the repository layout (repo/data/clean, repo/cases/...) is kept.
"""

import json
import os
import re

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "data", "clean")

# Categorical palette: same hex values as lectures/common/nn_plot.py and the Beamer preamble.
PAL = {"RO": "#2a78d6", "DE_LU": "#eb6834", "HU": "#1baf7a", "BG": "#eda100",
       "c5": "#e87ba4", "c6": "#008300", "c7": "#6250d6", "c8": "#e34948"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#222222"
MUTED = "#8a8a8a"
EVENT = "#e34948"     # colour used to shade/mark events (crisis windows, spikes)

SRC_EC = {"en": "energy-charts.info (Fraunhofer ISE), exchange/ENTSO-E data",
          "ro": "energy-charts.info (Fraunhofer ISE), date de bursă și ENTSO-E"}
SRC_YF = {"en": "Yahoo Finance daily quotes",
          "ro": "Yahoo Finance, cotații zilnice"}
SRC_EOD = {"en": "EODHD (Bucharest Stock Exchange closing prices, dividend/split-adjusted)",
           "ro": "EODHD (prețuri de închidere la Bursa de Valori București, ajustate pentru dividende și divizări de acțiuni)"}


def style():
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5,
        "legend.fontsize": 9.5, "legend.frameon": False, "lines.linewidth": 1.3,
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "figure.dpi": 100,
        "axes.formatter.use_locale": False,
    })


style()


def T(lang, en, ro):
    """Pick the label for the language."""
    return ro if lang == "ro" else en


def legend_bottom(fig, ncol=4, y=-0.02, handles=None, labels=None):
    """Legend centred under the axes, outside the plotting area."""
    if handles is None:
        handles, labels = [], []
        for ax in fig.axes:
            h, l_ = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l_):
                if ll not in labels and not ll.startswith("_"):
                    handles.append(hh)
                    labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y),
                   frameon=False, handlelength=2.0)


def day_ticks(ax, lang, interval=2):
    """Daily date ticks with the full four-digit year: 2024-07-10 (EN), 10.07.2024 (RO)."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y" if lang == "ro" else "%Y-%m-%d"))


def comma_axis(ax, lang, axis="y", nd=None):
    """Decimal comma on tick labels for Romanian charts. The number of decimals is the one matplotlib
    chooses for the English chart (same tick labels in both languages); ``nd`` is kept for compatibility."""
    if lang != "ro":
        return
    from matplotlib.ticker import ScalarFormatter

    class _Comma(ScalarFormatter):
        def __call__(self, v, pos=None):
            return super().__call__(v, pos).replace(".", ",").replace("-", "−").replace("\u2212", "−")
    f = _Comma(useOffset=False)
    f.set_scientific(False)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def save_both(make, name, outdir="fig", show_lang=None, headline=False):
    """Draw ``make(lang)`` for both languages, save fig/<lang>/<name>.pdf, show one inline.

    If ``headline`` is true the chart is also saved as fig_headline_<lang>.pdf (case root)."""
    shown = None
    for lang in ("en", "ro"):
        fig = make(lang)
        os.makedirs(os.path.join(outdir, lang), exist_ok=True)
        fig.savefig(os.path.join(outdir, lang, name + ".pdf"), transparent=True, bbox_inches="tight")
        if headline:
            fig.savefig(f"fig_headline_{lang}.pdf", transparent=True, bbox_inches="tight")
        if lang == (show_lang or LANG):
            shown = fig
        else:
            plt.close(fig)
    return shown


LANG = "en"


def set_lang(lang):
    global LANG
    LANG = lang


# ------------------------------------------------------------------ numbers and tables
def fmt(x, nd=2, lang="en", pct=False, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"
    if lang == "ro":
        s = s.replace(",", " ").replace(".", ",").replace(" ", ".")
    if pct:
        s += "\\%"
    return s


def write_numbers(nums, digits=None, path="."):
    """nums: {key: value}. Keys must be letters only (they become LaTeX macros \\Nkey).

    Floats are formatted with ``digits[key]`` decimals (default 2); strings are written as is
    (give a (en, ro) tuple for language-specific strings)."""
    digits = digits or {}
    js = {}
    for lang in ("en", "ro"):
        lines = ["% generated by the case notebook -- do not edit"]
        for k, v in nums.items():
            if not re.fullmatch(r"[A-Za-z]+", k):
                raise ValueError(f"macro key must be letters only: {k}")
            if isinstance(v, tuple):
                s = v[1] if lang == "ro" else v[0]
                js[k] = v[0]
            elif isinstance(v, (int, np.integer)):
                # calendar years (keys ending in "Year", 1900-2100) are printed without a thousands separator
                s = str(int(v)) if (k.endswith("Year") and 1900 <= int(v) <= 2100) else fmt(float(v), 0, lang)
                js[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                s = fmt(float(v), digits.get(k, 2), lang)
                if s.startswith("-"):
                    s = "$-$" + s[1:]
                js[k] = round(float(v), 6)
            else:
                s = str(v)
                js[k] = s
            lines.append(f"\\newcommand{{\\N{k}}}{{{s}}}")
        with open(os.path.join(path, f"numbers_{lang}.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(path, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, indent=1, ensure_ascii=False)


def write_table(df, name, header, digits=2, best=None, path=".", colfmt=None, first_col=None):
    """Write a booktabs table body (no float, no caption) for both languages.

    df: rows = items, columns = numeric metrics (index is the row label, may be a (en, ro) tuple).
    header: list of (en, ro) column headers, first one for the label column.
    best: {column: 'min' | 'max' | 'absmin'} -> bold the best value in that column.
    digits: int or {column: int}."""
    best = best or {}
    for lang in ("en", "ro"):
        cols = list(df.columns)
        spec = colfmt or ("l" + "r" * len(cols))
        out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
               " & ".join(T(lang, *h) for h in header) + r" \\", "\\midrule"]
        bidx = {}
        for c, how in best.items():
            s = pd.to_numeric(df[c], errors="coerce")
            if how == "min":
                bidx[c] = s.idxmin()
            elif how == "max":
                bidx[c] = s.idxmax()
            elif how == "absmin":
                bidx[c] = s.abs().idxmin()
        for idx, row in df.iterrows():
            lab = T(lang, *idx) if isinstance(idx, tuple) else str(idx)
            cells = [lab]
            for c in cols:
                v = row[c]
                nd = digits.get(c, 2) if isinstance(digits, dict) else digits
                if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                    s = fmt(float(v), nd, lang)
                elif isinstance(v, tuple):
                    s = T(lang, *v)
                else:
                    s = "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                if len(s) > 1 and s[0] == "-" and s[1].isdigit():
                    s = "$-$" + s[1:]
                if c in bidx and bidx[c] == idx:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
            out.append(" & ".join(cells) + r" \\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(path, f"tab_{name}_{lang}.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")


def show_table(df, best=None, digits=2):
    """Pretty display in the notebook with the best value per column highlighted."""
    best = best or {}
    sty = df.style.format(precision=digits)
    for c, how in best.items():
        if how == "min":
            sty = sty.highlight_min(subset=[c], props="font-weight:bold; background-color:#d8e8fa")
        elif how == "max":
            sty = sty.highlight_max(subset=[c], props="font-weight:bold; background-color:#d8e8fa")
    return sty


# ------------------------------------------------------------------ data loaders
def _p(name):
    return os.path.join(DATA, name)


def prices_daily():
    """Daily base day-ahead prices (CET/CEST day), EUR/MWh."""
    return pd.read_csv(_p("power_da_prices_daily_cet.csv"), parse_dates=["date_cet"]).set_index("date_cet")


def prices_hourly(tz="Europe/Bucharest"):
    x = pd.read_csv(_p("power_da_prices_hourly_utc.csv"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    x.index = x.index.tz_convert(tz)
    return x


def load_generation_ro_hourly(tz="Europe/Bucharest"):
    x = pd.read_csv(_p("ro_load_generation_hourly_utc.csv"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    x.index = x.index.tz_convert(tz)
    return x


def markets_daily(field="adjclose"):
    """Wide table date x ticker of Yahoo Finance daily prices (markets_daily.csv)."""
    x = pd.read_csv(_p("markets_daily.csv"), parse_dates=["date"])
    return x.pivot(index="date", columns="ticker", values=field)


def bvb_daily(field="adjclose"):
    x = pd.read_csv(_p("bvb_equities_daily.csv"), parse_dates=["date"])
    return x.pivot(index="date", columns="ticker", values=field)


def fraud():
    return pd.read_csv(_p("creditcard_fraud_sample.csv"))


def german_credit():
    return pd.read_csv(_p("german_credit.csv"))


def period(idx, lang="en"):
    """'Jan 2015 -- Oct 2026' style period string from a DatetimeIndex."""
    a, b = pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
    if lang == "ro":
        return f"{a:%d.%m.%Y}--{b:%d.%m.%Y}"
    return f"{a:%d %b %Y}--{b:%d %b %Y}"
