"""
nn_plot.py -- shared chart style, chart metadata and data loaders for all NN lectures.

Chart rules (course standard, see lectures/STYLE.md):
  * transparent background, no gridlines, no in-chart title repeating the slide title;
  * legend at the bottom, OUTSIDE the axes (``legend_bottom(fig)``; never ``ax.legend()``);
  * one y-axis per panel (no twin axes); fixed colour per role or entity (``PAL``, ``SERIES``);
  * four-digit years on every date axis (``year_ticks``, ``date_axis``); never '15 or 15;
  * every chart drawn twice: charts/en/<name>.pdf and charts/ro/<name>.pdf; the Romanian one is fully Romanian
    (labels, legend, month names) with a decimal comma and a true minus sign (``out(..., lang='ro')`` does the axes);
  * the source + data period are NOT drawn in the PDF: register them with ``chart_meta`` in the numbers JSON;
    the deck builder prints them under the figure.
  * drawn at display size: 6.4 in wide, about 2-2.6 in tall, 10-11 pt fonts.

Usage in lectures/Lk/analysis_Lk.py:
    import nn_plot as npl
    npl.style()
    N = {"charts": {}}
    for lang in ("en", "ro"):
        fig, ax = plt.subplots(figsize=(6.4, 2.4))
        ...
        npl.legend_bottom(fig)
        npl.out(fig, HERE, lang, "my_chart")
    npl.chart_meta(N, "my_chart", "yahoo", "2005-01-03 to 2026-10-30", "3 ianuarie 2005 - 30 octombrie 2026")
"""

import os
import random
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "data", "clean")
RAW = os.path.join(REPO, "data", "raw")

# ---------------------------------------------------------------- palette
# Categorical palette in a validated adjacent-pair order (same hex in nn_preamble.tex as Ser1..Ser8).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Fixed colour per role, identical in every lecture, so that a colour means the same thing all semester.
PAL = {
    "data": "#2a78d6",        # observed series / real values
    "model": "#eb6834",       # the neural network (main model)
    "benchmark": "#1baf7a",   # linear / logistic / naive benchmark
    "alt": "#eda100",         # second alternative model
    "train": "#2a78d6", "valid": "#eb6834", "test": "#6250d6",
    "positive": "#e34948",    # event class: fraud, default, illicit, spike
    "negative": "#2a78d6",    # non-event class
    "c5": "#e87ba4", "c6": "#008300", "c7": "#6250d6", "c8": "#e34948",
}
INK = "#222222"
MUTED = "#8a8a8a"

MON = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
       "ro": ["ian.", "feb.", "mar.", "apr.", "mai", "iun.", "iul.", "aug.", "sep.", "oct.", "nov.", "dec."]}

# Source lines printed under the charts (EN, RO); key = dataset of data/SOURCES.md
SRC = {
    "fraud": ("ULB Machine Learning Group and Worldline, credit-card transactions (OpenML 1597)",
              "ULB Machine Learning Group și Worldline, tranzacții cu carduri (OpenML 1597)"),
    "german": ("UCI Machine Learning Repository, Statlog German credit", "UCI Machine Learning Repository, Statlog German credit"),
    "hmeq": ("Baesens, Roesch and Scheule (2016), HMEQ data", "Baesens, Roesch și Scheule (2016), datele HMEQ"),
    "california": ("StatLib, California housing, 1990 census (Pace and Barry, 1997)",
                   "StatLib, locuințe din California, recensămîntul din 1990 (Pace și Barry, 1997)"),
    "yahoo": ("Yahoo Finance, daily quotes", "Yahoo Finance, cotații zilnice"),
    "bvb": ("EODHD, Bucharest Stock Exchange closing prices", "EODHD, prețuri de închidere la Bursa de Valori București"),
    "power": ("energy-charts.info (Fraunhofer ISE), from exchange and ENTSO-E data",
              "energy-charts.info (Fraunhofer ISE), din datele burselor și ale ENTSO-E"),
    "yield": ("European Central Bank, euro area AAA yield curve", "Banca Centrală Europeană, curba randamentelor AAA din zona euro"),
    "phrasebank": ("Financial PhraseBank (Malo et al., 2014)", "Financial PhraseBank (Malo et al., 2014)"),
    "tweets": ("Twitter financial news datasets (Hugging Face, zeroshot)", "Seturile Twitter financial news (Hugging Face, zeroshot)"),
    "speeches": ("European Central Bank, speeches dataset", "Banca Centrală Europeană, setul de discursuri"),
    "elliptic": ("Elliptic Bitcoin transaction graph (Weber et al., 2019)", "Graful tranzacțiilor Bitcoin Elliptic (Weber et al., 2019)"),
    "fashion_mnist": ("Fashion-MNIST, Zalando Research (Xiao et al., 2017)", "Fashion-MNIST, Zalando Research (Xiao et al., 2017)"),
    "cifar10": ("CIFAR-10 (Krizhevsky, 2009)", "CIFAR-10 (Krizhevsky, 2009)"),
    "eurosat": ("EuroSAT, Sentinel-2 images (Helber et al., 2019)", "EuroSAT, imagini Sentinel-2 (Helber et al., 2019)"),
}


def style():
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
        "legend.fontsize": 10.5, "legend.frameon": False, "lines.linewidth": 1.4,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.unicode_minus": True,
    })


def legend_bottom(fig, handles=None, labels=None, ncol=4, y=0.0):
    """Legend centred under the axes, outside the plotting area (the only legend placement of the course)."""
    if handles is None:
        handles, labels = [], []
        for ax in fig.axes:
            h, l_ = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l_):
                if ll not in labels and not ll.startswith("_"):
                    handles.append(hh)
                    labels.append(ll)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(ncol, len(handles)), bbox_to_anchor=(0.5, y),
                   frameon=False, handlelength=2.0, columnspacing=1.2)


def year_ticks(ax, years, x=None, star=None, rotate=None, fontsize=None):
    """Full four-digit year labels (never '15 or 15). `x` = tick positions (default: the years themselves);
    `star` = a partial year marked with an asterisk; labels are rotated 45 degrees when they would crowd the panel."""
    years = list(years)
    x = years if x is None else list(x)
    lab = [f"{int(y)}" + ("*" if star is not None and int(y) == int(star) else "") for y in years]
    ax.set_xticks(x)
    if rotate is None:
        w_in = ax.get_position().width * ax.figure.get_figwidth()
        rotate = len(lab) * 0.42 > w_in
    kw = {"fontsize": fontsize} if fontsize else {}
    if rotate:
        ax.set_xticklabels(lab, rotation=45, ha="right", rotation_mode="anchor", **kw)
    else:
        ax.set_xticklabels(lab, **kw)


def date_axis(ax, lang="en", span_years=None):
    """Date axis with four-digit years; for spans under two years, months (in the language of the chart) + year."""
    lo, hi = [mdates.num2date(v) for v in ax.get_xlim()]
    span = span_years if span_years is not None else (hi - lo).days / 365.25
    if span >= 2:
        step = 1 if span <= 8 else (2 if span <= 16 else 5)
        ax.xaxis.set_major_locator(mdates.YearLocator(step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1 if span < 0.6 else 3))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, p: f"{MON[lang][mdates.num2date(v).month - 1]}\n{mdates.num2date(v).year}"
            if mdates.num2date(v).month == 1 or p == 0 else MON[lang][mdates.num2date(v).month - 1]))


def ro_number(v, decimals=None):
    """Romanian number: decimal comma, true minus sign."""
    s = f"{v:g}" if decimals is None else f"{v:.{decimals}f}"
    return s.replace(".", ",").replace("-", "−")


def comma_axes(fig, lang, also_x=()):
    """Romanian charts: decimal comma and true minus on every numeric linear y axis (and the x axes in also_x)."""
    if lang != "ro":
        return
    f_ = FuncFormatter(lambda v, p: ro_number(v))
    for ax in fig.axes:
        if ax.get_yscale() == "linear" and not getattr(ax, "_keep_y", False):
            ax.yaxis.set_major_formatter(f_)
    for ax in also_x:
        ax.xaxis.set_major_formatter(f_)


def audit_figure(fig):
    """Course chart rules that can be checked automatically; returns a list of problems (empty = fine)."""
    bad = []
    for ax in fig.axes:
        if ax.get_legend() is not None:
            bad.append("legend inside an axes: use legend_bottom(fig)")
        if ax.get_title():
            bad.append(f"in-chart title '{ax.get_title()[:30]}': the slide title says it")
        try:
            conv = ax.xaxis.get_converter()
        except AttributeError:
            conv = ax.xaxis.converter
        is_date = conv is not None and "date" in type(conv).__name__.lower()
        for t in ax.get_xticklabels():
            s = t.get_text().strip().split("\n")[-1]
            if re.fullmatch(r"['’]\d{2}", s) or (is_date and re.fullmatch(r"\d{2}", s)):
                bad.append(f"two-digit year tick '{s}'")
        if ax.get_facecolor()[3] > 0:
            bad.append("axes background is not transparent")
    if fig.get_facecolor()[3] > 0:
        bad.append("figure background is not transparent")
    return bad


def save(fig, path, bottom=0.17):
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)


def out(fig, here, lang, name, bottom=0.17, also_x=(), png=False, strict=True):
    """Save a chart of lecture folder `here` as charts/<lang>/<name>.pdf (and .png if asked), after the RO number
    format and the automatic rule audit (strict: a rule violation stops the analysis)."""
    fig.canvas.draw()
    probs = audit_figure(fig)
    if probs and strict:
        raise ValueError(f"chart {name} ({lang}): " + "; ".join(probs))
    comma_axes(fig, lang, also_x)
    d = os.path.join(here, "charts", lang)
    os.makedirs(d, exist_ok=True)
    if png:
        fig.savefig(os.path.join(d, name + ".png"), transparent=True, bbox_inches="tight", dpi=200)
    save(fig, os.path.join(d, name + ".pdf"), bottom=bottom)


def chart_meta(N, name, src_key, period_en, period_ro, source_en=None, source_ro=None):
    """Register source + data period of a chart in the numbers JSON (printed under the figure by the builder)."""
    s_en, s_ro = SRC.get(src_key, (src_key, src_key))
    N.setdefault("charts", {})[name] = {"source_en": source_en or s_en, "source_ro": source_ro or s_ro,
                                        "period_en": period_en, "period_ro": period_ro}


def period(a, b, lang="en"):
    """'3 Jan 2005 to 30 Oct 2026' / '3 ianuarie 2005 - 30 octombrie 2026'."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if lang == "ro":
        mm = ["ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie", "august", "septembrie",
              "octombrie", "noiembrie", "decembrie"]
        return f"{a.day} {mm[a.month - 1]} {a.year} - {b.day} {mm[b.month - 1]} {b.year}"
    return f"{a.day} {MON['en'][a.month - 1]} {a.year} to {b.day} {MON['en'][b.month - 1]} {b.year}"


# ---------------------------------------------------------------- reproducibility
def seed_all(seed=20261108):
    """Same seed for Python, NumPy and PyTorch (CPU and MPS); report the seed with every trained result."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
    return seed


def device():
    """'mps' on Apple silicon, 'cuda' on an NVIDIA GPU (Colab), else 'cpu'."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------- loaders (data/clean, see data/SOURCES.md)
def _p(name):
    return os.path.join(DATA, name)


def fraud():
    """ULB credit-card fraud: all frauds + 10 % sample of legitimate transactions; column `weight` restores the full data."""
    return pd.read_csv(_p("creditcard_fraud_sample.csv"))


def german_credit():
    return pd.read_csv(_p("german_credit.csv"))


def hmeq():
    return pd.read_csv(_p("hmeq.csv"))


def california():
    return pd.read_csv(_p("california_housing.csv"))


def markets(field="adjclose", tickers=None):
    """Wide table date x ticker from markets_daily.csv (Yahoo Finance): SP500, EUROSTOXX50, DAX, VIX, EURUSD, BTCUSD, AAPL, ..."""
    x = pd.read_csv(_p("markets_daily.csv"), parse_dates=["date"])
    w = x.pivot(index="date", columns="ticker", values=field)
    return w[tickers] if tickers else w


def bvb(field="adjclose", tickers=None):
    """Wide table date x ticker of Bucharest Stock Exchange shares (EODHD), RON."""
    x = pd.read_csv(_p("bvb_equities_daily.csv"), parse_dates=["date"])
    w = x.pivot(index="date", columns="ticker", values=field)
    return w[tickers] if tickers else w


def prices_hourly(tz="Europe/Brussels"):
    """Hourly day-ahead prices RO, BG, HU, DE_LU (EUR/MWh); 15-min prices averaged to the hour from Oct 2025."""
    x = pd.read_csv(_p("power_da_prices_hourly_utc.csv"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    x.index = x.index.tz_convert(tz)
    return x


def prices_daily():
    return pd.read_csv(_p("power_da_prices_daily_cet.csv"), parse_dates=["date_cet"]).set_index("date_cet")


def load_generation_ro(tz="Europe/Bucharest"):
    """Romanian load and generation by source, mean MW per hour."""
    x = pd.read_csv(_p("ro_load_generation_hourly_utc.csv"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    x.index = x.index.tz_convert(tz)
    return x


def yield_curve():
    return pd.read_csv(_p("ecb_yield_curve_daily.csv"), parse_dates=["date"]).set_index("date")


def phrasebank(min_agreement=50):
    x = pd.read_csv(_p("financial_phrasebank.csv"))
    return x[x["min_agreement_pct"] >= min_agreement].reset_index(drop=True)


def tweets_sentiment():
    return pd.read_csv(_p("twitter_fin_sentiment.csv"))


def tweets_topic():
    return pd.read_csv(_p("twitter_fin_topic.csv"))


def ecb_speeches():
    return pd.read_csv(_p("ecb_speeches_2022plus.csv"), parse_dates=["date"])


def vision_root():
    """Folder where torchvision datasets (Fashion-MNIST, CIFAR-10, EuroSAT) are cached (download=True on first use)."""
    d = os.path.join(RAW, "vision")
    os.makedirs(d, exist_ok=True)
    return d


def elliptic_root():
    """Folder for torch_geometric.datasets.EllipticBitcoinDataset (downloaded on first use)."""
    d = os.path.join(RAW, "elliptic_pyg")
    os.makedirs(d, exist_ok=True)
    return d
