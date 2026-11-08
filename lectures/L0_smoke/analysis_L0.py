"""
analysis_L0.py -- pipeline smoke test (not part of the course site): one real-data chart and its numbers.
Writes lectures/L0_smoke/L0_numbers.json and charts/en|ro/sp500_drawdown.pdf.
Run from anywhere:  python3 lectures/L0_smoke/analysis_L0.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

npl.style()
N = {"charts": {}}

# ---------------------------------------------------------------- 0. data and cut-off (last complete month)
px = npl.markets("adjclose", ["SP500"])["SP500"].dropna()
end = (px.index.max() + np.timedelta64(1, "D")).to_period("M").to_timestamp() - np.timedelta64(1, "D")
px = px.loc[:end]
r = np.log(px).diff().dropna()
N["data"] = {"first": str(px.index.min().date()), "last": str(px.index.max().date()), "n_days": int(len(r))}

# ---------------------------------------------------------------- 1. annualised volatility and maximum drawdown
vol = float(r.std(ddof=1) * np.sqrt(252))
dd = px / px.cummax() - 1
N["sp500"] = {"vol_ann_pct": round(100 * vol, 2), "maxdd_pct": round(100 * float(dd.min()), 2),
              "maxdd_date": str(dd.idxmin().date())}

for lang in ("en", "ro"):
    T = {"en": ("Drawdown from the previous peak (%)", "S&P 500 drawdown"),
         "ro": ("Scăderea față de maximul anterior (%)", "Scăderea indicelui S&P 500")}[lang]
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.fill_between(dd.index, 100 * dd.values, 0, color=npl.PAL["data"], alpha=0.35, lw=0)
    ax.plot(dd.index, 100 * dd.values, color=npl.PAL["data"], lw=0.8, label=T[1])
    ax.set_ylabel(T[0])
    ax.set_xlim(dd.index.min(), dd.index.max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "sp500_drawdown")
npl.chart_meta(N, "sp500_drawdown", "yahoo", npl.period(px.index.min(), px.index.max(), "en") + ", trading days",
               npl.period(px.index.min(), px.index.max(), "ro") + ", zile de tranzacționare")

with open(os.path.join(HERE, "L0_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1)
print("wrote L0_numbers.json and", len(N["charts"]), "chart x 2 languages")
