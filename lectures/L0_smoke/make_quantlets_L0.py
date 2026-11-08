"""
make_quantlets_L0.py -- smoke test of the Quantlet maker: Quantlets/L0/NN_L0_sp500_drawdown/ (Metainfo.txt, notebook, PNG),
executed, and every number printed with show() compared with L0_numbers.json.
Run:  python3 lectures/L0_smoke/make_quantlets_L0.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

q = Quantlet(k=0, chart="sp500_drawdown",
             description="Drawdown of the S&P 500 index from its previous peak, daily adjusted closes, with the maximum "
                         "drawdown and the annualised volatility of daily log returns. Source: Yahoo Finance.",
             keywords="drawdown, volatility, S&P 500, log returns, risk, deep learning data",
             datafile="data/clean/markets_daily.csv (Yahoo Finance, ticker SP500)")
q.md("Drawdown $D_t = P_t / \\max_{s \\le t} P_s - 1$ of the S&P 500, data cut at the last complete month.")
q.code("x = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])\n"
       "px = x[x.ticker == 'SP500'].set_index('date')['adjclose'].dropna()\n"
       "end = (px.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)\n"
       "px = px.loc[:end]\nr = np.log(px).diff().dropna()\ndd = px / px.cummax() - 1\n"
       "show('sp500.maxdd_pct', 'maximum drawdown (%)', round(100 * float(dd.min()), 2))\n"
       "show('sp500.vol_ann_pct', 'annualised volatility (%)', round(100 * float(r.std(ddof=1) * np.sqrt(252)), 2))\n"
       "show('data.n_days', 'daily log returns', len(r))")
q.code("fig, ax = plt.subplots(figsize=(6.4, 2.3))\n"
       "ax.fill_between(dd.index, 100 * dd.values, 0, color=PAL['data'], alpha=0.35, lw=0)\n"
       "ax.plot(dd.index, 100 * dd.values, color=PAL['data'], lw=0.8, label='S&P 500 drawdown')\n"
       "ax.set_ylabel('Drawdown from the previous peak (%)')\nax.set_xlim(dd.index.min(), dd.index.max())\n"
       "legend_bottom(fig)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
       "fig.savefig('sp500_drawdown.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
q.write(execute=True)
n, bad = check_numbers(q, os.path.join(HERE, "L0_numbers.json"))
print(f"{n} numbers checked against L0_numbers.json", "OK" if not bad else bad)
sys.exit(1 if bad or n == 0 else 0)
