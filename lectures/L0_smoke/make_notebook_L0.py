"""
make_notebook_L0.py -- smoke test of the notebook maker: lecture notebook and seminar notebook, EN + RO, executed.
Run:  python3 lectures/L0_smoke/make_notebook_L0.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Pipeline smoke test", "Testul de funcționare al fluxului de lucru")
LOAD = ("x = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])\n"
        "px = x[x.ticker == 'SP500'].set_index('date')['adjclose'].dropna()\n"
        "end = (px.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)\n"
        "px = px.loc[:end]\nr = np.log(px).diff().dropna()\nprint(px.index.min().date(), px.index.max().date(), len(r))")


def lecture(nb):
    nb.md(nb.T("## §1 Drawdown of the S&P 500", "## §1 Scăderea indicelui S&P 500 față de maximul anterior"))
    nb.code(LOAD)
    nb.code("dd = px / px.cummax() - 1\n"
            "print('max drawdown %:', round(100 * dd.min(), 2), 'on', dd.idxmin().date())\n"
            "print('annualised volatility %:', round(100 * r.std() * np.sqrt(252), 2))\n"
            "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.plot(dd.index, 100 * dd, lw=0.8, label='S&P 500')\n"
            "ax.set_ylabel('%')\nfig.legend(loc='lower center', ncol=1)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\nplt.show()")


def seminar(nb):
    nb.md(nb.T("## Setup", "## Pregătire"))
    nb.code(LOAD)
    nb.task(nb.T("Compute the annualised volatility of the daily log returns (252 trading days).",
                 "Calculați volatilitatea anualizată a randamentelor logaritmice zilnice (252 de zile de tranzacționare)."))
    nb.solution("vol = r.std(ddof=1) * np.sqrt(252)\nprint(round(100 * vol, 2))")


if __name__ == "__main__":
    Notebook.build_all(0, TITLE, lecture, kind="lecture")
    Notebook.build_all(0, TITLE, seminar, kind="seminar")
