"""
make_quantlets_L5.py -- one Quantlet per chart of Lecture 5: Quantlets/L5/NN_L5_<chart>/ (Metainfo.txt, executed notebook,
<chart>.png). Data charts are recomputed from data/clean; charts of trained networks read <chart>_results.json, the saved
forecasts of analysis_L5.py (its cache). Every number printed with show() is compared with L5_numbers.json.
Run (after analysis_L5.py):  python3 lectures/L5/make_quantlets_L5.py
"""
import json
import os
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

print("loading the results of analysis_L5.py (from its cache) ...")
A = runpy.run_path(os.path.join(HERE, "analysis_L5.py"), run_name="analysis_L5_results")
NUM = os.path.join(HERE, "L5_numbers.json")
K = 5
QS = []


def loader(chart):
    return ("import json, urllib.request\n"
            f"_F = '{chart}_results.json'\n"
            f"R = json.load(open(_F)) if os.path.isfile(_F) else json.load(urllib.request.urlopen(REPO + 'Quantlets/L{K}/NN_L{K}_{chart}/' + _F))")


ONLY = [a for a in sys.argv[1:] if not a.startswith('-')]


def make(chart, description, keywords, datafile, intro, code, results=None):
    if ONLY and chart not in ONLY:
        return
    if results is not None:
        d = os.path.join(HERE, "..", "..", "Quantlets", f"L{K}", f"NN_L{K}_{chart}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{chart}_results.json"), "w") as f:
            json.dump(results, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else (str(o) if hasattr(o, "isoformat") else float(o)))
        code = loader(chart) + "\n" + code
    q = Quantlet(k=K, chart=chart, description=description, keywords=keywords, datafile=datafile)
    q.md(intro)
    q.code(code)
    q.write(execute=True)
    n, bad = check_numbers(q, NUM)
    print(f"  {q.name}: {n} numbers checked", "OK" if not bad else bad)
    QS.append((q.name, n, bad))


SAVE = "fig.savefig('{c}.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()"
SPX = """import matplotlib.dates as mdates
x = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])
d = x[x.ticker == 'SP500'].set_index('date')[['open', 'high', 'low', 'close', 'adjclose']].dropna()
END = (d.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)
d = d.loc[:END]; d = d[d.high > d.low]
gk = (0.5 * np.log(d.high / d.low) ** 2 - (2 * np.log(2) - 1) * np.log(d.close / d.open) ** 2).clip(lower=1e-8)
d['gk'] = gk; d['r'] = 100 * np.log(d.adjclose).diff(); d = d.dropna()"""
MK = "MSPX = 'S&P 500'"
L_ = lambda o: np.asarray(o).tolist()
dates = lambda ix: [str(t)[:10] for t in ix]

make("sp500_vol", "S&P 500 daily log returns and annualised Garman-Klass range volatility, from daily open, high, low and close.",
     "volatility clustering, Garman-Klass estimator, S&P 500, returns, range volatility",
     "data/clean/markets_daily.csv (Yahoo Finance, ticker SP500)",
     "$r_t=100\\ln(P_t/P_{t-1})$ and $\\hat\\sigma^2_{GK,t}=\\frac12\\ln^2(H_t/L_t)-(2\\ln2-1)\\ln^2(C_t/O_t)$; data cut at the last complete month.",
     SPX + "\nann = np.sqrt(252 * d.gk) * 100\nshow('data.n_days', 'trading days', len(d)); show('data.gk_ann_median', 'median annualised volatility (%)', round(float(ann.median()), 2))\n"
     "fig, axes = plt.subplots(2, 1, figsize=(6.4, 2.9), sharex=True)\naxes[0].plot(d.index, d.r, color=PAL['data'], lw=0.4, label='Daily log return (%)')\n"
     "axes[1].plot(d.index, ann, color=PAL['model'], lw=0.4, label='Garman-Klass volatility, annualised (%)'); axes[1].set_yscale('log')\n"
     "for ax in axes: ax.set_ylabel('%'); ax.set_xlim(d.index.min(), d.index.max())\naxes[1].xaxis.set_major_locator(mdates.YearLocator(2)); axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
     "legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.1, 1, 1))\n" + SAVE.format(c="sp500_vol"))

make("acf_abs", "Autocorrelation of S&P 500 daily returns and absolute returns, lags 1-50, with the 1.96/sqrt(T) band.",
     "autocorrelation, volatility clustering, long memory, absolute returns, S&P 500",
     "data/clean/markets_daily.csv (Yahoo Finance, ticker SP500)",
     "The memory of returns is in their size, not in their sign.",
     SPX + "\nfrom statsmodels.tsa.stattools import acf\nar, aa = acf(d.r, nlags=50, fft=True)[1:], acf(d.r.abs(), nlags=50, fft=True)[1:]; band = 1.96 / np.sqrt(len(d))\n"
     "show('acf.r1', 'rho_r(1)', round(float(ar[0]), 3)); show('acf.a1', 'rho_|r|(1)', round(float(aa[0]), 3)); show('acf.a50', 'rho_|r|(50)', round(float(aa[-1]), 3))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.3)); k = np.arange(1, 51)\nax.vlines(k - .15, 0, ar, color=PAL['data'], lw=1.6, label='returns $r_t$'); ax.vlines(k + .15, 0, aa, color=PAL['model'], lw=1.6, label='absolute returns $|r_t|$')\n"
     "ax.axhspan(-band, band, color='#8a8a8a', alpha=.25, lw=0, label='±1.96/√T band'); ax.axhline(0, color='#222222', lw=.6)\n"
     "ax.set_xlabel('Lag (trading days)'); ax.set_ylabel('Autocorrelation'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="acf_abs"))

te = A["s_te"]
VOL = A["VOL"]
res_vol = {"date": dates(A["tdate"][te]), "y": L_(A["y_true"][te]), "naive": L_(A["naive"][te]), "har": L_(A["har"][te]), "garch": L_(A["garch"][te]),
           **{c: L_(VOL[c]["pred"][:, te]) for c in ["rnn", "lstm", "gru"]}}
make("vol_forecast", "One-day-ahead forecasts of S&P 500 annualised volatility in 2020: HAR and LSTM (mean of 5 seeds) against the observed Garman-Klass volatility.",
     "volatility forecasting, LSTM, HAR, GARCH, COVID-19 crash, S&P 500",
     "data/clean/markets_daily.csv; saved test forecasts of analysis_L5.py (vol_forecast_results.json)",
     "Forecasts of $y_{t+1}=\\ln\\hat\\sigma^2_{GK,t+1}$ on the test period; the crash window 20 Feb - 30 Apr 2020. Retrain with lectures/L5/analysis_L5.py.",
     "import matplotlib.dates as mdates\nt = pd.to_datetime(R['date']); y = np.array(R['y']); lstm = np.array(R['lstm']).mean(0)\n"
     "cr = (t >= '2020-02-20') & (t <= '2020-04-30'); mse = lambda p: round(float(np.mean((np.asarray(p)[cr] - y[cr]) ** 2)), 3)\n"
     "show('vol.crash.har', 'crash MSE HAR', mse(R['har'])); show('vol.crash.lstm', 'crash MSE LSTM', mse(lstm)); show('vol.crash.garch', 'crash MSE GARCH', mse(R['garch'])); show('vol.crash.naive', 'crash MSE naive', mse(R['naive']))\n"
     "v = lambda a: np.sqrt(252 * np.exp(np.asarray(a))) * 100; m = t.year == 2020\nfig, ax = plt.subplots(figsize=(6.4, 2.4))\n"
     "ax.plot(t[m], v(y)[m], color=PAL['data'], lw=.8, alpha=.8, label='Observed (Garman-Klass)'); ax.plot(t[m], v(R['har'])[m], color=PAL['benchmark'], lw=1.4, label='HAR')\n"
     "ax.plot(t[m], v(lstm)[m], color=PAL['model'], lw=1.4, label='LSTM (mean of 5 seeds)'); ax.set_ylabel('Annualised volatility (%)')\n"
     "ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\\n%Y'))\nlegend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="vol_forecast"),
     res_vol)

make("vol_mse", "Test MSE of the log Garman-Klass variance of the S&P 500, 2020-2026: naive, GARCH(1,1), HAR and RNN, LSTM, GRU (5 seeds each), with Diebold-Mariano tests.",
     "volatility forecasting, RNN, LSTM, GRU, HAR, GARCH, Diebold-Mariano test, seeds",
     "data/clean/markets_daily.csv; saved test forecasts of analysis_L5.py (vol_mse_results.json)",
     "Benchmarks first; networks with 5 seeds; Diebold-Mariano test with Newey-West variance.",
     "from scipy.stats import norm\ny = np.array(R['y']); mse = lambda p: float(np.mean((np.asarray(p) - y) ** 2))\n"
     "def dm(e1, e2):\n    dd = e1 - e2; n = len(dd); dc = dd - dd.mean(); L = int(np.floor(4 * (n / 100) ** (2 / 9))); v = dc @ dc / n\n"
     "    for k in range(1, L + 1): v += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / n\n    s = dd.mean() / np.sqrt(v / n); return s, 2 * (1 - norm.cdf(abs(s)))\n"
     "for k in ['naive', 'garch', 'har']: show(f'vol.{k}.mse', f'{k} MSE', round(mse(R[k]), 4))\n"
     "for c in ['rnn', 'lstm', 'gru']: show(f'vol.{c}.mse_mean', f'{c} MSE, mean over seeds', round(float(np.mean([mse(p) for p in R[c]])), 4))\n"
     "ens = np.array(R['lstm']).mean(0); s, p = dm((ens - y) ** 2, (np.array(R['har']) - y) ** 2)\nshow('vol.dm_lstm_har.stat', 'DM LSTM vs HAR', round(float(s), 2)); show('vol.dm_lstm_har.p', 'p-value', round(float(p), 3))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.3)); names = ['Naive', 'GARCH', 'HAR', 'RNN', 'LSTM', 'GRU']\nvals = [mse(R[k]) for k in ['naive', 'garch', 'har']]\n"
     "for i, v_ in enumerate(vals): ax.bar(i, v_, color=PAL['benchmark'], width=.6)\n"
     "for i, c in enumerate(['rnn', 'lstm', 'gru']):\n    ms = [mse(p) for p in R[c]]; ax.bar(3 + i, np.mean(ms), color=PAL['model'], width=.6, alpha=.55)\n"
     "    ax.scatter(np.full(len(ms), 3 + i) + np.linspace(-.15, .15, len(ms)), ms, color='#222222', s=8, zorder=3)\n"
     "ax.set_xticks(range(6)); ax.set_xticklabels(names); ax.set_ylabel('Test MSE of log variance'); ax.set_ylim(0, max(vals) * 1.15)\n"
     "ax.bar([], [], color=PAL['benchmark'], label='benchmarks'); ax.bar([], [], color=PAL['model'], alpha=.55, label='recurrent networks (mean)'); ax.scatter([], [], color='#222222', s=8, label='one seed')\n"
     "legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="vol_mse"),
     res_vol)

make("lstm_gates", "Mean forget-gate and input-gate activations of the trained volatility LSTM (seed 20261108) over the S&P 500 test period 2020-2026.",
     "LSTM gates, forget gate, input gate, interpretability, volatility, S&P 500",
     "saved gate activations of analysis_L5.py (lstm_gates_results.json)",
     "Gate activations at the last step of each 22-day window, 5-day means.",
     "import matplotlib.dates as mdates\nt = pd.to_datetime(R['date']); f = np.array(R['f']); i = np.array(R['i'])\n"
     "show('gates.f_mean', 'mean forget gate', round(float(f.mean()), 3)); show('gates.i_mean', 'mean input gate', round(float(i.mean()), 3))\n"
     "show('gates.f_corr_y', 'corr(forget gate, log variance today)', round(float(np.corrcoef(f, R['ylag'])[0, 1]), 3))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.3)); ax.plot(t, pd.Series(f).rolling(5).mean(), color=PAL['model'], lw=1, label='forget gate $f_t$ (mean of 32 units)')\n"
     "ax.plot(t, pd.Series(i).rolling(5).mean(), color='#6250d6', lw=1, label='input gate $i_t$'); ax.set_ylabel('Gate activation (5-day mean)', fontsize=9.5)\n"
     "ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="lstm_gates"),
     {"date": dates(A["tdate"][te]), "f": L_(A["fg"]), "i": L_(A["ig"]), "ylag": L_(A["y"][A["T_"]][te])})

RT = A["RT"]
make("returns_r2", "Out-of-sample R^2 of next-day returns, 2020-2026: training mean, AR(1) and LSTM (5 seeds), for the S&P 500 and Banca Transilvania (TLV).",
     "return predictability, out-of-sample R2, LSTM, AR(1), efficient markets, S&P 500, Bucharest Stock Exchange",
     "data/clean/markets_daily.csv and bvb_equities_daily.csv; saved test forecasts of analysis_L5.py (returns_r2_results.json)",
     "$R^2_{OOS}=1-\\sum(r-\\hat r)^2/\\sum r^2$ against the zero forecast.",
     "def r2(y, p): y, p = np.asarray(y), np.asarray(p); return 100 * (1 - np.sum((y - p) ** 2) / np.sum(y ** 2))\n"
     "for tk in ['SP500', 'TLV']:\n    o = R[tk]; show(f'ret.{tk}.r2_ar', f'{tk} AR(1) R2 (%)', round(r2(o['y'], o['ar']), 3)); show(f'ret.{tk}.r2_lstm_mean', f'{tk} LSTM R2 (%), mean of seeds', round(float(np.mean([r2(o['y'], p) for p in o['lstm']])), 3))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.3)); xs = np.arange(3)\nfor j, (tk, c, l) in enumerate([('SP500', PAL['data'], 'S&P 500'), ('TLV', PAL['alt'], 'Banca Transilvania (TLV)')]):\n"
     "    o = R[tk]; v = [r2(o['y'], o['mean']), r2(o['y'], o['ar']), np.mean([r2(o['y'], p) for p in o['lstm']])]\n"
     "    ax.bar(xs + (j - .5) * .36, v, width=.36, color=c, label=l); ax.scatter(np.full(len(o['lstm']), 2 + (j - .5) * .36), [r2(o['y'], p) for p in o['lstm']], color='#222222', s=8, zorder=3)\n"
     "ax.axhline(0, color='#222222', lw=.8); ax.set_xticks(xs); ax.set_xticklabels(['Training mean', 'AR(1)', 'LSTM']); ax.set_ylabel('Out-of-sample $R^2$ (%)')\n"
     "ax.scatter([], [], color='#222222', s=8, label='LSTM, one seed'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="returns_r2"),
     {tk: (lambda o: {"y": L_(o["y"][o["mte"]]), "lstm": L_(o["pred"][:, o["mte"]]),
                      "ar": L_(np.polyval(np.polyfit(o["lag"][o["mtr"]], o["y"][o["mtr"]], 1), o["lag"])[o["mte"]]),
                      "mean": L_(np.full(int(o["mte"].sum()), o["y"][o["mtr"]].mean()))})(RT[tk]) for tk in ["SP500", "TLV"]})

make("load_series", "Daily mean Romanian electricity load and its 365-day mean, from hourly ENTSO-E data.",
     "electricity load, Romania, seasonality, ENTSO-E, energy-charts",
     "data/clean/ro_load_generation_hourly_utc.csv (energy-charts.info, ENTSO-E)",
     "Romanian hourly load, daily means, 2016 to the last complete month.",
     "import matplotlib.dates as mdates\nld = pd.read_csv(DATA + 'ro_load_generation_hourly_utc.csv', parse_dates=['timestamp_utc']).set_index('timestamp_utc')['load']\n"
     f"ld = ld[(ld.index >= '2016-01-01') & (ld.index < '{A['END'] + np.timedelta64(1, 'D')}'.split(' ')[0] + ' 00:00+00:00')]\n"
     "show('load.mean_mw', 'mean load (MW)', int(round(float(ld.mean()))))\ns = ld.resample('D').mean()\nfig, ax = plt.subplots(figsize=(6.4, 2.3))\n"
     "ax.plot(s.index, s / 1000, color=PAL['data'], lw=.6, label='daily mean load'); ax.plot(s.index, s.rolling(365, center=True).mean() / 1000, color=PAL['model'], lw=1.6, label='365-day mean')\n"
     "ax.set_ylabel('GW'); ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="load_series"))

meta = A["meta"]
LD = A["LD"]
mte = meta["mte"]
res_load = {"date": dates(meta["D"][mte]), "y": L_(meta["Y"][mte]), "nv": L_(meta["NV"][mte]), "yd": L_(meta["YD"][mte]),
            "ridge": L_(LD["main"]["ridge"]["pred"][mte]), **{c: L_(LD["main"][c]["pred"].mean(0)[mte]) for c in ["rnn", "lstm", "gru"]}}
make("load_errors", "Day-ahead Romanian hourly load, test days 2024-2026: MAPE of seasonal naive, naive, ridge ARX, RNN, LSTM and GRU (mean of 3 seeds), and MAE by hour.",
     "load forecasting, LSTM, GRU, RNN, ridge regression, MAPE, Diebold-Mariano, Romania, electricity",
     "data/clean/ro_load_generation_hourly_utc.csv; saved test forecasts of analysis_L5.py (load_errors_results.json)",
     "24 hourly values of day d+1 forecast at the end of day d (UTC days).",
     "Y = np.array(R['y']); mape = lambda p: round(float(np.mean(np.abs(np.asarray(p) - Y) / Y) * 100), 2)\n"
     "show('load.snaive', 'MAPE seasonal naive (%)', mape(R['nv'])); show('load.ridge', 'MAPE ridge ARX (%)', mape(R['ridge']))\n"
     "for c in ['rnn', 'lstm', 'gru']: show(f'load.{c}.ens', f'MAPE {c} (%)', mape(R[c]))\n"
     "show('load.mae_lstm_mw', 'MAE LSTM (MW)', int(round(float(np.mean(np.abs(np.array(R['lstm']) - Y))))))\n"
     "fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={'width_ratios': [1.15, 1]})\nks = ['nv', 'yd', 'ridge', 'rnn', 'lstm', 'gru']\n"
     "axes[0].bar(range(6), [mape(R[k]) for k in ks], color=[PAL['benchmark']] * 3 + [PAL['model']] * 3, width=.65)\n"
     "axes[0].set_xticks(range(6)); axes[0].set_xticklabels(['Naive\\n(week)', 'Naive\\n(day)', 'Ridge\\nARX', 'RNN', 'LSTM', 'GRU'], fontsize=7.5); axes[0].set_ylabel('Test MAPE (%)')\n"
     "for k, c, l, ls in [('nv', PAL['benchmark'], 'seasonal naive', '--'), ('ridge', PAL['alt'], 'ridge ARX', '-'), ('lstm', PAL['model'], 'LSTM', '-')]:\n"
     "    axes[1].plot(range(24), np.mean(np.abs(np.array(R[k]) - Y), 0), color=c, ls=ls, lw=1.5, label=l)\n"
     "axes[1].set_xlabel('Hour of day d+1 (UTC)', fontsize=9); axes[1].set_ylabel('MAE (MW)'); axes[1].set_xticks([0, 6, 12, 18, 23])\nlegend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="load_errors"),
     res_load)

sel = A["sel"]
make("load_fortnight", "Two winter test weeks of Romanian hourly load: observed, seasonal naive and LSTM day-ahead forecasts.",
     "load forecasting, LSTM, seasonal naive, Romania, electricity, day-ahead",
     "saved test forecasts of analysis_L5.py (load_fortnight_results.json)",
     "Fourteen consecutive test days, hourly.",
     "import matplotlib.dates as mdates\nt = pd.to_datetime(R['t']); show('load.n_test', 'test days of the whole test period', R['n_test'])\nfig, ax = plt.subplots(figsize=(6.4, 2.3))\n"
     "ax.plot(t, np.array(R['y']) / 1000, color=PAL['data'], lw=1.6, label='observed'); ax.plot(t, np.array(R['nv']) / 1000, color=PAL['benchmark'], lw=1, ls='--', label='seasonal naive (a week earlier)')\n"
     "ax.plot(t, np.array(R['lstm']) / 1000, color=PAL['model'], lw=1.2, label='LSTM'); ax.set_ylabel('GW')\n"
     "ax.xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\\n%Y')); ax.tick_params(axis='x', labelsize=8.5)\n"
     "legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="load_fortnight"),
     {"t": [str(meta["D"][i] + np.timedelta64(h, "h"))[:16] for i in sel for h in range(24)],
      "y": L_(np.concatenate([meta["Y"][i] for i in sel])), "nv": L_(np.concatenate([meta["NV"][i] for i in sel])),
      "lstm": L_(np.concatenate([LD["main"]["lstm"]["pred"].mean(0)[i] for i in sel])), "n_test": int(mte.sum())})

make("load_window", "Test MAPE of the day-ahead LSTM for Romanian load as a function of the input window (24, 72, 168, 336 hours), 2 seeds each, against ridge ARX.",
     "window length, LSTM, memory, load forecasting, Romania",
     "saved test MAPE of analysis_L5.py (load_window_results.json)",
     "How much history does the LSTM need?",
     "for w in ['24', '72', '168', '336']: show(f'load.window.{w}', f'MAPE, window {w} hours (%)', round(float(np.mean(R['win'][w])), 2))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.1)); ws = ['24', '72', '168', '336']\nfor i, w in enumerate(ws):\n"
     "    ax.bar(i, np.mean(R['win'][w]), color=PAL['model'], width=.55, alpha=.7); ax.scatter([i - .08, i + .08], R['win'][w], color='#222222', s=10, zorder=3)\n"
     "ax.axhline(R['ridge'], color=PAL['alt'], ls='--', lw=1.3, label='ridge ARX (168 hours)'); ax.set_xticks(range(4)); ax.set_xticklabels([w + ' hours' for w in ws])\n"
     "ax.set_ylabel('Test MAPE (%)'); ax.scatter([], [], color='#222222', s=10, label='LSTM, one seed'); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="load_window"),
     {"win": {str(w): v for w, v in LD["window"].items()}, "ridge": A["N"]["load"]["ridge"]})

make("grad_lag", "Sensitivity of the trained day-ahead load forecasts (hour 12 UTC) to each of the 168 input hours: RNN vs LSTM vs GRU, 200 test days.",
     "vanishing gradient, memory, sensitivity, RNN, LSTM, GRU, backpropagation through time",
     "saved gradient sensitivities of analysis_L5.py (grad_lag_results.json)",
     "$S(k)=\\frac1n\\sum_i|\\partial\\hat y_{i,12}/\\partial x_{i,T+1-k}|$, normalised by its maximum.",
     "for c in ['rnn', 'lstm', 'gru']:\n    s = np.array(R[c]); show(f'grad.{c}.lag168_rel', f'{c}: lag 168 relative to the last day maximum', round(float(s[167] / s[:24].max()), 3))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nfor c, col in [('rnn', PAL['benchmark']), ('lstm', PAL['model']), ('gru', '#6250d6')]: ax.plot(np.arange(1, 169), R[c], color=col, lw=1.2, label=c.upper())\n"
     "ax.set_yscale('log'); ax.set_xlabel('Lag of the input hour (hours before the forecast origin)', fontsize=9.5); ax.set_ylabel('|d forecast / d input|\\n(relative)', fontsize=9)\n"
     "ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168]); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="grad_lag"),
     {c: L_(A["sens"][c]) for c in ["rnn", "lstm", "gru"]})

bad = [q for q in QS if q[2] or q[1] == 0]
print(f"{len(QS)} Quantlets written;", "all numbers match" if not bad else f"problems: {bad}")
sys.exit(1 if bad else 0)
