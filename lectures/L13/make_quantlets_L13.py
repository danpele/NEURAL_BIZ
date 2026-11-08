"""
make_quantlets_L13.py -- one Quantlet per chart of Lecture 13 (Quantlets/L13/NN_L13_<chart>/), executed, and every number
printed with show() compared with L13_numbers.json.
Data charts recompute everything from data/clean; model charts recompute every number from the per-hour forecasts that
analysis_L13.py publishes in lectures/L13/results/ (the models themselves need a graphics card and the notebook of the lecture).
Run:  python3 lectures/L13/make_quantlets_L13.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

NUM = os.path.join(HERE, "L13_numbers.json")
RES = "RES = REPO + 'lectures/L13/results/'\n"
PRICES = ("x = pd.read_csv(DATA + 'power_da_prices_hourly_utc.csv', parse_dates=['timestamp_utc']).set_index('timestamp_utc')\n"
          "x.index = x.index.tz_convert('Europe/Brussels')\n"
          "def day_matrix(s):\n"
          "    df = pd.DataFrame({'v': s.values, 'd': s.index.tz_localize(None).normalize(), 'h': s.index.hour}, index=s.index)\n"
          "    return df.groupby(['d', 'h'])['v'].mean().unstack('h').reindex(columns=range(24))\n"
          "RO = day_matrix(x['RO']).loc[:'2026-10-31']\nDE = day_matrix(x['DE_LU']).loc[:'2026-10-31']\n")
FC = (RES + "F = pd.read_csv(RES + 'forecasts_hourly.csv', parse_dates=['date'])\n"
      "MODELS = ['naive', 'lear', 'dnn', 'nbeats', 'chronos2', 'chronos2_cov', 'bolt', 'timesfm', 'ens']\n")
SRC = "energy-charts.info (Fraunhofer ISE), Romanian day-ahead prices"
KW = "electricity price forecasting, day-ahead market, Romania, deep learning, foundation models, Chronos, LEAR"
qs = []

q = Quantlet(13, "price_history", "Daily mean day-ahead electricity prices in Romania and Germany-Luxembourg since 2015, with the means of 2019 and "
             "2022, the highest day and the correlation of daily means. Source: " + SRC + ".", KW, "data/clean/power_da_prices_hourly_utc.csv")
q.md("Daily means of hourly day-ahead prices on the CET calendar (the doubled autumn hour is averaged).")
q.code(PRICES + "dly, dde = RO.mean(axis=1), DE.mean(axis=1)\n"
       "show('facts.mean_2019', 'mean price 2019 (EUR/MWh)', round(float(np.nanmean(RO.loc['2019'].values)), 2))\n"
       "show('facts.mean_2022', 'mean price 2022 (EUR/MWh)', round(float(np.nanmean(RO.loc['2022'].values)), 2))\n"
       "show('facts.max_daily', 'highest daily mean (EUR/MWh)', round(float(dly.max()), 2))\n"
       "show('facts.corr_ro_de_daily', 'correlation RO-DE daily means', round(float(dly.corr(dde)), 3))")
q.code("fig, ax = plt.subplots(figsize=(6.4, 2.4))\nax.plot(dde.index, dde.values, color=PAL['benchmark'], lw=0.6, alpha=0.8, label='Germany-Luxembourg')\n"
       "ax.plot(dly.index, dly.values, color=PAL['data'], lw=0.6, label='Romania')\nax.set_ylabel('Daily mean price (EUR/MWh)')\n"
       "legend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('price_history.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "hourly_profile", "Mean Romanian day-ahead price by delivery hour in 2019, 2022, 2024 and 2026 (January-October): the solar "
             "trough at midday. Source: " + SRC + ".", KW + ", duck curve, solar", "data/clean/power_da_prices_hourly_utc.csv")
q.code(PRICES + "prof = {y: RO.loc[str(y)].mean(axis=0) for y in [2019, 2022, 2024, 2026]}\n"
       "show('facts.profile.2026.min', 'lowest hourly mean 2026 (EUR/MWh)', round(float(prof[2026].min()), 2))\n"
       "show('facts.profile.2026.max', 'highest hourly mean 2026 (EUR/MWh)', round(float(prof[2026].max()), 2))\n"
       "show('facts.profile.2026.min_hour', 'cheapest hour 2026', int(prof[2026].idxmin()))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nfor y, c in zip(prof, [SERIES[0], SERIES[7], SERIES[3], SERIES[1]]):\n"
       "    ax.plot(range(24), prof[y].values, marker='o', ms=2.5, color=c, label=str(y) + ('*' if y == 2026 else ''))\n"
       "ax.set_xlabel('Delivery hour (CET)')\nax.set_ylabel('Mean price (EUR/MWh)')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\n"
       "fig.savefig('hourly_profile.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "mae_models", "Test-year mean absolute error of nine day-ahead forecasts for Romania: naive, LEAR, DNN, N-BEATS, Chronos-2 "
             "(with and without covariates), Chronos-Bolt, TimesFM 2.5 and an ensemble; rMAE relative to the naive forecast.",
             KW, "lectures/L13/results/forecasts_hourly.csv (from analysis_L13.py)")
q.md("Every forecast uses only information available at gate closure (12:00 CET on the day before delivery).")
q.code(FC + "mae = {m: float((F[m] - F['actual']).abs().mean()) for m in MODELS}\n"
       "for m in MODELS:\n    show(f'acc.{m}.mae', f'MAE {m} (EUR/MWh)', round(mae[m], 2))\n"
       "show('acc.lear.rmae', 'rMAE LEAR', round(round(mae['lear'], 2) / round(mae['naive'], 2), 3))\n"
       "order = ['naive', 'lear', 'dnn', 'nbeats', 'bolt', 'timesfm', 'chronos2', 'chronos2_cov', 'ens']\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.6))\nax.barh(range(9), [mae[m] for m in order], color=[PAL['benchmark']] * 2 + [PAL['model']] * 2 + [PAL['alt']] * 4 + ['#6250d6'])\n"
       "ax.set_yticks(range(9))\nax.set_yticklabels(order)\nax.invert_yaxis()\nax.set_xlabel('MAE, test year (EUR/MWh)')\nfig.tight_layout()\n"
       "fig.savefig('mae_models.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "mae_by_hour", "Mean absolute error by delivery hour of the naive, LEAR, DNN, Chronos-2 and ensemble forecasts of Romanian "
             "day-ahead prices in the test year.", KW, "lectures/L13/results/forecasts_hourly.csv")
q.code(FC + "E = {m: (F[m] - F['actual']).abs().groupby(F['hour']).mean() for m in ['naive', 'lear', 'dnn', 'chronos2', 'ens']}\n"
       "show('byhour.lear.worst', 'LEAR worst hourly MAE', round(float(E['lear'].max()), 2))\n"
       "show('byhour.lear.worst_hour', 'LEAR worst hour', int(E['lear'].idxmax()))\n"
       "show('byhour.chronos2.worst', 'Chronos-2 worst hourly MAE', round(float(E['chronos2'].max()), 2))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nfor m, c in zip(E, [PAL['benchmark'], SERIES[5], PAL['model'], PAL['alt'], '#6250d6']):\n"
       "    ax.plot(range(24), E[m].values, marker='o', ms=2.5, color=c, label=m)\nax.set_xlabel('Delivery hour (CET)')\nax.set_ylabel('MAE (EUR/MWh)')\n"
       "legend_bottom(fig, ncol=5)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('mae_by_hour.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "forecast_week", "One week of hourly Romanian day-ahead prices (19-25 October 2026) with the LEAR, DNN and Chronos-2 forecasts.",
             KW, "lectures/L13/results/forecasts_hourly.csv")
q.code(FC + "W = F[(F.date >= '2026-10-19') & (F.date <= '2026-10-25')].reset_index(drop=True)\n"
       "for m in ['lear', 'dnn', 'chronos2']:\n    show(f'week.mae.{m}', f'week MAE {m}', round(float((W[m] - W['actual']).abs().mean()), 2))\n"
       "ts = W.date + pd.to_timedelta(W.hour, unit='h')\nfig, ax = plt.subplots(figsize=(6.4, 2.5))\n"
       "ax.plot(ts, W.actual, color=PAL['data'], lw=1.6, label='Actual price')\nax.plot(ts, W.lear, color=PAL['benchmark'], lw=1, label='LEAR')\n"
       "ax.plot(ts, W.dnn, color=PAL['model'], lw=1, label='DNN')\nax.plot(ts, W.chronos2, color=PAL['alt'], lw=1, ls='--', label='Chronos-2')\n"
       "ax.set_ylabel('EUR/MWh')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('forecast_week.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "dm_pvalues", "Multivariate Diebold-Mariano tests (daily L1 loss differential over 24 hours) between all pairs of nine "
             "day-ahead forecasts for Romania; one-sided p-values, chessboard plot.", KW + ", Diebold-Mariano", "lectures/L13/results/forecasts_hourly.csv")
q.code(FC + "from scipy import stats\nL1 = {m: (F[m] - F['actual']).abs().groupby(F['date']).sum().values for m in MODELS}\n"
       "def dm_p(a, b):\n    d = L1[a] - L1[b]\n    return float(1 - stats.norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))))\n"
       "for a, b in [('lear', 'dnn'), ('lear', 'chronos2'), ('lear', 'ens'), ('chronos2', 'ens')]:\n"
       "    show(f'dm.{a}__{b}', f'p-value: {b} better than {a}', round(dm_p(a, b), 4))\n"
       "P = np.array([[np.nan if a == b else dm_p(a, b) for b in MODELS] for a in MODELS])\n"
       "fig, ax = plt.subplots(figsize=(6.4, 3.0))\nim = ax.imshow(np.clip(P, 0, 0.1), cmap='RdYlGn_r', vmin=0, vmax=0.1)\n"
       "ax.set_xticks(range(9))\nax.set_xticklabels(MODELS, rotation=40, ha='right', fontsize=8)\nax.set_yticks(range(9))\nax.set_yticklabels(MODELS, fontsize=8)\n"
       "fig.colorbar(im, ax=ax, fraction=0.03)\nfig.savefig('dm_pvalues.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "training_curves", "Training and validation L1 loss by epoch of the DNN (first monthly recalibration) and of N-BEATS, with early "
             "stopping at the best validation epoch.", KW + ", early stopping", "lectures/L13/results/training_hist.json")
q.code(RES + "import json, urllib.request\nsrc = RES + 'training_hist.json'\n"
       "Hh = json.load(open(src)) if os.path.exists(src) else json.load(urllib.request.urlopen(src))\n"
       "d, n = np.array(Hh['dnn']), np.array(Hh['nbeats'])\n"
       "show('train.dnn_best_epoch', 'DNN best epoch', int(np.argmin(d[:, 1]) + 1))\nshow('train.nb_best_epoch', 'N-BEATS best epoch', int(np.argmin(n[:, 1]) + 1))\n"
       "fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3))\nfor ax, h, lab in ((axs[0], d, 'Epoch (DNN)'), (axs[1], n, 'Epoch (N-BEATS)')):\n"
       "    ax.plot(h[:, 0], color=PAL['train'], label='Training')\n    ax.plot(h[:, 1], color=PAL['valid'], label='Validation')\n    ax.set_xlabel(lab)\n"
       "axs[0].set_ylabel('L1 loss')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('training_curves.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "context_length", "Test-year mean absolute error of the zero-shot Chronos-2 forecast of Romanian day-ahead prices as a function "
             "of the context length (168 to 8,192 hours), against LEAR.", KW + ", context length", "lectures/L13/results/forecasts_hourly.csv")
q.code(FC + "C = [168, 512, 2048, 8192]\nm = {c: float((F[f'chronos2_ctx{c}'] - F['actual']).abs().mean()) for c in C}\n"
       "for c in C:\n    show(f'ctx.{c}', f'Chronos-2 MAE, context {c} h', round(m[c], 2))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.plot(range(4), [m[c] for c in C], marker='o', color=PAL['alt'], label='Chronos-2')\n"
       "ax.axhline(float((F['lear'] - F['actual']).abs().mean()), color=PAL['benchmark'], ls='--', label='LEAR')\n"
       "ax.set_xticks(range(4))\nax.set_xticklabels([f'{c} h' for c in C])\nax.set_ylabel('MAE (EUR/MWh)')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\n"
       "fig.savefig('context_length.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

CONF = (RES + "Q = pd.read_csv(RES + 'quantiles_hourly.csv', parse_dates=['date'])\n"
        "test = Q[(Q.date >= '2025-11-01') & (Q.date <= '2026-10-31')]\n"
        "def wide(col):\n    return Q.pivot(index='date', columns='hour', values=col)\n"
        "Y = wide('actual')\ndays = [d for d in sorted(test.date.unique()) if Y.loc[d].notna().all()]\n"
        "def conformal(lo_c, hi_c, alpha, cal=60):\n"
        "    lo, hi = wide(lo_c), wide(hi_c)\n    s = np.maximum(lo - Y, Y - hi)\n    out_lo, out_hi = [], []\n"
        "    for D in days:\n        w = s.loc[D - pd.Timedelta(days=cal):D - pd.Timedelta(days=1)].values.ravel()\n        w = w[~np.isnan(w)]\n"
        "        n = len(w)\n        q = np.quantile(w, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))\n"
        "        out_lo.append(lo.loc[D].values - q)\n        out_hi.append(hi.loc[D].values + q)\n    return np.array(out_lo), np.array(out_hi)\n"
        "y = Y.loc[days].values\n"
        "def cov(lo, hi):\n    return round(100 * float(((y >= lo) & (y <= hi)).mean()), 1)\n")
q = Quantlet(13, "coverage", "Empirical coverage of 50, 80 and 90 percent intervals for Romanian day-ahead prices in the test year: raw Chronos-2 "
             "quantiles, the same after a rolling split-conformal correction (60 days), and LEAR with empirical errors.",
             KW + ", conformal prediction, coverage", "lectures/L13/results/quantiles_hourly.csv")
q.code(CONF + "levels = {'50': ('ch2_q25', 'ch2_q75', 0.5), '80': ('ch2_q10', 'ch2_q90', 0.2), '90': ('ch2_q05', 'ch2_q95', 0.1)}\n"
       "raw, con = {}, {}\nfor k, (a, b, al) in levels.items():\n"
       "    raw[k] = cov(wide(a).loc[days].values, wide(b).loc[days].values)\n    con[k] = cov(*conformal(a, b, al))\n"
       "    show(f'cov.chronos2.{k}.cov', f'raw coverage {k} %', raw[k])\n    show(f'cov.chronos2_conf.{k}.cov', f'conformal coverage {k} %', con[k])\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nks = ['50', '80', '90']\n"
       "ax.bar(np.arange(3) - 0.2, [raw[k] for k in ks], 0.4, color=PAL['alt'], label='Chronos-2, raw')\n"
       "ax.bar(np.arange(3) + 0.2, [con[k] for k in ks], 0.4, color=PAL['model'], label='Chronos-2 + conformal')\n"
       "for j, k in enumerate(ks):\n    ax.plot([j - 0.45, j + 0.45], [int(k)] * 2, color='#222222')\n"
       "ax.set_xticks(range(3))\nax.set_xticklabels([f'nominal {k} %' for k in ks])\nax.set_ylabel('Empirical coverage (%)')\nlegend_bottom(fig)\n"
       "fig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('coverage.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "chronos_fan", "Zero-shot Chronos-2 quantile forecasts (50, 80 and 90 percent bands and the median) of Romanian day-ahead "
             "prices for 19-25 October 2026, with the realised prices.", KW + ", probabilistic forecast", "lectures/L13/results/quantiles_hourly.csv")
q.code(RES + "Q = pd.read_csv(RES + 'quantiles_hourly.csv', parse_dates=['date'])\nW = Q[(Q.date >= '2026-10-19') & (Q.date <= '2026-10-25')].reset_index(drop=True)\n"
       "show('week.mae.chronos2', 'week MAE of the Chronos-2 median', round(float((W.ch2_q50 - W.actual).abs().mean()), 2))\n"
       "ts = W.date + pd.to_timedelta(W.hour, unit='h')\nfig, ax = plt.subplots(figsize=(6.4, 2.5))\n"
       "for a, b, al in (('ch2_q05', 'ch2_q95', 0.18), ('ch2_q10', 'ch2_q90', 0.28), ('ch2_q25', 'ch2_q75', 0.4)):\n"
       "    ax.fill_between(ts, W[a], W[b], color=PAL['alt'], alpha=al, lw=0)\n"
       "ax.plot(ts, W.ch2_q50, color=PAL['alt'], label='Median')\nax.plot(ts, W.actual, color=PAL['data'], lw=1.4, label='Actual')\nax.set_ylabel('EUR/MWh')\n"
       "legend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('chronos_fan.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "load_mape", "Mean absolute percentage error of day-ahead forecasts of Romanian hourly load in the test year: same hour last "
             "week, TimesFM 2.5 and Chronos-2, both zero-shot.", "load forecasting, Romania, foundation models, TimesFM, Chronos, MAPE",
             "lectures/L13/results/load_actual.csv, load_forecasts.csv")
q.code(RES + "A = pd.read_csv(RES + 'load_actual.csv', index_col=0)\nLF = pd.read_csv(RES + 'load_forecasts.csv', index_col=[0, 1])\n"
       "mape = {}\nfor m in ['week', 'timesfm', 'chronos2']:\n    P = LF.loc[m].values\n"
       "    mape[m] = round(float(100 * np.abs((P - A.values) / A.values).mean()), 2)\n    show(f'load.{m}.mape', f'MAPE {m} (%)', mape[m])\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.barh(range(3), list(mape.values()), color=[PAL['benchmark'], PAL['alt'], PAL['alt']])\n"
       "ax.set_yticks(range(3))\nax.set_yticklabels(['Same hour last week', 'TimesFM 2.5', 'Chronos-2'])\nax.invert_yaxis()\nax.set_xlabel('MAPE (%)')\n"
       "fig.tight_layout()\nfig.savefig('load_mape.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(13, "var_sp500", "One-day 1 percent Value-at-Risk of the S&P 500, January 2020 to October 2026: GARCH(1,1)-t and zero-shot Chronos-2 "
             "quantiles with a rolling one-sided conformal shift; violations, Kupiec and Christoffersen tests.",
             "Value-at-Risk, backtesting, Kupiec, Christoffersen, GARCH, Chronos, conformal prediction, S&P 500", "lectures/L13/results/var_SP500.csv (from data/clean/markets_daily.csv)")
q.code(RES + "from scipy import stats\nT = pd.read_csv(RES + 'var_SP500.csv', parse_dates=['date']).set_index('date')\n"
       "def kupiec(h, a):\n    n, x = len(h), int(h.sum()); p = x / n\n"
       "    lr = -2 * n * np.log(1 - a) if x == 0 else -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - p) - x * np.log(p))\n"
       "    return float(1 - stats.chi2.cdf(lr, 1))\n"
       "for m in ['hs', 'garch', 'ch', 'chc']:\n    h = (T['r'] < T[f'{m}_01']).values\n"
       "    show(f'var.SP500.{m}_01.viol', f'1 % violations {m}', int(h.sum()))\n    show(f'var.SP500.{m}_01.kupiec_p', f'Kupiec p {m}', round(kupiec(h, 0.01), 3))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.5))\nax.plot(T.index, T.r, color=PAL['data'], lw=0.45, label='Daily log return (%)')\n"
       "ax.plot(T.index, T.garch_01, color=PAL['benchmark'], lw=0.9, label='1 % VaR, GARCH(1,1)-t')\nax.plot(T.index, T.chc_01, color=PAL['model'], lw=0.9, label='1 % VaR, Chronos-2 + conformal')\n"
       "hit = T.r < T.chc_01\nax.scatter(T.index[hit], T.r[hit], s=10, color=PAL['positive'], zorder=3, label='Violation')\nax.set_ylabel('%')\n"
       "legend_bottom(fig, ncol=2)\nfig.tight_layout(rect=(0, 0.15, 1, 1))\nfig.savefig('var_sp500.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

bad_all, n_all = [], 0
for q in qs:
    q.write(execute=True)
    n, bad = check_numbers(q, NUM)
    n_all += n
    bad_all += bad
    print(f"  {q.name}: {n} numbers checked", "OK" if not bad else bad)
print(f"{len(qs)} Quantlets, {n_all} numbers checked against L13_numbers.json:", "OK" if not bad_all else bad_all)
sys.exit(1 if bad_all else 0)
