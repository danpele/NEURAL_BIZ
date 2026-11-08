"""
make_notebook_L13.py -- Lecture 13 notebooks (EN + RO): lecture notebook (sections = the 'Experiment yourself' pointers of the
deck) and seminar notebook (student + solutions), all executed.
Heavy runs (a year of daily LEAR re-estimation, the DNN with 4 seeds x 12 months, foundation models on 425 days) are done by
analysis_L13.py; the notebook runs every model on a short window and reads the full-year forecasts from lectures/L13/results/.
Run:  python3 lectures/L13/make_notebook_L13.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Deep learning for energy and financial time series; time-series foundation models",
         "Deep learning pentru serii de timp din energie și finanțe; modele fundație pentru serii de timp")

LOAD = ("RES = '../lectures/L13/results/'\n"
        "x = pd.read_csv(DATA + 'power_da_prices_hourly_utc.csv', parse_dates=['timestamp_utc']).set_index('timestamp_utc')\n"
        "x.index = x.index.tz_convert('Europe/Brussels')\n"
        "def day_matrix(s):\n"
        "    df = pd.DataFrame({'v': s.values, 'd': s.index.tz_localize(None).normalize(), 'h': s.index.hour}, index=s.index)\n"
        "    return df.groupby(['d', 'h'])['v'].mean().unstack('h').reindex(columns=range(24))\n"
        "RO = day_matrix(x['RO']).loc[:'2026-10-31']\nDE = day_matrix(x['DE_LU']).loc[:'2026-10-31']\n"
        "F = pd.read_csv(RES + 'forecasts_hourly.csv', parse_dates=['date'])\n"
        "print(RO.shape, F.date.min().date(), F.date.max().date())")
INSTALL = ("try:\n    import chronos\nexcept ImportError:      # Google Colab\n"
           "    import subprocess\n    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'chronos-forecasting'], check=True)\n"
           "import sys, torch\nfrom chronos import BaseChronosPipeline")


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 Data, information set and the naive benchmark", "## §1 Datele, setul de informații și modelul naiv"))
    nb.code("import sys\n" + LOAD)
    nb.code("Y = F.pivot(index='date', columns='hour', values='actual')\n"
            "naive = F.pivot(index='date', columns='hour', values='naive')\n"
            "print('test days:', len(Y), ' mean price:', round(float(Y.values.mean()), 2))\n"
            "print('naive MAE (EUR/MWh):', round(float(np.abs(naive - Y).values.mean()), 2))")
    nb.md(T("§ *Experiment yourself*: replace the naive rule by `RO.shift(7)` for all days and compare.",
            "§ *Exercițiu practic*: înlocuiți regula naivă cu `RO.shift(7)` pentru toate zilele și comparați."))
    nb.md(T("## §2 LEAR: Lasso on asinh-transformed lags (one day re-estimated here)", "## §2 LEAR: Lasso pe decalaje transformate asinh (aici, reestimat pentru o singură zi)"))
    nb.code("from sklearn.linear_model import LassoLarsIC\n"
            "def features(D):\n"
            "    parts = [RO.loc[D - pd.Timedelta(days=l)].values for l in (1, 2, 3, 7)] + [DE.loc[D - pd.Timedelta(days=1)].values]\n"
            "    dow = np.zeros(7); dow[D.dayofweek] = 1\n    return np.concatenate(parts + [dow])\n"
            "D0 = pd.Timestamp('2026-10-20')\nwin = [d for d in pd.date_range(D0 - pd.Timedelta(days=1092), D0 - pd.Timedelta(days=1))\n"
            "       if d in RO.index and RO.loc[d].notna().all() and all((d - pd.Timedelta(days=l)) in RO.index for l in (1, 2, 3, 7))]\n"
            "Xw = np.array([features(d) for d in win]); Yw = RO.loc[win].values\nok = ~np.isnan(Xw).any(1)\nXw, Yw = Xw[ok], Yw[ok]\n"
            "med = np.median(Yw); mad = np.median(np.abs(Yw - med)) / 0.6745\nv = lambda a: np.arcsinh((a - med) / mad)\n"
            "A = Xw.copy(); A[:, :120] = v(A[:, :120]); b = features(D0); b[:120] = v(b[:120])\n"
            "mu, sd = A.mean(0), A.std(0) + 1e-9\npred = []\nfor h in range(24):\n"
            "    m = LassoLarsIC(criterion='aic').fit((A - mu) / sd, v(Yw[:, h]))\n    pred.append(np.sinh(m.predict(((b - mu) / sd)[None])[0]) * mad + med)\n"
            "print('MAE on', D0.date(), round(float(np.abs(np.array(pred) - RO.loc[D0].values).mean()), 2))\n"
            "print('full-year LEAR MAE:', round(float((F.lear - F.actual).abs().mean()), 2))")
    nb.md(T("## §3 The DNN and N-BEATS: training curves and seeds", "## §3 DNN și N-BEATS: curbele de antrenare și rulările repetate"))
    nb.code("import json\nH = json.load(open(RES + 'training_hist.json'))\nfig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3))\n"
            "for ax, k in zip(axs, ('dnn', 'nbeats')):\n    h = np.array(H[k]); ax.plot(h[:, 0], label='train'); ax.plot(h[:, 1], label='validation'); ax.set_xlabel(k)\n"
            "fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=2)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\nplt.show()\n"
            "for k in ('dnn', 'nbeats'):\n    print(k, 'test MAE:', round(float((F[k] - F.actual).abs().mean()), 2))")
    nb.md(T("§ *Experiment yourself*: in `analysis_L13.py`, change `HID` or the recalibration frequency of the DNN and rerun.",
            "§ *Exercițiu practic*: în `analysis_L13.py`, modificați `HID` sau frecvența de reantrenare a DNN și rulați din nou."))
    nb.md(T("## §4 Chronos-2 zero-shot: one week, then the full test year", "## §4 Chronos-2 fără reantrenare: o săptămînă, apoi întregul an de test"))
    nb.code(INSTALL + "\npipe = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map=DEVICE)\n"
            "s = x['RO'].loc[:'2026-10-31 23:00']\nweek = pd.date_range('2026-10-19', '2026-10-25')\nctx = [torch.tensor(s.loc[:str((d - pd.Timedelta(days=1)).date()) + ' 23:00'].values[-2048:], dtype=torch.float32) for d in week]\n"
            "q, _ = pipe.predict_quantiles(ctx, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])\n"
            "q = torch.stack([a.squeeze(0) for a in q]) if isinstance(q, list) else q\n"
            "med = q[:, :, 1].cpu().numpy()\nprint('Chronos-2 MAE, week of 19 Oct 2026:', round(float(np.abs(med - RO.loc[week].values).mean()), 2))\n"
            "mae = {m: round(float((F[m] - F.actual).abs().mean()), 2) for m in ['naive', 'lear', 'dnn', 'nbeats', 'bolt', 'timesfm', 'chronos2', 'chronos2_cov', 'ens']}\nprint(mae)")
    nb.md(T("## §5 Context length and covariates", "## §5 Lungimea contextului și covariabilele"))
    nb.code("print({c: round(float((F[f'chronos2_ctx{c}'] - F.actual).abs().mean()), 2) for c in (168, 512, 2048, 8192)})\n"
            "print('with covariates:', round(float((F.chronos2_cov - F.actual).abs().mean()), 2))")
    nb.md(T("## §6 Diebold–Mariano test on daily losses", "## §6 Testul Diebold–Mariano pe pierderile zilnice"))
    nb.code("from scipy import stats\nL1 = {m: (F[m] - F.actual).abs().groupby(F.date).sum().values for m in ['lear', 'dnn', 'chronos2', 'ens']}\n"
            "def dm(a, b):\n    d = L1[a] - L1[b]\n    return round(float(1 - stats.norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))), 4)\n"
            "print('p(Chronos-2 better than LEAR) =', dm('lear', 'chronos2'))\nprint('p(DNN better than LEAR) =', dm('lear', 'dnn'))")
    nb.md(T("## §7 Quantiles, pinball loss and split-conformal calibration", "## §7 Cuantile, pierderea pinball și calibrarea conformală"))
    nb.code("Q = pd.read_csv(RES + 'quantiles_hourly.csv', parse_dates=['date'])\nW = lambda c: Q.pivot(index='date', columns='hour', values=c)\nYq = W('actual')\n"
            "days = [d for d in Yq.index if d >= pd.Timestamp('2025-11-01') and Yq.loc[d].notna().all()]\n"
            "lo, hi = W('ch2_q10'), W('ch2_q90')\nscore = np.maximum(lo - Yq, Yq - hi)\nL_, H_ = [], []\nfor D in days:\n"
            "    w = score.loc[D - pd.Timedelta(days=60):D - pd.Timedelta(days=1)].values.ravel(); w = w[~np.isnan(w)]\n"
            "    qc = np.quantile(w, min(1, np.ceil((len(w) + 1) * 0.8) / len(w)))\n    L_.append(lo.loc[D] - qc); H_.append(hi.loc[D] + qc)\n"
            "y = Yq.loc[days].values\nprint('raw 80 % coverage:', round(100 * float(((y >= lo.loc[days].values) & (y <= hi.loc[days].values)).mean()), 1))\n"
            "print('conformal 80 % coverage:', round(100 * float(((y >= np.array(L_)) & (y <= np.array(H_))).mean()), 1))")
    nb.md(T("## §8 Romanian load: foundation models against the seasonal naive", "## §8 Consumul României: modele fundație față de modelul naiv sezonier"))
    nb.code("A = pd.read_csv(RES + 'load_actual.csv', index_col=0)\nLF = pd.read_csv(RES + 'load_forecasts.csv', index_col=[0, 1])\n"
            "for m in ['week', 'timesfm', 'chronos2']:\n    print(m, 'MAPE %:', round(float(100 * np.abs((LF.loc[m].values - A.values) / A.values).mean()), 2))")
    nb.md(T("## §9 Value-at-Risk with Chronos-2 and a conformal shift", "## §9 Valoarea expusă la risc cu Chronos-2 și o deplasare conformală"))
    nb.code("V = pd.read_csv(RES + 'var_SP500.csv', parse_dates=['date']).set_index('date')\n"
            "def kupiec(h, a):\n    n, k = len(h), int(h.sum()); p = k / n\n"
            "    lr = -2 * ((n - k) * np.log(1 - a) + k * np.log(a) - (n - k) * np.log(1 - p) - k * np.log(p)) if k else -2 * n * np.log(1 - a)\n"
            "    return round(float(1 - stats.chi2.cdf(lr, 1)), 3)\n"
            "for m in ['hs', 'garch', 'ch', 'chc']:\n    h = (V.r < V[m + '_01']).values\n    print(m, 'violations:', int(h.sum()), 'expected:', round(0.01 * len(h), 1), 'Kupiec p:', kupiec(h, 0.01))")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code("import sys\n" + LOAD)
    nb.task(T("Compute the MAE of the seasonal naive forecast (same hour, 7 days earlier) on the test days of `F` and compare it with the naive rule in `F.naive`.",
              "Calculați MAE a prognozei naive sezoniere (aceeași oră, cu 7 zile în urmă) pe zilele de test din `F` și comparați-o cu regula naivă din `F.naive`."))
    nb.solution("Y = F.pivot(index='date', columns='hour', values='actual')\nS7 = RO.shift(7).loc[Y.index]\n"
                "print(round(float(np.abs(S7.values - Y.values).mean()), 2), round(float((F.naive - F.actual).abs().mean()), 2))")
    nb.task(T("Compute the rMAE of every model in `F` relative to `F.naive`. Which models have rMAE < 0.8?",
              "Calculați rMAE pentru fiecare model din `F`, relativ la `F.naive`. Care modele au rMAE < 0,8?"))
    nb.solution("base = (F.naive - F.actual).abs().mean()\nfor m in ['lear', 'dnn', 'nbeats', 'bolt', 'timesfm', 'chronos2', 'chronos2_cov', 'ens']:\n"
                "    r = (F[m] - F.actual).abs().mean() / base\n    print(m, round(float(r), 3), '<0.8' if r < 0.8 else '')")
    nb.task(T("Run the multivariate Diebold–Mariano test between Chronos-2 with and without covariates. Is the difference significant at 5 %?",
              "Aplicați testul Diebold–Mariano multivariat între Chronos-2 cu și fără covariabile. Este diferența semnificativă la 5 %?"))
    nb.solution("from scipy import stats\nd = ((F.chronos2 - F.actual).abs().groupby(F.date).sum() - (F.chronos2_cov - F.actual).abs().groupby(F.date).sum()).values\n"
                "dm = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))\nprint('DM =', round(float(dm), 3), ' one-sided p =', round(float(1 - stats.norm.cdf(dm)), 4))")
    nb.task(T("Using `quantiles_hourly.csv`, compute the empirical coverage of the raw 90 % Chronos-2 interval (q05--q95) in the test year, by hour of the day. Which hours are worst covered?",
              "Folosind `quantiles_hourly.csv`, calculați acoperirea empirică a intervalului brut de 90 % al Chronos-2 (q05--q95) în anul de test, pe ore. Care ore sînt cel mai slab acoperite?"))
    nb.solution("Q = pd.read_csv(RES + 'quantiles_hourly.csv', parse_dates=['date'])\nQt = Q[(Q.date >= '2025-11-01')].dropna(subset=['actual'])\n"
                "cov = ((Qt.actual >= Qt.ch2_q05) & (Qt.actual <= Qt.ch2_q95)).groupby(Qt.hour).mean().mul(100).round(1)\nprint(cov.sort_values().head(5))")
    nb.task(T("Forecast the Bulgarian (BG) day-ahead prices of 19--25 October 2026 with Chronos-2 zero-shot (context 2,048 hours) and report the MAE against the naive forecast `BG(d-1)`.",
              "Prognozați prețurile pentru ziua următoare din Bulgaria (BG) pentru 19--25 octombrie 2026 cu Chronos-2 fără reantrenare (context de 2.048 de ore) și raportați MAE față de prognoza naivă `BG(d-1)`."))
    nb.solution(INSTALL + "\npipe = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map=DEVICE)\nBG = day_matrix(x['BG'])\ns = x['BG']\n"
                "week = pd.date_range('2026-10-19', '2026-10-25')\nctx = [torch.tensor(s.loc[:str((d - pd.Timedelta(days=1)).date()) + ' 23:00'].dropna().values[-2048:], dtype=torch.float32) for d in week]\n"
                "q, _ = pipe.predict_quantiles(ctx, prediction_length=24, quantile_levels=[0.5])\nq = torch.stack([a.squeeze(0) for a in q]) if isinstance(q, list) else q\n"
                "print('Chronos-2 MAE:', round(float(np.abs(q[:, :, 0].cpu().numpy() - BG.loc[week].values).mean()), 2))\n"
                "print('naive MAE:', round(float(np.abs(BG.shift(1).loc[week].values - BG.loc[week].values).mean()), 2))")
    nb.task(T("For Bitcoin (`var_BTCUSD.csv`), compute the 5 % violations and the Kupiec p-value of the raw and conformal Chronos-2 VaR. Does the conformal shift help?",
              "Pentru Bitcoin (`var_BTCUSD.csv`), calculați depășirile la 5 % și valoarea p a testului Kupiec pentru VaR Chronos-2 brută și conformală. Ajută deplasarea conformală?"))
    nb.solution("V = pd.read_csv(RES + 'var_BTCUSD.csv', parse_dates=['date']).set_index('date')\n"
                "def kupiec(h, a):\n    n, k = len(h), int(h.sum()); p = k / n\n"
                "    lr = -2 * ((n - k) * np.log(1 - a) + k * np.log(a) - (n - k) * np.log(1 - p) - k * np.log(p))\n    return float(1 - stats.chi2.cdf(lr, 1))\n"
                "for m in ['ch_05', 'chc_05']:\n    h = (V.r < V[m]).values\n    print(m, int(h.sum()), round(0.05 * len(h), 1), round(kupiec(h, 0.05), 3))")


if __name__ == "__main__":
    Notebook.build_all(13, TITLE, lecture, kind="lecture")
    Notebook.build_all(13, TITLE, seminar, kind="seminar")
