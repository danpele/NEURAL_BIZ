"""
analysis_L5.py -- Lecture 5 "Recurrent networks: RNN, LSTM and GRU for sequences and financial time series".

Real data (data/clean, see data/SOURCES.md): S&P 500 daily open/high/low/close (Yahoo Finance), Banca Transilvania (TLV)
daily adjusted closes (EODHD, Bucharest Stock Exchange), Romanian hourly electricity load (energy-charts.info, ENTSO-E).
Three tasks, each with its benchmarks first:
  A. next-day log variance of the S&P 500 (Garman-Klass range estimator): naive, HAR, GARCH(1,1) vs RNN, LSTM, GRU;
  B. next-day returns of the S&P 500 and TLV: zero / historical mean / AR(1) vs LSTM (out-of-sample R^2);
  C. day-ahead hourly Romanian load (24 values): seasonal naive, ridge ARX vs RNN, LSTM, GRU.
Networks are trained on the CPU (deterministic with the fixed seeds); results cached in lectures/L5/cache/*.pkl.
Writes lectures/L5/L5_numbers.json and charts/en|ro/*.pdf.   Run:  python3 lectures/L5/analysis_L5.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(6)
npl.style()
SEED = npl.seed_all()
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
N = {"charts": {}, "seed": SEED}
LANGS = ("en", "ro")
SEEDS = 5


def L(en, ro, lang):
    return en if lang == "en" else ro


def cached(name, fn):
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.isfile(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    r = fn()
    r["_seconds"] = round(time.time() - t0, 1)
    with open(p, "wb") as f:
        pickle.dump(r, f)
    print(f"  {name}: {r['_seconds']} s")
    return r


def r2(x):
    return round(float(x), 2)


def dm_test(e1, e2, h=1):
    """Diebold-Mariano test on squared-error (or given) loss differentials d = e1 - e2, Newey-West variance."""
    d = np.asarray(e1) - np.asarray(e2)
    n = len(d)
    dc = d - d.mean()
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    from scipy.stats import norm
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


# =====================================================================================================================
# 0. Data and cut-off (last complete month)
# =====================================================================================================================
cols = {f: npl.markets(f, ["SP500"])["SP500"] for f in ["open", "high", "low", "close", "adjclose"]}
spx = pd.DataFrame(cols).dropna()
END = (spx.index.max() + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.Timedelta(days=1)
spx = spx.loc[:END]
spx = spx[(spx.high > spx.low)]
gk = 0.5 * np.log(spx.high / spx.low) ** 2 - (2 * np.log(2) - 1) * np.log(spx.close / spx.open) ** 2
spx["gk"] = gk.clip(lower=1e-8)
spx["y"] = np.log(spx["gk"])                                   # log daily variance (target of task A)
spx["r"] = 100 * np.log(spx.adjclose).diff()                   # daily log return in %
spx = spx.dropna()
TR_END, VA_END = pd.Timestamp("2016-12-31"), pd.Timestamp("2019-12-31")
N["data"] = {"first": str(spx.index.min().date()), "last": str(spx.index.max().date()), "n_days": int(len(spx)),
             "train_end": "2016-12-31", "valid_end": "2019-12-31", "test_start": "2020-01-01"}
N["data"]["n_train"] = int((spx.index <= TR_END).sum())
N["data"]["n_valid"] = int(((spx.index > TR_END) & (spx.index <= VA_END)).sum())
N["data"]["n_test"] = int((spx.index > VA_END).sum())
ann = np.sqrt(252 * spx["gk"]) * 100
N["data"]["gk_ann_median"] = r2(ann.median())
N["data"]["gk_ann_max"] = r2(ann.max())
N["data"]["gk_ann_max_date"] = str(ann.idxmax().date())
N["data"]["r_sd"] = r2(spx.r.std())
N["data"]["r_kurt"] = r2(spx.r.kurt())

# chart: returns and range volatility
for lang in LANGS:
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 2.9), sharex=True)
    axes[0].plot(spx.index, spx.r, color=npl.PAL["data"], lw=0.4, label=L("Daily log return (%)", "Randamentul logaritmic zilnic (%)", lang))
    axes[1].plot(spx.index, ann, color=npl.PAL["model"], lw=0.4,
                 label=L("Garman–Klass volatility, annualised (%)", "Volatilitatea Garman–Klass, anualizată (%)", lang))
    axes[0].set_ylabel("%")
    axes[1].set_ylabel("%")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlim(spx.index.min(), spx.index.max())
    npl.date_axis(axes[1], lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "sp500_vol", bottom=0.1)
P_SPX = (npl.period(spx.index.min(), spx.index.max(), "en") + ", trading days",
         npl.period(spx.index.min(), spx.index.max(), "ro") + ", zile de tranzacționare")
npl.chart_meta(N, "sp500_vol", "yahoo", *P_SPX)

# chart: autocorrelation of returns and absolute returns
from statsmodels.tsa.stattools import acf  # noqa: E402

LAGS = 50
ac_r = acf(spx.r, nlags=LAGS, fft=True)[1:]
ac_a = acf(spx.r.abs(), nlags=LAGS, fft=True)[1:]
band = 1.96 / np.sqrt(len(spx))
N["acf"] = {"r1": round(float(ac_r[0]), 3), "a1": round(float(ac_a[0]), 3), "a50": round(float(ac_a[-1]), 3),
            "band": round(float(band), 3), "n_sig_r": int((np.abs(ac_r) > band).sum()), "n_sig_a": int((np.abs(ac_a) > band).sum())}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    k = np.arange(1, LAGS + 1)
    ax.vlines(k - 0.15, 0, ac_r, color=npl.PAL["data"], lw=1.6, label=L("returns $r_t$", "randamente $r_t$", lang))
    ax.vlines(k + 0.15, 0, ac_a, color=npl.PAL["model"], lw=1.6, label=L("absolute returns $|r_t|$", "randamente absolute $|r_t|$", lang))
    ax.axhspan(-band, band, color=npl.MUTED, alpha=0.25, lw=0, label=L("±1.96/√T band", "banda ±1,96/√T", lang))
    ax.axhline(0, color="#222222", lw=0.6)
    ax.set_xlabel(L("Lag (trading days)", "Decalajul (zile de tranzacționare)", lang))
    ax.set_ylabel(L("Autocorrelation", "Autocorelația", lang))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "acf_abs")
npl.chart_meta(N, "acf_abs", "yahoo", *P_SPX)

# =====================================================================================================================
# A. Volatility: next-day log Garman-Klass variance
# =====================================================================================================================
W = 22
y = spx.y.values.astype(np.float32)
rr = spx.r.values.astype(np.float32)
idx_dates = spx.index
tr_mask = idx_dates <= TR_END
mu_y, sd_y = y[tr_mask].mean(), y[tr_mask].std()
sd_r = rr[tr_mask].std()
feat = np.stack([(y - mu_y) / sd_y, rr / sd_r, np.abs(rr) / sd_r], 1).astype(np.float32)
# sample t uses days t-W+1..t to forecast y[t+1]
T_ = np.arange(W - 1, len(y) - 1)
Xseq = np.stack([feat[t - W + 1:t + 1] for t in T_])
Yt = ((y[T_ + 1] - mu_y) / sd_y).astype(np.float32)
tdate = idx_dates[T_ + 1]                                       # date of the forecast target
s_tr, s_va, s_te = tdate <= TR_END, (tdate > TR_END) & (tdate <= VA_END), tdate > VA_END

# benchmarks
y_true = y[T_ + 1]
naive = y[T_]
d5 = pd.Series(y).rolling(5).mean().values
d22 = pd.Series(y).rolling(22).mean().values
H = np.column_stack([np.ones(len(T_)), y[T_], d5[T_], d22[T_]])
beta = np.linalg.lstsq(H[s_tr], y_true[s_tr], rcond=None)[0]
har = H @ beta
N["har"] = {"b0": round(float(beta[0]), 3), "bd": round(float(beta[1]), 3), "bw": round(float(beta[2]), 3),
            "bm": round(float(beta[3]), 3)}
from arch import arch_model  # noqa: E402

r_all = spx.r.values
ntr = int(tr_mask.sum())
am = arch_model(r_all, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
gres = am.fit(last_obs=ntr, disp="off")
N["garch"] = {k.split("[")[0]: round(float(gres.params[k]), 3) for k in ["omega", "alpha[1]", "beta[1]", "nu"]}
N["garch"]["persistence"] = round(float(gres.params["alpha[1]"] + gres.params["beta[1]"]), 3)
fc = am.fix(gres.params).forecast(horizon=1, start=0, reindex=True)       # one-step forecasts with fixed parameters
sig2 = fc.variance.values[:, 0]                                 # forecast for t+1 made at t (in %^2)
lg = np.log(sig2[T_] / 1e4)
mz = np.linalg.lstsq(np.column_stack([np.ones(s_tr.sum()), lg[s_tr]]), y_true[s_tr], rcond=None)[0]
garch = mz[0] + mz[1] * lg                                      # mapped to the log range variance on the training part


class RecNet(nn.Module):
    def __init__(self, cell, n_in, hidden=32, n_out=1, n_static=0):
        super().__init__()
        self.cell = cell
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell](n_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden + n_static, n_out)

    def forward(self, x, s=None):
        h, _ = self.rnn(x)
        z = h[:, -1]
        if s is not None:
            z = torch.cat([z, s], 1)
        return self.out(z)


def train_rec(cell, Xtr, Ytr, Xva, Yva, seed, hidden=32, epochs=40, lr=1e-3, bs=64, Str=None, Sva=None, patience=6, clip=1.0):
    torch.manual_seed(SEED + seed)
    np.random.seed(SEED + seed)
    n_out = Ytr.shape[1] if Ytr.ndim > 1 else 1
    m = RecNet(cell, Xtr.shape[2], hidden, n_out, 0 if Str is None else Str.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xtr_, Ytr_ = torch.tensor(Xtr), torch.tensor(Ytr).view(len(Ytr), -1)
    Xva_, Yva_ = torch.tensor(Xva), torch.tensor(Yva).view(len(Yva), -1)
    Str_ = None if Str is None else torch.tensor(Str)
    Sva_ = None if Sva is None else torch.tensor(Sva)
    g = torch.Generator().manual_seed(SEED + seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr_), generator=g)
        for j in range(0, len(perm), bs):
            b = perm[j:j + bs]
            pred = m(Xtr_[b], None if Str_ is None else Str_[b])
            loss = ((pred - Ytr_[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), clip)       # gradient clipping (exploding gradients)
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(((m(Xva_, Sva_) - Yva_) ** 2).mean())
        hist.append(vl)
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:                                 # early stopping on the validation years
                break
    m.load_state_dict(best_state)
    return m, hist


def predict(m, X, S=None):
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(X), None if S is None else torch.tensor(S)).numpy()


def run_vol():
    out = {}
    for cell in ["rnn", "lstm", "gru"]:
        preds, hists, states = [], [], []
        for s in range(SEEDS):
            m, h = train_rec(cell, Xseq[s_tr], Yt[s_tr], Xseq[s_va], Yt[s_va], s)
            preds.append(predict(m, Xseq)[:, 0] * sd_y + mu_y)
            hists.append(h)
            if s == 0:
                states.append({k: v.clone() for k, v in m.state_dict().items()})
        out[cell] = {"pred": np.array(preds), "hist": hists, "state0": states[0]}
    return out


VOL = cached("vol", run_vol)


def mse(p, mask=s_te):
    return float(np.mean((p[mask] - y_true[mask]) ** 2))


base = mse(naive)
N["vol"] = {"W": W, "hidden": 32, "seeds": SEEDS}
for k, p in [("naive", naive), ("har", har), ("garch", garch)]:
    N["vol"][k] = {"mse": round(mse(p), 4), "r2": round(1 - mse(p) / float(np.var(y_true[s_te])), 3)}
for cell in ["rnn", "lstm", "gru"]:
    ms = [mse(p) for p in VOL[cell]["pred"]]
    ens = VOL[cell]["pred"].mean(0)
    N["vol"][cell] = {"mse_mean": round(float(np.mean(ms)), 4), "mse_sd": round(float(np.std(ms, ddof=1)), 4),
                      "mse_min": round(float(np.min(ms)), 4), "mse_max": round(float(np.max(ms)), 4),
                      "mse_ens": round(mse(ens), 4), "r2_ens": round(1 - mse(ens) / float(np.var(y_true[s_te])), 3),
                      "epochs": int(np.mean([len(h) for h in VOL[cell]["hist"]]))}
best_net = min(["rnn", "lstm", "gru"], key=lambda c: N["vol"][c]["mse_ens"])
N["vol"]["best_net"] = best_net.upper()
e_l = (VOL["lstm"]["pred"].mean(0)[s_te] - y_true[s_te]) ** 2
e_h = (har[s_te] - y_true[s_te]) ** 2
e_g = (garch[s_te] - y_true[s_te]) ** 2
st, pv = dm_test(e_l, e_h)
N["vol"]["dm_lstm_har"] = {"stat": r2(st), "p": round(pv, 3)}
st, pv = dm_test(e_l, e_g)
N["vol"]["dm_lstm_garch"] = {"stat": r2(st), "p": round(pv, 3)}
N["vol"]["lstm_vs_har_pct"] = round(100 * (N["vol"]["lstm"]["mse_ens"] / N["vol"]["har"]["mse"] - 1), 1)
N["vol"]["lstm_vs_naive_pct"] = round(100 * (N["vol"]["lstm"]["mse_ens"] / N["vol"]["naive"]["mse"] - 1), 1)
# 2020 crash sub-period
crash = s_te & (tdate >= pd.Timestamp("2020-02-20")) & (tdate <= pd.Timestamp("2020-04-30"))
N["vol"]["crash"] = {"har": round(mse(har, crash), 3), "lstm": round(mse(VOL["lstm"]["pred"].mean(0), crash), 3),
                     "garch": round(mse(garch, crash), 3), "naive": round(mse(naive, crash), 3)}

# chart: test-period forecasts (annualised volatility), 2020 window
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    msk = (tdate >= pd.Timestamp("2020-01-01")) & (tdate <= pd.Timestamp("2020-12-31"))
    to_vol = lambda v: np.sqrt(252 * np.exp(v)) * 100
    ax.plot(tdate[msk], to_vol(y_true[msk]), color=npl.PAL["data"], lw=0.8, alpha=0.8, label=L("Observed (Garman–Klass)", "Observat (Garman–Klass)", lang))
    ax.plot(tdate[msk], to_vol(har[msk]), color=npl.PAL["benchmark"], lw=1.4, label="HAR")
    ax.plot(tdate[msk], to_vol(VOL["lstm"]["pred"].mean(0)[msk]), color=npl.PAL["model"], lw=1.4,
            label=L("LSTM (mean of 5 seeds)", "LSTM (media pe 5 semințe)", lang))
    ax.set_ylabel(L("Annualised volatility (%)", "Volatilitatea anualizată (%)", lang))
    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "vol_forecast")
npl.chart_meta(N, "vol_forecast", "yahoo", "test year 2020 (one-day-ahead forecasts), trading days",
               "anul de test 2020 (prognoze pentru ziua următoare), zile de tranzacționare")

# chart: test MSE by model, seeds as points
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    names = [L("Naive", "Naiv", lang), "GARCH", "HAR", "RNN", "LSTM", "GRU"]
    vals = [N["vol"]["naive"]["mse"], N["vol"]["garch"]["mse"], N["vol"]["har"]["mse"]]
    colors = [npl.PAL["benchmark"]] * 3 + [npl.PAL["model"]] * 3
    for i, v in enumerate(vals):
        ax.bar(i, v, color=colors[i], width=0.6)
    for i, cell in enumerate(["rnn", "lstm", "gru"]):
        ms = [mse(p) for p in VOL[cell]["pred"]]
        ax.bar(3 + i, np.mean(ms), color=colors[3 + i], width=0.6, alpha=0.55)
        ax.scatter(np.full(len(ms), 3 + i) + np.linspace(-0.15, 0.15, len(ms)), ms, color="#222222", s=8, zorder=3)
    ax.set_xticks(range(6)); ax.set_xticklabels(names)
    ax.set_ylabel(L("Test MSE of log variance", "MSE pe test (log-varianță)", lang))
    ax.set_ylim(0, max(vals) * 1.15)
    ax.bar([], [], color=npl.PAL["benchmark"], label=L("benchmarks", "modele de referință", lang))
    ax.bar([], [], color=npl.PAL["model"], alpha=0.55, label=L("recurrent networks (mean)", "rețele recurente (media)", lang))
    ax.scatter([], [], color="#222222", s=8, label=L("one seed", "o sămînță", lang))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "vol_mse")
npl.chart_meta(N, "vol_mse", "yahoo", "test 2 Jan 2020 to " + npl.period(END, END, "en").split(" to ")[1] + "; 5 seeds per network",
               "test 2 ianuarie 2020 - " + npl.period(END, END, "ro").split(" - ")[1] + "; 5 semințe pentru fiecare rețea")

# forget gate of the LSTM (seed 0) over the test period
lstm0 = RecNet("lstm", 3, 32, 1)
lstm0.load_state_dict(VOL["lstm"]["state0"])


def lstm_gates(m, X):
    """Run the LSTM step by step and return the mean forget-gate and input-gate activations at the last step."""
    Wih, Whh = m.rnn.weight_ih_l0.detach(), m.rnn.weight_hh_l0.detach()
    b = (m.rnn.bias_ih_l0 + m.rnn.bias_hh_l0).detach()
    x = torch.tensor(X)
    n, T, _ = x.shape
    Hd = Whh.shape[1]
    h = torch.zeros(n, Hd)
    c = torch.zeros(n, Hd)
    for t in range(T):
        z = x[:, t] @ Wih.T + h @ Whh.T + b
        i, f, g, o = z.split(Hd, 1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
    return f.mean(1).numpy(), i.mean(1).numpy()


fg, ig = lstm_gates(lstm0, Xseq[s_te])
N["gates"] = {"f_mean": round(float(fg.mean()), 3), "i_mean": round(float(ig.mean()), 3),
              "f_corr_y": round(float(np.corrcoef(fg, y[T_][s_te])[0, 1]), 3),
              "i_corr_y": round(float(np.corrcoef(ig, y[T_][s_te])[0, 1]), 3)}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.plot(tdate[s_te], pd.Series(fg).rolling(5).mean(), color=npl.PAL["model"], lw=1.0,
            label=L("forget gate $f_t$ (mean of 32 units)", "poarta de uitare $f_t$ (media celor 32 de unități)", lang))
    ax.plot(tdate[s_te], pd.Series(ig).rolling(5).mean(), color=npl.PAL["c7"], lw=1.0,
            label=L("input gate $i_t$", "poarta de intrare $i_t$", lang))
    ax.set_ylabel(L("Gate activation (5-day mean)", "Activarea porții (media pe 5 zile)", lang), fontsize=9.5)
    ax.set_xlim(tdate[s_te].min(), tdate[s_te].max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "lstm_gates")
npl.chart_meta(N, "lstm_gates", "yahoo", "test period 2020-2026, LSTM seed 20261108, last step of each 22-day window",
               "perioada de test 2020-2026, LSTM cu sămînța 20261108, ultimul pas al fiecărei ferestre de 22 de zile")

# =====================================================================================================================
# B. Returns: is there anything to forecast? S&P 500 and Banca Transilvania (TLV)
# =====================================================================================================================
tlv = npl.bvb("adjclose", ["TLV"])["TLV"].dropna().loc[:END]
tlv = tlv[tlv.index >= "2008-01-01"]
tlv = tlv[tlv.diff() != 0]                                      # drop non-trading rows (unchanged price on holidays)
r_tlv = (100 * np.log(tlv).diff()).dropna()
RET = {"SP500": spx.r, "TLV": r_tlv}
N["ret"] = {"tlv_first": str(r_tlv.index.min().date()), "tlv_n": int(len(r_tlv)), "W": W}


def run_ret():
    out = {}
    for name, s in RET.items():
        v = s.values.astype(np.float32)
        d = s.index
        Tn = np.arange(W - 1, len(v) - 1)
        td = d[Tn + 1]
        mtr, mva, mte = td <= TR_END, (td > TR_END) & (td <= VA_END), td > VA_END
        sd = v[d <= TR_END].std()
        X = np.stack([v[t - W + 1:t + 1, None] / sd for t in Tn]).astype(np.float32)
        Yv = (v[Tn + 1] / sd).astype(np.float32)
        preds = []
        for s_ in range(SEEDS):
            m, _ = train_rec("lstm", X[mtr], Yv[mtr], X[mva], Yv[mva], s_, hidden=16)
            preds.append(predict(m, X)[:, 0] * sd)
        out[name] = {"pred": np.array(preds), "y": v[Tn + 1], "lag": v[Tn], "mtr": mtr, "mte": mte, "mva": mva, "td": td}
    return out


RT = cached("ret", run_ret)
for name, o in [(k, RT[k]) for k in RET]:
    yv, mtr, mte = o["y"], o["mtr"], o["mte"]
    ar = np.polyfit(o["lag"][mtr], yv[mtr], 1)
    p_ar = np.polyval(ar, o["lag"])
    p_mean = np.full_like(yv, yv[mtr].mean())
    sse0 = np.sum(yv[mte] ** 2)
    oos = lambda p: 1 - np.sum((yv[mte] - p[mte]) ** 2) / sse0           # out-of-sample R^2 against the zero forecast
    lst = [oos(p) for p in o["pred"]]
    hit = [float(np.mean(np.sign(p[mte]) == np.sign(yv[mte]))) for p in o["pred"]]
    N["ret"][name] = {"r2_mean": round(100 * oos(p_mean), 3), "r2_ar": round(100 * oos(p_ar), 3),
                      "r2_lstm_mean": round(100 * float(np.mean(lst)), 3), "r2_lstm_min": round(100 * float(np.min(lst)), 3),
                      "r2_lstm_max": round(100 * float(np.max(lst)), 3), "ar_slope": round(float(ar[0]), 3),
                      "hit_lstm": round(100 * float(np.mean(hit)), 1), "up_share": round(100 * float(np.mean(yv[mte] > 0)), 1),
                      "n_test": int(mte.sum())}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    labs = [L("Training mean", "Media de antrenare", lang), "AR(1)", "LSTM"]
    xs = np.arange(3)
    for j, (name, c) in enumerate([("SP500", npl.PAL["data"]), ("TLV", npl.PAL["alt"])]):
        vals = [N["ret"][name]["r2_mean"], N["ret"][name]["r2_ar"], N["ret"][name]["r2_lstm_mean"]]
        ax.bar(xs + (j - 0.5) * 0.36, vals, width=0.36, color=c, label="S&P 500" if name == "SP500" else L("Banca Transilvania (TLV)", "Banca Transilvania (TLV)", lang))
        lst = [100 * (1 - np.sum((RT[name]["y"][RT[name]["mte"]] - p[RT[name]["mte"]]) ** 2) / np.sum(RT[name]["y"][RT[name]["mte"]] ** 2)) for p in RT[name]["pred"]]
        ax.scatter(np.full(len(lst), 2 + (j - 0.5) * 0.36), lst, color="#222222", s=8, zorder=3)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labs)
    ax.set_ylabel(L("Out-of-sample $R^2$ (%)", "$R^2$ în afara eșantionului (%)", lang))
    ax.scatter([], [], color="#222222", s=8, label=L("LSTM, one seed", "LSTM, o sămînță", lang))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "returns_r2")
npl.chart_meta(N, "returns_r2", "yahoo", "test 2020 to Oct 2026; S\\&P 500 (Yahoo Finance) and TLV (EODHD, Bucharest Stock Exchange)",
               "test 2020 - octombrie 2026; S\\&P 500 (Yahoo Finance) și TLV (EODHD, Bursa de Valori București)",
               source_en="Yahoo Finance; EODHD", source_ro="Yahoo Finance; EODHD")

# =====================================================================================================================
# C. Romanian hourly load: day-ahead forecast of the 24 hours of day d+1 from the hours up to the end of day d (UTC)
# =====================================================================================================================
ld = npl.load_generation_ro(tz="UTC")["load"]
ld = ld[(ld.index >= "2016-01-01") & (ld.index < (END + pd.Timedelta(days=1)).tz_localize("UTC"))]
days = pd.date_range(ld.index.min().normalize(), ld.index.max().normalize(), freq="D", tz="UTC")
full = ld.reindex(pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h", tz="UTC"))
M = full.values.reshape(-1, 24)                                 # day x hour (UTC)
dd = pd.DatetimeIndex(days[:len(M)])
LTR, LVA = pd.Timestamp("2022-12-31", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
lmu, lsd = np.nanmean(M[dd <= LTR]), np.nanstd(M[dd <= LTR])
N["load"] = {"first": str(dd.min().date()), "last": str(dd.max().date()), "mean_mw": int(round(np.nanmean(M))),
             "missing_hours": int(np.isnan(M).sum()), "train_end": "2022-12-31", "valid_end": "2023-12-31"}


def load_samples(win):
    """Origin = end of day d; input = the last `win` hours as win/24 daily profiles; target = 24 hours of day d+1;
    calendar of day d+1 as static input."""
    nd = win // 24
    X, Y, S, D_, NV, YD = [], [], [], [], [], []
    flat = M.reshape(-1)
    for d in range(nd + 7, len(M) - 1):
        x = flat[(d + 1) * 24 - win:(d + 1) * 24]
        yy = M[d + 1]
        if np.isnan(x).any() or np.isnan(yy).any() or np.isnan(M[d - 6]).any():
            continue
        tgt = dd[d + 1]
        # one step per day: the 24 scaled hourly loads of that day + its day of the week (sequence of nd daily profiles)
        dows = np.zeros((nd, 7)); dows[np.arange(nd), [(dd[d - nd + 1 + j]).dayofweek for j in range(nd)]] = 1
        X.append(np.concatenate([((x - lmu) / lsd).reshape(nd, 24), dows], 1))
        dow = np.zeros(7); dow[tgt.dayofweek] = 1
        doy = tgt.dayofyear / 365.25
        S.append(np.concatenate([dow, [np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)]]))
        Y.append((yy - M[d - 6]) / lsd)                            # target: deviation from the seasonal naive forecast
        D_.append(tgt)
        NV.append(M[d - 6])                                      # seasonal naive: same weekday one week earlier
        YD.append(M[d])                                          # yesterday
    return (np.array(X, np.float32), np.array(Y, np.float32), np.array(S, np.float32), pd.DatetimeIndex(D_),
            np.array(NV), np.array(YD))


def run_load_main():
    out = {}
    X, Y, S, D_, NV, YD = load_samples(168)
    mtr, mva, mte = D_ <= LTR, (D_ > LTR) & (D_ <= LVA), D_ > LVA
    for cell in ["rnn", "lstm", "gru"]:
        def one(cell=cell):
            preds, st0 = [], None
            for s in range(3):
                m, h = train_rec(cell, X[mtr], Y[mtr], X[mva], Y[mva], s, hidden=48, epochs=40, lr=3e-3, bs=64,
                                 Str=S[mtr], Sva=S[mva], patience=5)
                preds.append(predict(m, X, S) * lsd + NV)
                if s == 0:
                    st0 = {k: v.clone() for k, v in m.state_dict().items()}
                print(f"    load {cell} seed {s}: epochs {len(h)}", flush=True)
            return {"pred": np.array(preds), "state0": st0}
        out[cell] = cached(f"load_{cell}", one)
    out["meta"] = {"D": D_, "NV": NV, "YD": YD, "Y": Y * lsd + NV, "mtr": mtr, "mva": mva, "mte": mte}
    from sklearn.linear_model import Ridge
    Z = np.column_stack([X[:, :, :24].reshape(len(X), -1), S])
    best = None
    for a in [0.1, 1, 10, 100]:
        rg = Ridge(alpha=a).fit(Z[mtr], Y[mtr])
        v = np.mean((rg.predict(Z[mva]) - Y[mva]) ** 2)
        if best is None or v < best[0]:
            best = (v, a, rg)
    out["ridge"] = {"pred": best[2].predict(Z) * lsd + NV, "alpha": best[1]}
    return out


def run_window(win):
    Xw, Yw, Sw, Dw, NVw, _ = load_samples(win)
    a_, b_, c_ = Dw <= LTR, (Dw > LTR) & (Dw <= LVA), Dw > LVA
    res = []
    for s in range(2):
        m, _ = train_rec("lstm", Xw[a_], Yw[a_], Xw[b_], Yw[b_], s, hidden=48, epochs=40, lr=3e-3, bs=64,
                         Str=Sw[a_], Sva=Sw[b_], patience=5)
        p = predict(m, Xw[c_], Sw[c_]) * lsd + NVw[c_]
        yt_ = Yw[c_] * lsd + NVw[c_]
        res.append(float(np.mean(np.abs(p - yt_) / yt_)) * 100)
    print("    window", win, res, flush=True)
    return {"mape": res}


def run_load():
    t0 = time.time()
    out = {"main": run_load_main(), "window": {}}
    for win in [24, 72, 168, 336]:
        out["window"][win] = cached(f"load_window_{win}", lambda win=win: run_window(win))["mape"]
    out["_seconds"] = round(sum(out["main"][c]["_seconds"] for c in ["rnn", "lstm", "gru"]) + time.time() - t0, 1)
    return out


LD = run_load()
meta = LD["main"]["meta"]
mte = meta["mte"]
Ytrue = meta["Y"]


def mape(p, m=mte):
    return float(np.mean(np.abs(p[m] - Ytrue[m]) / Ytrue[m]) * 100)


N["load"]["n_train"] = int(meta["mtr"].sum())
N["load"]["n_valid"] = int(meta["mva"].sum())
N["load"]["n_test"] = int(mte.sum())
N["load"]["test_first"] = str(meta["D"][mte].min().date())
N["load"]["test_last"] = str(meta["D"][mte].max().date())
N["load"]["ridge_alpha"] = LD["main"]["ridge"]["alpha"]
N["load"]["snaive"] = round(mape(meta["NV"]), 2)
N["load"]["yday"] = round(mape(meta["YD"]), 2)
N["load"]["ridge"] = round(mape(LD["main"]["ridge"]["pred"]), 2)
for cell in ["rnn", "lstm", "gru"]:
    ms = [mape(p) for p in LD["main"][cell]["pred"]]
    N["load"][cell] = {"mean": round(float(np.mean(ms)), 2), "sd": round(float(np.std(ms, ddof=1)), 2),
                       "ens": round(mape(LD["main"][cell]["pred"].mean(0)), 2)}
N["load"]["window"] = {str(w): round(float(np.mean(v)), 2) for w, v in LD["window"].items()}
N["load"]["seconds"] = LD["_seconds"]
best_l = min(["rnn", "lstm", "gru"], key=lambda c: N["load"][c]["ens"])
N["load"]["best"] = best_l.upper()
N["load"]["best_vs_ridge_pct"] = round(100 * (N["load"][best_l]["ens"] / N["load"]["ridge"] - 1), 1)
N["load"]["lstm_vs_snaive_pct"] = round(100 * (N["load"]["lstm"]["ens"] / N["load"]["snaive"] - 1), 1)
dl = np.mean(np.abs(LD["main"]["lstm"]["pred"].mean(0)[mte] - Ytrue[mte]), 1)
dr = np.mean(np.abs(LD["main"]["ridge"]["pred"][mte] - Ytrue[mte]), 1)
st, pv = dm_test(dl, dr)
N["load"]["dm_lstm_ridge"] = {"stat": r2(st), "p": round(pv, 3)}
N["load"]["mae_lstm_mw"] = int(round(float(dl.mean())))
N["load"]["mae_ridge_mw"] = int(round(float(dr.mean())))
N["load"]["mae_snaive_mw"] = int(round(float(np.mean(np.abs(meta["NV"][mte] - Ytrue[mte])))))
# MWh cost illustration: 1 MWh imbalance per MW of error per hour
N["load"]["lstm_gain_mwh_day"] = int(round(24 * (N["load"]["mae_snaive_mw"] - N["load"]["mae_lstm_mw"])))

# chart: load series (one year) and the daily profile
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    s = full.loc["2016-01-01":str(END.date())].resample("D").mean()
    ax.plot(s.index, s.values / 1000, color=npl.PAL["data"], lw=0.6, label=L("daily mean load", "consumul mediu zilnic", lang))
    ax.plot(s.index, s.rolling(365, center=True).mean() / 1000, color=npl.PAL["model"], lw=1.6,
            label=L("365-day mean", "media pe 365 de zile", lang))
    ax.set_ylabel("GW")
    ax.set_xlim(s.index.min(), s.index.max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "load_series")
npl.chart_meta(N, "load_series", "power", npl.period(dd.min(), dd.max(), "en") + ", hourly, mean MW per day",
               npl.period(dd.min(), dd.max(), "ro") + ", orar, MW medii pe zi")

# chart: a test fortnight
tst = np.where(mte)[0]
k0 = tst[np.argmax(meta["D"][tst] >= pd.Timestamp("2025-01-20", tz="UTC"))]
sel = np.arange(k0, k0 + 14)
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    tt = pd.DatetimeIndex(np.concatenate([meta["D"][i] + pd.to_timedelta(np.arange(24), "h") for i in sel]))
    ax.plot(tt, np.concatenate([Ytrue[i] for i in sel]) / 1000, color=npl.PAL["data"], lw=1.6, label=L("observed", "observat", lang))
    ax.plot(tt, np.concatenate([meta["NV"][i] for i in sel]) / 1000, color=npl.PAL["benchmark"], lw=1.0, ls="--",
            label=L("seasonal naive (a week earlier)", "naiv sezonier (cu o săptămînă înainte)", lang))
    ax.plot(tt, np.concatenate([LD["main"]["lstm"]["pred"].mean(0)[i] for i in sel]) / 1000, color=npl.PAL["model"], lw=1.2,
            label=L("LSTM", "LSTM", lang))
    ax.set_ylabel("GW")
    ax.set_xlim(tt.min(), tt.max())
    ax.xaxis.set_major_locator(__import__("matplotlib.dates", fromlist=["DayLocator"]).DayLocator(interval=2))
    mn = npl.MON[lang]
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: (lambda d_: f"{d_.day} {mn[d_.month - 1]}\n{d_.year}")(
        __import__("matplotlib.dates", fromlist=["num2date"]).num2date(v))))
    ax.tick_params(axis="x", labelsize=8.5)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "load_fortnight")
npl.chart_meta(N, "load_fortnight", "power", "14 test days from " + npl.period(meta["D"][sel[0]], meta["D"][sel[-1]], "en"),
               "14 zile de test, " + npl.period(meta["D"][sel[0]], meta["D"][sel[-1]], "ro"))
N["load"]["fortnight_first"] = str(meta["D"][sel[0]].date())

# chart: MAPE by model, MAE by hour
for lang in LANGS:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    names = [L("Naive\n(week)", "Naiv\n(săpt.)", lang), L("Naive\n(day)", "Naiv\n(zi)", lang), "Ridge\nARX", "RNN", "LSTM", "GRU"]
    vals = [N["load"]["snaive"], N["load"]["yday"], N["load"]["ridge"], N["load"]["rnn"]["ens"], N["load"]["lstm"]["ens"], N["load"]["gru"]["ens"]]
    cl = [npl.PAL["benchmark"]] * 3 + [npl.PAL["model"]] * 3
    axes[0].bar(range(6), vals, color=cl, width=0.65)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.08, f"{v:.2f}" if lang == "en" else f"{v:.2f}".replace(".", ","), ha="center", fontsize=7.5)
    axes[0].set_xticks(range(6)); axes[0].set_xticklabels(names, fontsize=7.5)
    axes[0].set_ylabel(L("Test MAPE (%)", "MAPE pe test (%)", lang))
    hr = np.arange(24)
    axes[1].plot(hr, np.mean(np.abs(meta["NV"][mte] - Ytrue[mte]), 0), color=npl.PAL["benchmark"], ls="--", lw=1.4,
                 label=L("seasonal naive", "naiv sezonier", lang))
    axes[1].plot(hr, np.mean(np.abs(LD["main"]["ridge"]["pred"][mte] - Ytrue[mte]), 0), color=npl.PAL["alt"], lw=1.4, label="ridge ARX")
    axes[1].plot(hr, np.mean(np.abs(LD["main"]["lstm"]["pred"].mean(0)[mte] - Ytrue[mte]), 0), color=npl.PAL["model"], lw=1.6, label="LSTM")
    axes[1].set_xlabel(L("Hour of day d+1 (UTC)", "Ora din ziua d+1 (UTC)", lang), fontsize=9)
    axes[1].set_ylabel(L("MAE (MW)", "MAE (MW)", lang))
    axes[1].set_xticks([0, 6, 12, 18, 23])
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "load_errors")
npl.chart_meta(N, "load_errors", "power", npl.period(meta["D"][mte].min(), meta["D"][mte].max(), "en") + " (test days)",
               npl.period(meta["D"][mte].min(), meta["D"][mte].max(), "ro") + " (zile de test)")

# chart: window length
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.1))
    ws = [24, 72, 168, 336]
    for i, w in enumerate(ws):
        v = LD["window"][w]
        ax.bar(i, np.mean(v), color=npl.PAL["model"], width=0.55, alpha=0.7)
        ax.scatter([i - 0.08, i + 0.08], v, color="#222222", s=10, zorder=3)
    ax.axhline(N["load"]["ridge"], color=npl.PAL["alt"], ls="--", lw=1.3, label=L("ridge ARX (168 hours)", "ridge ARX (168 de ore)", lang))
    ax.set_xticks(range(4))
    ax.set_xticklabels([L(f"{w} hours", f"{w} de ore" if w >= 20 else f"{w} ore", lang) for w in ws])
    ax.set_ylabel(L("Test MAPE (%)", "MAPE pe test (%)", lang))
    ax.set_ylim(0, max(max(np.mean(v) for v in LD["window"].values()), N["load"]["ridge"]) * 1.2)
    ax.scatter([], [], color="#222222", s=10, label=L("LSTM, one seed", "LSTM, o sămînță", lang))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "load_window")
npl.chart_meta(N, "load_window", "power", "test days 2024-2026; LSTM with 2 seeds per window",
               "zile de test 2024-2026; LSTM cu 2 semințe pentru fiecare fereastră")

# chart: sensitivity of the forecast to each input lag (gradient), trained RNN vs LSTM vs GRU on load
X168, Y168, S168, D168, _, _ = load_samples(168)
te_ids = np.where(D168 > LVA)[0][:200]
sens = {}
for cell in ["rnn", "lstm", "gru"]:
    m = RecNet(cell, 31, 48, 24, 9)
    m.load_state_dict(LD["main"][cell]["state0"])
    m.eval()
    x = torch.tensor(X168[te_ids], requires_grad=True)
    out = m(x, torch.tensor(S168[te_ids]))
    out[:, 12].sum().backward()                                  # forecast of hour 12 of day d+1
    gnorm = x.grad[:, :, :24].abs().mean(0).numpy().reshape(-1)[::-1]   # lag 1 = last hour of day d
    sens[cell] = gnorm / gnorm.max()
N["grad"] = {c: {"lag168_rel": round(float(sens[c][167] / sens[c][:24].max()), 3),
                 "lag24_rel": round(float(sens[c][23] / sens[c][0]), 3)} for c in sens}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for cell, c in [("rnn", npl.PAL["benchmark"]), ("lstm", npl.PAL["model"]), ("gru", npl.PAL["c7"])]:
        ax.plot(np.arange(1, 169), sens[cell], color=c, lw=1.2, label=cell.upper())
    ax.set_yscale("log")
    ax._keep_y = True
    ax.set_xlabel(L("Lag of the input hour (hours before the forecast origin)", "Decalajul orei de intrare (ore înainte de momentul prognozei)", lang), fontsize=9.5)
    ax.set_ylabel(L("|∂ forecast / ∂ input|\n(relative)", "|∂ prognoză / ∂ intrare|\n(relativ)", lang), fontsize=9)
    ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168])
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "grad_lag")
npl.chart_meta(N, "grad_lag", "power", "200 test days of 2024; models with seed 20261108; forecast for noon (hour 12, universal time)",
               "200 de zile de test din 2024; modele cu sămînța 20261108; prognoza pentru ora 12 (timpul universal)")

with open(os.path.join(HERE, "L5_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote L5_numbers.json and", len(N["charts"]), "charts x 2 languages")
