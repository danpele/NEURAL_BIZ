"""
analysis_L13.py -- Lecture 13: deep learning for energy and financial time series; time-series foundation models.
All computation of the lecture: numbers -> L13_numbers.json, charts -> charts/en|ro/*.pdf.

Real data only (data/clean, see data/SOURCES.md):
  * Romanian day-ahead electricity prices (OPCOM, via energy-charts.info), with DE-LU and HU prices and Romanian load;
  * S&P 500 and Bitcoin daily closes (Yahoo Finance).
Models: naive benchmarks, LEAR (Lasso-estimated autoregressive), a deep neural network (DNN, Lago et al. 2021 style),
N-BEATS, and three pretrained time-series foundation models used zero-shot (Chronos-2, Chronos-Bolt, TimesFM 2.5);
split-conformal correction of the foundation-model quantiles; Value-at-Risk (VaR) backtests.

Heavy steps are cached in lectures/L13/cache (git-ignored); per-day forecasts and errors are written to
lectures/L13/results/*.csv (small, committed) so that the Quantlets can recompute every number.
Run:  python3 lectures/L13/analysis_L13.py          (first run about 15-25 min on an Apple M-series laptop, then seconds)
"""
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")      # avoid thread oversubscription with PyTorch
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

warnings.filterwarnings("ignore")
npl.style()
SEED = npl.seed_all()
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(HERE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RES, exist_ok=True)
N = {"charts": {}, "seed": SEED}
T0 = time.time()

# ---------------------------------------------------------------- 0. cut-off and splits (fixed a priori)
TEST_START, TEST_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-31")    # last 12 complete months
CAL_DAYS = 60                     # conformal calibration window (days of past forecasts)
FM_START = TEST_START - pd.Timedelta(days=CAL_DAYS + 5)
WIN = 1092                        # LEAR / DNN calibration window, days (3 years, as in Lago et al. 2021)
CTX = 2048                        # foundation-model context, hours (about 85 days)
VAR_START, VAR_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-10-31")
N["split"] = {"test_start": str(TEST_START.date()), "test_end": str(TEST_END.date()), "win_days": WIN, "ctx_hours": CTX,
              "cal_days": CAL_DAYS, "var_start": str(VAR_START.date()), "var_end": str(VAR_END.date())}


def L(lang, en, ro):
    return en if lang == "en" else ro


# ---------------------------------------------------------------- 1. data: hourly prices and load on the CET day-hour grid
P = npl.prices_hourly()                                   # Europe/Brussels (CET/CEST), EUR/MWh
LD = npl.load_generation_ro().tz_convert("Europe/Brussels")


def day_matrix(s):
    """date x hour (0-23) on the CET calendar; the doubled autumn hour is averaged, the missing spring hour stays missing."""
    df = pd.DataFrame({"v": s.values, "d": s.index.tz_localize(None).normalize(), "h": s.index.hour}, index=s.index)
    m = df.groupby(["d", "h"])["v"].mean().unstack("h")
    return m.reindex(columns=range(24))


RO = day_matrix(P["RO"]).loc[:TEST_END]
DE = day_matrix(P["DE_LU"]).loc[:TEST_END]
HU = day_matrix(P["HU"]).loc[:TEST_END]
LO = day_matrix(LD["load"]).loc[:TEST_END]
full = RO.notna().all(axis=1)
N["data"] = {"first": str(RO.index.min().date()), "last": str(RO.index.max().date()), "n_days": int(len(RO)),
             "n_full_days": int(full.sum()), "n_hours": int(RO.notna().sum().sum())}
test_days = [d for d in RO.loc[TEST_START:TEST_END].index if full.get(d, False)]
N["data"]["n_test_days"] = len(test_days)
N["data"]["n_test_hours"] = 24 * len(test_days)
y_test = RO.loc[test_days]
N["data"]["test_mean"] = round(float(y_test.values.mean()), 2)
N["data"]["test_std"] = round(float(y_test.values.std(ddof=1)), 2)
N["data"]["test_min"] = round(float(y_test.values.min()), 2)
N["data"]["test_max"] = round(float(y_test.values.max()), 2)
N["data"]["test_neg_hours"] = int((y_test.values < 0).sum())

# hourly UTC series used by the foundation models (context = every hour before the delivery day)
ser = P["RO"].loc[:TEST_END.tz_localize("Europe/Brussels") + pd.Timedelta(days=1)]
ser_de, ser_hu = P["DE_LU"].reindex(ser.index), P["HU"].reindex(ser.index)
ser_lo = LD["load"].reindex(ser.index)

# ---------------------------------------------------------------- 2. stylised facts (charts 1-2)
dly = RO.mean(axis=1)
dde = DE.mean(axis=1)
yrs = sorted({d.year for d in RO.index})
N["facts"] = {"mean_2019": round(float(np.nanmean(RO.loc["2019"].values)), 2),
              "mean_2022": round(float(np.nanmean(RO.loc["2022"].values)), 2),
              "max_daily": round(float(dly.max()), 2), "max_daily_date": str(dly.idxmax().date()),
              "corr_ro_de_daily": round(float(dly.corr(dde)), 3)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(dde.index, dde.values, color=npl.PAL["benchmark"], lw=0.6, alpha=0.8, label=L(lang, "Germany-Luxembourg", "Germania-Luxemburg"))
    ax.plot(dly.index, dly.values, color=npl.PAL["data"], lw=0.6, label=L(lang, "Romania", "România"))
    ax.set_ylabel(L(lang, "Daily mean price (EUR/MWh)", "Prețul mediu zilnic (EUR/MWh)"))
    ax.set_xlim(dly.index.min(), dly.index.max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "price_history")
npl.chart_meta(N, "price_history", "power", npl.period(dly.index.min(), dly.index.max(), "en") + ", daily means of hourly prices",
               npl.period(dly.index.min(), dly.index.max(), "ro") + ", medii zilnice ale prețurilor orare")

prof_years = [2019, 2022, 2024, 2026]
prof = {y: RO.loc[str(y)].mean(axis=0) for y in prof_years}
N["facts"]["profile"] = {str(y): {"min_hour": int(prof[y].idxmin()), "min": round(float(prof[y].min()), 2),
                                  "max_hour": int(prof[y].idxmax()), "max": round(float(prof[y].max()), 2)} for y in prof_years}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for i, y in enumerate(prof_years):
        lab = f"{y}" + ("*" if y == 2026 else "")
        ax.plot(range(24), prof[y].values, marker="o", ms=2.5, color=npl.SERIES[[0, 7, 3, 1][i]], label=lab)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel(L(lang, "Delivery hour (CET)", "Ora de livrare (CET)"))
    ax.set_ylabel(L(lang, "Mean price (EUR/MWh)", "Prețul mediu (EUR/MWh)"))
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "hourly_profile")
npl.chart_meta(N, "hourly_profile", "power", "2019, 2022, 2024 and 2026 (*January-October), Romania",
               "2019, 2022, 2024 și 2026 (*ianuarie-octombrie), România")

# ---------------------------------------------------------------- 3. features for LEAR / DNN (information at gate closure on day D-1)
dates = RO.index


def feat_row(D):
    """Features for delivery day D: RO prices of D-1, D-2, D-3, D-7; DE-LU prices of D-1; RO load of D-2; day of week.
    All of them are known before the day-ahead auction for D closes (12:00 CET on D-1)."""
    parts = []
    for lag in (1, 2, 3, 7):
        dd = D - pd.Timedelta(days=lag)
        parts.append(RO.loc[dd].values if dd in RO.index else np.full(24, np.nan))
    dd = D - pd.Timedelta(days=1)
    parts.append(DE.loc[dd].values if dd in DE.index else np.full(24, np.nan))
    dd = D - pd.Timedelta(days=2)
    parts.append(LO.loc[dd].values if dd in LO.index else np.full(24, np.nan))
    dow = np.zeros(7)
    dow[D.dayofweek] = 1
    return np.concatenate(parts + [dow])


fpath = os.path.join(CACHE, "features.npz")
if os.path.isfile(fpath):
    z = np.load(fpath, allow_pickle=True)
    XF, YF, FD = z["X"], z["Y"], pd.DatetimeIndex(z["D"])
else:
    rows, ys, ds = [], [], []
    for D in dates:
        if D < pd.Timestamp("2016-01-01") or not full.get(D, False):
            continue
        x = feat_row(D)
        if np.isnan(x).any():
            continue
        rows.append(x)
        ys.append(RO.loc[D].values)
        ds.append(D)
    XF, YF, FD = np.array(rows), np.array(ys), pd.DatetimeIndex(ds)
    np.savez(fpath, X=XF, Y=YF, D=np.array(ds, dtype="datetime64[ns]"))
NPRICE = 24 * 5          # price columns (RO lags + DE) get the variance-stabilising transform
N["features"] = {"n": int(XF.shape[1]), "n_samples": int(len(FD))}


def vst_fit(y):
    med = np.median(y)
    mad = np.median(np.abs(y - med)) / 0.6745
    return med, max(mad, 1e-6)


def vst(y, med, mad):          # asinh transform after median/MAD normalisation (Uniejewski, Weron and Ziel, 2018)
    return np.arcsinh((y - med) / mad)


def ivst(z, med, mad):
    return np.sinh(z) * mad + med


def prep(Xw, Yw, Xt):
    """Transform prices (asinh), standardise load (z-score) with statistics of the training window only."""
    med, mad = vst_fit(Yw.ravel())
    A = Xw.copy()
    B = Xt.copy()
    A[:, :NPRICE] = vst(A[:, :NPRICE], med, mad)
    B[:, :NPRICE] = vst(B[:, :NPRICE], med, mad)
    mu = A[:, NPRICE:NPRICE + 24].mean()
    sd = A[:, NPRICE:NPRICE + 24].std()
    A[:, NPRICE:NPRICE + 24] = (A[:, NPRICE:NPRICE + 24] - mu) / sd
    B[:, NPRICE:NPRICE + 24] = (B[:, NPRICE:NPRICE + 24] - mu) / sd
    return A, vst(Yw, med, mad), B, (med, mad)


# ---------------------------------------------------------------- 4. naive benchmarks and LEAR (daily recalibration)
pos = {d: i for i, d in enumerate(FD)}
eval_days = [d for d in test_days if d in pos]
lear_days = [d for d in FD if FM_START <= d <= TEST_END]
lpath = os.path.join(CACHE, "lear.csv")
if os.path.isfile(lpath):
    LEAR = pd.read_csv(lpath, index_col=0, parse_dates=True)
else:
    from sklearn.linear_model import LassoLarsIC
    from joblib import Parallel, delayed

    def lear_day(D):
        i = pos[D]
        lo = np.searchsorted(FD.values, (D - pd.Timedelta(days=WIN)).to_datetime64())
        A, Z, B, (med, mad) = prep(XF[lo:i], YF[lo:i], XF[i:i + 1])
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1
        A, B = (A - mu) / sd, (B - mu) / sd
        out = []
        for h in range(24):
            m = LassoLarsIC(criterion="aic", max_iter=2500).fit(A, Z[:, h])
            out.append(float(m.predict(B)[0]))
        return ivst(np.array(out), med, mad)

    t = time.time()
    preds = Parallel(n_jobs=8)(delayed(lear_day)(D) for D in lear_days)
    LEAR = pd.DataFrame(np.array(preds), index=pd.DatetimeIndex(lear_days), columns=range(24))
    LEAR.to_csv(lpath)
    print(f"LEAR: {len(lear_days)} days x 24 hours in {time.time() - t:.0f} s")
LEAR.columns = range(24)
NAIVE = pd.DataFrame({D: (RO.loc[D - pd.Timedelta(days=7)].values if D.dayofweek in (0, 5, 6)
                          else RO.loc[D - pd.Timedelta(days=1)].values) for D in eval_days}).T
NAIVE.columns = range(24)
WEEK = pd.DataFrame({D: RO.loc[D - pd.Timedelta(days=7)].values for D in eval_days}).T
WEEK.columns = range(24)

# ---------------------------------------------------------------- 5. DNN (Lago et al. 2021 style): MLP, 24 outputs, monthly recalibration, 4 seeds
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DNN_SEEDS = [SEED, SEED + 1, SEED + 2, SEED + 3]


def inst_stats(Xa):
    """Level and scale of each sample from its own recent prices (instance normalisation, as in RevIN, Kim et al. 2022):
    m_d = median of the prices of day d-1; s_d = MAD of the 96 lagged Romanian prices / 0.6745 + 1 EUR/MWh."""
    P = Xa[:, :96]
    m = np.median(P[:, :24], 1, keepdims=True)
    s = np.median(np.abs(P - np.median(P, 1, keepdims=True)), 1, keepdims=True) / 0.6745 + 1.0
    return m, s


def prep_inst(Xw, Yw, Xt):
    """DNN inputs: price features and target transformed per sample, asinh((y - m_d) / s_d); then z-scores with window statistics."""
    out = []
    for Xa in (Xw, Xt):
        m, s_ = inst_stats(Xa)
        A_ = Xa.copy()
        A_[:, :NPRICE] = np.arcsinh((A_[:, :NPRICE] - m) / s_)
        out.append((A_, m, s_))
    (A_, mw, sw), (B_, mt, st) = out
    Z_ = np.arcsinh((Yw - mw) / sw)
    mu, sd = A_.mean(0), A_.std(0)
    sd[sd == 0] = 1
    return (A_ - mu) / sd, Z_, (B_ - mu) / sd, (mt, st)
HID = (256, 128)


def train_mlp(A, Z, seed, epochs=400, patience=30, lr=1e-3, drop=0.1, log=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(A)
    nv = int(0.2 * n)                     # chronological validation split: the last 20 % of the window
    Xtr, Ytr = torch.tensor(A[:-nv], dtype=torch.float32), torch.tensor(Z[:-nv], dtype=torch.float32)
    Xva, Yva = torch.tensor(A[-nv:], dtype=torch.float32), torch.tensor(Z[-nv:], dtype=torch.float32)
    net = nn.Sequential(nn.Linear(A.shape[1], HID[0]), nn.ReLU(), nn.Dropout(drop),
                        nn.Linear(HID[0], HID[1]), nn.ReLU(), nn.Dropout(drop), nn.Linear(HID[1], 24))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad, hist = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tl = 0.0
        for j in range(0, len(Xtr), 64):
            b = perm[j:j + 64]
            opt.zero_grad()
            loss = (net(Xtr[b]) - Ytr[b]).abs().mean()            # L1 loss on the transformed prices
            loss.backward()
            opt.step()
            tl += float(loss) * len(b)
        net.eval()
        with torch.no_grad():
            vl = float((net(Xva) - Yva).abs().mean())
        hist.append((tl / len(Xtr), vl))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, hist


dpath = os.path.join(CACHE, "dnn.csv")
hpath = os.path.join(CACHE, "dnn_hist.json")
if os.path.isfile(dpath):
    DNNall = pd.read_csv(dpath, index_col=[0, 1], parse_dates=[1])
    HIST = json.load(open(hpath))
else:
    t = time.time()
    torch.set_num_threads(8)
    months = pd.period_range(FM_START, TEST_END, freq="M")
    out, HIST = [], {}
    for mth in months:
        days_m = [d for d in lear_days if d.to_period("M") == mth]
        if not days_m:
            continue
        i0 = pos[days_m[0]]
        lo = np.searchsorted(FD.values, (days_m[0] - pd.Timedelta(days=WIN)).to_datetime64())
        idx = [pos[d] for d in days_m]
        A, Z, B, (med, mad) = prep_inst(XF[lo:i0], YF[lo:i0], XF[idx])
        for s in DNN_SEEDS:
            net, hist = train_mlp(A, Z, s)
            if mth == pd.Period(TEST_START, "M"):
                HIST[str(s)] = hist
            with torch.no_grad():
                pr = ivst(net(torch.tensor(B, dtype=torch.float32)).numpy(), med, mad)
            for d, row in zip(days_m, pr):
                out.append([s, d] + list(row))
    DNNall = pd.DataFrame(out, columns=["seed", "date"] + list(range(24))).set_index(["seed", "date"])
    DNNall.to_csv(dpath)
    json.dump(HIST, open(hpath, "w"))
    print(f"DNN: {len(months)} recalibrations x {len(DNN_SEEDS)} seeds in {time.time() - t:.0f} s")
DNNall.columns = range(24)
DNN = DNNall.groupby(level="date").mean()
N["dnn"] = {"hidden1": HID[0], "hidden2": HID[1], "n_seeds": len(DNN_SEEDS), "dropout": 0.1, "lr": 0.001, "batch": 64,
            "patience": 30, "n_params": int(XF.shape[1] * HID[0] + HID[0] + HID[0] * HID[1] + HID[1] + HID[1] * 24 + 24)}

# ---------------------------------------------------------------- 6. N-BEATS (generic), trained once on hourly windows before the test year
NB_IN, NB_OUT = 168, 24


class Block(nn.Module):
    def __init__(self, width=256, layers=4):
        super().__init__()
        mods, d = [], NB_IN
        for _ in range(layers):
            mods += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.fc = nn.Sequential(*mods)
        self.back = nn.Linear(width, NB_IN)        # backcast: the part of the input this block explains
        self.fore = nn.Linear(width, NB_OUT)       # forecast: this block's contribution to the output

    def forward(self, x):
        h = self.fc(x)
        return self.back(h), self.fore(h)


class NBeats(nn.Module):
    def __init__(self, n_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])

    def forward(self, x):
        res, f = x, 0
        for b in self.blocks:
            bc, fc = b(res)
            res = res - bc                          # doubly residual stacking (Oreshkin et al., 2020)
            f = f + fc
        return f


def nb_windows(arr_utc, ends):
    X, Y = [], []
    for e in ends:
        x = arr_utc[e - NB_IN:e]
        y = arr_utc[e:e + NB_OUT]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


def nb_norm(X):
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1.0
    return mu, sd


NB_SEEDS = [SEED, SEED + 1, SEED + 2]
nbpath = os.path.join(CACHE, "nbeats.csv")
nbhpath = os.path.join(CACHE, "nbeats_hist.json")
sv = ser.values.astype(float)
sidx = ser.index
if os.path.isfile(nbpath):
    NBall = pd.read_csv(nbpath, index_col=[0, 1], parse_dates=[1])
    NBH = json.load(open(nbhpath))
else:
    t = time.time()
    dev = npl.device()
    tr_end = int(np.searchsorted(sidx, (FM_START).tz_localize("Europe/Brussels")))
    tr_start = int(np.searchsorted(sidx, pd.Timestamp("2019-01-01").tz_localize("Europe/Brussels")))
    ends = np.arange(tr_start + NB_IN, tr_end - NB_OUT)
    X, Y = nb_windows(sv, ends)
    mu, sd = nb_norm(X)
    Xn, Yn = (X - mu) / sd, (Y - mu) / sd
    nv = int(0.1 * len(Xn))
    out, NBH = [], {}
    # forecast origins: midnight CET starting each evaluation day
    origins = []
    for D in lear_days:
        e = int(np.searchsorted(sidx, D.tz_localize("Europe/Brussels")))
        origins.append((D, e))
    for s in NB_SEEDS:
        torch.manual_seed(s)
        net = NBeats().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt, Yt = torch.tensor(Xn[:-nv], dtype=torch.float32), torch.tensor(Yn[:-nv], dtype=torch.float32)
        Xv, Yv = torch.tensor(Xn[-nv:], dtype=torch.float32).to(dev), torch.tensor(Yn[-nv:], dtype=torch.float32).to(dev)
        g = torch.Generator().manual_seed(s)
        best, bs, bad, hist = np.inf, None, 0, []
        for ep in range(60):
            net.train()
            perm = torch.randperm(len(Xt), generator=g)
            tl = 0.0
            for j in range(0, len(Xt), 512):
                b = perm[j:j + 512]
                xb, yb = Xt[b].to(dev), Yt[b].to(dev)
                opt.zero_grad()
                loss = (net(xb) - yb).abs().mean()
                loss.backward()
                opt.step()
                tl += float(loss) * len(b)
            net.eval()
            with torch.no_grad():
                vl = float((net(Xv) - Yv).abs().mean())
            hist.append((tl / len(Xt), vl))
            if vl < best - 1e-5:
                best, bad, bs = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
        net.load_state_dict(bs)
        net.eval()
        NBH[str(s)] = hist
        XX = np.array([sv[e - NB_IN:e] for _, e in origins])
        XX = pd.DataFrame(XX).ffill(axis=1).bfill(axis=1).values        # rare missing context hours: carried forward (input only)
        m_, s_ = nb_norm(XX)
        with torch.no_grad():
            pr = net(torch.tensor((XX - m_) / s_, dtype=torch.float32).to(dev)).cpu().numpy() * s_ + m_
        for (D, e), row in zip(origins, pr):
            out.append([s, D] + list(row))
    NBall = pd.DataFrame(out, columns=["seed", "date"] + list(range(24))).set_index(["seed", "date"])
    NBall.to_csv(nbpath)
    json.dump(NBH, open(nbhpath, "w"))
    N_nb_train = len(Xn) - nv
    print(f"N-BEATS: {len(NB_SEEDS)} seeds in {time.time() - t:.0f} s, {N_nb_train} training windows")
NBall.columns = range(24)
NBEATS = NBall.groupby(level="date").mean()
N["nbeats"] = {"n_blocks": 6, "width": 256, "layers": 4, "input": NB_IN, "output": NB_OUT, "n_seeds": len(NB_SEEDS),
               "n_params": int(sum(p.numel() for p in NBeats().parameters()))}

# ---------------------------------------------------------------- 7. foundation models, zero-shot: Chronos-2, Chronos-Bolt (base), TimesFM 2.5
QL = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]


def origins_for(days):
    return [(D, int(np.searchsorted(sidx, D.tz_localize("Europe/Brussels")))) for D in days]


def to_day(pred_steps, e):
    """Map the next 25 hourly steps after origin e to the 24 CET hours of the delivery day (autumn hour averaged)."""
    ts = sidx[e - 1] + pd.to_timedelta(np.arange(1, pred_steps.shape[0] + 1), unit="h")
    df = pd.DataFrame(pred_steps, index=ts)
    df = df[df.index.tz_localize(None).normalize() == (sidx[e - 1] + pd.Timedelta(hours=1)).tz_localize(None).normalize()]
    return df.groupby(df.index.hour).mean().reindex(range(24))


def run_fm(name, ctx=CTX, covariates=False, days=None):
    days = days or lear_days
    path = os.path.join(CACHE, f"fm_{name}_{ctx}{'_cov' if covariates else ''}.parquet")
    if os.path.isfile(path):
        return pd.read_parquet(path)
    t = time.time()
    org = origins_for(days)
    ctxs = [sv[max(0, e - ctx):e] for _, e in org]
    H = 25
    if name in ("chronos2", "bolt"):
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained({"chronos2": "amazon/chronos-2", "bolt": "amazon/chronos-bolt-base"}[name],
                                                   device_map=npl.device())
        qs = QL if name == "chronos2" else [q for q in QL if 0.1 <= q <= 0.9]
        allq = []
        for j in range(0, len(ctxs), 32):
            if covariates:
                inp = [{"target": c, "past_covariates": {"de": ser_de.values[max(0, e - ctx):e].astype(float),
                                                          "hu": ser_hu.values[max(0, e - ctx):e].astype(float),
                                                          "load": ser_lo.values[max(0, e - ctx):e].astype(float)}}
                       for c, (_, e) in zip(ctxs[j:j + 32], org[j:j + 32])]
            else:
                inp = [torch.tensor(c, dtype=torch.float32) for c in ctxs[j:j + 32]]
            q, _ = pipe.predict_quantiles(inp, prediction_length=H, quantile_levels=qs)
            q = q if isinstance(q, torch.Tensor) else torch.stack([x.squeeze(0) for x in q])
            allq.append(q.float().cpu().numpy())
        Q = np.concatenate(allq)                          # (n, H, nq)
    else:
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=ctx, max_horizon=32, normalize_inputs=True,
                                         use_continuous_quantile_head=True, fix_quantile_crossing=True))
        cc = [pd.Series(c).ffill().bfill().values for c in ctxs]       # TimesFM needs a complete context (input only)
        _, qf = m.forecast(horizon=H, inputs=cc)
        qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        Q = qf[:, :, 1:]                                  # column 0 = mean, then the deciles
    rows = []
    for (D, e), qd in zip(org, Q):
        for k, ql in enumerate(qs):
            day = to_day(qd[:, k:k + 1], e)[0].values
            rows.append([D, ql] + list(day))
    df = pd.DataFrame(rows, columns=["date", "q"] + [str(h) for h in range(24)])
    df.to_parquet(path)
    print(f"{name} ctx={ctx}{' +cov' if covariates else ''}: {len(org)} origins in {time.time() - t:.0f} s")
    return df


def qmat(df, q):
    x = df[np.isclose(df.q, q)].set_index("date")[[str(h) for h in range(24)]]
    x.columns = range(24)
    x.index = pd.DatetimeIndex(x.index)
    return x


FM = {"chronos2": run_fm("chronos2"), "bolt": run_fm("bolt"), "timesfm": run_fm("timesfm"),
      "chronos2_cov": run_fm("chronos2", covariates=True)}
CH2 = qmat(FM["chronos2"], 0.5)
BOLT = qmat(FM["bolt"], 0.5)
TFM = qmat(FM["timesfm"], 0.5)
CH2C = qmat(FM["chronos2_cov"], 0.5)
ENS = (LEAR + DNN.reindex(LEAR.index) + CH2.reindex(LEAR.index)) / 3

# ---------------------------------------------------------------- 8. point accuracy on the test year
MODELS = [("naive", NAIVE), ("lear", LEAR), ("dnn", DNN), ("nbeats", NBEATS), ("chronos2", CH2), ("chronos2_cov", CH2C),
          ("bolt", BOLT), ("timesfm", TFM), ("ens", ENS)]
NAME = {"naive": ("Naive (weekly/daily)", "Naiv (săptămînal/zilnic)"), "week": ("Same hour last week", "Aceeași oră, săptămîna trecută"),
        "lear": ("LEAR", "LEAR"), "dnn": ("DNN", "DNN"), "nbeats": ("N-BEATS", "N-BEATS"),
        "chronos2": ("Chronos-2", "Chronos-2"), "chronos2_cov": ("Chronos-2 + covariates", "Chronos-2 + covariabile"),
        "bolt": ("Chronos-Bolt", "Chronos-Bolt"), "timesfm": ("TimesFM 2.5", "TimesFM 2.5"),
        "ens": ("Ensemble LEAR+DNN+Chronos-2", "Ansamblu LEAR+DNN+Chronos-2")}
Y = RO.loc[eval_days]
ERR = {}
N["acc"] = {}
for k, M in MODELS:
    E = M.reindex(eval_days).values - Y.values
    assert not np.isnan(E).any(), k
    ERR[k] = E
    mae = float(np.abs(E).mean())
    N["acc"][k] = {"mae": round(mae, 2), "rmse": round(float(np.sqrt((E ** 2).mean())), 2)}
for k in N["acc"]:
    N["acc"][k]["rmae"] = round(N["acc"][k]["mae"] / N["acc"]["naive"]["mae"], 3)
best = min((k for k, _ in MODELS), key=lambda k: N["acc"][k]["mae"])
N["acc_best"] = best
N["acc_dnn_seeds"] = [round(float(np.abs(DNNall.xs(s, level="seed").reindex(eval_days).values - Y.values).mean()), 2) for s in DNN_SEEDS]
N["acc_dnn_seed_min"], N["acc_dnn_seed_max"] = min(N["acc_dnn_seeds"]), max(N["acc_dnn_seeds"])
N["acc_nb_seeds"] = [round(float(np.abs(NBall.xs(s, level="seed").reindex(eval_days).values - Y.values).mean()), 2) for s in NB_SEEDS]
N["acc_nb_seed_min"], N["acc_nb_seed_max"] = min(N["acc_nb_seeds"]), max(N["acc_nb_seeds"])
pd.DataFrame({k: np.abs(E).mean(1) for k, E in ERR.items()}, index=pd.DatetimeIndex(eval_days)).to_csv(
    os.path.join(RES, "daily_mae_test.csv"))

order = ["naive", "lear", "dnn", "nbeats", "bolt", "timesfm", "chronos2", "chronos2_cov", "ens"]
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    vals = [N["acc"][k]["mae"] for k in order]
    col = [npl.PAL["benchmark"] if k in ("naive", "lear") else (npl.PAL["model"] if k in ("dnn", "nbeats") else
           (npl.PAL["alt"] if k != "ens" else npl.PAL["c7"])) for k in order]
    ax.barh(range(len(order)), vals, color=col)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, (f"{v:.2f}" if lang == "en" else f"{v:.2f}".replace(".", ",")), va="center", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[k][0 if lang == "en" else 1] for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(L(lang, "Mean absolute error, test year (EUR/MWh)", "Eroarea absolută medie, anul de test (EUR/MWh)"))
    ax.set_xlim(0, max(vals) * 1.15)
    from matplotlib.patches import Patch
    h = [Patch(color=npl.PAL["benchmark"]), Patch(color=npl.PAL["model"]), Patch(color=npl.PAL["alt"]), Patch(color=npl.PAL["c7"])]
    npl.legend_bottom(fig, h, [L(lang, "Benchmarks", "Repere"), L(lang, "Networks trained here", "Rețele antrenate aici"),
                               L(lang, "Foundation models, zero-shot", "Modele fundație, fără reantrenare"),
                               L(lang, "Ensemble", "Ansamblu")], ncol=4)
    npl.out(fig, HERE, lang, "mae_models", also_x=[ax])
per_en = npl.period(eval_days[0], eval_days[-1], "en") + f", {len(eval_days)} days x 24 hours, Romania"
per_ro = npl.period(eval_days[0], eval_days[-1], "ro") + f", {len(eval_days)} de zile x 24 de ore, România"
npl.chart_meta(N, "mae_models", "power", per_en, per_ro)

# MAE by hour
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for k, c in [("naive", npl.PAL["benchmark"]), ("lear", npl.SERIES[5]), ("dnn", npl.PAL["model"]), ("chronos2", npl.PAL["alt"]),
                 ("ens", npl.PAL["c7"])]:
        ax.plot(range(24), np.abs(ERR[k]).mean(0), marker="o", ms=2.5, color=c, label=NAME[k][0 if lang == "en" else 1])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel(L(lang, "Delivery hour (CET)", "Ora de livrare (CET)"))
    ax.set_ylabel(L(lang, "MAE (EUR/MWh)", "MAE (EUR/MWh)"))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "mae_by_hour")
npl.chart_meta(N, "mae_by_hour", "power", per_en, per_ro)
mh = {k: np.abs(ERR[k]).mean(0) for k in ("lear", "dnn", "chronos2")}
N["byhour"] = {k: {"worst_hour": int(np.argmax(v)), "worst": round(float(v.max()), 2), "best_hour": int(np.argmin(v)),
                   "best": round(float(v.min()), 2)} for k, v in mh.items()}

# example week (last full week of the test period)
wk = [d for d in eval_days if d >= pd.Timestamp("2026-10-19") and d <= pd.Timestamp("2026-10-25")]
N["week"] = {"start": str(wk[0].date()), "end": str(wk[-1].date())}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ts = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in wk for h in range(24)])
    ax.plot(ts, RO.loc[wk].values.ravel(), color=npl.PAL["data"], lw=1.6, label=L(lang, "Actual price", "Prețul realizat"))
    ax.plot(ts, LEAR.loc[wk].values.ravel(), color=npl.PAL["benchmark"], lw=1.0, label="LEAR")
    ax.plot(ts, DNN.loc[wk].values.ravel(), color=npl.PAL["model"], lw=1.0, label="DNN")
    ax.plot(ts, CH2.loc[wk].values.ravel(), color=npl.PAL["alt"], lw=1.0, ls="--", label="Chronos-2")
    ax.set_ylabel(L(lang, "EUR/MWh", "EUR/MWh"))
    ax.set_xlim(ts[0], ts[-1] + pd.Timedelta(hours=1))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "forecast_week")
npl.chart_meta(N, "forecast_week", "power", npl.period(wk[0], wk[-1], "en") + ", hourly, Romania",
               npl.period(wk[0], wk[-1], "ro") + ", orar, România")
N["week"]["mae"] = {k: round(float(np.abs(M.loc[wk].values - RO.loc[wk].values).mean()), 2)
                    for k, M in [("lear", LEAR), ("dnn", DNN), ("chronos2", CH2)]}

# ---------------------------------------------------------------- 9. Diebold-Mariano tests (multivariate, Ziel and Weron 2018)
from scipy import stats  # noqa: E402

dm_models = ["naive", "lear", "dnn", "nbeats", "bolt", "timesfm", "chronos2", "chronos2_cov", "ens"]
PV = np.full((len(dm_models), len(dm_models)), np.nan)
for i, a in enumerate(dm_models):
    for j, b in enumerate(dm_models):
        if i == j:
            continue
        d = np.abs(ERR[a]).sum(1) - np.abs(ERR[b]).sum(1)          # daily loss differential, L1 norm over 24 hours
        dm = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        PV[i, j] = 1 - stats.norm.cdf(dm)                          # H1: column model b is more accurate than row model a
N["dm"] = {f"{a}__{b}": round(float(PV[i, j]), 4) for i, a in enumerate(dm_models) for j, b in enumerate(dm_models) if i != j}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("dm", ["#008300", "#eda100", "#e34948", "#f4f4f4"], N=256)
    M = np.where(np.isnan(PV), np.nan, np.clip(PV, 0, 0.1))
    im = ax.imshow(M, cmap=cm, vmin=0, vmax=0.1, aspect="auto")
    lab = [NAME[k][0 if lang == "en" else 1].replace("Ensemble LEAR+DNN+Chronos-2", "Ensemble").replace(
        "Ansamblu LEAR+DNN+Chronos-2", "Ansamblu").replace("Naive (weekly/daily)", "Naive").replace(
        "Naiv (săptămînal/zilnic)", "Naiv").replace(" + covariates", "+cov.").replace(" + covariabile", "+cov.") for k in dm_models]
    ax.set_xticks(range(len(dm_models)))
    ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(dm_models)))
    ax.set_yticklabels(lab, fontsize=8.5)
    ax.tick_params(length=0)
    ax.set_xlabel(L(lang, "Model B (column): is it better?", "Modelul B (coloana): este mai bun?"), fontsize=9)
    ax.set_ylabel(L(lang, "Model A (row)", "Modelul A (rîndul)"), fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(L(lang, "p-value", "valoarea p"), fontsize=9)
    cb.ax._keep_y = True
    if lang == "ro":
        from matplotlib.ticker import FuncFormatter
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: npl.ro_number(round(v, 3))))
    cb.ax.tick_params(labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    npl.out(fig, HERE, lang, "dm_pvalues", bottom=0.02)
npl.chart_meta(N, "dm_pvalues", "power", per_en, per_ro)
N["dm_n_days"] = len(eval_days)

# ---------------------------------------------------------------- 10. training curves (DNN first recalibration, N-BEATS)
for lang in ("en", "ro"):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3))
    h = np.array(HIST[str(SEED)])
    axs[0].plot(h[:, 0], color=npl.PAL["train"], label=L(lang, "Training", "Antrenare"))
    axs[0].plot(h[:, 1], color=npl.PAL["valid"], label=L(lang, "Validation", "Validare"))
    axs[0].set_xlabel(L(lang, "Epoch (DNN)", "Epoca (DNN)"))
    axs[0].set_ylabel(L(lang, "L1 loss (asinh scale)", "Pierderea L1 (scara asinh)"), fontsize=9.5)
    h = np.array(NBH[str(SEED)])
    axs[1].plot(h[:, 0], color=npl.PAL["train"])
    axs[1].plot(h[:, 1], color=npl.PAL["valid"])
    axs[1].set_xlabel(L(lang, "Epoch (N-BEATS)", "Epoca (N-BEATS)"))
    axs[1].set_ylabel(L(lang, "L1 loss (window scale)", "Pierderea L1 (scara ferestrei)"), fontsize=9.5)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "training_curves")
npl.chart_meta(N, "training_curves", "power", "training windows before 1 Nov 2025, Romania",
               "ferestre de antrenare dinainte de 1 noiembrie 2025, România")
hd = np.array(HIST[str(SEED)])
hn = np.array(NBH[str(SEED)])
N["train"] = {"dnn_epochs": int(len(hd)), "dnn_best_epoch": int(np.argmin(hd[:, 1]) + 1), "nb_epochs": int(len(hn)),
              "nb_best_epoch": int(np.argmin(hn[:, 1]) + 1)}

# ---------------------------------------------------------------- 11. context length of Chronos-2
CTXS = [168, 512, 2048, 8192]
ctx_mae = {}
for c in CTXS:
    df = FM["chronos2"] if c == CTX else run_fm("chronos2", ctx=c, days=eval_days)
    ctx_mae[c] = float(np.abs(qmat(df, 0.5).reindex(eval_days).values - Y.values).mean())
N["ctx"] = {str(c): round(v, 2) for c, v in ctx_mae.items()}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.plot(range(len(CTXS)), [ctx_mae[c] for c in CTXS], marker="o", color=npl.PAL["alt"], label="Chronos-2")
    ax.axhline(N["acc"]["lear"]["mae"], color=npl.PAL["benchmark"], ls="--", lw=1, label=L(lang, "LEAR (3-year window)", "LEAR (fereastră de 3 ani)"))
    ax.set_xticks(range(len(CTXS)))
    ax.set_xticklabels([f"{c} h\n({c / 24:.0f} " + L(lang, "days)", "zile)") for c in CTXS])
    ax.set_xlabel(L(lang, "Context length given to the model", "Lungimea contextului primit de model"))
    ax.set_ylabel(L(lang, "MAE (EUR/MWh)", "MAE (EUR/MWh)"))
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "context_length")
npl.chart_meta(N, "context_length", "power", per_en, per_ro)

# ---------------------------------------------------------------- 12. probabilistic forecasts: pinball, coverage, split-conformal correction
def pinball(qpred, y, q):
    u = y - qpred
    return np.maximum(q * u, (q - 1) * u)


DEC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# LEAR + empirical errors (historical simulation of the last CAL_DAYS days of LEAR errors, per hour)
lear_err = LEAR - RO.reindex(LEAR.index)


def lear_q(D, q):
    past = lear_err.loc[D - pd.Timedelta(days=CAL_DAYS):D - pd.Timedelta(days=1)].dropna()
    return LEAR.loc[D].values - np.quantile(past.values, 1 - q, axis=0)


prob = {}
for k in ("chronos2", "bolt", "timesfm", "chronos2_cov"):
    prob[k] = float(np.mean([pinball(qmat(FM[k], q).reindex(eval_days).values, Y.values, q).mean() for q in DEC]))
prob["lear_hs"] = float(np.mean([pinball(np.array([lear_q(D, q) for D in eval_days]), Y.values, q).mean() for q in DEC]))
N["pinball"] = {k: round(v, 2) for k, v in prob.items()}


def interval(model, alpha, conformal):
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    if model == "lear_hs":
        lo = np.array([lear_q(D, lo_q) for D in eval_days])
        hi = np.array([lear_q(D, hi_q) for D in eval_days])
        return lo, hi
    qlo, qhi = qmat(FM[model], lo_q), qmat(FM[model], hi_q)
    lo, hi = qlo.reindex(eval_days).values.copy(), qhi.reindex(eval_days).values.copy()
    if conformal:   # split-conformal quantile regression (Romano, Patterson and Candès, 2019), rolling calibration window
        yy = RO.reindex(qlo.index)
        score = np.maximum(qlo - yy, yy - qhi)
        for i, D in enumerate(eval_days):
            s = score.loc[D - pd.Timedelta(days=CAL_DAYS):D - pd.Timedelta(days=1)].values.ravel()
            s = s[~np.isnan(s)]
            n = len(s)
            Qc = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
            lo[i] -= Qc
            hi[i] += Qc
    return lo, hi


N["cov"] = {}
ALPHAS = [0.5, 0.2, 0.1]
for model, conf in [("chronos2", False), ("chronos2", True), ("lear_hs", False), ("timesfm", False), ("timesfm", True)]:
    key = model + ("_conf" if conf else "")
    N["cov"][key] = {}
    for a in ALPHAS:
        if model == "timesfm" and a != 0.2:
            continue                                  # TimesFM gives deciles only: the 80 % interval (0.1 to 0.9)
        lo, hi = interval(model, a, conf)
        cov = float(((Y.values >= lo) & (Y.values <= hi)).mean())
        N["cov"][key][str(int(100 * (1 - a)))] = {"cov": round(100 * cov, 1), "width": round(float((hi - lo).mean()), 2)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    keys = [("chronos2", L(lang, "Chronos-2, raw quantiles", "Chronos-2, cuantile brute"), npl.PAL["alt"]),
            ("chronos2_conf", L(lang, "Chronos-2 + conformal", "Chronos-2 + conformal"), npl.PAL["model"]),
            ("lear_hs", L(lang, "LEAR + empirical errors", "LEAR + erori empirice"), npl.PAL["benchmark"])]
    w = 0.26
    for i, (k, lab, c) in enumerate(keys):
        ax.bar(np.arange(3) + (i - 1) * w, [N["cov"][k][str(int(100 * (1 - a)))]["cov"] for a in ALPHAS], w, color=c, label=lab)
    for j, a in enumerate(ALPHAS):
        ax.plot([j - 1.6 * w, j + 1.6 * w], [100 * (1 - a)] * 2, color="#222222", lw=1.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([L(lang, f"nominal {int(100 * (1 - a))} %", f"nominal {int(100 * (1 - a))} %") for a in ALPHAS])
    ax.set_ylabel(L(lang, "Empirical coverage (%)", "Acoperirea empirică (%)"))
    ax.set_ylim(0, 100)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "coverage")
npl.chart_meta(N, "coverage", "power", per_en, per_ro)

# fan chart for the example week
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ts = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in wk for h in range(24)])
    for (a, b), al in [((0.05, 0.95), 0.18), ((0.1, 0.9), 0.28), ((0.25, 0.75), 0.4)]:
        ax.fill_between(ts, qmat(FM["chronos2"], a).loc[wk].values.ravel(), qmat(FM["chronos2"], b).loc[wk].values.ravel(),
                        color=npl.PAL["alt"], alpha=al, lw=0, label=L(lang, f"{int(round(100 * (b - a)))} % interval", f"interval de {int(round(100 * (b - a)))} %"))
    ax.plot(ts, CH2.loc[wk].values.ravel(), color=npl.PAL["alt"], lw=1.0, label=L(lang, "Median", "Mediana"))
    ax.plot(ts, RO.loc[wk].values.ravel(), color=npl.PAL["data"], lw=1.4, label=L(lang, "Actual", "Realizat"))
    ax.set_ylabel("EUR/MWh")
    ax.set_xlim(ts[0], ts[-1] + pd.Timedelta(hours=1))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    npl.legend_bottom(fig, ncol=5)
    npl.out(fig, HERE, lang, "chronos_fan")
npl.chart_meta(N, "chronos_fan", "power", npl.period(wk[0], wk[-1], "en") + ", hourly, Romania",
               npl.period(wk[0], wk[-1], "ro") + ", orar, România")

# ---------------------------------------------------------------- 13. Romanian load: seasonal naive vs foundation models
lo_days = [d for d in eval_days if d in LO.index and LO.loc[d].notna().all() and (d - pd.Timedelta(days=7)) in LO.index
           and LO.loc[d - pd.Timedelta(days=7)].notna().all()]
lpath2 = os.path.join(CACHE, "load_fm.parquet")
if os.path.isfile(lpath2):
    LF = pd.read_parquet(lpath2)
else:
    from chronos import BaseChronosPipeline
    t = time.time()
    lv = ser_lo.values.astype(float)
    org = origins_for(lo_days)
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=npl.device())
    qs = [0.5]
    outs = []
    for j in range(0, len(org), 32):
        inp = [torch.tensor(lv[max(0, e - CTX):e], dtype=torch.float32) for _, e in org[j:j + 32]]
        q, _ = pipe.predict_quantiles(inp, prediction_length=25, quantile_levels=qs)
        q = torch.stack([x.squeeze(0) for x in q]) if not isinstance(q, torch.Tensor) else q
        outs.append(q.float().cpu().numpy())
    Qc = np.concatenate(outs)
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=32, normalize_inputs=True,
                                     use_continuous_quantile_head=True, fix_quantile_crossing=True))
    _, qf = m.forecast(horizon=25, inputs=[pd.Series(lv[max(0, e - CTX):e]).ffill().bfill().values for _, e in org])
    rows = []
    for (D, e), a, b in zip(org, Qc, qf):
        rows.append([D, "chronos2"] + list(to_day(a[:, :1], e)[0].values))
        rows.append([D, "timesfm"] + list(to_day(b[:, 5:6], e)[0].values))
    LF = pd.DataFrame(rows, columns=["date", "model"] + [str(h) for h in range(24)])
    LF.to_parquet(lpath2)
    print(f"load forecasts in {time.time() - t:.0f} s")
LY = LO.loc[lo_days].values
N["load"] = {"n_days": len(lo_days), "mean_mw": round(float(LY.mean()), 0)}
lpred = {"week": LO.loc[[d - pd.Timedelta(days=7) for d in lo_days]].values}
for mname in ("chronos2", "timesfm"):
    x = LF[LF.model == mname].set_index("date")[[str(h) for h in range(24)]]
    x.index = pd.DatetimeIndex(x.index)
    lpred[mname] = x.reindex(lo_days).values
for k, v in lpred.items():
    N["load"][k] = {"mae": round(float(np.abs(v - LY).mean()), 1), "mape": round(float(100 * np.abs((v - LY) / LY).mean()), 2)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ks = ["week", "timesfm", "chronos2"]
    labs = [L(lang, "Same hour last week", "Aceeași oră, săptămîna trecută"), "TimesFM 2.5", "Chronos-2"]
    vals = [N["load"][k]["mape"] for k in ks]
    ax.barh(range(3), vals, color=[npl.PAL["benchmark"], npl.PAL["alt"], npl.PAL["alt"]])
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, (f"{v:.2f}" if lang == "en" else f"{v:.2f}".replace(".", ",")) + " %", va="center", fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.2)
    ax.set_xlabel(L(lang, "Mean absolute percentage error of the hourly load (%)", "Eroarea procentuală absolută medie a consumului orar (%)"))
    from matplotlib.patches import Patch
    npl.legend_bottom(fig, [Patch(color=npl.PAL["benchmark"]), Patch(color=npl.PAL["alt"])],
                      [L(lang, "Seasonal naive benchmark", "Reper naiv sezonier"), L(lang, "Foundation models, zero-shot", "Modele fundație, fără reantrenare")])
    npl.out(fig, HERE, lang, "load_mape", also_x=[ax])
npl.chart_meta(N, "load_mape", "power", npl.period(lo_days[0], lo_days[-1], "en") + f", {len(lo_days)} days, Romanian load",
               npl.period(lo_days[0], lo_days[-1], "ro") + f", {len(lo_days)} de zile, consumul României")

# ---------------------------------------------------------------- 14. Value-at-Risk with a foundation model: S&P 500 and Bitcoin
from arch import arch_model  # noqa: E402

PX = npl.markets("adjclose", ["SP500", "BTCUSD"])
N["var"] = {}


def kupiec(hits, a):
    n, x = len(hits), int(hits.sum())
    p = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - a)
    else:
        lr = -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - p) - x * np.log(p))
    return float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(hits):
    h = hits.astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(h[:-1], h[1:]):
        n00 += (a == 0) & (b == 0); n01 += (a == 0) & (b == 1); n10 += (a == 1) & (b == 0); n11 += (a == 1) & (b == 1)
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(pr, a, b):
        return (a * np.log(1 - pr) if a and pr < 1 else 0) + (b * np.log(pr) if b and pr > 0 else 0)
    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(1 - stats.chi2.cdf(lr, 1))


def tick(q, r, a):
    return float(np.mean(np.maximum(a * (r - q), (a - 1) * (r - q))))


VAR_RES = {}
for tic, ctxv in [("SP500", 1024), ("BTCUSD", 1024)]:
    px = PX[tic].dropna()
    r = 100 * np.log(px).diff().dropna()
    r = r.loc[:VAR_END]
    days = r.loc[VAR_START:VAR_END].index
    cal_start = r.index[r.index.get_loc(days[0]) - 250]
    fdays = r.loc[cal_start:VAR_END].index
    path = os.path.join(CACHE, f"var_{tic}.parquet")
    if os.path.isfile(path):
        V = pd.read_parquet(path)
    else:
        t = time.time()
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=npl.device())
        ii = [r.index.get_loc(d) for d in fdays]
        ctxs = [torch.tensor(r.values[max(0, i - ctxv):i], dtype=torch.float32) for i in ii]
        outs = []
        for j in range(0, len(ctxs), 64):
            q, _ = pipe.predict_quantiles(ctxs[j:j + 64], prediction_length=1, quantile_levels=[0.01, 0.05])
            q = torch.stack([x.squeeze(0) for x in q]) if not isinstance(q, torch.Tensor) else q
            outs.append(q.float().cpu().numpy()[:, 0, :])
        Qv = np.concatenate(outs)
        V = pd.DataFrame({"r": r.loc[fdays].values, "ch_01": Qv[:, 0], "ch_05": Qv[:, 1]}, index=fdays)
        # historical simulation, 250 days
        V["hs_01"] = [np.quantile(r.values[i - 250:i], 0.01) for i in ii]
        V["hs_05"] = [np.quantile(r.values[i - 250:i], 0.05) for i in ii]
        # GARCH(1,1) with Student-t errors, 1000-day window, re-estimated every 21 trading days
        g01, g05, params = [], [], None
        for n_, i in enumerate(ii):
            win = r.values[i - 1000:i]
            am = arch_model(win, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
            if n_ % 21 == 0 or params is None:
                params = am.fit(disp="off", show_warning=False).params
            f = am.fix(params).forecast(horizon=1, reindex=False)
            mu, s2 = float(f.mean.values[-1, 0]), float(f.variance.values[-1, 0])
            nu = float(params["nu"])
            sc = np.sqrt(s2 * (nu - 2) / nu)
            g01.append(mu + sc * stats.t.ppf(0.01, nu))
            g05.append(mu + sc * stats.t.ppf(0.05, nu))
        V["garch_01"], V["garch_05"] = g01, g05
        V.index.name = "date"
        V.to_parquet(path)
        print(f"VaR {tic}: {len(fdays)} days in {time.time() - t:.0f} s")
    # one-sided split-conformal shift of the Chronos-2 quantile, rolling 250-day calibration window
    for a, s in ((0.01, "01"), (0.05, "05")):
        sc = (V[f"ch_{s}"] - V["r"]).values
        adj = np.full(len(V), np.nan)
        for i in range(250, len(V)):
            w = sc[i - 250:i]
            n = len(w)
            adj[i] = V[f"ch_{s}"].values[i] - np.quantile(w, min(1.0, np.ceil((n + 1) * (1 - a)) / n))
        V[f"chc_{s}"] = adj
    T = V.loc[VAR_START:VAR_END]
    VAR_RES[tic] = T
    T.to_csv(os.path.join(RES, f"var_{tic}.csv"))
    N["var"][tic] = {"n": int(len(T)), "first": str(T.index[0].date()), "last": str(T.index[-1].date())}
    for m in ("hs", "garch", "ch", "chc"):
        for a, s in ((0.01, "01"), (0.05, "05")):
            hits = (T["r"] < T[f"{m}_{s}"]).values
            N["var"][tic][f"{m}_{s}"] = {"viol": int(hits.sum()), "rate": round(100 * hits.mean(), 2),
                                         "kupiec_p": round(kupiec(hits, a), 3), "chris_p": round(christoffersen(hits), 3),
                                         "tick": round(tick(T[f"{m}_{s}"].values, T["r"].values, a), 4)}
    N["var"][tic]["exp_01"] = round(0.01 * len(T), 1)
    N["var"][tic]["exp_05"] = round(0.05 * len(T), 1)

T = VAR_RES["SP500"]
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.plot(T.index, T["r"], color=npl.PAL["data"], lw=0.45, label=L(lang, "Daily log return (%)", "Randamentul logaritmic zilnic (%)"))
    ax.plot(T.index, T["garch_01"], color=npl.PAL["benchmark"], lw=0.9, label=L(lang, "1 % VaR, GARCH(1,1)-t", "VaR 1 %, GARCH(1,1)-t"))
    ax.plot(T.index, T["chc_01"], color=npl.PAL["model"], lw=0.9, label=L(lang, "1 % VaR, Chronos-2 + conformal", "VaR 1 %, Chronos-2 + conformal"))
    hit = T["r"] < T["chc_01"]
    ax.scatter(T.index[hit], T["r"][hit], s=10, color=npl.PAL["positive"], zorder=3, label=L(lang, "Violation (Chronos-2 + conformal)", "Depășire (Chronos-2 + conformal)"))
    ax.set_ylabel("%")
    ax.set_xlim(T.index[0], T.index[-1])
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "var_sp500")
npl.chart_meta(N, "var_sp500", "yahoo", npl.period(T.index[0], T.index[-1], "en") + ", trading days, S\\&P 500",
               npl.period(T.index[0], T.index[-1], "ro") + ", zile de tranzacționare, S\\&P 500")

# ---------------------------------------------------------------- 15. per-hour results for the Quantlets (small CSV files, committed)
rows = []
for D in eval_days:
    for h in range(24):
        rows.append([D.date(), h, RO.loc[D, h]] + [M.loc[D, h] for _, M in MODELS] +
                    [qmat(run_fm("chronos2", ctx=c, days=eval_days) if c != CTX else FM["chronos2"], 0.5).loc[D, h] for c in CTXS])
pd.DataFrame(rows, columns=["date", "hour", "actual"] + [k for k, _ in MODELS] + [f"chronos2_ctx{c}" for c in CTXS]).to_csv(
    os.path.join(RES, "forecasts_hourly.csv"), index=False)
qcols = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
QQ = {q: qmat(FM["chronos2"], q) for q in qcols}
TQ = {q: qmat(FM["timesfm"], q) for q in (0.1, 0.9)}
rows = []
for D in QQ[0.5].index:
    if D not in RO.index:
        continue
    for h in range(24):
        rows.append([D.date(), h, RO.loc[D, h], LEAR.loc[D, h]] + [QQ[q].loc[D, h] for q in qcols] +
                    [TQ[q].loc[D, h] for q in TQ])
pd.DataFrame(rows, columns=["date", "hour", "actual", "lear"] + [f"ch2_q{int(round(100 * q)):02d}" for q in qcols] +
             [f"tfm_q{int(round(100 * q)):02d}" for q in TQ]).to_csv(os.path.join(RES, "quantiles_hourly.csv"), index=False)
pd.DataFrame({"date": [d.date() for d in lo_days], **{f"h{h}": LO.loc[lo_days, h].values for h in range(24)}}).to_csv(
    os.path.join(RES, "load_actual.csv"), index=False)
pd.concat({k: pd.DataFrame(v, index=[d.date() for d in lo_days]) for k, v in lpred.items()}, names=["model", "date"]).to_csv(
    os.path.join(RES, "load_forecasts.csv"))

json.dump({"dnn": HIST[str(SEED)], "nbeats": NBH[str(SEED)]}, open(os.path.join(RES, "training_hist.json"), "w"))

# ---------------------------------------------------------------- 16. save numbers
N["runtime_min"] = round((time.time() - T0) / 60, 1)
with open(os.path.join(HERE, "L13_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1)
print("wrote L13_numbers.json and", len(N["charts"]), "charts x 2 languages;", f"{N['runtime_min']} min")
print(json.dumps(N["acc"], indent=0))
