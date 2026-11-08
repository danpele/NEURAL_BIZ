"""Inputs of the Lecture 5 interactive page (interactive/L5_data.js), from data/clean and the saved runs of analysis_L5.py
(lectures/L5/cache/*.pkl), with the same data preparation, splits and benchmarks (HAR, GARCH(1,1), ridge ARX). Nothing is
retrained. Result cached in lectures/L5/cache/interactive_v1.json.
Run alone to refresh the cache:  python3 lectures/L5/interactive_data_L5.py --fresh
"""
import json
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
CDIR = os.path.join(HERE, "cache")
CACHE = os.path.join(CDIR, "interactive_v1.json")


def compute():
    import numpy as np
    import pandas as pd
    import nn_plot as npl
    out = {}
    pk = lambda n: pickle.load(open(os.path.join(CDIR, n + ".pkl"), "rb"))  # noqa: E731

    # ---------------------------------------------------------------- S&P 500, as analysis_L5.py
    cols = {f: npl.markets(f, ["SP500"])["SP500"] for f in ["open", "high", "low", "close", "adjclose"]}
    spx = pd.DataFrame(cols).dropna()
    END = (spx.index.max() + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.Timedelta(days=1)
    spx = spx.loc[:END]
    spx = spx[(spx.high > spx.low)]
    gk = 0.5 * np.log(spx.high / spx.low) ** 2 - (2 * np.log(2) - 1) * np.log(spx.close / spx.open) ** 2
    spx["gk"] = gk.clip(lower=1e-8)
    spx["y"] = np.log(spx["gk"])
    spx["r"] = 100 * np.log(spx.adjclose).diff()
    spx = spx.dropna()
    TR_END, VA_END = pd.Timestamp("2016-12-31"), pd.Timestamp("2019-12-31")
    W = 22
    y = spx.y.values.astype(np.float32)
    idx_dates = spx.index
    T_ = np.arange(W - 1, len(y) - 1)
    tdate = idx_dates[T_ + 1]
    s_tr, s_te = tdate <= TR_END, tdate > VA_END
    y_true = y[T_ + 1]
    naive = y[T_]
    d5 = pd.Series(y).rolling(5).mean().values
    d22 = pd.Series(y).rolling(22).mean().values
    H = np.column_stack([np.ones(len(T_)), y[T_], d5[T_], d22[T_]])
    beta = np.linalg.lstsq(H[s_tr], y_true[s_tr], rcond=None)[0]
    har = H @ beta
    from arch import arch_model
    r_all = spx.r.values
    ntr = int((idx_dates <= TR_END).sum())
    am = arch_model(r_all, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    gres = am.fit(last_obs=ntr, disp="off")
    fc = am.fix(gres.params).forecast(horizon=1, start=0, reindex=True)
    sig2 = fc.variance.values[:, 0]
    lg = np.log(sig2[T_] / 1e4)
    mz = np.linalg.lstsq(np.column_stack([np.ones(s_tr.sum()), lg[s_tr]]), y_true[s_tr], rcond=None)[0]
    garch = mz[0] + mz[1] * lg
    VOL = pk("vol")
    r4 = lambda a: [round(float(v), 4) for v in a]  # noqa: E731
    out["vf"] = {"d": [str(d.date()) for d in tdate[s_te]], "obs": r4(y_true[s_te]), "naive": r4(naive[s_te]), "har": r4(har[s_te]),
                 "garch": r4(garch[s_te]), **{c: r4(VOL[c]["pred"].mean(0)[s_te]) for c in ("rnn", "lstm", "gru")}}

    # daily returns (%) for the autocorrelation block: S&P 500 and Banca Transilvania (TLV)
    tlv = npl.bvb("adjclose", ["TLV"])["TLV"].dropna().loc[:END]
    tlv = tlv[tlv.index >= "2008-01-01"]
    tlv = tlv[tlv.diff() != 0]
    r_tlv = (100 * np.log(tlv).diff()).dropna()
    out["ac"] = {"SP500": [round(float(v), 3) for v in spx.r.values], "TLV": [round(float(v), 3) for v in r_tlv.values],
                 "range": {"SP500": [str(spx.index.min().date()), str(spx.index.max().date())],
                           "TLV": [str(r_tlv.index.min().date()), str(r_tlv.index.max().date())]}}

    # returns: test-period targets, AR(1) and training-mean benchmarks, LSTM forecasts (5 seeds)
    RT = pk("ret")
    out["rt"] = {}
    for name in ("SP500", "TLV"):
        o = RT[name]
        yv, mtr, mte = o["y"], o["mtr"], o["mte"]
        ar = np.polyfit(o["lag"][mtr], yv[mtr], 1)
        out["rt"][name] = {"d": [str(pd.Timestamp(d).date()) for d in o["td"][mte]], "y": r4(yv[mte]),
                           "ar": r4(np.polyval(ar, o["lag"][mte])), "mean": round(float(yv[mtr].mean()), 5),
                           "lstm": [r4(p[mte]) for p in o["pred"]], "ar_coef": [round(float(v), 4) for v in ar]}

    # ---------------------------------------------------------------- Romanian load, day-ahead (analysis_L5.py, section C)
    ld = npl.load_generation_ro(tz="UTC")["load"]
    ld = ld[(ld.index >= "2016-01-01") & (ld.index < (END + pd.Timedelta(days=1)).tz_localize("UTC"))]
    days = pd.date_range(ld.index.min().normalize(), ld.index.max().normalize(), freq="D", tz="UTC")
    full = ld.reindex(pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h", tz="UTC"))
    M = full.values.reshape(-1, 24)
    dd = pd.DatetimeIndex(days[:len(M)])
    LTR, LVA = pd.Timestamp("2022-12-31", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
    lmu, lsd = np.nanmean(M[dd <= LTR]), np.nanstd(M[dd <= LTR])

    def load_samples(win):
        nd = win // 24
        X, Y, S, D_, NV, YD = [], [], [], [], [], []
        flat = M.reshape(-1)
        for d in range(nd + 7, len(M) - 1):
            x = flat[(d + 1) * 24 - win:(d + 1) * 24]
            yy = M[d + 1]
            if np.isnan(x).any() or np.isnan(yy).any() or np.isnan(M[d - 6]).any():
                continue
            tgt = dd[d + 1]
            dows = np.zeros((nd, 7)); dows[np.arange(nd), [(dd[d - nd + 1 + j]).dayofweek for j in range(nd)]] = 1
            X.append(np.concatenate([((x - lmu) / lsd).reshape(nd, 24), dows], 1))
            dow = np.zeros(7); dow[tgt.dayofweek] = 1
            doy = tgt.dayofyear / 365.25
            S.append(np.concatenate([dow, [np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)]]))
            Y.append((yy - M[d - 6]) / lsd)
            D_.append(tgt)
            NV.append(M[d - 6])
            YD.append(M[d])
        return (np.array(X, np.float32), np.array(Y, np.float32), np.array(S, np.float32), pd.DatetimeIndex(D_),
                np.array(NV), np.array(YD))

    X, Y, S, D_, NV, YD = load_samples(168)
    mtr, mva, mte = D_ <= LTR, (D_ > LTR) & (D_ <= LVA), D_ > LVA
    from sklearn.linear_model import Ridge
    Z = np.column_stack([X[:, :, :24].reshape(len(X), -1), S])
    best = None
    for a in [0.1, 1, 10, 100]:
        rg = Ridge(alpha=a).fit(Z[mtr], Y[mtr])
        v = np.mean((rg.predict(Z[mva]) - Y[mva]) ** 2)
        if best is None or v < best[0]:
            best = (v, a, rg)
    ridge = best[2].predict(Z) * lsd + NV
    Ytrue = Y * lsd + NV
    ri = lambda a: np.round(a[mte]).astype(int).tolist()  # noqa: E731
    out["ld"] = {"d": [str(d.date()) for d in D_[mte]], "obs": ri(Ytrue), "snaive": ri(NV), "ridge": ri(ridge),
                 **{c: ri(pk(f"load_{c}")["pred"].mean(0)) for c in ("lstm", "gru")}, "alpha": best[1]}
    return out


def load(fresh=False):
    if os.path.exists(CACHE) and not fresh:
        return json.load(open(CACHE))
    r = compute()
    json.dump(r, open(CACHE, "w"), separators=(",", ":"))
    return r


if __name__ == "__main__":
    import numpy as np
    d = load("--fresh" in sys.argv)
    v = d["vf"]
    o = np.array(v["obs"])
    for k in ("naive", "har", "garch", "rnn", "lstm", "gru"):
        print(k, round(float(np.mean((np.array(v[k]) - o) ** 2)), 4))
    L_ = d["ld"]
    o = np.array(L_["obs"], float)
    for k in ("snaive", "ridge", "lstm", "gru"):
        print(k, round(float(np.mean(np.abs(np.array(L_[k]) - o) / o) * 100), 2))
    for n, r in d["rt"].items():
        y = np.array(r["y"])
        print(n, len(y), "R2 ar", round(100 * (1 - np.sum((y - r["ar"]) ** 2) / np.sum(y ** 2)), 3),
              "lstm", [round(100 * (1 - np.sum((y - np.array(p)) ** 2) / np.sum(y ** 2)), 3) for p in r["lstm"]])
    print(len(d["ac"]["SP500"]), len(d["ac"]["TLV"]), d["ld"]["d"][0], d["ld"]["d"][-1], len(d["ld"]["d"]))
