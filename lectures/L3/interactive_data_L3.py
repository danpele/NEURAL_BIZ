"""Inputs of the Lecture 3 interactive page (interactive/L3_data.js), from data/clean and the caches of analysis_L3.py, with
the same splits, models and seeds. New runs (learning curves for a grid of dropout x weight decay on German credit, the
early-stopping curve on card fraud, test scores of the fraud models, hold-out AUCs for several test shares) are cached in
lectures/L3/cache/interactive_v1.json.   Run alone to refresh the cache:  python3 lectures/L3/interactive_data_L3.py --fresh
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
CDIR = os.path.join(HERE, "cache")
CACHE = os.path.join(CDIR, "interactive_v1.json")
DPS = [0.0, 0.3, 0.5]
WDS = [0.0, 0.1, 1.0]
SHARES = [0.1, 0.2, 0.3, 0.5]


def compute():
    import numpy as np
    import nn_plot as npl
    import xgboost as xgb                                    # before torch, as in analysis_L3.py
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    torch.set_num_threads(1)
    SEED = npl.seed_all()

    class Net(nn.Module):
        def __init__(self, d, hidden=(128, 128), p=0.0, bn=False):
            super().__init__()
            layers, k = [], d
            for h in hidden:
                layers.append(nn.Linear(k, h))
                if bn:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                if p > 0:
                    layers.append(nn.Dropout(p))
                k = h
            layers.append(nn.Linear(k, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def proba(m, X):
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(X))).numpy().ravel()

    def fit(Xtr, ytr, Xva, yva, hidden=(128, 128), p=0.0, wd=0.0, lr=1e-3, epochs=200, bs=128, seed=0, early=True, patience=20,
            record=False, metric=None):
        """fit() of analysis_L3.py (AdamW, early stopping on the validation loss); `metric(m)` is recorded after each epoch."""
        torch.manual_seed(SEED + seed)
        rng = np.random.default_rng(SEED + seed)
        m = Net(Xtr.shape[1], hidden, p, False)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        lf = nn.BCEWithLogitsLoss()
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
        Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32).view(-1, 1)
        best, state, bad, hist = np.inf, None, 0, []
        for ep in range(epochs):
            m.train()
            idx = rng.permutation(len(Xt))
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                opt.zero_grad()
                lf(m(Xt[b]), yt[b]).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                vl = float(lf(m(Xv), yv))
                tl = float(lf(m(Xt), yt)) if record else np.nan
            hist.append([tl, vl] + ([metric(m)] if metric else []))
            if not np.isfinite(vl):
                break
            if vl < best - 1e-5:
                best, bad = vl, 0
                state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if early and bad >= patience:
                    break
        if early and state is not None:
            m.load_state_dict(state)
        m.eval()
        return m, hist

    out = {}
    # ---------------------------------------------------------------- German credit, 600 / 200 / 200 (analysis_L3.py)
    g = npl.german_credit()
    yg = g["default"].values.astype(int)
    Xg_raw = g.drop(columns="default")

    def encode_german(tr, others):
        cat = [c for c in tr.columns if tr[c].dtype == object]
        num = [c for c in tr.columns if c not in cat]
        levels = {c: sorted(tr[c].unique()) for c in cat}

        def f(d):
            x = d[num].astype(float).copy()
            for c in cat:
                for lv in levels[c][1:]:
                    x[f"{c}_{lv}"] = (d[c] == lv).astype(float)
            return x
        xtr = f(tr)
        mu, sd = xtr.mean(), xtr.std().replace(0, 1)
        return [((xtr - mu) / sd).values.astype(np.float32)] + [((f(o) - mu) / sd).values.astype(np.float32) for o in others]

    gi = np.arange(len(g))
    g_tr, g_te = train_test_split(gi, test_size=0.2, stratify=yg, random_state=SEED)
    g_tr, g_va = train_test_split(g_tr, test_size=0.25, stratify=yg[g_tr], random_state=SEED)
    Gtr, Gva, Gte = encode_german(Xg_raw.iloc[g_tr], [Xg_raw.iloc[g_va], Xg_raw.iloc[g_te]])
    yva_ = yg[g_va]
    curves = {}
    for p in DPS:
        for wd in WDS:
            _, h = fit(Gtr, yg[g_tr], Gva, yva_, hidden=(256, 256), p=p, wd=wd, epochs=150, early=False, record=True, bs=64,
                       metric=lambda m: float(roc_auc_score(yva_, proba(m, Gva))))
            h = np.array(h)
            curves[f"{p}|{wd}"] = {"tr": np.round(h[:, 0], 4).tolist(), "va": np.round(h[:, 1], 4).tolist(),
                                   "auc": np.round(h[:, 2], 4).tolist()}
    lr_g = LogisticRegression(C=1.0, max_iter=5000).fit(Gtr, yg[g_tr])
    pv = lr_g.predict_proba(Gva)[:, 1]
    out["ov"] = {"c": curves, "logit_va": float(-np.mean(yva_ * np.log(pv) + (1 - yva_) * np.log(1 - pv))),
                 "logit_auc": float(roc_auc_score(yva_, pv)), "dps": DPS, "wds": WDS}

    # hold-out noise: logistic regression on 200 random stratified splits, for several test shares
    ho = {}
    for sh in SHARES:
        r = []
        for s in range(200):
            a, b = train_test_split(np.arange(len(yg)), test_size=sh, stratify=yg, random_state=SEED + s)
            xa, xb = encode_german(Xg_raw.iloc[a], [Xg_raw.iloc[b]])
            r.append(float(roc_auc_score(yg[b], LogisticRegression(C=1.0, max_iter=5000).fit(xa, yg[a]).predict_proba(xb)[:, 1])))
        ho[str(sh)] = [round(v, 4) for v in r]
    out["ho"] = ho

    # ---------------------------------------------------------------- card fraud, time-ordered split (analysis_L3.py)
    fr = npl.fraud().sort_values("Time").reset_index(drop=True)
    vcols = [f"V{i}" for i in range(1, 29)]
    Xf0 = np.c_[fr[vcols].values, np.log1p(fr["Amount"].values)].astype(np.float32)
    yf = fr["Class"].values.astype(int)
    wf = fr["weight"].values
    amt = fr["Amount"].values
    c1, c2 = np.quantile(fr["Time"], [0.6, 0.8])
    f_tr = (fr["Time"] <= c1).values
    f_va = ((fr["Time"] > c1) & (fr["Time"] <= c2)).values
    f_te = (fr["Time"] > c2).values
    fmu, fsd = Xf0[f_tr].mean(0), Xf0[f_tr].std(0)
    Xf = (Xf0 - fmu) / fsd

    # early-stopping picture: MLP 256-256 without regularisation, 120 epochs, validation PR-AUC after each epoch
    yv_, wv_ = yf[f_va], wf[f_va]
    _, h = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yv_, hidden=(256, 256), epochs=120, early=False, record=True, bs=256, seed=0,
               metric=lambda m: float(average_precision_score(yv_, proba(m, Xf[f_va]), sample_weight=wv_)))
    h = np.array(h)
    out["es"] = {"tr": [float(f"{v:.4e}") for v in h[:, 0]], "va": [round(v, 5) for v in h[:, 1]], "ap": np.round(h[:, 2], 4).tolist()}

    # final models and their test scores (cost of fraud)
    p_lr = LogisticRegression(max_iter=5000).fit(Xf[f_tr], yf[f_tr]).predict_proba(Xf[f_te])[:, 1]
    mlps = [fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=60, bs=256, seed=s, early=True, wd=0.1, p=0.3)[0]
            for s in range(5)]
    p_ml = np.mean([proba(m, Xf[f_te]) for m in mlps], axis=0)
    xgm = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                            eval_metric="aucpr", early_stopping_rounds=100, random_state=SEED, n_jobs=1)
    xgm.fit(Xf[f_tr], yf[f_tr], eval_set=[(Xf[f_va], yf[f_va])], verbose=False)
    p_xg = xgm.predict_proba(Xf[f_te])[:, 1]
    out["fc"] = {"y": yf[f_te].tolist(), "w": np.round(wf[f_te], 3).tolist(), "amt": np.round(amt[f_te], 2).tolist(),
                 "logit": [float(f"{v:.4e}") for v in p_lr], "mlp": [float(f"{v:.4e}") for v in p_ml],
                 "xgb": [float(f"{v:.4e}") for v in p_xg]}
    return out


def load(fresh=False):
    import numpy as np
    if os.path.exists(CACHE) and not fresh:
        r = json.load(open(CACHE))
    else:
        r = compute()
        json.dump(r, open(CACHE, "w"))
    dd = json.load(open(os.path.join(CDIR, "double_descent_v1.json")))
    P = sorted(int(k) for k in dd)
    r["dd"] = {"p": P, "tr": [[x[0] for x in dd[str(p)]] for p in P], "te": [[x[1] for x in dd[str(p)]] for p in P], "n": 400}
    r["dd"]["tr"] = [[float(f"{v:.4g}") for v in a] for a in r["dd"]["tr"]]
    r["dd"]["te"] = [[float(f"{v:.4g}") for v in a] for a in r["dd"]["te"]]
    return r


if __name__ == "__main__":
    d = load("--fresh" in sys.argv)
    import numpy as np
    c = d["ov"]["c"]
    print("no reg: val min", min(c["0.0|0.0"]["va"]), "last", c["0.0|0.0"]["va"][-1], "| dp .5 wd .1: min", min(c["0.5|0.1"]["va"]),
          "last", c["0.5|0.1"]["va"][-1])
    print("es best epoch", int(np.argmin(d["es"]["va"])) + 1, "min", min(d["es"]["va"]))
    print({k: (round(np.mean(v), 3), round(np.std(v, ddof=1), 3)) for k, v in d["ho"].items()})
    from sklearn.metrics import average_precision_score
    for k in ("logit", "mlp", "xgb"):
        print(k, round(average_precision_score(d["fc"]["y"], d["fc"][k], sample_weight=d["fc"]["w"]), 3))
