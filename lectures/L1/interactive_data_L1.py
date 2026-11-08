"""Inputs of the Lecture 1 interactive page (interactive/L1_data.js), recomputed from data/clean with the code and seeds of
analysis_L1.py (same splits, same models). Trained results are cached in lectures/L1/cache/interactive_v1.json.
Called by build_interactive_L1.py; run alone to refresh the cache:  python3 lectures/L1/interactive_data_L1.py --fresh
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
CACHE = os.path.join(HERE, "cache", "interactive_v1.json")


def compute():
    import numpy as np
    import nn_plot as npl
    import xgboost as xgb                     # before torch (both ship an OpenMP runtime), as in analysis_L1.py
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(1)
    SEED = npl.seed_all()
    out = {}

    class MLP(nn.Module):
        def __init__(self, d, hidden=(64, 64), act="relu"):
            super().__init__()
            A = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[act]
            layers, k = [], d
            for h in hidden:
                layers += [nn.Linear(k, h), A()]
                k = h
            layers.append(nn.Linear(k, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), lr=1e-3, epochs=300, bs=128, patience=25, seed=0):
        torch.manual_seed(SEED + seed)
        rng = np.random.default_rng(SEED + seed)
        m = MLP(Xtr.shape[1], hidden)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        lf = nn.BCEWithLogitsLoss()
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
        Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32).view(-1, 1)
        best, state, bad = np.inf, None, 0
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
            if vl < best - 1e-5:
                best, bad = vl, 0
                state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        m.load_state_dict(state)
        m.eval()
        return m

    def proba(m, X):
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(np.asarray(X, np.float32)))).numpy().ravel()

    # ------------------------------------------------------------------ HMEQ, same split and encoding as analysis_L1.py
    import pandas as pd
    h = npl.hmeq()
    yh = h.pop("BAD").values.astype(int)
    num = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]

    def encode(tr, others):
        med = tr[num].median()

        def f(d):
            x = pd.DataFrame(index=d.index)
            for c in num:
                x[c] = d[c].fillna(med[c])
                x["miss_" + c] = d[c].isna().astype(float)
            for c, levels in (("REASON", ["DebtCon", "HomeImp"]), ("JOB", ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"])):
                for lv in levels:
                    x[f"{c}_{lv}"] = (d[c] == lv).astype(float)
            return x
        xtr = f(tr)
        mu, sd = xtr.mean(), xtr.std().replace(0, 1)
        return [((xtr - mu) / sd).values.astype(np.float32)] + [((f(o) - mu) / sd).values.astype(np.float32) for o in others]

    idx = np.arange(len(h))
    i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=yh, random_state=SEED)
    i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=yh[i_tr], random_state=SEED)
    Xtr, Xva, Xte = encode(h.iloc[i_tr], [h.iloc[i_va], h.iloc[i_te]])
    ytr, yva, yte = yh[i_tr], yh[i_va], yh[i_te]

    # 1. one neuron on two variables (debt-to-income, age of the oldest credit line)
    two = ["DEBTINC", "CLAGE"]
    m2 = h[two].notna().all(axis=1).values
    i_tr2, i_va2, i_te2 = i_tr[m2[i_tr]], i_va[m2[i_va]], i_te[m2[i_te]]
    mu2, sd2 = h.iloc[i_tr2][two].mean(), h.iloc[i_tr2][two].std()
    Z = lambda ii: ((h.iloc[ii][two] - mu2) / sd2).values.astype(np.float32)  # noqa: E731
    lr2 = LogisticRegression(C=1e6, max_iter=5000).fit(Z(i_tr2), yh[i_tr2])
    ml2 = train_mlp(Z(i_tr2), yh[i_tr2], Z(i_va2), yh[i_va2], hidden=(64, 64), seed=0, epochs=400, patience=40)
    gx, gy = np.linspace(0, 70, 71), np.linspace(0, 600, 61)
    G = np.array([[a, b] for b in gy for a in gx])
    gz = ((G - mu2.values) / sd2.values).astype(np.float32)
    te = h.iloc[i_te2]
    out["nb"] = {"mu": [float(v) for v in mu2.values], "sd": [float(v) for v in sd2.values],
                 "w": [float(v) for v in lr2.coef_[0]], "b": float(lr2.intercept_[0]),
                 "x1": te["DEBTINC"].round(1).tolist(), "x2": te["CLAGE"].round(0).tolist(), "y": yh[i_te2].tolist(),
                 "gx": gx.tolist(), "gy": gy.tolist(), "mlp": np.round(proba(ml2, gz).reshape(len(gy), len(gx)), 3).tolist(),
                 "auc_logit": float(roc_auc_score(yh[i_te2], lr2.predict_proba(Z(i_te2))[:, 1])),
                 "auc_mlp": float(roc_auc_score(yh[i_te2], proba(ml2, Z(i_te2)))), "n_train": int(len(i_tr2))}

    # 2. test-set scores of logistic regression, MLP 64-64 (seed 0) and XGBoost (ROC and threshold)
    p_lr = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    p_ml = proba(train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), seed=0), Xte)
    xg = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                           eval_metric="logloss", early_stopping_rounds=50, random_state=SEED, n_jobs=4)
    xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p_xg = xg.predict_proba(Xte)[:, 1]
    out["roc"] = {"y": yte.tolist(), "logit": np.round(p_lr, 4).tolist(), "mlp": np.round(p_ml, 4).tolist(),
                  "xgb": np.round(p_xg, 4).tolist()}

    # 3. universal approximation: MLPs with H sigmoid units on latitude (California housing), as analysis_L1.py
    ca = npl.california()
    x1 = ca["Latitude"].values.astype(np.float32)
    y1 = ca["MedHouseVal_100k_usd"].values.astype(np.float32)
    keep = y1 < 5.0
    x1, y1 = x1[keep], y1[keep]
    i_a, i_b = train_test_split(np.arange(len(x1)), test_size=0.25, random_state=SEED)
    mx_, sx_ = x1[i_a].mean(), x1[i_a].std()
    xs = np.linspace(32.5, 42, 191).astype(np.float32)
    ua = {"x": np.round(xs, 2).tolist(), "fits": {}}
    Xa = torch.tensor((x1[i_a, None] - mx_) / sx_)
    Ya = torch.tensor(y1[i_a, None])
    Xb = torch.tensor((x1[i_b, None] - mx_) / sx_)
    Yb = torch.tensor(y1[i_b, None])
    Xs = torch.tensor((xs[:, None] - mx_) / sx_)
    for w in (1, 2, 3, 5, 10, 20, 50):
        torch.manual_seed(SEED + w)
        mU = MLP(1, (w,), "sigmoid")
        opt = torch.optim.Adam(mU.parameters(), lr=0.02)
        for _ in range(4000):
            opt.zero_grad()
            nn.MSELoss()(mU(Xa), Ya).backward()
            opt.step()
        with torch.no_grad():
            hid = torch.sigmoid(mU.net[0](Xs))                     # hidden activations, 191 x w
            v = mU.net[2].weight[0]                                 # output weights
            units = (hid * v).numpy().T                             # contribution of each unit
            ua["fits"][str(w)] = {"f": np.round(mU(Xs).numpy().ravel(), 3).tolist(), "c": float(mU.net[2].bias[0]),
                                  "u": np.round(units, 3).tolist(),
                                  "mse_te": float(nn.MSELoss()(mU(Xb), Yb)), "mse_tr": float(nn.MSELoss()(mU(Xa), Ya))}
    b_lin = np.polyfit(x1[i_a], y1[i_a], 1)
    ua["lin"] = np.round(np.polyval(b_lin, xs), 3).tolist()
    ua["mse_lin"] = float(np.mean((np.polyval(b_lin, x1[i_b]) - y1[i_b]) ** 2))
    bins = np.quantile(x1[i_a], np.linspace(0, 1, 41))
    bc = np.digitize(x1[i_a], bins[1:-1])
    ua["bx"] = [round(float(x1[i_a][bc == k].mean()), 3) for k in range(40)]
    ua["by"] = [round(float(y1[i_a][bc == k].mean()), 3) for k in range(40)]
    ua["n"] = int(keep.sum())
    out["ua"] = ua
    return out


def load(fresh=False):
    if os.path.exists(CACHE) and not fresh:
        return json.load(open(CACHE))
    r = compute()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(r, open(CACHE, "w"))
    return r


if __name__ == "__main__":
    r = load("--fresh" in sys.argv)
    print("AUC two-variable logit / MLP:", round(r["nb"]["auc_logit"], 3), round(r["nb"]["auc_mlp"], 3))
    from sklearn.metrics import roc_auc_score
    for k in ("logit", "mlp", "xgb"):
        print(k, round(roc_auc_score(r["roc"]["y"], r["roc"][k]), 3))
    for w, f in r["ua"]["fits"].items():
        print("H", w, round(f["mse_te"], 3))
    print("lin", round(r["ua"]["mse_lin"], 3))
