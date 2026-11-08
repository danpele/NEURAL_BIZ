"""
analysis_L3.py -- Lecture 3 "Regularisation and generalisation". All computation of the lecture:
writes lectures/L3/L3_numbers.json and charts/en|ro/*.pdf.

Data (data/SOURCES.md): ULB credit-card fraud sample (28,924 transactions, all 492 frauds; time-ordered split; the column
`weight` restores the full-data class mix in every precision-type metric), German credit (1,000 applicants; small-data
overfitting, weight decay, dropout, validation noise), California housing (bias-variance and double descent; batch norm).
Models: MLPs in PyTorch with weight decay, L1, dropout, early stopping and batch normalisation; logistic regression and
XGBoost as benchmarks; random-feature regression (two-layer network with a frozen first layer) for double descent.
Seeds 20261108 + s. Hardware: CPU (PyTorch, 1 thread). Long runs cached in lectures/L3/cache/ (git-ignored).
Run:  python3 lectures/L3/analysis_L3.py  [--fresh]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import xgboost as xgb  # noqa: E402  (import before torch: both ship an OpenMP runtime)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

npl.style()
SEED = npl.seed_all()
torch.set_num_threads(1)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
FRESH = "--fresh" in sys.argv
N = {"charts": {}, "seed": SEED}
PAL = npl.PAL
T0 = time.time()


def cached(name, fn):
    p = os.path.join(CACHE, name + ".json")
    if os.path.exists(p) and not FRESH:
        return json.load(open(p))
    r = fn()
    json.dump(r, open(p, "w"))
    return r


def r2(x, d=3):
    return float(round(float(x), d))


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


def fit(Xtr, ytr, Xva, yva, hidden=(128, 128), p=0.0, bn=False, wd=0.0, l1=0.0, lr=1e-3, epochs=200, bs=128, seed=0,
        early=True, patience=20, loss="bce", opt_name="adam", record=False, w_va=None):
    """Adam (or SGD) with decoupled weight decay (AdamW), optional L1 penalty, dropout, batch norm and early stopping
    on the validation loss. Returns the model (best validation epoch if early) and the loss history."""
    torch.manual_seed(SEED + seed)
    rng = np.random.default_rng(SEED + seed)
    m = Net(Xtr.shape[1], hidden, p, bn)
    if opt_name == "adam":
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss() if loss == "bce" else nn.MSELoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    best, state, bad, hist, best_ep = np.inf, None, 0, [], 0
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            if len(b) < 2 and bn:
                continue
            opt.zero_grad()
            ls = lf(m(Xt[b]), yt[b])
            if l1 > 0:
                ls = ls + l1 * sum(w.abs().sum() for n_, w in m.named_parameters() if "weight" in n_ and w.dim() == 2)
            ls.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lf(m(Xv), yv))
            tl = float(lf(m(Xt), yt)) if record else np.nan
        hist.append((tl, vl))
        if not np.isfinite(vl):
            break
        if vl < best - 1e-5:
            best, bad, best_ep = vl, 0, ep + 1
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if early and bad >= patience:
                break
    if early and state is not None:
        m.load_state_dict(state)
    m.eval()
    return m, hist, best_ep


def proba(m, X):
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy().ravel()


# =============================================================================================== German credit
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


# ---- 1. overfitting picture: 600 / 200 / 200 split, big MLP without any regularisation
gi = np.arange(len(g))
g_tr, g_te = train_test_split(gi, test_size=0.2, stratify=yg, random_state=SEED)
g_tr, g_va = train_test_split(g_tr, test_size=0.25, stratify=yg[g_tr], random_state=SEED)
Gtr, Gva, Gte = encode_german(Xg_raw.iloc[g_tr], [Xg_raw.iloc[g_va], Xg_raw.iloc[g_te]])
N["german"] = {"n_train": int(len(g_tr)), "n_val": int(len(g_va)), "n_test": int(len(g_te)), "d": int(Gtr.shape[1])}
m_over, h_over, be_over = fit(Gtr, yg[g_tr], Gva, yg[g_va], hidden=(256, 256), epochs=150, early=False, record=True, bs=64)
m_reg, h_reg, be_reg = fit(Gtr, yg[g_tr], Gva, yg[g_va], hidden=(256, 256), p=0.5, wd=0.1, epochs=150, early=False, record=True, bs=64)
ho, hr = np.array(h_over), np.array(h_reg)
n_par = (Gtr.shape[1] + 1) * 256 + 257 * 256 + 257
lr_g = LogisticRegression(C=1.0, max_iter=5000).fit(Gtr, yg[g_tr])
ll_logit_va = float(-np.mean(yg[g_va] * np.log(lr_g.predict_proba(Gva)[:, 1]) + (1 - yg[g_va]) * np.log(1 - lr_g.predict_proba(Gva)[:, 1])))
N["overfit"] = {"params": int(n_par), "best_ep": int(np.argmin(ho[:, 1]) + 1), "val_min": r2(ho[:, 1].min()), "val_last": r2(ho[-1, 1]),
                "train_last": r2(ho[-1, 0], 4), "reg_val_min": r2(hr[:, 1].min()), "reg_val_last": r2(hr[-1, 1]), "epochs": 150,
                "logit_val": r2(ll_logit_va), "auc_over_last": r2(roc_auc_score(yg[g_te], proba(m_over, Gte))),
                "auc_reg_last": r2(roc_auc_score(yg[g_te], proba(m_reg, Gte)))}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    e = np.arange(1, len(ho) + 1)
    ax.plot(e, ho[:, 0], color=PAL["train"], label="Training loss, no regularisation" if not ro else "Pierderea de antrenare, fără regularizare")
    ax.plot(e, ho[:, 1], color=PAL["valid"], label="Validation loss, no regularisation" if not ro else "Pierderea de validare, fără regularizare")
    ax.plot(e, hr[:, 1], color=PAL["valid"], ls="--", label="Validation loss, dropout 0.5 + weight decay" if not ro else
            "Pierderea de validare, dropout 0,5 + penalizare")
    ax.axhline(ll_logit_va, color=PAL["benchmark"], lw=1.1, label="Logistic regression (validation)" if not ro else "Regresia logistică (validare)")
    ax.axvline(N["overfit"]["best_ep"], color=npl.MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("Epoch" if not ro else "Epoca")
    ax.set_ylabel("Cross-entropy" if not ro else "Entropia încrucișată")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "overfit_german")
npl.chart_meta(N, "overfit_german", "german", f"{len(g_tr)} training and {len(g_va)} validation applicants; MLP 256-256",
               f"{len(g_tr)} de solicitanți de antrenare și {len(g_va)} de validare; MLP 256-256")

# ---- 2. weight decay and dropout sweeps, 5-fold cross-validation (inner validation split for early stopping)
LAMS = [0.0, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xg_raw, yg))


def fold_data(f):
    a, b = FOLDS[f]
    a_tr, a_va = train_test_split(a, test_size=0.2, stratify=yg[a], random_state=SEED)
    xa, xv, xb = encode_german(Xg_raw.iloc[a_tr], [Xg_raw.iloc[a_va], Xg_raw.iloc[b]])
    return xa, yg[a_tr], xv, yg[a_va], xb, yg[b]


FD = [fold_data(f) for f in range(5)]


def sweep(kind):
    out = {}
    grid = LAMS if kind == "wd" else PS
    for v in grid:
        res = []
        for f in range(5):
            xa, ya, xv, yv, xb, yb = FD[f]
            kw = {"wd": v} if kind == "wd" else {"p": v}
            m, _, _ = fit(xa, ya, xv, yv, hidden=(256, 256), epochs=150, bs=64, early=False, seed=f, **kw)
            wn = float(sum((w.detach() ** 2).sum() for n_, w in m.named_parameters() if "weight" in n_).sqrt())
            res.append([roc_auc_score(yb, proba(m, xb)), wn])
        out[str(v)] = res
    return out


swd = cached("sweep_wd_v1", lambda: sweep("wd"))
sdp = cached("sweep_dp_v1", lambda: sweep("dp"))
auc_lr_cv = [roc_auc_score(FD[f][5], LogisticRegression(C=1.0, max_iter=5000).fit(FD[f][0], FD[f][1]).predict_proba(FD[f][4])[:, 1]) for f in range(5)]
KEY = lambda v: f"{float(v):g}".replace(".", "p")          # 0.01 -> "0p01" (placeholders split on dots)
N["wd"] = {KEY(k): {"auc": r2(np.mean([x[0] for x in v])), "sd": r2(np.std([x[0] for x in v], ddof=1)), "norm": r2(np.mean([x[1] for x in v]), 1)}
           for k, v in swd.items()}
N["dp"] = {KEY(k): {"auc": r2(np.mean([x[0] for x in v])), "sd": r2(np.std([x[0] for x in v], ddof=1))} for k, v in sdp.items()}
best_lam = max(LAMS, key=lambda v: np.mean([x[0] for x in swd[str(v)]]))
best_p = max(PS, key=lambda v: np.mean([x[0] for x in sdp[str(v)]]))
N["cv"] = {"auc_logit": r2(np.mean(auc_lr_cv)), "sd_logit": r2(np.std(auc_lr_cv, ddof=1)), "best_lam": best_lam, "best_p": best_p,
           "auc_best_lam": N["wd"][KEY(best_lam)]["auc"], "auc_best_p": N["dp"][KEY(best_p)]["auc"],
           "auc_none": N["wd"]["0"]["auc"], "best_lam_key": KEY(best_lam), "best_p_key": KEY(best_p)}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    xl = np.arange(len(LAMS))
    mwd = np.array([np.mean([x[0] for x in swd[str(v)]]) for v in LAMS])
    swd_ = np.array([np.std([x[0] for x in swd[str(v)]], ddof=1) for v in LAMS])
    axs[0].errorbar(xl, mwd, yerr=swd_ / np.sqrt(5), fmt="o-", color=PAL["model"], ms=4, capsize=2,
                    label="MLP 256-256 (mean of 5 folds ± s.e.)" if not ro else "MLP 256-256 (media a 5 pliuri ± e.s.)")
    axs[0].set_xticks(xl)
    axs[0].set_xticklabels(["0"] + [f"{v:g}" for v in LAMS[1:]], fontsize=8.5, rotation=45)
    axs[0].set_xlabel("Weight decay $\\lambda$" if not ro else "Penalizarea ponderilor $\\lambda$")
    xp = np.arange(len(PS))
    mdp = np.array([np.mean([x[0] for x in sdp[str(v)]]) for v in PS])
    sdp_ = np.array([np.std([x[0] for x in sdp[str(v)]], ddof=1) for v in PS])
    axs[1].errorbar(xp, mdp, yerr=sdp_ / np.sqrt(5), fmt="o-", color=PAL["model"], ms=4, capsize=2)
    axs[1].set_xticks(xp)
    axs[1].set_xticklabels([f"{v:g}" for v in PS], fontsize=8.5)
    axs[1].set_xlabel("Dropout rate $p$" if not ro else "Rata de dropout $p$")
    for ax in axs:
        ax.axhline(np.mean(auc_lr_cv), color=PAL["benchmark"], lw=1.1, label="Logistic regression" if not ro else "Regresia logistică")
    axs[0].set_ylabel("Held-out AUC" if not ro else "AUC pe pliul de test")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "wd_dropout")
npl.chart_meta(N, "wd_dropout", "german", "1,000 applicants, 5-fold cross-validation, no early stopping (150 epochs)",
               "1\\,000 de solicitanți, validare încrucișată cu 5 pliuri, fără oprire timpurie (150 de epoci)")

# ---- 3. L1 vs L2: the first-layer weights
def penal_net(kind, lam, seed=0, epochs=300):
    torch.manual_seed(SEED + seed)
    m = Net(Gtr.shape[1], (64,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(Gtr), torch.tensor(yg[g_tr], dtype=torch.float32).view(-1, 1)
    for ep in range(epochs):
        opt.zero_grad()                         # full batch: the penalty path is easier to read
        W = m.net[0].weight
        pen = lam * (W.abs().sum() if kind == "l1" else (W ** 2).sum())
        (nn.BCEWithLogitsLoss()(m(Xt), yt) + pen).backward()
        opt.step()
    m.eval()
    return m


ml1 = penal_net("l1", 1e-3)
ml2 = penal_net("l2", 1e-2)
W1l1 = ml1.net[0].weight.detach().numpy().ravel()
W1l2 = ml2.net[0].weight.detach().numpy().ravel()
TH = 1e-2
# inputs whose 64 outgoing weights are all near zero under L1 = inputs switched off
dead_in = int(np.sum(np.abs(ml1.net[0].weight.detach().numpy()).max(0) < TH))
N["l1l2"] = {"share_small_l1_pct": r2(100 * np.mean(np.abs(W1l1) < TH), 1), "share_small_l2_pct": r2(100 * np.mean(np.abs(W1l2) < TH), 1),
             "thr": TH, "n_weights": int(W1l1.size), "lam_l1": 1e-3, "lam_l2": 1e-2, "inputs_off_l1": dead_in,
             "auc_l1": r2(roc_auc_score(yg[g_te], proba(ml1, Gte))), "auc_l2": r2(roc_auc_score(yg[g_te], proba(ml2, Gte)))}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    bins = np.linspace(-0.6, 0.6, 81)
    ax.hist(W1l2, bins=bins, color=PAL["data"], alpha=0.55, label="L2 penalty $\\lambda\\sum w^2$" if not ro else "Penalizare L2 $\\lambda\\sum w^2$")
    ax.hist(W1l1, bins=bins, color=PAL["model"], alpha=0.55, label="L1 penalty $\\lambda\\sum |w|$" if not ro else "Penalizare L1 $\\lambda\\sum |w|$")
    ax.set_yscale("log")
    ax.set_xlabel("First-layer weight $w$" if not ro else "Ponderea din primul strat $w$")
    ax.set_ylabel("Count (log scale)" if not ro else "Frecvența (scară log.)")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "l1_l2_weights", also_x=[ax] if ro else ())
npl.chart_meta(N, "l1_l2_weights", "german", f"{len(g_tr)} training applicants; MLP with 64 hidden units, {W1l1.size:,} first-layer weights",
               f"{len(g_tr)} de solicitanți de antrenare; MLP cu 64 de neuroni ascunși, {W1l1.size:,} de ponderi în primul strat".replace(",", "\\,"))

# =============================================================================================== California: double descent + batch norm
ca = npl.california()
feat = [c for c in ca.columns if c != "MedHouseVal_100k_usd"]
Xc = ca[feat].values.astype(np.float64)
yc = ca["MedHouseVal_100k_usd"].values.astype(np.float64)
NDD = 400
P_GRID = [10, 25, 50, 100, 200, 300, 350, 380, 400, 420, 450, 500, 600, 800, 1200, 2000, 4000, 8000]


def double_descent():
    out = {str(p): [] for p in P_GRID}
    for s in range(10):
        rng = np.random.default_rng(SEED + s)
        idx = rng.permutation(len(Xc))
        tr, te = idx[:NDD], idx[NDD:NDD + 5000]
        mu, sd = Xc[tr].mean(0), Xc[tr].std(0)
        A, B = (Xc[tr] - mu) / sd, (Xc[te] - mu) / sd
        Wr = rng.normal(0, 1 / np.sqrt(A.shape[1]), (A.shape[1], max(P_GRID)))
        br = rng.normal(0, 1, max(P_GRID))
        for p in P_GRID:
            Fa, Fb = np.maximum(A @ Wr[:, :p] + br[:p], 0), np.maximum(B @ Wr[:, :p] + br[:p], 0)
            yt_ = yc[tr] - yc[tr].mean()
            if p <= NDD:                                              # least squares (minimum norm if rank-deficient)
                beta = np.linalg.lstsq(Fa, yt_, rcond=None)[0]
            else:                                                     # minimum-norm interpolant F'(FF')^-1 y
                beta = Fa.T @ np.linalg.lstsq(Fa @ Fa.T, yt_, rcond=None)[0]
            out[str(p)].append([float(np.mean((Fa @ beta + yc[tr].mean() - yc[tr]) ** 2)),
                                float(np.mean((Fb @ beta + yc[tr].mean() - yc[te]) ** 2))])
    return out


dd = cached("double_descent_v1", double_descent)
tr_med = np.array([np.median([x[0] for x in dd[str(p)]]) for p in P_GRID])
te_med = np.array([np.median([x[1] for x in dd[str(p)]]) for p in P_GRID])
under = [p for p in P_GRID if p < NDD]
over = [p for p in P_GRID if p > NDD]
best_under = min(under, key=lambda p: te_med[P_GRID.index(p)])
N["dd"] = {"n": NDD, "peak_p": int(P_GRID[int(np.argmax(te_med))]), "peak_mse": r2(te_med.max(), 1),
           "best_under_p": best_under, "best_under_mse": r2(te_med[P_GRID.index(best_under)]),
           "big_p": P_GRID[-1], "big_mse": r2(te_med[-1]), "seeds": 10}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(P_GRID, te_med, "o-", color=PAL["test"], ms=3.5, label="Test MSE (median of 10 draws)" if not ro else "MSE pe test (mediana a 10 extrageri)")
    ax.plot(P_GRID, tr_med, "o-", color=PAL["train"], ms=3.5, label="Training MSE" if not ro else "MSE de antrenare")
    ax.axvline(NDD, color=npl.MUTED, ls="--", lw=1, label=f"interpolation threshold p = n = {NDD}" if not ro else f"pragul de interpolare p = n = {NDD}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1e3)
    ax.set_xlabel("Hidden units $p$ (random ReLU features, log scale)" if not ro else "Neuroni ascunși $p$ (caracteristici ReLU aleatoare, scară log.)")
    ax.set_ylabel("MSE (log scale)" if not ro else "MSE (scară log.)")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "double_descent")
npl.chart_meta(N, "double_descent", "california", f"{NDD} training and 5,000 test block groups per draw, 10 draws",
               f"{NDD} de grupuri de antrenare și 5\\,000 de testare la fiecare extragere, 10 extrageri")

# batch normalisation on a deep MLP, SGD with a large learning rate
ci_tr, ci_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED)
ci_tr, ci_va = train_test_split(ci_tr, test_size=0.25, random_state=SEED)
cmu, csd = Xc[ci_tr].mean(0), Xc[ci_tr].std(0)
Cz = ((Xc - cmu) / csd).astype(np.float32)
yc32 = yc.astype(np.float32)


def bn_runs():
    out = {}
    for bn in (False, True):
        for lr in (0.01, 0.1):
            key = f"{'bn' if bn else 'plain'}|{lr}"
            out[key] = []
            for s in range(3):
                _, h, _ = fit(Cz[ci_tr], yc32[ci_tr], Cz[ci_va], yc32[ci_va], hidden=(64,) * 6, bn=bn, lr=lr, epochs=40, bs=128,
                              seed=s, early=False, loss="mse", opt_name="sgd")
                v = [min(x[1], 10.0) if np.isfinite(x[1]) else 10.0 for x in h]
                v += [10.0] * (40 - len(v))
                out[key].append(v)
    return out


bnr = cached("batchnorm_v2", bn_runs)
N["bn"] = {k.replace("|", "_").replace(".", ""): {"val_final": r2(np.median([v[-1] for v in vals])), "val_ep5": r2(np.median([v[4] for v in vals])),
                                                  "val_best": r2(np.median([min(v) for v in vals])),
                                                  "diverged": int(sum(max(v) >= 10.0 for v in vals))}
           for k, vals in bnr.items()}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for key, c, ls, nm in (("plain|0.01", PAL["data"], "-", ("no BN, $\\eta$=0.01", "fără BN, $\\eta$=0,01")),
                           ("plain|0.1", PAL["data"], "--", ("no BN, $\\eta$=0.1 (diverges)", "fără BN, $\\eta$=0,1 (diverge)")),
                           ("bn|0.01", PAL["model"], "-", ("batch norm, $\\eta$=0.01", "normalizare pe loturi, $\\eta$=0,01")),
                           ("bn|0.1", PAL["model"], "--", ("batch norm, $\\eta$=0.1", "normalizare pe loturi, $\\eta$=0,1"))):
        v = np.median(np.array(bnr[key]), axis=0)
        ax.plot(np.arange(1, 41), v, color=c, ls=ls, label=nm[ro])
    ax.set_ylim(0.2, 1.5)
    ax.set_xlabel("Epoch" if not ro else "Epoca")
    ax.set_ylabel("Validation MSE (median)" if not ro else "MSE pe validare (mediana)")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "batchnorm")
npl.chart_meta(N, "batchnorm", "california", "6 hidden layers of 64 units, SGD with momentum 0.9, 3 seeds",
               "6 straturi ascunse de 64 de neuroni, SGD cu moment 0,9, 3 inițializări")

# =============================================================================================== fraud: time-ordered split
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
N["fraud"] = {"n": int(len(fr)), "frauds": int(yf.sum()), "n_train": int(f_tr.sum()), "n_val": int(f_va.sum()), "n_test": int(f_te.sum()),
              "fr_train": int(yf[f_tr].sum()), "fr_val": int(yf[f_va].sum()), "fr_test": int(yf[f_te].sum()),
              "cut1_h": r2(c1 / 3600, 1), "cut2_h": r2(c2 / 3600, 1), "prev_full_pct": r2(100 * yf.sum() / wf.sum(), 3),
              "prev_sample_pct": r2(100 * yf.mean(), 2), "n_full": int(round(wf.sum())), "legit_weight": r2(wf[yf == 0].mean(), 2),
              "fraud_amount_median": r2(np.median(amt[yf == 1]), 2), "legit_amount_median": r2(np.median(amt[yf == 0]), 2),
              "fraud_amount_sum_test": r2(amt[f_te & (yf == 1)].sum(), 0)}
# hourly fraud rate (population-weighted), for the data frame
hour = (fr["Time"] // 3600).astype(int)
rate_h = pd.Series(yf).groupby(hour).sum() / pd.Series(wf).groupby(hour).sum()
cnt_h = pd.Series(yf).groupby(hour).sum()
N["fraud"]["rate_h_max_pct"] = r2(100 * rate_h.max(), 2)
N["fraud"]["rate_h_max_hour"] = int(rate_h.idxmax())
N["fraud"]["rate_h_median_pct"] = r2(100 * rate_h.median(), 3)
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.bar(rate_h.index, 100 * rate_h.values, color=PAL["positive"], width=0.8, label="Fraud rate in the hour (%)" if not ro else "Rata fraudei în ora respectivă (%)")
    for cut, nm, yy_ in ((c1, ("validation", "validare"), 0.97), (c2, ("test", "test"), 0.80)):
        ax.axvline(cut / 3600, color=npl.INK, lw=1, ls="--")
        ax.text(cut / 3600 + 0.4, yy_ * 100 * rate_h.max(), nm[ro], fontsize=8.5, va="top")
    ax.set_xlabel("Hours since the first transaction (two days)" if not ro else "Ore de la prima tranzacție (două zile)")
    ax.set_ylabel("Fraud rate (%)" if not ro else "Rata fraudei (%)")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "fraud_by_hour")
npl.chart_meta(N, "fraud_by_hour", "fraud", "September 2013, 2 days, 284,807 transactions (weighted sample)",
               "septembrie 2013, 2 zile, 284\\,807 tranzacții (eșantion ponderat)")


def ap(y, p, w):
    return float(average_precision_score(y, p, sample_weight=w))


# ---- early stopping picture
m_es, h_es, be_es = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=120, early=False, record=True, bs=256, seed=0)
hes = np.array(h_es)
PAT = 20
stop_ep = None
best_ = np.inf
bad_ = 0
for i, v in enumerate(hes[:, 1]):
    if v < best_ - 1e-5:
        best_, bad_ = v, 0
    else:
        bad_ += 1
        if bad_ >= PAT:
            stop_ep = i + 1
            break
N["es"] = {"best_ep": int(np.argmin(hes[:, 1]) + 1), "stop_ep": stop_ep, "patience": PAT, "val_min": r2(hes[:, 1].min(), 4),
           "val_last": r2(hes[-1, 1], 4), "train_last": r2(hes[-1, 0], 5), "epochs": 120}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    e = np.arange(1, len(hes) + 1)
    ax.plot(e, hes[:, 0], color=PAL["train"], label="Training loss" if not ro else "Pierderea de antrenare")
    ax.plot(e, hes[:, 1], color=PAL["valid"], label="Validation loss" if not ro else "Pierderea de validare")
    ax.axvline(N["es"]["best_ep"], color=npl.INK, lw=1, ls="--", label="Best validation epoch (weights kept)" if not ro else "Epoca cea mai bună (ponderile păstrate)")
    if stop_ep:
        ax.axvspan(N["es"]["best_ep"], stop_ep, color=npl.MUTED, alpha=0.15, lw=0, label=f"Patience = {PAT} epochs" if not ro else f"Răbdare = {PAT} de epoci")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch" if not ro else "Epoca")
    ax.set_ylabel("Cross-entropy (log scale)" if not ro else "Entropia încrucișată (scară log.)", fontsize=9.5)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "early_stopping")
npl.chart_meta(N, "early_stopping", "fraud", f"time-ordered split: {int(f_tr.sum()):,} training, {int(f_va.sum()):,} validation transactions",
               f"împărțire cronologică: {int(f_tr.sum()):,} de tranzacții de antrenare, {int(f_va.sum()):,} de validare".replace(",", "\\,"))

# ---- regularisation ablation on fraud (5 seeds, test PR-AUC with population weights)
CONF = {"none": dict(early=False), "es": dict(early=True), "wd": dict(early=False, wd=0.1), "dropout": dict(early=False, p=0.3),
        "bn": dict(early=False, bn=True), "all": dict(early=True, wd=0.1, p=0.3)}
CN = {"none": ("none (60 epochs)", "nimic (60 de epoci)"), "es": ("early stopping", "oprire timpurie"), "wd": ("weight decay 0.1", "penalizare 0,1"),
      "dropout": ("dropout 0.3", "dropout 0,3"), "bn": ("batch norm", "normalizare pe loturi"), "all": ("early stop. + decay + dropout", "oprire + penalizare + dropout")}


def ablation():
    out = {}
    for k, kw in CONF.items():
        out[k] = []
        for s in range(5):
            m, h, be = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=60, bs=256, seed=s, **kw)
            pt = proba(m, Xf[f_te])
            out[k].append([ap(yf[f_te], pt, wf[f_te]), float(roc_auc_score(yf[f_te], pt, sample_weight=wf[f_te])), be, len(h)])
    return out


ab = cached("ablation_v1", ablation)
N["abl"] = {k: {"ap": r2(np.mean([x[0] for x in v])), "ap_sd": r2(np.std([x[0] for x in v], ddof=1)),
                "auc": r2(np.mean([x[1] for x in v])), "epochs": r2(np.mean([x[3] for x in v]), 0)} for k, v in ab.items()}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for i, k in enumerate(CONF):
        v = np.array([x[0] for x in ab[k]])
        ax.scatter(v, np.full(5, i) + np.linspace(-0.12, 0.12, 5), s=16, color=PAL["model"] if k != "none" else PAL["data"])
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=npl.INK, lw=1.6)
    ax.set_yticks(range(len(CONF)))
    ax.set_yticklabels([CN[k][ro] for k in CONF], fontsize=8.5)
    ax.scatter([], [], s=16, color=PAL["model"], label="MLP 256-256, one seed" if not ro else "MLP 256-256, o inițializare")
    ax.plot([], [], color=npl.INK, lw=1.6, label="Mean of 5 seeds" if not ro else "Media a 5 inițializări")
    ax.set_xlabel("Test PR-AUC (population-weighted)" if not ro else "PR-AUC pe test (ponderat la populație)")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "reg_ablation", also_x=[ax] if ro else ())
npl.chart_meta(N, "reg_ablation", "fraud", f"time-ordered test set: last {int(f_te.sum()):,} transactions ({int(yf[f_te].sum())} frauds); 5 seeds",
               f"setul de test cronologic: ultimele {int(f_te.sum()):,} de tranzacții ({int(yf[f_te].sum())} de fraude); 5 inițializări".replace(",", "\\,"))

# ---- final comparison on the test set: logistic, regularised MLP, XGBoost (PR curves) + cost
lrf = LogisticRegression(max_iter=5000).fit(Xf[f_tr], yf[f_tr])
p_lr = lrf.predict_proba(Xf[f_te])[:, 1]
mlps = [fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=60, bs=256, seed=s, early=True, wd=0.1, p=0.3)[0] for s in range(5)]
p_ml = np.mean([proba(m, Xf[f_te]) for m in mlps], axis=0)
xgm = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                        early_stopping_rounds=100, random_state=SEED, n_jobs=1)
xgm.fit(Xf[f_tr], yf[f_tr], eval_set=[(Xf[f_va], yf[f_va])], verbose=False)
p_xg = xgm.predict_proba(Xf[f_te])[:, 1]
yt_, wt_ = yf[f_te], wf[f_te]
N["final"] = {}
for k, p in (("logit", p_lr), ("mlp", p_ml), ("xgb", p_xg)):
    pr, rc, th = precision_recall_curve(yt_, p, sample_weight=wt_)
    r_at = float(np.max(rc[pr >= 0.8])) if (pr >= 0.8).any() else 0.0
    N["final"][k] = {"ap": r2(ap(yt_, p, wt_)), "auc": r2(roc_auc_score(yt_, p, sample_weight=wt_)), "recall_at_p80_pct": r2(100 * r_at, 1)}
N["final"]["base_ap"] = r2(yt_.sum() / wt_.sum(), 4)
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for k, p, c, nm in (("logit", p_lr, PAL["benchmark"], ("Logistic regression", "Regresia logistică")),
                        ("mlp", p_ml, PAL["model"], ("Regularised MLP (5-seed ensemble)", "MLP regularizat (ansamblu de 5)")),
                        ("xgb", p_xg, PAL["alt"], ("XGBoost", "XGBoost"))):
        pr, rc, _ = precision_recall_curve(yt_, p, sample_weight=wt_)
        ax.plot(100 * rc, 100 * pr, color=c, label=f"{nm[ro]} (PR-AUC {N['final'][k]['ap']:.3f})".replace(".", "," if ro else "."))
    ax.axhline(100 * N["final"]["base_ap"], color=npl.MUTED, lw=0.8, ls=":", label="Fraud rate (random ranking)" if not ro else "Rata fraudei (ordonare aleatoare)")
    ax.set_xlabel("Recall: frauds caught (%)" if not ro else "Sensibilitatea: fraude detectate (%)")
    ax.set_ylabel("Precision (%)" if not ro else "Precizia (%)")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "fraud_pr", also_x=[ax] if ro else ())
npl.chart_meta(N, "fraud_pr", "fraud", f"time-ordered test set: {int(f_te.sum()):,} transactions, {int(yf[f_te].sum())} frauds, population-weighted",
               f"setul de test cronologic: {int(f_te.sum()):,} de tranzacții, {int(yf[f_te].sum())} de fraude, ponderat la populație".replace(",", "\\,"))

# cost: a missed fraud loses its amount; each alert costs a fixed review cost REVIEW (EUR, a constant chosen a priori)
REVIEW = 5.0


def cost_curve(p):
    ths = np.unique(np.quantile(p, np.linspace(0.90, 0.99995, 300)))
    out = []
    for t_ in ths:
        alert = p >= t_
        out.append([float((wt_[alert]).sum()), float(REVIEW * wt_[alert].sum() + amt[f_te][(~alert) & (yt_ == 1)].sum())])
    return np.array(out)


N["cost"] = {"review": REVIEW, "no_model": r2(amt[f_te][yt_ == 1].sum(), 0)}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for k, p, c, nm in (("logit", p_lr, PAL["benchmark"], ("Logistic regression", "Regresia logistică")),
                        ("mlp", p_ml, PAL["model"], ("Regularised MLP", "MLP regularizat")), ("xgb", p_xg, PAL["alt"], ("XGBoost", "XGBoost"))):
        cc = cost_curve(p)
        j = int(np.argmin(cc[:, 1]))
        N["cost"][k] = {"min": r2(cc[j, 1], 0), "alerts": r2(cc[j, 0], 0)}
        ax.plot(cc[:, 0], cc[:, 1] / 1000, color=c, label=nm[ro])
        ax.plot(cc[j, 0], cc[j, 1] / 1000, "o", color=c, ms=5)
    ax.axhline(N["cost"]["no_model"] / 1000, color=npl.MUTED, ls=":", lw=1, label="No alerts: all test frauds lost" if not ro else "Fără alerte: toate fraudele pierdute")
    ax.set_xscale("log")
    ax.set_xlabel("Alerts sent to review (population-weighted, log scale)" if not ro else "Alerte trimise la verificare (ponderat, scară log.)")
    ax.set_ylabel("Cost (thousand EUR)" if not ro else "Cost (mii EUR)")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "fraud_cost")
npl.chart_meta(N, "fraud_cost", "fraud", f"test period (last 20\\,\\% of the two days); review cost {REVIEW:.0f} EUR per alert",
               f"perioada de test (ultimele 20\\,\\% din cele două zile); costul verificării {REVIEW:.0f} EUR pe alertă")

# ---- validation design: time split vs random split vs oversampling before the split
def design_runs():
    out = {"time": [], "random": [], "oversample_first": []}
    for s in range(5):
        # (a) honest: time-ordered split (as above)
        m, _, _ = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(128, 128), epochs=40, bs=256, seed=s, early=True)
        out["time"].append(ap(yf[f_te], proba(m, Xf[f_te]), wf[f_te]))
        # (b) random stratified split of the same data (ignores time)
        idx = np.arange(len(yf))
        a, b = train_test_split(idx, test_size=0.2, stratify=yf, random_state=SEED + s)
        a, v = train_test_split(a, test_size=0.25, stratify=yf[a], random_state=SEED + s)
        mu_, sd_ = Xf0[a].mean(0), Xf0[a].std(0)
        Z_ = (Xf0 - mu_) / sd_
        m, _, _ = fit(Z_[a], yf[a], Z_[v], yf[v], hidden=(128, 128), epochs=40, bs=256, seed=s, early=True)
        out["random"].append(ap(yf[b], proba(m, Z_[b]), wf[b]))
        # (c) leakage: duplicate every fraud 20 times BEFORE the random split -> copies end up in train and test
        rng = np.random.default_rng(SEED + s)
        fi = np.where(yf == 1)[0]
        big = np.r_[idx, np.repeat(fi, 19)]
        rng.shuffle(big)
        yb_ = yf[big]
        a, b = train_test_split(np.arange(len(big)), test_size=0.2, stratify=yb_, random_state=SEED + s)
        a, v = train_test_split(a, test_size=0.25, stratify=yb_[a], random_state=SEED + s)
        Xb_ = Xf0[big]
        mu_, sd_ = Xb_[a].mean(0), Xb_[a].std(0)
        Z_ = (Xb_ - mu_) / sd_
        m, _, _ = fit(Z_[a], yb_[a], Z_[v], yb_[v], hidden=(128, 128), epochs=40, bs=256, seed=s, early=True)
        wb = np.where(yb_ == 1, 1.0 / 20, wf[big])                    # each duplicated fraud counts 1/20, as in the original data
        out["oversample_first"].append(ap(yb_[b], proba(m, Z_[b]), wb[b]))
    return out


dr = cached("design_v1", design_runs)
N["design"] = {k: {"ap": r2(np.mean(v)), "sd": r2(np.std(v, ddof=1))} for k, v in dr.items()}
DN = {"time": ("Time-ordered split\n(what deployment sees)", "Împărțire cronologică\n(ce vede modelul în producție)"),
      "random": ("Random split\n(ignores time)", "Împărțire aleatoare\n(ignoră timpul)"),
      "oversample_first": ("Oversampling before\nthe split (leakage)", "Supraeșantionare înainte\nde împărțire (scurgere)")}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for i, k in enumerate(DN):
        v = np.array(dr[k])
        ax.bar(i, v.mean(), color=[PAL["test"], PAL["alt"], PAL["positive"]][i], width=0.55, alpha=0.85)
        ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), v, color=npl.INK, s=10, zorder=3,
                   label=("One seed" if not ro else "O inițializare") if i == 0 else "_")
    ax.set_xticks(range(3))
    ax.set_xticklabels([DN[k][ro] for k in DN], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Estimated PR-AUC" if not ro else "PR-AUC estimată")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "split_leakage")
npl.chart_meta(N, "split_leakage", "fraud", "the same 28,924 transactions, MLP 128-128 with early stopping, 5 seeds",
               "aceleași 28\\,924 de tranzacții, MLP 128-128 cu oprire timpurie, 5 inițializări")

# ---- how noisy is a single hold-out estimate? German credit, logistic regression, 200 random 70/30 splits
def holdouts():
    out = []
    for s in range(200):
        a, b = train_test_split(np.arange(len(yg)), test_size=0.3, stratify=yg, random_state=SEED + s)
        xa, xb = encode_german(Xg_raw.iloc[a], [Xg_raw.iloc[b]])
        out.append(float(roc_auc_score(yg[b], LogisticRegression(C=1.0, max_iter=5000).fit(xa, yg[a]).predict_proba(xb)[:, 1])))
    return out


ho_ = np.array(cached("holdout_v1", holdouts))
N["holdout"] = {"n_splits": 200, "mean": r2(ho_.mean()), "sd": r2(ho_.std(ddof=1)), "min": r2(ho_.min()), "max": r2(ho_.max()),
                "p05": r2(np.percentile(ho_, 5)), "p95": r2(np.percentile(ho_, 95))}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.hist(ho_, bins=30, color=PAL["benchmark"], alpha=0.8, label="AUC of one random 70/30 split" if not ro else "AUC pe o împărțire aleatoare 70/30")
    ax.axvline(ho_.mean(), color=npl.INK, lw=1.2, label="Mean of 200 splits" if not ro else "Media celor 200 de împărțiri")
    for q in (np.percentile(ho_, 5), np.percentile(ho_, 95)):
        ax.axvline(q, color=npl.INK, lw=0.8, ls="--")
    ax.set_xlabel("Test AUC of logistic regression, German credit" if not ro else "AUC pe test al regresiei logistice, German credit")
    ax.set_ylabel("Number of splits" if not ro else "Numărul de împărțiri")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "holdout_noise", also_x=[ax] if ro else ())
npl.chart_meta(N, "holdout_noise", "german", "1,000 applicants, 200 random stratified splits", "1\\,000 de solicitanți, 200 de împărțiri aleatoare stratificate")

N["runtime_min"] = r2((time.time() - T0) / 60, 1)
with open(os.path.join(HERE, "L3_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print("wrote L3_numbers.json and", len(N["charts"]), "charts x 2 languages in", N["runtime_min"], "min")
