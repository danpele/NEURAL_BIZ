"""
analysis_L2.py -- Lecture 2 "Training: backpropagation and optimisation". All computation of the lecture:
writes lectures/L2/L2_numbers.json and charts/en|ro/*.pdf.

Data (data/SOURCES.md): California housing (20,640 block groups; regression testbed for the optimisers),
ULB credit-card fraud sample (28,924 transactions; classification, time-ordered split), S&P 500 daily closes
(Yahoo Finance; 1-day 99 % value at risk with a quantile network).
Models: MLPs in PyTorch, trained with SGD, momentum, RMSProp, Adam, AdamW and learning-rate schedules.
Seeds 20261108 + s. Hardware: CPU (PyTorch, 1 thread) for the small nets; the CPU/GPU timing frame also uses Apple MPS.
Long runs are cached in lectures/L2/cache/ (git-ignored). Run:  python3 lectures/L2/analysis_L2.py  [--fresh]
"""
import json
import os
import platform
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
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from scipy import stats  # noqa: E402

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


# =============================================================================================== California data
ca = npl.california()
feat = [c for c in ca.columns if c != "MedHouseVal_100k_usd"]
Xc = ca[feat].values.astype(np.float32)
yc = ca["MedHouseVal_100k_usd"].values.astype(np.float32)
i_tr, i_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, random_state=SEED)
mu, sd = Xc[i_tr].mean(0), Xc[i_tr].std(0)
Z = (Xc - mu) / sd
Xtr, Xva, Xte = Z[i_tr], Z[i_va], Z[i_te]
ytr, yva, yte = yc[i_tr], yc[i_va], yc[i_te]
lin = LinearRegression().fit(Xtr, ytr)
mse_lin_val = float(np.mean((lin.predict(Xva) - yva) ** 2))
mse_lin_te = float(np.mean((lin.predict(Xte) - yte) ** 2))
N["cal"] = {"n": int(len(ca)), "n_train": int(len(i_tr)), "n_val": int(len(i_va)), "n_test": int(len(i_te)), "d": len(feat),
            "mse_lin_val": r2(mse_lin_val), "mse_lin_test": r2(mse_lin_te), "var_y": r2(np.var(yte))}


def make_mlp(d, hidden=(64, 64), act=nn.ReLU, out=1):
    layers, k = [], d
    for h in hidden:
        layers += [nn.Linear(k, h), act()]
        k = h
    layers.append(nn.Linear(k, out))
    return nn.Sequential(*layers)


def make_opt(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    raise ValueError(name)


def run_reg(opt_name, lr, seed=0, epochs=60, bs=128, sched=None, hidden=(64, 64), track_steps=False):
    """Train an MLP on California housing; returns validation MSE per epoch (and the test MSE of the last epoch)."""
    torch.manual_seed(SEED + seed)
    rng = np.random.default_rng(SEED + seed)
    m = make_mlp(Xtr.shape[1], hidden)
    opt = make_opt(opt_name, m.parameters(), lr)
    steps_per_epoch = int(np.ceil(len(Xtr) / bs))
    total = epochs * steps_per_epoch
    if sched == "step":
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=20 * steps_per_epoch, gamma=0.1)
    elif sched == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    elif sched == "onecycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.3)
    else:
        sch = None
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr).view(-1, 1)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva).view(-1, 1)
    lf = nn.MSELoss()
    val, step_loss, lrs = [], [], []
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            loss = lf(m(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            lrs.append(opt.param_groups[0]["lr"])
            if sch is not None:
                sch.step()
            if track_steps:
                step_loss.append(float(loss))
        m.eval()
        with torch.no_grad():
            v = float(lf(m(Xv), yv))
        val.append(v if np.isfinite(v) else 1e9)
    with torch.no_grad():
        te = float(lf(m(torch.tensor(Xte)), torch.tensor(yte).view(-1, 1)))
    out = {"val": val, "test": te if np.isfinite(te) else 1e9}
    if track_steps:
        out["steps"] = step_loss
    if sched is not None:
        out["lr"] = lrs[::steps_per_epoch]
    return out


# =============================================================================================== 1. the loss surface of a real linear regression
x_inc = ca["MedInc"].values.astype(float)
y_all = yc.astype(float)


def surface(xf):
    X = np.c_[np.ones_like(xf), xf]
    Hs = 2 * X.T @ X / len(xf)
    w_star = np.linalg.lstsq(X, y_all, rcond=None)[0]
    return X, Hs, w_star


def gd_path(X, Hs, w0, lr, steps, beta=0.0):
    w, v, path = np.array(w0, float), np.zeros(2), [np.array(w0, float)]
    for _ in range(steps):
        g = 2 * X.T @ (X @ w - y_all) / len(y_all)
        v = beta * v + g
        w = w - lr * v
        path.append(w.copy())
    return np.array(path)


Xraw, Hraw, wraw = surface(x_inc)
Xstd, Hstd, wstd = surface((x_inc - x_inc.mean()) / x_inc.std())
ev_raw, ev_std = np.linalg.eigvalsh(Hraw), np.linalg.eigvalsh(Hstd)
lr_raw = 1.0 / ev_raw.max()          # step that is stable for the steepest direction
lr_std = 1.0 / ev_std.max()
w0 = np.array([4.0, -0.5])
w0s = np.array([4.0, -1.0])
p_gd_raw = gd_path(Xraw, Hraw, w0, lr_raw, 60)
p_mom_raw = gd_path(Xraw, Hraw, w0, lr_raw * 0.5, 60, beta=0.9)
p_gd_std = gd_path(Xstd, Hstd, w0s, lr_std, 60)


def mse_at(X, w):
    return float(np.mean((X @ w - y_all) ** 2))


N["surface"] = {"cond_raw": r2(ev_raw.max() / ev_raw.min(), 1), "cond_std": r2(ev_std.max() / ev_std.min(), 2),
                "lr_raw": r2(lr_raw, 4), "lr_std": r2(lr_std, 3), "lr_max_raw": r2(2 * lr_raw, 4),
                "lam_max_raw": r2(ev_raw.max(), 2), "lam_min_raw": r2(ev_raw.min(), 3), "slope": r2(wraw[1], 4), "intercept": r2(wraw[0], 4),
                "mse_star": r2(mse_at(Xraw, wraw), 4), "mse_gd_raw_60": r2(mse_at(Xraw, p_gd_raw[-1]), 4),
                "mse_mom_raw_60": r2(mse_at(Xraw, p_mom_raw[-1]), 4), "mse_gd_std_60": r2(mse_at(Xstd, p_gd_std[-1]), 4),
                "mse_start": r2(mse_at(Xraw, w0), 3), "steps": 60}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, (X, ws, paths, xl) in zip(axs, (
            (Xraw, wraw, [(p_gd_raw, PAL["data"], "GD"), (p_mom_raw, PAL["model"], "momentum" if not ro else "moment")],
             "raw income" if not ro else "venit brut"),
            (Xstd, wstd, [(p_gd_std, PAL["data"], "GD")], "standardised income" if not ro else "venit standardizat"))):
        a = np.linspace(min(ws[0], -0.5) - 1.0, 5.0, 160)
        bgrid = np.linspace(-1.2, 1.4, 160)
        A_, B_ = np.meshgrid(a, bgrid)
        L = (np.mean(y_all ** 2) - 2 * (A_ * np.mean(X[:, 0] * y_all) + B_ * np.mean(X[:, 1] * y_all))
             + A_ ** 2 * np.mean(X[:, 0] ** 2) + 2 * A_ * B_ * np.mean(X[:, 0] * X[:, 1]) + B_ ** 2 * np.mean(X[:, 1] ** 2))
        ax.contour(A_, B_, L, levels=np.quantile(L, [0.01, 0.03, 0.08, 0.15, 0.3, 0.5, 0.7]), colors=[npl.MUTED], linewidths=0.6)
        for pth, c, lab in paths:
            ax.plot(pth[:, 0], pth[:, 1], "o-", ms=2, lw=1, color=c, label=lab)
        ax.plot(*ws, "*", color=npl.INK, ms=9, label="Least-squares minimum" if not ro else "Minimul MCMMP")
        ax.set_xlabel("Intercept $a$" if not ro else "Termenul liber $a$")
        ax.set_ylabel("Slope $b$" if not ro else "Panta $b$")
        ax.text(0.97, 0.95, xl, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "gd_paths", also_x=list(fig.axes) if ro else ())
npl.chart_meta(N, "gd_paths", "california", f"{len(ca):,} block groups, 1990 census", f"{len(ca):,} de grupuri de locuințe, recensămîntul din 1990".replace(",", "\\,"))

# =============================================================================================== 2. backpropagation by hand on one real block group
torch.manual_seed(SEED)
xb = torch.tensor(Xte[0, [0, 6]], dtype=torch.float64)     # median income and latitude (standardised)
yb = torch.tensor(float(yte[0]), dtype=torch.float64)
W1 = (torch.randn(2, 2, dtype=torch.float64) * 0.5).round(decimals=2)
b1 = torch.zeros(2, dtype=torch.float64)
w2 = (torch.randn(2, dtype=torch.float64) * 0.5).round(decimals=2)
b2 = torch.tensor(0.0, dtype=torch.float64)
z1 = W1 @ xb + b1
h1 = torch.relu(z1)
yhat = w2 @ h1 + b2
Lb = 0.5 * (yhat - yb) ** 2
d2 = yhat - yb
gw2 = d2 * h1
d1 = d2 * w2 * (z1 > 0).double()
gW1 = torch.outer(d1, xb)
# check with autograd
P = [W1.clone().requires_grad_(), b1.clone().requires_grad_(), w2.clone().requires_grad_(), b2.clone().requires_grad_()]
L2_ = 0.5 * (P[2] @ torch.relu(P[0] @ xb + P[1]) + P[3] - yb) ** 2
L2_.backward()
N["bp"] = {"x1": r2(xb[0], 2), "x2": r2(xb[1], 2), "y": r2(yb, 3),
           "W11": r2(W1[0, 0], 2), "W12": r2(W1[0, 1], 2), "W21": r2(W1[1, 0], 2), "W22": r2(W1[1, 1], 2),
           "w21": r2(w2[0], 2), "w22": r2(w2[1], 2),
           "z1": r2(z1[0], 3), "z2": r2(z1[1], 3), "h1": r2(h1[0], 3), "h2": r2(h1[1], 3), "yhat": r2(yhat, 3), "L": r2(Lb, 3),
           "d2": r2(d2, 3), "gw21": r2(gw2[0], 3), "gw22": r2(gw2[1], 3), "d11": r2(d1[0], 3), "d12": r2(d1[1], 3),
           "gW11": r2(gW1[0, 0], 3), "gW12": r2(gW1[0, 1], 3), "gW21": r2(gW1[1, 0], 3), "gW22": r2(gW1[1, 1], 3),
           "max_abs_diff_autograd": float(max((gW1 - P[0].grad).abs().max(), (gw2 - P[2].grad).abs().max()))}

# gradient check of a full MLP 64-64 on a mini-batch: autograd vs central finite differences (float64)
torch.manual_seed(SEED)
mg = make_mlp(Xtr.shape[1], (64, 64)).double()
Xg = torch.tensor(Xtr[:256], dtype=torch.float64)
yg = torch.tensor(ytr[:256], dtype=torch.float64).view(-1, 1)
lossg = nn.MSELoss()(mg(Xg), yg)
grads = torch.autograd.grad(lossg, list(mg.parameters()))
rel = []
rng = np.random.default_rng(SEED)
eps = 1e-6
with torch.no_grad():
    for _ in range(30):
        li = rng.integers(0, len(grads))
        p = list(mg.parameters())[li]
        flat = p.view(-1)
        j = rng.integers(0, flat.numel())
        old = flat[j].item()
        flat[j] = old + eps
        lp = nn.MSELoss()(mg(Xg), yg).item()
        flat[j] = old - eps
        lm = nn.MSELoss()(mg(Xg), yg).item()
        flat[j] = old
        fd = (lp - lm) / (2 * eps)
        an = grads[li].view(-1)[j].item()
        rel.append(abs(fd - an) / max(1e-12, abs(fd) + abs(an)))
N["gradcheck"] = {"n_checked": 30, "max_rel_err": float(f"{max(rel):.1e}"), "n_params": int(sum(p.numel() for p in mg.parameters()))}

# =============================================================================================== 3. learning-rate sweep (plain SGD)
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def lr_sweep():
    out = {}
    for lr in LRS:
        out[str(lr)] = [run_reg("sgd", lr, seed=s, epochs=30)["val"][-1] for s in range(3)]
    return out


sw = cached("lr_sweep_v1", lr_sweep)
DIV = 5.0                                           # a run is "diverged" when its validation MSE exceeds this (or is not finite)
sw_m = {k: (np.median(v) if max(v) < DIV else np.nan) for k, v in sw.items()}
ok = {k: v for k, v in sw_m.items() if np.isfinite(v)}
best_lr = min(ok, key=ok.get)
N["lrsweep"] = {"best_lr": float(best_lr), "best_mse": r2(ok[best_lr]), "smallest_mse": r2(sw_m[str(LRS[0])]),
                "first_diverged": float(min([float(k) for k, v in sw_m.items() if not np.isfinite(v)], default=np.nan)),
                "epochs": 30, "batch": 128}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    xs = np.array(LRS)
    for s in range(3):
        v = np.array([sw[str(lr)][s] for lr in LRS])
        good = v < DIV
        ax.plot(xs[good], v[good], "o", color=PAL["model"], ms=4, alpha=0.6,
                label="SGD, one seed" if (s == 0 and not ro) else ("SGD, o inițializare" if s == 0 else "_"))
        if (~good).any():
            ax.plot(xs[~good], np.full((~good).sum(), 1.55), "x", color=PAL["positive"], ms=7,
                    label=("Diverged" if not ro else "Divergent") if s == 0 else "_")
    med = np.array([sw_m[str(lr)] for lr in LRS])
    ax.plot(xs, med, "-", color=PAL["model"], lw=1.2)
    ax.axhline(mse_lin_val, color=PAL["benchmark"], lw=1.2, label="Linear regression" if not ro else "Regresia liniară")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.65)
    ax.set_xlabel("Learning rate $\\eta$ (log scale)" if not ro else "Rata de învățare $\\eta$ (scară logaritmică)")
    ax.set_ylabel("Validation MSE" if not ro else "MSE pe validare")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "lr_sweep")
npl.chart_meta(N, "lr_sweep", "california", f"{len(i_tr):,} training and {len(i_va):,} validation block groups; 30 epochs",
               f"{len(i_tr):,} de grupuri de antrenare și {len(i_va):,} de validare; 30 de epoci".replace(",", "\\,"))

# =============================================================================================== 4. optimisers, tuned on validation
OPTS = {"sgd": [0.01, 0.03, 0.1], "momentum": [0.001, 0.003, 0.01], "rmsprop": [3e-4, 1e-3, 3e-3],
        "adam": [3e-4, 1e-3, 3e-3], "adamw": [3e-4, 1e-3, 3e-3]}
ONAME = {"sgd": ("SGD", "SGD"), "momentum": ("SGD + momentum", "SGD + moment"), "rmsprop": ("RMSProp", "RMSProp"),
         "adam": ("Adam", "Adam"), "adamw": ("AdamW", "AdamW")}
OCOL = {"sgd": PAL["data"], "momentum": PAL["c7"], "rmsprop": PAL["alt"], "adam": PAL["model"], "adamw": PAL["c5"]}


def opt_tune():
    out = {}
    for o, grid in OPTS.items():
        for lr in grid:
            out[f"{o}|{lr}"] = [run_reg(o, lr, seed=s, epochs=60) for s in range(3)]
    return out


ot = cached("opt_tune_v1", opt_tune)
best = {}
for o, grid in OPTS.items():
    scores = {lr: np.mean([min(r["val"]) for r in ot[f"{o}|{lr}"]]) for lr in grid}
    best[o] = min(scores, key=scores.get)


def opt_final():
    return {o: [run_reg(o, best[o], seed=10 + s, epochs=60) for s in range(5)] for o in OPTS}


of = cached("opt_final_v1", opt_final)
N["opt"] = {}
for o in OPTS:
    vals = np.array([r["val"] for r in of[o]])
    te = np.array([r["test"] for r in of[o]])
    ep_to = [int(np.argmax(np.array(v) < mse_lin_val * 0.6)) + 1 if (np.array(v) < mse_lin_val * 0.6).any() else None for v in vals]
    N["opt"][o] = {"lr": float(best[o]), "val_final": r2(vals[:, -1].mean()), "test_mean": r2(te.mean()), "test_sd": r2(te.std(ddof=1)),
                   "test_min": r2(te.min()), "test_max": r2(te.max()), "val_ep10": r2(vals[:, 9].mean()),
                   "ep_to_target": None if None in ep_to else r2(np.mean(ep_to), 1)}
N["opt_target"] = r2(mse_lin_val * 0.6)
# paired comparison Adam vs SGD over 5 seeds (test MSE)
ta, ts = np.array([r["test"] for r in of["adam"]]), np.array([r["test"] for r in of["sgd"]])
tt = stats.ttest_rel(ts, ta)
N["opt_test"] = {"diff_sgd_adam": r2((ts - ta).mean()), "t": r2(tt.statistic, 2), "p": r2(tt.pvalue)}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for o in OPTS:
        v = np.array([r["val"] for r in of[o]]).mean(0)
        ax.plot(np.arange(1, len(v) + 1), v, color=OCOL[o], label=f"{ONAME[o][ro]} ($\\eta$={best[o]:g})".replace(".", "," if ro else "."))
    ax.axhline(mse_lin_val, color=PAL["benchmark"], lw=1.1, ls="--", label="Linear regression" if not ro else "Regresia liniară")
    ax.set_ylim(0.2, 0.62)
    ax.set_xlabel("Epoch" if not ro else "Epoca")
    ax.set_ylabel("Validation MSE" if not ro else "MSE pe validare")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "optimizers")
npl.chart_meta(N, "optimizers", "california", f"{len(i_tr):,} training and {len(i_va):,} validation block groups",
               f"{len(i_tr):,} de grupuri de antrenare și {len(i_va):,} de validare".replace(",", "\\,"))
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for i, o in enumerate(OPTS):
        te = np.array([r["test"] for r in of[o]])
        ax.scatter(te, np.full(5, i) + np.linspace(-0.12, 0.12, 5), color=OCOL[o], s=16, label=ONAME[o][ro])
        ax.plot([te.mean()] * 2, [i - 0.3, i + 0.3], color=npl.INK, lw=1.6)
    ax.axvline(mse_lin_te, color=PAL["benchmark"], lw=1.2, ls="--", label="Linear regression" if not ro else "Regresia liniară")
    ax.set_yticks(range(len(OPTS)))
    ax.set_yticklabels([ONAME[o][ro] for o in OPTS], fontsize=9)
    ax.set_xlabel("Test MSE after 60 epochs (5 seeds; bar = mean)" if not ro else "MSE pe test după 60 de epoci (5 inițializări; bara = media)")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "optimizer_seeds", also_x=[ax] if ro else ())
npl.chart_meta(N, "optimizer_seeds", "california", f"{len(i_te):,} test block groups", f"{len(i_te):,} de grupuri de testare".replace(",", "\\,"))

# =============================================================================================== 5. batch size
BS = [16, 128, 1024]


def batch_runs():
    out = {}
    for b in BS:
        t0 = time.time()
        r = run_reg("adam", 1e-3, seed=0, epochs=40, bs=b)
        r["sec"] = time.time() - t0
        out[str(b)] = r
    return out


br = cached("batch_v1", batch_runs)
N["batch"] = {str(b): {"val_final": r2(br[str(b)]["val"][-1]), "val_ep5": r2(br[str(b)]["val"][4]),
                       "steps_per_epoch": int(np.ceil(len(i_tr) / b)), "sec": r2(br[str(b)]["sec"], 1)} for b in BS}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for b, c in zip(BS, (PAL["data"], PAL["model"], PAL["c7"])):
        v = np.array(br[str(b)]["val"])
        spe = int(np.ceil(len(i_tr) / b))
        axs[0].plot(np.arange(1, len(v) + 1), v, color=c, label=(f"batch {b}" if not ro else f"lot de {b}"))
        axs[1].plot(np.arange(1, len(v) + 1) * spe, v, color=c)
    axs[1].set_xscale("log")
    axs[0].set_ylim(0.2, 0.7)
    axs[0].set_xlabel("Epoch" if not ro else "Epoca")
    axs[1].set_xlabel("Gradient steps (log scale)" if not ro else "Pași de gradient (scară log.)")
    axs[0].set_ylabel("Validation MSE" if not ro else "MSE pe validare")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "batch_size")
npl.chart_meta(N, "batch_size", "california", "Adam, learning rate 0.001, 40 epochs", "Adam, rata de învățare 0,001, 40 de epoci")

# =============================================================================================== 6. learning-rate schedules
SCH = {"const": None, "step": "step", "cosine": "cosine", "onecycle": "onecycle"}
SNAME = {"const": ("constant", "constantă"), "step": ("step decay (x0.1 every 20 epochs)", "scădere în trepte (x0,1 la 20 de epoci)"),
         "cosine": ("cosine annealing", "descreștere cosinus"), "onecycle": ("one-cycle (warm-up 30%)", "un singur ciclu (încălzire 30%)")}
SCOL = {"const": PAL["data"], "step": PAL["alt"], "cosine": PAL["model"], "onecycle": PAL["c7"]}


def sched_runs():
    out = {}
    for k, s_ in SCH.items():
        rr = [run_reg("momentum", 0.01, seed=20 + s, epochs=60, sched=s_) for s in range(3)]
        if s_ is None:
            for r in rr:
                r["lr"] = [0.01] * 60
        out[k] = rr
    return out


sr = cached("sched_v1", sched_runs)
N["sched"] = {k: {"val_final": r2(np.mean([r["val"][-1] for r in sr[k]])), "test": r2(np.mean([r["test"] for r in sr[k]])),
                  "val_sd": r2(np.std([r["val"][-1] for r in sr[k]], ddof=1))} for k in SCH}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for k in SCH:
        lrs = np.array(sr[k][0]["lr"][:60])
        axs[0].plot(np.arange(1, len(lrs) + 1), lrs, color=SCOL[k], label=SNAME[k][ro])
        v = np.array([r["val"] for r in sr[k]]).mean(0)
        axs[1].plot(np.arange(1, len(v) + 1), v, color=SCOL[k])
    axs[0].set_ylabel("Learning rate $\\eta_t$" if not ro else "Rata de învățare $\\eta_t$")
    axs[1].set_ylabel("Validation MSE" if not ro else "MSE pe validare")
    axs[1].set_ylim(0.22, 0.45)
    for ax in axs:
        ax.set_xlabel("Epoch" if not ro else "Epoca")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "schedules")
npl.chart_meta(N, "schedules", "california", "SGD with momentum 0.9, peak rate 0.01, 3 seeds",
               "SGD cu moment 0,9, rata maximă 0,01, 3 inițializări")

# =============================================================================================== 7. vanishing gradients and initialisation
def grad_norms(act, init, depth=10, width=64, seed=0):
    torch.manual_seed(SEED + seed)
    layers, k = [], Xtr.shape[1]
    lin_layers = []
    for _ in range(depth):
        l_ = nn.Linear(k, width)
        lin_layers.append(l_)
        layers += [l_, act()]
        k = width
    last = nn.Linear(k, 1)
    lin_layers.append(last)
    m = nn.Sequential(*layers, last)
    for l_ in lin_layers:
        if init == "xavier":
            nn.init.xavier_normal_(l_.weight)
        elif init == "he":
            nn.init.kaiming_normal_(l_.weight, nonlinearity="relu")
        elif init == "small":
            nn.init.normal_(l_.weight, 0, 0.01)
        elif init == "large":
            nn.init.normal_(l_.weight, 0, 1.0)
        nn.init.zeros_(l_.bias)
    xb_ = torch.tensor(Xtr[:512])
    yb_ = torch.tensor(ytr[:512]).view(-1, 1)
    loss = nn.MSELoss()(m(xb_), yb_)
    loss.backward()
    return [float(l_.weight.grad.norm()) for l_ in lin_layers[:-1]], m


VG = {"sig": (nn.Sigmoid, "xavier", ("sigmoid, Xavier init.", "sigmoidă, inițializare Xavier")),
      "tanh": (nn.Tanh, "xavier", ("tanh, Xavier init.", "tanh, inițializare Xavier")),
      "relu_he": (nn.ReLU, "he", ("ReLU, He init.", "ReLU, inițializare He")),
      "relu_small": (nn.ReLU, "small", ("ReLU, N(0, 0.01²) init.", "ReLU, inițializare N(0; 0,01²)"))}
gn = {k: np.mean([grad_norms(a, i, seed=s)[0] for s in range(5)], axis=0) for k, (a, i, _) in VG.items()}
N["vanish"] = {k: {"first": float(f"{gn[k][0]:.2e}"), "last": float(f"{gn[k][-1]:.2e}"), "ratio": float(f"{gn[k][0] / gn[k][-1]:.2e}")}
               for k in VG}
N["vanish"]["depth"] = 10
VCOL = {"sig": PAL["data"], "tanh": PAL["benchmark"], "relu_he": PAL["model"], "relu_small": PAL["c7"]}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for k, (_, _, nm) in VG.items():
        ax.plot(np.arange(1, 11), gn[k], "o-", ms=3.5, color=VCOL[k], label=nm[ro])
    ax.set_yscale("log")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Layer (1 = next to the input)" if not ro else "Stratul (1 = lîngă intrare)")
    ax.set_ylabel("$\\|\\partial\\mathcal{L}/\\partial W_l\\|$ (log scale)" if not ro else "$\\|\\partial\\mathcal{L}/\\partial W_l\\|$ (scară log.)")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "vanishing")
npl.chart_meta(N, "vanishing", "california", "10 hidden layers of 64 units, one batch of 512 training block groups, 5 seeds",
               "10 straturi ascunse de 64 de neuroni, un lot de 512 grupuri de antrenare, 5 inițializări")

# =============================================================================================== 8. fraud: SGD vs Adam on a time-ordered split
fr = npl.fraud().sort_values("Time").reset_index(drop=True)
vcols = [f"V{i}" for i in range(1, 29)]
Xf = np.c_[fr[vcols].values, np.log1p(fr["Amount"].values)].astype(np.float32)
yf = fr["Class"].values.astype(np.float32)
wf = fr["weight"].values
t_cut1, t_cut2 = np.quantile(fr["Time"], [0.6, 0.8])
f_tr = (fr["Time"] <= t_cut1).values
f_va = ((fr["Time"] > t_cut1) & (fr["Time"] <= t_cut2)).values
f_te = (fr["Time"] > t_cut2).values
fmu, fsd = Xf[f_tr].mean(0), Xf[f_tr].std(0)
Xf = (Xf - fmu) / fsd
N["fraud"] = {"n": int(len(fr)), "frauds": int(yf.sum()), "n_train": int(f_tr.sum()), "n_val": int(f_va.sum()), "n_test": int(f_te.sum()),
              "fr_train": int(yf[f_tr].sum()), "fr_val": int(yf[f_va].sum()), "fr_test": int(yf[f_te].sum()),
              "cut1_h": r2(t_cut1 / 3600, 1), "cut2_h": r2(t_cut2 / 3600, 1),
              "prev_full_pct": r2(100 * yf.sum() / wf.sum(), 3)}


def fraud_run(opt_name, lr, seed, epochs=30, bs=256):
    torch.manual_seed(SEED + seed)
    rng = np.random.default_rng(SEED + seed)
    m = make_mlp(Xf.shape[1], (64, 64))
    opt = make_opt(opt_name, m.parameters(), lr)
    Xt, yt = torch.tensor(Xf[f_tr]), torch.tensor(yf[f_tr]).view(-1, 1)
    Xv = torch.tensor(Xf[f_va])
    lf = nn.BCEWithLogitsLoss()
    ap = []
    for ep in range(epochs):
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            lf(m(Xt[b]), yt[b]).backward()
            opt.step()
        with torch.no_grad():
            pv = torch.sigmoid(m(Xv)).numpy().ravel()
        ap.append(average_precision_score(yf[f_va], pv, sample_weight=wf[f_va]))
    with torch.no_grad():
        pt = torch.sigmoid(m(torch.tensor(Xf[f_te]))).numpy().ravel()
    return {"ap": ap, "ap_test": float(average_precision_score(yf[f_te], pt, sample_weight=wf[f_te])),
            "auc_test": float(roc_auc_score(yf[f_te], pt, sample_weight=wf[f_te]))}


def fraud_all():
    return {k: [fraud_run(o, lr, s) for s in range(5)] for k, (o, lr) in
            {"sgd": ("sgd", 0.01), "momentum": ("momentum", 0.01), "adam": ("adam", 1e-3)}.items()}


fa = cached("fraud_opt_v1", fraud_all)
lrf = LogisticRegression(max_iter=5000).fit(Xf[f_tr], yf[f_tr])
p_lrf = lrf.predict_proba(Xf[f_te])[:, 1]
N["fraud"]["ap_logit_test"] = r2(average_precision_score(yf[f_te], p_lrf, sample_weight=wf[f_te]))
for k in fa:
    aps = np.array([r["ap"] for r in fa[k]])
    N["fraud"][k] = {"ap_ep1": r2(aps[:, 0].mean()), "ap_ep5": r2(aps[:, 4].mean()), "ap_final": r2(aps[:, -1].mean()),
                     "ap_test": r2(np.mean([r["ap_test"] for r in fa[k]])), "ap_test_sd": r2(np.std([r["ap_test"] for r in fa[k]], ddof=1)),
                     "auc_test": r2(np.mean([r["auc_test"] for r in fa[k]]))}
FN_ = {"sgd": ("SGD, $\\eta$=0.01", "SGD, $\\eta$=0,01"), "momentum": ("SGD + momentum 0.9, $\\eta$=0.01", "SGD + moment 0,9, $\\eta$=0,01"),
       "adam": ("Adam, $\\eta$=0.001", "Adam, $\\eta$=0,001")}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for k, c in (("sgd", PAL["data"]), ("momentum", PAL["c7"]), ("adam", PAL["model"])):
        aps = np.array([r["ap"] for r in fa[k]])
        ax.fill_between(np.arange(1, aps.shape[1] + 1), aps.min(0), aps.max(0), color=c, alpha=0.15, lw=0)
        ax.plot(np.arange(1, aps.shape[1] + 1), aps.mean(0), color=c, label=FN_[k][ro])
    ax.set_xlabel("Epoch" if not ro else "Epoca")
    ax.set_ylabel("Validation PR-AUC" if not ro else "PR-AUC pe validare")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "fraud_optimizers")
npl.chart_meta(N, "fraud_optimizers", "fraud", f"September 2013, 2 days; train = first {N['fraud']['cut1_h']:.1f} hours, "
               f"validation to hour {N['fraud']['cut2_h']:.1f}; 5 seeds",
               f"septembrie 2013, 2 zile; antrenare = primele {N['fraud']['cut1_h']:.1f} ore, validare pînă la ora {N['fraud']['cut2_h']:.1f}; "
               "5 inițializări".replace(".", ","))

# =============================================================================================== 9. 1-day 99% VaR of the S&P 500 with a quantile network
px = npl.markets("adjclose", ["SP500"])["SP500"].dropna()
end = (px.index.max() + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.Timedelta(days=1)
px = px.loc[:end]
r = 100 * np.log(px).diff().dropna()
feat_df = pd.DataFrame({"r": r, "ar": r.abs(), "s5": r.rolling(5).std(), "s22": r.rolling(22).std(), "s66": r.rolling(66).std(),
                        "m5": r.rolling(5).mean()})
target = r.shift(-1)
dfv = pd.concat([feat_df, target.rename("y")], axis=1).dropna()
TAU = 0.01
tr_end, va_end = pd.Timestamp("2016-12-31"), pd.Timestamp("2019-12-31")
v_tr = dfv.index <= tr_end
v_va = (dfv.index > tr_end) & (dfv.index <= va_end)
v_te = dfv.index > va_end
Xv_ = dfv[feat_df.columns].values.astype(np.float32)
vmu, vsd = Xv_[v_tr].mean(0), Xv_[v_tr].std(0)
Xv_ = (Xv_ - vmu) / vsd
yv_ = dfv["y"].values.astype(np.float32)
tdate = pd.Series(r.index, index=r.index).shift(-1).reindex(dfv.index).values      # date of the return being forecast


def pinball(q, y, tau=TAU):
    u = y - q
    return torch.mean(torch.maximum(tau * u, (tau - 1) * u))


def var_run(seed, epochs=400):
    torch.manual_seed(SEED + seed)
    m = make_mlp(Xv_.shape[1], (32, 32))
    with torch.no_grad():
        m[-1].bias.fill_(float(np.quantile(yv_[v_tr], TAU)))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(Xv_[v_tr]), torch.tensor(yv_[v_tr]).view(-1, 1)
    Xva_, yva_ = torch.tensor(Xv_[v_va]), torch.tensor(yv_[v_va]).view(-1, 1)
    best, state, bad, rng = np.inf, None, 0, np.random.default_rng(SEED + seed)
    for ep in range(epochs):
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            opt.zero_grad()
            pinball(m(Xt[b]), yt[b]).backward()
            opt.step()
        with torch.no_grad():
            vl = float(pinball(m(Xva_), yva_))
        if vl < best - 1e-6:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    m.load_state_dict(state)
    with torch.no_grad():
        return m(torch.tensor(Xv_)).numpy().ravel().tolist()


VAR_SEEDS = 5
vq = np.array(cached("var_v1", lambda: [var_run(s) for s in range(VAR_SEEDS)]))
q_nn = vq.mean(0)                      # ensemble of 5 seeds
yy = dfv["y"].values
# benchmarks: 250-day historical simulation and RiskMetrics EWMA-normal (lambda 0.94), both from data up to day t
hs = r.rolling(250).quantile(TAU).reindex(dfv.index).values
ew = r.pow(2).ewm(alpha=0.06, adjust=False).mean().pow(0.5).reindex(dfv.index).values
q_ew = stats.norm.ppf(TAU) * ew


def kupiec(viol, p=TAU):
    n, x = len(viol), int(viol.sum())
    ph = x / n
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return x, float(lr), float(stats.chi2.sf(lr, 1))


def pin_np(q, y, tau=TAU):
    u = y - q
    return float(np.mean(np.maximum(tau * u, (tau - 1) * u)))


N["var"] = {"tau_pct": 1, "n_test": int(v_te.sum()), "test_start": str(pd.Timestamp(tdate[v_te][0]).date()),
            "test_end": str(pd.Timestamp(tdate[v_te][-1]).date()),
            "train_start": str(dfv.index[v_tr][0].date()), "expected": r2(TAU * v_te.sum(), 1), "seeds": VAR_SEEDS}
for k, q in (("nn", q_nn), ("hs", hs), ("ewma", q_ew)):
    viol = yy[v_te] < q[v_te]
    x, lr_, p_ = kupiec(viol)
    N["var"][k] = {"viol": x, "rate_pct": r2(100 * x / v_te.sum(), 2), "lr": r2(lr_, 2), "p": r2(p_), "pinball": r2(pin_np(q[v_te], yy[v_te]), 4),
                   "mean_var": r2(-np.mean(q[v_te]), 2)}
seed_rates = [100 * np.mean(yy[v_te] < vq[s][v_te]) for s in range(VAR_SEEDS)]
N["var"]["nn_seed_rate_min"] = r2(min(seed_rates), 2)
N["var"]["nn_seed_rate_max"] = r2(max(seed_rates), 2)
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    idx_te = pd.DatetimeIndex(tdate[v_te])
    ax.plot(idx_te, yy[v_te], color=npl.MUTED, lw=0.5, label="S&P 500 daily log return (%)" if not ro else "Randamentul logaritmic zilnic S&P 500 (%)")
    ax.plot(idx_te, q_nn[v_te], color=PAL["model"], lw=1.0, label="Quantile MLP, 99% VaR" if not ro else "MLP cuantilic, VaR 99%")
    ax.plot(idx_te, hs[v_te], color=PAL["benchmark"], lw=1.0, label="Historical simulation, 250 days" if not ro else "Simulare istorică, 250 de zile")
    vi = yy[v_te] < q_nn[v_te]
    ax.plot(idx_te[vi], yy[v_te][vi], "o", color=PAL["positive"], ms=3.5, label="Violation of the MLP VaR" if not ro else "Depășire a VaR MLP")
    ax.set_ylim(-13, 9)
    ax.set_xlim(idx_te.min(), idx_te.max())
    ax.set_ylabel("%")
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "var_backtest")
npl.chart_meta(N, "var_backtest", "yahoo", npl.period(tdate[v_te][0], tdate[v_te][-1], "en") + ", trading days (test period)",
               npl.period(tdate[v_te][0], tdate[v_te][-1], "ro") + ", zile de tranzacționare (perioada de test)")

# =============================================================================================== 10. CPU vs GPU (MPS) timing on this machine
def timing(dev, steps=50, width=2048, bs=2048):
    torch.manual_seed(SEED)
    m = make_mlp(Xtr.shape[1], (width, width)).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    x = torch.tensor(Xtr[:bs], device=dev)        # one batch of real California inputs
    y = torch.tensor(ytr[:bs], device=dev).view(-1, 1)
    for _ in range(3):
        opt.zero_grad(); nn.MSELoss()(m(x), y).backward(); opt.step()
    if dev == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(); nn.MSELoss()(m(x), y).backward(); opt.step()
    if dev == "mps":
        torch.mps.synchronize()
    return (time.time() - t0) / steps * 1000


def timing_all():
    torch.set_num_threads(4)
    out = {"cpu": timing("cpu")}
    torch.set_num_threads(1)
    if torch.backends.mps.is_available():
        out["mps"] = timing("mps")
    out["machine"] = platform.machine() + ", " + platform.system()
    return out


tm = cached("timing_v1", timing_all)
N["timing"] = {"cpu_ms": r2(tm["cpu"], 1), "mps_ms": r2(tm.get("mps", float("nan")), 1),
               "speedup": r2(tm["cpu"] / tm["mps"], 1) if "mps" in tm else None, "machine": tm["machine"],
               "params": int((Xtr.shape[1] + 1) * 2048 + 2049 * 2048 + 2049)}
N["runtime_min"] = r2((time.time() - T0) / 60, 1)
with open(os.path.join(HERE, "L2_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print("wrote L2_numbers.json and", len(N["charts"]), "charts x 2 languages in", N["runtime_min"], "min")
