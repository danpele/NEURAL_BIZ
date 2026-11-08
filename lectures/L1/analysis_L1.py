"""
analysis_L1.py -- Lecture 1 "Deep learning for business: linear models, the perceptron and the MLP".
All computation of the lecture: writes lectures/L1/L1_numbers.json and charts/en|ro/*.pdf.

Data (data/SOURCES.md): German credit (UCI Statlog, 1,000 applicants), HMEQ home-equity loans (5,960 loans),
California housing (20,640 block groups, for the universal-approximation picture).
Models: logistic regression (scikit-learn), perceptron (NumPy, Rosenblatt's rule), multilayer perceptrons (PyTorch),
gradient boosting (XGBoost) as the tabular benchmark. Seeds: 20261108 + s, s = 0..4.
Hardware: tabular nets are tiny, so they are trained on the CPU (PyTorch, 1 thread); about 3 minutes on an Apple M-series laptop.
Run:  python3 lectures/L1/analysis_L1.py   (add --fresh to ignore the cache in lectures/L1/cache/)
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
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from scipy import stats  # noqa: E402

npl.style()
SEED = npl.seed_all()
torch.set_num_threads(1)          # one thread: tiny nets, and avoids an OpenMP clash with XGBoost on macOS
DEV = "cpu"
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
FRESH = "--fresh" in sys.argv
N = {"charts": {}, "seed": SEED}
T0 = time.time()
PAL = npl.PAL


def cached(name, fn):
    p = os.path.join(CACHE, name + ".json")
    if os.path.exists(p) and not FRESH:
        return json.load(open(p))
    r = fn()
    json.dump(r, open(p, "w"))
    return r


def r2(x, d=3):
    return float(round(float(x), d))


# =============================================================================================== data
def prep_hmeq():
    h = npl.hmeq()
    y = h.pop("BAD").values.astype(int)
    return h, y


def encode_hmeq(tr, others):
    """Median imputation + missing indicators + one-hot, all fitted on the training part only."""
    num = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
    med = tr[num].median()
    mu, sd = None, None

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
    out = [((xtr - mu) / sd)]
    for o in others:
        out.append((f(o) - mu) / sd)
    return [o.values.astype(np.float32) for o in out], list(xtr.columns)


def encode_german(tr, others):
    cat = [c for c in tr.columns if tr[c].dtype == object]
    num = [c for c in tr.columns if c not in cat]
    levels = {c: sorted(tr[c].unique()) for c in cat}

    def f(d):
        x = d[num].astype(float).copy()
        for c in cat:
            for lv in levels[c][1:]:                       # first level = reference category
                x[f"{c}_{lv}"] = (d[c] == lv).astype(float)
        return x
    xtr = f(tr)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    return [((xtr - mu) / sd).values.astype(np.float32)] + [((f(o) - mu) / sd).values.astype(np.float32) for o in others], list(xtr.columns)


# =============================================================================================== models
class MLP(nn.Module):
    def __init__(self, d, hidden=(64, 64), act="relu", out=1):
        super().__init__()
        A = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[act]
        layers, k = [], d
        for h in hidden:
            layers += [nn.Linear(k, h), A()]
            k = h
        layers.append(nn.Linear(k, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), act="relu", lr=1e-3, epochs=300, bs=128, patience=25, seed=0,
              loss="bce", wd=0.0):
    """Adam + mini-batches; keeps the weights of the epoch with the lowest validation loss (early stopping, Lecture 3)."""
    torch.manual_seed(SEED + seed)
    rng = np.random.default_rng(SEED + seed)
    m = MLP(Xtr.shape[1], hidden, act)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss() if loss == "bce" else nn.MSELoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    best, best_state, bad, hist = np.inf, None, 0, []
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
            tl = float(lf(m(Xt), yt))
        hist.append((tl, vl))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, hist


def proba(m, X):
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy().ravel()


def logit_fit(Xtr, ytr):
    return LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr)


def xgb_fit(Xtr, ytr, Xva, yva, seed=0):
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", early_stopping_rounds=50, random_state=SEED + seed, n_jobs=4)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


# =============================================================================================== 0. datasets
g = npl.german_credit()
h, yh = prep_hmeq()
ca = npl.california()
fr = npl.fraud()
N["data"] = {
    "german_n": int(len(g)), "german_bad": int(g["default"].sum()), "german_bad_pct": r2(100 * g["default"].mean(), 1),
    "german_vars": int(g.shape[1] - 1),
    "hmeq_n": int(len(h)), "hmeq_bad": int(yh.sum()), "hmeq_bad_pct": r2(100 * yh.mean(), 1), "hmeq_vars": int(h.shape[1]),
    "hmeq_debtinc_missing_pct": r2(100 * h["DEBTINC"].isna().mean(), 1),
    "california_n": int(len(ca)), "fraud_n_full": 284807, "fraud_frauds": int(fr["Class"].sum()),
    "fraud_rate_full_pct": r2(100 * fr["Class"].sum() / fr["weight"].sum(), 3),
}
# default rate when DEBTINC is missing vs present (why missing indicators matter)
N["data"]["hmeq_bad_debtinc_missing_pct"] = r2(100 * yh[h["DEBTINC"].isna().values].mean(), 1)
N["data"]["hmeq_bad_debtinc_present_pct"] = r2(100 * yh[h["DEBTINC"].notna().values].mean(), 1)

# =============================================================================================== 1. default rates (EDA)
chk = g.groupby("checking_status")["default"].agg(["mean", "size"])
dq = h["DELINQ"].copy()
dq_cat = np.where(dq.isna(), "na", np.where(dq >= 3, "3+", dq.fillna(0).astype(int).astype(str)))
dqr = pd.Series(yh).groupby(dq_cat).agg(["mean", "size"])
N["eda"] = {"chk": {k: {"pct": r2(100 * v["mean"], 1), "n": int(v["size"])} for k, v in chk.iterrows()},
            "delinq": {k.replace("+", "plus"): {"pct": r2(100 * v["mean"], 1), "n": int(v["size"])} for k, v in dqr.iterrows()}}
for lang in ("en", "ro"):
    ro = lang == "ro"
    labs = {"A11": ("< 0\nDM", "< 0\nDM"), "A12": ("0-200\nDM", "0-200\nDM"), "A13": ("≥ 200\nDM", "≥ 200\nDM"),
            "A14": ("no\naccount", "fără\ncont")}
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ks = ["A11", "A12", "A13", "A14"]
    axs[0].bar(range(4), [100 * chk.loc[k, "mean"] for k in ks], color=PAL["positive"], width=0.6,
               label="Default rate (%)" if not ro else "Rata de neplată (%)")
    axs[0].set_xticks(range(4))
    axs[0].set_xticklabels([labs[k][ro] for k in ks], fontsize=9)
    axs[0].set_ylabel("Default rate (%)" if not ro else "Rata de neplată (%)")
    axs[0].set_xlabel("German credit: checking account" if not ro else "German credit: contul curent")
    order = ["0", "1", "2", "3+", "na"]
    axs[1].bar(range(5), [100 * dqr.loc[k, "mean"] for k in order], color=PAL["positive"], width=0.6)
    axs[1].set_xticks(range(5))
    axs[1].set_xticklabels(order[:4] + (["missing"] if not ro else ["lipsă"]), fontsize=9)
    axs[1].set_xlabel("HMEQ: delinquent credit lines" if not ro else "HMEQ: linii de credit cu restanțe")
    for ax in axs:
        ax.axhline(100 * (g["default"].mean() if ax is axs[0] else yh.mean()), color=npl.MUTED, lw=1, ls="--",
                   label="Sample average" if not ro else "Media eșantionului")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "default_rates")
npl.chart_meta(N, "default_rates", "german", "1,000 applicants (German credit); 5,960 loans (HMEQ)",
               "1\\,000 de solicitanți (German credit); 5\\,960 de credite (HMEQ)",
               source_en="UCI Statlog German credit; Baesens, Roesch and Scheule (2016), HMEQ",
               source_ro="UCI Statlog German credit; Baesens, Roesch și Scheule (2016), HMEQ")

# =============================================================================================== 2. HMEQ split (60/20/20, stratified)
idx = np.arange(len(h))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=yh, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=yh[i_tr], random_state=SEED)
(Xtr, Xva, Xte), cols = encode_hmeq(h.iloc[i_tr], [h.iloc[i_va], h.iloc[i_te]])
ytr, yva, yte = yh[i_tr], yh[i_va], yh[i_te]
N["hmeq"] = {"n_train": int(len(i_tr)), "n_val": int(len(i_va)), "n_test": int(len(i_te)), "d": int(Xtr.shape[1]),
             "test_bad": int(yte.sum())}

# ---- logistic regression (benchmark) + standardised coefficients with 95% CI (statsmodels)
import statsmodels.api as sm  # noqa: E402
base = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC", "miss_DEBTINC"]
ci = [cols.index(c) for c in base]
lm = sm.Logit(ytr, sm.add_constant(Xtr[:, ci].astype(float))).fit(disp=0)
coef = pd.DataFrame({"b": lm.params[1:], "lo": lm.conf_int()[1:, 0], "hi": lm.conf_int()[1:, 1]}, index=base)
coef = coef.reindex(coef["b"].abs().sort_values().index)
N["logit"] = {"intercept": r2(lm.params[0]), "coef": {c: r2(coef.loc[c, "b"]) for c in base},
              "or": {c: r2(np.exp(coef.loc[c, "b"]), 2) for c in base}, "n_params": len(base) + 1,
              "pseudo_r2": r2(lm.prsquared)}
lr_full = logit_fit(Xtr, ytr)
p_lr = lr_full.predict_proba(Xte)[:, 1]
NM = {"LOAN": ("loan amount", "valoarea creditului"), "MORTDUE": ("mortgage due", "ipoteca datorată"),
      "VALUE": ("property value", "valoarea proprietății"), "YOJ": ("years on job", "ani la locul de muncă"),
      "DEROG": ("derogatory reports", "rapoarte negative"), "DELINQ": ("delinquent lines", "linii cu restanțe"),
      "CLAGE": ("age of oldest line", "vechimea celei mai vechi linii"), "NINQ": ("recent inquiries", "interogări recente"),
      "CLNO": ("number of lines", "numărul de linii"), "DEBTINC": ("debt-to-income", "datorii/venit"),
      "miss_DEBTINC": ("debt-to-income missing", "datorii/venit lipsă")}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    yy = np.arange(len(coef))
    ax.errorbar(coef["b"], yy, xerr=[coef["b"] - coef["lo"], coef["hi"] - coef["b"]], fmt="o", color=PAL["benchmark"],
                ms=4, lw=1.2, label="Coefficient and 95% interval" if not ro else "Coeficient și interval de 95%")
    ax.axvline(0, color=npl.MUTED, lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([NM[c][ro] for c in coef.index], fontsize=8.5)
    ax.set_xlabel("Change in log-odds of default per 1 sd" if not ro else "Variația log-șansei de neplată la 1 abatere standard")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "logit_coef", also_x=[ax] if ro else ())
npl.chart_meta(N, "logit_coef", "hmeq", f"{len(i_tr):,} training loans", f"{len(i_tr):,} de credite de antrenare".replace(",", "\\,"))

# ---- forward pass of the logistic model for one real test applicant (the first test defaulter with all variables present)
j = int(next(k for k in np.where(yte == 1)[0] if h.iloc[i_te[k]][base[:-1]].notna().all()))
xj = Xte[j, ci].astype(float)
zj = float(lm.params[0] + lm.params[1:] @ xj)
contrib = lm.params[1:] * xj
top = np.argsort(-np.abs(contrib))[:3]
rawj = h.iloc[i_te[j]]
N["one"] = {"z": r2(zj, 2), "p": r2(1 / (1 + np.exp(-zj)), 3), "p_pct": r2(100 / (1 + np.exp(-zj)), 1),
            "loan": int(rawj["LOAN"]), "debtinc": r2(rawj["DEBTINC"], 1), "delinq": int(rawj["DELINQ"]),
            "derog": int(rawj["DEROG"]), "clage": r2(rawj["CLAGE"], 1),
            "top": [{"var": base[k], "x": r2(xj[k], 2), "b": r2(lm.params[1 + k], 3), "c": r2(contrib[k], 2)} for k in top],
            "rest": r2(zj - lm.params[0] - contrib[top].sum(), 2)}

# =============================================================================================== 3. perceptron on HMEQ (not separable)
def perceptron_run(X, y, epochs=60, seed=0):
    rng = np.random.default_rng(SEED + seed)
    s = 2 * y - 1
    w, b = np.zeros(X.shape[1]), 0.0
    err, upd = [], 0
    best = (1.0, w.copy(), b)
    for ep in range(epochs):
        for i in rng.permutation(len(X)):
            if s[i] * (X[i] @ w + b) <= 0:
                w += s[i] * X[i]
                b += s[i]
                upd += 1
        e = float(np.mean(np.sign(X @ w + b + 1e-12) != s))
        err.append(e)
        if e < best[0]:
            best = (e, w.copy(), b)
    return err, best, upd


perr, pbest, pupd = perceptron_run(Xtr.astype(float), ytr)
lr_err = float(np.mean((lr_full.predict_proba(Xtr)[:, 1] > 0.5) != ytr))
N["perceptron"] = {"epochs": len(perr), "err_last_pct": r2(100 * perr[-1], 1), "err_min_pct": r2(100 * min(perr), 1),
                   "err_max_pct": r2(100 * max(perr), 1), "updates": int(pupd), "logit_err_pct": r2(100 * lr_err, 1),
                   "majority_err_pct": r2(100 * ytr.mean(), 1)}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(np.arange(1, len(perr) + 1), 100 * np.array(perr), color=PAL["model"], lw=1.3,
            label="Perceptron after each epoch" if not ro else "Perceptronul după fiecare epocă")
    ax.axhline(100 * lr_err, color=PAL["benchmark"], lw=1.3, ls="-", label="Logistic regression" if not ro else "Regresia logistică")
    ax.axhline(100 * ytr.mean(), color=npl.MUTED, lw=1, ls="--",
               label="Always predict 'no default'" if not ro else "Mereu „fără neplată”")
    ax.set_xlabel("Epoch (one pass over the training loans)" if not ro else "Epoca (o trecere prin creditele de antrenare)")
    ax.set_ylabel("Training error (%)" if not ro else "Eroarea de antrenare (%)")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "perceptron_errors")
npl.chart_meta(N, "perceptron_errors", "hmeq", f"{len(i_tr):,} training loans", f"{len(i_tr):,} de credite de antrenare".replace(",", "\\,"))

# =============================================================================================== 4. activations (mechanism)
z = np.linspace(-4, 4, 401)
sig = 1 / (1 + np.exp(-z))
gelu = 0.5 * z * (1 + np.vectorize(lambda u: __import__("math").erf(u / np.sqrt(2)))(z))
acts = {"sigmoid": (sig, sig * (1 - sig)), "tanh": (np.tanh(z), 1 - np.tanh(z) ** 2),
        "ReLU": (np.maximum(z, 0), (z > 0).astype(float)),
        "GELU": (gelu, np.gradient(gelu, z))}
N["act"] = {"sigmoid_max_grad": 0.25, "tanh_max_grad": 1.0}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for (nm, (f, d)), c in zip(acts.items(), [PAL["data"], PAL["benchmark"], PAL["model"], PAL["c7"]]):
        axs[0].plot(z, f, color=c, label=nm)
        axs[1].plot(z, d, color=c)
    axs[0].set_ylim(-1.2, 2.2)
    axs[0].set_ylabel("$a(z)$")
    axs[1].set_ylabel("$a'(z)$")
    for ax in axs:
        ax.set_xlabel("$z$")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "activations", also_x=list(fig.axes) if ro else ())
npl.chart_meta(N, "activations", "formula", "functions plotted on [-4, 4]", "funcții reprezentate pe [-4, 4]",
               source_en="analytic functions (no data)", source_ro="funcții analitice (fără date)")

# =============================================================================================== 5. XOR (mechanism)
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
yx = np.array([0, 1, 1, 0], dtype=np.float32)
torch.manual_seed(SEED)
best_x = None
for s in range(20):
    torch.manual_seed(SEED + s)
    mx = MLP(2, (2,), "tanh")
    o = torch.optim.Adam(mx.parameters(), lr=0.05)
    for _ in range(2000):
        o.zero_grad()
        nn.BCEWithLogitsLoss()(mx(torch.tensor(Xx)), torch.tensor(yx).view(-1, 1)).backward()
        o.step()
    pr = proba(mx, Xx)
    if np.all((pr > 0.5) == yx.astype(bool)):
        best_x = mx
        break
lx = LogisticRegression(C=1e6).fit(Xx, yx)
gx = np.linspace(-0.4, 1.4, 200)
GX, GY = np.meshgrid(gx, gx)
grid = np.c_[GX.ravel(), GY.ravel()].astype(np.float32)
N["xor"] = {"logit_acc_pct": int(round(100 * np.mean((lx.predict_proba(Xx)[:, 1] > 0.5) == yx))),
            "mlp_acc_pct": int(round(100 * np.mean((proba(best_x, Xx) > 0.5) == yx)))}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, P, nm in ((axs[0], lx.predict_proba(grid)[:, 1], "lin"), (axs[1], proba(best_x, grid), "mlp")):
        ax.contourf(GX, GY, P.reshape(GX.shape), levels=[0, 0.5, 1], colors=[PAL["negative"], PAL["positive"]], alpha=0.15)
        ax.contour(GX, GY, P.reshape(GX.shape), levels=[0.5], colors=[npl.INK], linewidths=1)
        ax.scatter(Xx[yx == 0, 0], Xx[yx == 0, 1], s=60, color=PAL["negative"], marker="o", label="XOR = 0", zorder=3)
        ax.scatter(Xx[yx == 1, 0], Xx[yx == 1, 1], s=70, color=PAL["positive"], marker="X", label="XOR = 1", zorder=3)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel(("$x_1$   (one linear boundary)" if not ro else "$x_1$   (o singură frontieră liniară)") if nm == "lin"
                      else ("$x_1$   (MLP, 2 hidden units)" if not ro else "$x_1$   (MLP, 2 neuroni ascunși)"))
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "xor")
npl.chart_meta(N, "xor", "formula", "the four XOR inputs", "cele patru intrări XOR",
               source_en="logical XOR table (no data)", source_ro="tabelul funcției logice XOR (fără date)")

# =============================================================================================== 6. HMEQ: two-feature boundaries
two = ["DEBTINC", "CLAGE"]
m2 = h[two].notna().all(axis=1).values
i_tr2, i_te2 = i_tr[m2[i_tr]], i_te[m2[i_te]]
i_va2 = i_va[m2[i_va]]
mu2, sd2 = h.iloc[i_tr2][two].mean(), h.iloc[i_tr2][two].std()
Z = lambda ii: ((h.iloc[ii][two] - mu2) / sd2).values.astype(np.float32)  # noqa: E731
lr2 = logit_fit(Z(i_tr2), yh[i_tr2])
ml2, _ = train_mlp(Z(i_tr2), yh[i_tr2], Z(i_va2), yh[i_va2], hidden=(64, 64), seed=0, epochs=400, patience=40)
auc2 = {"logit": r2(roc_auc_score(yh[i_te2], lr2.predict_proba(Z(i_te2))[:, 1])),
        "mlp": r2(roc_auc_score(yh[i_te2], proba(ml2, Z(i_te2))))}
N["two"] = {"auc_logit": auc2["logit"], "auc_mlp": auc2["mlp"], "n_test": int(len(i_te2)), "n_train": int(len(i_tr2))}
dg = np.linspace(0, 70, 220)
cg = np.linspace(0, 600, 220)
DG, CG = np.meshgrid(dg, cg)
gz = (np.c_[DG.ravel(), CG.ravel()] - mu2.values) / sd2.values
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    te = h.iloc[i_te2]
    for ax, P in ((axs[0], lr2.predict_proba(gz)[:, 1]), (axs[1], proba(ml2, gz.astype(np.float32)))):
        cs = ax.contourf(DG, CG, P.reshape(DG.shape), levels=[0, 0.1, 0.2, 0.3, 0.5, 1], cmap="Oranges", alpha=0.55)
        ax.contour(DG, CG, P.reshape(DG.shape), levels=[0.2], colors=[npl.INK], linewidths=1)
        ok = yh[i_te2] == 0
        ax.scatter(te["DEBTINC"][ok], te["CLAGE"][ok], s=3, color=PAL["negative"], alpha=0.35, lw=0,
                   label="Repaid (test)" if not ro else "Rambursat (test)")
        ax.scatter(te["DEBTINC"][~ok], te["CLAGE"][~ok], s=5, color=PAL["positive"], alpha=0.8, lw=0,
                   label="Defaulted (test)" if not ro else "Neplată (test)")
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 600)
        ax.set_xlabel("Debt-to-income (%)" if not ro else "Datorii/venit (%)")
    axs[0].set_ylabel("Oldest line (months)" if not ro else "Cea mai veche linie (luni)")
    for ax, nm in zip(axs, (("logistic", "logistică"), ("MLP 64-64", "MLP 64-64"))):
        ax.text(0.97, 0.95, nm[ro], transform=ax.transAxes, ha="right", va="top", fontsize=9)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "boundary_hmeq")
npl.chart_meta(N, "boundary_hmeq", "hmeq", f"{len(i_tr2):,} training and {len(i_te2):,} test loans with both variables",
               f"{len(i_tr2):,} de credite de antrenare și {len(i_te2):,} de testare cu ambele variabile".replace(",", "\\,"))

# =============================================================================================== 7. HMEQ: width sweep, ROC, calibration
WIDTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEEDS = [0, 1, 2, 3, 4]


def width_sweep():
    out = {}
    for w in WIDTHS:
        a = []
        for s in SEEDS:
            m, _ = train_mlp(Xtr, ytr, Xva, yva, hidden=(w,), seed=s)
            a.append(roc_auc_score(yte, proba(m, Xte)))
        out[str(w)] = a
    return out


ws = cached("width_sweep_v1", width_sweep)
auc_lr = roc_auc_score(yte, p_lr)
xg = xgb_fit(Xtr, ytr, Xva, yva)
p_xg = xg.predict_proba(Xte)[:, 1]
auc_xg = roc_auc_score(yte, p_xg)


def deep_runs():
    return [float(roc_auc_score(yte, proba(train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), seed=s)[0], Xte))) for s in SEEDS]


deep = cached("deep_runs_v1", deep_runs)
N["width"] = {str(w): {"mean": r2(np.mean(ws[str(w)])), "sd": r2(np.std(ws[str(w)], ddof=1))} for w in WIDTHS}
best_w = max(WIDTHS, key=lambda w: np.mean(ws[str(w)]))
N["hmeq"].update({"auc_logit": r2(auc_lr), "auc_xgb": r2(auc_xg), "auc_mlp_mean": r2(np.mean(deep)),
                  "auc_mlp_sd": r2(np.std(deep, ddof=1)), "auc_mlp_min": r2(min(deep)), "auc_mlp_max": r2(max(deep)),
                  "best_width": best_w, "auc_w1": r2(np.mean(ws["1"])), "auc_best_width": r2(np.mean(ws[str(best_w)])),
                  "n_params_6464": int((Xtr.shape[1] + 1) * 64 + 65 * 64 + 65), "n_params_logit": int(Xtr.shape[1] + 1),
                  "n_params_32": int((Xtr.shape[1] + 1) * 32 + 33)})
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    mu = np.array([np.mean(ws[str(w)]) for w in WIDTHS])
    lo = np.array([np.min(ws[str(w)]) for w in WIDTHS])
    hi = np.array([np.max(ws[str(w)]) for w in WIDTHS])
    ax.fill_between(WIDTHS, lo, hi, color=PAL["model"], alpha=0.18, lw=0)
    ax.plot(WIDTHS, mu, "o-", color=PAL["model"], ms=4, label="MLP, one hidden layer (mean of 5 seeds, range)" if not ro
            else "MLP cu un strat ascuns (media a 5 inițializări, interval)")
    ax.axhline(auc_lr, color=PAL["benchmark"], lw=1.3, label="Logistic regression" if not ro else "Regresia logistică")
    ax.axhline(auc_xg, color=PAL["alt"], lw=1.3, ls="--", label="Gradient boosting (XGBoost)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.set_xlabel("Hidden units $H$" if not ro else "Neuroni ascunși $H$")
    ax.set_ylabel("Test AUC" if not ro else "AUC pe test")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "width_auc")
npl.chart_meta(N, "width_auc", "hmeq", f"{len(i_tr):,} training, {len(i_va):,} validation, {len(i_te):,} test loans",
               f"{len(i_tr):,} de credite de antrenare, {len(i_va):,} de validare, {len(i_te):,} de testare".replace(",", "\\,"))

m_best, _ = train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), seed=0)
p_ml = proba(m_best, Xte)
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for p, c, nm in ((p_lr, PAL["benchmark"], ("Logistic regression", "Regresia logistică")),
                     (p_ml, PAL["model"], ("MLP 64-64", "MLP 64-64")), (p_xg, PAL["alt"], ("XGBoost", "XGBoost"))):
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(100 * fpr, 100 * tpr, color=c, label=f"{nm[ro]} (AUC {roc_auc_score(yte, p):.3f})".replace(".", "," if ro else "."))
    ax.plot([0, 100], [0, 100], color=npl.MUTED, lw=0.8, ls=":")
    ax.set_xlabel("False positive rate: good loans rejected (%)" if not ro else "Rata alarmelor false: credite bune respinse (%)")
    ax.set_ylabel("Defaults caught (%)" if not ro else "Neplăți detectate (%)")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "roc_hmeq", also_x=[ax] if ro else ())
npl.chart_meta(N, "roc_hmeq", "hmeq", f"{len(i_te):,} test loans ({int(yte.sum())} defaults)",
               f"{len(i_te):,} de credite de testare ({int(yte.sum())} de neplăți)".replace(",", "\\,"))
fpr_l, tpr_l, _ = roc_curve(yte, p_lr)
fpr_m, tpr_m, _ = roc_curve(yte, p_ml)
fpr_x, tpr_x, _ = roc_curve(yte, p_xg)
N["hmeq"].update({"auc_mlp_seed0": r2(roc_auc_score(yte, p_ml)),
                  "tpr10_logit": r2(100 * np.interp(0.10, fpr_l, tpr_l), 1), "tpr10_mlp": r2(100 * np.interp(0.10, fpr_m, tpr_m), 1),
                  "tpr10_xgb": r2(100 * np.interp(0.10, fpr_x, tpr_x), 1)})

# DeLong-free paired bootstrap of the AUC difference MLP - logistic (test set, 2,000 resamples)
rng = np.random.default_rng(SEED)
diffs = []
for _ in range(2000):
    b = rng.integers(0, len(yte), len(yte))
    if yte[b].min() == yte[b].max():
        continue
    diffs.append(roc_auc_score(yte[b], p_ml[b]) - roc_auc_score(yte[b], p_lr[b]))
N["hmeq"]["dauc_mlp_lr"] = r2(roc_auc_score(yte, p_ml) - auc_lr)
N["hmeq"]["dauc_lo"] = r2(np.percentile(diffs, 2.5))
N["hmeq"]["dauc_hi"] = r2(np.percentile(diffs, 97.5))

# calibration (10 equal-count bins)
def calib(p, y, nb=10):
    q = np.quantile(p, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nb - 1)
    return np.array([p[b == k].mean() for k in range(nb)]), np.array([y[b == k].mean() for k in range(nb)])


N["calib"] = {"brier_logit": r2(brier_score_loss(yte, p_lr), 4), "brier_mlp": r2(brier_score_loss(yte, p_ml), 4),
              "brier_xgb": r2(brier_score_loss(yte, p_xg), 4), "brier_base": r2(brier_score_loss(yte, np.full(len(yte), ytr.mean())), 4)}
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for p, c, nm in ((p_lr, PAL["benchmark"], ("Logistic regression", "Regresia logistică")),
                     (p_ml, PAL["model"], ("MLP 64-64", "MLP 64-64")), (p_xg, PAL["alt"], ("XGBoost", "XGBoost"))):
        a, b = calib(p, yte)
        ax.plot(100 * a, 100 * b, "o-", color=c, ms=3.5, label=nm[ro])
    ax.plot([0, 100], [0, 100], color=npl.MUTED, lw=0.8, ls=":", label="Perfect calibration" if not ro else "Calibrare perfectă")
    ax.set_xlabel("Predicted probability of default (%), decile bins" if not ro else "Probabilitatea prezisă de neplată (%), pe decile")
    ax.set_ylabel("Observed default rate (%)" if not ro else "Rata observată (%)")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "calibration", also_x=[ax] if ro else ())
npl.chart_meta(N, "calibration", "hmeq", f"{len(i_te):,} test loans", f"{len(i_te):,} de credite de testare".replace(",", "\\,"))

# =============================================================================================== 8. German credit: repeated CV + cost
yg = g["default"].values.astype(int)
Xg_raw = g.drop(columns="default")
COST_FN, COST_FP = 5.0, 1.0          # UCI cost matrix: a missed bad risk costs 5, a rejected good applicant 1
THR_BAYES = COST_FP / (COST_FP + COST_FN)


def german_cv():
    res = {"logit": [], "mlp": [], "xgb": []}
    oof = {"logit": np.zeros(len(yg)), "mlp": np.zeros(len(yg)), "xgb": np.zeros(len(yg))}
    for rep in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for f, (a, b) in enumerate(skf.split(Xg_raw, yg)):
            a_tr, a_va = train_test_split(a, test_size=0.2, stratify=yg[a], random_state=SEED + rep)
            (xa, xv, xb), _ = encode_german(Xg_raw.iloc[a_tr], [Xg_raw.iloc[a_va], Xg_raw.iloc[b]])
            pl = LogisticRegression(C=1.0, max_iter=5000).fit(xa, yg[a_tr]).predict_proba(xb)[:, 1]
            mm, _ = train_mlp(xa, yg[a_tr], xv, yg[a_va], hidden=(64, 64), seed=rep * 10 + f, lr=1e-3, bs=64)
            pm = proba(mm, xb)
            px = xgb_fit(xa, yg[a_tr], xv, yg[a_va], seed=rep).predict_proba(xb)[:, 1]
            for k, p in (("logit", pl), ("mlp", pm), ("xgb", px)):
                res[k].append(roc_auc_score(yg[b], p))
                if rep == 0:
                    oof[k][b] = p
    return {"auc": res, "oof": {k: v.tolist() for k, v in oof.items()}}


gcv = cached("german_cv_v1", german_cv)
A = {k: np.array(v) for k, v in gcv["auc"].items()}
OOF = {k: np.array(v) for k, v in gcv["oof"].items()}


def nb_ttest(d, n_train_frac=0.8):
    """Nadeau-Bengio corrected resampled t-test for repeated k-fold CV (k=5: n_test/n_train = 1/4)."""
    J = len(d)
    var = np.var(d, ddof=1)
    t = np.mean(d) / np.sqrt((1 / J + 0.25) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


t_ml, p_ml_ = nb_ttest(A["mlp"] - A["logit"])
t_xg, p_xg_ = nb_ttest(A["xgb"] - A["logit"])


def cost(p, y, thr):
    pred = p >= thr
    return float((COST_FN * np.sum(~pred & (y == 1)) + COST_FP * np.sum(pred & (y == 0))) / len(y))


thr = np.linspace(0.02, 0.9, 89)
cst = {k: np.array([cost(OOF[k], yg, t) for t in thr]) for k in OOF}
N["german"] = {"auc_logit": r2(A["logit"].mean()), "auc_mlp": r2(A["mlp"].mean()), "auc_xgb": r2(A["xgb"].mean()),
               "sd_logit": r2(A["logit"].std(ddof=1)), "sd_mlp": r2(A["mlp"].std(ddof=1)), "sd_xgb": r2(A["xgb"].std(ddof=1)),
               "n_folds": int(len(A["logit"])), "d_mlp": r2((A["mlp"] - A["logit"]).mean()), "t_mlp": r2(t_ml, 2), "p_mlp": r2(p_ml_),
               "d_xgb": r2((A["xgb"] - A["logit"]).mean()), "t_xgb": r2(t_xg, 2), "p_xgb": r2(p_xg_),
               "thr_bayes": r2(THR_BAYES), "cost_fn": COST_FN, "cost_fp": COST_FP,
               "cost_all_accept": r2(COST_FN * yg.mean()), "cost_all_reject": r2(COST_FP * (1 - yg.mean())),
               "d_onehot": int(encode_german(Xg_raw, [])[0][0].shape[1])}
for k in ("logit", "mlp", "xgb"):
    N["german"][f"cost_{k}_bayes"] = r2(cost(OOF[k], yg, THR_BAYES))
    N["german"][f"cost_{k}_05"] = r2(cost(OOF[k], yg, 0.5))
    N["german"][f"thr_{k}_best"] = r2(thr[np.argmin(cst[k])], 2)
    N["german"][f"cost_{k}_best"] = r2(cst[k].min())
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for i, (k, c, nm) in enumerate((("logit", PAL["benchmark"], ("Logistic regression", "Regresia logistică")),
                                     ("mlp", PAL["model"], ("MLP 64-64", "MLP 64-64")), ("xgb", PAL["alt"], ("XGBoost", "XGBoost")))):
        jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(A[k]))
        ax.scatter(A[k], i + jit, s=10, color=c, alpha=0.7, label=nm[ro])
        ax.plot([A[k].mean()] * 2, [i - 0.3, i + 0.3], color=npl.INK, lw=1.8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Logistic" if not ro else "Logistică", "MLP", "XGBoost"])
    ax.set_xlabel("AUC on the held-out fold (5 x 5-fold cross-validation)" if not ro else
                  "AUC pe pliul de test (validare încrucișată 5 x 5)")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "german_cv", also_x=[ax] if ro else ())
npl.chart_meta(N, "german_cv", "german", "1,000 applicants, 25 held-out folds", "1\\,000 de solicitanți, 25 de pliuri de test")
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for k, c, nm in (("logit", PAL["benchmark"], ("Logistic regression", "Regresia logistică")), ("mlp", PAL["model"], ("MLP 64-64", "MLP 64-64")),
                     ("xgb", PAL["alt"], ("XGBoost", "XGBoost"))):
        ax.plot(thr, cst[k], color=c, label=nm[ro])
    ax.axvline(THR_BAYES, color=npl.INK, lw=1, ls="--", label="Cost-optimal cut-off 1/6" if not ro else "Pragul optim pentru cost 1/6")
    ax.axhline(COST_FN * yg.mean(), color=npl.MUTED, lw=1, ls=":", label="Accept everyone" if not ro else "Acceptăm pe toată lumea")
    ax.set_xlabel("Cut-off: reject if predicted PD ≥ threshold" if not ro else "Prag: respingem dacă PD prezisă ≥ prag")
    ax.set_ylabel("Cost per applicant" if not ro else "Cost pe solicitant")
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "cost_threshold", also_x=[ax] if ro else ())
npl.chart_meta(N, "cost_threshold", "german", "1,000 applicants, out-of-fold predictions", "1\\,000 de solicitanți, predicții în afara pliului")

# =============================================================================================== 9. universal approximation on California housing
x1 = ca["Latitude"].values.astype(np.float32)
y1 = ca["MedHouseVal_100k_usd"].values.astype(np.float32)
keep = y1 < 5.0                                   # drop the capped block groups (source cap at 5.00001)
x1, y1 = x1[keep], y1[keep]
i_a, i_b = train_test_split(np.arange(len(x1)), test_size=0.25, random_state=SEED)
mx_, sx_ = x1[i_a].mean(), x1[i_a].std()
fits, mse = {}, {}
xs = np.linspace(32.5, 42, 300).astype(np.float32)
for w in (1, 3, 50):
    torch.manual_seed(SEED + w)
    mU = MLP(1, (w,), "sigmoid")
    opt = torch.optim.Adam(mU.parameters(), lr=0.02)
    Xa = torch.tensor((x1[i_a, None] - mx_) / sx_)
    Ya = torch.tensor(y1[i_a, None])
    for it in range(4000):
        opt.zero_grad()
        nn.MSELoss()(mU(Xa), Ya).backward()
        opt.step()
    with torch.no_grad():
        fits[w] = mU(torch.tensor((xs[:, None] - mx_) / sx_)).numpy().ravel()
        mse[w] = float(nn.MSELoss()(mU(torch.tensor((x1[i_b, None] - mx_) / sx_)), torch.tensor(y1[i_b, None])))
b_lin = np.polyfit(x1[i_a], y1[i_a], 1)
mse_lin = float(np.mean((np.polyval(b_lin, x1[i_b]) - y1[i_b]) ** 2))
N["uat"] = {"n": int(keep.sum()), "n_capped": int((~keep).sum()), "mse_lin": r2(mse_lin), **{f"mse_w{w}": r2(mse[w]) for w in mse}}
bins = np.quantile(x1[i_a], np.linspace(0, 1, 41))
bc = np.digitize(x1[i_a], bins[1:-1])
bx = np.array([x1[i_a][bc == k].mean() for k in range(40)])
by = np.array([y1[i_a][bc == k].mean() for k in range(40)])
for lang in ("en", "ro"):
    ro = lang == "ro"
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.scatter(bx, by, s=12, color=PAL["data"], zorder=3, label="Data: mean value in 40 latitude bins" if not ro else
               "Date: valoarea medie pe 40 de clase de latitudine")
    ax.plot(xs, np.polyval(b_lin, xs), color=PAL["benchmark"], lw=1.2, label="Linear" if not ro else "Liniar")
    for w, c in ((1, PAL["c5"]), (3, PAL["alt"]), (50, PAL["model"])):
        ax.plot(xs, fits[w], color=c, lw=1.4, label=f"MLP, H = {w}")
    ax.set_xlabel("Latitude of the block group (degrees north)" if not ro else "Latitudinea grupului de locuințe (grade nord)")
    ax.set_ylabel("Value (100,000 USD)" if not ro else "Valoarea (100\\,000 USD)")
    ax.set_xlim(32.5, 42)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "uat_california", also_x=[ax] if ro else ())
npl.chart_meta(N, "uat_california", "california", f"{int(keep.sum()):,} block groups below the source cap",
               f"{int(keep.sum()):,} de grupuri de locuințe sub plafonul sursei".replace(",", "\\,"))

N["runtime_min"] = r2((time.time() - T0) / 60, 1)
with open(os.path.join(HERE, "L1_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print("wrote L1_numbers.json and", len(N["charts"]), "charts x 2 languages in", N["runtime_min"], "min")
