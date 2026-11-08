"""
analysis_L11.py -- Lecture 11 "Explainable AI (XAI): SHAP, LIME, gradient methods and the stability of explanations".

Real data only:
  A. Credit (HMEQ home-equity loans, Baesens et al., 2016): multilayer perceptron (MLP) against logistic regression and
     gradient boosting (XGBoost); Kernel SHAP, TreeSHAP and linear SHAP; LIME and its sensitivity to its own settings;
     a counterfactual explanation for a rejected applicant; stability of SHAP rankings across 10 training seeds.
  B. Fraud (ULB credit-card transactions, two days of September 2013): an MLP trained on day 1 and tested on day 2;
     saliency, gradient x input, SmoothGrad, integrated gradients (completeness), Kernel SHAP, LIME, permutation
     importance; agreement between methods, deletion (faithfulness) curves, local robustness, and the shift of the
     explanations from day 1 to day 2.
  C. California housing (1990 census): partial dependence and individual conditional expectation for an MLP.
Writes lectures/L11/L11_numbers.json and charts/en|ro/*.pdf; trained results cached in lectures/L11/cache/ (git-ignored).
Run:  python3 lectures/L11/analysis_L11.py   (about 5 minutes on an Apple M5 Max CPU the first time)
"""
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

# OpenMP on macOS: PyTorch must be loaded before scikit-learn / SHAP / LIME (the reverse order crashes in training), and
# XGBoost cannot share a process with PyTorch: the gradient-boosting part runs in a child process (`--xgb`) without torch.
XGB_MODE = len(sys.argv) > 1 and sys.argv[1] == "--xgb"
if not XGB_MODE:
    import torch  # noqa: E402
    import torch.nn as nn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402

warnings.filterwarnings("ignore")
npl.style()
SEED = 20261108
np.random.seed(SEED)
if not XGB_MODE:
    SEED = npl.seed_all()
    torch.set_num_threads(8)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
N = {"charts": {}, "seed": SEED}
T0 = time.time()
L = lambda lang, en, ro: en if lang == "en" else ro  # noqa: E731


def r(x, d=2):
    return float(round(float(x), d))


def cached(name, fn):
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.isfile(p):
        return pickle.load(open(p, "rb"))
    t = time.time()
    res = fn()
    res["_seconds"] = round(time.time() - t, 1)
    pickle.dump(res, open(p, "wb"))
    return res


# ================================================================ A. credit: HMEQ
h = npl.hmeq()
NUM = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
JOBS = ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]
y_all = h["BAD"].values
idx = np.arange(len(h))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=y_all[i_tr], random_state=SEED)
med = h.iloc[i_tr][NUM].median()


def design(df):
    X = pd.DataFrame(index=df.index)
    for c in NUM:
        X[c] = df[c].fillna(med[c])
    X["DEBTINC_missing"] = df["DEBTINC"].isna().astype(float)
    X["REASON_HomeImp"] = (df["REASON"] == "HomeImp").astype(float)
    for j in JOBS:
        X["JOB_" + j] = (df["JOB"] == j).astype(float)
    return X


XR = design(h)                                   # raw units (imputed with the training medians)
FEATS = list(XR.columns)
mu, sdv = XR.iloc[i_tr].mean(), XR.iloc[i_tr].std()
XS = ((XR - mu) / sdv).values.astype(np.float32)   # standardised with training moments
Xtr, Xva, Xte = XS[i_tr], XS[i_va], XS[i_te]
ytr, yva, yte = y_all[i_tr], y_all[i_va], y_all[i_te]
N["hmeq"] = {"n": int(len(h)), "bad_pct": r(100 * y_all.mean(), 1), "n_train": int(len(i_tr)), "n_valid": int(len(i_va)),
             "n_test": int(len(i_te)), "n_feat": len(FEATS), "debtinc_missing_pct": r(100 * h["DEBTINC"].isna().mean(), 1),
             "bad_if_debtinc_missing": r(100 * h.loc[h.DEBTINC.isna(), "BAD"].mean(), 1),
             "bad_if_debtinc_present": r(100 * h.loc[h.DEBTINC.notna(), "BAD"].mean(), 1)}
rng = np.random.default_rng(SEED)
BG = Xtr[rng.choice(len(Xtr), 100, replace=False)]           # background sample: 100 training applicants
i_sub = rng.choice(len(Xte), 400, replace=False)             # 400 test applicants for the global plots
Xsub = Xte[i_sub]
SEEDS10 = [SEED + i for i in range(10)]
if XGB_MODE:
    import xgboost as xgb
    out = {"pred": [], "shap_seed": []}
    for sd in SEEDS10:
        m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              random_state=sd, eval_metric="logloss", n_jobs=4)
        m.fit(Xtr, ytr)
        out["pred"].append(m.predict_proba(Xte)[:, 1])
        tex = shap.TreeExplainer(m, data=BG, model_output="raw", feature_perturbation="interventional")
        if sd == SEEDS10[0]:
            out["shap_main"] = tex.shap_values(Xsub)
            out["base"] = float(np.atleast_1d(tex.expected_value)[0])
        out["shap_seed"].append(tex.shap_values(Xsub[:200]))
    pickle.dump(out, open(os.path.join(CACHE, "xgb_v1.pkl"), "wb"))
    print("XGBoost part written")
    sys.exit(0)
if not os.path.isfile(os.path.join(CACHE, "xgb_v1.pkl")):
    import subprocess
    subprocess.run([sys.executable, os.path.abspath(__file__), "--xgb"], check=True)
XG = pickle.load(open(os.path.join(CACHE, "xgb_v1.pkl"), "rb"))
NICE = {"en": {"DEBTINC_missing": "DEBTINC missing", "REASON_HomeImp": "REASON = HomeImp"},
        "ro": {"DEBTINC_missing": "DEBTINC lipsă", "REASON_HomeImp": "REASON = HomeImp"}}


def nice(f, lang):
    return NICE[lang].get(f, f.replace("JOB_", "JOB = "))


class MLP(nn.Module):
    def __init__(self, d, h1=64, h2=32, p=0.1, out=1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(p), nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p),
                               nn.Linear(h2, out))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def train_mlp(seed, Xa, ya, Xv, yv, w_pos=1.0, epochs=300, lr=1e-3, wd=1e-4, batch=256, patience=25, task="clf",
              sw_valid=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = MLP(Xa.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xa_t, ya_t, Xv_t = torch.from_numpy(Xa), torch.from_numpy(ya.astype(np.float32)), torch.from_numpy(Xv)
    best, state, bad, ep_best = -np.inf, None, 0, 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(Xa))
        for j in range(0, len(Xa), batch):
            b = perm[j:j + batch]
            out = net(Xa_t[b])
            if task == "clf":
                w = torch.where(ya_t[b] > 0, torch.tensor(w_pos), torch.tensor(1.0))
                loss = nn.functional.binary_cross_entropy_with_logits(out, ya_t[b], weight=w)
            else:
                loss = ((out - ya_t[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv_t).numpy()
        score = (average_precision_score(yv, pv, sample_weight=sw_valid) if w_pos != 1.0 else roc_auc_score(yv, pv)) \
            if task == "clf" else -np.mean((pv - yv) ** 2)
        if score > best:
            best, state, bad, ep_best = score, {k: v.clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad > patience:
                break
    net.load_state_dict(state)
    net.eval()
    return net, ep_best


def logit_fn(net):
    def f(X):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy().astype(np.float64)
    return f


sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def credit_models():
    out = {"mlp": []}
    for sd in SEEDS10:
        net, ep = train_mlp(sd, Xtr, ytr, Xva, yva)
        out["mlp"].append({"seed": sd, "state": {k: v.clone() for k, v in net.state_dict().items()}, "epoch": ep})
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
    out["lr"] = lr
    return out


CM = cached("credit_models_v1", credit_models)
nets = []
for x in CM["mlp"]:
    n_ = MLP(Xtr.shape[1])
    n_.load_state_dict(x["state"])
    n_.eval()
    nets.append(n_)
net0, lrm = nets[0], CM["lr"]
f0 = logit_fn(net0)


def metr(y, p):
    return roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)


m_mlp = np.array([metr(yte, sig(logit_fn(n_)(Xte))) for n_ in nets])
m_xgb = np.array([metr(yte, p_) for p_ in XG["pred"]])
m_lr = metr(yte, lrm.predict_proba(Xte)[:, 1])
N["perf"] = {"mlp_auc": r(m_mlp[:, 0].mean(), 3), "mlp_auc_sd": r(m_mlp[:, 0].std(ddof=1), 3),
             "mlp_pr": r(m_mlp[:, 1].mean(), 3), "mlp_brier": r(m_mlp[:, 2].mean(), 3),
             "xgb_auc": r(m_xgb[:, 0].mean(), 3), "xgb_auc_sd": r(m_xgb[:, 0].std(ddof=1), 3),
             "xgb_pr": r(m_xgb[:, 1].mean(), 3), "xgb_brier": r(m_xgb[:, 2].mean(), 3),
             "lr_auc": r(m_lr[0], 3), "lr_pr": r(m_lr[1], 3), "lr_brier": r(m_lr[2], 3),
             "mlp0_auc": r(m_mlp[0, 0], 3), "mlp_epoch_mean": r(np.mean([x["epoch"] for x in CM["mlp"]]), 0),
             "base_pr": r(yte.mean(), 3), "seconds": r(CM["_seconds"], 0)}

# ---------------------------------------------------------------- SHAP: Kernel SHAP (MLP), TreeSHAP (XGBoost), linear SHAP
def kernel_shap(f, X, nsamples=600, seed=SEED):
    np.random.seed(seed)
    ex = shap.KernelExplainer(f, BG)
    return np.array(ex.shap_values(X, nsamples=nsamples, silent=True)), float(np.mean(f(BG)))


def shap_main():
    sv, base = kernel_shap(f0, Xsub)
    return {"mlp": sv, "mlp_base": base, "xgb": XG["shap_main"], "xgb_base": XG["base"]}


SH = cached("shap_main_v1", shap_main)
sv_mlp, sv_xgb = SH["mlp"], SH["xgb"]
sv_lr = (Xsub - BG.mean(0)) * lrm.coef_[0]                   # linear SHAP: beta_j (x_j - E x_j), exact for a linear logit
add_err = np.abs(sv_mlp.sum(1) + SH["mlp_base"] - f0(Xsub)).max()
imp = {k: np.abs(v).mean(0) for k, v in (("mlp", sv_mlp), ("xgb", sv_xgb), ("lr", sv_lr))}
order = np.argsort(-imp["mlp"])
top10 = order[:10]
N["shap"] = {"add_err": float(f"{add_err:.1e}"), "base_logit": r(SH["mlp_base"], 3), "base_pd": r(100 * sig(SH["mlp_base"]), 1),
             "n_bg": 100, "n_explained": int(len(Xsub)), "nsamples": 600,
             "top1": FEATS[order[0]], "top2": FEATS[order[1]], "top3": FEATS[order[2]],
             "top1_val": r(imp["mlp"][order[0]], 3), "top2_val": r(imp["mlp"][order[1]], 3),
             "xgb_top1": FEATS[int(np.argmax(imp["xgb"]))], "lr_top1": FEATS[int(np.argmax(imp["lr"]))],
             "rho_mlp_xgb": r(stats.spearmanr(imp["mlp"], imp["xgb"])[0], 2),
             "rho_mlp_lr": r(stats.spearmanr(imp["mlp"], imp["lr"])[0], 2), "seconds": r(SH["_seconds"], 0)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    yy = np.arange(len(top10))[::-1]
    w = 0.27
    for k, (key, c, lab) in enumerate((("mlp", npl.PAL["model"], L(lang, "MLP (Kernel SHAP)", "MLP (Kernel SHAP)")),
                                        ("xgb", npl.PAL["alt"], L(lang, "XGBoost (TreeSHAP)", "XGBoost (TreeSHAP)")),
                                        ("lr", npl.PAL["benchmark"], L(lang, "logistic regression (linear SHAP)", "regresie logistică (SHAP liniar)")))):
        ax.barh(yy + (1 - k) * w, imp[key][top10], height=w, color=c, label=lab)
    ax.set_yticks(yy)
    ax.set_yticklabels([nice(FEATS[j], lang) for j in top10], fontsize=9)
    ax.set_xlabel(L(lang, "Mean |SHAP value| (log-odds of default)", "Media |valorii SHAP| (log-șansele de neplată)"))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "shap_global", also_x=(ax,))
npl.chart_meta(N, "shap_global", "hmeq", "400 test applicants, 100 background applicants", "400 de solicitanți din test, 100 de solicitanți de referință")

# beeswarm of the MLP
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for row, j in enumerate(top10[::-1]):
        v = sv_mlp[:, j]
        xv = Xsub[:, j]
        q = stats.rankdata(xv) / len(xv)
        jit = (np.random.default_rng(j).random(len(v)) - 0.5) * 0.6
        sc = ax.scatter(v, row + jit, c=q, cmap="coolwarm", s=5, lw=0, vmin=0, vmax=1)
    ax.axvline(0, color=npl.MUTED, lw=0.6)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([nice(FEATS[j], lang) for j in top10[::-1]], fontsize=9)
    ax.set_xlabel(L(lang, "SHAP value (contribution to the log-odds of default)", "Valoarea SHAP (contribuția la log-șansele de neplată)"))
    cax = fig.add_axes([0.35, 0.02, 0.3, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(L(lang, ["low value", "high value"], ["valoare mică", "valoare mare"]), fontsize=9)
    cb.outline.set_visible(False)
    cax._keep_y = True
    npl.out(fig, HERE, lang, "shap_beeswarm", bottom=0.12)
npl.chart_meta(N, "shap_beeswarm", "hmeq", "400 test applicants; colour = rank of the feature value",
               "400 de solicitanți din test; culoarea = rangul valorii variabilei")

# one rejected applicant: highest predicted PD among those with a predicted PD in (0.5, 0.8) who did default
p_sub = sig(f0(Xsub))
cand = np.where((p_sub > 0.5) & (p_sub < 0.8) & (yte[i_sub] == 1))[0]
k_app = int(cand[np.argmin(np.abs(p_sub[cand] - 0.65))])
x_app = Xsub[k_app]
raw_app = XR.iloc[i_te[i_sub[k_app]]]
phi = sv_mlp[k_app]
ordl = np.argsort(-np.abs(phi))
N["app"] = {"row": int(XR.index[i_te[i_sub[k_app]]]), "pd": r(100 * p_sub[k_app], 1), "logit": r(f0(x_app[None])[0], 3),
            "base": r(SH["mlp_base"], 3), "sum_phi": r(phi.sum(), 3),
            "f1": FEATS[ordl[0]], "f1_phi": r(phi[ordl[0]], 3), "f2": FEATS[ordl[1]], "f2_phi": r(phi[ordl[1]], 3),
            "f3": FEATS[ordl[2]], "f3_phi": r(phi[ordl[2]], 3),
            "debtinc": r(raw_app["DEBTINC"], 1), "delinq": int(raw_app["DELINQ"]), "derog": int(raw_app["DEROG"]),
            "clage": r(raw_app["CLAGE"], 0), "loan": int(raw_app["LOAN"]), "ninq": int(raw_app["NINQ"]),
            "debtinc_missing": int(raw_app["DEBTINC_missing"]), "default": int(yte[i_sub[k_app]])}
for nm in ("f1", "f2", "f3"):
    f_ = N["app"][nm]
    N["app"][nm + "_raw"] = r(raw_app[f_], 1)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    k = 8
    js = ordl[:k]
    rest = phi.sum() - phi[js].sum()
    vals = list(phi[js]) + [rest]
    labs = [f"{nice(FEATS[j], lang)} = {npl.ro_number(raw_app[FEATS[j]], 1) if lang == 'ro' else f'{raw_app[FEATS[j]]:.1f}'}" for j in js] + \
        [L(lang, f"{len(FEATS) - k} other features", f"celelalte {len(FEATS) - k} variabile")]
    start = SH["mlp_base"]
    for i, v in enumerate(vals):
        ax.barh(len(vals) - 1 - i, v, left=start, color=npl.PAL["positive"] if v > 0 else npl.PAL["negative"], height=0.7)
        start += v
    ax.axvline(SH["mlp_base"], color=npl.MUTED, ls="--", lw=0.8)
    ax.axvline(start, color="#222222", lw=0.8)
    ax.barh([0], [0], color=npl.PAL["positive"], label=L(lang, "raises the risk", "crește riscul"))
    ax.barh([0], [0], color=npl.PAL["negative"], label=L(lang, "lowers the risk", "reduce riscul"))
    ax.plot([], [], color=npl.MUTED, ls="--", label=L(lang, "$E[f(X)]$ (average applicant)", "$E[f(X)]$ (solicitantul mediu)"))
    ax.plot([], [], color="#222222", label=L(lang, "$f(x)$ for this applicant", "$f(x)$ pentru acest solicitant"))
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs[::-1], fontsize=8.5)
    ax.set_xlabel(L(lang, "Log-odds of default", "Log-șansele de neplată"))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "shap_waterfall", also_x=(ax,))
npl.chart_meta(N, "shap_waterfall", "hmeq", f"test applicant (row {N['app']['row']}); values in original units",
               f"solicitant din test (rîndul {N['app']['row']}); valori în unitățile originale")

# dependence plot: SHAP of DEBTINC vs DEBTINC (imputed rows marked)
jd = FEATS.index("DEBTINC")
miss = Xsub[:, FEATS.index("DEBTINC_missing")] > 0
raw_d = Xsub[:, jd] * sdv["DEBTINC"] + mu["DEBTINC"]
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.scatter(raw_d[~miss], sv_mlp[~miss, jd], s=7, color=npl.PAL["data"], lw=0, label=L(lang, "DEBTINC reported", "DEBTINC raportat"))
    ax.scatter(raw_d[miss], sv_mlp[miss, jd], s=9, color=npl.PAL["positive"], lw=0, label=L(lang, "DEBTINC missing (median imputed)", "DEBTINC lipsă (imputat cu mediana)"))
    ax.axhline(0, color=npl.MUTED, lw=0.6)
    ax.set_xlabel(L(lang, "Debt-to-income ratio DEBTINC (%)", "Raportul datorie/venit DEBTINC (%)"))
    ax.set_ylabel(L(lang, "SHAP value of DEBTINC", "Valoarea SHAP a DEBTINC"))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "shap_dependence", also_x=(ax,))
npl.chart_meta(N, "shap_dependence", "hmeq", "400 test applicants, MLP", "400 de solicitanți din test, MLP")
hi = (~miss) & (raw_d > 45)
lo_ = (~miss) & (raw_d < 30)
N["dep"] = {"shap_hi": r(sv_mlp[hi, jd].mean(), 2), "shap_lo": r(sv_mlp[lo_, jd].mean(), 2), "n_hi": int(hi.sum()),
            "shap_missing_flag": r(sv_mlp[miss, FEATS.index("DEBTINC_missing")].mean(), 2)}

# ---------------------------------------------------------------- LIME on the same applicant
lime_ex = lambda kw=None, seed=0: LimeTabularExplainer(  # noqa: E731
    Xtr, mode="regression", feature_names=FEATS, discretize_continuous=False, sample_around_instance=True,
    random_state=seed, kernel_width=kw)
D_ = Xtr.shape[1]


def lime_weights(x, num_samples=5000, kw=None, seed=0, f=f0):
    e = lime_ex(kw, seed).explain_instance(x, f, num_features=D_, num_samples=num_samples)
    w = np.zeros(D_)
    for j, v in e.local_exp[0]:
        w[j] = v
    return w


def lime_runs():
    out = {}
    out["main"] = lime_weights(x_app)
    out["s500"] = np.array([lime_weights(x_app, 500, seed=s) for s in range(30)])
    out["s5000"] = np.array([lime_weights(x_app, 5000, seed=s) for s in range(30)])
    kws = [0.5, 1.0, 2.0, 3.0, 4.12, 6.0, 10.0]
    out["kw"] = kws
    out["kw_w"] = np.array([lime_weights(x_app, 5000, kw=k, seed=0) for k in kws])
    return out


LM = cached("lime_v1", lime_runs)
lime_contrib = LM["main"] * x_app                              # local slope x standardised value (deviation from the mean)
top_l = np.argsort(-np.abs(LM["main"]))
N["lime"] = {"kw_default": r(0.75 * np.sqrt(D_), 2), "top1": FEATS[top_l[0]], "top2": FEATS[top_l[1]],
             "rho_shap": r(stats.spearmanr(np.abs(lime_contrib), np.abs(phi))[0], 2),
             "top1_same_500": r(100 * np.mean([np.argmax(np.abs(w)) == np.argmax(np.abs(LM["s5000"][0])) for w in LM["s500"]]), 0),
             "top1_same_5000": r(100 * np.mean([np.argmax(np.abs(w)) == np.argmax(np.abs(LM["s5000"][0])) for w in LM["s5000"]]), 0),
             "seconds": r(LM["_seconds"], 0)}


def jacc5(W):
    S = [set(np.argsort(-np.abs(w))[:5]) for w in W]
    return np.mean([len(S[a] & S[b]) / len(S[a] | S[b]) for a in range(len(S)) for b in range(a + 1, len(S))])


N["lime"]["jacc_500"] = r(jacc5(LM["s500"]), 2)
N["lime"]["jacc_5000"] = r(jacc5(LM["s5000"]), 2)
kwj = top_l[0]
N["lime"]["kw_top1_min"] = r(LM["kw_w"][:, kwj].min(), 3)
N["lime"]["kw_top1_max"] = r(LM["kw_w"][:, kwj].max(), 3)
N["lime"]["kw_sign_flips"] = int(sum(np.ptp(np.sign(LM["kw_w"][:, j])) > 0 for j in top_l[:5]))
cv500 = np.abs(LM["s500"][:, top_l[:5]].std(0) / LM["s500"][:, top_l[:5]].mean(0)).mean()
cv5000 = np.abs(LM["s5000"][:, top_l[:5]].std(0) / LM["s5000"][:, top_l[:5]].mean(0)).mean()
N["lime"]["cv_500"] = r(100 * cv500, 0)
N["lime"]["cv_5000"] = r(100 * cv5000, 0)

for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    js = ordl[:8]
    yy = np.arange(len(js))[::-1]
    ax.barh(yy + 0.2, phi[js], height=0.38, color=npl.PAL["model"], label=L(lang, "Kernel SHAP $\\phi_j$", "Kernel SHAP $\\phi_j$"))
    ax.barh(yy - 0.2, lime_contrib[js], height=0.38, color=npl.PAL["alt"], label=L(lang, "LIME slope $\\times$ standardised value", "LIME: panta $\\times$ valoarea standardizată"))
    ax.axvline(0, color=npl.MUTED, lw=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([nice(FEATS[j], lang) for j in js], fontsize=9)
    ax.set_xlabel(L(lang, "Contribution to the log-odds of default", "Contribuția la log-șansele de neplată"))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "lime_vs_shap", also_x=(ax,))
npl.chart_meta(N, "lime_vs_shap", "hmeq", f"test applicant (row {N['app']['row']}); LIME with 5,000 samples",
               f"solicitant din test (rîndul {N['app']['row']}); LIME cu 5 000 de eșantioane")

for lang in ("en", "ro"):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    js = top_l[:5]
    pos = np.arange(len(js))
    b1 = axs[0].boxplot([LM["s500"][:, j] for j in js], positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=False)
    b2 = axs[0].boxplot([LM["s5000"][:, j] for j in js], positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=False)
    for b, c in ((b1, npl.PAL["alt"]), (b2, npl.PAL["model"])):
        for p_ in b["boxes"]:
            p_.set_facecolor(c)
            p_.set_alpha(0.6)
        for m_ in b["medians"]:
            m_.set_color("#222222")
    axs[0].set_xticks(pos)
    axs[0].set_xticklabels([nice(FEATS[j], lang).replace(" missing", "\nmissing").replace(" lipsă", "\nlipsă") for j in js], fontsize=8)
    axs[0].set_ylabel(L(lang, "LIME slope", "Panta LIME"))
    axs[0].axhline(0, color=npl.MUTED, lw=0.6)
    axs[0].bar([0], [0], color=npl.PAL["alt"], alpha=0.6, label=L(lang, "500 samples (30 runs)", "500 de eșantioane (30 de rulări)"))
    axs[0].bar([0], [0], color=npl.PAL["model"], alpha=0.6, label=L(lang, "5,000 samples (30 runs)", "5 000 de eșantioane (30 de rulări)"))
    for k, j in enumerate(js):
        axs[1].plot(LM["kw"], LM["kw_w"][:, j], marker="o", ms=3, color=npl.SERIES[k % 8])
    axs[1].axvline(0.75 * np.sqrt(D_), color=npl.MUTED, ls="--", lw=0.8)
    axs[1].set_xscale("log")
    axs[1].set_xticks([0.5, 1, 2, 4, 10])
    axs[1].set_xticklabels(["0.5", "1", "2", "4", "10"] if lang == "en" else ["0,5", "1", "2", "4", "10"])
    axs[1].set_xlabel(L(lang, "Kernel width $\\sigma$ (dashed: default)", "Lățimea nucleului $\\sigma$ (punctat: implicit)"))
    axs[1].axhline(0, color=npl.MUTED, lw=0.6)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "lime_instability")
npl.chart_meta(N, "lime_instability", "hmeq", f"test applicant (row {N['app']['row']}); right panel: same five features, 5,000 samples",
               f"solicitant din test (rîndul {N['app']['row']}); dreapta: aceleași cinci variabile, 5 000 de eșantioane")

# ---------------------------------------------------------------- counterfactual (Wachter et al., 2017) for the applicant
ACTION = ["LOAN", "DEBTINC", "NINQ", "DELINQ", "DEROG", "CLNO", "MORTDUE"]
mad = (XR.iloc[i_tr][NUM] - XR.iloc[i_tr][NUM].median()).abs().median().replace(0, np.nan)
mad = mad.fillna(XR.iloc[i_tr][NUM].std())


def counterfactual(x0, target=-0.5, lam=10.0, steps=3000, lr=0.01):
    """Wachter et al. (2017): min_x' lam (f(x') - target)_+^2 + sum_j |x'_j - x_j| / MAD_j over the actionable features,
    with every raw value kept >= 0 (standardised space); counts are rounded to integers at the end."""
    x = torch.tensor(x0, requires_grad=True)
    m = torch.tensor([1.0 if f in ACTION else 0.0 for f in FEATS])
    scale = torch.tensor([sdv[f] / mad[f] if f in ACTION else 0.0 for f in FEATS], dtype=torch.float32)
    lo = torch.tensor([float((0 - mu[f]) / sdv[f]) for f in FEATS], dtype=torch.float32)
    x0t = torch.tensor(x0)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        xx = x0t + m * (x - x0t)
        loss = lam * torch.relu(net0(xx[None])[0] - target) ** 2 + (scale * (xx - x0t).abs()).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.copy_(torch.maximum(x, lo))
    xx = (x0t + m * (x - x0t)).detach().numpy()
    raw = xx * sdv.values + mu.values
    for j, f in enumerate(FEATS):
        if f in ("NINQ", "DELINQ", "DEROG", "CLNO"):
            raw[j] = np.round(raw[j])
        if f == "LOAN":
            raw[j] = np.round(raw[j], -2)
    return ((raw - mu.values) / sdv.values).astype(np.float32)


x_cf = counterfactual(x_app.copy())
raw_cf = x_cf * sdv.values + mu.values
chg = [(f, float(raw_app[f]), float(raw_cf[j])) for j, f in enumerate(FEATS)
       if f in ACTION and abs(raw_cf[j] - raw_app[f]) > 0.02 * (abs(raw_app[f]) + 1)]
chg = sorted(chg, key=lambda t: -abs(t[2] - t[1]) / mad[t[0]])[:4]
# documentary counterfactual: the applicant reports the debt-to-income ratio (value = training median, flag = 0)
x_doc = x_app.copy()
x_doc[FEATS.index("DEBTINC_missing")] = (0 - mu["DEBTINC_missing"]) / sdv["DEBTINC_missing"]
N["cf"] = {"pd_before": r(100 * sig(f0(x_app[None])[0]), 1), "pd_after": r(100 * sig(f0(x_cf[None])[0]), 1),
           "pd_doc": r(100 * sig(f0(x_doc[None])[0]), 1), "debtinc_median": r(med["DEBTINC"], 1),
           "n_changed": len(chg), "rows": [{"f": f, "before": r(a, 1), "after": r(b, 1)} for f, a, b in chg],
           "actionable": ", ".join(ACTION)}
for k_, (f, a, b) in enumerate(chg):
    N["cf"][f"f{k_}"], N["cf"][f"b{k_}"], N["cf"][f"a{k_}"] = f, r(a, 1), r(b, 1)


# ---------------------------------------------------------------- stability of SHAP rankings across 10 seeds
def seed_shap():
    out = {"mlp": [], "xgb": XG["shap_seed"]}
    Xs = Xsub[:200]
    for k, n_ in enumerate(nets):
        sv, _ = kernel_shap(logit_fn(n_), Xs, nsamples=600, seed=SEED)
        out["mlp"].append(sv)
    return out


SS = cached("seed_shap_v1", seed_shap)
imp_m = np.array([np.abs(s).mean(0) for s in SS["mlp"]])
imp_x = np.array([np.abs(s).mean(0) for s in SS["xgb"]])


def mean_tau(I):
    return np.mean([stats.kendalltau(I[a], I[b])[0] for a in range(len(I)) for b in range(a + 1, len(I))])


def top3_overlap(I):
    S = [set(np.argsort(-i)[:3]) for i in I]
    return np.mean([len(S[a] & S[b]) / 3 for a in range(len(S)) for b in range(a + 1, len(S))])


# local stability: top feature of each of the 200 applicants, share of seed pairs that agree
def local_agree(SV):
    tops = np.array([np.argmax(np.abs(s), 1) for s in SV])         # seeds x applicants
    return np.mean([(tops[a] == tops[b]).mean() for a in range(len(tops)) for b in range(a + 1, len(tops))])


rank_m = np.array([stats.rankdata(-i) for i in imp_m])
rank_x = np.array([stats.rankdata(-i) for i in imp_x])
N["stab"] = {"tau_mlp": r(mean_tau(imp_m), 2), "tau_xgb": r(mean_tau(imp_x), 2),
             "top3_mlp": r(100 * top3_overlap(imp_m), 0), "top3_xgb": r(100 * top3_overlap(imp_x), 0),
             "local_mlp": r(100 * local_agree(SS["mlp"]), 0), "local_xgb": r(100 * local_agree(SS["xgb"]), 0),
             "auc_range_mlp_lo": r(m_mlp[:, 0].min(), 3), "auc_range_mlp_hi": r(m_mlp[:, 0].max(), 3),
             "top1_mlp_share": r(100 * np.mean(np.argmax(imp_m, 1) == np.bincount(np.argmax(imp_m, 1)).argmax()), 0),
             "n_seeds": len(nets), "seconds": r(SS["_seconds"], 0)}
ord_s = np.argsort(rank_m.mean(0))[:8]
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    xs = np.arange(len(ord_s))
    for k in range(len(nets)):
        jit = (k - 4.5) * 0.025
        ax.scatter(xs - 0.15 + jit, rank_m[k, ord_s], s=12, color=npl.PAL["model"], lw=0, alpha=0.8,
                   label=L(lang, "MLP, one seed", "MLP, o valoare inițială") if k == 0 else None)
        ax.scatter(xs + 0.15 + jit, rank_x[k, ord_s], s=12, color=npl.PAL["alt"], lw=0, alpha=0.8,
                   label=L(lang, "XGBoost, one seed", "XGBoost, o valoare inițială") if k == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([nice(FEATS[j], lang).replace(" missing", "\nmissing").replace(" lipsă", "\nlipsă") for j in ord_s], fontsize=8)
    ax.set_ylabel(L(lang, "Rank by mean |SHAP| (1 = most important)", "Rangul după media |SHAP| (1 = cea mai importantă)"))
    ax.invert_yaxis()
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "seed_stability")
npl.chart_meta(N, "seed_stability", "hmeq", "10 training seeds; 200 test applicants", "10 valori inițiale de antrenare; 200 de solicitanți din test")

print(f"credit part done ({time.time() - T0:.0f}s)")

# ================================================================ B. fraud: gradient methods, agreement, faithfulness, shift
fr = npl.fraud()
VF = [f"V{i}" for i in range(1, 29)]
fr["logAmount"] = np.log1p(fr["Amount"])
FF = VF + ["logAmount"]
day1 = fr["Time"] < 86400
d1 = fr[day1]
d2 = fr[~day1]
j_tr, j_va = train_test_split(np.arange(len(d1)), test_size=0.25, stratify=d1["Class"], random_state=SEED)
fmu, fsd = d1.iloc[j_tr][FF].mean(), d1.iloc[j_tr][FF].std()
Z1 = ((d1[FF] - fmu) / fsd).values.astype(np.float32)
Z2 = ((d2[FF] - fmu) / fsd).values.astype(np.float32)
y1, y2 = d1["Class"].values, d2["Class"].values
w1, w2 = d1["weight"].values, d2["weight"].values
N["fraud"] = {"n_day1": int(len(d1)), "n_day2": int(len(d2)), "fr_day1": int(y1.sum()), "fr_day2": int(y2.sum()),
              "rate_full_day2_pct": r(100 * (w2 * y2).sum() / w2.sum(), 3), "n_feat": len(FF)}


def fraud_model():
    pos_w = float((w1[j_tr] * (1 - y1[j_tr])).sum() / (w1[j_tr] * y1[j_tr]).sum()) ** 0.5
    net, ep = train_mlp(SEED, Z1[j_tr], y1[j_tr], Z1[j_va], y1[j_va], w_pos=pos_w, epochs=200, patience=20,
                        sw_valid=w1[j_va])
    return {"state": {k: v.clone() for k, v in net.state_dict().items()}, "epoch": ep, "pos_w": pos_w}


FM = cached("fraud_model_v1", fraud_model)
fnet = MLP(len(FF))
fnet.load_state_dict(FM["state"])
fnet.eval()
ff = logit_fn(fnet)
p2 = sig(ff(Z2))
lr_f = LogisticRegression(C=1.0, max_iter=3000, class_weight={0: 1, 1: FM["pos_w"]}).fit(Z1[j_tr], y1[j_tr], sample_weight=w1[j_tr])
N["fraud"].update({"mlp_prauc": r(average_precision_score(y2, p2, sample_weight=w2), 3),
                   "mlp_auc": r(roc_auc_score(y2, p2, sample_weight=w2), 3),
                   "lr_prauc": r(average_precision_score(y2, lr_f.predict_proba(Z2)[:, 1], sample_weight=w2), 3),
                   "lr_auc": r(roc_auc_score(y2, lr_f.predict_proba(Z2)[:, 1], sample_weight=w2), 3),
                   "epoch": int(FM["epoch"])})

# baseline x' = mean legitimate training transaction (standardised); frauds of day 2 are explained
base_f = Z1[j_tr][y1[j_tr] == 0].mean(0)
F2 = Z2[y2 == 1]
F1v = Z1[j_va][y1[j_va] == 1]


def grad_of(X):
    x = torch.tensor(np.asarray(X, dtype=np.float32), requires_grad=True)
    out = fnet(x).sum()
    g, = torch.autograd.grad(out, x)
    return g.numpy()


def integrated_gradients(X, m=64, base=None):
    base = base_f if base is None else base
    al = (np.arange(m) + 0.5) / m                                   # midpoint Riemann sum
    tot = np.zeros_like(X)
    for a in al:
        tot += grad_of(base + a * (X - base))
    return (X - base) * tot / m


def smoothgrad(X, n=50, s=0.15, seed=SEED):
    g = np.zeros_like(X)
    rg = np.random.default_rng(seed)
    for _ in range(n):
        g += grad_of(X + rg.normal(0, s, X.shape).astype(np.float32))
    return g / n


def attributions(X):
    A = {"saliency": np.abs(grad_of(X)), "gradxinput": grad_of(X) * (X - base_f), "smoothgrad": smoothgrad(X) * (X - base_f),
         "ig": integrated_gradients(X)}
    return A


def fraud_xai():
    A = attributions(F2)
    np.random.seed(SEED)
    bgf = Z1[j_tr][np.random.default_rng(SEED).choice(len(j_tr), 100, replace=False)]
    ex = shap.KernelExplainer(ff, bgf)
    A["shap"] = np.array(ex.shap_values(F2, nsamples=800, silent=True))
    lex = LimeTabularExplainer(Z1[j_tr], mode="regression", feature_names=FF, discretize_continuous=False,
                               sample_around_instance=True, random_state=SEED)
    LW = []
    for x in F2:
        e = lex.explain_instance(x, ff, num_features=len(FF), num_samples=2000)
        w = np.zeros(len(FF))
        for j, v in e.local_exp[0]:
            w[j] = v
        LW.append(w * (x - base_f))
    A["lime"] = np.array(LW)
    # IG completeness vs number of steps
    comp = {}
    gap = ff(F2) - ff(base_f[None])[0]
    for m in (2, 4, 8, 16, 32, 64, 128, 256):
        ig = integrated_gradients(F2, m=m)
        comp[m] = float(np.median(np.abs(ig.sum(1) - gap) / np.abs(gap)))
    # permutation importance on day 2 (weighted PR-AUC drop, 5 repeats)
    rg = np.random.default_rng(SEED)
    base_ap = average_precision_score(y2, ff(Z2), sample_weight=w2)
    pimp = np.zeros(len(FF))
    for j in range(len(FF)):
        drops = []
        for _ in range(5):
            Zp = Z2.copy()
            Zp[:, j] = Zp[rg.permutation(len(Zp)), j]
            drops.append(base_ap - average_precision_score(y2, ff(Zp), sample_weight=w2))
        pimp[j] = np.mean(drops)
    # day-1 (held-out) frauds for the shift comparison
    A1 = integrated_gradients(F1v)
    return {"A": A, "comp": comp, "pimp": pimp, "ig_day1": A1, "bg": bgf}


FX = cached("fraud_xai_v1", fraud_xai)
A = FX["A"]
METH = ["saliency", "gradxinput", "smoothgrad", "ig", "shap", "lime"]
MN = {"en": {"saliency": "saliency", "gradxinput": "gradient × input", "smoothgrad": "SmoothGrad", "ig": "integrated gradients",
             "shap": "Kernel SHAP", "lime": "LIME", "perm": "permutation"},
      "ro": {"saliency": "saliență", "gradxinput": "gradient × intrare", "smoothgrad": "SmoothGrad", "ig": "gradienți integrați",
             "shap": "Kernel SHAP", "lime": "LIME", "perm": "permutare"}}
gimp = {m: np.abs(A[m]).mean(0) for m in METH}
gimp["perm"] = FX["pimp"]
ML = METH + ["perm"]
R = np.array([[stats.spearmanr(gimp[a], gimp[b])[0] for b in ML] for a in ML])
gap = ff(F2) - ff(base_f[None])[0]
N["ig"] = {"comp_" + str(m): float(f"{v:.2e}") for m, v in FX["comp"].items()}
N["ig"]["comp64_pct"] = r(100 * FX["comp"][64], 2)
N["ig"]["comp4_pct"] = r(100 * FX["comp"][4], 1)
N["ig"]["sum_check"] = r(np.median(np.abs(A["ig"].sum(1) - gap)), 4)
N["ig"]["n_frauds"] = int(len(F2))
N["agree"] = {"ig_shap": r(R[ML.index("ig"), ML.index("shap")], 2), "sal_shap": r(R[ML.index("saliency"), ML.index("shap")], 2),
              "lime_shap": r(R[ML.index("lime"), ML.index("shap")], 2), "perm_shap": r(R[ML.index("perm"), ML.index("shap")], 2),
              "sal_ig": r(R[ML.index("saliency"), ML.index("ig")], 2), "min": r(R[np.triu_indices(len(ML), 1)].min(), 2),
              "max": r(R[np.triu_indices(len(ML), 1)].max(), 2),
              "top_ig": FF[int(np.argmax(gimp["ig"]))], "top_shap": FF[int(np.argmax(gimp["shap"]))],
              "top_sal": FF[int(np.argmax(gimp["saliency"]))], "top_perm": FF[int(np.argmax(gimp["perm"]))]}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for a in range(len(ML)):
        for b in range(len(ML)):
            v = R[a, b]
            ax.text(b, a, (f"{v:.2f}" if lang == "en" else npl.ro_number(v, 2)), ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "#222222")
    labs = [MN[lang][m] for m in ML]
    ax.set_xticks(range(len(ML)))
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(ML)))
    ax.set_yticklabels(labs, fontsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    npl.out(fig, HERE, lang, "method_agreement", bottom=0.0)
npl.chart_meta(N, "method_agreement", "fraud", f"Spearman correlation of global importances; {len(F2)} frauds of day 2",
               f"corelația Spearman a importanțelor globale; {len(F2)} fraude din ziua a doua")

for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ms = sorted(FX["comp"])
    ax.loglog(ms, [100 * FX["comp"][m] for m in ms], marker="o", color=npl.PAL["model"],
              label=L(lang, "median relative completeness error", "eroarea relativă mediană de completitudine"))
    ax.set_xlabel(L(lang, "Number of Riemann steps $m$", "Numărul de pași Riemann $m$"))
    ax.set_ylabel(L(lang, "Error (%)", "Eroare (%)"))
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax._keep_y = True
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "ig_completeness")
npl.chart_meta(N, "ig_completeness", "fraud", f"{len(F2)} frauds of day 2; baseline = mean legitimate transaction of day 1",
               f"{len(F2)} fraude din ziua a doua; referința = tranzacția legitimă medie din ziua întîi")

# one fraud: normalised attributions of four methods
kf = int(np.argsort(sig(ff(F2)))[len(F2) // 2])                     # the median-score fraud of day 2
N["onefraud"] = {"p": r(100 * sig(ff(F2[kf:kf + 1]))[0], 1)}
topf = np.argsort(-np.abs(A["shap"][kf]))[:8]
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    w = 0.2
    yy = np.arange(len(topf))[::-1]
    for k, (m, c) in enumerate((("saliency", npl.PAL["data"]), ("gradxinput", npl.PAL["c7"]), ("ig", npl.PAL["model"]),
                                ("shap", npl.PAL["alt"]))):
        v = A[m][kf] / np.abs(A[m][kf]).sum()
        ax.barh(yy + (1.5 - k) * w, v[topf], height=w, color=c, label=MN[lang][m])
    ax.axvline(0, color=npl.MUTED, lw=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([FF[j] for j in topf], fontsize=9)
    ax.set_xlabel(L(lang, "Share of total |attribution| (signed)", "Ponderea în |atribuirea| totală (cu semn)"))
    npl.legend_bottom(fig, ncol=4)
    npl.out(fig, HERE, lang, "fraud_local", also_x=(ax,))
npl.chart_meta(N, "fraud_local", "fraud", "one fraud of day 2 (median fraud score)", "o fraudă din ziua a doua (scorul median al fraudelor)")
N["onefraud"]["top_shap"] = FF[topf[0]]
N["onefraud"]["top_ig"] = FF[int(np.argmax(np.abs(A["ig"][kf])))]
N["onefraud"]["top_sal"] = FF[int(np.argmax(np.abs(A["saliency"][kf])))]


# deletion curves: replace the k most important features (per transaction) by the baseline
def deletion(Amat, X, K=12, rand=False, seed=SEED):
    rg = np.random.default_rng(seed)
    out = np.zeros(K + 1)
    for i, x in enumerate(X):
        o = rg.permutation(len(x)) if rand else np.argsort(-np.abs(Amat[i]))
        xs = np.repeat(x[None], K + 1, 0)
        for k in range(1, K + 1):
            xs[k:, o[k - 1]] = base_f[o[k - 1]]
        out += sig(ff(xs))
    return out / len(X)


DEL = {m: deletion(A[m], F2) for m in ("saliency", "ig", "shap", "lime")}
DEL["random"] = deletion(None, F2, rand=True)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for m, c in (("ig", npl.PAL["model"]), ("shap", npl.PAL["alt"]), ("lime", npl.PAL["c5"]), ("saliency", npl.PAL["data"]),
                 ("random", npl.MUTED)):
        ax.plot(range(13), 100 * DEL[m], marker="o", ms=3, color=c, label=MN[lang].get(m, L(lang, "random order", "ordine aleatoare")))
    ax.set_xlabel(L(lang, "Number of top-ranked features reset to the baseline", "Numărul de variabile de top readuse la referință"))
    ax.set_ylabel(L(lang, "Mean fraud score (%)", "Scorul mediu de fraudă (%)"))
    npl.legend_bottom(fig, ncol=5)
    npl.out(fig, HERE, lang, "deletion_curve")
npl.chart_meta(N, "deletion_curve", "fraud", f"{len(F2)} frauds of day 2", f"{len(F2)} fraude din ziua a doua")
N["del"] = {m: r(100 * DEL[m][3], 1) for m in DEL}
N["del"]["start"] = r(100 * DEL["ig"][0], 1)
N["del"]["aopc_ig"] = r(100 * (DEL["ig"][0] - DEL["ig"][1:]).mean(), 1)
N["del"]["aopc_shap"] = r(100 * (DEL["shap"][0] - DEL["shap"][1:]).mean(), 1)
N["del"]["aopc_sal"] = r(100 * (DEL["saliency"][0] - DEL["saliency"][1:]).mean(), 1)
N["del"]["aopc_lime"] = r(100 * (DEL["lime"][0] - DEL["lime"][1:]).mean(), 1)
N["del"]["aopc_rand"] = r(100 * (DEL["random"][0] - DEL["random"][1:]).mean(), 1)


# local robustness: relative change of the attribution vector under small input noise (sigma = 0.05 sd)
def robustness():
    rg = np.random.default_rng(SEED)
    Xs = F2[:60]
    res = {m: [] for m in ("saliency", "gradxinput", "ig", "smoothgrad")}
    for _ in range(10):
        Xn = Xs + rg.normal(0, 0.05, Xs.shape).astype(np.float32)
        A0 = {"saliency": np.abs(grad_of(Xs)), "gradxinput": grad_of(Xs) * (Xs - base_f), "ig": integrated_gradients(Xs, m=32),
              "smoothgrad": smoothgrad(Xs, n=50) * (Xs - base_f)}
        A1 = {"saliency": np.abs(grad_of(Xn)), "gradxinput": grad_of(Xn) * (Xn - base_f), "ig": integrated_gradients(Xn, m=32),
              "smoothgrad": smoothgrad(Xn, n=50, seed=SEED + 1) * (Xn - base_f)}
        for m in res:
            res[m].append(np.linalg.norm(A1[m] - A0[m], axis=1) / np.linalg.norm(Xn - Xs, axis=1) /
                          (np.linalg.norm(A0[m], axis=1) + 1e-9))
    return {m: np.concatenate(v) for m, v in res.items()}


RB = cached("robust_v1", robustness)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ms = ["saliency", "gradxinput", "smoothgrad", "ig"]
    bp = ax.boxplot([RB[m] for m in ms], vert=False, patch_artist=True, showfliers=False, widths=0.55)
    for p_, c in zip(bp["boxes"], [npl.PAL["data"], npl.PAL["c7"], npl.PAL["benchmark"], npl.PAL["model"]]):
        p_.set_facecolor(c)
        p_.set_alpha(0.6)
    for m_ in bp["medians"]:
        m_.set_color("#222222")
    ax.set_yticks(range(1, len(ms) + 1))
    ax.set_yticklabels([MN[lang][m] for m in ms], fontsize=9)
    ax.set_xlabel(L(lang, "Relative change of the explanation per unit of input change", "Variația relativă a explicației pe unitatea de variație a intrării"))
    npl.out(fig, HERE, lang, "local_robustness", also_x=(ax,), bottom=0.0)
npl.chart_meta(N, "local_robustness", "fraud", "60 frauds of day 2 x 10 perturbations (sigma = 0.05 standard deviations)",
               "60 de fraude din ziua a doua x 10 perturbări (sigma = 0,05 abateri standard)")
N["rob"] = {m: r(np.median(RB[m]), 3) for m in RB if not m.startswith("_")}

# shift of explanations from day 1 to day 2 (integrated gradients, mean |attribution| of frauds)
g1 = np.abs(FX["ig_day1"]).mean(0)
g2 = np.abs(A["ig"]).mean(0)
top_t = np.argsort(-(g1 + g2))[:10]
p1n, p2n = g1 / g1.sum(), g2 / g2.sum()
mpn = 0.5 * (p1n + p2n)
js_div = 0.5 * (p1n * np.log(p1n / mpn)).sum() + 0.5 * (p2n * np.log(p2n / mpn)).sum()
N["shift"] = {"rho": r(stats.spearmanr(g1, g2)[0], 2), "js": r(js_div, 3), "n1": int(len(F1v)), "n2": int(len(F2)),
              "top1_day1": FF[int(np.argmax(g1))], "top1_day2": FF[int(np.argmax(g2))],
              "top5_overlap": int(len(set(np.argsort(-g1)[:5]) & set(np.argsort(-g2)[:5])))}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    xs = np.arange(len(top_t))
    ax.bar(xs - 0.2, p1n[top_t] * 100, width=0.4, color=npl.PAL["train"], label=L(lang, f"day 1, held-out frauds ({len(F1v)})", f"ziua 1, fraude rezervate ({len(F1v)})"))
    ax.bar(xs + 0.2, p2n[top_t] * 100, width=0.4, color=npl.PAL["test"], label=L(lang, f"day 2, test frauds ({len(F2)})", f"ziua 2, fraude de test ({len(F2)})"))
    ax.set_xticks(xs)
    ax.set_xticklabels([FF[j] for j in top_t], fontsize=9)
    ax.set_ylabel(L(lang, "Share of total |IG| (%)", "Ponderea în |IG| total (%)"))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "time_shift")
npl.chart_meta(N, "time_shift", "fraud", "model trained on day 1; integrated gradients", "model antrenat pe ziua 1; gradienți integrați")
print(f"fraud part done ({time.time() - T0:.0f}s)")

# ================================================================ C. California housing: PDP and ICE
ca = npl.california()
CF = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
k_tr, k_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED)
k_tr, k_va = train_test_split(k_tr, test_size=0.2, random_state=SEED)
cmu, csd = ca.iloc[k_tr][CF].mean(), ca.iloc[k_tr][CF].std()
CZ = ((ca[CF] - cmu) / csd).values.astype(np.float32)
cy = ca["MedHouseVal_100k_usd"].values.astype(np.float32)


def cal_model():
    net, ep = train_mlp(SEED, CZ[k_tr], cy[k_tr], CZ[k_va], cy[k_va], task="reg", epochs=200, patience=20, lr=2e-3)
    return {"state": {k: v.clone() for k, v in net.state_dict().items()}, "epoch": ep}


CAM = cached("cal_model_v1", cal_model)
cnet = MLP(len(CF))
cnet.load_state_dict(CAM["state"])
cnet.eval()
cf_ = logit_fn(cnet)
pred_te = cf_(CZ[k_te])
from sklearn.linear_model import LinearRegression  # noqa: E402
lin = LinearRegression().fit(CZ[k_tr], cy[k_tr])
N["cal"] = {"r2_mlp": r(1 - ((pred_te - cy[k_te]) ** 2).sum() / ((cy[k_te] - cy[k_te].mean()) ** 2).sum(), 3),
            "r2_lin": r(lin.score(CZ[k_te], cy[k_te]), 3), "n": int(len(ca)), "rmse_mlp": r(np.sqrt(((pred_te - cy[k_te]) ** 2).mean()), 3)}
grid_raw = np.linspace(np.percentile(ca.MedInc, 2), np.percentile(ca.MedInc, 98), 40)
grid_z = (grid_raw - cmu["MedInc"]) / csd["MedInc"]
rgc = np.random.default_rng(SEED)
ice_rows = CZ[k_te][rgc.choice(len(k_te), 60, replace=False)]
pdp_rows = CZ[k_te][rgc.choice(len(k_te), 1000, replace=False)]


def ice(rows):
    out = np.zeros((len(rows), len(grid_z)))
    for g, z in enumerate(grid_z):
        Xg = rows.copy()
        Xg[:, 0] = z
        out[:, g] = cf_(Xg)
    return out


ICE = ice(ice_rows)
PDP = ice(pdp_rows).mean(0)
N["pdp"] = {"lo": r(grid_raw[0], 2), "hi": r(grid_raw[-1], 2), "pdp_lo": r(PDP[0], 2), "pdp_hi": r(PDP[-1], 2),
            "slope_lin": r(lin.coef_[0] / csd["MedInc"], 3),
            "ice_spread_hi": r(ICE[:, -1].std(), 2),
            "corr_rooms": r(np.corrcoef(ca.MedInc, ca.AveRooms)[0, 1], 2)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for i, row in enumerate(ICE):
        ax.plot(grid_raw, row, color=npl.PAL["data"], alpha=0.25, lw=0.7, label=L(lang, "ICE: one block group", "ICE: un grup de locuințe") if i == 0 else None)
    ax.plot(grid_raw, PDP, color=npl.PAL["model"], lw=2.2, label=L(lang, "partial dependence (mean of 1,000)", "dependența parțială (media a 1 000)"))
    ax.set_xlabel(L(lang, "Median income MedInc (tens of thousands of USD)", "Venitul median MedInc (zeci de mii de USD)"))
    ax.set_ylabel(L(lang, "Predicted value (100,000 USD)", "Valoarea prognozată (100 000 USD)"))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "pdp_ice", also_x=(ax,))
npl.chart_meta(N, "pdp_ice", "california", "MLP; 60 test block groups (ICE), 1,000 for the PDP", "MLP; 60 de grupuri de test (ICE), 1 000 pentru PDP")

N["runtime_s"] = r(time.time() - T0, 0)
with open(os.path.join(HERE, "L11_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, default=float)
print("wrote L11_numbers.json and", len(N["charts"]), "charts x 2 languages in", N["runtime_s"], "s")
