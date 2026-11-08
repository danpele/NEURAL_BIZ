"""
make_quantlets_L11.py -- one Quantlet per chart of Lecture 11: Quantlets/L11/NN_L11_<chart>/ (Metainfo.txt, executed
notebook, <chart>.png). Every number printed with show() is compared with L11_numbers.json. Deterministic numbers
(data counts, logistic benchmarks) are checked; explanations are recomputed with smaller budgets (fewer applicants,
fewer Kernel SHAP samples), so they are printed, not checked. PyTorch is imported before scikit-learn (macOS OpenMP).
Run:  python3 lectures/L11/make_quantlets_L11.py [chart ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

NUM = os.path.join(HERE, "L11_numbers.json")
SEED = 20261108
CORE = """import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import shap
SEED = 20261108; np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
class MLP(nn.Module):
    def __init__(self, d, h1=64, h2=32, p=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(p), nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p), nn.Linear(h2, 1))
    def forward(self, x): return self.f(x).squeeze(-1)
def train_mlp(seed, Xa, ya, Xv, yv, w_pos=1.0, epochs=300, patience=25, task='clf', lr=1e-3, sw_valid=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = MLP(Xa.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xa), torch.from_numpy(ya.astype(np.float32)); best, state, bad = -np.inf, None, 0
    for ep in range(epochs):
        net.train(); perm = rng.permutation(len(Xa))
        for j in range(0, len(Xa), 256):
            b = torch.from_numpy(perm[j:j + 256]); out = net(Xt[b])
            if task == 'clf':
                w = torch.where(yt[b] > 0, torch.tensor(w_pos), torch.tensor(1.0))
                loss = nn.functional.binary_cross_entropy_with_logits(out, yt[b], weight=w)
            else:
                loss = ((out - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pv = net(torch.from_numpy(Xv)).numpy()
        sc = (roc_auc_score(yv, pv) if w_pos == 1.0 else average_precision_score(yv, pv, sample_weight=sw_valid)) if task == 'clf' else -np.mean((pv - yv) ** 2)
        if sc > best: best, state, bad = sc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience: break
    net.load_state_dict(state); net.eval(); return net
def logit_fn(net):
    def f(X):
        with torch.no_grad(): return net(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy().astype(np.float64)
    return f
sig = lambda z: 1 / (1 + np.exp(-z))"""
HMEQ = """h = pd.read_csv(DATA + 'hmeq.csv')
NUM = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
y_all = h['BAD'].values; idx = np.arange(len(h))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=y_all[i_tr], random_state=SEED)
med = h.iloc[i_tr][NUM].median()
XR = pd.DataFrame({c: h[c].fillna(med[c]) for c in NUM})
XR['DEBTINC_missing'] = h['DEBTINC'].isna().astype(float); XR['REASON_HomeImp'] = (h['REASON'] == 'HomeImp').astype(float)
for j in ['Mgr', 'Office', 'Other', 'ProfExe', 'Sales', 'Self']: XR['JOB_' + j] = (h['JOB'] == j).astype(float)
FEATS = list(XR.columns); mu, sdv = XR.iloc[i_tr].mean(), XR.iloc[i_tr].std()
XS = ((XR - mu) / sdv).values.astype(np.float32)
Xtr, Xva, Xte = XS[i_tr], XS[i_va], XS[i_te]; ytr, yva, yte = y_all[i_tr], y_all[i_va], y_all[i_te]
rng = np.random.default_rng(SEED); BG = Xtr[rng.choice(len(Xtr), 100, replace=False)]; i_sub = rng.choice(len(Xte), 400, replace=False); Xsub = Xte[i_sub]
lrm = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
show('hmeq.n', 'loans', int(len(h))); show('hmeq.n_test', 'test loans', int(len(i_te)))
show('perf.lr_auc', 'logistic regression test AUC', round(float(roc_auc_score(yte, lrm.predict_proba(Xte)[:, 1])), 3))
net0 = train_mlp(SEED, Xtr, ytr, Xva, yva); f0 = logit_fn(net0)
print('MLP test AUC (seed 20261108):', round(roc_auc_score(yte, f0(Xte)), 3))"""
KSHAP = """np.random.seed(SEED); ex = shap.KernelExplainer(f0, BG)
NE = 150
sv = np.array(ex.shap_values(Xsub[:NE], nsamples=400, silent=True)); base = float(f0(BG).mean())
print('additivity error:', float(np.abs(sv.sum(1) + base - f0(Xsub[:NE])).max()))"""
FRAUD = """fr = pd.read_csv(DATA + 'creditcard_fraud_sample.csv'); fr['logAmount'] = np.log1p(fr['Amount'])
FF = [f'V{i}' for i in range(1, 29)] + ['logAmount']; d1, d2 = fr[fr.Time < 86400], fr[fr.Time >= 86400]
j_tr, j_va = train_test_split(np.arange(len(d1)), test_size=0.25, stratify=d1['Class'], random_state=SEED)
fmu, fsd = d1.iloc[j_tr][FF].mean(), d1.iloc[j_tr][FF].std()
Z1 = ((d1[FF] - fmu) / fsd).values.astype(np.float32); Z2 = ((d2[FF] - fmu) / fsd).values.astype(np.float32)
y1, y2, w1, w2 = d1['Class'].values, d2['Class'].values, d1['weight'].values, d2['weight'].values
show('fraud.n_day1', 'transactions in the sample, day 1', int(len(d1))); show('fraud.fr_day2', 'frauds on day 2', int(y2.sum()))
pw = float((w1[j_tr] * (1 - y1[j_tr])).sum() / (w1[j_tr] * y1[j_tr]).sum()) ** 0.5
fnet = train_mlp(SEED, Z1[j_tr], y1[j_tr], Z1[j_va], y1[j_va], w_pos=pw, epochs=200, patience=20, sw_valid=w1[j_va]); ff = logit_fn(fnet)
print('MLP day-2 weighted PR-AUC:', round(average_precision_score(y2, sig(ff(Z2)), sample_weight=w2), 3))
base_f = Z1[j_tr][y1[j_tr] == 0].mean(0); F2 = Z2[y2 == 1]; F1v = Z1[j_va][y1[j_va] == 1]
def grad_of(X):
    x = torch.tensor(np.asarray(X, dtype=np.float32), requires_grad=True); g, = torch.autograd.grad(fnet(x).sum(), x); return g.numpy()
def ig(X, m=64):
    tot = np.zeros_like(X)
    for a in (np.arange(m) + 0.5) / m: tot += grad_of(base_f + a * (X - base_f))
    return (X - base_f) * tot / m
def smoothgrad(X, n=50, s=0.15, seed=SEED):
    g = np.zeros_like(X); rg = np.random.default_rng(seed)
    for _ in range(n): g += grad_of(X + rg.normal(0, s, X.shape).astype(np.float32))
    return g / n"""
CAL = """ca = pd.read_csv(DATA + 'california_housing.csv')
CF = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
k_tr, k_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED); k_tr, k_va = train_test_split(k_tr, test_size=0.2, random_state=SEED)
cmu, csd = ca.iloc[k_tr][CF].mean(), ca.iloc[k_tr][CF].std(); CZ = ((ca[CF] - cmu) / csd).values.astype(np.float32)
cy = ca['MedHouseVal_100k_usd'].values.astype(np.float32)
from sklearn.linear_model import LinearRegression
lin = LinearRegression().fit(CZ[k_tr], cy[k_tr]); show('cal.r2_lin', 'linear regression test R2', round(float(lin.score(CZ[k_te], cy[k_te])), 3))
show('cal.n', 'block groups', int(len(ca)))"""
SAVE = "fig.savefig('{c}.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()"
SRC_H = "data/clean/hmeq.csv (Baesens, Roesch and Scheule, 2016)"
SRC_F = "data/clean/creditcard_fraud_sample.csv (ULB Machine Learning Group and Worldline, OpenML 1597)"
SRC_C = "data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)"
QS = []


def ql(chart, desc, kw, datafile, *codes):
    q = Quantlet(k=11, chart=chart, description=desc, keywords=kw, datafile=datafile)
    for c in codes:
        q.code(c)
    QS.append(q)


ql("shap_global", "Global importance (mean absolute SHAP value) of an MLP credit-scoring model (Kernel SHAP) and of a logistic regression (linear SHAP) on HMEQ home-equity loans.",
   "SHAP, Shapley values, global feature importance, credit scoring, explainable AI, MLP", SRC_H, CORE, HMEQ, KSHAP,
   "sv_lr = (Xsub[:NE] - BG.mean(0)) * lrm.coef_[0]\nimp = np.abs(sv).mean(0); imp_lr = np.abs(sv_lr).mean(0); top = np.argsort(-imp)[:10]\n"
   "print('top features (MLP):', [FEATS[j] for j in top[:3]])\nfig, ax = plt.subplots(figsize=(6.4, 2.7)); yy = np.arange(10)[::-1]\n"
   "ax.barh(yy + 0.2, imp[top], height=0.4, color=PAL['model'], label='MLP (Kernel SHAP)')\n"
   "ax.barh(yy - 0.2, imp_lr[top], height=0.4, color=PAL['benchmark'], label='logistic regression (linear SHAP)')\n"
   "ax.set_yticks(yy); ax.set_yticklabels([FEATS[j] for j in top], fontsize=9); ax.set_xlabel('Mean |SHAP value| (log-odds)')\n"
   "legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="shap_global"))
ql("shap_beeswarm", "Beeswarm of Kernel SHAP values of an MLP credit model on HMEQ: each dot is one applicant, colour = rank of the feature value.",
   "SHAP beeswarm, Kernel SHAP, local explanations, credit risk, explainable AI", SRC_H, CORE, HMEQ, KSHAP,
   "top = np.argsort(-np.abs(sv).mean(0))[:10][::-1]\nfig, ax = plt.subplots(figsize=(6.4, 2.9))\n"
   "for row, j in enumerate(top):\n    q = stats.rankdata(Xsub[:NE, j]) / NE\n"
   "    sc = ax.scatter(sv[:, j], row + (np.random.default_rng(j).random(NE) - 0.5) * 0.6, c=q, cmap='coolwarm', s=6, lw=0, vmin=0, vmax=1)\n"
   "ax.axvline(0, color='#8a8a8a', lw=0.6); ax.set_yticks(range(10)); ax.set_yticklabels([FEATS[j] for j in top], fontsize=9)\n"
   "ax.set_xlabel('SHAP value (log-odds of default)'); cb = fig.colorbar(sc, orientation='horizontal', pad=0.25, ticks=[0, 1]); cb.ax.set_xticklabels(['low value', 'high value'])\n"
   "fig.tight_layout()\n" + SAVE.format(c="shap_beeswarm"))
ql("shap_waterfall", "SHAP waterfall for one rejected HMEQ applicant: from the average log-odds of default to this applicant's score, feature by feature.",
   "SHAP waterfall, local explanation, adverse action, credit decision, explainable AI", SRC_H, CORE, HMEQ, KSHAP,
   "p = sig(f0(Xsub[:NE])); cand = np.where((p > 0.5) & (p < 0.8) & (yte[i_sub[:NE]] == 1))[0]\n"
   "k = int(cand[np.argmin(np.abs(p[cand] - 0.65))]) if len(cand) else int(np.argmax(p)); phi = sv[k]; o = np.argsort(-np.abs(phi))[:8]\n"
   "raw = XR.iloc[i_te[i_sub[k]]]; print('PD of this applicant (%):', round(100 * p[k], 1))\n"
   "vals = list(phi[o]) + [phi.sum() - phi[o].sum()]; labs = [f'{FEATS[j]} = {raw[FEATS[j]]:.1f}' for j in o] + ['other features']\n"
   "fig, ax = plt.subplots(figsize=(6.4, 2.8)); start = base\n"
   "for i, v in enumerate(vals):\n    ax.barh(len(vals) - 1 - i, v, left=start, color=PAL['positive'] if v > 0 else PAL['negative'], height=0.7); start += v\n"
   "ax.axvline(base, color='#8a8a8a', ls='--', lw=0.8); ax.axvline(start, color='#222222', lw=0.8)\n"
   "ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs[::-1], fontsize=8.5); ax.set_xlabel('Log-odds of default')\n"
   "ax.barh([0], [0], color=PAL['positive'], label='raises the risk'); ax.barh([0], [0], color=PAL['negative'], label='lowers the risk')\n"
   "legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="shap_waterfall"))
ql("shap_dependence", "SHAP dependence of the debt-to-income ratio (DEBTINC) in an MLP credit model on HMEQ, with median-imputed missing values marked.",
   "SHAP dependence plot, missing values, debt-to-income, credit scoring", SRC_H, CORE, HMEQ, KSHAP,
   "show('hmeq.debtinc_missing_pct', 'DEBTINC missing (%)', round(float(100 * h['DEBTINC'].isna().mean()), 1))\n"
   "show('hmeq.bad_if_debtinc_missing', 'default rate if DEBTINC missing (%)', round(float(100 * h.loc[h.DEBTINC.isna(), 'BAD'].mean()), 1))\n"
   "jd = FEATS.index('DEBTINC'); miss = Xsub[:NE, FEATS.index('DEBTINC_missing')] > 0; raw = Xsub[:NE, jd] * sdv['DEBTINC'] + mu['DEBTINC']\n"
   "fig, ax = plt.subplots(figsize=(6.4, 2.4)); ax.scatter(raw[~miss], sv[~miss, jd], s=7, color=PAL['data'], label='DEBTINC reported')\n"
   "ax.scatter(raw[miss], sv[miss, jd], s=9, color=PAL['positive'], label='DEBTINC missing (median imputed)'); ax.axhline(0, color='#8a8a8a', lw=0.6)\n"
   "ax.set_xlabel('DEBTINC (%)'); ax.set_ylabel('SHAP value of DEBTINC'); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="shap_dependence"))
LIME = """from lime.lime_tabular import LimeTabularExplainer
p = sig(f0(Xsub[:NE])); cand = np.where((p > 0.5) & (p < 0.8) & (yte[i_sub[:NE]] == 1))[0]
k = int(cand[np.argmin(np.abs(p[cand] - 0.65))]) if len(cand) else int(np.argmax(p)); x_app = Xsub[k]
def lime_w(x, n=5000, kw=None, seed=0):
    e = LimeTabularExplainer(Xtr, mode='regression', feature_names=FEATS, discretize_continuous=False, sample_around_instance=True,
                             random_state=seed, kernel_width=kw).explain_instance(x, f0, num_features=len(FEATS), num_samples=n)
    w = np.zeros(len(FEATS))
    for j, v in e.local_exp[0]: w[j] = v
    return w
show('lime.kw_default', 'default LIME kernel width 0.75 sqrt(d)', round(float(0.75 * np.sqrt(len(FEATS))), 2))"""
ql("lime_vs_shap", "LIME (local linear surrogate) against Kernel SHAP for the same HMEQ applicant: contribution of each feature to the log-odds of default.",
   "LIME, SHAP, local surrogate, explanation comparison, credit scoring", SRC_H, CORE, HMEQ, KSHAP, LIME,
   "w = lime_w(x_app); lc = w * x_app; phi = sv[k]; o = np.argsort(-np.abs(phi))[:8]; yy = np.arange(8)[::-1]\n"
   "print('Spearman |LIME| vs |SHAP|:', round(stats.spearmanr(np.abs(lc), np.abs(phi))[0], 2))\n"
   "fig, ax = plt.subplots(figsize=(6.4, 2.6)); ax.barh(yy + 0.2, phi[o], height=0.38, color=PAL['model'], label='Kernel SHAP')\n"
   "ax.barh(yy - 0.2, lc[o], height=0.38, color=PAL['alt'], label='LIME slope x standardised value'); ax.axvline(0, color='#8a8a8a', lw=0.6)\n"
   "ax.set_yticks(yy); ax.set_yticklabels([FEATS[j] for j in o], fontsize=9); ax.set_xlabel('Contribution to the log-odds of default')\n"
   "legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="lime_vs_shap"))
ql("lime_instability", "Instability of LIME: spread of the local slopes over repeated runs with 500 and 5,000 samples, and their dependence on the kernel width.",
   "LIME instability, kernel width, sampling noise, explanation stability", SRC_H, CORE, HMEQ, KSHAP, LIME,
   "W500 = np.array([lime_w(x_app, 500, seed=s) for s in range(15)]); W5k = np.array([lime_w(x_app, 5000, seed=s) for s in range(15)])\n"
   "top = np.argsort(-np.abs(W5k[0]))[:5]; kws = [0.5, 1, 2, 3, 4.12, 6, 10]; KW = np.array([lime_w(x_app, 5000, kw=k_) for k_ in kws])\n"
   "fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5)); pos = np.arange(5)\n"
   "axs[0].boxplot([W500[:, j] for j in top], positions=pos - 0.18, widths=0.3, showfliers=False)\n"
   "axs[0].boxplot([W5k[:, j] for j in top], positions=pos + 0.18, widths=0.3, showfliers=False)\n"
   "axs[0].set_xticks(pos); axs[0].set_xticklabels([FEATS[j] for j in top], fontsize=7, rotation=30); axs[0].set_ylabel('LIME slope')\n"
   "for i, j in enumerate(top): axs[1].plot(kws, KW[:, j], marker='o', ms=3, label=FEATS[j])\n"
   "axs[1].set_xscale('log'); axs[1].set_xlabel('kernel width'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="lime_instability"))
ql("seed_stability", "Rank of each feature by mean |SHAP| for MLP credit models retrained with different seeds (Rashomon effect): equal accuracy, different explanations.",
   "Rashomon effect, explanation stability, random seeds, SHAP ranking, Kendall tau", SRC_H, CORE, HMEQ,
   "R = []\nfor s in range(4):\n    n_ = train_mlp(SEED + s, Xtr, ytr, Xva, yva); f_ = logit_fn(n_); np.random.seed(SEED)\n"
   "    v = np.array(shap.KernelExplainer(f_, BG).shap_values(Xsub[:80], nsamples=300, silent=True)); R.append(stats.rankdata(-np.abs(v).mean(0)))\n"
   "    print('seed', SEED + s, 'AUC', round(roc_auc_score(yte, f_(Xte)), 3))\nR = np.array(R); o = np.argsort(R.mean(0))[:8]\n"
   "print('mean Kendall tau:', round(np.mean([stats.kendalltau(R[a], R[b])[0] for a in range(4) for b in range(a + 1, 4)]), 2))\n"
   "fig, ax = plt.subplots(figsize=(6.4, 2.6))\nfor s in range(4): ax.scatter(np.arange(8) + (s - 1.5) * 0.06, R[s, o], s=14, color=PAL['model'], label='MLP, one seed' if s == 0 else None)\n"
   "ax.set_xticks(range(8)); ax.set_xticklabels([FEATS[j] for j in o], fontsize=8, rotation=20); ax.set_ylabel('Rank by mean |SHAP|'); ax.invert_yaxis()\n"
   "legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="seed_stability"))
ql("method_agreement", "Spearman correlation between the global feature importances of saliency, gradient x input, SmoothGrad, integrated gradients and permutation importance for a fraud MLP.",
   "explanation agreement, saliency, integrated gradients, SmoothGrad, permutation importance, fraud detection", SRC_F, CORE, FRAUD,
   "A = {'saliency': np.abs(grad_of(F2)), 'gradient x input': grad_of(F2) * (F2 - base_f), 'SmoothGrad': smoothgrad(F2) * (F2 - base_f), 'integrated gradients': ig(F2)}\n"
   "G = pd.DataFrame({k_: np.abs(v).mean(0) for k_, v in A.items()}, index=FF)\nC = G.corr(method='spearman'); print(C.round(2))\n"
   "fig, ax = plt.subplots(figsize=(6.4, 2.9)); ax.imshow(C.values, cmap='Blues', vmin=0, vmax=1)\n"
   "for a in range(len(C)):\n    for b in range(len(C)): ax.text(b, a, f'{C.values[a, b]:.2f}', ha='center', va='center', fontsize=8)\n"
   "ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=30, ha='right', fontsize=8); ax.set_yticks(range(len(C))); ax.set_yticklabels(C.index, fontsize=8)\n"
   "fig.tight_layout()\n" + SAVE.format(c="method_agreement"))
ql("ig_completeness", "Completeness of integrated gradients for a fraud MLP: relative gap between the sum of attributions and f(x) - f(baseline) as a function of the number of Riemann steps.",
   "integrated gradients, completeness axiom, Riemann sum, attribution, fraud detection", SRC_F, CORE, FRAUD,
   "gap = ff(F2) - ff(base_f[None])[0]; ms = [2, 4, 8, 16, 32, 64, 128, 256]\nerr = [100 * np.median(np.abs(ig(F2, m).sum(1) - gap) / np.abs(gap)) for m in ms]\n"
   "print(dict(zip(ms, np.round(err, 4))))\nfig, ax = plt.subplots(figsize=(6.4, 2.3)); ax.loglog(ms, err, marker='o', color=PAL['model'], label='median relative completeness error')\n"
   "ax.set_xlabel('Riemann steps m'); ax.set_ylabel('Error (%)'); legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="ig_completeness"))
ql("fraud_local", "Attributions of saliency, gradient x input and integrated gradients for one fraudulent transaction of day 2 (median fraud score), normalised to shares of the total.",
   "local explanation, saliency, integrated gradients, fraud detection, credit-card transactions", SRC_F, CORE, FRAUD,
   "kf = int(np.argsort(sig(ff(F2)))[len(F2) // 2]); x = F2[kf:kf + 1]\n"
   "A = {'saliency': np.abs(grad_of(x))[0], 'gradient x input': (grad_of(x) * (x - base_f))[0], 'integrated gradients': ig(x)[0]}\n"
   "top = np.argsort(-np.abs(A['integrated gradients']))[:8]; yy = np.arange(8)[::-1]\nfig, ax = plt.subplots(figsize=(6.4, 2.6))\n"
   "for i, (k_, c) in enumerate(zip(A, (PAL['data'], '#6250d6', PAL['model']))):\n    v = A[k_] / np.abs(A[k_]).sum(); ax.barh(yy + (1 - i) * 0.27, v[top], height=0.27, color=c, label=k_)\n"
   "ax.set_yticks(yy); ax.set_yticklabels([FF[j] for j in top]); ax.axvline(0, color='#8a8a8a', lw=0.6); ax.set_xlabel('Share of total |attribution|')\n"
   "legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="fraud_local"))
ql("deletion_curve", "Faithfulness of explanations for a fraud MLP: mean fraud score after resetting the k top-ranked features to the baseline, for integrated gradients, saliency and a random order.",
   "deletion curve, faithfulness, AOPC, integrated gradients, saliency, fraud detection", SRC_F, CORE, FRAUD,
   "def deletion(Am, K=12, rand=False):\n    rg = np.random.default_rng(SEED); out = np.zeros(K + 1)\n    for i, x in enumerate(F2):\n"
   "        o = rg.permutation(len(x)) if rand else np.argsort(-np.abs(Am[i])); xs = np.repeat(x[None], K + 1, 0)\n"
   "        for k_ in range(1, K + 1): xs[k_:, o[k_ - 1]] = base_f[o[k_ - 1]]\n        out += sig(ff(xs))\n    return out / len(F2)\n"
   "D = {'integrated gradients': deletion(ig(F2)), 'saliency': deletion(np.abs(grad_of(F2))), 'random order': deletion(None, rand=True)}\n"
   "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nfor (k_, v), c in zip(D.items(), (PAL['model'], PAL['data'], '#8a8a8a')): ax.plot(range(13), 100 * v, marker='o', ms=3, color=c, label=k_)\n"
   "ax.set_xlabel('Top-ranked features reset to the baseline'); ax.set_ylabel('Mean fraud score (%)'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="deletion_curve"))
ql("local_robustness", "Local robustness of gradient explanations for a fraud MLP: relative change of the attribution vector per unit of a small input perturbation.",
   "local Lipschitz, explanation robustness, saliency, SmoothGrad, integrated gradients", SRC_F, CORE, FRAUD,
   "rg = np.random.default_rng(SEED); Xs = F2[:40]; res = {k_: [] for k_ in ('saliency', 'gradient x input', 'SmoothGrad', 'integrated gradients')}\n"
   "fA = {'saliency': lambda X: np.abs(grad_of(X)), 'gradient x input': lambda X: grad_of(X) * (X - base_f),\n"
   "      'SmoothGrad': lambda X: smoothgrad(X, n=30) * (X - base_f), 'integrated gradients': lambda X: ig(X, 32)}\n"
   "for _ in range(5):\n    Xn = Xs + rg.normal(0, 0.05, Xs.shape).astype(np.float32)\n    for k_, f_ in fA.items():\n"
   "        a0, a1 = f_(Xs), f_(Xn); res[k_].append(np.linalg.norm(a1 - a0, axis=1) / np.linalg.norm(Xn - Xs, axis=1) / (np.linalg.norm(a0, axis=1) + 1e-9))\n"
   "fig, ax = plt.subplots(figsize=(6.4, 2.3)); ax.boxplot([np.concatenate(v) for v in res.values()], vert=False, showfliers=False)\n"
   "ax.set_yticks(range(1, 5)); ax.set_yticklabels(list(res)); ax.set_xlabel('Relative change of the explanation per unit of input change')\n"
   "fig.tight_layout()\n" + SAVE.format(c="local_robustness"))
ql("time_shift", "Shift of explanations from day 1 to day 2 for a fraud MLP trained on day 1: share of the total |integrated gradients| by feature for held-out day-1 frauds and day-2 frauds.",
   "explanation drift, distribution shift, integrated gradients, forward-time validation, fraud detection", SRC_F, CORE, FRAUD,
   "show('shift.n1', 'held-out frauds, day 1', int(len(F1v))); show('shift.n2', 'test frauds, day 2', int(len(F2)))\n"
   "g1, g2 = np.abs(ig(F1v)).mean(0), np.abs(ig(F2)).mean(0); p1, p2 = g1 / g1.sum(), g2 / g2.sum()\n"
   "print('Spearman day 1 vs day 2:', round(stats.spearmanr(g1, g2)[0], 2)); top = np.argsort(-(g1 + g2))[:10]; xs = np.arange(10)\n"
   "fig, ax = plt.subplots(figsize=(6.4, 2.5)); ax.bar(xs - 0.2, 100 * p1[top], width=0.4, color=PAL['train'], label='day 1, held-out frauds')\n"
   "ax.bar(xs + 0.2, 100 * p2[top], width=0.4, color=PAL['test'], label='day 2, test frauds'); ax.set_xticks(xs); ax.set_xticklabels([FF[j] for j in top])\n"
   "ax.set_ylabel('Share of total |IG| (%)'); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="time_shift"))
ql("pdp_ice", "Partial dependence (PDP) and individual conditional expectation (ICE) of the predicted house value on median income, MLP on California housing.",
   "partial dependence, ICE curves, global explanation, MLP regression, California housing", SRC_C, CORE, CAL,
   "cnet = train_mlp(SEED, CZ[k_tr], cy[k_tr], CZ[k_va], cy[k_va], task='reg', epochs=200, patience=20, lr=2e-3); cf = logit_fn(cnet)\n"
   "grid = np.linspace(np.percentile(ca.MedInc, 2), np.percentile(ca.MedInc, 98), 40); gz = (grid - cmu['MedInc']) / csd['MedInc']\n"
   "rg = np.random.default_rng(SEED); rows = CZ[k_te][rg.choice(len(k_te), 60, replace=False)]; prow = CZ[k_te][rg.choice(len(k_te), 1000, replace=False)]\n"
   "def ice(R):\n    out = np.zeros((len(R), len(gz)))\n    for g_, z in enumerate(gz):\n        Xg = R.copy(); Xg[:, 0] = z; out[:, g_] = cf(Xg)\n    return out\n"
   "I = ice(rows); PD = ice(prow).mean(0); fig, ax = plt.subplots(figsize=(6.4, 2.4))\n"
   "for i, row in enumerate(I): ax.plot(grid, row, color=PAL['data'], alpha=0.25, lw=0.7, label='ICE: one block group' if i == 0 else None)\n"
   "ax.plot(grid, PD, color=PAL['model'], lw=2.2, label='partial dependence'); ax.set_xlabel('MedInc (tens of thousands of USD)'); ax.set_ylabel('Predicted value (100,000 USD)')\n"
   "legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="pdp_ice"))

if __name__ == "__main__":
    only = sys.argv[1:]
    bad_all = []
    for q in QS:
        if only and q.chart not in only:
            continue
        q.write(execute=True)
        n, bad = check_numbers(q, NUM)
        print(f"  {q.name}: {n} numbers checked", "OK" if not bad else bad)
        bad_all += bad
        if n == 0:
            bad_all.append(f"{q.name}: no number checked")
    print("QUANTLETS", "OK" if not bad_all else "FAILED", len(QS))
    sys.exit(1 if bad_all else 0)
