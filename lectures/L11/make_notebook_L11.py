"""
make_notebook_L11.py -- Lecture 11 notebooks (EN + RO): lecture notebook (executed) and seminar notebooks.
The lecture notebook reproduces the methods of the deck on the same real data (HMEQ, ULB credit-card fraud, California
housing) with smaller explanation budgets; the seminar applies them to German credit. XGBoost is left out of the
notebooks: on macOS it cannot share a process with PyTorch (the deck computes it in a separate process).
Run:  python3 lectures/L11/make_notebook_L11.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Explainable AI: SHAP, LIME, gradient methods and the stability of explanations",
         "Inteligența artificială explicabilă: SHAP, LIME, metode bazate pe gradient și stabilitatea explicațiilor")

MLP = """import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import shap
from lime.lime_tabular import LimeTabularExplainer
torch.set_num_threads(4)
class MLP(nn.Module):
    def __init__(self, d, h1=64, h2=32, p=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(p), nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p), nn.Linear(h2, 1))
    def forward(self, x): return self.f(x).squeeze(-1)
def train_mlp(seed, Xa, ya, Xv, yv, w_pos=1.0, epochs=300, patience=25, task='clf', lr=1e-3):
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
        sc = (roc_auc_score(yv, pv) if w_pos == 1.0 else average_precision_score(yv, pv)) if task == 'clf' else -np.mean((pv - yv) ** 2)
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
JOBS = ['Mgr', 'Office', 'Other', 'ProfExe', 'Sales', 'Self']
y_all = h['BAD'].values; idx = np.arange(len(h))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=y_all[i_tr], random_state=SEED)
med = h.iloc[i_tr][NUM].median()
XR = pd.DataFrame({c: h[c].fillna(med[c]) for c in NUM})
XR['DEBTINC_missing'] = h['DEBTINC'].isna().astype(float); XR['REASON_HomeImp'] = (h['REASON'] == 'HomeImp').astype(float)
for j in JOBS: XR['JOB_' + j] = (h['JOB'] == j).astype(float)
FEATS = list(XR.columns); mu, sd = XR.iloc[i_tr].mean(), XR.iloc[i_tr].std()
XS = ((XR - mu) / sd).values.astype(np.float32)
Xtr, Xva, Xte = XS[i_tr], XS[i_va], XS[i_te]; ytr, yva, yte = y_all[i_tr], y_all[i_va], y_all[i_te]
print(len(h), 'loans;', round(100 * y_all.mean(), 1), '% defaults;', len(FEATS), 'features')"""


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 Credit data (HMEQ) and the models to explain", "## §1 Datele de credit (HMEQ) și modelele de explicat"))
    nb.code(MLP)
    nb.code(HMEQ)
    nb.code("net = train_mlp(SEED, Xtr, ytr, Xva, yva); f0 = logit_fn(net)\nlr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)\n"
            "for name, p in (('MLP', sig(f0(Xte))), ('logistic', lr.predict_proba(Xte)[:, 1])):\n"
            "    print(f'{name:9s} AUC {roc_auc_score(yte, p):.3f}  PR-AUC {average_precision_score(yte, p):.3f}')")
    nb.md(T("## §2 Kernel SHAP: global importance, beeswarm, one applicant",
            "## §2 Kernel SHAP: importanța globală, diagrama roi, un solicitant"))
    nb.code("rng = np.random.default_rng(SEED)\nBG = Xtr[rng.choice(len(Xtr), 100, replace=False)]\ni_sub = rng.choice(len(Xte), 400, replace=False); Xsub = Xte[i_sub]\n"
            "np.random.seed(SEED); ex = shap.KernelExplainer(f0, BG)\nsv = np.array(ex.shap_values(Xsub[:150], nsamples=400, silent=True)); base = f0(BG).mean()\n"
            "print('additivity error:', np.abs(sv.sum(1) + base - f0(Xsub[:150])).max())\n"
            "imp = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False); print(imp.head(8).round(3))\n"
            "sv_lr = (Xsub[:150] - BG.mean(0)) * lr.coef_[0]   # linear SHAP of the logistic regression\n"
            "print('linear SHAP top features:', list(pd.Series(np.abs(sv_lr).mean(0), index=FEATS).sort_values(ascending=False).index[:5]))")
    nb.code("top = imp.index[:8][::-1]\nfig, ax = plt.subplots(figsize=(6.4, 2.9))\n"
            "for row, f in enumerate(top):\n    j = FEATS.index(f); q = Xsub[:150, j].argsort().argsort() / 150\n"
            "    sc = ax.scatter(sv[:, j], row + (np.random.default_rng(j).random(150) - 0.5) * 0.6, c=q, cmap='coolwarm', s=6)\n"
            "ax.set_yticks(range(len(top))); ax.set_yticklabels(top); ax.set_xlabel('SHAP value (log-odds)'); plt.colorbar(sc, orientation='horizontal', pad=0.25)\n"
            "fig.tight_layout(); plt.show()")
    nb.code("p_sub = sig(f0(Xsub[:150]))\ncand = np.where((p_sub > 0.5) & (p_sub < 0.8) & (yte[i_sub[:150]] == 1))[0]\n"
            "k = int(cand[np.argmin(np.abs(p_sub[cand] - 0.65))]) if len(cand) else int(np.argmax(p_sub)); x_app = Xsub[k]\n"
            "phi = pd.Series(sv[k], index=FEATS); print('PD', round(100 * p_sub[k], 1), '%; base log-odds', round(base, 3))\n"
            "print(phi.reindex(phi.abs().sort_values(ascending=False).index).head(6).round(3))")
    nb.md(T("## §3 LIME and its sensitivity to its own settings", "## §3 LIME și sensibilitatea la propriile setări"))
    nb.code("def lime_w(x, n=5000, kw=None, seed=0):\n"
            "    e = LimeTabularExplainer(Xtr, mode='regression', feature_names=FEATS, discretize_continuous=False, sample_around_instance=True,\n"
            "                             random_state=seed, kernel_width=kw).explain_instance(x, f0, num_features=len(FEATS), num_samples=n)\n"
            "    w = np.zeros(len(FEATS))\n    for j, v in e.local_exp[1]: w[j] = v\n    return w\n"
            "W500 = np.array([lime_w(x_app, 500, seed=s) for s in range(15)]); W5k = np.array([lime_w(x_app, 5000, seed=s) for s in range(15)])\n"
            "top1 = lambda W: np.mean([np.argmax(np.abs(w)) == np.argmax(np.abs(W5k[0])) for w in W])\n"
            "print('same top feature, 500 samples:', top1(W500), '| 5,000 samples:', top1(W5k))\n"
            "for kw in (0.5, 1, 2, 4.12, 10):\n    w = lime_w(x_app, kw=kw); print('kernel width', kw, '-> top feature', FEATS[int(np.argmax(np.abs(w)))], round(w.max(), 3))")
    nb.md(T("## §4 A counterfactual explanation", "## §4 O explicație contrafactuală"))
    nb.code("ACTION = ['LOAN', 'DEBTINC', 'NINQ', 'DELINQ', 'DEROG', 'CLNO', 'MORTDUE']\n"
            "mad = (XR.iloc[i_tr][NUM] - XR.iloc[i_tr][NUM].median()).abs().median().replace(0, np.nan).fillna(XR.iloc[i_tr][NUM].std())\n"
            "m = torch.tensor([1.0 if f in ACTION else 0.0 for f in FEATS]); scale = torch.tensor([sd[f] / mad[f] if f in ACTION else 0.0 for f in FEATS], dtype=torch.float32)\n"
            "x0 = torch.tensor(x_app); x = x0.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=0.01)\n"
            "for _ in range(3000):\n    xx = x0 + m * (x - x0); loss = 10 * net(xx[None])[0] ** 2 + (scale * (xx - x0).abs()).sum()\n"
            "    opt.zero_grad(); loss.backward(); opt.step()\n"
            "xc = (x0 + m * (x - x0)).detach().numpy(); raw0, raw1 = x_app * sd.values + mu.values, xc * sd.values + mu.values\n"
            "print('PD before', round(100 * sig(f0(x_app[None])[0]), 1), '% -> after', round(100 * sig(f0(xc[None])[0]), 1), '%')\n"
            "print(pd.DataFrame({'before': raw0, 'after': raw1}, index=FEATS).loc[ACTION].round(1))")
    nb.md(T("## §5 Fraud: saliency, gradient x input, SmoothGrad, integrated gradients",
            "## §5 Fraude: saliența, gradient x intrare, SmoothGrad, gradienții integrați"))
    nb.code("fr = pd.read_csv(DATA + 'creditcard_fraud_sample.csv'); fr['logAmount'] = np.log1p(fr['Amount'])\n"
            "FF = [f'V{i}' for i in range(1, 29)] + ['logAmount']; d1, d2 = fr[fr.Time < 86400], fr[fr.Time >= 86400]\n"
            "j_tr, j_va = train_test_split(np.arange(len(d1)), test_size=0.25, stratify=d1['Class'], random_state=SEED)\n"
            "fmu, fsd = d1.iloc[j_tr][FF].mean(), d1.iloc[j_tr][FF].std()\n"
            "Z1 = ((d1[FF] - fmu) / fsd).values.astype(np.float32); Z2 = ((d2[FF] - fmu) / fsd).values.astype(np.float32)\n"
            "y1, y2, w1, w2 = d1['Class'].values, d2['Class'].values, d1['weight'].values, d2['weight'].values\n"
            "pw = float((w1[j_tr] * (1 - y1[j_tr])).sum() / (w1[j_tr] * y1[j_tr]).sum()) ** 0.5\n"
            "fnet = train_mlp(SEED, Z1[j_tr], y1[j_tr], Z1[j_va], y1[j_va], w_pos=pw, epochs=200, patience=20); ff = logit_fn(fnet)\n"
            "print('day 2 weighted PR-AUC:', round(average_precision_score(y2, ff(Z2), sample_weight=w2), 3))\n"
            "base_f = Z1[j_tr][y1[j_tr] == 0].mean(0); F2 = Z2[y2 == 1]\n"
            "def grad_of(X):\n    x = torch.tensor(np.asarray(X, dtype=np.float32), requires_grad=True); g, = torch.autograd.grad(fnet(x).sum(), x); return g.numpy()\n"
            "def ig(X, m=64):\n    tot = np.zeros_like(X)\n    for a in (np.arange(m) + 0.5) / m: tot += grad_of(base_f + a * (X - base_f))\n    return (X - base_f) * tot / m\n"
            "gap = ff(F2) - ff(base_f[None])[0]\nfor m_ in (4, 16, 64, 256):\n"
            "    print('IG steps', m_, 'median completeness error %:', round(100 * np.median(np.abs(ig(F2, m_).sum(1) - gap) / np.abs(gap)), 3))")
    nb.md(T("## §6 Agreement between methods, faithfulness (deletion) and local robustness",
            "## §6 Concordanța metodelor, fidelitatea (eliminarea) și robustețea locală"))
    nb.code("from scipy import stats\nA = {'saliency': np.abs(grad_of(F2)), 'gradxinput': grad_of(F2) * (F2 - base_f), 'ig': ig(F2)}\n"
            "G = pd.DataFrame({k: np.abs(v).mean(0) for k, v in A.items()}, index=FF)\nprint(G.corr(method='spearman').round(2))\n"
            "def deletion(Am, K=12):\n    out = np.zeros(K + 1)\n    for i, x in enumerate(F2):\n        o = np.argsort(-np.abs(Am[i])); xs = np.repeat(x[None], K + 1, 0)\n"
            "        for k in range(1, K + 1): xs[k:, o[k - 1]] = base_f[o[k - 1]]\n        out += sig(ff(xs))\n    return out / len(F2)\n"
            "for k_, v in A.items(): print(k_, 'score after resetting 3 features:', round(100 * deletion(v)[3], 1), '%')")
    nb.md(T("## §7 Stability: three training seeds and the shift from day 1 to day 2",
            "## §7 Stabilitatea: trei valori inițiale și deplasarea de la ziua 1 la ziua 2"))
    nb.code("ranks = []\nfor s in range(3):\n    n_ = train_mlp(SEED + s, Xtr, ytr, Xva, yva); e_ = shap.KernelExplainer(logit_fn(n_), BG)\n"
            "    v = np.array(e_.shap_values(Xsub[:80], nsamples=300, silent=True)); ranks.append(stats.rankdata(-np.abs(v).mean(0)))\n"
            "    print('seed', SEED + s, 'AUC', round(roc_auc_score(yte, logit_fn(n_)(Xte)), 3), 'top-3', [FEATS[j] for j in np.argsort(ranks[-1])[:3]])\n"
            "print('Kendall tau seed 1 vs 2:', round(stats.kendalltau(ranks[0], ranks[1])[0], 2))\n"
            "F1v = Z1[j_va][y1[j_va] == 1]; g1, g2 = np.abs(ig(F1v)).mean(0), np.abs(A['ig']).mean(0)\n"
            "print('Spearman day 1 vs day 2 (IG importances):', round(stats.spearmanr(g1, g2)[0], 2))")
    nb.md(T("## §8 Partial dependence and ICE on California housing", "## §8 Dependența parțială și ICE pe locuințele din California"))
    nb.code("ca = pd.read_csv(DATA + 'california_housing.csv'); CF = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']\n"
            "k_tr, k_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED); k_tr, k_va = train_test_split(k_tr, test_size=0.2, random_state=SEED)\n"
            "cmu, csd = ca.iloc[k_tr][CF].mean(), ca.iloc[k_tr][CF].std(); CZ = ((ca[CF] - cmu) / csd).values.astype(np.float32)\n"
            "cy = ca['MedHouseVal_100k_usd'].values.astype(np.float32)\n"
            "cnet = train_mlp(SEED, CZ[k_tr], cy[k_tr], CZ[k_va], cy[k_va], task='reg', epochs=200, patience=20, lr=2e-3); cf = logit_fn(cnet)\n"
            "grid = np.linspace(np.percentile(ca.MedInc, 2), np.percentile(ca.MedInc, 98), 40); gz = (grid - cmu['MedInc']) / csd['MedInc']\n"
            "rows = CZ[k_te][np.random.default_rng(SEED).choice(len(k_te), 60, replace=False)]\n"
            "ice = np.array([[cf(np.c_[np.full(1, z), r[None, 1:]])[0] for z in gz] for r in rows])\n"
            "fig, ax = plt.subplots(figsize=(6.4, 2.4)); ax.plot(grid, ice.T, color='#2a78d6', alpha=0.25, lw=0.7); ax.plot(grid, ice.mean(0), color='#eb6834', lw=2.2)\n"
            "ax.set_xlabel('MedInc'); ax.set_ylabel('predicted value (100,000 USD)'); fig.tight_layout(); plt.show()")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup: German credit (Statlog), 1,000 applicants", "## Pregătire: German credit (Statlog), 1.000 de solicitanți"))
    nb.code(MLP)
    nb.code("g = pd.read_csv(DATA + 'german_credit.csv'); y = g['default'].values\n"
            "X = pd.get_dummies(g.drop(columns='default'), drop_first=True).astype(float); FEATS = list(X.columns)\n"
            "idx = np.arange(len(g)); i_tr, i_te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)\n"
            "i_tr, i_va = train_test_split(i_tr, test_size=0.2, stratify=y[i_tr], random_state=SEED)\n"
            "mu, sd = X.iloc[i_tr].mean(), X.iloc[i_tr].std().replace(0, 1)\nXS = ((X - mu) / sd).values.astype(np.float32)\n"
            "Xtr, Xva, Xte, ytr, yva, yte = XS[i_tr], XS[i_va], XS[i_te], y[i_tr], y[i_va], y[i_te]\nprint(len(g), 'applicants,', len(FEATS), 'features')")
    nb.task(T("Train the MLP (seed 20261108) and a logistic regression; report the test AUC and PR-AUC of both.",
              "Antrenați MLP-ul (valoarea inițială 20261108) și o regresie logistică; raportați AUC și PR-AUC pe test pentru ambele."))
    nb.solution("net = train_mlp(SEED, Xtr, ytr, Xva, yva); f0 = logit_fn(net); lr = LogisticRegression(C=0.5, max_iter=3000).fit(Xtr, ytr)\n"
                "for n_, p in (('MLP', sig(f0(Xte))), ('logistic', lr.predict_proba(Xte)[:, 1])):\n"
                "    print(n_, round(roc_auc_score(yte, p), 3), round(average_precision_score(yte, p), 3))")
    nb.task(T("Compute Kernel SHAP values of the MLP for 100 test applicants (100 background applicants, 400 samples). Check additivity and list the 8 most important features.",
              "Calculați valorile Kernel SHAP ale MLP-ului pentru 100 de solicitanți din test (100 de solicitanți de referință, 400 de eșantioane). Verificați aditivitatea și enumerați cele mai importante 8 variabile."))
    nb.solution("rng = np.random.default_rng(SEED); BG = Xtr[rng.choice(len(Xtr), 100, replace=False)]; Xs = Xte[:100]\n"
                "np.random.seed(SEED); sv = np.array(shap.KernelExplainer(f0, BG).shap_values(Xs, nsamples=400, silent=True))\n"
                "print('additivity error', np.abs(sv.sum(1) + f0(BG).mean() - f0(Xs)).max())\n"
                "imp = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False); print(imp.head(8).round(3))")
    nb.task(T("Fairness check: how large are the SHAP contributions of age_years and of the personal_status_sex dummies compared with the top feature? What would you tell a regulator?",
              "Verificarea echității: cît de mari sînt contribuțiile SHAP ale variabilei age_years și ale variabilelor personal_status_sex comparativ cu variabila de top? Ce i-ați spune unui reglementator?"))
    nb.solution("prot = [f for f in FEATS if f.startswith('personal_status_sex') or f == 'age_years']\nprint(imp[prot].round(3)); print('top feature', imp.index[0], round(imp.iloc[0], 3))")
    nb.task(T("Explain the riskiest test applicant with LIME 20 times (500 samples, seeds 0-19). How often is the top feature the same?",
              "Explicați cu LIME de 20 de ori solicitantul cu cel mai mare risc din test (500 de eșantioane, valori inițiale 0-19). Cît de des este aceeași variabila de top?"))
    nb.solution("x = Xte[np.argmax(f0(Xte))]\ndef lime_w(x, n, seed):\n"
                "    e = LimeTabularExplainer(Xtr, mode='regression', discretize_continuous=False, sample_around_instance=True, random_state=seed).explain_instance(x, f0, num_features=len(FEATS), num_samples=n)\n"
                "    w = np.zeros(len(FEATS))\n    for j, v in e.local_exp[1]: w[j] = v\n    return w\n"
                "tops = [np.argmax(np.abs(lime_w(x, 500, s))) for s in range(20)]\nprint(pd.Series([FEATS[t] for t in tops]).value_counts())")
    nb.task(T("Integrated gradients for the same applicant (baseline = mean good applicant of the training set, 128 steps). Verify completeness.",
              "Gradienții integrați pentru același solicitant (referința = solicitantul bun mediu din setul de antrenare, 128 de pași). Verificați completitudinea."))
    nb.solution("base = Xtr[ytr == 0].mean(0)\ndef grad_of(X):\n    t = torch.tensor(np.asarray(X, dtype=np.float32), requires_grad=True); g_, = torch.autograd.grad(net(t).sum(), t); return g_.numpy()\n"
                "al = (np.arange(128) + 0.5) / 128; igv = (x - base) * np.mean([grad_of((base + a * (x - base))[None])[0] for a in al], 0)\n"
                "print('sum IG', round(igv.sum(), 4), '| f(x) - f(baseline)', round(f0(x[None])[0] - f0(base[None])[0], 4))\n"
                "print(pd.Series(igv, index=FEATS).abs().sort_values(ascending=False).head(5).round(3))")
    nb.task(T("Rashomon check: retrain the MLP with three more seeds. Are the test AUCs close? Do the top-5 SHAP features stay the same (Kendall tau of the importances)?",
              "Verificarea efectului Rashomon: reantrenați MLP-ul cu încă trei valori inițiale. Sînt AUC-urile pe test apropiate? Rămîn aceleași primele 5 variabile SHAP (tau Kendall al importanțelor)?"))
    nb.solution("from scipy import stats\nI = [np.abs(sv).mean(0)]\nfor s in (1, 2, 3):\n    n_ = train_mlp(SEED + s, Xtr, ytr, Xva, yva); fs = logit_fn(n_)\n"
                "    np.random.seed(SEED); v = np.array(shap.KernelExplainer(fs, BG).shap_values(Xs, nsamples=400, silent=True)); I.append(np.abs(v).mean(0))\n"
                "    print('seed', SEED + s, 'AUC', round(roc_auc_score(yte, fs(Xte)), 3), [FEATS[j] for j in np.argsort(-I[-1])[:5]])\n"
                "print('mean Kendall tau:', round(np.mean([stats.kendalltau(I[a], I[b])[0] for a in range(4) for b in range(a + 1, 4)]), 2))")


if __name__ == "__main__":
    Notebook.build_all(11, TITLE, lecture, kind="lecture")
    Notebook.build_all(11, TITLE, seminar, kind="seminar")
