"""
make_notebook_L1.py -- Lecture 1 notebooks (EN + RO): the lecture notebook (reproduces the numbers and charts of the slides,
sections § = "Experiment yourself" pointers) and the seminar notebook (student version + solutions), all executed.
Run:  python3 lectures/L1/make_notebook_L1.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")   # macOS: PyTorch and XGBoost ship different OpenMP runtimes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Deep learning for business: from linear models to the perceptron and the MLP",
         "Deep learning pentru afaceri: de la modelele liniare la perceptron și la MLP")

HELPERS = r'''import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
torch.set_num_threads(1)
NUM = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']

def encode_hmeq(tr, others):
    """median fill + missing indicators + one-hot, fitted on the training part only (no leakage)"""
    med = tr[NUM].median()
    def f(d):
        x = pd.DataFrame(index=d.index)
        for c in NUM:
            x[c] = d[c].fillna(med[c]); x['miss_' + c] = d[c].isna().astype(float)
        for c, lv in (('REASON', ['DebtCon', 'HomeImp']), ('JOB', ['Mgr', 'Office', 'Other', 'ProfExe', 'Sales', 'Self'])):
            for v in lv:
                x[f'{c}_{v}'] = (d[c] == v).astype(float)
        return x
    xtr = f(tr); mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    return [((xtr - mu) / sd).values.astype(np.float32)] + [((f(o) - mu) / sd).values.astype(np.float32) for o in others], list(xtr.columns)

def mlp(d, hidden=(64, 64), act=nn.ReLU):
    layers, k = [], d
    for h in hidden:
        layers += [nn.Linear(k, h), act()]; k = h
    return nn.Sequential(*layers, nn.Linear(k, 1))

def train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), lr=1e-3, epochs=300, bs=128, patience=25, seed=0):
    """Adam on mini-batches; keeps the weights of the best validation epoch"""
    torch.manual_seed(SEED + seed); rng = np.random.default_rng(SEED + seed)
    m = mlp(Xtr.shape[1], hidden); opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        m.train(); idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; opt.zero_grad(); lf(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lf(m(Xv), yv))
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state); m.eval()
    return m

def proba(m, X):
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy().ravel()'''

HMEQ = r'''h = pd.read_csv(DATA + 'hmeq.csv'); yh = h.pop('BAD').values.astype(int)
idx = np.arange(len(h))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=yh, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=yh[i_tr], random_state=SEED)
(Xtr, Xva, Xte), cols = encode_hmeq(h.iloc[i_tr], [h.iloc[i_va], h.iloc[i_te]])
ytr, yva, yte = yh[i_tr], yh[i_va], yh[i_te]
print('train / validation / test:', len(i_tr), len(i_va), len(i_te), '| inputs d =', Xtr.shape[1])'''


def lecture(nb):
    T = nb.T
    nb.md(T("On macOS, if the kernel stops when XGBoost and PyTorch run in the same session, start Jupyter with "
            "`OMP_NUM_THREADS=1 jupyter lab` (the two libraries ship different OpenMP runtimes). Colab is not affected.",
            "Pe macOS, dacă nucleul se oprește cînd XGBoost și PyTorch rulează în aceeași sesiune, porniți Jupyter cu "
            "`OMP_NUM_THREADS=1 jupyter lab` (cele două biblioteci folosesc versiuni diferite ale OpenMP). Colab nu este afectat."))
    nb.code("import xgboost as xgb\n" + HELPERS)
    nb.md(T("## §1 The data: default rates by applicant group", "## §1 Datele: ratele de neplată pe grupuri de solicitanți"))
    nb.code("g = pd.read_csv(DATA + 'german_credit.csv')\n"
            "print(len(g), 'applicants, default rate %:', round(100 * g['default'].mean(), 1))\n"
            "print((100 * g.groupby('checking_status')['default'].mean()).round(1))\n"
            "hm = pd.read_csv(DATA + 'hmeq.csv')\n"
            "d = hm['DELINQ']; cat = np.where(d.isna(), 'missing', np.where(d >= 3, '3+', d.fillna(0).astype(int).astype(str)))\n"
            "print((100 * hm.groupby(cat)['BAD'].mean()).round(1))\n"
            "fig, ax = plt.subplots(1, 2, figsize=(9, 3))\n"
            "(100 * g.groupby('checking_status')['default'].mean()).plot.bar(ax=ax[0], color='#e34948'); ax[0].set_ylabel('default rate (%)')\n"
            "(100 * hm.groupby(cat)['BAD'].mean()).plot.bar(ax=ax[1], color='#e34948'); plt.tight_layout(); plt.show()")
    nb.md(T("## §2 Logistic regression as one neuron (HMEQ)", "## §2 Regresia logistică drept un neuron (HMEQ)"))
    nb.code(HMEQ)
    nb.code("import statsmodels.api as sm\n"
            "base = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC', 'miss_DEBTINC']\n"
            "ci = [cols.index(c) for c in base]\n"
            "lm = sm.Logit(ytr, sm.add_constant(Xtr[:, ci].astype(float))).fit(disp=0)\n"
            "print(pd.DataFrame({'coef': lm.params[1:], 'odds ratio': np.exp(lm.params[1:])}, index=base).round(3))\n"
            "print('McFadden pseudo-R2:', round(lm.prsquared, 3))\n"
            "print('default rate when DEBTINC is missing / present (%):',\n"
            "      round(100 * yh[h['DEBTINC'].isna().values].mean(), 1), round(100 * yh[h['DEBTINC'].notna().values].mean(), 1))")
    nb.md(T("Forward pass of one real test applicant: the score is the sum of the contributions $w_jx_j$.",
            "Propagarea înainte pentru un solicitant real din setul de testare: scorul este suma contribuțiilor $w_jx_j$."))
    nb.code("j = next(k for k in np.where(yte == 1)[0] if h.iloc[i_te[k]][base[:-1]].notna().all())\n"
            "x = Xte[j, ci].astype(float); contrib = lm.params[1:] * x\n"
            "z = lm.params[0] + contrib.sum()\n"
            "print(pd.Series(contrib, index=base).sort_values(key=abs, ascending=False).round(2).head(3))\n"
            "print('score z =', round(z, 2), ' PD =', round(100 / (1 + np.exp(-z)), 1), '%')")
    nb.md(T("## §3 Perceptron convergence on real loan data", "## §3 Convergența perceptronului pe date reale de credit"))
    nb.code("rng = np.random.default_rng(SEED); s = 2 * ytr - 1; X = Xtr.astype(float)\n"
            "w, b, err = np.zeros(X.shape[1]), 0.0, []\n"
            "for ep in range(60):\n"
            "    for i in rng.permutation(len(X)):\n"
            "        if s[i] * (X[i] @ w + b) <= 0:\n"
            "            w += s[i] * X[i]; b += s[i]\n"
            "    err.append(np.mean(np.sign(X @ w + b + 1e-12) != s))\n"
            "lr = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr)\n"
            "print('perceptron error % min / max:', round(100 * min(err), 1), round(100 * max(err), 1))\n"
            "print('logistic regression error %:', round(100 * np.mean((lr.predict_proba(Xtr)[:, 1] > 0.5) != ytr), 1))\n"
            "plt.figure(figsize=(8, 3)); plt.plot(100 * np.array(err)); plt.ylabel('training error (%)'); plt.xlabel('epoch'); plt.show()")
    nb.md(T("## §4 The MLP: universal approximation on California housing", "## §4 MLP: aproximarea universală pe datele California"))
    nb.code("ca = pd.read_csv(DATA + 'california_housing.csv')\n"
            "x1, y1 = ca['Latitude'].values.astype(np.float32), ca['MedHouseVal_100k_usd'].values.astype(np.float32)\n"
            "keep = y1 < 5.0; x1, y1 = x1[keep], y1[keep]           # drop the block groups at the source cap\n"
            "a, bb = train_test_split(np.arange(len(x1)), test_size=0.25, random_state=SEED)\n"
            "mx, sx = x1[a].mean(), x1[a].std()\n"
            "for H in (1, 3, 50):\n"
            "    torch.manual_seed(SEED + H); m = mlp(1, (H,), nn.Sigmoid); opt = torch.optim.Adam(m.parameters(), lr=0.02)\n"
            "    Xa, Ya = torch.tensor((x1[a, None] - mx) / sx), torch.tensor(y1[a, None])\n"
            "    for it in range(4000):\n"
            "        opt.zero_grad(); nn.MSELoss()(m(Xa), Ya).backward(); opt.step()\n"
            "    with torch.no_grad():\n"
            "        mse = float(nn.MSELoss()(m(torch.tensor((x1[bb, None] - mx) / sx)), torch.tensor(y1[bb, None])))\n"
            "    print('H =', H, ' held-out MSE =', round(mse, 3))\n"
            "bl = np.polyfit(x1[a], y1[a], 1); print('linear held-out MSE =', round(np.mean((np.polyval(bl, x1[bb]) - y1[bb]) ** 2), 3))")
    nb.md(T("§ Experiment yourself: replace the sigmoid by `nn.ReLU` and compare the shapes of the fitted curves.",
            "§ Exercițiu: înlocuiți sigmoida cu `nn.ReLU` și comparați formele curbelor estimate."))
    nb.md(T("## §5 Case study: logistic regression, MLP and XGBoost on HMEQ", "## §5 Studiu de caz: regresia logistică, MLP și XGBoost pe HMEQ"))
    nb.code("p_lr = lr.predict_proba(Xte)[:, 1]\n"
            "xg = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,\n"
            "                       eval_metric='logloss', early_stopping_rounds=50, random_state=SEED, n_jobs=1)\n"
            "xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False); p_xg = xg.predict_proba(Xte)[:, 1]\n"
            "aucs = [roc_auc_score(yte, proba(train_mlp(Xtr, ytr, Xva, yva, seed=s), Xte)) for s in range(5)]\n"
            "print('test AUC  logistic', round(roc_auc_score(yte, p_lr), 3), '| XGBoost', round(roc_auc_score(yte, p_xg), 3),\n"
            "      '| MLP 64-64 mean of 5 seeds', round(np.mean(aucs), 3), '(sd', round(np.std(aucs, ddof=1), 3), ')')")
    nb.code("m0 = train_mlp(Xtr, ytr, Xva, yva, seed=0); p_ml = proba(m0, Xte)\n"
            "rng = np.random.default_rng(SEED); diffs = []\n"
            "for _ in range(2000):\n"
            "    b = rng.integers(0, len(yte), len(yte))\n"
            "    if yte[b].min() == yte[b].max():\n"
            "        continue\n"
            "    diffs.append(roc_auc_score(yte[b], p_ml[b]) - roc_auc_score(yte[b], p_lr[b]))\n"
            "print('AUC gain MLP - logistic:', round(roc_auc_score(yte, p_ml) - roc_auc_score(yte, p_lr), 3),\n"
            "      ' 95% bootstrap interval:', np.round(np.percentile(diffs, [2.5, 97.5]), 3))\n"
            "for nm, p in (('logistic', p_lr), ('MLP', p_ml), ('XGBoost', p_xg)):\n"
            "    fpr, tpr, _ = roc_curve(yte, p); plt.plot(fpr, tpr, label=nm)\n"
            "    print(nm, 'Brier', round(brier_score_loss(yte, p), 4), ' defaults caught at 10% FPR:', round(100 * np.interp(0.1, fpr, tpr), 1))\n"
            "plt.plot([0, 1], [0, 1], ':'); plt.xlabel('false positive rate'); plt.ylabel('true positive rate'); plt.legend(); plt.show()")
    nb.md(T("§ Experiment yourself: vary the width of one hidden layer, `hidden=(H,)` with H from 1 to 256, and plot the test AUC "
            "against H (the slide uses 5 seeds per width).",
            "§ Exercițiu: variați lățimea unui strat ascuns, `hidden=(H,)` cu H de la 1 la 256, și reprezentați AUC pe test în funcție "
            "de H (slide-ul folosește 5 inițializări pentru fiecare lățime)."))
    nb.md(T("## §6 German credit: cross-validation, a corrected t-test and the cost-optimal cut-off",
            "## §6 German credit: validare încrucișată, testul t corectat și pragul optim pentru cost"))
    nb.code("from scipy import stats\n"
            "yg = g['default'].values; Xg = g.drop(columns='default')\n"
            "def enc(tr, others):\n"
            "    cat = [c for c in tr.columns if tr[c].dtype == object]; num = [c for c in tr.columns if c not in cat]\n"
            "    lv = {c: sorted(tr[c].unique()) for c in cat}\n"
            "    f = lambda d: pd.concat([d[num].astype(float)] + [(d[c] == v).astype(float).rename(f'{c}_{v}') for c in cat for v in lv[c][1:]], axis=1)\n"
            "    xt = f(tr); mu, sd = xt.mean(), xt.std().replace(0, 1)\n"
            "    return [((xt - mu) / sd).values.astype(np.float32)] + [((f(o) - mu) / sd).values.astype(np.float32) for o in others]\n"
            "A = {'logit': [], 'mlp': []}; oof = {'logit': np.zeros(len(yg)), 'mlp': np.zeros(len(yg))}\n"
            "for rep in range(5):\n"
            "    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + rep).split(Xg, yg)):\n"
            "        a_tr, a_va = train_test_split(a, test_size=0.2, stratify=yg[a], random_state=SEED + rep)\n"
            "        xa, xv, xb = enc(Xg.iloc[a_tr], [Xg.iloc[a_va], Xg.iloc[b]])\n"
            "        pl = LogisticRegression(C=1.0, max_iter=5000).fit(xa, yg[a_tr]).predict_proba(xb)[:, 1]\n"
            "        pm = proba(train_mlp(xa, yg[a_tr], xv, yg[a_va], seed=rep * 10 + f, bs=64), xb)\n"
            "        A['logit'].append(roc_auc_score(yg[b], pl)); A['mlp'].append(roc_auc_score(yg[b], pm))\n"
            "        if rep == 0:\n"
            "            oof['logit'][b], oof['mlp'][b] = pl, pm\n"
            "d = np.array(A['mlp']) - np.array(A['logit']); J = len(d)\n"
            "t = d.mean() / np.sqrt((1 / J + 0.25) * d.var(ddof=1))      # Nadeau-Bengio correction, n_test/n_train = 1/4\n"
            "print('mean AUC logistic', round(np.mean(A['logit']), 3), '| MLP', round(np.mean(A['mlp']), 3))\n"
            "print('corrected t =', round(t, 2), ' p =', round(2 * stats.t.sf(abs(t), J - 1), 3))")
    nb.code("cost = lambda p, thr: (5 * np.sum((p < thr) & (yg == 1)) + 1 * np.sum((p >= thr) & (yg == 0))) / len(yg)\n"
            "for k in ('logit', 'mlp'):\n"
            "    print(k, ' cost per applicant at 1/6:', round(cost(oof[k], 1 / 6), 3), ' at 1/2:', round(cost(oof[k], 0.5), 3))\n"
            "print('accept everyone:', round(5 * yg.mean(), 2))")
    nb.md(T("## §7 A minimal MLP in PyTorch", "## §7 Un MLP minimal în PyTorch"))
    nb.code("model = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))\n"
            "print(model)\n"
            "print('parameters:', sum(p.numel() for p in model.parameters()), '(logistic regression:', Xtr.shape[1] + 1, ')')")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(HELPERS)
    nb.code(HMEQ)
    nb.task(T("Fit a logistic regression on the HMEQ training set and report the test AUC.",
              "Estimați o regresie logistică pe setul de antrenare HMEQ și raportați AUC pe setul de testare."))
    nb.solution("lr = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr)\nprint(round(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]), 3))")
    nb.task(T("Count the parameters of an MLP with two hidden layers of 32 units for these inputs, by formula and with PyTorch.",
              "Numărați parametrii unui MLP cu două straturi ascunse de 32 de neuroni pentru aceste intrări, cu formula și cu PyTorch."))
    nb.solution("d = Xtr.shape[1]\nprint((d + 1) * 32 + 33 * 32 + 33, sum(p.numel() for p in mlp(d, (32, 32)).parameters()))")
    nb.task(T("Train an MLP 32-32 with 3 seeds and report the mean and the standard deviation of the test AUC.",
              "Antrenați un MLP 32-32 cu 3 inițializări și raportați media și abaterea standard a AUC pe test."))
    nb.solution("a = [roc_auc_score(yte, proba(train_mlp(Xtr, ytr, Xva, yva, hidden=(32, 32), seed=s), Xte)) for s in range(3)]\n"
                "print(np.round(np.mean(a), 3), np.round(np.std(a, ddof=1), 3))")
    nb.task(T("Replace ReLU by tanh in the hidden layers. Does the test AUC change by more than the seed spread?",
              "Înlocuiți ReLU cu tanh în straturile ascunse. Se schimbă AUC pe test cu mai mult decît variația dintre inițializări?"))
    nb.solution("def train_act(act, seed):\n"
                "    torch.manual_seed(SEED + seed); m = mlp(Xtr.shape[1], (32, 32), act); opt = torch.optim.Adam(m.parameters(), 1e-3)\n"
                "    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)\n"
                "    for ep in range(40):\n"
                "        for i in range(0, len(Xt), 128):\n"
                "            opt.zero_grad(); nn.BCEWithLogitsLoss()(m(Xt[i:i + 128]), yt[i:i + 128]).backward(); opt.step()\n"
                "    return roc_auc_score(yte, proba(m, Xte))\n"
                "for act in (nn.ReLU, nn.Tanh):\n"
                "    a = [train_act(act, s) for s in range(3)]; print(act.__name__, np.round(np.mean(a), 3), np.round(np.std(a, ddof=1), 3))")
    nb.task(T("A missed default costs 5 and a rejected good loan costs 1. Compute the test cost per applicant of the logistic model at the "
              "cut-offs 1/2 and 1/6.",
              "O neplată ratată costă 5, iar un credit bun respins costă 1. Calculați costul pe solicitant al modelului logistic pe setul de "
              "testare, la pragurile 1/2 și 1/6."))
    nb.solution("p = lr.predict_proba(Xte)[:, 1]\n"
                "for thr in (0.5, 1 / 6):\n"
                "    print(round(thr, 3), round((5 * np.sum((p < thr) & (yte == 1)) + np.sum((p >= thr) & (yte == 0))) / len(yte), 3))")
    nb.task(T("Drop the missing-value indicators from the inputs. How much test AUC does the logistic model lose? Explain why.",
              "Eliminați indicatorii de valori lipsă din intrări. Cît pierde modelul logistic din AUC pe test? Explicați de ce."))
    nb.solution("keep = [i for i, c in enumerate(cols) if not c.startswith('miss_')]\n"
                "lr2 = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr[:, keep], ytr)\n"
                "print(round(roc_auc_score(yte, lr2.predict_proba(Xte[:, keep])[:, 1]), 3))")


if __name__ == "__main__":
    Notebook.build_all(1, TITLE, lecture, kind="lecture")
    Notebook.build_all(1, TITLE, seminar, kind="seminar")
