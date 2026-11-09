"""
make_notebook_L3.py -- Lecture 3 notebooks (EN + RO): lecture notebook (sections § = "Experiment yourself" pointers) and seminar
notebook (student version + solutions), all executed. Run:  python3 lectures/L3/make_notebook_L3.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Regularisation and generalisation", "Regularizare și generalizare")

HELP = r'''import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
torch.set_num_threads(1)

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
        layers.append(nn.Linear(k, 1)); self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def fit(Xtr, ytr, Xva, yva, hidden=(128, 128), p=0.0, bn=False, wd=0.0, lr=1e-3, epochs=200, bs=128, seed=0,
        early=True, patience=20, loss='bce', opt_name='adam', record=False):
    """AdamW or SGD with momentum; dropout, batch norm, weight decay, early stopping on the validation loss"""
    torch.manual_seed(SEED + seed); rng = np.random.default_rng(SEED + seed)
    m = Net(Xtr.shape[1], hidden, p, bn)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd) if opt_name == 'adam' else torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss() if loss == 'bce' else nn.MSELoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    best, state, bad, hist, best_ep = np.inf, None, 0, [], 0
    for ep in range(epochs):
        m.train(); idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            if len(b) < 2 and bn:
                continue
            opt.zero_grad(); lf(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lf(m(Xv), yv)); tl = float(lf(m(Xt), yt)) if record else np.nan
        hist.append((tl, vl))
        if not np.isfinite(vl):
            break
        if vl < best - 1e-5:
            best, bad, best_ep = vl, 0, ep + 1; state = {k: v.clone() for k, v in m.state_dict().items()}
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

g = pd.read_csv(DATA + 'german_credit.csv'); yg = g['default'].values.astype(int); Xg_raw = g.drop(columns='default')
def encode_german(tr, others):
    cat = [c for c in tr.columns if tr[c].dtype == object]; num = [c for c in tr.columns if c not in cat]
    levels = {c: sorted(tr[c].unique()) for c in cat}
    def f(d):
        x = d[num].astype(float).copy()
        for c in cat:
            for lv in levels[c][1:]:
                x[f'{c}_{lv}'] = (d[c] == lv).astype(float)
        return x
    xtr = f(tr); mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    return [((xtr - mu) / sd).values.astype(np.float32)] + [((f(o) - mu) / sd).values.astype(np.float32) for o in others]
g_tr, g_te = train_test_split(np.arange(len(g)), test_size=0.2, stratify=yg, random_state=SEED)
g_tr, g_va = train_test_split(g_tr, test_size=0.25, stratify=yg[g_tr], random_state=SEED)
Gtr, Gva, Gte = encode_german(Xg_raw.iloc[g_tr], [Xg_raw.iloc[g_va], Xg_raw.iloc[g_te]])

fr = pd.read_csv(DATA + 'creditcard_fraud_sample.csv').sort_values('Time').reset_index(drop=True)
Xf0 = np.c_[fr[[f'V{i}' for i in range(1, 29)]].values, np.log1p(fr['Amount'].values)].astype(np.float32)
yf = fr['Class'].values.astype(int); wf = fr['weight'].values; amt = fr['Amount'].values
c1, c2 = np.quantile(fr['Time'], [0.6, 0.8])
f_tr = (fr['Time'] <= c1).values; f_va = ((fr['Time'] > c1) & (fr['Time'] <= c2)).values; f_te = (fr['Time'] > c2).values
Xf = (Xf0 - Xf0[f_tr].mean(0)) / Xf0[f_tr].std(0)
ap = lambda y, p, w: float(average_precision_score(y, p, sample_weight=w))
print('German credit train/val/test:', len(g_tr), len(g_va), len(g_te), '| fraud train/val/test:', f_tr.sum(), f_va.sum(), f_te.sum(),
      '| frauds in test:', yf[f_te].sum())'''


def lecture(nb):
    T = nb.T
    nb.code(HELP)
    nb.md(T("## §1 Overfitting on 600 applicants", "## §1 Supraînvățarea pe 600 de solicitanți"))
    nb.code("_, h, _ = fit(Gtr, yg[g_tr], Gva, yg[g_va], hidden=(256, 256), epochs=150, early=False, record=True, bs=64)\n"
            "h = np.array(h); print('best validation epoch', int(np.argmin(h[:, 1]) + 1), '| val loss min', round(h[:, 1].min(), 3), '| after 150 epochs', round(h[-1, 1], 3))\n"
            "plt.plot(h[:, 0], label='train'); plt.plot(h[:, 1], label='validation'); plt.ylim(0, 1.6); plt.legend(); plt.xlabel('epoch'); plt.show()")
    nb.md(T("## §2 Double descent with random ReLU features", "## §2 Dubla coborîre cu caracteristici ReLU aleatoare"))
    nb.code("ca = pd.read_csv(DATA + 'california_housing.csv'); Xc = ca.drop(columns='MedHouseVal_100k_usd').values; yc = ca['MedHouseVal_100k_usd'].values\n"
            "rng = np.random.default_rng(SEED); idx = rng.permutation(len(Xc)); tr, te = idx[:400], idx[400:5400]\n"
            "mu, sd = Xc[tr].mean(0), Xc[tr].std(0); A, B = (Xc[tr] - mu) / sd, (Xc[te] - mu) / sd\n"
            "Wr = rng.normal(0, 1 / np.sqrt(8), (8, 8000)); br = rng.normal(0, 1, 8000)\n"
            "for p in (50, 200, 400, 800, 8000):\n"
            "    Fa, Fb = np.maximum(A @ Wr[:, :p] + br[:p], 0), np.maximum(B @ Wr[:, :p] + br[:p], 0); yt = yc[tr] - yc[tr].mean()\n"
            "    beta = np.linalg.lstsq(Fa, yt, rcond=None)[0] if p <= 400 else Fa.T @ np.linalg.lstsq(Fa @ Fa.T, yt, rcond=None)[0]\n"
            "    print(p, 'test MSE', round(np.mean((Fb @ beta + yc[tr].mean() - yc[te]) ** 2), 3))")
    nb.md(T("## §3 L1 against L2 penalties", "## §3 Penalizarea L1 față de L2"))
    nb.code("def penal_net(kind, lam, epochs=300):\n"
            "    torch.manual_seed(SEED); m = Net(Gtr.shape[1], (64,)); opt = torch.optim.Adam(m.parameters(), lr=1e-3)\n"
            "    Xt, yt = torch.tensor(Gtr), torch.tensor(yg[g_tr], dtype=torch.float32).view(-1, 1)\n"
            "    for ep in range(epochs):\n"
            "        opt.zero_grad(); W = m.net[0].weight\n"
            "        (nn.BCEWithLogitsLoss()(m(Xt), yt) + lam * (W.abs().sum() if kind == 'l1' else (W ** 2).sum())).backward(); opt.step()\n"
            "    return m.eval()\n"
            "for kind, lam in (('l1', 1e-3), ('l2', 1e-2)):\n"
            "    w = penal_net(kind, lam).net[0].weight.detach().numpy().ravel(); print(kind, 'share of |w| < 0.01:', round(100 * np.mean(np.abs(w) < 0.01), 1), '%')")
    nb.md(T("## §4 Weight decay and dropout under cross-validation", "## §4 Penalizarea ponderilor și dropout în validarea încrucișată"))
    nb.code("FD = []\n"
            "for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xg_raw, yg):\n"
            "    a_tr, a_va = train_test_split(a, test_size=0.2, stratify=yg[a], random_state=SEED)\n"
            "    xa, xv, xb = encode_german(Xg_raw.iloc[a_tr], [Xg_raw.iloc[a_va], Xg_raw.iloc[b]]); FD.append((xa, yg[a_tr], xv, yg[a_va], xb, yg[b]))\n"
            "run = lambda f, **kw: roc_auc_score(FD[f][5], proba(fit(*FD[f][:4], hidden=(256, 256), epochs=150, bs=64, early=False, seed=f, **kw)[0], FD[f][4]))\n"
            "for kw in ({}, {'wd': 1.0}, {'p': 0.5}):\n"
            "    print(kw, 'mean held-out AUC', round(np.mean([run(f, **kw) for f in range(5)]), 3))\n"
            "print('logistic', round(np.mean([roc_auc_score(FD[f][5], LogisticRegression(C=1.0, max_iter=5000).fit(FD[f][0], FD[f][1]).predict_proba(FD[f][4])[:, 1]) for f in range(5)]), 3))")
    nb.md(T("§ Experiment yourself: replace `hidden=(256, 256)` by `hidden=(16,)` and compare with logistic regression.",
            "§ Exercițiu: înlocuiți `hidden=(256, 256)` cu `hidden=(16,)` și comparați cu regresia logistică."))
    nb.md(T("## §5 Early stopping on card fraud", "## §5 Oprirea timpurie la fraudele cu carduri"))
    nb.code("_, h, best = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=120, early=False, record=True, bs=256)\n"
            "h = np.array(h); print('best validation epoch', int(np.argmin(h[:, 1]) + 1), '| validation loss there', round(h[:, 1].min(), 4), '| at the end', round(h[-1, 1], 4))\n"
            "plt.semilogy(h[:, 0], label='train'); plt.semilogy(h[:, 1], label='validation'); plt.legend(); plt.xlabel('epoch'); plt.show()")
    nb.md(T("## §6 Batch normalisation and the learning rate", "## §6 Normalizarea pe loturi și rata de învățare"))
    nb.code("ci_tr, ci_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED); ci_tr, ci_va = train_test_split(ci_tr, test_size=0.25, random_state=SEED)\n"
            "Cz = ((Xc - Xc[ci_tr].mean(0)) / Xc[ci_tr].std(0)).astype(np.float32); y32 = yc.astype(np.float32)\n"
            "for bn in (False, True):\n"
            "    _, h, _ = fit(Cz[ci_tr], y32[ci_tr], Cz[ci_va], y32[ci_va], hidden=(64,) * 6, bn=bn, lr=0.1, epochs=40, bs=128, early=False, loss='mse', opt_name='sgd')\n"
            "    v = [x[1] for x in h]; print('batch norm' if bn else 'no batch norm', '| epochs run', len(v), '| best validation MSE', round(np.nanmin(v), 3))")
    nb.md(T("## §7 Variability of a single hold-out split", "## §7 Variabilitatea unei singure împărțiri"))
    nb.code("aucs = []\n"
            "for s in range(200):\n"
            "    a, b = train_test_split(np.arange(len(yg)), test_size=0.3, stratify=yg, random_state=SEED + s); xa, xb = encode_german(Xg_raw.iloc[a], [Xg_raw.iloc[b]])\n"
            "    aucs.append(roc_auc_score(yg[b], LogisticRegression(C=1.0, max_iter=5000).fit(xa, yg[a]).predict_proba(xb)[:, 1]))\n"
            "print('AUC range', round(min(aucs), 3), '-', round(max(aucs), 3), '| 90% interval', np.round(np.percentile(aucs, [5, 95]), 3))\n"
            "plt.hist(aucs, bins=30); plt.xlabel('test AUC'); plt.show()")
    nb.md(T("## §8 Validation design and leakage", "## §8 Schema de validare și scurgerea de informație"))
    nb.code("res = {'time': [], 'random': [], 'oversample_first': []}\n"
            "for s in range(3):\n"
            "    m, _, _ = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(128, 128), epochs=40, bs=256, seed=s); res['time'].append(ap(yf[f_te], proba(m, Xf[f_te]), wf[f_te]))\n"
            "    idx = np.arange(len(yf)); a, b = train_test_split(idx, test_size=0.2, stratify=yf, random_state=SEED + s); a, v = train_test_split(a, test_size=0.25, stratify=yf[a], random_state=SEED + s)\n"
            "    Z = (Xf0 - Xf0[a].mean(0)) / Xf0[a].std(0); m, _, _ = fit(Z[a], yf[a], Z[v], yf[v], hidden=(128, 128), epochs=40, bs=256, seed=s); res['random'].append(ap(yf[b], proba(m, Z[b]), wf[b]))\n"
            "    rng = np.random.default_rng(SEED + s); big = np.r_[idx, np.repeat(np.where(yf == 1)[0], 19)]; rng.shuffle(big); yb = yf[big]\n"
            "    a, b = train_test_split(np.arange(len(big)), test_size=0.2, stratify=yb, random_state=SEED + s); a, v = train_test_split(a, test_size=0.25, stratify=yb[a], random_state=SEED + s)\n"
            "    Xb = Xf0[big]; Z = (Xb - Xb[a].mean(0)) / Xb[a].std(0); m, _, _ = fit(Z[a], yb[a], Z[v], yb[v], hidden=(128, 128), epochs=40, bs=256, seed=s)\n"
            "    res['oversample_first'].append(ap(yb[b], proba(m, Z[b]), np.where(yb == 1, 1 / 20, wf[big])[b]))\n"
            "print({k: round(np.mean(v), 3) for k, v in res.items()})")
    nb.md(T("## §9 The fraud data: rare and clustered", "## §9 Datele despre fraude: rare și grupate"))
    nb.code("hour = (fr['Time'] // 3600).astype(int); rate = pd.Series(yf).groupby(hour).sum() / pd.Series(wf).groupby(hour).sum()\n"
            "print('fraud rate in the full data %:', round(100 * yf.sum() / wf.sum(), 3), '| highest hourly rate %:', round(100 * rate.max(), 2))\n"
            "(100 * rate).plot.bar(figsize=(10, 3)); plt.ylabel('fraud rate (%)'); plt.show()")
    nb.md(T("## §10 Case study: regularised MLP, logistic regression and XGBoost on fraud", "## §10 Studiu de caz: MLP regularizat, regresie logistică și XGBoost pentru fraudă"))
    nb.code("import xgboost as xgb   # on macOS start Jupyter with OMP_NUM_THREADS=1 if the kernel stops here\n"
            "p_lr = LogisticRegression(max_iter=5000).fit(Xf[f_tr], yf[f_tr]).predict_proba(Xf[f_te])[:, 1]\n"
            "p_ml = np.mean([proba(fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=60, bs=256, seed=s, wd=0.1, p=0.3)[0], Xf[f_te]) for s in range(5)], axis=0)\n"
            "xgm = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='aucpr',\n"
            "                        early_stopping_rounds=100, random_state=SEED, n_jobs=1)\n"
            "xgm.fit(Xf[f_tr], yf[f_tr], eval_set=[(Xf[f_va], yf[f_va])], verbose=False); p_xg = xgm.predict_proba(Xf[f_te])[:, 1]\n"
            "for nm, p in (('logistic', p_lr), ('MLP', p_ml), ('XGBoost', p_xg)):\n"
            "    print(nm, 'test PR-AUC', round(ap(yf[f_te], p, wf[f_te]), 3))\n"
            "    pr, rc, _ = precision_recall_curve(yf[f_te], p, sample_weight=wf[f_te]); plt.plot(rc, pr, label=nm)\n"
            "plt.xlabel('recall'); plt.ylabel('precision'); plt.legend(); plt.show()")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(HELP)
    nb.task(T("Train an MLP 256-256 on German credit for 100 epochs without regularisation. At which epoch is the validation loss lowest?",
              "Antrenați un MLP 256-256 pe German credit timp de 100 de epoci, fără regularizare. La ce epocă este pierderea de validare minimă?"))
    nb.solution("_, h, _ = fit(Gtr, yg[g_tr], Gva, yg[g_va], hidden=(256, 256), epochs=100, early=False, record=True, bs=64)\nprint(int(np.argmin([x[1] for x in h]) + 1))")
    nb.task(T("Repeat with early stopping (patience 10) and report the test AUC of the restored weights.",
              "Repetați cu oprire timpurie (răbdare 10) și raportați AUC pe test pentru ponderile restaurate."))
    nb.solution("m, _, be = fit(Gtr, yg[g_tr], Gva, yg[g_va], hidden=(256, 256), epochs=100, patience=10, bs=64)\nprint(be, round(roc_auc_score(yg[g_te], proba(m, Gte)), 3))")
    nb.task(T("Compare a small MLP (one hidden layer of 16 units, early stopping) with logistic regression on the German test set.",
              "Comparați un MLP mic (un strat ascuns de 16 neuroni, oprire timpurie) cu regresia logistică pe setul de test German credit."))
    nb.solution("m, _, _ = fit(Gtr, yg[g_tr], Gva, yg[g_va], hidden=(16,), epochs=200, bs=64)\n"
                "print('MLP', round(roc_auc_score(yg[g_te], proba(m, Gte)), 3), '| logistic', round(roc_auc_score(yg[g_te], LogisticRegression(C=1.0, max_iter=5000).fit(Gtr, yg[g_tr]).predict_proba(Gte)[:, 1]), 3))")
    nb.task(T("On the fraud data, compute the test accuracy of the rule 'never fraud' with the population weights. Why is accuracy useless here?",
              "Pe datele despre fraude, calculați acuratețea pe test a regulii „niciodată fraudă”, cu ponderile populației. De ce este acuratețea inutilă aici?"))
    nb.solution("print(round(100 * np.sum(wf[f_te] * (yf[f_te] == 0)) / wf[f_te].sum(), 3), '% accuracy without catching a single fraud')")
    nb.task(T("Train the MLP 128-128 with early stopping on the time-ordered fraud split and report the precision at 80 % recall (population-weighted).",
              "Antrenați MLP 128-128 cu oprire timpurie pe împărțirea cronologică a fraudelor și raportați precizia la o sensibilitate de 80 % (ponderat la populație)."))
    nb.solution("m, _, _ = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(128, 128), epochs=40, bs=256)\n"
                "pr, rc, _ = precision_recall_curve(yf[f_te], proba(m, Xf[f_te]), sample_weight=wf[f_te])\nprint(round(100 * pr[rc >= 0.8].max(), 1))")
    nb.task(T("Correct the probabilities of that model for the 10 % undersampling of legitimate payments with $p=\\beta p_s/(\\beta p_s-p_s+1)$, $\\beta=0.1$, "
              "and compare the mean predicted fraud probability with the true fraud rate.",
              "Corectați probabilitățile acestui model pentru subeșantionarea de 10 % a plăților legitime cu $p=\\beta p_s/(\\beta p_s-p_s+1)$, $\\beta=0{,}1$, "
              "și comparați probabilitatea medie de fraudă prezisă cu rata reală a fraudei."))
    nb.solution("ps = proba(m, Xf[f_te]); pc = 0.1 * ps / (0.1 * ps - ps + 1)\n"
                "w = wf[f_te]; print(round(100 * np.sum(w * ps) / w.sum(), 3), round(100 * np.sum(w * pc) / w.sum(), 3), round(100 * np.sum(w * yf[f_te]) / w.sum(), 3))")


if __name__ == "__main__":
    Notebook.build_all(3, TITLE, lecture, kind="lecture")
    Notebook.build_all(3, TITLE, seminar, kind="seminar")
