"""
make_quantlets_L3.py -- one Quantlet per chart of Lecture 3: Quantlets/L3/NN_L3_<chart>/ (Metainfo.txt, executed notebook, PNG).
Every number printed with show() is compared with L3_numbers.json.
Run:  python3 lectures/L3/make_quantlets_L3.py  [chart ...]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

BASE = r'''import xgboost as xgb            # import before torch (OpenMP runtimes on macOS)
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
torch.set_num_threads(1)
SEED = 20261108

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
    """AdamW (decoupled weight decay) or SGD with momentum; dropout, batch norm, early stopping on the validation loss"""
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
'''
GERMAN = r'''
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
gi = np.arange(len(g))
g_tr, g_te = train_test_split(gi, test_size=0.2, stratify=yg, random_state=SEED)
g_tr, g_va = train_test_split(g_tr, test_size=0.25, stratify=yg[g_tr], random_state=SEED)
Gtr, Gva, Gte = encode_german(Xg_raw.iloc[g_tr], [Xg_raw.iloc[g_va], Xg_raw.iloc[g_te]])
'''
FRAUD = r'''
fr = pd.read_csv(DATA + 'creditcard_fraud_sample.csv').sort_values('Time').reset_index(drop=True)
Xf0 = np.c_[fr[[f'V{i}' for i in range(1, 29)]].values, np.log1p(fr['Amount'].values)].astype(np.float32)
yf = fr['Class'].values.astype(int); wf = fr['weight'].values; amt = fr['Amount'].values
c1, c2 = np.quantile(fr['Time'], [0.6, 0.8])
f_tr = (fr['Time'] <= c1).values; f_va = ((fr['Time'] > c1) & (fr['Time'] <= c2)).values; f_te = (fr['Time'] > c2).values
Xf = (Xf0 - Xf0[f_tr].mean(0)) / Xf0[f_tr].std(0)
ap = lambda y, p, w: float(average_precision_score(y, p, sample_weight=w))
'''
CAL = r'''
ca = pd.read_csv(DATA + 'california_housing.csv'); feat = [c for c in ca.columns if c != 'MedHouseVal_100k_usd']
Xc = ca[feat].values.astype(np.float64); yc = ca['MedHouseVal_100k_usd'].values.astype(np.float64)
'''

Q = {}
Q["overfit_german"] = dict(
    desc="Training and validation cross-entropy of an MLP 256-256 on 600 German credit applicants without regularisation and with dropout 0.5 plus weight decay, against logistic regression: overfitting within a few epochs.",
    kw="overfitting, generalisation gap, training curve, validation loss, dropout, weight decay, German credit",
    data="data/clean/german_credit.csv (UCI Statlog German credit)",
    code=BASE + GERMAN + r'''m_o, h_o, _ = fit(Gtr, yg[g_tr], Gva, yg[g_va], hidden=(256, 256), epochs=150, early=False, record=True, bs=64)
m_r, h_r, _ = fit(Gtr, yg[g_tr], Gva, yg[g_va], hidden=(256, 256), p=0.5, wd=0.1, epochs=150, early=False, record=True, bs=64)
ho, hr = np.array(h_o), np.array(h_r)
pl = LogisticRegression(C=1.0, max_iter=5000).fit(Gtr, yg[g_tr]).predict_proba(Gva)[:, 1]
ll = float(-np.mean(yg[g_va] * np.log(pl) + (1 - yg[g_va]) * np.log(1 - pl)))
show('overfit.best_ep', 'best validation epoch', int(np.argmin(ho[:, 1]) + 1)); show('overfit.val_min', 'best validation loss', round(float(ho[:, 1].min()), 3))
show('overfit.val_last', 'validation loss after 150 epochs', round(float(ho[-1, 1]), 3)); show('overfit.logit_val', 'logistic validation loss', round(ll, 3))
fig, ax = plt.subplots(figsize=(6.4, 2.6)); e = np.arange(1, len(ho) + 1)
ax.plot(e, ho[:, 0], color=PAL['train'], label='Training loss, no regularisation'); ax.plot(e, ho[:, 1], color=PAL['valid'], label='Validation loss, no regularisation')
ax.plot(e, hr[:, 1], color=PAL['valid'], ls='--', label='Validation loss, dropout 0.5 + weight decay')
ax.axhline(ll, color=PAL['benchmark'], lw=1.1, label='Logistic regression (validation)'); ax.axvline(int(np.argmin(ho[:, 1]) + 1), color='#8a8a8a', lw=0.8, ls=':')
ax.set_ylim(0, 1.6); ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig('overfit_german.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["double_descent"] = dict(
    desc="Double descent on California housing: training and test MSE of a two-layer network with p random, frozen ReLU features and a minimum-norm output layer, n = 400 training block groups, median of 10 draws.",
    kw="double descent, interpolation threshold, bias-variance trade-off, random features, minimum-norm least squares, California housing",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code="SEED = 20261108\n" + CAL + r'''NDD = 400; P = [10, 25, 50, 100, 200, 300, 350, 380, 400, 420, 450, 500, 600, 800, 1200, 2000, 4000, 8000]
out = {p: [] for p in P}
for s in range(10):
    rng = np.random.default_rng(SEED + s); idx = rng.permutation(len(Xc)); tr, te = idx[:NDD], idx[NDD:NDD + 5000]
    mu, sd = Xc[tr].mean(0), Xc[tr].std(0); A, B = (Xc[tr] - mu) / sd, (Xc[te] - mu) / sd
    Wr = rng.normal(0, 1 / np.sqrt(A.shape[1]), (A.shape[1], max(P))); br = rng.normal(0, 1, max(P))
    for p in P:
        Fa, Fb = np.maximum(A @ Wr[:, :p] + br[:p], 0), np.maximum(B @ Wr[:, :p] + br[:p], 0); yt = yc[tr] - yc[tr].mean()
        beta = np.linalg.lstsq(Fa, yt, rcond=None)[0] if p <= NDD else Fa.T @ np.linalg.lstsq(Fa @ Fa.T, yt, rcond=None)[0]
        out[p].append([np.mean((Fa @ beta + yc[tr].mean() - yc[tr]) ** 2), np.mean((Fb @ beta + yc[tr].mean() - yc[te]) ** 2)])
trm = np.array([np.median([x[0] for x in out[p]]) for p in P]); tem = np.array([np.median([x[1] for x in out[p]]) for p in P])
show('dd.peak_p', 'width of the test-error peak', int(P[int(np.argmax(tem))])); show('dd.best_under_mse', 'best test MSE below the threshold', round(float(min(tem[[i for i, p in enumerate(P) if p < NDD]])), 3))
show('dd.big_mse', 'test MSE at p = 8000', round(float(tem[-1]), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.plot(P, tem, 'o-', color=PAL['test'], ms=3.5, label='Test MSE (median of 10 draws)'); ax.plot(P, trm, 'o-', color=PAL['train'], ms=3.5, label='Training MSE')
ax.axvline(NDD, color='#8a8a8a', ls='--', lw=1, label='interpolation threshold p = n = 400'); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.01, 1e3)
ax.set_xlabel('Hidden units p (random ReLU features, log scale)'); ax.set_ylabel('MSE (log scale)')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('double_descent.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["l1_l2_weights"] = dict(
    desc="First-layer weights of an MLP with 64 hidden units on German credit trained with an L1 penalty against an L2 penalty: L1 piles the weights at zero.",
    kw="L1 penalty, L2 penalty, lasso, ridge, sparsity, weight decay, German credit",
    data="data/clean/german_credit.csv (UCI Statlog German credit)",
    code=BASE + GERMAN + r'''def penal_net(kind, lam, epochs=300):
    torch.manual_seed(SEED); m = Net(Gtr.shape[1], (64,)); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(Gtr), torch.tensor(yg[g_tr], dtype=torch.float32).view(-1, 1)
    for ep in range(epochs):
        opt.zero_grad(); W = m.net[0].weight
        (nn.BCEWithLogitsLoss()(m(Xt), yt) + lam * (W.abs().sum() if kind == 'l1' else (W ** 2).sum())).backward(); opt.step()
    m.eval(); return m
m1, m2 = penal_net('l1', 1e-3), penal_net('l2', 1e-2)
w1, w2 = m1.net[0].weight.detach().numpy().ravel(), m2.net[0].weight.detach().numpy().ravel()
show('l1l2.share_small_l1_pct', 'share of |w| < 0.01 with L1 (%)', round(float(100 * np.mean(np.abs(w1) < 0.01)), 1))
show('l1l2.share_small_l2_pct', 'share of |w| < 0.01 with L2 (%)', round(float(100 * np.mean(np.abs(w2) < 0.01)), 1))
fig, ax = plt.subplots(figsize=(6.4, 2.4)); bins = np.linspace(-0.6, 0.6, 81)
ax.hist(w2, bins=bins, color=PAL['data'], alpha=0.55, label='L2 penalty'); ax.hist(w1, bins=bins, color=PAL['model'], alpha=0.55, label='L1 penalty')
ax.set_yscale('log'); ax.set_xlabel('First-layer weight w'); ax.set_ylabel('Count (log scale)')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('l1_l2_weights.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["wd_dropout"] = dict(
    desc="Held-out AUC (5-fold cross-validation) of an MLP 256-256 on German credit for a grid of weight-decay strengths and dropout rates, against logistic regression.",
    kw="weight decay, dropout, regularisation strength, cross-validation, AUC, small data, German credit",
    data="data/clean/german_credit.csv (UCI Statlog German credit)",
    code=BASE + GERMAN + r'''LAMS = [0.0, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]; PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xg_raw, yg)); FD = []
for f in range(5):
    a, b = FOLDS[f]; a_tr, a_va = train_test_split(a, test_size=0.2, stratify=yg[a], random_state=SEED)
    xa, xv, xb = encode_german(Xg_raw.iloc[a_tr], [Xg_raw.iloc[a_va], Xg_raw.iloc[b]]); FD.append((xa, yg[a_tr], xv, yg[a_va], xb, yg[b]))
run = lambda f, **kw: roc_auc_score(FD[f][5], proba(fit(*FD[f][:4], hidden=(256, 256), epochs=150, bs=64, early=False, seed=f, **kw)[0], FD[f][4]))
awd = {v: [run(f, wd=v) for f in range(5)] for v in LAMS}; adp = {v: [run(f, p=v) for f in range(5)] for v in PS}
alr = np.mean([roc_auc_score(FD[f][5], LogisticRegression(C=1.0, max_iter=5000).fit(FD[f][0], FD[f][1]).predict_proba(FD[f][4])[:, 1]) for f in range(5)])
show('cv.auc_none', 'AUC without regularisation', round(float(np.mean(awd[0.0])), 3)); show('cv.auc_logit', 'AUC logistic regression', round(float(alr), 3))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
for ax, grid, res, lab in ((axs[0], LAMS, awd, 'Weight decay'), (axs[1], PS, adp, 'Dropout rate p')):
    x = np.arange(len(grid)); m = np.array([np.mean(res[v]) for v in grid]); s = np.array([np.std(res[v], ddof=1) for v in grid]) / np.sqrt(5)
    ax.errorbar(x, m, yerr=s, fmt='o-', color=PAL['model'], ms=4, capsize=2, label='MLP 256-256 (mean of 5 folds ± s.e.)')
    ax.axhline(alr, color=PAL['benchmark'], lw=1.1, label='Logistic regression'); ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in grid], fontsize=8.5); ax.set_xlabel(lab)
axs[0].set_ylabel('Held-out AUC')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('wd_dropout.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["early_stopping"] = dict(
    desc="Training and validation loss of an MLP 256-256 on the ULB credit-card fraud data with a time-ordered split, with the best validation epoch and the patience window of early stopping.",
    kw="early stopping, patience, validation loss, overfitting, fraud detection, time-ordered split",
    data="data/clean/creditcard_fraud_sample.csv (ULB Machine Learning Group and Worldline, OpenML 1597)",
    code=BASE + FRAUD + r'''_, h, _ = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=120, early=False, record=True, bs=256, seed=0)
h = np.array(h); best = int(np.argmin(h[:, 1]) + 1); b_, bad, stop = np.inf, 0, None
for i, v in enumerate(h[:, 1]):
    if v < b_ - 1e-5:
        b_, bad = v, 0
    else:
        bad += 1
        if bad >= 20:
            stop = i + 1; break
show('es.best_ep', 'best validation epoch', best); show('es.stop_ep', 'stopping epoch with patience 20', stop); show('es.val_min', 'best validation loss', round(float(h[:, 1].min()), 4))
fig, ax = plt.subplots(figsize=(6.4, 2.5)); e = np.arange(1, len(h) + 1)
ax.plot(e, h[:, 0], color=PAL['train'], label='Training loss'); ax.plot(e, h[:, 1], color=PAL['valid'], label='Validation loss')
ax.axvline(best, color='#222222', lw=1, ls='--', label='Best validation epoch (weights kept)'); ax.axvspan(best, stop, color='#8a8a8a', alpha=0.15, lw=0, label='Patience = 20 epochs')
ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy (log scale)')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('early_stopping.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["batchnorm"] = dict(
    desc="Validation MSE of a 6-layer MLP on California housing trained with SGD and momentum at learning rates 0.01 and 0.1, with and without batch normalisation (median of 3 seeds).",
    kw="batch normalisation, learning rate, divergence, deep MLP, SGD with momentum, California housing",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=BASE + CAL + r'''ci_tr, ci_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED); ci_tr, ci_va = train_test_split(ci_tr, test_size=0.25, random_state=SEED)
Cz = ((Xc - Xc[ci_tr].mean(0)) / Xc[ci_tr].std(0)).astype(np.float32); y32 = yc.astype(np.float32)
res = {}
for bn in (False, True):
    for lr in (0.01, 0.1):
        runs = []
        for s in range(3):
            _, h, _ = fit(Cz[ci_tr], y32[ci_tr], Cz[ci_va], y32[ci_va], hidden=(64,) * 6, bn=bn, lr=lr, epochs=40, bs=128, seed=s, early=False, loss='mse', opt_name='sgd')
            v = [min(x[1], 10.0) if np.isfinite(x[1]) else 10.0 for x in h]; runs.append(v + [10.0] * (40 - len(v)))
        res[(bn, lr)] = np.array(runs)
show('bn.plain_01.diverged', 'seeds that diverge without BN at lr 0.1', int(sum(r.max() >= 10.0 for r in res[(False, 0.1)])))
show('bn.bn_01.val_best', 'best validation MSE with BN at lr 0.1', round(float(np.median(res[(True, 0.1)].min(1))), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.5))
for (bn, lr), c, ls, nm in (((False, 0.01), PAL['data'], '-', 'no BN, lr=0.01'), ((False, 0.1), PAL['data'], '--', 'no BN, lr=0.1 (diverges)'),
                           ((True, 0.01), PAL['model'], '-', 'batch norm, lr=0.01'), ((True, 0.1), PAL['model'], '--', 'batch norm, lr=0.1')):
    ax.plot(np.arange(1, 41), np.median(res[(bn, lr)], axis=0), color=c, ls=ls, label=nm)
ax.set_ylim(0.2, 1.5); ax.set_xlabel('Epoch'); ax.set_ylabel('Validation MSE (median)')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('batchnorm.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["holdout_noise"] = dict(
    desc="Distribution of the test AUC of logistic regression on German credit over 200 random stratified 70/30 hold-out splits: how noisy a single split is on small data.",
    kw="hold-out validation, variance of the estimate, AUC, cross-validation, small data, German credit",
    data="data/clean/german_credit.csv (UCI Statlog German credit)",
    code=BASE + GERMAN + r'''aucs = []
for s in range(200):
    a, b = train_test_split(np.arange(len(yg)), test_size=0.3, stratify=yg, random_state=SEED + s); xa, xb = encode_german(Xg_raw.iloc[a], [Xg_raw.iloc[b]])
    aucs.append(roc_auc_score(yg[b], LogisticRegression(C=1.0, max_iter=5000).fit(xa, yg[a]).predict_proba(xb)[:, 1]))
aucs = np.array(aucs)
show('holdout.min', 'lowest AUC', round(float(aucs.min()), 3)); show('holdout.max', 'highest AUC', round(float(aucs.max()), 3)); show('holdout.sd', 'standard deviation', round(float(aucs.std(ddof=1)), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.hist(aucs, bins=30, color=PAL['benchmark'], alpha=0.8, label='AUC of one random 70/30 split'); ax.axvline(aucs.mean(), color='#222222', lw=1.2, label='Mean of 200 splits')
for q in np.percentile(aucs, [5, 95]):
    ax.axvline(q, color='#222222', lw=0.8, ls='--')
ax.set_xlabel('Test AUC of logistic regression, German credit'); ax.set_ylabel('Number of splits')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('holdout_noise.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["split_leakage"] = dict(
    desc="PR-AUC of the same MLP on the ULB card-fraud data estimated with a time-ordered split, a random split, and with oversampling of frauds before the split (leakage); 5 seeds each.",
    kw="data leakage, oversampling, time-ordered split, validation design, PR-AUC, fraud detection",
    data="data/clean/creditcard_fraud_sample.csv (ULB Machine Learning Group and Worldline, OpenML 1597)",
    code=BASE + FRAUD + r'''out = {'time': [], 'random': [], 'oversample_first': []}
for s in range(5):
    m, _, _ = fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(128, 128), epochs=40, bs=256, seed=s); out['time'].append(ap(yf[f_te], proba(m, Xf[f_te]), wf[f_te]))
    idx = np.arange(len(yf)); a, b = train_test_split(idx, test_size=0.2, stratify=yf, random_state=SEED + s); a, v = train_test_split(a, test_size=0.25, stratify=yf[a], random_state=SEED + s)
    Z = (Xf0 - Xf0[a].mean(0)) / Xf0[a].std(0); m, _, _ = fit(Z[a], yf[a], Z[v], yf[v], hidden=(128, 128), epochs=40, bs=256, seed=s); out['random'].append(ap(yf[b], proba(m, Z[b]), wf[b]))
    rng = np.random.default_rng(SEED + s); big = np.r_[idx, np.repeat(np.where(yf == 1)[0], 19)]; rng.shuffle(big); yb = yf[big]
    a, b = train_test_split(np.arange(len(big)), test_size=0.2, stratify=yb, random_state=SEED + s); a, v = train_test_split(a, test_size=0.25, stratify=yb[a], random_state=SEED + s)
    Xb = Xf0[big]; Z = (Xb - Xb[a].mean(0)) / Xb[a].std(0); m, _, _ = fit(Z[a], yb[a], Z[v], yb[v], hidden=(128, 128), epochs=40, bs=256, seed=s)
    wb = np.where(yb == 1, 1.0 / 20, wf[big]); out['oversample_first'].append(ap(yb[b], proba(m, Z[b]), wb[b]))
for k in out:
    show(f'design.{k}.ap', f'{k}: mean PR-AUC', round(float(np.mean(out[k])), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.5))
for i, (k, lab, c) in enumerate((('time', 'Time-ordered split\n(what deployment sees)', '#6250d6'), ('random', 'Random split\n(ignores time)', PAL['alt']),
                                 ('oversample_first', 'Oversampling before\nthe split (leakage)', PAL['positive']))):
    v = np.array(out[k]); ax.bar(i, v.mean(), color=c, width=0.55, alpha=0.85)
    ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), v, color='#222222', s=10, zorder=3, label='One seed' if i == 0 else '_')
ax.set_xticks(range(3)); ax.set_xticklabels(['Time-ordered split\n(what deployment sees)', 'Random split\n(ignores time)', 'Oversampling before\nthe split (leakage)'], fontsize=8.5)
ax.set_ylim(0, 1.05); ax.set_ylabel('Estimated PR-AUC')
legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('split_leakage.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["fraud_by_hour"] = dict(
    desc="Hourly fraud rate (weighted to the full data) over the two days of the ULB credit-card fraud data, with the start of the validation and test periods of the time-ordered split.",
    kw="fraud detection, class imbalance, time-ordered split, credit cards, ULB fraud data",
    data="data/clean/creditcard_fraud_sample.csv (ULB Machine Learning Group and Worldline, OpenML 1597)",
    code="SEED = 20261108\n" + FRAUD + r'''hour = (fr['Time'] // 3600).astype(int); rate = pd.Series(yf).groupby(hour).sum() / pd.Series(wf).groupby(hour).sum()
show('fraud.prev_full_pct', 'fraud rate in the full data (%)', round(float(100 * yf.sum() / wf.sum()), 3)); show('fraud.rate_h_max_pct', 'highest hourly fraud rate (%)', round(float(100 * rate.max()), 2))
fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.bar(rate.index, 100 * rate.values, color=PAL['positive'], width=0.8, label='Fraud rate in the hour (%)')
for cut, nm, y_ in ((c1, 'validation', 0.97), (c2, 'test', 0.80)):
    ax.axvline(cut / 3600, color='#222222', lw=1, ls='--'); ax.text(cut / 3600 + 0.4, y_ * 100 * rate.max(), nm, fontsize=8.5, va='top')
ax.set_xlabel('Hours since the first transaction (two days)'); ax.set_ylabel('Fraud rate (%)')
legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('fraud_by_hour.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["reg_ablation"] = dict(
    desc="Test PR-AUC (population-weighted) of an MLP 256-256 on the ULB card-fraud data with no regularisation, early stopping, weight decay, dropout, batch normalisation and all combined; 5 seeds each, time-ordered split.",
    kw="regularisation, ablation, early stopping, weight decay, dropout, batch normalisation, fraud detection, PR-AUC",
    data="data/clean/creditcard_fraud_sample.csv (ULB Machine Learning Group and Worldline, OpenML 1597)",
    code=BASE + FRAUD + r'''CONF = {'none': dict(early=False), 'es': dict(early=True), 'wd': dict(early=False, wd=0.1), 'dropout': dict(early=False, p=0.3),
        'bn': dict(early=False, bn=True), 'all': dict(early=True, wd=0.1, p=0.3)}
NAME = {'none': 'none (60 epochs)', 'es': 'early stopping', 'wd': 'weight decay 0.1', 'dropout': 'dropout 0.3', 'bn': 'batch norm', 'all': 'early stop. + decay + dropout'}
res = {k: [ap(yf[f_te], proba(fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=60, bs=256, seed=s, **kw)[0], Xf[f_te]), wf[f_te]) for s in range(5)] for k, kw in CONF.items()}
for k in ('none', 'es', 'all'):
    show(f'abl.{k}.ap', f'{NAME[k]}: mean test PR-AUC', round(float(np.mean(res[k])), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.5))
for i, k in enumerate(CONF):
    v = np.array(res[k]); ax.scatter(v, np.full(5, i) + np.linspace(-0.12, 0.12, 5), s=16, color=PAL['model'] if k != 'none' else PAL['data'])
    ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color='#222222', lw=1.6)
ax.set_yticks(range(len(CONF))); ax.set_yticklabels([NAME[k] for k in CONF], fontsize=8.5)
ax.scatter([], [], s=16, color=PAL['model'], label='MLP 256-256, one seed'); ax.plot([], [], color='#222222', lw=1.6, label='Mean of 5 seeds')
ax.set_xlabel('Test PR-AUC (population-weighted)')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('reg_ablation.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
FINAL = r'''lrf = LogisticRegression(max_iter=5000).fit(Xf[f_tr], yf[f_tr]); p_lr = lrf.predict_proba(Xf[f_te])[:, 1]
p_ml = np.mean([proba(fit(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], hidden=(256, 256), epochs=60, bs=256, seed=s, early=True, wd=0.1, p=0.3)[0], Xf[f_te]) for s in range(5)], axis=0)
xgm = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='aucpr',
                        early_stopping_rounds=100, random_state=SEED, n_jobs=1)
xgm.fit(Xf[f_tr], yf[f_tr], eval_set=[(Xf[f_va], yf[f_va])], verbose=False); p_xg = xgm.predict_proba(Xf[f_te])[:, 1]
yt, wt = yf[f_te], wf[f_te]
'''
Q["fraud_pr"] = dict(
    desc="Precision-recall curves (population-weighted) on the time-ordered test hours of the ULB card-fraud data: logistic regression, a regularised MLP (ensemble of 5 seeds) and XGBoost.",
    kw="precision-recall curve, PR-AUC, imbalanced classification, fraud detection, XGBoost, regularised MLP",
    data="data/clean/creditcard_fraud_sample.csv (ULB Machine Learning Group and Worldline, OpenML 1597)",
    code=BASE + FRAUD + FINAL + r'''fig, ax = plt.subplots(figsize=(6.4, 2.6))
for k, p, c, nm in (('logit', p_lr, PAL['benchmark'], 'Logistic regression'), ('mlp', p_ml, PAL['model'], 'Regularised MLP (5-seed ensemble)'), ('xgb', p_xg, PAL['alt'], 'XGBoost')):
    a = ap(yt, p, wt); show(f'final.{k}.ap', f'{nm}: test PR-AUC', round(a, 3))
    pr, rc, _ = precision_recall_curve(yt, p, sample_weight=wt); ax.plot(100 * rc, 100 * pr, color=c, label=f'{nm} (PR-AUC {a:.3f})')
ax.axhline(100 * yt.sum() / wt.sum(), color='#8a8a8a', lw=0.8, ls=':', label='Fraud rate (random ranking)')
ax.set_xlabel('Recall: frauds caught (%)'); ax.set_ylabel('Precision (%)')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('fraud_pr.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["fraud_cost"] = dict(
    desc="Total cost on the test hours of the ULB card-fraud data (missed frauds lose their amount, each alert costs a review of 5 EUR) against the number of alerts, for logistic regression, a regularised MLP and XGBoost.",
    kw="cost-sensitive evaluation, alert budget, fraud detection, threshold, expected cost, XGBoost, MLP",
    data="data/clean/creditcard_fraud_sample.csv (ULB Machine Learning Group and Worldline, OpenML 1597)",
    code=BASE + FRAUD + FINAL + r'''REVIEW = 5.0
def cost_curve(p):
    out = []
    for t in np.unique(np.quantile(p, np.linspace(0.90, 0.99995, 300))):
        al = p >= t; out.append([wt[al].sum(), REVIEW * wt[al].sum() + amt[f_te][(~al) & (yt == 1)].sum()])
    return np.array(out)
show('cost.no_model', 'test losses with no alerts (EUR)', round(float(amt[f_te][yt == 1].sum()), 0))
fig, ax = plt.subplots(figsize=(6.4, 2.5))
for k, p, c, nm in (('logit', p_lr, PAL['benchmark'], 'Logistic regression'), ('mlp', p_ml, PAL['model'], 'Regularised MLP'), ('xgb', p_xg, PAL['alt'], 'XGBoost')):
    cc = cost_curve(p); j = int(np.argmin(cc[:, 1])); show(f'cost.{k}.min', f'{nm}: lowest cost (EUR)', round(float(cc[j, 1]), 0))
    ax.plot(cc[:, 0], cc[:, 1] / 1000, color=c, label=nm); ax.plot(cc[j, 0], cc[j, 1] / 1000, 'o', color=c, ms=5)
ax.axhline(amt[f_te][yt == 1].sum() / 1000, color='#8a8a8a', ls=':', lw=1, label='No alerts: all test frauds lost')
ax.set_xscale('log'); ax.set_xlabel('Alerts sent to review (population-weighted, log scale)'); ax.set_ylabel('Cost (thousand EUR)')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('fraud_cost.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')

if __name__ == "__main__":
    only = sys.argv[1:]
    bad_all = []
    for chart, d in Q.items():
        if only and chart not in only:
            continue
        q = Quantlet(k=3, chart=chart, description=d["desc"], keywords=d["kw"], datafile=d["data"])
        q.md(f"Lecture 3 of *Neural Networks and Deep Learning*: the chart `{chart}` of the slides, recomputed from the public data. "
             "Numbers printed with `show()` are the ones on the slide.")
        q.code(d["code"])
        q.write(execute=True)
        n, bad = check_numbers(q, os.path.join(HERE, "L3_numbers.json"))
        print(f"  {chart}: {n} numbers checked", "OK" if not bad else bad, flush=True)
        bad_all += bad if n else [f"{chart}: no number checked"]
    sys.exit(1 if bad_all else 0)
