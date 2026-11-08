"""
make_quantlets_L1.py -- one Quantlet per chart of Lecture 1: Quantlets/L1/NN_L1_<chart>/ (Metainfo.txt, executed notebook, PNG).
Every number printed with show() is compared with L1_numbers.json.
Run:  python3 lectures/L1/make_quantlets_L1.py  [chart ...]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")   # macOS: PyTorch and XGBoost ship different OpenMP runtimes
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

SEED = "SEED = 20261108\n"
TORCH = ("import xgboost as xgb            # import before torch (OpenMP runtimes on macOS)\n"
         "import torch, torch.nn as nn\ntorch.set_num_threads(1)\n"
         "from sklearn.linear_model import LogisticRegression\n"
         "from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss\n"
         "from sklearn.model_selection import train_test_split, StratifiedKFold\n" + SEED)
MLP = r'''
class MLP(nn.Module):
    def __init__(self, d, hidden=(64, 64), act='relu'):
        super().__init__()
        A = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}[act]
        layers, k = [], d
        for h in hidden:
            layers += [nn.Linear(k, h), A()]; k = h
        layers.append(nn.Linear(k, 1)); self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), act='relu', lr=1e-3, epochs=300, bs=128, patience=25, seed=0):
    """Adam on mini-batches; keeps the weights of the best validation epoch"""
    torch.manual_seed(SEED + seed); rng = np.random.default_rng(SEED + seed)
    m = MLP(Xtr.shape[1], hidden, act); opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss()
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
        return torch.sigmoid(m(torch.tensor(X))).numpy().ravel()

def xgb_fit(Xtr, ytr, Xva, yva, seed=0):
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', early_stopping_rounds=50, random_state=SEED + seed, n_jobs=1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m
'''
HMEQ = r'''
NUM = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
def encode_hmeq(tr, others):
    """median fill + missing indicators + one-hot, fitted on the training part only"""
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

h = pd.read_csv(DATA + 'hmeq.csv'); yh = h.pop('BAD').values.astype(int)
i_tr, i_te = train_test_split(np.arange(len(h)), test_size=0.2, stratify=yh, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=yh[i_tr], random_state=SEED)
(Xtr, Xva, Xte), cols = encode_hmeq(h.iloc[i_tr], [h.iloc[i_va], h.iloc[i_te]])
ytr, yva, yte = yh[i_tr], yh[i_va], yh[i_te]
'''
GERMAN = r'''
g = pd.read_csv(DATA + 'german_credit.csv'); yg = g['default'].values.astype(int); Xg = g.drop(columns='default')
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
'''
GCV = r'''
def german_cv():
    res = {'logit': [], 'mlp': [], 'xgb': []}; oof = {k: np.zeros(len(yg)) for k in res}
    for rep in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for f, (a, b) in enumerate(skf.split(Xg, yg)):
            a_tr, a_va = train_test_split(a, test_size=0.2, stratify=yg[a], random_state=SEED + rep)
            xa, xv, xb = encode_german(Xg.iloc[a_tr], [Xg.iloc[a_va], Xg.iloc[b]])
            pl = LogisticRegression(C=1.0, max_iter=5000).fit(xa, yg[a_tr]).predict_proba(xb)[:, 1]
            pm = proba(train_mlp(xa, yg[a_tr], xv, yg[a_va], hidden=(64, 64), seed=rep * 10 + f, bs=64), xb)
            px = xgb_fit(xa, yg[a_tr], xv, yg[a_va], seed=rep).predict_proba(xb)[:, 1]
            for k, p in (('logit', pl), ('mlp', pm), ('xgb', px)):
                res[k].append(roc_auc_score(yg[b], p))
                if rep == 0:
                    oof[k][b] = p
    return {k: np.array(v) for k, v in res.items()}, oof
A, OOF = german_cv()
'''

Q = {}
Q["default_rates"] = dict(
    desc="Default rates of German credit applicants by checking-account status and of HMEQ home-equity loans by number of delinquent credit lines, against the sample average.",
    kw="credit risk, default rate, probability of default, exploratory data analysis, German credit, HMEQ",
    data="data/clean/german_credit.csv (UCI Statlog German credit); data/clean/hmeq.csv (Baesens, Roesch and Scheule, 2016)",
    code=r'''g = pd.read_csv(DATA + 'german_credit.csv'); hm = pd.read_csv(DATA + 'hmeq.csv')
chk = g.groupby('checking_status')['default'].mean()
d = hm['DELINQ']; cat = np.where(d.isna(), 'na', np.where(d >= 3, '3+', d.fillna(0).astype(int).astype(str)))
dq = hm.groupby(cat)['BAD'].mean()
show('eda.chk.A11.pct', 'default rate, negative checking balance (%)', round(100 * chk['A11'], 1))
show('eda.chk.A14.pct', 'default rate, no checking account (%)', round(100 * chk['A14'], 1))
show('eda.delinq.0.pct', 'HMEQ default rate, no delinquent line (%)', round(100 * dq['0'], 1))
show('eda.delinq.3plus.pct', 'HMEQ default rate, 3+ delinquent lines (%)', round(100 * dq['3+'], 1))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
ks = ['A11', 'A12', 'A13', 'A14']
axs[0].bar(range(4), 100 * chk[ks].values, color=PAL['positive'], width=0.6, label='Default rate (%)')
axs[0].set_xticks(range(4)); axs[0].set_xticklabels(['< 0\nDM', '0-200\nDM', '≥ 200\nDM', 'no\naccount'], fontsize=9)
axs[0].set_ylabel('Default rate (%)'); axs[0].set_xlabel('German credit: checking account')
order = ['0', '1', '2', '3+', 'na']
axs[1].bar(range(5), 100 * dq[order].values, color=PAL['positive'], width=0.6)
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(['0', '1', '2', '3+', 'missing'], fontsize=9)
axs[1].set_xlabel('HMEQ: delinquent credit lines')
for ax, m in zip(axs, (g['default'].mean(), hm['BAD'].mean())):
    ax.axhline(100 * m, color='#8a8a8a', lw=1, ls='--', label='Sample average')
legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('default_rates.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["logit_coef"] = dict(
    desc="Logistic regression of default on standardised HMEQ variables (with a missing-value indicator for the debt-to-income ratio): coefficients and 95 % confidence intervals.",
    kw="logistic regression, odds ratio, credit scoring, missing values, HMEQ, one-neuron network",
    data="data/clean/hmeq.csv (Baesens, Roesch and Scheule, 2016)",
    code=TORCH + HMEQ + r'''import statsmodels.api as sm
base = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC', 'miss_DEBTINC']
ci = [cols.index(c) for c in base]
lm = sm.Logit(ytr, sm.add_constant(Xtr[:, ci].astype(float))).fit(disp=0)
coef = pd.DataFrame({'b': lm.params[1:], 'lo': lm.conf_int()[1:, 0], 'hi': lm.conf_int()[1:, 1]}, index=base)
coef = coef.reindex(coef['b'].abs().sort_values().index)
show('logit.or.miss_DEBTINC', 'odds ratio, debt-to-income missing', round(float(np.exp(coef.loc['miss_DEBTINC', 'b'])), 2))
show('logit.or.DELINQ', 'odds ratio, delinquent lines (per sd)', round(float(np.exp(coef.loc['DELINQ', 'b'])), 2))
show('logit.pseudo_r2', 'McFadden pseudo-R2', round(float(lm.prsquared), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.9)); yy = np.arange(len(coef))
ax.errorbar(coef['b'], yy, xerr=[coef['b'] - coef['lo'], coef['hi'] - coef['b']], fmt='o', color=PAL['benchmark'], ms=4, lw=1.2,
            label='Coefficient and 95% interval')
ax.axvline(0, color='#8a8a8a', lw=0.8); ax.set_yticks(yy); ax.set_yticklabels(coef.index, fontsize=8.5)
ax.set_xlabel('Change in log-odds of default per 1 sd')
legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('logit_coef.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["perceptron_errors"] = dict(
    desc="Rosenblatt's perceptron trained on the HMEQ loans for 60 epochs: the training error never settles because the classes are not linearly separable; logistic regression and the majority rule as references.",
    kw="perceptron, learning rule, convergence theorem, linear separability, credit scoring, HMEQ",
    data="data/clean/hmeq.csv (Baesens, Roesch and Scheule, 2016)",
    code=TORCH + HMEQ + r'''rng = np.random.default_rng(SEED); s = 2 * ytr - 1; X = Xtr.astype(float)
w, b, err, upd = np.zeros(X.shape[1]), 0.0, [], 0
for ep in range(60):
    for i in rng.permutation(len(X)):
        if s[i] * (X[i] @ w + b) <= 0:
            w += s[i] * X[i]; b += s[i]; upd += 1
    err.append(float(np.mean(np.sign(X @ w + b + 1e-12) != s)))
lr = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr); lr_err = float(np.mean((lr.predict_proba(Xtr)[:, 1] > 0.5) != ytr))
show('perceptron.err_min_pct', 'perceptron, lowest training error (%)', round(100 * min(err), 1))
show('perceptron.err_max_pct', 'perceptron, highest training error (%)', round(100 * max(err), 1))
show('perceptron.updates', 'number of updates', upd)
show('perceptron.logit_err_pct', 'logistic regression training error (%)', round(100 * lr_err, 1))
fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.plot(np.arange(1, 61), 100 * np.array(err), color=PAL['model'], lw=1.3, label='Perceptron after each epoch')
ax.axhline(100 * lr_err, color=PAL['benchmark'], lw=1.3, label='Logistic regression')
ax.axhline(100 * ytr.mean(), color='#8a8a8a', lw=1, ls='--', label="Always predict 'no default'")
ax.set_xlabel('Epoch (one pass over the training loans)'); ax.set_ylabel('Training error (%)')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('perceptron_errors.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["activations"] = dict(
    desc="Activation functions of neural networks (sigmoid, tanh, ReLU, GELU) and their derivatives; analytic curves, a mechanism illustration without data.",
    kw="activation function, sigmoid, tanh, ReLU, GELU, vanishing gradient, multilayer perceptron",
    data="none (analytic functions)",
    code=r'''from math import erf
z = np.linspace(-4, 4, 401); sig = 1 / (1 + np.exp(-z))
gelu = 0.5 * z * (1 + np.vectorize(lambda u: erf(u / np.sqrt(2)))(z))
acts = {'sigmoid': (sig, sig * (1 - sig)), 'tanh': (np.tanh(z), 1 - np.tanh(z) ** 2), 'ReLU': (np.maximum(z, 0), (z > 0).astype(float)),
        'GELU': (gelu, np.gradient(gelu, z))}
show('act.sigmoid_max_grad', 'largest derivative of the sigmoid', round(float((sig * (1 - sig)).max()), 2))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
for (nm, (f, d)), c in zip(acts.items(), [PAL['data'], PAL['benchmark'], PAL['model'], '#6250d6']):
    axs[0].plot(z, f, color=c, label=nm); axs[1].plot(z, d, color=c)
axs[0].set_ylim(-1.2, 2.2); axs[0].set_ylabel('$a(z)$'); axs[1].set_ylabel("$a'(z)$")
for ax in axs:
    ax.set_xlabel('$z$')
legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('activations.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["xor"] = dict(
    desc="The XOR problem: no linear boundary (logistic regression) separates the four points; an MLP with two tanh hidden units does. Mechanism illustration on the logical XOR table.",
    kw="XOR, perceptron, linear separability, hidden layer, multilayer perceptron, Minsky and Papert",
    data="none (logical XOR table)",
    code=TORCH + MLP + r'''Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32); yx = np.array([0, 1, 1, 0], dtype=np.float32)
best = None
for s in range(20):
    torch.manual_seed(SEED + s); mx = MLP(2, (2,), 'tanh'); o = torch.optim.Adam(mx.parameters(), lr=0.05)
    for _ in range(2000):
        o.zero_grad(); nn.BCEWithLogitsLoss()(mx(torch.tensor(Xx)), torch.tensor(yx).view(-1, 1)).backward(); o.step()
    if np.all((proba(mx, Xx) > 0.5) == yx.astype(bool)):
        best = mx; break
lx = LogisticRegression(C=1e6).fit(Xx, yx)
show('xor.logit_acc_pct', 'accuracy of the linear boundary (%)', int(round(100 * np.mean((lx.predict_proba(Xx)[:, 1] > 0.5) == yx))))
show('xor.mlp_acc_pct', 'accuracy of the MLP (%)', int(round(100 * np.mean((proba(best, Xx) > 0.5) == yx))))
gx = np.linspace(-0.4, 1.4, 200); GX, GY = np.meshgrid(gx, gx); grid = np.c_[GX.ravel(), GY.ravel()].astype(np.float32)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
for ax, P, lab in ((axs[0], lx.predict_proba(grid)[:, 1], 'one linear boundary'), (axs[1], proba(best, grid), 'MLP, 2 hidden units')):
    ax.contourf(GX, GY, P.reshape(GX.shape), levels=[0, 0.5, 1], colors=[PAL['negative'], PAL['positive']], alpha=0.15)
    ax.contour(GX, GY, P.reshape(GX.shape), levels=[0.5], colors=['#222222'], linewidths=1)
    ax.scatter(Xx[yx == 0, 0], Xx[yx == 0, 1], s=60, color=PAL['negative'], label='XOR = 0', zorder=3)
    ax.scatter(Xx[yx == 1, 0], Xx[yx == 1, 1], s=70, color=PAL['positive'], marker='X', label='XOR = 1', zorder=3)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel('$x_1$   (' + lab + ')'); ax.set_ylabel('$x_2$')
legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('xor.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["boundary_hmeq"] = dict(
    desc="Predicted probability of default on two HMEQ variables (debt-to-income, age of the oldest credit line): straight bands of logistic regression against the curved regions of an MLP 64-64, with test loans.",
    kw="decision boundary, multilayer perceptron, logistic regression, credit scoring, AUC, HMEQ",
    data="data/clean/hmeq.csv (Baesens, Roesch and Scheule, 2016)",
    code=TORCH + MLP + HMEQ + r'''two = ['DEBTINC', 'CLAGE']; m2 = h[two].notna().all(axis=1).values
a, v, t = i_tr[m2[i_tr]], i_va[m2[i_va]], i_te[m2[i_te]]
mu2, sd2 = h.iloc[a][two].mean(), h.iloc[a][two].std()
Z = lambda ii: ((h.iloc[ii][two] - mu2) / sd2).values.astype(np.float32)
lr2 = LogisticRegression(C=1e6, max_iter=5000).fit(Z(a), yh[a])
ml2 = train_mlp(Z(a), yh[a], Z(v), yh[v], hidden=(64, 64), seed=0, epochs=400, patience=40)
show('two.auc_logit', 'test AUC, logistic, two inputs', round(roc_auc_score(yh[t], lr2.predict_proba(Z(t))[:, 1]), 3))
show('two.auc_mlp', 'test AUC, MLP 64-64, two inputs', round(roc_auc_score(yh[t], proba(ml2, Z(t))), 3))
DG, CG = np.meshgrid(np.linspace(0, 70, 220), np.linspace(0, 600, 220))
gz = (np.c_[DG.ravel(), CG.ravel()] - mu2.values) / sd2.values
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True); te = h.iloc[t]; ok = yh[t] == 0
for ax, P, nm in ((axs[0], lr2.predict_proba(gz)[:, 1], 'logistic'), (axs[1], proba(ml2, gz.astype(np.float32)), 'MLP 64-64')):
    ax.contourf(DG, CG, P.reshape(DG.shape), levels=[0, 0.1, 0.2, 0.3, 0.5, 1], cmap='Oranges', alpha=0.55)
    ax.contour(DG, CG, P.reshape(DG.shape), levels=[0.2], colors=['#222222'], linewidths=1)
    ax.scatter(te['DEBTINC'][ok], te['CLAGE'][ok], s=3, color=PAL['negative'], alpha=0.35, lw=0, label='Repaid (test)')
    ax.scatter(te['DEBTINC'][~ok], te['CLAGE'][~ok], s=5, color=PAL['positive'], alpha=0.8, lw=0, label='Defaulted (test)')
    ax.set_xlim(0, 70); ax.set_ylim(0, 600); ax.set_xlabel('Debt-to-income (%)')
    ax.text(0.97, 0.95, nm, transform=ax.transAxes, ha='right', va='top', fontsize=9)
axs[0].set_ylabel('Oldest line (months)')
legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('boundary_hmeq.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["width_auc"] = dict(
    desc="Test AUC of a one-hidden-layer MLP on HMEQ as a function of the number of hidden units (1 to 256, 5 seeds each), against logistic regression and gradient boosting (XGBoost).",
    kw="multilayer perceptron, width, hidden units, AUC, credit scoring, XGBoost, HMEQ",
    data="data/clean/hmeq.csv (Baesens, Roesch and Scheule, 2016)",
    code=TORCH + MLP + HMEQ + r'''W = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ws = {w: [roc_auc_score(yte, proba(train_mlp(Xtr, ytr, Xva, yva, hidden=(w,), seed=s), Xte)) for s in range(5)] for w in W}
auc_lr = roc_auc_score(yte, LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
auc_xg = roc_auc_score(yte, xgb_fit(Xtr, ytr, Xva, yva).predict_proba(Xte)[:, 1])
show('hmeq.auc_w1', 'AUC with 1 hidden unit (mean of 5 seeds)', round(float(np.mean(ws[1])), 3))
show('hmeq.auc_best_width', 'AUC with 256 hidden units', round(float(np.mean(ws[256])), 3))
show('hmeq.auc_logit', 'AUC, logistic regression', round(auc_lr, 3))
show('hmeq.auc_xgb', 'AUC, XGBoost', round(auc_xg, 3))
mu = np.array([np.mean(ws[w]) for w in W])
fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.fill_between(W, [min(ws[w]) for w in W], [max(ws[w]) for w in W], color=PAL['model'], alpha=0.18, lw=0)
ax.plot(W, mu, 'o-', color=PAL['model'], ms=4, label='MLP, one hidden layer (mean of 5 seeds, range)')
ax.axhline(auc_lr, color=PAL['benchmark'], lw=1.3, label='Logistic regression')
ax.axhline(auc_xg, color=PAL['alt'], lw=1.3, ls='--', label='Gradient boosting (XGBoost)')
ax.set_xscale('log', base=2); ax.set_xticks(W); ax.set_xticklabels([str(w) for w in W])
ax.set_xlabel('Hidden units $H$'); ax.set_ylabel('Test AUC')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('width_auc.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["roc_hmeq"] = dict(
    desc="ROC curves of logistic regression, an MLP 64-64 and XGBoost on the HMEQ test loans, with the share of defaults caught at a 10 % false-positive rate.",
    kw="ROC curve, AUC, true positive rate, credit scoring, multilayer perceptron, XGBoost, HMEQ",
    data="data/clean/hmeq.csv (Baesens, Roesch and Scheule, 2016)",
    code=TORCH + MLP + HMEQ + r'''p_lr = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
p_ml = proba(train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), seed=0), Xte)
p_xg = xgb_fit(Xtr, ytr, Xva, yva).predict_proba(Xte)[:, 1]
fig, ax = plt.subplots(figsize=(6.4, 2.7))
for key, p, c, nm in (('logit', p_lr, PAL['benchmark'], 'Logistic regression'), ('mlp', p_ml, PAL['model'], 'MLP 64-64'), ('xgb', p_xg, PAL['alt'], 'XGBoost')):
    fpr, tpr, _ = roc_curve(yte, p)
    show(f'hmeq.tpr10_{key}', f'{nm}: defaults caught at 10% FPR (%)', round(100 * float(np.interp(0.10, fpr, tpr)), 1))
    ax.plot(100 * fpr, 100 * tpr, color=c, label=f'{nm} (AUC {roc_auc_score(yte, p):.3f})')
ax.plot([0, 100], [0, 100], color='#8a8a8a', lw=0.8, ls=':')
ax.set_xlabel('False positive rate: good loans rejected (%)'); ax.set_ylabel('Defaults caught (%)')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('roc_hmeq.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["calibration"] = dict(
    desc="Reliability diagram (deciles of predicted probability of default against observed default rate) and Brier scores of logistic regression, an MLP 64-64 and XGBoost on the HMEQ test loans.",
    kw="calibration, reliability diagram, Brier score, probability of default, multilayer perceptron, HMEQ",
    data="data/clean/hmeq.csv (Baesens, Roesch and Scheule, 2016)",
    code=TORCH + MLP + HMEQ + r'''p_lr = LogisticRegression(C=1e6, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
p_ml = proba(train_mlp(Xtr, ytr, Xva, yva, hidden=(64, 64), seed=0), Xte)
p_xg = xgb_fit(Xtr, ytr, Xva, yva).predict_proba(Xte)[:, 1]
def calib(p, y, nb=10):
    q = np.quantile(p, np.linspace(0, 1, nb + 1)); b = np.clip(np.searchsorted(q, p, side='right') - 1, 0, nb - 1)
    return np.array([p[b == k].mean() for k in range(nb)]), np.array([y[b == k].mean() for k in range(nb)])
fig, ax = plt.subplots(figsize=(6.4, 2.6))
for key, p, c, nm in (('logit', p_lr, PAL['benchmark'], 'Logistic regression'), ('mlp', p_ml, PAL['model'], 'MLP 64-64'), ('xgb', p_xg, PAL['alt'], 'XGBoost')):
    show(f'calib.brier_{key}', f'Brier score, {nm}', round(brier_score_loss(yte, p), 4))
    a, b = calib(p, yte); ax.plot(100 * a, 100 * b, 'o-', color=c, ms=3.5, label=nm)
ax.plot([0, 100], [0, 100], color='#8a8a8a', lw=0.8, ls=':', label='Perfect calibration')
ax.set_xlabel('Predicted probability of default (%), decile bins'); ax.set_ylabel('Observed default rate (%)')
legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('calibration.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["german_cv"] = dict(
    desc="AUC of logistic regression, an MLP 64-64 and XGBoost on German credit over 5 x 5-fold cross-validation (25 held-out folds), with the Nadeau-Bengio corrected t-test.",
    kw="cross-validation, corrected resampled t-test, AUC, credit scoring, small data, German credit",
    data="data/clean/german_credit.csv (UCI Statlog German credit)",
    code=TORCH + MLP + GERMAN + GCV + r'''from scipy import stats
d = A['mlp'] - A['logit']; J = len(d); t = d.mean() / np.sqrt((1 / J + 0.25) * d.var(ddof=1))
show('german.auc_logit', 'mean AUC, logistic', round(float(A['logit'].mean()), 3))
show('german.auc_mlp', 'mean AUC, MLP 64-64', round(float(A['mlp'].mean()), 3))
show('german.auc_xgb', 'mean AUC, XGBoost', round(float(A['xgb'].mean()), 3))
show('german.t_mlp', 'corrected t, MLP minus logistic', round(float(t), 2))
show('german.p_mlp', 'p-value', round(float(2 * stats.t.sf(abs(t), J - 1)), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.4))
for i, (k, c, nm) in enumerate((('logit', PAL['benchmark'], 'Logistic regression'), ('mlp', PAL['model'], 'MLP 64-64'), ('xgb', PAL['alt'], 'XGBoost'))):
    jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(A[k]))
    ax.scatter(A[k], i + jit, s=10, color=c, alpha=0.7, label=nm); ax.plot([A[k].mean()] * 2, [i - 0.3, i + 0.3], color='#222222', lw=1.8)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['Logistic', 'MLP', 'XGBoost'])
ax.set_xlabel('AUC on the held-out fold (5 x 5-fold cross-validation)')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('german_cv.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["cost_threshold"] = dict(
    desc="Expected cost per German credit applicant (UCI cost matrix: a missed bad risk costs 5, a rejected good applicant 1) as a function of the cut-off, out-of-fold predictions of logistic regression, an MLP and XGBoost; cost-optimal cut-off 1/6.",
    kw="cost-sensitive classification, cut-off, Bayes decision rule, credit scoring, German credit",
    data="data/clean/german_credit.csv (UCI Statlog German credit)",
    code=TORCH + MLP + GERMAN + GCV + r'''cost = lambda p, thr: float((5 * np.sum((p < thr) & (yg == 1)) + np.sum((p >= thr) & (yg == 0))) / len(yg))
thr = np.linspace(0.02, 0.9, 89)
show('german.cost_logit_bayes', 'cost per applicant, logistic, cut-off 1/6', round(cost(OOF['logit'], 1 / 6), 3))
show('german.cost_logit_05', 'cost per applicant, logistic, cut-off 1/2', round(cost(OOF['logit'], 0.5), 3))
show('german.cost_mlp_bayes', 'cost per applicant, MLP, cut-off 1/6', round(cost(OOF['mlp'], 1 / 6), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.5))
for k, c, nm in (('logit', PAL['benchmark'], 'Logistic regression'), ('mlp', PAL['model'], 'MLP 64-64'), ('xgb', PAL['alt'], 'XGBoost')):
    ax.plot(thr, [cost(OOF[k], t) for t in thr], color=c, label=nm)
ax.axvline(1 / 6, color='#222222', lw=1, ls='--', label='Cost-optimal cut-off 1/6')
ax.axhline(5 * yg.mean(), color='#8a8a8a', lw=1, ls=':', label='Accept everyone')
ax.set_xlabel('Cut-off: reject if predicted PD ≥ threshold'); ax.set_ylabel('Cost per applicant')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('cost_threshold.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["uat_california"] = dict(
    desc="Universal approximation on real data: one-hidden-layer sigmoid MLPs with 1, 3 and 50 units fitted to California house values as a function of latitude, against a straight line; held-out MSE.",
    kw="universal approximation theorem, hidden units, nonlinear regression, California housing, multilayer perceptron",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=TORCH + MLP + r'''ca = pd.read_csv(DATA + 'california_housing.csv')
x1, y1 = ca['Latitude'].values.astype(np.float32), ca['MedHouseVal_100k_usd'].values.astype(np.float32)
keep = y1 < 5.0; x1, y1 = x1[keep], y1[keep]
a, b = train_test_split(np.arange(len(x1)), test_size=0.25, random_state=SEED); mx, sx = x1[a].mean(), x1[a].std()
xs = np.linspace(32.5, 42, 300).astype(np.float32); fits = {}
for H in (1, 3, 50):
    torch.manual_seed(SEED + H); m = MLP(1, (H,), 'sigmoid'); opt = torch.optim.Adam(m.parameters(), lr=0.02)
    Xa, Ya = torch.tensor((x1[a, None] - mx) / sx), torch.tensor(y1[a, None])
    for it in range(4000):
        opt.zero_grad(); nn.MSELoss()(m(Xa), Ya).backward(); opt.step()
    with torch.no_grad():
        fits[H] = m(torch.tensor((xs[:, None] - mx) / sx)).numpy().ravel()
        show(f'uat.mse_w{H}', f'held-out MSE, H = {H}', round(float(nn.MSELoss()(m(torch.tensor((x1[b, None] - mx) / sx)), torch.tensor(y1[b, None]))), 3))
bl = np.polyfit(x1[a], y1[a], 1); show('uat.mse_lin', 'held-out MSE, linear', round(float(np.mean((np.polyval(bl, x1[b]) - y1[b]) ** 2)), 3))
bins = np.quantile(x1[a], np.linspace(0, 1, 41)); bc = np.digitize(x1[a], bins[1:-1])
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.scatter([x1[a][bc == k].mean() for k in range(40)], [y1[a][bc == k].mean() for k in range(40)], s=12, color=PAL['data'], zorder=3,
           label='Data: mean value in 40 latitude bins')
ax.plot(xs, np.polyval(bl, xs), color=PAL['benchmark'], lw=1.2, label='Linear')
for H, c in ((1, '#e87ba4'), (3, PAL['alt']), (50, PAL['model'])):
    ax.plot(xs, fits[H], color=c, lw=1.4, label=f'MLP, H = {H}')
ax.set_xlabel('Latitude of the block group (degrees north)'); ax.set_ylabel('Value (100,000 USD)'); ax.set_xlim(32.5, 42)
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('uat_california.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')

if __name__ == "__main__":
    only = sys.argv[1:]
    bad_all = []
    for chart, d in Q.items():
        if only and chart not in only:
            continue
        q = Quantlet(k=1, chart=chart, description=d["desc"], keywords=d["kw"], datafile=d["data"])
        q.md(f"Lecture 1 of *Neural Networks and Deep Learning*: the chart `{chart}` of the slides, recomputed from the public data. "
             "Numbers printed with `show()` are the ones on the slide.")
        q.code(d["code"])
        q.write(execute=True)
        n, bad = check_numbers(q, os.path.join(HERE, "L1_numbers.json"))
        print(f"  {chart}: {n} numbers checked", "OK" if not bad else bad)
        bad_all += bad if n else [f"{chart}: no number checked"]
    sys.exit(1 if bad_all else 0)
