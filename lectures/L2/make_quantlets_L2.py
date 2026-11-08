"""
make_quantlets_L2.py -- one Quantlet per chart of Lecture 2: Quantlets/L2/NN_L2_<chart>/ (Metainfo.txt, executed notebook, PNG).
Every number printed with show() is compared with L2_numbers.json.
Run:  python3 lectures/L2/make_quantlets_L2.py  [chart ...]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

CAL = r'''import torch, torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy import stats
torch.set_num_threads(1)
SEED = 20261108
ca = pd.read_csv(DATA + 'california_housing.csv')
feat = [c for c in ca.columns if c != 'MedHouseVal_100k_usd']
Xc = ca[feat].values.astype(np.float32); yc = ca['MedHouseVal_100k_usd'].values.astype(np.float32)
i_tr, i_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, random_state=SEED)
mu, sd = Xc[i_tr].mean(0), Xc[i_tr].std(0); Z = (Xc - mu) / sd
Xtr, Xva, Xte, ytr, yva, yte = Z[i_tr], Z[i_va], Z[i_te], yc[i_tr], yc[i_va], yc[i_te]
lin = LinearRegression().fit(Xtr, ytr)
mse_lin_val = float(np.mean((lin.predict(Xva) - yva) ** 2)); mse_lin_te = float(np.mean((lin.predict(Xte) - yte) ** 2))

def make_mlp(d, hidden=(64, 64), act=nn.ReLU):
    layers, k = [], d
    for h in hidden:
        layers += [nn.Linear(k, h), act()]; k = h
    return nn.Sequential(*layers, nn.Linear(k, 1))

def make_opt(name, params, lr):
    return {'sgd': lambda: torch.optim.SGD(params, lr=lr), 'momentum': lambda: torch.optim.SGD(params, lr=lr, momentum=0.9),
            'rmsprop': lambda: torch.optim.RMSprop(params, lr=lr, alpha=0.99), 'adam': lambda: torch.optim.Adam(params, lr=lr),
            'adamw': lambda: torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)}[name]()

def run_reg(opt_name, lr, seed=0, epochs=60, bs=128, sched=None):
    torch.manual_seed(SEED + seed); rng = np.random.default_rng(SEED + seed)
    m = make_mlp(Xtr.shape[1]); opt = make_opt(opt_name, m.parameters(), lr)
    spe = int(np.ceil(len(Xtr) / bs)); total = epochs * spe
    sch = {'step': lambda: torch.optim.lr_scheduler.StepLR(opt, step_size=20 * spe, gamma=0.1),
           'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total),
           'onecycle': lambda: torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.3)}.get(sched, lambda: None)()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr).view(-1, 1); Xv, yv = torch.tensor(Xva), torch.tensor(yva).view(-1, 1)
    val, lrs = [], []
    for ep in range(epochs):
        m.train(); idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; opt.zero_grad(); nn.MSELoss()(m(Xt[b]), yt[b]).backward(); opt.step()
            lrs.append(opt.param_groups[0]['lr'])
            if sch is not None:
                sch.step()
        m.eval()
        with torch.no_grad():
            v = float(nn.MSELoss()(m(Xv), yv)); val.append(v if np.isfinite(v) else 1e9)
    with torch.no_grad():
        te = float(nn.MSELoss()(m(torch.tensor(Xte)), torch.tensor(yte).view(-1, 1)))
    return {'val': val, 'test': te if np.isfinite(te) else 1e9, 'lr': lrs[::spe]}
'''
BEST = "BEST = {'sgd': 0.1, 'momentum': 0.01, 'rmsprop': 0.003, 'adam': 0.003, 'adamw': 0.003}   # chosen on validation in analysis_L2.py (best of 3 rates)\n"
NAMES = "ONAME = {'sgd': 'SGD', 'momentum': 'SGD + momentum', 'rmsprop': 'RMSProp', 'adam': 'Adam', 'adamw': 'AdamW'}\n" \
        "OCOL = {'sgd': PAL['data'], 'momentum': '#6250d6', 'rmsprop': PAL['alt'], 'adam': PAL['model'], 'adamw': '#e87ba4'}\n"
FRAUD = r'''
fr = pd.read_csv(DATA + 'creditcard_fraud_sample.csv').sort_values('Time').reset_index(drop=True)
Xf = np.c_[fr[[f'V{i}' for i in range(1, 29)]].values, np.log1p(fr['Amount'].values)].astype(np.float32)
yf, wf = fr['Class'].values.astype(np.float32), fr['weight'].values
c1, c2 = np.quantile(fr['Time'], [0.6, 0.8]); f_tr = (fr['Time'] <= c1).values
f_va = ((fr['Time'] > c1) & (fr['Time'] <= c2)).values; f_te = (fr['Time'] > c2).values
Xf = (Xf - Xf[f_tr].mean(0)) / Xf[f_tr].std(0)
'''

Q = {}
Q["gd_paths"] = dict(
    desc="Gradient descent and momentum on the real mean-squared-error surface of a one-variable regression of California house values on median income: raw (ill-conditioned) against standardised income.",
    kw="gradient descent, momentum, condition number, Hessian, feature scaling, loss surface, California housing",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=r'''ca = pd.read_csv(DATA + 'california_housing.csv'); x = ca['MedInc'].values.astype(float); y = ca['MedHouseVal_100k_usd'].values.astype(float)
def setup(xf):
    X = np.c_[np.ones_like(xf), xf]; return X, 2 * X.T @ X / len(xf), np.linalg.lstsq(X, y, rcond=None)[0]
def gd(X, w0, lr, steps, beta=0.0):
    w, v, p = np.array(w0, float), np.zeros(2), [np.array(w0, float)]
    for _ in range(steps):
        v = beta * v + 2 * X.T @ (X @ w - y) / len(y); w = w - lr * v; p.append(w.copy())
    return np.array(p)
Xr, Hr, wr = setup(x); Xs, Hs, ws = setup((x - x.mean()) / x.std())
er, es = np.linalg.eigvalsh(Hr), np.linalg.eigvalsh(Hs)
pg = gd(Xr, [4.0, -0.5], 1 / er.max(), 60); pm = gd(Xr, [4.0, -0.5], 0.5 / er.max(), 60, 0.9); ps = gd(Xs, [4.0, -1.0], 1 / es.max(), 60)
mse = lambda X, w: float(np.mean((X @ w - y) ** 2))
show('surface.cond_raw', 'condition number, raw income', round(float(er.max() / er.min()), 1))
show('surface.mse_gd_raw_60', 'MSE after 60 GD steps, raw', round(mse(Xr, pg[-1]), 4))
show('surface.mse_mom_raw_60', 'MSE after 60 momentum steps, raw', round(mse(Xr, pm[-1]), 4))
show('surface.mse_gd_std_60', 'MSE after 60 GD steps, standardised', round(mse(Xs, ps[-1]), 4))
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
for ax, (X, w_, paths, lab) in zip(axs, ((Xr, wr, [(pg, PAL['data'], 'GD'), (pm, PAL['model'], 'momentum')], 'raw income'),
                                         (Xs, ws, [(ps, PAL['data'], 'GD')], 'standardised income'))):
    A, Bm = np.meshgrid(np.linspace(min(w_[0], -0.5) - 1, 5, 160), np.linspace(-1.2, 1.4, 160))
    L = (np.mean(y ** 2) - 2 * (A * np.mean(X[:, 0] * y) + Bm * np.mean(X[:, 1] * y)) + A ** 2 * np.mean(X[:, 0] ** 2)
         + 2 * A * Bm * np.mean(X[:, 0] * X[:, 1]) + Bm ** 2 * np.mean(X[:, 1] ** 2))
    ax.contour(A, Bm, L, levels=np.quantile(L, [0.01, 0.03, 0.08, 0.15, 0.3, 0.5, 0.7]), colors=['#8a8a8a'], linewidths=0.6)
    for p, c, l in paths:
        ax.plot(p[:, 0], p[:, 1], 'o-', ms=2, lw=1, color=c, label=l)
    ax.plot(*w_, '*', color='#222222', ms=9, label='Least-squares minimum')
    ax.set_xlabel('Intercept $a$'); ax.set_ylabel('Slope $b$'); ax.text(0.97, 0.95, lab, transform=ax.transAxes, ha='right', va='top', fontsize=9)
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('gd_paths.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["vanishing"] = dict(
    desc="Gradient norm per layer at initialisation in a 10-layer MLP on one batch of real California data: sigmoid and tanh with Xavier initialisation, ReLU with He and with too small weights (mean of 5 seeds).",
    kw="vanishing gradient, exploding gradient, Xavier initialisation, He initialisation, backpropagation, deep networks",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=CAL + r'''def grad_norms(act, init, depth=10, width=64, seed=0):
    torch.manual_seed(SEED + seed); layers, k, lins = [], Xtr.shape[1], []
    for _ in range(depth):
        l = nn.Linear(k, width); lins.append(l); layers += [l, act()]; k = width
    last = nn.Linear(k, 1); lins.append(last); m = nn.Sequential(*layers, last)
    for l in lins:
        {'xavier': lambda w: nn.init.xavier_normal_(w), 'he': lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
         'small': lambda w: nn.init.normal_(w, 0, 0.01)}[init](l.weight); nn.init.zeros_(l.bias)
    nn.MSELoss()(m(torch.tensor(Xtr[:512])), torch.tensor(ytr[:512]).view(-1, 1)).backward()
    return [float(l.weight.grad.norm()) for l in lins[:-1]]
VG = {'sig': (nn.Sigmoid, 'xavier', 'sigmoid, Xavier init.'), 'tanh': (nn.Tanh, 'xavier', 'tanh, Xavier init.'),
      'relu_he': (nn.ReLU, 'he', 'ReLU, He init.'), 'relu_small': (nn.ReLU, 'small', 'ReLU, N(0, 0.01²) init.')}
fig, ax = plt.subplots(figsize=(6.4, 2.6))
for k, (a, i, nm), c in zip(VG, VG.values(), (PAL['data'], PAL['benchmark'], PAL['model'], '#6250d6')):
    gn = np.mean([grad_norms(a, i, seed=s) for s in range(5)], axis=0)
    show(f'vanish.{k}.first', f'{nm}: first-layer gradient norm', float(f'{gn[0]:.2e}'))
    ax.plot(range(1, 11), gn, 'o-', ms=3.5, color=c, label=nm)
ax.set_yscale('log'); ax.set_xticks(range(1, 11)); ax.set_xlabel('Layer (1 = next to the input)'); ax.set_ylabel('Gradient norm (log scale)')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig('vanishing.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["lr_sweep"] = dict(
    desc="Validation MSE of an MLP 64-64 on California housing after 30 epochs of plain SGD, for nine learning rates from 0.0001 to 1 and three seeds; diverged runs marked.",
    kw="learning rate, stochastic gradient descent, divergence, hyperparameter tuning, California housing",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=CAL + r'''LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
sw = {lr: [run_reg('sgd', lr, seed=s, epochs=30)['val'][-1] for s in range(3)] for lr in LRS}
med = {lr: (np.median(v) if max(v) < 5.0 else np.nan) for lr, v in sw.items()}
best = min([lr for lr in LRS if np.isfinite(med[lr])], key=lambda lr: med[lr])
show('lrsweep.best_mse', 'best median validation MSE', round(float(med[best]), 3))
show('lrsweep.smallest_mse', 'validation MSE at lr = 1e-4', round(float(med[1e-4]), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.5))
for s in range(3):
    v = np.array([sw[lr][s] for lr in LRS]); good = v < 5.0
    ax.plot(np.array(LRS)[good], v[good], 'o', color=PAL['model'], ms=4, alpha=0.6, label='SGD, one seed' if s == 0 else '_')
    if (~good).any():
        ax.plot(np.array(LRS)[~good], np.full((~good).sum(), 1.55), 'x', color=PAL['positive'], ms=7, label='Diverged' if s == 0 else '_')
ax.plot(LRS, [med[lr] for lr in LRS], '-', color=PAL['model'], lw=1.2)
ax.axhline(mse_lin_val, color=PAL['benchmark'], lw=1.2, label='Linear regression')
ax.set_xscale('log'); ax.set_ylim(0, 1.65); ax.set_xlabel('Learning rate (log scale)'); ax.set_ylabel('Validation MSE')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('lr_sweep.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["batch_size"] = dict(
    desc="Validation MSE of an MLP 64-64 on California housing trained with Adam for batch sizes 16, 128 and 1,024, per epoch and per gradient step.",
    kw="mini-batch, batch size, stochastic gradient descent, Adam, training budget, California housing",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=CAL + r'''fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
for b, c in zip((16, 128, 1024), (PAL['data'], PAL['model'], '#6250d6')):
    v = np.array(run_reg('adam', 1e-3, seed=0, epochs=40, bs=b)['val']); spe = int(np.ceil(len(Xtr) / b))
    show(f'batch.{b}.val_ep5', f'batch {b}: validation MSE after 5 epochs', round(float(v[4]), 3))
    axs[0].plot(np.arange(1, 41), v, color=c, label=f'batch {b}'); axs[1].plot(np.arange(1, 41) * spe, v, color=c)
axs[1].set_xscale('log'); axs[0].set_ylim(0.2, 0.7); axs[0].set_xlabel('Epoch'); axs[1].set_xlabel('Gradient steps (log scale)'); axs[0].set_ylabel('Validation MSE')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('batch_size.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["optimizers"] = dict(
    desc="Validation MSE per epoch of an MLP 64-64 on California housing trained with SGD, SGD with momentum, RMSProp, Adam and AdamW at their best learning rates (mean of 5 seeds), against linear regression.",
    kw="optimiser, SGD, momentum, RMSProp, Adam, AdamW, training curve, California housing",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=CAL + BEST + NAMES + r'''fig, ax = plt.subplots(figsize=(6.4, 2.6))
for o, lr in BEST.items():
    v = np.array([run_reg(o, lr, seed=10 + s)['val'] for s in range(5)])
    show(f'opt.{o}.val_ep10', f'{ONAME[o]}: validation MSE after 10 epochs', round(float(v[:, 9].mean()), 3))
    ax.plot(np.arange(1, 61), v.mean(0), color=OCOL[o], label=f'{ONAME[o]} (lr={lr:g})')
ax.axhline(mse_lin_val, color=PAL['benchmark'], lw=1.1, ls='--', label='Linear regression')
ax.set_ylim(0.2, 0.62); ax.set_xlabel('Epoch'); ax.set_ylabel('Validation MSE')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig('optimizers.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["optimizer_seeds"] = dict(
    desc="Test MSE of an MLP 64-64 on California housing for five seeds of each optimiser (SGD, momentum, RMSProp, Adam, AdamW), with a paired t-test of SGD against Adam.",
    kw="random seed, reproducibility, optimiser comparison, paired t-test, Adam, SGD, California housing",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=CAL + BEST + NAMES + r'''te = {o: np.array([run_reg(o, lr, seed=10 + s)['test'] for s in range(5)]) for o, lr in BEST.items()}
for o in ('sgd', 'adam', 'adamw'):
    show(f'opt.{o}.test_mean', f'{ONAME[o]}: mean test MSE', round(float(te[o].mean()), 3))
    show(f'opt.{o}.test_sd', f'{ONAME[o]}: sd of test MSE', round(float(te[o].std(ddof=1)), 3))
tt = stats.ttest_rel(te['sgd'], te['adam']); show('opt_test.p', 'paired t-test SGD vs Adam, p', round(float(tt.pvalue), 3))
fig, ax = plt.subplots(figsize=(6.4, 2.4))
for i, o in enumerate(BEST):
    ax.scatter(te[o], np.full(5, i) + np.linspace(-0.12, 0.12, 5), color=OCOL[o], s=16, label=ONAME[o])
    ax.plot([te[o].mean()] * 2, [i - 0.3, i + 0.3], color='#222222', lw=1.6)
ax.axvline(mse_lin_te, color=PAL['benchmark'], lw=1.2, ls='--', label='Linear regression')
ax.set_yticks(range(5)); ax.set_yticklabels([ONAME[o] for o in BEST], fontsize=9); ax.set_xlabel('Test MSE after 60 epochs (5 seeds; bar = mean)')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig('optimizer_seeds.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["schedules"] = dict(
    desc="Learning-rate schedules (constant, step decay, cosine annealing, one-cycle) for SGD with momentum and the resulting validation MSE of an MLP 64-64 on California housing (mean of 3 seeds).",
    kw="learning-rate schedule, cosine annealing, step decay, one-cycle, warm-up, SGD with momentum, California housing",
    data="data/clean/california_housing.csv (StatLib, Pace and Barry, 1997)",
    code=CAL + r'''SN = {None: ('const', 'constant', PAL['data']), 'step': ('step', 'step decay (x0.1 every 20 epochs)', PAL['alt']),
      'cosine': ('cosine', 'cosine annealing', PAL['model']), 'onecycle': ('onecycle', 'one-cycle (warm-up 30%)', '#6250d6')}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
for s_, (key, nm, c) in SN.items():
    rr = [run_reg('momentum', 0.01, seed=20 + s, sched=s_) for s in range(3)]
    show(f'sched.{key}.val_final', f'{nm}: final validation MSE', round(float(np.mean([r['val'][-1] for r in rr])), 3))
    axs[0].plot(np.arange(1, 61), rr[0]['lr'][:60] if s_ else [0.01] * 60, color=c, label=nm)
    axs[1].plot(np.arange(1, 61), np.mean([r['val'] for r in rr], axis=0), color=c)
axs[0].set_ylabel('Learning rate'); axs[1].set_ylabel('Validation MSE'); axs[1].set_ylim(0.22, 0.45)
for ax in axs:
    ax.set_xlabel('Epoch')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig('schedules.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["fraud_optimizers"] = dict(
    desc="Validation PR-AUC (weighted to the true fraud rate) per epoch of an MLP 64-64 on the ULB credit-card fraud data with a time-ordered split, trained with SGD, SGD with momentum and Adam (5 seeds).",
    kw="fraud detection, imbalanced classification, PR-AUC, optimiser, Adam, SGD, time-ordered split",
    data="data/clean/creditcard_fraud_sample.csv (ULB Machine Learning Group and Worldline, OpenML 1597)",
    code=CAL + FRAUD + r'''def fraud_run(opt_name, lr, seed, epochs=30, bs=256):
    torch.manual_seed(SEED + seed); rng = np.random.default_rng(SEED + seed); m = make_mlp(Xf.shape[1]); opt = make_opt(opt_name, m.parameters(), lr)
    Xt, yt, Xv = torch.tensor(Xf[f_tr]), torch.tensor(yf[f_tr]).view(-1, 1), torch.tensor(Xf[f_va]); ap = []
    for ep in range(epochs):
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; opt.zero_grad(); nn.BCEWithLogitsLoss()(m(Xt[b]), yt[b]).backward(); opt.step()
        with torch.no_grad():
            ap.append(average_precision_score(yf[f_va], torch.sigmoid(m(Xv)).numpy().ravel(), sample_weight=wf[f_va]))
    return ap
fig, ax = plt.subplots(figsize=(6.4, 2.5))
for k, lr, c, nm in (('sgd', 0.01, PAL['data'], 'SGD, lr=0.01'), ('momentum', 0.01, '#6250d6', 'SGD + momentum 0.9, lr=0.01'), ('adam', 1e-3, PAL['model'], 'Adam, lr=0.001')):
    a = np.array([fraud_run(k, lr, s) for s in range(5)])
    show(f'fraud.{k}.ap_ep1', f'{nm}: validation PR-AUC after 1 epoch', round(float(a[:, 0].mean()), 3))
    ax.fill_between(np.arange(1, 31), a.min(0), a.max(0), color=c, alpha=0.15, lw=0); ax.plot(np.arange(1, 31), a.mean(0), color=c, label=nm)
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation PR-AUC')
legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('fraud_optimizers.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')
Q["var_backtest"] = dict(
    desc="One-day 99 % value at risk of the S&P 500 from a quantile MLP trained with the pinball loss (ensemble of 5 seeds), against 250-day historical simulation, with violations and the Kupiec test, test period from 2020.",
    kw="value at risk, quantile regression, pinball loss, backtesting, Kupiec test, S&P 500, neural network",
    data="data/clean/markets_daily.csv (Yahoo Finance, ticker SP500)",
    code=CAL + r'''x = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])
px = x[x.ticker == 'SP500'].set_index('date')['adjclose'].dropna()
end = (px.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1); px = px.loc[:end]
r = 100 * np.log(px).diff().dropna()
F = pd.DataFrame({'r': r, 'ar': r.abs(), 's5': r.rolling(5).std(), 's22': r.rolling(22).std(), 's66': r.rolling(66).std(), 'm5': r.rolling(5).mean()})
dfv = pd.concat([F, r.shift(-1).rename('y')], axis=1).dropna(); TAU = 0.01
v_tr = dfv.index <= pd.Timestamp('2016-12-31'); v_va = (dfv.index > pd.Timestamp('2016-12-31')) & (dfv.index <= pd.Timestamp('2019-12-31')); v_te = dfv.index > pd.Timestamp('2019-12-31')
Xv = dfv[F.columns].values.astype(np.float32); Xv = (Xv - Xv[v_tr].mean(0)) / Xv[v_tr].std(0); yv = dfv['y'].values.astype(np.float32)
tdate = pd.Series(r.index, index=r.index).shift(-1).reindex(dfv.index).values
pin = lambda q, y: torch.mean(torch.maximum(TAU * (y - q), (TAU - 1) * (y - q)))
def var_run(seed, epochs=400):
    torch.manual_seed(SEED + seed); m = make_mlp(Xv.shape[1], (32, 32))
    with torch.no_grad():
        m[-1].bias.fill_(float(np.quantile(yv[v_tr], TAU)))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); rng = np.random.default_rng(SEED + seed)
    Xt, yt = torch.tensor(Xv[v_tr]), torch.tensor(yv[v_tr]).view(-1, 1); Xa, ya = torch.tensor(Xv[v_va]), torch.tensor(yv[v_va]).view(-1, 1)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]; opt.zero_grad(); pin(m(Xt[b]), yt[b]).backward(); opt.step()
        with torch.no_grad():
            vl = float(pin(m(Xa), ya))
        if vl < best - 1e-6:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    m.load_state_dict(state)
    with torch.no_grad():
        return m(torch.tensor(Xv)).numpy().ravel()
q = np.mean([var_run(s) for s in range(5)], axis=0)
hs = r.rolling(250).quantile(TAU).reindex(dfv.index).values; yy = dfv['y'].values
for k, qq in (('nn', q), ('hs', hs)):
    viol = yy[v_te] < qq[v_te]; n, xk = len(viol), int(viol.sum()); ph = xk / n
    lr = -2 * ((n - xk) * np.log(1 - TAU) + xk * np.log(TAU) - (n - xk) * np.log(1 - ph) - xk * np.log(ph))
    show(f'var.{k}.viol', f'{k}: violations', xk); show(f'var.{k}.p', f'{k}: Kupiec p-value', round(float(stats.chi2.sf(lr, 1)), 3))
idx = pd.DatetimeIndex(tdate[v_te])
fig, ax = plt.subplots(figsize=(6.6, 2.7))
ax.plot(idx, yy[v_te], color='#8a8a8a', lw=0.5, label='S&P 500 daily log return (%)')
ax.plot(idx, q[v_te], color=PAL['model'], lw=1.0, label='Quantile MLP, 99% VaR')
ax.plot(idx, hs[v_te], color=PAL['benchmark'], lw=1.0, label='Historical simulation, 250 days')
vi = yy[v_te] < q[v_te]; ax.plot(idx[vi], yy[v_te][vi], 'o', color=PAL['positive'], ms=3.5, label='Violation of the MLP VaR')
ax.set_ylim(-13, 9); ax.set_xlim(idx.min(), idx.max()); ax.set_ylabel('%')
legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig('var_backtest.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()''')

if __name__ == "__main__":
    only = sys.argv[1:]
    bad_all = []
    for chart, d in Q.items():
        if only and chart not in only:
            continue
        q = Quantlet(k=2, chart=chart, description=d["desc"], keywords=d["kw"], datafile=d["data"])
        q.md(f"Lecture 2 of *Neural Networks and Deep Learning*: the chart `{chart}` of the slides, recomputed from the public data. "
             "Numbers printed with `show()` are the ones on the slide.")
        q.code(d["code"])
        q.write(execute=True)
        n, bad = check_numbers(q, os.path.join(HERE, "L2_numbers.json"))
        print(f"  {chart}: {n} numbers checked", "OK" if not bad else bad, flush=True)
        bad_all += bad if n else [f"{chart}: no number checked"]
    sys.exit(1 if bad_all else 0)
