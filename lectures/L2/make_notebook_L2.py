"""
make_notebook_L2.py -- Lecture 2 notebooks (EN + RO): lecture notebook (sections § = "Experiment yourself" pointers) and seminar
notebook (student version + solutions), all executed. Run:  python3 lectures/L2/make_notebook_L2.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Training: backpropagation and optimisation", "Antrenarea rețelelor: propagarea înapoi a erorii și optimizarea")

SETUP = r'''import torch, torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from scipy import stats
torch.set_num_threads(1)
ca = pd.read_csv(DATA + 'california_housing.csv')
feat = [c for c in ca.columns if c != 'MedHouseVal_100k_usd']
Xc = ca[feat].values.astype(np.float32); yc = ca['MedHouseVal_100k_usd'].values.astype(np.float32)
i_tr, i_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, random_state=SEED)
mu, sd = Xc[i_tr].mean(0), Xc[i_tr].std(0); Z = (Xc - mu) / sd          # scaling fitted on the training rows only
Xtr, Xva, Xte, ytr, yva, yte = Z[i_tr], Z[i_va], Z[i_te], yc[i_tr], yc[i_va], yc[i_te]
print('train / validation / test:', len(i_tr), len(i_va), len(i_te))
print('linear regression, validation MSE:', round(float(np.mean((LinearRegression().fit(Xtr, ytr).predict(Xva) - yva) ** 2)), 3))

def make_mlp(d, hidden=(64, 64), act=nn.ReLU):
    layers, k = [], d
    for h in hidden:
        layers += [nn.Linear(k, h), act()]; k = h
    return nn.Sequential(*layers, nn.Linear(k, 1))

def make_opt(name, params, lr):
    return {'sgd': lambda: torch.optim.SGD(params, lr=lr), 'momentum': lambda: torch.optim.SGD(params, lr=lr, momentum=0.9),
            'rmsprop': lambda: torch.optim.RMSprop(params, lr=lr), 'adam': lambda: torch.optim.Adam(params, lr=lr),
            'adamw': lambda: torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)}[name]()

def run_reg(opt_name, lr, seed=0, epochs=60, bs=128, sched=None):
    """MLP 64-64 on California housing; returns the validation MSE after each epoch and the final test MSE"""
    torch.manual_seed(SEED + seed); rng = np.random.default_rng(SEED + seed)
    m = make_mlp(Xtr.shape[1]); opt = make_opt(opt_name, m.parameters(), lr)
    spe = int(np.ceil(len(Xtr) / bs)); total = epochs * spe
    sch = {'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total),
           'step': lambda: torch.optim.lr_scheduler.StepLR(opt, step_size=20 * spe, gamma=0.1),
           'onecycle': lambda: torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.3)}.get(sched, lambda: None)()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr).view(-1, 1); Xv, yv = torch.tensor(Xva), torch.tensor(yva).view(-1, 1)
    val = []
    for ep in range(epochs):
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; opt.zero_grad(); nn.MSELoss()(m(Xt[b]), yt[b]).backward(); opt.step()
            if sch is not None:
                sch.step()
        with torch.no_grad():
            v = float(nn.MSELoss()(m(Xv), yv)); val.append(v if np.isfinite(v) else 1e9)
    with torch.no_grad():
        te = float(nn.MSELoss()(m(torch.tensor(Xte)), torch.tensor(yte).view(-1, 1)))
    return val, te'''


def lecture(nb):
    T = nb.T
    nb.code(SETUP)
    nb.md(T("## §1 The loss surface of a real linear regression", "## §1 Suprafața pierderii unei regresii liniare reale"))
    nb.code("x = ca['MedInc'].values.astype(float); y = yc.astype(float)\n"
            "def path(xf, w0, lr, steps=60, beta=0.0):\n"
            "    X = np.c_[np.ones_like(xf), xf]; w, v = np.array(w0, float), np.zeros(2)\n"
            "    for _ in range(steps):\n"
            "        g = 2 * X.T @ (X @ w - y) / len(y); v = beta * v + g; w = w - lr * v\n"
            "    return X, w\n"
            "for name, xf in (('raw income', x), ('standardised income', (x - x.mean()) / x.std())):\n"
            "    X = np.c_[np.ones_like(xf), xf]; ev = np.linalg.eigvalsh(2 * X.T @ X / len(y))\n"
            "    X, w = path(xf, [4.0, -0.5 if name == 'raw income' else -1.0], 1 / ev.max())\n"
            "    print(name, '| condition number', round(ev.max() / ev.min(), 1), '| MSE after 60 GD steps', round(np.mean((X @ w - y) ** 2), 4))\n"
            "X, w = path(x, [4.0, -0.5], 0.5 / np.linalg.eigvalsh(2 * np.c_[np.ones_like(x), x].T @ np.c_[np.ones_like(x), x] / len(y)).max(), beta=0.9)\n"
            "print('raw income with momentum 0.9: MSE after 60 steps', round(np.mean((X @ w - y) ** 2), 4))")
    nb.md(T("## §2 Backpropagation by hand and the gradient check", "## §2 Propagarea înapoi de mînă și verificarea gradientului"))
    nb.code("torch.manual_seed(SEED)\n"
            "x = torch.tensor(Xte[0, [0, 6]], dtype=torch.float64); y = torch.tensor(float(yte[0]), dtype=torch.float64)\n"
            "W1 = (torch.randn(2, 2, dtype=torch.float64) * 0.5).round(decimals=2); w2 = (torch.randn(2, dtype=torch.float64) * 0.5).round(decimals=2)\n"
            "z = W1 @ x; h = torch.relu(z); yhat = w2 @ h\n"
            "d2 = yhat - y; gw2 = d2 * h; d1 = d2 * w2 * (z > 0).double(); gW1 = torch.outer(d1, x)\n"
            "print('forward: z', z.numpy().round(3), 'h', h.numpy().round(3), 'yhat', round(float(yhat), 3), 'y', round(float(y), 3))\n"
            "print('backward: dL/dw2', gw2.numpy().round(3), '\\ndL/dW1\\n', gW1.numpy().round(3))\n"
            "W1a, w2a = W1.clone().requires_grad_(), w2.clone().requires_grad_()\n"
            "(0.5 * (w2a @ torch.relu(W1a @ x) - y) ** 2).backward()\n"
            "print('autograd agrees:', torch.allclose(W1a.grad, gW1), torch.allclose(w2a.grad, gw2))")
    nb.code("torch.manual_seed(SEED); m = make_mlp(Xtr.shape[1]).double()\n"
            "Xg, yg = torch.tensor(Xtr[:256], dtype=torch.float64), torch.tensor(ytr[:256], dtype=torch.float64).view(-1, 1)\n"
            "grads = torch.autograd.grad(nn.MSELoss()(m(Xg), yg), list(m.parameters()))\n"
            "rng = np.random.default_rng(SEED); rel = []\n"
            "with torch.no_grad():\n"
            "    for _ in range(30):\n"
            "        li = rng.integers(0, len(grads)); p = list(m.parameters())[li].view(-1); j = rng.integers(0, p.numel()); old = p[j].item()\n"
            "        p[j] = old + 1e-6; lp = nn.MSELoss()(m(Xg), yg).item(); p[j] = old - 1e-6; lm = nn.MSELoss()(m(Xg), yg).item(); p[j] = old\n"
            "        fd = (lp - lm) / 2e-6; an = grads[li].view(-1)[j].item(); rel.append(abs(fd - an) / max(1e-12, abs(fd) + abs(an)))\n"
            "print('largest relative error, autograd vs finite differences:', f'{max(rel):.1e}')")
    nb.md(T("## §3 Vanishing gradients and initialisation", "## §3 Dispariția gradienților și inițializarea"))
    nb.code("def grad_norms(act, init, depth=10, width=64, seed=0):\n"
            "    torch.manual_seed(SEED + seed); layers, k, lins = [], Xtr.shape[1], []\n"
            "    for _ in range(depth):\n"
            "        l = nn.Linear(k, width); lins.append(l); layers += [l, act()]; k = width\n"
            "    last = nn.Linear(k, 1); lins.append(last); m = nn.Sequential(*layers, last)\n"
            "    for l in lins:\n"
            "        {'xavier': lambda w: nn.init.xavier_normal_(w), 'he': lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),\n"
            "         'small': lambda w: nn.init.normal_(w, 0, 0.01)}[init](l.weight); nn.init.zeros_(l.bias)\n"
            "    nn.MSELoss()(m(torch.tensor(Xtr[:512])), torch.tensor(ytr[:512]).view(-1, 1)).backward()\n"
            "    return [float(l.weight.grad.norm()) for l in lins[:-1]]\n"
            "for nm, a, i in (('sigmoid, Xavier', nn.Sigmoid, 'xavier'), ('tanh, Xavier', nn.Tanh, 'xavier'), ('ReLU, He', nn.ReLU, 'he'),\n"
            "                 ('ReLU, N(0, 0.01^2)', nn.ReLU, 'small')):\n"
            "    g = np.mean([grad_norms(a, i, seed=s) for s in range(5)], axis=0)\n"
            "    print(f'{nm:20s} first layer {g[0]:.2e}   last layer {g[-1]:.2e}')\n"
            "    plt.semilogy(range(1, 11), g, 'o-', label=nm)\n"
            "plt.xlabel('layer'); plt.ylabel('gradient norm'); plt.legend(); plt.show()")
    nb.md(T("## §4 The learning rate and the batch size", "## §4 Rata de învățare și mărimea lotului"))
    nb.code("for lr in (1e-4, 1e-3, 1e-2, 1e-1, 3e-1):\n"
            "    v = [run_reg('sgd', lr, seed=s, epochs=30)[0][-1] for s in range(3)]\n"
            "    print(f'SGD lr={lr:g}: validation MSE of 3 seeds', np.round(v, 3))")
    nb.md(T("§ Experiment yourself: rerun the loop with `bs=16` and `bs=1024` in `run_reg` and compare the validation MSE after 5 epochs.",
            "§ Exercițiu: rulați din nou bucla cu `bs=16` și `bs=1024` în `run_reg` și comparați MSE pe validare după 5 epoci."))
    nb.md(T("## §5 Optimisers: SGD, momentum, RMSProp, Adam, AdamW (5 seeds each)", "## §5 Optimizatori: SGD, moment, RMSProp, Adam, AdamW (cîte 5 inițializări)"))
    nb.code("BEST = {'sgd': 0.1, 'momentum': 0.01, 'rmsprop': 0.003, 'adam': 0.003, 'adamw': 0.003}   # best of three rates on validation (see the slides)\n"
            "res = {o: [run_reg(o, lr, seed=10 + s) for s in range(5)] for o, lr in BEST.items()}\n"
            "for o, r in res.items():\n"
            "    te = np.array([x[1] for x in r]); print(f'{o:8s} test MSE {te.mean():.3f} (sd {te.std(ddof=1):.3f})')\n"
            "    plt.plot(np.mean([x[0] for x in r], axis=0), label=o)\n"
            "plt.ylim(0.2, 0.6); plt.xlabel('epoch'); plt.ylabel('validation MSE'); plt.legend(); plt.show()\n"
            "ts, ta = np.array([x[1] for x in res['sgd']]), np.array([x[1] for x in res['adam']])\n"
            "print('paired t-test SGD vs Adam:', stats.ttest_rel(ts, ta))")
    nb.md(T("## §6 Learning-rate schedules", "## §6 Programele ratei de învățare"))
    nb.code("for s_ in (None, 'step', 'cosine', 'onecycle'):\n"
            "    r = [run_reg('momentum', 0.01, seed=20 + s, sched=s_) for s in range(3)]\n"
            "    print(f'{str(s_):9s} final validation MSE {np.mean([x[0][-1] for x in r]):.3f}   test MSE {np.mean([x[1] for x in r]):.3f}')")
    nb.md(T("## §7 Card fraud: SGD against Adam on a time-ordered split", "## §7 Fraude cu carduri: SGD față de Adam, cu împărțire cronologică"))
    nb.code("fr = pd.read_csv(DATA + 'creditcard_fraud_sample.csv').sort_values('Time').reset_index(drop=True)\n"
            "Xf = np.c_[fr[[f'V{i}' for i in range(1, 29)]].values, np.log1p(fr['Amount'].values)].astype(np.float32)\n"
            "yf, wf = fr['Class'].values.astype(np.float32), fr['weight'].values\n"
            "c1, c2 = np.quantile(fr['Time'], [0.6, 0.8]); tr = (fr['Time'] <= c1).values; va = ((fr['Time'] > c1) & (fr['Time'] <= c2)).values\n"
            "Xf = (Xf - Xf[tr].mean(0)) / Xf[tr].std(0)\n"
            "def fraud_run(opt_name, lr, seed, epochs=30, bs=256):\n"
            "    torch.manual_seed(SEED + seed); rng = np.random.default_rng(SEED + seed); m = make_mlp(Xf.shape[1]); opt = make_opt(opt_name, m.parameters(), lr)\n"
            "    Xt, yt, Xv = torch.tensor(Xf[tr]), torch.tensor(yf[tr]).view(-1, 1), torch.tensor(Xf[va]); ap = []\n"
            "    for ep in range(epochs):\n"
            "        idx = rng.permutation(len(Xt))\n"
            "        for i in range(0, len(idx), bs):\n"
            "            b = idx[i:i + bs]; opt.zero_grad(); nn.BCEWithLogitsLoss()(m(Xt[b]), yt[b]).backward(); opt.step()\n"
            "        with torch.no_grad():\n"
            "            ap.append(average_precision_score(yf[va], torch.sigmoid(m(Xv)).numpy().ravel(), sample_weight=wf[va]))\n"
            "    return ap\n"
            "for o, lr in (('sgd', 0.01), ('momentum', 0.01), ('adam', 1e-3)):\n"
            "    a = np.array([fraud_run(o, lr, s) for s in range(5)]); print(f'{o:8s} PR-AUC after epoch 1: {a[:, 0].mean():.3f}, epoch 30: {a[:, -1].mean():.3f}')")
    nb.md(T("## §8 Case study: 99 % VaR of the S&P 500 with a quantile network", "## §8 Studiu de caz: VaR de 99 % pentru S&P 500 cu o rețea cuantilică"))
    nb.code("x = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])\n"
            "px = x[x.ticker == 'SP500'].set_index('date')['adjclose'].dropna()\n"
            "end = (px.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1); px = px.loc[:end]\n"
            "r = 100 * np.log(px).diff().dropna()\n"
            "F = pd.DataFrame({'r': r, 'ar': r.abs(), 's5': r.rolling(5).std(), 's22': r.rolling(22).std(), 's66': r.rolling(66).std(), 'm5': r.rolling(5).mean()})\n"
            "dfv = pd.concat([F, r.shift(-1).rename('y')], axis=1).dropna(); TAU = 0.01\n"
            "v_tr = dfv.index <= '2016-12-31'; v_va = (dfv.index > '2016-12-31') & (dfv.index <= '2019-12-31'); v_te = dfv.index > '2019-12-31'\n"
            "Xv = dfv[F.columns].values.astype(np.float32); Xv = (Xv - Xv[v_tr].mean(0)) / Xv[v_tr].std(0); yv = dfv['y'].values.astype(np.float32)\n"
            "pin = lambda q, y: torch.mean(torch.maximum(TAU * (y - q), (TAU - 1) * (y - q)))\n"
            "def var_run(seed, epochs=400):\n"
            "    torch.manual_seed(SEED + seed); m = make_mlp(Xv.shape[1], (32, 32))\n"
            "    with torch.no_grad():\n"
            "        m[-1].bias.fill_(float(np.quantile(yv[v_tr], TAU)))\n"
            "    opt = torch.optim.Adam(m.parameters(), lr=1e-3); rng = np.random.default_rng(SEED + seed)\n"
            "    Xt, yt = torch.tensor(Xv[v_tr]), torch.tensor(yv[v_tr]).view(-1, 1); Xa, ya = torch.tensor(Xv[v_va]), torch.tensor(yv[v_va]).view(-1, 1)\n"
            "    best, state, bad = np.inf, None, 0\n"
            "    for ep in range(epochs):\n"
            "        idx = rng.permutation(len(Xt))\n"
            "        for i in range(0, len(idx), 256):\n"
            "            b = idx[i:i + 256]; opt.zero_grad(); pin(m(Xt[b]), yt[b]).backward(); opt.step()\n"
            "        with torch.no_grad():\n"
            "            vl = float(pin(m(Xa), ya))\n"
            "        if vl < best - 1e-6:\n"
            "            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}\n"
            "        else:\n"
            "            bad += 1\n"
            "            if bad >= 40:\n"
            "                break\n"
            "    m.load_state_dict(state)\n"
            "    with torch.no_grad():\n"
            "        return m(torch.tensor(Xv)).numpy().ravel()\n"
            "q = np.mean([var_run(s) for s in range(5)], axis=0)\n"
            "hs = r.rolling(250).quantile(TAU).reindex(dfv.index).values; yy = dfv['y'].values\n"
            "def kupiec(viol, p=TAU):\n"
            "    n, k = len(viol), int(viol.sum()); ph = k / n\n"
            "    lr = -2 * ((n - k) * np.log(1 - p) + k * np.log(p) - (n - k) * np.log(1 - ph) - k * np.log(ph))\n"
            "    return k, round(lr, 2), round(stats.chi2.sf(lr, 1), 3)\n"
            "print('test days', v_te.sum(), ' expected violations', round(TAU * v_te.sum(), 1))\n"
            "print('quantile MLP (violations, LR, p):', kupiec(yy[v_te] < q[v_te]))\n"
            "print('historical simulation (violations, LR, p):', kupiec(yy[v_te] < hs[v_te]))")
    nb.md(T("## §9 Training time on CPU and GPU", "## §9 Timpul de antrenare pe CPU și GPU"))
    nb.code("import time\n"
            "dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')\n"
            "def timing(d, steps=20):\n"
            "    m = make_mlp(Xtr.shape[1], (2048, 2048)).to(d); opt = torch.optim.Adam(m.parameters(), 1e-3)\n"
            "    x, y = torch.tensor(Xtr[:2048], device=d), torch.tensor(ytr[:2048], device=d).view(-1, 1)\n"
            "    for _ in range(3):\n"
            "        opt.zero_grad(); nn.MSELoss()(m(x), y).backward(); opt.step()\n"
            "    t0 = time.time()\n"
            "    for _ in range(steps):\n"
            "        opt.zero_grad(); nn.MSELoss()(m(x), y).backward(); opt.step()\n"
            "    float(m(x).sum())                              # waits for the device to finish\n"
            "    return (time.time() - t0) / steps * 1000\n"
            "print('CPU ms per step:', round(timing('cpu'), 1))\n"
            "if dev != 'cpu':\n"
            "    print(dev, 'ms per step:', round(timing(dev), 1))")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(SETUP)
    nb.task(T("Compute by hand (NumPy) the gradient of the MSE of a linear regression $\\hat y=\\mathbf X\\mathbf w$ on the California training set at "
              "$\\mathbf w=\\mathbf 0$, and check it with PyTorch autograd.",
              "Calculați manual (NumPy) gradientul MSE pentru regresia liniară $\\hat y=\\mathbf X\\mathbf w$ pe setul de antrenare California, în "
              "$\\mathbf w=\\mathbf 0$, și verificați-l cu diferențierea automată din PyTorch."))
    nb.solution("X = np.c_[np.ones(len(Xtr)), Xtr].astype(np.float64); w = np.zeros(X.shape[1])\n"
                "g = 2 * X.T @ (X @ w - ytr) / len(ytr)\n"
                "wt = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)\n"
                "torch.mean((torch.tensor(X) @ wt - torch.tensor(ytr, dtype=torch.float64)) ** 2).backward()\n"
                "print(np.allclose(g, wt.grad.numpy()))")
    nb.task(T("Find the largest eigenvalue of the Hessian $2\\mathbf X^\\top\\mathbf X/n$ of that regression and run gradient descent with "
              "$\\eta=1.9/\\lambda_{\\max}$ and $\\eta=2.1/\\lambda_{\\max}$ for 100 steps. What happens?",
              "Găsiți cea mai mare valoare proprie a hessianei $2\\mathbf X^\\top\\mathbf X/n$ a acestei regresii și rulați coborîrea pe gradient cu "
              "$\\eta=1{,}9/\\lambda_{\\max}$ și $\\eta=2{,}1/\\lambda_{\\max}$ timp de 100 de pași. Ce se întîmplă?"))
    nb.solution("Hm = 2 * X.T @ X / len(ytr); lam = np.linalg.eigvalsh(Hm).max()\n"
                "for c in (1.9, 2.1):\n"
                "    w = np.zeros(X.shape[1])\n"
                "    for _ in range(100):\n"
                "        w -= c / lam * 2 * X.T @ (X @ w - ytr) / len(ytr)\n"
                "    print(c, np.mean((X @ w - ytr) ** 2))")
    nb.task(T("Train the MLP 64-64 with Adam at learning rates 1e-4, 1e-3 and 1e-2 (3 seeds each, 30 epochs). Which rate would you keep?",
              "Antrenați MLP 64-64 cu Adam la ratele de învățare 1e-4, 1e-3 și 1e-2 (cîte 3 inițializări, 30 de epoci). Ce rată ați păstra?"))
    nb.solution("for lr in (1e-4, 1e-3, 1e-2):\n"
                "    print(lr, np.round([min(run_reg('adam', lr, seed=s, epochs=30)[0]) for s in range(3)], 3))")
    nb.task(T("Compare a constant learning rate with cosine annealing for SGD with momentum (3 seeds). Report the mean test MSE of each.",
              "Comparați o rată de învățare constantă cu descreșterea cosinus pentru SGD cu moment (3 inițializări). Raportați MSE medie pe test pentru fiecare."))
    nb.solution("for s_ in (None, 'cosine'):\n"
                "    print(s_, round(np.mean([run_reg('momentum', 0.01, seed=s, sched=s_)[1] for s in range(3)]), 3))")
    nb.task(T("Write the pinball loss $\\rho_\\tau(u)=u(\\tau-\\mathbb 1\\{u<0\\})$ and check numerically that its minimiser over a constant $q$ on "
              "the S&P 500 daily log returns is the empirical 1 % quantile.",
              "Scrieți pierderea pinball $\\rho_\\tau(u)=u(\\tau-\\mathbb 1\\{u<0\\})$ și verificați numeric că minimul ei după o constantă $q$, pe "
              "randamentele logaritmice zilnice S&P 500, este cuantila empirică de 1 %."))
    nb.solution("x = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])\n"
                "r = 100 * np.log(x[x.ticker == 'SP500'].set_index('date')['adjclose'].dropna()).diff().dropna().values\n"
                "grid = np.linspace(-6, 0, 3001); loss = [np.mean(np.maximum(0.01 * (r - q), (0.01 - 1) * (r - q))) for q in grid]\n"
                "print(round(grid[int(np.argmin(loss))], 3), round(np.quantile(r, 0.01), 3))")


if __name__ == "__main__":
    Notebook.build_all(2, TITLE, lecture, kind="lecture")
    Notebook.build_all(2, TITLE, seminar, kind="seminar")
