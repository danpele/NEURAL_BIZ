"""
make_quantlets_L10.py -- one Quantlet per chart of Lecture 10: Quantlets/L10/NN_L10_<chart>/ (Metainfo.txt, executed
notebook, <chart>.png). Every number printed with show() is compared with L10_numbers.json. Deterministic numbers
(dynamic programming, benchmarks) are reproduced exactly; the learning agents are retrained with one seed and a smaller
budget (the slides report five seeds), so their numbers are printed as "one seed" and not checked.
Run:  python3 lectures/L10/make_quantlets_L10.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

NUM = os.path.join(HERE, "L10_numbers.json")
PRICES = """import torch, torch.nn as nn
x = pd.read_csv(DATA + 'power_da_prices_hourly_utc.csv', parse_dates=['timestamp_utc']).set_index('timestamp_utc')
x.index = x.index.tz_convert('Europe/Brussels')
END = '2026-10-31'                                   # last complete month of the course data
x = x.loc[:END + ' 23:59']
x['day'] = x.index.tz_localize(None).normalize(); x['hour'] = x.index.hour
def day_matrix(zone):
    d = x[['day', 'hour', zone]].dropna(); n = d.groupby('day')[zone].count()
    return d[d.day.isin(n[n == 24].index)].pivot_table(index='day', columns='hour', values=zone).sort_index()
RO = day_matrix('RO')"""
BATTERY = """E_MAX, ETA = 2, 0.85
ETA_C = ETA_D = np.sqrt(ETA); ACTS = np.array([-1, 0, 1])
def reward(p, a): return np.where(a > 0, ETA_D * p, np.where(a < 0, -p / ETA_C, 0.0))
def dp_days(P):
    D = P.shape[0]; V = np.zeros((D, 25, E_MAX + 1)); A = np.zeros((D, 24, E_MAX + 1), dtype=int)
    for h in range(23, -1, -1):
        best = np.full((D, E_MAX + 1), -np.inf); arg = np.zeros((D, E_MAX + 1), dtype=int)
        for s in range(E_MAX + 1):
            for a in ACTS:
                if 0 <= s - a <= E_MAX:
                    q = reward(P[:, h], a) + V[:, h + 1, s - a]; b = q > best[:, s]; best[b, s] = q[b]; arg[b, s] = a
        V[:, h] = best; A[:, h] = arg
    return V, A"""
SPLIT = """P = RO.values; days = RO.index
ok = np.array([i >= 7 and (days[i] - days[i-1]).days == 1 and (days[i] - days[i-7]).days == 7 for i in range(len(days))])
iTR = np.where(ok & (days.year >= 2019) & (days.year <= 2023))[0]; iVA = np.where(ok & (days.year == 2024))[0]
iTE = np.where(ok & (days.year >= 2025))[0]
V, A = dp_days(P)
plan = np.zeros((len(P), 24), int); s = np.zeros(len(P), int)
for h in range(24):
    a = A[np.arange(len(P)), h, s]; plan[:, h] = a; s = s - a
def run_plan(Pd, acts):
    s = np.zeros(len(Pd), int); tot = np.zeros(len(Pd))
    for h in range(24):
        a = acts[:, h].copy(); a[(a > 0) & (s == 0)] = 0; a[(a < 0) & (s == E_MAX)] = 0; tot += reward(Pd[:, h], a); s = s - a
    return tot
naive = np.full(len(P), np.nan); naive[1:] = run_plan(P[1:], plan[:-1])
QB = P.argsort(1).argsort(1) // 6
def mask_of(s):
    m = np.ones((len(s), 3), bool); m[:, 0] = s < E_MAX; m[:, 2] = s > 0; return m
def greedy_q(Q, idx):
    s = np.zeros(len(idx), int); tot = np.zeros(len(idx)); B = QB[idx - 1]
    for h in range(24):
        q = Q[h, B[:, h], s].copy(); q[~mask_of(s)] = -np.inf; a = ACTS[q.argmax(1)]; tot += reward(P[idx, h], a); s = s - a
    return tot
def q_learning(seed, epochs=40):
    rng = np.random.default_rng(seed); Q = np.zeros((24, 4, E_MAX + 1, 3)); best = (-np.inf, None); hist = []
    for ep in range(epochs):
        eps = max(0.05, 1.0 - ep / (0.6 * epochs))
        for d in rng.permutation(iTR):
            s, b = 0, QB[d - 1]
            for h in range(24):
                feas = [k for k, a in enumerate(ACTS) if 0 <= s - a <= E_MAX]
                k = feas[rng.integers(len(feas))] if rng.random() < eps else feas[int(np.argmax(Q[h, b[h], s, feas]))]
                r_ = float(reward(P[d, h], ACTS[k])) / 100.0; s2 = s - ACTS[k]
                tgt = r_ if h == 23 else r_ + Q[h + 1, b[h + 1], s2, [j for j, a2 in enumerate(ACTS) if 0 <= s2 - a2 <= E_MAX]].max()
                Q[h, b[h], s, k] += 0.05 * (tgt - Q[h, b[h], s, k]); s = s2
        va = greedy_q(Q, iVA).mean(); hist.append(va)
        if va > best[0]: best = (va, Q.copy())
    return best[1], hist"""
MARKETS = """import torch, torch.nn as nn
m = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])
w = m.pivot(index='date', columns='ticker', values='close').loc[:'2026-10-31']
px = w['SP500'].dropna(); vix = w['VIX'].reindex(px.index).ffill()"""
TRADE = """lr_ = np.log(px).diff()
F = pd.DataFrame({'r1': lr_, 'r5': np.log(px).diff(5), 'r21': np.log(px).diff(21), 'r63': np.log(px).diff(63),
                  'rv21': lr_.rolling(21).std() * np.sqrt(252), 'vix': vix / 100, 'ma200': px / px.rolling(200).mean() - 1}).dropna()
fwd = lr_.shift(-1).reindex(F.index); F, fwd = F.iloc[:-1], fwd.iloc[:-1]
tr_m, va_m, te_m = F.index.year <= 2016, (F.index.year >= 2017) & (F.index.year <= 2019), F.index.year >= 2020
X = ((F - F[tr_m].mean()) / F[tr_m].std()).values.astype(np.float32); RET = fwd.values.astype(np.float32); COST = 0.0005
def perf(pos, ret):
    pos = np.asarray(pos, float); return pos * ret - COST * np.abs(np.diff(np.concatenate([[0.0], pos])))
def stats_of(x):
    eq = np.exp(np.cumsum(x)); dd = eq / np.maximum.accumulate(eq) - 1
    return {'ann_ret': 100 * (np.exp(x.mean() * 252) - 1), 'ann_vol': 100 * x.std() * np.sqrt(252),
            'sharpe': x.mean() / x.std() * np.sqrt(252), 'maxdd': 100 * dd.min()}
ite, iva, itr = np.where(te_m)[0], np.where(va_m)[0], np.where(tr_m)[0]
bh = perf(np.ones(len(ite)), RET[ite]); ma_pos = (F['ma200'].values[ite] > 0).astype(float); ma = perf(ma_pos, RET[ite])
class AC(nn.Module):
    def __init__(s, d):
        super().__init__(); s.body = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        s.pi = nn.Linear(64, 2); s.v = nn.Linear(64, 1)
    def forward(s, x):
        h = s.body(x); return s.pi(h), s.v(h).squeeze(-1)
def run_policy(net, idx):
    pos, prev = [], 0.0
    with torch.no_grad():
        for i in idx:
            a = int(net(torch.from_numpy(np.append(X[i], prev).astype(np.float32)))[0].argmax()); pos.append(float(a)); prev = float(a)
    return np.array(pos)
def ppo(seed, iters=60, B=64, T=126):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = AC(X.shape[1] + 1)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4); hist, best = [], (-np.inf, None)
    for it in range(iters):
        st = rng.choice(itr[:-T], B); prev = np.zeros(B, np.float32); O, A_, LP, R_, V_ = [], [], [], [], []
        with torch.no_grad():
            for t in range(T):
                o = torch.from_numpy(np.concatenate([X[st + t], prev[:, None]], 1).astype(np.float32)); lg, v = net(o)
                dist = torch.distributions.Categorical(logits=lg); a = dist.sample(); pos = a.numpy().astype(np.float32)
                R_.append(torch.from_numpy(100 * (pos * RET[st + t] - COST * np.abs(pos - prev))))
                O.append(o); A_.append(a); LP.append(dist.log_prob(a)); V_.append(v); prev = pos
            O, A_, LP, R_, V_ = map(torch.stack, (O, A_, LP, R_, V_)); adv = torch.zeros_like(R_); g = torch.zeros(B)
            for t in range(T - 1, -1, -1):
                vn = V_[t + 1] if t + 1 < T else torch.zeros(B); g = R_[t] + 0.99 * vn - V_[t] + 0.99 * 0.95 * g; adv[t] = g
            ret = adv + V_
        O, A_, LP, adv, ret = (z.reshape(T * B, *z.shape[2:]) for z in (O, A_, LP, adv, ret)); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(4):
            for b in torch.from_numpy(rng.permutation(T * B)).split(1024):
                lg, v = net(O[b]); dist = torch.distributions.Categorical(logits=lg); rho = torch.exp(dist.log_prob(A_[b]) - LP[b])
                loss = -torch.min(rho * adv[b], rho.clamp(0.8, 1.2) * adv[b]).mean() + 0.5 * ((v - ret[b]) ** 2).mean() - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if it % 5 == 4:
            s_tr = stats_of(perf(run_policy(net, itr[-756:]), RET[itr[-756:]]))['sharpe']; s_va = stats_of(perf(run_policy(net, iva), RET[iva]))['sharpe']
            hist.append((it + 1, s_tr, s_va))
            if s_va > best[0]: best = (s_va, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); return net, hist"""
HEDGE = """from scipy.stats import norm
S = px; VX = vix / 100; n = 21; lS = np.log(S.values); st = np.arange(len(S) - n)
paths = np.stack([np.exp(lS[i:i + n + 1] - lS[i]) for i in st]).astype(np.float32)
vols = np.stack([VX.values[i:i + n + 1] for i in st]).astype(np.float32); y0 = S.index[st].year
TAU = (n - np.arange(n + 1)) / 252.0
def bs_call(Sx, sg, tau):
    tau = np.maximum(tau, 1e-8); d1 = (np.log(Sx) + 0.5 * sg ** 2 * tau) / (sg * np.sqrt(tau))
    return Sx * norm.cdf(d1) - norm.cdf(d1 - sg * np.sqrt(tau)), norm.cdf(d1)
prem = bs_call(1.0, vols[:, 0], TAU[0])[0]; payoff = np.maximum(paths[:, -1] - 1.0, 0)
itr, ite = np.where(y0 <= 2016)[0], np.where(y0 >= 2020)[0]
def pnl_of(delta, c, idx):
    Pp = paths[idx]; z = np.zeros((len(delta), 1)); trade = np.abs(np.diff(np.concatenate([z, delta, z], 1), axis=1))
    return prem[idx] + (delta * np.diff(Pp, axis=1)).sum(1) - payoff[idx] - c * (trade * Pp).sum(1)
def es(loss, a=0.95): q = np.quantile(loss, a); return float(loss[loss >= q].mean())
bs_delta = np.stack([bs_call(paths[:, k], vols[:, k], TAU[k])[1] for k in range(n)], 1)
class Hedger(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x): return torch.sigmoid(s.f(x)).squeeze(-1)
def feats(Sk, vk, k, prev): return torch.stack([torch.log(Sk) * 10.0, torch.full_like(Sk, float(TAU[k])) * 12.0, vk * 4.0, prev], -1)
def deltas(net, idx):
    Pt, Vt = torch.from_numpy(paths[idx]), torch.from_numpy(vols[idx]); prev = torch.zeros(len(idx)); out = []
    for k in range(n):
        prev = net(feats(Pt[:, k], Vt[:, k], k, prev)); out.append(prev)
    return torch.stack(out, 1)
def train_hedger(seed, c, iters=600):
    torch.manual_seed(seed); net = Hedger(); w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam(list(net.parameters()) + [w], lr=3e-3); Pt = torch.from_numpy(paths[itr])
    pr = torch.from_numpy(prem[itr].astype(np.float32)); py = torch.from_numpy(payoff[itr].astype(np.float32))
    for it in range(iters):
        d = deltas(net, itr); z = torch.zeros(len(itr), 1); trd = torch.abs(torch.diff(torch.cat([z, d, z], 1), dim=1))
        loss = -100 * (pr + (d * (Pt[:, 1:] - Pt[:, :-1])).sum(1) - py - c * (trd * Pt).sum(1))
        obj = w + torch.relu(loss - w).mean() / 0.05; opt.zero_grad(); obj.backward(); opt.step()
    return net"""

SRC_P = "data/clean/power_da_prices_hourly_utc.csv (energy-charts.info, Fraunhofer ISE, from exchange and ENTSO-E data)"
SRC_M = "data/clean/markets_daily.csv (Yahoo Finance: SP500 and VIX closes)"
QS = []


def ql(chart, desc, kw, datafile, cells):
    q = Quantlet(k=10, chart=chart, description=desc, keywords=kw, datafile=datafile)
    for kind, txt in cells:
        (q.md if kind == "md" else q.code)(txt)
    QS.append(q)


SAVE = "fig.savefig('{c}.png', dpi=200, transparent=True, bbox_inches='tight'); plt.show()"

ql("spread_monthly", "Monthly mean of the daily max-min spread of day-ahead prices in Romania and Germany-Luxembourg, the income source of a battery.",
   "electricity prices, day-ahead market, price spread, battery storage, Romania, reinforcement learning", SRC_P,
   [("md", "Daily spread = max - min of the 24 hourly day-ahead prices of a delivery day (CET), days with 24 complete hours."),
    ("code", PRICES + "\nDE = day_matrix('DE_LU')\nsp = RO.max(axis=1) - RO.min(axis=1)\ny = sp.groupby(sp.index.year).mean()\n"
     "show('spread.ro_2019', 'mean daily spread RO 2019 (EUR/MWh)', round(float(y.loc[2019]), 1))\n"
     "show('spread.ro_2025', 'mean daily spread RO 2025 (EUR/MWh)', round(float(y.loc[2025]), 1))\n"
     "show('spread.ro_ytd', 'mean daily spread RO 2026 YTD (EUR/MWh)', round(float(y.loc[2026]), 1))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.4))\n"
     "for M, c, lab in ((RO, PAL['data'], 'Romania'), (DE, PAL['alt'], 'Germany-Luxembourg')):\n"
     "    s_ = (M.max(axis=1) - M.min(axis=1)).resample('MS').mean(); ax.plot(s_.index, s_.values, color=c, lw=1.3, label=lab)\n"
     "ax.set_ylabel('Daily max - min (EUR/MWh)'); legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="spread_monthly"))])

ql("dp_revenue_year", "Perfect-foresight revenue of a 2 MWh battery (85 % round trip) by year and bidding zone, by exact dynamic programming (Bellman backward induction).",
   "dynamic programming, Bellman equation, battery arbitrage, perfect foresight, electricity prices, reinforcement learning", SRC_P,
   [("md", "$V_h(s)=\\max_a\\{r_h(a)+V_{h+1}(s-a)\\}$, $V_{24}=0$; revenue of a day = $V_0(0)$."),
    ("code", PRICES + "\n" + BATTERY + "\nout = {}\nfor z in ('RO', 'HU', 'DE_LU', 'BG'):\n    M = day_matrix(z); V, _ = dp_days(M.values)\n"
     "    rev = pd.Series(V[:, 0, 0], index=M.index); out[z] = rev.groupby(rev.index.year).sum() / 1000\n"
     "show('dp.ro_2025_k', 'RO 2025 (thousand EUR)', round(float(out['RO'].loc[2025]), 1))\n"
     "show('dp.ro_2019_k', 'RO 2019 (thousand EUR)', round(float(out['RO'].loc[2019]), 1))\n"
     "show('dp.hu_2025_k', 'HU 2025 (thousand EUR)', round(float(out['HU'].loc[2025]), 1))\n"
     "show('dp.de_2025_k', 'DE_LU 2025 (thousand EUR)', round(float(out['DE_LU'].loc[2025]), 1))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.4)); years = sorted(out['RO'].index)\n"
     "for i, (z, c) in enumerate(zip(('RO', 'HU', 'DE_LU', 'BG'), (PAL['data'], PAL['benchmark'], PAL['alt'], '#6250d6'))):\n"
     "    ys = [y for y in years if y in out[z].index]; ax.bar(np.array(ys) + (i - 1.5) * 0.2, out[z].loc[ys].values, width=0.2, color=c, label=z)\n"
     "ax.set_xticks(years); ax.set_xticklabels([str(y) for y in years], rotation=45); ax.set_ylabel('Revenue (thousand EUR)')\n"
     "legend_bottom(fig); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="dp_revenue_year"))])

ql("dp_day", "Optimal charge and discharge schedule of a 2 MWh battery on the 2026 day with the largest price spread in Romania (exact dynamic programming).",
   "dynamic programming, battery dispatch, electricity prices, optimal control, Romania", SRC_P,
   [("code", PRICES + "\n" + BATTERY + "\nV, A = dp_days(RO.values)\nr26 = RO.loc['2026']; day = (r26.max(axis=1) - r26.min(axis=1)).idxmax(); di = RO.index.get_loc(day)\n"
     "s, plan = 0, []\nfor h in range(24):\n    a = int(A[di, h, s]); plan.append(a); s -= a\np = RO.loc[day].values\n"
     "show('day.date', 'day', str(day.date()))\nshow('day.rev', 'revenue of the day (EUR)', round(float(V[di, 0, 0]), 0))\n"
     "show('day.pmax', 'maximum price (EUR/MWh)', round(float(p.max()), 2))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.5))\ncols = [PAL['benchmark'] if a < 0 else (PAL['positive'] if a > 0 else '#c8c8c8') for a in plan]\n"
     "ax.bar(range(24), p, color=cols, width=0.8)\nfor c, lab in ((PAL['benchmark'], 'charge (buy)'), (PAL['positive'], 'discharge (sell)'), ('#c8c8c8', 'idle')):\n"
     "    ax.bar([0], [0], color=c, label=lab)\nax.set_xlabel('Delivery hour (CET)'); ax.set_ylabel('Price (EUR/MWh)')\n"
     "legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="dp_day"))])

ql("dp_value", "Value function V_h(s) of the battery problem on one real day: the euros still to be earned from hour h with s MWh stored (the shadow price of stored energy).",
   "value function, Bellman equation, dynamic programming, battery, shadow price", SRC_P,
   [("code", PRICES + "\n" + BATTERY + "\nV, A = dp_days(RO.values)\nr26 = RO.loc['2026']; day = (r26.max(axis=1) - r26.min(axis=1)).idxmax(); di = RO.index.get_loc(day)\n"
     "show('day.v12_s0', 'V_12(0) (EUR)', round(float(V[di, 12, 0]), 0))\nshow('day.v12_s2', 'V_12(2) (EUR)', round(float(V[di, 12, 2]), 0))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nfor s_, c in zip(range(3), (PAL['data'], PAL['model'], '#6250d6')):\n"
     "    ax.plot(range(25), V[di, :, s_] / 1000, marker='o', ms=2.5, color=c, label=f'stored energy {s_} MWh')\n"
     "ax.set_xlabel('Hour h (CET)'); ax.set_ylabel('$V_h(s)$ (thousand EUR)'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="dp_value"))])

ql("q_policy", "Policy learned by tabular Q-learning for a battery that sees only yesterday's price ranking: action by hour and price quartile with 1 MWh stored.",
   "Q-learning, temporal difference, tabular reinforcement learning, battery dispatch, electricity prices", SRC_P,
   [("code", PRICES + "\n" + BATTERY + "\n" + SPLIT + "\nQ, hist = q_learning(20261108)\nq_te = greedy_q(Q, iTE)\n"
     "show('split.test_days', 'test days', int(len(iTE)))\nshow('bat.pf_k', 'perfect foresight, test (thousand EUR)', round(float(V[iTE, 0, 0].sum() / 1000), 1))\n"
     "print('Q-learning, first seed, test (thousand EUR):', round(q_te.sum() / 1000, 1))\n"
     "pol = np.array([[ACTS[int(np.argmax(Q[h, b, 1]))] for h in range(24)] for b in range(4)])\n"
     "from matplotlib.colors import ListedColormap\ncm = ListedColormap([PAL['benchmark'], '#e6e6e6', PAL['positive']])\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.0)); ax.imshow(pol, aspect='auto', cmap=cm, vmin=-1, vmax=1, origin='lower')\n"
     "ax.set_yticks(range(4)); ax.set_yticklabels(['cheapest 6 h', '2nd quartile', '3rd quartile', 'dearest 6 h'], fontsize=9)\n"
     "for v, lab in ((-1, 'charge'), (0, 'idle'), (1, 'discharge')): ax.bar([0], [0], color=cm((v + 1) / 2), label=lab)\n"
     "ax.set_xlabel('Hour h (CET); stored energy 1 MWh'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="q_policy"))])

DQN = """def zz(v): return (v - v.mean()) / (v.std() + 1.0)
FEAT = {d: np.concatenate([zz(P[d-1]), zz(P[d-7]), np.eye(7)[days[d].dayofweek]]) for d in np.concatenate([iTR, iVA, iTE])}
def obs(dl, h, s):
    return np.concatenate([np.stack([FEAT[d] for d in dl]), np.tile(np.eye(24)[h], (len(dl), 1)), np.eye(E_MAX + 1)[s]], 1).astype(np.float32)
d_in = obs(iTR[:1], 0, np.zeros(1, int)).shape[1]
def mk(): return nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))
def greedy_dqn(net, idx):
    s = np.zeros(len(idx), int); tot = np.zeros(len(idx))
    with torch.no_grad():
        for h in range(24):
            q = net(torch.from_numpy(obs(idx, h, s))).numpy(); q[~mask_of(s)] = -1e9; a = ACTS[q.argmax(1)]
            tot += reward(P[idx, h], a); s = s - a
    return tot
def dqn(seed, epochs=15, updates=20):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net, tgt = mk(), mk(); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=3e-4); buf = []; best = (-np.inf, None); hist = []
    for ep in range(epochs):
        eps = max(0.05, 1 - ep / (0.6 * epochs)); s = np.zeros(len(iTR), int)
        for h in range(24):
            o = obs(iTR, h, s); m = mask_of(s)
            with torch.no_grad(): q = net(torch.from_numpy(o)).numpy()
            q[~m] = -1e9; k = q.argmax(1); rnd = rng.random(len(iTR)) < eps; k[rnd] = (rng.random((rnd.sum(), 3)) * m[rnd]).argmax(1)
            a = ACTS[k]; s2 = s - a
            buf.append((o, k, reward(P[iTR, h], a) / 100, obs(iTR, min(h + 1, 23), s2), mask_of(s2), np.full(len(iTR), h == 23)))
            buf = buf[-96:]; s = s2; O, K, R, O2, M2, DN = (np.concatenate(z) for z in zip(*buf))
            for _ in range(updates):
                j = rng.integers(0, len(O), 512)
                with torch.no_grad():
                    q2 = net(torch.from_numpy(O2[j])); q2[torch.from_numpy(~M2[j])] = -1e9
                    y = torch.from_numpy(R[j].astype(np.float32)) + tgt(torch.from_numpy(O2[j])).gather(1, q2.argmax(1, keepdim=True)).squeeze(1) * torch.from_numpy(~DN[j]).float()
                loss = nn.functional.smooth_l1_loss(net(torch.from_numpy(O[j])).gather(1, torch.from_numpy(K[j]).long()[:, None]).squeeze(1), y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
                with torch.no_grad():
                    for pt, pn in zip(tgt.parameters(), net.parameters()): pt.mul_(0.995).add_(0.005 * pn)
        va = greedy_dqn(net, iVA).mean(); hist.append(va)
        if va > best[0]: best = (va, {k_: v.clone() for k_, v in net.state_dict().items()})
    net.load_state_dict(best[1]); return net, hist"""

ql("battery_test_cum", "Cumulative test revenue (2025 - October 2026) of a battery run by a double deep Q-network and by tabular Q-learning, against perfect foresight and yesterday's optimal plan.",
   "deep Q-network, double DQN, experience replay, Q-learning, battery arbitrage, out-of-sample test", SRC_P,
   [("md", "One seed and a smaller training budget than the slides (five seeds); the benchmarks are deterministic and match the slides."),
    ("code", PRICES + "\n" + BATTERY + "\n" + SPLIT + "\n" + DQN + "\nQ, _ = q_learning(20261108); q_te = greedy_q(Q, iTE)\nnet, _ = dqn(20261108); d_te = greedy_dqn(net, iTE)\n"
     "pf, nv = V[iTE, 0, 0], naive[iTE]\nshow('bat.pf_k', 'perfect foresight (thousand EUR)', round(float(pf.sum() / 1000), 1))\n"
     "show('bat.naive_k', \"yesterday's plan (thousand EUR)\", round(float(nv.sum() / 1000), 1))\n"
     "show('bat.naive_cap', \"capture ratio of yesterday's plan (%)\", round(float(100 * nv.sum() / pf.sum()), 1))\n"
     "print('one seed: DQN', round(d_te.sum() / 1000, 1), '| Q-learning', round(q_te.sum() / 1000, 1), 'thousand EUR')\n"
     "dd = days[iTE]; fig, ax = plt.subplots(figsize=(6.4, 2.5))\n"
     "for v, c, lab in ((pf, PAL['data'], 'perfect foresight (upper bound)'), (d_te, PAL['model'], 'double DQN (one seed)'),\n"
     "                  (q_te, PAL['alt'], 'tabular Q-learning (one seed)'), (nv, PAL['benchmark'], \"yesterday's optimal plan\")):\n"
     "    ax.plot(dd, np.cumsum(v) / 1000, color=c, label=lab)\nax.set_ylabel('Cumulative revenue (thousand EUR)')\n"
     "legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.16, 1, 1))\n" + SAVE.format(c="battery_test_cum"))])

ql("battery_learning", "Learning curves of tabular Q-learning and a double deep Q-network for battery dispatch: revenue per day on the 2024 validation year after each training epoch.",
   "learning curve, Q-learning, deep Q-network, validation, battery dispatch", SRC_P,
   [("code", PRICES + "\n" + BATTERY + "\n" + SPLIT + "\n" + DQN + "\n_, hq = q_learning(20261108)\n_, hd = dqn(20261108)\n"
     "show('bat.naive_valid_day', \"yesterday's plan, 2024 (EUR per day)\", round(float(naive[iVA].mean()), 0))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nax.plot(range(1, len(hd) + 1), hd, color=PAL['model'], label='double DQN (validation 2024)')\n"
     "ax.plot(range(1, len(hq) + 1), hq, color=PAL['alt'], label='tabular Q-learning (validation 2024)')\n"
     "ax.axhline(naive[iVA].mean(), color=PAL['benchmark'], ls='--', label=\"yesterday's optimal plan\")\n"
     "ax.set_xlabel('Training epoch'); ax.set_ylabel('EUR per day'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="battery_learning"))])

ql("trading_learning", "PPO actor-critic agent trading the S&P 500 (flat or long, 5 bp cost): Sharpe ratio on training and validation years during training.",
   "proximal policy optimisation, PPO, actor-critic, policy gradient, trading, S&P 500, validation", SRC_M,
   [("code", MARKETS + "\n" + TRADE + "\nnet, hist = ppo(20261108)\nH = np.array(hist)\n"
     "show('trade.bh_valid_sharpe', 'buy-and-hold Sharpe 2017-2019', round(float(stats_of(perf(np.ones(len(iva)), RET[iva]))['sharpe']), 2))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nax.plot(H[:, 0], H[:, 1], color=PAL['train'], label='training (2014-2016)')\n"
     "ax.plot(H[:, 0], H[:, 2], color=PAL['valid'], label='validation (2017-2019)')\n"
     "ax.set_xlabel('PPO iteration'); ax.set_ylabel('Sharpe ratio (annualised)'); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="trading_learning"))])

ql("trading_equity", "Test-period value of 1 USD: PPO agent (one seed) against buy-and-hold and the 200-day moving-average rule on the S&P 500, after costs.",
   "PPO, reinforcement learning trading, buy and hold, moving average rule, Sharpe ratio, backtest", SRC_M,
   [("code", MARKETS + "\n" + TRADE + "\nnet, _ = ppo(20261108)\npos = run_policy(net, ite); pp = perf(pos, RET[ite])\n"
     "show('trade.bh_sharpe', 'buy-and-hold Sharpe', round(float(stats_of(bh)['sharpe']), 2))\n"
     "show('trade.ma_sharpe', 'moving-average Sharpe', round(float(stats_of(ma)['sharpe']), 2))\n"
     "show('trade.bh_dd', 'buy-and-hold maximum drawdown (%)', round(float(stats_of(bh)['maxdd']), 1))\n"
     "print('PPO, one seed: Sharpe', round(stats_of(pp)['sharpe'], 2))\n"
     "dte = F.index[ite]; fig, ax = plt.subplots(figsize=(6.4, 2.5))\nax.plot(dte, np.exp(np.cumsum(bh)), color=PAL['data'], label='buy and hold')\n"
     "ax.plot(dte, np.exp(np.cumsum(ma)), color=PAL['benchmark'], label='200-day moving-average rule')\n"
     "ax.plot(dte, np.exp(np.cumsum(pp)), color=PAL['model'], label='PPO agent (one seed)')\n"
     "ax.set_ylabel('Value of 1 USD'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="trading_equity"))])

ql("trading_exposure", "Share of capital invested by a PPO trading agent on the S&P 500 (21-day mean) against the 200-day moving-average rule, test period from 2020.",
   "PPO, policy interpretation, market exposure, trend following, S&P 500", SRC_M,
   [("code", MARKETS + "\n" + TRADE + "\nnet, _ = ppo(20261108)\npos = run_policy(net, ite)\n"
     "show('trade.ma_inv', 'moving-average rule: days invested (%)', round(float(100 * ma_pos.mean()), 0))\n"
     "show('trade.ma_trades', 'moving-average rule: trades', int(np.abs(np.diff(np.concatenate([[0], ma_pos]))).sum()))\n"
     "dte = F.index[ite]; fig, ax = plt.subplots(figsize=(6.4, 2.3))\n"
     "ax.plot(dte, 100 * pd.Series(pos, index=dte).rolling(21).mean(), color=PAL['model'], label='PPO exposure (one seed, 21-day mean)')\n"
     "ax.plot(dte, 100 * pd.Series(ma_pos, index=dte).rolling(21).mean(), color=PAL['benchmark'], lw=1.0, label='moving-average rule')\n"
     "ax.set_ylabel('% invested'); ax.set_ylim(-3, 103); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="trading_exposure"))])

ql("ppo_clip", "The clipped surrogate objective of proximal policy optimisation (PPO) as a function of the probability ratio, for a positive and a negative advantage (analytic, no data).",
   "PPO, clipped surrogate objective, policy gradient, trust region, mechanism", "none (analytic function)",
   [("code", "show('trade.cost_bp', 'trading cost used elsewhere in the lecture (bp)', 5)\n"
     "rr = np.linspace(0.4, 1.6, 400); fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.2))\n"
     "for ax, A_, nm in ((axs[0], 1.0, 'A>0'), (axs[1], -1.0, 'A<0')):\n"
     "    ax.plot(rr, rr * A_, color='#8a8a8a', ls='--', lw=1.0, label='unclipped rho A')\n"
     "    ax.plot(rr, np.minimum(rr * A_, np.clip(rr, 0.8, 1.2) * A_), color=PAL['model'], lw=1.8, label='PPO, epsilon = 0.2')\n"
     "    ax.axvline(1, color='#8a8a8a', lw=0.6); ax.set_xlabel(f'probability ratio rho ({nm})')\n"
     "axs[0].set_ylabel('objective'); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="ppo_clip"))])

ql("hedge_pnl", "Hedging profit and loss of a short one-month at-the-money call on the S&P 500 on real 21-day paths (test from 2020): no hedge, Black-Scholes delta with the VIX, deep hedge.",
   "deep hedging, expected shortfall, Black-Scholes delta, option hedging, transaction costs, S&P 500, VIX", SRC_M,
   [("code", MARKETS + "\n" + HEDGE + "\nC = 0.001\nnh = 100 * pnl_of(np.zeros_like(bs_delta[ite]), C, ite); bsd = 100 * pnl_of(bs_delta[ite], C, ite)\n"
     "net = train_hedger(20261108, C)\nwith torch.no_grad(): dh = 100 * pnl_of(deltas(net, ite).numpy(), C, ite)\n"
     "show('hedge.n_test', 'test paths', int(len(ite)))\nshow('hedge.nh_std', 'no hedge: std (%)', round(float(nh.std()), 2))\n"
     "show('hedge.bs_std', 'delta hedge: std (%)', round(float(bsd.std()), 2))\nshow('hedge.bs_es', 'delta hedge: ES95 of loss (%)', round(es(-bsd), 2))\n"
     "show('hedge.nh_es', 'no hedge: ES95 of loss (%)', round(es(-nh), 2))\nprint('deep hedge, one seed: std', round(dh.std(), 2), 'ES95', round(es(-dh), 2))\n"
     "bins = np.linspace(-8, 4, 61); fig, ax = plt.subplots(figsize=(6.4, 2.4))\n"
     "ax.hist(np.clip(nh, -8, 4), bins=bins, color='#8a8a8a', alpha=0.5, label='no hedge')\n"
     "ax.hist(np.clip(bsd, -8, 4), bins=bins, histtype='step', lw=1.6, color=PAL['benchmark'], label='Black-Scholes delta')\n"
     "ax.hist(np.clip(dh, -8, 4), bins=bins, histtype='step', lw=1.6, color=PAL['model'], label='deep hedge (one seed)')\n"
     "ax.set_xlabel('Hedging profit and loss (% of the index)'); ax.set_ylabel('Start days'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="hedge_pnl"))])

ql("hedge_delta", "Hedge ratio learned by a deep-hedging network against the Black-Scholes delta, 10 trading days before maturity with the VIX at 20.",
   "deep hedging, hedge ratio, Black-Scholes delta, no-trade band, transaction costs", SRC_M,
   [("code", MARKETS + "\n" + HEDGE + "\nnet = train_hedger(20261108, 0.001)\ngrid = np.linspace(0.9, 1.1, 41).astype(np.float32); k10 = n - 10\n"
     "prev = torch.from_numpy(bs_call(grid, 0.2, TAU[k10])[1].astype(np.float32))\n"
     "with torch.no_grad(): dg = net(feats(torch.from_numpy(grid), torch.full((41,), 0.2), k10, prev)).numpy()\n"
     "show('hedge.atm_bs', 'Black-Scholes delta at the money', round(float(bs_call(1.0, 0.2, 10 / 252)[1]), 2))\n"
     "print('deep hedge at the money, one seed:', round(float(dg[20]), 2))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.3)); ax.plot(grid, dg, color=PAL['model'], label='deep hedge (one seed)')\n"
     "ax.plot(grid, bs_call(grid, 0.2, 10 / 252)[1], color=PAL['benchmark'], ls='--', label='Black-Scholes delta')\n"
     "ax.set_xlabel('Moneyness S/K (10 days to maturity, VIX = 20)'); ax.set_ylabel('Hedge ratio'); legend_bottom(fig, ncol=2)\n"
     "fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="hedge_delta"))])

ql("hedge_costs", "Expected shortfall (95 %) of the hedging loss as a function of the transaction cost: Black-Scholes delta against a deep hedge trained for each cost level.",
   "deep hedging, transaction costs, expected shortfall, Rockafellar-Uryasev, option hedging", SRC_M,
   [("code", MARKETS + "\n" + HEDGE + "\nCG = [0.0, 0.0005, 0.001, 0.002, 0.005]\nbs_es = [es(-100 * pnl_of(bs_delta[ite], c, ite)) for c in CG]\n"
     "dh_es = []\nfor c in CG:\n    net = train_hedger(20261108, c, iters=300)\n    with torch.no_grad(): dh_es.append(es(-100 * pnl_of(deltas(net, ite).numpy(), c, ite)))\n"
     "show('hedge.bs_es_0', 'delta hedge ES95, no cost (%)', round(bs_es[0], 2))\nshow('hedge.bs_es_50', 'delta hedge ES95, 50 bp (%)', round(bs_es[-1], 2))\n"
     "print('deep hedge, one seed, 300 steps:', [round(v, 2) for v in dh_es])\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.3)); xb = [c * 1e4 for c in CG]\n"
     "ax.plot(xb, bs_es, marker='o', color=PAL['benchmark'], label='Black-Scholes delta'); ax.plot(xb, dh_es, marker='o', color=PAL['model'], label='deep hedge (one seed)')\n"
     "ax.set_xlabel('Transaction cost (bp)'); ax.set_ylabel('ES95 of the loss (%)'); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, 0.14, 1, 1))\n" + SAVE.format(c="hedge_costs"))])

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
