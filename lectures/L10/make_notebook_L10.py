"""
make_notebook_L10.py -- Lecture 10 notebooks (EN + RO): lecture notebook (executed) and seminar notebooks.
The lecture notebook reproduces the method of every chart of the deck on the same real data; the learning agents are
trained with one seed and fewer epochs than in analysis_L10.py (five seeds), so their numbers are close to, not equal
to, the slides. Section numbers (§) are the "Experiment yourself" pointers of the slides.
Run:  python3 lectures/L10/make_notebook_L10.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Reinforcement learning: trading, hedging and battery dispatch",
         "Reinforcement learning: tranzacționare, acoperirea riscului și dispecerizarea bateriilor")

LOAD_PRICES = """import torch, torch.nn as nn
x = pd.read_csv(DATA + 'power_da_prices_hourly_utc.csv', parse_dates=['timestamp_utc']).set_index('timestamp_utc')
x.index = x.index.tz_convert('Europe/Brussels')
END = '2026-10-31'                                  # last complete month of the course data
x = x.loc[:END + ' 23:59']
x['day'] = x.index.tz_localize(None).normalize(); x['hour'] = x.index.hour

def day_matrix(zone):
    # delivery days (CET) x 24 hours; only days with 24 reported hours
    d = x[['day', 'hour', zone]].dropna()
    n = d.groupby('day')[zone].count()
    return d[d.day.isin(n[n == 24].index)].pivot_table(index='day', columns='hour', values=zone).sort_index()

RO = day_matrix('RO')
print(RO.index.min().date(), RO.index.max().date(), len(RO), 'complete days')"""

BATTERY = """E_MAX, ETA = 2, 0.85                 # 2 MWh, 1 MWh of stored energy per hour, 85 % round-trip efficiency
ETA_C = ETA_D = np.sqrt(ETA)
ACTS = np.array([-1, 0, 1])          # charge, idle, discharge

def reward(p, a):
    return np.where(a > 0, ETA_D * p, np.where(a < 0, -p / ETA_C, 0.0))

def dp_days(P):
    # backward induction V_h(s) = max_a { r(p_h, a) + V_{h+1}(s - a) }, all days at once
    D = P.shape[0]
    V = np.zeros((D, 25, E_MAX + 1)); A = np.zeros((D, 24, E_MAX + 1), dtype=int)
    for h in range(23, -1, -1):
        best = np.full((D, E_MAX + 1), -np.inf); arg = np.zeros((D, E_MAX + 1), dtype=int)
        for s in range(E_MAX + 1):
            for a in ACTS:
                if 0 <= s - a <= E_MAX:
                    q = reward(P[:, h], a) + V[:, h + 1, s - a]
                    b = q > best[:, s]; best[b, s] = q[b]; arg[b, s] = a
        V[:, h] = best; A[:, h] = arg
    return V, A"""


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 Data: Romanian day-ahead prices and the daily spread",
            "## §1 Datele: prețurile pentru ziua următoare din România și diferența zilnică"))
    nb.code(LOAD_PRICES)
    nb.code("spread = (RO.max(axis=1) - RO.min(axis=1))\n"
            "print(spread.groupby(spread.index.year).mean().round(1))\n"
            "fig, ax = plt.subplots(figsize=(6.4, 2.4))\n"
            "ax.plot(spread.resample('MS').mean(), label='RO: monthly mean of max - min')\n"
            "ax.set_ylabel('EUR/MWh'); fig.legend(loc='lower center'); fig.tight_layout(rect=(0, 0.1, 1, 1)); plt.show()")
    nb.md(T("## §2 Dynamic programming: the perfect-foresight battery",
            "## §2 Programarea dinamică: bateria cu previziune perfectă"))
    nb.code(BATTERY)
    nb.code("V, A = dp_days(RO.values)\nrev = pd.Series(V[:, 0, 0], index=RO.index)\n"
            "print('perfect-foresight revenue per year (thousand EUR):')\nprint((rev.groupby(rev.index.year).sum() / 1000).round(1))")
    nb.code("day = (RO.loc['2026'].max(axis=1) - RO.loc['2026'].min(axis=1)).idxmax(); di = RO.index.get_loc(day)\n"
            "s, plan = 0, []\nfor h in range(24):\n    a = int(A[di, h, s]); plan.append(a); s -= a\n"
            "print(day.date(), 'revenue', round(V[di, 0, 0]), 'EUR; plan:', plan)\n"
            "print('value of the stored energy at noon (EUR per MWh):', round((V[di, 12, 2] - V[di, 12, 0]) / 2, 2))")
    nb.md(T("## §3 Tabular Q-learning with yesterday's prices",
            "## §3 Q-learning tabelar cu prețurile de ieri"))
    nb.code("P = RO.values; days = RO.index\n"
            "ok = np.array([i >= 7 and (days[i] - days[i-1]).days == 1 and (days[i] - days[i-7]).days == 7 for i in range(len(days))])\n"
            "iTR = np.where(ok & (days.year >= 2019) & (days.year <= 2023))[0]\n"
            "iVA = np.where(ok & (days.year == 2024))[0]\niTE = np.where(ok & (days.year >= 2025))[0]\n"
            "QB = P.argsort(1).argsort(1) // 6           # quartile of each hour's price within its day\n"
            "def mask_of(s):\n    m = np.ones((len(s), 3), bool); m[:, 0] = s < E_MAX; m[:, 2] = s > 0; return m\n"
            "def greedy_q(Q, idx):\n    s = np.zeros(len(idx), int); tot = np.zeros(len(idx)); B = QB[idx - 1]\n"
            "    for h in range(24):\n        q = Q[h, B[:, h], s].copy(); q[~mask_of(s)] = -np.inf; a = ACTS[q.argmax(1)]\n"
            "        tot += reward(P[idx, h], a); s = s - a\n    return tot\n"
            "rng = np.random.default_rng(SEED); Q = np.zeros((24, 4, E_MAX + 1, 3)); best = (-np.inf, None)\n"
            "for ep in range(15):\n    eps = max(0.05, 1 - ep / 9)\n    for d in rng.permutation(iTR):\n"
            "        s, b = 0, QB[d - 1]\n        for h in range(24):\n"
            "            feas = [k for k, a in enumerate(ACTS) if 0 <= s - a <= E_MAX]\n"
            "            k = feas[rng.integers(len(feas))] if rng.random() < eps else feas[int(np.argmax(Q[h, b[h], s, feas]))]\n"
            "            r_ = float(reward(P[d, h], ACTS[k])) / 100; s2 = s - ACTS[k]\n"
            "            tgt = r_ if h == 23 else r_ + Q[h + 1, b[h + 1], s2, [j for j, a2 in enumerate(ACTS) if 0 <= s2 - a2 <= E_MAX]].max()\n"
            "            Q[h, b[h], s, k] += 0.05 * (tgt - Q[h, b[h], s, k]); s = s2\n"
            "    va = greedy_q(Q, iVA).mean()\n    if va > best[0]: best = (va, Q.copy())\n"
            "Q = best[1]; V, _ = dp_days(P)\nq_te = greedy_q(Q, iTE)\n"
            "print('test revenue, Q-learning (thousand EUR):', round(q_te.sum() / 1000, 1), '| perfect foresight:', round(V[iTE, 0, 0].sum() / 1000, 1))\n"
            "print('capture ratio (%):', round(100 * q_te.sum() / V[iTE, 0, 0].sum(), 1))")
    nb.md(T("## §4 Double deep Q-network (DQN)", "## §4 Rețeaua Q profundă dublă (DQN)"))
    nb.code("def zz(v): return (v - v.mean()) / (v.std() + 1.0)\n"
            "FEAT = {d: np.concatenate([zz(P[d-1]), zz(P[d-7]), np.eye(7)[days[d].dayofweek]]) for d in np.concatenate([iTR, iVA, iTE])}\n"
            "def obs(dl, h, s):\n    return np.concatenate([np.stack([FEAT[d] for d in dl]), np.tile(np.eye(24)[h], (len(dl), 1)), np.eye(E_MAX + 1)[s]], 1).astype(np.float32)\n"
            "d_in = obs(iTR[:1], 0, np.zeros(1, int)).shape[1]\n"
            "mk = lambda: nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))\n"
            "torch.manual_seed(SEED); net, tgt = mk(), mk(); tgt.load_state_dict(net.state_dict())\n"
            "opt = torch.optim.Adam(net.parameters(), lr=5e-4)\n"
            "def greedy(idx):\n    s = np.zeros(len(idx), int); tot = np.zeros(len(idx))\n    with torch.no_grad():\n"
            "        for h in range(24):\n            q = net(torch.from_numpy(obs(idx, h, s))).numpy(); q[~mask_of(s)] = -1e9\n"
            "            a = ACTS[q.argmax(1)]; tot += reward(P[idx, h], a); s = s - a\n    return tot\n"
            "buf = []; best = (-np.inf, None)\n"
            "for ep in range(20):\n    eps = max(0.05, 1 - ep / 12); s = np.zeros(len(iTR), int)\n    for h in range(24):\n"
            "        o = obs(iTR, h, s); m = mask_of(s)\n        with torch.no_grad(): q = net(torch.from_numpy(o)).numpy()\n"
            "        q[~m] = -1e9; k = q.argmax(1); rnd = rng.random(len(iTR)) < eps\n"
            "        k[rnd] = (rng.random((rnd.sum(), 3)) * m[rnd]).argmax(1); a = ACTS[k]; s2 = s - a\n"
            "        buf.append((o, k, reward(P[iTR, h], a) / 100, obs(iTR, min(h + 1, 23), s2), mask_of(s2), np.full(len(iTR), h == 23)))\n"
            "        buf = buf[-96:]; s = s2\n"
            "        O, K, R, O2, M2, DN = (np.concatenate(z) for z in zip(*buf))\n"
            "        for _ in range(6):\n            j = rng.integers(0, len(O), 512)\n            with torch.no_grad():\n"
            "                q2 = net(torch.from_numpy(O2[j])); q2[torch.from_numpy(~M2[j])] = -1e9\n"
            "                y = torch.from_numpy(R[j].astype(np.float32)) + tgt(torch.from_numpy(O2[j])).gather(1, q2.argmax(1, keepdim=True)).squeeze(1) * torch.from_numpy(~DN[j]).float()\n"
            "            loss = nn.functional.smooth_l1_loss(net(torch.from_numpy(O[j])).gather(1, torch.from_numpy(K[j]).long()[:, None]).squeeze(1), y)\n"
            "            opt.zero_grad(); loss.backward(); opt.step()\n"
            "            with torch.no_grad():\n                for pt, pn in zip(tgt.parameters(), net.parameters()): pt.mul_(0.99).add_(0.01 * pn)\n"
            "    va = greedy(iVA).mean()\n    if va > best[0]: best = (va, {k_: v.clone() for k_, v in net.state_dict().items()})\n"
            "net.load_state_dict(best[1]); d_te = greedy(iTE)\n"
            "print('test revenue, DQN (thousand EUR):', round(d_te.sum() / 1000, 1), '| capture ratio (%):', round(100 * d_te.sum() / V[iTE, 0, 0].sum(), 1))")
    nb.code("fig, ax = plt.subplots(figsize=(6.4, 2.5))\nax.plot(days[iTE], np.cumsum(V[iTE, 0, 0]) / 1000, label='perfect foresight')\n"
            "ax.plot(days[iTE], np.cumsum(d_te) / 1000, label='double DQN')\nax.plot(days[iTE], np.cumsum(q_te) / 1000, label='Q-learning')\n"
            "ax.set_ylabel('thousand EUR'); fig.legend(loc='lower center', ncol=3); fig.tight_layout(rect=(0, 0.12, 1, 1)); plt.show()")
    nb.md(T("## §5 PPO for trading the S&P 500", "## §5 PPO pentru tranzacționarea indicelui S&P 500"))
    nb.code("m = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])\n"
            "w = m.pivot(index='date', columns='ticker', values='close').loc[:END]\npx = w['SP500'].dropna(); vix = w['VIX'].reindex(px.index).ffill()\n"
            "lr = np.log(px).diff()\nF = pd.DataFrame({'r1': lr, 'r5': np.log(px).diff(5), 'r21': np.log(px).diff(21), 'r63': np.log(px).diff(63),\n"
            "                  'rv21': lr.rolling(21).std() * np.sqrt(252), 'vix': vix / 100, 'ma200': px / px.rolling(200).mean() - 1}).dropna()\n"
            "fwd = lr.shift(-1).reindex(F.index); F, fwd = F.iloc[:-1], fwd.iloc[:-1]\n"
            "tr, va, te = F.index.year <= 2016, (F.index.year >= 2017) & (F.index.year <= 2019), F.index.year >= 2020\n"
            "X = ((F - F[tr].mean()) / F[tr].std()).values.astype(np.float32); RET = fwd.values.astype(np.float32); COST = 0.0005\n"
            "def perf(pos, ret):\n    pos = np.asarray(pos, float); return pos * ret - COST * np.abs(np.diff(np.concatenate([[0.], pos])))\n"
            "def sharpe(x): return x.mean() / x.std() * np.sqrt(252)\nprint(F.index[te][0].date(), F.index[te][-1].date())")
    nb.code("class AC(nn.Module):\n    def __init__(s, d):\n        super().__init__(); s.b = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())\n"
            "        s.pi = nn.Linear(64, 2); s.v = nn.Linear(64, 1)\n    def forward(s, x):\n        h = s.b(x); return s.pi(h), s.v(h).squeeze(-1)\n"
            "def run(net, idx):\n    pos, prev = [], 0.\n    with torch.no_grad():\n        for i in idx:\n"
            "            a = int(net(torch.from_numpy(np.append(X[i], prev).astype(np.float32)))[0].argmax()); pos.append(a); prev = float(a)\n    return np.array(pos)\n"
            "torch.manual_seed(SEED); ac = AC(X.shape[1] + 1); opt = torch.optim.Adam(ac.parameters(), lr=3e-4)\n"
            "itr, iva, ite = np.where(tr)[0], np.where(va)[0], np.where(te)[0]; B, Tn = 64, 126; best = (-np.inf, None)\n"
            "for it in range(60):\n    st = rng.choice(itr[:-Tn], B); prev = np.zeros(B, np.float32); O, A_, LP, R, V_ = [], [], [], [], []\n"
            "    with torch.no_grad():\n        for t in range(Tn):\n"
            "            o = torch.from_numpy(np.concatenate([X[st + t], prev[:, None]], 1).astype(np.float32)); lg, v = ac(o)\n"
            "            dist = torch.distributions.Categorical(logits=lg); a = dist.sample(); pos = a.numpy().astype(np.float32)\n"
            "            R.append(torch.from_numpy(100 * (pos * RET[st + t] - COST * np.abs(pos - prev)))); O.append(o); A_.append(a); LP.append(dist.log_prob(a)); V_.append(v); prev = pos\n"
            "        O, A_, LP, R, V_ = map(torch.stack, (O, A_, LP, R, V_)); adv = torch.zeros_like(R); g = torch.zeros(B)\n"
            "        for t in range(Tn - 1, -1, -1):\n            vn = V_[t + 1] if t + 1 < Tn else torch.zeros(B)\n"
            "            g = R[t] + 0.99 * vn - V_[t] + 0.99 * 0.95 * g; adv[t] = g\n        ret = adv + V_\n"
            "    O, A_, LP, adv, ret = (z.reshape(Tn * B, *z.shape[2:]) for z in (O, A_, LP, adv, ret)); adv = (adv - adv.mean()) / (adv.std() + 1e-8)\n"
            "    for _ in range(4):\n        for b in torch.from_numpy(rng.permutation(Tn * B)).split(1024):\n"
            "            lg, v = ac(O[b]); dist = torch.distributions.Categorical(logits=lg); rho = torch.exp(dist.log_prob(A_[b]) - LP[b])\n"
            "            loss = -torch.min(rho * adv[b], rho.clamp(0.8, 1.2) * adv[b]).mean() + 0.5 * ((v - ret[b]) ** 2).mean() - 0.01 * dist.entropy().mean()\n"
            "            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()\n"
            "    if it % 5 == 4:\n        sv = sharpe(perf(run(ac, iva), RET[iva]))\n"
            "        if sv > best[0]: best = (sv, {k_: v.clone() for k_, v in ac.state_dict().items()})\n"
            "ac.load_state_dict(best[1]); pos = run(ac, ite)\n"
            "bh = perf(np.ones(len(ite)), RET[ite]); ma = perf((F['ma200'].values[ite] > 0).astype(float), RET[ite]); pp = perf(pos, RET[ite])\n"
            "print('test Sharpe  buy-and-hold', round(sharpe(bh), 2), '| MA rule', round(sharpe(ma), 2), '| PPO (one seed)', round(sharpe(pp), 2))\n"
            "print('PPO share of days invested (%):', round(100 * pos.mean()))")
    nb.md(T("## §6 Deep hedging of a one-month call", "## §6 Acoperirea neuronală a unei opțiuni call pe o lună"))
    nb.code("from scipy.stats import norm\nS = px; VX = vix / 100; n = 21; lS = np.log(S.values)\n"
            "st = np.arange(len(S) - n); paths = np.stack([np.exp(lS[i:i + n + 1] - lS[i]) for i in st]).astype(np.float32)\n"
            "vols = np.stack([VX.values[i:i + n + 1] for i in st]).astype(np.float32); y0 = S.index[st].year\n"
            "TAU = (n - np.arange(n + 1)) / 252\n"
            "def bs(Sx, sg, tau):\n    d1 = (np.log(Sx) + 0.5 * sg**2 * tau) / (sg * np.sqrt(tau)); return Sx * norm.cdf(d1) - norm.cdf(d1 - sg * np.sqrt(tau)), norm.cdf(d1)\n"
            "prem = bs(1.0, vols[:, 0], TAU[0])[0]; payoff = np.maximum(paths[:, -1] - 1, 0); C = 0.001\n"
            "def es(l, a=0.95): return l[l >= np.quantile(l, a)].mean()\n"
            "bsd = np.stack([bs(paths[:, k], vols[:, k], TAU[k])[1] for k in range(n)], 1)\n"
            "itr, ite = np.where(y0 <= 2016)[0], np.where(y0 >= 2020)[0]\n"
            "net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)); wv = torch.zeros(1, requires_grad=True)\n"
            "def deltas(idx):\n    Pt, Vt = torch.from_numpy(paths[idx]), torch.from_numpy(vols[idx]); prev = torch.zeros(len(idx)); out = []\n"
            "    for k in range(n):\n        f = torch.stack([10 * torch.log(Pt[:, k]), torch.full_like(prev, 12 * TAU[k]), 4 * Vt[:, k], prev], -1)\n"
            "        prev = torch.sigmoid(net(f)).squeeze(-1); out.append(prev)\n    return torch.stack(out, 1)\n"
            "torch.manual_seed(SEED); opt = torch.optim.Adam(list(net.parameters()) + [wv], lr=3e-3)\n"
            "Pt = torch.from_numpy(paths[itr]); pr = torch.from_numpy(prem[itr].astype(np.float32)); py = torch.from_numpy(payoff[itr].astype(np.float32))\n"
            "for it in range(300):\n    d = deltas(itr); z = torch.zeros(len(itr), 1); trd = torch.abs(torch.diff(torch.cat([z, d, z], 1), dim=1))\n"
            "    loss = -100 * (pr + (d * (Pt[:, 1:] - Pt[:, :-1])).sum(1) - py - C * (trd * Pt).sum(1))\n"
            "    obj = wv + torch.relu(loss - wv).mean() / 0.05; opt.zero_grad(); obj.backward(); opt.step()\n"
            "with torch.no_grad(): dh = deltas(ite).numpy()\n"
            "def pnl_te(d):\n    z = np.zeros((len(d), 1)); trd = np.abs(np.diff(np.concatenate([z, d, z], 1), axis=1))\n"
            "    return 100 * (prem[ite] + (d * np.diff(paths[ite], axis=1)).sum(1) - payoff[ite] - C * (trd * paths[ite]).sum(1))\n"
            "for name, d in (('no hedge', np.zeros_like(bsd[ite])), ('Black-Scholes delta', bsd[ite]), ('deep hedge', dh)):\n"
            "    p = pnl_te(d)\n"
            "    print(f'{name:22s} mean {p.mean():6.2f}  sd {p.std():5.2f}  ES95 of loss {es(-p):5.2f}  (% of the index level)')")
    nb.md(T("**Experiment yourself.** Change the round-trip efficiency, the transaction cost or the network size and re-run "
            "the section; compare with the five-seed results of the slides.",
            "**Exercițiu practic.** Modificați randamentul ciclului de stocare, costul de tranzacționare sau dimensiunea rețelei "
            "și rulați din nou secțiunea; comparați cu rezultatele pe cinci valori inițiale de pe slide-uri."))


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup: prices and the battery model", "## Pregătire: prețurile și modelul bateriei"))
    nb.code(LOAD_PRICES)
    nb.code(BATTERY)
    nb.task(T("Compute the perfect-foresight revenue of the battery for every year in Hungary (HU) and Germany-Luxembourg "
              "(DE_LU) and compare it with Romania. Which market paid the most in 2025?",
              "Calculați venitul bateriei cu previziune perfectă, pe ani, pentru Ungaria (HU) și Germania-Luxemburg (DE_LU) "
              "și comparați-l cu România. Care piață a plătit cel mai mult în 2025?"))
    nb.solution("for z in ('RO', 'HU', 'DE_LU'):\n    M = day_matrix(z); V, _ = dp_days(M.values)\n"
                "    rev = pd.Series(V[:, 0, 0], index=M.index); print(z, (rev.groupby(rev.index.year).sum() / 1000).round(1).loc[2023:].to_dict())")
    nb.task(T("Re-run the dynamic programme with a round-trip efficiency of 0.90 and 0.75. How much does the 2025 revenue in "
              "Romania change? Explain why efficiency matters more on days with a small spread.",
              "Rulați din nou programarea dinamică cu un randament al ciclului de 0,90 și 0,75. Cît se modifică venitul din 2025 "
              "în România? Explicați de ce randamentul contează mai mult în zilele cu o diferență mică de preț."))
    nb.solution("for eta in (0.90, 0.85, 0.75):\n    ETA_C = ETA_D = np.sqrt(eta)\n    V, _ = dp_days(RO.loc['2025'].values)\n"
                "    print(eta, round(V[:, 0, 0].sum() / 1000, 1), 'thousand EUR')\nETA_C = ETA_D = np.sqrt(0.85)")
    nb.task(T("Naive benchmark: apply yesterday's optimal plan to today's prices in 2025-2026. What share of the "
              "perfect-foresight revenue does it capture?",
              "Reperul naiv: aplicați planul optim de ieri la prețurile de azi, în 2025-2026. Ce parte din venitul cu "
              "previziune perfectă obține?"))
    nb.solution("P = RO.values; V, A = dp_days(P)\nplan = np.zeros((len(P), 24), int); s = np.zeros(len(P), int)\n"
                "for h in range(24):\n    a = A[np.arange(len(P)), h, s]; plan[:, h] = a; s = s - a\n"
                "def run_plan(Pd, acts):\n    s = np.zeros(len(Pd), int); tot = np.zeros(len(Pd))\n"
                "    for h in range(24):\n        a = acts[:, h].copy(); a[(a > 0) & (s == 0)] = 0; a[(a < 0) & (s == E_MAX)] = 0\n"
                "        tot += reward(Pd[:, h], a); s = s - a\n    return tot\n"
                "te = np.where((RO.index.year >= 2025)[1:])[0] + 1\n"
                "nv = run_plan(P[te], plan[te - 1]); print('capture ratio (%):', round(100 * nv.sum() / V[te, 0, 0].sum(), 1))")
    nb.task(T("A one-hour battery: set E_MAX = 1 and recompute the 2025 perfect-foresight revenue. Is the second hour of "
              "storage worth half of the revenue?",
              "O baterie de o oră: setați E_MAX = 1 și recalculați venitul cu previziune perfectă din 2025. Valorează a doua oră "
              "de stocare jumătate din venit?"))
    nb.solution("E_MAX = 1\nV1, _ = dp_days(RO.loc['2025'].values)\nE_MAX = 2\nV2, _ = dp_days(RO.loc['2025'].values)\n"
                "print('1 MWh:', round(V1[:, 0, 0].sum() / 1000, 1), '| 2 MWh:', round(V2[:, 0, 0].sum() / 1000, 1), 'thousand EUR')")
    nb.md(T("## Trading and hedging", "## Tranzacționare și acoperirea riscului"))
    nb.code("m = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])\n"
            "w = m.pivot(index='date', columns='ticker', values='close').loc[:END]\npx = w['SP500'].dropna(); vix = w['VIX'].reindex(px.index).ffill()\n"
            "r = np.log(px).diff().shift(-1).dropna()\nma = (px / px.rolling(200).mean() - 1).reindex(r.index)")
    nb.task(T("Evaluate the 200-day moving-average rule on 2020 - October 2026 with costs of 0, 5, 20 and 50 basis points per trade "
              "and compare its Sharpe ratio with buy-and-hold.",
              "Evaluați regula mediei mobile de 200 de zile pe 2020 - octombrie 2026, cu costuri de 0, 5, 20 și 50 de puncte de bază pe "
              "tranzacție, și comparați raportul Sharpe cu strategia cumpără și păstrează."))
    nb.solution("te = r.index.year >= 2020\npos = (ma[te] > 0).astype(float).values; ret = r[te].values\n"
                "sh = lambda x: x.mean() / x.std() * np.sqrt(252)\nprint('buy and hold', round(sh(ret), 2))\n"
                "for c in (0, 5, 20, 50):\n    x = pos * ret - c / 1e4 * np.abs(np.diff(np.concatenate([[0], pos])))\n    print(c, 'bp:', round(sh(x), 2))")
    nb.task(T("Delta-hedge the one-month at-the-money call with the 21-day realised volatility instead of the VIX. Compare the "
              "standard deviation of the hedging profit and loss with the VIX-based hedge (no transaction costs).",
              "Acoperiți opțiunea call la bani pe o lună cu delta calculat din volatilitatea realizată pe 21 de zile în loc de VIX. "
              "Comparați abaterea standard a rezultatului acoperirii cu cea a acoperirii bazate pe VIX (fără costuri)."))
    nb.solution("from scipy.stats import norm\nn = 21; lS = np.log(px.values); rv = (np.log(px).diff().rolling(21).std() * np.sqrt(252)).values\n"
                "st = np.arange(30, len(px) - n); st = st[px.index[st].year >= 2020]\n"
                "def delta(Sx, sg, tau): return norm.cdf((np.log(Sx) + 0.5 * sg**2 * tau) / (sg * np.sqrt(tau)))\n"
                "def call(sg, tau): d1 = 0.5 * sg * np.sqrt(tau); return norm.cdf(d1) - norm.cdf(-d1)\n"
                "for name, vol in (('VIX', vix.values / 100), ('realised', rv)):\n    res = []\n    for i in st:\n"
                "        S = np.exp(lS[i:i + n + 1] - lS[i]); tau = (n - np.arange(n)) / 252\n"
                "        d = delta(S[:-1], vol[i:i + n], tau); res.append(call(vol[i], n / 252) + (d * np.diff(S)).sum() - max(S[-1] - 1, 0))\n"
                "    print(name, 'sd of P&L (% of index):', round(100 * np.std(res), 2))")


if __name__ == "__main__":
    Notebook.build_all(10, TITLE, lecture, kind="lecture")
    Notebook.build_all(10, TITLE, seminar, kind="seminar")
