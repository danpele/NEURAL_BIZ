"""
analysis_L10.py -- Lecture 10 "Reinforcement learning for business decisions": all computation and charts.

Three real problems, one per family of methods:
  A. Battery dispatch on Romanian day-ahead prices (energy-charts.info): Bellman equation solved exactly by dynamic
     programming (perfect-foresight upper bound), tabular Q-learning and a double deep Q-network (DQN) that see only
     yesterday's prices, against the naive "yesterday's optimal schedule".
  B. Trading the S&P 500 (Yahoo Finance): proximal policy optimisation (PPO, actor-critic) for a flat/long position
     with transaction costs, against buy-and-hold and the 200-day moving-average rule; five seeds.
  C. Hedging a one-month at-the-money call on the S&P 500 with the VIX as implied volatility: a deep-hedging policy
     (Buehler et al., 2019) trained on real 21-day paths to minimise the expected shortfall of the hedging loss,
     against the Black-Scholes delta hedge.
Writes lectures/L10/L10_numbers.json and charts/en|ro/*.pdf. Trained results are cached in lectures/L10/cache/
(git-ignored); delete the cache to retrain (about 4 minutes on an Apple M5 Max CPU).
Run:  python3 lectures/L10/analysis_L10.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

npl.style()
SEED = npl.seed_all()
torch.set_num_threads(8)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
N = {"charts": {}, "seed": SEED}
T0 = time.time()
SEEDS = [SEED + i for i in range(5)]
N["n_seeds"] = len(SEEDS)


def r(x, d=2):
    return float(round(float(x), d))


def cached(name, fn):
    p = os.path.join(CACHE, name + ".json")
    if os.path.isfile(p):
        return json.load(open(p))
    t = time.time()
    res = fn()
    res["_seconds"] = round(time.time() - t, 1)
    json.dump(res, open(p, "w"))
    return res


L = lambda lang, en, ro: en if lang == "en" else ro  # noqa: E731

# ================================================================ 0. cut-off: last complete month
mk = npl.markets("close", ["SP500", "VIX"])
mka = npl.markets("adjclose", ["SP500"])
last = mk.dropna().index.max()
END = (last + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.Timedelta(days=1)
N["cutoff"] = {"end": str(END.date()), "year": int(END.year), "month": int(END.month)}

# ================================================================ A. battery dispatch
ph = npl.prices_hourly("Europe/Brussels")
ph = ph.loc[:str(END.date()) + " 23:59"]
ph["day"] = ph.index.tz_localize(None).normalize()
ph["hour"] = ph.index.hour


def day_matrix(zone):
    """Delivery days (CET) x 24 hours; only days with 24 reported hours (clock-change and incomplete days dropped)."""
    x = ph[["day", "hour", zone]].dropna()
    cnt = x.groupby("day")[zone].count()
    full = cnt[cnt == 24].index
    m = x[x.day.isin(full)].pivot_table(index="day", columns="hour", values=zone)
    return m.sort_index()


E_MAX, ETA = 2, 0.85                  # 2 MWh battery, 1 MWh of stored energy per hour, 85 % round trip (definitions)
ETA_C = ETA_D = np.sqrt(ETA)
ACTS = np.array([-1, 0, 1])           # -1 charge, 0 idle, +1 discharge (MWh of stored energy)
N["battery"] = {"e_max": E_MAX, "eta_pct": int(ETA * 100)}


def reward(p, a):
    """Cash flow (EUR) of action a at price p: discharge sells eta_d MWh, charge buys 1/eta_c MWh."""
    return np.where(a > 0, ETA_D * p, np.where(a < 0, -p / ETA_C, 0.0))


def dp_days(P):
    """Exact backward induction for every day (rows of P), SoC in {0,1,2}, start empty, free terminal value.
    Returns value V[d, h, s] (h = 0..24), optimal actions A[d, h, s]."""
    D = P.shape[0]
    V = np.zeros((D, 25, E_MAX + 1))
    A = np.zeros((D, 24, E_MAX + 1), dtype=int)
    for h in range(23, -1, -1):
        best = np.full((D, E_MAX + 1), -np.inf)
        arg = np.zeros((D, E_MAX + 1), dtype=int)
        for s in range(E_MAX + 1):
            for a in ACTS:
                s2 = s - a
                if s2 < 0 or s2 > E_MAX:
                    continue
                q = reward(P[:, h], a) + V[:, h + 1, s2]
                better = q > best[:, s]
                best[better, s] = q[better]
                arg[better, s] = a
        V[:, h] = best
        A[:, h] = arg
    return V, A


def run_schedule(P, acts):
    """Apply a fixed action plan (D x 24) to prices P from SoC 0; infeasible actions become idle."""
    s = np.zeros(P.shape[0], dtype=int)
    tot = np.zeros(P.shape[0])
    for h in range(24):
        a = acts[:, h].copy()
        a[(a > 0) & (s == 0)] = 0
        a[(a < 0) & (s == E_MAX)] = 0
        tot += reward(P[:, h], a)
        s = s - a
    return tot


ZONES = ["RO", "HU", "DE_LU", "BG"]
ZNAME = {"RO": ("Romania", "România"), "HU": ("Hungary", "Ungaria"), "DE_LU": ("Germany-Luxembourg", "Germania-Luxemburg"),
         "BG": ("Bulgaria", "Bulgaria")}
mats = {z: day_matrix(z) for z in ZONES}
RO = mats["RO"]
N["data"] = {"ro_first": str(RO.index.min().date()), "ro_last": str(RO.index.max().date()), "ro_days": int(len(RO)),
             "sp_first": str(mk.dropna().index.min().date())}

# A1. monthly daily spread (max - min) -- motivation
spread = {z: (mats[z].max(axis=1) - mats[z].min(axis=1)).resample("MS").mean() for z in ["RO", "DE_LU"]}
yr_sp = (RO.max(axis=1) - RO.min(axis=1)).groupby(RO.index.year).mean()
N["spread"] = {"ro_2019": r(yr_sp.loc[2019], 1), "ro_2025": r(yr_sp.loc[2025], 1),
               "ro_ytd": r(yr_sp.loc[END.year], 1), "ro_max_month": str(spread["RO"].idxmax().strftime("%Y-%m")),
               "ro_max_value": r(spread["RO"].max(), 1)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for z, c in (("RO", npl.PAL["data"]), ("DE_LU", npl.PAL["alt"])):
        ax.plot(spread[z].index, spread[z].values, color=c, lw=1.3, label=ZNAME[z][0 if lang == "en" else 1])
    ax.set_ylabel(L(lang, "Daily max - min (EUR/MWh)", "Max. - min. zilnic (EUR/MWh)"))
    ax.set_xlim(spread["RO"].index.min(), spread["RO"].index.max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "spread_monthly")
npl.chart_meta(N, "spread_monthly", "power", npl.period(RO.index.min(), RO.index.max(), "en") + ", monthly mean of the daily spread",
               npl.period(RO.index.min(), RO.index.max(), "ro") + ", media lunară a diferenței zilnice")

# A2. perfect-foresight DP on every day and every zone
annual = {}
for z in ZONES:
    m = mats[z]
    V, _ = dp_days(m.values)
    rev = pd.Series(V[:, 0, 0], index=m.index)
    g = rev.groupby(rev.index.year)
    annual[z] = {int(y): {"sum_k": float(v.sum() / 1000), "n": int(len(v)), "mean": float(v.mean())} for y, v in g}
N["dp"] = {"annual_k": {z: {str(y): r(annual[z][y]["sum_k"], 1) for y in annual[z]} for z in ZONES},
           "ro_2025_k": r(annual["RO"][2025]["sum_k"], 1), "ro_2019_k": r(annual["RO"][2019]["sum_k"], 1),
           "ro_ytd_k": r(annual["RO"][END.year]["sum_k"], 1), "ro_ytd_days": annual["RO"][END.year]["n"],
           "de_2025_k": r(annual["DE_LU"][2025]["sum_k"], 1), "hu_2025_k": r(annual["HU"][2025]["sum_k"], 1),
           "bg_2025_k": r(annual["BG"][2025]["sum_k"], 1),
           "ro_2022_k": r(annual["RO"][2022]["sum_k"], 1)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    years = sorted(annual["RO"])
    w = 0.2
    for i, (z, c) in enumerate(zip(ZONES, [npl.PAL["data"], npl.PAL["benchmark"], npl.PAL["alt"], npl.PAL["c7"]])):
        ys = [y for y in years if y in annual[z]]
        ax.bar(np.array(ys) + (i - 1.5) * w, [annual[z][y]["sum_k"] for y in ys], width=w, color=c,
               label=ZNAME[z][0 if lang == "en" else 1])
    ax.set_ylabel(L(lang, "Revenue (thousand EUR)", "Venit (mii EUR)"))
    npl.year_ticks(ax, years, star=END.year)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "dp_revenue_year")
npl.chart_meta(N, "dp_revenue_year", "power",
               f"{RO.index.min().year}-{END.year} (*{END.year}: January-{END.strftime('%B')}); days with 24 complete hours",
               f"{RO.index.min().year}-{END.year} (*{END.year}: ianuarie-{['ianuarie','februarie','martie','aprilie','mai','iunie','iulie','august','septembrie','octombrie','noiembrie','decembrie'][END.month-1]}); zile cu 24 de ore complete")

# A3. one real day: the DP schedule and the value function
VRO, ARO = dp_days(RO.values)
recent = RO.loc[str(END.year)]
idx_recent = recent.index
spreads_recent = (recent.max(axis=1) - recent.min(axis=1))
day = spreads_recent.idxmax()
di = RO.index.get_loc(day)
p_day = RO.loc[day].values
s, sched, socs = 0, [], [0]
for h in range(24):
    a = int(ARO[di, h, s])
    sched.append(a)
    s = s - a
    socs.append(s)
N["day"] = {"date": str(day.date()), "rev": r(VRO[di, 0, 0], 0), "pmin": r(p_day.min(), 2), "pmax": r(p_day.max(), 2),
            "h_min": int(p_day.argmin()), "h_max": int(p_day.argmax()),
            "v0_empty": r(VRO[di, 0, 0], 0), "n_charge": int(sum(1 for a in sched if a < 0)),
            "n_dis": int(sum(1 for a in sched if a > 0))}
# value at noon with SoC 0 vs 2: the price of stored energy
N["day"]["v12_s0"] = r(VRO[di, 12, 0], 0)
N["day"]["v12_s2"] = r(VRO[di, 12, 2], 0)
N["day"]["v12_gain_per_mwh"] = r((VRO[di, 12, 2] - VRO[di, 12, 0]) / 2, 2)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    hrs = np.arange(24)
    cols = [npl.PAL["benchmark"] if a < 0 else (npl.PAL["positive"] if a > 0 else "#c8c8c8") for a in sched]
    ax.bar(hrs, p_day, color=cols, width=0.8)
    ax.bar([0], [0], color=npl.PAL["benchmark"], label=L(lang, "charge (buy)", "încărcare (cumpărare)"))
    ax.bar([0], [0], color=npl.PAL["positive"], label=L(lang, "discharge (sell)", "descărcare (vînzare)"))
    ax.bar([0], [0], color="#c8c8c8", label=L(lang, "idle", "inactiv"))
    ax.set_xlabel(L(lang, "Delivery hour (CET)", "Ora de livrare (CET)"))
    ax.set_ylabel(L(lang, "Price (EUR/MWh)", "Preț (EUR/MWh)"))
    ax.set_xticks(range(0, 24, 3))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "dp_day")
npl.chart_meta(N, "dp_day", "power", f"Romania, {npl.period(day, day, 'en').split(' to ')[0]}, hourly day-ahead prices",
               f"România, {npl.period(day, day, 'ro').split(' - ')[0]}, prețuri orare pentru ziua următoare")

for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for s_, c in zip(range(E_MAX + 1), [npl.PAL["data"], npl.PAL["model"], npl.PAL["c7"]]):
        ax.plot(range(25), VRO[di, :, s_] / 1000, marker="o", ms=2.5, color=c,
                label=L(lang, f"stored energy {s_} MWh", f"energie stocată {s_} MWh"))
    ax.set_xlabel(L(lang, "Hour h (CET)", "Ora h (CET)"))
    ax.set_ylabel(L(lang, "$V_h(s)$ (thousand EUR)", "$V_h(s)$ (mii EUR)"))
    ax.set_xticks(range(0, 25, 3))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "dp_value")
npl.chart_meta(N, "dp_value", "power", f"Romania, {npl.period(day, day, 'en').split(' to ')[0]}; own computation (backward induction)",
               f"România, {npl.period(day, day, 'ro').split(' - ')[0]}; calcul propriu (inducție înapoi)")

# A4. learning agents: train 2019-2023, validate 2024, test 2025 - cut-off; information = yesterday's (and last week's) prices
P = RO.values
days = RO.index
prev_ok = np.array([i >= 7 and (days[i] - days[i - 1]).days == 1 and (days[i] - days[i - 7]).days == 7 for i in range(len(days))])
yr = days.year
TR = prev_ok & (yr >= 2019) & (yr <= 2023)
VA = prev_ok & (yr == 2024)
TE = prev_ok & (yr >= 2025)
iTR, iVA, iTE = np.where(TR)[0], np.where(VA)[0], np.where(TE)[0]
N["split"] = {"train_days": int(TR.sum()), "valid_days": int(VA.sum()), "test_days": int(TE.sum()),
              "test_first": str(days[iTE[0]].date()), "test_last": str(days[iTE[-1]].date())}

# benchmarks on the test days
pf = VRO[:, 0, 0]
_, Aall = dp_days(P)
yest_plan = np.zeros((len(days), 24), dtype=int)
s0 = np.zeros(len(days), dtype=int)
for h in range(24):                         # yesterday's optimal plan from SoC 0
    a = Aall[np.arange(len(days)), h, s0]
    yest_plan[:, h] = a
    s0 = s0 - a
naive = np.full(len(days), np.nan)
naive[1:] = run_schedule(P[1:], yest_plan[:-1])


def rank_bins(prow):
    """Quartile of each hour's price within the day (0 = cheapest six hours)."""
    rk = prow.argsort(axis=1).argsort(axis=1)
    return rk // 6


QB = rank_bins(P)                           # bin of each hour on each day; the agent uses the previous day's row


def q_learning(seed, epochs=40, alpha=0.05, gamma=1.0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((24, 4, E_MAX + 1, 3))
    hist = []
    best = (-np.inf, None, 0)
    for ep in range(epochs):
        eps = max(0.05, 1.0 - ep / (0.6 * epochs))
        order = rng.permutation(iTR)
        for d in order:
            s = 0
            b = QB[d - 1]
            for h in range(24):
                feas = [k for k, a in enumerate(ACTS) if 0 <= s - a <= E_MAX]
                if rng.random() < eps:
                    k = feas[rng.integers(len(feas))]
                else:
                    k = feas[int(np.argmax(Q[h, b[h], s, feas]))]
                a = ACTS[k]
                rew = float(reward(P[d, h], a)) / 100.0
                s2 = s - a
                if h == 23:
                    tgt = rew
                else:
                    feas2 = [j for j, a2 in enumerate(ACTS) if 0 <= s2 - a2 <= E_MAX]
                    tgt = rew + gamma * Q[h + 1, b[h + 1], s2, feas2].max()
                Q[h, b[h], s, k] += alpha * (tgt - Q[h, b[h], s, k])
                s = s2
        tr = greedy_q(Q, iTR).mean()
        va = greedy_q(Q, iVA).mean()
        hist.append((tr, va))
        if va > best[0]:
            best = (va, Q.copy(), ep)
    return best[1], hist, best[2]


def greedy_q(Q, idx):
    """Greedy tabular policy on days idx, vectorised over days."""
    idx = np.asarray(idx)
    s = np.zeros(len(idx), dtype=int)
    tot = np.zeros(len(idx))
    B = QB[idx - 1]
    for h in range(24):
        q = Q[h, B[:, h], s].copy()
        q[~mask_of(s)] = -np.inf
        a = ACTS[q.argmax(1)]
        tot += reward(P[idx, h], a)
        s = s - a
    return tot


def feats_day(d):
    """Information available before day d: yesterday's 24 prices and last week's 24 prices, standardised within each day."""
    y = P[d - 1]
    w = P[d - 7]
    z = lambda v: (v - v.mean()) / (v.std() + 1.0)  # noqa: E731
    dow = np.zeros(7)
    dow[days[d].dayofweek] = 1
    return np.concatenate([z(y), z(w), dow])


FEAT = {d: feats_day(d) for d in np.concatenate([iTR, iVA, iTE])}


def obs(dlist, h, s):
    base = np.stack([FEAT[d] for d in dlist])
    hh = np.zeros((len(dlist), 24))
    hh[:, h] = 1
    ss = np.zeros((len(dlist), E_MAX + 1))
    ss[np.arange(len(dlist)), s] = 1
    return np.concatenate([base, hh, ss], axis=1).astype(np.float32)


def mask_of(s):
    m = np.ones((len(s), 3), dtype=bool)
    m[:, 0] = s < E_MAX      # charge
    m[:, 2] = s > 0          # discharge
    return m


class QNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, x):
        return self.f(x)


def greedy_dqn(net, idx):
    s = np.zeros(len(idx), dtype=int)
    tot = np.zeros(len(idx))
    with torch.no_grad():
        for h in range(24):
            q = net(torch.from_numpy(obs(idx, h, s))).numpy()
            q[~mask_of(s)] = -1e9
            k = q.argmax(1)
            a = ACTS[k]
            tot += reward(P[idx, h], a)
            s = s - a
    return tot


def dqn(seed, epochs=40, gamma=1.0, lr=3e-4, batch=512, updates=40, tau=0.005):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d_in = obs(iTR[:1], 0, np.zeros(1, dtype=int)).shape[1]
    net, tgt = QNet(d_in), QNet(d_in)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    CAP = 24 * len(iTR) * 4                                  # replay memory: the last four epochs
    RB = {"o": np.zeros((CAP, d_in), np.float32), "a": np.zeros(CAP, np.int64), "r": np.zeros(CAP, np.float32),
          "o2": np.zeros((CAP, d_in), np.float32), "m2": np.zeros((CAP, 3), bool), "done": np.zeros(CAP, bool)}
    ptr, full = 0, 0
    hist, best = [], (-np.inf, None, 0)
    for ep in range(epochs):
        eps = max(0.05, 1.0 - ep / (0.6 * epochs))
        s = np.zeros(len(iTR), dtype=int)
        for h in range(24):
            o = obs(iTR, h, s)
            m = mask_of(s)
            with torch.no_grad():
                q = net(torch.from_numpy(o)).numpy()
            q[~m] = -1e9
            k = q.argmax(1)
            rnd = rng.random(len(iTR)) < eps
            if rnd.any():
                u = rng.random((int(rnd.sum()), 3)) * m[rnd]
                k[rnd] = u.argmax(1)
            a = ACTS[k]
            rew = reward(P[iTR, h], a) / 100.0
            s2 = s - a
            o2 = obs(iTR, min(h + 1, 23), s2)
            n = len(iTR)
            sl = np.arange(ptr, ptr + n) % CAP
            RB["o"][sl], RB["a"][sl], RB["r"][sl], RB["o2"][sl] = o, k, rew, o2
            RB["m2"][sl], RB["done"][sl] = mask_of(s2), (h == 23)
            ptr, full = (ptr + n) % CAP, min(full + n, CAP)
            s = s2
            O, A_, R_, O2, M2, DN = (RB[kk] for kk in ("o", "a", "r", "o2", "m2", "done"))
            for _ in range(updates):
                j = rng.integers(0, full, batch)
                o_t, o2_t = torch.from_numpy(O[j]), torch.from_numpy(O2[j])
                with torch.no_grad():
                    q2_online = net(o2_t)
                    q2_online[torch.from_numpy(~M2[j])] = -1e9
                    a2 = q2_online.argmax(1, keepdim=True)            # double DQN: online net chooses,
                    q2 = tgt(o2_t).gather(1, a2).squeeze(1)           # target net evaluates
                    y = torch.from_numpy(R_[j].astype(np.float32)) + gamma * q2 * torch.from_numpy(~DN[j]).float()
                qsa = net(o_t).gather(1, torch.from_numpy(A_[j]).long().unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(qsa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                with torch.no_grad():
                    for pt, pn in zip(tgt.parameters(), net.parameters()):
                        pt.mul_(1 - tau).add_(tau * pn)
        tr = greedy_dqn(net, iTR).mean()
        va = greedy_dqn(net, iVA).mean()
        hist.append((tr, va))
        if va > best[0]:
            best = (va, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    return net, hist, best[2]


def battery_q_runs():
    out = {"q": []}
    for sd in SEEDS:
        Q, hist, be = q_learning(sd)
        out["q"].append({"seed": sd, "hist": hist, "best_ep": be, "test": greedy_q(Q, iTE).tolist()})
        if sd == SEEDS[0]:
            np.save(os.path.join(CACHE, "q_table.npy"), Q)
        print(f"  Q-learning seed {sd}: {np.mean(out['q'][-1]['test']):.1f} EUR/day ({time.time() - T0:.0f}s)")
    return out


def battery_dqn_runs():
    out = {"dqn": []}
    for sd in SEEDS:
        net, hist, be = dqn(sd)
        te = greedy_dqn(net, iTE)
        out["dqn"].append({"seed": sd, "hist": hist, "best_ep": be, "test": te.tolist()})
        print(f"  DQN seed {sd}: {te.mean():.1f} EUR/day ({time.time() - T0:.0f}s)")
    return out


BRQ = cached("battery_v3", battery_q_runs)          # tabular Q-learning, 5 seeds
BRD = cached("battery_dqn_v4", battery_dqn_runs)    # double DQN, 5 seeds
BR = {"q": BRQ["q"], "dqn": BRD["dqn"], "_seconds": BRQ["_seconds"] + BRD["_seconds"]}
pf_te, nv_te = pf[iTE], naive[iTE]
qs = np.array([x["test"] for x in BR["q"]])      # seeds x days
ds = np.array([x["test"] for x in BR["dqn"]])
cap = lambda v: 100 * v.sum() / pf_te.sum()       # noqa: E731
q_caps = np.array([cap(v) for v in qs])
d_caps = np.array([cap(v) for v in ds])
N["bat"] = {"pf_k": r(pf_te.sum() / 1000, 1), "naive_k": r(nv_te.sum() / 1000, 1), "naive_cap": r(cap(nv_te), 1),
            "q_k": r(qs.sum(1).mean() / 1000, 1), "q_cap": r(q_caps.mean(), 1), "q_cap_sd": r(q_caps.std(ddof=1), 1),
            "dqn_k": r(ds.sum(1).mean() / 1000, 1), "dqn_cap": r(d_caps.mean(), 1), "dqn_cap_sd": r(d_caps.std(ddof=1), 1),
            "dqn_cap_min": r(d_caps.min(), 1), "dqn_cap_max": r(d_caps.max(), 1),
            "q_cap_min": r(q_caps.min(), 1), "q_cap_max": r(q_caps.max(), 1),
            "pf_day": r(pf_te.mean(), 0), "seconds": r(BR["_seconds"], 0),
            "q_states": 24 * 4 * (E_MAX + 1), "dqn_inputs": int(obs(iTR[:1], 0, np.zeros(1, dtype=int)).shape[1])}
# paired test: DQN (seed mean) vs naive, daily differences, Diebold-Mariano style t statistic with Newey-West (lag 7)
diff = ds.mean(0) - nv_te


def nw_t(x, lag=7):
    x = x - x.mean()
    n = len(x)
    g0 = (x @ x) / n
    s = g0 + 2 * sum((1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n for k in range(1, lag + 1))
    return float((diff.mean()) / np.sqrt(s / n))


from scipy import stats  # noqa: E402
tdm = nw_t(diff)
N["bat"]["dm_t"] = r(tdm, 2)
N["bat"]["dm_p"] = r(2 * (1 - stats.norm.cdf(abs(tdm))), 3)
N["bat"]["dqn_minus_naive_day"] = r(diff.mean(), 1)
N["bat"]["dqn_beats_naive_seeds"] = int((ds.sum(1) > nv_te.sum()).sum())
N["bat"]["q_beats_naive_seeds"] = int((qs.sum(1) > nv_te.sum()).sum())

for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    dd = days[iTE]
    ax.plot(dd, np.cumsum(pf_te) / 1000, color=npl.PAL["data"], label=L(lang, "perfect foresight (upper bound)", "previziune perfectă (limita superioară)"))
    ax.plot(dd, np.cumsum(ds.mean(0)) / 1000, color=npl.PAL["model"], label=L(lang, "double DQN (mean of 5 seeds)", "DQN dublu (media a 5 valori inițiale)"))
    ax.plot(dd, np.cumsum(qs.mean(0)) / 1000, color=npl.PAL["alt"], label=L(lang, "tabular Q-learning", "Q-learning tabelar"))
    ax.plot(dd, np.cumsum(nv_te) / 1000, color=npl.PAL["benchmark"], label=L(lang, "yesterday's optimal plan", "planul optim de ieri"))
    ax.set_ylabel(L(lang, "Cumulative revenue (thousand EUR)", "Venit cumulat (mii EUR)"))
    ax.set_xlim(dd.min(), dd.max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "battery_test_cum")
npl.chart_meta(N, "battery_test_cum", "power", npl.period(days[iTE[0]], days[iTE[-1]], "en") + " (test); trained on 2019-2023, validated on 2024",
               npl.period(days[iTE[0]], days[iTE[-1]], "ro") + " (test); antrenare 2019-2023, validare 2024")

# training curves (validation revenue per day by epoch), mean and range over seeds
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for key, c, lab in (("dqn", npl.PAL["model"], L(lang, "double DQN", "DQN dublu")),
                        ("q", npl.PAL["alt"], L(lang, "tabular Q-learning", "Q-learning tabelar"))):
        H = np.array([[h[1] for h in x["hist"]] for x in BR[key]])
        ep = np.arange(1, H.shape[1] + 1)
        ax.fill_between(ep, H.min(0), H.max(0), color=c, alpha=0.2, lw=0)
        ax.plot(ep, H.mean(0), color=c, label=lab + L(lang, " (validation 2024)", " (validare 2024)"))
    ax.axhline(naive[iVA].mean(), color=npl.PAL["benchmark"], ls="--", lw=1.1, label=L(lang, "yesterday's optimal plan", "planul optim de ieri"))
    ax.set_xlabel(L(lang, "Training epoch (one pass over the 2019-2023 days)", "Epoca de antrenare (o trecere prin zilele 2019-2023)"))
    ax.set_ylabel(L(lang, "EUR per day", "EUR pe zi"))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "battery_learning")
npl.chart_meta(N, "battery_learning", "power", "validation year 2024; band = range over 5 seeds",
               "anul de validare 2024; banda = intervalul celor 5 valori inițiale")
N["bat"]["naive_valid_day"] = r(naive[iVA].mean(), 0)
N["bat"]["dqn_valid_day"] = r(np.mean([max(h[1] for h in x["hist"]) for x in BR["dqn"]]), 0)
N["bat"]["q_valid_day"] = r(np.mean([max(h[1] for h in x["hist"]) for x in BR["q"]]), 0)
N["bat"]["epochs"] = len(BR["dqn"][0]["hist"])

# learned tabular policy: action at SoC 1 by hour and yesterday's price quartile
Qt = np.load(os.path.join(CACHE, "q_table.npy"))
pol = np.zeros((4, 24))
for b in range(4):
    for h in range(24):
        pol[b, h] = ACTS[int(np.argmax(Qt[h, b, 1]))]
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    from matplotlib.colors import ListedColormap
    cm = ListedColormap([npl.PAL["benchmark"], "#e6e6e6", npl.PAL["positive"]])
    ax.imshow(pol, aspect="auto", cmap=cm, vmin=-1, vmax=1, origin="lower")
    ax.set_yticks(range(4))
    ax.set_yticklabels(L(lang, ["cheapest 6 h", "2nd quartile", "3rd quartile", "dearest 6 h"],
                         ["cele mai ieftine 6 h", "quartila a 2-a", "quartila a 3-a", "cele mai scumpe 6 h"]), fontsize=9)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel(L(lang, "Hour h (CET); stored energy 1 MWh", "Ora h (CET); energie stocată 1 MWh"))
    for v, lab in ((-1, L(lang, "charge", "încărcare")), (0, L(lang, "idle", "inactiv")), (1, L(lang, "discharge", "descărcare"))):
        ax.bar([0], [0], color=cm((v + 1) / 2), label=lab)
    for sp in ax.spines.values():
        sp.set_visible(False)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "q_policy")
npl.chart_meta(N, "q_policy", "power", "rows = quartile of the hour's price yesterday; seed " + str(SEEDS[0]),
               "rînduri = quartila prețului orei respective în ziua precedentă; valoarea inițială " + str(SEEDS[0]))

# ================================================================ B. trading the S&P 500 with PPO
px = mk["SP500"].dropna().loc[:END]
vix = mk["VIX"].reindex(px.index).ffill()
lr_ = np.log(px).diff()
F = pd.DataFrame({
    "r1": lr_, "r5": np.log(px).diff(5), "r21": np.log(px).diff(21), "r63": np.log(px).diff(63),
    "rv21": lr_.rolling(21).std() * np.sqrt(252), "vix": vix / 100, "ma200": px / px.rolling(200).mean() - 1,
}).dropna()
fwd = lr_.shift(-1).reindex(F.index)                   # next day's log return (earned by the position chosen at close t)
F, fwd = F.iloc[:-1], fwd.iloc[:-1]
tr_m = (F.index.year <= 2016)
va_m = (F.index.year >= 2017) & (F.index.year <= 2019)
te_m = (F.index.year >= 2020)
mu, sd = F[tr_m].mean(), F[tr_m].std()
X = ((F - mu) / sd).values.astype(np.float32)
RET = fwd.values.astype(np.float32)
COST = 0.0005                                          # 5 basis points per unit of position changed (definition)
N["trade"] = {"cost_bp": 5, "train_first": str(F.index[tr_m][0].date()), "train_last": str(F.index[tr_m][-1].date()),
              "valid_first": str(F.index[va_m][0].date()), "valid_last": str(F.index[va_m][-1].date()),
              "test_first": str(F.index[te_m][0].date()), "test_last": str(F.index[te_m][-1].date()),
              "n_feat": int(X.shape[1]) + 1, "n_test": int(te_m.sum())}


class ActorCritic(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, 2)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def perf(pos, ret, cost=COST):
    """Daily strategy log returns from positions (chosen at close t, earning ret[t]); cost on every change."""
    pos = np.asarray(pos, float)
    chg = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pos * ret - cost * chg


def stats_of(x):
    x = np.asarray(x)
    eq = np.exp(np.cumsum(x))
    dd = eq / np.maximum.accumulate(eq) - 1
    return {"ann_ret": 100 * (np.exp(x.mean() * 252) - 1), "ann_vol": 100 * x.std() * np.sqrt(252),
            "sharpe": x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0, "maxdd": 100 * dd.min()}


def run_policy(net, idx, greedy=True):
    pos, prev = [], 0.0
    with torch.no_grad():
        for i in idx:
            o = torch.from_numpy(np.concatenate([X[i], [prev]]).astype(np.float32))
            logits, _ = net(o)
            a = int(logits.argmax()) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
            pos.append(float(a))
            prev = float(a)
    return np.array(pos)


def ppo(seed, iters=150, B=64, T=126, gamma=0.99, lam=0.95, clip=0.2, ent=0.01, lr=3e-4, k_epochs=4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ActorCritic(X.shape[1] + 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    itr = np.where(tr_m)[0]
    iva = np.where(va_m)[0]
    hist, best = [], (-np.inf, None, 0)
    for it in range(iters):
        starts = rng.choice(itr[:-T], B)
        prev = np.zeros(B, dtype=np.float32)
        O, A_, LP, R_, V_ = [], [], [], [], []
        with torch.no_grad():
            for t in range(T):
                o = torch.from_numpy(np.concatenate([X[starts + t], prev[:, None]], 1).astype(np.float32))
                logits, v = net(o)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                pos = a.numpy().astype(np.float32)
                rew = pos * RET[starts + t] - COST * np.abs(pos - prev)
                O.append(o); A_.append(a); LP.append(dist.log_prob(a)); R_.append(torch.from_numpy(100 * rew)); V_.append(v)
                prev = pos
            O, A_, LP, R_, V_ = torch.stack(O), torch.stack(A_), torch.stack(LP), torch.stack(R_), torch.stack(V_)
            adv = torch.zeros_like(R_)
            g = torch.zeros(B)
            for t in range(T - 1, -1, -1):                       # generalised advantage estimation
                v_next = V_[t + 1] if t + 1 < T else torch.zeros(B)
                delta = R_[t] + gamma * v_next - V_[t]
                g = delta + gamma * lam * g
                adv[t] = g
            ret_t = adv + V_
        O, A_, LP, adv, ret_t = (z.reshape(T * B, *z.shape[2:]) for z in (O, A_, LP, adv, ret_t))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(k_epochs):
            perm = torch.from_numpy(rng.permutation(T * B))
            for j in range(0, T * B, 1024):
                b = perm[j:j + 1024]
                logits, v = net(O[b])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A_[b]) - LP[b])
                s1, s2 = ratio * adv[b], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
                loss = -torch.min(s1, s2).mean() + 0.5 * ((v - ret_t[b]) ** 2).mean() - ent * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if it % 5 == 4 or it == iters - 1:
            s_tr = stats_of(perf(run_policy(net, itr[-756:]), RET[itr[-756:]]))["sharpe"]
            s_va = stats_of(perf(run_policy(net, iva), RET[iva]))["sharpe"]
            hist.append((it + 1, s_tr, s_va))
            if s_va > best[0]:
                best = (s_va, {k: v.clone() for k, v in net.state_dict().items()}, it + 1)
    net.load_state_dict(best[1])
    return net, hist, best[2]


ite = np.where(te_m)[0]


def trading_runs():
    out = []
    for sd in SEEDS:
        net, hist, be = ppo(sd)
        pos = run_policy(net, ite)
        out.append({"seed": sd, "hist": hist, "best_it": be, "pos": pos.tolist()})
        print(f"  PPO seed {sd}: test Sharpe {stats_of(perf(pos, RET[ite]))['sharpe']:.2f} ({time.time() - T0:.0f}s)")
    return {"data": out}


TRR = cached("trading_v2", trading_runs)
bh = perf(np.ones(len(ite)), RET[ite])
ma_pos = (F["ma200"].values[ite] > 0).astype(float)
ma = perf(ma_pos, RET[ite])
ppo_r = np.array([perf(np.array(x["pos"]), RET[ite]) for x in TRR["data"]])
st_bh, st_ma = stats_of(bh), stats_of(ma)
st_ppo = [stats_of(x) for x in ppo_r]
sh_ppo = np.array([x["sharpe"] for x in st_ppo])
pos_all = np.array([x["pos"] for x in TRR["data"]])
N["trade"].update({
    "bh_sharpe": r(st_bh["sharpe"]), "bh_ret": r(st_bh["ann_ret"], 1), "bh_vol": r(st_bh["ann_vol"], 1), "bh_dd": r(st_bh["maxdd"], 1),
    "ma_sharpe": r(st_ma["sharpe"]), "ma_ret": r(st_ma["ann_ret"], 1), "ma_vol": r(st_ma["ann_vol"], 1), "ma_dd": r(st_ma["maxdd"], 1),
    "ma_inv": r(100 * ma_pos.mean(), 0),
    "ppo_sharpe": r(sh_ppo.mean()), "ppo_sharpe_min": r(sh_ppo.min()), "ppo_sharpe_max": r(sh_ppo.max()),
    "ppo_ret": r(np.mean([x["ann_ret"] for x in st_ppo]), 1), "ppo_vol": r(np.mean([x["ann_vol"] for x in st_ppo]), 1),
    "ppo_dd": r(np.mean([x["maxdd"] for x in st_ppo]), 1), "ppo_inv": r(100 * pos_all.mean(), 0),
    "ppo_trades": r(np.mean([np.abs(np.diff(np.concatenate([[0], p]))).sum() for p in pos_all]), 0),
    "ma_trades": int(np.abs(np.diff(np.concatenate([[0], ma_pos]))).sum()),
    "ppo_beats_bh": int((sh_ppo > st_bh["sharpe"]).sum()),
    "best_it_mean": r(np.mean([x["best_it"] for x in TRR["data"]]), 0), "seconds": r(TRR["_seconds"], 0),
})
# stationary block bootstrap (mean block 21 days) of the Sharpe difference: PPO (seed-mean returns) minus buy-and-hold
rng = np.random.default_rng(SEED)
pm = ppo_r.mean(0)
nT = len(pm)
diffs = []
for _ in range(2000):
    idx = []
    while len(idx) < nT:
        st = rng.integers(0, nT)
        idx.extend(range(st, min(st + rng.geometric(1 / 21), nT)))
    idx = np.array(idx[:nT])
    a, b = pm[idx], bh[idx]
    diffs.append(a.mean() / a.std() * np.sqrt(252) - b.mean() / b.std() * np.sqrt(252))
diffs = np.array(diffs)
N["trade"]["dsh"] = r(stats_of(pm)["sharpe"] - st_bh["sharpe"])
N["trade"]["dsh_lo"] = r(np.percentile(diffs, 2.5))
N["trade"]["dsh_hi"] = r(np.percentile(diffs, 97.5))
N["trade"]["pm_sharpe"] = r(stats_of(pm)["sharpe"])

dte = F.index[ite]
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    eqs = np.exp(np.cumsum(ppo_r, 1))
    ax.fill_between(dte, eqs.min(0), eqs.max(0), color=npl.PAL["model"], alpha=0.2, lw=0)
    ax.plot(dte, np.exp(np.cumsum(bh)), color=npl.PAL["data"], label=L(lang, "buy and hold", "cumpără și păstrează"))
    ax.plot(dte, np.exp(np.cumsum(ma)), color=npl.PAL["benchmark"], label=L(lang, "200-day moving-average rule", "regula mediei mobile de 200 de zile"))
    ax.plot(dte, np.exp(np.cumsum(pm)), color=npl.PAL["model"], label=L(lang, "PPO agent (mean of 5 seeds; band = range)", "agentul PPO (media a 5 valori inițiale; banda = interval)"))
    ax.set_ylabel(L(lang, "Value of 1 USD", "Valoarea a 1 USD"))
    ax.set_xlim(dte.min(), dte.max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "trading_equity")
npl.chart_meta(N, "trading_equity", "yahoo", npl.period(dte.min(), dte.max(), "en") + " (test), S\\&P 500, 5 bp cost per trade",
               npl.period(dte.min(), dte.max(), "ro") + " (test), S\\&P 500, cost de 5 pb pe tranzacție")

bhv = stats_of(perf(np.ones(va_m.sum()), RET[va_m]))["sharpe"]
N["trade"]["bh_valid_sharpe"] = r(bhv)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    H = np.array([[h[2] for h in x["hist"]] for x in TRR["data"]])
    Ht = np.array([[h[1] for h in x["hist"]] for x in TRR["data"]])
    it_ = np.array([h[0] for h in TRR["data"][0]["hist"]])
    ax.fill_between(it_, Ht.min(0), Ht.max(0), color=npl.PAL["train"], alpha=0.15, lw=0)
    ax.plot(it_, Ht.mean(0), color=npl.PAL["train"], label=L(lang, "training (2014-2016)", "antrenare (2014-2016)"))
    ax.fill_between(it_, H.min(0), H.max(0), color=npl.PAL["valid"], alpha=0.2, lw=0)
    ax.plot(it_, H.mean(0), color=npl.PAL["valid"], label=L(lang, "validation (2017-2019)", "validare (2017-2019)"))
    ax.axhline(bhv, color=npl.PAL["data"], ls="--", lw=1.0, label=L(lang, "buy and hold, validation", "cumpără și păstrează, validare"))
    ax.set_xlabel(L(lang, "PPO iteration (64 episodes of 126 days)", "Iterația PPO (64 de episoade a cîte 126 de zile)"))
    ax.set_ylabel(L(lang, "Sharpe ratio (annualised)", "Raportul Sharpe (anualizat)"))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "trading_learning")
npl.chart_meta(N, "trading_learning", "yahoo", "training windows from " + npl.period(F.index[tr_m][0], F.index[tr_m][-1], "en") +
               "; band = range over 5 seeds", "ferestre de antrenare din " + npl.period(F.index[tr_m][0], F.index[tr_m][-1], "ro") +
               "; banda = intervalul celor 5 valori inițiale")

for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    expo = pd.Series(pos_all.mean(0), index=dte).rolling(21).mean()
    ax.plot(dte, 100 * expo.values, color=npl.PAL["model"], label=L(lang, "PPO exposure (% invested, 21-day mean)", "expunerea PPO (% investit, medie pe 21 de zile)"))
    ax.plot(dte, 100 * pd.Series(ma_pos, index=dte).rolling(21).mean().values, color=npl.PAL["benchmark"], lw=1.0,
            label=L(lang, "moving-average rule", "regula mediei mobile"))
    ax.set_ylabel(L(lang, "% invested", "% investit"))
    ax.set_ylim(-3, 103)
    ax.set_xlim(dte.min(), dte.max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "trading_exposure")
npl.chart_meta(N, "trading_exposure", "yahoo", npl.period(dte.min(), dte.max(), "en") + " (test)",
               npl.period(dte.min(), dte.max(), "ro") + " (test)")
covid = (dte >= "2020-02-20") & (dte <= "2020-04-30")
y22 = (dte.year == 2022)
N["trade"]["ppo_inv_covid"] = r(100 * pos_all[:, covid].mean(), 0)
N["trade"]["ma_inv_covid"] = r(100 * ma_pos[covid].mean(), 0)
N["trade"]["ppo_inv_2022"] = r(100 * pos_all[:, y22].mean(), 0)
N["trade"]["ma_inv_2022"] = r(100 * ma_pos[y22].mean(), 0)

# PPO clipped objective (mechanism illustration, analytic)
for lang in ("en", "ro"):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.2))
    rr = np.linspace(0.4, 1.6, 400)
    for ax, A, nm in ((axs[0], 1.0, "$A>0$"), (axs[1], -1.0, "$A<0$")):
        ax.plot(rr, rr * A, color=npl.MUTED, ls="--", lw=1.0, label=L(lang, "unclipped $\\rho A$", "obiectivul neplafonat $\\rho A$"))
        ax.plot(rr, np.minimum(rr * A, np.clip(rr, 0.8, 1.2) * A), color=npl.PAL["model"], lw=1.8,
                label=L(lang, "PPO: $\\min(\\rho A,\\ \\mathrm{clip}(\\rho,1-\\epsilon,1+\\epsilon)A)$, $\\epsilon=0.2$",
                        "PPO: $\\min(\\rho A,\\ \\mathrm{clip}(\\rho,1-\\epsilon,1+\\epsilon)A)$, $\\epsilon=0{,}2$"))
        ax.axvline(1, color=npl.MUTED, lw=0.6)
        ax.set_xlabel(L(lang, "probability ratio $\\rho$ ", "raportul probabilităților $\\rho$ ") + "(" + nm + ")")
    axs[0].set_ylabel(L(lang, "objective", "obiectiv"))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "ppo_clip", also_x=tuple(axs))
npl.chart_meta(N, "ppo_clip", "mechanism", "analytic function, no data", "funcție analitică, fără date",
               source_en="Schulman et al. (2017), clipped surrogate objective",
               source_ro="Schulman et al. (2017), obiectivul surogat plafonat")

# ================================================================ C. deep hedging of a one-month call on the S&P 500
S = mk["SP500"].dropna().loc[:END]
VX = mk["VIX"].reindex(S.index).ffill() / 100
NSTEP = 21
logS = np.log(S.values)
starts = np.arange(0, len(S) - NSTEP)
paths = np.stack([np.exp(logS[i:i + NSTEP + 1] - logS[i]) for i in starts]).astype(np.float32)   # S_k / S_0
vols = np.stack([VX.values[i:i + NSTEP + 1] for i in starts]).astype(np.float32)
sd0 = S.index[starts]
h_tr = sd0.year <= 2016
h_va = (sd0.year >= 2017) & (sd0.year <= 2019)
h_te = sd0.year >= 2020
from scipy.stats import norm  # noqa: E402


def bs_call(Sx, sig, tau, K=1.0):
    tau = np.maximum(tau, 1e-8)
    d1 = (np.log(Sx / K) + 0.5 * sig ** 2 * tau) / (sig * np.sqrt(tau))
    return Sx * norm.cdf(d1) - K * norm.cdf(d1 - sig * np.sqrt(tau)), norm.cdf(d1)


TAU = (NSTEP - np.arange(NSTEP + 1)) / 252.0
prem = bs_call(1.0, vols[:, 0], TAU[0])[0]
payoff = np.maximum(paths[:, -1] - 1.0, 0)


def pnl_of(delta, c, idx):
    """The hedger sold the call at the Black-Scholes price (VIX as implied volatility, rate 0) and holds delta_k units
    of the index over (k, k+1]; proportional cost c on every unit traded, including the final unwinding."""
    Pp = paths[idx]
    dS = np.diff(Pp, axis=1)
    z = np.zeros((len(delta), 1))
    trade = np.abs(np.diff(np.concatenate([z, delta, z], 1), axis=1))
    costs = c * (trade * Pp).sum(1)
    return prem[idx] + (delta * dS).sum(1) - payoff[idx] - costs


bs_delta = np.stack([bs_call(paths[:, k], vols[:, k], TAU[k])[1] for k in range(NSTEP)], 1)


def es(loss, a=0.95):
    q = np.quantile(loss, a)
    return float(loss[loss >= q].mean())


class Hedger(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return torch.sigmoid(self.f(x)).squeeze(-1)


def hedge_feats(Sk, vk, k, prev):
    tau = torch.full_like(Sk, float(TAU[k]))
    return torch.stack([torch.log(Sk) * 10.0, tau * 12.0, vk * 4.0, prev], -1)


def deep_hedge_delta(net, idx):
    Pt, Vt = torch.from_numpy(paths[idx]), torch.from_numpy(vols[idx])
    prev = torch.zeros(len(idx))
    out = []
    for k in range(NSTEP):
        d = net(hedge_feats(Pt[:, k], Vt[:, k], k, prev))
        out.append(d)
        prev = d
    return torch.stack(out, 1)


def train_hedger(seed, c, iters=600, alpha=0.95):
    torch.manual_seed(seed)
    net = Hedger()
    w = torch.zeros(1, requires_grad=True)                   # value-at-risk level in the Rockafellar-Uryasev form
    opt = torch.optim.Adam(list(net.parameters()) + [w], lr=3e-3)
    itr_ = np.where(h_tr)[0]
    Pt = torch.from_numpy(paths[itr_])
    pr, py = torch.from_numpy(prem[itr_].astype(np.float32)), torch.from_numpy(payoff[itr_].astype(np.float32))
    for it in range(iters):
        delta = deep_hedge_delta(net, itr_)
        dS = Pt[:, 1:] - Pt[:, :-1]
        z = torch.zeros(len(itr_), 1)
        trade = torch.abs(torch.diff(torch.cat([z, delta, z], 1), dim=1))
        pnl = pr + (delta * dS).sum(1) - py - c * (trade * Pt).sum(1)
        loss_ = -pnl * 100
        obj = w + torch.relu(loss_ - w).mean() / (1 - alpha)
        opt.zero_grad()
        obj.backward()
        opt.step()
    return net


C_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005]
C_MAIN = 0.001                                                # 10 bp (definition)
ihte = np.where(h_te)[0]


def hedge_runs():
    out = {"main": [], "grid": {}}
    for sd in SEEDS:
        net = train_hedger(sd, C_MAIN)
        with torch.no_grad():
            d = deep_hedge_delta(net, ihte).numpy()
            grid = np.linspace(0.9, 1.1, 41).astype(np.float32)
            k10 = NSTEP - 10
            prev = torch.from_numpy(bs_call(grid, 0.2, TAU[k10])[1].astype(np.float32))
            dg = net(hedge_feats(torch.from_numpy(grid), torch.full((41,), 0.2), k10, prev)).numpy()
        out["main"].append({"seed": sd, "pnl": pnl_of(d, C_MAIN, ihte).tolist(), "grid": dg.tolist(),
                            "turnover": float(np.abs(np.diff(np.concatenate([np.zeros((len(d), 1)), d], 1), axis=1)).sum(1).mean())})
        print(f"  hedge seed {sd} done ({time.time() - T0:.0f}s)")
    for c in C_GRID:
        res = []
        for sd in SEEDS[:3]:
            net = train_hedger(sd, c)
            with torch.no_grad():
                d = deep_hedge_delta(net, ihte).numpy()
            res.append(es(-pnl_of(d, c, ihte) * 100))
        out["grid"][str(c)] = res
    return out


HR = cached("hedge_v1", hedge_runs)
nh = pnl_of(np.zeros_like(bs_delta[ihte]), C_MAIN, ihte) * 100   # in % of the initial index level
bsd = pnl_of(bs_delta[ihte], C_MAIN, ihte) * 100
dhs = np.array([x["pnl"] for x in HR["main"]]) * 100
es_dh = np.array([es(-x) for x in dhs])
std_dh = np.array([x.std() for x in dhs])
N["hedge"] = {"n_train": int(h_tr.sum()), "n_valid": int(h_va.sum()), "n_test": int(h_te.sum()), "cost_bp": 10,
              "test_first": str(sd0[ihte[0]].date()), "test_last": str(sd0[ihte[-1]].date()),
              "prem_mean": r(100 * prem[ihte].mean(), 2), "payoff_mean": r(100 * payoff[ihte].mean(), 2),
              "nh_std": r(nh.std(), 2), "nh_es": r(es(-nh), 2), "nh_mean": r(nh.mean(), 2),
              "bs_std": r(bsd.std(), 2), "bs_es": r(es(-bsd), 2), "bs_mean": r(bsd.mean(), 2),
              "dh_std": r(std_dh.mean(), 2), "dh_es": r(es_dh.mean(), 2), "dh_es_sd": r(es_dh.std(ddof=1), 2),
              "dh_mean": r(dhs.mean(), 2), "dh_es_min": r(es_dh.min(), 2), "dh_es_max": r(es_dh.max(), 2),
              "dh_beats_bs": int((es_dh < es(-bsd)).sum()),
              "bs_turnover": r(np.abs(np.diff(np.concatenate([np.zeros((len(ihte), 1)), bs_delta[ihte]], 1), axis=1)).sum(1).mean(), 2),
              "dh_turnover": r(np.mean([x["turnover"] for x in HR["main"]]), 2), "seconds": r(HR["_seconds"], 0),
              "vix_mean_test": r(100 * vols[ihte, 0].mean(), 1)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    bins = np.linspace(-8, 4, 61)
    ax.hist(np.clip(nh, -8, 4), bins=bins, color=npl.MUTED, alpha=0.5, label=L(lang, "no hedge", "fără acoperire"))
    ax.hist(np.clip(bsd, -8, 4), bins=bins, histtype="step", lw=1.6, color=npl.PAL["benchmark"], label=L(lang, "Black-Scholes delta", "delta Black-Scholes"))
    ax.hist(np.clip(dhs[0], -8, 4), bins=bins, histtype="step", lw=1.6, color=npl.PAL["model"], label=L(lang, "deep hedge (first seed)", "acoperire neuronală (prima valoare inițială)"))
    ax.set_xlabel(L(lang, "Hedging profit and loss (% of the initial index level; below -8 shown at -8)",
                    "Rezultatul acoperirii (% din nivelul inițial al indicelui; sub -8 apare la -8)"))
    ax.set_ylabel(L(lang, "Start days", "Zile de început"))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "hedge_pnl", also_x=(ax,))
npl.chart_meta(N, "hedge_pnl", "yahoo", "21-day paths starting " + npl.period(sd0[ihte[0]], sd0[ihte[-1]], "en") + " (test); S\\&P 500 and VIX; 10 bp cost",
               "traiectorii de 21 de zile începute " + npl.period(sd0[ihte[0]], sd0[ihte[-1]], "ro") + " (test); S\\&P 500 și VIX; cost 10 pb")

grid = np.linspace(0.9, 1.1, 41)
G = np.array([x["grid"] for x in HR["main"]])
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.fill_between(grid, G.min(0), G.max(0), color=npl.PAL["model"], alpha=0.2, lw=0)
    ax.plot(grid, G.mean(0), color=npl.PAL["model"], label=L(lang, "deep hedge (mean of 5 seeds; band = range)", "acoperire neuronală (media a 5 valori inițiale; banda = interval)"))
    ax.plot(grid, bs_call(grid, 0.2, 10 / 252)[1], color=npl.PAL["benchmark"], ls="--", label=L(lang, "Black-Scholes delta", "delta Black-Scholes"))
    ax.set_xlabel(L(lang, "Moneyness $S_t/K$ (10 trading days to maturity, VIX = 20)", "Raportul $S_t/K$ (10 zile pînă la scadență, VIX = 20)"))
    ax.set_ylabel(L(lang, "Hedge ratio $\\delta_t$", "Raportul de acoperire $\\delta_t$"))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "hedge_delta", also_x=(ax,))
npl.chart_meta(N, "hedge_delta", "yahoo", "policy trained on 21-day paths starting 2005-2016", "politică antrenată pe traiectorii de 21 de zile începute în 2005-2016")
mid = np.argmin(np.abs(grid - 1.0))
N["hedge"]["atm_dh"] = r(G.mean(0)[mid], 2)
N["hedge"]["atm_bs"] = r(bs_call(1.0, 0.2, 10 / 252)[1], 2)

bs_es_grid = [es(-pnl_of(bs_delta[ihte], c, ihte) * 100) for c in C_GRID]
dh_es_grid = [np.mean(HR["grid"][str(c)]) for c in C_GRID]
N["hedge"]["bs_es_0"] = r(bs_es_grid[0], 2)
N["hedge"]["dh_es_0"] = r(dh_es_grid[0], 2)
N["hedge"]["bs_es_50"] = r(bs_es_grid[-1], 2)
N["hedge"]["dh_es_50"] = r(dh_es_grid[-1], 2)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    xb = [c * 1e4 for c in C_GRID]
    ax.plot(xb, bs_es_grid, marker="o", color=npl.PAL["benchmark"], label=L(lang, "Black-Scholes delta", "delta Black-Scholes"))
    ax.plot(xb, dh_es_grid, marker="o", color=npl.PAL["model"], label=L(lang, "deep hedge (mean of 3 seeds)", "acoperire neuronală (media a 3 valori inițiale)"))
    ax.set_xlabel(L(lang, "Transaction cost (basis points of the traded value)", "Costul de tranzacționare (puncte de bază din valoarea tranzacționată)"))
    ax.set_ylabel(L(lang, "ES$_{95}$ of the loss (%)", "ES$_{95}$ al pierderii (%)"))
    ax.set_xticks(xb)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "hedge_costs", also_x=(ax,))
npl.chart_meta(N, "hedge_costs", "yahoo", "test paths starting " + npl.period(sd0[ihte[0]], sd0[ihte[-1]], "en"),
               "traiectorii de test începute " + npl.period(sd0[ihte[0]], sd0[ihte[-1]], "ro"))

N["runtime_s"] = r(time.time() - T0, 0)
with open(os.path.join(HERE, "L10_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1)
print("wrote L10_numbers.json and", len(N["charts"]), "charts x 2 languages in", N["runtime_s"], "s")
