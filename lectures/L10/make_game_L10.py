"""
make_game_L10.py -- data of the Lecture 10 game "You or the agent? A battery on the Romanian day-ahead market"
(interactive/L10_game.html, L10_game_ro.html).

Each card is a real test day (2025 to 24 October 2026) of the battery problem of analysis_L10.py: 2 MWh, 1 MWh of stored
energy per hour, 85 % round trip, start empty, hourly Romanian day-ahead prices (energy-charts.info). Before deciding,
the student sees what the tabular Q-learning agent of the lecture sees: yesterday's 24 prices. He chooses one of four
simple plans; then he sees the realised cash flow of every plan on the day's real prices, the cash flow of the agent
(tabular Q-learning, first seed, Q-table cached in lectures/L10/cache/q_table.npy) and the perfect-foresight revenue
(dynamic programming). The hours of the three fixed plans come from the mean price profile of 2019-2024 (training and
validation years), never from the test days.

Same data, cut-off, battery, split and agent as analysis_L10.py; the script checks that the perfect-foresight and naive
totals equal L10_numbers.json and that the replayed agent equals the cached test revenue (cache/battery_v3.json, seed 1).
The pool of cards: 40 test days drawn at random (seed of the course); every game draws 8 of them.

Run:  python3 lectures/L10/make_game_L10.py      ->  interactive/L10_game_data.js and the two pages
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402

SEED = npl.seed_all()
OUT = os.path.join(HERE, "..", "..", "interactive", "L10_game_data.js")
NUMS = json.load(open(os.path.join(HERE, "L10_numbers.json")))

# ---------------------------------------------------------------- data, battery and split exactly as in analysis_L10.py
mk = npl.markets("close", ["SP500", "VIX"])
last = mk.dropna().index.max()
END = (last + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.Timedelta(days=1)
assert str(END.date()) == NUMS["cutoff"]["end"]
ph = npl.prices_hourly("Europe/Brussels")
ph = ph.loc[:str(END.date()) + " 23:59"]
ph["day"] = ph.index.tz_localize(None).normalize()
ph["hour"] = ph.index.hour
x = ph[["day", "hour", "RO"]].dropna()
cnt = x.groupby("day")["RO"].count()
RO = x[x.day.isin(cnt[cnt == 24].index)].pivot_table(index="day", columns="hour", values="RO").sort_index()

E_MAX, ETA = 2, 0.85
ETA_C = ETA_D = np.sqrt(ETA)
ACTS = np.array([-1, 0, 1])


def reward(p, a):
    return np.where(a > 0, ETA_D * p, np.where(a < 0, -p / ETA_C, 0.0))


def dp_days(P):
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
    """Fixed plan (D x 24) from an empty battery; infeasible actions become idle. Returns revenue and executed actions."""
    s = np.zeros(P.shape[0], dtype=int)
    tot = np.zeros(P.shape[0])
    done = np.zeros_like(acts)
    for h in range(24):
        a = acts[:, h].copy()
        a[(a > 0) & (s == 0)] = 0
        a[(a < 0) & (s == E_MAX)] = 0
        tot += reward(P[:, h], a)
        done[:, h] = a
        s = s - a
    return tot, done


P = RO.values
days = RO.index
prev_ok = np.array([i >= 7 and (days[i] - days[i - 1]).days == 1 and (days[i] - days[i - 7]).days == 7 for i in range(len(days))])
yr = days.year
iTE = np.where(prev_ok & (yr >= 2025))[0]
assert len(iTE) == NUMS["split"]["test_days"] and str(days[iTE[0]].date()) == NUMS["split"]["test_first"]

V, Aall = dp_days(P)
pf = V[:, 0, 0]
yest_plan = np.zeros((len(days), 24), dtype=int)
s0 = np.zeros(len(days), dtype=int)
for h in range(24):
    a = Aall[np.arange(len(days)), h, s0]
    yest_plan[:, h] = a
    s0 = s0 - a
naive = np.full(len(days), np.nan)
naive[1:] = run_schedule(P[1:], yest_plan[:-1])[0]
assert round(pf[iTE].sum() / 1000, 1) == NUMS["bat"]["pf_k"], pf[iTE].sum()
assert round(naive[iTE].sum() / 1000, 1) == NUMS["bat"]["naive_k"], naive[iTE].sum()

# ---------------------------------------------------------------- the tabular Q-learning agent (first seed), replayed
QB = P.argsort(axis=1).argsort(axis=1) // 6                 # quartile of each hour within the day; agent uses yesterday's row
Q = np.load(os.path.join(HERE, "cache", "q_table.npy"))
BQ = json.load(open(os.path.join(HERE, "cache", "battery_v3.json")))["q"][0]
assert BQ["seed"] == SEED
s = np.zeros(len(iTE), dtype=int)
agent = np.zeros(len(iTE))
agent_act = np.zeros((len(iTE), 24), dtype=int)
B = QB[iTE - 1]
for h in range(24):
    q = Q[h, B[:, h], s].copy()
    m = np.ones((len(s), 3), dtype=bool)
    m[:, 0] = s < E_MAX
    m[:, 2] = s > 0
    q[~m] = -np.inf
    a = ACTS[q.argmax(1)]
    agent += reward(P[iTE, h], a)
    agent_act[:, h] = a
    s = s - a
assert np.allclose(agent, BQ["test"]), "replayed agent differs from cache/battery_v3.json"

# ---------------------------------------------------------------- the four plans
prof = RO[(yr >= 2019) & (yr <= 2024)].mean().values        # mean hourly price 2019-2024 (training + validation)
print("mean profile 2019-2024 (EUR/MWh):", np.round(prof).astype(int).tolist())
H = {"night": (2, 3), "morning": (7, 8), "midday": (12, 13), "evening": (19, 20)}
assert set(np.argsort(prof)[:2]) <= {2, 3, 4} and set(np.argsort(-prof)[:2]) <= {18, 19, 20}


def fixed(charge, discharge):
    a = np.zeros(24, dtype=int)
    a[list(charge)] = -1
    a[list(discharge)] = 1
    return a


FIX = [fixed(H["night"], H["evening"]), fixed(H["midday"], H["evening"]),
       fixed(H["night"] + H["midday"], H["morning"] + H["evening"])]
plans_rev, plans_act = [], []
for a in FIX:
    rv, ac = run_schedule(P[iTE], np.tile(a, (len(iTE), 1)))
    plans_rev.append(rv)
    plans_act.append(ac)
rv, ac = run_schedule(P[iTE], yest_plan[iTE - 1])            # plan 4: yesterday's optimal plan (the lecture's naive rule)
plans_rev.append(rv)
plans_act.append(ac)
R = np.array(plans_rev).T                                   # test days x 4
ACT = np.stack(plans_act, 1)                                # test days x 4 x 24
dist = np.abs(ACT - agent_act[:, None, :]).sum(2)           # hourly distance of each plan to what the agent did
closest = np.array([min(range(4), key=lambda k: (dist[i, k], abs(R[i, k] - agent[i]))) for i in range(len(iTE))])
best = R.argmax(1)

tot = {"plans_k": [round(float(v) / 1000, 1) for v in R.sum(0)], "agent_k": round(float(agent.sum()) / 1000, 1),
       "best_share": [round(100 * float((best == k).mean()), 0) for k in range(4)],
       "plan_ge_agent": [int((R[:, k] >= agent - 1e-9).sum()) for k in range(4)],
       "closest_share": [round(100 * float((closest == k).mean()), 0) for k in range(4)]}
print("test totals:", tot)

rng = np.random.default_rng(SEED)
pick = np.sort(rng.choice(len(iTE), 40, replace=False))
pool = []
for i in pick:
    d = iTE[i]
    pool.append({"date": str(days[d].date()), "dow": int(days[d].dayofweek),
                 "yp": [round(float(v), 1) for v in P[d - 1]], "tp": [round(float(v), 1) for v in P[d]],
                 "yplan": yest_plan[d - 1].tolist(), "agent_act": agent_act[i].tolist(),
                 "rev": [round(float(v), 1) for v in R[i]], "agent": round(float(agent[i]), 1), "pf": round(float(pf[d]), 1),
                 "best": int(best[i]), "closest": int(closest[i])})
b = NUMS["bat"]
info = {"seed": SEED, "per_game": 8, "pool": len(pool), "n_test": int(len(iTE)), "test_first": NUMS["split"]["test_first"],
        "test_last": NUMS["split"]["test_last"], "hours": H, "pf_k": b["pf_k"], "naive_k": b["naive_k"], "naive_cap": b["naive_cap"],
        "q_k": b["q_k"], "q_cap": b["q_cap"], "dqn_k": b["dqn_k"], "dqn_cap": b["dqn_cap"], **tot}
with open(OUT, "w", encoding="utf-8") as f:
    f.write("// generated by lectures/L10/make_game_L10.py -- do not edit\n")
    f.write("window.NN_GAME_L10 = " + json.dumps({"info": info, "pool": pool}, ensure_ascii=False) + ";\n")
print(OUT, {k: info[k] for k in ("n_test", "pool", "agent_k", "plans_k")})


# ---------------------------------------------------------------- the two pages
def ro(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


pk, ak = info["plans_k"], info["agent_k"]
from game_page import write_game  # noqa: E402

write_game(10, en=dict(
    title="You or the agent? A battery on the Romanian day-ahead market",
    lead="You run a 2 MWh battery on the Romanian day-ahead electricity market. Each card is a real day of 2025 or 2026. "
         "Like the agent of the lecture, you plan the day before its prices are known: you see only yesterday's 24 hourly prices "
         "and choose one of four simple plans. Then you see what your plan, the agent and a trader with perfect foresight earned "
         "on that day's real prices.",
    how="The battery charges or discharges 1 MWh of stored energy per hour, starts the day empty and loses 15% of the energy "
        "over a full cycle (85% round-trip efficiency). The agent is the tabular Q-learning agent of the lecture (reinforcement "
        "learning: it learned which hours to charge and discharge from 1,746 days of 2019-2023); its state is the hour, the stored "
        "energy and the rank of that hour among yesterday's prices. A card is won against the agent when your plan earns at least as "
        "much as the agent; the score counts the cards on which you chose the best of the four plans.",
    lesson=[f"<li>Over the {info['n_test']} test days (1 January 2025 to 24 October 2026), perfect foresight earns {info['pf_k']:.1f} thousand EUR, "
            f"yesterday's optimal plan {info['naive_k']:.1f} thousand ({info['naive_cap']:.1f}% of the maximum), tabular Q-learning "
            f"{info['q_k']:.1f} thousand ({info['q_cap']:.1f}%) and the deep Q-network (DQN) {info['dqn_k']:.1f} thousand ({info['dqn_cap']:.1f}%), means of five seeds.</li>",
            f"<li>The three fixed plans of the game earn {pk[0]:.1f} (night to evening), {pk[1]:.1f} (midday to evening) and {pk[2]:.1f} thousand EUR "
            f"(two cycles) on the same days; the agent of the game (first seed) earns {ak:.1f} thousand.</li>",
            "<li>A simple rule that uses all 24 of yesterday's prices beats both learning agents: reinforcement learning must always be "
            "benchmarked against the rule already in use.</li>"],
    src="Data: hourly day-ahead prices for Romania, energy-charts.info (Fraunhofer Institute for Solar Energy Systems, ISE), up to 31 October 2026; "
        f"hours in Central European Time (CET). The {info['pool']} days of the game are drawn at random from the {info['n_test']} test days (seed {SEED}); "
        f"every game draws {info['per_game']} of them. The hours of the fixed plans come from the mean price profile of 2019-2024. "
        "Code: lectures/L10/make_game_L10.py."),
  ro=dict(
    title="Dumneavoastră sau agentul? O baterie pe piața românească pentru ziua următoare",
    lead="Administrați o baterie de 2 MWh pe piața românească de energie electrică pentru ziua următoare. Fiecare card este o zi reală "
         "din 2025 sau 2026. Ca și agentul din curs, planificați ziua înainte de a-i cunoaște prețurile: vedeți doar cele 24 de prețuri orare "
         "de ieri și alegeți unul dintre patru planuri simple. Apoi vedeți cît au cîștigat planul dumneavoastră, agentul și un operator "
         "cu previziune perfectă la prețurile reale ale zilei.",
    how="Bateria încarcă sau descarcă 1 MWh de energie stocată pe oră, începe ziua goală și pierde 15 % din energie pe un ciclu complet "
        "(randament dus-întors de 85 %). Agentul este agentul Q-learning tabelar din curs (reinforcement learning: a învățat în ce ore "
        "să încarce și să descarce din 1.746 de zile din 2019-2023); starea sa este ora, energia stocată și rangul orei respective "
        "între prețurile de ieri. Un card este cîștigat în fața agentului cînd planul dumneavoastră aduce cel puțin cît agentul; scorul "
        "numără cardurile pe care ați ales cel mai bun dintre cele patru planuri.",
    lesson=[f"<li>Pe cele {info['n_test']} de zile de test (1 ianuarie 2025 - 24 octombrie 2026), previziunea perfectă aduce {ro(info['pf_k'])} mii EUR, "
            f"planul optim de ieri {ro(info['naive_k'])} mii ({ro(info['naive_cap'])} % din maxim), Q-learning tabelar {ro(info['q_k'])} mii "
            f"({ro(info['q_cap'])} %), iar rețeaua Q profundă (DQN, deep Q-network) {ro(info['dqn_k'])} mii ({ro(info['dqn_cap'])} %), medii pe cinci valori inițiale.</li>",
            f"<li>Cele trei planuri fixe din joc aduc, în aceleași zile, {ro(pk[0])} (noaptea - seara), {ro(pk[1])} (la prînz - seara) și {ro(pk[2])} mii EUR "
            f"(două cicluri); agentul din joc (prima valoare inițială) aduce {ro(ak)} mii.</li>",
            "<li>O regulă simplă care folosește toate cele 24 de prețuri de ieri îi întrece pe ambii agenți: un agent de reinforcement learning "
            "trebuie comparat întotdeauna cu regula folosită deja în practică.</li>"],
    src="Date: prețurile orare pentru ziua următoare în România, energy-charts.info (Institutul Fraunhofer pentru Sisteme de Energie Solară, ISE), "
        f"pînă la 31 octombrie 2026; orele sînt în ora Europei Centrale (CET). Cele {info['pool']} de zile din joc sînt extrase aleator dintre cele "
        f"{info['n_test']} de zile de test (sămînța {SEED}); fiecare joc extrage {info['per_game']} dintre ele. Orele planurilor fixe provin din "
        "profilul mediu al prețurilor din 2019-2024. Cod: lectures/L10/make_game_L10.py."),
  plotly=True)
print("pages written")
