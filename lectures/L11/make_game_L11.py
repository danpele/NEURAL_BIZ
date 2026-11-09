"""
make_game_L11.py -- data of the Lecture 11 game "Which feature drove the decision?" (interactive/L11_game.html, L11_game_ro.html).

Each card is a real test applicant of the HMEQ credit data of analysis_L11.py, with the decision of the lecture's network
(MLP 64-32, first of the 10 training seeds; refuse when the predicted probability of default is at least 50 %). The student
picks, among four features, the one with the largest Kernel SHAP contribution for this applicant; then sees the SHAP top
features, LIME's answer (model answer = the option with the largest LIME contribution, slope x standardised value, as on
the LIME-versus-SHAP slide) and how often the top SHAP feature stays the same over the 10 training seeds of the network.

Sources (all from analysis_L11.py, same data, split, seed and settings):
  - Kernel SHAP of the first network for 400 test applicants: cache/shap_main_v1.pkl;
  - Kernel SHAP of the 10 networks (seeds) for the first 200 of them: cache/seed_shap_v1.pkl;
  - the networks: cache/credit_models_v1.pkl;
  - LIME is cached for the lecture's applicant only (cache/lime_v2.pkl); for the game applicants it is recomputed with the
    lecture's function and settings (5,000 samples, default kernel width, seed 0); the script checks that the recomputation
    reproduces the cached LIME weights of the lecture's applicant.
The pool: 30 applicants drawn at random (seed of the course) among those 200 test applicants, 12 refused and 18 approved
by the network; every game draws 8 of them.

Run:  python3 lectures/L11/make_game_L11.py      ->  interactive/L11_game_data.js and the two pages
"""
import json
import os
import pickle
import sys
import warnings

import torch  # noqa: F401  (before scikit-learn, as in analysis_L11.py)
import torch.nn as nn
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402

warnings.filterwarnings("ignore")
SEED = npl.seed_all()
torch.set_num_threads(4)
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "..", "..", "interactive", "L11_game_data.js")
NUMS = json.load(open(os.path.join(HERE, "L11_numbers.json")))
load = lambda n: pickle.load(open(os.path.join(CACHE, n + ".pkl"), "rb"))  # noqa: E731

# ---------------------------------------------------------------- data and split exactly as in analysis_L11.py
h = npl.hmeq()
NUM = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
JOBS = ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]
y_all = h["BAD"].values
idx = np.arange(len(h))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=y_all[i_tr], random_state=SEED)
med = h.iloc[i_tr][NUM].median()


def design(df):
    X = pd.DataFrame(index=df.index)
    for c in NUM:
        X[c] = df[c].fillna(med[c])
    X["DEBTINC_missing"] = df["DEBTINC"].isna().astype(float)
    X["REASON_HomeImp"] = (df["REASON"] == "HomeImp").astype(float)
    for j in JOBS:
        X["JOB_" + j] = (df["JOB"] == j).astype(float)
    return X


XR = design(h)
FEATS = list(XR.columns)
mu, sdv = XR.iloc[i_tr].mean(), XR.iloc[i_tr].std()
XS = ((XR - mu) / sdv).values.astype(np.float32)
Xtr, Xte = XS[i_tr], XS[i_te]
yte = y_all[i_te]
rng = np.random.default_rng(SEED)
BG = Xtr[rng.choice(len(Xtr), 100, replace=False)]
i_sub = rng.choice(len(Xte), 400, replace=False)
Xsub = Xte[i_sub]
assert len(i_te) == NUMS["hmeq"]["n_test"]
assert NUMS["app"]["f1"] == "DEBTINC_missing" and NUMS["lime"]["top1"] == "DELINQ"      # named in the end-screen text


class MLP(nn.Module):
    def __init__(self, d, h1=64, h2=32, p=0.1, out=1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(p), nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p),
                               nn.Linear(h2, out))

    def forward(self, x):
        return self.f(x).squeeze(-1)


CM = load("credit_models_v1")
net0 = MLP(Xtr.shape[1])
net0.load_state_dict(CM["mlp"][0]["state"])
net0.eval()


def f0(X):
    with torch.no_grad():
        return net0(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy().astype(np.float64)


sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
SH = load("shap_main_v1")
sv = SH["mlp"]
assert np.abs(sv.sum(1) + SH["mlp_base"] - f0(Xsub)).max() < 1e-4, "SHAP cache does not match the network"
SS = load("seed_shap_v1")
tops_seed = np.array([np.argmax(np.abs(s), 1) for s in SS["mlp"]])            # 10 seeds x 200 applicants


# ---------------------------------------------------------------- LIME with the lecture's settings, checked on its applicant
def lime_weights(x, num_samples=5000, kw=None, seed=0):
    e = LimeTabularExplainer(Xtr, mode="regression", feature_names=FEATS, discretize_continuous=False, sample_around_instance=True,
                             random_state=seed, kernel_width=kw).explain_instance(x, f0, num_features=len(FEATS), num_samples=num_samples)
    w = np.zeros(len(FEATS))
    for j, v in e.local_exp[1]:
        w[j] = v
    return w


p_sub = sig(f0(Xsub))
cand = np.where((p_sub > 0.5) & (p_sub < 0.8) & (yte[i_sub] == 1))[0]
k_app = int(cand[np.argmin(np.abs(p_sub[cand] - 0.65))])
assert int(XR.index[i_te[i_sub[k_app]]]) == NUMS["app"]["row"]
LM = load("lime_v2")
assert np.allclose(lime_weights(Xsub[k_app]), LM["main"], atol=1e-8), "LIME recomputation differs from cache/lime_v2.pkl"

# ---------------------------------------------------------------- the pool: 12 refused + 18 approved among the first 200
ref = np.where(p_sub[:200] >= 0.5)[0]
acc = np.where(p_sub[:200] < 0.5)[0]
pick = np.concatenate([rng.choice(ref, 12, replace=False), rng.choice(acc, 18, replace=False)])
pool, agree_top, agree_opt = [], 0, 0
for k in pick:
    x = Xsub[k]
    phi = sv[k]
    # LIME slope x standardised value (contribution); lime_weights reads local_exp[1], the fitted slopes in regression mode
    lime = lime_weights(x) * x
    ords = np.argsort(-np.abs(phi))
    opts = [int(ords[0]), int(ords[1])]
    lt = int(np.argmax(np.abs(lime)))
    if lt not in opts:
        opts.append(lt)
    rest = [int(j) for j in ords[2:10] if j not in opts]
    opts += [int(j) for j in rng.choice(rest, 4 - len(opts), replace=False)]
    opts = [opts[j] for j in rng.permutation(4)]
    lime_pick = int(np.argmax([abs(lime[j]) for j in opts]))
    agree_top += int(lt == ords[0])
    agree_opt += int(opts[lime_pick] == ords[0])
    r = h.iloc[i_te[i_sub[k]]]
    v = lambda c, d=0: None if pd.isna(r[c]) else round(float(r[c]), d)  # noqa: E731
    top5 = [int(j) for j in ords[:5]]
    pool.append({"row": int(XR.index[i_te[i_sub[k]]]),
                 "loan": v("LOAN"), "mortdue": v("MORTDUE"), "value": v("VALUE"), "reason": None if pd.isna(r["REASON"]) else r["REASON"],
                 "job": None if pd.isna(r["JOB"]) else r["JOB"], "yoj": v("YOJ", 1), "derog": v("DEROG"), "delinq": v("DELINQ"),
                 "clage": v("CLAGE"), "ninq": v("NINQ"), "clno": v("CLNO"), "debtinc": v("DEBTINC", 1),
                 "p": round(float(p_sub[k]), 3), "bad": int(yte[i_sub[k]]),
                 "opts": [FEATS[j] for j in opts], "shap_opt": [round(float(phi[j]), 3) for j in opts],
                 "lime_opt": [round(float(lime[j]), 3) for j in opts], "truth": int(np.argmax([abs(phi[j]) for j in opts])),
                 "lime_pick": lime_pick, "lime_top": FEATS[lt], "lime_top_val": round(float(lime[lt]), 3),
                 "top5": [FEATS[j] for j in top5], "top5_shap": [round(float(phi[j]), 3) for j in top5],
                 "top5_lime": [round(float(lime[j]), 3) for j in top5],
                 "seeds_same": int((tops_seed[:, k] == ords[0]).sum())})
    assert pool[-1]["truth"] == opts.index(int(ords[0]))
info = {"seed": SEED, "per_game": 8, "pool": len(pool), "pool_ref": 12, "pool_acc": 18, "n": NUMS["hmeq"]["n"],
        "n_test": NUMS["hmeq"]["n_test"], "auc": NUMS["perf"]["mlp0_auc"], "base_pd": NUMS["shap"]["base_pd"],
        "agree_top": agree_top, "agree_opt": agree_opt, "seeds": NUMS["stab"]["n_seeds"],
        "lime_rho": NUMS["lime"]["rho_shap"], "lime_top1_500": NUMS["lime"]["top1_same_500"], "local_mlp": NUMS["stab"]["local_mlp"],
        "app_shap": NUMS["app"]["f1"], "app_lime": NUMS["lime"]["top1"]}
with open(OUT, "w", encoding="utf-8") as f:
    f.write("// generated by lectures/L11/make_game_L11.py -- do not edit\n")
    f.write("window.NN_GAME_L11 = " + json.dumps({"info": info, "pool": pool}, ensure_ascii=False) + ";\n")
print(OUT, info)


# ---------------------------------------------------------------- the two pages
def ro(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


from game_page import write_game  # noqa: E402

write_game(11, en=dict(
    title="Which feature drove the decision?",
    lead="A bank's network has approved or refused a real home-equity loan application. For each of 8 applicants you see the file and the "
         "network's decision, and you choose, among four features, the one that pushed this decision the most. Then you see the answer of "
         "SHAP (SHapley Additive exPlanations, Kernel SHAP) and whether LIME (local interpretable model-agnostic explanations) agrees.",
    how=f"The network is the multilayer perceptron (MLP) of Lecture 11, trained on 3,576 other loans (area under the receiver operating characteristic curve, AUC, {info['auc']:.3f} "
        "on the test loans); it refuses a loan when its probability of default is at least 50%. The right answer is the option with the largest "
        "absolute Kernel SHAP contribution to the log-odds of default; LIME's answer is the option with the largest absolute LIME slope × standardised value. "
        "Both explanations were computed with the settings of the lecture.",
    lesson=[f"<li>LIME and SHAP name the same top feature for {info['agree_top']} of the {info['pool']} applicants of the game. For the applicant studied "
            "in the lecture, SHAP puts first the missing debt-to-income ratio (DEBTINC missing) and LIME the delinquent credit lines (DELINQ); "
            f"the Spearman correlation of their absolute contributions is {info['lime_rho']:.2f}.</li>",
            f"<li>Explanations are estimates and move with chance: with 500 samples LIME keeps the same top feature in only {info['lime_top1_500']:.0f}% of 30 runs, "
            f"and across the {info['seeds']} training seeds of the same network an applicant's top SHAP feature agrees for only {info['local_mlp']:.0f}% of the pairs of seeds.</li>",
            "<li>Before an explanation reaches a customer or a supervisor, check it: two methods, several seeds, fixed and reported settings.</li>"],
    src=f"Data: HMEQ home-equity loans (Baesens, Roesch and Scheule, 2016; {info['n']:,} loans). The {info['pool']} applicants of the game are drawn at random from the "
        f"test applicants explained in the lecture ({info['pool_ref']} refused and {info['pool_acc']} approved by the network; seed {SEED}); every game draws "
        f"{info['per_game']} of them. Kernel SHAP and the seed comparison come from the lecture's cached results; LIME is recomputed with the lecture's settings "
        "(5,000 samples, default kernel width, seed 0). Code: lectures/L11/make_game_L11.py."),
  ro=dict(
    title="Ce variabilă a determinat decizia?",
    lead="Rețeaua unei bănci a aprobat sau a respins o cerere reală de credit garantat cu locuința. Pentru fiecare dintre cei 8 solicitanți vedeți "
         "dosarul și decizia rețelei și alegeți, dintre patru variabile, pe cea care a împins cel mai mult această decizie. Apoi vedeți răspunsul "
         "SHAP (SHapley Additive exPlanations, Kernel SHAP) și dacă LIME (local interpretable model-agnostic explanations, explicații locale "
         "interpretabile, independente de model) este de acord.",
    how=f"Rețeaua este perceptronul multistrat (MLP) din cursul 11, antrenat pe alte 3.576 de credite (aria de sub curba ROC, receiver operating characteristic, AUC, {ro(info['auc'], 3)} "
        "pe creditele de test); respinge un credit cînd probabilitatea de neplată este de cel puțin 50 %. Răspunsul corect este varianta cu cea mai mare "
        "contribuție Kernel SHAP, în valoare absolută, la log-șansele de neplată; răspunsul LIME este varianta cu cea mai mare valoare absolută a produsului "
        "dintre panta LIME și valoarea standardizată. Ambele explicații au fost calculate cu setările din curs.",
    lesson=[f"<li>LIME și SHAP indică aceeași variabilă principală pentru {info['agree_top']} dintre cei {info['pool']} de solicitanți din joc. Pentru solicitantul "
            "studiat în curs, SHAP pune pe primul loc lipsa raportului datorii/venit (DEBTINC lipsă), iar LIME liniile de credit cu restanțe (DELINQ); "
            f"corelația Spearman a contribuțiilor lor absolute este {ro(info['lime_rho'], 2)}.</li>",
            f"<li>Explicațiile sînt estimări și variază cu hazardul: cu 500 de eșantioane, LIME păstrează aceeași variabilă principală doar în {ro(info['lime_top1_500'])} % "
            f"din 30 de rulări, iar între cele {info['seeds']} valori inițiale de antrenare ale aceleiași rețele variabila SHAP principală a unui solicitant "
            f"coincide doar pentru {ro(info['local_mlp'])} % din perechile de valori inițiale.</li>",
            "<li>Înainte ca o explicație să ajungă la un client sau la autoritatea de supraveghere, ea trebuie verificată: două metode, mai multe valori "
            "inițiale, setări fixate și raportate.</li>"],
    src=f"Date: creditele HMEQ garantate cu locuința (Baesens, Roesch și Scheule, 2016; {ro(info['n'])} de credite). Cei {info['pool']} de solicitanți din joc "
        f"sînt extrași aleator dintre solicitanții de test explicați în curs ({info['pool_ref']} respinși și {info['pool_acc']} aprobați de rețea; sămînța {SEED}); "
        f"fiecare joc extrage {info['per_game']} dintre ei. Kernel SHAP și comparația între valorile inițiale provin din rezultatele salvate ale cursului; LIME este "
        "recalculat cu setările din curs (5.000 de eșantioane, lățimea implicită a nucleului, sămînța 0). Cod: lectures/L11/make_game_L11.py."),
  plotly=False)
print("pages written")
