"""
make_game_L1.py -- data of the Lecture 1 game "You or the network?" (interactive/L1_game.html, L1_game_ro.html).

The student decides on real HMEQ home-equity loans (approve or refuse) and then sees what the network of the lecture
decided and whether the loan defaulted. Same data, split (60/20/20, stratified, seed of the course) and network
(MLP 64-64, ReLU, Adam, early stopping, seed 0) as analysis_L1.py, so the test AUC equals auc_mlp_seed0 of L1_numbers.json.
The pool of cards is drawn at random from the TEST loans only (12 defaults, 24 repaid, seed of the course); every game
draws 10 cards from this pool.

Run:  python3 lectures/L1/make_game_L1.py      ->  interactive/L1_game_data.js
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = npl.seed_all()
torch.set_num_threads(1)
OUT = os.path.join(HERE, "..", "..", "interactive", "L1_game_data.js")
NUM = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]


def encode(tr, others):
    med = tr[NUM].median()

    def f(d):
        x = pd.DataFrame(index=d.index)
        for c in NUM:
            x[c] = d[c].fillna(med[c])
            x["miss_" + c] = d[c].isna().astype(float)
        for c, levels in (("REASON", ["DebtCon", "HomeImp"]), ("JOB", ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"])):
            for lv in levels:
                x[f"{c}_{lv}"] = (d[c] == lv).astype(float)
        return x
    xtr = f(tr)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    return [((xtr - mu) / sd).values.astype(np.float32)] + [((f(o) - mu) / sd).values.astype(np.float32) for o in others]


class MLP(nn.Module):
    def __init__(self, d, hidden=(64, 64)):
        super().__init__()
        layers, k = [], d
        for h in hidden:
            layers += [nn.Linear(k, h), nn.ReLU()]
            k = h
        layers.append(nn.Linear(k, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xva, yva, lr=1e-3, epochs=300, bs=128, patience=25, seed=0):
    """Identical to train_mlp of analysis_L1.py (Adam, mini-batches, best validation epoch)."""
    torch.manual_seed(SEED + seed)
    rng = np.random.default_rng(SEED + seed)
    m = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0.0)
    lf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        m.train()
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            lf(m(Xt[b]), yt[b]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lf(m(Xv), yv))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m


h = npl.hmeq()
y = h.pop("BAD").values.astype(int)
idx = np.arange(len(h))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=y[i_tr], random_state=SEED)
Xtr, Xva, Xte = encode(h.iloc[i_tr], [h.iloc[i_va], h.iloc[i_te]])
m = train_mlp(Xtr, y[i_tr], Xva, y[i_va])
with torch.no_grad():
    p = torch.sigmoid(m(torch.tensor(Xte))).numpy().ravel()
yte = y[i_te]
auc = roc_auc_score(yte, p)
acc = float(((p >= 0.5).astype(int) == yte).mean())

rng = np.random.default_rng(SEED)
bad_pos = rng.choice(np.where(yte == 1)[0], 12, replace=False)
good_pos = rng.choice(np.where(yte == 0)[0], 24, replace=False)
pool = []
for k in np.concatenate([bad_pos, good_pos]):
    r = h.iloc[i_te[k]]
    v = lambda c, d=0: None if pd.isna(r[c]) else round(float(r[c]), d)  # noqa: E731
    pool.append({"loan": v("LOAN"), "mortdue": v("MORTDUE"), "value": v("VALUE"), "reason": None if pd.isna(r["REASON"]) else r["REASON"],
                 "job": None if pd.isna(r["JOB"]) else r["JOB"], "yoj": v("YOJ", 1), "derog": v("DEROG"), "delinq": v("DELINQ"),
                 "clage": v("CLAGE"), "ninq": v("NINQ"), "clno": v("CLNO"), "debtinc": v("DEBTINC", 1),
                 "p": round(float(p[k]), 3), "bad": int(yte[k])})
info = {"n": int(len(h)), "n_test": int(len(i_te)), "bad_pct": round(100 * float(y.mean()), 1), "auc": round(float(auc), 3),
        "acc": round(acc, 3), "seed": SEED, "pool_bad": 12, "pool_good": 24, "per_game": 10}
with open(OUT, "w", encoding="utf-8") as f:
    f.write("// generated by lectures/L1/make_game_L1.py -- do not edit\n")
    f.write("window.NN_GAME_L1 = " + json.dumps({"info": info, "pool": pool}, ensure_ascii=False) + ";\n")
print(OUT, info)

# ------------------------------------------------------------------ the two pages (interactive/L1_game.html, L1_game_ro.html)
from game_page import write_game  # noqa: E402

write_game(1, en=dict(
    title="You or the network? A credit game",
    lead="You are the credit officer of a bank. You see 10 real home-equity loan applications and decide: approve or refuse. After each decision "
         "you see what the network of Lecture 1 decided and whether the client repaid. In the full data 19.9% of the loans defaulted; in the game "
         "one application in three is a default.",
    how="The network is the multilayer perceptron (MLP 64-64) of the lecture, trained on 3,576 other loans; it never saw these applications. "
        "It refuses a loan when its probability of default is at least 50%.",
    lesson=[f"<li>On all {info['n_test']:,} test loans the network decides correctly in {100 * info['acc']:.1f}% of cases (AUC, the area under the ROC curve, {info['auc']:.3f}).</li>",
            "<li>A signal people tend to overlook: when the debt-to-income ratio is missing from the file, 62.0% of the loans default; when it is reported, 8.6%. The network learned this from the data.</li>",
            "<li>50% is not necessarily the right threshold: if a default costs five times as much as a lost good client, the cost-minimising threshold is 1/6 (slide 40 of the lecture).</li>"],
    src="Data: HMEQ home-equity loans (Baesens, Roesch and Scheule, 2016; 5,960 loans). The 36 applications of the game are drawn at random from the test loans "
        "(12 defaults, 24 repaid; seed 20261108); every game draws 10 of them. Code: lectures/L1/make_game_L1.py."),
  ro=dict(
    title="Dumneavoastră sau rețeaua? Un joc de creditare",
    lead="Sînteți analistul de credit al unei bănci. Vedeți 10 cereri reale de credit garantat cu locuința și decideți: aprobați sau respingeți. "
         "După fiecare decizie vedeți ce a decis rețeaua din cursul 1 și dacă clientul a rambursat. În datele complete, 19,9 % dintre credite au intrat "
         "în neplată; în joc, una din trei cereri este o neplată.",
    how="Rețeaua este perceptronul multistrat (MLP 64-64) din curs, antrenat pe alte 3.576 de credite; nu a văzut aceste cereri. "
        "Respinge un credit cînd probabilitatea de neplată este de cel puțin 50 %.",
    lesson=[f"<li>Pe toate cele {info['n_test']:,} de credite de test, rețeaua decide corect în {100 * info['acc']:.1f} % din cazuri (AUC, aria de sub curba ROC, {info['auc']:.3f}).</li>".replace(f"{info['n_test']:,}", f"{info['n_test']:,}".replace(",", ".")).replace(f"{100 * info['acc']:.1f}", f"{100 * info['acc']:.1f}".replace(".", ",")).replace(f"{info['auc']:.3f}", f"{info['auc']:.3f}".replace(".", ",")),
            "<li>Un semnal pe care oamenii îl trec ușor cu vederea: cînd raportul datorii/venit lipsește din dosar, 62,0 % dintre credite ajung în neplată; cînd este raportat, 8,6 %. Rețeaua a învățat asta din date.</li>",
            "<li>Pragul de 50 % nu este neapărat cel potrivit: dacă o neplată costă de cinci ori cît un client bun pierdut, pragul care minimizează costul este 1/6 (slide-ul 40 din curs).</li>"],
    src="Date: creditele HMEQ garantate cu locuința (Baesens, Roesch și Scheule, 2016; 5.960 de credite). Cele 36 de cereri din joc sînt extrase aleator "
        "dintre creditele de test (12 neplăți, 24 rambursate; sămînța 20261108); fiecare joc extrage 10 dintre ele. Cod: lectures/L1/make_game_L1.py."))
print("pages written")
