"""
make_game_L3.py -- data of the Lecture 3 game "Stop early or train longer?" (interactive/L3_game.html, L3_game_ro.html).

The student sees the training curve (and, on most cards, the validation curve) of a real run of the lecture and chooses one of
three epochs at which to stop. The right answer is the epoch whose network has the lowest loss on the TEST set; the model is
early stopping, which picks the epoch with the lowest validation loss among the three.

Runs (same data, splits, networks and seeds as analysis_L3.py and interactive_data_L3.py):
  * German credit, 600 / 200 / 200 applicants, MLP 256-256, AdamW, 150 epochs, batches of 64, dropout {0, 0.3, 0.5} x
    weight decay {0, 0.1, 1}, seed 0 (the 9 curves of cache/interactive_v1.json, key "ov");
  * card fraud, time-ordered split, MLP 256-256 without regularisation, 120 epochs, batches of 256, seed 0 (key "es").
The cache holds training and validation curves only, so the runs are repeated here with the test loss and test score recorded
after every epoch (recording does not touch the random streams). The script checks that the training and validation curves
equal the cache exactly and that the numbers equal L3_numbers.json. About 15 seconds on one CPU thread.

Run:  python3 lectures/L3/make_game_L3.py      ->  interactive/L3_game_data.js, L3_game.html, L3_game_ro.html
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import xgboost  # noqa: E402,F401  (import before torch, as in analysis_L3.py: both ship an OpenMP runtime)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

torch.set_num_threads(1)
SEED = npl.seed_all()
OUT = os.path.join(HERE, "..", "..", "interactive", "L3_game_data.js")
N = json.load(open(os.path.join(HERE, "L3_numbers.json")))
CACHE = json.load(open(os.path.join(HERE, "cache", "interactive_v1.json")))


class Net(nn.Module):
    def __init__(self, d, hidden=(256, 256), p=0.0):
        super().__init__()
        layers, k = [], d
        for h in hidden:
            layers += [nn.Linear(k, h), nn.ReLU()] + ([nn.Dropout(p)] if p > 0 else [])
            k = h
        layers.append(nn.Linear(k, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def proba(m, X):
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy().ravel()


def run(Xtr, ytr, Xva, yva, Xte, yte, p, wd, epochs, bs, score):
    """fit(..., early=False, record=True) of analysis_L3.py (AdamW, lr 0.001, seed 0), plus the test loss and score per epoch."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    m = Net(Xtr.shape[1], (256, 256), p)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss()
    T = lambda X, y: (torch.tensor(X), torch.tensor(y, dtype=torch.float32).view(-1, 1))  # noqa: E731
    (Xt, yt), (Xv, yv), (Xe, ye) = T(Xtr, ytr), T(Xva, yva), T(Xte, yte)
    H = []
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
            H.append([float(lf(m(Xt), yt)), float(lf(m(Xv), yv)), float(lf(m(Xe), ye)), score(proba(m, Xte))])
    return np.array(H)


# ------------------------------------------------------------------ German credit (600 / 200 / 200, stratified)
g = npl.german_credit()
yg = g["default"].values.astype(int)
Xg_raw = g.drop(columns="default")


def encode_german(tr, others):
    cat = [c for c in tr.columns if tr[c].dtype == object]
    num = [c for c in tr.columns if c not in cat]
    levels = {c: sorted(tr[c].unique()) for c in cat}

    def f(d):
        x = d[num].astype(float).copy()
        for c in cat:
            for lv in levels[c][1:]:
                x[f"{c}_{lv}"] = (d[c] == lv).astype(float)
        return x
    xtr = f(tr)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    return [((xtr - mu) / sd).values.astype(np.float32)] + [((f(o) - mu) / sd).values.astype(np.float32) for o in others]


gi = np.arange(len(g))
g_tr, g_te = train_test_split(gi, test_size=0.2, stratify=yg, random_state=SEED)
g_tr, g_va = train_test_split(g_tr, test_size=0.25, stratify=yg[g_tr], random_state=SEED)
Gtr, Gva, Gte = encode_german(Xg_raw.iloc[g_tr], [Xg_raw.iloc[g_va], Xg_raw.iloc[g_te]])
runs = []
for p in (0.0, 0.3, 0.5):
    for wd in (0.0, 0.1, 1.0):
        H = run(Gtr, yg[g_tr], Gva, yg[g_va], Gte, yg[g_te], p, wd, 150, 64, lambda q: float(roc_auc_score(yg[g_te], q)))
        c = CACHE["ov"]["c"][f"{p}|{wd}"]
        assert np.array_equal(np.round(H[:, 0], 4), c["tr"]) and np.array_equal(np.round(H[:, 1], 4), c["va"]), (p, wd)
        runs.append({"data": "german", "p": p, "wd": wd, "tr": np.round(H[:, 0], 4).tolist(), "va": np.round(H[:, 1], 4).tolist(),
                     "te": np.round(H[:, 2], 4).tolist(), "sc": np.round(H[:, 3], 3).tolist()})
o = runs[0]
assert int(np.argmin(o["va"])) + 1 == N["overfit"]["best_ep"] and round(min(o["va"]), 3) == N["overfit"]["val_min"]
assert round(o["va"][-1], 3) == N["overfit"]["val_last"] and round(o["sc"][-1], 3) == N["overfit"]["auc_over_last"]
assert round(runs[7]["sc"][-1], 3) == N["overfit"]["auc_reg_last"]                      # dropout 0.5, weight decay 0.1

# ------------------------------------------------------------------ card fraud (time-ordered 60 / 20 / 20)
fr = npl.fraud().sort_values("Time").reset_index(drop=True)
Xf0 = np.c_[fr[[f"V{i}" for i in range(1, 29)]].values, np.log1p(fr["Amount"].values)].astype(np.float32)
yf, wf = fr["Class"].values.astype(int), fr["weight"].values
c1, c2 = np.quantile(fr["Time"], [0.6, 0.8])
f_tr = (fr["Time"] <= c1).values
f_va = ((fr["Time"] > c1) & (fr["Time"] <= c2)).values
f_te = (fr["Time"] > c2).values
Xf = (Xf0 - Xf0[f_tr].mean(0)) / Xf0[f_tr].std(0)
H = run(Xf[f_tr], yf[f_tr], Xf[f_va], yf[f_va], Xf[f_te], yf[f_te], 0.0, 0.0, 120, 256,
        lambda q: float(average_precision_score(yf[f_te], q, sample_weight=wf[f_te])))
assert np.array_equal(np.round(H[:, 1], 5), CACHE["es"]["va"])
assert int(np.argmin(H[:, 1])) + 1 == N["es"]["best_ep"] and round(H[-1, 1], 4) == N["es"]["val_last"]
runs.append({"data": "fraud", "p": 0.0, "wd": 0.0, "tr": [float(f"{v:.3e}") for v in H[:, 0]], "va": np.round(H[:, 1], 5).tolist(),
             "te": np.round(H[:, 2], 5).tolist(), "sc": np.round(H[:, 3], 3).tolist()})

best_te = [int(np.argmin(r_["te"])) + 1 for r_ in runs[:9]]
best_va = [int(np.argmin(r_["va"])) + 1 for r_ in runs[:9]]
info = {"seed": SEED, "german": {"cand": [1, 3, 5, 10, 40, 150], "n_train": int(len(g_tr)), "n_val": int(len(g_va)), "n_test": int(len(g_te))},
        "fraud": {"cand": [1, 4, 10, 20, 60, 120], "n_train": int(f_tr.sum()), "n_val": int(f_va.sum()), "n_test": int(f_te.sum())},
        "best_te": [min(best_te), max(best_te)], "best_va": [min(best_va), max(best_va)], "fraud_best_te": int(np.argmin(H[:, 2])) + 1,
        "n_blind": 3, "n_cards": 9}
with open(OUT, "w", encoding="utf-8") as f:
    f.write("// generated by lectures/L3/make_game_L3.py -- do not edit\n")
    f.write("window.NN_GAME_L3 = " + json.dumps({"info": info, "runs": runs}, ensure_ascii=False, separators=(",", ":")) + ";\n")
print(OUT, info, os.path.getsize(OUT))

# ------------------------------------------------------------------ the two pages
from game_page import write_game  # noqa: E402

ov, ab = N["overfit"], N["abl"]
ro = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")  # noqa: E731

write_game(3, en=dict(
    title="Stop early or train longer? A game on training curves",
    lead="You train a neural network and must decide when to stop. Each chart is a real run of Lecture 3: the training loss and, on most cards, "
         "the loss on a validation set the network does not train on. Pick one of three epochs. The right answer is the epoch whose network makes "
         "the smallest error on new data: a test set nobody looked at during training. Nine charts, about four minutes.",
    how="Seven charts come from German credit applications (600 for training, 200 for validation, 200 for test); two from card payments split by "
        "time. The loss is the cross-entropy: the lower, the better the predicted probabilities. On the first three charts you see only the training "
        "loss, as in a report that shows the fit to the training data. The model you play against is early stopping: it picks the epoch with "
        "the lowest validation loss.",
    lesson=[f"<li>On German credit the training loss of the network without regularisation falls to {ov['train_last']:.4f} after "
            f"{ov['epochs']} epochs, while its validation loss rises from {ov['val_min']:.3f} at epoch {ov['best_ep']} to {ov['val_last']:.3f}: "
            "the network memorises its 600 applicants. A falling training loss says nothing about new clients.</li>",
            f"<li>In the 9 German-credit runs of the game, with or without dropout and weight decay, the test loss is lowest between epochs "
            f"{info['best_te'][0]} and {info['best_te'][1]}, and the validation loss points to epochs {info['best_va'][0]} to {info['best_va'][1]}.</li>",
            f"<li>On card fraud, early stopping raises the test area under the precision-recall curve (PR-AUC) from {ab['none']['ap']:.3f} to "
            f"{ab['es']['ap']:.3f} (5 seeds); weight decay ({ab['wd']['ap']:.3f}) or dropout ({ab['dropout']['ap']:.3f}) alone do not.</li>"],
    src="Data: German credit (Hofmann, 1994), University of California, Irvine (UCI) repository, 1,000 applicants split 60/20/20 at random; card "
        "fraud of the Université libre de Bruxelles (ULB) and Worldline, Open Machine Learning platform (OpenML) dataset 1597, split by time. Runs: the networks of analysis_L3.py (seed 20261108), repeated with the test loss recorded after "
        "every epoch. Code: lectures/L3/make_game_L3.py."),
  ro=dict(
    title="Oprire timpurie sau antrenare mai lungă? Un joc cu curbe de învățare",
    lead="Antrenați o rețea neuronală și trebuie să decideți cînd vă opriți. Fiecare grafic este o antrenare reală din cursul 3: pierderea de "
         "antrenare și, pe majoritatea cartonașelor, pierderea pe un set de validare pe care rețeaua nu se antrenează. Alegeți una din trei epoci. "
         "Răspunsul corect este epoca în care rețeaua greșește cel mai puțin pe date noi: un set de test pe care nu l-a folosit nimeni în timpul "
         "antrenării. Nouă grafice, circa patru minute.",
    how="Șapte grafice provin din cererile de credit germane (600 pentru antrenare, 200 pentru validare, 200 pentru test), două din plățile cu "
        "carduri împărțite cronologic. Pierderea este entropia încrucișată: cu cît este mai mică, cu atît probabilitățile estimate sînt mai bune. "
        "Pe primele trei grafice vedeți doar pierderea de antrenare, ca într-un raport care arată numai potrivirea pe datele de antrenare. "
        "Modelul cu care vă comparați este oprirea timpurie: alege epoca cu cea mai mică pierdere de validare.",
    lesson=[f"<li>Pe creditele germane, pierderea de antrenare a rețelei fără regularizare scade la {ro(ov['train_last'], 4)} după "
            f"{ov['epochs']} de epoci, în timp ce pierderea de validare crește de la {ro(ov['val_min'])} la epoca {ov['best_ep']} la "
            f"{ro(ov['val_last'])}: rețeaua învață pe de rost cei 600 de solicitanți. O pierdere de antrenare care scade nu spune nimic despre "
            "clienții noi.</li>",
            f"<li>În cele 9 antrenări pe creditele germane din joc, cu sau fără dropout și penalizarea ponderilor, pierderea pe test este minimă "
            f"între epocile {info['best_te'][0]} și {info['best_te'][1]}, iar pierderea de validare indică epocile {info['best_va'][0]}–"
            f"{info['best_va'][1]}.</li>",
            f"<li>Pe fraudele cu carduri, oprirea timpurie crește aria de sub curba precizie-sensibilitate (PR-AUC) pe test de la "
            f"{ro(ab['none']['ap'])} la {ro(ab['es']['ap'])} (5 inițializări); penalizarea ponderilor ({ro(ab['wd']['ap'])}) sau dropout "
            f"({ro(ab['dropout']['ap'])}) singure nu o cresc.</li>"],
    src="Date: creditele germane (Hofmann, 1994), depozitul University of California, Irvine (UCI), 1.000 de solicitanți împărțiți aleator 60/20/20; "
        "fraudele cu carduri ale Université libre de Bruxelles (ULB) și Worldline, setul 1597 al platformei Open Machine Learning (OpenML), împărțite cronologic. Antrenări: rețelele din analysis_L3.py (sămînța 20261108), repetate cu "
        "pierderea pe test înregistrată după fiecare epocă. Cod: lectures/L3/make_game_L3.py."))
print("pages written")
