"""Inputs of the Lecture 2 interactive page (interactive/L2_data.js), from data/clean and the caches of analysis_L2.py
(lectures/L2/cache/*.json), with the same splits, models and seeds. New runs (validation curves of the learning-rate sweep,
gradient norms by depth) are cached in lectures/L2/cache/interactive_v1.json.
Run alone to refresh the cache:  python3 lectures/L2/interactive_data_L2.py --fresh
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
CDIR = os.path.join(HERE, "cache")
CACHE = os.path.join(CDIR, "interactive_v1.json")
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
DEPTHS = [2, 4, 6, 8, 10, 12, 16, 20]


def california():
    import numpy as np
    import nn_plot as npl
    from sklearn.model_selection import train_test_split
    SEED = 20261108
    ca = npl.california()
    feat = [c for c in ca.columns if c != "MedHouseVal_100k_usd"]
    Xc = ca[feat].values.astype(np.float32)
    yc = ca["MedHouseVal_100k_usd"].values.astype(np.float32)
    i_tr, i_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED)
    i_tr, i_va = train_test_split(i_tr, test_size=0.25, random_state=SEED)
    mu, sd = Xc[i_tr].mean(0), Xc[i_tr].std(0)
    Z = (Xc - mu) / sd
    return ca, Z[i_tr], Z[i_va], Z[i_te], yc[i_tr], yc[i_va], yc[i_te]


def compute():
    import numpy as np
    import nn_plot as npl
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    SEED = npl.seed_all()
    ca, Xtr, Xva, Xte, ytr, yva, yte = california()

    def run_sgd(lr, seed=0, epochs=30, bs=128):
        """run_reg('sgd', lr) of analysis_L2.py: MLP 64-64, plain SGD, validation MSE after each epoch."""
        torch.manual_seed(SEED + seed)
        rng = np.random.default_rng(SEED + seed)
        m = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr).view(-1, 1)
        Xv, yv = torch.tensor(Xva), torch.tensor(yva).view(-1, 1)
        lf, val = nn.MSELoss(), []
        for ep in range(epochs):
            m.train()
            idx = rng.permutation(len(Xt))
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                opt.zero_grad()
                lf(m(Xt[b]), yt[b]).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                v = float(lf(m(Xv), yv))
            val.append(v if np.isfinite(v) else 1e9)
        return val

    out = {"lr": {str(lr): run_sgd(lr) for lr in LRS}}

    # gradient norm of each layer at initialisation, one batch of 512 training block groups (grad_norms of analysis_L2.py)
    ACT = {"sig": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}

    def grad_norms(act, init, depth, seed):
        torch.manual_seed(SEED + seed)
        layers, k, lins = [], Xtr.shape[1], []
        for _ in range(depth):
            l_ = nn.Linear(k, 64)
            lins.append(l_)
            layers += [l_, ACT[act]()]
            k = 64
        last = nn.Linear(k, 1)
        lins.append(last)
        m = nn.Sequential(*layers, last)
        for l_ in lins:
            if init == "xavier":
                nn.init.xavier_normal_(l_.weight)
            elif init == "he":
                nn.init.kaiming_normal_(l_.weight, nonlinearity="relu")
            elif init == "small":
                nn.init.normal_(l_.weight, 0, 0.01)
            elif init == "large":
                nn.init.normal_(l_.weight, 0, 1.0)
            nn.init.zeros_(l_.bias)
        loss = nn.MSELoss()(m(torch.tensor(Xtr[:512])), torch.tensor(ytr[:512]).view(-1, 1))
        loss.backward()
        return [float(l_.weight.grad.norm()) for l_ in lins[:-1]]

    vg = {}
    for a in ACT:
        for ini in ("xavier", "he", "small", "large"):
            for d in DEPTHS:
                g = np.mean([grad_norms(a, ini, d, s) for s in range(5)], axis=0)
                vg[f"{a}|{ini}|{d}"] = [float(f"{v:.3e}") for v in g]
    out["vg"] = vg
    return out


def load(fresh=False):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    if os.path.exists(CACHE) and not fresh:
        r = json.load(open(CACHE))
    else:
        r = compute()
        json.dump(r, open(CACHE, "w"))
    ca, Xtr, Xva, Xte, ytr, yva, yte = california()
    lin = LinearRegression().fit(Xtr, ytr)
    out = {"mse_lin_val": float(np.mean((lin.predict(Xva) - yva) ** 2)), "n_tr": int(len(ytr)), "n_va": int(len(yva))}

    # 1. loss surface of the simple regression value ~ a + b * median income (all block groups, as analysis_L2.py)
    x = ca["MedInc"].values.astype(float)
    y = ca["MedHouseVal_100k_usd"].values.astype(float)
    gd = {}
    for k, xf in (("raw", x), ("std", (x - x.mean()) / x.std())):
        gd[k] = {"mx": float(xf.mean()), "mxx": float(np.mean(xf * xf)), "my": float(y.mean()), "mxy": float(np.mean(xf * y)),
                 "myy": float(np.mean(y * y))}
    out["gd"] = gd

    # 2. learning-rate sweep: validation curves of seed 0 (new run) and the final MSE of the 3 seeds (analysis cache)
    sw = json.load(open(os.path.join(CDIR, "lr_sweep_v1.json")))
    out["lr"] = {"lrs": LRS, "curve": {k: [min(v, 1e3) for v in r["lr"][k]] for k in r["lr"]},
                 "final": {k: [min(v, 1e3) for v in sw[k]] for k in sw}}

    # 3. optimisers x learning rates, 3 seeds, 60 epochs (analysis cache opt_tune_v1)
    ot = json.load(open(os.path.join(CDIR, "opt_tune_v1.json")))
    out["opt"] = {k: [[round(min(v, 1e3), 4) for v in s["val"]] for s in runs] for k, runs in ot.items()}

    # 4. batch size (analysis cache batch_v1): Adam, lr 0.001, 40 epochs
    br = json.load(open(os.path.join(CDIR, "batch_v1.json")))
    out["bs"] = {b: {"val": [round(v, 4) for v in br[b]["val"]], "sec": round(br[b]["sec"], 1),
                     "spe": int(np.ceil(len(ytr) / int(b)))} for b in br}

    # 5. vanishing gradients
    out["vg"] = {"depths": DEPTHS, "v": r["vg"]}
    return out


if __name__ == "__main__":
    d = load("--fresh" in sys.argv)
    print({k: round(v[-1], 3) for k, v in d["lr"]["curve"].items()})
    print({k: [round(x, 3) for x in v] for k, v in d["lr"]["final"].items()})
    print(d["vg"]["v"]["sig|xavier|10"][:3], d["vg"]["v"]["relu|he|10"][:3])
