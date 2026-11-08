"""Inputs of the Lecture 4 interactive page (interactive/L4_data.js): EuroSAT tiles (Sentinel-2), Fashion-MNIST test images, the
feature maps of the trained CNN (lectures/L4/cache/fm_cnn_seed0.pt) and the saved results of analysis_L4.py (cache/*.pkl).
Nothing is retrained. Result cached in lectures/L4/cache/interactive_v1.json.
Run alone to refresh the cache:  python3 lectures/L4/interactive_data_L4.py --fresh
"""
import json
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
CDIR = os.path.join(HERE, "cache")
CACHE = os.path.join(CDIR, "interactive_v1.json")
BUDGET = [10, 25, 50, 100, 250, 1000]


def compute():
    import numpy as np
    import nn_plot as npl
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    SEED = npl.seed_all()
    out = {}

    # EuroSAT: one tile per class (the 6th tile of each class, as the highway tile of the conv_edges chart), grey levels 0-255
    z = np.load(os.path.join(CDIR, "eurosat_u8.npz"))
    Xes, yes = z["X"], z["y"]
    out["tiles"] = [np.round(Xes[np.where(yes == c)[0][5]].astype(np.float32).mean(2)).astype(int).tolist() for c in range(10)]

    # Fashion-MNIST: the first test image of each class; feature maps of the trained CNN (seed 20261108)
    import torchvision
    fm_tr = torchvision.datasets.FashionMNIST(npl.vision_root(), train=True, download=False)
    fm_te = torchvision.datasets.FashionMNIST(npl.vision_root(), train=False, download=False)
    Xfm = fm_tr.data.numpy().astype(np.float32) / 255.0
    Xte = fm_te.data.numpy().astype(np.float32) / 255.0
    yte = fm_te.targets.numpy()
    perm = np.random.RandomState(SEED).permutation(len(Xfm))
    i_tr = perm[:50000]
    MU, SD = float(Xfm[i_tr].mean()), float(Xfm[i_tr].std())

    class FMCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
            self.b2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
            self.b3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 10))

        def forward(self, x):
            return self.head(self.b3(self.b2(self.b1(x))))

    cnn = FMCNN()
    cnn.load_state_dict(torch.load(os.path.join(CDIR, "fm_cnn_seed0.pt"), map_location="cpu"))
    cnn.eval()
    ids = [int(np.where(yte == c)[0][0]) for c in range(10)]
    imgs, m1, m3, probs = [], [], [], []
    with torch.no_grad():
        for i in ids:
            x = torch.tensor((Xte[i] - MU) / SD)[None, None]
            a1 = cnn.b1(x)
            a3 = cnn.b3(cnn.b2(a1))
            p = torch.softmax(cnn.head(a3), 1)[0].numpy()
            imgs.append(np.round(Xte[i] * 255).astype(int).tolist())
            for a, dst in ((a1[0].numpy(), m1), (a3[0].numpy(), m3)):
                order = np.argsort(-a.mean((1, 2)))                       # channels from the most to the least active
                mx = float(a.max()) or 1.0
                dst.append({"order": order.tolist(), "max": round(mx, 3),
                            "m": np.round(99 * a / mx).astype(int).tolist()})
            probs.append(np.round(p, 4).tolist())
    out["fm"] = {"ids": ids, "img": imgs, "b1": m1, "b3": m3, "p": probs}

    # confusion matrices on the 10,000 test images (seed 20261108 runs of analysis_L4.py)
    FM = pickle.load(open(os.path.join(CDIR, "fmnist.pkl"), "rb"))
    cm = {}
    for k in ("softmax", "mlp", "cnn"):
        pred = FM[k][0]["pred"]
        M = np.zeros((10, 10), int)
        np.add.at(M, (yte, pred), 1)
        cm[k] = M.tolist()
    out["cm"] = {"m": cm, "params": {k: FM[k][0]["params"] for k in ("softmax", "mlp", "cnn")},
                 "acc": {k: [round(float(r["test_acc"]), 4) for r in FM[k]] for k in ("softmax", "mlp", "cnn")}}

    # transfer learning on EuroSAT: label budget (2 seeds each) and validation curves of the full-data runs
    bu = {k: {} for k in ("scratch", "finetune", "probe")}
    for n in BUDGET:
        p = pickle.load(open(os.path.join(CDIR, f"eurosat_budget_{n}.pkl"), "rb"))
        for k in bu:
            bu[k][str(n)] = [round(float(v), 4) for v in p[k][n]]
    ES = pickle.load(open(os.path.join(CDIR, "eurosat_main.pkl"), "rb"))
    out["tl"] = {"budget": BUDGET, "acc": bu,
                 "hist": {k: [[round(float(a), 3), round(float(b), 4)] for a, b in ES[k]["hist"]] for k in ("scratch", "finetune")},
                 "probe_va": round(float(ES["probe"]["valid_acc"]), 4),
                 "full": {k: round(float(ES[k]["test_acc"]), 4) for k in ("scratch", "finetune", "probe")},
                 "params": {"scratch": int(ES["scratch"]["params"]), "finetune": int(ES["finetune"]["params"])}}

    # CIFAR-10 training curves (augmentation, plain vs residual 32-layer networks)
    cf = {}
    for k in ("cnn", "cnn_aug", "plain32", "res32"):
        r = pickle.load(open(os.path.join(CDIR, f"cifar_{k}.pkl"), "rb"))
        cf[k] = {h: [round(float(v), 4) for v in r["hist"][h]] for h in ("train_loss", "train_acc", "valid_acc")}
        cf[k]["test"] = round(float(r["test_acc"]), 4)
        cf[k]["params"] = int(r["params"])
    out["cf"] = cf
    return out


def load(fresh=False):
    if os.path.exists(CACHE) and not fresh:
        return json.load(open(CACHE))
    r = compute()
    json.dump(r, open(CACHE, "w"), separators=(",", ":"))
    return r


if __name__ == "__main__":
    d = load("--fresh" in sys.argv)
    import numpy as np
    for k, M in d["cm"]["m"].items():
        M = np.array(M)
        print(k, "acc", M.trace() / M.sum(), "shirt recall", M[6, 6] / M[6].sum(), "shirt->tshirt", M[6, 0])
    print("fm pred", [int(np.argmax(p)) for p in d["fm"]["p"]], [round(max(p), 3) for p in d["fm"]["p"]])
    print({k: v for k, v in d["tl"]["acc"].items()})
    print({k: (v["test"], v["train_acc"][-1], v["valid_acc"][-1], v["train_loss"][-1]) for k, v in d["cf"].items()})
