"""
analysis_L4.py -- Lecture 4 "Convolutional neural networks (CNN) and transfer learning": all computation and charts.

Data (real, on demand, see data/SOURCES.md): Fashion-MNIST (Zalando product images), CIFAR-10 (photographs),
EuroSAT (Sentinel-2 satellite images of land use). Pre-trained weights: ResNet-18 trained on ImageNet (torchvision).
Writes lectures/L4/L4_numbers.json and charts/en|ro/*.pdf. Trained results are cached in lectures/L4/cache/*.pkl
(git-ignored); delete a file there to retrain that experiment.
Hardware of the reference run: Apple silicon GPU (MPS backend of PyTorch); full first run about 20 minutes.
Run:  python3 lectures/L4/analysis_L4.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

npl.style()
SEED = npl.seed_all()
DEV = npl.device()
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
N = {"charts": {}, "seed": SEED, "device": DEV}
LANGS = ("en", "ro")

FM_EN = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FM_RO = ["Tricou", "Pantaloni", "Pulover", "Rochie", "Palton", "Sandală", "Cămașă", "Pantof sport", "Geantă", "Ghete"]
ES_EN = ["Annual crop", "Forest", "Herbaceous veg.", "Highway", "Industrial", "Pasture", "Permanent crop",
         "Residential", "River", "Sea/lake"]
ES_RO = ["Culturi anuale", "Pădure", "Vegetație ierboasă", "Autostradă", "Zonă industrială", "Pășune",
         "Culturi permanente", "Zonă rezidențială", "Rîu", "Mare/lac"]
CI_EN = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CI_RO = ["avion", "automobil", "pasăre", "pisică", "cerb", "cîine", "broască", "cal", "navă", "camion"]


def cached(name, fn):
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.isfile(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    r = fn()
    r["_seconds"] = round(time.time() - t0, 1)
    with open(p, "wb") as f:
        pickle.dump(r, f)
    print(f"  {name}: trained in {r['_seconds']} s")
    return r


def L(en, ro, lang):
    return en if lang == "en" else ro


def nparams(m):
    return int(sum(p.numel() for p in m.parameters()))


# =====================================================================================================================
# 1. Fashion-MNIST: softmax regression, MLP and CNN (3 seeds each)
# =====================================================================================================================
import torchvision  # noqa: E402

fm_tr = torchvision.datasets.FashionMNIST(npl.vision_root(), train=True, download=True)
fm_te = torchvision.datasets.FashionMNIST(npl.vision_root(), train=False, download=True)
Xfm = fm_tr.data.numpy().astype(np.float32) / 255.0
yfm = fm_tr.targets.numpy()
Xfm_te = fm_te.data.numpy().astype(np.float32) / 255.0
yfm_te = fm_te.targets.numpy()
perm = np.random.RandomState(SEED).permutation(len(Xfm))
i_tr, i_va = perm[:50000], perm[50000:]
FM_MU, FM_SD = float(Xfm[i_tr].mean()), float(Xfm[i_tr].std())     # scaling fitted on the training part only
N["fmnist"] = {"n_train": int(len(i_tr)), "n_valid": int(len(i_va)), "n_test": int(len(Xfm_te)), "size": 28,
               "mu": round(FM_MU, 4), "sd": round(FM_SD, 4)}


class FMCNN(nn.Module):
    """Two convolutional blocks (3x3 kernels, batch norm, ReLU, 2x2 max pooling), a third convolution and global average pooling."""
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


def fm_model(kind):
    if kind == "softmax":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    if kind == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
                             nn.Dropout(0.2), nn.Linear(128, 10))
    return FMCNN()


def evaluate(model, X, y, bs=1000):
    model.eval()
    out = []
    with torch.no_grad():
        for j in range(0, len(X), bs):
            out.append(model(X[j:j + bs]).float().cpu())
    logits = torch.cat(out)
    loss = float(F.cross_entropy(logits, torch.as_tensor(y)))
    pred = logits.argmax(1).numpy()
    return loss, float((pred == y).mean()), pred


FM_EPOCHS, FM_BS, FM_LR = 12, 128, 1e-3


def train_fm(kind, seed):
    npl.seed_all(SEED + seed)
    Xt = torch.tensor((Xfm[i_tr] - FM_MU) / FM_SD).unsqueeze(1).to(DEV)
    Xv = torch.tensor((Xfm[i_va] - FM_MU) / FM_SD).unsqueeze(1).to(DEV)
    Xs = torch.tensor((Xfm_te - FM_MU) / FM_SD).unsqueeze(1).to(DEV)
    yt = torch.tensor(yfm[i_tr]).to(DEV)
    m = fm_model(kind).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=FM_LR, weight_decay=1e-4)
    steps = FM_EPOCHS * int(np.ceil(len(i_tr) / FM_BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3 * FM_LR, total_steps=steps)
    hist = {"train_loss": [], "valid_loss": [], "valid_acc": []}
    best, best_state = -1, None
    g = torch.Generator().manual_seed(SEED + seed)
    for ep in range(FM_EPOCHS):
        m.train()
        idx = torch.randperm(len(i_tr), generator=g)
        tot = 0.0
        for j in range(0, len(idx), FM_BS):
            b = idx[j:j + FM_BS].to(DEV)
            loss = F.cross_entropy(m(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tot += float(loss) * len(b)
        vl, va, _ = evaluate(m, Xv, yfm[i_va])
        hist["train_loss"].append(tot / len(i_tr))
        hist["valid_loss"].append(vl)
        hist["valid_acc"].append(va)
        if va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)                          # early stopping: epoch with the best validation accuracy
    tl, ta, pred = evaluate(m, Xs, yfm_te)
    res = {"hist": hist, "test_acc": ta, "test_loss": tl, "best_epoch": int(np.argmax(hist["valid_acc"]) + 1),
           "params": nparams(m), "pred": pred}
    if kind == "cnn" and seed == 0:
        torch.save(best_state, os.path.join(CACHE, "fm_cnn_seed0.pt"))
    return res


FM_KINDS = ["softmax", "mlp", "cnn"]
FM = cached("fmnist", lambda: {k: [train_fm(k, s) for s in range(3)] for k in FM_KINDS})
for k in FM_KINDS:
    acc = np.array([r["test_acc"] for r in FM[k]]) * 100
    N["fmnist"][k] = {"acc_mean": round(float(acc.mean()), 2), "acc_sd": round(float(acc.std(ddof=1)), 2),
                      "acc_min": round(float(acc.min()), 2), "acc_max": round(float(acc.max()), 2),
                      "params": FM[k][0]["params"], "best_epoch": FM[k][0]["best_epoch"],
                      "valid_acc_last": round(100 * FM[k][0]["hist"]["valid_acc"][-1], 2)}
N["fmnist"]["epochs"] = FM_EPOCHS
N["fmnist"]["batch"] = FM_BS
N["fmnist"]["seconds"] = FM["_seconds"]
N["fmnist"]["cnn_vs_mlp_pp"] = round(N["fmnist"]["cnn"]["acc_mean"] - N["fmnist"]["mlp"]["acc_mean"], 2)
N["fmnist"]["cnn_err_cut_pct"] = round(100 * (1 - (100 - N["fmnist"]["cnn"]["acc_mean"]) / (100 - N["fmnist"]["mlp"]["acc_mean"])), 1)
N["fmnist"]["params_ratio"] = round(N["fmnist"]["mlp"]["params"] / N["fmnist"]["cnn"]["params"], 2)
# paired comparison over the three seeds (same seeds for both models)
d = np.array([a["test_acc"] - b["test_acc"] for a, b in zip(FM["cnn"], FM["mlp"])]) * 100
N["fmnist"]["diff_t"] = round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), 2)
# McNemar test CNN vs MLP, seed 0, on the 10,000 test images
pc, pm = FM["cnn"][0]["pred"] == yfm_te, FM["mlp"][0]["pred"] == yfm_te
b01, b10 = int((pc & ~pm).sum()), int((~pc & pm).sum())
N["fmnist"]["mcnemar"] = {"cnn_only": b01, "mlp_only": b10, "chi2": round((abs(b01 - b10) - 1) ** 2 / (b01 + b10), 1)}

# confusion matrix of the CNN (seed 0)
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

cm = confusion_matrix(yfm_te, FM["cnn"][0]["pred"])
rec = cm.diagonal() / cm.sum(1)
N["fmnist"]["recall"] = {FM_EN[i]: round(100 * float(rec[i]), 1) for i in range(10)}
N["fmnist"]["worst_class"] = FM_EN[int(rec.argmin())]
N["fmnist"]["worst_class_ro"] = FM_RO[int(rec.argmin())]
N["fmnist"]["worst_recall"] = round(100 * float(rec.min()), 1)
off = cm.copy()
np.fill_diagonal(off, 0)
a_, b_ = np.unravel_index(off.argmax(), off.shape)
N["fmnist"]["top_conf"] = {"true": FM_EN[a_], "pred": FM_EN[b_], "true_ro": FM_RO[a_], "pred_ro": FM_RO[b_], "n": int(off[a_, b_])}
N["fmnist"]["shirt_n"] = int(cm[6].sum())

# ---- chart: sample images
for lang in LANGS:
    names = FM_EN if lang == "en" else FM_RO
    fig, axes = plt.subplots(2, 10, figsize=(6.4, 1.75))
    for c in range(10):
        ids = np.where(yfm[i_tr] == c)[0][:2]
        for r in range(2):
            ax = axes[r, c]
            ax.imshow(Xfm[i_tr][ids[r]], cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[1, c].set_xlabel(names[c], fontsize=7.5)
    npl.out(fig, HERE, lang, "fmnist_samples", bottom=0.0)
npl.chart_meta(N, "fmnist_samples", "fashion_mnist", "60,000 training images, 28 x 28 grey levels",
               "60.000 de imagini de antrenare, 28 x 28 niveluri de gri")

# ---- chart: learning curves (seed 0)
for lang in LANGS:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    lab = {"softmax": L("Softmax regression", "Regresie softmax", lang), "mlp": "MLP", "cnn": "CNN"}
    col = {"softmax": npl.PAL["benchmark"], "mlp": npl.PAL["alt"], "cnn": npl.PAL["model"]}
    ep = np.arange(1, FM_EPOCHS + 1)
    for k in FM_KINDS:
        h = FM[k][0]["hist"]
        axes[0].plot(ep, h["train_loss"], color=col[k], ls="--", lw=1.2)
        axes[0].plot(ep, h["valid_loss"], color=col[k], lw=1.6, label=lab[k])
        axes[1].plot(ep, 100 * np.array(h["valid_acc"]), color=col[k], lw=1.6, marker="o", ms=2.5)
    axes[0].set_ylabel(L("Cross-entropy", "Entropia încrucișată", lang))
    axes[1].set_ylabel(L("Validation accuracy (%)", "Acuratețea de validare (%)", lang))
    for ax in axes:
        ax.set_xlabel(L("Epoch", "Epoca", lang))
    axes[0].plot([], [], color="k", ls="--", lw=1.2, label=L("training (dashed)", "antrenare (punctat)", lang))
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "fmnist_curves")
npl.chart_meta(N, "fmnist_curves", "fashion_mnist", "50,000 training and 10,000 validation images; seed 20261108",
               "50.000 de imagini de antrenare și 10.000 de validare; sămînța 20261108")

# ---- chart: accuracy and parameters
for lang in LANGS:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    lab = [L("Softmax\nregression", "Regresie\nsoftmax", lang), "MLP", "CNN"]
    cols_ = [npl.PAL["benchmark"], npl.PAL["alt"], npl.PAL["model"]]
    acc = [N["fmnist"][k]["acc_mean"] for k in FM_KINDS]
    sd = [N["fmnist"][k]["acc_sd"] for k in FM_KINDS]
    axes[0].bar(lab, acc, yerr=sd, color=cols_, capsize=3)
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.4, (f"{v:.2f}" if lang == "en" else f"{v:.2f}".replace(".", ",")), ha="center", fontsize=9)
    axes[0].set_ylim(80, 95)
    axes[0].set_ylabel(L("Test accuracy (%)", "Acuratețea pe test (%)", lang))
    pr = [N["fmnist"][k]["params"] / 1000 for k in FM_KINDS]
    axes[1].bar(lab, pr, color=cols_)
    for i, v in enumerate(pr):
        axes[1].text(i, v + 4, f"{v:.1f}" if lang == "en" else f"{v:.1f}".replace(".", ","), ha="center", fontsize=9)
    axes[1].set_ylabel(L("Parameters (thousands)", "Parametri (mii)", lang))
    npl.out(fig, HERE, lang, "fmnist_compare", bottom=0.02)
npl.chart_meta(N, "fmnist_compare", "fashion_mnist", "10,000 test images; mean and standard deviation over 3 seeds",
               "10.000 de imagini de test; media și abaterea standard pe 3 semințe")

# ---- chart: confusion matrix
for lang in LANGS:
    names = FM_EN if lang == "en" else FM_RO
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    cmn = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 10:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6.5, color="white" if cmn[i, j] > 50 else "#222222")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel(L("Predicted class", "Clasa prezisă", lang), fontsize=9)
    ax.set_ylabel(L("True class", "Clasa reală", lang), fontsize=9)
    ax._keep_y = True
    npl.out(fig, HERE, lang, "fmnist_confusion", bottom=0.0)
npl.chart_meta(N, "fmnist_confusion", "fashion_mnist", "10,000 test images, CNN seed 20261108",
               "10.000 de imagini de test, CNN cu sămînța 20261108")

# ---- learned filters and feature maps (CNN seed 0)
cnn = FMCNN().to(DEV)
cnn.load_state_dict(torch.load(os.path.join(CACHE, "fm_cnn_seed0.pt"), map_location=DEV))
cnn.eval()
W1 = cnn.b1[0].weight.detach().cpu().numpy()[:, 0]          # 32 x 3 x 3
for lang in LANGS:
    fig, axes = plt.subplots(2, 16, figsize=(6.4, 1.0))
    for i, ax in enumerate(axes.flat):
        w = W1[i]
        ax.imshow(w, cmap="RdBu_r", vmin=-np.abs(W1).max(), vmax=np.abs(W1).max())
        ax.set_xticks([]); ax.set_yticks([])
    npl.out(fig, HERE, lang, "fmnist_filters", bottom=0.0)
npl.chart_meta(N, "fmnist_filters", "fashion_mnist", "first convolutional layer of the CNN (32 filters of 3 x 3), seed 20261108",
               "primul strat convoluțional al CNN (32 de filtre de 3 x 3), sămînța 20261108")

img_id = int(np.where(yfm_te == 9)[0][0])                   # first ankle boot of the test set
x0 = torch.tensor((Xfm_te[img_id] - FM_MU) / FM_SD)[None, None].to(DEV)
with torch.no_grad():
    a1 = cnn.b1(x0)
    a2 = cnn.b2(a1)
    a3 = cnn.b3(a2)
maps = [a1[0].cpu().numpy(), a3[0].cpu().numpy()]
N["fmnist"]["fmap_sizes"] = {"l1": list(maps[0].shape), "l3": list(maps[1].shape)}
N["fmnist"]["fmap_l3_size"] = int(maps[1].shape[1])
for lang in LANGS:
    fig, axes = plt.subplots(2, 9, figsize=(6.4, 1.6))
    for r in range(2):
        axes[r, 0].imshow(Xfm_te[img_id], cmap="gray_r")
        order = np.argsort(-maps[r].mean((1, 2)))[:8]
        for c in range(8):
            axes[r, c + 1].imshow(maps[r][order[c]], cmap="viridis")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel(L("block 1\n14×14", "blocul 1\n14×14", lang), fontsize=7.5)
    axes[1, 0].set_ylabel(L("block 3\n7×7", "blocul 3\n7×7", lang), fontsize=7.5)
    npl.out(fig, HERE, lang, "fmnist_featuremaps", bottom=0.0)
npl.chart_meta(N, "fmnist_featuremaps", "fashion_mnist", "one test image (ankle boot); 8 most active channels per block",
               "o imagine de test (ghete); cele mai active 8 canale din fiecare bloc")

# =====================================================================================================================
# 2. Mechanism on a real image: hand-set filters on a EuroSAT image; receptive field arithmetic
# =====================================================================================================================
ES_ROOT = os.path.join(npl.vision_root(), "eurosat", "2750")
ES_DIRS = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop",
           "Residential", "River", "SeaLake"]


def load_eurosat():
    p = os.path.join(CACHE, "eurosat_u8.npz")
    if os.path.isfile(p):
        z = np.load(p)
        return z["X"], z["y"]
    from PIL import Image
    X, y = [], []
    for c, dname in enumerate(ES_DIRS):
        for fn in sorted(os.listdir(os.path.join(ES_ROOT, dname))):
            if fn.lower().endswith(".jpg"):
                X.append(np.asarray(Image.open(os.path.join(ES_ROOT, dname, fn)).convert("RGB")))
                y.append(c)
    X, y = np.stack(X), np.array(y)
    np.savez(p, X=X, y=y)
    return X, y


Xes, yes = load_eurosat()
N["eurosat"] = {"n": int(len(Xes)), "size": int(Xes.shape[1]), "classes": 10,
                "per_class_min": int(np.bincount(yes).min()), "per_class_max": int(np.bincount(yes).max())}

img = Xes[np.where(yes == 3)[0][5]].astype(np.float32).mean(2) / 255.0      # a highway tile, grey levels
K = {"sx": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
     "sy": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
     "blur": np.ones((3, 3), np.float32) / 9}
convd = {k: F.conv2d(torch.tensor(img)[None, None], torch.tensor(v)[None, None], padding=1)[0, 0].numpy() for k, v in K.items()}
N["conv"] = {"in": 64, "out_same": int(convd["sx"].shape[0]),
             "out_valid": 64 - 3 + 1, "out_s2": (64 + 2 * 1 - 3) // 2 + 1}
for lang in LANGS:
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 1.9))
    axes[0].imshow(img, cmap="gray")
    axes[1].imshow(convd["sx"], cmap="RdBu_r")
    axes[2].imshow(convd["sy"], cmap="RdBu_r")
    axes[3].imshow(convd["blur"], cmap="gray")
    labs = [L("Input (highway)", "Intrare (autostradă)", lang), L("Vertical edges", "Muchii verticale", lang),
            L("Horizontal edges", "Muchii orizontale", lang), L("Mean 3×3 (blur)", "Media 3×3 (estompare)", lang)]
    for ax, l_ in zip(axes, labs):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(l_, fontsize=8.5)
    npl.out(fig, HERE, lang, "conv_edges", bottom=0.0)
npl.chart_meta(N, "conv_edges", "eurosat", "one 64 x 64 Sentinel-2 tile (class Highway), grey levels",
               "o imagine Sentinel-2 de 64 x 64 (clasa Autostradă), niveluri de gri")

# receptive field: r_l = r_{l-1} + (k_l - 1) * prod_{i<l} s_i
def rf(layers):
    r, j, out = 1, 1, []
    for k, s in layers:
        r += (k - 1) * j
        j *= s
        out.append(r)
    return out


plain = rf([(3, 1)] * 12)
pooled = rf([(3, 1), (3, 1), (2, 2)] * 4)
pooled = [pooled[i] for i in range(len(pooled)) if (i % 3) != 2][:8]
N["rf"] = {"plain_6": plain[5], "plain_12": plain[11], "pooled_8conv": pooled[7], "plain_8": plain[7]}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.plot(range(1, 13), plain, marker="o", ms=3, color=npl.PAL["benchmark"],
            label=L("3×3 convolutions, stride 1", "convoluții 3×3, pas 1", lang))
    ax.plot(range(1, 9), pooled, marker="s", ms=3, color=npl.PAL["model"],
            label=L("pairs of 3×3 convolutions + 2×2 pooling", "perechi de convoluții 3×3 + agregare 2×2", lang))
    ax.set_xlabel(L("Number of convolutional layers", "Numărul de straturi convoluționale", lang))
    ax.set_ylabel(L("Receptive field (pixels)", "Cîmpul receptiv (pixeli)", lang))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "receptive_field")
npl.chart_meta(N, "receptive_field", "formula", "recursion of the receptive field, no data",
               "recurența cîmpului receptiv, fără date",
               source_en="own computation from the receptive-field recursion", source_ro="calcul propriu din recurența cîmpului receptiv")

# =====================================================================================================================
# 3. CIFAR-10: a small CNN with and without augmentation; plain vs residual 32-layer networks (degradation)
# =====================================================================================================================
cf_tr = torchvision.datasets.CIFAR10(npl.vision_root(), train=True, download=True)
cf_te = torchvision.datasets.CIFAR10(npl.vision_root(), train=False, download=True)
Xc = cf_tr.data
yc = np.array(cf_tr.targets)
Xc_te = cf_te.data
yc_te = np.array(cf_te.targets)
pc_ = np.random.RandomState(SEED).permutation(len(Xc))
c_tr, c_va = pc_[:45000], pc_[45000:]
C_MU = Xc[c_tr].reshape(-1, 3).mean(0) / 255.0
C_SD = Xc[c_tr].reshape(-1, 3).std(0) / 255.0
N["cifar"] = {"n_train": len(c_tr), "n_valid": len(c_va), "n_test": len(Xc_te), "size": 32}


def to_t(X):
    x = torch.tensor(X).permute(0, 3, 1, 2).float() / 255.0
    return (x - torch.tensor(C_MU).view(1, 3, 1, 1).float()) / torch.tensor(C_SD).view(1, 3, 1, 1).float()


def augment(x, g):
    """Random crop with 4-pixel zero padding and random horizontal flip, on the device."""
    n = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    ox = torch.randint(0, 9, (n,), generator=g)
    oy = torch.randint(0, 9, (n,), generator=g)
    idx = torch.arange(32)
    rows = (oy[:, None] + idx[None, :]).to(x.device)
    cols = (ox[:, None] + idx[None, :]).to(x.device)
    xp = xp[torch.arange(n, device=x.device)[:, None, None], :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    flip = (torch.rand(n, generator=g) < 0.5).to(x.device)
    xp[flip] = xp[flip].flip(3)
    return xp


class Block(nn.Module):
    def __init__(self, cin, cout, stride, residual=True):
        super().__init__()
        self.res = residual
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        if self.res:
            o = o + (x if self.sc is None else self.sc(x))
        return F.relu(o)


def resnet_cifar(n=5, residual=True):
    """He et al. (2016) CIFAR network with 6n+2 layers: 16, 32, 64 channels; residual=False gives the plain network."""
    layers = [nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU()]
    cin = 16
    for cout, st in [(16, 1), (32, 2), (64, 2)]:
        for i in range(n):
            layers.append(Block(cin, cout, st if i == 0 else 1, residual))
            cin = cout
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)]
    return nn.Sequential(*layers)


def small_cnn():
    def blk(a, b):
        return [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1),
                nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*blk(3, 32), *blk(32, 64), *blk(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(128, 10))


C_EPOCHS, C_BS = 12, 128


def train_cifar(kind, aug):
    npl.seed_all(SEED)
    m = {"cnn": small_cnn, "plain32": lambda: resnet_cifar(5, False), "res32": lambda: resnet_cifar(5, True)}[kind]().to(DEV)
    Xt, Xv, Xs = to_t(Xc[c_tr]).to(DEV), to_t(Xc[c_va]).to(DEV), to_t(Xc_te).to(DEV)
    yt = torch.tensor(yc[c_tr]).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = C_EPOCHS * int(np.ceil(len(c_tr) / C_BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps)
    g = torch.Generator().manual_seed(SEED)
    hist = {"train_loss": [], "valid_acc": [], "train_acc": []}
    for ep in range(C_EPOCHS):
        m.train()
        idx = torch.randperm(len(c_tr), generator=g)
        tot, cor = 0.0, 0
        for j in range(0, len(idx), C_BS):
            b = idx[j:j + C_BS].to(DEV)
            xb = augment(Xt[b], g) if aug else Xt[b]
            out = m(xb)
            loss = F.cross_entropy(out, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tot += float(loss) * len(b)
            cor += int((out.argmax(1) == yt[b]).sum())
        _, va, _ = evaluate(m, Xv, yc[c_va], bs=500)
        hist["train_loss"].append(tot / len(c_tr))
        hist["train_acc"].append(cor / len(c_tr))
        hist["valid_acc"].append(va)
        print(f"    cifar {kind} aug={aug} epoch {ep + 1}: loss {tot / len(c_tr):.3f} valid {va:.3f}")
    _, ta, pred = evaluate(m, Xs, yc_te, bs=500)          # fixed schedule, test set used once at the end
    return {"hist": hist, "test_acc": ta, "params": nparams(m), "pred": pred}


CF = {"cnn": cached("cifar_cnn", lambda: train_cifar("cnn", False)),
      "cnn_aug": cached("cifar_cnn_aug", lambda: train_cifar("cnn", True)),
      "plain32": cached("cifar_plain32", lambda: train_cifar("plain32", True)),
      "res32": cached("cifar_res32", lambda: train_cifar("res32", True))}
CF["_seconds"] = sum(CF[k]["_seconds"] for k in ["cnn", "cnn_aug", "plain32", "res32"])
for k in ["cnn", "cnn_aug", "plain32", "res32"]:
    h = CF[k]["hist"]
    N["cifar"][k] = {"test_acc": round(100 * CF[k]["test_acc"], 2), "params": CF[k]["params"],
                     "train_acc": round(100 * h["train_acc"][-1], 2), "valid_acc": round(100 * h["valid_acc"][-1], 2),
                     "train_loss": round(h["train_loss"][-1], 3),
                     "gap": round(100 * (h["train_acc"][-1] - h["valid_acc"][-1]), 2)}
N["cifar"]["epochs"] = C_EPOCHS
N["cifar"]["seconds"] = CF["_seconds"]
N["cifar"]["aug_gain"] = round(N["cifar"]["cnn_aug"]["test_acc"] - N["cifar"]["cnn"]["test_acc"], 2)
N["cifar"]["res_gain"] = round(N["cifar"]["res32"]["test_acc"] - N["cifar"]["plain32"]["test_acc"], 2)
cmc = confusion_matrix(yc_te, CF["res32"]["pred"])
rc = cmc.diagonal() / cmc.sum(1)
N["cifar"]["res32_worst"] = CI_EN[int(rc.argmin())]
N["cifar"]["res32_worst_ro"] = CI_RO[int(rc.argmin())]
N["cifar"]["res32_worst_recall"] = round(100 * float(rc.min()), 1)
N["cifar"]["res32_best"] = CI_EN[int(rc.argmax())]
N["cifar"]["res32_best_ro"] = CI_RO[int(rc.argmax())]
N["cifar"]["res32_best_recall"] = round(100 * float(rc.max()), 1)

for lang in LANGS:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ep = np.arange(1, C_EPOCHS + 1)
    for k, c, l_ in [("cnn", npl.PAL["benchmark"], L("CNN, no augmentation", "CNN, fără augmentare", lang)),
                     ("cnn_aug", npl.PAL["model"], L("CNN, augmentation", "CNN, cu augmentare", lang))]:
        h = CF[k]["hist"]
        axes[0].plot(ep, 100 * np.array(h["train_acc"]), color=c, ls="--", lw=1.2)
        axes[0].plot(ep, 100 * np.array(h["valid_acc"]), color=c, lw=1.6, label=l_)
    axes[0].set_ylabel(L("Accuracy (%)", "Acuratețea (%)", lang))
    for k, c, l_ in [("plain32", npl.PAL["alt"], L("Plain, 32 layers", "Simplă, 32 de straturi", lang)),
                     ("res32", npl.PAL["c7"], L("ResNet, 32 layers", "ResNet, 32 de straturi", lang))]:
        axes[1].plot(ep, CF[k]["hist"]["train_loss"], color=c, lw=1.6, label=l_)
    axes[1].set_ylabel(L("Training loss", "Pierderea la antrenare", lang))
    for ax in axes:
        ax.set_xlabel(L("Epoch", "Epoca", lang))
    axes[0].plot([], [], color="k", ls="--", lw=1.2, label=L("training (dashed)", "antrenare (punctat)", lang))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "cifar_curves", bottom=0.22)
npl.chart_meta(N, "cifar_curves", "cifar10", "45,000 training and 5,000 validation images; seed 20261108",
               "45.000 de imagini de antrenare și 5.000 de validare; sămînța 20261108")

# =====================================================================================================================
# 4. EuroSAT: CNN from scratch, ResNet-18 as a frozen feature extractor, fine-tuned ResNet-18; label budget
# =====================================================================================================================
rs = np.random.RandomState(SEED)
e_tr, e_va, e_te = [], [], []
for c in range(10):
    ids = rs.permutation(np.where(yes == c)[0])
    n = len(ids)
    e_tr += list(ids[:int(0.6 * n)]); e_va += list(ids[int(0.6 * n):int(0.8 * n)]); e_te += list(ids[int(0.8 * n):])
e_tr, e_va, e_te = np.array(e_tr), np.array(e_va), np.array(e_te)
N["eurosat"].update({"n_train": len(e_tr), "n_valid": len(e_va), "n_test": len(e_te)})
IMN_MU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_SD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RES = 112


def es_tensor(ids, size=None):
    x = torch.tensor(Xes[ids]).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMN_MU) / IMN_SD
    if size and size != 64:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


def es_aug(x, g):
    """Satellite tiles have no 'up': random flips and 90-degree rotations."""
    n = x.shape[0]
    f = (torch.rand(n, generator=g) < 0.5).to(x.device)
    x = torch.where(f[:, None, None, None], x.flip(3), x)
    k = int(torch.randint(0, 4, (1,), generator=g))
    return torch.rot90(x, k, (2, 3))


def scratch_es():
    def blk(a, b):
        return [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1),
                nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*blk(3, 32), *blk(32, 64), *blk(64, 128), *blk(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(256, 10))


def resnet18(pretrained=True):
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Linear(512, 10)
    return m


def fit_es(kind, tr_ids, steps, seed=0, curve=False, size=None):
    """kind: 'scratch' (64 px) or 'finetune' (ResNet-18, all layers, 112 px). Fixed number of steps; for the curve,
    validation accuracy at regular intervals (and the weights of the best validation point)."""
    npl.seed_all(SEED + seed)
    size = size or (64 if kind == "scratch" else RES)
    m = (scratch_es() if kind == "scratch" else resnet18(True)).to(DEV)
    if kind == "scratch":
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
        max_lr = 2e-3
    else:
        opt = torch.optim.AdamW([{"params": [p for n_, p in m.named_parameters() if not n_.startswith("fc")], "lr": 3e-4},
                                 {"params": m.fc.parameters(), "lr": 3e-3}], weight_decay=1e-4)
        max_lr = [3e-4, 3e-3]
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=steps)
    Xt = es_tensor(tr_ids, size).to(DEV)
    yt = torch.tensor(yes[tr_ids]).to(DEV)
    Xv = es_tensor(e_va, size).to(DEV) if curve else None
    g = torch.Generator().manual_seed(SEED + seed)
    bs = 64
    hist, best, best_state, step = [], -1, None, 0
    every = max(1, steps // 12)
    while step < steps:
        idx = torch.randperm(len(tr_ids), generator=g)
        for j in range(0, len(idx), bs):
            if step >= steps:
                break
            m.train()
            b = idx[j:j + bs].to(DEV)
            loss = F.cross_entropy(m(es_aug(Xt[b], g)), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            step += 1
            if curve and (step % every == 0 or step == steps):
                _, va, _ = evaluate(m, Xv, yes[e_va], bs=500)
                hist.append((step * bs / len(tr_ids), va))
                if va > best:
                    best, best_state = va, {k: v.detach().clone() for k, v in m.state_dict().items()}
    if curve:
        m.load_state_dict(best_state)
    Xs = es_tensor(e_te, size).to(DEV)
    _, ta, pred = evaluate(m, Xs, yes[e_te], bs=500)
    return m, {"hist": hist, "test_acc": ta, "pred": pred, "params": nparams(m)}


def features_es():
    m = resnet18(True).to(DEV)
    m.fc = nn.Identity()
    m.eval()
    out = []
    with torch.no_grad():
        for j in range(0, len(Xes), 500):
            out.append(m(es_tensor(np.arange(j, min(j + 500, len(Xes))), RES).to(DEV)).cpu().numpy())
    return {"feat": np.concatenate(out)}


FEAT = cached("eurosat_features", features_es)["feat"]
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def linear_probe(tr_ids, C=None):
    sc = StandardScaler().fit(FEAT[tr_ids])
    Ctry = [C] if C else [0.01, 0.1, 1.0]
    best = None
    for c_ in Ctry:                                         # C chosen on the validation set
        lr = LogisticRegression(C=c_, max_iter=2000).fit(sc.transform(FEAT[tr_ids]), yes[tr_ids])
        va = lr.score(sc.transform(FEAT[e_va]), yes[e_va])
        if best is None or va > best[0]:
            best = (va, c_, lr)
    pred = best[2].predict(sc.transform(FEAT[e_te]))
    return {"test_acc": float((pred == yes[e_te]).mean()), "valid_acc": best[0], "C": best[1], "pred": pred}


def run_es_main():
    steps_full = 6 * int(np.ceil(len(e_tr) / 64))
    _, s = fit_es("scratch", e_tr, 3 * steps_full, curve=True)
    mft, f = fit_es("finetune", e_tr, steps_full, curve=True)
    torch.save(mft.state_dict(), os.path.join(CACHE, "eurosat_finetune.pt"))
    p = linear_probe(e_tr)
    return {"scratch": s, "finetune": f, "probe": p, "steps_full": steps_full}


ES = cached("eurosat_main", run_es_main)
N["eurosat"]["scratch_acc"] = round(100 * ES["scratch"]["test_acc"], 2)
N["eurosat"]["finetune_acc"] = round(100 * ES["finetune"]["test_acc"], 2)
N["eurosat"]["probe_acc"] = round(100 * ES["probe"]["test_acc"], 2)
N["eurosat"]["probe_C"] = ES["probe"]["C"]
N["eurosat"]["scratch_params"] = ES["scratch"]["params"]
N["eurosat"]["resnet_params"] = ES["finetune"]["params"]
N["eurosat"]["scratch_epochs"] = 18
N["eurosat"]["ft_epochs"] = 6
N["eurosat"]["res"] = RES
N["eurosat"]["seconds"] = ES["_seconds"]
N["eurosat"]["ft_err_cut_pct"] = round(100 * (1 - (1 - ES["finetune"]["test_acc"]) / (1 - ES["scratch"]["test_acc"])), 1)
cme = confusion_matrix(yes[e_te], ES["finetune"]["pred"])
off = cme.copy(); np.fill_diagonal(off, 0)
a_, b_ = np.unravel_index(off.argmax(), off.shape)
N["eurosat"]["top_conf"] = {"true": ES_EN[a_], "pred": ES_EN[b_], "true_ro": ES_RO[a_], "pred_ro": ES_RO[b_], "n": int(off[a_, b_])}
N["eurosat"]["n_errors_ft"] = int(off.sum())
rce = cme.diagonal() / cme.sum(1)
N["eurosat"]["worst"] = ES_EN[int(rce.argmin())]
N["eurosat"]["worst_ro"] = ES_RO[int(rce.argmin())]
N["eurosat"]["worst_recall"] = round(100 * float(rce.min()), 1)
N["eurosat"]["macro_f1_ft"] = round(100 * f1_score(yes[e_te], ES["finetune"]["pred"], average="macro"), 2)

# label budget: n labelled tiles per class
BUDGET = [10, 25, 50, 100, 250, 1000]


def run_budget_n(n):
    out = {"scratch": {}, "finetune": {}, "probe": {}}
    if True:
        for s in range(2):
            rs_ = np.random.RandomState(SEED + 100 * s + n)
            ids = np.concatenate([rs_.choice(e_tr[yes[e_tr] == c], n, replace=False) for c in range(10)])
            steps = 300 if n <= 250 else 600
            out["probe"].setdefault(n, []).append(linear_probe(ids)["test_acc"])
            out["finetune"].setdefault(n, []).append(fit_es("finetune", ids, steps, seed=s)[1]["test_acc"])
            out["scratch"].setdefault(n, []).append(fit_es("scratch", ids, 2 * steps, seed=s)[1]["test_acc"])
            print(f"    budget {n}/class seed {s}: probe {out['probe'][n][-1]:.3f} ft {out['finetune'][n][-1]:.3f} "
                  f"scratch {out['scratch'][n][-1]:.3f}")
    return out


def run_budget():
    parts = [cached(f"eurosat_budget_{n}", lambda n=n: run_budget_n(n)) for n in BUDGET]
    out = {k: {n: p[k][n] for n, p in zip(BUDGET, parts)} for k in ["scratch", "finetune", "probe"]}
    out["_seconds"] = sum(p["_seconds"] for p in parts)
    return out


BU = run_budget()
N["budget"] = {k: {str(n): round(100 * float(np.mean(BU[k][n])), 1) for n in BUDGET} for k in ["scratch", "finetune", "probe"]}
N["budget"]["seconds"] = BU["_seconds"]
N["budget"]["n_small"] = BUDGET[0]
N["budget"]["tiles_small"] = BUDGET[0] * 10

for lang in LANGS:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for k, c, l_ in [("scratch", npl.PAL["benchmark"], L("CNN from scratch", "CNN de la zero", lang)),
                     ("finetune", npl.PAL["model"], L("ResNet-18 fine-tuned", "ResNet-18 ajustat fin", lang))]:
        h = np.array(ES[k]["hist"])
        axes[0].plot(h[:, 0], 100 * h[:, 1], color=c, lw=1.6, marker="o", ms=2.5, label=l_)
    axes[0].axhline(100 * ES["probe"]["valid_acc"], color=npl.PAL["alt"], lw=1.4, ls="--",
                    label=L("ResNet-18 frozen + logistic", "ResNet-18 înghețat + logistic", lang))
    axes[0].set_xlabel(L("Epochs (passes over the training set)", "Epoci (treceri prin setul de antrenare)", lang), fontsize=9)
    axes[0].set_ylabel(L("Validation accuracy (%)", "Acuratețea de validare (%)", lang))
    axes[0].set_ylim(50, 100)
    for k, c in [("scratch", npl.PAL["benchmark"]), ("finetune", npl.PAL["model"]), ("probe", npl.PAL["alt"])]:
        axes[1].plot(BUDGET, [N["budget"][k][str(n)] for n in BUDGET], color=c, lw=1.6, marker="o", ms=3)
    axes[1].set_xscale("log")
    axes[1].set_xticks(BUDGET)
    axes[1].set_xticklabels([str(b) for b in BUDGET])
    axes[1].set_xlabel(L("Labelled tiles per class", "Imagini etichetate pe clasă", lang), fontsize=9)
    axes[1].set_ylabel(L("Test accuracy (%)", "Acuratețea pe test (%)", lang))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "transfer_curves", bottom=0.14)
npl.chart_meta(N, "transfer_curves", "eurosat", "16,200 training, 5,400 validation, 5,400 test tiles; right: mean of 2 seeds",
               "16.200 de imagini de antrenare, 5.400 de validare, 5.400 de test; dreapta: media pe 2 semințe")

# EuroSAT samples
for lang in LANGS:
    names = ES_EN if lang == "en" else ES_RO
    fig, axes = plt.subplots(2, 10, figsize=(6.4, 1.8))
    for c in range(10):
        ids = e_tr[yes[e_tr] == c][:2]
        for r in range(2):
            axes[r, c].imshow(Xes[ids[r]])
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[1, c].set_xlabel(names[c].replace(" ", "\n", 1), fontsize=6.8)
    npl.out(fig, HERE, lang, "eurosat_samples", bottom=0.0)
npl.chart_meta(N, "eurosat_samples", "eurosat", "27,000 tiles of 64 x 64 pixels, 10 classes",
               "27.000 de imagini de 64 x 64 pixeli, 10 clase")

# EuroSAT confusion matrix of the fine-tuned ResNet-18
for lang in LANGS:
    names = ES_EN if lang == "en" else ES_RO
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    cmn = cme / cme.sum(1, keepdims=True) * 100
    ax.imshow(cmn, cmap="Oranges", vmin=0, vmax=100, aspect="auto")
    for i in range(10):
        for j in range(10):
            if cme[i, j] >= 1:
                ax.text(j, i, str(cme[i, j]), ha="center", va="center", fontsize=6.5, color="white" if cmn[i, j] > 50 else "#222222")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7.2)
    ax.set_yticklabels(names, fontsize=7.2)
    ax.set_xlabel(L("Predicted class", "Clasa prezisă", lang), fontsize=9)
    ax.set_ylabel(L("True class", "Clasa reală", lang), fontsize=9)
    ax._keep_y = True
    npl.out(fig, HERE, lang, "eurosat_confusion", bottom=0.0)
npl.chart_meta(N, "eurosat_confusion", "eurosat", "5,400 test tiles, fine-tuned ResNet-18, seed 20261108",
               "5.400 de imagini de test, ResNet-18 ajustat fin, sămînța 20261108")

# Grad-CAM on the fine-tuned model
mft = resnet18(False).to(DEV)
mft.load_state_dict(torch.load(os.path.join(CACHE, "eurosat_finetune.pt"), map_location=DEV))
mft.eval()
acts = {}
mft.layer3.register_forward_hook(lambda m_, i_, o: acts.__setitem__("a", o))
show_cls = [0, 3, 4, 7, 8, 9]
cam_ids = [int(e_te[yes[e_te] == c][3]) for c in show_cls]
cams, probs = [], []
for i_ in cam_ids:
    x = es_tensor(np.array([i_]), RES).to(DEV)
    out = mft(x)
    c = int(out.argmax(1))
    probs.append((c, float(F.softmax(out, 1)[0, c])))
    a = acts["a"]
    grad = torch.autograd.grad(out[0, c], a)[0]
    w = grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    cams.append(cam / (cam.max() + 1e-9))
N["gradcam"] = {"n": len(cam_ids), "map": 7, "probs_min": round(100 * min(p for _, p in probs), 1),
                "all_correct": bool(all(c == yes[i_] for (c, _), i_ in zip(probs, cam_ids)))}
for lang in LANGS:
    names = ES_EN if lang == "en" else ES_RO
    fig, axes = plt.subplots(2, 6, figsize=(6.4, 2.3))
    for j, i_ in enumerate(cam_ids):
        axes[0, j].imshow(Xes[i_])
        axes[1, j].imshow(Xes[i_].mean(2), cmap="gray")
        axes[1, j].imshow(cams[j], cmap="jet", alpha=0.45)
        for r in range(2):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        pct = f"{100 * probs[j][1]:.0f}%"
        axes[1, j].set_xlabel(names[probs[j][0]].replace(" ", "\n", 1) + f"\n{pct}", fontsize=7)
    npl.out(fig, HERE, lang, "gradcam", bottom=0.0)
npl.chart_meta(N, "gradcam", "eurosat", "six test tiles; Grad-CAM on the 7 x 7 maps of block 3 of the fine-tuned ResNet-18",
               "șase imagini de test; Grad-CAM pe hărțile de 7 x 7 ale blocului 3 din ResNet-18 ajustat fin")

# business illustration: land-use shares of the test tiles, true vs predicted (fine-tuned model)
true_sh = np.bincount(yes[e_te], minlength=10) / len(e_te) * 100
pred_sh = np.bincount(ES["finetune"]["pred"], minlength=10) / len(e_te) * 100
N["landuse"] = {"max_abs_share_err_pp": round(float(np.abs(true_sh - pred_sh).max()), 2),
                "crop_true": round(float(true_sh[0] + true_sh[6]), 2), "crop_pred": round(float(pred_sh[0] + pred_sh[6]), 2)}

with open(os.path.join(HERE, "L4_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote L4_numbers.json and", len(N["charts"]), "charts x 2 languages")
