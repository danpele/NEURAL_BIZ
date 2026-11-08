"""
analysis_L14.py -- Lecture 14: multimodal and vision-language models; synthesis of the course.
All computation of the lecture: numbers -> L14_numbers.json, charts -> charts/en|ro/*.pdf.

Real data only (see data/SOURCES.md):
  * EuroSAT (Helber et al., 2019): 27,000 Sentinel-2 RGB images of land use in Europe, 10 classes (on demand, torchvision);
  * S&P 500 daily closes (Yahoo Finance) rendered as chart images (as in Jheng, Pele, Tak and Găman, 2026);
  * euro area AAA yield curve (European Central Bank) rendered as table images.
Open models (Hugging Face): CLIP ViT-B/32 and ViT-B/16 (Radford et al., 2021), Qwen2-VL-2B-Instruct (Wang et al., 2024).
A small convolutional network (CNN) is trained from scratch as the benchmark (Lecture 4).

Embeddings and model answers are cached in lectures/L14/cache (git-ignored); per-item results used by the Quantlets are
written to lectures/L14/results/*.csv.
Run:  python3 lectures/L14/analysis_L14.py      (first run about 25-35 min on an Apple M-series laptop, then seconds)
"""
import io
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")      # avoid thread oversubscription with PyTorch
os.environ.setdefault("OMP_NUM_THREADS", "4")
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
npl.style()
SEED = npl.seed_all()
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(HERE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RES, exist_ok=True)
N = {"charts": {}, "seed": SEED}
T0 = time.time()

import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

DEV = npl.device()


def L(lang, en, ro):
    return en if lang == "en" else ro


def fmt_num(v, lang, d=1):
    s = f"{v:.{d}f}"
    return s if lang == "en" else s.replace(".", ",")


# ---------------------------------------------------------------- 0. EuroSAT and split (fixed a priori)
ES = torchvision.datasets.EuroSAT(npl.vision_root(), download=True)
CLS = ES.classes
y_all = np.array(ES.targets)
idx_tr, idx_te = train_test_split(np.arange(len(ES)), test_size=0.2, stratify=y_all, random_state=SEED)
N["data"] = {"n": int(len(ES)), "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)), "n_classes": len(CLS), "px": 64}
NAME_EN = {"AnnualCrop": "Annual crop", "Forest": "Forest", "HerbaceousVegetation": "Herbaceous vegetation", "Highway": "Highway",
           "Industrial": "Industrial", "Pasture": "Pasture", "PermanentCrop": "Permanent crop", "Residential": "Residential",
           "River": "River", "SeaLake": "Sea or lake"}
NAME_RO = {"AnnualCrop": "Culturi anuale", "Forest": "Pădure", "HerbaceousVegetation": "Vegetație ierboasă", "Highway": "Autostradă",
           "Industrial": "Zonă industrială", "Pasture": "Pășune", "PermanentCrop": "Culturi permanente", "Residential": "Zonă rezidențială",
           "River": "Rîu", "SeaLake": "Mare sau lac"}
# class descriptions used by Radford et al. (2021) for EuroSAT
PROMPT_NAMES = {"AnnualCrop": "annual crop land", "Forest": "forest", "HerbaceousVegetation": "brushland or shrubland",
                "Highway": "highway or road", "Industrial": "industrial buildings or commercial buildings", "Pasture": "pasture land",
                "PermanentCrop": "permanent crop land", "Residential": "residential buildings or homes or apartments",
                "River": "river", "SeaLake": "lake or sea"}
TEMPL = {"name": ["{}"], "photo": ["a photo of {}."], "sat": ["a satellite photo of {}."],
         "ens": ["a centered satellite photo of {}.", "a centered satellite photo of a {}.", "a centered satellite photo of the {}.",
                 "a satellite photo of {}.", "an aerial photo of {}."]}
CLIPS = {"B32": "openai/clip-vit-base-patch32", "B16": "openai/clip-vit-base-patch16"}


def load_clip(k):
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(CLIPS[k]).to(DEV).eval(), CLIPProcessor.from_pretrained(CLIPS[k])


def img_emb(model, proc, images, bs=128):
    out = []
    with torch.no_grad():
        for j in range(0, len(images), bs):
            px = proc(images=images[j:j + bs], return_tensors="pt")["pixel_values"].to(DEV)
            e = model.get_image_features(pixel_values=px)
            e = e if isinstance(e, torch.Tensor) else e.pooler_output
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def txt_emb(model, proc, texts):
    with torch.no_grad():
        t = proc(text=texts, return_tensors="pt", padding=True).to(DEV)
        e = model.get_text_features(**t)
        e = e if isinstance(e, torch.Tensor) else e.pooler_output
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


EMB, ZS, TIMES = {}, {}, {}
for k in CLIPS:
    path = os.path.join(CACHE, f"eurosat_{k}.npz")
    model, proc = load_clip(k)
    if not os.path.isfile(path):
        t = time.time()
        imgs = [ES[i][0] for i in range(len(ES))]
        E = img_emb(model, proc, imgs)
        np.savez(path, E=E, sec=time.time() - t)
        print(f"CLIP {k}: {len(E)} images in {time.time() - t:.0f} s")
    z = np.load(path)
    EMB[k] = z["E"]
    TIMES[k] = float(z["sec"]) / len(ES)
    for tk, tl in TEMPL.items():
        W = []
        for c in CLS:
            names = [c] if tk == "name" else [PROMPT_NAMES[c]]
            e = txt_emb(model, proc, [tpl.format(n) for tpl in tl for n in names]).mean(0)
            W.append(e / np.linalg.norm(e))
        W = np.array(W)
        S = EMB[k][idx_te] @ W.T
        ZS[(k, tk)] = S
    N.setdefault("clip", {})[k] = {"params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                                    "ms_per_image": round(1000 * TIMES[k], 2), "dim": int(EMB[k].shape[1])}
    del model
yte = y_all[idx_te]
N["zs"] = {f"{k}_{tk}": round(100 * float((ZS[(k, tk)].argmax(1) == yte).mean()), 1) for (k, tk) in ZS}
N["zs_best"] = max(N["zs"], key=N["zs"].get)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    tks = ["name", "photo", "sat", "ens"]
    lab = {"name": L(lang, "Folder name\n(\"AnnualCrop\")", "Numele dosarului\n(„AnnualCrop”)"),
           "photo": L(lang, "\"a photo of\n{class}.\"", "„a photo of\n{class}.”"),
           "sat": L(lang, "\"a satellite photo\nof {class}.\"", "„a satellite photo\nof {class}.”"),
           "ens": L(lang, "Average of 5\nsatellite templates", "Media a 5 șabloane\nsatelitare")}
    w = 0.38
    for j, (k, c) in enumerate((("B32", npl.PAL["data"]), ("B16", npl.PAL["model"]))):
        v = [N["zs"][f"{k}_{tk}"] for tk in tks]
        ax.bar(np.arange(4) + (j - 0.5) * w, v, w, color=c, label=f"CLIP ViT-{k[0]}/{k[1:]}")
        for i, x in enumerate(v):
            ax.text(i + (j - 0.5) * w, x + 1, fmt_num(x, lang), ha="center", fontsize=8.5)
    ax.axhline(10, color="#999999", ls=":", lw=1, label=L(lang, "Chance (10 classes)", "Nivelul hazardului (10 clase)"))
    ax.set_xticks(range(4))
    ax.set_xticklabels([lab[t] for t in tks], fontsize=8.5)
    ax.set_ylabel(L(lang, "Test accuracy (%)", "Acuratețea pe test (%)"))
    ax.set_ylim(0, 100)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "zeroshot_prompts")
npl.chart_meta(N, "zeroshot_prompts", "eurosat", f"{len(idx_te)} test images, no training",
               f"{len(idx_te)} de imagini de test, fără antrenare")

# confusion matrix of the best zero-shot model
bk, bt = N["zs_best"].split("_")
pred_zs = ZS[(bk, bt)].argmax(1)
CM = confusion_matrix(yte, pred_zs, labels=range(10))
N["zs_class"] = {c: round(100 * float(CM[i, i] / CM[i].sum()), 1) for i, c in enumerate(CLS)}
N["zs_worst"] = min(N["zs_class"], key=N["zs_class"].get)
N["zs_best_class"] = max(N["zs_class"], key=N["zs_class"].get)
off = CM.copy()
np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
N["zs_top_conf"] = {"true": CLS[i], "pred": CLS[j], "n": int(off[i, j]), "n_true": int(CM[i].sum())}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    R = CM / CM.sum(1, keepdims=True) * 100
    ax.imshow(R, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    nm = [NAME_EN[c] if lang == "en" else NAME_RO[c] for c in CLS]
    ax.set_xticks(range(10))
    ax.set_xticklabels(nm, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels(nm, fontsize=8.5)
    ax.tick_params(length=0)
    for a in range(10):
        for b in range(10):
            if R[a, b] >= 1:
                ax.text(b, a, f"{R[a, b]:.0f}", ha="center", va="center", fontsize=8, color="white" if R[a, b] > 55 else "#222222")
    ax.set_xlabel(L(lang, "Predicted class (zero-shot)", "Clasa prezisă (zero-shot)"), fontsize=9.5)
    ax.set_ylabel(L(lang, "True class", "Clasa reală"), fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax._keep_y = True
    npl.out(fig, HERE, lang, "zeroshot_confusion", bottom=0.02)
npl.chart_meta(N, "zeroshot_confusion", "eurosat", f"{len(idx_te)} test images; row percentages",
               f"{len(idx_te)} de imagini de test; procente pe rînd")

# text-to-image search: rank the test images by their similarity to the prompt of each class, precision of the top 100
S_best = ZS[(bk, bt)]
prec = {}
for i, c in enumerate(CLS):
    top = np.argsort(-S_best[:, i])[:100]
    prec[c] = float((yte[top] == i).mean())
N["search"] = {"k": 100, "mean": round(100 * float(np.mean(list(prec.values()))), 1), "per_class": {c: round(100 * v, 1) for c, v in prec.items()},
               "best": max(prec, key=prec.get), "worst": min(prec, key=prec.get)}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    order_c = sorted(CLS, key=lambda c: -prec[c])
    ax.bar(range(10), [100 * prec[c] for c in order_c], color=npl.PAL["model"], label=L(lang, "Precision of the 100 images closest to the class prompt",
                                                                                      "Precizia celor 100 de imagini cele mai apropiate de promptul clasei"))
    ax.axhline(10, color="#999999", ls=":", lw=1, label=L(lang, "Random images (10 %)", "Imagini aleatoare (10 %)"))
    ax.set_xticks(range(10))
    ax.set_xticklabels([NAME_EN[c] if lang == "en" else NAME_RO[c] for c in order_c], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel(L(lang, "Precision at 100 (%)", "Precizia la 100 (%)"))
    ax.set_ylim(0, 105)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "text_search")
npl.chart_meta(N, "text_search", "eurosat", f"{len(idx_te)} test images; model " + f"CLIP ViT-{bk[0]}/{bk[1:]}, 5 templates",
               f"{len(idx_te)} de imagini de test; modelul CLIP ViT-{bk[0]}/{bk[1:]}, 5 șabloane")

# ---------------------------------------------------------------- 1. few-shot: CLIP linear probe vs CNN trained from scratch
KS = [1, 2, 4, 8, 16, 32, 64, 128]
SEEDS = [SEED + s for s in range(5)]
ytr_all = y_all[idx_tr]


def sample_k(k, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(idx_tr[ytr_all == c], k, replace=False) for c in range(10)])


probe = {}
for k in KS:
    for s in SEEDS:
        ii = sample_k(k, s)
        lr = LogisticRegression(max_iter=3000, C=10.0).fit(EMB["B16"][ii], y_all[ii])
        probe[(k, s)] = float((lr.predict(EMB["B16"][idx_te]) == yte).mean())
lr = LogisticRegression(max_iter=3000, C=10.0).fit(EMB["B16"][idx_tr], ytr_all)
N["probe_full"] = round(100 * float((lr.predict(EMB["B16"][idx_te]) == yte).mean()), 1)
lr32 = LogisticRegression(max_iter=3000, C=10.0).fit(EMB["B32"][idx_tr], ytr_all)
N["probe_full_b32"] = round(100 * float((lr32.predict(EMB["B32"][idx_te]) == yte).mean()), 1)

# CNN from scratch (three convolution blocks), same few-shot samples, three seeds; and the full training set
X_all = np.stack([np.asarray(ES[i][0], dtype=np.uint8) for i in range(len(ES))]) if not os.path.isfile(os.path.join(CACHE, "eurosat_px.npy")) else None
if X_all is not None:
    np.save(os.path.join(CACHE, "eurosat_px.npy"), X_all)
X_all = np.load(os.path.join(CACHE, "eurosat_px.npy"))
MEAN = X_all[idx_tr].reshape(-1, 3).mean(0) / 255
STD = X_all[idx_tr].reshape(-1, 3).std(0) / 255


def to_t(ii):
    x = torch.tensor(X_all[ii], dtype=torch.float32).permute(0, 3, 1, 2) / 255
    return (x - torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)) / torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)


def cnn():
    def blk(i, o):
        return torch.nn.Sequential(torch.nn.Conv2d(i, o, 3, padding=1), torch.nn.BatchNorm2d(o), torch.nn.ReLU(),
                                   torch.nn.Conv2d(o, o, 3, padding=1), torch.nn.BatchNorm2d(o), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
    return torch.nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                               torch.nn.Dropout(0.3), torch.nn.Linear(128, 10))


def aug(x, g):
    if torch.rand(1, generator=g) < 0.5:
        x = x.flip(3)
    if torch.rand(1, generator=g) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,), generator=g)), (2, 3))


def train_cnn(ii, seed, steps=None, epochs=None):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = cnn().to(DEV)
    X, Yt = to_t(ii), torch.tensor(y_all[ii])
    n = len(ii)
    bs = min(64, n)
    if epochs is None:
        epochs = max(30, int(3000 / max(1, n // bs)))                  # about 3,000 gradient steps for small samples
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((n + bs - 1) // bs))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            xb = aug(X[b], g).to(DEV)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(xb), Yt[b].to(DEV))
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    acc, Xt = [], idx_te
    with torch.no_grad():
        for j in range(0, len(Xt), 512):
            acc.append((net(to_t(Xt[j:j + 512]).to(DEV)).argmax(1).cpu().numpy() == yte[j:j + 512]))
    return float(np.concatenate(acc).mean()), net


cpath = os.path.join(CACHE, "cnn_fewshot.json")
if os.path.isfile(cpath):
    CNNR = json.load(open(cpath))
else:
    t = time.time()
    CNNR = {}
    for k in [4, 16, 64, 128]:
        for s in SEEDS[:3]:
            CNNR[f"{k}_{s}"] = train_cnn(sample_k(k, s), s)[0]
    for s in SEEDS[:3]:
        CNNR[f"full_{s}"] = train_cnn(idx_tr, s, epochs=12)[0]
    CNNR["_seconds"] = time.time() - t
    json.dump(CNNR, open(cpath, "w"))
    print(f"CNN few-shot in {time.time() - t:.0f} s")
CK = [4, 16, 64, 128]
N["fewshot"] = {"probe": {str(k): {"mean": round(100 * float(np.mean([probe[(k, s)] for s in SEEDS])), 1),
                                   "min": round(100 * min(probe[(k, s)] for s in SEEDS), 1),
                                   "max": round(100 * max(probe[(k, s)] for s in SEEDS), 1)} for k in KS},
                "cnn": {str(k): {"mean": round(100 * float(np.mean([CNNR[f'{k}_{s}'] for s in SEEDS[:3]])), 1),
                                 "min": round(100 * min(CNNR[f'{k}_{s}'] for s in SEEDS[:3]), 1),
                                 "max": round(100 * max(CNNR[f'{k}_{s}'] for s in SEEDS[:3]), 1)} for k in CK}}
fullv = [CNNR[f"full_{s}"] for s in SEEDS[:3]]
N["cnn_full"] = {"mean": round(100 * float(np.mean(fullv)), 1), "min": round(100 * min(fullv), 1), "max": round(100 * max(fullv), 1),
                 "params_k": round(sum(p.numel() for p in cnn().parameters()) / 1e3, 1)}
# shots per class at which the probe first beats zero-shot
zsb = N["zs"][N["zs_best"]]
N["probe_beats_zs_k"] = next((k for k in KS if N["fewshot"]["probe"][str(k)]["mean"] > zsb), None)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x = np.log2(KS)
    pm = [N["fewshot"]["probe"][str(k)]["mean"] for k in KS]
    ax.fill_between(x, [N["fewshot"]["probe"][str(k)]["min"] for k in KS], [N["fewshot"]["probe"][str(k)]["max"] for k in KS],
                    color=npl.PAL["model"], alpha=0.2, lw=0)
    ax.plot(x, pm, marker="o", color=npl.PAL["model"], label=L(lang, "CLIP ViT-B/16 embeddings + logistic (5 seeds)", "Embedding CLIP ViT-B/16 + logistică (5 valori aleatoare)"))
    xc = np.log2(CK)
    ax.fill_between(xc, [N["fewshot"]["cnn"][str(k)]["min"] for k in CK], [N["fewshot"]["cnn"][str(k)]["max"] for k in CK],
                    color=npl.PAL["benchmark"], alpha=0.2, lw=0)
    ax.plot(xc, [N["fewshot"]["cnn"][str(k)]["mean"] for k in CK], marker="s", color=npl.PAL["benchmark"],
            label=L(lang, "CNN trained from scratch (3 seeds)", "CNN antrenată de la zero (3 valori aleatoare)"))
    ax.axhline(zsb, color=npl.PAL["alt"], ls="--", lw=1.2, label=L(lang, "CLIP zero-shot (no labelled image)", "CLIP zero-shot (nicio imagine etichetată)"))
    ax.axhline(N["cnn_full"]["mean"], color=npl.PAL["benchmark"], ls=":", lw=1.2,
               label=L(lang, f"CNN on all {len(idx_tr):,} training images", f"CNN pe toate cele {len(idx_tr):,} de imagini de antrenare".replace(",", ".")))
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel(L(lang, "Labelled images per class", "Imagini etichetate pe clasă"))
    ax.set_ylabel(L(lang, "Test accuracy (%)", "Acuratețea pe test (%)"))
    ax.set_ylim(0, 100)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "fewshot_eurosat")
npl.chart_meta(N, "fewshot_eurosat", "eurosat", f"{len(idx_te)} test images; shaded = range over seeds",
               f"{len(idx_te)} de imagini de test; zona colorată = intervalul pe valorile generatorului aleator")

# ---------------------------------------------------------------- 2. example predictions and the patch view of a vision transformer
rng = np.random.default_rng(SEED)
ex = [int(rng.choice(np.where(yte == c)[0])) for c in rng.choice(10, 8, replace=False)]
P_ex = np.exp(100 * ZS[(bk, bt)][ex])
P_ex = P_ex / P_ex.sum(1, keepdims=True)
N["examples"] = {"n_right": int(sum(int(pred_zs[e] == yte[e]) for e in ex)), "n": len(ex)}
for lang in ("en", "ro"):
    fig, axs = plt.subplots(2, 4, figsize=(6.4, 3.6))
    nm = NAME_EN if lang == "en" else NAME_RO
    for a, e in zip(axs.ravel(), ex):
        a.imshow(X_all[idx_te[e]])
        a.set_xticks([])
        a.set_yticks([])
        ok = pred_zs[e] == yte[e]
        a.set_xlabel(L(lang, "true: ", "real: ") + nm[CLS[yte[e]]] + "\n" + L(lang, "CLIP: ", "CLIP: ") + nm[CLS[pred_zs[e]]] +
                     f" ({fmt_num(100 * P_ex[ex.index(e)].max(), lang, 0)} %)", fontsize=7.5, color=npl.SERIES[5] if ok else npl.PAL["positive"])
        for s in a.spines.values():
            s.set_color(npl.SERIES[5] if ok else npl.PAL["positive"])
            s.set_linewidth(2)
            s.set_visible(True)
    npl.out(fig, HERE, lang, "zeroshot_examples", bottom=0.0)
npl.chart_meta(N, "zeroshot_examples", "eurosat", "8 random test images (seeded); green = correct, red = wrong",
               "8 imagini de test alese aleator (valoare fixă); verde = corect, roșu = greșit")

# patches of a vision transformer (mechanism on a real image) + attention of the class token (CLIP ViT-B/16)
from transformers import CLIPModel, CLIPProcessor  # noqa: E402
cm16 = CLIPModel.from_pretrained(CLIPS["B16"], attn_implementation="eager").to(DEV).eval()
pr16 = CLIPProcessor.from_pretrained(CLIPS["B16"])
att_idx = [int(np.where(yte == CLS.index(c))[0][3]) for c in ("Highway", "River", "Industrial", "Residential")]


def rollout(img):
    px = pr16(images=[img], return_tensors="pt")["pixel_values"].to(DEV)
    with torch.no_grad():
        out = cm16.vision_model(pixel_values=px, output_attentions=True)
    R = torch.eye(out.attentions[0].shape[-1], device=DEV)
    for A in out.attentions:                                 # attention rollout (Abnar and Zuidema, 2020)
        A = A.mean(1)[0]
        A = A + torch.eye(A.shape[-1], device=DEV)
        A = A / A.sum(-1, keepdim=True)
        R = A @ R
    m = R[0, 1:].reshape(14, 14).float().cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-9), px[0].float().cpu().numpy()


for lang in ("en", "ro"):
    fig, axs = plt.subplots(2, 4, figsize=(6.4, 3.4))
    nm = NAME_EN if lang == "en" else NAME_RO
    for c_, e in enumerate(att_idx):
        img = Image.fromarray(X_all[idx_te[e]])
        m, px = rollout(img)
        big = np.asarray(img.resize((224, 224), Image.NEAREST))
        axs[0, c_].imshow(big)
        for g in range(0, 225, 16):
            axs[0, c_].axhline(g - 0.5, color="white", lw=0.4)
            axs[0, c_].axvline(g - 0.5, color="white", lw=0.4)
        axs[1, c_].imshow(big)
        axs[1, c_].imshow(np.kron(m, np.ones((16, 16))), cmap="inferno", alpha=0.55)
        axs[0, c_].set_xlabel(nm[CLS[yte[e]]], fontsize=8)
        axs[1, c_].set_xlabel(L(lang, "class-token attention", "atenția tokenului de clasă"), fontsize=7.5)
        for a in axs[:, c_]:
            a.set_xticks([])
            a.set_yticks([])
    npl.out(fig, HERE, lang, "vit_patches_attention", bottom=0.0)
npl.chart_meta(N, "vit_patches_attention", "eurosat", "4 test images upscaled to 224 x 224 pixels = 14 x 14 patches of 16 x 16; model CLIP ViT-B/16",
               "4 imagini de test mărite la 224 x 224 de pixeli = 14 x 14 petice de 16 x 16; modelul CLIP ViT-B/16")
N["vit"] = {"patch": 16, "grid": 14, "n_tokens": 197, "layers": len(cm16.vision_model.encoder.layers), "heads": int(cm16.config.vision_config.num_attention_heads),
            "width": int(cm16.config.vision_config.hidden_size)}

# ---------------------------------------------------------------- 3. markets as images: S&P 500 regime classification (after Jheng, Pele, Tak and Găman, 2026)
px = npl.markets("adjclose", ["SP500"])["SP500"].dropna().loc[:"2026-10-31"]
r = np.log(px).diff().dropna()
vol20 = r.rolling(20).std() * np.sqrt(252) * 100
thr = vol20.rolling(750, min_periods=500).quantile(0.9).shift(1)          # rolling 90th percentile, past values only
WIN, STRIDE = 120, 5
ends = [i for i in range(len(px)) if i >= 770 and (len(px) - 1 - i) % STRIDE == 0]
win = pd.DataFrame({"end": [px.index[i] for i in ends]})
win["stress"] = [int(vol20.loc[px.index[i]] > thr.loc[px.index[i]]) if px.index[i] in thr.index and np.isfinite(thr.loc[px.index[i]]) else -1 for i in ends]
win = win[win.stress >= 0].reset_index(drop=True)
ends = [px.index.get_loc(d) for d in win.end]
win["ret120"] = [float(np.log(px.iloc[i] / px.iloc[i - WIN + 1])) for i in ends]
win["ret20"] = [float(np.log(px.iloc[i] / px.iloc[i - 19])) for i in ends]
win["vol60"] = [float(r.loc[:px.index[i]].iloc[-60:].std() * np.sqrt(252) * 100) for i in ends]
win["dd"] = [float(px.iloc[i] / px.iloc[i - WIN + 1:i + 1].max() - 1) for i in ends]
win["up"] = (win.ret120 > 0).astype(int)
SPLIT = pd.Timestamp("2019-01-01")
tr = win.end < SPLIT
N["charts_task"] = {"n": int(len(win)), "n_train": int(tr.sum()), "n_test": int((~tr).sum()), "window": WIN, "stride": STRIDE,
                    "first": str(win.end.min().date()), "last": str(win.end.max().date()),
                    "base_test": round(100 * float(win.stress[~tr].mean()), 1), "base_train": round(100 * float(win.stress[tr].mean()), 1)}


def render(i, size=224):
    seg = px.iloc[i - WIN + 1:i + 1].values
    seg = seg / seg[0] * 100
    dd = seg / np.maximum.accumulate(seg) - 1
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    a1 = fig.add_axes([0.08, 0.36, 0.9, 0.6])
    a1.plot(seg, color="#1f4e9a", lw=1.2)
    a1.set_xticks([])
    a1.tick_params(labelsize=5)
    a2 = fig.add_axes([0.08, 0.05, 0.9, 0.26])
    a2.fill_between(range(len(dd)), 100 * dd, 0, color="#c0392b", lw=0)
    a2.set_xticks([])
    a2.set_ylim(min(-20, 100 * dd.min() * 1.05), 0)
    a2.tick_params(labelsize=5)
    for a in (a1, a2):
        a.set_facecolor("white")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.patch.set_facecolor("white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


ipath = os.path.join(CACHE, "chart_imgs.npy")
if not os.path.isfile(ipath):
    t = time.time()
    np.save(ipath, np.stack([np.asarray(render(i)) for i in ends]))
    print(f"rendered {len(ends)} chart images in {time.time() - t:.0f} s")
IM = np.load(ipath)
epath = os.path.join(CACHE, "chart_clip.npz")
model, proc = load_clip("B16")
if not os.path.isfile(epath):
    np.savez(epath, E=img_emb(model, proc, [Image.fromarray(x) for x in IM]))
CE = np.load(epath)["E"]
txt = txt_emb(model, proc, ["a line chart of a calm stock market that rises steadily",
                            "a line chart of a volatile stock market with a sharp crash and a deep drawdown"])
del model
zs_score = CE @ txt[1] - CE @ txt[0]
ytr_c, yte_c = win.stress[tr].values, win.stress[~tr].values
N["regime"] = {}
N["regime"]["clip_zs"] = {"auc": round(float(roc_auc_score(yte_c, zs_score[~tr])), 3), "ap": round(float(average_precision_score(yte_c, zs_score[~tr])), 3)}
lrp = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced").fit(CE[tr], ytr_c)
s_probe = lrp.predict_proba(CE[~tr])[:, 1]
N["regime"]["clip_probe"] = {"auc": round(float(roc_auc_score(yte_c, s_probe)), 3), "ap": round(float(average_precision_score(yte_c, s_probe)), 3)}
F = ["ret20", "ret120", "vol60", "dd"]
mu, sd = win.loc[tr, F].mean(), win.loc[tr, F].std()
lrn = LogisticRegression(max_iter=4000, class_weight="balanced").fit(((win.loc[tr, F] - mu) / sd).values, ytr_c)
s_num = lrn.predict_proba(((win.loc[~tr, F] - mu) / sd).values)[:, 1]
N["regime"]["numeric"] = {"auc": round(float(roc_auc_score(yte_c, s_num)), 3), "ap": round(float(average_precision_score(yte_c, s_num)), 3)}

# vision-language model (Qwen2-VL-2B-Instruct): regime question and trend question on the test images; table reading
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: E402

VL = "Qwen/Qwen2-VL-2B-Instruct"
_vl = {}


def vl():
    if not _vl:
        p = AutoProcessor.from_pretrained(VL, min_pixels=128 * 28 * 28, max_pixels=320 * 28 * 28)
        p.tokenizer.padding_side = "left"
        m = Qwen2VLForConditionalGeneration.from_pretrained(VL, dtype=torch.float16 if DEV != "cpu" else torch.float32).to(DEV).eval()
        _vl["p"], _vl["m"] = p, m
    return _vl["p"], _vl["m"]


def vl_yesno(images, question, words, bs=8):
    """Probability of the first answer word versus the second (first-token probabilities, renormalised)."""
    p, m = vl()
    tok = p.tokenizer
    ids = [[tok.encode(v)[0] for v in (w, w.capitalize(), " " + w)] for w in words]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    txt = p.apply_chat_template(msgs, add_generation_prompt=True)
    out = []
    with torch.no_grad():
        for j in range(0, len(images), bs):
            ims = images[j:j + bs]
            inp = p(text=[txt] * len(ims), images=ims, return_tensors="pt", padding=True).to(DEV)
            pr = torch.softmax(m(**inp).logits[:, -1].float(), -1)
            a = torch.stack([pr[:, list(set(i))].sum(1) for i in ids], 1)
            out.append((a[:, 0] / a.sum(1)).cpu().numpy())
    return np.concatenate(out)


te_idx = np.where(~tr)[0]
vpath = os.path.join(CACHE, "vl_charts.npz")
if not os.path.isfile(vpath):
    t = time.time()
    ims = [Image.fromarray(IM[i]) for i in te_idx]
    s_vl = vl_yesno(ims, "This image shows a stock index over the last 120 trading days (top: price; bottom: drawdown from the "
                         "running maximum). Is the market in a turbulent, high-volatility state at the end of the window? Answer yes or no.",
                    ["yes", "no"])
    s_up = vl_yesno(ims, "This image shows a stock index over 120 trading days (top panel: price). Did the price end higher or lower "
                         "than where it started? Answer higher or lower.", ["higher", "lower"])
    np.savez(vpath, s_vl=s_vl, s_up=s_up, sec=(time.time() - t) / (2 * len(ims)))
    print(f"Qwen2-VL on {len(ims)} charts in {time.time() - t:.0f} s")
z = np.load(vpath)
s_vl, s_up = z["s_vl"], z["s_up"]
N["vl"] = {"ms_per_image": round(1000 * float(z["sec"]), 0)}
N["regime"]["vlm_zs"] = {"auc": round(float(roc_auc_score(yte_c, s_vl)), 3), "ap": round(float(average_precision_score(yte_c, s_vl)), 3)}
up_true = win.up.values[te_idx]
N["vl"]["trend_acc"] = round(100 * float(((s_up > 0.5).astype(int) == up_true).mean()), 1)
N["vl"]["trend_auc"] = round(float(roc_auc_score(up_true, s_up)), 3)
N["vl"]["trend_base"] = round(100 * float(max(up_true.mean(), 1 - up_true.mean())), 1)
N["vl"]["trend_n"] = int(len(up_true))
pd.DataFrame({"end": win.end.dt.date, "stress": win.stress, "up": win.up, "split": np.where(tr, "train", "test"),
              "ret20": win.ret20, "ret120": win.ret120, "vol60": win.vol60, "dd": win.dd,
              "clip_zs": zs_score, "clip_probe": np.r_[np.full(tr.sum(), np.nan), s_probe],
              "numeric": np.r_[np.full(tr.sum(), np.nan), s_num], "vlm_stress": np.r_[np.full(tr.sum(), np.nan), s_vl],
              "vlm_up": np.r_[np.full(tr.sum(), np.nan), s_up]}).to_csv(os.path.join(RES, "chart_regime.csv"), index=False)

from sklearn.metrics import roc_curve  # noqa: E402
for lang in ("en", "ro"):
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    calm = te_idx[np.argmin(np.where(yte_c == 0, win.vol60.values[te_idx], np.inf))]
    stress = te_idx[np.argmax(np.where(yte_c == 1, -win.dd.values[te_idx], -np.inf))]
    for a, i, lab in ((axs[0], calm, L(lang, "calm window", "fereastră calmă")), (axs[1], stress, L(lang, "stress window", "fereastră de criză"))):
        a.imshow(IM[i])
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel(lab + f"\n{win.end[i].date()}", fontsize=8)
    for key, c, lab in (("numeric", npl.PAL["benchmark"], L(lang, "Numbers + logistic", "Cifre + logistică")),
                        ("clip_probe", npl.PAL["model"], L(lang, "CLIP image + logistic", "Imagine CLIP + logistică")),
                        ("clip_zs", npl.PAL["alt"], L(lang, "CLIP zero-shot", "CLIP zero-shot")),
                        ("vlm_zs", npl.PAL["c7"], L(lang, "Qwen2-VL zero-shot", "Qwen2-VL zero-shot"))):
        sc = {"numeric": s_num, "clip_probe": s_probe, "clip_zs": zs_score[~tr], "vlm_zs": s_vl}[key]
        fpr, tpr, _ = roc_curve(yte_c, sc)
        axs[2].plot(fpr, tpr, color=c, lw=1.3, label=lab + f" (AUC {fmt_num(N['regime'][key]['auc'], lang, 2)})")
    axs[2].plot([0, 1], [0, 1], color="#999999", ls=":", lw=1)
    axs[2].set_xlabel(L(lang, "False positive rate", "Rata alarmelor false"), fontsize=9)
    axs[2].set_ylabel(L(lang, "True positive rate", "Rata detecțiilor"), fontsize=9)
    axs[2].tick_params(labelsize=8)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "chart_regime", also_x=[axs[2]])
npl.chart_meta(N, "chart_regime", "yahoo", f"S\\&P 500, windows of {WIN} trading days every {STRIDE} days; test {npl.period(win.end[~tr].min(), win.end.max(), 'en')}",
               f"S\\&P 500, ferestre de {WIN} de zile de tranzacționare la fiecare {STRIDE} zile; test {npl.period(win.end[~tr].min(), win.end.max(), 'ro')}")

# table reading: euro area yield curve tables rendered as images, the vision-language model reads the 10-year yield
YC = npl.yield_curve().dropna()
rng = np.random.default_rng(SEED)
tab_dates = sorted(rng.choice(YC.index[YC.index >= "2015-01-01"], 60, replace=False))
MAT = ["y_3M", "y_6M", "y_1Y", "y_2Y", "y_3Y", "y_5Y", "y_7Y", "y_10Y", "y_15Y", "y_20Y", "y_30Y"]


def render_table(d):
    vals = YC.loc[d, MAT].values
    fig = plt.figure(figsize=(4.6, 1.3), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    t = ax.table(cellText=[[f"{v:.2f}" for v in vals]], colLabels=[m[2:] for m in MAT], rowLabels=["Yield (%)"], loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.6)
    fig.text(0.02, 0.9, f"Euro area AAA yield curve, {pd.Timestamp(d).date()}", fontsize=8)
    fig.patch.set_facecolor("white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


tpath = os.path.join(CACHE, "vl_tables.json")
if not os.path.isfile(tpath):
    p, m = vl()
    outs = []
    t = time.time()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text":
            "What is the 10-year yield (column 10Y) in this table? Reply with the number only."}]}]
    txt = p.apply_chat_template(msgs, add_generation_prompt=True)
    with torch.no_grad():
        for j in range(0, len(tab_dates), 6):
            ims = [render_table(d) for d in tab_dates[j:j + 6]]
            inp = p(text=[txt] * len(ims), images=ims, return_tensors="pt", padding=True).to(DEV)
            g = m.generate(**inp, max_new_tokens=8, do_sample=False)
            outs += p.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
    json.dump({"text": outs, "sec": (time.time() - t) / len(tab_dates)}, open(tpath, "w"))
TB = json.load(open(tpath))
true10 = np.round(YC.loc[tab_dates, "y_10Y"].values, 2)
read = np.array([float(re.search(r"-?\d+(?:\.\d+)?", s).group(0)) if re.search(r"-?\d+(?:\.\d+)?", s) else np.nan for s in TB["text"]])
N["table"] = {"n": len(tab_dates), "exact": round(100 * float(np.mean(np.isclose(read, true10, atol=0.001))), 1),
              "mae": round(float(np.nanmean(np.abs(read - true10))), 3), "parsed": round(100 * float(np.isfinite(read).mean()), 1)}
wrong = np.where(~np.isclose(read, true10, atol=0.001))[0]
if len(wrong):
    w0 = wrong[0]
    N["table"]["wrong_example"] = {"date": str(pd.Timestamp(tab_dates[w0]).date()), "true": float(true10[w0]), "read": float(read[w0]) if np.isfinite(read[w0]) else None}
pd.DataFrame({"date": [pd.Timestamp(d).date() for d in tab_dates], "true_10y": true10, "answer": TB["text"]}).to_csv(
    os.path.join(RES, "table_reading.csv"), index=False)
render_table(tab_dates[0]).save(os.path.join(RES, "table_example.png"))
VL_PARAMS = sum(q.numel() for q in vl()[1].parameters()) / 1e9
del _vl["m"]

# ---------------------------------------------------------------- 4a. small result files for the Quantlets
pd.DataFrame({"index": idx_te, "true": [CLS[i] for i in yte],
              **{f"{k}_{tk}": [CLS[i] for i in ZS[(k, tk)].argmax(1)] for (k, tk) in ZS}}).to_csv(os.path.join(RES, "eurosat_zeroshot.csv"), index=False)
pd.DataFrame(ZS[(bk, bt)], columns=CLS).assign(true=[CLS[i] for i in yte]).to_csv(os.path.join(RES, "eurosat_scores_best.csv"), index=False)
fsr = [["probe", k, s, probe[(k, s)]] for k in KS for s in SEEDS] + [["cnn", k, s, CNNR[f"{k}_{s}"]] for k in CK for s in SEEDS[:3]] + \
      [["cnn_full", len(idx_tr) // 10, s, CNNR[f"full_{s}"]] for s in SEEDS[:3]]
pd.DataFrame(fsr, columns=["method", "k", "seed", "acc"]).to_csv(os.path.join(RES, "fewshot_eurosat.csv"), index=False)

# ---------------------------------------------------------------- 4. cost and size summary (measured)
N["cost"] = {"cnn_params_k": N["cnn_full"]["params_k"], "b32_params_m": N["clip"]["B32"]["params_m"], "b16_params_m": N["clip"]["B16"]["params_m"],
             "vl_params_b": round(VL_PARAMS, 2), "hardware": f"Apple silicon, PyTorch {torch.__version__.split('+')[0]}, device {DEV}"}

N["runtime_min"] = round((time.time() - T0) / 60, 1)
with open(os.path.join(HERE, "L14_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print("wrote L14_numbers.json and", len(N["charts"]), "charts x 2 languages;", f"{N['runtime_min']} min")
print(json.dumps({k: N[k] for k in ("zs", "probe_full", "cnn_full", "regime", "vl", "table")}, indent=0))
