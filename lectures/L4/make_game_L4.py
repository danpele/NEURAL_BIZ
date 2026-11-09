"""
make_game_L4.py -- data of the Lecture 4 game "You or the CNN?" (interactive/L4_game.html, L4_game_ro.html).

The student sees a real EuroSAT test tile (Sentinel-2, 64 x 64 pixels) and picks its land-use class among four options
(the true class, the class predicted by the model if it differs, and random other classes, shuffled), then sees what
the fine-tuned ResNet-18 of the lecture predicted and the true class.

Same data and split as analysis_L4.py (stratified 60/20/20 per class with RandomState(seed of the course)); the model's
answers are the cached test predictions of the fine-tuned ResNet-18 (cache/eurosat_main.pkl, key finetune.pred), whose
accuracy equals eurosat.finetune_acc of L4_numbers.json (checked below). The confidence shown is the softmax probability
of the same model (cache/eurosat_finetune.pt, 112 px input, as in the lecture), computed for the pool tiles only.

Pool, drawn at random from the 5,400 TEST tiles with the seed of the course: 3 tiles per class among those the model
classifies correctly (30) and 10 of the 67 tiles it gets wrong. Every game draws 8 correct and 2 wrong tiles from the pool,
so the errors are over-represented (20% of the cards versus 1.24% of the test set); the page says so.

Run:  python3 lectures/L4/make_game_L4.py   ->  interactive/L4_game_data.js, interactive/game_img/L4/*.png, the two pages
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402

SEED = npl.seed_all()
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(ROOT, "interactive", "L4_game_data.js")
IMG = os.path.join(ROOT, "interactive", "game_img", "L4")
os.makedirs(IMG, exist_ok=True)
CACHE = os.path.join(HERE, "cache")
NUM = json.load(open(os.path.join(HERE, "L4_numbers.json"), encoding="utf-8"))

# --------------------------------------------------------------------- data and split, exactly as in analysis_L4.py
z = np.load(os.path.join(CACHE, "eurosat_u8.npz"))
Xes, yes = z["X"], z["y"]
rs = np.random.RandomState(SEED)
e_tr, e_va, e_te = [], [], []
for c in range(10):
    ids = rs.permutation(np.where(yes == c)[0])
    n = len(ids)
    e_tr += list(ids[:int(0.6 * n)]); e_va += list(ids[int(0.6 * n):int(0.8 * n)]); e_te += list(ids[int(0.8 * n):])
e_te = np.array(e_te)
with open(os.path.join(CACHE, "eurosat_main.pkl"), "rb") as f:
    ES = pickle.load(f)
pred = np.asarray(ES["finetune"]["pred"])
yte = yes[e_te]
acc = 100 * float((pred == yte).mean())
assert len(e_te) == NUM["eurosat"]["n_test"]
assert round(acc, 2) == NUM["eurosat"]["finetune_acc"], (acc, NUM["eurosat"]["finetune_acc"])
assert round(100 * ES["finetune"]["test_acc"], 2) == NUM["eurosat"]["finetune_acc"]
n_err = int((pred != yte).sum())
assert n_err == NUM["eurosat"]["n_errors_ft"]

# --------------------------------------------------------------------- pool: random, stratified by class and by model correctness
rng = np.random.default_rng(SEED)
right_pos = np.concatenate([rng.choice(np.where((yte == c) & (pred == yte))[0], 3, replace=False) for c in range(10)])
wrong_pos = rng.choice(np.where(pred != yte)[0], 10, replace=False)

# softmax confidence of the fine-tuned ResNet-18 (same preprocessing as es_tensor of analysis_L4.py, 112 px)
m = torchvision.models.resnet18(weights=None)
m.fc = nn.Linear(512, 10)
m.load_state_dict(torch.load(os.path.join(CACHE, "eurosat_finetune.pt"), map_location="cpu"))
m.eval()
MU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
SD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
pos = np.concatenate([right_pos, wrong_pos])
x = (torch.tensor(Xes[e_te[pos]]).permute(0, 3, 1, 2).float() / 255.0 - MU) / SD
x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
with torch.no_grad():
    prob = F.softmax(m(x), 1).numpy()
same = int((prob.argmax(1) == pred[pos]).sum())
print(f"recomputed argmax equals the cached prediction on {same} of {len(pos)} pool tiles")

pool = []
for j, k in enumerate(pos):
    name = f"t{int(e_te[k])}.png"
    Image.fromarray(Xes[e_te[k]]).save(os.path.join(IMG, name), optimize=True)
    pc = int(pred[k])
    pool.append({"img": name, "y": int(yte[k]), "pred": pc, "conf": round(float(prob[j, pc]), 3), "ok": int(pc == yte[k])})
    if prob[j].argmax() != pc:    # should not happen; keep the cached prediction and flag it
        print("  argmax differs for tile", int(e_te[k]))

info = {"n": NUM["eurosat"]["n"], "n_test": int(len(e_te)), "acc": round(acc, 2), "n_err": n_err, "seed": SEED,
        "pool_right": len(right_pos), "pool_wrong": len(wrong_pos), "per_game": 10, "wrong_per_game": 2,
        "scratch_acc": NUM["eurosat"]["scratch_acc"], "top_conf": NUM["eurosat"]["top_conf"],
        "b10_ft": NUM["budget"]["finetune"]["10"], "b10_scratch": NUM["budget"]["scratch"]["10"]}
with open(OUT, "w", encoding="utf-8") as f:
    f.write("// generated by lectures/L4/make_game_L4.py -- do not edit\n")
    f.write("window.NN_GAME_L4 = " + json.dumps({"info": info, "pool": pool}, ensure_ascii=False) + ";\n")
sizes = [os.path.getsize(os.path.join(IMG, p["img"])) for p in pool]
print(OUT, info, f"images: {len(sizes)}, max {max(sizes)} B, total {sum(sizes)} B")

# ------------------------------------------------------------------ the two pages (interactive/L4_game.html, L4_game_ro.html)
from game_page import write_game  # noqa: E402


def ro(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


tc = info["top_conf"]
err_pct = 100 * info["n_err"] / info["n_test"]
write_game(4, en=dict(
    title="You or the CNN? Land use from satellite images",
    lead="You see 10 real Sentinel-2 satellite images from the EuroSAT data set. Each image covers 640 m × 640 m of land in Europe "
         "(64 × 64 pixels of 10 m). For each image, choose its land-use class among four options. After each answer you see what the "
         "convolutional neural network (CNN) of Lecture 4 predicted and the true class.",
    how="The network is a ResNet-18 (residual network with 18 layers) pre-trained on the ImageNet photographs and fine-tuned on 16,200 "
        "EuroSAT images (transfer learning). It never saw the images of the game: they come from the 5,400 test images. "
        "Two cards in each game are images that the network classifies wrongly.",
    lesson=[f"<li>On all {info['n_test']:,} test images the fine-tuned network is right in {info['acc']:.2f}% of cases: {info['n_err']} errors, "
            f"most often {tc['true'].lower()} taken for {tc['pred'].lower()} ({tc['n']} images).</li>",
            f"<li>Transfer learning matters most when labels are scarce: with only 10 labelled images per class, the fine-tuned network reaches "
            f"{info['b10_ft']:.1f}%, a CNN trained from scratch {info['b10_scratch']:.1f}%.</li>",
            f"<li>The game over-represents the network's errors (2 cards in 10, against {err_pct:.2f}% of the test images), so its score in the game "
            "is lower than on the full test set.</li>"],
    src=f"Data: EuroSAT, Sentinel-2 images (Helber et al., 2019; 27,000 images, 10 classes). The 40 images of the game are drawn at random "
        f"from the test images with the seed {SEED}: 3 per class among those the network classifies correctly and 10 of its {info['n_err']} errors; "
        "every game draws 8 correct and 2 wrong ones. Model: cached predictions of lectures/L4/analysis_L4.py. Code: lectures/L4/make_game_L4.py."),
  ro=dict(
    title="Dumneavoastră sau rețeaua convoluțională? Utilizarea terenului din imagini satelitare",
    lead="Vedeți 10 imagini satelitare reale Sentinel-2 din setul de date EuroSAT. Fiecare imagine acoperă 640 m × 640 m de teren din Europa "
         "(64 × 64 pixeli de 10 m). Pentru fiecare imagine alegeți clasa de utilizare a terenului dintre patru variante. După fiecare răspuns "
         "vedeți ce a prezis rețeaua neuronală convoluțională (CNN) din cursul 4 și clasa corectă.",
    how="Rețeaua este un ResNet-18 (rețea reziduală cu 18 straturi) preantrenat pe fotografiile ImageNet și ajustat fin pe 16.200 de imagini "
        "EuroSAT (transferul învățării). Nu a văzut imaginile din joc: ele provin din cele 5.400 de imagini de test. "
        "În fiecare joc, două carduri sînt imagini pe care rețeaua le clasifică greșit.",
    lesson=[f"<li>Pe toate cele {info['n_test']:,} de imagini de test, rețeaua ajustată fin are dreptate în {ro(info['acc'])} % din cazuri: "
            f"{info['n_err']} de erori, cel mai des {tc['true_ro'].lower()} luate drept {tc['pred_ro'].lower()} ({tc['n']} imagini).</li>".replace(f"{info['n_test']:,}", f"{info['n_test']:,}".replace(",", ".")),
            f"<li>Transferul învățării contează cel mai mult cînd etichetele sînt puține: cu doar 10 imagini etichetate pe clasă, rețeaua ajustată fin "
            f"ajunge la {ro(info['b10_ft'], 1)} %, iar o CNN antrenată de la zero la {ro(info['b10_scratch'], 1)} %.</li>",
            f"<li>Jocul suprareprezintă erorile rețelei (2 carduri din 10, față de {ro(err_pct)} % din imaginile de test), așa că scorul ei în joc "
            "este mai mic decît pe întregul set de test.</li>"],
    src=f"Date: EuroSAT, imagini Sentinel-2 (Helber et al., 2019; 27.000 de imagini, 10 clase). Cele 40 de imagini din joc sînt extrase aleator "
        f"dintre imaginile de test, cu sămînța {SEED}: 3 pe clasă dintre cele clasificate corect de rețea și 10 dintre cele {info['n_err']} de erori ale ei; "
        "fiecare joc extrage 8 corecte și 2 greșite. Modelul: predicțiile salvate de lectures/L4/analysis_L4.py. Cod: lectures/L4/make_game_L4.py."))
print("pages written")
