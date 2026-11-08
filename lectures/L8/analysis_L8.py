"""
analysis_L8.py -- Lecture 8 "Generative models: autoencoders, VAE, GAN, diffusion": all computation on real data.
Writes lectures/L8/L8_numbers.json, charts/en|ro/*.pdf and lectures/L8/results/*.csv (small files read by the Quantlets).

Four experiments, each with a benchmark:
  A. Autoencoder as a fraud detector (ULB credit-card transactions): trained on legitimate transactions only, in time
     order (first 60 % of the two days = training, next 20 % = validation, last 20 % = test); benchmark: principal
     component analysis (PCA) reconstruction (= a linear autoencoder) and a supervised logistic regression.
     Precision and PR-AUC are weighted with the column `weight` so that they refer to the full data set.
  B. Variational autoencoder (VAE) on Fashion-MNIST: 2-D latent space for the picture, 16-D for samples, beta in {0.5, 1, 4}.
  C. Generative adversarial network (GAN) on S&P 500 daily log returns, training 2005-2019, test 2020 onwards;
     benchmark: Gaussian distribution and historical simulation for the 1 % Value-at-Risk.
  D. Denoising diffusion model (DDPM) on daily euro-area AAA yield curves (11 maturities); benchmarks: multivariate
     Gaussian and a VAE.
Seeds: 20261108 for the data splits, 1-3 for the networks (mean and standard deviation over seeds where stated).
Run:  python3 lectures/L8/analysis_L8.py    (about 8 minutes on an Apple-silicon laptop; trained results cached in cache/)
"""
import json
import math
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
import torch.nn.functional as F  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve  # noqa: E402
from scipy import stats  # noqa: E402

npl.style()
SEED = npl.seed_all()
DEV = npl.device()
N = {"charts": {}, "seed": SEED, "device": DEV}
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(HERE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RES, exist_ok=True)
SEEDS = [1, 2, 3]
t0 = time.time()


def seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


# ====================================================================== A. autoencoder for fraud detection
fr = npl.fraud().sort_values("Time").reset_index(drop=True)
cols = [f"V{i}" for i in range(1, 29)]
Xf = np.column_stack([fr[cols].values, np.log1p(fr["Amount"].values)]).astype(np.float32)
yf = fr["Class"].values
wf = fr["weight"].values
q60, q80 = np.quantile(fr["Time"], [0.6, 0.8])
itr = (fr.Time <= q60).values
iva = ((fr.Time > q60) & (fr.Time <= q80)).values
ite = (fr.Time > q80).values
leg_tr = itr & (yf == 0)
mu, sd = Xf[leg_tr].mean(0), Xf[leg_tr].std(0) + 1e-6
Xs = (Xf - mu) / sd
N["fraud"] = {"n_rows": int(len(fr)), "n_fraud": int(yf.sum()), "n_train_legit": int(leg_tr.sum()),
              "n_test": int(ite.sum()), "n_test_fraud": int(yf[ite].sum()),
              "test_fraud_rate_pct": round(100 * float((yf[ite] * wf[ite]).sum() / wf[ite].sum()), 3),
              "n_full_test": int(round(wf[ite].sum())), "d_in": int(Xf.shape[1]),
              "hours_train": round(q60 / 3600, 1), "hours_total": round(fr.Time.max() / 3600, 1)}


class AE(nn.Module):
    def __init__(self, d, k=8, h=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.ReLU(), nn.Linear(k, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(s, k=8, epochs=60, patience=6):
    seed(s)
    m = AE(Xs.shape[1], k)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    xt = torch.tensor(Xs[leg_tr])
    xv = torch.tensor(Xs[iva & (yf == 0)])
    best, state, wait, hist = 1e9, None, 0, []
    g = torch.Generator().manual_seed(s)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(xt), generator=g)
        tl = 0.0
        for b in range(0, len(perm), 256):
            j = perm[b:b + 256]
            loss = F.mse_loss(m(xt[j]), xt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss) * len(j)
        m.eval()
        with torch.no_grad():
            vl = float(F.mse_loss(m(xv), xv))
        hist.append((ep, tl / len(xt), vl))
        if vl < best - 1e-5:
            best, wait, state = vl, 0, {k_: v.clone() for k_, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        err = ((m(torch.tensor(Xs)) - torch.tensor(Xs)) ** 2).mean(1).numpy()
    return err, hist


def wap(y, s, mask):
    return float(average_precision_score(y[mask], s[mask], sample_weight=wf[mask]))


acache = os.path.join(CACHE, "ae.npz")
if os.path.exists(acache):
    za = np.load(acache, allow_pickle=True)
    ERR, AEH, BOT = list(za["ERR"]), za["AEH"].item(), za["BOT"].item()
else:
    ERR, AEH, BOT = [], {}, {}
    for s in SEEDS:
        e, h = train_ae(s)
        ERR.append(e)
        AEH[s] = h
        print("AE seed", s, "test PR-AUC", round(wap(yf, e, ite), 4))
    for k in [2, 4, 8, 16, 24]:
        BOT[k] = [wap(yf, train_ae(s, k=k)[0], ite) for s in SEEDS[:2]]
        print("bottleneck", k, BOT[k])
    np.savez_compressed(acache, ERR=np.array(ERR), AEH=np.array(AEH, dtype=object), BOT=np.array(BOT, dtype=object))
pca = PCA(n_components=8, random_state=SEED).fit(Xs[leg_tr])
err_pca = ((pca.inverse_transform(pca.transform(Xs)) - Xs) ** 2).mean(1)
logit = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(Xs[itr], yf[itr])
p_log = logit.predict_proba(Xs)[:, 1]
err_ae = np.mean(ERR, 0)
ap = [wap(yf, e, ite) for e in ERR]
N["ae"] = {"ap": round(100 * float(np.mean(ap)), 1), "ap_sd": round(100 * float(np.std(ap, ddof=1)), 1),
           "auc": round(100 * float(np.mean([roc_auc_score(yf[ite], e[ite], sample_weight=wf[ite]) for e in ERR])), 1),
           "ap_pca": round(100 * wap(yf, err_pca, ite), 1),
           "auc_pca": round(100 * float(roc_auc_score(yf[ite], err_pca[ite], sample_weight=wf[ite])), 1),
           "ap_logit": round(100 * wap(yf, p_log, ite), 1),
           "auc_logit": round(100 * float(roc_auc_score(yf[ite], p_log[ite], sample_weight=wf[ite])), 1),
           "k": 8, "params": int(sum(p.numel() for p in AE(29).parameters())),
           "pca_var_pct": round(100 * float(pca.explained_variance_ratio_.sum()), 1),
           "epochs_seed1": len(AEH[1])}
N["ae"]["base_ap"] = round(N["fraud"]["test_fraud_rate_pct"], 3)
# share of the frauds among the 0.1 % highest scores of the test day (weighted), and ratio of median errors
for nm, sc_ in [("ae", err_ae), ("pca", err_pca), ("logit", p_log)]:
    s_te, y_te, w_te = sc_[ite], yf[ite], wf[ite]
    o = np.argsort(-s_te)
    cw = np.cumsum(w_te[o])
    top = o[: int(np.searchsorted(cw, 0.001 * w_te.sum())) + 1]
    N["ae"][f"top01_frauds_{nm}"] = int(y_te[top].sum())
    N["ae"][f"top01_recall_{nm}"] = round(100 * float(y_te[top].sum() / y_te.sum()), 1)
N["ae"]["n_alerts_top01"] = int(round(0.001 * wf[ite].sum()))
N["ae"]["median_ratio"] = round(float(np.median(err_ae[ite & (yf == 1)]) / np.median(err_ae[ite & (yf == 0)])), 1)
N["ae"]["bottleneck"] = {str(k): round(100 * float(np.mean(v)), 1) for k, v in BOT.items()}
N["ae"]["best_k"] = int(max(BOT, key=lambda k: np.mean(BOT[k])))
pd.DataFrame({"y": yf[ite], "weight": wf[ite], "err_ae": np.round(err_ae[ite], 6), "err_pca": np.round(err_pca[ite], 6),
              "p_logit": np.round(p_log[ite], 6)}).to_csv(os.path.join(RES, "fraud_test_scores.csv"), index=False)
pd.DataFrame({"k": list(BOT), "ap": [float(np.mean(v)) for v in BOT.values()]}).to_csv(os.path.join(RES, "ae_bottleneck.csv"), index=False)
print("A done", round(time.time() - t0), "s", N["ae"])

# ====================================================================== B. VAE on Fashion-MNIST
import torchvision  # noqa: E402

ftr = torchvision.datasets.FashionMNIST(npl.vision_root(), train=True, download=True)
fte = torchvision.datasets.FashionMNIST(npl.vision_root(), train=False, download=True)
Xtr_img = (ftr.data.float() / 255).view(-1, 784)
Xte_img = (fte.data.float() / 255).view(-1, 784)
yte_img = fte.targets.numpy()
CL = {"en": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
      "ro": ["Tricou", "Pantaloni", "Pulover", "Rochie", "Palton", "Sandală", "Cămașă", "Pantof sport", "Geantă", "Gheată"]}


class VAE(nn.Module):
    def __init__(self, k=2, h=400):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(784, h), nn.ReLU(), nn.Linear(h, h // 2), nn.ReLU())
        self.mu, self.lv = nn.Linear(h // 2, k), nn.Linear(h // 2, k)
        self.d = nn.Sequential(nn.Linear(k, h // 2), nn.ReLU(), nn.Linear(h // 2, h), nn.ReLU(), nn.Linear(h, 784))

    def forward(self, x):
        h = self.e(x)
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.d(z), mu, lv


def vae_loss(logits, x, mu, lv, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp()) / len(x)
    return rec + beta * kl, rec, kl


def train_vae(s, k, beta=1.0, epochs=20):
    seed(s)
    m = VAE(k).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(s)
    hist = []
    xt = Xtr_img.to(DEV)
    xv = Xte_img.to(DEV)
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(xt), generator=g).to(DEV)
        for b in range(0, len(perm), 128):
            x = xt[perm[b:b + 128]]
            lg, mu, lv = m(x)
            loss, _, _ = vae_loss(lg, x, mu, lv, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            lg, mu, lv = m(xv)
            _, r, kl = vae_loss(lg, xv, mu, lv, beta)
            klj = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean(0)
        hist.append((ep, float(r), float(kl)))
    return m, hist, klj.cpu().numpy()


vcache = os.path.join(CACHE, "vae.pt")
if os.path.exists(vcache):
    VC = torch.load(vcache, weights_only=False)
else:
    VC = {}
    m2, h2, _ = train_vae(1, 2)
    VC["m2"] = m2.cpu().state_dict()
    VC["h2"] = h2
    VC["beta"] = {}
    for beta in [0.5, 1.0, 4.0]:
        m16, h16, klj = train_vae(1, 16, beta)
        VC["beta"][beta] = {"rec": h16[-1][1], "kl": h16[-1][2], "active": int((klj > 0.05).sum()), "hist": h16}
        if beta == 1.0:
            VC["m16"] = m16.cpu().state_dict()
        print("beta", beta, VC["beta"][beta]["rec"], VC["beta"][beta]["kl"], VC["beta"][beta]["active"])
    torch.save(VC, vcache)
m2 = VAE(2)
m2.load_state_dict(VC["m2"])
m2.eval()
m16 = VAE(16)
m16.load_state_dict(VC["m16"])
m16.eval()
with torch.no_grad():
    _, mu2, _ = m2(Xte_img)
    mu2 = m2.mu(m2.e(Xte_img)).numpy()
    seed(5)
    rec16 = torch.sigmoid(m16.d(m16.mu(m16.e(Xte_img[:10])))).numpy()
    samp16 = torch.sigmoid(m16.d(torch.randn(10, 16))).numpy()
# how well the 2-D latent separates the classes: k-nearest-neighbour accuracy on the latent means (5-fold on test)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
knn = float(np.mean(cross_val_score(KNeighborsClassifier(15), mu2, yte_img, cv=5)))
N["vae"] = {"knn_acc_2d": round(100 * knn, 1), "n_train": int(len(Xtr_img)), "n_test": int(len(Xte_img)),
            "rec_2d": round(VC["h2"][-1][1], 1), "kl_2d": round(VC["h2"][-1][2], 2),
            "beta": {{0.5: "b05", 1.0: "b1", 4.0: "b4"}[b]: {"rec": round(v["rec"], 1), "kl": round(v["kl"], 1), "active": v["active"]} for b, v in VC["beta"].items()},
            "epochs": len(VC["h2"]), "params_16": int(sum(p.numel() for p in VAE(16).parameters()))}
N["vae"]["elbo_16"] = round(-(VC["beta"][1.0]["rec"] + VC["beta"][1.0]["kl"]), 1)
pd.DataFrame({"z1": np.round(mu2[:, 0], 5), "z2": np.round(mu2[:, 1], 5), "y": yte_img}).to_csv(os.path.join(RES, "vae_latent.csv"), index=False)
pd.DataFrame(VC["beta"][1.0]["hist"], columns=["epoch", "rec", "kl"]).to_csv(os.path.join(RES, "vae_elbo.csv"), index=False)
pd.DataFrame(np.round(np.vstack([Xte_img[:10].numpy(), rec16, samp16]), 4)).assign(row=["real"] * 10 + ["reconstructed"] * 10 + ["generated"] * 10).to_csv(os.path.join(RES, "vae_samples.csv"), index=False)
print("B done", round(time.time() - t0), "s", N["vae"])

# ====================================================================== C. GAN on S&P 500 daily log returns
px = npl.markets("adjclose", ["SP500"])["SP500"].dropna()
end = (px.index.max() + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.Timedelta(days=1)
px = px.loc[:end]
r = np.log(px).diff().dropna()
rtr, rte = r.loc[:"2019-12-31"], r.loc["2020-01-01":]
s_tr = float(rtr.std())
ztr = (rtr.values / s_tr).astype(np.float32)


class G(nn.Module):
    def __init__(self, nz=8, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nz, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, 1))

    def forward(self, z):
        return self.net(z)


class D(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x)


def train_gan(s, steps=6000, bs=256):
    seed(s)
    g_, d_ = G(), D()
    og = torch.optim.Adam(g_.parameters(), lr=2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(d_.parameters(), lr=2e-4, betas=(0.5, 0.999))
    xr = torch.tensor(ztr).unsqueeze(1)
    hist = []
    gen = torch.Generator().manual_seed(s)
    for it in range(1, steps + 1):
        j = torch.randint(0, len(xr), (bs,), generator=gen)
        real = xr[j]
        fake = g_(torch.randn(bs, 8))
        ld = F.binary_cross_entropy_with_logits(d_(real), torch.ones(bs, 1)) + \
            F.binary_cross_entropy_with_logits(d_(fake.detach()), torch.zeros(bs, 1))
        od.zero_grad()
        ld.backward()
        od.step()
        lg = F.binary_cross_entropy_with_logits(d_(fake), torch.ones(bs, 1))   # non-saturating generator loss
        og.zero_grad()
        lg.backward()
        og.step()
        if it % 50 == 0:
            hist.append((it, float(ld), float(lg)))
    with torch.no_grad():
        torch.manual_seed(100 + s)
        samp = g_(torch.randn(200000, 8)).numpy().ravel()
    return samp, hist


gcache = os.path.join(CACHE, "gan.npz")
if os.path.exists(gcache):
    zg = np.load(gcache, allow_pickle=True)
    GS, GH = list(zg["GS"]), zg["GH"].item()
else:
    GS, GH = [], {}
    for s in SEEDS:
        samp, h = train_gan(s)
        GS.append(samp)
        GH[s] = h
        print("GAN seed", s, "kurt", round(float(stats.kurtosis(samp)), 2), "q01", round(float(np.quantile(samp, 0.01)), 3))
    np.savez_compressed(gcache, GS=np.array(GS), GH=np.array(GH, dtype=object))
real_k = float(stats.kurtosis(ztr))
gk = [float(stats.kurtosis(g_)) for g_ in GS]
q = 0.01
var_real = float(np.quantile(ztr, q)) * s_tr
var_gan = [float(np.quantile(g_, q)) * s_tr for g_ in GS]
var_norm = float(stats.norm.ppf(q, loc=rtr.mean(), scale=s_tr))
viol = {"hs": float(np.mean(rte.values < var_real)), "gan": float(np.mean([np.mean(rte.values < v) for v in var_gan])),
        "norm": float(np.mean(rte.values < var_norm))}
ks_gan = [float(stats.ks_2samp(ztr, g_[:20000]).statistic) for g_ in GS]
ks_norm = float(stats.kstest(ztr, "norm", args=(ztr.mean(), ztr.std())).statistic)


def acf_abs(x, lags=20):
    a = np.abs(x) - np.abs(x).mean()
    return np.array([np.sum(a[l:] * a[:-l]) / np.sum(a * a) for l in range(1, lags + 1)])


acf_real = acf_abs(ztr)
acf_gan = acf_abs(GS[0][:len(ztr)])
N["gan"] = {"n_train": int(len(rtr)), "n_test": int(len(rte)), "train_first": str(rtr.index.min().date()),
            "test_first": str(rte.index.min().date()), "test_last": str(rte.index.max().date()),
            "kurt_real": round(real_k, 2), "kurt_gan": round(float(np.mean(gk)), 2), "kurt_gan_sd": round(float(np.std(gk, ddof=1)), 2),
            "var_real_pct": round(100 * var_real, 2), "var_gan_pct": round(100 * float(np.mean(var_gan)), 2),
            "var_gan_sd": round(100 * float(np.std(var_gan, ddof=1)), 2), "var_norm_pct": round(100 * var_norm, 2),
            "viol_hs_pct": round(100 * viol["hs"], 2), "viol_gan_pct": round(100 * viol["gan"], 2), "viol_norm_pct": round(100 * viol["norm"], 2),
            "ks_gan": round(float(np.mean(ks_gan)), 3), "ks_norm": round(ks_norm, 3),
            "acf1_real": round(float(acf_real[0]), 2), "acf1_gan": round(float(acf_gan[0]), 2),
            "acf20_real": round(float(acf_real[-1]), 2),
            "params_g": int(sum(p.numel() for p in G().parameters())), "params_d": int(sum(p.numel() for p in D().parameters())),
            "steps": 6000, "sd_train_pct": round(100 * s_tr, 2)}
N["gan"]["n_viol_expected"] = int(round(q * len(rte)))
pd.DataFrame({"lag": np.arange(1, 21), "real": np.round(acf_real, 5), "gan": np.round(acf_gan, 5)}).to_csv(os.path.join(RES, "gan_acf.csv"), index=False)
pd.DataFrame({"gan_seed1": np.round(GS[0][:50000] * s_tr, 7)}).to_csv(os.path.join(RES, "gan_samples.csv"), index=False)
pd.DataFrame(GH[1], columns=["iteration", "loss_d", "loss_g"]).round(5).to_csv(os.path.join(RES, "gan_losses.csv"), index=False)
print("C done", round(time.time() - t0), "s", N["gan"])

# ====================================================================== D. diffusion on yield curves
yc = npl.yield_curve()
endy = (yc.index.max() + pd.Timedelta(days=1)).to_period("M").to_timestamp() - pd.Timedelta(days=1)
yc = yc.loc[:endy].dropna()
MAT = ["y_3M", "y_6M", "y_1Y", "y_2Y", "y_3Y", "y_5Y", "y_7Y", "y_10Y", "y_15Y", "y_20Y", "y_30Y"]
MAT_Y = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
Y = yc[MAT].values.astype(np.float32)
ym, ys = Y.mean(0), Y.std(0)
Yz = (Y - ym) / ys
TT = 200
betas = torch.linspace(1e-4, 0.02, TT)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)


class Denoiser(nn.Module):
    def __init__(self, d=11, h=256, te=32):
        super().__init__()
        self.te = te
        self.net = nn.Sequential(nn.Linear(d + te, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))

    def temb(self, t):
        half = self.te // 2
        f = torch.exp(-math.log(1000) * torch.arange(half) / half)
        a = t.float().unsqueeze(1) * f.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], 1))


def train_ddpm(s, steps=12000, bs=512):
    seed(s)
    m = Denoiser()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    x0all = torch.tensor(Yz)
    gen = torch.Generator().manual_seed(s)
    hist = []
    for it in range(1, steps + 1):
        x0 = x0all[torch.randint(0, len(x0all), (bs,), generator=gen)]
        t = torch.randint(0, TT, (bs,), generator=gen)
        eps = torch.randn(x0.shape, generator=gen)
        ab = abar[t].unsqueeze(1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        loss = F.mse_loss(m(xt, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if it % 100 == 0:
            hist.append((it, float(loss)))
    return m, hist


@torch.no_grad()
def sample_ddpm(m, n, s):
    torch.manual_seed(1000 + s)
    x = torch.randn(n, 11)
    for t in reversed(range(TT)):
        tt = torch.full((n,), t)
        eps = m(x, tt)
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
    return x.numpy()


class VAEy(nn.Module):
    def __init__(self, k=2, h=64):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(11, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu, self.lv = nn.Linear(h, k), nn.Linear(h, k)
        self.d = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 11))

    def forward(self, x):
        h = self.e(x)
        mu, lv = self.mu(h), self.lv(h)
        return self.d(mu + torch.exp(0.5 * lv) * torch.randn_like(mu)), mu, lv


def train_vaey(s, epochs=300):
    seed(s)
    m = VAEy()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    x = torch.tensor(Yz)
    gen = torch.Generator().manual_seed(s)
    for ep in range(epochs):
        perm = torch.randperm(len(x), generator=gen)
        for b in range(0, len(x), 256):
            xb = x[perm[b:b + 256]]
            rec, mu, lv = m(xb)
            # Gaussian decoder with unit variance: squared error / 2 per coordinate
            loss = 0.5 * ((rec - xb) ** 2).sum(1).mean() - 0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp(), 1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


dcache = os.path.join(CACHE, "ddpm.npz")
if os.path.exists(dcache):
    zd = np.load(dcache, allow_pickle=True)
    DS, DH, VS, VZ = list(zd["DS"]), zd["DH"].item(), list(zd["VS"]), zd["VZ"]
else:
    DS, DH, VS = [], {}, []
    for s in SEEDS:
        s0 = time.time()
        m, h = train_ddpm(s)
        DS.append(sample_ddpm(m, 5000, s))
        DH[s] = h
        print("DDPM seed", s, round(time.time() - s0), "s")
        mv = train_vaey(s)
        mv.eval()
        with torch.no_grad():
            torch.manual_seed(2000 + s)
            VS.append(mv.d(torch.randn(5000, 2)).numpy())
            if s == 1:
                VZ = mv.mu(mv.e(torch.tensor(Yz))).numpy()
    np.savez_compressed(dcache, DS=np.array(DS), DH=np.array(DH, dtype=object), VS=np.array(VS), VZ=VZ)
rngg = np.random.default_rng(SEED)
GA = [rngg.multivariate_normal(np.zeros(11), np.cov(Yz.T), 5000) for _ in SEEDS]


def feat(Z):
    Yy = Z * ys + ym
    return {"y10": Yy[:, 7], "slope": Yy[:, 7] - Yy[:, 0], "curv": 2 * Yy[:, 5] - Yy[:, 0] - Yy[:, 10], "y2": Yy[:, 3]}


FR = feat(Yz)
N["yc"] = {"n_days": int(len(Yz)), "first": str(yc.index.min().date()), "last": str(yc.index.max().date()),
           "pct_inverted_real": round(100 * float(np.mean(FR["slope"] < 0)), 1),
           "pct_neg10_real": round(100 * float(np.mean(FR["y10"] < 0)), 1),
           "y10_min": round(float(FR["y10"].min()), 2), "y10_max": round(float(FR["y10"].max()), 2),
           "T": TT, "steps": 12000, "params": int(sum(p.numel() for p in Denoiser().parameters()))}
for nm, SS in [("ddpm", DS), ("gauss", GA), ("vae", VS)]:
    w10 = [stats.wasserstein_distance(FR["y10"], feat(z)["y10"]) for z in SS]
    wsl = [stats.wasserstein_distance(FR["slope"], feat(z)["slope"]) for z in SS]
    inv = [100 * float(np.mean(feat(z)["slope"] < 0)) for z in SS]
    neg = [100 * float(np.mean(feat(z)["y10"] < 0)) for z in SS]
    N["yc"][nm] = {"w_y10": round(float(np.mean(w10)), 3), "w_y10_sd": round(float(np.std(w10, ddof=1)), 3),
                   "w_slope": round(float(np.mean(wsl)), 3), "pct_inverted": round(float(np.mean(inv)), 1),
                   "pct_neg10": round(float(np.mean(neg)), 1)}
# correlation of the VAE latent means with level and slope
lev = Y.mean(1)
slo = FR["slope"]
c = np.corrcoef(np.column_stack([VZ, lev, slo]).T)
N["yc"]["vae_corr"] = {"z1_level": round(float(c[0, 2]), 2), "z1_slope": round(float(c[0, 3]), 2),
                       "z2_level": round(float(c[1, 2]), 2), "z2_slope": round(float(c[1, 3]), 2)}
N["yc"]["vae_corr"]["max_level"] = round(float(max(abs(c[0, 2]), abs(c[1, 2]))), 2)
N["yc"]["vae_corr"]["max_slope"] = round(float(max(abs(c[0, 3]), abs(c[1, 3]))), 2)
pd.DataFrame({"y10_real": np.round(FR["y10"], 4)}).to_csv(os.path.join(RES, "yc_real_y10.csv"), index=False)
pd.DataFrame({"ddpm": np.round(feat(DS[0])["y10"], 4), "gauss": np.round(feat(GA[0])["y10"], 4),
              "vae": np.round(feat(VS[0])["y10"], 4)}).to_csv(os.path.join(RES, "yc_generated_y10.csv"), index=False)
pd.DataFrame({"date": yc.index.strftime("%Y-%m-%d"), "z1": np.round(VZ[:, 0], 4), "z2": np.round(VZ[:, 1], 4),
              "level": np.round(lev, 4), "slope": np.round(slo, 4)}).to_csv(os.path.join(RES, "yc_vae_latent.csv"), index=False)
print("D done", round(time.time() - t0), "s", N["yc"])

# ====================================================================== charts
PER_F = {"en": "two days of September 2013; test = last 20\\,\\% of the time span", "ro": "două zile din septembrie 2013; test = ultimele 20\\,\\% din interval"}
for lang in ("en", "ro"):
    ro = lang == "ro"
    # A1 reconstruction error
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    le = np.log10(err_ae[ite])
    bins = np.linspace(np.percentile(le, 0.1), le.max(), 60)
    ax.hist(le[yf[ite] == 0], bins=bins, weights=wf[ite][yf[ite] == 0], density=True, alpha=0.5, color=npl.PAL["negative"],
            label="tranzacții legitime" if ro else "legitimate transactions")
    ax.hist(le[yf[ite] == 1], bins=bins, density=True, alpha=0.6, color=npl.PAL["positive"], label="fraude" if ro else "frauds")
    ax.set_xlabel("log$_{10}$ eroarea de reconstrucție" if ro else "log$_{10}$ reconstruction error")
    ax.set_ylabel("densitate" if ro else "density")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "ae_error_dist", also_x=[ax])

    # A2 precision-recall
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for sc_, c_, ls, lab in [(err_ae, npl.PAL["model"], "-", "autoencoder"),
                             (err_pca, npl.PAL["benchmark"], "--", "PCA, 8 componente" if ro else "PCA, 8 components"),
                             (p_log, npl.PAL["alt"], ":", "regresie logistică (cu etichete)" if ro else "logistic regression (uses labels)")]:
        p_, r_, _ = precision_recall_curve(yf[ite], sc_[ite], sample_weight=wf[ite])
        ax.plot(r_, p_, ls, color=c_, label=lab)
    ax.set_xlabel("recall (fraude detectate)" if ro else "recall (frauds detected)")
    ax.set_ylabel("precizie" if ro else "precision")
    ax.set_ylim(0, 1.02)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "ae_pr", also_x=[ax])

    # A3 bottleneck
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ks = sorted(BOT)
    ax.plot(ks, [100 * np.mean(BOT[k]) for k in ks], "o-", color=npl.PAL["model"], label="autoencoder")
    ax.axhline(N["ae"]["ap_pca"], color=npl.PAL["benchmark"], ls="--", label="PCA, 8 componente" if ro else "PCA, 8 components")
    ax.set_xticks(ks)
    ax.set_xlabel("dimensiunea stratului îngust (bottleneck) $k$" if ro else "bottleneck size $k$")
    ax.set_ylabel("PR-AUC, test (%)")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "ae_bottleneck")

    # B1 latent space
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for c in range(10):
        mm = yte_img == c
        ax.scatter(mu2[mm, 0], mu2[mm, 1], s=2, alpha=0.5, color=plt.cm.tab10(c), lw=0, label=CL[lang][c])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    h_, l_ = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    hh = [Line2D([], [], marker="o", ls="", color=plt.cm.tab10(c), ms=5) for c in range(10)]
    npl.legend_bottom(fig, hh, CL[lang], ncol=5)
    npl.out(fig, HERE, lang, "vae_latent", bottom=0.22)

    # B2 images: real, reconstructed, generated
    fig, axs = plt.subplots(3, 10, figsize=(6.4, 2.2))
    for j in range(10):
        for i, img in enumerate([Xte_img[j].numpy(), rec16[j], samp16[j]]):
            axs[i, j].imshow(img.reshape(28, 28), cmap="Greys")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            for sp_ in axs[i, j].spines.values():
                sp_.set_visible(False)
    rl = ["reale", "reconstruite", "generate"] if ro else ["real", "reconstructed", "generated"]
    for i in range(3):
        axs[i, 0].set_ylabel(rl[i], fontsize=8.5, rotation=0, ha="right", va="center")
    npl.out(fig, HERE, lang, "vae_samples", bottom=0.0)

    # B3 ELBO terms by epoch
    hb = np.array(VC["beta"][1.0]["hist"])
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3))
    axs[0].plot(hb[:, 0], hb[:, 1], "o-", ms=3, color=npl.PAL["data"], label="eroarea de reconstrucție" if ro else "reconstruction error")
    axs[1].plot(hb[:, 0], hb[:, 2], "s-", ms=3, color=npl.PAL["model"], label="divergența KL" if ro else "KL divergence")
    for a_ in axs:
        a_.set_xlabel("epoca" if ro else "epoch")
    axs[0].set_ylabel("nats / imagine" if ro else "nats per image")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "vae_elbo")

    # C1 density of returns (log scale)
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    b = np.linspace(-8, 8, 121)
    ax.hist(ztr, bins=b, density=True, histtype="stepfilled", alpha=0.35, color=npl.PAL["data"],
            label="S&P 500, 2005-2019")
    ax.hist(GS[0], bins=b, density=True, histtype="step", lw=1.5, color=npl.PAL["model"], label="GAN")
    xx = np.linspace(-8, 8, 400)
    ax.plot(xx, stats.norm.pdf(xx), "--", color=npl.PAL["benchmark"], label="normală" if ro else "Gaussian")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("randamentul zilnic standardizat" if ro else "standardised daily return")
    ax.set_ylabel("densitate (log)" if ro else "density (log)")
    ax._keep_y = True
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "gan_density", also_x=[ax])

    # C2 losses
    hg = np.array(GH[1])
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.plot(hg[:, 0], hg[:, 1], color=npl.PAL["data"], lw=1, label="discriminator")
    ax.plot(hg[:, 0], hg[:, 2], color=npl.PAL["model"], lw=1, label="generator")
    ax.axhline(2 * math.log(2), color=npl.MUTED, ls="--", lw=1, label="2 log 2 (echilibru)" if ro else "2 log 2 (equilibrium)")
    ax.axhline(math.log(2), color=npl.MUTED, ls=":", lw=1, label="log 2")
    ax.set_xlabel("iterația" if ro else "iteration")
    ax.set_ylabel("entropia încrucișată" if ro else "cross-entropy")
    npl.legend_bottom(fig, ncol=4)
    npl.out(fig, HERE, lang, "gan_losses")

    # C3 ACF of absolute returns
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    lags = np.arange(1, 21)
    ax.bar(lags - 0.2, acf_real, 0.4, color=npl.PAL["data"], label="S&P 500, 2005-2019")
    ax.bar(lags + 0.2, acf_gan, 0.4, color=npl.PAL["model"], label="GAN")
    ax.axhline(0, color=npl.INK, lw=0.6)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("decalajul (zile)" if ro else "lag (days)")
    ax.set_ylabel("autocorelația lui |r|" if ro else "autocorrelation of |r|")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "gan_acf")

    # D1 forward diffusion of a real curve (the last curve of the sample)
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    x0 = torch.tensor(Yz[-1])
    eps = torch.tensor(np.random.default_rng(7).standard_normal(11), dtype=torch.float32)
    for t, c_ in zip([0, 20, 60, 120, 199], [npl.PAL["data"], npl.PAL["benchmark"], npl.PAL["alt"], npl.PAL["c5"], npl.PAL["positive"]]):
        xt = abar[t].sqrt() * x0 + (1 - abar[t]).sqrt() * eps
        ax.plot(MAT_Y, (xt.numpy() * ys + ym), "o-", ms=3, color=c_, label=f"t = {t}" if t else ("t = 0 (curba reală)" if ro else "t = 0 (real curve)"))
    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 2, 5, 10, 30])
    ax.set_xticklabels(["3L" if ro else "3M", "1A" if ro else "1Y", "2A" if ro else "2Y", "5A" if ro else "5Y", "10A" if ro else "10Y", "30A" if ro else "30Y"])
    ax.minorticks_off()
    ax.set_xlabel("scadența" if ro else "maturity")
    ax.set_ylabel("randament (%)" if ro else "yield (%)")
    npl.legend_bottom(fig, ncol=5)
    npl.out(fig, HERE, lang, "diffusion_forward")
    if lang == "en":
        N["yc"]["forward_date"] = str(yc.index[-1].date())

    # D2 density of the 10-year yield: real vs generated
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    b = np.linspace(-2, 6, 81)
    ax.hist(FR["y10"], bins=b, density=True, histtype="stepfilled", alpha=0.35, color=npl.PAL["data"], label="reale" if ro else "real")
    ax.hist(feat(DS[0])["y10"], bins=b, density=True, histtype="step", lw=1.6, color=npl.PAL["model"], label="difuzie (DDPM)" if ro else "diffusion (DDPM)")
    ax.hist(feat(GA[0])["y10"], bins=b, density=True, histtype="step", lw=1.2, ls="--", color=npl.PAL["benchmark"], label="normală multivariată" if ro else "multivariate Gaussian")
    ax.hist(feat(VS[0])["y10"], bins=b, density=True, histtype="step", lw=1.2, ls=":", color=npl.PAL["alt"], label="VAE")
    ax.set_xlabel("randamentul la 10 ani (%)" if ro else "10-year yield (%)")
    ax.set_ylabel("densitate" if ro else "density")
    npl.legend_bottom(fig, ncol=4)
    npl.out(fig, HERE, lang, "yc_density", also_x=[ax])

    # D3 level-slope scatter: real vs diffusion vs Gaussian
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.4), sharex=True, sharey=True)
    for ax, (Z, c_, lab) in zip(axs, [(Yz, npl.PAL["data"], "reale" if ro else "real"), (DS[0], npl.PAL["model"], "DDPM"),
                                     (GA[0], npl.PAL["benchmark"], "normală" if ro else "Gaussian")]):
        f_ = feat(Z)
        ii = np.random.default_rng(3).choice(len(Z), min(2500, len(Z)), replace=False)
        ax.scatter(f_["y10"][ii], f_["slope"][ii], s=2, alpha=0.4, color=c_, lw=0, label=lab)
        ax.axhline(0, color=npl.MUTED, lw=0.6)
        ax.set_xlabel("10A (%)" if ro else "10Y (%)")
    axs[0].set_ylabel("panta 10A−3L (pp)" if ro else "slope 10Y−3M (pp)")
    from matplotlib.lines import Line2D
    hh = [Line2D([], [], marker="o", ls="", color=c_, ms=5) for c_ in (npl.PAL["data"], npl.PAL["model"], npl.PAL["benchmark"])]
    npl.legend_bottom(fig, hh, ["reale" if ro else "real", "DDPM", "normală multivariată" if ro else "multivariate Gaussian"], ncol=3)
    npl.out(fig, HERE, lang, "yc_scatter", also_x=list(axs))

    # D4 VAE latent coordinates over time
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.plot(yc.index, VZ[:, 0], color=npl.PAL["model"], lw=0.9, label="$z_1$")
    ax.plot(yc.index, VZ[:, 1], color=npl.PAL["data"], lw=0.9, label="$z_2$")
    ax.set_ylabel("media latentă" if ro else "latent mean")
    ax.set_xlim(yc.index.min(), yc.index.max())
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "yc_latent_time")

# chart metadata
for nm in ["ae_error_dist", "ae_pr", "ae_bottleneck"]:
    npl.chart_meta(N, nm, "fraud", "own estimates; " + PER_F["en"], "estimări proprii; " + PER_F["ro"])
for nm in ["vae_latent", "vae_samples", "vae_elbo"]:
    npl.chart_meta(N, nm, "fashion_mnist", "own estimates; 60,000 training and 10,000 test images",
                   "estimări proprii; 60.000 de imagini de antrenare și 10.000 de test")
pe_en = npl.period(rtr.index.min(), rtr.index.max(), "en") + ", trading days"
pe_ro = npl.period(rtr.index.min(), rtr.index.max(), "ro") + ", zile de tranzacționare"
for nm in ["gan_density", "gan_losses", "gan_acf"]:
    npl.chart_meta(N, nm, "yahoo", "own estimates; S\\&P 500 " + pe_en, "estimări proprii; S\\&P 500 " + pe_ro)
yp_en = npl.period(yc.index.min(), yc.index.max(), "en") + ", business days"
yp_ro = npl.period(yc.index.min(), yc.index.max(), "ro") + ", zile lucrătoare"
npl.chart_meta(N, "diffusion_forward", "yield", "curve of " + npl.period(yc.index[-1], yc.index[-1], "en").split(" to ")[0] + "; noise schedule of the lecture (mechanism)",
               "curba din " + npl.period(yc.index[-1], yc.index[-1], "ro").split(" - ")[0] + "; programul de zgomot din curs (mecanism)")
for nm in ["yc_density", "yc_scatter", "yc_latent_time"]:
    npl.chart_meta(N, nm, "yield", "own estimates; " + yp_en, "estimări proprii; " + yp_ro)
N["runtime_min"] = round((time.time() - t0) / 60, 1)
with open(os.path.join(HERE, "L8_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print("wrote L8_numbers.json and", len(N["charts"]), "charts x 2 languages in", N["runtime_min"], "min")
