"""
analysis_L6.py -- Lecture 6 "NLP foundations: text representation and word embeddings".

Real data (data/clean, see data/SOURCES.md): Financial PhraseBank (Malo et al., 2014), Twitter financial news sentiment and
topic (Hugging Face, zeroshot), speeches of the European Central Bank (ECB) 2022-2026 (corpus for our own embeddings).
  1. tokens, Zipf's law and the vocabulary of central-bank speeches;
  2. our own word2vec (skip-gram with negative sampling) and GloVe, trained in PyTorch on the ECB speeches;
  3. sentiment classification: majority, bag of words, TF-IDF (+ logistic regression, + MLP), mean embeddings,
     an embedding-bag network (random or word2vec initialisation); 5 random splits (PhraseBank), 3 seeds (tweets);
  4. topic classification of finance tweets (20 classes).
Writes lectures/L6/L6_numbers.json and charts/en|ro/*.pdf; trained results cached in lectures/L6/cache/*.pkl.
Run:  python3 lectures/L6/analysis_L6.py
"""
import json
import os
import pickle
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

torch.set_num_threads(4)
npl.style()
SEED = npl.seed_all()
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
N = {"charts": {}, "seed": SEED}
LANGS = ("en", "ro")


def L(en, ro, lang):
    return en if lang == "en" else ro


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
    print(f"  {name}: {r['_seconds']} s")
    return r


TOK = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*")


def tok(s):
    """Lower-case word tokens; every number becomes the token <num>; URLs and @mentions are dropped."""
    s = re.sub(r"https?://\S+|@\w+", " ", str(s).lower())
    return ["<num>" if t[0].isdigit() else t for t in TOK.findall(s)]


# =====================================================================================================================
# 0. Data
# =====================================================================================================================
pb = npl.phrasebank(50)
tw = npl.tweets_sentiment()
tp = npl.tweets_topic()
sp = npl.ecb_speeches()
N["data"] = {"pb_n": len(pb), "pb_pos": int((pb.label == "positive").sum()), "pb_neu": int((pb.label == "neutral").sum()),
             "pb_neg": int((pb.label == "negative").sum()), "pb_agree100": int((pb.min_agreement_pct == 100).sum()),
             "tw_train": int((tw.split == "train").sum()), "tw_valid": int((tw.split == "valid").sum()),
             "tp_train": int((tp.split == "train").sum()), "tp_valid": int((tp.split == "valid").sum()),
             "tp_classes": int(tp.label.nunique()),
             "sp_n": len(sp), "sp_first": str(sp.date.min().date()), "sp_last": str(sp.date.max().date())}
N["data"]["pb_neu_pct"] = round(100 * N["data"]["pb_neu"] / N["data"]["pb_n"], 1)
N["data"]["tw_neu_pct"] = round(100 * float((tw[tw.split == "valid"].label_name == "neutral").mean()), 1)
P_PB = ("4,838 sentences of financial news, 5-8 annotators each", "4.838 de fraze din știri financiare, 5-8 adnotatori pentru fiecare")
P_SP = (npl.period(sp.date.min(), sp.date.max(), "en") + ", " + f"{len(sp)} speeches",
        npl.period(sp.date.min(), sp.date.max(), "ro") + ", " + f"{len(sp)} de discursuri")

# example of the pipeline on one real sentence
ok_ = pb.sentence.str.contains("operating profit rose") & pb.sentence.map(lambda z: z.isascii()) & (pb.min_agreement_pct == 100)
i_ex = int(np.argmax(ok_.values))
ex = pb.sentence.iloc[i_ex]
N["example"] = {"sentence": ex, "tokens": " | ".join(tok(ex)[:14]), "n_tokens": len(tok(ex)), "label": pb.label.iloc[i_ex]}

# =====================================================================================================================
# 1. Corpus statistics: ECB speeches (Zipf), vocabulary
# =====================================================================================================================
docs = [tok(t) for t in sp.contents]
cnt = Counter(w for d in docs for w in d)
n_tok = sum(cnt.values())
freq = np.array(sorted(cnt.values(), reverse=True), dtype=float)
rk = np.arange(1, len(freq) + 1)
sel = (rk >= 10) & (rk <= 10000)
slope, icpt = np.polyfit(np.log(rk[sel]), np.log(freq[sel]), 1)
top10_share = freq[:10].sum() / n_tok
N["corpus"] = {"n_tokens": int(n_tok), "n_types": len(cnt), "zipf_slope": round(float(slope), 3),
               "top10_share_pct": round(100 * float(top10_share), 1), "hapax": int(sum(1 for v in cnt.values() if v == 1)),
               "top": ", ".join(w for w, _ in cnt.most_common(8)),
               "types_min5": int(sum(1 for v in cnt.values() if v >= 5))}
N["corpus"]["hapax_pct"] = round(100 * N["corpus"]["hapax"] / N["corpus"]["n_types"], 1)
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.loglog(rk, freq, color=npl.PAL["data"], lw=1.4, label=L("word frequency", "frecvența cuvîntului", lang))
    ax.loglog(rk[sel], np.exp(icpt) * rk[sel] ** slope, color=npl.PAL["model"], lw=1.2, ls="--",
              label=L(f"fitted power law, slope {slope:.2f}", f"lege putere estimată, panta {slope:.2f}".replace(".", ","), lang))
    ax.set_xlabel(L("Rank of the word", "Rangul cuvîntului", lang))
    ax.set_ylabel(L("Frequency", "Frecvența", lang))
    ax._keep_y = True
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "zipf")
npl.chart_meta(N, "zipf", "speeches", *P_SP)

# =====================================================================================================================
# 2. Our own embeddings on the ECB speeches: word2vec (SGNS) and GloVe
# =====================================================================================================================
MIN_COUNT, DIM, WIN = 5, 100, 5
vocab = [w for w, c in cnt.most_common() if c >= MIN_COUNT]
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
counts = np.array([cnt[w] for w in vocab], dtype=float)
corpus_ids = [np.array([w2i[w] for w in d if w in w2i]) for d in docs]
N["w2v"] = {"V": V, "dim": DIM, "win": WIN, "neg": 5, "min_count": MIN_COUNT, "t": 1e-4}


def train_sgns(epochs=5, neg=5, t=1e-4, bs=4096, lr=0.003, seed=0):
    rng = np.random.RandomState(SEED + seed)
    torch.manual_seed(SEED + seed)
    f = counts / counts.sum()
    keep = np.minimum(1.0, np.sqrt(t / f) + t / f)               # subsampling of frequent words (Mikolov et al., 2013)
    noise = torch.tensor(counts ** 0.75 / (counts ** 0.75).sum(), dtype=torch.float)
    Win = nn.Embedding(V, DIM)
    Wout = nn.Embedding(V, DIM)
    nn.init.uniform_(Win.weight, -0.5 / DIM, 0.5 / DIM)
    nn.init.zeros_(Wout.weight)
    opt = torch.optim.SparseAdam(list(Win.parameters()) + list(Wout.parameters()), lr=lr) if False else \
        torch.optim.Adam(list(Win.parameters()) + list(Wout.parameters()), lr=lr)
    hist = []
    for ep in range(epochs):
        cen, ctx = [], []
        for d in corpus_ids:
            d = d[rng.rand(len(d)) < keep[d]]
            for off in range(1, WIN + 1):
                b = rng.randint(1, WIN + 1, len(d))               # dynamic window: offset used with probability b >= off
                ok = b[:-off] >= off if off < len(d) else np.zeros(0, bool)
                if off < len(d):
                    cen.append(d[:-off][ok]); ctx.append(d[off:][ok])
                    cen.append(d[off:][ok]); ctx.append(d[:-off][ok])
        cen = torch.tensor(np.concatenate(cen)); ctx = torch.tensor(np.concatenate(ctx))
        perm = torch.randperm(len(cen))
        tot, nb = 0.0, 0
        for j in range(0, len(perm), bs):
            b = perm[j:j + bs]
            c, o = cen[b], ctx[b]
            ng = torch.multinomial(noise, len(b) * neg, replacement=True).view(len(b), neg)
            vc = Win(c)
            pos = F.logsigmoid((vc * Wout(o)).sum(1))
            negs = F.logsigmoid(-(Wout(ng) @ vc.unsqueeze(2)).squeeze(2)).sum(1)
            loss = -(pos + negs).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        hist.append(tot / nb)
        print(f"    sgns epoch {ep + 1}: {tot / nb:.4f}  pairs {len(cen):,}")
    return {"E": Win.weight.detach().numpy().copy(), "hist": hist, "pairs": int(len(cen))}


def train_glove(epochs=25, xmax=100.0, alpha=0.75, win=10, bs=65536, lr=0.05, seed=0):
    torch.manual_seed(SEED + seed)
    co = Counter()
    for d in corpus_ids:
        for off in range(1, win + 1):
            if off >= len(d):
                break
            a, b = d[:-off], d[off:]
            for i_, j_ in zip(a.tolist(), b.tolist()):
                co[(i_, j_)] += 1.0 / off
                co[(j_, i_)] += 1.0 / off
    keys = np.array(list(co.keys()))
    X = torch.tensor(np.array(list(co.values())), dtype=torch.float)
    I, J = torch.tensor(keys[:, 0]), torch.tensor(keys[:, 1])
    Wm, Wc = nn.Embedding(V, DIM), nn.Embedding(V, DIM)
    bm, bc = nn.Embedding(V, 1), nn.Embedding(V, 1)
    for e in (Wm, Wc):
        nn.init.uniform_(e.weight, -0.5 / DIM, 0.5 / DIM)
    nn.init.zeros_(bm.weight); nn.init.zeros_(bc.weight)
    params = [p for m in (Wm, Wc, bm, bc) for p in m.parameters()]
    opt = torch.optim.Adagrad(params, lr=lr)
    fw = torch.clamp((X / xmax) ** alpha, max=1.0)
    lx = torch.log(X)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for j in range(0, len(perm), bs):
            b = perm[j:j + bs]
            pred = (Wm(I[b]) * Wc(J[b])).sum(1) + bm(I[b])[:, 0] + bc(J[b])[:, 0]
            loss = (fw[b] * (pred - lx[b]) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        hist.append(tot / len(X))
        if ep % 5 == 4:
            print(f"    glove epoch {ep + 1}: {tot / len(X):.4f}")
    return {"E": (Wm.weight + Wc.weight).detach().numpy().copy(), "hist": hist, "nnz": int(len(X)),
            "x": X.numpy().copy()}


W2V = cached("w2v", train_sgns)
GLV = cached("glove", train_glove)
N["w2v"]["pairs"] = W2V["pairs"]
N["w2v"]["loss_first"] = round(W2V["hist"][0], 3)
N["w2v"]["loss_last"] = round(W2V["hist"][-1], 3)
N["w2v"]["epochs"] = len(W2V["hist"])
N["w2v"]["seconds"] = W2V["_seconds"]
N["glove"] = {"nnz": GLV["nnz"], "epochs": len(GLV["hist"]), "loss_first": round(GLV["hist"][0], 4),
              "loss_last": round(GLV["hist"][-1], 4), "seconds": GLV["_seconds"], "xmax": 100, "alpha": 0.75, "win": 10,
              "share_below_xmax": round(100 * float((GLV["x"] < 100).mean()), 2),
              "density_pct": round(100 * GLV["nnz"] / V ** 2, 3)}


def unit(E):
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)


Ew, Eg = unit(W2V["E"]), unit(GLV["E"])
PROBES = ["inflation", "energy", "banks", "climate", "digital", "wages", "ukraine", "rates"]
PROBES = [w for w in PROBES if w in w2i]
nbrs = {}
for name, E in [("w2v", Ew), ("glove", Eg)]:
    nbrs[name] = {}
    for w in PROBES:
        s = E @ E[w2i[w]]
        s[w2i[w]] = -1
        top = np.argsort(-s)[:5]
        nbrs[name][w] = [(vocab[i], round(float(s[i]), 2)) for i in top]
N["nbrs"] = {name: {w: ", ".join(x for x, _ in v) for w, v in d.items()} for name, d in nbrs.items()}
N["nbrs_sim"] = {name: {w: v[0][1] for w, v in d.items()} for name, d in nbrs.items()}
N["probes"] = PROBES
overlap = [len(set(x for x, _ in nbrs["w2v"][w]) & set(x for x, _ in nbrs["glove"][w])) for w in PROBES]
N["nbrs_overlap"] = round(float(np.mean(overlap)), 2)


def cos(a, b, E=Ew):
    return float(E[w2i[a]] @ E[w2i[b]])


PAIRS = [("inflation", "prices"), ("inflation", "climate"), ("banks", "lending"), ("energy", "gas"), ("wages", "labour")]
N["cos"] = {f"{a}_{b}": round(cos(a, b), 2) for a, b in PAIRS if a in w2i and b in w2i}

# analogy test on the corpus: a : b :: c : ?
ANALOGIES = [("germany", "german", "france"), ("italy", "italian", "spain"), ("banks", "bank", "households"),
             ("rise", "rising", "fall"), ("higher", "lower", "increase")]
an = {}
for a, b, c in ANALOGIES:
    if all(w in w2i for w in (a, b, c)):
        v = Ew[w2i[b]] - Ew[w2i[a]] + Ew[w2i[c]]
        s = Ew @ (v / np.linalg.norm(v))
        for w in (a, b, c):
            s[w2i[w]] = -1
        an[f"{a}:{b}::{c}"] = vocab[int(np.argmax(s))]
N["analogy"] = an

# chart: 2-D principal components of word groups
GROUPS = {"monetary": ["inflation", "prices", "rates", "interest", "monetary", "policy", "tightening", "price"],
          "energy": ["energy", "gas", "oil", "electricity", "fossil", "renewable", "fuel", "commodity"],
          "banking": ["banks", "lending", "credit", "loans", "deposits", "capital", "liquidity", "supervisory"],
          "digital": ["digital", "payments", "payment", "cash", "crypto", "innovation", "technology", "data"]}
GNAME = {"monetary": ("monetary policy", "politică monetară"), "energy": ("energy", "energie"),
         "banking": ("banking", "sistem bancar"), "digital": ("digital and payments", "digitalizare și plăți")}
words = [(g, w) for g, ws in GROUPS.items() for w in ws if w in w2i]
Z = np.array([Ew[w2i[w]] for _, w in words])
Zc = Z - Z.mean(0)
U, Sv, Vt = np.linalg.svd(Zc, full_matrices=False)
P2 = Zc @ Vt[:2].T
N["pca"] = {"var2_pct": round(100 * float((Sv[:2] ** 2).sum() / (Sv ** 2).sum()), 1), "n_words": len(words)}
gc = {"monetary": npl.PAL["data"], "energy": npl.PAL["model"], "banking": npl.PAL["benchmark"], "digital": npl.PAL["c7"]}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for g in GROUPS:
        ii = [k for k, (gg, _) in enumerate(words) if gg == g]
        ax.scatter(P2[ii, 0], P2[ii, 1], color=gc[g], s=16, label=GNAME[g][0 if lang == "en" else 1])
        for k in ii:
            ax.annotate(words[k][1], (P2[k, 0], P2[k, 1]), fontsize=7.5, color=gc[g], xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(L("First principal component", "Prima componentă principală", lang))
    ax.set_ylabel(L("Second principal\ncomponent", "A doua componentă\nprincipală", lang))
    npl.legend_bottom(fig, ncol=4)
    npl.out(fig, HERE, lang, "w2v_pca", also_x=())
npl.chart_meta(N, "w2v_pca", "speeches", *P_SP)

# chart: training losses of SGNS and GloVe + GloVe weighting function on the real co-occurrence counts
for lang in LANGS:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    axes[0].plot(np.arange(1, len(W2V["hist"]) + 1), W2V["hist"], color=npl.PAL["model"], marker="o", ms=3, label="word2vec (SGNS)")
    axes[0].set_xlabel(L("Epoch", "Epoca", lang))
    axes[0].set_ylabel(L("Negative-sampling loss", "Pierderea cu eșantionare\nnegativă", lang), fontsize=9.5)
    xs = np.sort(GLV["x"])
    axes[1].hist(np.log10(GLV["x"]), bins=60, color=npl.PAL["data"], alpha=0.7,
                 label=L("co-occurrence counts $X_{ij}$", "frecvențele de co-apariție $X_{ij}$", lang))
    axes[1].set_yscale("log")
    axes[1].set_xlabel(L("log$_{10}$ $X_{ij}$", "log$_{10}$ $X_{ij}$", lang))
    axes[1].set_ylabel(L("Number of word pairs", "Numărul de perechi", lang), fontsize=9.5)
    axes[1].axvline(2, color=npl.PAL["alt"], ls="--", lw=1.3, label=L("$x_{\\max}=100$ (weight 1 above)", "$x_{\\max}=100$ (pondere 1 peste)", lang))
    axes[1]._keep_y = True
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "emb_training", also_x=(axes[1],))
npl.chart_meta(N, "emb_training", "speeches", *P_SP)

# =====================================================================================================================
# 3. Sentiment classification
# =====================================================================================================================
LAB = ["negative", "neutral", "positive"]
pb_y = pb.label.map({k: i for i, k in enumerate(LAB)}).values
pb_tok = [tok(s) for s in pb.sentence]
tw_lab = {"bearish": 0, "neutral": 1, "bullish": 2}
tw_y = tw.label_name.map(tw_lab).values
tw_tok = [tok(s) for s in tw.text]


def mean_emb(toks, E):
    out = np.zeros((len(toks), E.shape[1]), np.float32)
    for k, t in enumerate(toks):
        ids = [w2i[w] for w in t if w in w2i]
        if ids:
            out[k] = E[ids].mean(0)
    return out


def ngrams(t):
    return t + [a + "_" + b for a, b in zip(t, t[1:])]


class EmbBag(nn.Module):
    """Mean of word (and bigram) embeddings -> dropout -> linear softmax layer (a fastText-style classifier)."""
    def __init__(self, n_vocab, dim, n_cls, init=None):
        super().__init__()
        self.emb = nn.EmbeddingBag(n_vocab, dim, mode="mean", padding_idx=0)
        if init is not None:
            with torch.no_grad():
                self.emb.weight[:init.shape[0]] = torch.tensor(init)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(dim, n_cls)

    def forward(self, ids, offs):
        return self.out(self.drop(self.emb(ids, offs)))


def run_embbag(tr_t, tr_y, va_t, va_y, te_t, n_cls, seed, dim=100, init=False, epochs=30, lr=5e-3, bigrams=True):
    torch.manual_seed(SEED + seed)
    np.random.seed(SEED + seed)
    feats = (lambda t: ngrams(t)) if bigrams else (lambda t: t)
    if init:                                                       # word2vec rows first, then the new tokens of the task
        voc = {"<pad>": 0}
        for w in vocab:
            voc[w] = len(voc)
    else:
        voc = {"<pad>": 0}
    c = Counter(x for t in tr_t for x in feats(t))
    for w, k in c.items():
        if k >= 2 and w not in voc:
            voc[w] = len(voc)
    pre = None
    if init:
        pre = np.zeros((1 + V, dim), np.float32)
        pre[1:] = W2V["E"][:, :dim] / np.linalg.norm(W2V["E"], axis=1, keepdims=True).mean()

    def enc(ts):
        ids, offs = [], []
        for t in ts:
            offs.append(len(ids))
            x = [voc[w] for w in feats(t) if w in voc] or [0]
            ids += x
        return torch.tensor(ids), torch.tensor(offs)

    m = EmbBag(len(voc), dim, n_cls, pre)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr_y_ = torch.tensor(tr_y)
    w_cls = torch.tensor(len(tr_y) / (n_cls * np.bincount(tr_y, minlength=n_cls)), dtype=torch.float)
    va_ids = enc(va_t)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        m.train()
        perm = np.random.permutation(len(tr_t))
        for j in range(0, len(perm), 64):
            b = perm[j:j + 64]
            ids, offs = enc([tr_t[k] for k in b])
            loss = F.cross_entropy(m(ids, offs), tr_y_[b], weight=w_cls)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            f1 = f1_score(va_y, m(*va_ids).argmax(1).numpy(), average="macro")
        if f1 > best:
            best, best_state, bad = f1, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return m(*enc(te_t)).argmax(1).numpy()


def run_tfidf_mlp(Xtr, ytr, Xva, yva, Xte, n_cls, seed):
    torch.manual_seed(SEED + seed)
    Xtr_, Xva_, Xte_ = [torch.tensor(x.toarray(), dtype=torch.float) for x in (Xtr, Xva, Xte)]
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, n_cls))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    w_cls = torch.tensor(len(ytr) / (n_cls * np.bincount(ytr, minlength=n_cls)), dtype=torch.float)
    yt = torch.tensor(ytr)
    best, state, bad = -1, None, 0
    g = torch.Generator().manual_seed(SEED + seed)
    for ep in range(40):
        m.train()
        perm = torch.randperm(len(yt), generator=g)
        for j in range(0, len(perm), 64):
            b = perm[j:j + 64]
            loss = F.cross_entropy(m(Xtr_[b]), yt[b], weight=w_cls)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            f1 = f1_score(yva, m(Xva_).argmax(1).numpy(), average="macro")
        if f1 > best:
            best, state, bad = f1, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return m(Xte_).argmax(1).numpy()


def lr_tuned(Xtr, ytr, Xva, yva, Xte):
    best = None
    for C in [0.1, 1.0, 10.0]:                                     # penalty chosen on the validation part
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, C, m)
    return best[2].predict(Xte), best[1], best[2]


MODELS = ["majority", "bow_lr", "tfidf_lr", "tfidf_mlp", "w2v_lr", "embbag", "embbag_w2v"]


def evaluate_split(tr, va, te, toks, yy, seed, n_cls=3):
    jt = lambda ids: [" ".join(toks[i]) for i in ids]
    res, preds = {}, {}
    maj = np.bincount(yy[tr]).argmax()
    preds["majority"] = np.full(len(te), maj)
    bow = CountVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, min_df=2).fit(jt(tr))
    preds["bow_lr"], _, mb = lr_tuned(bow.transform(jt(tr)), yy[tr], bow.transform(jt(va)), yy[va], bow.transform(jt(te)))
    tf = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True).fit(jt(tr))
    Xtr, Xva, Xte = tf.transform(jt(tr)), tf.transform(jt(va)), tf.transform(jt(te))
    preds["tfidf_lr"], _, _ = lr_tuned(Xtr, yy[tr], Xva, yy[va], Xte)
    preds["tfidf_mlp"] = run_tfidf_mlp(Xtr, yy[tr], Xva, yy[va], Xte, n_cls, seed)
    E = W2V["E"]
    me = lambda ids: mean_emb([toks[i] for i in ids], E)
    preds["w2v_lr"], _, _ = lr_tuned(me(tr), yy[tr], me(va), yy[va], me(te))
    tt = lambda ids: [toks[i] for i in ids]
    preds["embbag"] = run_embbag(tt(tr), yy[tr], tt(va), yy[va], tt(te), n_cls, seed)
    preds["embbag_w2v"] = run_embbag(tt(tr), yy[tr], tt(va), yy[va], tt(te), n_cls, seed, init=True)
    for k, p in preds.items():
        res[k] = {"f1": f1_score(yy[te], p, average="macro"), "acc": accuracy_score(yy[te], p)}
    return res, preds, mb, bow


def pb_split(seed):
    rs = np.random.RandomState(SEED + seed)
    tr, va, te = [], [], []
    for c in range(3):
        ids = rs.permutation(np.where(pb_y == c)[0])
        n = len(ids)
        tr += list(ids[:int(0.7 * n)]); va += list(ids[int(0.7 * n):int(0.85 * n)]); te += list(ids[int(0.85 * n):])
    return np.array(tr), np.array(va), np.array(te)


def run_sentiment():
    out = {"pb": [], "tw": []}
    for s in range(5):
        tr, va, te = pb_split(s)
        res, preds, mb, bow = evaluate_split(tr, va, te, pb_tok, pb_y, s)
        out["pb"].append({"res": res, "te": te, "preds": preds,
                          "coef": mb.coef_ if s == 0 else None, "vocab": bow.get_feature_names_out() if s == 0 else None})
        print("    phrasebank split", s, {k: round(v["f1"], 3) for k, v in res.items()})
    tr_all = np.where(tw.split == "train")[0]
    te = np.where(tw.split == "valid")[0]
    for s in range(3):
        rs = np.random.RandomState(SEED + s)
        p = rs.permutation(tr_all)
        va, tr = p[:int(0.15 * len(p))], p[int(0.15 * len(p)):]
        res, preds, _, _ = evaluate_split(tr, va, te, tw_tok, tw_y, s)
        out["tw"].append({"res": res, "preds": preds})
        print("    tweets seed", s, {k: round(v["f1"], 3) for k, v in res.items()})
    return out


SE = cached("sentiment", run_sentiment)
N["sent"] = {"splits_pb": 5, "seeds_tw": 3}
for ds in ["pb", "tw"]:
    N["sent"][ds] = {}
    for k in MODELS:
        f = np.array([r["res"][k]["f1"] for r in SE[ds]]) * 100
        a = np.array([r["res"][k]["acc"] for r in SE[ds]]) * 100
        N["sent"][ds][k] = {"f1": round(float(f.mean()), 2), "f1_sd": round(float(f.std(ddof=1)), 2),
                            "acc": round(float(a.mean()), 2)}
    best = max(MODELS, key=lambda k: N["sent"][ds][k]["f1"])
    N["sent"][ds]["best"] = best
# paired comparison on the 5 PhraseBank splits: TF-IDF + LR vs embedding bag with word2vec start
d = np.array([r["res"]["tfidf_lr"]["f1"] - r["res"]["embbag_w2v"]["f1"] for r in SE["pb"]]) * 100
N["sent"]["pb_diff_tfidf_embw2v"] = round(float(d.mean()), 2)
N["sent"]["pb_diff_t"] = round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), 2)
d2 = np.array([r["res"]["embbag_w2v"]["f1"] - r["res"]["embbag"]["f1"] for r in SE["pb"]]) * 100
N["sent"]["pb_w2v_gain"] = round(float(d2.mean()), 2)
d3 = np.array([r["res"]["embbag_w2v"]["f1"] - r["res"]["embbag"]["f1"] for r in SE["tw"]]) * 100
N["sent"]["tw_w2v_gain"] = round(float(d3.mean()), 2)

MNAME = {"majority": ("Majority", "Clasa majoritară"), "bow_lr": ("BoW + logistic", "BoW + logistic"),
         "tfidf_lr": ("TF-IDF + logistic", "TF-IDF + logistic"), "tfidf_mlp": ("TF-IDF + MLP", "TF-IDF + MLP"),
         "w2v_lr": ("mean word2vec + logistic", "media word2vec + logistic"),
         "embbag": ("embedding bag (random)", "embedding bag (aleator)"),
         "embbag_w2v": ("embedding bag (word2vec start)", "embedding bag (start word2vec)")}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    yv = np.arange(len(MODELS))
    for j, (ds, c, lab) in enumerate([("pb", npl.PAL["data"], "Financial PhraseBank"),
                                     ("tw", npl.PAL["model"], L("Finance tweets", "Tweet-uri financiare", lang))]):
        f = [N["sent"][ds][k]["f1"] for k in MODELS]
        sd = [N["sent"][ds][k]["f1_sd"] for k in MODELS]
        ax.barh(yv + (j - 0.5) * 0.38, f, height=0.38, xerr=sd, color=c, capsize=2, label=lab)
    ax.set_yticks(yv)
    ax.set_yticklabels([MNAME[k][0 if lang == "en" else 1] for k in MODELS], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(L("Macro-F1 on the test set (%)", "Macro-F1 pe setul de test (%)", lang))
    ax._keep_y = True
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "sentiment_models", also_x=(ax,))
npl.chart_meta(N, "sentiment_models", "phrasebank", "PhraseBank: 5 random 70/15/15 splits; tweets: published validation split as test, 3 seeds",
               "PhraseBank: 5 împărțiri aleatoare 70/15/15; tweet-uri: setul de validare publicat, folosit ca test, 3 semințe",
               source_en="Financial PhraseBank (Malo et al., 2014); Twitter financial news (Hugging Face, zeroshot)",
               source_ro="Financial PhraseBank (Malo et al., 2014); Twitter financial news (Hugging Face, zeroshot)")

# chart: most informative words (logistic regression on bag of words, split 0)
coef, voc_ = SE["pb"][0]["coef"], SE["pb"][0]["vocab"]
top_pos = np.argsort(-(coef[2] - coef[0]))[:8]
top_neg = np.argsort(coef[2] - coef[0])[:8]
N["words"] = {"pos": ", ".join(voc_[top_pos[:5]]), "neg": ", ".join(voc_[top_neg[:5]])}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    vals = np.concatenate([(coef[2] - coef[0])[top_neg][::-1], (coef[2] - coef[0])[top_pos][::-1]])
    labs = list(voc_[top_neg][::-1]) + list(voc_[top_pos][::-1])
    cl = [npl.PAL["positive"]] * 8 + [npl.PAL["benchmark"]] * 8
    ax.barh(range(16), vals, color=cl)
    ax.set_yticks(range(16)); ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0, color="#222222", lw=0.7)
    ax.set_xlabel(L("$\\beta_{\\mathrm{positive}}-\\beta_{\\mathrm{negative}}$ (log-odds per occurrence)",
                    "$\\beta_{\\mathrm{pozitiv}}-\\beta_{\\mathrm{negativ}}$ (log-șanse la o apariție)", lang))
    ax.barh([], [], color=npl.PAL["benchmark"], label=L("pushes towards positive", "împinge spre pozitiv", lang))
    ax.barh([], [], color=npl.PAL["positive"], label=L("pushes towards negative", "împinge spre negativ", lang))
    ax._keep_y = True
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "top_words", also_x=(ax,))
npl.chart_meta(N, "top_words", "phrasebank", *P_PB)

# chart: accuracy by annotator agreement
agree = pb.min_agreement_pct.values
acc_ag = {k: {} for k in ["tfidf_lr", "embbag_w2v", "majority"]}
for k in acc_ag:
    for a in [50, 66, 75, 100]:
        v = []
        for r in SE["pb"]:
            te = r["te"]
            msk = agree[te] == a
            v.append(float((r["preds"][k][msk] == pb_y[te][msk]).mean()))
        acc_ag[k][a] = 100 * np.mean(v)
N["agree"] = {k: {str(a): round(v, 1) for a, v in d_.items()} for k, d_ in acc_ag.items()}
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    xs = np.arange(4)
    for j, (k, c) in enumerate([("majority", npl.MUTED), ("tfidf_lr", npl.PAL["benchmark"]), ("embbag_w2v", npl.PAL["model"])]):
        ax.bar(xs + (j - 1) * 0.27, [acc_ag[k][a] for a in [50, 66, 75, 100]], width=0.27, color=c,
               label=MNAME[k][0 if lang == "en" else 1])
    ax.set_xticks(xs)
    ax.set_xticklabels([L(f"{a}% agree", f"acord {a}%", lang) for a in [50, 66, 75, 100]])
    ax.set_ylabel(L("Test accuracy (%)", "Acuratețea pe test (%)", lang))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "agreement")
npl.chart_meta(N, "agreement", "phrasebank", "test sentences of 5 random splits, by share of annotators that agree",
               "frazele de test din 5 împărțiri aleatoare, după ponderea adnotatorilor care sînt de acord")

# chart: confusion matrix of TF-IDF + LR, all 5 PhraseBank test sets pooled
cmp = sum(confusion_matrix(pb_y[r["te"]], r["preds"]["tfidf_lr"], labels=[0, 1, 2]) for r in SE["pb"])
N["cm"] = {"neg_as_pos": int(cmp[0, 2]), "pos_as_neg": int(cmp[2, 0]), "neg_as_neu": int(cmp[0, 1]),
           "pos_as_neu": int(cmp[2, 1]), "n": int(cmp.sum()),
           "neg_recall": round(100 * cmp[0, 0] / cmp[0].sum(), 1), "pos_recall": round(100 * cmp[2, 2] / cmp[2].sum(), 1),
           "neu_recall": round(100 * cmp[1, 1] / cmp[1].sum(), 1)}

# learning curve: fraction of the training part (split 0), transfer from the ECB embeddings
FR = [0.05, 0.1, 0.25, 0.5, 1.0]


def run_curve():
    out = {"tfidf_lr": [], "embbag": [], "embbag_w2v": [], "w2v_lr": []}
    for s in range(3):
        tr, va, te = pb_split(s)
        rs = np.random.RandomState(SEED + 7 + s)
        for fr in FR:
            sub = rs.permutation(tr)[:int(fr * len(tr))]
            jt = lambda ids: [" ".join(pb_tok[i]) for i in ids]
            tf = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True).fit(jt(sub))
            p, _, _ = lr_tuned(tf.transform(jt(sub)), pb_y[sub], tf.transform(jt(va)), pb_y[va], tf.transform(jt(te)))
            out["tfidf_lr"].append((s, fr, f1_score(pb_y[te], p, average="macro")))
            me = lambda ids: mean_emb([pb_tok[i] for i in ids], W2V["E"])
            p, _, _ = lr_tuned(me(sub), pb_y[sub], me(va), pb_y[va], me(te))
            out["w2v_lr"].append((s, fr, f1_score(pb_y[te], p, average="macro")))
            tt = lambda ids: [pb_tok[i] for i in ids]
            for k, ini in [("embbag", False), ("embbag_w2v", True)]:
                p = run_embbag(tt(sub), pb_y[sub], tt(va), pb_y[va], tt(te), 3, s, init=ini)
                out[k].append((s, fr, f1_score(pb_y[te], p, average="macro")))
        print("    curve seed", s)
    dims = {}
    tr, va, te = pb_split(0)
    tt = lambda ids: [pb_tok[i] for i in ids]
    for dmn in [8, 16, 32, 64, 128, 256]:
        dims[dmn] = [f1_score(pb_y[te], run_embbag(tt(tr), pb_y[tr], tt(va), pb_y[va], tt(te), 3, s, dim=dmn), average="macro")
                     for s in range(3)]
    out["dims"] = dims
    return out


CU = cached("curve", run_curve)
ntr = len(pb_split(0)[0])
curve = {k: [100 * np.mean([f for s, fr, f in CU[k] if fr == x]) for x in FR] for k in ["tfidf_lr", "embbag", "embbag_w2v", "w2v_lr"]}
N["curve"] = {k: {f"p{int(round(100 * x))}": round(v[i], 1) for i, x in enumerate(FR)} for k, v in curve.items()}
N["curve"]["n_small"] = int(FR[0] * ntr)
N["curve"]["n_full"] = ntr
N["dims"] = {str(k): round(100 * float(np.mean(v)), 1) for k, v in CU["dims"].items()}
for lang in LANGS:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1.4, 1]})
    nn_ = [int(x * ntr) for x in FR]
    for k, c in [("tfidf_lr", npl.PAL["benchmark"]), ("w2v_lr", npl.PAL["alt"]), ("embbag", npl.PAL["c7"]), ("embbag_w2v", npl.PAL["model"])]:
        axes[0].plot(nn_, curve[k], marker="o", ms=3, color=c, lw=1.5, label=MNAME[k][0 if lang == "en" else 1])
    axes[0].set_xscale("log")
    axes[0].set_xticks(nn_); axes[0].set_xticklabels([str(x) for x in nn_], fontsize=8)
    axes[0].set_xlabel(L("Labelled training sentences", "Fraze etichetate de antrenare", lang), fontsize=9)
    axes[0].set_ylabel(L("Test macro-F1 (%)", "Macro-F1 pe test (%)", lang))
    ds_ = sorted(CU["dims"])
    axes[1].plot(ds_, [100 * np.mean(CU["dims"][k]) for k in ds_], marker="s", ms=3, color=npl.PAL["c7"], lw=1.5)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ds_); axes[1].set_xticklabels([str(x) for x in ds_], fontsize=8)
    axes[1].set_xlabel(L("Embedding dimension $d$", "Dimensiunea vectorilor $d$", lang), fontsize=9)
    axes[1].set_ylabel(L("Test macro-F1 (%)", "Macro-F1 pe test (%)", lang))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "learning_curve", bottom=0.22)
npl.chart_meta(N, "learning_curve", "phrasebank", "left: 3 random splits; right: embedding bag (random start), split 1, 3 seeds",
               "stînga: 3 împărțiri aleatoare; dreapta: embedding bag (start aleator), împărțirea 1, 3 semințe")

# =====================================================================================================================
# 4. Topic classification of finance tweets (20 classes): TF-IDF + logistic regression vs embedding bag
# =====================================================================================================================
tp_tok = [tok(s) for s in tp.text]
tp_y = tp.label.values
names_tp = tp.drop_duplicates("label").set_index("label").label_name.sort_index()


def run_topic():
    tr_all = np.where(tp.split == "train")[0]
    te = np.where(tp.split == "valid")[0]
    rs = np.random.RandomState(SEED)
    p = rs.permutation(tr_all)
    va, tr = p[:int(0.15 * len(p))], p[int(0.15 * len(p)):]
    jt = lambda ids: [" ".join(tp_tok[i]) for i in ids]
    tf = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True).fit(jt(tr))
    p_lr, C, _ = lr_tuned(tf.transform(jt(tr)), tp_y[tr], tf.transform(jt(va)), tp_y[va], tf.transform(jt(te)))
    tt = lambda ids: [tp_tok[i] for i in ids]
    p_eb = run_embbag(tt(tr), tp_y[tr], tt(va), tp_y[va], tt(te), 20, 0, init=True)
    return {"te": te, "lr": p_lr, "eb": p_eb, "C": C}


TP = cached("topic", run_topic)
yt_ = tp_y[TP["te"]]
f1c = f1_score(yt_, TP["lr"], average=None)
N["topic"] = {"lr_f1": round(100 * f1_score(yt_, TP["lr"], average="macro"), 2), "lr_acc": round(100 * accuracy_score(yt_, TP["lr"]), 2),
              "eb_f1": round(100 * f1_score(yt_, TP["eb"], average="macro"), 2), "eb_acc": round(100 * accuracy_score(yt_, TP["eb"]), 2),
              "best_class": names_tp.iloc[int(f1c.argmax())], "best_f1": round(100 * float(f1c.max()), 1),
              "worst_class": names_tp.iloc[int(f1c.argmin())], "worst_f1": round(100 * float(f1c.min()), 1),
              "n_test": int(len(yt_)), "majority_acc": round(100 * float(np.mean(yt_ == np.bincount(tp_y[tp.split == "train"]).argmax())), 2)}
TOPIC_RO = {"Analyst Update": "Actualizări ale analiștilor", "Fed | Central Banks": "Fed și băncile centrale",
            "Company | Product News": "Știri despre companii și produse", "Treasuries | Corporate Debt": "Titluri de stat și obligațiuni corporative",
            "Dividend": "Dividende", "Earnings": "Rezultate financiare", "Energy | Oil": "Energie și petrol",
            "Financials": "Sectorul financiar", "Currencies": "Valute", "General News | Opinion": "Știri generale și opinii",
            "Gold | Metals | Materials": "Aur, metale și materiale", "IPO": "Oferte publice inițiale", "Legal | Regulation": "Juridic și reglementare",
            "M&A | Investments": "Fuziuni, achiziții și investiții", "Macro": "Macroeconomie", "Markets": "Piețe",
            "Politics": "Politică", "Personnel Change": "Schimbări de personal", "Stock Commentary": "Comentarii despre acțiuni",
            "Stock Movement": "Mișcări ale acțiunilor"}
N["topic"]["best_class_ro"] = TOPIC_RO.get(N["topic"]["best_class"], N["topic"]["best_class"])
N["topic"]["worst_class_ro"] = TOPIC_RO.get(N["topic"]["worst_class"], N["topic"]["worst_class"])
order = np.argsort(f1c)
for lang in LANGS:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labs = [names_tp.iloc[i] if lang == "en" else TOPIC_RO.get(names_tp.iloc[i], names_tp.iloc[i]) for i in order]
    ax.barh(range(20), 100 * f1c[order], color=npl.PAL["benchmark"], label=L("TF-IDF + logistic", "TF-IDF + logistic", lang))
    f1e = f1_score(yt_, TP["eb"], average=None)
    ax.scatter(100 * f1e[order], range(20), color=npl.PAL["model"], s=14, zorder=3,
               label=L("embedding bag (word2vec start)", "embedding bag (start word2vec)", lang))
    ax.set_yticks(range(20)); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel(L("F1 on the test tweets (%)", "F1 pe tweet-urile de test (%)", lang), fontsize=9.5)
    ax._keep_y = True
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "topic_f1", also_x=(ax,), bottom=0.08)
npl.chart_meta(N, "topic_f1", "tweets", f"{N['data']['tp_valid']:,} test tweets (published validation split), 20 topics",
               f"{N['data']['tp_valid']:,} de tweet-uri de test (setul de validare publicat), 20 de teme".replace(",", "."))

# OOV: share of tokens of each task that the ECB vocabulary covers
def cover(toks):
    tot = sum(len(t) for t in toks)
    return 100 * sum(1 for t in toks for w in t if w in w2i) / tot


N["oov"] = {"pb_cover": round(cover(pb_tok), 1), "tw_cover": round(cover(tw_tok), 1)}

with open(os.path.join(HERE, "L6_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote L6_numbers.json and", len(N["charts"]), "charts x 2 languages")
