"""
analysis_L7.py -- Lecture 7 "Attention and transformers": all computation on real financial text.
Writes lectures/L7/L7_numbers.json, charts/en|ro/*.pdf and lectures/L7/results/*.csv (small files read by the Quantlets).

Data: Twitter financial news sentiment (bearish / bullish / neutral; published train and valid splits), Financial
PhraseBank sentences, ECB speeches (lengths only). The published "valid" split is our TEST set (used once); 10 % of the
published train split (stratified, seed 20261108) is the validation set for early stopping and for the choice of C.
Models: TF-IDF + logistic regression (benchmark); embedding bag (no attention); a small transformer trained from scratch
(with and without positional encoding), 3 seeds each; FinBERT (Araci, 2019; ProsusAI) zero-shot, linear probe and
fine-tuned (3 seeds). Bootstrap (1,000 draws of the test tweets) for the differences in macro-F1.
Run:  python3 lectures/L7/analysis_L7.py     (about 12 minutes on an Apple-silicon laptop, MPS; results cached in cache/)
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
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()
npl.style()
SEED = npl.seed_all()
DEV = npl.device()
N = {"charts": {}, "seed": SEED, "device": DEV}
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(HERE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RES, exist_ok=True)
SEEDS = [1, 2, 3]
MAXLEN = 64
CLASSES = ["bearish", "bullish", "neutral"]
t0 = time.time()

# ---------------------------------------------------------------- 0. data
tw = npl.tweets_sentiment()
tr_all = tw[tw.split == "train"].reset_index(drop=True)
te = tw[tw.split == "valid"].reset_index(drop=True)
tr, va = train_test_split(tr_all, test_size=0.10, stratify=tr_all.label, random_state=SEED)
tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
ytr, yva, yte = tr.label.values, va.label.values, te.label.values
N["data"] = {"n_train_pub": int(len(tr_all)), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
             "pct_neutral_test": round(100 * float(np.mean(yte == 2)), 1), "pct_bearish_test": round(100 * float(np.mean(yte == 0)), 1),
             "pct_bullish_test": round(100 * float(np.mean(yte == 1)), 1),
             "n_bearish_test": int((yte == 0).sum()), "n_bullish_test": int((yte == 1).sum()), "n_neutral_test": int((yte == 2).sum())}
assert list(tw.drop_duplicates("label").sort_values("label").label_name) == CLASSES

tok = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=True)
N["tok"] = {"vocab": int(tok.vocab_size)}

# ---------------------------------------------------------------- 1. context lengths (tokens): tweets, PhraseBank, ECB speeches
pb = npl.phrasebank()
sp = npl.ecb_speeches()
len_tw = np.array([len(x) for x in tok(list(tw.text), add_special_tokens=True)["input_ids"]])
len_pb = np.array([len(x) for x in tok(list(pb.sentence), add_special_tokens=True)["input_ids"]])
len_sp = np.array([len(x) for x in tok(list(sp.contents.astype(str)), add_special_tokens=True, truncation=False)["input_ids"]])
pd.DataFrame({"corpus": ["tweets"] * len(len_tw) + ["phrasebank"] * len(len_pb) + ["speeches"] * len(len_sp),
              "tokens": np.concatenate([len_tw, len_pb, len_sp])}).to_csv(os.path.join(RES, "token_lengths.csv"), index=False)
N["len"] = {"tw_median": int(np.median(len_tw)), "tw_p99": int(np.percentile(len_tw, 99)), "tw_max": int(len_tw.max()),
            "pb_median": int(np.median(len_pb)), "sp_median": int(np.median(len_sp)), "sp_max": int(len_sp.max()),
            "sp_pct_over512": round(100 * float(np.mean(len_sp > 512)), 1), "n_speeches": int(len(sp)),
            "n_pb": int(len(pb)), "tw_pct_over64": round(100 * float(np.mean(len_tw > MAXLEN)), 2),
            "sp_first": str(sp.date.min().date()), "sp_last": str(sp.date.max().date())}
N["len"]["cost_ratio"] = int(round((np.median(len_sp) / np.median(len_tw)) ** 2))
print("lengths", N["len"], round(time.time() - t0), "s")


def enc(texts):
    e = tok(list(texts), padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


Itr, Mtr = enc(tr.text)
Iva, Mva = enc(va.text)
Ite, Mte = enc(te.text)


def scores(y, yhat):
    return {"macro_f1": f1_score(y, yhat, average="macro"), "acc": accuracy_score(y, yhat),
            "rec_bearish": recall_score(y, yhat, labels=[0], average="macro"),
            "prec_bearish": precision_score(y, yhat, labels=[0], average="macro", zero_division=0)}


# ---------------------------------------------------------------- 2. benchmark: TF-IDF + logistic regression (C on validation)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(tr.text)
Xva, Xte = vec.transform(va.text), vec.transform(te.text)
bestC, bestf = None, -1
for C in [0.3, 1, 3, 10, 30]:
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    f = f1_score(yva, m.predict(Xva), average="macro")
    if f > bestf:
        bestC, bestf = C, f
lr_tfidf = LogisticRegression(C=bestC, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
PRED = {"tfidf": [lr_tfidf.predict(Xte)]}
N["tfidf"] = {"C": bestC, "n_features": int(Xtr.shape[1])}
print("tfidf", scores(yte, PRED["tfidf"][0]))


# ---------------------------------------------------------------- 3. small transformer from scratch
class TinyTransformer(nn.Module):
    """Token embedding + positional encoding -> L encoder blocks (multi-head self-attention, feed-forward, residual,
    layer normalisation) -> mean pooling over the real tokens -> linear layer to 3 classes."""

    def __init__(self, vocab, d=64, heads=4, layers=2, ff=128, drop=0.1, pos=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = pos
        pe = torch.zeros(MAXLEN, d)
        p = torch.arange(MAXLEN).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(p * div), torch.cos(p * div)
        self.register_buffer("pe", pe)
        self.layers = layers
        if layers:
            blk = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(blk, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(d, 3)

    def forward(self, ids, mask):
        h = self.emb(ids) * math.sqrt(self.emb.embedding_dim)
        if self.pos:
            h = h + self.pe[: ids.shape[1]]
        h = self.drop(h)
        if self.layers:
            h = self.enc(h, src_key_padding_mask=(mask == 0))
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1))


def train_tiny(seed, frac=1.0, pos=True, layers=2, epochs=20, patience=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.arange(len(ytr))
    if frac < 1:
        idx, _ = train_test_split(idx, train_size=frac, stratify=ytr, random_state=seed)
    model = TinyTransformer(tok.vocab_size, pos=pos, layers=layers).to(DEV)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr)), dtype=torch.float32, device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    yt = torch.tensor(ytr)
    best, best_state, wait, hist = -1, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.tensor(idx)[torch.randperm(len(idx), generator=g)]
        tl = 0.0
        for b in range(0, len(perm), 64):
            j = perm[b:b + 64]
            out = model(Itr[j].to(DEV), Mtr[j].to(DEV))
            loss = F.cross_entropy(out, yt[j].to(DEV), weight=w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(j)
        model.eval()
        with torch.no_grad():
            ov = model(Iva.to(DEV), Mva.to(DEV))
            vl = float(F.cross_entropy(ov, torch.tensor(yva, device=DEV), weight=w))
            fv = f1_score(yva, ov.argmax(1).cpu().numpy(), average="macro")
        hist.append((ep, tl / len(idx), vl, fv))
        if fv > best:
            best, wait = fv, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(Ite[b:b + 512].to(DEV), Mte[b:b + 512].to(DEV)).argmax(1).cpu() for b in range(0, len(yte), 512)]).numpy()
    return pred, hist, model


tcache = os.path.join(CACHE, "tiny.json")
if os.path.exists(tcache):
    TC = json.load(open(tcache))
else:
    TC = {}
    for name, kw in [("tiny", {}), ("tiny_nopos", {"pos": False}), ("embbag", {"layers": 0})]:
        TC[name] = {"pred": [], "hist": [], "sec": []}
        for s in SEEDS:
            s0 = time.time()
            p, h, mdl = train_tiny(s, **kw)
            TC[name]["pred"].append(p.tolist())
            TC[name]["hist"].append(h)
            TC[name]["sec"].append(time.time() - s0)
            print(name, s, scores(yte, p), round(time.time() - s0, 1), "s")
        TC[name]["params"] = int(sum(q.numel() for q in mdl.parameters()))
        TC[name]["params_noemb"] = int(sum(q.numel() for n_, q in mdl.named_parameters() if not n_.startswith("emb")))
    json.dump(TC, open(tcache, "w"))
for name in ["tiny", "tiny_nopos", "embbag"]:
    PRED[name] = [np.array(p) for p in TC[name]["pred"]]

# ---------------------------------------------------------------- 4. FinBERT: zero-shot, linear probe, fine-tuned
fin_map = {0: 1, 1: 0, 2: 2}       # FinBERT label (positive, negative, neutral) -> ours (bearish 0, bullish 1, neutral 2)
our2fin = {v: k for k, v in fin_map.items()}


def finbert(attn=False):
    kw = {"attn_implementation": "eager"} if attn else {}
    return AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", local_files_only=True, **kw).to(DEV)


def run_logits(model, I, M, hidden=False):
    outs, hs = [], []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(I), 256):
            o = model(input_ids=I[b:b + 256].to(DEV), attention_mask=M[b:b + 256].to(DEV), output_hidden_states=hidden)
            outs.append(o.logits.float().cpu())
            if hidden:
                h = o.hidden_states[-1]
                m = M[b:b + 256].to(DEV).unsqueeze(-1).float()
                hs.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    return torch.cat(outs), (torch.cat(hs) if hidden else None)


fcache = os.path.join(CACHE, "finbert.json")
if os.path.exists(fcache):
    FC = json.load(open(fcache))
else:
    FC = {}
    fb = finbert()
    s0 = time.time()
    lg, hte = run_logits(fb, Ite, Mte, hidden=True)
    FC["zs_pred"] = [fin_map[int(i)] for i in lg.argmax(1)]
    FC["zs_sec"] = time.time() - s0
    _, htr = run_logits(fb, Itr, Mtr, hidden=True)
    _, hva = run_logits(fb, Iva, Mva, hidden=True)
    np.savez_compressed(os.path.join(CACHE, "finbert_emb.npz"), htr=htr.numpy(), hva=hva.numpy(), hte=hte.numpy())
    FC["params"] = int(sum(q.numel() for q in fb.parameters()))
    del fb
    # fine-tuning: all weights, 3 epochs, AdamW, learning rate 2e-5 with linear decay, batch 32; best epoch on validation
    FC["ft_pred"], FC["ft_hist"], FC["ft_sec"] = [], [], []
    yfin = torch.tensor([our2fin[int(v)] for v in ytr])
    for s in SEEDS:
        torch.manual_seed(s)
        s0 = time.time()
        fb = finbert()
        opt = torch.optim.AdamW(fb.parameters(), lr=2e-5, weight_decay=0.01)
        steps = 3 * math.ceil(len(ytr) / 32)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: max(0.0, 1 - k / steps))
        g = torch.Generator().manual_seed(s)
        best, best_pred, hist = -1, None, []
        for ep in range(1, 4):
            fb.train()
            perm = torch.randperm(len(ytr), generator=g)
            tl = 0.0
            for b in range(0, len(perm), 32):
                j = perm[b:b + 32]
                out = fb(input_ids=Itr[j].to(DEV), attention_mask=Mtr[j].to(DEV))
                loss = F.cross_entropy(out.logits, yfin[j].to(DEV))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
                tl += float(loss) * len(j)
            lv, _ = run_logits(fb, Iva, Mva)
            fv = f1_score(yva, [fin_map[int(i)] for i in lv.argmax(1)], average="macro")
            hist.append((ep, tl / len(ytr), fv))
            if fv > best:
                best = fv
                lt, _ = run_logits(fb, Ite, Mte)
                best_pred = [fin_map[int(i)] for i in lt.argmax(1)]
            print("finbert ft seed", s, "epoch", ep, round(fv, 4), round(time.time() - s0), "s")
        FC["ft_pred"].append(best_pred)
        FC["ft_hist"].append(hist)
        FC["ft_sec"].append(time.time() - s0)
        if s == SEEDS[0]:
            torch.save(fb.state_dict(), os.path.join(CACHE, "finbert_ft_seed1_last.pt"))
        del fb
        if DEV == "mps":
            torch.mps.empty_cache()
    json.dump(FC, open(fcache, "w"))
PRED["finbert_zs"] = [np.array(FC["zs_pred"])]
PRED["finbert_ft"] = [np.array(p) for p in FC["ft_pred"]]
ez = np.load(os.path.join(CACHE, "finbert_emb.npz"))
htr, hva, hte = ez["htr"], ez["hva"], ez["hte"]
bestC, bestf = None, -1
for C in [0.01, 0.03, 0.1, 0.3, 1]:
    m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(htr, ytr)
    f = f1_score(yva, m.predict(hva), average="macro")
    if f > bestf:
        bestC, bestf = C, f
probe = LogisticRegression(C=bestC, max_iter=5000, class_weight="balanced").fit(htr, ytr)
PRED["finbert_probe"] = [probe.predict(hte)]
N["probe"] = {"C": bestC, "dim": int(htr.shape[1])}
print("finbert done", round(time.time() - t0), "s")

# ---------------------------------------------------------------- 5. results table + bootstrap
MODELS = ["tfidf", "embbag", "tiny_nopos", "tiny", "finbert_zs", "finbert_probe", "finbert_ft"]
N["models"] = {}
rows = []
for mname in MODELS:
    ss = [scores(yte, p) for p in PRED[mname]]
    N["models"][mname] = {k: round(100 * float(np.mean([s_[k] for s_ in ss])), 1) for k in ss[0]}
    N["models"][mname]["macro_f1_sd"] = round(100 * float(np.std([s_["macro_f1"] for s_ in ss], ddof=1)), 1) if len(ss) > 1 else 0.0
    N["models"][mname]["n_runs"] = len(ss)
    for i, s_ in enumerate(ss):
        rows.append({"model": mname, "run": i + 1, **{k: round(float(v), 6) for k, v in s_.items()}})
pd.DataFrame(rows).to_csv(os.path.join(RES, "model_metrics.csv"), index=False)
predtab = pd.DataFrame({"y": yte, **{m_: PRED[m_][0] for m_ in MODELS}})
predtab.to_csv(os.path.join(RES, "test_predictions.csv"), index=False)
N["models"]["tiny"]["params"] = TC["tiny"]["params"]
N["models"]["tiny"]["params_noemb"] = TC["tiny"]["params_noemb"]
N["models"]["embbag"]["params"] = TC["embbag"]["params"]
N["models"]["finbert_ft"]["params_m"] = round(FC["params"] / 1e6, 1)
N["models"]["finbert_ft"]["min_per_seed"] = round(float(np.mean(FC["ft_sec"])) / 60, 1)
N["models"]["tiny"]["sec_per_seed"] = round(float(np.mean(TC["tiny"]["sec"])), 1)
N["models"]["tiny"]["epochs_median"] = int(np.median([len(h) for h in TC["tiny"]["hist"]]))
N["gain_ft_vs_tfidf"] = round(N["models"]["finbert_ft"]["macro_f1"] - N["models"]["tfidf"]["macro_f1"], 1)
N["gain_pos"] = round(N["models"]["tiny"]["macro_f1"] - N["models"]["tiny_nopos"]["macro_f1"], 1)
N["gain_ft_vs_zs"] = round(N["models"]["finbert_ft"]["macro_f1"] - N["models"]["finbert_zs"]["macro_f1"], 1)

rng = np.random.default_rng(SEED)
pairs = {"ft_tfidf": ("finbert_ft", "tfidf"), "tiny_tfidf": ("tiny", "tfidf"), "tiny_nopos": ("tiny", "tiny_nopos"),
         "probe_tfidf": ("finbert_probe", "tfidf")}
N["boot"] = {}
bcache = os.path.join(CACHE, "boot.json")
BC = json.load(open(bcache)) if os.path.exists(bcache) else {}
for key, (a, b) in pairs.items():
    if key in BC:
        N["boot"][key] = BC[key]
        continue
    pa, pb_ = PRED[a], PRED[b]
    d_ = []
    for _ in range(1000):
        i = rng.integers(0, len(yte), len(yte))
        fa = np.mean([f1_score(yte[i], p[i], average="macro") for p in pa])
        fb_ = np.mean([f1_score(yte[i], p[i], average="macro") for p in pb_])
        d_.append(fa - fb_)
    N["boot"][key] = {"mean": round(100 * float(np.mean(d_)), 1), "lo": round(100 * float(np.percentile(d_, 2.5)), 1),
                      "hi": round(100 * float(np.percentile(d_, 97.5)), 1)}
    BC[key] = N["boot"][key]
json.dump(BC, open(bcache, "w"))
print(json.dumps(N["models"], indent=1), N["boot"])

# confusion of the fine-tuned model (seed 1)
cm = confusion_matrix(yte, PRED["finbert_ft"][0], labels=[0, 1, 2])
N["cm"] = {"bear_as_neutral": int(cm[0, 2]), "bull_as_neutral": int(cm[1, 2]), "bear_as_bull": int(cm[0, 1]),
           "bull_as_bear": int(cm[1, 0]), "bear_ok": int(cm[0, 0]), "bull_ok": int(cm[1, 1]), "neu_ok": int(cm[2, 2])}
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(os.path.join(RES, "confusion_finbert_ft.csv"))
cm_zs = confusion_matrix(yte, PRED["finbert_zs"][0], labels=[0, 1, 2])
N["cm_zs"] = {"bear_ok": int(cm_zs[0, 0]), "bull_ok": int(cm_zs[1, 1]), "neu_ok": int(cm_zs[2, 2])}

# ---------------------------------------------------------------- 6. data scaling (seed 1)
scache = os.path.join(CACHE, "scaling.csv")
FRACS = [0.05, 0.1, 0.25, 0.5, 1.0]
if os.path.exists(scache):
    sc = pd.read_csv(scache)
else:
    rows = []
    for fr in FRACS:
        idx = np.arange(len(ytr))
        if fr < 1:
            idx, _ = train_test_split(idx, train_size=fr, stratify=ytr, random_state=1)
        v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        xa = v2.fit_transform(tr.text.iloc[idx])
        f_tfidf = f1_score(yte, LogisticRegression(C=N["tfidf"]["C"], max_iter=3000, class_weight="balanced").fit(xa, ytr[idx]).predict(v2.transform(te.text)), average="macro")
        f_probe = f1_score(yte, LogisticRegression(C=N["probe"]["C"], max_iter=5000, class_weight="balanced").fit(htr[idx], ytr[idx]).predict(hte), average="macro")
        p, _, _ = train_tiny(1, frac=fr) if fr < 1 else (PRED["tiny"][0], None, None)
        rows.append({"frac": fr, "n": len(idx), "tfidf": f_tfidf, "tiny": f1_score(yte, p, average="macro"), "probe": f_probe})
        print("scaling", rows[-1])
    sc = pd.DataFrame(rows)
    sc.to_csv(scache, index=False)
sc.to_csv(os.path.join(RES, "data_scaling.csv"), index=False)
N["scaling"] = {"n_small": int(sc.n.iloc[0]), "tiny_small": round(100 * sc.tiny.iloc[0], 1), "tfidf_small": round(100 * sc.tfidf.iloc[0], 1),
                "probe_small": round(100 * sc.probe.iloc[0], 1), "n_full": int(sc.n.iloc[-1])}

# ---------------------------------------------------------------- 7. attention inside FinBERT (pretrained, eager attention)
fa = finbert(attn=True)
fa.eval()
# 7.1 entropy of the attention rows by layer, 500 test tweets
sel = np.random.default_rng(SEED).choice(len(yte), 500, replace=False)
ent = np.zeros(12)
on_sep = np.zeros(12)
on_cls = np.zeros(12)
cnt_rows = 0
with torch.no_grad():
    for b in range(0, 500, 100):
        j = torch.tensor(sel[b:b + 100])
        I_, M_ = Ite[j].to(DEV), Mte[j].to(DEV)
        o = fa(input_ids=I_, attention_mask=M_, output_attentions=True)
        lens = M_.sum(1)
        for li, A in enumerate(o.attentions):          # (batch, heads, n, n)
            A = A.float()
            for r in range(A.shape[0]):
                n_ = int(lens[r])
                a = A[r, :, :n_, :n_]
                e = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1) / math.log(n_)
                ent[li] += float(e.mean())
                on_cls[li] += float(a[:, :, 0].mean())
                on_sep[li] += float(a[:, :, n_ - 1].mean())
        cnt_rows += A.shape[0]
ent, on_cls, on_sep = ent / cnt_rows, on_cls / cnt_rows, on_sep / cnt_rows
ae = pd.DataFrame({"layer": np.arange(1, 13), "entropy": ent, "on_cls": on_cls, "on_sep": on_sep})
ae.to_csv(os.path.join(RES, "attention_entropy.csv"), index=False)
N["attn"] = {"ent_l1": round(float(ent[0]), 2), "ent_min": round(float(ent.min()), 2), "ent_min_layer": int(ent.argmin() + 1),
             "sep_max_pct": round(100 * float(on_sep.max()), 1), "sep_max_layer": int(on_sep.argmax() + 1),
             "sep_l1_pct": round(100 * float(on_sep[0]), 1), "n_tweets": 500, "heads": 12, "layers": 12, "d_model": 768,
             "d_head": 64}
# 7.2 one real test tweet: attention of the last layer, averaged over the heads
cand = te[(te.label == 0)].copy()
cand["n"] = [len(tok(x)["input_ids"]) for x in cand.text]
cand = cand[(cand.n >= 12) & (cand.n <= 16) & (~cand.text.str.contains("http"))]
ex = cand.iloc[0]
e1 = tok(ex.text, return_tensors="pt")
with torch.no_grad():
    o = fa(input_ids=e1["input_ids"].to(DEV), attention_mask=e1["attention_mask"].to(DEV), output_attentions=True)
toks = tok.convert_ids_to_tokens(e1["input_ids"][0])
A_last = o.attentions[-1][0].float().mean(0).cpu().numpy()
A_mid = o.attentions[5][0].float().mean(0).cpu().numpy()
pz = torch.softmax(o.logits.float(), 1)[0].cpu().numpy()
N["example"] = {"text": ex.text.replace("$", "\\$").replace("&", "and"), "n_tokens": len(toks), "label": "bearish",
                "p_neg_pct": round(100 * float(pz[1]), 1), "p_pos_pct": round(100 * float(pz[0]), 1)}
pd.DataFrame(A_last, index=toks, columns=toks).to_csv(os.path.join(RES, "attention_example_last.csv"))
pd.DataFrame(A_mid, index=toks, columns=toks).to_csv(os.path.join(RES, "attention_example_layer6.csv"))
# word that receives most attention from [CLS] in layer 6 (excluding special tokens)
row = A_mid[0, 1:-1]
N["example"]["cls_top_token"] = toks[1 + int(row.argmax())].replace("#", "")
N["example"]["cls_top_pct"] = round(100 * float(row.max()), 1)
del fa

# ---------------------------------------------------------------- 8. charts
PER_TW = {"en": f"{N['data']['n_test']:,} test tweets (published validation split)",
          "ro": f"{N['data']['n_test']:,} tweet-uri de test (setul de validare publicat)".replace(",", ".")}
for lang in ("en", "ro"):
    ro = lang == "ro"
    # 8.1 context lengths
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    bins = np.logspace(np.log10(4), np.log10(30000), 60)
    for v, c, lab in [(len_tw, npl.PAL["data"], "tweet-uri financiare" if ro else "finance tweets"),
                      (len_pb, npl.PAL["benchmark"], "propoziții Financial PhraseBank" if ro else "Financial PhraseBank sentences"),
                      (len_sp, npl.PAL["model"], "discursuri BCE" if ro else "ECB speeches")]:
        ax.hist(v, bins=bins, weights=np.full(len(v), 100.0 / len(v)), histtype="stepfilled", alpha=0.45, color=c, label=lab)
    ax.axvline(512, color=npl.INK, lw=1, ls="--", label="fereastra BERT: 512 tokeni" if ro else "BERT window: 512 tokens")
    ax.set_xscale("log")
    ax.set_xlabel("lungimea textului în tokeni (scară logaritmică)" if ro else "text length in tokens (log scale)")
    ax.set_ylabel("% din texte" if ro else "% of texts")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "context_lengths")

    # 8.2 positional encoding (mechanism)
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    d = 64
    p = np.arange(MAXLEN)[:, None]
    div = np.exp(np.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = np.zeros((MAXLEN, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(p * div), np.cos(p * div)
    im = ax.imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.set_xlabel("poziția tokenului $t$" if ro else "token position $t$")
    ax.set_ylabel("dimensiunea $i$" if ro else "dimension $i$")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=9)
    npl.out(fig, HERE, lang, "positional_encoding")

    # 8.3 attention example (layer 6 and last layer)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1))
    tl = [t_.replace("##", "") for t_ in toks]
    for ax, A_, lab in [(axs[0], A_mid, "stratul 6" if ro else "layer 6"), (axs[1], A_last, "stratul 12" if ro else "layer 12")]:
        ax.imshow(A_, cmap="Oranges", vmin=0, vmax=min(1, A_.max()))
        ax.set_xticks(range(len(tl)))
        ax.set_xticklabels(tl, rotation=90, fontsize=7.5)
        ax.set_yticks(range(len(tl)))
        ax.set_yticklabels(tl, fontsize=7.5)
        ax.set_xlabel(lab + (": cheile (keys)" if ro else ": keys"), fontsize=9)
    axs[0].set_ylabel("interogările (queries)" if ro else "queries", fontsize=9)
    npl.out(fig, HERE, lang, "attention_example")

    # 8.4 attention entropy and attention on [SEP] by layer
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    axs[0].plot(ae.layer, ae.entropy, "o-", color=npl.PAL["model"], label="entropia normalizată" if ro else "normalised entropy")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("stratul" if ro else "layer")
    axs[0].set_ylabel("entropie / log n" if ro else "entropy / log n")
    axs[1].plot(ae.layer, 100 * ae.on_sep, "s-", color=npl.PAL["data"], label="atenția pe [SEP]" if ro else "attention on [SEP]")
    axs[1].plot(ae.layer, 100 * ae.on_cls, "^-", color=npl.PAL["alt"], label="atenția pe [CLS]" if ro else "attention on [CLS]")
    axs[1].set_xlabel("stratul" if ro else "layer")
    axs[1].set_ylabel("ponderea atenției (%)" if ro else "attention weight (%)")
    for a_ in axs:
        a_.set_xticks([1, 3, 6, 9, 12])
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "attention_entropy")

    # 8.5 learning curve of the small transformer (seed 1)
    h = np.array(TC["tiny"]["hist"][0])
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.plot(h[:, 0], h[:, 1], "o-", color=npl.PAL["train"], label="antrenare" if ro else "training")
    ax.plot(h[:, 0], h[:, 2], "s-", color=npl.PAL["valid"], label="validare" if ro else "validation")
    be = int(h[np.argmax(h[:, 3]), 0])
    ax.axvline(be, color=npl.MUTED, ls="--", lw=1, label="epoca reținută" if ro else "epoch kept")
    ax.set_xlabel("epoca" if ro else "epoch")
    ax.set_ylabel("entropia încrucișată" if ro else "cross-entropy")
    ax.set_xticks(h[:, 0].astype(int))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "tiny_learning_curve")
    if lang == "en":
        N["models"]["tiny"]["best_epoch_seed1"] = be
        N["models"]["tiny"]["minloss_epoch_seed1"] = int(h[np.argmin(h[:, 2]), 0])
        N["models"]["tiny"]["minloss_seed1"] = round(float(h[:, 2].min()), 3)
        N["models"]["tiny"]["lastloss_seed1"] = round(float(h[-1, 2]), 3)
        N["models"]["tiny"]["epochs_seed1"] = int(h[-1, 0])
        pd.DataFrame(h, columns=["epoch", "train_loss", "val_loss", "val_macro_f1"]).round(6).to_csv(os.path.join(RES, "tiny_history.csv"), index=False)

    # 8.6 model comparison
    lab = {"en": {"tfidf": "TF-IDF + logistic", "embbag": "Embedding bag (no attention)", "tiny_nopos": "Transformer, no positions",
                  "tiny": "Small transformer", "finbert_zs": "FinBERT zero-shot", "finbert_probe": "FinBERT frozen + logistic",
                  "finbert_ft": "FinBERT fine-tuned"},
           "ro": {"tfidf": "TF-IDF + logistic", "embbag": "Medie de embedding-uri (fără atenție)", "tiny_nopos": "Transformer fără poziții",
                  "tiny": "Transformer mic", "finbert_zs": "FinBERT fără antrenare", "finbert_probe": "FinBERT înghețat + logistic",
                  "finbert_ft": "FinBERT ajustat (fine-tuned)"}}[lang]
    col = {"tfidf": npl.PAL["benchmark"], "embbag": npl.MUTED, "tiny_nopos": npl.PAL["alt"], "tiny": npl.PAL["model"],
           "finbert_zs": npl.PAL["c5"], "finbert_probe": npl.PAL["c7"], "finbert_ft": npl.PAL["c6"]}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    v = [N["models"][m_]["macro_f1"] for m_ in MODELS]
    e = [N["models"][m_]["macro_f1_sd"] for m_ in MODELS]
    ax.barh(range(len(MODELS)), v, xerr=e, color=[col[m_] for m_ in MODELS], height=0.62, error_kw={"lw": 1, "capsize": 2})
    for i, x in enumerate(v):
        ax.text(x + 1.2, i, (f"{x:.1f}".replace(".", ",") if ro else f"{x:.1f}"), va="center", fontsize=9.5)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([lab[m_] for m_ in MODELS], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("macro-F1 pe tweet-urile de test (%)" if ro else "macro-F1 on the test tweets (%)")
    npl.out(fig, HERE, lang, "model_comparison", also_x=[ax])

    # 8.7 confusion matrix (FinBERT fine-tuned, seed 1)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
    cn = ["bearish", "bullish", "neutral"] if not ro else ["negativ (bearish)", "pozitiv (bullish)", "neutru"]
    for ax, M_, ttl in [(axs[0], cm_zs, "FinBERT fără antrenare" if ro else "FinBERT zero-shot"),
                        (axs[1], cm, "FinBERT ajustat" if ro else "FinBERT fine-tuned")]:
        Mn = M_ / M_.sum(1, keepdims=True)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M_[i, j]:,}".replace(",", "." if ro else ","), ha="center", va="center",
                        color="white" if Mn[i, j] > 0.6 else npl.INK, fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels([c_.split(" ")[0] for c_ in cn], fontsize=8.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels([c_.split(" ")[0] for c_ in cn], fontsize=8.5)
        ax.set_xlabel(("prezis" if ro else "predicted") + "\n" + ttl, fontsize=9)
    axs[0].set_ylabel("observat" if ro else "observed", fontsize=9)
    npl.out(fig, HERE, lang, "confusion")

    # 8.8 data scaling
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.semilogx(sc.n, 100 * sc.tfidf, "o-", color=npl.PAL["benchmark"], label="TF-IDF + logistic")
    ax.semilogx(sc.n, 100 * sc.tiny, "s-", color=npl.PAL["model"], label="transformer mic" if ro else "small transformer")
    ax.semilogx(sc.n, 100 * sc.probe, "^-", color=npl.PAL["c7"], label="FinBERT înghețat + logistic" if ro else "FinBERT frozen + logistic")
    ax.set_xlabel("tweet-uri de antrenare (scară logaritmică)" if ro else "training tweets (log scale)")
    ax.set_ylabel("macro-F1, test (%)")
    ax.set_xticks(sc.n)
    ax.set_xticklabels([f"{int(x):,}".replace(",", "." if ro else ",") for x in sc.n])
    ax.minorticks_off()
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "data_scaling")

npl.chart_meta(N, "context_lengths", "tweets", "finance tweets; Financial PhraseBank; European Central Bank speeches " + npl.period(N["len"]["sp_first"], N["len"]["sp_last"], "en"),
               "tweet-uri financiare; Financial PhraseBank; discursurile Băncii Centrale Europene " + npl.period(N["len"]["sp_first"], N["len"]["sp_last"], "ro"),
               source_en="Twitter financial news (zeroshot); Malo et al. (2014); European Central Bank speeches; tokenizer of the ProsusAI model",
               source_ro="Twitter financial news (zeroshot); Malo et al. (2014); discursurile Băncii Centrale Europene; tokenizatorul modelului ProsusAI")
npl.chart_meta(N, "positional_encoding", "Vaswani et al. (2017), formula", "mechanism illustration, $d=64$, 64 positions",
               "ilustrarea mecanismului, $d=64$, 64 de poziții", source_en="Vaswani et al. (2017), sinusoidal formula",
               source_ro="Vaswani et al. (2017), formula sinusoidală")
npl.chart_meta(N, "attention_example", "tweets", "one test tweet; pretrained FinBERT, attention averaged over 12 heads",
               "un tweet de test; FinBERT preantrenat, atenția mediată pe 12 capete")
npl.chart_meta(N, "attention_entropy", "tweets", "500 test tweets; pretrained FinBERT", "500 de tweet-uri de test; FinBERT preantrenat")
npl.chart_meta(N, "tiny_learning_curve", "tweets", "own estimates; small transformer, seed 1", "estimări proprii; transformer mic, sămînța 1")
for nm in ["model_comparison", "confusion", "data_scaling"]:
    npl.chart_meta(N, nm, "tweets", "own estimates; " + PER_TW["en"], "estimări proprii; " + PER_TW["ro"])
N["runtime_min"] = round((time.time() - t0) / 60, 1)
with open(os.path.join(HERE, "L7_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print("wrote L7_numbers.json and", len(N["charts"]), "charts x 2 languages in", N["runtime_min"], "min")
