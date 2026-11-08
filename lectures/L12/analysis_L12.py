"""
analysis_L12.py -- Lecture 12: large language models (LLM) in finance and prompt engineering.
All computation of the lecture: numbers -> L12_numbers.json, charts -> charts/en|ro/*.pdf.

Real data only (data/clean, see data/SOURCES.md): Financial PhraseBank (Malo et al., 2014), Twitter financial news
sentiment (zeroshot, Hugging Face), speeches of the European Central Bank (ECB) 2022-2026, S&P 500 daily closes.
Open models (Hugging Face): Qwen2.5-0.5B/1.5B/3B-Instruct (Qwen Team, 2024), BAAI bge-small-en-v1.5 (embeddings),
ProsusAI FinBERT tokenizer (WordPiece). Everything runs locally (Apple MPS); no paid application programming interface.

Model outputs (label probabilities, generated answers, Value-at-Risk numbers) are cached in lectures/L12/cache
(git-ignored) and the per-item results needed by the Quantlets are written to lectures/L12/results/*.csv (small).
Run:  python3 lectures/L12/analysis_L12.py        (first run about 30-40 min on an Apple M-series laptop, then seconds)
"""
import copy
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
npl.style()
SEED = npl.seed_all()
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(HERE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RES, exist_ok=True)
N = {"charts": {}, "seed": SEED}
T0 = time.time()

import torch  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split, GridSearchCV  # noqa: E402

DEV = npl.device()
QW = {"0.5B": "Qwen/Qwen2.5-0.5B-Instruct", "1.5B": "Qwen/Qwen2.5-1.5B-Instruct", "3B": "Qwen/Qwen2.5-3B-Instruct"}
QW_PARAMS = {"0.5B": 0.49, "1.5B": 1.54, "3B": 3.09}      # billions of parameters (model cards)
N["models"] = {k: {"params_b": v} for k, v in QW_PARAMS.items()}


def L(lang, en, ro):
    return en if lang == "en" else ro


def fmt_num(v, lang, d=2):
    s = f"{v:.{d}f}"
    return s if lang == "en" else s.replace(".", ",")


_LM = {}


def lm(size):
    """Load a Qwen2.5 instruct model once (float16 on MPS/CUDA, float32 on CPU)."""
    if size not in _LM:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _LM.clear()
        if DEV == "mps":
            torch.mps.empty_cache()
        tok = AutoTokenizer.from_pretrained(QW[size])
        m = AutoModelForCausalLM.from_pretrained(QW[size], dtype=torch.float16 if DEV != "cpu" else torch.float32).to(DEV).eval()
        _LM[size] = (tok, m)
    return _LM[size]


def jcache(name, fn):
    path = os.path.join(CACHE, name)
    if os.path.isfile(path):
        return json.load(open(path))
    t = time.time()
    out = fn()
    out = out if isinstance(out, dict) else {"v": out}
    out["_seconds"] = round(time.time() - t, 2)
    json.dump(out, open(path, "w"))
    print(f"cached {name} in {out['_seconds']:.0f} s")
    return out


# ---------------------------------------------------------------- 0. data and splits (fixed a priori)
PB = npl.phrasebank(50)                                   # all 4,838 sentences with their annotator-agreement level
PB100 = PB[PB.min_agreement_pct == 100].reset_index(drop=True)
pb_tr, pb_te = train_test_split(PB100, test_size=0.30, stratify=PB100.label, random_state=SEED)
pb_tr, pb_te = pb_tr.reset_index(drop=True), pb_te.reset_index(drop=True)
TW = npl.tweets_sentiment()
tw_tr = TW[TW.split == "train"].reset_index(drop=True)
tw_va = TW[TW.split == "valid"]
tw_te = tw_va.groupby("label_name", group_keys=False).apply(lambda g: g.sample(frac=900 / len(tw_va), random_state=SEED)).reset_index(drop=True)
N["data"] = {"pb_n": int(len(PB)), "pb100_n": int(len(PB100)), "pb_tr": int(len(pb_tr)), "pb_te": int(len(pb_te)),
             "pb_share_neutral": round(100 * float((PB100.label == "neutral").mean()), 1),
             "pb_share_pos": round(100 * float((PB100.label == "positive").mean()), 1),
             "pb_share_neg": round(100 * float((PB100.label == "negative").mean()), 1),
             "tw_tr": int(len(tw_tr)), "tw_va": int(len(tw_va)), "tw_te": int(len(tw_te)),
             "tw_share_neutral": round(100 * float((tw_te.label_name == "neutral").mean()), 1)}
for lev in (50, 66, 75, 100):
    N["data"][f"pb_lev{lev}"] = int((PB.min_agreement_pct == lev).sum())
LAB = {"pb": ["positive", "negative", "neutral"], "tw": ["bullish", "bearish", "neutral"]}
SETS = {"pb": (pb_tr.sentence.tolist(), pb_tr.label.tolist(), pb_te.sentence.tolist(), pb_te.label.tolist()),
        "tw": (tw_tr.text.tolist(), tw_tr.label_name.tolist(), tw_te.text.tolist(), tw_te.label_name.tolist())}

# ---------------------------------------------------------------- 1. tokenisation (charts: tokens per sentence)
from transformers import AutoTokenizer  # noqa: E402

tq = AutoTokenizer.from_pretrained(QW["1.5B"])
tb = AutoTokenizer.from_pretrained("ProsusAI/finbert")
nw = PB.sentence.str.split().str.len().values
nq = np.array([len(tq.encode(s)) for s in PB.sentence])
nb = np.array([len(tb.encode(s, add_special_tokens=False)) for s in PB.sentence])
SP = npl.ecb_speeches()
sp_last = SP.sort_values("date").iloc[-1]
N["tok"] = {"vocab_qwen": int(len(tq)), "vocab_bert": int(len(tb)), "words_mean": round(float(nw.mean()), 1),
            "qwen_mean": round(float(nq.mean()), 1), "bert_mean": round(float(nb.mean()), 1),
            "qwen_per_word": round(float(nq.sum() / nw.sum()), 2), "bert_per_word": round(float(nb.sum() / nw.sum()), 2)}
sp_words = SP.contents.str.split().str.len()
sp_tok = np.array([len(tq.encode(c)) for c in SP.contents])
N["tok"]["ecb_tokens_mean"] = int(round(float(sp_tok.mean()), 0))
N["tok"]["ecb_tokens_max"] = int(sp_tok.max())
N["tok"]["ecb_per_word"] = round(float(sp_tok.sum() / sp_words.sum()), 2)
N["tok"]["ecb_total_m"] = round(float(sp_tok.sum()) / 1e6, 2)
ex = PB100.sentence.iloc[int(np.argmin(np.abs(PB100.sentence.str.split().str.len() - 14)))]
N["tok"]["example"] = ex
N["tok"]["example_qwen"] = " | ".join(tq.convert_tokens_to_string([t]).replace("|", "/") for t in tq.tokenize(ex))
N["tok"]["example_n_qwen"] = len(tq.tokenize(ex))
N["tok"]["example_n_bert"] = len(tb.tokenize(ex))
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    bins = np.arange(0, 90, 2)
    ax.hist(nw, bins=bins, color=npl.PAL["data"], alpha=0.55, label=L(lang, "Words", "Cuvinte"))
    ax.hist(nq, bins=bins, histtype="step", lw=1.5, color=npl.PAL["model"], label=L(lang, "Qwen2.5 tokens (byte-pair encoding)", "Tokeni Qwen2.5 (codificare byte-pair)"))
    ax.hist(nb, bins=bins, histtype="step", lw=1.5, color=npl.PAL["benchmark"], label=L(lang, "FinBERT tokens (WordPiece)", "Tokeni FinBERT (WordPiece)"))
    ax.set_xlabel(L(lang, "Length of a sentence", "Lungimea unei propoziții"))
    ax.set_ylabel(L(lang, "Number of sentences", "Numărul de propoziții"))
    ax.set_xlim(0, 88)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "tokens_per_sentence")
npl.chart_meta(N, "tokens_per_sentence", "phrasebank", f"{len(PB):,} sentences", f"{len(PB):,} de propoziții".replace(",", "."))

# ---------------------------------------------------------------- 2. next-token distribution and temperature (real prompt from an ECB speech)
sents = re.split(r"(?<=[.!?])\s+", sp_last.contents)
cand = [s for s in sents if s.lower().startswith("inflation") and 10 <= len(s.split()) <= 40]
if not cand:
    cand = [s for s in sents if " inflation " in s.lower() and 10 <= len(s.split()) <= 40]
base = cand[0]
words = base.split()
cut = min(len(words) - 2, max(5, [i for i, w in enumerate(words) if w.lower().startswith("inflation")][0] + 3))
PROMPT = " ".join(words[:cut])
N["next"] = {"prompt": PROMPT, "speech_date": str(sp_last.date.date()), "speaker": sp_last.speakers, "true_next": words[cut]}


def nexttok():
    tok, m = lm("1.5B")
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        lg = m(ids).logits[0, -1].float().cpu()
    out = {}
    for T in (0.5, 1.0, 2.0):
        p = torch.softmax(lg / T, -1)
        top = torch.topk(p, 8)
        ent = float(-(p * torch.log(p.clamp_min(1e-30))).sum())
        out[str(T)] = {"tok": [tok.decode([i]) for i in top.indices.tolist()], "p": [float(x) for x in top.values], "entropy": ent}
    return out


NT = jcache("next_token.json", nexttok)
N["next"]["entropy"] = {T: round(NT[T]["entropy"], 2) for T in ("0.5", "1.0", "2.0")}
N["next"]["top1"] = NT["1.0"]["tok"][0].strip()
N["next"]["top1_p"] = round(100 * NT["1.0"]["p"][0], 1)
N["next"]["top1_p05"] = round(100 * NT["0.5"]["p"][0], 1)
N["next"]["top1_p2"] = round(100 * NT["2.0"]["p"][0], 1)
toks = NT["1.0"]["tok"]
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    w = 0.27
    for j, (T, c) in enumerate([("0.5", npl.PAL["data"]), ("1.0", npl.PAL["model"]), ("2.0", npl.PAL["benchmark"])]):
        lg = {t: p for t, p in zip(NT[T]["tok"], NT[T]["p"])}
        ax.bar(np.arange(len(toks)) + (j - 1) * w, [100 * lg.get(t, np.nan) for t in toks], w, color=c,
               label=L(lang, f"temperature {T}", f"temperatura {T.replace('.', ',')}"))
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels([repr(t.replace(" ", "␣"))[1:-1] for t in toks], fontsize=9)
    ax.set_ylabel(L(lang, "Probability (%)", "Probabilitatea (%)"))
    ax.set_xlabel(L(lang, "Most likely next tokens at temperature 1 (␣ = space)", "Cei mai probabili tokeni următori la temperatura 1 (␣ = spațiu)"))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "next_token")
npl.chart_meta(N, "next_token", "speeches", f"prompt from the ECB speech of {npl.period(sp_last.date, sp_last.date, 'en').split(' to ')[0]}; model Qwen2.5-1.5B-Instruct",
               f"începutul unei fraze din discursul BCE din {npl.period(sp_last.date, sp_last.date, 'ro').split(' - ')[0]}; modelul Qwen2.5-1.5B-Instruct")

# ---------------------------------------------------------------- 3. perplexity on ECB speeches: model size and training cut-off
def body(row):
    c = row.contents
    head = f"SPEECH {row.title} {row.subtitle}"
    if c.startswith(head):
        c = c[len(head):]
    m = re.search(r"\b(19|20)\d{2}\b", c[:120])
    return c[m.end():].strip() if m else c


SP = SP.sort_values("date").reset_index(drop=True)
SP["body"] = SP.apply(body, axis=1)
old = SP[(SP.date < "2024-01-01")].sample(40, random_state=SEED)
new = SP[(SP.date >= "2025-07-01")].sample(40, random_state=SEED)
N["ppl_data"] = {"old_first": str(old.date.min().date()), "old_last": str(old.date.max().date()),
                 "new_first": str(new.date.min().date()), "new_last": str(new.date.max().date()), "n_each": 40, "n_tokens": 1024}


def ppl_size(size):
    tok, m = lm(size)
    out = {}
    for name, df in (("old", old), ("new", new)):
        nll, cnt = 0.0, 0
        for txt in df.body:
            ids = tok(txt, return_tensors="pt").input_ids[:, :1024].to(DEV)
            with torch.no_grad():
                lo = m(ids).logits[0, :-1].float()
            l = torch.nn.functional.cross_entropy(lo, ids[0, 1:], reduction="sum")
            nll += float(l)
            cnt += ids.shape[1] - 1
        out[name] = nll / cnt
    return out


PPL = {s: jcache(f"ppl_{s}.json", lambda s=s: ppl_size(s)) for s in ("0.5B", "1.5B", "3B")}
N["ppl"] = {s: {"old": round(float(np.exp(PPL[s]["old"])), 2), "new": round(float(np.exp(PPL[s]["new"])), 2),
                "nll_new": round(PPL[s]["new"], 3)} for s in PPL}
# own scaling estimate: log loss = a + b log(params) on the three sizes (post-cut-off speeches)
xs = np.log([QW_PARAMS[s] for s in ("0.5B", "1.5B", "3B")])
ys = np.log([PPL[s]["new"] for s in ("0.5B", "1.5B", "3B")])
b_, a_ = np.polyfit(xs, ys, 1)
N["ppl_slope"] = round(float(b_), 3)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    sizes = ["0.5B", "1.5B", "3B"]
    w = 0.36
    ax.bar(np.arange(3) - w / 2, [N["ppl"][s]["old"] for s in sizes], w, color=npl.PAL["data"],
           label=L(lang, "Speeches of 2022-2023 (before the model's training cut-off)", "Discursuri din 2022-2023 (înainte de limita datelor de antrenare)"))
    ax.bar(np.arange(3) + w / 2, [N["ppl"][s]["new"] for s in sizes], w, color=npl.PAL["model"],
           label=L(lang, "Speeches from July 2025 (after the cut-off)", "Discursuri din iulie 2025 (după limită)"))
    for i, s in enumerate(sizes):
        for dx, k in ((-w / 2, "old"), (w / 2, "new")):
            ax.text(i + dx, N["ppl"][s][k] + 0.1, fmt_num(N["ppl"][s][k], lang), ha="center", fontsize=8.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Qwen2.5-{s} ({fmt_num(QW_PARAMS[s], lang)} " + L(lang, "bn parameters)", "mld. parametri)") for s in sizes], fontsize=9)
    ax.set_ylabel(L(lang, "Perplexity", "Perplexitatea"))
    ax.set_ylim(0, max(N["ppl"][s]["old"] for s in sizes) * 1.2)
    npl.legend_bottom(fig, ncol=1)
    npl.out(fig, HERE, lang, "perplexity_size")
npl.chart_meta(N, "perplexity_size", "speeches", "40 + 40 speeches, first 1,024 tokens of each", "40 + 40 de discursuri, primii 1.024 de tokeni din fiecare")

# ---------------------------------------------------------------- 4. prompting: label probabilities from the next-token distribution
INSTR = {
    "pb": ["Classify the sentiment of this sentence from financial news, from the point of view of an investor in the company. "
           "Answer with one word: positive, negative or neutral.",
           "Is the following financial news sentence positive, negative or neutral for the company's investors? Reply with a single word.",
           "You read a sentence from a company press release. Label its sentiment as positive, negative or neutral. One word only.",
           "Sentiment analysis. Possible labels: neutral, negative, positive. Give only the label for the sentence below.",
           "Would this news make a shareholder feel positive, negative or neutral? Answer with exactly one of these three words."],
    "tw": ["Classify this finance tweet as bullish, bearish or neutral for the market. Answer with one word.",
           "Is the following tweet bullish, bearish or neutral? Reply with a single word.",
           "You read a tweet about markets. Label it bullish, bearish or neutral. One word only.",
           "Market sentiment. Possible labels: neutral, bearish, bullish. Give only the label for the tweet below.",
           "Would this tweet make a trader bullish, bearish or neutral? Answer with exactly one of these three words."],
}
SYS = "You are a careful financial analyst."
FIELD = {"pb": "Sentence", "tw": "Tweet"}


def label_ids(tok, labs):
    ids = {}
    for l in labs:
        s = set()
        for v in (l, " " + l, l.capitalize(), " " + l.capitalize()):
            s.add(tok.encode(v)[0])
        ids[l] = sorted(s)
    return ids


def build_prompt(tok, ds, text, instr, shots=()):
    user = instr + "\n\n"
    for st, sl in shots:
        user += f"{FIELD[ds]}: {st}\nAnswer: {sl}\n\n"
    user += f"{FIELD[ds]}: {text}\nAnswer:"
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score(size, ds, texts, instr, shots=(), bs=16):
    """Probability of each label = probability mass of its first token (any capitalisation) at the answer position,
    renormalised over the labels. Shared prompt prefix (instruction + examples) is encoded once (key-value cache)."""
    tok, m = lm(size)
    labs = LAB[ds]
    ids = label_ids(tok, labs)
    prompts = [build_prompt(tok, ds, t, instr, shots) for t in texts]
    pre = os.path.commonprefix(prompts)
    pre = pre[:pre.rfind("\n") + 1] if "\n" in pre else ""
    P = []
    with torch.no_grad():
        cache = None
        if pre:
            pi = tok(pre, return_tensors="pt").input_ids.to(DEV)
            cache = m(pi, use_cache=True).past_key_values
        for j in range(0, len(prompts), bs):
            suf = [p[len(pre):] for p in prompts[j:j + bs]]
            enc = tok(suf, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False).to(DEV)
            n = len(suf)
            if cache is not None:
                c2 = copy.deepcopy(cache)
                c2.batch_repeat_interleave(n)
                am = torch.cat([torch.ones(n, pi.shape[1], device=DEV, dtype=enc.attention_mask.dtype), enc.attention_mask], 1)
                out = m(input_ids=enc.input_ids, attention_mask=am, past_key_values=c2, use_cache=True)
            else:
                out = m(**enc)
            last = enc.attention_mask.sum(1) - 1
            pr = torch.softmax(out.logits[torch.arange(n), last].float(), -1)
            P.append(torch.stack([pr[:, ids[l]].sum(1) for l in labs], 1).cpu().numpy())
    P = np.concatenate(P)
    return (P / P.sum(1, keepdims=True)).tolist()


def shots_for(ds, k, seed):
    """k examples per class from the training pool, in random order (seeded)."""
    Xtr, ytr = SETS[ds][0], SETS[ds][1]
    rng = np.random.default_rng(seed)
    idx = []
    for l in LAB[ds]:
        cand = [i for i, y in enumerate(ytr) if y == l and len(Xtr[i]) < 220]
        idx += list(rng.choice(cand, k, replace=False))
    rng.shuffle(idx)
    return [(Xtr[i], ytr[i]) for i in idx]


def metrics(ds, P, y=None, labs=None):
    labs = labs or LAB[ds]
    y = y if y is not None else SETS[ds][3]
    pred = [labs[i] for i in np.argmax(np.array(P), 1)]
    return {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred, labels=labs, average="macro")}


def boot_ci(ds, pred, y=None, B=1000):
    y = np.array(y if y is not None else SETS[ds][3])
    pred = np.array(pred)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[i], pred[i], labels=LAB[ds], average="macro"))
    return np.percentile(vals, [2.5, 97.5])


RESULTS = {}           # (ds, method) -> list of probability rows
TIMES = {}             # (ds, method) -> seconds per item
for ds in ("pb", "tw"):
    Xtr, ytr, Xte, yte = SETS[ds]
    # majority class
    maj = max(set(ytr), key=ytr.count)
    RESULTS[(ds, "majority")] = [[1.0 if l == maj else 0.0 for l in LAB[ds]] for _ in Xte]
    # TF-IDF + logistic regression (trained on the training pool; C by 5-fold cross-validation)
    t = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    A = vec.fit_transform(Xtr)
    gs = GridSearchCV(LogisticRegression(max_iter=3000, class_weight="balanced"), {"C": [0.3, 1, 3, 10, 30]}, cv=5,
                      scoring="f1_macro").fit(A, ytr)
    pr = gs.predict_proba(vec.transform(Xte))
    TIMES[(ds, "tfidf")] = (time.time() - t) / len(Xte)
    RESULTS[(ds, "tfidf")] = pr[:, [list(gs.classes_).index(l) for l in LAB[ds]]].tolist()
    N.setdefault("tfidf", {})[ds] = {"C": float(gs.best_params_["C"]), "n_features": int(A.shape[1])}
    # zero-shot with three model sizes
    for s in ("0.5B", "1.5B", "3B"):
        r = jcache(f"zs_{ds}_{s}.json", lambda s=s: {"P": score(s, ds, Xte, INSTR[ds][0])})
        RESULTS[(ds, f"zs_{s}")] = r["P"]
        TIMES[(ds, f"zs_{s}")] = r["_seconds"] / len(Xte)
    # few-shot: 4 examples per class (12 in the prompt), three seeds for the choice of examples
    for s in ("1.5B", "3B"):
        for sd in (0, 1, 2):
            r = jcache(f"fs_{ds}_{s}_k4_s{sd}.json", lambda s=s, sd=sd: {"P": score(s, ds, Xte, INSTR[ds][0], shots_for(ds, 4, SEED + sd))})
            RESULTS[(ds, f"fs_{s}_s{sd}")] = r["P"]
            TIMES[(ds, f"fs_{s}")] = r["_seconds"] / len(Xte)

# few-shot curve (Financial PhraseBank, 1.5B and 3B, k = 0, 1, 2, 4, 8 per class, 3 seeds)
KS = [0, 1, 2, 4, 8]
FS = {}
for s in ("1.5B", "3B"):
    for k in KS:
        for sd in (0, 1, 2):
            if k == 0:
                P = RESULTS[("pb", f"zs_{s}")]
            elif k == 4:
                P = RESULTS[("pb", f"fs_{s}_s{sd}")]
            else:
                P = jcache(f"fs_pb_{s}_k{k}_s{sd}.json", lambda s=s, k=k, sd=sd: {"P": score(s, "pb", SETS["pb"][2], INSTR["pb"][0], shots_for("pb", k, SEED + sd))})["P"]
            FS[(s, k, sd)] = metrics("pb", P)["f1"]

# prompt sensitivity: five paraphrases of the instruction, zero-shot 3B and 1.5B, both datasets
SENS = {}
for ds in ("pb", "tw"):
    for s in ("1.5B", "3B"):
        for i, ins in enumerate(INSTR[ds]):
            P = RESULTS[(ds, f"zs_{s}")] if i == 0 else jcache(f"sens_{ds}_{s}_{i}.json", lambda s=s, ins=ins, ds=ds: {"P": score(s, ds, SETS[ds][2], ins)})["P"]
            SENS[(ds, s, i)] = metrics(ds, P)["f1"]


# chain-of-thought (zero-shot, 3B): the model writes two short reasoning sentences, then the label
def cot(ds, texts, size="3B", bs=16):
    tok, m = lm(size)
    labs = LAB[ds]
    ins = (INSTR[ds][0].split(" Answer")[0] + f" First explain your reasoning in at most two short sentences, then give the final "
           f"answer on a new line as 'Label: <{'|'.join(labs)}>'.")
    outs = []
    with torch.no_grad():
        for j in range(0, len(texts), bs):
            ps = [tok.apply_chat_template([{"role": "system", "content": SYS},
                                           {"role": "user", "content": ins + f"\n\n{FIELD[ds]}: {t}"}], tokenize=False,
                                          add_generation_prompt=True) for t in texts[j:j + bs]]
            enc = tok(ps, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(DEV)
            g = m.generate(**enc, max_new_tokens=90, do_sample=False, pad_token_id=tok.eos_token_id)
            outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return {"text": outs}


def parse_label(txt, labs):
    m_ = re.search(r"label\s*:\s*\**\s*([a-zA-Z]+)", txt, re.I)
    if m_ and m_.group(1).lower() in labs:
        return m_.group(1).lower()
    found = [(txt.lower().rfind(l), l) for l in labs if l in txt.lower()]
    return max(found)[1] if found else None


for ds in ("pb", "tw"):
    r = jcache(f"cot_{ds}_3B.json", lambda ds=ds: cot(ds, SETS[ds][2]))
    lab = [parse_label(x, LAB[ds]) for x in r["text"]]
    N.setdefault("cot", {})[ds] = {"parsed_pct": round(100 * np.mean([x is not None for x in lab]), 1)}
    lab = [x if x is not None else "neutral" for x in lab]
    RESULTS[(ds, "cot_3B")] = [[1.0 if l == x else 0.0 for l in LAB[ds]] for x in lab]
    TIMES[(ds, "cot_3B")] = r["_seconds"] / len(SETS[ds][2])
    k = int(np.argmax([len(x) for x in r["text"]][:40]))
    N["cot"][ds]["example_text"] = SETS[ds][2][k]
    N["cot"][ds]["example_out"] = r["text"][k].strip().replace("\n", " ")
    N["cot"][ds]["mean_words"] = round(float(np.mean([len(x.split()) for x in r["text"]])), 1)


# embedding probe: mean-pooled hidden states of Qwen2.5-1.5B + logistic regression (a light form of fine-tuning)
def embed(size, texts, bs=32):
    tok, m = lm(size)
    E = []
    with torch.no_grad():
        for j in range(0, len(texts), bs):
            enc = tok(texts[j:j + bs], return_tensors="pt", padding=True, padding_side="right").to(DEV)
            h = m(**enc, output_hidden_states=True).hidden_states[-1].float()
            msk = enc.attention_mask.unsqueeze(-1).float()
            E.append(((h * msk).sum(1) / msk.sum(1)).cpu().numpy())
    return np.concatenate(E)


for ds in ("pb", "tw"):
    path = os.path.join(CACHE, f"emb_{ds}.npz")
    if not os.path.isfile(path):
        t = time.time()
        Etr, Ete = embed("1.5B", SETS[ds][0]), embed("1.5B", SETS[ds][2])
        np.savez(path, tr=Etr, te=Ete, sec=time.time() - t)
    z = np.load(path)
    mu, sd = z["tr"].mean(0), z["tr"].std(0) + 1e-6
    gs = GridSearchCV(LogisticRegression(max_iter=4000, class_weight="balanced"), {"C": [0.01, 0.03, 0.1, 0.3]}, cv=5,
                      scoring="f1_macro").fit((z["tr"] - mu) / sd, SETS[ds][1])
    pr = gs.predict_proba((z["te"] - mu) / sd)
    RESULTS[(ds, "probe")] = pr[:, [list(gs.classes_).index(l) for l in LAB[ds]]].tolist()
    TIMES[(ds, "probe")] = float(z["sec"]) / (len(SETS[ds][0]) + len(SETS[ds][2]))
    N.setdefault("probe", {})[ds] = {"C": float(gs.best_params_["C"]), "dim": int(z["tr"].shape[1])}

# ---------------------------------------------------------------- 5. results table, bootstrap intervals, charts
METH = ["majority", "tfidf", "zs_0.5B", "zs_1.5B", "zs_3B", "fs_1.5B", "fs_3B", "cot_3B", "probe"]
MNAME = {"majority": ("Majority class", "Clasa majoritară"), "tfidf": ("TF-IDF + logistic", "TF-IDF + logistică"),
         "zs_0.5B": ("Zero-shot 0.5B", "Zero-shot 0,5B"), "zs_1.5B": ("Zero-shot 1.5B", "Zero-shot 1,5B"),
         "zs_3B": ("Zero-shot 3B", "Zero-shot 3B"), "fs_1.5B": ("Few-shot 1.5B (12 ex.)", "Few-shot 1,5B (12 ex.)"),
         "fs_3B": ("Few-shot 3B (12 ex.)", "Few-shot 3B (12 ex.)"), "cot_3B": ("Chain-of-thought 3B", "Chain-of-thought 3B"),
         "probe": ("Embeddings 1.5B + logistic", "Embedding 1,5B + logistică")}
N["cls"] = {}
for ds in ("pb", "tw"):
    N["cls"][ds] = {}
    labs = LAB[ds]
    for mth in METH:
        if mth.startswith("fs_"):
            runs = [metrics(ds, RESULTS[(ds, f"{mth}_s{sd}")]) for sd in (0, 1, 2)]
            P0 = RESULTS[(ds, f"{mth}_s0")]
            f1s = [r["f1"] for r in runs]
            pred = [labs[i] for i in np.argmax(np.array(P0), 1)]
            lo, hi = boot_ci(ds, pred)
            N["cls"][ds][mth] = {"f1": round(100 * float(np.mean(f1s)), 1), "acc": round(100 * float(np.mean([r["acc"] for r in runs])), 1),
                                 "f1_min": round(100 * min(f1s), 1), "f1_max": round(100 * max(f1s), 1),
                                 "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
        else:
            P = RESULTS[(ds, mth)]
            r = metrics(ds, P)
            pred = [labs[i] for i in np.argmax(np.array(P), 1)]
            lo, hi = boot_ci(ds, pred)
            N["cls"][ds][mth] = {"f1": round(100 * r["f1"], 1), "acc": round(100 * r["acc"], 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
        N["cls"][ds][mth]["ms"] = round(1000 * TIMES.get((ds, mth), 0.0), 2)
    best = max((m for m in METH if m != "majority"), key=lambda m: N["cls"][ds][m]["f1"])
    N["cls"][ds]["best"] = best
    N["cls"][ds]["best_llm"] = max((m for m in METH if m not in ("majority", "tfidf", "probe")), key=lambda m: N["cls"][ds][m]["f1"])
    pd.DataFrame({"text": SETS[ds][2], "label": SETS[ds][3],
                  **{f"{m}_{l}": np.array(RESULTS[(ds, m if not m.startswith('fs_') else m + '_s0')])[:, j]
                     for m in METH if m != "majority" for j, l in enumerate(labs)}}).to_csv(os.path.join(RES, f"predictions_{ds}.csv"), index=False)

for lang in ("en", "ro"):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    order = METH
    for ax, ds in zip(axs, ("pb", "tw")):
        v = [N["cls"][ds][m]["f1"] for m in order]
        lo = [N["cls"][ds][m]["lo"] for m in order]
        hi = [N["cls"][ds][m]["hi"] for m in order]
        col = [npl.PAL["benchmark"] if m in ("majority", "tfidf") else (npl.PAL["c7"] if m == "probe" else npl.PAL["model"]) for m in order]
        ax.barh(range(len(order)), v, color=col, xerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)],
                error_kw={"lw": 0.8, "capsize": 2, "ecolor": "#555555"})
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([MNAME[m][0 if lang == "en" else 1] for m in order], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel(L(lang, "Macro-F1 (%)", "Macro-F1 (%)") + (" - PhraseBank" if ds == "pb" else " - Twitter"), fontsize=9.5)
    from matplotlib.patches import Patch
    npl.legend_bottom(fig, [Patch(color=npl.PAL["benchmark"]), Patch(color=npl.PAL["model"]), Patch(color=npl.PAL["c7"])],
                      [L(lang, "Benchmarks", "Repere"), L(lang, "Prompted LLM", "LLM cu prompt"),
                       L(lang, "LLM embeddings + trained classifier", "Embedding LLM + clasificator antrenat")], ncol=3)
    npl.out(fig, HERE, lang, "prompt_results", also_x=list(axs))
npl.chart_meta(N, "prompt_results", "Financial PhraseBank (Malo et al., 2014) and Twitter financial news (zeroshot)",
               f"test sets: {len(pb_te)} sentences (100\\,\\% agreement) and {len(tw_te)} tweets; 95\\,\\% bootstrap intervals",
               f"seturi de test: {len(pb_te)} de propoziții (acord 100\\,\\%) și {len(tw_te)} de tweet-uri; intervale bootstrap de 95\\,\\%",
               source_ro="Financial PhraseBank (Malo et al., 2014) și Twitter financial news (zeroshot)")

N["fewshot"] = {s: {str(k): {"mean": round(100 * float(np.mean([FS[(s, k, sd)] for sd in (0, 1, 2)])), 1),
                             "min": round(100 * min(FS[(s, k, sd)] for sd in (0, 1, 2)), 1),
                             "max": round(100 * max(FS[(s, k, sd)] for sd in (0, 1, 2)), 1)} for k in KS} for s in ("1.5B", "3B")}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for s, c in (("1.5B", npl.PAL["data"]), ("3B", npl.PAL["model"])):
        mean = [N["fewshot"][s][str(k)]["mean"] for k in KS]
        ax.plot(range(len(KS)), mean, marker="o", color=c, label=f"Qwen2.5-{s}" if lang == "en" else f"Qwen2.5-{s.replace('.', ',')}")
        for i, k in enumerate(KS):
            ax.plot([i, i], [N["fewshot"][s][str(k)]["min"], N["fewshot"][s][str(k)]["max"]], color=c, lw=2.5, alpha=0.4)
    ax.axhline(N["cls"]["pb"]["tfidf"]["f1"], color=npl.PAL["benchmark"], ls="--", lw=1,
               label=L(lang, f"TF-IDF + logistic ({len(pb_tr):,} labelled sentences)", f"TF-IDF + logistică ({len(pb_tr):,} de propoziții etichetate)".replace(",", ".")))
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([str(3 * k) for k in KS])
    ax.set_xlabel(L(lang, "Examples in the prompt (balanced over the 3 classes)", "Exemple în prompt (echilibrate pe cele 3 clase)"))
    ax.set_ylabel(L(lang, "Macro-F1 (%)", "Macro-F1 (%)"))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "fewshot_curve")
npl.chart_meta(N, "fewshot_curve", "phrasebank", f"test set of {len(pb_te)} sentences; bars = range over 3 seeds",
               f"setul de test de {len(pb_te)} de propoziții; barele = intervalul pe 3 valori ale generatorului aleator")

N["sens"] = {}
for ds in ("pb", "tw"):
    for s in ("1.5B", "3B"):
        v = [100 * SENS[(ds, s, i)] for i in range(5)]
        N["sens"][f"{ds}_{s}"] = {"min": round(min(v), 1), "max": round(max(v), 1), "range": round(max(v) - min(v), 1),
                                  "vals": [round(x, 1) for x in v]}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    rows = [("pb", "1.5B"), ("pb", "3B"), ("tw", "1.5B"), ("tw", "3B")]
    for r_, (ds, s) in enumerate(rows):
        v = N["sens"][f"{ds}_{s}"]["vals"]
        ax.plot([min(v), max(v)], [r_, r_], color="#bbbbbb", lw=3, zorder=1)
        for i, x in enumerate(v):
            ax.scatter(x, r_, s=36, color=npl.SERIES[i], zorder=2, label=L(lang, f"Instruction {i + 1}", f"Instrucțiunea {i + 1}") if r_ == 0 else None)
    ax.set_yticks(range(4))
    ax.set_yticklabels([("PhraseBank" if ds == "pb" else "Twitter") + f", {s if lang == 'en' else s.replace('.', ',')}" for ds, s in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(L(lang, "Macro-F1 (%), zero-shot, same model and data", "Macro-F1 (%), zero-shot, același model și aceleași date"))
    npl.legend_bottom(fig, ncol=5)
    npl.out(fig, HERE, lang, "prompt_sensitivity", also_x=[ax])
npl.chart_meta(N, "prompt_sensitivity", "Financial PhraseBank and Twitter financial news", "test sets as above; five paraphrases of the instruction",
               "seturile de test de mai sus; cinci reformulări ale instrucțiunii", source_ro="Financial PhraseBank și Twitter financial news")


# calibration: reliability of the 3B zero-shot probabilities versus TF-IDF + logistic
def reliability(P, y, labs, bins=10):
    P = np.array(P)
    conf = P.max(1)
    ok = np.array([labs[i] == t for i, t in zip(P.argmax(1), y)])
    edges = np.linspace(1 / 3, 1, bins + 1)
    rows, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        msk = (conf >= a) & (conf <= b if b == 1 else conf < b)
        if msk.sum() >= 5:
            rows.append((conf[msk].mean(), ok[msk].mean(), msk.sum()))
        if msk.sum():
            ece += msk.mean() * abs(conf[msk].mean() - ok[msk].mean())
    return rows, ece


N["calib"] = {}
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot([1 / 3, 1], [1 / 3, 1], color="#999999", lw=1, ls=":", label=L(lang, "Perfect calibration", "Calibrare perfectă"))
    for mth, c in (("zs_3B", npl.PAL["model"]), ("fs_3B_s0", npl.PAL["alt"]), ("tfidf", npl.PAL["benchmark"])):
        rows, ece = reliability(RESULTS[("pb", mth)], SETS["pb"][3], LAB["pb"])
        N["calib"][mth] = round(100 * ece, 1)
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", color=c,
                label={"zs_3B": L(lang, "Zero-shot 3B", "Zero-shot 3B"), "fs_3B_s0": L(lang, "Few-shot 3B", "Few-shot 3B"),
                       "tfidf": L(lang, "TF-IDF + logistic", "TF-IDF + logistică")}[mth] + f" (ECE {fmt_num(100 * ece, lang, 1)} %)")
    ax.set_xlabel(L(lang, "Predicted probability of the chosen label", "Probabilitatea prezisă a etichetei alese"))
    ax.set_ylabel(L(lang, "Observed accuracy", "Acuratețea observată"))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "calibration", also_x=[ax])
npl.chart_meta(N, "calibration", "phrasebank", f"test set of {len(pb_te)} sentences, 10 bins", f"setul de test de {len(pb_te)} de propoziții, 10 intervale")

# accuracy by annotator agreement (the 100 % test part + all sentences with 50-75 % agreement, none used in training)
amb = PB[PB.min_agreement_pct < 100].reset_index(drop=True)
P_amb = {mth: jcache(f"amb_{mth}.json", lambda mth=mth: {"P": score(mth.split("_")[1], "pb", amb.sentence.tolist(), INSTR["pb"][0],
                                                                      shots_for("pb", 4, SEED) if mth.startswith("fs") else ())})["P"]
         for mth in ("zs_3B", "fs_3B")}
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
A = vec.fit_transform(SETS["pb"][0])
lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=N["tfidf"]["pb"]["C"]).fit(A, SETS["pb"][1])
P_amb["tfidf"] = lr.predict_proba(vec.transform(amb.sentence))[:, [list(lr.classes_).index(l) for l in LAB["pb"]]].tolist()
N["agree"] = {}
for mth in ("tfidf", "zs_3B", "fs_3B"):
    N["agree"][mth] = {}
    for lev in (100, 75, 66, 50):
        if lev == 100:
            P = RESULTS[("pb", mth if mth != "fs_3B" else "fs_3B_s0")]
            y = SETS["pb"][3]
        else:
            msk = (amb.min_agreement_pct == lev).values
            P = np.array(P_amb[mth])[msk]
            y = amb.label[msk].tolist()
        N["agree"][mth][str(lev)] = round(100 * metrics("pb", P, y)["acc"], 1)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    levs = [100, 75, 66, 50]
    w = 0.27
    for j, (mth, c, lab) in enumerate((("tfidf", npl.PAL["benchmark"], L(lang, "TF-IDF + logistic", "TF-IDF + logistică")),
                                      ("zs_3B", npl.PAL["model"], "Zero-shot 3B"), ("fs_3B", npl.PAL["alt"], "Few-shot 3B"))):
        ax.bar(np.arange(4) + (j - 1) * w, [N["agree"][mth][str(l)] for l in levs], w, color=c, label=lab)
    ax.set_xticks(range(4))
    ax.set_xticklabels([L(lang, f"{l} % agree", f"acord {l} %") + f"\n(n = {N['data']['pb_lev' + str(l)] if l < 100 else len(pb_te)})" for l in levs], fontsize=9)
    ax.set_ylabel(L(lang, "Accuracy (%)", "Acuratețea (%)"))
    ax.set_ylim(0, 100)
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "agreement")
npl.chart_meta(N, "agreement", "phrasebank", "sentences never used for training or for examples",
               "propoziții nefolosite nici la antrenare, nici ca exemple")

# confusion matrix of the best prompted model (Financial PhraseBank)
bl = N["cls"]["pb"]["best_llm"]
P = RESULTS[("pb", bl if not bl.startswith("fs_") else bl + "_s0")]
pred = [LAB["pb"][i] for i in np.argmax(np.array(P), 1)]
cm = confusion_matrix(SETS["pb"][3], pred, labels=LAB["pb"])
N["confusion"] = {"model": bl, "m": cm.tolist()}

# cost-accuracy: measured time per item on this machine
N["hardware"] = "Apple silicon, PyTorch " + torch.__version__.split("+")[0] + f", device {DEV}"
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for mth in METH[1:]:
        x, y = N["cls"]["pb"][mth]["ms"], N["cls"]["pb"][mth]["f1"]
        c = npl.PAL["benchmark"] if mth == "tfidf" else (npl.PAL["c7"] if mth == "probe" else npl.PAL["model"])
        ax.scatter(max(x, 0.01), y, s=40, color=c, zorder=3)
        ax.annotate(MNAME[mth][0 if lang == "en" else 1], (max(x, 0.01), y), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(L(lang, "Milliseconds per sentence on a laptop (log scale)", "Milisecunde pe propoziție, pe un laptop (scară logaritmică)"))
    ax.set_ylabel(L(lang, "Macro-F1 (%)", "Macro-F1 (%)"))
    ax.set_xlim(0.005, 2000)
    from matplotlib.lines import Line2D
    npl.legend_bottom(fig, [Line2D([], [], marker="o", ls="", color=npl.PAL["benchmark"]), Line2D([], [], marker="o", ls="", color=npl.PAL["model"]),
                            Line2D([], [], marker="o", ls="", color=npl.PAL["c7"])],
                      [L(lang, "Benchmark", "Reper"), L(lang, "Prompted LLM", "LLM cu prompt"), L(lang, "Embeddings + classifier", "Embedding + clasificator")], ncol=3)
    if lang == "ro":
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: npl.ro_number(v)))
    npl.out(fig, HERE, lang, "cost_accuracy")
npl.chart_meta(N, "cost_accuracy", "phrasebank", f"test set of {len(pb_te)} sentences; time measured on a laptop graphics processor",
               f"setul de test de {len(pb_te)} de propoziții; timp măsurat pe procesorul grafic al unui laptop")

# ---------------------------------------------------------------- 6. retrieval-augmented generation (RAG) on ECB speeches
def chunks(text, size=200, overlap=50):
    w = text.split()
    out = []
    for i in range(0, max(1, len(w) - overlap), size - overlap):
        out.append(" ".join(w[i:i + size]))
        if i + size >= len(w):
            break
    return out


C_txt, C_sp = [], []
for i, r in SP.iterrows():
    for c in chunks(r.body):
        C_txt.append(c)
        C_sp.append(i)
C_sp = np.array(C_sp)
N["rag"] = {"n_speeches": int(len(SP)), "n_chunks": int(len(C_txt)), "chunk_words": 200, "overlap": 50}
queries = SP.title.tolist()

# lexical: TF-IDF cosine and BM25
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
Ct = vec.fit_transform(C_txt)
S_tfidf = (vec.transform(queries) @ Ct.T).toarray()


def bm25(qs, docs, k1=1.5, b=0.75):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words="english")
    D = cv.fit_transform(docs).tocsc().astype(float)
    dl = np.asarray(D.sum(1)).ravel()
    avg = dl.mean()
    df = np.asarray((D > 0).sum(0)).ravel()
    idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
    Q = cv.transform(qs)
    S = np.zeros((len(qs), len(docs)))
    D = D.tocsr()
    for qi in range(len(qs)):
        terms = Q[qi].indices
        for tj in terms:
            col = D[:, tj].toarray().ravel()
            S[qi] += idf[tj] * col * (k1 + 1) / (col + k1 * (1 - b + b * dl / avg))
    return S


S_bm25 = bm25(queries, C_txt)


def dense():
    from transformers import AutoModel
    tk = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    md = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(DEV).eval()

    def enc(txts, pre=""):
        out = []
        with torch.no_grad():
            for j in range(0, len(txts), 64):
                e = tk([pre + t for t in txts[j:j + 64]], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
                h = md(**e).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out)
    t = time.time()
    Cd = enc(C_txt)
    Qd = enc(queries, "Represent this sentence for searching relevant passages: ")
    return Qd, Cd, time.time() - t


dpath = os.path.join(CACHE, "rag_dense.npz")
if not os.path.isfile(dpath):
    Qd, Cd, sec = dense()
    np.savez(dpath, Q=Qd, C=Cd, sec=sec)
z = np.load(dpath)
S_dense = z["Q"] @ z["C"].T
N["rag"]["dense_dim"] = int(z["C"].shape[1])
N["rag"]["embed_seconds"] = round(float(z["sec"]), 1)


def speech_rank(S):
    """Rank of the correct speech when speeches are ordered by their best chunk."""
    ranks = []
    for qi in range(S.shape[0]):
        best = pd.Series(S[qi]).groupby(C_sp).max().sort_values(ascending=False)
        ranks.append(int(np.where(best.index.values == qi)[0][0]) + 1)
    return np.array(ranks)


def rrf(*Ss, k=60):
    out = np.zeros_like(Ss[0])
    for S in Ss:
        r = np.argsort(np.argsort(-S, 1), 1) + 1
        out += 1.0 / (k + r)
    return out


S_hyb = rrf(S_bm25, S_dense)
RANK = {"tfidf": speech_rank(S_tfidf), "bm25": speech_rank(S_bm25), "dense": speech_rank(S_dense), "hybrid": speech_rank(S_hyb)}
KR = [1, 3, 5, 10]
for k_, v in RANK.items():
    N["rag"][k_] = {f"r{k}": round(100 * float((v <= k).mean()), 1) for k in KR}
    N["rag"][k_]["mrr"] = round(float((1 / v).mean()), 3)
pd.DataFrame({"date": SP.date.dt.date, "speaker": SP.speakers, "title": SP.title, **{f"rank_{k}": v for k, v in RANK.items()}}).to_csv(
    os.path.join(RES, "rag_ranks.csv"), index=False)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    for k_, c, lab in (("tfidf", npl.PAL["benchmark"], "TF-IDF"), ("bm25", npl.SERIES[5], "BM25"),
                       ("dense", npl.PAL["model"], L(lang, "Dense embeddings (bge-small)", "Embedding dense (bge-small)")),
                       ("hybrid", npl.PAL["c7"], L(lang, "Hybrid (BM25 + dense)", "Hibrid (BM25 + dense)"))):
        ax.plot(KR, [N["rag"][k_][f"r{k}"] for k in KR], marker="o", color=c, label=lab)
    ax.set_xticks(KR)
    ax.set_xlabel(L(lang, "k (number of speeches returned)", "k (numărul de discursuri returnate)"))
    ax.set_ylabel(L(lang, "Recall@k (%)", "Recall@k (%)"))
    npl.legend_bottom(fig, ncol=4)
    npl.out(fig, HERE, lang, "rag_recall")
npl.chart_meta(N, "rag_recall", "speeches", f"{len(SP)} speeches, {len(C_txt)} chunks; query = speech title",
               f"{len(SP)} de discursuri, {len(C_txt)} de fragmente; interogarea = titlul discursului")

# question answering: which Executive Board member gave the speech? closed-book vs RAG vs oracle context
qa = SP[(SP.date >= "2025-01-01") & (~SP.speakers.str.contains(","))].reset_index()
N["rag"]["qa_n"] = int(len(qa))
N["rag"]["qa_first"] = str(qa.date.min().date())
board = sorted(SP.speakers[~SP.speakers.str.contains(",")].unique())


def qa_run(mode, size="3B", topk=3, bs=8):
    tok, m = lm(size)
    outs = []
    with torch.no_grad():
        prompts = []
        for _, r in qa.iterrows():
            q_forced = (f"Which member of the European Central Bank's Executive Board delivered the speech titled \"{r.title}\" in "
                        f"{r.date.year}? Answer with the person's full name only.")
            q_abst = q_forced + " If you do not know, answer \"I don't know\"."
            if mode == "closed":
                user = q_forced                         # forced answer: measures hallucination
            elif mode == "closed_abstain":
                user = q_abst                           # abstention allowed
            else:
                if mode == "rag":
                    best = pd.Series(S_hyb[r["index"]]).groupby(C_sp).idxmax()
                    order = pd.Series(S_hyb[r["index"]]).groupby(C_sp).max().sort_values(ascending=False).index[:topk]
                    ids = [int(best[o]) for o in order]
                else:
                    ids = [int(np.where(C_sp == r["index"])[0][0])]
                ctx = "\n\n".join(f"[Document {n + 1}] Speech by {SP.speakers[C_sp[i]]}, {SP.date[C_sp[i]].date()}: "
                                  f"\"{SP.title[C_sp[i]]}\"\n{C_txt[i][:800]}" for n, i in enumerate(ids))
                user = ("Below are excerpts of ECB speeches, each preceded by its speaker, date and title.\n\n" + ctx +
                        f"\n\nWho delivered the speech titled \"{r.title}\"? Answer with the full name only, or \"I don't know\" if the documents do not say.")
            prompts.append(tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                                   tokenize=False, add_generation_prompt=True))
        for j in range(0, len(prompts), bs):
            enc = tok(prompts[j:j + bs], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(DEV)
            g = m.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
            outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return {"text": outs}


def judge(ans, truth):
    a = ans.lower()
    if "don't know" in a or "do not know" in a or "unknown" in a:
        return "abstain"
    last = truth.split()[-1].lower()
    return "correct" if last in a else "wrong"


QA = {}
for mode in ("closed", "closed_abstain", "rag", "oracle"):
    r = jcache(f"qa2_{mode}.json", lambda mode=mode: qa_run(mode))
    QA[mode] = [judge(a, t) for a, t in zip(r["text"], qa.speakers)]
    N["rag"][f"qa_{mode}"] = {k: round(100 * float(np.mean([x == k for x in QA[mode]])), 1) for k in ("correct", "wrong", "abstain")}
    if mode == "closed":
        guess = pd.Series([a.strip() for a, j in zip(r["text"], QA[mode]) if j == "wrong"]).value_counts()
        N["rag"]["closed_top_wrong"] = guess.index[0] if len(guess) else ""
        N["rag"]["closed_top_wrong_n"] = int(guess.iloc[0]) if len(guess) else 0
maj_sp = qa.speakers.value_counts()
N["rag"]["qa_majority"] = {"name": maj_sp.index[0], "pct": round(100 * float(maj_sp.iloc[0] / len(qa)), 1)}
N["rag"]["qa_r3_hybrid"] = round(100 * float((RANK["hybrid"][qa["index"].values] <= 3).mean()), 1)
pd.DataFrame({"date": qa.date.dt.date, "speaker": qa.speakers, "title": qa.title, **{m: QA[m] for m in QA}}).to_csv(
    os.path.join(RES, "rag_qa.csv"), index=False)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    modes = [("closed", L(lang, "No documents, answer forced", "Fără documente, răspuns obligatoriu")),
             ("closed_abstain", L(lang, "No documents, \"I don't know\" allowed", "Fără documente, „nu știu” permis")),
             ("rag", L(lang, "RAG: top-3 retrieved speeches", "RAG: primele 3 discursuri regăsite")),
             ("oracle", L(lang, "Oracle: the right speech given", "Oracol: discursul corect furnizat"))]
    left = np.zeros(len(modes))
    for k_, c, lab in (("correct", npl.PAL["benchmark"], L(lang, "Correct", "Corect")),
                       ("wrong", npl.PAL["positive"], L(lang, "Wrong name (hallucination)", "Nume greșit (halucinație)")),
                       ("abstain", "#bbbbbb", L(lang, "\"I don't know\"", "„Nu știu”"))):
        v = np.array([N["rag"][f"qa_{m}"][k_] for m, _ in modes])
        ax.barh(range(len(modes)), v, left=left, color=c, label=lab)
        for i in range(len(modes)):
            if v[i] >= 6:
                ax.text(left[i] + v[i] / 2, i, fmt_num(v[i], lang, 1), ha="center", va="center", fontsize=8.5, color="white" if k_ != "abstain" else "#222222")
        left += v
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([m[1] for m in modes], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(L(lang, "Share of questions (%)", "Ponderea întrebărilor (%)"))
    npl.legend_bottom(fig, ncol=3)
    npl.out(fig, HERE, lang, "rag_qa", also_x=[ax])
npl.chart_meta(N, "rag_qa", "speeches", f"{len(qa)} single-speaker speeches from {npl.period(qa.date.min(), qa.date.max(), 'en')}; model Qwen2.5-3B-Instruct",
               f"{len(qa)} de discursuri cu un singur vorbitor, {npl.period(qa.date.min(), qa.date.max(), 'ro')}; modelul Qwen2.5-3B-Instruct")

# ---------------------------------------------------------------- 7. LLM-VaR with an open model (after Pele et al., 2026): S&P 500
from scipy import stats  # noqa: E402
from arch import arch_model  # noqa: E402

px = npl.markets("adjclose", ["SP500"])["SP500"].dropna()
r = 100 * np.log(px).diff().dropna()
r = r.loc[:"2026-10-31"]
VT = r.loc["2025-01-01":"2026-10-31"].index
WN = 250
N["llmvar"] = {"first": str(VT[0].date()), "last": str(VT[-1].date()), "n": int(len(VT)), "window": WN, "model": "Qwen2.5-3B-Instruct"}


def llm_var(alpha, size="3B", bs=8):
    tok, m = lm(size)
    outs = []
    prompts = []
    for d in VT:
        i = r.index.get_loc(d)
        w = r.values[i - WN:i]
        txt = ", ".join(f"{x:.2f}" for x in w)
        user = (f"Here are the last {WN} daily log returns of the S&P 500 index, in percent, from oldest to newest:\n{txt}\n\n"
                f"Estimate the Value-at-Risk at the {int(100 * alpha)}% level for tomorrow: the {alpha:.2f} quantile of tomorrow's "
                f"daily return distribution, in percent. Reply with a single number only (it will usually be negative).")
        prompts.append(tok.apply_chat_template([{"role": "system", "content": "You are a quantitative risk analyst."},
                                                {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True) +
                       f"The {alpha:.2f} quantile of tomorrow's return is ")    # answer prefix: the model must write the number
    with torch.no_grad():
        for j in range(0, len(prompts), bs):
            enc = tok(prompts[j:j + bs], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(DEV)
            g = m.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
            outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return {"text": outs}


def parse_num(s):
    m_ = re.search(r"[-−]?\d+(?:\.\d+)?", s.replace(",", "."))
    if not m_:
        return np.nan
    v = float(m_.group(0).replace("−", "-"))
    return -abs(v)


def kupiec(hits, a):
    n, x = len(hits), int(hits.sum())
    p = x / n
    lr = -2 * n * np.log(1 - a) if x == 0 else -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - p) - x * np.log(p))
    return float(1 - stats.chi2.cdf(lr, 1))


VAR = pd.DataFrame({"r": r.loc[VT].values}, index=VT)
for a, s in ((0.05, "05"), (0.01, "01")):
    out = jcache(f"llmvar2_{s}.json", lambda a=a: llm_var(a))["text"]
    v = np.array([parse_num(x) for x in out])
    N["llmvar"][f"parsed_{s}"] = round(100 * float(np.isfinite(v).mean()), 1)
    v = pd.Series(v, index=VT).ffill().values             # unparsable answer: keep the previous day's estimate
    VAR[f"llm_{s}"] = v
    VAR[f"hs_{s}"] = [np.quantile(r.values[r.index.get_loc(d) - WN:r.index.get_loc(d)], a) for d in VT]
gpath = os.path.join(CACHE, "garch_var.json")
if not os.path.isfile(gpath):
    g = {"01": [], "05": []}
    params = None
    for n_, d in enumerate(VT):
        i = r.index.get_loc(d)
        am = arch_model(r.values[i - 1000:i], mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        if n_ % 21 == 0 or params is None:
            params = am.fit(disp="off", show_warning=False).params
        f = am.fix(params).forecast(horizon=1, reindex=False)
        mu, s2, nu = float(f.mean.values[-1, 0]), float(f.variance.values[-1, 0]), float(params["nu"])
        sc = np.sqrt(s2 * (nu - 2) / nu)
        g["01"].append(mu + sc * stats.t.ppf(0.01, nu))
        g["05"].append(mu + sc * stats.t.ppf(0.05, nu))
    json.dump(g, open(gpath, "w"))
g = json.load(open(gpath))
VAR["garch_01"], VAR["garch_05"] = g["01"], g["05"]
VAR.index.name = "date"
VAR.to_csv(os.path.join(RES, "llmvar_sp500.csv"), float_format="%.4f")
for mname in ("llm", "hs", "garch"):
    for a, s in ((0.05, "05"), (0.01, "01")):
        hits = (VAR["r"] < VAR[f"{mname}_{s}"]).values
        N["llmvar"][f"{mname}_{s}"] = {"viol": int(hits.sum()), "rate": round(100 * hits.mean(), 2), "kupiec_p": round(kupiec(hits, a), 3),
                                       "mean": round(float(VAR[f"{mname}_{s}"].mean()), 2)}
N["llmvar"]["exp_05"] = round(0.05 * len(VT), 1)
N["llmvar"]["exp_01"] = round(0.01 * len(VT), 1)
N["llmvar"]["corr_llm_garch_05"] = round(float(VAR[["llm_05", "garch_05"]].corr().iloc[0, 1]), 2)
for lang in ("en", "ro"):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.plot(VAR.index, VAR["r"], color=npl.PAL["data"], lw=0.5, label=L(lang, "Daily log return (%)", "Randamentul logaritmic zilnic (%)"))
    ax.plot(VAR.index, VAR["garch_05"], color=npl.PAL["benchmark"], lw=1.0, label=L(lang, "5 % VaR, GARCH(1,1)-t", "VaR 5 %, GARCH(1,1)-t"))
    ax.plot(VAR.index, VAR["llm_05"], color=npl.PAL["model"], lw=1.0, label=L(lang, "5 % LLM-VaR, Qwen2.5-3B", "LLM-VaR 5 %, Qwen2.5-3B"))
    hit = VAR["r"] < VAR["llm_05"]
    ax.scatter(VAR.index[hit], VAR["r"][hit], s=10, color=npl.PAL["positive"], zorder=3, label=L(lang, "Violation of the LLM-VaR", "Depășire a LLM-VaR"))
    ax.set_ylabel("%")
    ax.set_xlim(VAR.index[0], VAR.index[-1])
    npl.date_axis(ax, lang)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "llm_var")
npl.chart_meta(N, "llm_var", "yahoo", npl.period(VT[0], VT[-1], "en") + ", S\\&P 500, trading days",
               npl.period(VT[0], VT[-1], "ro") + ", S\\&P 500, zile de tranzacționare")

# ---------------------------------------------------------------- 8. small result files for the Quantlets
def labels_of(P, ds):
    return [LAB[ds][i] for i in np.argmax(np.array(P), 1)]


fsl = {}
for s in ("1.5B", "3B"):
    for k in KS:
        for sd in (0, 1, 2):
            P = (RESULTS[("pb", f"zs_{s}")] if k == 0 else RESULTS[("pb", f"fs_{s}_s{sd}")] if k == 4 else
                 json.load(open(os.path.join(CACHE, f"fs_pb_{s}_k{k}_s{sd}.json")))["P"])
            fsl[f"{s}_k{k}_s{sd}"] = labels_of(P, "pb")
pd.DataFrame({"label": SETS["pb"][3], **fsl}).to_csv(os.path.join(RES, "fewshot_labels_pb.csv"), index=False)
for ds in ("pb", "tw"):
    sl = {}
    for s in ("1.5B", "3B"):
        for i in range(5):
            P = RESULTS[(ds, f"zs_{s}")] if i == 0 else json.load(open(os.path.join(CACHE, f"sens_{ds}_{s}_{i}.json")))["P"]
            sl[f"{s}_i{i + 1}"] = labels_of(P, ds)
    pd.DataFrame({"label": SETS[ds][3], **sl}).to_csv(os.path.join(RES, f"sensitivity_labels_{ds}.csv"), index=False)
pd.DataFrame({"sentence": amb.sentence, "label": amb.label, "agreement": amb.min_agreement_pct,
              **{m: labels_of(P_amb[m], "pb") for m in P_amb}}).to_csv(os.path.join(RES, "agreement_labels.csv"), index=False)
json.dump({"next_token": {T: NT[T] for T in ("0.5", "1.0", "2.0")}, "ppl_nll": {s: {k: PPL[s][k] for k in ("old", "new")} for s in PPL},
           "params_b": QW_PARAMS, "ms_per_item": {ds: {m: N["cls"][ds][m]["ms"] for m in METH} for ds in ("pb", "tw")}}, open(os.path.join(RES, "lm_outputs.json"), "w"), indent=1)

N["runtime_min"] = round((time.time() - T0) / 60, 1)


def _key(k):
    """No dots inside keys (the @{a.b.c} placeholders split on dots): 0.5B -> 05B, 1.5B -> 15B, temperature 0.5 -> t05."""
    k = k.replace("0.5B", "05B").replace("1.5B", "15B")
    return {"0.5": "t05", "1.0": "t10", "2.0": "t20"}.get(k, k)


def _clean(x):
    if isinstance(x, dict):
        return {_key(k): _clean(v) for k, v in x.items()}
    return x


N = _clean(N)
with open(os.path.join(HERE, "L12_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print("wrote L12_numbers.json and", len(N["charts"]), "charts x 2 languages;", f"{N['runtime_min']} min")
print(json.dumps(N["cls"], indent=0)[:3000])
