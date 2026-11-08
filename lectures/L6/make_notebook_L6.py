"""
make_notebook_L6.py -- Lecture 6 notebooks (NLP foundations and word embeddings), EN + RO, executed:
notebooks/L6_notebook[_ro].ipynb and the seminar notebooks L6_seminar[_ro].ipynb (+ _solutions, git-ignored).
One split and fewer epochs than analysis_L6.py, so numbers are close to, not identical with, the slides.
Run:  python3 lectures/L6/make_notebook_L6.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("NLP foundations: text representation and word embeddings",
         "Bazele NLP: reprezentarea textului și reprezentarea vectorială a cuvintelor")

LOAD = r"""import re
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*")
def tok(s):
    s = re.sub(r'https?://\S+|@\w+', ' ', str(s).lower())
    return ['<num>' if t[0].isdigit() else t for t in TOK.findall(s)]
pb = pd.read_csv(DATA + 'financial_phrasebank.csv')
tw = pd.read_csv(DATA + 'twitter_fin_sentiment.csv')
sp = pd.read_csv(DATA + 'ecb_speeches_2022plus.csv', parse_dates=['date'])
LAB = ['negative', 'neutral', 'positive']
pb_y = pb.label.map({k: i for i, k in enumerate(LAB)}).values
pb_tok = [tok(s) for s in pb.sentence]
print(len(pb), len(tw), len(sp))"""

SPLIT = r"""rs = np.random.RandomState(SEED); tr, va, te = [], [], []
for c in range(3):
    ids = rs.permutation(np.where(pb_y == c)[0]); n = len(ids)
    tr += list(ids[:int(.7 * n)]); va += list(ids[int(.7 * n):int(.85 * n)]); te += list(ids[int(.85 * n):])
tr, va, te = map(np.array, (tr, va, te))
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
J = lambda ids: [' '.join(pb_tok[i]) for i in ids]
def lr_tuned(Xtr, ytr, Xva, yva, Xte):
    best = max(((f1_score(yva, LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(Xtr, ytr).predict(Xva), average='macro'), C)
                for C in [0.1, 1.0, 10.0]))
    m = LogisticRegression(C=best[1], max_iter=3000, class_weight='balanced').fit(Xtr, ytr)
    return m.predict(Xte), m
print(len(tr), len(va), len(te))"""

SGNS = r"""docs = [tok(t) for t in sp.contents]
cnt = Counter(w for d in docs for w in d)
vocab = [w for w, c in cnt.most_common() if c >= 5]; w2i = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
counts = np.array([cnt[w] for w in vocab], float); f = counts / counts.sum()
keep = np.minimum(1, np.sqrt(1e-4 / f) + 1e-4 / f)
ids = [np.array([w2i[w] for w in d if w in w2i]) for d in docs]
rng = np.random.RandomState(SEED); torch.manual_seed(SEED)
DIM, WIN, NEG = 100, 5, 5
Win, Wout = nn.Embedding(V, DIM), nn.Embedding(V, DIM)
nn.init.uniform_(Win.weight, -.5 / DIM, .5 / DIM); nn.init.zeros_(Wout.weight)
noise = torch.tensor(counts ** .75 / (counts ** .75).sum(), dtype=torch.float)
opt = torch.optim.Adam(list(Win.parameters()) + list(Wout.parameters()), lr=0.003)
for ep in range(3):
    cen, ctx = [], []
    for d in ids:
        d = d[rng.rand(len(d)) < keep[d]]
        for off in range(1, WIN + 1):
            if off >= len(d): break
            b = rng.randint(1, WIN + 1, len(d) - off) >= off
            cen += [d[:-off][b], d[off:][b]]; ctx += [d[off:][b], d[:-off][b]]
    cen, ctx = torch.tensor(np.concatenate(cen)), torch.tensor(np.concatenate(ctx)); perm = torch.randperm(len(cen)); tot = 0
    for j in range(0, len(perm), 4096):
        b = perm[j:j + 4096]; v = Win(cen[b]); ng = torch.multinomial(noise, len(b) * NEG, replacement=True).view(len(b), NEG)
        loss = -(F.logsigmoid((v * Wout(ctx[b])).sum(1)) + F.logsigmoid(-(Wout(ng) @ v.unsqueeze(2)).squeeze(2)).sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    print('epoch', ep + 1, 'loss', round(tot / (len(perm) / 4096), 3), 'pairs', len(cen))
E = Win.weight.detach().numpy(); En = E / np.linalg.norm(E, axis=1, keepdims=True)
def nearest(w, k=5):
    s = En @ En[w2i[w]]; s[w2i[w]] = -1; return [(vocab[i], round(float(s[i]), 2)) for i in np.argsort(-s)[:k]]"""


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 Tokens and Zipf's law", "## §1 Unități lexicale și legea lui Zipf"))
    nb.code(LOAD)
    nb.code("print(pb.sentence[3]); print(tok(pb.sentence[3]))\ncnt_all = Counter(w for t in sp.contents for w in tok(t))\n"
            "freq = np.array(sorted(cnt_all.values(), reverse=True), float); rk = np.arange(1, len(freq) + 1); s = (rk >= 10) & (rk <= 10000)\n"
            "print('tokens', int(freq.sum()), 'types', len(freq), 'Zipf slope', round(np.polyfit(np.log(rk[s]), np.log(freq[s]), 1)[0], 3))\n"
            "plt.loglog(rk, freq); plt.xlabel('rank'); plt.ylabel('frequency'); plt.show()")
    nb.md(T("## §2 Our own word2vec on ECB speeches", "## §2 Propriul nostru word2vec pe discursurile BCE") + "\n\n" +
          T("Skip-gram with negative sampling: $\\ell=-\\log\\sigma(\\mathbf u_o^\\top\\mathbf v_w)-\\sum_k\\log\\sigma(-\\mathbf u_{n_k}^\\top\\mathbf v_w)$. Three epochs here (five on the slides).",
            "Skip-gram cu eșantionare negativă: $\\ell=-\\log\\sigma(\\mathbf u_o^\\top\\mathbf v_w)-\\sum_k\\log\\sigma(-\\mathbf u_{n_k}^\\top\\mathbf v_w)$. Aici, trei epoci (pe slide-uri, cinci)."))
    nb.code(SGNS)
    nb.code("for w in ['inflation', 'energy', 'banks', 'climate', 'digital']:\n    print(w, nearest(w))")
    nb.md(T("## §3 Sentiment on the Financial PhraseBank", "## §3 Sentimentul pe Financial PhraseBank"))
    nb.code(SPLIT)
    nb.code("res = {}\nmaj = np.bincount(pb_y[tr]).argmax(); res['majority'] = np.full(len(te), maj)\n"
            "bow = CountVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, min_df=2).fit(J(tr))\n"
            "res['bow_lr'], mb = lr_tuned(bow.transform(J(tr)), pb_y[tr], bow.transform(J(va)), pb_y[va], bow.transform(J(te)))\n"
            "tf = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(J(tr))\n"
            "res['tfidf_lr'], _ = lr_tuned(tf.transform(J(tr)), pb_y[tr], tf.transform(J(va)), pb_y[va], tf.transform(J(te)))\n"
            "me = lambda I: np.array([E[[w2i[w] for w in pb_tok[i] if w in w2i]].mean(0) if any(w in w2i for w in pb_tok[i]) else np.zeros(DIM) for i in I])\n"
            "res['w2v_lr'], _ = lr_tuned(me(tr), pb_y[tr], me(va), pb_y[va], me(te))\n"
            "for k, p in res.items():\n    print(f'{k:10s} macro-F1 {100 * f1_score(pb_y[te], p, average=\"macro\"):.2f}  accuracy {100 * accuracy_score(pb_y[te], p):.2f}')\n"
            "voc = bow.get_feature_names_out(); d = mb.coef_[2] - mb.coef_[0]\nprint('positive:', list(voc[np.argsort(-d)[:8]])); print('negative:', list(voc[np.argsort(d)[:8]]))")
    nb.md(T("## §4 An embedding-bag network", "## §4 O rețea embedding bag") + "\n\n" +
          T("$\\mathbf z = \\frac1{|S|}\\sum_{s\\in S}\\mathbf e_s$, then a softmax layer; random start or start from our word2vec.",
            "$\\mathbf z = \\frac1{|S|}\\sum_{s\\in S}\\mathbf e_s$, apoi un strat softmax; start aleator sau start din word2vec-ul nostru."))
    nb.code("def embbag(init, dim=100, epochs=30):\n    torch.manual_seed(SEED); np.random.seed(SEED)\n"
            "    ng = lambda t: t + [a + '_' + b for a, b in zip(t, t[1:])]\n    voc = {'<pad>': 0}\n"
            "    if init:\n        for w in vocab: voc[w] = len(voc)\n"
            "    for w, k in Counter(x for i in tr for x in ng(pb_tok[i])).items():\n        if k >= 2 and w not in voc: voc[w] = len(voc)\n"
            "    emb = nn.EmbeddingBag(len(voc), dim, mode='mean'); out = nn.Linear(dim, 3)\n"
            "    if init:\n        with torch.no_grad(): emb.weight[1:V + 1] = torch.tensor(E / np.linalg.norm(E, axis=1, keepdims=True).mean())\n"
            "    def enc(I):\n        ids, offs = [], []\n        for i in I:\n            offs.append(len(ids)); ids += [voc[w] for w in ng(pb_tok[i]) if w in voc] or [0]\n"
            "        return torch.tensor(ids), torch.tensor(offs)\n"
            "    params = list(emb.parameters()) + list(out.parameters()); opt = torch.optim.Adam(params, lr=5e-3)\n"
            "    w = torch.tensor(len(tr) / (3 * np.bincount(pb_y[tr])), dtype=torch.float); best, state, bad = -1, None, 0\n"
            "    for ep in range(epochs):\n        perm = np.random.permutation(tr)\n        for j in range(0, len(perm), 64):\n"
            "            b = perm[j:j + 64]; loss = F.cross_entropy(out(F.dropout(emb(*enc(b)), .3)), torch.tensor(pb_y[b]), weight=w)\n"
            "            opt.zero_grad(); loss.backward(); opt.step()\n"
            "        with torch.no_grad(): f = f1_score(pb_y[va], out(emb(*enc(va))).argmax(1).numpy(), average='macro')\n"
            "        if f > best: best, bad, state = f, 0, ([p.clone() for p in params])\n        else:\n            bad += 1\n            if bad >= 4: break\n"
            "    with torch.no_grad():\n        for p, q in zip(params, state): p.copy_(q)\n        return out(emb(*enc(te))).argmax(1).numpy()\n"
            "for init in (False, True):\n    p = embbag(init); print('embedding bag', 'word2vec start' if init else 'random start', 'macro-F1', round(100 * f1_score(pb_y[te], p, average='macro'), 2))")
    nb.md(T("## §5 Topics of finance tweets", "## §5 Temele tweet-urilor financiare"))
    nb.code("tp = pd.read_csv(DATA + 'twitter_fin_topic.csv'); a, b = tp[tp.split == 'train'], tp[tp.split == 'valid']\n"
            "tf2 = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(a.text)\n"
            "m = LogisticRegression(C=10, max_iter=3000, class_weight='balanced').fit(tf2.transform(a.text), a.label)\np = m.predict(tf2.transform(b.text))\n"
            "print('topic macro-F1', round(100 * f1_score(b.label, p, average='macro'), 2), ' accuracy', round(100 * accuracy_score(b.label, p), 2))")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(LOAD)
    nb.code(SPLIT)
    nb.task(T("Compute by hand the TF-IDF weight (sublinear tf, smoothed idf) of a word that occurs twice in a sentence and in 30 of the 3,000 training sentences; check with `TfidfVectorizer` on a toy corpus.",
              "Calculați de mînă ponderea TF-IDF (tf sublinear, idf netezit) a unui cuvînt care apare de două ori într-o frază și în 30 din cele 3.000 de fraze de antrenare; verificați cu `TfidfVectorizer` pe un corpus mic."))
    nb.solution("print('tf =', round(1 + np.log(2), 4), ' idf =', round(np.log(3001 / 31) + 1, 4), ' product =', round((1 + np.log(2)) * (np.log(3001 / 31) + 1), 4))\n"
                "toy = ['profit rose rose', 'profit fell', 'sales were flat']\nv = TfidfVectorizer(sublinear_tf=True, norm=None).fit(toy)\n"
                "print(dict(zip(v.get_feature_names_out(), np.round(v.transform(toy[:1]).toarray()[0], 4))))")
    nb.task(T("Remove English stop words (`stop_words='english'`) from the TF-IDF model of the lecture. What happens to the macro-F1, and which removed words carried sentiment?",
              "Eliminați cuvintele de legătură din engleză (`stop_words='english'`) din modelul TF-IDF al cursului. Ce se întîmplă cu macro-F1 și ce cuvinte eliminate purtau sentiment?"))
    nb.solution("from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS\nfor sw in (None, 'english'):\n"
                "    tf = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=sw).fit(J(tr))\n"
                "    p, _ = lr_tuned(tf.transform(J(tr)), pb_y[tr], tf.transform(J(va)), pb_y[va], tf.transform(J(te)))\n"
                "    print(sw, round(100 * f1_score(pb_y[te], p, average='macro'), 2))\nprint([w for w in ['up', 'down', 'not', 'less', 'more', 'no'] if w in ENGLISH_STOP_WORDS])")
    nb.task(T("Train the TF-IDF + logistic model only on sentences with 100% annotator agreement and test on the same test set. Is cleaner training data worth fewer examples?",
              "Antrenați modelul TF-IDF + logistic numai pe frazele cu acord unanim al adnotatorilor și testați pe același set de test. Merită datele mai curate numărul mai mic de exemple?"))
    nb.solution("clean = tr[pb.min_agreement_pct.values[tr] == 100]\n"
                "tf = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(J(clean))\n"
                "p, _ = lr_tuned(tf.transform(J(clean)), pb_y[clean], tf.transform(J(va)), pb_y[va], tf.transform(J(te)))\n"
                "print(len(clean), 'of', len(tr), 'training sentences; macro-F1', round(100 * f1_score(pb_y[te], p, average='macro'), 2))")
    nb.task(T("Apply the tweet-sentiment model (TF-IDF + logistic trained on the tweets' train split) to the PhraseBank test sentences, mapping bearish/neutral/bullish to negative/neutral/positive. How much does the domain change cost?",
              "Aplicați modelul de sentiment pentru tweet-uri (TF-IDF + logistic antrenat pe setul de antrenare al tweet-urilor) frazelor de test din PhraseBank, cu corespondența pesimist/neutru/optimist → negativ/neutru/pozitiv. Cît costă schimbarea domeniului?"))
    nb.solution("a = tw[tw.split == 'train']; ya = a.label_name.map({'bearish': 0, 'neutral': 1, 'bullish': 2}).values\n"
                "tf = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(a.text)\n"
                "m = LogisticRegression(C=1, max_iter=3000, class_weight='balanced').fit(tf.transform(a.text), ya)\n"
                "p = m.predict(tf.transform(pb.sentence.values[te])); print('cross-domain macro-F1', round(100 * f1_score(pb_y[te], p, average='macro'), 2))")
    nb.task(T("Build a sentiment index of ECB speeches: apply the PhraseBank TF-IDF model to every sentence of each speech, average (positive − negative) by quarter and plot it.",
              "Construiți un indice de sentiment al discursurilor BCE: aplicați modelul TF-IDF PhraseBank fiecărei fraze din fiecare discurs, mediați (pozitiv − negativ) pe trimestre și reprezentați grafic."))
    nb.solution("tf = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(J(tr))\n"
                "_, m = lr_tuned(tf.transform(J(tr)), pb_y[tr], tf.transform(J(va)), pb_y[va], tf.transform(J(te)))\nscore = []\n"
                "for txt in sp.contents:\n    sents = [' '.join(tok(s)) for s in re.split(r'(?<=[.!?])\\s+', txt) if len(s) > 30]\n"
                "    pr = m.predict_proba(tf.transform(sents)); score.append(float((pr[:, 2] - pr[:, 0]).mean()))\n"
                "idx = pd.Series(score, index=sp.date).sort_index().resample('QE').mean()\nidx.plot(marker='o'); plt.ylabel('mean P(pos) - P(neg)'); plt.show(); print(idx.round(3))")


if __name__ == "__main__":
    Notebook.build_all(6, TITLE, lecture, kind="lecture")
    Notebook.build_all(6, TITLE, seminar, kind="seminar")
