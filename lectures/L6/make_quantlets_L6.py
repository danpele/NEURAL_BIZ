"""
make_quantlets_L6.py -- one Quantlet per chart of Lecture 6: Quantlets/L6/NN_L6_<chart>/ (Metainfo.txt, executed notebook,
<chart>.png). The Zipf chart is recomputed from data/clean; charts of trained models read <chart>_results.json, the saved
outputs of analysis_L6.py (its cache). Every number printed with show() is compared with L6_numbers.json.
Run (after analysis_L6.py):  python3 lectures/L6/make_quantlets_L6.py
"""
import json
import os
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

print("loading the results of analysis_L6.py (from its cache) ...")
A = runpy.run_path(os.path.join(HERE, "analysis_L6.py"), run_name="analysis_L6_results")
NUM = os.path.join(HERE, "L6_numbers.json")
K = 6
QS = []
L_ = lambda o: np.asarray(o).tolist()


def loader(chart):
    return ("import json, urllib.request\n"
            f"_F = '{chart}_results.json'\n"
            f"R = json.load(open(_F)) if os.path.isfile(_F) else json.load(urllib.request.urlopen(REPO + 'Quantlets/L{K}/NN_L{K}_{chart}/' + _F))")


ONLY = [a for a in sys.argv[1:] if not a.startswith('-')]


def make(chart, description, keywords, datafile, intro, code, results=None):
    if ONLY and chart not in ONLY:
        return
    if results is not None:
        d = os.path.join(HERE, "..", "..", "Quantlets", f"L{K}", f"NN_L{K}_{chart}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{chart}_results.json"), "w") as f:
            json.dump(results, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
        code = loader(chart) + "\n" + code
    q = Quantlet(k=K, chart=chart, description=description, keywords=keywords, datafile=datafile)
    q.md(intro)
    q.code(code)
    q.write(execute=True)
    n, bad = check_numbers(q, NUM)
    print(f"  {q.name}: {n} numbers checked", "OK" if not bad else bad)
    QS.append((q.name, n, bad))


SAVE = "fig.savefig('{c}.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()"
TOK = r"""import re
from collections import Counter
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*")
def tok(s):
    s = re.sub(r'https?://\S+|@\w+', ' ', str(s).lower())
    return ['<num>' if t[0].isdigit() else t for t in TOK.findall(s)]"""

make("zipf", "Rank-frequency plot of the words of the European Central Bank speeches 2022-2026, log-log, with the fitted power law (Zipf's law).",
     "Zipf's law, word frequency, tokenisation, corpus statistics, ECB speeches, NLP",
     "data/clean/ecb_speeches_2022plus.csv (European Central Bank, speeches dataset)",
     "$f(r)\\approx Cr^{-a}$, fitted by least squares on log rank and log frequency for ranks 10 to 10,000.",
     TOK + "\nsp = pd.read_csv(DATA + 'ecb_speeches_2022plus.csv')\ncnt = Counter(w for t in sp.contents for w in tok(t))\n"
     "freq = np.array(sorted(cnt.values(), reverse=True), float); rk = np.arange(1, len(freq) + 1); s = (rk >= 10) & (rk <= 10000)\n"
     "a, c = np.polyfit(np.log(rk[s]), np.log(freq[s]), 1)\nshow('corpus.n_tokens', 'tokens', int(freq.sum())); show('corpus.n_types', 'distinct words', len(freq)); show('corpus.zipf_slope', 'slope', round(float(a), 3))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.3)); ax.loglog(rk, freq, color=PAL['data'], lw=1.4, label='word frequency')\n"
     "ax.loglog(rk[s], np.exp(c) * rk[s] ** a, color=PAL['model'], lw=1.2, ls='--', label=f'fitted power law, slope {a:.2f}')\n"
     "ax.set_xlabel('Rank of the word'); ax.set_ylabel('Frequency'); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="zipf"))

coef, voc = A["coef"], A["voc_"]
make("top_words", "The words with the largest positive and negative difference between the positive and negative coefficients of a bag-of-words logistic regression on the Financial PhraseBank.",
     "sentiment, bag of words, logistic regression, financial lexicon, Financial PhraseBank",
     "Financial PhraseBank (Malo et al., 2014); saved coefficients of analysis_L6.py, split 1 (top_words_results.json)",
     "$\\beta_{positive}-\\beta_{negative}$ per word: a sentiment dictionary learned from finance text.",
     "w = np.array(R['words']); d = np.array(R['diff']); pos = np.argsort(-d)[:8]; neg = np.argsort(d)[:8]\n"
     "show('words.pos', 'most positive words', ', '.join(w[pos[:5]])); show('words.neg', 'most negative words', ', '.join(w[neg[:5]]))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.6)); v = np.concatenate([d[neg][::-1], d[pos][::-1]]); lab = list(w[neg][::-1]) + list(w[pos][::-1])\n"
     "ax.barh(range(16), v, color=[PAL['positive']] * 8 + [PAL['benchmark']] * 8); ax.set_yticks(range(16)); ax.set_yticklabels(lab, fontsize=8); ax.axvline(0, color='#222222', lw=.7)\n"
     "ax.set_xlabel('beta_positive - beta_negative (log-odds per occurrence)'); ax.barh([], [], color=PAL['benchmark'], label='pushes towards positive'); ax.barh([], [], color=PAL['positive'], label='pushes towards negative')\n"
     "legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, .12, 1, 1))\n" + SAVE.format(c="top_words"),
     {"words": L_(voc), "diff": L_(coef[2] - coef[0])})

make("w2v_pca", "Two-dimensional principal-component projection of 32 words in four groups (monetary policy, energy, banking, digital payments) in our word2vec space trained on ECB speeches.",
     "word2vec, embeddings, PCA, semantic space, ECB speeches, NLP",
     "data/clean/ecb_speeches_2022plus.csv; saved word vectors of analysis_L6.py (w2v_pca_results.json)",
     "Word vectors of skip-gram with negative sampling ($d=100$), centred and projected on two principal components.",
     "Z = np.array(R['vec']); g = R['group']; w = R['word']; Zc = Z - Z.mean(0); U, S, Vt = np.linalg.svd(Zc, full_matrices=False); P = Zc @ Vt[:2].T\n"
     "show('pca.var2_pct', 'variance kept by 2 components (%)', round(100 * float((S[:2] ** 2).sum() / (S ** 2).sum()), 1))\n"
     "col = {'monetary': PAL['data'], 'energy': PAL['model'], 'banking': PAL['benchmark'], 'digital': '#6250d6'}\n"
     "name = {'monetary': 'monetary policy', 'energy': 'energy', 'banking': 'banking', 'digital': 'digital and payments'}\nfig, ax = plt.subplots(figsize=(6.4, 2.9))\n"
     "for k in col:\n    ii = [i for i, x in enumerate(g) if x == k]; ax.scatter(P[ii, 0], P[ii, 1], color=col[k], s=16, label=name[k])\n"
     "    for i in ii: ax.annotate(w[i], (P[i, 0], P[i, 1]), fontsize=7.5, color=col[k], xytext=(2, 2), textcoords='offset points')\n"
     "ax.set_xlabel('First principal component'); ax.set_ylabel('Second principal\\ncomponent'); legend_bottom(fig, ncol=4); fig.tight_layout(rect=(0, .12, 1, 1))\n" + SAVE.format(c="w2v_pca"),
     {"vec": L_(A["Z"]), "group": [g for g, _ in A["words"]], "word": [w for _, w in A["words"]]})

W2V, GLV = A["W2V"], A["GLV"]
hc, he = np.histogram(np.log10(GLV["x"]), bins=60)
make("emb_training", "Training loss of word2vec (SGNS) by epoch and distribution of the GloVe co-occurrence counts, both on ECB speeches.",
     "word2vec, GloVe, negative sampling, co-occurrence, weighting function, embeddings",
     "data/clean/ecb_speeches_2022plus.csv; saved losses and co-occurrence histogram of analysis_L6.py (emb_training_results.json)",
     "Left: SGNS loss per (centre, context) pair. Right: histogram of $\\log_{10}X_{ij}$; GloVe gives weight 1 above $x_{max}=100$.",
     "show('w2v.loss_first', 'SGNS loss, epoch 1', round(R['hist'][0], 3)); show('w2v.loss_last', 'SGNS loss, last epoch', round(R['hist'][-1], 3)); show('glove.nnz', 'non-zero co-occurrence pairs', R['nnz'])\n"
     "fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4)); axes[0].plot(np.arange(1, len(R['hist']) + 1), R['hist'], color=PAL['model'], marker='o', ms=3, label='word2vec (SGNS)')\n"
     "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Negative-sampling loss', fontsize=9.5); e = np.array(R['edges'])\n"
     "axes[1].bar(e[:-1], R['counts'], width=np.diff(e), align='edge', color=PAL['data'], alpha=.7, label='co-occurrence counts $X_{ij}$'); axes[1].set_yscale('log')\n"
     "axes[1].axvline(2, color=PAL['alt'], ls='--', lw=1.3, label='$x_{max}=100$ (weight 1 above)'); axes[1].set_xlabel('log$_{10}$ $X_{ij}$'); axes[1].set_ylabel('Number of word pairs', fontsize=9.5)\n"
     "legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="emb_training"),
     {"hist": W2V["hist"], "nnz": GLV["nnz"], "counts": L_(hc), "edges": L_(he)})

SE = A["SE"]
MODELS = A["MODELS"]
res_sent = {ds: {k: {"f1": [r["res"][k]["f1"] for r in SE[ds]], "acc": [r["res"][k]["acc"] for r in SE[ds]]} for k in MODELS} for ds in ["pb", "tw"]}
make("sentiment_models", "Test macro-F1 of seven text representations and classifiers on the Financial PhraseBank (5 random splits) and on finance tweets (3 seeds).",
     "sentiment analysis, TF-IDF, bag of words, word2vec, embedding bag, MLP, macro-F1, finance tweets",
     "Financial PhraseBank and Twitter financial news; saved test scores of analysis_L6.py (sentiment_models_results.json)",
     "Macro-F1 = mean of the F1 scores of the three classes; bars are means, whiskers standard deviations.",
     "M = ['majority', 'bow_lr', 'tfidf_lr', 'tfidf_mlp', 'w2v_lr', 'embbag', 'embbag_w2v']\n"
     "NAME = ['Majority', 'BoW + logistic', 'TF-IDF + logistic', 'TF-IDF + MLP', 'mean word2vec + logistic', 'embedding bag (random)', 'embedding bag (word2vec start)']\n"
     "for ds in ['pb', 'tw']:\n    for k in ['tfidf_lr', 'embbag_w2v']: show(f'sent.{ds}.{k}.f1', f'{ds} {k} macro-F1 (%)', round(float(np.mean(np.array(R[ds][k]['f1']) * 100)), 2))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.7)); y = np.arange(len(M))\nfor j, (ds, c, l) in enumerate([('pb', PAL['data'], 'Financial PhraseBank'), ('tw', PAL['model'], 'Finance tweets')]):\n"
     "    ax.barh(y + (j - .5) * .38, [100 * np.mean(R[ds][k]['f1']) for k in M], height=.38, xerr=[100 * np.std(R[ds][k]['f1'], ddof=1) for k in M], color=c, capsize=2, label=l)\n"
     "ax.set_yticks(y); ax.set_yticklabels(NAME, fontsize=8.5); ax.invert_yaxis(); ax.set_xlabel('Macro-F1 on the test set (%)'); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, .12, 1, 1))\n" + SAVE.format(c="sentiment_models"),
     res_sent)

pb_y, agree = A["pb_y"], A["agree"]
res_ag = {k: [[float((r["preds"][k][agree[r["te"]] == a] == pb_y[r["te"]][agree[r["te"]] == a]).mean()) for r in SE["pb"]] for a in [50, 66, 75, 100]]
          for k in ["majority", "tfidf_lr", "embbag_w2v"]}
make("agreement", "Test accuracy on the Financial PhraseBank by the share of annotators who agreed on the label (50, 66, 75, 100%), 5 random splits.",
     "label noise, annotator agreement, sentiment, TF-IDF, embedding bag, Financial PhraseBank",
     "Financial PhraseBank; saved test predictions of analysis_L6.py (agreement_results.json)",
     "Test accuracy is much lower on the sentences on which the annotators disagree.",
     "for k in ['tfidf_lr']:\n    show(f'agree.{k}.100', 'accuracy, unanimous sentences (%)', round(100 * float(np.mean(R[k][3])), 1)); show(f'agree.{k}.50', 'accuracy, 50% agreement (%)', round(100 * float(np.mean(R[k][0])), 1))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.2)); xs = np.arange(4)\nfor j, (k, c, l) in enumerate([('majority', '#8a8a8a', 'Majority'), ('tfidf_lr', PAL['benchmark'], 'TF-IDF + logistic'), ('embbag_w2v', PAL['model'], 'embedding bag (word2vec start)')]):\n"
     "    ax.bar(xs + (j - 1) * .27, [100 * np.mean(v) for v in R[k]], width=.27, color=c, label=l)\n"
     "ax.set_xticks(xs); ax.set_xticklabels([f'{a}% agree' for a in [50, 66, 75, 100]]); ax.set_ylabel('Test accuracy (%)'); legend_bottom(fig, ncol=3); fig.tight_layout(rect=(0, .15, 1, 1))\n" + SAVE.format(c="agreement"),
     res_ag)

CU = A["CU"]
make("learning_curve", "Test macro-F1 on the Financial PhraseBank against the number of labelled sentences (TF-IDF, mean word2vec, embedding bag random or word2vec start), and against the embedding dimension.",
     "learning curve, transfer learning, word2vec, few labels, embedding dimension, sentiment",
     "Financial PhraseBank; saved test scores of analysis_L6.py (learning_curve_results.json)",
     "Pre-trained vectors and the label budget.",
     "FR = [0.05, 0.1, 0.25, 0.5, 1.0]; n = [int(f * R['ntr']) for f in FR]\ncur = {k: [100 * np.mean([f for s, fr, f in R[k] if fr == x]) for x in FR] for k in ['tfidf_lr', 'w2v_lr', 'embbag', 'embbag_w2v']}\n"
     "for k in cur: show(f'curve.{k}.p5', f'{k}, {n[0]} sentences', round(cur[k][0], 1))\nfig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={'width_ratios': [1.4, 1]})\n"
     "for k, c, l in [('tfidf_lr', PAL['benchmark'], 'TF-IDF + logistic'), ('w2v_lr', PAL['alt'], 'mean word2vec + logistic'), ('embbag', '#6250d6', 'embedding bag (random)'), ('embbag_w2v', PAL['model'], 'embedding bag (word2vec start)')]:\n"
     "    axes[0].plot(n, cur[k], marker='o', ms=3, color=c, lw=1.5, label=l)\n"
     "axes[0].set_xscale('log'); axes[0].set_xticks(n); axes[0].set_xticklabels([str(x) for x in n], fontsize=8); axes[0].set_xlabel('Labelled training sentences', fontsize=9); axes[0].set_ylabel('Test macro-F1 (%)')\n"
     "d = sorted(R['dims'], key=int); axes[1].plot([int(x) for x in d], [100 * np.mean(R['dims'][x]) for x in d], marker='s', ms=3, color='#6250d6'); axes[1].set_xscale('log', base=2)\n"
     "axes[1].set_xticks([int(x) for x in d]); axes[1].set_xticklabels(d, fontsize=8); axes[1].set_xlabel('Embedding dimension d', fontsize=9); axes[1].set_ylabel('Test macro-F1 (%)')\n"
     "legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, .2, 1, 1))\n" + SAVE.format(c="learning_curve"),
     {**{k: [list(x) for x in CU[k]] for k in ["tfidf_lr", "w2v_lr", "embbag", "embbag_w2v"]}, "dims": {str(k): v for k, v in CU["dims"].items()}, "ntr": A["ntr"]})

from sklearn.metrics import f1_score  # noqa: E402
TP = A["TP"]
yt_ = A["yt_"]
make("topic_f1", "F1 per topic on the test tweets of the Twitter financial news topic data (20 classes): TF-IDF + logistic regression and an embedding bag with word2vec start.",
     "topic classification, finance tweets, TF-IDF, logistic regression, embedding bag, F1",
     "Twitter financial news topic (Hugging Face, zeroshot); saved test predictions of analysis_L6.py (topic_f1_results.json)",
     "Per-class F1 on the published validation split, used as test.",
     "from sklearn.metrics import f1_score\ny = np.array(R['y']); lr = np.array(R['lr']); eb = np.array(R['eb']); names = R['names']\n"
     "show('topic.lr_f1', 'macro-F1 TF-IDF + logistic (%)', round(100 * f1_score(y, lr, average='macro'), 2)); show('topic.eb_f1', 'macro-F1 embedding bag (%)', round(100 * f1_score(y, eb, average='macro'), 2))\n"
     "f1c = f1_score(y, lr, average=None); f1e = f1_score(y, eb, average=None); o = np.argsort(f1c)\nfig, ax = plt.subplots(figsize=(6.4, 3.4))\n"
     "ax.barh(range(20), 100 * f1c[o], color=PAL['benchmark'], label='TF-IDF + logistic'); ax.scatter(100 * f1e[o], range(20), color=PAL['model'], s=14, zorder=3, label='embedding bag (word2vec start)')\n"
     "ax.set_yticks(range(20)); ax.set_yticklabels([names[i] for i in o], fontsize=7); ax.set_xlabel('F1 on the test tweets (%)', fontsize=9.5); legend_bottom(fig, ncol=2); fig.tight_layout(rect=(0, .08, 1, 1))\n" + SAVE.format(c="topic_f1"),
     {"y": L_(yt_), "lr": L_(TP["lr"]), "eb": L_(TP["eb"]), "names": list(A["names_tp"].values)})

bad = [q for q in QS if q[2] or q[1] == 0]
print(f"{len(QS)} Quantlets written;", "all numbers match" if not bad else f"problems: {bad}")
sys.exit(1 if bad else 0)
