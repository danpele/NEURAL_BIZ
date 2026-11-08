"""
make_quantlets_L12.py -- one Quantlet per chart of Lecture 12 (Quantlets/L12/NN_L12_<chart>/), executed, and every number
printed with show() compared with L12_numbers.json.
The language models need a graphics card and about 40 minutes (lecture notebook); the Quantlets recompute every number and
chart from the per-item model outputs that analysis_L12.py publishes in lectures/L12/results/, and from data/clean.
Run:  python3 lectures/L12/make_quantlets_L12.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

NUM = os.path.join(HERE, "L12_numbers.json")
RES = ("RES = REPO + 'lectures/L12/results/'\nimport json, urllib.request\n"
       "def load_json(name):\n    p = RES + name\n    return json.load(open(p)) if os.path.exists(p) else json.load(urllib.request.urlopen(p))\n")
F1 = ("from sklearn.metrics import f1_score, accuracy_score\n"
      "LAB = {'pb': ['positive', 'negative', 'neutral'], 'tw': ['bullish', 'bearish', 'neutral']}\n"
      "def f1(y, p, ds):\n    return f1_score(y, p, labels=LAB[ds], average='macro')\n"
      "def pred(df, m, ds):\n    P = df[[f'{m}_{l}' for l in LAB[ds]]].values\n    return [LAB[ds][i] for i in P.argmax(1)]\n")
KW = "large language models, prompt engineering, financial sentiment, Qwen2.5, zero-shot, few-shot"
qs = []

q = Quantlet(12, "tokens_per_sentence", "Length of the 4,838 Financial PhraseBank sentences in words, in Qwen2.5 byte-pair-encoding tokens and "
             "in FinBERT WordPiece tokens; tokens per word.", "tokenisation, byte-pair encoding, WordPiece, Qwen2.5, FinBERT, Financial PhraseBank",
             "data/clean/financial_phrasebank.csv; tokenizers Qwen/Qwen2.5-1.5B-Instruct and ProsusAI/finbert (Hugging Face)")
q.code("from transformers import AutoTokenizer\nPB = pd.read_csv(DATA + 'financial_phrasebank.csv')\n"
       "tq = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')\ntb = AutoTokenizer.from_pretrained('ProsusAI/finbert')\n"
       "nw = PB.sentence.str.split().str.len().values\nnq = np.array([len(tq.encode(s)) for s in PB.sentence])\n"
       "nb = np.array([len(tb.encode(s, add_special_tokens=False)) for s in PB.sentence])\n"
       "show('tok.qwen_per_word', 'Qwen tokens per word', round(float(nq.sum() / nw.sum()), 2))\n"
       "show('tok.bert_per_word', 'WordPiece tokens per word', round(float(nb.sum() / nw.sum()), 2))\n"
       "show('tok.qwen_mean', 'mean Qwen tokens per sentence', round(float(nq.mean()), 1))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nbins = np.arange(0, 90, 2)\nax.hist(nw, bins=bins, color=PAL['data'], alpha=0.55, label='Words')\n"
       "ax.hist(nq, bins=bins, histtype='step', lw=1.5, color=PAL['model'], label='Qwen2.5 tokens')\n"
       "ax.hist(nb, bins=bins, histtype='step', lw=1.5, color=PAL['benchmark'], label='FinBERT tokens')\nax.set_xlabel('Length of a sentence')\n"
       "ax.set_ylabel('Number of sentences')\nlegend_bottom(fig, ncol=3)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\n"
       "fig.savefig('tokens_per_sentence.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "next_token", "Top next-token probabilities of Qwen2.5-1.5B-Instruct after the start of a sentence from the latest ECB speech, "
             "at temperatures 0.5, 1 and 2, with the entropy of each distribution.", "next-token prediction, temperature, softmax, entropy, ECB speeches",
             "lectures/L12/results/lm_outputs.json (model outputs); data/clean/ecb_speeches_2022plus.csv")
q.code(RES + "NT = load_json('lm_outputs.json')['next_token']\n"
       "for T in ('0.5', '1.0', '2.0'):\n    show('next.entropy.t' + T.replace('.', ''), f'entropy at T={T} (nats)', round(NT[T]['entropy'], 2))\n"
       "show('next.top1_p', 'probability of the top token at T=1 (%)', round(100 * NT['1.0']['p'][0], 1))\n"
       "toks = NT['1.0']['tok']\nfig, ax = plt.subplots(figsize=(6.4, 2.4))\n"
       "for j, (T, c) in enumerate([('0.5', PAL['data']), ('1.0', PAL['model']), ('2.0', PAL['benchmark'])]):\n"
       "    d = dict(zip(NT[T]['tok'], NT[T]['p']))\n    ax.bar(np.arange(len(toks)) + (j - 1) * 0.27, [100 * d.get(t, np.nan) for t in toks], 0.27, color=c, label=f'temperature {T}')\n"
       "ax.set_xticks(range(len(toks)))\nax.set_xticklabels([repr(t)[1:-1] for t in toks])\nax.set_ylabel('Probability (%)')\nlegend_bottom(fig, ncol=3)\n"
       "fig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('next_token.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "perplexity_size", "Perplexity of Qwen2.5-0.5B, 1.5B and 3B on 40 ECB speeches of 2022-2023 and 40 speeches from July 2025 "
             "(first 1,024 tokens), and the slope of log loss on log parameters.", "perplexity, scaling, training cut-off, contamination, ECB speeches",
             "lectures/L12/results/lm_outputs.json (mean negative log-likelihood per token)")
q.code(RES + "J = load_json('lm_outputs.json')\nnll, P = J['ppl_nll'], J['params_b']\nsizes = ['0.5B', '1.5B', '3B']\n"
       "for s in sizes:\n    show('ppl.' + s.replace('.', '') + '.old', f'perplexity {s}, 2022-2023', round(float(np.exp(nll[s]['old'])), 2))\n"
       "    show('ppl.' + s.replace('.', '') + '.new', f'perplexity {s}, from July 2025', round(float(np.exp(nll[s]['new'])), 2))\n"
       "b = np.polyfit(np.log([P[s] for s in sizes]), np.log([nll[s]['new'] for s in sizes]), 1)[0]\nshow('ppl_slope', 'slope of log loss on log parameters', round(float(b), 3))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.bar(np.arange(3) - 0.18, [np.exp(nll[s]['old']) for s in sizes], 0.36, color=PAL['data'], label='2022-2023')\n"
       "ax.bar(np.arange(3) + 0.18, [np.exp(nll[s]['new']) for s in sizes], 0.36, color=PAL['model'], label='from July 2025')\nax.set_xticks(range(3))\n"
       "ax.set_xticklabels(['Qwen2.5-' + s for s in sizes])\nax.set_ylabel('Perplexity')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\n"
       "fig.savefig('perplexity_size.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "prompt_results", "Macro-F1 with 95 percent bootstrap intervals of prompted Qwen2.5 models (zero-shot, few-shot, chain-of-thought), "
             "LLM embeddings with logistic regression, TF-IDF with logistic regression and the majority class, on Financial PhraseBank and Twitter "
             "financial news.", KW, "lectures/L12/results/predictions_pb.csv, predictions_tw.csv")
q.code(RES + F1 + "M = ['tfidf', 'zs_0.5B', 'zs_1.5B', 'zs_3B', 'fs_1.5B', 'fs_3B', 'cot_3B', 'probe']\nR = {}\n"
       "for ds in ('pb', 'tw'):\n    df = pd.read_csv(RES + f'predictions_{ds}.csv')\n    for m in M:\n        R[(ds, m)] = 100 * f1(df.label, pred(df, m, ds), ds)\n"
       "show('cls.pb.tfidf.f1', 'PhraseBank macro-F1 TF-IDF', round(R[('pb', 'tfidf')], 1))\nshow('cls.pb.zs_3B.f1', 'PhraseBank macro-F1 zero-shot 3B', round(R[('pb', 'zs_3B')], 1))\n"
       "show('cls.tw.zs_3B.f1', 'Twitter macro-F1 zero-shot 3B', round(R[('tw', 'zs_3B')], 1))\nshow('cls.pb.probe.f1', 'PhraseBank macro-F1 embeddings + logistic', round(R[('pb', 'probe')], 1))\n"
       "fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)\nfor ax, ds in zip(axs, ('pb', 'tw')):\n"
       "    ax.barh(range(len(M)), [R[(ds, m)] for m in M], color=[PAL['benchmark']] + [PAL['model']] * 6 + ['#6250d6'])\n"
       "    ax.set_yticks(range(len(M)))\n    ax.set_yticklabels(M)\n    ax.invert_yaxis()\n    ax.set_xlim(0, 100)\n    ax.set_xlabel('Macro-F1 (%) - ' + ds)\n"
       "fig.tight_layout()\nfig.savefig('prompt_results.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "fewshot_curve", "Macro-F1 on Financial PhraseBank of Qwen2.5-1.5B and 3B with 0, 3, 6, 12 and 24 balanced examples in the "
             "prompt (three seeds for the choice of examples), against TF-IDF with logistic regression.", KW + ", in-context learning",
             "lectures/L12/results/fewshot_labels_pb.csv, predictions_pb.csv")
q.code(RES + F1 + "L = pd.read_csv(RES + 'fewshot_labels_pb.csv')\nKS = [0, 1, 2, 4, 8]\nmean = {}\n"
       "for s in ('1.5B', '3B'):\n    for k in KS:\n        v = [100 * f1(L.label, L[f'{s}_k{k}_s{sd}'], 'pb') for sd in (0, 1, 2)]\n        mean[(s, k)] = np.mean(v)\n"
       "        show('fewshot.' + s.replace('.', '') + f'.{k}.mean', f'{s}, {3 * k} examples: mean macro-F1', round(float(np.mean(v)), 1))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nfor s, c in (('1.5B', PAL['data']), ('3B', PAL['model'])):\n"
       "    ax.plot(range(5), [mean[(s, k)] for k in KS], marker='o', color=c, label='Qwen2.5-' + s)\nax.set_xticks(range(5))\nax.set_xticklabels([str(3 * k) for k in KS])\n"
       "ax.set_xlabel('Examples in the prompt')\nax.set_ylabel('Macro-F1 (%)')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\n"
       "fig.savefig('fewshot_curve.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "prompt_sensitivity", "Macro-F1 of zero-shot Qwen2.5-1.5B and 3B under five paraphrases of the same instruction, on Financial "
             "PhraseBank and Twitter financial news: the range is a measure of prompt risk.", KW + ", prompt sensitivity, model risk",
             "lectures/L12/results/sensitivity_labels_pb.csv, sensitivity_labels_tw.csv")
q.code(RES + F1 + "rows = []\nfor ds in ('pb', 'tw'):\n    L = pd.read_csv(RES + f'sensitivity_labels_{ds}.csv')\n    for s in ('1.5B', '3B'):\n"
       "        v = [100 * f1(L.label, L[f'{s}_i{i}'], ds) for i in range(1, 6)]\n        rows.append((ds, s, v))\n"
       "        show(f'sens.{ds}_' + s.replace('.', '') + '.range', f'{ds} {s}: range over the 5 wordings', round(max(v) - min(v), 1))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nfor r, (ds, s, v) in enumerate(rows):\n    ax.plot([min(v), max(v)], [r, r], color='#bbbbbb', lw=3)\n"
       "    ax.scatter(v, [r] * 5, color=SERIES[:5], zorder=3)\nax.set_yticks(range(4))\nax.set_yticklabels([f'{d}, {s}' for d, s, _ in rows])\nax.invert_yaxis()\n"
       "ax.set_xlabel('Macro-F1 (%)')\nfig.tight_layout()\nfig.savefig('prompt_sensitivity.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "calibration", "Reliability diagram and expected calibration error of the label probabilities of zero-shot and few-shot "
             "Qwen2.5-3B and of TF-IDF with logistic regression on the Financial PhraseBank test set.", KW + ", calibration, expected calibration error",
             "lectures/L12/results/predictions_pb.csv")
q.code(RES + F1 + "df = pd.read_csv(RES + 'predictions_pb.csv')\n"
       "def rel(m, bins=10):\n    P = df[[f'{m}_{l}' for l in LAB['pb']]].values\n    conf = P.max(1)\n    ok = np.array([LAB['pb'][i] == t for i, t in zip(P.argmax(1), df.label)])\n"
       "    edges = np.linspace(1 / 3, 1, bins + 1)\n    pts, ece = [], 0.0\n    for a, b in zip(edges[:-1], edges[1:]):\n"
       "        msk = (conf >= a) & (conf <= b if b == 1 else conf < b)\n        if msk.sum() >= 5:\n            pts.append((conf[msk].mean(), ok[msk].mean()))\n"
       "        if msk.sum():\n            ece += msk.mean() * abs(conf[msk].mean() - ok[msk].mean())\n    return pts, ece\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.6))\nax.plot([1 / 3, 1], [1 / 3, 1], ls=':', color='#999999', label='Perfect calibration')\n"
       "for m, key, c in (('zs_3B', 'zs_3B', PAL['model']), ('fs_3B', 'fs_3B_s0', PAL['alt']), ('tfidf', 'tfidf', PAL['benchmark'])):\n"
       "    pts, ece = rel(m)\n    show(f'calib.{key}', f'ECE {m} (%)', round(100 * ece, 1))\n    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', color=c, label=m)\n"
       "ax.set_xlabel('Predicted probability of the chosen label')\nax.set_ylabel('Observed accuracy')\nlegend_bottom(fig, ncol=2)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
       "fig.savefig('calibration.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "agreement", "Accuracy of TF-IDF with logistic regression and of zero-shot and few-shot Qwen2.5-3B on Financial PhraseBank "
             "sentences by annotator agreement (100, 75, 66 and 50 percent).", KW + ", label quality, annotator agreement",
             "lectures/L12/results/agreement_labels.csv, predictions_pb.csv")
q.code(RES + F1 + "A = pd.read_csv(RES + 'agreement_labels.csv')\nT = pd.read_csv(RES + 'predictions_pb.csv')\nacc = {}\n"
       "for m, mt in (('tfidf', 'tfidf'), ('zs_3B', 'zs_3B'), ('fs_3B', 'fs_3B')):\n    acc[(m, 100)] = 100 * accuracy_score(T.label, pred(T, mt, 'pb'))\n"
       "    for lev in (75, 66, 50):\n        s = A[A.agreement == lev]\n        acc[(m, lev)] = 100 * accuracy_score(s.label, s[m])\n"
       "    for lev in (100, 75, 66, 50):\n        show(f'agree.{m}.{lev}', f'{m} accuracy at {lev} % agreement', round(acc[(m, lev)], 1))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nfor j, (m, c) in enumerate((('tfidf', PAL['benchmark']), ('zs_3B', PAL['model']), ('fs_3B', PAL['alt']))):\n"
       "    ax.bar(np.arange(4) + (j - 1) * 0.27, [acc[(m, l)] for l in (100, 75, 66, 50)], 0.27, color=c, label=m)\n"
       "ax.set_xticks(range(4))\nax.set_xticklabels(['100 %', '75 %', '66 %', '50 %'])\nax.set_ylabel('Accuracy (%)')\nlegend_bottom(fig, ncol=3)\n"
       "fig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('agreement.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "cost_accuracy", "Macro-F1 against measured milliseconds per sentence on a laptop for TF-IDF, prompted Qwen2.5 models and "
             "LLM embeddings with logistic regression, Financial PhraseBank test set.", KW + ", cost, latency",
             "lectures/L12/results/predictions_pb.csv, lm_outputs.json")
q.code(RES + F1 + "df = pd.read_csv(RES + 'predictions_pb.csv')\nms = {k.replace('05B', '0.5B').replace('15B', '1.5B'): v for k, v in load_json('lm_outputs.json')['ms_per_item']['pb'].items()}\n"
       "M = ['tfidf', 'zs_0.5B', 'zs_1.5B', 'zs_3B', 'fs_1.5B', 'fs_3B', 'cot_3B', 'probe']\nf = {m: 100 * f1(df.label, pred(df, m, 'pb'), 'pb') for m in M}\n"
       "show('cls.pb.cot_3B.f1', 'chain-of-thought 3B macro-F1', round(f['cot_3B'], 1))\nshow('cls.pb.zs_05B.f1', 'zero-shot 0.5B macro-F1', round(f['zs_0.5B'], 1))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nfor m in M:\n    ax.scatter(max(ms[m], 0.01), f[m], color=PAL['model'])\n"
       "    ax.annotate(m, (max(ms[m], 0.01), f[m]), textcoords='offset points', xytext=(4, 3), fontsize=8)\n"
       "ax.set_xscale('log')\nax.set_xlabel('Milliseconds per sentence (log scale)')\nax.set_ylabel('Macro-F1 (%)')\nfig.tight_layout()\n"
       "fig.savefig('cost_accuracy.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "rag_recall", "Recall@k of four retrievers (TF-IDF, BM25, dense bge-small embeddings, hybrid reciprocal rank fusion) that "
             "search 400 ECB speeches by their title.", "retrieval-augmented generation, BM25, dense retrieval, ECB speeches, recall",
             "lectures/L12/results/rag_ranks.csv (from data/clean/ecb_speeches_2022plus.csv)")
q.code(RES + "Rk = pd.read_csv(RES + 'rag_ranks.csv')\nK = [1, 3, 5, 10]\nfig, ax = plt.subplots(figsize=(6.4, 2.3))\n"
       "for m, c in (('tfidf', PAL['benchmark']), ('bm25', SERIES[5]), ('dense', PAL['model']), ('hybrid', '#6250d6')):\n"
       "    r = [round(100 * float((Rk[f'rank_{m}'] <= k).mean()), 1) for k in K]\n    show(f'rag.{m}.r1', f'recall@1 {m} (%)', r[0])\n"
       "    show(f'rag.{m}.mrr', f'mean reciprocal rank {m}', round(float((1 / Rk[f'rank_{m}']).mean()), 3))\n    ax.plot(K, r, marker='o', color=c, label=m)\n"
       "ax.set_xticks(K)\nax.set_xlabel('k')\nax.set_ylabel('Recall@k (%)')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\n"
       "fig.savefig('rag_recall.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "rag_qa", "Which ECB board member gave the speech? Answers of Qwen2.5-3B-Instruct without documents, with the top-3 retrieved "
             "speeches (RAG) and with the right speech: share correct, wrong (hallucinated) and abstaining, speeches since 2025.",
             "retrieval-augmented generation, hallucination, question answering, ECB speeches", "lectures/L12/results/rag_qa.csv")
q.code(RES + "Q = pd.read_csv(RES + 'rag_qa.csv')\nmodes = ['closed', 'closed_abstain', 'rag', 'oracle']\nfig, ax = plt.subplots(figsize=(6.4, 2.6))\nleft = np.zeros(4)\n"
       "for k, c in (('correct', PAL['benchmark']), ('wrong', PAL['positive']), ('abstain', '#bbbbbb')):\n"
       "    v = np.array([round(100 * float((Q[m] == k).mean()), 1) for m in modes])\n"
       "    for m, x in zip(modes, v):\n        show(f'rag.qa_{m}.{k}', f'{m}: {k} (%)', float(x))\n"
       "    ax.barh(range(4), v, left=left, color=c, label=k)\n    left += v\n"
       "ax.set_yticks(range(4))\nax.set_yticklabels(modes)\nax.invert_yaxis()\nax.set_xlabel('Share of questions (%)')\nlegend_bottom(fig, ncol=3)\n"
       "fig.tight_layout(rect=(0, 0.12, 1, 1))\nfig.savefig('rag_qa.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(12, "llm_var", "LLM-VaR after Pele et al. (2026) with an open 3B model: the 5 and 1 percent Value-at-Risk of the S&P 500 written "
             "by Qwen2.5-3B-Instruct from the last 250 daily returns, against historical simulation and GARCH(1,1)-t, with Kupiec tests, 2025-2026.",
             "LLM-VaR, Value-at-Risk, backtesting, Kupiec, GARCH, S&P 500, large language models",
             "lectures/L12/results/llmvar_sp500.csv (from data/clean/markets_daily.csv)")
q.code(RES + "from scipy import stats\nV = pd.read_csv(RES + 'llmvar_sp500.csv', parse_dates=['date']).set_index('date')\n"
       "def kupiec(h, a):\n    n, x = len(h), int(h.sum()); p = x / n\n"
       "    lr = -2 * n * np.log(1 - a) if x == 0 else -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - (n - x) * np.log(1 - p) - x * np.log(p))\n"
       "    return float(1 - stats.chi2.cdf(lr, 1))\n"
       "for m in ('llm', 'hs', 'garch'):\n    for a, s in ((0.05, '05'), (0.01, '01')):\n        h = (V['r'] < V[f'{m}_{s}']).values\n"
       "        show(f'llmvar.{m}_{s}.viol', f'{m} violations at {s}', int(h.sum()))\n        show(f'llmvar.{m}_{s}.kupiec_p', f'{m} Kupiec p at {s}', round(kupiec(h, a), 3))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.5))\nax.plot(V.index, V.r, color=PAL['data'], lw=0.5, label='Daily log return (%)')\n"
       "ax.plot(V.index, V.garch_05, color=PAL['benchmark'], label='5 % VaR, GARCH(1,1)-t')\nax.plot(V.index, V.llm_05, color=PAL['model'], label='5 % LLM-VaR')\n"
       "hit = V.r < V.llm_05\nax.scatter(V.index[hit], V.r[hit], s=10, color=PAL['positive'], zorder=3, label='Violation')\nax.set_ylabel('%')\n"
       "legend_bottom(fig, ncol=2)\nfig.tight_layout(rect=(0, 0.15, 1, 1))\nfig.savefig('llm_var.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

bad_all, n_all = [], 0
for q in qs:
    q.write(execute=True)
    n, bad = check_numbers(q, NUM)
    n_all += n
    bad_all += bad
    print(f"  {q.name}: {n} numbers checked", "OK" if not bad else bad)
print(f"{len(qs)} Quantlets, {n_all} numbers checked against L12_numbers.json:", "OK" if not bad_all else bad_all)
sys.exit(1 if bad_all else 0)
