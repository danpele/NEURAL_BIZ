"""
make_quantlets_L7.py -- one Quantlet per chart of Lecture 7: Quantlets/L7/NN_L7_<chart>/ (Metainfo.txt, executed notebook, PNG).
The notebooks read the public text data (data/clean) and, for trained models, the small result files written by
analysis_L7.py (lectures/L7/results/*.csv: test predictions, metrics, attention weights, training history). Every number
printed with show() is checked against L7_numbers.json.
Run:  python3 lectures/L7/make_quantlets_L7.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

RES = "REPO + 'lectures/L7/results/'"
DES = "Lecture 7 of Neural Networks and Deep Learning (NN), attention and transformers. "
QS = []
SAVE = "fig.tight_layout(rect=(0, 0.12, 1, 1))\nfig.savefig('{c}.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()"
SAVE0 = "fig.tight_layout()\nfig.savefig('{c}.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()"


def Q(chart, desc, kw, data, md, code, save=SAVE):
    q = Quantlet(k=7, chart=chart, description=DES + desc, keywords=kw, datafile=data)
    q.md(md).code(code + "\n" + save.format(c=chart))
    QS.append(q)


Q("context_lengths", "Length in WordPiece tokens (FinBERT tokenizer) of finance tweets, Financial PhraseBank sentences and ECB speeches, against the 512-token window of BERT.",
  "context window, tokenization, transformer, ECB speeches, tweets, quadratic attention cost",
  "data/clean/twitter_fin_sentiment.csv, financial_phrasebank.csv, ecb_speeches_2022plus.csv; lectures/L7/results/token_lengths.csv",
  "Token counts computed with the FinBERT tokenizer (ProsusAI/finbert) in lectures/L7/analysis_L7.py.",
  "t = pd.read_csv(" + RES + " + 'token_lengths.csv')\nsp, tw, pb = (t.tokens[t.corpus == c].values for c in ['speeches', 'tweets', 'phrasebank'])\n"
  "show('len.tw_median', 'median tokens, tweets', int(np.median(tw)))\nshow('len.sp_median', 'median tokens, ECB speeches', int(np.median(sp)))\n"
  "show('len.sp_pct_over512', 'speeches above 512 tokens (%)', round(100 * float(np.mean(sp > 512)), 1))\n"
  "bins = np.logspace(np.log10(4), np.log10(30000), 60)\nfig, ax = plt.subplots(figsize=(6.4, 2.4))\n"
  "for v, c, lab in [(tw, PAL['data'], 'finance tweets'), (pb, PAL['benchmark'], 'Financial PhraseBank sentences'), (sp, PAL['model'], 'ECB speeches')]:\n"
  "    ax.hist(v, bins=bins, weights=np.full(len(v), 100.0 / len(v)), histtype='stepfilled', alpha=0.45, color=c, label=lab)\n"
  "ax.axvline(512, color='#222222', lw=1, ls='--', label='BERT window: 512 tokens')\nax.set_xscale('log')\n"
  "ax.set_xlabel('text length in tokens (log scale)'); ax.set_ylabel('% of texts')\nlegend_bottom(fig, ncol=2)")

Q("positional_encoding", "Sinusoidal positional encoding of Vaswani et al. (2017) for 64 positions and d = 64 dimensions (mechanism illustration).",
  "positional encoding, transformer, sinusoid, attention, order",
  "none (formula)", "$p_{t,2i}=\\sin(t/10000^{2i/d})$, $p_{t,2i+1}=\\cos(t/10000^{2i/d})$.",
  "d, n = 64, 64\np = np.arange(n)[:, None]\ndiv = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))\npe = np.zeros((n, d))\npe[:, 0::2], pe[:, 1::2] = np.sin(p * div), np.cos(p * div)\n"
  "show('attn.d_head', 'dimension per head in BERT-base', 64)\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nim = ax.imshow(pe.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')\n"
  "ax.set_xlabel('token position $t$'); ax.set_ylabel('dimension $i$')\nfig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)", save=SAVE0)

Q("attention_example", "Attention weights of pretrained FinBERT, averaged over the 12 heads, in layers 6 and 12, for one bearish test tweet.",
  "attention map, FinBERT, BERT, self-attention, interpretability, financial sentiment",
  "lectures/L7/results/attention_example_layer6.csv, attention_example_last.csv (from data/clean/twitter_fin_sentiment.csv)",
  "Rows = queries, columns = keys; each row sums to one.",
  "A6 = pd.read_csv(" + RES + " + 'attention_example_layer6.csv', index_col=0)\nA12 = pd.read_csv(" + RES + " + 'attention_example_last.csv', index_col=0)\n"
  "toks = [str(x).replace('##', '') for x in A6.index]\nshow('example.n_tokens', 'tokens in the tweet', len(toks))\n"
  "show('example.cls_top_pct', 'largest attention of [CLS] on a word, layer 6 (%)', round(100 * float(A6.values[0, 1:-1].max()), 1))\n"
  "fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1))\nfor ax, A, lab in [(axs[0], A6.values, 'layer 6: keys'), (axs[1], A12.values, 'layer 12: keys')]:\n"
  "    ax.imshow(A, cmap='Oranges', vmin=0, vmax=min(1, A.max()))\n    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=90, fontsize=7.5)\n"
  "    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=7.5); ax.set_xlabel(lab, fontsize=9)\naxs[0].set_ylabel('queries', fontsize=9)", save=SAVE0)

Q("attention_entropy", "Normalised entropy of the attention rows and the share of attention on [CLS] and [SEP], by layer of pretrained FinBERT, 500 test tweets.",
  "attention entropy, BERT layers, attention sink, separator token, interpretability",
  "lectures/L7/results/attention_entropy.csv", "Entropy of each attention row divided by $\\log n$ (1 = uniform), averaged over heads, rows and tweets.",
  "a = pd.read_csv(" + RES + " + 'attention_entropy.csv')\nshow('attn.ent_l1', 'normalised entropy, layer 1', round(float(a.entropy[0]), 2))\n"
  "show('attn.sep_max_pct', 'largest share of attention on [SEP] (%)', round(100 * float(a.on_sep.max()), 1))\n"
  "fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))\naxs[0].plot(a.layer, a.entropy, 'o-', color=PAL['model'], label='normalised entropy'); axs[0].set_ylim(0, 1)\n"
  "axs[1].plot(a.layer, 100 * a.on_sep, 's-', color=PAL['data'], label='attention on [SEP]')\naxs[1].plot(a.layer, 100 * a.on_cls, '^-', color=PAL['alt'], label='attention on [CLS]')\n"
  "for x in axs: x.set_xlabel('layer'); x.set_xticks([1, 3, 6, 9, 12])\naxs[0].set_ylabel('entropy / log n'); axs[1].set_ylabel('attention weight (%)')\nlegend_bottom(fig, ncol=3)")

Q("tiny_learning_curve", "Training and validation cross-entropy of the small transformer (2 blocks, 4 heads, d = 64) by epoch, seed 1, with the epoch kept by early stopping.",
  "learning curve, early stopping, transformer from scratch, overfitting",
  "lectures/L7/results/tiny_history.csv", "Early stopping on the validation macro-F1, patience 4.",
  "h = pd.read_csv(" + RES + " + 'tiny_history.csv')\nbe = int(h.epoch[h.val_macro_f1.idxmax()])\nshow('models.tiny.best_epoch_seed1', 'epoch kept', be)\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.plot(h.epoch, h.train_loss, 'o-', color=PAL['train'], label='training')\nax.plot(h.epoch, h.val_loss, 's-', color=PAL['valid'], label='validation')\n"
  "ax.axvline(be, color='#8a8a8a', ls='--', lw=1, label='epoch kept')\nax.set_xlabel('epoch'); ax.set_ylabel('cross-entropy'); ax.set_xticks(h.epoch)\nlegend_bottom(fig, ncol=3)")

Q("model_comparison", "Macro-F1 on 2,388 test tweets for TF-IDF + logistic regression, an embedding bag, a small transformer with and without positions, and FinBERT zero-shot, frozen + logistic and fine-tuned (mean and sd over seeds).",
  "transformer, FinBERT, fine-tuning, TF-IDF, sentiment analysis, finance tweets, macro-F1",
  "lectures/L7/results/model_metrics.csv (from data/clean/twitter_fin_sentiment.csv)", "Mean over the runs of each model (3 seeds for the trained networks).",
  "m = pd.read_csv(" + RES + " + 'model_metrics.csv')\ng = m.groupby('model').macro_f1.agg(['mean', 'std']).fillna(0)\n"
  "show('models.tfidf.macro_f1', 'macro-F1, TF-IDF (%)', round(100 * float(g.loc['tfidf', 'mean']), 1))\n"
  "show('models.finbert_ft.macro_f1', 'macro-F1, FinBERT fine-tuned (%)', round(100 * float(g.loc['finbert_ft', 'mean']), 1))\n"
  "show('models.tiny.macro_f1', 'macro-F1, small transformer (%)', round(100 * float(g.loc['tiny', 'mean']), 1))\n"
  "order = ['tfidf', 'embbag', 'tiny_nopos', 'tiny', 'finbert_zs', 'finbert_probe', 'finbert_ft']\n"
  "names = ['TF-IDF + logistic', 'Embedding bag (no attention)', 'Transformer, no positions', 'Small transformer', 'FinBERT zero-shot', 'FinBERT frozen + logistic', 'FinBERT fine-tuned']\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.8))\nax.barh(range(7), [100 * g.loc[o, 'mean'] for o in order], xerr=[100 * g.loc[o, 'std'] for o in order], color=[PAL['benchmark'], '#8a8a8a', PAL['alt'], PAL['model'], '#e87ba4', '#6250d6', '#008300'], height=0.62)\n"
  "ax.set_yticks(range(7)); ax.set_yticklabels(names, fontsize=9.5); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('macro-F1 on the test tweets (%)')", save=SAVE0)

Q("confusion", "Confusion matrices of FinBERT on the test tweets, zero-shot and after fine-tuning (seed 1).",
  "confusion matrix, FinBERT, fine-tuning, domain shift, bearish news",
  "lectures/L7/results/test_predictions.csv (from data/clean/twitter_fin_sentiment.csv)", "Rows = observed class, columns = predicted class; 0 bearish, 1 bullish, 2 neutral.",
  "p = pd.read_csv(" + RES + " + 'test_predictions.csv')\nfrom sklearn.metrics import confusion_matrix\n"
  "cz, cf = confusion_matrix(p.y, p.finbert_zs, labels=[0, 1, 2]), confusion_matrix(p.y, p.finbert_ft, labels=[0, 1, 2])\n"
  "show('cm_zs.bear_ok', 'bearish found, zero-shot', int(cz[0, 0]))\nshow('cm.bear_ok', 'bearish found, fine-tuned', int(cf[0, 0]))\nshow('cm.bear_as_neutral', 'bearish read as neutral, fine-tuned', int(cf[0, 2]))\n"
  "cn = ['bearish', 'bullish', 'neutral']\nfig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))\nfor ax, M, ttl in [(axs[0], cz, 'FinBERT zero-shot'), (axs[1], cf, 'FinBERT fine-tuned')]:\n"
  "    Mn = M / M.sum(1, keepdims=True); ax.imshow(Mn, cmap='Blues', vmin=0, vmax=1)\n"
  "    for i in range(3):\n        for j in range(3):\n            ax.text(j, i, f'{M[i, j]:,}', ha='center', va='center', color='white' if Mn[i, j] > 0.6 else '#222222', fontsize=9)\n"
  "    ax.set_xticks(range(3)); ax.set_xticklabels(cn, fontsize=8.5); ax.set_yticks(range(3)); ax.set_yticklabels(cn, fontsize=8.5); ax.set_xlabel('predicted\\n' + ttl, fontsize=9)\n"
  "axs[0].set_ylabel('observed', fontsize=9)", save=SAVE0)

Q("data_scaling", "Test macro-F1 as a function of the number of training tweets (5-100 % of the training set, seed 1): TF-IDF, small transformer from scratch, frozen FinBERT + logistic regression.",
  "learning curve, sample size, pretraining, transfer learning, FinBERT",
  "lectures/L7/results/data_scaling.csv", "Stratified subsamples of the training tweets; the test set is fixed.",
  "s = pd.read_csv(" + RES + " + 'data_scaling.csv')\nshow('scaling.n_small', 'smallest training set', int(s.n[0]))\n"
  "show('scaling.probe_small', 'macro-F1, frozen FinBERT, smallest set (%)', round(100 * float(s.probe[0]), 1))\nshow('scaling.tfidf_small', 'macro-F1, TF-IDF, smallest set (%)', round(100 * float(s.tfidf[0]), 1))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nax.semilogx(s.n, 100 * s.tfidf, 'o-', color=PAL['benchmark'], label='TF-IDF + logistic')\n"
  "ax.semilogx(s.n, 100 * s.tiny, 's-', color=PAL['model'], label='small transformer')\nax.semilogx(s.n, 100 * s.probe, '^-', color='#6250d6', label='FinBERT frozen + logistic')\n"
  "ax.set_xticks(s.n); ax.set_xticklabels([f'{int(x):,}' for x in s.n]); ax.minorticks_off()\nax.set_xlabel('training tweets (log scale)'); ax.set_ylabel('macro-F1, test (%)')\nlegend_bottom(fig, ncol=3)")

if __name__ == "__main__":
    only = sys.argv[1:]
    bad_all, n_all = [], 0
    for q in QS:
        if only and q.chart not in only:
            continue
        q.write(execute=True)
        n, bad = check_numbers(q, os.path.join(HERE, "L7_numbers.json"))
        n_all += n
        bad_all += bad
        print(f"  {q.name}: {n} numbers checked", "OK" if not bad and n else (bad or "NO NUMBERS"))
    print(f"{n_all} numbers checked;", "OK" if not bad_all else bad_all)
    sys.exit(1 if bad_all else 0)
