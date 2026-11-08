"""
make_quantlets_L14.py -- one Quantlet per chart of Lecture 14 (Quantlets/L14/NN_L14_<chart>/), executed, and every number
printed with show() compared with L14_numbers.json.
CLIP embeddings of 27,000 images and the vision-language model need a graphics card (lecture notebook); the Quantlets recompute
the numbers from the per-image predictions in lectures/L14/results/, and run CLIP only on a few images for the attention chart.
Run:  python3 lectures/L14/make_quantlets_L14.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

NUM = os.path.join(HERE, "L14_numbers.json")
RES = "RES = REPO + 'lectures/L14/results/'\n"
KW = "vision-language models, CLIP, zero-shot classification, EuroSAT, satellite images, vision transformer"
EUROSAT = ("import torchvision\nroot = os.path.join(REPO, 'data', 'raw', 'vision') if os.path.isdir(os.path.join(REPO, 'data')) else 'vision'\n"
           "os.makedirs(root, exist_ok=True)\nES = torchvision.datasets.EuroSAT(root, download=True)\n")
qs = []

q = Quantlet(14, "zeroshot_prompts", "Zero-shot accuracy of CLIP ViT-B/32 and ViT-B/16 on 5,400 EuroSAT test images for four prompt designs: "
             "folder name, 'a photo of', 'a satellite photo of' and an ensemble of five satellite templates.", KW + ", prompt engineering",
             "lectures/L14/results/eurosat_zeroshot.csv (predictions of analysis_L14.py on EuroSAT)")
q.code(RES + "Z = pd.read_csv(RES + 'eurosat_zeroshot.csv')\nT = ['name', 'photo', 'sat', 'ens']\nacc = {}\n"
       "for k in ('B32', 'B16'):\n    for t in T:\n        acc[(k, t)] = round(100 * float((Z[f'{k}_{t}'] == Z['true']).mean()), 1)\n        show(f'zs.{k}_{t}', f'zero-shot accuracy {k}, {t} (%)', acc[(k, t)])\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nfor j, (k, c) in enumerate((('B32', PAL['data']), ('B16', PAL['model']))):\n"
       "    ax.bar(np.arange(4) + (j - 0.5) * 0.38, [acc[(k, t)] for t in T], 0.38, color=c, label='CLIP ViT-' + k)\n"
       "ax.axhline(10, ls=':', color='#999999', label='Chance')\nax.set_xticks(range(4))\nax.set_xticklabels(T)\nax.set_ylabel('Test accuracy (%)')\nax.set_ylim(0, 100)\n"
       "legend_bottom(fig, ncol=3)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('zeroshot_prompts.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(14, "zeroshot_confusion", "Confusion matrix (row percentages) of the best zero-shot CLIP classifier on the EuroSAT test set; per-class "
             "accuracy.", KW + ", confusion matrix", "lectures/L14/results/eurosat_zeroshot.csv")
q.code(RES + "import json, urllib.request\nZ = pd.read_csv(RES + 'eurosat_zeroshot.csv')\nCLS = sorted(Z['true'].unique())\n"
       "best = max([c for c in Z.columns if c[:3] in ('B32', 'B16')], key=lambda c: (Z[c] == Z['true']).mean())\n"
       "CM = pd.crosstab(Z['true'], Z[best]).reindex(index=CLS, columns=CLS, fill_value=0).values\n"
       "for i, c in enumerate(CLS):\n    show(f'zs_class.{c}', f'zero-shot accuracy, {c} (%)', round(100 * float(CM[i, i] / CM[i].sum()), 1))\n"
       "R = CM / CM.sum(1, keepdims=True) * 100\nfig, ax = plt.subplots(figsize=(6.4, 3.4))\nax.imshow(R, cmap='Blues', vmin=0, vmax=100)\n"
       "ax.set_xticks(range(10))\nax.set_xticklabels(CLS, rotation=40, ha='right', fontsize=8)\nax.set_yticks(range(10))\nax.set_yticklabels(CLS, fontsize=8)\n"
       "for a in range(10):\n    for b in range(10):\n        if R[a, b] >= 1:\n            ax.text(b, a, f'{R[a, b]:.0f}', ha='center', va='center', fontsize=7)\n"
       "ax.set_xlabel('Predicted class')\nax.set_ylabel('True class')\nfig.savefig('zeroshot_confusion.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(14, "text_search", "Text-to-image search on the EuroSAT test set with CLIP: the 100 images most similar to each class prompt and "
             "the share that truly belong to the class (precision at 100).", KW + ", image retrieval, precision at k",
             "lectures/L14/results/eurosat_scores_best.csv (image-prompt similarities of the best zero-shot CLIP model)")
q.code(RES + "Sc = pd.read_csv(RES + 'eurosat_scores_best.csv')\nCLS = [c for c in Sc.columns if c != 'true']\n"
       "prec = {c: float((Sc.sort_values(c, ascending=False).head(100)['true'] == c).mean()) for c in CLS}\n"
       "show('search.mean', 'mean precision at 100 (%)', round(100 * float(np.mean(list(prec.values()))), 1))\n"
       "for c in CLS:\n    show(f'search.per_class.{c}', f'precision at 100, {c} (%)', round(100 * prec[c], 1))\n"
       "o = sorted(CLS, key=lambda c: -prec[c])\nfig, ax = plt.subplots(figsize=(6.4, 2.4))\nax.bar(range(10), [100 * prec[c] for c in o], color=PAL['model'])\n"
       "ax.axhline(10, ls=':', color='#999999')\nax.set_xticks(range(10))\nax.set_xticklabels(o, rotation=30, ha='right', fontsize=8)\nax.set_ylabel('Precision at 100 (%)')\n"
       "fig.tight_layout()\nfig.savefig('text_search.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(14, "fewshot_eurosat", "EuroSAT test accuracy against labelled images per class: logistic regression on CLIP ViT-B/16 embeddings "
             "(5 seeds) and a CNN trained from scratch (3 seeds), with zero-shot CLIP and the CNN on all training images as references.",
             KW + ", few-shot learning, linear probe, transfer learning", "lectures/L14/results/fewshot_eurosat.csv, eurosat_zeroshot.csv")
q.code(RES + "Fq = pd.read_csv(RES + 'fewshot_eurosat.csv')\nZ = pd.read_csv(RES + 'eurosat_zeroshot.csv')\n"
       "g = Fq.groupby(['method', 'k'])['acc'].mean().mul(100)\n"
       "for k in (16, 128):\n    show(f'fewshot.probe.{k}.mean', f'probe, {k} per class (%)', round(float(g[('probe', k)]), 1))\n"
       "    show(f'fewshot.cnn.{k}.mean', f'CNN, {k} per class (%)', round(float(g[('cnn', k)]), 1))\n"
       "show('cnn_full.mean', 'CNN on all training images (%)', round(float(Fq[Fq.method == 'cnn_full'].acc.mean() * 100), 1))\n"
       "zs = max(100 * (Z[c] == Z['true']).mean() for c in Z.columns if c[:3] in ('B32', 'B16'))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.6))\npk = g['probe']\nck = g['cnn']\n"
       "ax.plot(np.log2(pk.index), pk.values, marker='o', color=PAL['model'], label='CLIP ViT-B/16 + logistic')\n"
       "ax.plot(np.log2(ck.index), ck.values, marker='s', color=PAL['benchmark'], label='CNN from scratch')\n"
       "ax.axhline(zs, ls='--', color=PAL['alt'], label='CLIP zero-shot')\nax.set_xticks(np.log2(pk.index))\nax.set_xticklabels(pk.index)\n"
       "ax.set_xlabel('Labelled images per class')\nax.set_ylabel('Test accuracy (%)')\nax.set_ylim(0, 100)\nlegend_bottom(fig, ncol=3)\n"
       "fig.tight_layout(rect=(0, 0.1, 1, 1))\nfig.savefig('fewshot_eurosat.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(14, "zeroshot_examples", "Eight random EuroSAT test images with their true class and the zero-shot CLIP prediction.", KW,
             "EuroSAT (torchvision, on demand); lectures/L14/results/eurosat_zeroshot.csv")
q.code(RES + EUROSAT + "Z = pd.read_csv(RES + 'eurosat_zeroshot.csv')\nbest = max([c for c in Z.columns if c[:3] in ('B32', 'B16')], key=lambda c: (Z[c] == Z['true']).mean())\n"
       "rng = np.random.default_rng(20261108)\nyte = Z['true'].map({c: i for i, c in enumerate(ES.classes)}).values\n"
       "ex = [int(rng.choice(np.where(yte == c)[0])) for c in rng.choice(10, 8, replace=False)]\n"
       "show('examples.n_right', 'correct among the 8 examples', int(sum(Z[best][e] == Z['true'][e] for e in ex)))\n"
       "fig, axs = plt.subplots(2, 4, figsize=(6.4, 3.6))\nfor a, e in zip(axs.ravel(), ex):\n    a.imshow(ES[int(Z['index'][e])][0])\n    a.set_xticks([]); a.set_yticks([])\n"
       "    a.set_xlabel(f\"true: {Z['true'][e]}\\nCLIP: {Z[best][e]}\", fontsize=7, color='green' if Z[best][e] == Z['true'][e] else 'red')\n"
       "fig.savefig('zeroshot_examples.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(14, "vit_patches_attention", "Patch grid (14 x 14 patches of 16 x 16 pixels) and attention rollout of the class token of CLIP ViT-B/16 "
             "on four EuroSAT test images (highway, river, industrial, residential).", KW + ", attention rollout",
             "EuroSAT (torchvision, on demand); model openai/clip-vit-base-patch16 (Hugging Face)")
q.code(RES + EUROSAT + "import torch\nfrom PIL import Image\nfrom transformers import CLIPModel, CLIPProcessor\n"
       "m = CLIPModel.from_pretrained('openai/clip-vit-base-patch16', attn_implementation='eager').eval()\npr = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')\n"
       "Z = pd.read_csv(RES + 'eurosat_zeroshot.csv')\nidx = [int(Z['index'][np.where(Z['true'] == c)[0][3]]) for c in ('Highway', 'River', 'Industrial', 'Residential')]\n"
       "show('vit.layers', 'transformer layers', len(m.vision_model.encoder.layers))\nshow('vit.heads', 'attention heads', int(m.config.vision_config.num_attention_heads))\n"
       "fig, axs = plt.subplots(2, 4, figsize=(6.4, 3.4))\nfor c, i in enumerate(idx):\n    img = ES[i][0]\n"
       "    with torch.no_grad():\n        out = m.vision_model(pixel_values=pr(images=[img], return_tensors='pt')['pixel_values'], output_attentions=True)\n"
       "    R = torch.eye(197)\n    for A in out.attentions:\n        A = A.mean(1)[0] + torch.eye(197)\n        R = (A / A.sum(-1, keepdim=True)) @ R\n"
       "    att = R[0, 1:].reshape(14, 14).numpy()\n    big = np.asarray(img.resize((224, 224), Image.NEAREST))\n    axs[0, c].imshow(big)\n"
       "    for g in range(0, 225, 16):\n        axs[0, c].axhline(g - 0.5, color='white', lw=0.4); axs[0, c].axvline(g - 0.5, color='white', lw=0.4)\n"
       "    axs[1, c].imshow(big)\n    axs[1, c].imshow(np.kron(att, np.ones((16, 16))), cmap='inferno', alpha=0.55)\n"
       "    for a in axs[:, c]:\n        a.set_xticks([]); a.set_yticks([])\n"
       "fig.savefig('vit_patches_attention.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

q = Quantlet(14, "chart_regime", "Market-stress detection on S&P 500 chart windows (120 trading days): logistic regression on numbers, logistic "
             "regression on CLIP image embeddings, CLIP zero-shot and Qwen2-VL zero-shot; ROC curves and AUC on 2019-2026, after Jheng, Pele, Tak "
             "and Găman (2026).", "vision-language models, CLIP, Qwen2-VL, market regimes, anomaly detection, S&P 500, ROC",
             "lectures/L14/results/chart_regime.csv (scores of analysis_L14.py; data/clean/markets_daily.csv)")
q.code(RES + "from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve\nC = pd.read_csv(RES + 'chart_regime.csv')\nT = C[C.split == 'test']\n"
       "S = {'numeric': T.numeric, 'clip_probe': T.clip_probe, 'clip_zs': T.clip_zs, 'vlm_zs': T.vlm_stress}\n"
       "for k, s in S.items():\n    show(f'regime.{k}.auc', f'AUC {k}', round(float(roc_auc_score(T.stress, s)), 3))\n"
       "    show(f'regime.{k}.ap', f'average precision {k}', round(float(average_precision_score(T.stress, s)), 3))\n"
       "show('charts_task.base_test', 'stress share in the test windows (%)', round(100 * float(T.stress.mean()), 1))\n"
       "fig, ax = plt.subplots(figsize=(6.4, 2.6))\nfor (k, s), c in zip(S.items(), [PAL['benchmark'], PAL['model'], PAL['alt'], '#6250d6']):\n"
       "    fpr, tpr, _ = roc_curve(T.stress, s)\n    ax.plot(fpr, tpr, color=c, label=k)\nax.plot([0, 1], [0, 1], ls=':', color='#999999')\n"
       "ax.set_xlabel('False positive rate')\nax.set_ylabel('True positive rate')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.1, 1, 1))\n"
       "fig.savefig('chart_regime.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")
qs.append(q)

bad_all, n_all = [], 0
for q in qs:
    q.write(execute=True)
    n, bad = check_numbers(q, NUM)
    n_all += n
    bad_all += bad
    print(f"  {q.name}: {n} numbers checked", "OK" if not bad else bad)
print(f"{len(qs)} Quantlets, {n_all} numbers checked against L14_numbers.json:", "OK" if not bad_all else bad_all)
sys.exit(1 if bad_all else 0)
