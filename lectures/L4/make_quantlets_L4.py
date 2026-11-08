"""
make_quantlets_L4.py -- one Quantlet per chart of Lecture 4: Quantlets/L4/NN_L4_<chart>/ (Metainfo.txt, executed notebook,
<chart>.png). Charts of trained networks read <chart>_results.json, the saved outputs of analysis_L4.py (its cache), written
next to the notebook; the training code itself is in analysis_L4.py and in notebooks/L4_notebook.ipynb.
Every number printed with show() is compared with L4_numbers.json.
Run (after analysis_L4.py):  python3 lectures/L4/make_quantlets_L4.py
"""
import json
import os
import runpy
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

print("loading the results of analysis_L4.py (from its cache) ...")
A = runpy.run_path(os.path.join(HERE, "analysis_L4.py"), run_name="analysis_L4_results")
NUM = os.path.join(HERE, "L4_numbers.json")
K = 4
VISION = ("VISION = REPO + 'data/raw/vision' if os.path.isdir(REPO + 'data') else 'vision'\nos.makedirs(VISION, exist_ok=True)\n"
          "import torch, torch.nn.functional as F, torchvision")


def loader(chart):
    return ("import json, urllib.request\n"
            f"_F = '{chart}_results.json'\n"
            f"R = json.load(open(_F)) if os.path.isfile(_F) else json.load(urllib.request.urlopen(REPO + 'Quantlets/L{K}/NN_L{K}_{chart}/' + _F))")


def save_results(chart, obj):
    d = os.path.join(HERE, "..", "..", "Quantlets", f"L{K}", f"NN_L{K}_{chart}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{chart}_results.json"), "w") as f:
        json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def tolist(x):
    return np.asarray(x).tolist()


QS = []


ONLY = [a for a in sys.argv[1:] if not a.startswith('-')]


def make(chart, description, keywords, datafile, intro, code, results=None):
    if ONLY and chart not in ONLY:
        return
    if results is not None:
        save_results(chart, results)
        code = loader(chart) + "\n" + code
    q = Quantlet(k=K, chart=chart, description=description, keywords=keywords, datafile=datafile)
    q.md(intro)
    q.code(code)
    q.write(execute=True)
    n, bad = check_numbers(q, NUM)
    print(f"  {q.name}: {n} numbers checked", "OK" if not bad else bad)
    QS.append((q.name, n, bad))


SAVE = "fig.savefig('{c}.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()"
FM_EN = A["FM_EN"]
ES_EN = A["ES_EN"]
N = A["N"]

# 1 Fashion-MNIST samples
make("fmnist_samples", "Two training images of each of the ten Fashion-MNIST product classes (Zalando catalogue photos, 28 x 28 grey levels).",
     "Fashion-MNIST, images, product classification, convolutional neural network, e-commerce",
     "Fashion-MNIST (Zalando Research), downloaded with torchvision",
     "Fashion-MNIST: the same validation split as the lecture (seed 20261108), two training images per class.",
     VISION + "\nfm = torchvision.datasets.FashionMNIST(VISION, train=True, download=True)\nX = fm.data.numpy() / 255; y = fm.targets.numpy()\n"
     "itr = np.random.RandomState(20261108).permutation(len(X))[:50000]\n"
     f"CLASSES = {FM_EN}\n"
     "show('fmnist.n_train', 'training images', len(itr))\n"
     "fig, axes = plt.subplots(2, 10, figsize=(6.4, 1.75))\nfor c in range(10):\n    ids = np.where(y[itr] == c)[0][:2]\n"
     "    for r in range(2):\n        axes[r, c].imshow(X[itr][ids[r]], cmap='gray_r'); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])\n"
     "    axes[1, c].set_xlabel(CLASSES[c], fontsize=7.5)\n" + SAVE.format(c="fmnist_samples"))

# 2 hand-made filters
make("conv_edges", "Sobel edge filters and a mean filter applied by convolution (padding 1) to a real 64 x 64 Sentinel-2 tile (EuroSAT, class Highway).",
     "convolution, Sobel filter, edge detection, padding, stride, satellite images, EuroSAT",
     "EuroSAT (Helber et al., 2019), Sentinel-2 RGB tiles, downloaded with torchvision",
     "Output size $\\lfloor (H+2p-k)/s\\rfloor+1$ and three fixed 3 x 3 kernels on a real tile.",
     VISION + "\nfrom PIL import Image\ntorchvision.datasets.EuroSAT(VISION, download=True)\nd = os.path.join(VISION, 'eurosat', '2750', 'Highway')\n"
     "img = np.asarray(Image.open(os.path.join(d, sorted(os.listdir(d))[5])).convert('RGB')).astype(np.float32).mean(2) / 255\n"
     "Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)\nker = [Kx, Kx.T, np.ones((3, 3), np.float32) / 9]\n"
     "out = [F.conv2d(torch.tensor(img)[None, None], torch.tensor(k)[None, None], padding=1)[0, 0].numpy() for k in ker]\n"
     "osz = lambda H, k, p, s: (H + 2 * p - k) // s + 1\nshow('conv.out_same', 'k=3, p=1, s=1', osz(64, 3, 1, 1))\nshow('conv.out_valid', 'k=3, p=0, s=1', osz(64, 3, 0, 1))\n"
     "show('conv.out_s2', 'k=3, p=1, s=2', osz(64, 3, 1, 2))\n"
     "fig, axes = plt.subplots(1, 4, figsize=(6.4, 1.9))\nfor ax, im, cm, l in zip(axes, [img] + out, ['gray', 'RdBu_r', 'RdBu_r', 'gray'],\n"
     "                             ['Input (highway)', 'Vertical edges', 'Horizontal edges', 'Mean 3x3 (blur)']):\n"
     "    ax.imshow(im, cmap=cm); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(l, fontsize=8.5)\n" + SAVE.format(c="conv_edges"))

# 3 receptive field
make("receptive_field", "Receptive field of stacked 3 x 3 convolutions, with and without 2 x 2 pooling, from the recursion r_l = r_(l-1) + (k_l - 1) prod s_i.",
     "receptive field, convolution, pooling, stride, depth, CNN architecture", "none (formula)",
     "Receptive field $r_l=r_{l-1}+(k_l-1)\\prod_{i<l}s_i$, $r_0=1$.",
     "def rf(layers):\n    r, j, out = 1, 1, []\n    for k, s in layers:\n        r += (k - 1) * j; j *= s; out.append(r)\n    return out\n"
     "plain = rf([(3, 1)] * 12)\npooled = [v for i, v in enumerate(rf([(3, 1), (3, 1), (2, 2)] * 4)) if i % 3 != 2][:8]\n"
     "show('rf.plain_12', 'twelve 3x3 layers', plain[11]); show('rf.pooled_8conv', 'eight 3x3 layers with pooling', pooled[7])\n"
     "fig, ax = plt.subplots(figsize=(6.4, 2.2))\nax.plot(range(1, 13), plain, marker='o', ms=3, color=PAL['benchmark'], label='3x3 convolutions, stride 1')\n"
     "ax.plot(range(1, 9), pooled, marker='s', ms=3, color=PAL['model'], label='pairs of 3x3 convolutions + 2x2 pooling')\n"
     "ax.set_xlabel('Number of convolutional layers'); ax.set_ylabel('Receptive field (pixels)')\nlegend_bottom(fig, ncol=2)\n"
     "fig.tight_layout(rect=(0, 0.15, 1, 1))\n" + SAVE.format(c="receptive_field"))

# 4 learning curves
FM = A["FM"]
make("fmnist_curves", "Training and validation cross-entropy and validation accuracy by epoch of softmax regression, MLP and CNN on Fashion-MNIST (seed 20261108).",
     "learning curves, cross-entropy, CNN, MLP, softmax regression, Fashion-MNIST, early stopping",
     "Fashion-MNIST; saved training histories (fmnist_curves_results.json, from lectures/L4/cache/fmnist.pkl)",
     "Learning curves of the three models of the lecture (AdamW, batch 128, 12 epochs, one-cycle learning rate). Retrain with lectures/L4/analysis_L4.py.",
     "for k in ['softmax', 'mlp', 'cnn']:\n    show(f'fmnist.{k}.best_epoch', f'{k}: best validation epoch', int(np.argmax(R[k]['valid_acc']) + 1))\n"
     "fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5)); ep = np.arange(1, 13)\n"
     "for k, c, l in [('softmax', PAL['benchmark'], 'Softmax regression'), ('mlp', PAL['alt'], 'MLP'), ('cnn', PAL['model'], 'CNN')]:\n"
     "    axes[0].plot(ep, R[k]['train_loss'], color=c, ls='--', lw=1.2); axes[0].plot(ep, R[k]['valid_loss'], color=c, lw=1.6, label=l)\n"
     "    axes[1].plot(ep, 100 * np.array(R[k]['valid_acc']), color=c, lw=1.6, marker='o', ms=2.5)\n"
     "axes[0].set_ylabel('Cross-entropy'); axes[1].set_ylabel('Validation accuracy (%)')\nfor ax in axes: ax.set_xlabel('Epoch')\n"
     "axes[0].plot([], [], color='k', ls='--', label='training (dashed)')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.15, 1, 1))\n" + SAVE.format(c="fmnist_curves"),
     {k: FM[k][0]["hist"] for k in ["softmax", "mlp", "cnn"]})

# 5 compare
make("fmnist_compare", "Test accuracy (mean and standard deviation over three seeds) and number of parameters of softmax regression, MLP and CNN on Fashion-MNIST.",
     "CNN, MLP, parameters, weight sharing, test accuracy, seeds, McNemar test",
     "Fashion-MNIST; saved test accuracies of 3 seeds per model and test predictions (fmnist_compare_results.json)",
     "Accuracy and size of the three models; McNemar's test CNN vs MLP on the 10,000 test images (seed 20261108).",
     "for k in ['softmax', 'mlp', 'cnn']:\n    a = 100 * np.array(R[k]['acc'])\n"
     "    show(f'fmnist.{k}.acc_mean', f'{k}: mean test accuracy (%)', round(float(a.mean()), 2)); show(f'fmnist.{k}.acc_sd', f'{k}: sd', round(float(a.std(ddof=1)), 2))\n"
     "    show(f'fmnist.{k}.params', f'{k}: parameters', R[k]['params'])\n"
     "y, pc, pm = np.array(R['y']), np.array(R['cnn_pred']), np.array(R['mlp_pred'])\nb, c = int(((pc == y) & (pm != y)).sum()), int(((pc != y) & (pm == y)).sum())\n"
     "show('fmnist.mcnemar.cnn_only', 'right only for the CNN', b); show('fmnist.mcnemar.mlp_only', 'right only for the MLP', c)\n"
     "show('fmnist.mcnemar.chi2', 'McNemar chi2', round((abs(b - c) - 1) ** 2 / (b + c), 1))\n"
     "fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3)); lab = ['Softmax\\nregression', 'MLP', 'CNN']; cols = [PAL['benchmark'], PAL['alt'], PAL['model']]\n"
     "acc = [100 * np.mean(R[k]['acc']) for k in ['softmax', 'mlp', 'cnn']]; sd = [100 * np.std(R[k]['acc'], ddof=1) for k in ['softmax', 'mlp', 'cnn']]\n"
     "axes[0].bar(lab, acc, yerr=sd, color=cols, capsize=3); axes[0].set_ylim(80, 95); axes[0].set_ylabel('Test accuracy (%)')\n"
     "axes[1].bar(lab, [R[k]['params'] / 1000 for k in ['softmax', 'mlp', 'cnn']], color=cols); axes[1].set_ylabel('Parameters (thousands)')\n"
     "fig.tight_layout()\n" + SAVE.format(c="fmnist_compare"),
     {**{k: {"acc": [r["test_acc"] for r in FM[k]], "params": FM[k][0]["params"]} for k in ["softmax", "mlp", "cnn"]},
      "y": tolist(A["yfm_te"]), "cnn_pred": tolist(FM["cnn"][0]["pred"]), "mlp_pred": tolist(FM["mlp"][0]["pred"])})

# 6 confusion
make("fmnist_confusion", "Confusion matrix of the CNN on the 10,000 Fashion-MNIST test images (seed 20261108).",
     "confusion matrix, recall, CNN, Fashion-MNIST, product classification",
     "Fashion-MNIST; saved confusion matrix (fmnist_confusion_results.json)",
     "Where the CNN errs: recall by class and the largest confusion.",
     f"CLASSES = {FM_EN}\ncm = np.array(R['cm']); rec = cm.diagonal() / cm.sum(1)\n"
     "show('fmnist.worst_recall', 'lowest recall (%), ' + CLASSES[int(rec.argmin())], round(100 * float(rec.min()), 1))\n"
     "off = cm.copy(); np.fill_diagonal(off, 0); i, j = np.unravel_index(off.argmax(), off.shape)\nshow('fmnist.top_conf.n', f'{CLASSES[i]} taken for {CLASSES[j]}', int(off[i, j]))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 3.3)); cmn = cm / cm.sum(1, keepdims=True) * 100\nax.imshow(cmn, cmap='Blues', vmin=0, vmax=100, aspect='auto')\n"
     "for i in range(10):\n    for j in range(10):\n        if cm[i, j] >= 10: ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6.5, color='white' if cmn[i, j] > 50 else '#222222')\n"
     "ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_xticklabels(CLASSES, rotation=40, ha='right', fontsize=7.5); ax.set_yticklabels(CLASSES, fontsize=7.5)\n"
     "ax.set_xlabel('Predicted class', fontsize=9); ax.set_ylabel('True class', fontsize=9)\n" + SAVE.format(c="fmnist_confusion"),
     {"cm": tolist(A["cm"])})

# 7 filters
make("fmnist_filters", "The 32 learned 3 x 3 filters of the first convolutional layer of the Fashion-MNIST CNN (seed 20261108).",
     "learned filters, convolution kernels, edge detectors, CNN, Fashion-MNIST",
     "Fashion-MNIST; saved first-layer weights of the trained CNN (fmnist_filters_results.json)",
     "First-layer kernels of the trained CNN: red positive, blue negative weights.",
     "W1 = np.array(R['W1']); show('fmnist.cnn.params', 'parameters of the CNN', R['params'])\nfig, axes = plt.subplots(2, 16, figsize=(6.4, 1.0)); v = np.abs(W1).max()\n"
     "for i, ax in enumerate(axes.flat):\n    ax.imshow(W1[i], cmap='RdBu_r', vmin=-v, vmax=v); ax.set_xticks([]); ax.set_yticks([])\n" + SAVE.format(c="fmnist_filters"),
     {"W1": tolist(A["W1"]), "params": FM["cnn"][0]["params"]})

# 8 feature maps
make("fmnist_featuremaps", "One Fashion-MNIST test image (ankle boot) and the eight most active feature maps of blocks 1 and 3 of the trained CNN.",
     "feature maps, activations, CNN, hierarchy of features, Fashion-MNIST",
     "Fashion-MNIST; saved feature maps of the trained CNN (fmnist_featuremaps_results.json)",
     "Feature maps of block 1 (14 x 14, after pooling) and block 3 (7 x 7).",
     "img = np.array(R['img']); maps = [np.array(R['m1']), np.array(R['m3'])]\nshow('fmnist.fmap_l3_size', 'size of the block-3 maps', maps[1].shape[1])\n"
     "fig, axes = plt.subplots(2, 9, figsize=(6.4, 1.6))\nfor r in range(2):\n    axes[r, 0].imshow(img, cmap='gray_r')\n"
     "    order = np.argsort(-maps[r].mean((1, 2)))[:8]\n    for c in range(8): axes[r, c + 1].imshow(maps[r][order[c]], cmap='viridis')\n"
     "    for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])\naxes[0, 0].set_ylabel('block 1\\n14x14', fontsize=7.5); axes[1, 0].set_ylabel('block 3\\n7x7', fontsize=7.5)\n" + SAVE.format(c="fmnist_featuremaps"),
     {"img": tolist(A["Xfm_te"][A["img_id"]]), "m1": tolist(A["maps"][0]), "m3": tolist(A["maps"][1])})

# 9 cifar
CF = A["CF"]
make("cifar_curves", "CIFAR-10: a small CNN with and without data augmentation, and plain vs residual 32-layer networks (training loss), seed 20261108.",
     "CIFAR-10, data augmentation, ResNet, residual connection, degradation problem, CNN",
     "CIFAR-10 (Krizhevsky, 2009); saved training histories and test accuracies (cifar_curves_results.json)",
     "Augmentation narrows the train-validation gap; the shortcut makes a 32-layer network trainable.",
     "for k in ['cnn', 'cnn_aug', 'plain32', 'res32']:\n    show(f'cifar.{k}.test_acc', f'{k}: test accuracy (%)', round(100 * R[k]['test_acc'], 2))\n"
     "show('cifar.plain32.train_loss', 'plain-32 final training loss', round(R['plain32']['train_loss'][-1], 3)); show('cifar.res32.train_loss', 'ResNet-32 final training loss', round(R['res32']['train_loss'][-1], 3))\n"
     "fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5)); ep = np.arange(1, len(R['cnn']['train_loss']) + 1)\n"
     "for k, c, l in [('cnn', PAL['benchmark'], 'CNN, no augmentation'), ('cnn_aug', PAL['model'], 'CNN, augmentation')]:\n"
     "    axes[0].plot(ep, 100 * np.array(R[k]['train_acc']), color=c, ls='--', lw=1.2); axes[0].plot(ep, 100 * np.array(R[k]['valid_acc']), color=c, lw=1.6, label=l)\n"
     "for k, c, l in [('plain32', PAL['alt'], 'Plain, 32 layers'), ('res32', '#6250d6', 'ResNet, 32 layers')]:\n    axes[1].plot(ep, R[k]['train_loss'], color=c, lw=1.6, label=l)\n"
     "axes[0].set_ylabel('Accuracy (%)'); axes[1].set_ylabel('Training loss')\nfor ax in axes: ax.set_xlabel('Epoch')\n"
     "axes[0].plot([], [], color='k', ls='--', label='training (dashed)'); legend_bottom(fig, ncol=3)\nfig.tight_layout(rect=(0, 0.2, 1, 1))\n" + SAVE.format(c="cifar_curves"),
     {k: {**CF[k]["hist"], "test_acc": CF[k]["test_acc"]} for k in ["cnn", "cnn_aug", "plain32", "res32"]})

# 10 eurosat samples
make("eurosat_samples", "Two training tiles of each of the ten EuroSAT land-use classes (Sentinel-2, 64 x 64 RGB).",
     "EuroSAT, Sentinel-2, satellite images, land use, agriculture, insurance, real estate",
     "EuroSAT (Helber et al., 2019), downloaded with torchvision",
     "The ten land-use classes; same stratified split as the lecture (seed 20261108).",
     VISION + "\nfrom PIL import Image\ntorchvision.datasets.EuroSAT(VISION, download=True)\nroot = os.path.join(VISION, 'eurosat', '2750')\n"
     "DIRS = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']\n"
     f"NAMES = {ES_EN}\nrs = np.random.RandomState(20261108); n_all = 0\n"
     "fig, axes = plt.subplots(2, 10, figsize=(6.4, 1.8))\nfor c, d in enumerate(DIRS):\n    files = sorted(f for f in os.listdir(os.path.join(root, d)) if f.endswith('.jpg')); n_all += len(files)\n"
     "    tr = np.sort(rs.permutation(len(files))[:int(0.6 * len(files))])\n"
     "    for r in range(2):\n        axes[r, c].imshow(Image.open(os.path.join(root, d, files[tr[r]]))); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])\n"
     "    axes[1, c].set_xlabel(NAMES[c].replace(' ', '\\n', 1), fontsize=6.8)\nshow('eurosat.n', 'tiles', n_all)\n" + SAVE.format(c="eurosat_samples"))

# 11 transfer curves
ES = A["ES"]
BU = A["BU"]
make("transfer_curves", "EuroSAT: validation accuracy of a CNN from scratch and of a fine-tuned ResNet-18, frozen ResNet-18 + logistic regression, and test accuracy against the label budget.",
     "transfer learning, fine-tuning, feature extraction, ResNet-18, ImageNet, label budget, EuroSAT",
     "EuroSAT; saved learning curves and test accuracies (transfer_curves_results.json, from lectures/L4/cache)",
     "Transfer learning: from scratch vs frozen features vs fine-tuning, and what happens with few labels.",
     "show('eurosat.scratch_acc', 'scratch, test accuracy (%)', round(100 * R['scratch_test'], 2)); show('eurosat.finetune_acc', 'fine-tuned, test accuracy (%)', round(100 * R['finetune_test'], 2))\n"
     "show('eurosat.probe_acc', 'frozen + logistic, test accuracy (%)', round(100 * R['probe_test'], 2))\nB = R['budget']; n = [10, 25, 50, 100, 250, 1000]\n"
     "for k in ['scratch', 'finetune', 'probe']:\n    show(f'budget.{k}.10', f'{k}, 10 tiles/class (%)', round(100 * float(np.mean(B[k]['10'])), 1))\n"
     "fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))\nfor k, c, l in [('scratch', PAL['benchmark'], 'CNN from scratch'), ('finetune', PAL['model'], 'ResNet-18 fine-tuned')]:\n"
     "    h = np.array(R[k + '_hist']); axes[0].plot(h[:, 0], 100 * h[:, 1], color=c, lw=1.6, marker='o', ms=2.5, label=l)\n"
     "axes[0].axhline(100 * R['probe_valid'], color=PAL['alt'], lw=1.4, ls='--', label='ResNet-18 frozen + logistic')\n"
     "axes[0].set_xlabel('Epochs (passes over the training set)', fontsize=9); axes[0].set_ylabel('Validation accuracy (%)'); axes[0].set_ylim(50, 100)\n"
     "for k, c in [('scratch', PAL['benchmark']), ('finetune', PAL['model']), ('probe', PAL['alt'])]:\n"
     "    axes[1].plot(n, [100 * np.mean(B[k][str(x)]) for x in n], color=c, lw=1.6, marker='o', ms=3)\n"
     "axes[1].set_xscale('log'); axes[1].set_xticks(n); axes[1].set_xticklabels([str(x) for x in n])\n"
     "axes[1].set_xlabel('Labelled tiles per class', fontsize=9); axes[1].set_ylabel('Test accuracy (%)')\nlegend_bottom(fig, ncol=3)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n" + SAVE.format(c="transfer_curves"),
     {"scratch_hist": ES["scratch"]["hist"], "finetune_hist": ES["finetune"]["hist"], "probe_valid": ES["probe"]["valid_acc"],
      "scratch_test": ES["scratch"]["test_acc"], "finetune_test": ES["finetune"]["test_acc"], "probe_test": ES["probe"]["test_acc"],
      "budget": {k: {str(n): BU[k][n] for n in A["BUDGET"]} for k in ["scratch", "finetune", "probe"]}})

# 12 eurosat confusion
make("eurosat_confusion", "Confusion matrix of the fine-tuned ResNet-18 on the 5,400 EuroSAT test tiles (seed 20261108).",
     "confusion matrix, macro-F1, transfer learning, ResNet-18, EuroSAT, land use",
     "EuroSAT; saved confusion matrix (eurosat_confusion_results.json)",
     "Errors of the fine-tuned network: number of errors, largest confusion, macro-F1.",
     f"NAMES = {ES_EN}\ncm = np.array(R['cm']); off = cm.copy(); np.fill_diagonal(off, 0)\nshow('eurosat.n_errors_ft', 'errors on the test tiles', int(off.sum()))\n"
     "P = cm.diagonal() / cm.sum(0); Rc = cm.diagonal() / cm.sum(1)\nshow('eurosat.macro_f1_ft', 'macro-F1 (%)', round(100 * float(np.mean(2 * P * Rc / (P + Rc))), 2))\n"
     "fig, ax = plt.subplots(figsize=(6.4, 3.3)); cmn = cm / cm.sum(1, keepdims=True) * 100\nax.imshow(cmn, cmap='Oranges', vmin=0, vmax=100, aspect='auto')\n"
     "for i in range(10):\n    for j in range(10):\n        if cm[i, j] >= 1: ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6.5, color='white' if cmn[i, j] > 50 else '#222222')\n"
     "ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_xticklabels(NAMES, rotation=40, ha='right', fontsize=7.2); ax.set_yticklabels(NAMES, fontsize=7.2)\n"
     "ax.set_xlabel('Predicted class', fontsize=9); ax.set_ylabel('True class', fontsize=9)\n" + SAVE.format(c="eurosat_confusion"),
     {"cm": tolist(A["cme"])})

# 13 grad-cam
make("gradcam", "Grad-CAM maps of the fine-tuned ResNet-18 for six EuroSAT test tiles: where in the tile the class decision comes from.",
     "Grad-CAM, class activation map, explainability, ResNet-18, EuroSAT, satellite images",
     "EuroSAT; saved tiles, Grad-CAM maps and predicted probabilities (gradcam_results.json)",
     "$M^c=\\mathrm{ReLU}(\\sum_o\\alpha_o^cA_o)$, $\\alpha_o^c$ the mean gradient of the class score over map $o$ (block 3, 7 x 7, upsampled to 64 x 64).",
     f"NAMES = {ES_EN}\nimgs = np.array(R['imgs'], dtype=np.uint8); cams = np.array(R['cams'])\nshow('gradcam.n', 'tiles shown', len(imgs))\n"
     "fig, axes = plt.subplots(2, 6, figsize=(6.4, 2.3))\nfor j in range(6):\n    axes[0, j].imshow(imgs[j]); axes[1, j].imshow(imgs[j].mean(2), cmap='gray'); axes[1, j].imshow(cams[j], cmap='jet', alpha=0.45)\n"
     "    for r in range(2): axes[r, j].set_xticks([]); axes[r, j].set_yticks([])\n"
     "    axes[1, j].set_xlabel(NAMES[R['cls'][j]].replace(' ', '\\n', 1) + f\"\\n{100 * R['prob'][j]:.0f}%\", fontsize=7)\n" + SAVE.format(c="gradcam"),
     {"imgs": tolist(A["Xes"][A["cam_ids"]]), "cams": tolist(np.round(np.array(A["cams"]), 4)),
      "cls": [c for c, _ in A["probs"]], "prob": [p for _, p in A["probs"]]})

bad = [q for q in QS if q[2] or q[1] == 0]
print(f"{len(QS)} Quantlets written;", "all numbers match" if not bad else f"problems: {bad}")
sys.exit(1 if bad else 0)
