"""
make_notebook_L4.py -- Lecture 4 notebooks (CNN and transfer learning), EN + RO, executed:
notebooks/L4_notebook[_ro].ipynb and the seminar notebooks L4_seminar[_ro].ipynb (+ _solutions, git-ignored).
Shorter training than analysis_L4.py (one seed, fewer epochs), so numbers are close to, not identical with, the slides.
Run:  python3 lectures/L4/make_notebook_L4.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Convolutional neural networks (CNN) and transfer learning", "Rețele neuronale convoluționale (CNN) și transferul învățării")

SETUP = r"""import torch, torch.nn as nn, torch.nn.functional as F, torchvision
VISION = '../data/raw/vision'                 # torchvision downloads Fashion-MNIST and EuroSAT here on first use
os.makedirs(VISION, exist_ok=True)
fm_tr = torchvision.datasets.FashionMNIST(VISION, train=True, download=True)
fm_te = torchvision.datasets.FashionMNIST(VISION, train=False, download=True)
X = fm_tr.data.numpy().astype(np.float32) / 255; y = fm_tr.targets.numpy()
Xte = fm_te.data.numpy().astype(np.float32) / 255; yte = fm_te.targets.numpy()
perm = np.random.RandomState(SEED).permutation(len(X)); itr, iva = perm[:50000], perm[50000:]
MU, SD = X[itr].mean(), X[itr].std()            # scaling fitted on the training part only
T = lambda a: torch.tensor((a - MU) / SD).unsqueeze(1)
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(len(itr), len(iva), len(Xte), round(float(MU), 4), round(float(SD), 4))"""

TRAIN = r"""def evaluate(m, Xt, yt, bs=1000):
    m.eval(); out = []
    with torch.no_grad():
        for j in range(0, len(Xt), bs):
            out.append(m(Xt[j:j + bs].to(DEVICE)).cpu())
    p = torch.cat(out).argmax(1).numpy()
    return float((p == yt).mean()), p

def fit(m, Xtr, ytr, Xva, yva, epochs=6, lr=1e-3, bs=128):
    torch.manual_seed(SEED); m = m.to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3 * lr, total_steps=epochs * int(np.ceil(len(Xtr) / bs)))
    yt = torch.tensor(ytr); best, state = -1, None
    for ep in range(epochs):
        m.train(); idx = torch.randperm(len(Xtr))
        for j in range(0, len(idx), bs):
            b = idx[j:j + bs]
            loss = F.cross_entropy(m(Xtr[b].to(DEVICE)), yt[b].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        va, _ = evaluate(m, Xva, yva)
        if va > best: best, state = va, {k: v.detach().clone() for k, v in m.state_dict().items()}
        print(f'epoch {ep + 1}: validation accuracy {va:.4f}')
    m.load_state_dict(state)                  # early stopping: best validation epoch
    return m

def cnn():
    blk = lambda a, b: [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
    return nn.Sequential(*blk(1, 32), *blk(32, 32), nn.MaxPool2d(2), *blk(32, 64), *blk(64, 64), nn.MaxPool2d(2),
                         *blk(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 10))
nparams = lambda m: sum(p.numel() for p in m.parameters())"""

ES = r"""from PIL import Image
es = torchvision.datasets.EuroSAT(VISION, download=True)            # 27,000 Sentinel-2 tiles, 64 x 64 RGB
root = os.path.join(VISION, 'eurosat', '2750')
DIRS = sorted(os.listdir(root)); DIRS = [d for d in DIRS if os.path.isdir(os.path.join(root, d))]
XE, YE = [], []
for c, d in enumerate(DIRS):
    for f in sorted(os.listdir(os.path.join(root, d))):
        if f.endswith('.jpg'):
            XE.append(np.asarray(Image.open(os.path.join(root, d, f)).convert('RGB'))); YE.append(c)
XE, YE = np.stack(XE), np.array(YE)
rs = np.random.RandomState(SEED); etr, eva, ete = [], [], []
for c in range(10):
    ids = rs.permutation(np.where(YE == c)[0]); n = len(ids)
    etr += list(ids[:int(.6 * n)]); eva += list(ids[int(.6 * n):int(.8 * n)]); ete += list(ids[int(.8 * n):])
etr, eva, ete = map(np.array, (etr, eva, ete))
MUI = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); SDI = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def es_t(ids, size=112):
    x = (torch.tensor(XE[ids]).permute(0, 3, 1, 2).float() / 255 - MUI) / SDI
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
print(DIRS, len(etr), len(eva), len(ete))"""


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 Product photos as data (Fashion-MNIST)", "## §1 Fotografiile de produse ca date (Fashion-MNIST)"))
    nb.code(SETUP)
    nb.code("fig, axes = plt.subplots(2, 10, figsize=(10, 2.4))\nfor c in range(10):\n    ids = np.where(y[itr] == c)[0][:2]\n"
            "    for r in range(2):\n        axes[r, c].imshow(X[itr][ids[r]], cmap='gray_r'); axes[r, c].axis('off')\n"
            "    axes[0, c].set_title(CLASSES[c], fontsize=7)\nplt.show()")
    nb.md(T("## §2 A convolution by hand", "## §2 O convoluție calculată de mînă") + "\n\n" +
          T("Output size $\\lfloor (H+2p-k)/s \\rfloor + 1$; Sobel kernels on a real image.",
            "Dimensiunea ieșirii $\\lfloor (H+2p-k)/s \\rfloor + 1$; nuclee Sobel pe o imagine reală."))
    nb.code("out_size = lambda H, k, p, s: (H + 2 * p - k) // s + 1\nprint(out_size(64, 3, 1, 1), out_size(64, 3, 0, 1), out_size(64, 3, 1, 2))\n"
            "Kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])\nimg = torch.tensor((X[itr][0] - MU) / SD)[None, None]\n"
            "ex = F.conv2d(img, Kx[None, None], padding=1)[0, 0]; ey = F.conv2d(img, Kx.T[None, None], padding=1)[0, 0]\n"
            "fig, ax = plt.subplots(1, 3, figsize=(7, 2.4))\nfor a, im in zip(ax, [img[0, 0], ex, ey]):\n    a.imshow(im, cmap='RdBu_r'); a.axis('off')\nplt.show()")
    nb.md(T("## §3 Softmax regression, MLP and CNN", "## §3 Regresia softmax, MLP și CNN") + "\n\n" +
          T("One seed and 6 epochs here (the slides: 3 seeds, 12 epochs).", "Aici, o singură sămînță și 6 epoci (pe slide-uri: 3 semințe, 12 epoci)."))
    nb.code(TRAIN)
    nb.code("Xtr, Xva, Xs = T(X[itr]), T(X[iva]), T(Xte)\nres = {}\n"
            "models = {'softmax': nn.Sequential(nn.Flatten(), nn.Linear(784, 10)),\n"
            "          'mlp': nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10)),\n"
            "          'cnn': cnn()}\nfor k, m in models.items():\n    m = fit(m, Xtr, y[itr], Xva, y[iva])\n    acc, pred = evaluate(m, Xs, yte)\n"
            "    res[k] = (acc, nparams(m), pred); print(k, 'test accuracy', round(100 * acc, 2), 'parameters', nparams(m))\nCNN = m")
    nb.code("pc, pm = res['cnn'][2] == yte, res['mlp'][2] == yte\nb, c = int((pc & ~pm).sum()), int((~pc & pm).sum())\n"
            "print('McNemar: b =', b, 'c =', c, 'chi2 =', round((abs(b - c) - 1) ** 2 / (b + c), 1))\n"
            "from sklearn.metrics import confusion_matrix\ncm = confusion_matrix(yte, res['cnn'][2])\n"
            "print('recall by class:', dict(zip(CLASSES, np.round(100 * cm.diagonal() / cm.sum(1), 1))))")
    nb.code("W1 = CNN[0].weight.detach().cpu().numpy()[:, 0]\nfig, axes = plt.subplots(2, 16, figsize=(10, 1.5))\n"
            "for i, a in enumerate(axes.flat):\n    a.imshow(W1[i], cmap='RdBu_r'); a.axis('off')\nplt.show()")
    nb.md(T("## §4 Residual connections (a short CIFAR-10 run)", "## §4 Conexiunile reziduale (o rulare scurtă pe CIFAR-10)") + "\n\n" +
          T("A residual block computes $\\mathbf y = \\mathrm{ReLU}(\\mathbf x + F(\\mathbf x))$. The slides train 32-layer networks for 15 epochs; here 2 epochs on 10,000 images show the start of the gap.",
            "Un bloc rezidual calculează $\\mathbf y = \\mathrm{ReLU}(\\mathbf x + F(\\mathbf x))$. Pe slide-uri, rețelele de 32 de straturi sînt antrenate 15 epoci; aici, 2 epoci pe 10.000 de imagini arată începutul diferenței."))
    nb.code("cf = torchvision.datasets.CIFAR10(VISION, train=True, download=True)\nXc = torch.tensor(cf.data[:10000]).permute(0, 3, 1, 2).float() / 255\n"
            "Xc = (Xc - Xc.mean((0, 2, 3), keepdim=True)) / Xc.std((0, 2, 3), keepdim=True); yc = torch.tensor(cf.targets[:10000])\n"
            "class Block(nn.Module):\n    def __init__(s, c, res):\n        super().__init__(); s.res = res\n"
            "        s.f = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))\n"
            "    def forward(s, x):\n        return F.relu(s.f(x) + (x if s.res else 0))\n"
            "def net(res, n=15):\n    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), *[Block(16, res) for _ in range(n)],\n"
            "                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))\n"
            "for res in (False, True):\n    torch.manual_seed(SEED); m = net(res).to(DEVICE); opt = torch.optim.SGD(m.parameters(), 0.05, momentum=0.9)\n"
            "    for ep in range(2):\n        idx = torch.randperm(10000); tot = 0\n        for j in range(0, 10000, 128):\n"
            "            b = idx[j:j + 128]; loss = F.cross_entropy(m(Xc[b].to(DEVICE)), yc[b].to(DEVICE))\n"
            "            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)\n"
            "    print('residual' if res else 'plain', '32-layer network, training loss after 2 epochs:', round(tot / 10000, 3))")
    nb.md(T("## §5 Transfer learning on EuroSAT", "## §5 Transferul învățării pe EuroSAT"))
    nb.code(ES)
    nb.code("from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\n"
            "body = torchvision.models.resnet18(weights='IMAGENET1K_V1'); body.fc = nn.Identity(); body = body.to(DEVICE).eval()\n"
            "with torch.no_grad():\n    FE = np.concatenate([body(es_t(np.arange(j, min(j + 500, len(XE)))).to(DEVICE)).cpu().numpy() for j in range(0, len(XE), 500)])\n"
            "sc = StandardScaler().fit(FE[etr])\nfor n in [10, 100, None]:\n"
            "    ids = etr if n is None else np.concatenate([np.random.RandomState(SEED).choice(etr[YE[etr] == c], n, replace=False) for c in range(10)])\n"
            "    lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(FE[ids]), YE[ids])\n"
            "    print('frozen ResNet-18 + logistic,', 'all' if n is None else n, 'tiles/class: test accuracy', round(100 * lr.score(sc.transform(FE[ete]), YE[ete]), 2))")
    nb.code("torch.manual_seed(SEED)\nft = torchvision.models.resnet18(weights='IMAGENET1K_V1'); ft.fc = nn.Linear(512, 10); ft = ft.to(DEVICE)\n"
            "opt = torch.optim.AdamW([{'params': [p for n, p in ft.named_parameters() if not n.startswith('fc')], 'lr': 3e-4},\n"
            "                         {'params': ft.fc.parameters(), 'lr': 3e-3}], weight_decay=1e-4)\nyt = torch.tensor(YE)\n"
            "for ep in range(2):\n    ft.train(); idx = np.random.permutation(etr)\n    for j in range(0, len(idx), 64):\n"
            "        b = idx[j:j + 64]; xb = es_t(b).to(DEVICE)\n        if np.random.rand() < .5: xb = xb.flip(3)\n"
            "        loss = F.cross_entropy(ft(torch.rot90(xb, np.random.randint(4), (2, 3))), yt[b].to(DEVICE))\n"
            "        opt.zero_grad(); loss.backward(); opt.step()\n"
            "ft.eval()\nwith torch.no_grad():\n    pe = np.concatenate([ft(es_t(ete[j:j + 500]).to(DEVICE)).argmax(1).cpu().numpy() for j in range(0, len(ete), 500)])\n"
            "print('fine-tuned ResNet-18 (2 epochs): test accuracy', round(100 * float((pe == YE[ete]).mean()), 2))")
    nb.md(T("## §6 Grad-CAM", "## §6 Grad-CAM") + "\n\n" +
          T("$M^c = \\mathrm{ReLU}(\\sum_o \\alpha_o^c A_o)$ with $\\alpha_o^c$ the mean gradient of the class score over map $o$.",
            "$M^c = \\mathrm{ReLU}(\\sum_o \\alpha_o^c A_o)$, unde $\\alpha_o^c$ este gradientul mediu al scorului clasei pe harta $o$."))
    nb.code("acts = {}\nh = ft.layer3.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))\nfig, axes = plt.subplots(1, 6, figsize=(10, 2))\n"
            "for a, c in zip(axes, [0, 3, 4, 7, 8, 9]):\n    i = ete[YE[ete] == c][3]; out = ft(es_t(np.array([i])).to(DEVICE)); k = int(out.argmax())\n"
            "    g = torch.autograd.grad(out[0, k], acts['a'])[0]\n    cam = F.relu((g.mean((2, 3), keepdim=True) * acts['a']).sum(1, keepdim=True))\n"
            "    cam = F.interpolate(cam, size=(64, 64), mode='bilinear')[0, 0].detach().cpu().numpy()\n"
            "    a.imshow(XE[i].mean(2), cmap='gray'); a.imshow(cam, cmap='jet', alpha=.45); a.set_title(DIRS[k], fontsize=7); a.axis('off')\nh.remove(); plt.show()")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(SETUP)
    nb.code(TRAIN)
    nb.task(T("Compute by hand the output size and the number of parameters of a convolution with $k=5$, $p=2$, $s=2$, 3 input channels and 16 filters, on a $64\\times64$ image; check with `nn.Conv2d`.",
              "Calculați de mînă dimensiunea ieșirii și numărul de parametri ai unei convoluții cu $k=5$, $p=2$, $s=2$, 3 canale de intrare și 16 filtre, pe o imagine de $64\\times64$; verificați cu `nn.Conv2d`."))
    nb.solution("print((64 + 2 * 2 - 5) // 2 + 1, (5 * 5 * 3 + 1) * 16)\nconv = nn.Conv2d(3, 16, 5, stride=2, padding=2)\n"
                "print(conv(torch.zeros(1, 3, 64, 64)).shape, sum(p.numel() for p in conv.parameters()))")
    nb.task(T("Train the CNN of the lecture with half the filters in every layer (16, 16, 32, 32, 64) for 4 epochs; compare test accuracy and parameters with the full CNN.",
              "Antrenați CNN-ul din curs cu jumătate din filtre în fiecare strat (16, 16, 32, 32, 64), timp de 4 epoci; comparați acuratețea pe test și numărul de parametri cu CNN-ul complet."))
    nb.solution("blk = lambda a, b: [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]\n"
                "half = nn.Sequential(*blk(1, 16), *blk(16, 16), nn.MaxPool2d(2), *blk(16, 32), *blk(32, 32), nn.MaxPool2d(2), *blk(32, 64),\n"
                "                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 10))\n"
                "Xtr, Xva, Xs = T(X[itr]), T(X[iva]), T(Xte)\nres = {}\nfor k, m in {'half': half, 'full': cnn()}.items():\n"
                "    m = fit(m, Xtr, y[itr], Xva, y[iva], epochs=4); acc, pred = evaluate(m, Xs, yte); res[k] = pred\n"
                "    print(k, round(100 * acc, 2), nparams(m))")
    nb.task(T("Run McNemar's test between the two networks of Task 2 on the 10,000 test images. Is the difference significant at 5%?",
              "Aplicați testul McNemar între cele două rețele din sarcina 2, pe cele 10.000 de imagini de test. Este diferența semnificativă la 5%?"))
    nb.solution("from scipy.stats import chi2\na, b = res['full'] == yte, res['half'] == yte\nn1, n2 = int((a & ~b).sum()), int((~a & b).sum())\n"
                "stat = (abs(n1 - n2) - 1) ** 2 / (n1 + n2); print(n1, n2, round(stat, 2), 'p =', round(1 - chi2.cdf(stat, 1), 4))")
    nb.md(T("## EuroSAT", "## EuroSAT"))
    nb.code(ES)
    nb.task(T("Extract frozen ResNet-18 features (512 per tile) and fit a logistic regression with 25 labelled tiles per class. Report the test accuracy and the most confused pair of classes.",
              "Extrageți trăsăturile înghețate ResNet-18 (512 pentru fiecare imagine) și estimați o regresie logistică cu 25 de imagini etichetate pe clasă. Raportați acuratețea pe test și perechea de clase cel mai des confundată."))
    nb.solution("from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import confusion_matrix\n"
                "body = torchvision.models.resnet18(weights='IMAGENET1K_V1'); body.fc = nn.Identity(); body = body.to(DEVICE).eval()\n"
                "with torch.no_grad():\n    FE = np.concatenate([body(es_t(np.arange(j, min(j + 500, len(XE)))).to(DEVICE)).cpu().numpy() for j in range(0, len(XE), 500)])\n"
                "ids = np.concatenate([np.random.RandomState(SEED).choice(etr[YE[etr] == c], 25, replace=False) for c in range(10)])\n"
                "sc = StandardScaler().fit(FE[ids]); lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(FE[ids]), YE[ids])\n"
                "p = lr.predict(sc.transform(FE[ete])); print('accuracy', round(100 * (p == YE[ete]).mean(), 2))\n"
                "cm = confusion_matrix(YE[ete], p); np.fill_diagonal(cm, 0); i, j = np.unravel_index(cm.argmax(), cm.shape); print(DIRS[i], '->', DIRS[j], cm[i, j])")
    nb.task(T("Business question: with the model of Task 4, estimate the share of 'AnnualCrop' + 'PermanentCrop' tiles in the test set and compare it with the true share.",
              "Întrebare de afaceri: cu modelul din sarcina 4, estimați ponderea imaginilor 'AnnualCrop' + 'PermanentCrop' în setul de test și comparați-o cu ponderea reală."))
    nb.solution("crop = [DIRS.index('AnnualCrop'), DIRS.index('PermanentCrop')]\n"
                "print('true %', round(100 * np.isin(YE[ete], crop).mean(), 2), 'predicted %', round(100 * np.isin(p, crop).mean(), 2))")
    nb.task(T("Repeat Task 4 with 10, 50 and 250 tiles per class and plot accuracy against the label budget (log scale). Where would you stop labelling?",
              "Repetați sarcina 4 cu 10, 50 și 250 de imagini pe clasă și reprezentați grafic acuratețea în funcție de bugetul de etichete (scară logaritmică). Unde v-ați opri cu etichetarea?"))
    nb.solution("acc = []\nfor n in [10, 25, 50, 250]:\n    ids = np.concatenate([np.random.RandomState(SEED).choice(etr[YE[etr] == c], n, replace=False) for c in range(10)])\n"
                "    sc = StandardScaler().fit(FE[ids]); lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(FE[ids]), YE[ids])\n"
                "    acc.append(100 * lr.score(sc.transform(FE[ete]), YE[ete]))\nplt.semilogx([10, 25, 50, 250], acc, 'o-'); plt.xlabel('tiles per class'); plt.ylabel('test accuracy (%)'); plt.show(); print(np.round(acc, 2))")


if __name__ == "__main__":
    Notebook.build_all(4, TITLE, lecture, kind="lecture")
    Notebook.build_all(4, TITLE, seminar, kind="seminar")
