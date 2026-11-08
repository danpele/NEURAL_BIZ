"""
make_notebook_L14.py -- Lecture 14 notebooks (EN + RO): lecture notebook (sections = the 'Experiment yourself' pointers of the
deck) and seminar notebook (student + solutions), all executed.
CLIP runs live on a sample of EuroSAT images; the full runs (27,000 images, the CNN, the vision-language model) are done by
analysis_L14.py and read from lectures/L14/results/.
Run:  python3 lectures/L14/make_notebook_L14.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Multimodal and vision-language models; synthesis of the course", "Modele multimodale și vizual-lingvistice; sinteza cursului")
SETUP = ("RES = '../lectures/L14/results/'\nimport torch, torchvision\nfrom PIL import Image\nfrom transformers import CLIPModel, CLIPProcessor\n"
         "root = '../data/raw/vision'\nos.makedirs(root, exist_ok=True)\nES = torchvision.datasets.EuroSAT(root, download=True)\n"
         "Z = pd.read_csv(RES + 'eurosat_zeroshot.csv')\nCLS = ES.classes\nprint(len(ES), 'images;', len(Z), 'test images in the lecture split')")
CLIP = ("clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE).eval()\nproc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')\n"
        "def img_emb(idx):\n    out = []\n    with torch.no_grad():\n        for j in range(0, len(idx), 64):\n"
        "            px = proc(images=[ES[int(i)][0] for i in idx[j:j + 64]], return_tensors='pt')['pixel_values'].to(DEVICE)\n"
        "            e = clip.get_image_features(pixel_values=px)\n            e = e if isinstance(e, torch.Tensor) else e.pooler_output\n"
        "            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())\n    return np.concatenate(out)\n"
        "def txt_emb(texts):\n    with torch.no_grad():\n        t = proc(text=texts, return_tensors='pt', padding=True).to(DEVICE)\n"
        "        e = clip.get_text_features(**t)\n        e = e if isinstance(e, torch.Tensor) else e.pooler_output\n"
        "        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()\n"
        "sample = Z.sample(500, random_state=SEED)\nE = img_emb(sample['index'].values)\ny = sample['true'].map({c: i for i, c in enumerate(CLS)}).values")
NAMES = ("NAMES = {'AnnualCrop': 'annual crop land', 'Forest': 'forest', 'HerbaceousVegetation': 'brushland or shrubland', 'Highway': 'highway or road',\n"
         "         'Industrial': 'industrial buildings or commercial buildings', 'Pasture': 'pasture land', 'PermanentCrop': 'permanent crop land',\n"
         "         'Residential': 'residential buildings or homes or apartments', 'River': 'river', 'SeaLake': 'lake or sea'}")


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 A vision transformer sees patches", "## §1 Un transformer pentru imagini vede petice"))
    nb.code(SETUP)
    nb.code("img = ES[int(Z['index'][0])][0].resize((224, 224), Image.NEAREST)\nfig, ax = plt.subplots(figsize=(3, 3))\nax.imshow(img)\n"
            "for g in range(0, 225, 32):\n    ax.axhline(g - 0.5, color='white', lw=0.5); ax.axvline(g - 0.5, color='white', lw=0.5)\n"
            "ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('7 x 7 = 49 patches of 32 x 32 pixels (ViT-B/32)')\nplt.show()")
    nb.md(T("## §2 CLIP zero-shot on EuroSAT (live on 500 images)", "## §2 CLIP zero-shot pe EuroSAT (live, pe 500 de imagini)"))
    nb.code(CLIP + "\n" + NAMES + "\nfor tpl in ['{}', 'a satellite photo of {}.']:\n"
            "    W = txt_emb([tpl.format(NAMES[c] if '{}.' in tpl else c) for c in CLS])\n"
            "    print(repr(tpl), 'accuracy on 500 images:', round(100 * float(((E @ W.T).argmax(1) == y).mean()), 1), '%')\n"
            "print('full test set, lecture results:', {c: round(100 * float((Z[c] == Z['true']).mean()), 1) for c in Z.columns if c[:3] in ('B32', 'B16')})")
    nb.md(T("§ *Experiment yourself*: write prompts for a class pair that CLIP confuses (pasture and herbaceous vegetation) and check whether the accuracy improves.",
            "§ *Exercițiu practic*: scrieți prompturi pentru o pereche de clase pe care CLIP le confundă (pășune și vegetație ierboasă) și verificați dacă acuratețea crește."))
    nb.md(T("## §3 A linear probe needs few labels", "## §3 Un clasificator liniar are nevoie de puține etichete"))
    nb.code("from sklearn.linear_model import LogisticRegression\ntrain_idx = np.setdiff1d(np.arange(len(ES)), Z['index'].values)\n"
            "ytr_all = np.array(ES.targets)[train_idx]\nrng = np.random.default_rng(SEED)\nk16 = np.concatenate([rng.choice(train_idx[ytr_all == c], 16, replace=False) for c in range(10)])\n"
            "clf = LogisticRegression(max_iter=3000, C=10).fit(img_emb(k16), np.array(ES.targets)[k16])\n"
            "print('ViT-B/32 probe, 16 images per class, accuracy on the 500-image sample:', round(100 * float((clf.predict(E) == y).mean()), 1), '%')\n"
            "Fq = pd.read_csv(RES + 'fewshot_eurosat.csv')\nprint(Fq.groupby(['method', 'k']).acc.mean().mul(100).round(1))")
    nb.md(T("## §4 Market regimes from chart images", "## §4 Regimurile pieței din imaginile graficelor"))
    nb.code("from sklearn.metrics import roc_auc_score\nC = pd.read_csv(RES + 'chart_regime.csv')\nTt = C[C.split == 'test']\n"
            "print('stress share in test:', round(100 * Tt.stress.mean(), 1), '%')\n"
            "print({m: round(roc_auc_score(Tt.stress, Tt[m]), 3) for m in ('numeric', 'clip_probe', 'clip_zs', 'vlm_stress')})")
    nb.md(T("## §5 What a vision-language model reads from tables", "## §5 Ce citește un model vizual-lingvistic din tabele"))
    nb.code("TR = pd.read_csv(RES + 'table_reading.csv')\nTR['read'] = pd.to_numeric(TR.answer.str.extract(r'(-?\\d+(?:\\.\\d+)?)')[0], errors='coerce')\n"
            "print('exact answers:', round(100 * float(np.isclose(TR.read, TR.true_10y, atol=0.001).mean()), 1), '%')\nprint(TR[~np.isclose(TR.read, TR.true_10y, atol=0.001)].head())\n"
            "fig, ax = plt.subplots(figsize=(4.6, 1.3))\nax.imshow(Image.open(RES + 'table_example.png'))\nax.axis('off')\nplt.show()")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(SETUP)
    nb.code(CLIP + "\n" + NAMES)
    nb.task(T("Build a zero-shot classifier with your own template (for example 'an aerial image of {}') and report its accuracy on the 500 sampled images.",
              "Construiți un clasificator zero-shot cu propriul șablon (de exemplu „an aerial image of {}”) și raportați acuratețea pe cele 500 de imagini eșantionate."))
    nb.solution("W = txt_emb(['an aerial image of {}.'.format(NAMES[c]) for c in CLS])\nprint(round(100 * float(((E @ W.T).argmax(1) == y).mean()), 1))")
    nb.task(T("Average the embeddings of three templates per class (prompt ensemble). Does the accuracy improve over the single template?",
              "Faceți media embedding-urilor a trei șabloane pentru fiecare clasă (ansamblu de prompturi). Crește acuratețea față de un singur șablon?"))
    nb.solution("tpls = ['a satellite photo of {}.', 'an aerial photo of {}.', 'a centered satellite photo of {}.']\n"
                "W = np.array([txt_emb([t.format(NAMES[c]) for t in tpls]).mean(0) for c in CLS])\nW = W / np.linalg.norm(W, axis=1, keepdims=True)\n"
                "print(round(100 * float(((E @ W.T).argmax(1) == y).mean()), 1))")
    nb.task(T("Train a logistic regression on CLIP ViT-B/32 embeddings with 4 and with 32 images per class (seed fixed). Plot the two accuracies next to the zero-shot one.",
              "Antrenați o regresie logistică pe embedding-urile CLIP ViT-B/32 cu 4 și cu 32 de imagini pe clasă (valoarea generatorului aleator fixă). Reprezentați grafic cele două acurateți lîngă cea zero-shot."))
    nb.solution("from sklearn.linear_model import LogisticRegression\ntrain_idx = np.setdiff1d(np.arange(len(ES)), Z['index'].values)\nyt = np.array(ES.targets)\n"
                "rng = np.random.default_rng(SEED)\nacc = {}\nfor k in (4, 32):\n    ii = np.concatenate([rng.choice(train_idx[yt[train_idx] == c], k, replace=False) for c in range(10)])\n"
                "    acc[k] = 100 * float((LogisticRegression(max_iter=3000, C=10).fit(img_emb(ii), yt[ii]).predict(E) == y).mean())\n"
                "W = txt_emb(['a satellite photo of {}.'.format(NAMES[c]) for c in CLS])\nacc['zero-shot'] = 100 * float(((E @ W.T).argmax(1) == y).mean())\n"
                "plt.bar([str(k) for k in acc], list(acc.values()))\nplt.ylabel('accuracy (%)')\nplt.show()\nprint(acc)")
    nb.task(T("From `eurosat_zeroshot.csv`, list the three most frequent confusions of the best zero-shot model.",
              "Din `eurosat_zeroshot.csv`, listați cele mai frecvente trei confuzii ale celui mai bun model zero-shot."))
    nb.solution("best = max([c for c in Z.columns if c[:3] in ('B32', 'B16')], key=lambda c: (Z[c] == Z['true']).mean())\n"
                "w = Z[Z[best] != Z['true']]\nprint(best)\nprint(w.groupby(['true', best]).size().sort_values(ascending=False).head(3))")
    nb.task(T("For the chart-regime task, compute a 95 % bootstrap interval of the difference in AUC between the numeric model and the CLIP probe on the test windows.",
              "Pentru sarcina regimurilor de piață, calculați un interval bootstrap de 95 % pentru diferența de AUC dintre modelul pe cifre și clasificatorul pe CLIP, pe ferestrele de test."))
    nb.solution("from sklearn.metrics import roc_auc_score\nC = pd.read_csv(RES + 'chart_regime.csv')\nTt = C[C.split == 'test'].reset_index(drop=True)\n"
                "rng = np.random.default_rng(SEED)\nd = []\nfor _ in range(1000):\n    i = rng.integers(0, len(Tt), len(Tt))\n    s = Tt.loc[i]\n"
                "    if s.stress.nunique() == 2:\n        d.append(roc_auc_score(s.stress, s.numeric) - roc_auc_score(s.stress, s.clip_probe))\n"
                "print(np.round(np.percentile(d, [2.5, 97.5]), 3), '(the windows overlap, so this interval is too narrow; a block bootstrap is better)')")


if __name__ == "__main__":
    Notebook.build_all(14, TITLE, lecture, kind="lecture")
    Notebook.build_all(14, TITLE, seminar, kind="seminar")
