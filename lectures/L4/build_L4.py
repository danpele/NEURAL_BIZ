r"""
build_L4.py -- Lecture 4 "Convolutional neural networks (CNN) and transfer learning", English and Romanian.
Slide text only; every result is a placeholder @{key} filled from L4_numbers.json (analysis_L4.py).
Run:  python3 lectures/L4/build_L4.py   ("en" or "ro" builds one language; exit 0 only if every check passes)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, block, cols, hier, items, reference_frames,  # noqa: E402
                      run_build, table)

ABBR = {
    "AI": ("artificial intelligence", "inteligență artificială"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "BN": ("batch normalisation", "normalizare pe loturi (batch normalisation)"),
    "CAM": ("class activation map", "harta de activare a clasei (class activation map)"),
    "CIFAR": ("Canadian Institute for Advanced Research (CIFAR-10 image set)", "Canadian Institute for Advanced Research (setul de imagini CIFAR-10)"),
    "CNN": ("convolutional neural network", "rețea neuronală convoluțională"),
    "CPU": ("central processing unit", "procesor central"),
    "GAP": ("global average pooling", "agregare prin medie globală (global average pooling)"),
    "GPU": ("graphics processing unit", "procesor grafic"),
    "ICBE": ("International Conference on Business Excellence", "International Conference on Business Excellence"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "MNIST": ("Modified National Institute of Standards and Technology (digit images)", "Modified National Institute of Standards and Technology (imagini cu cifre)"),
    "MPS": ("Metal Performance Shaders (Apple GPU back end of PyTorch)", "Metal Performance Shaders (procesorul grafic Apple, în PyTorch)"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată"),
    "RGB": ("red, green, blue (colour channels)", "roșu, verde, albastru (canalele de culoare)"),
    "SGD": ("stochastic gradient descent", "coborîre stohastică pe gradient"),
    "VGG": ("Visual Geometry Group, University of Oxford", "Visual Geometry Group, Universitatea Oxford"),
    "VLM": ("vision--language model", "model viziune--limbaj"),
    "GPT": ("generative pre-trained transformer", "transformer generativ preantrenat"),
    "EU": ("European Union", None),
    "UE": (None, "Uniunea Europeană"),
}
IGNORE = {"AdamW", "BF", "BIZ", "GP", "HW", "HWC", "IMAGENET1K", "MIT", "NEURAL", "SOURCES", "EuroSAT", "PICBE", "JSTARS", "IEEE", "CVPR", "NIPS", "ICLR", "ECCV", "ICCV", "OK", "II", "SSRN", "JMLR", "PMLR", "ICML", "NeurIPS"}
UNITS = set()

REFS = [
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}, ch.~10. Springer. \\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Goodfellow, I., Bengio, Y.\\ \\& Courville, A.\\ (2016). \\textit{Deep Learning}, ch.~9. MIT Press. \\href{https://www.deeplearningbook.org}{deeplearningbook.org}",
    "He, K., Zhang, X., Ren, S.\\ \\& Sun, J.\\ (2016). Deep residual learning for image recognition. \\textit{CVPR}, 770--778. \\href{https://doi.org/10.1109/CVPR.2016.90}{doi:10.1109/CVPR.2016.90}",
    "Helber, P., Bischke, B., Dengel, A.\\ \\& Borth, D.\\ (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. \\textit{IEEE JSTARS}, 12(7), 2217--2226. \\href{https://doi.org/10.1109/JSTARS.2019.2918242}{doi:10.1109/JSTARS.2019.2918242}",
    "Ioffe, S.\\ \\& Szegedy, C.\\ (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. \\textit{ICML}, PMLR 37, 448--456. \\href{https://arxiv.org/abs/1502.03167}{arXiv:1502.03167}",
    "Jheng, S.-L., Pele, D.\\,T., Tak, R.\\ \\& Găman, Ș.\\ (2026). A multimodal vision-language framework for financial anomaly detection. \\textit{Proceedings of the International Conference on Business Excellence}, 20(1), 191--201. \\href{https://doi.org/10.2478/picbe-2026-0020}{doi:10.2478/picbe-2026-0020}",
    "Găman, Ș., Winkel, J., Zuo, X., Härdle, W.\\,K.\\ \\& Pele, D.\\,T.\\ (2026). Quantlet: the code snippet knowledge platform. \\textit{Computational Statistics}, 41(3), 66. \\href{https://doi.org/10.1007/s00180-026-01742-6}{doi:10.1007/s00180-026-01742-6}",
    "Krizhevsky, A.\\ (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto. \\href{https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf}{cs.toronto.edu}",
    "Krizhevsky, A., Sutskever, I.\\ \\& Hinton, G.\\,E.\\ (2012). ImageNet classification with deep convolutional neural networks. \\textit{NIPS}, 25, 1097--1105. \\href{https://doi.org/10.1145/3065386}{doi:10.1145/3065386}",
    "LeCun, Y., Bottou, L., Bengio, Y.\\ \\& Haffner, P.\\ (1998). Gradient-based learning applied to document recognition. \\textit{Proceedings of the IEEE}, 86(11), 2278--2324. \\href{https://doi.org/10.1109/5.726791}{doi:10.1109/5.726791}",
    "McNemar, Q.\\ (1947). Note on the sampling error of the difference between correlated proportions or percentages. \\textit{Psychometrika}, 12(2), 153--157. \\href{https://doi.org/10.1007/BF02295996}{doi:10.1007/BF02295996}",
    "Selvaraju, R.\\,R., Cogswell, M., Das, A., Vedantam, R., Parikh, D.\\ \\& Batra, D.\\ (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. \\textit{ICCV}, 618--626. \\href{https://doi.org/10.1109/ICCV.2017.74}{doi:10.1109/ICCV.2017.74}",
    "Simonyan, K.\\ \\& Zisserman, A.\\ (2015). Very deep convolutional networks for large-scale image recognition. \\textit{ICLR}. \\href{https://arxiv.org/abs/1409.1556}{arXiv:1409.1556}",
    "Xiao, H., Rasul, K.\\ \\& Vollgraf, R.\\ (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. \\href{https://arxiv.org/abs/1708.07747}{arXiv:1708.07747}",
    "Yosinski, J., Clune, J., Bengio, Y.\\ \\& Lipson, H.\\ (2014). How transferable are features in deep neural networks? \\textit{NIPS}, 27, 3320--3328. \\href{https://arxiv.org/abs/1411.1792}{arXiv:1411.1792}",
    "Zeiler, M.\\,D.\\ \\& Fergus, R.\\ (2014). Visualizing and understanding convolutional networks. \\textit{ECCV}, 818--833. \\href{https://doi.org/10.1007/978-3-319-10590-1_53}{doi:10.1007/978-3-319-10590-1\\_53}",
]


def code(src):
    """Code listing without verbatim (frames are not fragile): escaped \\texttt lines, spaces kept."""
    rep = {"\\": "\\textbackslash{}", "_": "\\_", "#": "\\#", "{": "\\{", "}": "\\}", "%": "\\%", "&": "\\&",
           "^": "\\^{}", "~": "\\~{}", "$": "\\$"}
    out = []
    for line in src.strip("\n").split("\n"):
        esc = "".join(rep.get(c, c) for c in line)
        esc = re.sub(r"^( +)", lambda m: "\\hspace*{" + str(0.5 * len(m.group(1))) + "em}", esc)
        out.append(esc.replace("  ", "\\ \\ ") or "\\mbox{}")
    return "{\\scriptsize\\ttfamily\\setlength{\\parindent}{0pt}" + "\\\\\n".join(out) + "\\par}"


def it(*xs, sep="2pt"):
    def one(x):
        if isinstance(x, tuple):
            return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                    "\n".join("      \\item " + (s[0].upper() + s[1:] if s[0].islower() else s) for s in x[1:] if s) + "\n    \\end{itemize}")
        return hier(x)
    return (f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{{sep}}}\n" + "\n".join("  \\item " + one(x) for x in xs if x) +
            "\n\\end{itemize}")


def note(*xs):
    return "\n\\par\\vspace{0.5mm}{\\scriptsize " + items(*xs, opt="\\setlength{\\itemsep}{0pt}", nest=False) + "}"


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L4_numbers.json"), lang)
    D = Deck(4, "Convolutional neural networks (CNN) and transfer learning" if lang == "en" else
             "Rețele neuronale convoluționale (CNN) și transferul învățării", V, lang, heights=H)
    t = D.t
    ro = lang == "ro"
    OWN = t("(own computation)", "(calcul propriu)")
    NB = "\\S"

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Write a convolutional layer as a formula: kernel, padding, stride, channels, parameters and receptive field.",
          "Să scrieți un strat convoluțional ca formulă: nucleu, completarea marginilor, pas, canale, număr de parametri și cîmp receptiv."),
        t("Explain why a convolutional neural network (CNN) needs far fewer parameters than a multilayer perceptron (MLP) on images, and verify it on real data.",
          "Să explicați de ce o rețea neuronală convoluțională (CNN) are nevoie de mult mai puțini parametri decît un perceptron multistrat (MLP) pe imagini și să verificați acest lucru pe date reale."),
        t("Read the classic architectures, from LeNet to ResNet, and the residual connection that makes deep networks trainable.",
          "Să citiți arhitecturile clasice, de la LeNet la ResNet, și conexiunea reziduală care face antrenabile rețelele adînci."),
        t("Use transfer learning (feature extraction and fine-tuning) when labelled images are expensive, and measure what it saves.",
          "Să folosiți transferul învățării (extragerea trăsăturilor și ajustarea fină) atunci cînd imaginile etichetate sînt scumpe și să măsurați cît economisește."),
        t("Explain a prediction with a gradient-weighted class activation map (CAM) and turn image predictions into business numbers.",
          "Să explicați o predicție cu o hartă de activare a clasei (CAM) ponderată cu gradienți și să transformați predicțiile pe imagini în cifre utile în afaceri."),
        sep="4pt"))
    abbreviation_frames(D, ABBR)

    D.frame(t("Real data only: three image collections", "Numai date reale: trei colecții de imagini"),
            note(t("The Modified National Institute of Standards and Technology (MNIST) digits are replaced in Fashion-MNIST by clothing photos; the Canadian Institute for Advanced Research (CIFAR) released CIFAR-10.",
                   "Cifrele Modified National Institute of Standards and Technology (MNIST) sînt înlocuite în Fashion-MNIST cu fotografii de îmbrăcăminte; Canadian Institute for Advanced Research (CIFAR) a publicat CIFAR-10.")) +
            table("@{}p{0.2\\textwidth}p{0.3\\textwidth}p{0.2\\textwidth}p{0.22\\textwidth}@{}",
                  [t("Data set", "Setul de date"), t("Content", "Conținut"), t("Split (train/valid/test)", "Împărțire (antr./valid./test)"), t("Business use", "Utilizare în afaceri")],
                  [["Fashion-MNIST", t("Zalando catalogue photos, 10 product types, 28$\\times$28 grey", "fotografii din catalogul Zalando, 10 tipuri de produse, 28$\\times$28, gri"),
                    "@{fmnist.n_train|,d} / @{fmnist.n_valid|,d} / @{fmnist.n_test|,d}", t("automatic product tagging", "etichetarea automată a produselor")],
                   ["CIFAR-10", t("photographs, 10 object classes, 32$\\times$32 colour", "fotografii, 10 clase de obiecte, 32$\\times$32, color"),
                    "@{cifar.n_train|,d} / @{cifar.n_valid|,d} / @{cifar.n_test|,d}", t("benchmark for architectures", "banc de probă pentru arhitecturi")],
                   ["EuroSAT", t("Sentinel-2 satellite tiles of Europe, 10 land-use classes, 64$\\times$64 colour", "imagini satelitare Sentinel-2 din Europa, 10 clase de utilizare a terenului, 64$\\times$64, color"),
                    "@{eurosat.n_train|,d} / @{eurosat.n_valid|,d} / @{eurosat.n_test|,d}", t("agriculture, insurance, real estate", "agricultură, asigurări, imobiliare")]]) +
            note(t("Sources: Zalando Research (Xiao et al., 2017); Krizhevsky (2009); Helber et al. (2019), Copernicus Sentinel-2 data; all downloaded by the course scripts (data/SOURCES.md).",
                   "Surse: Zalando Research (Xiao et al., 2017); Krizhevsky (2009); Helber et al. (2019), date Copernicus Sentinel-2; toate sînt descărcate de scripturile cursului (data/SOURCES.md)."),
                 t("Validation parts are used for early stopping and tuning; every test set is used once, at the end.",
                   "Seturile de validare servesc la oprirea timpurie și la reglaj; fiecare set de test este folosit o singură dată, la final."),
                 t("Pre-trained weights: ResNet-18 trained on ImageNet (torchvision); training on an Apple graphics processing unit (GPU) through Metal Performance Shaders (MPS), seed @{seed}.",
                   "Ponderi preantrenate: ResNet-18 antrenat pe ImageNet (torchvision); antrenare pe procesorul grafic (GPU) Apple prin Metal Performance Shaders (MPS), sămînța @{seed}.")),
            "footnotesize", cite=t("(data/SOURCES.md)", "(data/SOURCES.md)"))

    # ------------------------------------------------------------------ 1. Why convolutions
    D.section(t("Images as data: why convolutions", "Imaginile ca date: de ce convoluții"))
    D.figframe(t("A product photo is a matrix of numbers", "O fotografie de produs este o matrice de numere"),
               t("each image is a $28\\times28$ array of grey levels; a model must map it to one of 10 product types.",
                 "fiecare imagine este o matrice de $28\\times28$ niveluri de gri; modelul trebuie să o asocieze unuia dintre cele 10 tipuri de produse."),
               "\\[ \\mathbf{X}\\in[0,1]^{28\\times 28},\\qquad \\hat{\\mathbf p}=f_{\\btheta}(\\mathbf X)\\in\\Delta^{9},\\qquad \\hat y=\\argmax_k \\hat p_k \\]",
               "fmnist_samples",
               bullets=[t("Fashion-MNIST: @{fmnist.n_train|,d} training, @{fmnist.n_valid|,d} validation and @{fmnist.n_test|,d} test images from the Zalando catalogue.",
                          "Fashion-MNIST: @{fmnist.n_train|,d} de imagini de antrenare, @{fmnist.n_valid|,d} de validare și @{fmnist.n_test|,d} de test, din catalogul Zalando.")],
               job=t("an online retailer tags thousands of new product photos a day; a classifier sets the category, a person checks the doubtful ones.",
                     "un magazin online etichetează zilnic mii de fotografii noi de produse; clasificatorul stabilește categoria, iar un operator verifică doar cazurile îndoielnice."),
               height="0.36\\textheight", cite=t("(Xiao et al., 2017)", "(Xiao et al., 2017)"))

    D.concept(t("A dense layer on pixels wastes parameters and ignores space", "Un strat dens pe pixeli irosește parametri și ignoră spațiul"),
              t("an MLP treats pixel (1,1) and pixel (1,2) as unrelated inputs and needs one weight per pixel and unit.",
                "un MLP tratează pixelii (1,1) și (1,2) ca intrări fără legătură și are nevoie de cîte o pondere pentru fiecare pixel și fiecare unitate."),
              "\\[ \\mathbf h=\\sigma(\\mW\\,\\mathrm{vec}(\\mathbf X)+\\vb),\\quad \\mW\\in\\R^{m\\times HWC} \\]"
              "\\where{$H,W$ height and width; $C$ channels; $m$ hidden units; a $224\\times224$ colour image and $m=1000$ give $224\\cdot224\\cdot3\\cdot1000\\approx1.5\\cdot10^{8}$ weights in the first layer alone}"
              if not ro else
              "\\[ \\mathbf h=\\sigma(\\mW\\,\\mathrm{vec}(\\mathbf X)+\\vb),\\quad \\mW\\in\\R^{m\\times HWC} \\]"
              "\\where{$H,W$ înălțimea și lățimea; $C$ numărul de canale; $m$ unitățile ascunse; o imagine color de $224\\times224$ și $m=1000$ dau $224\\cdot224\\cdot3\\cdot1000\\approx1{,}5\\cdot10^{8}$ ponderi numai în primul strat}",
              it(t(("\\textbf{Three ideas of a CNN}", "local connectivity: a unit sees a small patch", "weight sharing: the same filter at every position",
                    "equivariance: shifting the input shifts the feature map, $f(T_{\\delta}\\mathbf X)=T_{\\delta}f(\\mathbf X)$"),
                   ("\\textbf{Cele trei idei ale unei CNN}", "conectivitate locală: o unitate vede doar o mică zonă", "ponderi partajate: același filtru în toate pozițiile",
                    "echivarianță: o deplasare a intrării deplasează harta de trăsături, $f(T_{\\delta}\\mathbf X)=T_{\\delta}f(\\mathbf X)$")),
                 t("On our data: MLP @{fmnist.mlp.params|,d} parameters, CNN @{fmnist.cnn.params|,d}, @{fmnist.params_ratio|.2f} times fewer.",
                   "Pe datele noastre: MLP cu @{fmnist.mlp.params|,d} de parametri, CNN cu @{fmnist.cnn.params|,d}, de @{fmnist.params_ratio|.2f} ori mai puțini.")),
              t("fewer parameters mean less data needed for the same accuracy, which is what labelled business images cost.",
                "mai puțini parametri înseamnă mai puține date necesare pentru aceeași acuratețe, iar imaginile etichetate costă bani."),
              cite=t("(LeCun et al., 1998; Bishop \\& Bishop, 2024, ch.~10)", "(LeCun et al., 1998; Bishop \\& Bishop, 2024, cap.~10)"))

    # ------------------------------------------------------------------ 2. The convolutional layer
    D.section(t("The convolutional layer, formula by formula", "Stratul convoluțional, formulă cu formulă"))
    D.concept(t("Convolution: the same small filter slides over the image", "Convoluția: același filtru mic glisează peste imagine"),
              t("each output value is a weighted sum of a $k\\times k$ neighbourhood, with weights shared across positions.",
                "fiecare ieșire este o sumă ponderată a unei vecinătăți de $k\\times k$, cu aceleași ponderi peste tot."),
              "\\[ Z_{o}(i,j)=b_o+\\sum_{c=1}^{C_{\\mathrm{in}}}\\sum_{u=0}^{k-1}\\sum_{v=0}^{k-1}K_{o,c}(u,v)\\,X_c(i s+u-p,\\;j s+v-p) \\]"
              + t("\\where{$X_c$ input channel $c$; $K_{o,c}$ the $k\\times k$ kernel from channel $c$ to output map $o$; $b_o$ bias; $s$ stride; $p$ zero padding}",
                  "\\where{$X_c$ canalul de intrare $c$; $K_{o,c}$ nucleul de $k\\times k$ de la canalul $c$ la harta de ieșire $o$; $b_o$ termenul liber; $s$ pasul (stride); $p$ completarea cu zerouri (padding)}"),
              it(t("Feature map of filter $o$: $A_o=\\ReLU(Z_o)$, with the rectified linear unit (ReLU) $\\max(0,z)$; $K$ is learned by backpropagation.",
                   "Harta de trăsături a filtrului $o$: $A_o=\\ReLU(Z_o)$, cu unitatea liniară rectificată (ReLU) $\\max(0,z)$; $K$ este învățat prin propagarea înapoi a erorii.")),
              t("a filter is a pattern detector that is tested at every position of the image at the cost of $k^2C_{\\mathrm{in}}$ weights.",
                "un filtru este un detector de tipare testat în fiecare poziție a imaginii, cu un cost de doar $k^2C_{\\mathrm{in}}$ ponderi."),
              cite=t("(Goodfellow et al., 2016, ch.~9)", "(Goodfellow et al., 2016, cap.~9)"), wl="0.64", wr="0.33")

    D.figframe(t("Hand-made filters already find roads and field edges", "Filtrele construite de mînă găsesc deja drumurile și marginile parcelelor"),
               t("a Sobel kernel responds where the brightness changes along one direction; a mean kernel blurs.",
                 "un nucleu Sobel răspunde acolo unde luminozitatea se schimbă pe o direcție; un nucleu de medie estompează imaginea."),
               "\\[ K_x=\\begin{psmallmatrix}-1&0&1\\\\-2&0&2\\\\-1&0&1\\end{psmallmatrix},\\quad K_y=K_x^{\\top},\\quad K_{\\mathrm{mean}}=\\tfrac19\\mathbf 1_{3\\times3} \\]",
               "conv_edges",
               bullets=[t("Padding $p=1$ keeps the $@{conv.in}\\times@{conv.in}$ size; a CNN learns dozens of such kernels per layer instead of fixing them.",
                          "Completarea $p=1$ păstrează dimensiunea de $@{conv.in}\\times@{conv.in}$; o CNN învață zeci de astfel de nuclee în fiecare strat, în loc să le fixeze.")],
               height="0.36\\textheight", cite=OWN)

    D.concept(t("Output size, padding and stride", "Dimensiunea ieșirii, completarea și pasul"),
              t("three integers decide how large each feature map is.", "trei numere întregi decid cît de mare este fiecare hartă de trăsături."),
              "\\[ H_{\\mathrm{out}}=\\Big\\lfloor\\frac{H_{\\mathrm{in}}+2p-k}{s}\\Big\\rfloor+1 \\]"
              + t("\\where{$H_{\\mathrm{in}}$ input height (same formula for width); $k$ kernel size; $p$ padding; $s$ stride}",
                  "\\where{$H_{\\mathrm{in}}$ înălțimea intrării (aceeași formulă pentru lățime); $k$ dimensiunea nucleului; $p$ completarea; $s$ pasul}"),
              it(t(("\\textbf{Worked examples}, $H_{\\mathrm{in}}=@{conv.in}$, $k=3$",
                    "$p=1,s=1$: $H_{\\mathrm{out}}=@{conv.out_same}$ (``same'')", "$p=0,s=1$: $H_{\\mathrm{out}}=@{conv.out_valid}$ (``valid'')",
                    "$p=1,s=2$: $H_{\\mathrm{out}}=@{conv.out_s2}$ (downsampling)"),
                   ("\\textbf{Exemple}, $H_{\\mathrm{in}}=@{conv.in}$, $k=3$",
                    "$p=1,s=1$: $H_{\\mathrm{out}}=@{conv.out_same}$ (``same'')", "$p=0,s=1$: $H_{\\mathrm{out}}=@{conv.out_valid}$ (``valid'')",
                    "$p=1,s=2$: $H_{\\mathrm{out}}=@{conv.out_s2}$ (subeșantionare)")),
                 t("Parameters of the layer: $(k^2C_{\\mathrm{in}}+1)\\,C_{\\mathrm{out}}$, independent of the image size.",
                   "Parametrii stratului: $(k^2C_{\\mathrm{in}}+1)\\,C_{\\mathrm{out}}$, independenți de dimensiunea imaginii.")),
              t("the same network runs on any image size; only the cost in operations, $H_{\\mathrm{out}}W_{\\mathrm{out}}k^2C_{\\mathrm{in}}C_{\\mathrm{out}}$, grows.",
                "aceeași rețea rulează pe imagini de orice dimensiune; crește doar costul în operații, $H_{\\mathrm{out}}W_{\\mathrm{out}}k^2C_{\\mathrm{in}}C_{\\mathrm{out}}$."),
              cite=OWN)

    D.concept(t("Pooling summarises a neighbourhood; global pooling summarises the image", "Agregarea rezumă o vecinătate; agregarea globală rezumă imaginea"),
              t("max pooling keeps the strongest response in each window and makes the map smaller and slightly shift-invariant.",
                "agregarea prin maxim păstrează cel mai puternic răspuns din fiecare fereastră, micșorează harta și o face ușor invariantă la deplasări."),
              "\\[ P_o(i,j)=\\max_{0\\le u,v<2}A_o(2i+u,2j+v) \\]\\vspace{-2mm}\\[ g_o=\\frac{1}{HW}\\sum_{i,j}A_o(i,j) \\]"
              + t("\\where{$P_o$ max pooling $2\\times2$, stride 2 (halves height and width); $g_o$ global average pooling (GAP), one number per channel}",
                  "\\where{$P_o$ agregarea prin maxim $2\\times2$ cu pasul 2 (înjumătățește înălțimea și lățimea); $g_o$ agregarea prin medie globală (GAP), cîte un număr pentru fiecare canal}"),
              it(t("Our Fashion-MNIST CNN: maps $28\\times28\\to14\\times14\\to7\\times7$, then GAP gives 128 numbers and one linear layer gives the 10 class scores.",
                   "CNN-ul nostru pentru Fashion-MNIST: hărți $28\\times28\\to14\\times14\\to7\\times7$, apoi GAP dă 128 de numere, iar un strat liniar dă cele 10 scoruri ale claselor."),
                 t("Gradient of max pooling: it flows only to the arg-max position of each window.",
                   "Gradientul agregării prin maxim: trece doar prin poziția maximului din fiecare fereastră."),
                 t("GAP replaces large dense layers (thousands of weights) by an average (zero weights).",
                   "GAP înlocuiește straturile dense mari (mii de ponderi) cu o medie (zero ponderi).")),
              t("GAP is the reason a modern CNN has few parameters in its head and can be explained with class activation maps.",
                "GAP este motivul pentru care o CNN modernă are puțini parametri în partea finală și poate fi explicată cu hărți de activare ale claselor."),
              cite=t("(Bishop \\& Bishop, 2024, sec.~10.2)", "(Bishop \\& Bishop, 2024, secț.~10.2)"))

    D.figframe(t("Depth and pooling widen what a unit can see", "Adîncimea și agregarea lărgesc ce poate vedea o unitate"),
               t("the receptive field is the patch of the input that can influence one unit.",
                 "cîmpul receptiv este zona din intrare care poate influența o unitate."),
               "\\[ r_l=r_{l-1}+(k_l-1)\\prod_{i<l}s_i,\\qquad r_0=1 \\]",
               "receptive_field",
               bullets=[t("Twelve $3\\times3$ layers at stride 1 see only $@{rf.plain_12}\\times@{rf.plain_12}$ pixels; eight with $2\\times2$ pooling after every pair see $@{rf.pooled_8conv}\\times@{rf.pooled_8conv}$.",
                          "Douăsprezece straturi $3\\times3$ cu pasul 1 văd doar $@{rf.plain_12}\\times@{rf.plain_12}$ pixeli; opt straturi cu agregare $2\\times2$ după fiecare pereche văd $@{rf.pooled_8conv}\\times@{rf.pooled_8conv}$.")],
               height="0.36\\textheight", cite=OWN)

    D.concept(t("Backpropagation through a convolution is again a convolution", "Propagarea înapoi printr-o convoluție este tot o convoluție"),
              t("because the kernel is shared, its gradient sums the contributions of every position.",
                "pentru că nucleul este partajat, gradientul lui adună contribuțiile tuturor pozițiilor."),
              "\\[ \\frac{\\partial\\loss}{\\partial K_{o,c}(u,v)}=\\sum_{i,j}\\delta_o(i,j)\\,X_c(is+u-p,\\,js+v-p) \\]\\vspace{-2mm}\\[ \\delta_o(i,j)=\\frac{\\partial\\loss}{\\partial Z_o(i,j)} \\]"
              + t("\\where{$\\delta_o$ the error map of output $o$, computed by backpropagation from the layer above; for $s=1$, $\\partial\\loss/\\partial X_c=\\sum_o \\delta_o \\star \\mathrm{rot}_{180}(K_{o,c})$, a full convolution}",
                  "\\where{$\\delta_o$ harta erorilor pentru ieșirea $o$, calculată prin propagarea înapoi din stratul superior; pentru $s=1$, $\\partial\\loss/\\partial X_c=\\sum_o \\delta_o \\star \\mathrm{rot}_{180}(K_{o,c})$, o convoluție completă}"),
              it(t("Weight sharing turns millions of position-specific gradients into one sum per weight.",
                   "Partajarea ponderilor transformă milioane de gradienți specifici fiecărei poziții într-o singură sumă pentru fiecare pondere."),
                 t("In code, autograd does it: \\texttt{loss.backward()} on a \\texttt{nn.Conv2d}.",
                   "În cod, diferențierea automată face totul: \\texttt{loss.backward()} pe un \\texttt{nn.Conv2d}."),
                 t("Derivation in the appendix (convolution as a matrix product).", "Derivarea completă este în anexă (convoluția ca produs de matrice).")),
              t("one forward and one backward pass cost about the same number of operations, so training time scales with the forward cost.",
                "o trecere înainte și una înapoi costă aproximativ același număr de operații, deci timpul de antrenare crește odată cu costul trecerii înainte."),
              cite=t("(Goodfellow et al., 2016, sec.~9.5)", "(Goodfellow et al., 2016, secț.~9.5)"))

    D.concept(t("Batch normalisation keeps every channel on the same scale", "Normalizarea pe loturi menține fiecare canal pe aceeași scală"),
              t("each channel is standardised over the batch and the positions, then rescaled by two learned numbers.",
                "fiecare canal este standardizat pe lot și pe poziții, apoi rescalat cu doi parametri învățați."),
              "\\[ \\hat a=\\frac{a-\\mu_{\\mathcal B,o}}{\\sqrt{\\sigma^2_{\\mathcal B,o}+\\epsilon}},\\qquad \\mathrm{BN}(a)=\\gamma_o\\hat a+\\beta_o \\]"
              + t("\\where{$\\mu_{\\mathcal B,o},\\sigma^2_{\\mathcal B,o}$ mean and variance of channel $o$ over the batch $\\mathcal B$ and all positions; $\\gamma_o,\\beta_o$ learned; at test time running averages replace the batch statistics}",
                  "\\where{$\\mu_{\\mathcal B,o},\\sigma^2_{\\mathcal B,o}$ media și varianța canalului $o$ pe lotul $\\mathcal B$ și pe toate pozițiile; $\\gamma_o,\\beta_o$ învățați; la testare, mediile mobile înlocuiesc statisticile lotului}"),
              it(t("Every network of this lecture uses batch normalisation (BN) after each convolution: Conv $\\to$ BN $\\to$ ReLU.",
                   "Toate rețelele din acest curs folosesc normalizarea pe loturi (BN) după fiecare convoluție: Conv $\\to$ BN $\\to$ ReLU."),
                 t("It allows larger learning rates; our CIFAR runs use a one-cycle schedule up to 0.1 (chosen a priori).",
                   "Permite rate de învățare mai mari; antrenările noastre pe CIFAR folosesc o programare one-cycle pînă la 0,1 (aleasă a priori)."),
                 t("Remember \\texttt{model.eval()} before prediction: otherwise BN uses the test batch.",
                   "Nu uitați \\texttt{model.eval()} înainte de predicție: altfel BN folosește statisticile lotului de test.")),
              t("a model that scores well in the notebook but badly in production often forgot eval mode or the input scaling of training.",
                "un model care merge bine în notebook, dar prost în producție, a uitat adesea modul eval sau scalarea intrărilor din antrenare."),
              cite=t("(Ioffe \\& Szegedy, 2015)", "(Ioffe \\& Szegedy, 2015)"))

    # ------------------------------------------------------------------ 3. Fashion-MNIST case study
    D.section(t("Case study: product photos, from softmax to CNN", "Studiu de caz: fotografii de produse, de la softmax la CNN"))
    D.frame(t("Experimental protocol for the three models", "Protocolul experimental pentru cele trei modele"), cols(it(
        t(("\\textbf{Benchmark}: softmax (multinomial logistic) regression", "$\\hat{\\mathbf p}=\\softmax(\\mW\\mathrm{vec}(\\mathbf X)+\\vb)$, @{fmnist.softmax.params|,d} parameters"),
          ("\\textbf{Reper}: regresia softmax (logistică multinomială)", "$\\hat{\\mathbf p}=\\softmax(\\mW\\mathrm{vec}(\\mathbf X)+\\vb)$, @{fmnist.softmax.params|,d} de parametri")),
        t(("\\textbf{MLP}: 784--256--128--10, ReLU, dropout 0.2", "@{fmnist.mlp.params|,d} parameters"),
          ("\\textbf{MLP}: 784--256--128--10, ReLU, dropout 0,2", "@{fmnist.mlp.params|,d} de parametri")),
        t(("\\textbf{CNN}: 2 blocks [Conv3$\\times$3--BN--ReLU]$\\times$2 + max pool, Conv 128, GAP, linear", "@{fmnist.cnn.params|,d} parameters"),
          ("\\textbf{CNN}: 2 blocuri [Conv3$\\times$3--BN--ReLU]$\\times$2 + agregare maxim, Conv 128, GAP, liniar", "@{fmnist.cnn.params|,d} de parametri")), sep="3pt"),
        it(t(("\\textbf{Same training for all}", "AdamW, batch @{fmnist.batch}, @{fmnist.epochs} epochs, one-cycle learning rate",
              "cross-entropy $\\loss=-\\frac1n\\sum_i\\log\\hat p_{i,y_i}$", "pixels standardised with the training mean @{fmnist.mu|.4f} and standard deviation @{fmnist.sd|.4f}"),
             ("\\textbf{Aceeași antrenare pentru toate}", "AdamW, loturi de @{fmnist.batch}, @{fmnist.epochs} epoci, rată de învățare one-cycle",
              "entropia încrucișată $\\loss=-\\frac1n\\sum_i\\log\\hat p_{i,y_i}$", "pixeli standardizați cu media de antrenare @{fmnist.mu|.4f} și abaterea standard @{fmnist.sd|.4f}")),
           t(("\\textbf{Fair comparison}", "early stopping on the @{fmnist.n_valid|,d} validation images", "3 seeds per model; test images used once"),
             ("\\textbf{Comparație corectă}", "oprire timpurie pe cele @{fmnist.n_valid|,d} de imagini de validare", "3 semințe pentru fiecare model; imaginile de test sînt folosite o singură dată")), sep="3pt"), "0.5", "0.47")
        + D.bottom(t("total training time of the nine runs: @{fmnist.seconds|.0f} seconds on the GPU of a laptop.",
                     "timpul total de antrenare al celor nouă rulări: @{fmnist.seconds|.0f} de secunde pe procesorul grafic al unui laptop."), tryit=NB + "3"),
        "footnotesize", cite=OWN)

    D.figframe(t("The CNN learns faster and generalises better", "CNN învață mai repede și generalizează mai bine"),
               t("validation loss falls lowest for the CNN; the MLP fits the training data but stops improving on new images.",
                 "pierderea de validare scade cel mai mult pentru CNN; MLP se potrivește datelor de antrenare, dar nu mai progresează pe imagini noi."),
               None, "fmnist_curves",
               bullets=[t("Best validation epoch (seed @{seed}): CNN @{fmnist.cnn.best_epoch}, MLP @{fmnist.mlp.best_epoch}, softmax @{fmnist.softmax.best_epoch}, out of @{fmnist.epochs}.",
                          "Cea mai bună epocă pe validare (sămînța @{seed}): CNN @{fmnist.cnn.best_epoch}, MLP @{fmnist.mlp.best_epoch}, softmax @{fmnist.softmax.best_epoch}, din @{fmnist.epochs}.")],
               height="0.5\\textheight", cite=OWN, tryit=NB + "3")

    D.figframe(t("Higher accuracy with fewer parameters", "Acuratețe mai mare cu mai puțini parametri"),
               t("weight sharing buys accuracy and saves parameters at the same time.",
                 "partajarea ponderilor aduce acuratețe și economisește parametri în același timp."),
               None, "fmnist_compare",
               bullets=[t("Test accuracy (mean $\\pm$ sd, 3 seeds): softmax @{fmnist.softmax.acc_mean|.2f}\\,$\\pm$\\,@{fmnist.softmax.acc_sd|.2f}\\,\\%, MLP @{fmnist.mlp.acc_mean|.2f}\\,$\\pm$\\,@{fmnist.mlp.acc_sd|.2f}\\,\\%, CNN @{fmnist.cnn.acc_mean|.2f}\\,$\\pm$\\,@{fmnist.cnn.acc_sd|.2f}\\,\\%.",
                          "Acuratețea pe test (medie $\\pm$ abatere standard, 3 semințe): softmax @{fmnist.softmax.acc_mean|.2f}\\,$\\pm$\\,@{fmnist.softmax.acc_sd|.2f}\\,\\%, MLP @{fmnist.mlp.acc_mean|.2f}\\,$\\pm$\\,@{fmnist.mlp.acc_sd|.2f}\\,\\%, CNN @{fmnist.cnn.acc_mean|.2f}\\,$\\pm$\\,@{fmnist.cnn.acc_sd|.2f}\\,\\%."),
                        t("The CNN removes @{fmnist.cnn_err_cut_pct|.1f}\\,\\% of the MLP's errors with @{fmnist.params_ratio|.2f} times fewer parameters.",
                          "CNN elimină @{fmnist.cnn_err_cut_pct|.1f}\\,\\% din erorile MLP, cu de @{fmnist.params_ratio|.2f} ori mai puțini parametri.")],
               height="0.42\\textheight", cite=OWN)

    D.concept(t("Is the gain real? McNemar's test on the same test images", "Este cîștigul real? Testul McNemar pe aceleași imagini de test"),
              t("two classifiers scored on the same images differ only through the images on which exactly one of them is right.",
                "două clasificatoare evaluate pe aceleași imagini diferă doar prin imaginile pe care exact unul dintre ele le clasifică bine."),
              "\\[ \\chi^2=\\frac{(|b-c|-1)^2}{b+c}\\;\\overset{H_0}{\\sim}\\;\\chi^2_1 \\]"
              + t("\\where{$b$ images right for the CNN only; $c$ right for the MLP only; $H_0$: both have the same error rate; 5\\,\\% critical value 3.84}",
                  "\\where{$b$ imaginile corecte doar pentru CNN; $c$ corecte doar pentru MLP; $H_0$: ambele au aceeași rată de eroare; valoarea critică la 5\\,\\% este 3,84}"),
              it(t("Seed @{seed}: $b=@{fmnist.mcnemar.cnn_only}$, $c=@{fmnist.mcnemar.mlp_only}$, $\\chi^2=@{fmnist.mcnemar.chi2|.1f}$: the null is rejected.",
                   "Sămînța @{seed}: $b=@{fmnist.mcnemar.cnn_only}$, $c=@{fmnist.mcnemar.mlp_only}$, $\\chi^2=@{fmnist.mcnemar.chi2|.1f}$: ipoteza nulă este respinsă."),
                 t("Across the 3 seeds, the paired $t$ statistic of the accuracy gap is @{fmnist.diff_t|.2f}.",
                   "Pe cele 3 semințe, statistica $t$ pereche a diferenței de acuratețe este @{fmnist.diff_t|.2f}."),
                 t("Seeds measure the training noise; McNemar measures the test-set noise; report both.",
                   "Semințele măsoară zgomotul antrenării; testul McNemar măsoară zgomotul setului de test; raportați-le pe amîndouă.")),
              t("before replacing a production model, show that the gain survives a test on the same cases, not only a higher average.",
                "înainte de a înlocui un model aflat în producție, arătați că avantajul rezistă unui test pe aceleași cazuri, nu doar o medie mai mare."),
              cite=t("(McNemar, 1947)", "(McNemar, 1947)"))

    D.figframe(t("The errors are the products that people also confuse", "Erorile sînt la produsele pe care le confundă și oamenii"),
               t("most mistakes are between upper-body garments of similar shape.",
                 "cele mai multe greșeli apar între articolele pentru partea de sus a corpului, cu forme asemănătoare."),
               None, "fmnist_confusion",
               bullets=[t("Weakest class: @{fmnist.worst_class} (recall @{fmnist.worst_recall|.1f}\\,\\%); largest confusion: @{fmnist.top_conf.true} taken for @{fmnist.top_conf.pred}, @{fmnist.top_conf.n} images.",
                          "Clasa cea mai slabă: @{fmnist.worst_class_ro} (sensibilitate @{fmnist.worst_recall|.1f}\\,\\%); cea mai mare confuzie: @{fmnist.top_conf.true_ro} clasificat drept @{fmnist.top_conf.pred_ro}, @{fmnist.top_conf.n} de imagini.")],
               job=t("route only low-confidence photos of the weak classes to a human: the error cost falls where it is highest.",
                     "trimiteți la verificare umană doar fotografiile cu încredere scăzută din clasele slabe: costul erorilor scade acolo unde este cel mai mare."),
               height="0.5\\textheight", cite=OWN)

    D.figframe(t("First-layer filters are edge and contrast detectors", "Filtrele din primul strat detectează muchii și contraste"),
               t("nobody told the network to look for edges; gradient descent found them.",
                 "nimeni nu i-a cerut rețelei să caute muchii; coborîrea pe gradient le-a găsit singură."),
               None, "fmnist_filters",
               bullets=[t("Red = positive weight, blue = negative: many kernels are oriented differences, like the Sobel kernels set by hand.",
                          "Roșu = pondere pozitivă, albastru = negativă: multe nuclee sînt diferențe orientate, ca nucleele Sobel alese de mînă."),
                        t("Deeper layers combine them into parts (collars, soles, handles) (Zeiler \\& Fergus, 2014).",
                          "Straturile mai adînci le combină în părți (gulere, tălpi, mînere) (Zeiler \\& Fergus, 2014).")],
               height="0.2\\textheight", cite=t("(own computation; Zeiler \\& Fergus, 2014)", "(calcul propriu; Zeiler \\& Fergus, 2014)"))

    D.figframe(t("Feature maps: from edges to a coarse map of parts", "Hărțile de trăsături: de la muchii la o hartă grosieră a părților"),
               t("early maps keep the outline of the boot; deep maps are small but say where the informative parts are.",
                 "hărțile timpurii păstrează conturul ghetei; cele adînci sînt mici, dar arată unde se află părțile informative."),
               None, "fmnist_featuremaps",
               bullets=[t("Block 1: @{fmnist.fmap_sizes.l1.0} maps of @{fmnist.fmap_sizes.l1.1}$\\times$@{fmnist.fmap_sizes.l1.2}; block 3: @{fmnist.fmap_sizes.l3.0} maps of @{fmnist.fmap_sizes.l3.1}$\\times$@{fmnist.fmap_sizes.l3.2}, then averaged by GAP.",
                          "Blocul 1: @{fmnist.fmap_sizes.l1.0} de hărți de @{fmnist.fmap_sizes.l1.1}$\\times$@{fmnist.fmap_sizes.l1.2}; blocul 3: @{fmnist.fmap_sizes.l3.0} de hărți de @{fmnist.fmap_sizes.l3.1}$\\times$@{fmnist.fmap_sizes.l3.2}, apoi mediate de GAP.")],
               height="0.36\\textheight", cite=OWN)

    # ------------------------------------------------------------------ 4. Architectures
    D.section(t("Architectures: from LeNet to ResNet", "Arhitecturi: de la LeNet la ResNet"))
    D.frame(t("Image-recognition architectures, 1998--2021", "Arhitecturi pentru recunoașterea imaginilor, 1998--2021"),
            table("@{}p{0.17\\textwidth}p{0.07\\textwidth}p{0.68\\textwidth}@{}",
                  [t("Network", "Rețeaua"), t("Year", "Anul"), t("Key idea", "Ideea principală")],
                  [["LeNet-5", "1998", t("convolution + subsampling for handwritten digits (the MNIST benchmark); trained by backpropagation",
                                         "convoluție + subeșantionare pentru cifre scrise de mînă (banc de probă MNIST); antrenată prin propagarea înapoi a erorii")],
                   ["AlexNet", "2012", t("deep CNN on ImageNet trained on GPUs; ReLU, dropout, data augmentation", "CNN adîncă pe ImageNet antrenată pe GPU; ReLU, dropout, augmentarea datelor")],
                   [t("Visual Geometry Group (VGG)", "Visual Geometry Group (VGG)"), "2015", t("only $3\\times3$ kernels: two stacked $3\\times3$ see $5\\times5$ with $18C^2$ instead of $25C^2$ weights",
                                     "numai nuclee $3\\times3$: două nuclee $3\\times3$ suprapuse văd $5\\times5$ cu $18C^2$ în loc de $25C^2$ ponderi")],
                   ["ResNet", "2016", t("residual connections $\\mathbf y=\\mathbf x+F(\\mathbf x)$: networks of 100+ layers become trainable",
                                        "conexiuni reziduale $\\mathbf y=\\mathbf x+F(\\mathbf x)$: rețelele cu peste 100 de straturi devin antrenabile")],
                   [t("Vision transformer", "Transformer vizual"), "2021", t("image patches as tokens with attention (Lecture 7)", "fragmente de imagine tratate ca unități lexicale, cu atenție (cursul 7)")]]) +
            note(t("ResNet-18, used for transfer learning below, has @{eurosat.resnet_params|,d} parameters (our count, with a 10-class head).",
                   "ResNet-18, folosit mai jos pentru transferul învățării, are @{eurosat.resnet_params|,d} de parametri (numărați de noi, cu un strat final de 10 clase).")),
            "footnotesize", cite=t("(LeCun et al., 1998; Krizhevsky et al., 2012; Simonyan \\& Zisserman, 2015; He et al., 2016)",
                                   "(LeCun et al., 1998; Krizhevsky et al., 2012; Simonyan \\& Zisserman, 2015; He et al., 2016)"))

    D.concept(t("Residual connections and the identity path", "Conexiunile reziduale și calea identității"),
              t("a block adds its output to its input, so the gradient always has a direct path back.",
                "un bloc își adună ieșirea la intrare, astfel încît gradientul are mereu un drum direct înapoi."),
              "\\[ \\mathbf y=\\ReLU\\big(\\mathbf x+F(\\mathbf x;\\mW)\\big),\\qquad \\frac{\\partial\\loss}{\\partial\\mathbf x}=\\frac{\\partial\\loss}{\\partial\\mathbf y}\\Big(\\mathbf I+\\frac{\\partial F}{\\partial\\mathbf x}\\Big) \\]"
              + t("\\where{$F$ two Conv--BN layers; when the shape changes, $\\mathbf x$ passes through a $1\\times1$ convolution with stride 2; the derivative is written for the pre-activation sum}",
                  "\\where{$F$ două straturi Conv--BN; cînd forma se schimbă, $\\mathbf x$ trece printr-o convoluție $1\\times1$ cu pasul 2; derivata este scrisă pentru suma dinaintea activării}"),
              it(t("Plain deep networks suffer from degradation: deeper ones have higher training error, which is an optimisation problem, not overfitting.",
                   "Rețelele adînci simple suferă de degradare: cele mai adînci au eroare de antrenare mai mare, o problemă de optimizare, nu de supraînvățare."),
                 t("If the best block is the identity, it suffices to set $F=0$.", "Dacă cel mai bun bloc este identitatea, este suficient ca $F=0$."),
                 t("On CIFAR-10 (next frame): same 32 layers, with and without the shortcut.",
                   "Pe CIFAR-10 (diapozitivul următor): aceleași 32 de straturi, cu și fără scurtătură.")),
              t("residual blocks are the default in every modern vision and language model, including transformers.",
                "blocurile reziduale sînt alegerea implicită în toate modelele moderne de viziune și de limbaj, inclusiv în transformere."),
              cite=t("(He et al., 2016)", "(He et al., 2016)"))

    D.figframe(t("Augmentation narrows the gap; residuals make depth trainable", "Augmentarea reduce decalajul; conexiunile reziduale fac adîncimea antrenabilă"),
               t("left: the same CNN on Canadian Institute for Advanced Research (CIFAR-10) photos; right: 32-layer networks that differ only by the shortcut.",
                 "stînga: aceeași CNN pe fotografiile Canadian Institute for Advanced Research (CIFAR-10); dreapta: rețele de 32 de straturi care diferă doar prin scurtătură."),
               None, "cifar_curves",
               bullets=[t("Test accuracy: CNN @{cifar.cnn.test_acc|.2f}\\,\\% without and @{cifar.cnn_aug.test_acc|.2f}\\,\\% with augmentation (train--valid gap @{cifar.cnn.gap|.1f} vs @{cifar.cnn_aug.gap|.1f} points).",
                          "Acuratețea pe test: CNN @{cifar.cnn.test_acc|.2f}\\,\\% fără și @{cifar.cnn_aug.test_acc|.2f}\\,\\% cu augmentare (decalaj antrenare--validare @{cifar.cnn.gap|.1f} față de @{cifar.cnn_aug.gap|.1f} puncte)."),
                        t("Plain-32 @{cifar.plain32.test_acc|.2f}\\,\\% vs ResNet-32 @{cifar.res32.test_acc|.2f}\\,\\%; final training loss @{cifar.plain32.train_loss|.3f} vs @{cifar.res32.train_loss|.3f}.",
                          "Rețeaua simplă cu 32 de straturi @{cifar.plain32.test_acc|.2f}\\,\\%, ResNet-32 @{cifar.res32.test_acc|.2f}\\,\\%; pierderea finală la antrenare @{cifar.plain32.train_loss|.3f} față de @{cifar.res32.train_loss|.3f}.")],
               height="0.42\\textheight", cite=OWN, tryit=NB + "4")

    D.concept(t("Data augmentation as prior information", "Augmentarea datelor ca informație a priori"),
              t("we train on random transformations that should not change the label, which averages the loss over them.",
                "antrenăm pe transformări aleatoare care nu ar trebui să schimbe eticheta, adică mediem pierderea peste ele."),
              "\\[ \\min_{\\btheta}\\;\\frac1n\\sum_{i=1}^n\\E_{T\\sim\\mathcal T}\\Big[\\ell\\big(f_{\\btheta}(T(\\mathbf X_i)),y_i\\big)\\Big] \\]"
              + t("\\where{$\\mathcal T$ the set of label-preserving transformations; CIFAR-10: random crops after 4-pixel padding and horizontal flips; EuroSAT: flips and rotations by $90^\\circ$}",
                  "\\where{$\\mathcal T$ mulțimea transformărilor care păstrează eticheta; CIFAR-10: decupări aleatoare după o completare de 4 pixeli și oglindiri orizontale; EuroSAT: oglindiri și rotații cu $90^\\circ$}"),
              it(t("Gain on CIFAR-10 for the same network and budget: @{cifar.aug_gain|+.2f} percentage points of test accuracy.",
                   "Cîștigul pe CIFAR-10, pentru aceeași rețea și același buget: @{cifar.aug_gain|+.2f} puncte procentuale de acuratețe pe test."),
                 t("Choose $\\mathcal T$ by domain: a flipped digit 6 is not a 6, but a satellite tile has no ``up''.",
                   "Alegeți $\\mathcal T$ după domeniu: o cifră 6 oglindită nu mai este 6, dar o imagine satelitară nu are ``sus''."),
                 t("ResNet-32 per-class recall: best @{cifar.res32_best} @{cifar.res32_best_recall|.1f}\\,\\%, worst @{cifar.res32_worst} @{cifar.res32_worst_recall|.1f}\\,\\%.",
                   "Sensibilitatea pe clase a ResNet-32: cea mai bună @{cifar.res32_best_ro} @{cifar.res32_best_recall|.1f}\\,\\%, cea mai slabă @{cifar.res32_worst_ro} @{cifar.res32_worst_recall|.1f}\\,\\%.")),
              t("augmentation is free extra data; it is the first lever to pull before buying more labels.",
                "augmentarea înseamnă date suplimentare gratuite; este prima pîrghie de folosit înainte de a cumpăra mai multe etichete."),
              cite=OWN)

    # ------------------------------------------------------------------ 5. Transfer learning on EuroSAT
    D.section(t("Transfer learning: land use from satellite images", "Transferul învățării: utilizarea terenului din imagini satelitare"))
    D.figframe(t("Satellite tiles: land use for lenders, insurers and investors", "Imagini satelitare: utilizarea terenului pentru creditori, asigurători și investitori"),
               t("ten land-use classes from Sentinel-2 images of European Union (EU) countries; the question is how few labels we can afford.",
                 "zece clase de utilizare a terenului din imagini Sentinel-2 ale unor țări din Uniunea Europeană (UE); întrebarea este cu cît de puține etichete ne descurcăm."),
               "\\[ \\mathbf X\\in[0,1]^{64\\times64\\times3},\\qquad y\\in\\{\\text{" + t("crop", "culturi") + "},\\dots,\\text{" + t("sea/lake", "mare/lac") + "}\\} \\]",
               "eurosat_samples",
               bullets=[t("@{eurosat.n|,d} red, green, blue (RGB) tiles of $@{eurosat.size}\\times@{eurosat.size}$ pixels; @{eurosat.per_class_min|,d} to @{eurosat.per_class_max|,d} per class.",
                          "@{eurosat.n|,d} de imagini roșu, verde, albastru (RGB) de $@{eurosat.size}\\times@{eurosat.size}$ pixeli; între @{eurosat.per_class_min|,d} și @{eurosat.per_class_max|,d} pe clasă.")],
               job=t("an agricultural lender checks that a farm loan finances crops; an insurer maps flood-exposed residential areas.",
                     "un creditor agricol verifică dacă un împrumut finanțează culturi; un asigurător cartografiază zonele rezidențiale expuse inundațiilor."),
               height="0.36\\textheight", cite=t("(Helber et al., 2019)", "(Helber et al., 2019)"))

    D.concept(t("Transfer learning: reuse features learned on a million photos", "Transferul învățării: refolosim trăsături învățate pe un milion de fotografii"),
              t("the early layers of a network trained on ImageNet detect edges and textures that are useful for any image.",
                "primele straturi ale unei rețele antrenate pe ImageNet detectează muchii și texturi utile pentru orice imagine."),
              "\\[ f_{\\btheta}(\\mathbf X)=g_{\\boldsymbol\\psi}\\big(\\phi_{\\boldsymbol\\omega}(\\mathbf X)\\big),\\quad \\boldsymbol\\omega^{(0)}=\\boldsymbol\\omega_{\\mathrm{ImageNet}} \\]"
              + t("\\where{$\\phi_{\\boldsymbol\\omega}$ the convolutional body (512 features after GAP); $g_{\\boldsymbol\\psi}$ a new linear head with 10 outputs}",
                  "\\where{$\\phi_{\\boldsymbol\\omega}$ corpul convoluțional (512 trăsături după GAP); $g_{\\boldsymbol\\psi}$ un nou strat final liniar cu 10 ieșiri}"),
              it(t(("\\textbf{Feature extraction}", "$\\boldsymbol\\omega$ frozen; fit only $\\boldsymbol\\psi$ (here a logistic regression, $C$ chosen on validation)"),
                   ("\\textbf{Extragerea trăsăturilor}", "$\\boldsymbol\\omega$ înghețat; se estimează doar $\\boldsymbol\\psi$ (aici o regresie logistică, cu $C$ ales pe validare)")),
                 t(("\\textbf{Fine-tuning}", "all weights trained, smaller learning rate for the body ($3\\cdot10^{-4}$) than for the head ($3\\cdot10^{-3}$), chosen a priori"),
                   ("\\textbf{Ajustarea fină}", "toate ponderile sînt antrenate, cu rată de învățare mai mică pentru corp ($3\\cdot10^{-4}$) decît pentru stratul final ($3\\cdot10^{-3}$), alese a priori")),
                 t(("\\textbf{From scratch}", "a 4-block CNN, @{eurosat.scratch_params|,d} parameters, random start"),
                   ("\\textbf{De la zero}", "o CNN cu 4 blocuri, @{eurosat.scratch_params|,d} de parametri, inițializare aleatoare"))),
              t("feature extraction is the cheapest: one pass over the images and a logistic regression.",
                "extragerea trăsăturilor este cea mai ieftină: o singură trecere prin imagini și o regresie logistică."),
              cite=t("(Yosinski et al., 2014)", "(Yosinski et al., 2014)"))

    D.frame(t("Protocol: same split, same test, three strategies", "Protocolul: aceeași împărțire, același test, trei strategii"), cols(it(
        t(("\\textbf{Split}, stratified by class", "@{eurosat.n_train|,d} train / @{eurosat.n_valid|,d} validation / @{eurosat.n_test|,d} test tiles"),
          ("\\textbf{Împărțirea}, stratificată pe clase", "@{eurosat.n_train|,d} de imagini de antrenare / @{eurosat.n_valid|,d} de validare / @{eurosat.n_test|,d} de test")),
        t(("\\textbf{Input}", "tiles resized to $@{eurosat.res}\\times@{eurosat.res}$ for ResNet-18, scaled with the ImageNet mean and standard deviation per channel",
           "$\\tilde x_c=(x_c-\\mu_c)/s_c$"),
          ("\\textbf{Intrarea}", "imaginile sînt redimensionate la $@{eurosat.res}\\times@{eurosat.res}$ pentru ResNet-18 și scalate cu media și abaterea standard ImageNet ale fiecărui canal",
           "$\\tilde x_c=(x_c-\\mu_c)/s_c$")), sep="3pt"),
        it(t(("\\textbf{Training}", "AdamW, batch 64, one-cycle schedule; flips and $90^\\circ$ rotations",
              "from scratch @{eurosat.scratch_epochs} epochs; fine-tuning @{eurosat.ft_epochs} epochs", "model kept at its best validation accuracy"),
             ("\\textbf{Antrenarea}", "AdamW, loturi de 64, programare one-cycle; oglindiri și rotații cu $90^\\circ$",
              "de la zero @{eurosat.scratch_epochs} epoci; ajustare fină @{eurosat.ft_epochs} epoci", "se păstrează modelul cu cea mai bună acuratețe de validare")),
           t(("\\textbf{Cost}", "@{eurosat.seconds|.0f} seconds for the three main runs on a laptop GPU"),
             ("\\textbf{Costul}", "@{eurosat.seconds|.0f} de secunde pentru cele trei rulări principale, pe procesorul grafic al unui laptop")), sep="3pt"), "0.5", "0.47")
        + D.bottom(t("fixing the test tiles before any modelling is what makes the three strategies comparable.",
                     "fixarea imaginilor de test înaintea oricărei modelări face comparabile cele trei strategii."), tryit=NB + "5"),
        "footnotesize", cite=OWN)

    D.figframe(t("Fine-tuning wins, and transfer matters most when labels are few", "Ajustarea fină cîștigă, iar transferul contează cel mai mult cînd etichetele sînt puține"),
               t("left: learning with all labels; right: test accuracy when only $n$ labelled tiles per class are available.",
                 "stînga: învățarea cu toate etichetele; dreapta: acuratețea pe test cînd sînt disponibile doar $n$ imagini etichetate pe clasă."),
               None, "transfer_curves",
               bullets=[t("Test accuracy, all labels: from scratch @{eurosat.scratch_acc|.2f}\\,\\%, frozen + logistic @{eurosat.probe_acc|.2f}\\,\\%, fine-tuned @{eurosat.finetune_acc|.2f}\\,\\%.",
                          "Acuratețea pe test, cu toate etichetele: de la zero @{eurosat.scratch_acc|.2f}\\,\\%, înghețat + logistic @{eurosat.probe_acc|.2f}\\,\\%, ajustat fin @{eurosat.finetune_acc|.2f}\\,\\%."),
                        t("With @{budget.n_small} tiles per class: scratch @{budget.scratch.10|.1f}\\,\\%, frozen @{budget.probe.10|.1f}\\,\\%, fine-tuned @{budget.finetune.10|.1f}\\,\\%.",
                          "Cu @{budget.n_small} imagini pe clasă: de la zero @{budget.scratch.10|.1f}\\,\\%, înghețat @{budget.probe.10|.1f}\\,\\%, ajustat fin @{budget.finetune.10|.1f}\\,\\%.")],
               height="0.42\\textheight", cite=OWN, tryit=NB + "5")

    rows = []
    for n in ["10", "25", "50", "100", "250", "1000"]:
        rows.append([n, f"@{{budget.scratch.{n}|.1f}}", f"@{{budget.probe.{n}|.1f}}", f"@{{budget.finetune.{n}|.1f}}"])
    D.frame(t("The label budget: what transfer learning saves", "Bugetul de etichete: cît economisește transferul învățării"), cols(
        table("@{}rrrr@{}", [t("Tiles/class", "Imagini/clasă"), t("Scratch", "De la zero"), t("Frozen", "Înghețat"), t("Fine-tuned", "Ajustat fin")], rows) +
        note(t("Test accuracy (\\%), mean of 2 random draws of the labelled tiles; same 5,400 test tiles.",
               "Acuratețea pe test (\\%), media a 2 extrageri aleatoare ale imaginilor etichetate; aceleași 5.400 de imagini de test.")),
        it(t("With @{budget.tiles_small} labelled tiles in total, the pre-trained network already beats a CNN trained from scratch on the same labels by a wide margin.",
             "Cu doar @{budget.tiles_small} de imagini etichetate în total, rețeaua preantrenată depășește clar o CNN antrenată de la zero pe aceleași etichete."),
           t("The scratch CNN needs the full budget to approach the fine-tuned network.",
             "CNN-ul de la zero are nevoie de întregul buget pentru a se apropia de rețeaua ajustată fin."),
           t("The run of the whole table took @{budget.seconds|.0f} seconds.", "Rularea întregului tabel a durat @{budget.seconds|.0f} de secunde."), sep="4pt"), "0.47", "0.5")
        + D.bottom(t("labelling a tile costs an expert's minute; transfer learning turns weeks of labelling into an afternoon.",
                     "etichetarea unei imagini costă un minut de lucru al unui expert; transferul învățării transformă săptămîni de etichetare într-o după-amiază.")),
        "footnotesize", cite=OWN)

    D.figframe(t("Remaining errors: annual versus permanent crops", "Erorile rămase: culturi anuale față de culturi permanente"),
               t("the fine-tuned network mixes classes that look alike from space.",
                 "rețeaua ajustată fin confundă clase care arată la fel din spațiu."),
               None, "eurosat_confusion",
               bullets=[t("@{eurosat.n_errors_ft} errors on @{eurosat.n_test|,d} tiles; largest: @{eurosat.top_conf.true} taken for @{eurosat.top_conf.pred} (@{eurosat.top_conf.n}); macro-F1 @{eurosat.macro_f1_ft|.2f}\\,\\%.",
                          "@{eurosat.n_errors_ft} de erori la @{eurosat.n_test|,d} de imagini; cea mai mare: @{eurosat.top_conf.true_ro} clasificat drept @{eurosat.top_conf.pred_ro} (@{eurosat.top_conf.n}); macro-F1 @{eurosat.macro_f1_ft|.2f}\\,\\%.")],
               height="0.5\\textheight", cite=OWN)

    D.concept(t("Grad-CAM: where in the tile does the decision come from?", "Grad-CAM: din ce parte a imaginii vine decizia?"),
              t("weight each feature map by how much the class score reacts to it, sum, and keep the positive part.",
                "ponderăm fiecare hartă de trăsături după cît reacționează scorul clasei la ea, adunăm și păstrăm partea pozitivă."),
              "\\[ \\alpha^{c}_o=\\frac{1}{HW}\\sum_{i,j}\\frac{\\partial z_c}{\\partial A_o(i,j)} \\]\\vspace{-2mm}\\[ M^{c}=\\ReLU\\Big(\\sum_o\\alpha^{c}_oA_o\\Big) \\]"
              + t("\\where{$z_c$ the score (logit) of class $c$; $A_o$ feature map $o$ of the chosen layer; $M^c$ the class activation map (CAM), upsampled to the tile}",
                  "\\where{$z_c$ scorul (logitul) clasei $c$; $A_o$ harta de trăsături $o$ din stratul ales; $M^c$ harta de activare a clasei (CAM), mărită la dimensiunea imaginii}"),
              D.fig("gradcam", "0.55\\textheight"),
              t("an auditor can check that ``industrial'' is decided by the roofs, not by a cloud or a border artefact.",
                "un auditor poate verifica dacă ``zonă industrială'' este decisă de acoperișuri, nu de un nor sau de un artefact de margine."),
              cite=t("(Selvaraju et al., 2017; own computation)", "(Selvaraju et al., 2017; calcul propriu)"), wl="0.46", wr="0.51")

    D.concept(t("From tiles to a decision: land-use shares", "De la imagini la decizie: ponderile utilizării terenului"),
              t("a portfolio question is about shares and totals, so errors that cancel matter less than errors that do not.",
                "o întrebare de portofoliu privește ponderi și totaluri, deci erorile care se compensează contează mai puțin decît cele care nu se compensează."),
              "\\[ \\hat\\pi_k=\\frac1n\\sum_{i=1}^n\\mathbf 1\\{\\hat y_i=k\\},\\qquad \\E[\\hat\\pi_k]=\\sum_j\\pi_j\\,P(\\hat y=k\\mid y=j) \\]"
              + t("\\where{$\\pi_k$ true share of class $k$ in a region; $P(\\hat y=k\\mid y=j)$ the confusion matrix, estimated on the test tiles}",
                  "\\where{$\\pi_k$ ponderea reală a clasei $k$ într-o regiune; $P(\\hat y=k\\mid y=j)$ matricea de confuzie, estimată pe imaginile de test}"),
              it(t("Test tiles: share of crops (annual + permanent) @{landuse.crop_true|.2f}\\,\\% true vs @{landuse.crop_pred|.2f}\\,\\% predicted.",
                   "Imaginile de test: ponderea culturilor (anuale + permanente) este @{landuse.crop_true|.2f}\\,\\% în realitate și @{landuse.crop_pred|.2f}\\,\\% prezisă."),
                 t("Largest error of any class share: @{landuse.max_abs_share_err_pp|.2f} percentage points.",
                   "Cea mai mare eroare a ponderii oricărei clase: @{landuse.max_abs_share_err_pp|.2f} puncte procentuale."),
                 t("Inverting the confusion matrix corrects the shares when the error rates are stable (domain shift breaks this).",
                   "Inversarea matricei de confuzie corectează ponderile atunci cînd ratele de eroare sînt stabile (schimbarea domeniului strică acest lucru).")),
              t("for an insurer or an agricultural lender the output is a share of hectares per region, with an error bar, not a label per tile.",
                "pentru un asigurător sau un creditor agricol, rezultatul este ponderea hectarelor pe regiune, cu o marjă de eroare, nu o etichetă pentru fiecare imagine."),
              cite=OWN)

    D.frame(t("When transfer learning helps, and when it can mislead", "Cînd ajută transferul învățării și cînd poate induce în eroare"), cols(it(
        t(("\\textbf{Helps most}", "few labels (right panel of the transfer chart)", "a source domain with similar low-level statistics (edges, textures)",
           "a limited compute budget: frozen features cost one forward pass"),
          ("\\textbf{Ajută cel mai mult}", "cînd sînt puține etichete (panoul din dreapta al graficului de transfer)", "cînd domeniul sursă are statistici de nivel jos similare (muchii, texturi)",
           "cînd puterea de calcul este limitată: trăsăturile înghețate costă o singură trecere înainte")), sep="4pt"),
        it(t(("\\textbf{Can mislead}", "domain shift: other season, other sensor, other country than the training tiles",
              "spurious cues learned from the source (watermarks, borders)", "the test set must come from the population where the model will be used"),
             ("\\textbf{Poate induce în eroare}", "schimbarea domeniului: alt anotimp, alt senzor, altă țară decît imaginile de antrenare",
              "indicii false învățate din sursă (filigrane, margini)", "setul de test trebuie să provină din populația în care va fi folosit modelul")), sep="4pt"), "0.49", "0.48")
        + D.bottom(t("before deployment, label a small sample from the target region and re-measure; the fine-tuned model is cheap to update.",
                     "înainte de utilizare, etichetați un mic eșantion din regiunea țintă și măsurați din nou; modelul ajustat fin se actualizează ieftin.")),
        "footnotesize", cite=t("(Yosinski et al., 2014; Helber et al., 2019)", "(Yosinski et al., 2014; Helber et al., 2019)"))

    D.concept(t("Convolutions beyond photos: sequences and charts", "Convoluțiile dincolo de fotografii: secvențe și grafice"),
              t("a one-dimensional convolution slides over time, the same way a two-dimensional one slides over space.",
                "o convoluție unidimensională glisează în timp, la fel cum una bidimensională glisează în spațiu."),
              "\\[ z_t=b+\\sum_{c}\\sum_{u=0}^{k-1}K_c(u)\\,x_{c,\\,t-u} \\]"
              + t("\\where{$x_{c,t}$ input series $c$ at time $t$ (returns, volume, load); causal: only past values}",
                  "\\where{$x_{c,t}$ seria de intrare $c$ la momentul $t$ (randamente, volum, consum); cauzală: doar valori trecute}"),
              it(t("A one-dimensional CNN is a learned bank of moving-average filters; Lecture 5 compares it with recurrent networks.",
                   "O CNN unidimensională este un set învățat de filtre de tip medie mobilă; cursul 5 o compară cu rețelele recurente."),
                 t("Charts of markets can be read as images by a vision--language model (VLM): research of our group, next section.",
                   "Graficele piețelor pot fi citite ca imagini de un model viziune--limbaj (VLM): cercetarea grupului nostru, în secțiunea următoare."),
                 t("Same rules: real data, a time-ordered split, a simple benchmark first.",
                   "Aceleași reguli: date reale, împărțire în ordine cronologică, întîi un model de referință simplu.")),
              t("a risk desk can screen hundreds of chart panels per day with a model, and keep the analyst for the flagged ones.",
                "un departament de risc poate examina zilnic sute de grafice cu un model, iar analistul se ocupă doar de cele semnalate."),
              cite=t("(Bishop \\& Bishop, 2024, ch.~10)", "(Bishop \\& Bishop, 2024, cap.~10)"))

    # ------------------------------------------------------------------ wrap-up
    D.parts.append("\\section*{" + t("Wrap-up", "Recapitulare") + "}\n")
    D.quiz_frame(t("output size", "dimensiunea ieșirii"),
                 t("A $64\\times64$ tile passes a $3\\times3$ convolution with padding 1 and stride 2; how large is the output map?",
                   "O imagine de $64\\times64$ trece printr-o convoluție $3\\times3$ cu completare 1 și pasul 2; cît de mare este harta de ieșire?"),
                 t("$\\lfloor(64+2-3)/2\\rfloor+1=@{conv.out_s2}$, so $@{conv.out_s2}\\times@{conv.out_s2}$.",
                   "$\\lfloor(64+2-3)/2\\rfloor+1=@{conv.out_s2}$, deci $@{conv.out_s2}\\times@{conv.out_s2}$."))
    D.quiz_frame(t("parameters", "parametrii"),
                 t("Why does our CNN beat the MLP on Fashion-MNIST with fewer parameters?",
                   "De ce depășește CNN-ul nostru MLP pe Fashion-MNIST, deși are mai puțini parametri?"),
                 t("Weight sharing and local connectivity encode the structure of images; the CNN reaches @{fmnist.cnn.acc_mean|.2f}\\,\\% with @{fmnist.cnn.params|,d} parameters against @{fmnist.mlp.acc_mean|.2f}\\,\\% with @{fmnist.mlp.params|,d}.",
                   "Partajarea ponderilor și conectivitatea locală codifică structura imaginilor; CNN atinge @{fmnist.cnn.acc_mean|.2f}\\,\\% cu @{fmnist.cnn.params|,d} de parametri, față de @{fmnist.mlp.acc_mean|.2f}\\,\\% cu @{fmnist.mlp.params|,d}."))
    D.quiz_frame(t("depth", "adîncimea"),
                 t("A plain 32-layer network has a higher training loss than a residual one of the same depth; is that overfitting?",
                   "O rețea simplă de 32 de straturi are o pierdere la antrenare mai mare decît una reziduală de aceeași adîncime; este supraînvățare?"),
                 t("No: overfitting lowers the training loss; this is the degradation problem (@{cifar.plain32.train_loss|.3f} vs @{cifar.res32.train_loss|.3f}), solved by the identity shortcut.",
                   "Nu: supraînvățarea scade pierderea la antrenare; aceasta este problema degradării (@{cifar.plain32.train_loss|.3f} față de @{cifar.res32.train_loss|.3f}), rezolvată de scurtătura identitate."))
    D.quiz_frame(t("a label budget", "un buget de etichete"),
                 t("You can label only 100 satellite tiles in total; which strategy do you choose?",
                   "Puteți eticheta în total doar 100 de imagini satelitare; ce strategie alegeți?"),
                 t("A pre-trained network: with @{budget.n_small} tiles per class, fine-tuning gives @{budget.finetune.10|.1f}\\,\\% and frozen features @{budget.probe.10|.1f}\\,\\%, against @{budget.scratch.10|.1f}\\,\\% from scratch.",
                   "O rețea preantrenată: cu @{budget.n_small} imagini pe clasă, ajustarea fină dă @{budget.finetune.10|.1f}\\,\\%, iar trăsăturile înghețate @{budget.probe.10|.1f}\\,\\%, față de @{budget.scratch.10|.1f}\\,\\% de la zero."))
    D.quiz_frame(t("explanation", "explicația"),
                 t("A Grad-CAM map of an ``industrial'' tile lights up a cloud; what do you do?",
                   "Harta Grad-CAM a unei imagini ``zonă industrială'' evidențiază un nor; ce faceți?"),
                 t("Treat it as a warning of a spurious cue: inspect similar tiles, add cloudy examples or a cloud mask, and re-test on the target region.",
                   "O tratați ca pe un semnal de indiciu fals: verificați imagini asemănătoare, adăugați exemple cu nori sau o mască de nori și testați din nou pe regiunea țintă."))

    D.frame(t("Wrap-up", "Recapitulare"), "\\vspace{-2mm}\\begin{multicols}{2}\\scriptsize\\tableofcontents\\end{multicols}\n" + it(
        t("A convolution is a shared local filter; padding, stride and pooling set the map size, depth and pooling the receptive field.",
          "O convoluție este un filtru local partajat; completarea, pasul și agregarea stabilesc dimensiunea hărții, iar adîncimea și agregarea cîmpul receptiv."),
        t("On real images the CNN beats the MLP with fewer parameters; residual blocks make depth trainable; augmentation is free data.",
          "Pe imagini reale, CNN depășește MLP cu mai puțini parametri; blocurile reziduale fac adîncimea antrenabilă; augmentarea înseamnă date gratuite."),
        t("Transfer learning wins when labels are scarce. Next: sequences and recurrent networks (Lecture 5).",
          "Transferul învățării cîștigă atunci cînd etichetele sînt rare; urmează secvențele și rețelele recurente (cursul 5)."), sep="1pt"))

    D.frame(t("Research from our group: vision models for finance", "Cercetări ale grupului nostru: modele de viziune în finanțe"), it(
        t(("\\textbf{A multimodal vision--language framework for financial anomaly detection} (Jheng, Pele, Tak \\& Găman, 2026, PICBE), Best Paper Award at the International Conference on Business Excellence (ICBE) 2026",
           "market indicators projected onto a standardised multi-panel image; market regime classification treated as visual recognition",
           "zero-shot vision--language model (GPT-4.1, a generative pre-trained transformer) with rolling-quantile labels and structured natural-language rationales",
           "out-of-sample tests on the S\\&P 500: promising fidelity in detecting systemic anomalies, with interpretable explanations"),
          ("\\textbf{Un cadru multimodal viziune--limbaj pentru detectarea anomaliilor financiare} (Jheng, Pele, Tak \\& Găman, 2026, PICBE), premiul Best Paper la International Conference on Business Excellence (ICBE) 2026",
           "indicatorii de piață sînt proiectați într-o imagine standardizată, cu mai multe panouri; clasificarea regimurilor de piață devine o problemă de recunoaștere vizuală",
           "model viziune--limbaj folosit fără antrenare suplimentară (GPT-4.1, transformer generativ preantrenat), cu etichete din cuantile glisante și justificări structurate în limbaj natural",
           "teste în afara eșantionului pe S\\&P 500: detectarea anomaliilor sistemice cu o fidelitate promițătoare și cu explicații interpretabile")),
        t(("\\textbf{Quantlet: the code snippet knowledge platform} (Găman, Winkel, Zuo, Härdle \\& Pele, 2026, Computational Statistics)",
           "the platform on which every chart of this lecture is published as a reproducible Quantlet"),
          ("\\textbf{Quantlet: platforma de cunoștințe pentru fragmente de cod} (Găman, Winkel, Zuo, Härdle \\& Pele, 2026, Computational Statistics)",
           "platforma pe care fiecare grafic din acest curs este publicat ca Quantlet reproductibil")), sep="4pt") +
        note(t("Findings as stated in the papers; numbers on the other slides come from our own computations.",
               "Rezultatele de pe acest diapozitiv sînt cele raportate în lucrări; celelalte cifre din curs provin din calculele noastre.")),
        "footnotesize", cite=t("(Bucharest University of Economic Studies; Romanian Academy, Institute for Economic Forecasting)",
                               "(Academia de Studii Economice din București; Academia Română, Institutul de Prognoză Economică)"))

    D.frame(t("Want to learn more?", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Bishop \\& Bishop (2024)}, ch.~10: convolutional networks, with the derivations of this lecture.",
          "\\textbf{Bishop \\& Bishop (2024)}, cap.~10: rețele convoluționale, cu derivările din acest curs."),
        t("\\textbf{Goodfellow, Bengio \\& Courville (2016)}, ch.~9: convolution, pooling, variants.",
          "\\textbf{Goodfellow, Bengio \\& Courville (2016)}, cap.~9: convoluție, agregare, variante."),
        t("\\textbf{He et al. (2016)}: the ResNet paper, short and readable.", "\\textbf{He et al. (2016)}: articolul ResNet, scurt și ușor de citit."), sep="4pt"),
        it(t("\\textbf{This course}: code, data, notebooks --- ", "\\textbf{Acest curs}: cod, date, notebook-uri Jupyter --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Experiment yourself}: the notebook reproduces the charts of this lecture --- ",
             "\\textbf{Exercițiu practic}: notebook-ul reproduce graficele acestui curs --- ") + "\\href{\\nnrepo/\\nnnb}{notebooks/}.",
           t("\\textbf{Seminar 4}: transfer learning on EuroSAT with your own label budget.",
             "\\textbf{Seminarul 4}: transferul învățării pe EuroSAT, cu propriul buget de etichete."), sep="4pt"), "0.52", "0.45"))

    reference_frames(D, REFS, per_frame=9)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.frame(t("Appendix: convolution as a matrix product", "Anexă: convoluția ca produs de matrice"), it(
        t("Stack every $k\\times k\\times C_{\\mathrm{in}}$ patch of the input as a row of $\\mathbf P\\in\\R^{H_{\\mathrm{out}}W_{\\mathrm{out}}\\times k^2C_{\\mathrm{in}}}$ (``im2col'').",
          "Fiecare zonă de $k\\times k\\times C_{\\mathrm{in}}$ a intrării devine un rînd al matricei $\\mathbf P\\in\\R^{H_{\\mathrm{out}}W_{\\mathrm{out}}\\times k^2C_{\\mathrm{in}}}$ (``im2col'')."),
        t("Stack the kernels as columns of $\\mathbf K\\in\\R^{k^2C_{\\mathrm{in}}\\times C_{\\mathrm{out}}}$; then $\\mathbf Z=\\mathbf P\\mathbf K+\\mathbf 1\\vb^{\\top}$.",
          "Nucleele devin coloanele matricei $\\mathbf K\\in\\R^{k^2C_{\\mathrm{in}}\\times C_{\\mathrm{out}}}$; atunci $\\mathbf Z=\\mathbf P\\mathbf K+\\mathbf 1\\vb^{\\top}$."),
        t("Gradients follow from the matrix rules of Lecture 2: $\\partial\\loss/\\partial\\mathbf K=\\mathbf P^{\\top}\\boldsymbol\\Delta$ and $\\partial\\loss/\\partial\\mathbf P=\\boldsymbol\\Delta\\mathbf K^{\\top}$, with $\\boldsymbol\\Delta=\\partial\\loss/\\partial\\mathbf Z$.",
          "Gradienții rezultă din regulile matriceale din cursul 2: $\\partial\\loss/\\partial\\mathbf K=\\mathbf P^{\\top}\\boldsymbol\\Delta$ și $\\partial\\loss/\\partial\\mathbf P=\\boldsymbol\\Delta\\mathbf K^{\\top}$, cu $\\boldsymbol\\Delta=\\partial\\loss/\\partial\\mathbf Z$."),
        t("Adding back the rows of $\\partial\\loss/\\partial\\mathbf P$ to their pixel positions (``col2im'') gives $\\partial\\loss/\\partial\\mathbf X$: the full convolution of the main text.",
          "Readunarea rîndurilor din $\\partial\\loss/\\partial\\mathbf P$ la pozițiile pixelilor lor (``col2im'') dă $\\partial\\loss/\\partial\\mathbf X$: convoluția completă din textul principal."),
        t("A $1\\times1$ convolution is the special case $k=1$: a dense layer applied to the channels of every pixel.",
          "O convoluție $1\\times1$ este cazul particular $k=1$: un strat dens aplicat canalelor fiecărui pixel."), sep="3pt"), "footnotesize")
    D.frame(t("Appendix: the CNN of the case study in PyTorch", "Anexă: CNN-ul din studiul de caz, în PyTorch"),
            code(
            "block = lambda a, b: [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]\n"
            "cnn = nn.Sequential(*block(1, 32), *block(32, 32), nn.MaxPool2d(2),     # 28 -> 14\n"
            "                    *block(32, 64), *block(64, 64), nn.MaxPool2d(2),    # 14 -> 7\n"
            "                    *block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n"
            "                    nn.Dropout(0.3), nn.Linear(128, 10))\n"
            "resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')\n"
            "resnet.fc = nn.Linear(512, 10)                # new head for the 10 land-use classes\n"
            ) + it(t("Full code: lectures/L4/analysis\\_L4.py and the notebook of this lecture.",
                                      "Codul complet: lectures/L4/analysis\\_L4.py și notebook-ul acestui curs.")), "footnotesize")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 4, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
