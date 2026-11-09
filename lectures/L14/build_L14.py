r"""
build_L14.py -- Lecture 14 "Multimodal and vision-language models; synthesis of the course" (NN course), EN + RO.
Writes lecture14.tex / lecture14_ro.tex from the slide text below and L14_numbers.json (analysis_L14.py), compiles both and
runs every check of lectures/STYLE.md (nn_build.run_build).
Run:  python3 lectures/L14/analysis_L14.py && python3 lectures/L14/make_quantlets_L14.py && python3 lectures/L14/build_L14.py
Hand-typed numbers are definitions only (image sizes, patch size 16, window lengths and quantile levels fixed a priori).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, block, cols, fmt, heat, hier, items,  # noqa: E402
                      reference_frames, run_build, table)

ABBR = {
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "AUC": ("area under the receiver operating characteristic curve", "aria de sub curba caracteristică de operare (ROC)"),
    "BCE": (None, "Banca Centrală Europeană"),
    "CLIP": ("contrastive language--image pre-training", "preantrenare contrastivă limbaj--imagine"),
    "CNN": ("convolutional neural network", "rețea neuronală convoluțională"),
    "ECB": ("European Central Bank", None),
    "ESG": ("environmental, social and governance", "criterii de mediu, sociale și de guvernanță"),
    "GAN": ("generative adversarial network", "rețea generativă adversarială"),
    "GNN": ("graph neural network", "rețea neuronală pe grafuri"),
    "GPT": ("generative pre-trained transformer", "transformer generativ preantrenat"),
    "ICBE": ("International Conference on Business Excellence", "Conferința internațională Business Excellence"),
    "InfoNCE": ("information noise-contrastive estimation loss", "funcția de pierdere contrastivă InfoNCE (information noise-contrastive estimation)"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "LLaVA": ("large language and vision assistant", "asistentul lingvistic și vizual LLaVA (large language and vision assistant)"),
    "LLM": ("large language model", "model lingvistic mare"),
    "LSTM": ("long short-term memory network", "rețea cu memorie pe termen scurt și lung"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "OCR": ("optical character recognition", "recunoașterea optică a caracterelor"),
    "RAG": ("retrieval-augmented generation", "generare augmentată prin regăsire"),
    "RGB": ("red, green, blue (colour channels)", "roșu, verde, albastru (canalele de culoare)"),
    "RL": ("reinforcement learning", "învățare prin recompensă"),
    "ROC": ("receiver operating characteristic", "caracteristica de operare a receptorului"),
    "SHAP": ("Shapley additive explanations", "explicații aditive Shapley"),
    "VAE": ("variational autoencoder", "autoencoder variațional"),
    "VaR": ("Value-at-Risk", "valoarea expusă la risc (Value-at-Risk)"),
    "ViT": ("vision transformer", "transformer pentru imagini (vision transformer)"),
    "VLM": ("vision-language model", "model vizual-lingvistic"),
    "XAI": ("explainable artificial intelligence", "inteligență artificială explicabilă"),
}
UNITS = set()
IGNORE = {"AI", "EU", "UE", "AdamW", "BEATS", "BIZ", "NEURAL", "EuroSAT", "IEEE-CIS", "LLM-ES", "LN", "MSA", "OpenCLIP", "QwenLM", "VL", "ViT-B", "PICBE", "NAJEF", "ESWA", "JSTARS", "IEEE", "ICML", "ICLR", "AAA", "II", "OK", "TS", "EU", "UE", "AI", "NLP", "BET", "MSCA", "AIDA", "IDA"}

REFS = [
    "Abnar, S.\\ \\& Zuidema, W.\\ (2020). Quantifying attention flow in transformers. \\textit{Proceedings of the 58th Annual Meeting of the "
    "Association for Computational Linguistics}, 4190--4197. \\href{https://doi.org/10.18653/v1/2020.acl-main.385}{doi:10.18653/v1/2020.acl-main.385}",
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}. Springer. "
    "\\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Diaconu, D.\\ \\& Pele, D.\\,T.\\ (2026). Explainable AI in financial fraud detection: Evidence from the IEEE-CIS fraud dataset. "
    "\\textit{Proceedings of the International Conference on Business Excellence}, 20(1), 118--132. \\href{https://doi.org/10.2478/picbe-2026-0015}{doi:10.2478/picbe-2026-0015}",
    "Dosovitskiy, A.\\ et al.\\ (2021). An image is worth 16x16 words: Transformers for image recognition at scale. "
    "\\textit{International Conference on Learning Representations}. \\href{https://arxiv.org/abs/2010.11929}{arXiv:2010.11929}",
    "Găman, Ș., Tak, R., Jheng, S.-L.\\ \\& Pele, D.\\,T.\\ (2026). ESG-aware asset selection and portfolio optimization: A comparative study "
    "of multi-criteria decision-making methods. \\textit{Proceedings of the International Conference on Business Excellence}, 20(1), 133--147. "
    "\\href{https://doi.org/10.2478/picbe-2026-0016}{doi:10.2478/picbe-2026-0016}",
    "Helber, P., Bischke, B., Dengel, A.\\ \\& Borth, D.\\ (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land "
    "cover classification. \\textit{IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 12(7), 2217--2226. "
    "\\href{https://doi.org/10.1109/JSTARS.2019.2918242}{doi:10.1109/JSTARS.2019.2918242}",
    "Jheng, S.-L., Pele, D.\\,T., Tak, R.\\ \\& Găman, Ș.\\ (2026). A multimodal vision-language framework for financial anomaly detection. "
    "\\textit{Proceedings of the International Conference on Business Excellence}, 20(1), 191--201. \\href{https://doi.org/10.2478/picbe-2026-0020}{doi:10.2478/picbe-2026-0020}",
    "Liu, H., Li, C., Wu, Q.\\ \\& Lee, Y.\\,J.\\ (2023). Visual instruction tuning. \\textit{Advances in Neural Information Processing Systems}, 36. "
    "\\href{https://arxiv.org/abs/2304.08485}{arXiv:2304.08485}",
    "Oord, A.\\ van den, Li, Y.\\ \\& Vinyals, O.\\ (2018). Representation learning with contrastive predictive coding. "
    "\\href{https://arxiv.org/abs/1807.03748}{arXiv:1807.03748}",
    "Pele, D.\\,T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A.\\,T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S.\\ \\& "
    "Härdle, W.\\,K.\\ (2026). In the beginning was the Word: LLM-VaR and LLM-ES. \\textit{Expert Systems with Applications}, 295, 128676. "
    "\\href{https://doi.org/10.1016/j.eswa.2025.128676}{doi:10.1016/j.eswa.2025.128676}",
    "Radford, A.\\ et al.\\ (2021). Learning transferable visual models from natural language supervision. \\textit{Proceedings of the 38th "
    "International Conference on Machine Learning}, 8748--8763. \\href{https://arxiv.org/abs/2103.00020}{arXiv:2103.00020}",
    "Wang, P.\\ et al.\\ (2024). Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution. "
    "\\href{https://arxiv.org/abs/2409.12191}{arXiv:2409.12191}",
]


def it(*xs, sep="2pt"):
    def one(x):
        if isinstance(x, tuple):
            return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                    "\n".join("      \\item " + (s[0].upper() + s[1:] if s[0].islower() else s) for s in x[1:] if s) + "\n    \\end{itemize}")
        return hier(x)
    return (f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{{sep}}}\n" + "\n".join("  \\item " + one(x) for x in xs if x) + "\n\\end{itemize}")


def note(*xs):
    return "\n\\par\\vspace{0.5mm}{\\scriptsize " + items(*xs, opt="\\setlength{\\itemsep}{0pt}", nest=False) + "}"


NAME = {"en": {"AnnualCrop": "annual crop", "Forest": "forest", "HerbaceousVegetation": "herbaceous vegetation", "Highway": "highway",
               "Industrial": "industrial", "Pasture": "pasture", "PermanentCrop": "permanent crop", "Residential": "residential",
               "River": "river", "SeaLake": "sea or lake"},
        "ro": {"AnnualCrop": "culturi anuale", "Forest": "pădure", "HerbaceousVegetation": "vegetație ierboasă", "Highway": "autostradă",
               "Industrial": "zonă industrială", "Pasture": "pășune", "PermanentCrop": "culturi permanente", "Residential": "zonă rezidențială",
               "River": "rîu", "SeaLake": "mare sau lac"}}


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L14_numbers.json"), lang)
    N = V.N
    D = Deck(14, "Multimodal and vision-language models; synthesis of the course" if lang == "en" else
             "Modele multimodale și vizual-lingvistice; sinteza cursului", V, lang, heights=H)
    t = D.t
    ro = lang == "ro"
    NB = "notebook"
    nm = NAME[lang]
    zb = N["zs_best"]
    zbk, zbt = zb.split("_")
    zs_best_name = f"CLIP ViT-{zbk[0]}/{zbk[1:]}"
    R = N["regime"]

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Explain how a vision transformer turns an image into tokens and where its attention goes, on real satellite images.",
          "Să explicați cum transformă un transformer pentru imagini o imagine în tokeni și unde se îndreaptă atenția sa, pe imagini satelitare reale."),
        t("Write the contrastive loss of contrastive language--image pre-training (CLIP), use it as a zero-shot classifier and compare it with a convolutional neural network (CNN) trained on labelled images.",
          "Să scrieți funcția de pierdere a preantrenării contrastive limbaj--imagine (CLIP), să o folosiți ca clasificator zero-shot și să o comparați cu o rețea neuronală convoluțională (CNN) antrenată pe imagini etichetate."),
        t("Describe how vision-language models connect an image encoder to a language model, and test what they can read in charts and tables.",
          "Să descrieți cum leagă modelele vizual-lingvistice un codificator de imagini de un model de limbaj și să testați ce pot citi din grafice și tabele."),
        t("Synthesise the course: choose a model for a business problem, from logistic regression to foundation models, with the right evaluation.",
          "Să sintetizați cursul: să alegeți modelul pentru o problemă de afaceri, de la regresia logistică la modelele fundație, cu evaluarea potrivită."),
        sep="4pt"))
    abbreviation_frames(D, ABBR)

    # ------------------------------------------------------------------ 1. why multimodal
    D.section(t("Why multimodal models", "De ce modele multimodale"))
    D.frame(t("Images, documents and charts as business data", "Imagini, documente și grafice ca date economice"), it(
        (t("\\textbf{Satellite and aerial images}", "\\textbf{Imagini satelitare și aeriene}"),
         t("crop monitoring for agricultural credit and insurance; land use for environmental, social and governance (ESG) reporting",
           "monitorizarea culturilor pentru creditul și asigurarea agricolă; utilizarea terenului pentru raportările de mediu, sociale și de guvernanță (ESG)")),
        (t("\\textbf{Documents}", "\\textbf{Documente}"),
         t("scanned contracts, invoices, annual reports: text, tables and layout together", "contracte scanate, facturi, rapoarte anuale: text, tabele și așezare în pagină, împreună")),
        (t("\\textbf{Charts}", "\\textbf{Grafice}"),
         t("analysts read price charts; can a model read them, and should it, when the numbers exist?", "analiștii citesc graficele de preț; le poate citi un model și are rost, cînd cifrele există?")),
        (t("\\textbf{This lecture}", "\\textbf{Acest curs}"),
         t("CLIP and a vision-language model (VLM), measured on real images", "CLIP și un model vizual-lingvistic (VLM), evaluate pe imagini reale")),
        sep="3pt"), "small", cite="(Radford et al., 2021; Bishop \\& Bishop, 2024)")
    D.frame(t("Business application: land use for credit, insurance and ESG", "Aplicația economică: utilizarea terenului pentru credit, asigurări și ESG"), it(
        (t("\\textbf{Agricultural credit}", "\\textbf{Creditul agricol}"),
         t("a bank checks that the financed plots are cropland and stay cropland over the season", "o bancă verifică dacă parcelele finanțate sînt teren arabil și rămîn așa pe tot sezonul")),
        (t("\\textbf{Insurance}", "\\textbf{Asigurări}"),
         t("flood and drought claims are compared with images before and after the event", "daunele pentru inundații și secetă se compară cu imagini de dinainte și de după eveniment")),
        (t("\\textbf{ESG reporting}", "\\textbf{Raportarea ESG}"),
         t("forest cover and industrial sites around the assets of a portfolio", "suprafața împădurită și zonele industriale din jurul activelor unui portofoliu")),
        (t("\\textbf{The data problem}", "\\textbf{Problema datelor}"),
         t("images are free (Sentinel-2 every few days), labels are expensive: hence zero-shot and few-shot models",
           "imaginile sînt gratuite (Sentinel-2, la cîteva zile), etichetele sînt scumpe: de aici modelele zero-shot și cu puține exemple")),
        sep="3pt"), "small", cite="(Helber et al., 2019)")
    d = N["data"]
    ct = N["charts_task"]
    D.frame(t("Real data only: satellite images, market charts, yield-curve tables", "Numai date reale: imagini satelitare, grafice de piață, tabele ale curbei randamentelor"),
            table("@{}p{0.3\\textwidth}p{0.3\\textwidth}p{0.33\\textwidth}@{}", [t("Data", "Date"), t("Size", "Dimensiune"), t("Task", "Sarcina")],
                  [[t("EuroSAT: Sentinel-2 images of Europe, red, green, blue (RGB) channels", "EuroSAT: imagini Sentinel-2 ale Europei, canalele roșu, verde, albastru (RGB)"),
                    t(f"{d['n']:,} images, {d['px']}$\\times${d['px']} pixels, {d['n_classes']} classes".replace(",", "\\,"),
                      f"{d['n']:,} de imagini, {d['px']}$\\times${d['px']} pixeli, {d['n_classes']} clase".replace(",", ".")),
                    t("land-use classification: zero-shot, few-shot, full training", "clasificarea utilizării terenului: zero-shot, cu puține exemple, antrenare completă")],
                   [t("S\\&P 500 daily closes (Yahoo Finance), rendered as chart images", "Prețurile zilnice S\\&P 500 (Yahoo Finance), desenate ca imagini"),
                    t(f"{ct['n']:,} windows of {ct['window']} days".replace(",", "\\,"), f"{ct['n']:,} de ferestre de {ct['window']} de zile".replace(",", ".")),
                    t("market regime from the picture vs from the numbers", "regimul pieței din imagine față de cel din cifre")],
                   [t("Euro area AAA yield curve, European Central Bank (ECB)", "Curba randamentelor AAA din zona euro, Banca Centrală Europeană (BCE)"),
                    t(f"{N['table']['n']} tables rendered as images", f"{N['table']['n']} de tabele desenate ca imagini"),
                    t("can a VLM read a number from a table?", "poate un VLM să citească un număr dintr-un tabel?")]]) +
            note(t("EuroSAT split 80/20 stratified (seed @{seed}): @{data.n_train|,d} training and @{data.n_test|,d} test images.",
                   "EuroSAT împărțit 80/20 stratificat (valoarea generatorului aleator: @{seed}): @{data.n_train|,d} de imagini de antrenare și @{data.n_test|,d} de test.")),
            "small", cite="(Helber et al., 2019)")

    # ------------------------------------------------------------------ 2. ViT
    D.section(t("Vision transformers", "Transformere pentru imagini"))
    D.concept(t("A vision transformer reads an image as a sequence of patches", "Un transformer pentru imagini citește imaginea ca o secvență de petice"),
              t("cut the image into 16$\\times$16 patches, flatten each patch into a vector, and treat the vectors as words.",
                "imaginea se taie în petice de 16$\\times$16 pixeli, fiecare petic devine un vector, iar vectorii sînt tratați ca niște cuvinte."),
              "\\[ \\mathbf z_0=\\big[\\mathbf x_{\\text{cls}};\\ \\mathbf E\\mathbf p_1;\\ \\dots;\\ \\mathbf E\\mathbf p_{196}\\big]+\\mathbf E_{\\text{pos}},\\quad \\mathbf p_j\\in\\R^{16\\cdot16\\cdot3} \\]"
              "\\[ \\mathbf z_\\ell=\\mathbf z_{\\ell-1}+\\mathrm{MSA}(\\mathrm{LN}(\\mathbf z_{\\ell-1})),\\ \\ \\mathbf z_\\ell\\leftarrow\\mathbf z_\\ell+\\mathrm{MLP}(\\mathrm{LN}(\\mathbf z_\\ell)) \\]" +
              "\\where{" + t("$224\\times224$ image = $14\\times14$ = 196 patches + one class token = @{vit.n_tokens} tokens; $\\mathbf E$ = linear patch embedding (width @{vit.width}); "
                             "multi-head self-attention with @{vit.heads} heads, @{vit.layers} layers; layer normalisation; multilayer perceptron (MLP).",
                             "imagine $224\\times224$ = $14\\times14$ = 196 de petice + un token de clasă = @{vit.n_tokens} de tokeni; $\\mathbf E$ = proiecția liniară a peticelor (lățime @{vit.width}); "
                             "atenție multi-cap cu @{vit.heads} capete, @{vit.layers} straturi; normalizarea pe strat; perceptron multistrat (MLP).") + "}",
              it(t("\\textbf{Vision transformer (ViT)}: no convolution; locality must be learned from data.",
                   "\\textbf{Transformerul pentru imagini (ViT)}: fără convoluție; localitatea trebuie învățată din date."),
                 t("\\textbf{Hence the need for huge pre-training}: CLIP saw 400 million image--text pairs.",
                   "\\textbf{De aici nevoia de preantrenare uriașă}: CLIP a văzut 400 de milioane de perechi imagine--text.")),
              t("with few labelled images, a pre-trained ViT beats a ViT trained from scratch by a wide margin.",
                "cu puține imagini etichetate, un ViT preantrenat întrece cu mult un ViT antrenat de la zero."),
              cite="(Dosovitskiy et al., 2021)", tryit=NB + " \\S1")
    D.figframe(t("Patches and attention on real EuroSAT images", "Petice și atenție pe imagini EuroSAT reale"),
               t("top: the 14$\\times$14 patch grid; bottom: where the class token looks, averaged over all layers.",
                 "sus: grila de 14$\\times$14 petice; jos: unde privește tokenul de clasă, mediat pe toate straturile."),
               None, "vit_patches_attention",
               bullets=[t("Attention rollout: $\\tilde{\\mathbf A}=\\prod_{\\ell}(\\frac12\\bar{\\mathbf A}_\\ell+\\frac12\\mathbf I)$, row of the class token; $\\bar{\\mathbf A}_\\ell$ = attention of layer $\\ell$ averaged over heads.",
                          "Propagarea atenției: $\\tilde{\\mathbf A}=\\prod_{\\ell}(\\frac12\\bar{\\mathbf A}_\\ell+\\frac12\\mathbf I)$, rîndul tokenului de clasă; $\\bar{\\mathbf A}_\\ell$ = atenția stratului $\\ell$, mediată pe capete.")],
               height="0.68\\textheight", cite="(Abnar \\& Zuidema, 2020)", tryit=NB + " \\S1")

    # ------------------------------------------------------------------ 3. CLIP
    D.section(t("CLIP: images and text in one space", "CLIP: imagini și text în același spațiu"))
    D.concept(t("CLIP learns by matching images with their captions", "CLIP învață potrivind imaginile cu descrierile lor"),
              t("in a batch of $n$ image--caption pairs, each image must find its own caption, and vice versa.",
                "într-un lot de $n$ perechi imagine--descriere, fiecare imagine își caută descrierea, și invers."),
              "\\[ s_{ij}=\\frac{\\mathbf u_i^{\\top}\\mathbf v_j}{\\tau},\\quad \\mathbf u_i=\\frac{f(I_i)}{\\lVert f(I_i)\\rVert},\\ \\mathbf v_j=\\frac{g(T_j)}{\\lVert g(T_j)\\rVert} \\]"
              "\\[ \\loss=-\\frac{1}{2n}\\sum_{i=1}^{n}\\Big[\\log\\frac{e^{s_{ii}}}{\\sum_{j}e^{s_{ij}}}+\\log\\frac{e^{s_{ii}}}{\\sum_{j}e^{s_{ji}}}\\Big] \\]" +
              "\\where{" + t("$f$ = image encoder (ViT); $g$ = text encoder (transformer); $\\tau$ = learned temperature; information noise-contrastive estimation (InfoNCE) loss in both directions.",
                             "$f$ = codificatorul de imagini (ViT); $g$ = codificatorul de text; $\\tau$ = temperatura învățată; pierderea contrastivă (InfoNCE), în ambele direcții.") + "}",
              it(t("\\textbf{ViT-B/32}: @{clip.B32.params_m|.1f} million parameters; \\textbf{ViT-B/16}: @{clip.B16.params_m|.1f} million.",
                   "\\textbf{ViT-B/32}: @{clip.B32.params_m|.1f} milioane de parametri; \\textbf{ViT-B/16}: @{clip.B16.params_m|.1f} milioane."),
                 t("\\textbf{No class labels} were used in training: only captions from the web.",
                   "\\textbf{Nicio etichetă de clasă} nu a fost folosită la antrenare: doar descrieri de pe internet.")),
              t("one embedding space for images and text enables search, zero-shot labels and retrieval for retrieval-augmented generation (RAG).",
                "un singur spațiu de embedding pentru imagini și text permite căutarea, etichetarea zero-shot și regăsirea pentru generarea augmentată prin regăsire (RAG)."),
              cite="(Radford et al., 2021; Oord et al., 2018)", wl="0.58", wr="0.39")
    D.concept(t("Zero-shot classification: the class names become the classifier", "Clasificarea zero-shot: numele claselor devin clasificatorul"),
              t("embed one sentence per class, embed the image, and pick the closest sentence.", "se calculează embedding-ul unei propoziții pentru fiecare clasă și al imaginii, apoi se alege propoziția cea mai apropiată."),
              "\\[ \\mathbf w_c=\\mathrm{norm}\\Big(\\frac{1}{M}\\sum_{m=1}^{M}g\\big(\\text{template}_m(c)\\big)\\Big) \\]\\[ \\hat c(I)=\\argmax_c\\ \\mathbf u(I)^{\\top}\\mathbf w_c \\]" +
              "\\where{" + t("template$_m$ = sentence such as ``a satellite photo of \\{class\\}.''; $M$ = 1 or 5 templates (prompt ensemble); class descriptions as in Radford et al.\\ (2021).",
                             "template$_m$ = o propoziție ca „a satellite photo of \\{class\\}.”; $M$ = 1 sau 5 șabloane (ansamblu de prompturi); descrierile claselor ca la Radford et al.\\ (2021).") + "}",
              it(t("\\textbf{Best zero-shot}: " + zs_best_name + ", @{zs." + zb + "|.1f}\\,\\% on @{data.n_test|,d} test images, without one labelled example.",
                   "\\textbf{Cel mai bun zero-shot}: " + zs_best_name + ", @{zs." + zb + "|.1f}\\,\\% pe @{data.n_test|,d} de imagini de test, fără niciun exemplu etichetat."),
                 t("\\textbf{Chance} with 10 classes: 10\\,\\%.", "\\textbf{Nivelul hazardului} cu 10 clase: 10\\,\\%.")),
              t("the weight vectors are written in English, so a domain expert can edit the classifier without retraining.",
                "vectorii de ponderi sînt scriși în limba engleză, deci un expert al domeniului poate modifica clasificatorul fără reantrenare."),
              cite="(Radford et al., 2021)", tryit=NB + " \\S2")
    D.figframe(t("The words in the prompt change zero-shot accuracy", "Cuvintele din prompt schimbă acuratețea zero-shot"),
               t("telling CLIP that the image is a satellite photo is worth more than a bigger model.",
                 "a-i spune lui CLIP că imaginea este o fotografie satelitară valorează mai mult decît un model mai mare."),
               None, "zeroshot_prompts",
               bullets=[t("ViT-B/16: folder name @{zs.B16_name|.1f}\\,\\%, ``a satellite photo of'' @{zs.B16_sat|.1f}\\,\\%, 5-template ensemble @{zs.B16_ens|.1f}\\,\\%; "
                          "ViT-B/32 ensemble @{zs.B32_ens|.1f}\\,\\%.",
                          "ViT-B/16: numele dosarului @{zs.B16_name|.1f}\\,\\%, „a satellite photo of” @{zs.B16_sat|.1f}\\,\\%, ansamblul de 5 șabloane @{zs.B16_ens|.1f}\\,\\%; "
                          "ViT-B/32 ansamblu @{zs.B32_ens|.1f}\\,\\%.")],
               job=t("prompt wording is a hyper-parameter: select it on validation images, then freeze it.",
                     "formularea promptului este un hiperparametru: alegeți-o pe imagini de validare, apoi fixați-o."),
               height="0.5\\textheight", tryit=NB + " \\S2")
    D.figframe(t("Zero-shot predictions on eight random test images", "Predicțiile zero-shot pe opt imagini de test alese aleator"),
               t("green: correct; red: wrong; the percentage is the softmax probability of the chosen class.",
                 "verde: corect; roșu: greșit; procentul este probabilitatea softmax a clasei alese."),
               None, "zeroshot_examples",
               bullets=[t("@{examples.n_right} of @{examples.n} correct in this sample.", "@{examples.n_right} din @{examples.n} corecte în acest eșantion.")],
               height="0.72\\textheight", tryit=NB + " \\S2")
    tc = N["zs_top_conf"]
    D.figframe(t("Where zero-shot fails: similar-looking land uses", "Unde greșește zero-shot: utilizări ale terenului care arată la fel"),
               t("rows: true class; columns: predicted class; each row sums to 100\\,\\%.", "rîndurile: clasa reală; coloanele: clasa prezisă; fiecare rînd însumează 100\\,\\%."),
               None, "zeroshot_confusion",
               bullets=[t(f"Best class: {nm[N['zs_best_class']]} (@{{zs_class.{N['zs_best_class']}|.1f}}\\,\\%); worst: {nm[N['zs_worst']]} (@{{zs_class.{N['zs_worst']}|.1f}}\\,\\%); "
                          f"largest confusion: {tc['n']} of {tc['n_true']} {nm[tc['true']]} images called {nm[tc['pred']]}.",
                          f"Cea mai bună clasă: {nm[N['zs_best_class']]} (@{{zs_class.{N['zs_best_class']}|.1f}}\\,\\%); cea mai slabă: {nm[N['zs_worst']]} (@{{zs_class.{N['zs_worst']}|.1f}}\\,\\%); "
                          f"cea mai mare confuzie: {tc['n']} din {tc['n_true']} imagini „{nm[tc['true']]}” clasificate drept „{nm[tc['pred']]}”.")],
               height="0.6\\textheight", tryit=NB + " \\S2")
    se = N["search"]
    D.figframe(t("Text-to-image search with the same embeddings", "Căutarea imaginilor după text cu aceleași embedding-uri"),
               t("rank all test images by their similarity to a sentence; a bank can find flooded fields or new buildings without labels.",
                 "ordonăm toate imaginile de test după similaritatea cu o propoziție; o bancă poate găsi terenuri inundate sau clădiri noi fără etichete."),
               "\\[ \\mathrm{P@}k(c)=\\frac{1}{k}\\sum_{j\\in\\mathrm{top}_k(\\mathbf w_c)}\\mathbb 1\\{y_j=c\\},\\quad \\mathrm{top}_k(\\mathbf w_c)=\\text{" +
               t("the $k$ images with the largest", "cele $k$ imagini cu cel mai mare") + "}\\ \\mathbf u(I_j)^{\\top}\\mathbf w_c \\]", "text_search",
               bullets=[t(f"Mean precision of the top 100: @{{search.mean|.1f}}\\,\\%; best {nm[se['best']]} (@{{search.per_class.{se['best']}|.1f}}\\,\\%), "
                          f"worst {nm[se['worst']]} (@{{search.per_class.{se['worst']}|.1f}}\\,\\%).",
                          f"Precizia medie a primelor 100: @{{search.mean|.1f}}\\,\\%; cea mai bună: {nm[se['best']]} (@{{search.per_class.{se['best']}|.1f}}\\,\\%), "
                          f"cea mai slabă: {nm[se['worst']]} (@{{search.per_class.{se['worst']}|.1f}}\\,\\%).")],
               job=t("search first, label the hits, then train a probe: the cheapest way to build a labelled image set.",
                     "căutați întîi, etichetați rezultatele, apoi antrenați un clasificator: cea mai ieftină cale spre un set de imagini etichetate."),
               height="0.48\\textheight", tryit=NB + " \\S2")
    pk = N["probe_beats_zs_k"]
    D.concept(t("A linear probe: CLIP features plus logistic regression", "Un clasificator liniar pe trăsăturile CLIP: embedding plus regresie logistică"),
              t("keep CLIP frozen and train only a logistic regression on its image embeddings.", "CLIP rămîne fix și se antrenează doar o regresie logistică pe embedding-urile imaginilor."),
              "\\[ \\hat p(c\\mid I)=\\softmax\\big(\\mathbf W\\mathbf u(I)+\\mathbf b\\big)_c \\]\\[ \\min_{\\mathbf W,\\mathbf b}\\ -\\sum_{i}\\log\\hat p(c_i\\mid I_i)+\\frac{1}{2C}\\lVert\\mathbf W\\rVert_F^2 \\]" +
              "\\where{" + t("$\\mathbf u(I)$ = normalised ViT-B/16 embedding (@{clip.B16.dim} dimensions); $C=10$ fixed a priori; $k$ labelled images per class, 5 seeds.",
                             "$\\mathbf u(I)$ = embedding-ul normalizat ViT-B/16 (@{clip.B16.dim} de dimensiuni); $C=10$, fixat a priori; $k$ imagini etichetate pe clasă, 5 rulări.") + "}",
              it(t("\\textbf{All training images}: @{probe_full|.1f}\\,\\% (ViT-B/32: @{probe_full_b32|.1f}\\,\\%).",
                   "\\textbf{Toate imaginile de antrenare}: @{probe_full|.1f}\\,\\% (ViT-B/32: @{probe_full_b32|.1f}\\,\\%)."),
                 t(("\\textbf{Beats zero-shot} already with one labelled image per class: @{fewshot.probe.1.mean|.1f}\\,\\%." if pk == 1 else
                    f"\\textbf{{Beats zero-shot}} from {pk} labelled images per class.") if pk else "\\textbf{Never beats zero-shot} in the range tested.",
                   ("\\textbf{Depășește zero-shot} deja cu o singură imagine etichetată pe clasă: @{fewshot.probe.1.mean|.1f}\\,\\%." if pk == 1 else
                    f"\\textbf{{Depășește zero-shot}} de la {pk} imagini etichetate pe clasă.") if pk else "\\textbf{Nu depășește zero-shot} în intervalul testat.")),
              t("a probe trains in seconds on a laptop: the first thing to try when you have a few hundred labels.",
                "un astfel de clasificator se antrenează în cîteva secunde pe laptop: primul lucru de încercat cînd aveți cîteva sute de etichete."),
              tryit=NB + " \\S3")
    D.figframe(t("Pre-trained features and a CNN trained from scratch, by number of labels", "Trăsăturile preantrenate și o CNN antrenată de la zero, după numărul de etichete"),
               t("one labelled image per class already beats zero-shot; from about 64 per class a small network trained from scratch catches up.",
                 "o imagine etichetată pe clasă depășește deja zero-shot; de la circa 64 pe clasă, o rețea mică antrenată de la zero o ajunge din urmă."),
               None, "fewshot_eurosat",
               bullets=[t("16 images per class: probe @{fewshot.probe.16.mean|.1f}\\,\\%, convolutional neural network (CNN) @{fewshot.cnn.16.mean|.1f}\\,\\%; "
                          "128 per class: @{fewshot.probe.128.mean|.1f}\\,\\% vs @{fewshot.cnn.128.mean|.1f}\\,\\%; all @{data.n_train|,d} images: CNN @{cnn_full.mean|.1f}\\,\\%, probe @{probe_full|.1f}\\,\\%.",
                          "16 imagini pe clasă: clasificatorul pe embedding @{fewshot.probe.16.mean|.1f}\\,\\%, CNN @{fewshot.cnn.16.mean|.1f}\\,\\%; "
                          "128 pe clasă: @{fewshot.probe.128.mean|.1f}\\,\\% față de @{fewshot.cnn.128.mean|.1f}\\,\\%; toate cele @{data.n_train|,d} de imagini: CNN @{cnn_full.mean|.1f}\\,\\%, clasificatorul pe embedding @{probe_full|.1f}\\,\\%.")],
               job=t("with abundant labels a small CNN catches up; with scarce labels, start from a foundation model.",
                     "cu multe etichete, o CNN mică recuperează; cu puține etichete, porniți de la un model fundație."),
               height="0.46\\textheight", tryit=NB + " \\S3")
    cs = N["clip"]
    D.frame(t("Size and speed of the three image models", "Dimensiunea și viteza celor trei modele de imagini"),
            table("@{}lrrr@{}", [t("Model", "Model"), t("Parameters", "Parametri"), t("ms per image", "ms pe imagine"), t("Test accuracy (\\%)", "Acuratețea pe test (\\%)")],
                  [[t("CNN from scratch (all labels)", "CNN de la zero (toate etichetele)"), fmt(N["cnn_full"]["params_k"], ".1f", lang) + t(" thousand", " mii"), "--", fmt(N["cnn_full"]["mean"], ".1f", lang)],
                   [t("CLIP ViT-B/32 + probe", "CLIP ViT-B/32 + clasificator"), fmt(cs["B32"]["params_m"], ".1f", lang) + t(" million", " mil."), fmt(cs["B32"]["ms_per_image"], ".2f", lang), fmt(N["probe_full_b32"], ".1f", lang)],
                   [t("CLIP ViT-B/16 + probe", "CLIP ViT-B/16 + clasificator"), fmt(cs["B16"]["params_m"], ".1f", lang) + t(" million", " mil."), fmt(cs["B16"]["ms_per_image"], ".2f", lang), fmt(N["probe_full"], ".1f", lang)],
                   [t("Qwen2-VL-2B (chart questions)", "Qwen2-VL-2B (întrebări despre grafice)"), fmt(N["cost"]["vl_params_b"], ".2f", lang) + t(" billion", " mld."), fmt(N["vl"]["ms_per_image"], ",.0f", lang), "--"]]) +
            note(t("Time measured on a laptop (Apple silicon graphics processor) for embedding one image in batches; the VLM time includes the question.",
                   "Timp măsurat pe un laptop (procesor grafic Apple silicon), pentru calculul embedding-ului unei imagini în loturi; timpul VLM include întrebarea."),
                 t("A bigger patch (32 instead of 16 pixels) means 4 times fewer tokens: faster, slightly less accurate.",
                   "Un petic mai mare (32 de pixeli în loc de 16) înseamnă de 4 ori mai puțini tokeni: mai rapid, puțin mai puțin precis.")),
            "small", cite=t("(own computation)", "(calcul propriu)"))

    # ------------------------------------------------------------------ 4. VLMs
    D.section(t("Vision-language models: reading charts and tables", "Modele vizual-lingvistice: citirea graficelor și a tabelelor"))
    D.concept(t("A VLM plugs an image encoder into a language model", "Un VLM conectează un codificator de imagini la un model de limbaj"),
              t("image patches are projected into the word-embedding space of a large language model (LLM), which reads them as extra tokens.",
                "peticele imaginii sînt proiectate în spațiul embedding-urilor de cuvinte al unui model lingvistic mare (LLM), care le citește ca tokeni suplimentari."),
              "\\[ \\mathbf H_v=\\mathbf W_p\\,f_{\\text{ViT}}(I) \\]\\[ p(y_t\\mid y_{<t},I,q)=\\mathrm{LLM}_{\\btheta}\\big([\\mathbf H_v;\\ \\mathbf e(q);\\ \\mathbf e(y_{<t})]\\big) \\]" +
              "\\where{" + t("$f_{\\text{ViT}}$ = vision encoder; $\\mathbf W_p$ = projector (an MLP); $q$ = question; trained first on image--caption pairs, then on visual instructions.",
                             "$f_{\\text{ViT}}$ = codificatorul vizual; $\\mathbf W_p$ = proiectorul (un MLP); $q$ = întrebarea; antrenat întîi pe perechi imagine--descriere, apoi pe instrucțiuni vizuale.") + "}",
              it(t("\\textbf{Large language and vision assistant (LLaVA)} introduced this recipe; Qwen2-VL adds images of any resolution.",
                   "\\textbf{Asistentul lingvistic și vizual (LLaVA)} a introdus această rețetă; Qwen2-VL adaugă imagini de orice rezoluție."),
                 t("\\textbf{Our model}: Qwen2-VL-2B-Instruct, @{cost.vl_params_b|.2f} billion parameters, run locally.",
                   "\\textbf{Modelul nostru}: Qwen2-VL-2B-Instruct, @{cost.vl_params_b|.2f} miliarde de parametri, rulat local.")),
              t("the answer is generated text: constrain it (yes/no, a number) and score it like a classifier.",
                "răspunsul este text generat: restrîngeți-l (da/nu, un număr) și evaluați-l ca pe un clasificator."),
              cite="(Liu et al., 2023; Wang et al., 2024)")
    D.concept(t("Market regimes as a visual recognition task", "Regimurile pieței ca problemă de recunoaștere vizuală"),
              t("draw the last 120 days of the S\\&P 500 as an image and ask whether the market is in stress, as a chart analyst would.",
                "desenăm ultimele 120 de zile ale S\\&P 500 ca imagine și întrebăm dacă piața este în criză, cum ar face un analist tehnic."),
              "\\[ S_t=\\mathbb 1\\Big\\{\\sigma^{(20)}_t>Q_{0.9}\\big(\\sigma^{(20)}_{t-750:t-1}\\big)\\Big\\} \\]\\[ \\sigma^{(20)}_t=\\sqrt{252}\\,\\mathrm{sd}(r_{t-19:t}) \\]" +
              "\\where{" + t("$S_t$ = stress label at the end of the window: 20-day volatility above its rolling 90th percentile of the previous three years (past data only); "
                             "windows every @{charts_task.stride} days; train until 2018, test @{charts_task.n_test} windows from 2019.",
                             "$S_t$ = eticheta de criză la sfîrșitul ferestrei: volatilitatea pe 20 de zile peste percentila 90 glisantă din ultimii trei ani (doar date din trecut); "
                             "ferestre la fiecare @{charts_task.stride} zile; antrenare pînă în 2018, test pe @{charts_task.n_test} de ferestre din 2019.") + "}",
              it((t("\\textbf{Four readers of the same windows}", "\\textbf{Patru metode pe aceleași ferestre}"),
                  t("numbers + logistic regression (returns, volatility, drawdown)", "cifre + regresie logistică (randamente, volatilitate, scădere de la maxim)"),
                  t("CLIP image embedding + logistic regression", "embedding-ul CLIP al imaginii + regresie logistică"),
                  t("CLIP zero-shot: ``calm, steadily rising'' vs ``volatile, sharp crash''", "CLIP zero-shot: „calm, în creștere constantă” față de „volatil, prăbușire bruscă”"),
                  t("Qwen2-VL zero-shot: ``is the market turbulent?'' yes or no", "Qwen2-VL zero-shot: „este piața agitată?” da sau nu"))),
              t("a rolling-quantile label avoids hand-labelling and look-ahead.", "o etichetă bazată pe cuantile glisante evită etichetarea manuală și folosirea viitorului."),
              cite="(Jheng, Pele, Tak \\& Găman, 2026)", tryit=NB + " \\S4")
    D.figframe(t("Detecting market stress from charts and from numbers", "Detectarea crizelor de piață din grafice și din cifre"),
               t("left: a calm and a stress window as the models see them; right: receiver operating characteristic (ROC) curves on the test years.",
                 "stînga: o fereastră calmă și una de criză, așa cum le văd modelele; dreapta: curbele caracteristicii de operare a receptorului (ROC) pe anii de test."),
               None, "chart_regime",
               bullets=[t("Area under the ROC curve (AUC): numbers @{regime.numeric.auc|.3f}, CLIP probe @{regime.clip_probe.auc|.3f}, CLIP zero-shot @{regime.clip_zs.auc|.3f}, "
                          "Qwen2-VL zero-shot @{regime.vlm_zs.auc|.3f}; stress share in test @{charts_task.base_test|.1f}\\,\\%.",
                          "Aria de sub curba ROC (AUC): cifre @{regime.numeric.auc|.3f}, clasificator pe CLIP @{regime.clip_probe.auc|.3f}, CLIP zero-shot @{regime.clip_zs.auc|.3f}, "
                          "Qwen2-VL zero-shot @{regime.vlm_zs.auc|.3f}; ponderea crizelor în test @{charts_task.base_test|.1f}\\,\\%.")],
               height="0.55\\textheight", tryit=NB + " \\S4")
    rows = []
    for k, nmk in (("numeric", t("Numbers + logistic", "Cifre + logistică")), ("clip_probe", t("CLIP image + logistic", "Imagine CLIP + logistică")),
                   ("clip_zs", t("CLIP zero-shot", "CLIP zero-shot")), ("vlm_zs", t("Qwen2-VL zero-shot", "Qwen2-VL zero-shot"))):
        rows.append([nmk, heat(R[k]["auc"], 0.5, max(R[x]["auc"] for x in R), good_low=False, shown=fmt(R[k]["auc"], ".3f", lang)),
                     heat(R[k]["ap"], N["charts_task"]["base_test"] / 100, max(R[x]["ap"] for x in R), good_low=False, shown=fmt(R[k]["ap"], ".3f", lang)),
                     t("yes", "da") if k in ("numeric", "clip_probe") else t("no", "nu")])
    best_r = max(R, key=lambda k: R[k]["auc"])
    D.frame(t("Stress detection: discrimination and precision on the test years", "Detectarea crizelor: capacitatea de separare și precizia pe anii de test"),
            cols(table("@{}lrrl@{}", [t("Method", "Metoda"), "AUC", t("Avg.\\ prec.", "Prec.\\ medie"), t("Labels", "Etichete")], rows, size="scriptsize"),
                 it(t("\\textbf{Best}: " + {"numeric": "numbers", "clip_probe": "CLIP probe", "clip_zs": "CLIP zero-shot", "vlm_zs": "Qwen2-VL"}[best_r] + ".",
                      "\\textbf{Cel mai bun}: " + {"numeric": "cifrele", "clip_probe": "clasificatorul pe CLIP", "clip_zs": "CLIP zero-shot", "vlm_zs": "Qwen2-VL"}[best_r] + "."),
                    t("\\textbf{Average precision} of a random score = stress share, @{charts_task.base_test|.1f}\\,\\%.",
                      "\\textbf{Precizia medie} a unui scor aleator = ponderea crizelor, @{charts_task.base_test|.1f}\\,\\%."),
                    t("\\textbf{Lesson}: when the numbers exist, a picture adds a lossy step; images earn their place when the numbers are missing (scans, screenshots, satellite).",
                      "\\textbf{Lecția}: cînd cifrele există, imaginea adaugă un pas cu pierdere de informație; imaginile își au locul cînd cifrele lipsesc (scanări, capturi de ecran, imagini satelitare)."),
                    sep="4pt"), "0.5", "0.47"),
            "small", cite=t("(own computation, after Jheng, Pele, Tak \\& Găman, 2026)", "(calcul propriu, după Jheng, Pele, Tak \\& Găman, 2026)"), tryit=NB + " \\S4")
    vl = N["vl"]
    tb = N["table"]
    D.frame(t("What a 2-billion-parameter VLM can read: trends and table cells", "Ce poate citi un VLM de 2 miliarde de parametri: tendințe și celule de tabel"), it(
        (t("\\textbf{Trend question} on @{vl.trend_n} test charts: ``did the price end higher or lower?''",
           "\\textbf{Întrebarea despre tendință}, pe @{vl.trend_n} de grafice de test: „prețul a încheiat mai sus sau mai jos?”"),
         t("accuracy @{vl.trend_acc|.1f}\\,\\% at the 0.5 threshold, below always answering the majority (@{vl.trend_base|.1f}\\,\\%)",
           "acuratețe @{vl.trend_acc|.1f}\\,\\% la pragul de 0,5, sub răspunsul constant cu clasa majoritară (@{vl.trend_base|.1f}\\,\\%)"),
         t("yet AUC @{vl.trend_auc|.3f}: the ranking is right, the threshold is biased --- recalibrate on validation charts",
           "totuși AUC @{vl.trend_auc|.3f}: ordonarea este corectă, pragul este deplasat --- recalibrați pe grafice de validare")),
        (t("\\textbf{Table question} on @{table.n} ECB yield-curve tables: ``what is the 10-year yield?''",
           "\\textbf{Întrebarea despre tabel}, pe @{table.n} de tabele ale curbei randamentelor BCE: „care este randamentul la 10 ani?”"),
         t("exact answers @{table.exact|.1f}\\,\\%; mean absolute error @{table.mae|.3f} percentage points; numbers parsed @{table.parsed|.1f}\\,\\%",
           "răspunsuri exacte @{table.exact|.1f}\\,\\%; eroarea absolută medie @{table.mae|.3f} puncte procentuale; numere extrase @{table.parsed|.1f}\\,\\%")),
        (t("\\textbf{Why it matters}", "\\textbf{De ce contează}"),
         t("reading a cell is optical character recognition (OCR) plus locating the column: errors are silent and plausible",
           "citirea unei celule înseamnă recunoaștere optică a caracterelor (OCR) plus găsirea coloanei: erorile sînt tăcute și plauzibile"),
         t("for documents with known layout, extract the numbers with a parser and use the VLM only where layout varies",
           "pentru documente cu structură cunoscută, extrageți cifrele cu un program de citire și folosiți VLM doar unde structura variază")),
        sep="3pt"), "small", cite=t("(own computation; Wang et al., 2024)", "(calcul propriu; Wang et al., 2024)"), tryit=NB + " \\S5")
    D.frame(t("A document pipeline in a bank: where the VLM fits", "Fluxul documentelor într-o bancă: unde se potrivește un VLM"), it(
        (t("\\textbf{1. Capture}", "\\textbf{1. Preluarea}"), t("scanned page or electronic file; known templates go to a rule-based parser", "pagină scanată sau fișier electronic; șabloanele cunoscute merg la un program de citire bazat pe reguli")),
        (t("\\textbf{2. Read}", "\\textbf{2. Citirea}"), t("OCR and layout for text and tables; a VLM for free-form pages, charts and stamps",
                                                               "OCR și așezarea în pagină pentru text și tabele; un VLM pentru pagini libere, grafice și ștampile")),
        (t("\\textbf{3. Check}", "\\textbf{3. Verificarea}"), t("totals must add up, dates must be valid, values within ranges; a failed check goes to a person",
                                                                    "totalurile trebuie să se adune, datele să fie valide, valorile în intervale plauzibile; o verificare eșuată merge la un om")),
        (t("\\textbf{4. Measure}", "\\textbf{4. Măsurarea}"), t("exact-match rate per field on a labelled sample, as for the yield tables (@{table.exact|.1f}\\,\\% here)",
                                                                    "rata de potrivire exactă pe fiecare cîmp, pe un eșantion etichetat, ca la tabelele de randamente (aici @{table.exact|.1f}\\,\\%)")),
        sep="3pt"), "small")
    D.frame(t("Capabilities and limitations of multimodal models", "Capacitățile și limitele modelelor multimodale"),
            cols(it((t("\\textbf{Can}", "\\textbf{Pot}"),
                     t("label images with no training data (@{zs." + zb + "|.1f}\\,\\% on EuroSAT)", "eticheta imagini fără date de antrenare (@{zs." + zb + "|.1f}\\,\\% pe EuroSAT)"),
                     t("give features that need few labels (@{fewshot.probe.16.mean|.1f}\\,\\% with 16 per class)", "oferi trăsături care au nevoie de puține etichete (@{fewshot.probe.16.mean|.1f}\\,\\% cu 16 pe clasă)"),
                     t("search images by text, and text by images", "căuta imagini după text și text după imagini"),
                     t("answer simple questions about charts and documents", "răspunde la întrebări simple despre grafice și documente")), sep="3pt"),
                 it((t("\\textbf{Cannot (reliably)}", "\\textbf{Nu pot (în mod fiabil)}"),
                     t("replace the numbers behind a chart", "înlocui cifrele din spatele unui grafic"),
                     t("read every cell of a table correctly (@{table.exact|.1f}\\,\\% exact here)", "citi corect fiecare celulă a unui tabel (aici, @{table.exact|.1f}\\,\\% exact)"),
                     t("explain faithfully why they answered", "explica fidel de ce au răspuns astfel"),
                     t("separate look-alike classes without examples", "separa fără exemple clasele care arată la fel")), sep="3pt"), "0.49", "0.48") +
            note(t("Controls: a labelled test set, constrained answers, human review of low-confidence cases, logs of every prompt and image.",
                   "Măsuri de control: un set de test etichetat, răspunsuri restrînse, verificarea umană a cazurilor cu încredere scăzută, jurnalizarea fiecărui prompt și a fiecărei imagini.")),
            "small")

    # ------------------------------------------------------------------ 5. synthesis
    D.section(t("Synthesis of the course", "Sinteza cursului"))
    D.frame(t("Methods of the course by data type", "Metodele cursului după tipul de date"),
            table("@{}p{0.2\\textwidth}p{0.36\\textwidth}p{0.37\\textwidth}@{}", [t("Data", "Date"), t("Models", "Modele"), t("Lectures and business cases", "Cursuri și aplicații")],
                  [[t("Tables", "Tabele"), t("logistic regression, MLP, regularisation", "regresie logistică, MLP, regularizare"), t("1--3: credit scoring, fraud", "1--3: scoring de credit, fraudă")],
                   [t("Images", "Imagini"), t("CNN, transfer learning, ViT, CLIP", "CNN, transferul învățării, ViT, CLIP"), t("4, 14: land use, documents", "4, 14: utilizarea terenului, documente")],
                   [t("Sequences", "Secvențe"), t("long short-term memory (LSTM), basis-expansion networks, foundation models", "rețele cu memorie pe termen scurt și lung (LSTM), rețele cu expansiune în baze, modele fundație"), t("5, 13: prices, load, Value-at-Risk (VaR)", "5, 13: prețuri, consum, valoarea expusă la risc (VaR)")],
                   [t("Text", "Text"), t("embeddings, transformers, LLM, retrieval-augmented generation (RAG)", "embedding, transformere, LLM, generare augmentată prin regăsire (RAG)"), t("6, 7, 12: sentiment, central banks", "6, 7, 12: sentiment, bănci centrale")],
                   [t("Generation", "Generare"), t("variational autoencoder (VAE), generative adversarial network (GAN), diffusion", "autoencoder variațional (VAE), rețea generativă adversarială (GAN), difuzie"), t("8: synthetic data, anomalies", "8: date sintetice, anomalii")],
                   [t("Graphs", "Grafuri"), t("graph neural network (GNN)", "rețea neuronală pe grafuri (GNN)"), t("9: transaction networks", "9: rețele de tranzacții")],
                   [t("Decisions", "Decizii"), t("reinforcement learning (RL)", "învățare prin recompensă (RL)"), t("10: trading, hedging, batteries", "10: tranzacționare, acoperirea riscului, baterii")],
                   [t("Trust", "Încredere"), t("explainable artificial intelligence (XAI): Shapley additive explanations (SHAP), gradients", "inteligență artificială explicabilă (XAI): explicații aditive Shapley (SHAP), gradienți"), t("11: credit audits", "11: auditul modelelor de credit")]],
                  size="scriptsize"), "small")
    D.frame(t("Choosing a model for a business problem", "Alegerea modelului pentru o problemă de afaceri"), it(
        (t("\\textbf{1. Start with the decision and the cost of errors}", "\\textbf{1. Porniți de la decizie și de la costul erorilor}"),
         t("a missed fraud and a false alarm do not cost the same; choose the metric before the model", "o fraudă ratată și o alarmă falsă nu costă la fel; alegeți metrica înaintea modelului")),
        (t("\\textbf{2. Fit the benchmark first}", "\\textbf{2. Estimați întîi reperul}"),
         t("logistic regression, the naive forecast, a Lasso autoregression, a volatility model, word counts with a linear classifier: minutes, interpretable", "regresie logistică, prognoza naivă, o autoregresie Lasso, un model de volatilitate, frecvențe de cuvinte cu un clasificator liniar: cîteva minute, ușor de interpretat")),
        (t("\\textbf{3. Count your labels}", "\\textbf{3. Numărați etichetele}"),
         t("few labels: foundation model zero-shot or with a probe; many labels: a trained network", "puține etichete: model fundație zero-shot sau cu un clasificator pe embedding; multe etichete: o rețea antrenată")),
        (t("\\textbf{4. Match the architecture to the data}", "\\textbf{4. Potriviți arhitectura cu datele}"),
         t("tables MLP or boosting, images CNN or ViT, sequences recurrent or transformer, graphs GNN", "tabele: MLP sau boosting; imagini: CNN sau ViT; secvențe: rețele recurente sau transformer; grafuri: GNN")),
        (t("\\textbf{5. Prove it}", "\\textbf{5. Demonstrați}"),
         t("test once, several seeds, a significance test, calibration, an explanation for the regulator", "testare o singură dată, mai multe rulări, un test de semnificație, calibrare, o explicație pentru autoritatea de supraveghere")),
        sep="2pt"), "small")
    D.frame(t("Five evaluation rules that held in every lecture", "Cinci reguli de evaluare valabile în fiecare curs"), it(
        t("\\textbf{No leakage}: scalers, prompts and thresholds are chosen on training or validation data; time series are split in time.",
          "\\textbf{Fără scurgeri de informație}: scalările, prompturile și pragurile se aleg pe datele de antrenare sau de validare; seriile de timp se împart cronologic."),
        t("\\textbf{A benchmark next to every network}: here, a CNN trained from scratch next to CLIP, numbers next to chart images.",
          "\\textbf{Un reper lîngă fiecare rețea}: aici, o CNN antrenată de la zero lîngă CLIP, cifrele lîngă imaginile graficelor."),
        t("\\textbf{Seeds and intervals}: the few-shot results above are means over 5 seeds with their range.",
          "\\textbf{Rulări repetate și intervale}: rezultatele cu puține exemple de mai sus sînt medii pe 5 rulări, cu amplitudinea lor."),
        t("\\textbf{Metrics for rare events}: average precision and AUC for stress, not accuracy.",
          "\\textbf{Metrici pentru evenimente rare}: precizia medie și AUC pentru crize, nu acuratețea."),
        t("\\textbf{Real data, reproducible}: every number comes from a script and a public data set, with a Quantlet.",
          "\\textbf{Date reale, rezultate reproductibile}: fiecare cifră provine dintr-un program și dintr-un set de date public, cu un Quantlet."),
        sep="4pt"), "small")
    D.frame(t("Foundation models for images, text and time series", "Modelele fundație pentru imagini, text și serii de timp"), it(
        (t("\\textbf{The pattern}", "\\textbf{Tiparul}"), t("pre-train once on huge unlabelled data; use zero-shot, with a few examples, or with a small trained head",
                                                            "preantrenare o singură dată pe volume uriașe de date neetichetate; utilizare zero-shot, cu cîteva exemple sau cu un mic clasificator antrenat")),
        (t("\\textbf{Text} (Lecture 12)", "\\textbf{Text} (cursul 12)"), t("prompts and LLM embeddings for sentiment, RAG on ECB speeches, LLM-VaR",
                                                                            "prompturi și embedding LLM pentru sentiment, RAG pe discursurile BCE, LLM-VaR")),
        (t("\\textbf{Time series} (Lecture 13)", "\\textbf{Serii de timp} (cursul 13)"), t("Chronos-type models forecasting prices, load and VaR without training",
                                                                                            "modele de tip Chronos prognozează prețuri, consum și VaR fără antrenare")),
        (t("\\textbf{Images} (this lecture)", "\\textbf{Imagini} (acest curs)"), t("CLIP zero-shot @{zs." + zb + "|.1f}\\,\\%; probe on all labels @{probe_full|.1f}\\,\\%",
                                                                                   "CLIP zero-shot @{zs." + zb + "|.1f}\\,\\%; clasificator pe toate etichetele @{probe_full|.1f}\\,\\%")),
        (t("\\textbf{The common caveat}", "\\textbf{Avertismentul comun}"), t("pre-training data may contain your test set; evaluate on data the model cannot have seen",
                                                                             "datele de preantrenare pot conține setul dumneavoastră de test; evaluați pe date pe care modelul nu le-a putut vedea")),
        sep="3pt"), "small")
    D.frame(t("Ten formulas of the course (1/2): learning", "Zece formule ale cursului (1/2): învățarea"), table(
        "@{}p{0.27\\textwidth}p{0.47\\textwidth}p{0.18\\textwidth}@{}", [t("Idea", "Idee"), t("Formula", "Formula"), t("Lecture", "Cursul")],
        [[t("Logistic neuron", "Neuronul logistic"), "$\\hat p=\\sigma(\\mathbf w^{\\top}\\mathbf x+b)$", "1"],
         [t("Cross-entropy loss", "Pierderea entropie încrucișată"), "$\\loss=-\\sum_i[y_i\\log\\hat p_i+(1-y_i)\\log(1-\\hat p_i)]$", "1--2"],
         [t("Backpropagation", "Propagarea înapoi"), "$\\boldsymbol\\delta^{(\\ell)}=(\\mathbf W^{(\\ell+1)\\top}\\boldsymbol\\delta^{(\\ell+1)})\\odot f'(\\mathbf z^{(\\ell)})$", "2"],
         [t("Gradient step (Adam without moments)", "Pasul de gradient (Adam fără momente)"), "$\\btheta\\leftarrow\\btheta-\\eta\\nabla_{\\btheta}\\loss$", "2"],
         [t("Weight decay", "Penalizarea ponderilor"), "$\\loss+\\lambda\\lVert\\btheta\\rVert_2^2$", "3"]], size="footnotesize", stretch="1.4") +
        note(t("All symbols as defined in the lectures cited; $\\sigma$ = logistic function, $\\odot$ = element-wise product.",
               "Toate simbolurile, ca în cursurile citate; $\\sigma$ = funcția logistică, $\\odot$ = produsul element cu element.")), "small")
    D.frame(t("Ten formulas of the course (2/2): architectures", "Zece formule ale cursului (2/2): arhitecturi"), table(
        "@{}p{0.27\\textwidth}p{0.47\\textwidth}p{0.18\\textwidth}@{}", [t("Idea", "Idee"), t("Formula", "Formula"), t("Lecture", "Cursul")],
        [[t("Convolution", "Convoluția"), "$(\\mathbf K*\\mathbf X)_{ij}=\\sum_{u,v}K_{uv}X_{i+u,j+v}$", "4"],
         [t("Recurrent state", "Starea recurentă"), "$\\mathbf h_t=\\tanh(\\mathbf W\\mathbf h_{t-1}+\\mathbf U\\mathbf x_t+\\mathbf b)$", "5"],
         [t("Attention", "Atenția"), "$\\softmax(\\mathbf Q\\mathbf K^{\\top}/\\sqrt{d_k})\\mathbf V$", "7, 12--14"],
         [t("Graph convolution", "Convoluția pe graf"), "$\\mathbf H'=\\sigma(\\tilde{\\mathbf D}^{-1/2}\\tilde{\\mathbf A}\\tilde{\\mathbf D}^{-1/2}\\mathbf H\\mathbf W)$", "9"],
         [t("Contrastive image--text loss", "Pierderea contrastivă imagine--text"), "$-\\log\\frac{e^{s_{ii}}}{\\sum_j e^{s_{ij}}}$", "14"]], size="footnotesize", stretch="1.4") +
        note(t("$\\mathbf K$ = kernel; $\\tilde{\\mathbf A}$ = adjacency matrix with self-loops, $\\tilde{\\mathbf D}$ its degree matrix; $s_{ij}$ = scaled image--text similarity.",
               "$\\mathbf K$ = nucleul; $\\tilde{\\mathbf A}$ = matricea de adiacență cu bucle, $\\tilde{\\mathbf D}$ matricea gradelor; $s_{ij}$ = similaritatea scalată imagine--text.")), "small")
    D.frame(t("Responsible use: the same rules for every model of the course", "Utilizarea responsabilă: aceleași reguli pentru fiecare model din curs"), it(
        (t("\\textbf{Regulation}", "\\textbf{Reglementare}"),
         t("the European Union Artificial Intelligence Act (Regulation 2024/1689) lists credit scoring among high-risk uses: documentation, human oversight, data governance",
           "Regulamentul UE privind inteligența artificială (2024/1689) include scoringul de credit printre utilizările cu risc ridicat: documentație, supraveghere umană, guvernanța datelor")),
        (t("\\textbf{Documentation}", "\\textbf{Documentare}"),
         t("a model card: data, period, metric, benchmark, seeds, known failure cases (here: sea or lake read as highway)",
           "o fișă a modelului: date, perioadă, metrică, reper, rulări, cazuri cunoscute de eșec (aici: mare sau lac citit drept autostradă)")),
        (t("\\textbf{Monitoring}", "\\textbf{Monitorizare}"),
         t("re-test on new data: price regimes, new satellite seasons, new speech topics", "retestare pe date noi: regimuri de preț, noi sezoane satelitare, teme noi în discursuri")),
        (t("\\textbf{Privacy}", "\\textbf{Confidențialitate}"),
         t("local open models keep client documents and images in-house", "modelele deschise rulate local păstrează documentele și imaginile clienților în instituție")),
        sep="3pt"), "small")
    D.frame(t("What a good course project looks like", "Cum arată un proiect bun pentru acest curs"), cols(it(
        (t("\\textbf{Question and data}", "\\textbf{Întrebare și date}"), t("one business decision, one public data set, its period and licence",
                                                                          "o decizie de afaceri, un set de date public, perioada și licența lui")),
        (t("\\textbf{Benchmark first}", "\\textbf{Întîi reperul}"), t("a linear or naive model, reported next to every network", "un model liniar sau naiv, raportat lîngă fiecare rețea")),
        (t("\\textbf{Leakage check}", "\\textbf{Verificarea scurgerilor}"), t("time-ordered split, scalers fitted on training data, test used once",
                                                                                "împărțire în ordine cronologică, scalări estimate pe antrenare, testul folosit o singură dată")), sep="3pt"),
        it((t("\\textbf{Uncertainty}", "\\textbf{Incertitudine}"), t("several seeds, bootstrap intervals or a Diebold--Mariano test", "mai multe rulări, intervale bootstrap sau un test Diebold--Mariano")),
           (t("\\textbf{Explanation}", "\\textbf{Explicație}"), t("SHAP or gradients for the decision that matters", "SHAP sau gradienți pentru decizia care contează")),
           (t("\\textbf{Reproducible}", "\\textbf{Reproductibil}"), t("one script from raw data to every number and chart", "un singur program, de la datele brute la fiecare cifră și grafic")), sep="3pt"),
        "0.49", "0.48"), "small")
    D.frame(t("Open research problems for dissertations", "Probleme de cercetare deschise pentru lucrările de disertație"), cols(it(
        t("\\textbf{Calibration under drift}: conformal bands that stay valid when markets change regime.",
          "\\textbf{Calibrarea în prezența derivei}: benzi conformale care rămîn valide cînd piețele își schimbă regimul."),
        t("\\textbf{Leakage-proof evaluation} of foundation models trained on the open web.",
          "\\textbf{Evaluarea fără scurgeri de informație} a modelelor fundație antrenate pe internet."),
        t("\\textbf{Stable explanations}: do SHAP values of a credit model survive a retraining?",
          "\\textbf{Explicații stabile}: rezistă valorile SHAP ale unui model de credit la o reantrenare?"), sep="4pt"),
        it(t("\\textbf{Small models for small institutions}: when does a 2--3 billion model on a laptop suffice?",
             "\\textbf{Modele mici pentru instituții mici}: cînd ajunge un model de 2--3 miliarde de parametri pe un laptop?"),
           t("\\textbf{Multimodal risk signals}: satellite, text and prices in one early-warning model.",
             "\\textbf{Semnale de risc multimodale}: imagini satelitare, text și prețuri într-un singur model de avertizare timpurie."),
           t("\\textbf{Romanian data}: language models and forecasts for Romanian text, markets and energy.",
             "\\textbf{Date românești}: modele de limbaj și prognoze pentru textul, piețele și energia din România."), sep="4pt"), "0.49", "0.48"), "small")

    # ------------------------------------------------------------------ wrap-up
    D.parts.append("\\section*{" + t("Wrap-up", "Recapitulare") + "}\n")
    D.quiz_frame(t("zero-shot classifier", "clasificatorul zero-shot"),
                 t("A bank wants to flag flooded fields in satellite images but has no labelled flood image; what can it do with CLIP?",
                   "O bancă vrea să marcheze terenurile inundate în imagini satelitare, dar nu are nicio imagine etichetată; ce poate face cu CLIP?"),
                 t("Write class prompts (``a satellite photo of a flooded field'', ``\\ldots{} of a dry field''), embed them and classify by cosine similarity; then label a few hundred images to validate and train a probe.",
                   "Scrie prompturi pentru clase („a satellite photo of a flooded field”, „\\ldots{} of a dry field”), le calculează embedding-urile și clasifică după similaritatea cosinus; apoi etichetează cîteva sute de imagini pentru validare și pentru un clasificator pe embedding."))
    D.quiz_frame(t("pictures or numbers", "imagini sau cifre"),
                 t("Why did the numeric logistic model detect market stress at least as well as models reading the chart image?",
                   "De ce a detectat modelul logistic pe cifre crizele de piață cel puțin la fel de bine ca modelele care citesc imaginea graficului?"),
                 t("The image is drawn from the same numbers, with resolution lost; the numbers carry the volatility directly. Images help when no numbers exist.",
                   "Imaginea este desenată din aceleași cifre, cu pierdere de rezoluție; cifrele conțin direct volatilitatea. Imaginile ajută cînd nu există cifre."))
    D.quiz_frame(t("labels", "etichetele"),
                 t("With 16 labelled images per class, which do you choose: a CNN from scratch or a probe on CLIP embeddings?",
                   "Cu 16 imagini etichetate pe clasă, ce alegeți: o CNN de la zero sau un clasificator pe embedding-urile CLIP?"),
                 t("The probe: @{fewshot.probe.16.mean|.1f}\\,\\% against @{fewshot.cnn.16.mean|.1f}\\,\\% for the CNN on our data.",
                   "Clasificatorul pe embedding: @{fewshot.probe.16.mean|.1f}\\,\\%, față de @{fewshot.cnn.16.mean|.1f}\\,\\% pentru CNN, pe datele noastre."))
    D.frame(t("Wrap-up", "Recapitulare"), it(
        t("A ViT reads images as patch tokens; CLIP aligns them with text by a contrastive loss.",
          "Un ViT citește imaginile ca tokeni-petic; CLIP le aliniază cu textul printr-o pierdere contrastivă."),
        t("Zero-shot CLIP labels satellite images at @{zs." + zb + "|.1f}\\,\\% with no labels; a probe reaches @{probe_full|.1f}\\,\\%.",
          "CLIP zero-shot etichetează imaginile satelitare cu @{zs." + zb + "|.1f}\\,\\%, fără etichete; un clasificator pe embedding ajunge la @{probe_full|.1f}\\,\\%."),
        t("VLMs answer questions about charts and tables, but must be scored like any classifier.",
          "Modelele VLM răspund la întrebări despre grafice și tabele, dar trebuie evaluate ca orice clasificator."),
        t("The course in one line: benchmark first, real data, no leakage, several seeds, a test, an explanation.",
          "Cursul într-un rînd: întîi reperul, date reale, fără scurgeri de informație, mai multe rulări, un test, o explicație."),
        sep="4pt"))
    D.frame(t("Research from our group", "Cercetările grupului nostru"), it(
        t("\\textbf{Multimodal vision-language anomaly detection} (Best Paper Award, International Conference on Business Excellence (ICBE) 2026): "
          "market indicators are projected onto multi-panel images, a VLM (GPT-4.1, generative pre-trained transformer) classifies the regime zero-shot with "
          "rolling-quantile labels and explains its choice; validated out of sample on the S\\&P 500 (Jheng, Pele, Tak \\& Găman, 2026, PICBE).",
          "\\textbf{Detectarea anomaliilor cu modele vizual-lingvistice} (premiul pentru cea mai bună lucrare, Conferința internațională Business Excellence (ICBE) 2026): "
          "indicatorii de piață sînt proiectați pe imagini cu mai multe panouri, iar un VLM (GPT-4.1, transformer generativ preantrenat) clasifică regimul zero-shot, cu etichete "
          "bazate pe cuantile glisante, și își explică alegerea; validare pe date noi pentru S\\&P 500 (Jheng, Pele, Tak \\& Găman, 2026, PICBE)."),
        t("\\textbf{Explainable fraud detection}: tabular models and graph networks combined with SHAP and graph ablation (Diaconu \\& Pele, 2026, PICBE) --- Case study 1.",
          "\\textbf{Detectarea explicabilă a fraudei}: modele tabulare și rețele pe grafuri, combinate cu SHAP și ablația grafului (Diaconu \\& Pele, 2026, PICBE) --- studiul de caz 1."),
        t("\\textbf{LLM-VaR and LLM-ES}: language models as zero-shot risk models (Pele et al., 2026, Expert Systems with Applications) --- Lectures 12 and 13.",
          "\\textbf{LLM-VaR și LLM-ES}: modelele de limbaj ca modele de risc zero-shot (Pele et al., 2026, Expert Systems with Applications) --- cursurile 12 și 13."),
        t("\\textbf{ESG-aware portfolios}: multi-criteria selection of assets with ESG scores (Găman, Tak, Jheng \\& Pele, 2026, PICBE).",
          "\\textbf{Portofolii care țin cont de ESG}: selecția activelor după mai multe criterii, cu scoruri ESG (Găman, Tak, Jheng \\& Pele, 2026, PICBE)."),
        sep="3pt") + note(t("Findings as stated in the papers' abstracts; numbers on the other slides come from our own data.",
                            "Rezultatele sînt preluate din rezumatele lucrărilor; cifrele de pe celelalte slide-uri provin din datele noastre.")),
        "footnotesize", cite=t("(Bucharest University of Economic Studies, ASE; Romanian Academy, Institute for Economic Forecasting, IPE)",
                               "(Academia de Studii Economice din București, ASE; Academia Română, Institutul de Prognoză Economică, IPE)"))
    D.frame(t("Further reading and resources", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{OpenCLIP} models and benchmarks: ", "\\textbf{OpenCLIP}, modele și evaluări: ") + "\\href{https://github.com/mlfoundations/open_clip}{github.com/mlfoundations/open\\_clip}.",
        t("\\textbf{Qwen2-VL}: ", "\\textbf{Qwen2-VL}: ") + "\\href{https://github.com/QwenLM/Qwen2-VL}{github.com/QwenLM/Qwen2-VL}.",
        t("\\textbf{Bishop \\& Bishop (2024)}, chapters 12 and 20: ", "\\textbf{Bishop \\& Bishop (2024)}, capitolele 12 și 20: ") + "\\href{https://www.bishopbook.com}{bishopbook.com}.", sep="4pt"),
        it(t("\\textbf{Quantinar}: Digital Transformation on Finance: ", "\\textbf{Quantinar}: Digital Transformation on Finance: ") +
           "\\href{https://quantinar.com/course/674/digital-transformation-on-finance}{quantinar.com}.",
           t("\\textbf{EuroSAT} data: ", "\\textbf{EuroSAT}, date: ") + "\\href{https://github.com/phelber/EuroSAT}{github.com/phelber/EuroSAT}.",
           t("\\textbf{This lecture}: notebook, seminar and Quantlets --- ", "\\textbf{Acest curs}: notebook, seminar și Quantlets --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.", sep="4pt"), "0.5", "0.47"))
    reference_frames(D, REFS, per_frame=7)
    D.mark_main_end()

    D.parts.append("\\section*{" + t("Appendix", "Anexă") + "}\n")
    D.frame(t("Appendix A: the contrastive loss as a classification problem", "Anexa A: pierderea contrastivă ca problemă de clasificare"), it(
        t("Row $i$ of $\\mathbf S=(s_{ij})$ is a softmax classifier over $n$ captions with the correct class $i$: the loss is cross-entropy with labels $0,\\dots,n-1$.",
          "Rîndul $i$ al matricei $\\mathbf S=(s_{ij})$ este un clasificator softmax peste $n$ descrieri, cu clasa corectă $i$: pierderea este entropia încrucișată cu etichetele $0,\\dots,n-1$."),
        t("A larger batch means more negatives and a harder task; CLIP used $n$ = 32,768.",
          "Un lot mai mare înseamnă mai multe exemple negative și o sarcină mai grea; CLIP a folosit $n$ = 32.768."),
        t("Zero-shot classification is the same computation with the $n$ captions replaced by the $C$ class prompts.",
          "Clasificarea zero-shot este același calcul, cu cele $n$ descrieri înlocuite de cele $C$ prompturi ale claselor."),
        sep="4pt"), cite="(Radford et al., 2021; Oord et al., 2018)")
    D.frame(t("Appendix B: implementation details", "Anexa B: detalii de implementare"), it(
        t("CLIP: openai/clip-vit-base-patch32 and -patch16 (Hugging Face transformers), images resized to 224 pixels; logistic regression from scikit-learn.",
          "CLIP: openai/clip-vit-base-patch32 și -patch16 (Hugging Face transformers), imagini redimensionate la 224 de pixeli; regresia logistică din scikit-learn."),
        t("CNN: three convolution blocks (32, 64, 128 channels), AdamW with a one-cycle schedule, flips and rotations as augmentation; @{cnn_full.params_k|.1f} thousand parameters.",
          "CNN: trei blocuri convoluționale (32, 64, 128 de canale), AdamW cu program one-cycle, oglindiri și rotații ca augmentare; @{cnn_full.params_k|.1f} mii de parametri."),
        t("VLM: Qwen/Qwen2-VL-2B-Instruct, float16; yes/no answers scored by first-token probabilities; numbers by greedy decoding.",
          "VLM: Qwen/Qwen2-VL-2B-Instruct, float16; răspunsurile da/nu evaluate prin probabilitățile primului token; numerele prin decodare lacomă."),
        t("Seed @{seed}; the full analysis ran in @{runtime_min|.1f} minutes on its last run (embeddings and answers cached).",
          "Valoarea generatorului aleator: @{seed}; analiza completă a durat @{runtime_min|.1f} minute la ultima rulare (embedding-urile și răspunsurile stocate)."),
        sep="4pt"))
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 14, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
