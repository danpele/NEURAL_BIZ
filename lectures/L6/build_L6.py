r"""
build_L6.py -- Lecture 6 "NLP foundations: text representation and word embeddings", English and Romanian.
Slide text only; every result is a placeholder @{key} filled from L6_numbers.json (analysis_L6.py).
Run:  python3 lectures/L6/build_L6.py   ("en" or "ro" builds one language; exit 0 only if every check passes)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, cols, fmt, hier, items, reference_frames, run_build, table)  # noqa: E402

ABBR = {
    "AI": ("artificial intelligence", "inteligență artificială"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "BERT": ("bidirectional encoder representations from transformers", "reprezentări bidirecționale ale codificatorului transformer (BERT)"),
    "BoW": ("bag of words", "sac de cuvinte (bag of words)"),
    "CBOW": ("continuous bag of words (word2vec variant)", "sac continuu de cuvinte (varianta word2vec)"),
    "ECB": ("European Central Bank", None),
    "BCE": (None, "Banca Centrală Europeană"),
    "ES": ("expected shortfall", "pierderea medie așteptată peste VaR (expected shortfall)"),
    "ESG": ("environmental, social and governance", "mediu, social și guvernanță"),
    "FinBERT": ("BERT model adapted to financial text", "model BERT adaptat textelor financiare"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "LLM": ("large language model", "model lingvistic mare"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "NLP": ("natural language processing", "prelucrarea limbajului natural"),
    "OOV": ("out of vocabulary", "în afara vocabularului"),
    "PCA": ("principal component analysis", "analiza componentelor principale"),
    "PMI": ("pointwise mutual information", "informația mutuală punctuală"),
    "SVD": ("singular value decomposition", "descompunerea în valori singulare"),
    "SGNS": ("skip-gram with negative sampling", "skip-gram cu eșantionare negativă"),
    "TF-IDF": ("term frequency--inverse document frequency", "frecvența termenului--frecvența inversă în documente"),
    "VaR": ("value-at-risk", "valoarea expusă la risc (value-at-risk)"),
    "CPU": ("central processing unit", "procesor central"),
}
IGNORE = {"BIZ", "NEURAL", "SOURCES", "EACL", "EUR", "LLM-ES", "ESWA", "JASIST", "EMNLP", "NIPS", "NeurIPS", "ACL", "TACL", "ICLR", "OK", "II", "CNN", "LSTM", "GPT", "JMLR", "PICBE", "URL"}
UNITS = set()

REFS = [
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}, ch.~12. Springer. \\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Bojanowski, P., Grave, E., Joulin, A.\\ \\& Mikolov, T.\\ (2017). Enriching word vectors with subword information. \\textit{TACL}, 5, 135--146. \\href{https://doi.org/10.1162/tacl_a_00051}{doi:10.1162/tacl\\_a\\_00051}",
    "Conda, A.\\,I., Găman, Ș., Bâg, R.\\,C., Mazurencu-Marinescu-Pele, M., Pele, D.\\,T.\\ \\& Härdle, W.\\,K.\\ (2026). BitMood: AI analysis of Bitcoin trends via Facebook emotions. \\textit{Digital Finance}, 8(3), 42. \\href{https://doi.org/10.1007/s42521-026-00205-3}{doi:10.1007/s42521-026-00205-3}",
    "Firth, J.\\,R.\\ (1957). A synopsis of linguistic theory 1930--1955. In \\textit{Studies in Linguistic Analysis}, 1--32. Blackwell.",
    "Joulin, A., Grave, E., Bojanowski, P.\\ \\& Mikolov, T.\\ (2017). Bag of tricks for efficient text classification. \\textit{EACL}, 427--431. \\href{https://doi.org/10.18653/v1/E17-2068}{doi:10.18653/v1/E17-2068}",
    "Levy, O.\\ \\& Goldberg, Y.\\ (2014). Neural word embedding as implicit matrix factorization. \\textit{NIPS}, 27, 2177--2185. \\href{https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization}{papers.nips.cc}",
    "Loughran, T.\\ \\& McDonald, B.\\ (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. \\textit{Journal of Finance}, 66(1), 35--65. \\href{https://doi.org/10.1111/j.1540-6261.2010.01625.x}{doi:10.1111/j.1540-6261.2010.01625.x}",
    "Malo, P., Sinha, A., Korhonen, P., Wallenius, J.\\ \\& Takala, P.\\ (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. \\textit{JASIST}, 65(4), 782--796. \\href{https://doi.org/10.1002/asi.23062}{doi:10.1002/asi.23062}",
    "Mikolov, T., Sutskever, I., Chen, K., Corrado, G.\\ \\& Dean, J.\\ (2013). Distributed representations of words and phrases and their compositionality. \\textit{NIPS}, 26, 3111--3119. \\href{https://arxiv.org/abs/1310.4546}{arXiv:1310.4546}",
    "Pele, D.\\,T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A.\\,T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S.\\ \\& Härdle, W.\\,K.\\ (2026). In the beginning was the Word: LLM-VaR and LLM-ES. \\textit{Expert Systems with Applications}, 295, 128676. \\href{https://doi.org/10.1016/j.eswa.2025.128676}{doi:10.1016/j.eswa.2025.128676}",
    "Pennington, J., Socher, R.\\ \\& Manning, C.\\,D.\\ (2014). GloVe: Global vectors for word representation. \\textit{EMNLP}, 1532--1543. \\href{https://doi.org/10.3115/v1/D14-1162}{doi:10.3115/v1/D14-1162}",
    "Salton, G.\\ \\& Buckley, C.\\ (1988). Term-weighting approaches in automatic text retrieval. \\textit{Information Processing \\& Management}, 24(5), 513--523. \\href{https://doi.org/10.1016/0306-4573(88)90021-0}{doi:10.1016/0306-4573(88)90021-0}",
    "Zipf, G.\\,K.\\ (1949). \\textit{Human Behavior and the Principle of Least Effort}. Addison-Wesley.",
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


def tt(s):
    return "\\texttt{" + s.replace("_", "\\_").replace("<", "\\textless{}").replace(">", "\\textgreater{}") + "}"


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L6_numbers.json"), lang)
    N = V.N
    D = Deck(6, "NLP foundations: text representation and word embeddings" if lang == "en" else
             "Bazele NLP: reprezentarea textului și reprezentarea vectorială a cuvintelor", V, lang, heights=H)
    t = D.t
    ro = lang == "ro"
    OWN = t("(own computation)", "(calcul propriu)")
    NB = "\\S"
    ECB = t("ECB", "BCE")
    M = {"majority": t("Majority class", "Clasa majoritară"), "bow_lr": t("BoW + logistic", "BoW + logistic"),
         "tfidf_lr": t("TF-IDF + logistic", "TF-IDF + logistic"), "tfidf_mlp": t("TF-IDF + MLP", "TF-IDF + MLP"),
         "w2v_lr": t("mean word2vec + logistic", "media word2vec + logistic"), "embbag": t("embedding bag, random start", "embedding bag, start aleator"),
         "embbag_w2v": t("embedding bag, word2vec start", "embedding bag, start word2vec")}

    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Turn financial text into numbers for natural language processing (NLP): tokens, vocabulary, bag of words and term frequency--inverse document frequency (TF-IDF).",
          "Să transformați textul financiar în numere pentru prelucrarea limbajului natural (NLP): unități lexicale, vocabular, sac de cuvinte și frecvența termenului--frecvența inversă în documente (TF-IDF)."),
        t("Write the objectives of word2vec and GloVe and train both on a real corpus of central-bank speeches.",
          "Să scrieți funcțiile-obiectiv ale word2vec și GloVe și să le antrenați pe un corpus real de discursuri ale băncii centrale."),
        t("Build sentence classifiers from sparse and dense representations, with honest benchmarks and tests.",
          "Să construiți clasificatoare de fraze din reprezentări rare și dense, cu modele de referință și teste corecte."),
        t("Measure financial sentiment and topics on news and tweets, and know when a linear model is enough.",
          "Să măsurați sentimentul și temele financiare în știri și tweet-uri și să știți cînd un model liniar este suficient."), sep="4pt"))
    abbreviation_frames(D, ABBR)

    D.frame(t("The data: three real text collections", "Datele: trei colecții reale de texte"),
            table("@{}p{0.23\\textwidth}p{0.33\\textwidth}p{0.17\\textwidth}p{0.18\\textwidth}@{}",
                  [t("Corpus", "Corpusul"), t("Content", "Conținut"), t("Size", "Dimensiune"), t("Use here", "Utilizare aici")],
                  [["Financial PhraseBank", t("sentences of financial news on Finnish listed companies, labelled positive / neutral / negative by 5--8 annotators",
                                              "fraze din știri financiare despre companii listate în Finlanda, etichetate pozitiv / neutru / negativ de 5--8 adnotatori"),
                    t("@{data.pb_n|,d} sentences", "@{data.pb_n|,d} de fraze"), t("sentiment", "sentiment")],
                   [t("Twitter financial news", "Twitter financial news"), t("finance tweets: bearish / bullish / neutral, and 20 topics",
                                                                            "tweet-uri financiare: pesimist / optimist / neutru și 20 de teme"),
                    t("@{data.tw_train|,d} + @{data.tw_valid|,d} (sentiment), @{data.tp_train|,d} + @{data.tp_valid|,d} (topic)",
                      "@{data.tw_train|,d} + @{data.tw_valid|,d} (sentiment), @{data.tp_train|,d} + @{data.tp_valid|,d} (teme)"), t("sentiment, topics", "sentiment, teme")],
                   [t("European Central Bank (ECB) speeches", "Discursurile Băncii Centrale Europene (BCE)"),
                    t("full texts of the Executive Board and Supervisory Board", "textele integrale ale membrilor Comitetului executiv și ai Consiliului de supraveghere"),
                    t("@{data.sp_n} speeches, @{corpus.n_tokens|,d} tokens", "@{data.sp_n} de discursuri, @{corpus.n_tokens|,d} de unități lexicale"),
                    t("our own embeddings", "propriii noștri vectori de cuvinte")]]) +
            note(t("Speeches from @{data.sp_first|date} to @{data.sp_last|date}; PhraseBank by Malo et al. (2014); tweets as published on Hugging Face (train and validation splits).",
                   "Discursuri din perioada @{data.sp_first|date} -- @{data.sp_last|date}; PhraseBank de Malo et al. (2014); tweet-urile așa cum sînt publicate pe Hugging Face (seturile de antrenare și de validare)."),
                 t("Every model is trained on the central processing unit (CPU) of a laptop with seed @{seed}; test sets are used once.",
                   "Fiecare model este antrenat pe procesorul central (CPU) al unui laptop, cu sămînța @{seed}; seturile de test sînt folosite o singură dată.")),
            "footnotesize", cite=t("(data/SOURCES.md)", "(data/SOURCES.md)"))

    # ------------------------------------------------------------------ 1. text as data
    D.section(t("Text as data", "Textul ca date"))
    D.frame(t("Tokenisation of a sentence", "Împărțirea unei fraze în unități lexicale"), it(
        t(("\\textbf{A real PhraseBank sentence}", "``" + V.fmt("example.sentence") + "''", t("label given by the annotators: ", "eticheta dată de adnotatori: ") + "\\textbf{" + V.fmt("example.label") + "}"),
          ("\\textbf{O frază reală din PhraseBank}", "``" + V.fmt("example.sentence") + "''", "eticheta dată de adnotatori: \\textbf{" + {"positive": "pozitiv", "neutral": "neutru", "negative": "negativ"}[N["example"]["label"]] + "}")),
        t(("\\textbf{Our tokeniser}", "lower case, words and numbers; every number becomes " + tt("<num>") + "; links and @-mentions removed",
           "first tokens: " + tt(N["example"]["tokens"].replace(" | ", " ")) + " (@{example.n_tokens} in total)"),
          ("\\textbf{Tokenizatorul nostru}", "litere mici, cuvinte și numere; fiecare număr devine " + tt("<num>") + "; legăturile și mențiunile cu @ sînt eliminate",
           "primele unități: " + tt(N["example"]["tokens"].replace(" | ", " ")) + " (@{example.n_tokens} în total)")),
        t(("\\textbf{Choices that change results}", "lower-casing merges ``Bank'' and ``bank''; stemming merges ``rose'', ``rising''",
           "removing stop words deletes ``not'', ``down'', ``up'': harmful for sentiment"),
          ("\\textbf{Alegeri care schimbă rezultatele}", "trecerea la litere mici unește ``Bank'' și ``bank''; reducerea la rădăcină unește ``rose'' și ``rising''",
           "eliminarea cuvintelor de legătură șterge ``not'', ``down'', ``up'': dăunător pentru sentiment")), sep="4pt")
        + D.bottom(t("write the preprocessing down and keep it identical in training and production; a different tokeniser is a different model.",
                     "documentați preprocesarea și păstrați-o identică în antrenare și în producție; un alt tokenizator înseamnă un alt model."), tryit=NB + "1"),
        "footnotesize", cite=t("(Malo et al., 2014)", "(Malo et al., 2014)"))

    D.figframe(t("Word frequencies and Zipf's law", "Frecvențele cuvintelor și legea lui Zipf"),
               t("in @{data.sp_n} " + ("" if not ro else "de ") + "speeches of the European Central Bank, frequency falls as a power of rank.",
                 "în @{data.sp_n} de discursuri ale Băncii Centrale Europene, frecvența scade ca o putere a rangului."),
               "\\[ f(r)\\approx C\\,r^{-a}\\quad\\Longleftrightarrow\\quad\\ln f(r)=\\ln C-a\\ln r \\]",
               "zipf",
               bullets=[t("@{corpus.n_tokens|,d} tokens, @{corpus.n_types|,d} distinct words; estimated slope @{corpus.zipf_slope|.3f} (ranks 10 to 10,000); the 10 most frequent words make @{corpus.top10_share_pct|.1f}\\,\\% of all tokens.",
                          "@{corpus.n_tokens|,d} de unități lexicale, @{corpus.n_types|,d} de cuvinte distincte; panta estimată @{corpus.zipf_slope|.3f} (rangurile 10 -- 10.000); cele mai frecvente 10 cuvinte reprezintă @{corpus.top10_share_pct|.1f}\\,\\% din total."),
                        t("@{corpus.hapax_pct|.1f}\\,\\% of the distinct words occur once: no model can learn them from this corpus alone.",
                          "@{corpus.hapax_pct|.1f}\\,\\% dintre cuvintele distincte apar o singură dată: niciun model nu le poate învăța numai din acest corpus.")],
               height="0.36\\textheight", cite=t("(Zipf, 1949; own computation)", "(Zipf, 1949; calcul propriu)"))

    # ------------------------------------------------------------------ 2. sparse
    D.section(t("Sparse representations: bag of words and TF-IDF", "Reprezentări rare: sacul de cuvinte și TF-IDF"))
    D.concept(t("Bag of words: a document is a vector of counts", "Sacul de cuvinte: un document este un vector de frecvențe"),
              t("ignore the order, keep how many times each vocabulary word occurs.", "ignorăm ordinea și păstrăm de cîte ori apare fiecare cuvînt din vocabular."),
              "\\[ \\mathbf x_i=(c_{i1},\\dots,c_{iV})^{\\top} \\]\\vspace{-2mm}\\[ c_{ij}=\\#\\{\\text{" + t("occurrences of word", "apariții ale cuvîntului") + " } j \\text{ " + t("in document", "în documentul") + " } i\\} \\]"
              + t("\\where{$V$ vocabulary size (words seen at least twice in the training part); the document-term matrix $\\mathbf C\\in\\mathbb N^{n\\times V}$ is almost all zeros}",
                  "\\where{$V$ dimensiunea vocabularului (cuvintele întîlnite de cel puțin două ori în partea de antrenare); matricea documente--termeni $\\mathbf C\\in\\mathbb N^{n\\times V}$ este aproape în întregime zero}"),
              it(t("Bag of words (BoW) + multinomial logistic regression is the standard text baseline.", "Sacul de cuvinte (BoW) + regresia logistică multinomială este modelul de referință standard pentru text."),
                 t("$n$-grams add local order: ``profit rose'' and ``profit fell'' become features.", "$n$-gramele adaugă ordinea locală: ``profit rose'' și ``profit fell'' devin trăsături."),
                 t("Vocabulary is fitted on the training sentences only.", "Vocabularul este construit numai din frazele de antrenare.")),
              t("BoW models are fast, cheap and explainable word by word: auditors like them.",
                "modelele BoW sînt rapide, ieftine și explicabile cuvînt cu cuvînt: auditorilor le plac."),
              cite=t("(Salton \\& Buckley, 1988)", "(Salton \\& Buckley, 1988)"))

    D.concept(t("TF-IDF: lower weights for words that appear in every document", "TF-IDF: ponderi mai mici pentru cuvintele care apar în toate documentele"),
              t("a word is informative if it is frequent in this document and rare in the others.",
                "un cuvînt este informativ dacă este frecvent în acest document și rar în celelalte."),
              "\\[ w_{ij}=\\underbrace{(1+\\ln c_{ij})}_{\\text{tf}}\\cdot\\underbrace{\\Big(\\ln\\frac{1+n}{1+n_j}+1\\Big)}_{\\text{idf}},\\qquad \\tilde{\\mathbf x}_i=\\frac{\\mathbf w_i}{\\|\\mathbf w_i\\|_2} \\]"
              + t("\\where{$c_{ij}>0$ count of term $j$ in document $i$ ($w_{ij}=0$ if $c_{ij}=0$); $n$ documents; $n_j$ documents containing $j$; sublinear tf and smoothed idf, as in scikit-learn}",
                  "\\where{$c_{ij}>0$ frecvența termenului $j$ în documentul $i$ ($w_{ij}=0$ dacă $c_{ij}=0$); $n$ documente; $n_j$ documente care conțin $j$; tf sublinear și idf netezit, ca în scikit-learn}"),
              it(t("Term frequency--inverse document frequency (TF-IDF) on unigrams and bigrams, then logistic regression with balanced class weights.",
                   "Frecvența termenului--frecvența inversă în documente (TF-IDF) pe unigrame și bigrame, apoi regresie logistică cu ponderi echilibrate ale claselor."),
                 t("Words like ``the'', ``company'', " + tt("<num>") + " get a low idf; ``plunged'' a high one.",
                   "Cuvinte ca ``the'', ``company'', " + tt("<num>") + " primesc un idf mic; ``plunged'' unul mare."),
                 t("The $\\ell_2$ normalisation makes long and short sentences comparable.", "Normalizarea $\\ell_2$ face comparabile frazele lungi și cele scurte.")),
              t("TF-IDF + logistic regression is the model any NLP vendor must beat on your own documents.",
                "TF-IDF + regresie logistică este modelul pe care orice furnizor de NLP trebuie să-l depășească pe documentele dumneavoastră."),
              cite=t("(Salton \\& Buckley, 1988)", "(Salton \\& Buckley, 1988)"))

    D.figframe(t("The linear model learns a finance sentiment dictionary", "Modelul liniar învață un dicționar al sentimentului financiar"),
               t("the difference between the positive and negative coefficients of each word, logistic regression on BoW, PhraseBank.",
                 "diferența dintre coeficienții pozitiv și negativ ai fiecărui cuvînt, regresie logistică pe BoW, PhraseBank."),
               "\\[ P(y=k\\mid\\mathbf x)=\\frac{\\exp(\\boldsymbol\\beta_k^{\\top}\\mathbf x+\\alpha_k)}{\\sum_{l}\\exp(\\boldsymbol\\beta_l^{\\top}\\mathbf x+\\alpha_l)} \\]",
               "top_words",
               bullets=[t("Most positive: @{words.pos}; most negative: @{words.neg}.", "Cele mai pozitive: @{words.pos}; cele mai negative: @{words.neg}."),
                        t("General-language dictionaries misread finance words (``liability'', ``tax''): learn the weights on finance text.",
                          "Dicționarele de limbaj general interpretează greșit cuvintele financiare (``liability'', ``tax''): învățați ponderile pe texte financiare.")],
               height="0.42\\textheight", cite=t("(Loughran \\& McDonald, 2011; own computation)", "(Loughran \\& McDonald, 2011; calcul propriu)"))

    # ------------------------------------------------------------------ 3. dense
    D.section(t("Dense representations: word embeddings", "Reprezentări dense: vectorii de cuvinte (embeddings)"))
    D.concept(t("The distributional hypothesis", "Ipoteza distribuțională"),
              t("words used in similar contexts get similar vectors; similarity is the cosine of the angle.",
                "cuvintele folosite în contexte asemănătoare primesc vectori asemănători; asemănarea este cosinusul unghiului."),
              "\\[ \\mathbf e_w=\\mathbf E^{\\top}\\mathbf o_w\\in\\R^{d},\\qquad \\cos(w,v)=\\frac{\\mathbf e_w^{\\top}\\mathbf e_v}{\\|\\mathbf e_w\\|\\,\\|\\mathbf e_v\\|} \\]"
              + t("\\where{$\\mathbf o_w\\in\\{0,1\\}^{V}$ one-hot vector of word $w$; $\\mathbf E\\in\\R^{V\\times d}$ embedding matrix, $d=@{w2v.dim}\\ll V=@{w2v.V|,d}$; a lookup, not a product, in code}",
                  "\\where{$\\mathbf o_w\\in\\{0,1\\}^{V}$ vectorul one-hot al cuvîntului $w$; $\\mathbf E\\in\\R^{V\\times d}$ matricea vectorilor, $d=@{w2v.dim}\\ll V=@{w2v.V|,d}$; în cod este o căutare, nu un produs}"),
              it(t("One-hot vectors are orthogonal: ``inflation'' and ``prices'' have cosine 0.", "Vectorii one-hot sînt ortogonali: ``inflation'' și ``prices'' au cosinusul 0."),
                 t("In our own embeddings: cos(inflation, prices) = @{cos.inflation_prices|.2f}, cos(inflation, climate) = @{cos.inflation_climate|.2f}, cos(banks, lending) = @{cos.banks_lending|.2f}.",
                   "În vectorii noștri: cos(inflation, prices) = @{cos.inflation_prices|.2f}, cos(inflation, climate) = @{cos.inflation_climate|.2f}, cos(banks, lending) = @{cos.banks_lending|.2f}.")),
              t("dense vectors let a model trained on ``profit rose'' generalise to ``earnings increased''.",
                "vectorii denși permit unui model antrenat pe ``profit rose'' să generalizeze la ``earnings increased''."),
              cite=t("(Firth, 1957; Bishop \\& Bishop, 2024, sec.~12.2)", "(Firth, 1957; Bishop \\& Bishop, 2024, secț.~12.2)"))

    D.concept(t("word2vec skip-gram: predict the context from the word", "word2vec skip-gram: prezicem contextul din cuvînt"),
              t("each word in a window around the centre word is predicted by a softmax over the vocabulary.",
                "fiecare cuvînt din fereastra din jurul cuvîntului central este prezis printr-un softmax peste vocabular."),
              "\\[ \\max_{\\mathbf U,\\mathbf V}\\;\\frac1T\\sum_{t=1}^{T}\\sum_{0<|j|\\le c}\\log p(w_{t+j}\\mid w_t) \\]\\vspace{-2mm}\\[ p(o\\mid w)=\\frac{\\exp(\\mathbf u_o^{\\top}\\mathbf v_w)}{\\sum_{k=1}^{V}\\exp(\\mathbf u_k^{\\top}\\mathbf v_w)} \\]"
              + t("\\where{$\\mathbf v_w$ input (centre) vector; $\\mathbf u_o$ output (context) vector; $c$ window radius; the denominator costs $O(V)$ per pair}",
                  "\\where{$\\mathbf v_w$ vectorul de intrare (al cuvîntului central); $\\mathbf u_o$ vectorul de ieșire (al contextului); $c$ raza ferestrei; numitorul costă $O(V)$ pentru fiecare pereche}"),
              it(t("Continuous bag of words (CBOW) is the mirror image: predict the centre from the averaged context.",
                   "Sacul continuu de cuvinte (CBOW) este varianta în oglindă: prezice cuvîntul central din media contextului."),
                 t("The network is linear with one hidden layer; the embeddings are its first weight matrix.",
                   "Rețea liniară cu un strat ascuns; vectorii de cuvinte sînt prima ei matrice de ponderi."),
                 t("Our corpus: @{w2v.pairs|,d} (centre, context) pairs per epoch after subsampling.",
                   "Corpusul nostru: @{w2v.pairs|,d} de perechi (cuvînt central, context) pe epocă, după subeșantionare.")),
              t("the full softmax over @{w2v.V|,d} words is too slow; negative sampling replaces it.",
                "softmax-ul peste @{w2v.V|,d} de cuvinte este prea lent; îl înlocuim prin eșantionare negativă."),
              cite=t("(Mikolov et al., 2013)", "(Mikolov et al., 2013)"), wl="0.6", wr="0.37")

    D.concept(t("Negative sampling: tell real pairs from random ones", "Eșantionarea negativă: deosebim perechile reale de cele aleatoare"),
              t("replace the softmax by $K+1$ binary logistic regressions: one real context word, $K$ noise words.",
                "înlocuim softmax-ul cu $K+1$ regresii logistice binare: un cuvînt real de context și $K$ cuvinte de zgomot."),
              "\\[ \\ell(w,o)=-\\log\\sigma(\\mathbf u_o^{\\top}\\mathbf v_w)-\\sum_{k=1}^{K}\\log\\sigma(-\\mathbf u_{n_k}^{\\top}\\mathbf v_w) \\]\\vspace{-2mm}\\[ n_k\\sim P_n(w)\\propto f(w)^{3/4} \\]"
              + t("\\where{$\\sigma$ logistic function; $K=@{w2v.neg}$; $f(w)$ corpus frequency; frequent words kept with probability $\\min(1,\\sqrt{t/f(w)}+t/f(w))$, $t=10^{-4}$}",
                  "\\where{$\\sigma$ funcția logistică; $K=@{w2v.neg}$; $f(w)$ frecvența în corpus; cuvintele frecvente sînt păstrate cu probabilitatea $\\min(1,\\sqrt{t/f(w)}+t/f(w))$, $t=10^{-4}$}"),
              it(t("Skip-gram with negative sampling (SGNS), $d=@{w2v.dim}$, window @{w2v.win}, words with at least @{w2v.min_count} occurrences, Adam, @{w2v.epochs} epochs in @{w2v.seconds|.0f} seconds.",
                   "Skip-gram cu eșantionare negativă (SGNS), $d=@{w2v.dim}$, fereastră @{w2v.win}, cuvinte cu cel puțin @{w2v.min_count} apariții, Adam, @{w2v.epochs} epoci în @{w2v.seconds|.0f} de secunde."),
                 t("Loss per pair from @{w2v.loss_first|.3f} (epoch 1) to @{w2v.loss_last|.3f}.", "Pierderea pe pereche scade de la @{w2v.loss_first|.3f} (epoca 1) la @{w2v.loss_last|.3f}."),
                 t("SGNS implicitly factorises a shifted pointwise mutual information (PMI) matrix (appendix).",
                   "SGNS factorizează implicit o matrice deplasată de informație mutuală punctuală (PMI) (anexă).")),
              t("the same trick, contrast a true item with sampled negatives, trains recommender and fraud-embedding systems.",
                "același truc, contrastul dintre un element real și negative eșantionate, antrenează sisteme de recomandare și reprezentări pentru detectarea fraudei."),
              cite=t("(Mikolov et al., 2013; Levy \\& Goldberg, 2014)", "(Mikolov et al., 2013; Levy \\& Goldberg, 2014)"))

    D.concept(t("GloVe: fit the logarithm of the co-occurrence counts", "GloVe: aproximăm logaritmul frecvențelor de co-apariție"),
              t("count once how often words appear together, then fit dot products to their logarithms with a weighted least-squares loss.",
                "numărăm o singură dată cît de des apar cuvintele împreună, apoi aproximăm logaritmii acestor frecvențe prin produse scalare, cu o pierdere de tip cele mai mici pătrate ponderate."),
              "\\[ J=\\sum_{i,j:\\,X_{ij}>0}f(X_{ij})\\big(\\mathbf w_i^{\\top}\\tilde{\\mathbf w}_j+b_i+\\tilde b_j-\\ln X_{ij}\\big)^2 \\]\\vspace{-2mm}\\[ f(x)=\\min\\{1,(x/x_{\\max})^{\\alpha}\\} \\]"
              + t("\\where{$X_{ij}$ co-occurrences within @{glove.win} words, weighted $1/\\text{distance}$; $x_{\\max}=@{glove.xmax}$, $\\alpha=@{glove.alpha}$; final vector $\\mathbf w_i+\\tilde{\\mathbf w}_i$}",
                  "\\where{$X_{ij}$ co-aparițiile la cel mult @{glove.win} cuvinte distanță, ponderate cu $1/\\text{distanță}$; $x_{\\max}=@{glove.xmax}$, $\\alpha=@{glove.alpha}$; vectorul final este $\\mathbf w_i+\\tilde{\\mathbf w}_i$}"),
              D.fig("emb_training", "0.5\\textheight"),
              t("@{glove.nnz|,d} non-zero pairs (@{glove.density_pct|.3f}\\,\\% of $V^2$); @{glove.share_below_xmax|.2f}\\,\\% are below $x_{\\max}$ and get a weight below 1.",
                "@{glove.nnz|,d} de perechi nenule (@{glove.density_pct|.3f}\\,\\% din $V^2$); @{glove.share_below_xmax|.2f}\\,\\% sînt sub $x_{\\max}$ și primesc o pondere sub 1."),
              cite=t("(Pennington et al., 2014; own computation)", "(Pennington et al., 2014; calcul propriu)"), wl="0.5", wr="0.47")

    D.concept(t("Count-based word vectors: PMI and truncated matrix factorisation", "Vectori de cuvinte din frecvențe: PMI și factorizarea trunchiată a matricei"),
              t("before neural embeddings, dense vectors came from factorising a matrix of word associations.",
                "înaintea vectorilor neuronali, vectorii denși proveneau din factorizarea unei matrice de asocieri între cuvinte."),
              "\\[ \\mathrm{PMI}(w,c)=\\ln\\frac{P(w,c)}{P(w)P(c)} \\]\\vspace{-2mm}\\[ \\mathbf M_{+}\\approx\\mathbf U_d\\boldsymbol\\Sigma_d\\mathbf V_d^{\\top},\\qquad \\mathbf E=\\mathbf U_d\\boldsymbol\\Sigma_d^{1/2} \\]"
              + t("\\where{$P(w,c)$ share of co-occurrences of $w$ and $c$; $\\mathbf M_{+}$ the matrix of $\\max(\\mathrm{PMI},0)$; $\\mathbf U_d,\\boldsymbol\\Sigma_d,\\mathbf V_d$ the $d$ leading terms of the singular value decomposition (SVD)}",
                  "\\where{$P(w,c)$ ponderea co-aparițiilor lui $w$ și $c$; $\\mathbf M_{+}$ matricea valorilor $\\max(\\mathrm{PMI},0)$; $\\mathbf U_d,\\boldsymbol\\Sigma_d,\\mathbf V_d$ primii $d$ termeni ai descompunerii în valori singulare (SVD)}"),
              it(t("Pointwise mutual information (PMI) is positive when two words meet more often than chance.",
                   "Informația mutuală punctuală (PMI) este pozitivă cînd două cuvinte apar împreună mai des decît la întîmplare."),
                 t("Our co-occurrence table has @{glove.nnz|,d} non-zero pairs, @{glove.density_pct|.3f}\\,\\% of all $V^2$ cells: sparse, so the SVD is cheap.",
                   "Tabelul nostru de co-apariții are @{glove.nnz|,d} de perechi nenule, @{glove.density_pct|.3f}\\,\\% din cele $V^2$ celule: o matrice rară, deci SVD este ieftin."),
                 t("word2vec, GloVe and the SVD of PMI give similar neighbours on the same corpus.",
                   "word2vec, GloVe și SVD-ul matricei PMI dau vecini asemănători pe același corpus.")),
              t("a factorisation of counts is transparent and reproducible: a good first embedding for an internal corpus.",
                "o factorizare a frecvențelor este transparentă și reproductibilă: un prim vector de cuvinte bun pentru un corpus intern."),
              cite=t("(Levy \\& Goldberg, 2014)", "(Levy \\& Goldberg, 2014)"), wl="0.6", wr="0.37")

    rows = [[tt(w), V.fmt(f"nbrs.w2v.{w}").replace("_", "\\_"), V.fmt(f"nbrs.glove.{w}").replace("_", "\\_")] for w in N["probes"]]
    D.frame(t("Nearest neighbours in our central-bank embeddings", "Cei mai apropiați vecini în vectorii noștri, învățați din discursurile băncii centrale"),
            table("@{}p{0.13\\textwidth}p{0.4\\textwidth}p{0.4\\textwidth}@{}", [t("Word", "Cuvîntul"), "word2vec (SGNS)", "GloVe"], rows, size="scriptsize") +
            note(t("Five nearest words by cosine; both trained on the same @{corpus.n_tokens|,d} tokens of ECB speeches; mean overlap of the two top-5 lists: @{nbrs_overlap|.2f} words.",
                   "Cele mai apropiate cinci cuvinte după cosinus; ambele modele sînt antrenate pe aceleași @{corpus.n_tokens|,d} de unități lexicale din discursurile BCE; suprapunerea medie a celor două liste: @{nbrs_overlap|.2f} cuvinte.")) +
            D.bottom(t("a central-bank watcher can expand a search for ``inflation'' with its learned neighbours instead of a hand-made list.",
                       "un analist al politicii monetare poate extinde o căutare după ``inflation'' cu vecinii învățați, în locul unei liste făcute de mînă."), tryit=NB + "2"),
            "footnotesize", cite=OWN)

    D.figframe(t("Topics of central banking form regions of the embedding space", "Temele activității băncii centrale formează regiuni în spațiul vectorilor"),
               t("2-D principal component analysis (PCA) projection of 32 words chosen in four groups; the groups were not given to the model.",
                 "proiecția bidimensională prin analiza componentelor principale (PCA) a 32 de cuvinte alese din patru grupuri; grupurile nu au fost date modelului."),
               None, "w2v_pca",
               bullets=[t("Two components keep @{pca.var2_pct|.1f}\\,\\% of the variance of these @{pca.n_words} vectors, so the picture is a rough shadow of $\\R^{@{w2v.dim}}$.",
                          "Două componente păstrează @{pca.var2_pct|.1f}\\,\\% din varianța acestor @{pca.n_words} de vectori, deci imaginea este doar o umbră aproximativă a spațiului $\\R^{@{w2v.dim}}$.")],
               height="0.5\\textheight", cite=OWN)

    an = N["analogy"]
    an_rows = [[tt(k.replace(":", " : ").replace("  ", " ")), tt(v)] for k, v in an.items()]
    D.frame(t("Analogies, and the limits of one vector per word", "Analogii și limitele unui singur vector pentru fiecare cuvînt"), cols(
        table("@{}ll@{}", [t("$a:b::c:\\,?$", "$a:b::c:\\,?$"), t("answer", "răspuns")], an_rows, size="scriptsize") +
        note(t("Answer $=\\argmax_v\\cos(\\mathbf e_v,\\mathbf e_b-\\mathbf e_a+\\mathbf e_c)$, excluding $a,b,c$; our word2vec, one small corpus.",
               "Răspunsul $=\\argmax_v\\cos(\\mathbf e_v,\\mathbf e_b-\\mathbf e_a+\\mathbf e_c)$, excluzînd $a,b,c$; word2vec-ul nostru, un singur corpus mic.")),
        it(t(("\\textbf{Static embeddings}", "one vector per word type, whatever the sentence",
              "``interest'' (rate) and ``interest'' (attention) share a vector; so do ``bank'' and ``bank''"),
             ("\\textbf{Vectori statici}", "un singur vector pentru fiecare cuvînt, indiferent de frază",
              "``interest'' (dobîndă) și ``interest'' (atenție) au același vector; la fel ``bank''")),
           t(("\\textbf{Coverage}", "the ECB vocabulary covers @{oov.pb_cover|.1f}\\,\\% of PhraseBank tokens and @{oov.tw_cover|.1f}\\,\\% of tweet tokens",
              "the rest are out of vocabulary (OOV); subword models (fastText) and transformers (Lecture 7) address both problems"),
             ("\\textbf{Acoperirea}", "vocabularul BCE acoperă @{oov.pb_cover|.1f}\\,\\% din unitățile lexicale din PhraseBank și @{oov.tw_cover|.1f}\\,\\% din cele din tweet-uri",
              "restul sînt în afara vocabularului (OOV); modelele cu subcuvinte (fastText) și transformerele (cursul 7) rezolvă ambele probleme")), sep="4pt"), "0.46", "0.51"),
        "footnotesize", cite=t("(Mikolov et al., 2013; Bojanowski et al., 2017)", "(Mikolov et al., 2013; Bojanowski et al., 2017)"))

    D.concept(t("Subword vectors for words never seen", "Vectori din subcuvinte pentru cuvintele nevăzute"),
              t("build a word vector from the vectors of its character $n$-grams, so a new ticker or a typo still gets a vector.",
                "construim vectorul unui cuvînt din vectorii $n$-gramelor sale de caractere, astfel încît un simbol bursier nou sau o greșeală de scriere primește totuși un vector."),
              "\\[ \\mathbf e_w=\\sum_{g\\in\\mathcal G(w)}\\mathbf z_g \\]"
              + t("\\where{$\\mathcal G(w)$ the character $n$-grams of $w$ (for example 3 to 6 characters, with boundary symbols) and the word itself; $\\mathbf z_g$ learned $n$-gram vectors}",
                  "\\where{$\\mathcal G(w)$ $n$-gramele de caractere ale lui $w$ (de exemplu de 3--6 caractere, cu simboluri de margine) și cuvîntul însuși; $\\mathbf z_g$ vectori învățați ai $n$-gramelor}"),
              it(t("Coverage of our word-level ECB vocabulary: @{oov.pb_cover|.1f}\\,\\% of PhraseBank tokens, @{oov.tw_cover|.1f}\\,\\% of tweet tokens.",
                   "Acoperirea vocabularului nostru BCE la nivel de cuvînt: @{oov.pb_cover|.1f}\\,\\% din unitățile lexicale PhraseBank, @{oov.tw_cover|.1f}\\,\\% din cele din tweet-uri."),
                 t("Subword models (fastText) share information between ``profit'', ``profits'' and ``profitability''.",
                   "Modelele cu subcuvinte (fastText) împart informația între ``profit'', ``profits'' și ``profitability''."),
                 t("Transformers (Lecture 7) use the same idea with learned subword tokens.",
                   "Transformerele (cursul 7) folosesc aceeași idee, cu unități de subcuvinte învățate.")),
              t("for tweets and chat messages, with tickers and spelling variants, use subword vectors or a subword tokeniser.",
                "pentru tweet-uri și mesaje, cu simboluri bursiere și variante de scriere, folosiți vectori din subcuvinte sau un tokenizator pe subcuvinte."),
              cite=t("(Bojanowski et al., 2017)", "(Bojanowski et al., 2017)"))

    # ------------------------------------------------------------------ 4. classifiers
    D.section(t("Sentence classifiers: sparse versus dense", "Clasificatoare de fraze: reprezentări rare și dense"))
    D.concept(t("An embedding-bag network: averaged word vectors followed by a softmax layer", "O rețea embedding bag: media vectorilor de cuvinte și un strat softmax"),
              t("the simplest neural text classifier learns the embeddings and the classifier together.",
                "cel mai simplu clasificator neuronal de text învață împreună vectorii de cuvinte și clasificatorul."),
              "\\[ \\mathbf z=\\frac{1}{|S|}\\sum_{s\\in S}\\mathbf e_s \\]\\vspace{-2mm}\\[ \\hat{\\mathbf p}=\\softmax(\\mW\\,\\mathrm{dropout}(\\mathbf z)+\\vb) \\]\\vspace{-2mm}\\[ \\loss=-\\frac1n\\sum_i\\omega_{y_i}\\log\\hat p_{i,y_i} \\]"
              + t("\\where{$S$ the tokens and bigrams of the sentence; $\\mathbf e_s$ learned vectors ($d=100$); $\\omega_k=n/(3n_k)$ class weights against the dominance of ``neutral''}",
                  "\\where{$S$ unitățile lexicale și bigramele frazei; $\\mathbf e_s$ vectori învățați ($d=100$); $\\omega_k=n/(3n_k)$ ponderi ale claselor, împotriva dominanței clasei ``neutru''}"),
              it(t("Random start, or word2vec start: rows of $\\mathbf E$ initialised with our ECB vectors, then fine-tuned.",
                   "Start aleator sau start word2vec: rîndurile lui $\\mathbf E$ sînt inițializate cu vectorii noștri BCE, apoi ajustate."),
                 t("Also compared: mean word2vec vector + logistic regression (frozen), TF-IDF + a multilayer perceptron (MLP) with 128 hidden units.",
                   "Comparăm și: media vectorilor word2vec + regresie logistică (vectori înghețați), TF-IDF + un perceptron multistrat (MLP) cu 128 de unități ascunse.")),
              t("this is the architecture of fastText-type classifiers, used in production for millions of documents per hour.",
                "aceasta este arhitectura clasificatoarelor de tip fastText, folosite în producție pentru milioane de documente pe oră."),
              cite=t("(Joulin et al., 2017)", "(Joulin et al., 2017)"))

    D.concept(t("Logistic regression on text: the gradient", "Regresia logistică pe text: gradientul"),
              t("the multinomial logistic model is a one-layer network; its gradient compares predicted and true class shares.",
                "modelul logistic multinomial este o rețea cu un singur strat; gradientul lui compară probabilitățile prezise cu clasele reale."),
              "\\[ \\loss(\\mathbf B)=-\\sum_{i}\\sum_k y_{ik}\\ln\\hat p_{ik}+\\frac{1}{2C}\\|\\mathbf B\\|_F^2 \\]\\vspace{-2mm}\\[ \\nabla_{\\mathbf B}\\loss=\\mathbf X^{\\top}(\\hat{\\mathbf P}-\\mathbf Y)+\\mathbf B/C \\]"
              + t("\\where{$\\mathbf X$ the TF-IDF matrix ($n\\times V$, sparse); $\\mathbf Y$ one-hot labels; $\\hat{\\mathbf P}$ softmax probabilities; $C$ inverse penalty, chosen on validation from $\\{0.1,1,10\\}$}",
                  "\\where{$\\mathbf X$ matricea TF-IDF ($n\\times V$, rară); $\\mathbf Y$ etichetele one-hot; $\\hat{\\mathbf P}$ probabilitățile softmax; $C$ inversul penalizării, ales pe validare din $\\{0{,}1;1;10\\}$}"),
              it(t("Convex loss: one optimum, no seed to report; the multilayer perceptron (MLP) on the same inputs adds a hidden layer of 128 units.",
                   "Pierdere convexă: un singur optim, fără sămînță de raportat; perceptronul multistrat (MLP) pe aceleași intrări adaugă un strat ascuns de 128 de unități."),
                 t("PhraseBank macro-F1: logistic @{sent.pb.tfidf_lr.f1|.2f}\\,\\%, MLP @{sent.pb.tfidf_mlp.f1|.2f}\\,\\%: the hidden layer does not help here.",
                   "Macro-F1 pe PhraseBank: logistic @{sent.pb.tfidf_lr.f1|.2f}\\,\\%, MLP @{sent.pb.tfidf_mlp.f1|.2f}\\,\\%: stratul ascuns nu ajută aici.")),
              t("with a few thousand short sentences, the extra flexibility of a network buys variance, not accuracy.",
                "cu cîteva mii de fraze scurte, flexibilitatea suplimentară a unei rețele aduce varianță, nu acuratețe."),
              cite=OWN)

    D.concept(t("Sentence vectors built from word vectors", "Vectorul frazei construit din vectorii cuvintelor"),
              t("averaging word vectors is the simplest sentence representation, and it loses what matters for sentiment.",
                "media vectorilor de cuvinte este cea mai simplă reprezentare a unei fraze și pierde exact ce contează pentru sentiment."),
              "\\[ \\mathbf s_i=\\frac{1}{|S_i|}\\sum_{w\\in S_i}\\mathbf e_w,\\qquad \\mathbf s_i^{\\mathrm{w}}=\\frac{\\sum_{w}a_w\\mathbf e_w}{\\sum_w a_w},\\ \\ a_w=\\frac{\\alpha}{\\alpha+f(w)} \\]"
              + t("\\where{$S_i$ the in-vocabulary tokens of sentence $i$; $a_w$ a weight that down-weights frequent words; $\\alpha$ a small constant}",
                  "\\where{$S_i$ unitățile lexicale din vocabular ale frazei $i$; $a_w$ o pondere care micșorează influența cuvintelor frecvente; $\\alpha$ o constantă mică}"),
              it(t("Mean of our frozen ECB vectors + logistic regression: macro-F1 @{sent.pb.w2v_lr.f1|.2f}\\,\\% on PhraseBank, @{sent.tw.w2v_lr.f1|.2f}\\,\\% on tweets.",
                   "Media vectorilor BCE înghețați + regresie logistică: macro-F1 @{sent.pb.w2v_lr.f1|.2f}\\,\\% pe PhraseBank și @{sent.tw.w2v_lr.f1|.2f}\\,\\% pe tweet-uri."),
                 t("``rose'' and ``fell'' are close in a co-occurrence space (same contexts), so the average blurs the sign.",
                   "``rose'' și ``fell'' sînt apropiate într-un spațiu de co-apariții (aceleași contexte), deci media estompează semnul."),
                 t("Fine-tuning the vectors on labelled data (embedding bag) recovers most of the gap.",
                   "Ajustarea vectorilor pe date etichetate (embedding bag) recuperează cea mai mare parte a diferenței.")),
              t("generic embeddings are good for search and similarity, not automatically for sentiment.",
                "vectorii generali sînt buni pentru căutare și similaritate, nu în mod automat pentru sentiment."),
              cite=OWN)

    D.concept(t("Protocol and metric: macro-F1 on an imbalanced target", "Protocolul și metrica: macro-F1 pentru o țintă dezechilibrată"),
              t("``neutral'' is @{data.pb_neu_pct|.1f}\\,\\% of PhraseBank, so accuracy rewards a model that says ``neutral'' all the time.",
                "clasa ``neutru'' reprezintă @{data.pb_neu_pct|.1f}\\,\\% din PhraseBank, deci acuratețea răsplătește un model care spune mereu ``neutru''."),
              "\\[ \\mathrm{F1}_k=\\frac{2P_kR_k}{P_k+R_k},\\qquad \\text{macro-F1}=\\frac13\\sum_{k=1}^{3}\\mathrm{F1}_k \\]"
              + t("\\where{$P_k$ precision and $R_k$ recall of class $k$; each class counts equally, whatever its size}",
                  "\\where{$P_k$ precizia și $R_k$ sensibilitatea clasei $k$; fiecare clasă contează la fel, indiferent de mărime}"),
              it(t("PhraseBank: 5 random stratified splits 70/15/15; tweets: published validation split used as test, 3 seeds.",
                   "PhraseBank: 5 împărțiri aleatoare stratificate 70/15/15; tweet-uri: setul de validare publicat, folosit ca test, 3 semințe."),
                 t("Penalty $C$, early stopping and vocabulary are chosen on the validation part only.",
                   "Penalizarea $C$, oprirea timpurie și vocabularul sînt alese numai pe partea de validare.")),
              t("for an alert system, report the recall of the negative class next to the macro-F1: that is the class that costs money.",
                "pentru un sistem de alertă, raportați sensibilitatea clasei negative alături de macro-F1: aceasta este clasa care costă bani."),
              cite=OWN, tryit=NB + "3")

    D.figframe(t("On short financial sentences, TF-IDF with logistic regression has the highest macro-F1", "Pe frazele financiare scurte, TF-IDF cu regresie logistică are cel mai mare macro-F1"),
               t("test macro-F1 of seven representations and classifiers, PhraseBank and finance tweets (bars: mean, whiskers: standard deviation).",
                 "macro-F1 pe test pentru șapte reprezentări și clasificatoare, PhraseBank și tweet-uri financiare (bare: media, mustăți: abaterea standard)."),
               None, "sentiment_models",
               bullets=[t("Best on PhraseBank: @{sent.pb.tfidf_lr.f1|.2f} (TF-IDF + logistic) vs @{sent.pb.embbag_w2v.f1|.2f} (embedding bag, word2vec start); majority @{sent.pb.majority.f1|.2f}.",
                          "Pe PhraseBank: @{sent.pb.tfidf_lr.f1|.2f} (TF-IDF + logistic), față de @{sent.pb.embbag_w2v.f1|.2f} (embedding bag, start word2vec); clasa majoritară @{sent.pb.majority.f1|.2f}.")],
               height="0.5\\textheight", cite=OWN, tryit=NB + "3")

    rows = [[M[k], "@{sent.pb." + k + ".f1|.2f}\\,$\\pm$\\,@{sent.pb." + k + ".f1_sd|.2f}", "@{sent.pb." + k + ".acc|.2f}",
             "@{sent.tw." + k + ".f1|.2f}\\,$\\pm$\\,@{sent.tw." + k + ".f1_sd|.2f}", "@{sent.tw." + k + ".acc|.2f}"]
            for k in ["majority", "bow_lr", "tfidf_lr", "tfidf_mlp", "w2v_lr", "embbag", "embbag_w2v"]]
    D.frame(t("Sentiment results, with uncertainty", "Rezultatele pentru sentiment, cu incertitudine"),
            table("@{}lrrrr@{}", [t("Model", "Modelul"), t("PhraseBank macro-F1", "PhraseBank macro-F1"), t("acc.", "acur."),
                                  t("Tweets macro-F1", "Tweet-uri macro-F1"), t("acc.", "acur.")], rows, size="scriptsize") +
            note(t("Paired over the 5 PhraseBank splits: TF-IDF + logistic minus embedding bag (word2vec start) $=@{sent.pb_diff_tfidf_embw2v|+.2f}$ points, $t=@{sent.pb_diff_t|.2f}$ (4 degrees of freedom).",
                   "Pereche, pe cele 5 împărțiri PhraseBank: TF-IDF + logistic minus embedding bag (start word2vec) $=@{sent.pb_diff_tfidf_embw2v|+.2f}$ puncte, $t=@{sent.pb_diff_t|.2f}$ (4 grade de libertate)."),
                 t("Starting from our ECB word2vec vectors changes the embedding bag by @{sent.pb_w2v_gain|+.2f} points on PhraseBank and @{sent.tw_w2v_gain|+.2f} on tweets.",
                   "Pornirea de la vectorii noștri word2vec BCE schimbă rezultatul embedding bag cu @{sent.pb_w2v_gain|+.2f} puncte pe PhraseBank și cu @{sent.tw_w2v_gain|+.2f} pe tweet-uri.")) +
            D.bottom(t("the cheapest model within one standard deviation of the best is the one to deploy and explain.",
                       "modelul cel mai ieftin aflat la cel mult o abatere standard de cel mai bun este cel care trebuie folosit și explicat.")),
            "footnotesize", cite=OWN)

    rows_d = [[k, "@{dims." + k + "|.1f}"] for k in ["8", "16", "32", "64", "128", "256"]]
    D.frame(t("Choice of the embedding dimension", "Alegerea dimensiunii vectorului de cuvinte"), cols(
        table("@{}rr@{}", [t("Dimension $d$", "Dimensiunea $d$"), t("Macro-F1 (\\%)", "Macro-F1 (\\%)")], rows_d) +
        note(t("Embedding bag, random start, PhraseBank split 1, mean of 3 seeds.", "Embedding bag, start aleator, împărțirea 1 PhraseBank, media pe 3 semințe.")),
        it(t("Parameters of the embedding layer: $V\\times d$; with bigrams $V$ is several thousand, so $d$ drives the model size.",
             "Parametrii stratului de embedding: $V\\times d$; cu bigrame, $V$ are cîteva mii de elemente, deci $d$ determină mărimea modelului."),
           t("Here the score is flat in $d$: from @{dims.8|.1f} at $d=8$ to @{dims.128|.1f} at $d=128$.",
             "Aici scorul variază puțin cu $d$: de la @{dims.8|.1f} la $d=8$ la @{dims.128|.1f} la $d=128$."),
           t("A three-class task on short sentences needs few dimensions; general-purpose embeddings use 100--300 because they serve many tasks.",
             "O sarcină cu trei clase pe fraze scurte are nevoie de puține dimensiuni; vectorii de uz general folosesc 100--300 pentru că servesc multor sarcini."), sep="4pt"), "0.35", "0.62")
        + D.bottom(t("choose $d$ on validation data; a smaller model is cheaper to serve and to explain.",
                     "alegeți $d$ pe datele de validare; un model mai mic este mai ieftin de rulat și de explicat.")),
        "footnotesize", cite=OWN)

    D.figframe(t("Classification errors and annotator agreement", "Erorile de clasificare și acordul adnotatorilor"),
               t("test accuracy by the share of annotators who agreed on the label of the sentence.",
                 "acuratețea pe test în funcție de ponderea adnotatorilor care au fost de acord asupra etichetei frazei."),
               None, "agreement",
               bullets=[t("TF-IDF + logistic: @{agree.tfidf_lr.100|.1f}\\,\\% on unanimous sentences, @{agree.tfidf_lr.50|.1f}\\,\\% on 50\\,\\% agreement; @{data.pb_agree100|,d} of @{data.pb_n|,d} sentences are unanimous.",
                          "TF-IDF + logistic: @{agree.tfidf_lr.100|.1f}\\,\\% pe frazele cu acord unanim, @{agree.tfidf_lr.50|.1f}\\,\\% la un acord de 50\\,\\%; @{data.pb_agree100|,d} din @{data.pb_n|,d} de fraze au acord unanim.")],
               job=t("label noise caps every model: before buying a bigger model, check how much your own analysts agree.",
                     "zgomotul din etichete limitează orice model: înainte de a cumpăra un model mai mare, verificați cît de mult sînt de acord propriii analiști."),
               height="0.42\\textheight", cite=OWN)

    D.figframe(t("Learning curves with few labels: word2vec initialisation and TF-IDF", "Curbele de învățare cu puține etichete: inițializarea word2vec și TF-IDF"),
               t("left: test macro-F1 against the number of labelled sentences; right: embedding size of the embedding bag.",
                 "stînga: macro-F1 pe test în funcție de numărul de fraze etichetate; dreapta: dimensiunea vectorilor din embedding bag."),
               None, "learning_curve",
               bullets=[t("With @{curve.n_small} labelled sentences: TF-IDF @{curve.tfidf_lr.p5|.1f}, mean word2vec @{curve.w2v_lr.p5|.1f}, embedding bag random @{curve.embbag.p5|.1f}, word2vec start @{curve.embbag_w2v.p5|.1f}; with all @{curve.n_full|,d}: @{curve.tfidf_lr.p100|.1f}, @{curve.w2v_lr.p100|.1f}, @{curve.embbag.p100|.1f}, @{curve.embbag_w2v.p100|.1f}.",
                          "Cu @{curve.n_small} de fraze etichetate: TF-IDF @{curve.tfidf_lr.p5|.1f}, media word2vec @{curve.w2v_lr.p5|.1f}, embedding bag aleator @{curve.embbag.p5|.1f}, start word2vec @{curve.embbag_w2v.p5|.1f}; cu toate cele @{curve.n_full|,d}: @{curve.tfidf_lr.p100|.1f}, @{curve.w2v_lr.p100|.1f}, @{curve.embbag.p100|.1f}, @{curve.embbag_w2v.p100|.1f}.")],
               height="0.42\\textheight", cite=OWN, tryit=NB + "4")

    D.frame(t("Main error type: negative news classified as neutral", "Principalul tip de eroare: știri negative clasificate ca neutre"), cols(
        table("@{}lrrr@{}", [t("True \\textbackslash{} predicted", "Real \\textbackslash{} prezis"), t("neg.", "neg."), t("neu.", "neu."), t("pos.", "poz.")],
              [[t("negative", "negativ"), "--", "@{cm.neg_as_neu}", "@{cm.neg_as_pos}"],
               [t("positive", "pozitiv"), "@{cm.pos_as_neg}", "@{cm.pos_as_neu}", "--"]]) +
        note(t("TF-IDF + logistic, the 5 PhraseBank test sets pooled (@{cm.n|,d} sentences).",
               "TF-IDF + logistic, cele 5 seturi de test PhraseBank reunite (@{cm.n|,d} de fraze).")),
        it(t("Recall: negative @{cm.neg_recall|.1f}\\,\\%, neutral @{cm.neu_recall|.1f}\\,\\%, positive @{cm.pos_recall|.1f}\\,\\%.",
             "Sensibilitatea: negativ @{cm.neg_recall|.1f}\\,\\%, neutru @{cm.neu_recall|.1f}\\,\\%, pozitiv @{cm.pos_recall|.1f}\\,\\%."),
           t("Opposite-sign errors are rare; most errors move a sentence to neutral.", "Erorile de semn opus sînt rare; cele mai multe erori mută fraza în clasa neutră."),
           t("Typical miss: a fall described with neutral words and numbers (``sales were EUR 10 mn, down from 12'').",
             "Eroare tipică: o scădere descrisă cu cuvinte neutre și cifre (``sales were EUR 10 mn, down from 12'')."), sep="4pt"), "0.47", "0.5")
        + D.bottom(t("numbers carry sentiment in finance; mapping them all to one token loses ``down from''-type comparisons that a sequence model can read.",
                     "în finanțe, cifrele poartă sentiment; înlocuirea lor cu o singură unitate pierde comparațiile de tip ``down from'' pe care un model secvențial le poate citi.")),
        "footnotesize", cite=OWN)

    # ------------------------------------------------------------------ 5. topics
    D.section(t("Case study: topics of finance tweets", "Studiu de caz: temele tweet-urilor financiare"))
    D.figframe(t("Topic classification: F1 is high for earnings and the Fed, low for general news", "Clasificarea pe teme: F1 este mare pentru rezultate și Fed, mic pentru știrile generale"),
               t("per-topic F1 on the @{topic.n_test|,d} test tweets (published validation split).",
                 "F1 pe fiecare temă, pentru cele @{topic.n_test|,d} de tweet-uri de test (setul de validare publicat)."),
               None, "topic_f1",
               bullets=[t("Macro-F1: TF-IDF + logistic @{topic.lr_f1|.2f}\\,\\%, embedding bag (word2vec start) @{topic.eb_f1|.2f}\\,\\%; accuracy @{topic.lr_acc|.2f}\\,\\% vs @{topic.majority_acc|.2f}\\,\\% for the majority topic.",
                          "Macro-F1: TF-IDF + logistic @{topic.lr_f1|.2f}\\,\\%, embedding bag (start word2vec) @{topic.eb_f1|.2f}\\,\\%; acuratețe @{topic.lr_acc|.2f}\\,\\%, față de @{topic.majority_acc|.2f}\\,\\% pentru tema majoritară.")],
               height="0.55\\textheight", cite=OWN, tryit=NB + "5")

    D.frame(t("Applications of text models in finance", "Aplicații ale modelelor de text în finanțe"), cols(it(
        t(("\\textbf{News and tweet triage}", "route items by topic (best: @{topic.best_class}, F1 @{topic.best_f1|.1f}\\,\\%) and flag negative ones for the analyst"),
          ("\\textbf{Triajul știrilor și al tweet-urilor}", "direcționarea pe teme (cea mai bună: @{topic.best_class_ro}, F1 @{topic.best_f1|.1f}\\,\\%) și semnalarea celor negative pentru analist")),
        t(("\\textbf{Central-bank communication}", "track words near ``inflation'' or ``energy'' speech by speech"),
          ("\\textbf{Comunicarea băncii centrale}", "urmărirea cuvintelor din jurul lui ``inflation'' sau ``energy'', discurs cu discurs")),
        t(("\\textbf{Compliance and environmental, social and governance (ESG) screening}", "search reports with learned synonyms, not fixed keyword lists"),
          ("\\textbf{Conformitate și analiza de mediu, social și guvernanță (ESG)}", "căutarea în rapoarte cu sinonime învățate, nu cu liste fixe de cuvinte-cheie")), sep="3pt"),
        it(t(("\\textbf{Limits of today's models}", "word order beyond bigrams is lost", "one vector per word ignores context",
              "weakest topic: @{topic.worst_class}, F1 @{topic.worst_f1|.1f}\\,\\%"),
             ("\\textbf{Limitele modelelor de azi}", "ordinea cuvintelor dincolo de bigrame se pierde", "un singur vector pentru fiecare cuvînt ignoră contextul",
              "tema cea mai slabă: @{topic.worst_class_ro}, F1 @{topic.worst_f1|.1f}\\,\\%")),
           t(("\\textbf{Next: attention}", "contextual vectors from transformers, e.g. bidirectional encoder representations from transformers (BERT) and a BERT model adapted to financial text (FinBERT), in Lecture 7, compared fairly with these baselines"),
             ("\\textbf{Urmează: atenția}", "vectori contextuali din transformere, de exemplu reprezentări bidirecționale ale codificatorului transformer (BERT) și un model BERT adaptat textelor financiare (FinBERT), în cursul 7, comparați corect cu aceste modele de referință")), sep="3pt"), "0.5", "0.47")
        + D.bottom(t("start every text project with TF-IDF + logistic regression; move to embeddings or transformers only when the gain is shown on your data.",
                     "începeți orice proiect de text cu TF-IDF + regresie logistică; treceți la vectori de cuvinte sau transformere doar cînd cîștigul este demonstrat pe datele dumneavoastră.")),
        "footnotesize", cite=OWN)

    # ------------------------------------------------------------------ wrap-up
    D.parts.append("\\section*{" + t("Wrap-up", "Recapitulare") + "}\n")
    D.quiz_frame(t("idf", "idf"),
                 t("A word appears in every one of the $n$ training sentences; what is its smoothed idf?",
                   "Un cuvînt apare în fiecare dintre cele $n$ fraze de antrenare; care este idf-ul său netezit?"),
                 t("$\\ln\\frac{1+n}{1+n}+1=1$: the minimum, so it barely moves the normalised vector.",
                   "$\\ln\\frac{1+n}{1+n}+1=1$: valoarea minimă, deci aproape nu modifică vectorul normalizat."))
    D.quiz_frame(t("negative sampling", "eșantionarea negativă"),
                 t("Why does word2vec draw negative words from $f(w)^{3/4}$ and not from $f(w)$?",
                   "De ce extrage word2vec cuvintele negative din $f(w)^{3/4}$ și nu din $f(w)$?"),
                 t("The power 3/4 flattens the distribution: rare words are drawn more often than their frequency, frequent ones like ``the'' less, which gives more useful contrasts.",
                   "Puterea 3/4 aplatizează distribuția: cuvintele rare sînt extrase mai des decît frecvența lor, iar cele frecvente, ca ``the'', mai rar, ceea ce dă contraste mai utile."))
    D.quiz_frame(t("accuracy", "acuratețea"),
                 t("A vendor reports 60\\,\\% accuracy on PhraseBank sentiment; is it good?",
                   "Un furnizor raportează o acuratețe de 60\\,\\% pe sentimentul din PhraseBank; este un rezultat bun?"),
                 t("No: always ``neutral'' gives @{sent.pb.majority.acc|.2f}\\,\\%; ask for macro-F1 (our TF-IDF + logistic: @{sent.pb.tfidf_lr.f1|.2f}\\,\\%).",
                   "Nu: prognoza ``mereu neutru'' dă @{sent.pb.majority.acc|.2f}\\,\\%; cereți macro-F1 (TF-IDF + logistic, la noi: @{sent.pb.tfidf_lr.f1|.2f}\\,\\%)."))
    D.quiz_frame(t("transfer", "transferul"),
                 t("You have only @{curve.n_small} labelled sentences; which representation helps most?",
                   "Aveți doar @{curve.n_small} de fraze etichetate; ce reprezentare ajută cel mai mult?"),
                 t("Compare on validation data: here mean word2vec gives @{curve.w2v_lr.p5|.1f}, TF-IDF @{curve.tfidf_lr.p5|.1f}, embedding bag with word2vec start @{curve.embbag_w2v.p5|.1f} macro-F1.",
                   "Comparați pe datele de validare: aici media word2vec dă @{curve.w2v_lr.p5|.1f}, TF-IDF @{curve.tfidf_lr.p5|.1f}, embedding bag cu start word2vec @{curve.embbag_w2v.p5|.1f} macro-F1."))
    D.quiz_frame(t("polysemy", "polisemia"),
                 t("Why can a static embedding not separate ``interest rate'' from ``of interest to investors''?",
                   "De ce nu poate un vector static să deosebească ``interest rate'' de ``of interest to investors''?"),
                 t("It has one vector per word type, the average of all its uses; contextual models (Lecture 7) give a different vector in each sentence.",
                   "Are un singur vector pentru fiecare cuvînt, media tuturor utilizărilor lui; modelele contextuale (cursul 7) dau un vector diferit în fiecare frază."))

    D.frame(t("Wrap-up", "Recapitulare"), "\\vspace{-2mm}\\begin{multicols}{2}\\scriptsize\\tableofcontents\\end{multicols}\n" + it(
        t("Text becomes numbers through tokens and counts (BoW, TF-IDF) or dense vectors (word2vec, GloVe) learned from co-occurrence.",
          "Textul devine numere prin unități lexicale și frecvențe (BoW, TF-IDF) sau prin vectori denși (word2vec, GloVe) învățați din co-apariții."),
        t("On real financial sentences TF-IDF + logistic regression is the model to beat; a word2vec start helps the network only when labels are few. Next: attention and transformers (Lecture 7).",
          "Pe fraze financiare reale, TF-IDF + regresie logistică este modelul de depășit; startul word2vec ajută rețeaua doar cînd etichetele sînt puține; urmează atenția și transformerele (cursul 7)."), sep="1pt"))

    D.frame(t("Research from our group: language models for finance", "Cercetări ale grupului nostru: modele de limbaj în finanțe"), it(
        t(("\\textbf{Large language models (LLM) forecast value-at-risk (VaR) and expected shortfall (ES)} (Pele, Bolovăneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann \\& Härdle, 2026, Expert Systems with Applications)",
           "paper: ``In the beginning was the Word: LLM-VaR and LLM-ES''; LLM forecasts benchmarked against classical risk models with formal backtests",
           "text models move from sentiment scores to the risk numbers themselves (Lecture 12)"),
          ("\\textbf{Modelele lingvistice mari (LLM) prognozează valoarea expusă la risc (VaR) și pierderea medie peste VaR (ES)} (Pele, Bolovăneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann \\& Härdle, 2026, Expert Systems with Applications)",
           "lucrarea: ``In the beginning was the Word: LLM-VaR and LLM-ES''; prognozele LLM sînt comparate cu modelele clasice de risc prin teste retrospective formale",
           "modelele de text trec de la scoruri de sentiment la cifrele de risc propriu-zise (cursul 12)")),
        t(("\\textbf{BitMood: artificial intelligence (AI) analysis of Bitcoin trends via Facebook emotions} (Conda, Găman, Bâg, Mazurencu-Marinescu-Pele, Pele \\& Härdle, 2026, Digital Finance)",
           "120,000 Facebook posts, 2015--2023, classified with FinBERT into compound sentiment scores with the reactions of readers",
           "sentiment predicts Bitcoin trading volume; sentiment-based strategies outperform buy-and-hold; topic modelling tracks the themes of the discussion"),
          ("\\textbf{BitMood: analiza cu inteligență artificială (AI) a tendințelor Bitcoin prin emoțiile de pe Facebook} (Conda, Găman, Bâg, Mazurencu-Marinescu-Pele, Pele \\& Härdle, 2026, Digital Finance)",
           "120.000 de postări de pe Facebook, 2015--2023, clasificate cu FinBERT în scoruri compuse de sentiment, împreună cu reacțiile cititorilor",
           "sentimentul prognozează volumul tranzacțiilor cu Bitcoin; strategiile bazate pe sentiment depășesc strategia cumpără-și-păstrează; modelarea temelor urmărește subiectele discuției")), sep="4pt") +
        note(t("Findings as stated in the papers; numbers on the other slides come from our own computations.",
               "Rezultatele de pe acest diapozitiv sînt cele raportate în lucrări; celelalte cifre din curs provin din calculele noastre.")),
        "footnotesize", cite=t("(Bucharest University of Economic Studies; Romanian Academy, Institute for Economic Forecasting)",
                               "(Academia de Studii Economice din București; Academia Română, Institutul de Prognoză Economică)"))

    D.frame(t("Further reading and resources", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Mikolov et al. (2013)} and \\textbf{Pennington et al. (2014)}: the two embedding papers, both short.",
          "\\textbf{Mikolov et al. (2013)} și \\textbf{Pennington et al. (2014)}: cele două articole despre vectorii de cuvinte, ambele scurte."),
        t("\\textbf{Levy \\& Goldberg (2014)}: why word2vec is a matrix factorisation.", "\\textbf{Levy \\& Goldberg (2014)}: de ce word2vec este o factorizare de matrice."),
        t("\\textbf{Loughran \\& McDonald (2011)}: the finance sentiment dictionary and why general dictionaries fail.",
          "\\textbf{Loughran \\& McDonald (2011)}: dicționarul de sentiment financiar și de ce eșuează dicționarele generale."), sep="4pt"),
        it(t("\\textbf{This course}: code, data, notebooks --- ", "\\textbf{Acest curs}: cod, date, notebook-uri Jupyter --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Experiment yourself}: the notebook reproduces the charts of this lecture --- ",
             "\\textbf{Exercițiu practic}: notebook-ul reproduce graficele acestui curs --- ") + "\\href{\\nnrepo/\\nnnb}{notebooks/}.",
           t("\\textbf{Seminar 6}: a sentiment index of ECB speeches built with your own classifier.",
             "\\textbf{Seminarul 6}: un indice de sentiment al discursurilor BCE, construit cu propriul clasificator."), sep="4pt"), "0.52", "0.45"))

    reference_frames(D, REFS, per_frame=7)
    D.mark_main_end()

    D.frame(t("Appendix: the gradient of negative sampling", "Anexă: gradientul eșantionării negative"), it(
        t("For one pair, $\\ell=-\\log\\sigma(s_o)-\\sum_k\\log\\sigma(-s_k)$ with $s_o=\\mathbf u_o^{\\top}\\mathbf v_w$, $s_k=\\mathbf u_{n_k}^{\\top}\\mathbf v_w$.",
          "Pentru o pereche, $\\ell=-\\log\\sigma(s_o)-\\sum_k\\log\\sigma(-s_k)$, cu $s_o=\\mathbf u_o^{\\top}\\mathbf v_w$, $s_k=\\mathbf u_{n_k}^{\\top}\\mathbf v_w$."),
        t("Using $\\frac{d}{ds}\\log\\sigma(s)=1-\\sigma(s)$: $\\partial\\ell/\\partial\\mathbf v_w=-(1-\\sigma(s_o))\\mathbf u_o+\\sum_k\\sigma(s_k)\\mathbf u_{n_k}$.",
          "Folosind $\\frac{d}{ds}\\log\\sigma(s)=1-\\sigma(s)$: $\\partial\\ell/\\partial\\mathbf v_w=-(1-\\sigma(s_o))\\mathbf u_o+\\sum_k\\sigma(s_k)\\mathbf u_{n_k}$."),
        t("$\\partial\\ell/\\partial\\mathbf u_o=-(1-\\sigma(s_o))\\mathbf v_w$ and $\\partial\\ell/\\partial\\mathbf u_{n_k}=\\sigma(s_k)\\mathbf v_w$: real contexts are pulled closer, noise words pushed away.",
          "$\\partial\\ell/\\partial\\mathbf u_o=-(1-\\sigma(s_o))\\mathbf v_w$ și $\\partial\\ell/\\partial\\mathbf u_{n_k}=\\sigma(s_k)\\mathbf v_w$: contextele reale sînt atrase, cuvintele de zgomot sînt respinse."),
        t("At the optimum, $\\mathbf u_o^{\\top}\\mathbf v_w=\\mathrm{PMI}(w,o)-\\ln K$, with $\\mathrm{PMI}(w,o)=\\ln\\frac{P(w,o)}{P(w)P(o)}$ (Levy \\& Goldberg, 2014).",
          "La optim, $\\mathbf u_o^{\\top}\\mathbf v_w=\\mathrm{PMI}(w,o)-\\ln K$, cu $\\mathrm{PMI}(w,o)=\\ln\\frac{P(w,o)}{P(w)P(o)}$ (Levy \\& Goldberg, 2014)."), sep="3pt"), "footnotesize")
    D.frame(t("Appendix: the models in code", "Anexă: modelele în cod"),
            code(
            "tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train_text)\n"
            "clf = LogisticRegression(C=C, class_weight='balanced', max_iter=3000).fit(tf.transform(train_text), y)\n"
            "\n"
            "class EmbBag(nn.Module):                         # fastText-style classifier\n"
            "    def __init__(self, n_vocab, dim, n_cls):\n"
            "        super().__init__()\n"
            "        self.emb = nn.EmbeddingBag(n_vocab, dim, mode='mean')\n"
            "        self.out = nn.Linear(dim, n_cls)\n"
            "    def forward(self, ids, offsets):\n"
            "        return self.out(F.dropout(self.emb(ids, offsets), 0.3, self.training))\n"
            "# SGNS step: loss = -(logsigmoid((v*u_pos).sum(1)) + logsigmoid(-(U_neg @ v[...,None]).squeeze()).sum(1)).mean()\n"
            ) + it(t("Full code: lectures/L6/analysis\\_L6.py and the notebook of this lecture.",
                                      "Codul complet: lectures/L6/analysis\\_L6.py și notebook-ul acestui curs.")), "footnotesize")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 6, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
