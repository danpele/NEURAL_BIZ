r"""
build_L7.py -- Lecture 7 "Attention and transformers" (NN), English and Romanian from one source.
Writes lecture7.tex / lecture7_ro.tex from the bilingual slide text below and L7_numbers.json (analysis_L7.py),
compiles both and applies every check of lectures/STYLE.md (nn_build.run_build).
Run (from the repository root):
    python3 lectures/L7/analysis_L7.py
    python3 lectures/L7/make_quantlets_L7.py
    python3 lectures/L7/build_L7.py            # both languages; "en" or "ro" builds one
Hand-typed numbers are definitions only: 512-token window of BERT, 64-token cut of our tweets, 10 % validation share,
d = 64, 4 heads, 2 layers, learning rates 5e-4 and 2e-5, 3 epochs, 1,000 bootstrap draws, BERT-base sizes (12 layers,
12 heads, d = 768) as published.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, cols, fmt, heat, hier, items, reference_frames,  # noqa: E402
                      run_build, table)

ABBR = {
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "BCE": (None, "Banca Centrală Europeană"),
    "BERT": ("Bidirectional Encoder Representations from Transformers", "Bidirectional Encoder Representations from Transformers (reprezentări bidirecționale ale codificatorului transformer)"),
    "CNN": ("convolutional neural network", "rețea neuronală convoluțională"),
    "ECB": ("European Central Bank", None),
    "ES": ("expected shortfall", "pierderea așteptată în coadă (expected shortfall)"),
    "FFN": ("position-wise feed-forward network", "rețea directă aplicată fiecărei poziții (feed-forward)"),
    "FinBERT": ("BERT further trained on financial text", "BERT antrenat suplimentar pe texte financiare"),
    "GELU": ("Gaussian error linear unit", "unitate liniară cu eroare gaussiană (Gaussian error linear unit)"),
    "GPT": ("Generative Pre-trained Transformer", "Generative Pre-trained Transformer (transformer generativ preantrenat)"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "LLM": ("large language model", "model mare de limbaj"),
    "LSTM": ("long short-term memory network", "rețea cu memorie pe termen lung și scurt (long short-term memory)"),
    "MLM": ("masked language modelling", "modelarea limbajului cu cuvinte mascate (masked language modelling)"),
    "NLP": ("natural language processing", "prelucrarea limbajului natural"),
    "RNN": ("recurrent neural network", "rețea neuronală recurentă"),
    "TF-IDF": ("term frequency--inverse document frequency", "frecvența termenului--inversul frecvenței în documente (TF-IDF)"),
    "LN": ("layer normalisation", "normalizarea pe strat (layer normalisation)"),
    "MHA": ("multi-head attention", "atenția cu mai multe capete (multi-head attention)"),
    "MPS": ("Metal Performance Shaders (Apple graphics processor back end)", "Metal Performance Shaders (interfața procesorului grafic Apple)"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată (rectified linear unit)"),
    "VaR": ("Value-at-Risk", "valoarea la risc (Value-at-Risk)"),
}
IGNORE = {"CLS", "SEP", "PAD", "ICLR", "NeurIPS", "NIPS", "SSRN", "ECECSR", "IEEE", "ACM", "JASIST", "NAACL", "ACL", "II", "MIT", "PMLR",
          "ICML", "CEE", "JMLR", "PDF", "DOI", "AI", "IDA", "LLM-VaR", "LLM-ES", "NEURAL", "BIZ", "AdamW", "WordPiece", "PhraseBank",
          "EU", "TS", "BTIG", "CI", "LM", "QK", "XW", "ProsusAI", "BE", "RT", "BTC", "USD", "CEO", "IPO", "ETF", "SPAC", "GDP", "Fed", "UK", "US", "CPI", "NYSE", "ATH"}
UNITS = set()
REFS = [
    "Araci, D.\\ (2019). FinBERT: Financial sentiment analysis with pre-trained language models. \\href{https://arxiv.org/abs/1908.10063}{arXiv:1908.10063}",
    "Ba, J.\\,L., Kiros, J.\\,R.\\ \\& Hinton, G.\\,E.\\ (2016). Layer normalization. \\href{https://arxiv.org/abs/1607.06450}{arXiv:1607.06450}",
    "Bahdanau, D., Cho, K.\\ \\& Bengio, Y.\\ (2015). Neural machine translation by jointly learning to align and translate. "
    "\\textit{International Conference on Learning Representations}. \\href{https://arxiv.org/abs/1409.0473}{arXiv:1409.0473}",
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}, ch.\\ 12. Springer. "
    "\\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Clark, K., Khandelwal, U., Levy, O.\\ \\& Manning, C.\\,D.\\ (2019). What does BERT look at? An analysis of BERT's attention. "
    "\\textit{BlackboxNLP Workshop}, 276--286. \\href{https://doi.org/10.18653/v1/W19-4828}{doi:10.18653/v1/W19-4828}",
    "Devlin, J., Chang, M.-W., Lee, K.\\ \\& Toutanova, K.\\ (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. "
    "\\textit{Proceedings of NAACL}, 4171--4186. \\href{https://doi.org/10.18653/v1/N19-1423}{doi:10.18653/v1/N19-1423}",
    "Efron, B.\\ \\& Tibshirani, R.\\,J.\\ (1993). \\textit{An Introduction to the Bootstrap}. Chapman \\& Hall. "
    "\\href{https://doi.org/10.1201/9780429246593}{doi:10.1201/9780429246593}",
    "Jain, S.\\ \\& Wallace, B.\\,C.\\ (2019). Attention is not explanation. \\textit{Proceedings of NAACL}, 3543--3556. "
    "\\href{https://doi.org/10.18653/v1/N19-1357}{doi:10.18653/v1/N19-1357}",
    "Jheng, S.-L., Tak, R., Găman, Ș., Mazurencu-Marinescu-Pele, M.\\ \\& Pele, D.\\,T.\\ (2026). Can foundation models manage risk? Zero-shot VaR and ES "
    "forecasting with conformal calibration in CEE markets. \\textit{Economic Computation and Economic Cybernetics Studies and Research}, 60(2), 267--290. "
    "\\href{https://doi.org/10.24818/18423264/60.2.26.14}{doi:10.24818/18423264/60.2.26.14}",
    "Loshchilov, I.\\ \\& Hutter, F.\\ (2019). Decoupled weight decay regularization. \\textit{International Conference on Learning Representations}. "
    "\\href{https://arxiv.org/abs/1711.05101}{arXiv:1711.05101}",
    "Malo, P., Sinha, A., Korhonen, P., Wallenius, J.\\ \\& Takala, P.\\ (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. "
    "\\textit{Journal of the Association for Information Science and Technology}, 65(4), 782--796. \\href{https://doi.org/10.1002/asi.23062}{doi:10.1002/asi.23062}",
    "Pele, D.\\,T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A.\\,T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S.\\ \\& Härdle, W.\\,K.\\ (2026). "
    "In the beginning was the word: LLM-VaR and LLM-ES. SSRN Working Paper 5104383. \\href{https://doi.org/10.2139/ssrn.5104383}{doi:10.2139/ssrn.5104383}",
    "Radford, A., Narasimhan, K., Salimans, T.\\ \\& Sutskever, I.\\ (2018). Improving language understanding by generative pre-training. OpenAI technical report. "
    "\\href{https://openai.com/research/language-unsupervised}{openai.com/research/language-unsupervised}",
    "Tetlock, P.\\,C.\\ (2007). Giving content to investor sentiment: The role of media in the stock market. \\textit{Journal of Finance}, 62(3), 1139--1168. "
    "\\href{https://doi.org/10.1111/j.1540-6261.2007.01232.x}{doi:10.1111/j.1540-6261.2007.01232.x}",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.\\,N., Kaiser, Ł.\\ \\& Polosukhin, I.\\ (2017). Attention is all you need. "
    "\\textit{Advances in Neural Information Processing Systems}, 30. \\href{https://arxiv.org/abs/1706.03762}{arXiv:1706.03762}",
    "Wu, Y., Schuster, M., Chen, Z.\\ et al.\\ (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. "
    "\\href{https://arxiv.org/abs/1609.08144}{arXiv:1609.08144}",
]
IGNORE |= {"OpenAI", "BlackboxNLP"}


def it(*xs, sep="2pt"):
    def one(x):
        if isinstance(x, tuple):
            return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                    "\n".join("      \\item " + (s[0].upper() + s[1:] if s[0].islower() else s) for s in x[1:] if s) +
                    "\n    \\end{itemize}")
        return hier(x)
    return (f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{{sep}}}\n" + "\n".join("  \\item " + one(x) for x in xs if x) +
            "\n\\end{itemize}")


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L7_numbers.json"), lang)
    N = V.N
    ro = lang == "ro"
    D = Deck(7, "Attention and transformers" if not ro else "Atenția și transformerele", V, lang, heights=H)
    t = D.t
    NB = t("notebook \\S", "notebook, secțiunea ")
    _c = D.concept
    D.concept = lambda *a, **k: _c(*a, **{**{"wl": "0.58", "wr": "0.39"}, **k})
    M = N["models"]
    ECB = t("ECB", "BCE")

    def de(key):
        v = int(round(float(V.get(key))))
        return "de " if ro and (v >= 20 and (v % 100 >= 20 or v % 100 == 0)) else ""

    def n_(x, spec):
        return fmt(x, spec, lang)

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Explain why attention replaced recurrence for text: one step between any two words, computed in parallel.",
          "De ce atenția a înlocuit recurența pentru text; un singur pas între oricare două cuvinte, calculat în paralel"),
        t("Write scaled dot-product attention, multi-head attention, positional encoding and the transformer block with every symbol defined.",
          "Formulele atenției cu produs scalar scalat, ale atenției cu mai multe capete, ale codificării poziției și ale blocului transformer, cu toate simbolurile definite"),
        t("Compare, on real finance tweets, a small transformer trained from scratch, a pretrained Bidirectional Encoder Representations from Transformers (BERT) model for finance (FinBERT) and a term frequency--inverse document frequency (TF-IDF) benchmark.",
          "Compararea, pe tweet-uri financiare reale, a unui transformer mic antrenat de la zero, a unui model Bidirectional Encoder Representations from Transformers (BERT) preantrenat pentru finanțe (FinBERT) și a unui model de referință bazat pe frecvența termenului și inversul frecvenței în documente (TF-IDF)"),
        t("Read attention maps with care, and decide when pretraining, fine-tuning or a simple model is the right business choice.",
          "Citirea atentă a hărților de atenție; cînd preantrenarea, ajustarea fină sau un model simplu sînt alegerea potrivită pentru o firmă"),
        sep="5pt"))
    abbreviation_frames(D, ABBR)

    # ------------------------------------------------------------------ 1 why attention
    D.section(t("Why attention", "De ce atenția"))
    D.concept(t("Financial news moves prices; machines must read it fast", "Știrile financiare mișcă prețurile; mașinile trebuie să le citească repede"),
              t("Thousands of headlines and posts arrive every hour; a desk needs their tone in seconds, consistently, in the same units.",
                "Mii de titluri și postări sosesc în fiecare oră; o mesă de tranzacționare are nevoie de tonul lor în cîteva secunde, consecvent, pe aceeași scară."),
              "\\[ \\hat{\\mathbf p}(x_{1:n})=\\softmax\\big(W_o\\,\\vh(x_{1:n})+\\vb_o\\big)\\in\\Delta^{3} \\]"
              "\\where{" + t("$x_{1:n}$ = the tokens of one text; $\\vh$ = a fixed-size representation of the text; $\\Delta^3$ = probabilities of bearish, bullish, neutral; "
                             "the whole lecture is about how to compute $\\vh$.",
                             "$x_{1:n}$ = tokenii unui text; $\\vh$ = o reprezentare de dimensiune fixă a textului; $\\Delta^3$ = probabilitățile pentru negativ (bearish), pozitiv (bullish), neutru; "
                             "tot cursul este despre cum se calculează $\\vh$.") + "}",
              it((t("\\textbf{Our data}", "\\textbf{Datele noastre}"),
                  t("@{data.n_train_pub|,d} labelled finance tweets for training, @{data.n_test|,d} for the test", "@{data.n_train_pub|,d} de tweet-uri financiare etichetate pentru antrenare, @{data.n_test|,d} pentru test"),
                  t("test: @{data.pct_bearish_test|.1f}\\,\\% bearish, @{data.pct_bullish_test|.1f}\\,\\% bullish, @{data.pct_neutral_test|.1f}\\,\\% neutral",
                    "test: @{data.pct_bearish_test|.1f}\\,\\% negative, @{data.pct_bullish_test|.1f}\\,\\% pozitive, @{data.pct_neutral_test|.1f}\\,\\% neutre")),
                 (t("\\textbf{Evidence}", "\\textbf{Dovezi}"),
                  t("media pessimism predicts next-day returns and volume (Tetlock, 2007)", "pesimismul din presă anticipează randamentele și volumul din ziua următoare (Tetlock, 2007)")), sep="3pt"),
              t("a sentiment model is a measurement instrument: it must be accurate on the rare bearish news, not only on average.",
                "un model de sentiment este un instrument de măsură: trebuie să fie precis pe știrile negative rare, nu doar în medie."),
              cite="(Tetlock, 2007; Malo et al., 2014)")

    D.concept(t("Recurrent networks read word by word and forget", "Rețelele recurente citesc cuvînt cu cuvînt și uită"),
              t("A recurrent network squeezes everything read so far into one state vector, updated one token at a time.",
                "O rețea recurentă comprimă tot ce a citit pînă acum într-un singur vector de stare, actualizat token cu token."),
              "\\[ \\vh_t=\\phi\\big(W_h\\vh_{t-1}+W_x\\vx_t+\\vb\\big),\\qquad \\frac{\\partial\\vh_n}{\\partial\\vh_1}=\\prod_{t=2}^{n}\\frac{\\partial\\vh_t}{\\partial\\vh_{t-1}} \\]"
              "\\where{" + t("$\\vx_t$ = embedding of token $t$; $\\vh_t$ = state; $\\phi$ = tanh; the product of $n-1$ Jacobians shrinks or explodes with the distance between words.",
                             "$\\vx_t$ = reprezentarea tokenului $t$; $\\vh_t$ = starea; $\\phi$ = tanh; produsul a $n-1$ matrice jacobiene scade sau explodează odată cu distanța dintre cuvinte.") + "}",
              it((t("\\textbf{Two costs}", "\\textbf{Două costuri}"),
                  t("path between words $i$ and $j$: $|i-j|$ steps, so long links fade", "drumul dintre cuvintele $i$ și $j$: $|i-j|$ pași, deci legăturile lungi se estompează"),
                  t("$n$ sequential steps: no parallelism over the sentence", "$n$ pași succesivi: nicio paralelizare pe propoziție")),
                 (t("\\textbf{Gates help, not cure}", "\\textbf{Porțile ajută, dar nu vindecă}"),
                  t("long short-term memory (LSTM) of Lecture 5 eases but keeps the sequential path", "rețeaua cu memorie pe termen lung și scurt (LSTM) din cursul 5 atenuează, dar păstrează drumul secvențial")), sep="3pt"),
              t("in ``profit fell, but guidance was raised'', the verdict depends on a word far from the subject.",
                "în ``profitul a scăzut, dar prognoza a fost majorată'', verdictul depinde de un cuvînt aflat departe de subiect."),
              cite="(Bahdanau, Cho \\& Bengio, 2015; Bishop \\& Bishop, 2024, sec.\\ 12.1)")

    D.chart(t("Tweets are short; central-bank speeches are far beyond a 512-token window", "Tweet-urile sînt scurte; discursurile băncilor centrale depășesc mult fereastra de 512 tokeni"),
            "context_lengths", [
                t("\\textbf{Median length in tokens}: tweets @{len.tw_median}, Financial PhraseBank sentences @{len.pb_median}, European Central Bank (ECB) speeches @{len.sp_median|,d}.",
                  "\\textbf{Lungimea mediană în tokeni}: tweet-uri @{len.tw_median}, propoziții Financial PhraseBank @{len.pb_median}, discursuri ale Băncii Centrale Europene (BCE) @{len.sp_median|,d}"),
                t("\\textbf{Context limit}: @{len.sp_pct_over512|.1f}\\,\\% of the @{len.n_speeches} speeches exceed 512 tokens; only @{len.tw_pct_over64|.2f}\\,\\% of the tweets exceed 64.",
                  "\\textbf{Limita de context}: @{len.sp_pct_over512|.1f}\\,\\% dintre cele @{len.n_speeches} de discursuri depășesc 512 tokeni; doar @{len.tw_pct_over64|.2f}\\,\\% dintre tweet-uri depășesc 64")],
            height="0.52\\textheight", tryit=NB + "1",
            job=t("match the model to the text length: a tweet fits any transformer, a speech needs chunking or a long-context model.",
                  "potriviți modelul cu lungimea textului: un tweet încape în orice transformer, un discurs cere împărțire sau un model cu context lung."))

    # ------------------------------------------------------------------ 2 attention
    D.section(t("The attention mechanism", "Mecanismul atenției"))
    D.concept(t("Attention is a soft dictionary lookup", "Atenția este o căutare aproximativă într-un dicționar"),
              t("Each word asks a question (query), every word offers a label (key) and a content (value); the answer is the values weighted by how well keys match the query.",
                "Fiecare cuvînt pune o întrebare (query), fiecare cuvînt oferă o etichetă (key) și un conținut (value); răspunsul este media valorilor, ponderată după potrivirea cheilor cu întrebarea."),
              "\\[ \\vy_i=\\sum_{j=1}^{n}a_{ij}\\,\\mathbf v_j,\\qquad a_{ij}=\\frac{\\exp\\big(\\mathbf q_i^{\\top}\\mathbf k_j/\\sqrt{d_k}\\big)}{\\sum_{l=1}^{n}\\exp\\big(\\mathbf q_i^{\\top}\\mathbf k_l/\\sqrt{d_k}\\big)} \\]"
              "\\[ \\mathbf q_i=W_Q^{\\top}\\vx_i,\\quad \\mathbf k_j=W_K^{\\top}\\vx_j,\\quad \\mathbf v_j=W_V^{\\top}\\vx_j \\]"
              "\\where{" + t("$\\vx_i\\in\\R^{d}$ = token representation; $W_Q,W_K\\in\\R^{d\\times d_k}$, $W_V\\in\\R^{d\\times d_v}$ = learned projections; "
                             "$a_{ij}\\ge0$, $\\sum_j a_{ij}=1$: a probability over the words.",
                             "$\\vx_i\\in\\R^{d}$ = reprezentarea tokenului; $W_Q,W_K\\in\\R^{d\\times d_k}$, $W_V\\in\\R^{d\\times d_v}$ = proiecții învățate; "
                             "$a_{ij}\\ge0$, $\\sum_j a_{ij}=1$: o distribuție de probabilitate pe cuvinte.") + "}",
              it((t("\\textbf{Hard lookup}", "\\textbf{Căutarea exactă}"),
                  t("a dictionary returns the value of the one matching key", "un dicționar întoarce valoarea singurei chei potrivite")),
                 (t("\\textbf{Soft lookup}", "\\textbf{Căutarea aproximativă}"),
                  t("attention returns a blend of all values: differentiable, so $W_Q,W_K,W_V$ are learned", "atenția întoarce un amestec al tuturor valorilor: derivabil, deci $W_Q,W_K,W_V$ se învață"),
                  t("one step between any two words, whatever their distance", "un singur pas între oricare două cuvinte, oricare ar fi distanța")), sep="3pt"),
              t("each output word is a context-aware mixture: ``guidance'' can borrow meaning from ``raised'' ten words later.",
                "fiecare cuvînt de ieșire este un amestec care ține cont de context: ``prognoza'' poate împrumuta sens de la ``majorată'', aflat zece cuvinte mai departe."),
              tryit=NB + "2", cite="(Vaswani et al., 2017; Bishop \\& Bishop, 2024, sec.\\ 12.1)")

    D.concept(t("In matrix form: one multiplication attends all words at once", "În formă matriceală: o singură înmulțire calculează atenția pentru toate cuvintele"),
              t("Stacking queries, keys and values as matrices turns attention into two matrix products and a row-wise softmax.",
                "Așezînd întrebările, cheile și valorile în matrice, atenția devine două produse de matrice și un softmax pe rînduri."),
              "\\[ \\operatorname{Attention}(Q,K,V)=\\softmax\\Big(\\frac{QK^{\\top}}{\\sqrt{d_k}}+\\mathbf M\\Big)V \\]\\[ Q=XW_Q,\\quad K=XW_K,\\quad V=XW_V \\]"
              "\\where{" + t("$X\\in\\R^{n\\times d}$ = the $n$ token representations; $QK^{\\top}\\in\\R^{n\\times n}$ = all pairwise scores; "
                             "$\\mathbf M$ = mask, $0$ for allowed pairs and $-\\infty$ for padding (or for future tokens in a decoder).",
                             "$X\\in\\R^{n\\times d}$ = cele $n$ reprezentări ale tokenilor; $QK^{\\top}\\in\\R^{n\\times n}$ = toate scorurile perechilor; "
                             "$\\mathbf M$ = masca, $0$ pentru perechile permise și $-\\infty$ pentru completare (sau pentru tokenii viitori, într-un decodor).") + "}",
              it((t("\\textbf{Cost per layer}", "\\textbf{Costul pe strat}"),
                  "$O(n^2d)$ " + t("operations and $O(n^2)$ memory", "operații și memorie $O(n^2)$"),
                  t("a median speech costs about @{len.cost_ratio|,d} times a median tweet", "un discurs median costă de circa @{len.cost_ratio|,d} de ori mai mult decît un tweet median")),
                 (t("\\textbf{Parallel}", "\\textbf{Paralel}"),
                  t("no loop over positions: graphics processors multiply the whole sentence at once", "fără buclă pe poziții: procesoarele grafice înmulțesc toată propoziția deodată")), sep="3pt"),
              t("the $n^2$ term is why context windows are limited and long reports must be cut into pieces.",
                "termenul $n^2$ explică de ce ferestrele de context sînt limitate și rapoartele lungi trebuie tăiate în bucăți."),
              tryit=NB + "2", cite="(Vaswani et al., 2017)")

    D.concept(t("Why divide by $\\sqrt{d_k}$: keep the softmax out of saturation", "De ce împărțim la $\\sqrt{d_k}$: ferim softmax-ul de saturație"),
              t("Dot products of long random vectors grow with their length; without scaling the softmax becomes one-hot and its gradient vanishes.",
                "Produsele scalare ale unor vectori aleatori lungi cresc odată cu lungimea lor; fără scalare, softmax-ul devine one-hot și gradientul dispare."),
              "\\[ q_l,k_l\\iid(0,1)\\ \\Rightarrow\\ \\E[\\mathbf q^{\\top}\\mathbf k]=0,\\quad \\Var(\\mathbf q^{\\top}\\mathbf k)=\\sum_{l=1}^{d_k}\\Var(q_lk_l)=d_k \\]"
              "\\[ \\Var\\big(\\mathbf q^{\\top}\\mathbf k/\\sqrt{d_k}\\big)=1\\quad\\text{" + t("for every", "pentru orice") + "}\\ d_k \\]"
              "\\where{" + t("$d_k$ = dimension of queries and keys (64 per head in BERT); with unit-variance scores the softmax spreads its mass and its Jacobian $\\operatorname{diag}(\\mathbf a)-\\mathbf a\\mathbf a^{\\top}$ stays away from zero.",
                             "$d_k$ = dimensiunea întrebărilor și cheilor (64 pe cap în BERT); cu scoruri de varianță unu, softmax-ul își împarte masa, iar jacobianul $\\operatorname{diag}(\\mathbf a)-\\mathbf a\\mathbf a^{\\top}$ nu se apropie de zero.") + "}",
              it((t("\\textbf{Without scaling}", "\\textbf{Fără scalare}"),
                  t("with $d_k=64$ the scores have standard deviation 8, so one word takes almost all the weight", "cu $d_k=64$ scorurile au abaterea standard 8, deci un singur cuvînt ia aproape toată ponderea")),
                 (t("\\textbf{Temperature view}", "\\textbf{Interpretare ca temperatură}"),
                  t("$\\sqrt{d_k}$ is a fixed temperature that keeps attention soft at initialisation", "$\\sqrt{d_k}$ este o temperatură fixă care păstrează atenția difuză la inițializare")), sep="3pt"),
              t("a design constant, not a tuning knob: it is what lets deep stacks of attention train at all.",
                "o constantă de proiectare, nu un parametru de calibrat: datorită ei se pot antrena stive adînci de straturi de atenție."),
              cite="(Vaswani et al., 2017)")

    D.concept(t("Several heads look for several relations at once", "Mai multe capete caută simultan mai multe relații"),
              t("One head averages one kind of relation; several heads with smaller projections capture syntax, negation and entities in parallel.",
                "Un cap mediază un singur tip de relație; mai multe capete, cu proiecții mai mici, surprind în paralel sintaxa, negația și entitățile."),
              "\\[ \\operatorname{MHA}(X)=\\big[\\mathbf H_1\\,\\|\\,\\cdots\\,\\|\\,\\mathbf H_h\\big]W_O \\]\\[ \\mathbf H_m=\\operatorname{Attention}(XW_Q^{m},XW_K^{m},XW_V^{m}) \\]"
              "\\where{" + t("$h$ = number of heads; $W_Q^m,W_K^m,W_V^m\\in\\R^{d\\times d/h}$; $\\|$ = concatenation; $W_O\\in\\R^{d\\times d}$; parameters: $4d^2$ (+ biases), the same as one big head.",
                             "$h$ = numărul de capete; $W_Q^m,W_K^m,W_V^m\\in\\R^{d\\times d/h}$; $\\|$ = concatenare; $W_O\\in\\R^{d\\times d}$; parametri: $4d^2$ (plus termenii liberi), ca la un singur cap mare.") + "}",
              it((t("\\textbf{BERT-base}", "\\textbf{BERT-base}"),
                  t("@{attn.layers} layers $\\times$ @{attn.heads} heads, $d=$ @{attn.d_model}, $d/h=$ @{attn.d_head}", "@{attn.layers} straturi $\\times$ @{attn.heads} capete, $d=$ @{attn.d_model}, $d/h=$ @{attn.d_head}")),
                 (t("\\textbf{Our small transformer}", "\\textbf{Transformerul nostru mic}"),
                  t("2 layers $\\times$ 4 heads, $d=64$, $d/h=16$", "2 straturi $\\times$ 4 capete, $d=64$, $d/h=16$")), sep="3pt"),
              t("heads are cheap specialists: same cost as one head, more kinds of context per layer.",
                "capetele sînt specialiști ieftini: același cost ca un singur cap, mai multe tipuri de context pe strat."),
              tryit=NB + "2", cite="(Vaswani et al., 2017; Clark et al., 2019)")

    D.chart(t("What FinBERT attends to in a real bearish tweet", "La ce este atent FinBERT într-un tweet negativ real"), "attention_example", [
        t("\\textbf{Tweet} (test set, label bearish): ``@{example.text}''; pretrained FinBERT gives it only @{example.p_neg_pct|.1f}\\,\\% probability of negative: the bearish tone is implicit.",
          "\\textbf{Tweet-ul} (setul de test, etichetă negativă): ``@{example.text}''; FinBERT preantrenat îi dă doar @{example.p_neg_pct|.1f}\\,\\% probabilitate de a fi negativ: tonul negativ este implicit"),
        t("\\textbf{Layer 6}: the [CLS] token puts @{example.cls_top_pct|.1f}\\,\\% of its attention on ``@{example.cls_top_token}''; in layer 12 most weight flows to the separator tokens.",
          "\\textbf{Stratul 6}: tokenul [CLS] pune @{example.cls_top_pct|.1f}\\,\\% din atenție pe ``@{example.cls_top_token}''; în stratul 12, cea mai mare parte a ponderii merge spre tokenii separatori")],
        height="0.56\\textheight", tryit=NB + "4", cite=t("(own computation; mean over the 12 heads)", "(calcule proprii; media pe cele 12 capete)"))

    D.chart(t("Across layers, attention sharpens and parks on separator tokens", "De la un strat la altul, atenția se concentrează și se oprește pe tokenii separatori"), "attention_entropy", [
        t("\\textbf{Entropy} of the attention rows, divided by $\\log n$: @{attn.ent_l1|.2f} in layer 1, lowest @{attn.ent_min|.2f} in layer @{attn.ent_min_layer} (1 = uniform).",
          "\\textbf{Entropia} rîndurilor de atenție, împărțită la $\\log n$: @{attn.ent_l1|.2f} în stratul 1, minimul @{attn.ent_min|.2f} în stratul @{attn.ent_min_layer} (1 = uniformă)"),
        t("\\textbf{[SEP] as a sink}: up to @{attn.sep_max_pct|.1f}\\,\\% of all attention (layer @{attn.sep_max_layer}) goes to the separator: a ``no-op'' a head uses when it has nothing to add.",
          "\\textbf{[SEP] ca destinație implicită}: pînă la @{attn.sep_max_pct|.1f}\\,\\% din toată atenția (stratul @{attn.sep_max_layer}) merge spre separator, o operație nulă pe care un cap o folosește cînd nu are nimic de adăugat")],
        height="0.52\\textheight", tryit=NB + "4", cite=t("(own computation, @{attn.n_tweets} test tweets; Clark et al., 2019)", "(calcule proprii, @{attn.n_tweets} de tweet-uri de test; Clark et al., 2019)"))

    D.concept(t("Attention weights are a clue, not an explanation", "Ponderile de atenție sînt un indiciu, nu o explicație"),
              t("Where a head looks is not the same as why the model decides: values, later layers and the classifier all transform what attention selects.",
                "Unde se uită un cap nu este același lucru cu motivul deciziei: valorile, straturile următoare și clasificatorul transformă tot ce selectează atenția."),
              "\\[ \\vh_i^{(l+1)}=\\vh_i^{(l)}+\\sum_{m=1}^{h}\\sum_{j}a_{ij}^{(l,m)}\\,W_O^{m\\top}W_V^{m\\top}\\vh_j^{(l)}+\\dots \\]"
              "\\where{" + t("the contribution of word $j$ is $a_{ij}$ times a value vector whose norm can be large or small; the residual stream $\\vh_i^{(l)}$ carries information that attention never touches.",
                             "contribuția cuvîntului $j$ este $a_{ij}$ înmulțit cu un vector de valori a cărui normă poate fi mare sau mică; fluxul rezidual $\\vh_i^{(l)}$ duce informație pe care atenția nu o atinge.") + "}",
              it((t("\\textbf{Evidence}", "\\textbf{Dovezi}"),
                  t("different attention maps can give the same prediction (Jain \\& Wallace, 2019)", "hărți de atenție diferite pot da aceeași predicție (Jain \\& Wallace, 2019)"),
                  t("heavy attention on [SEP] carries almost no information (Clark et al., 2019)", "atenția mare pe [SEP] nu poartă aproape nicio informație (Clark et al., 2019)")),
                 (t("\\textbf{Better tools}", "\\textbf{Instrumente mai bune}"),
                  t("gradient-based attributions and occlusion tests (Lecture 11)", "atribuiri bazate pe gradient și teste de ocluzie (cursul 11)")), sep="3pt"),
              t("never show an attention map to a regulator as ``the reason'' for a decision.",
                "nu prezentați niciodată unui supraveghetor o hartă de atenție drept ``motivul'' unei decizii."),
              cite="(Jain \\& Wallace, 2019; Clark et al., 2019)")

    # ------------------------------------------------------------------ 3 transformer block
    D.section(t("The transformer architecture", "Arhitectura transformer"))
    D.chart(t("Attention ignores order; positional encoding puts it back", "Atenția ignoră ordinea; codificarea poziției o readuce"), "positional_encoding", [
        t("\\textbf{Formula}: $p_{t,2i}=\\sin(t/10000^{2i/d})$, $p_{t,2i+1}=\\cos(t/10000^{2i/d})$, added to the token embedding; low dimensions change fast, high ones slowly.",
          "\\textbf{Formula}: $p_{t,2i}=\\sin(t/10000^{2i/d})$, $p_{t,2i+1}=\\cos(t/10000^{2i/d})$, adunată la reprezentarea tokenului; dimensiunile mici se schimbă repede, cele mari încet"),
        t("\\textbf{Why}: self-attention is permutation-equivariant; without positions ``rates cut, stocks up'' equals ``stocks cut, rates up''.",
          "\\textbf{De ce}: auto-atenția este echivariantă la permutări; fără poziții, ``dobînzile scad, acțiunile cresc'' este totuna cu ``acțiunile scad, dobînzile cresc''")],
        height="0.48\\textheight", tryit=NB + "3", cite="\\simlabel\\ (Vaswani et al., 2017)")

    D.concept(t("A transformer block: attention, feed-forward, residuals, normalisation", "Un bloc transformer: atenție, rețea directă, conexiuni reziduale, normalizare"),
              t("Attention mixes information across words; a small two-layer network then transforms each word on its own; residual links and layer normalisation keep training stable.",
                "Atenția amestecă informația între cuvinte; o rețea mică cu două straturi transformă apoi fiecare cuvînt separat; conexiunile reziduale și normalizarea pe strat stabilizează antrenarea."),
              "\\[ Z=X+\\operatorname{MHA}\\big(\\operatorname{LN}(X)\\big),\\qquad X'=Z+\\operatorname{FFN}\\big(\\operatorname{LN}(Z)\\big) \\]"
              "\\[ \\operatorname{FFN}(\\vz)=W_2\\,\\operatorname{GELU}(W_1\\vz+\\vb_1)+\\vb_2,\\quad \\operatorname{LN}(\\vz)=\\boldsymbol\\gamma\\odot\\frac{\\vz-\\mu_{\\vz}}{\\sigma_{\\vz}}+\\boldsymbol\\beta \\]"
              "\\where{" + t("pre-norm variant; position-wise feed-forward network (FFN), $W_1\\in\\R^{4d\\times d}$; Gaussian error linear unit (GELU) or rectified linear unit (ReLU); "
                             "layer normalisation (LN) over the $d$ features of one token; parameters per block $\\approx12d^2$.",
                             "varianta cu normalizare înainte; rețeaua directă aplicată fiecărei poziții (FFN), $W_1\\in\\R^{4d\\times d}$; unitatea liniară cu eroare gaussiană (GELU) sau unitatea liniară rectificată (ReLU); "
                             "normalizarea pe strat (LN), pe cele $d$ variabile ale unui token; parametri pe bloc $\\approx12d^2$.") + "}",
              it((t("\\textbf{Residuals}", "\\textbf{Conexiunile reziduale}"),
                  t("gradients reach early layers directly (Lecture 4)", "gradientul ajunge direct la primele straturi (cursul 4)")),
                 (t("\\textbf{Our block}", "\\textbf{Blocul nostru}"),
                  t("$d=64$, FFN width 128, dropout 0.1; 2 blocks: @{models.tiny.params_noemb|,d} parameters outside the embeddings",
                    "$d=64$, lățimea FFN 128, dropout 0,1; 2 blocuri: @{models.tiny.params_noemb|,d} de parametri în afara reprezentărilor")), sep="3pt"),
              t("most parameters of a transformer sit in the feed-forward layers, not in attention.",
                "cei mai mulți parametri ai unui transformer se află în straturile directe, nu în atenție."),
              tryit=NB + "3", cite="(Vaswani et al., 2017; Ba, Kiros \\& Hinton, 2016)")

    D.concept(t("Encoders read, decoders write: two pretraining tasks", "Codificatoarele citesc, decodoarele scriu: două sarcini de preantrenare"),
              t("The same block reads or writes, depending on the mask and the pretraining task.",
                "Același bloc devine cititor sau scriitor în funcție de mască și de sarcina de preantrenare."),
              "\\[ \\loss_{\\text{MLM}}=-\\sum_{i\\in\\mathcal M}\\log p_{\\btheta}\\big(x_i\\mid x_{\\setminus\\mathcal M}\\big) \\]\\[ "
              "\\loss_{\\text{LM}}=-\\sum_{t=1}^{n}\\log p_{\\btheta}\\big(x_t\\mid x_{<t}\\big) \\]"
              "\\where{" + t("$\\mathcal M$ = 15\\,\\% of positions, masked at random: masked language modelling (MLM); "
                             "second line: next-token prediction, causal mask $M_{ij}=-\\infty$ for $j>i$.",
                             "$\\mathcal M$ = 15\\,\\% dintre poziții, mascate aleator: modelarea limbajului cu cuvinte mascate (MLM), bidirecțională; "
                             "dreapta: prezicerea tokenului următor, cu mască cauzală, $M_{ij}=-\\infty$ pentru $j>i$.") + "}",
              it((t("\\textbf{Encoder-only}", "\\textbf{Doar codificator}"),
                  t("Bidirectional Encoder Representations from Transformers (BERT): classification, extraction", "Bidirectional Encoder Representations from Transformers (BERT): clasificare, extragere de informații")),
                 (t("\\textbf{Decoder-only}", "\\textbf{Doar decodor}"),
                  t("Generative Pre-trained Transformer (GPT): text generation, today's large language models (LLM)", "Generative Pre-trained Transformer (GPT): generare de text, modelele mari de limbaj (LLM) de azi")),
                 sep="3pt"),
              t("for labelling texts use an encoder; for writing reports use a decoder (Lecture 12).",
                "pentru etichetarea textelor folosiți un codificator; pentru redactarea rapoartelor, un decodor (cursul 12)."),
              cite="(Devlin et al., 2019; Radford et al., 2018)")

    D.concept(t("Text becomes numbers through subword tokens", "Textul devine numere prin tokeni de subcuvinte"),
              t("A fixed vocabulary of word pieces covers any word, including tickers and new company names, by splitting rare words into frequent pieces.",
                "Un vocabular fix de bucăți de cuvinte acoperă orice cuvînt, inclusiv simbolurile bursiere și numele noi de firme, împărțind cuvintele rare în bucăți frecvente."),
              "\\[ x_{1:n}=\\big[\\texttt{[CLS]},w_1,\\dots,w_{n-2},\\texttt{[SEP]}\\big],\\qquad \\vx_t=E[x_t]+\\mathbf p_t+\\mathbf s_t \\]"
              "\\where{" + t("$E\\in\\R^{|\\mathcal V|\\times d}$ = embedding table; $\\mathbf p_t$ = position, $\\mathbf s_t$ = segment embedding; [CLS] summarises the text, [SEP] closes it; "
                             "WordPiece vocabulary of FinBERT: $|\\mathcal V|=$ @{tok.vocab|,d}.",
                             "$E\\in\\R^{|\\mathcal V|\\times d}$ = tabelul reprezentărilor; $\\mathbf p_t$ = poziția, $\\mathbf s_t$ = segmentul; [CLS] rezumă textul, [SEP] îl închide; "
                             "vocabularul WordPiece al FinBERT: $|\\mathcal V|=$ @{tok.vocab|,d}.") + "}",
              it((t("\\textbf{Our tweets}", "\\textbf{Tweet-urile noastre}"),
                  t("median @{len.tw_median} tokens, 99th percentile @{len.tw_p99}, maximum @{len.tw_max}", "mediana @{len.tw_median} de tokeni, percentila 99 este @{len.tw_p99}, maximul @{len.tw_max}"),
                  t("we cut at 64 tokens: @{len.tw_pct_over64|.2f}\\,\\% of the tweets lose a few tokens", "tăiem la 64 de tokeni: @{len.tw_pct_over64|.2f}\\,\\% dintre tweet-uri pierd cîțiva tokeni")),
                 (t("\\textbf{Embeddings dominate small models}", "\\textbf{Reprezentările domină modelele mici}"),
                  t("small transformer: @{models.tiny.params|,d} parameters, of which @{models.tiny.params_noemb|,d} outside the table", "transformerul mic: @{models.tiny.params|,d} de parametri, dintre care @{models.tiny.params_noemb|,d} în afara tabelului")), sep="3pt"),
              t("check how a tokenizer splits your tickers and product names before trusting a model on them.",
                "verificați cum împarte tokenizatorul simbolurile bursiere și numele produselor înainte de a avea încredere în model."),
              tryit=NB + "1", cite="(Wu et al., 2016; Devlin et al., 2019)")

    D.frame(t("Attention, recurrence and convolution compared", "Atenția, recurența și convoluția comparate"), table(
        "@{}p{0.26\\textwidth}p{0.20\\textwidth}p{0.22\\textwidth}p{0.22\\textwidth}@{}",
        [t("Layer type", "Tipul stratului"), t("Cost per layer", "Cost pe strat"), t("Sequential steps", "Pași secvențiali"), t("Longest path", "Drumul cel mai lung")],
        [[t("Self-attention", "Auto-atenție"), "$O(n^2 d)$", "$O(1)$", "$O(1)$"],
         [t("Recurrent (RNN, LSTM)", "Recurent (RNN, LSTM)"), "$O(n d^2)$", "$O(n)$", "$O(n)$"],
         [t("Convolution, kernel $k$", "Convoluție, nucleu $k$"), "$O(k n d^2)$", "$O(1)$", "$O(\\log_k n)$"]],
        size="footnotesize", stretch="1.3") + "\n\\vspace{2mm}\n" + it(
        t("\\textbf{Recurrent neural network (RNN)}: cheap per step, slow per sentence, long paths; \\textbf{convolutional neural network (CNN)}: parallel, but needs depth to link distant words.",
          "\\textbf{Rețeaua neuronală recurentă (RNN)}: ieftină pe pas, lentă pe propoziție, drumuri lungi; \\textbf{rețeaua neuronală convoluțională (CNN)}: paralelă, dar are nevoie de adîncime ca să lege cuvinte îndepărtate"),
        t("\\textbf{Self-attention}: constant path and full parallelism, paid with $n^2$ memory; for tweets ($n\\approx$ @{len.tw_median}) the quadratic term is negligible.",
          "\\textbf{Auto-atenția}: drum constant și paralelism complet, plătite cu memorie $n^2$; pentru tweet-uri ($n\\approx$ @{len.tw_median}) termenul pătratic este neglijabil")),
        "footnotesize", cite="(Vaswani et al., 2017, Table 1)")

    # ------------------------------------------------------------------ 4 case study
    D.section(t("Case study: sentiment of finance tweets", "Studiu de caz: sentimentul tweet-urilor financiare"))
    D.frame(t("Seven models, one held-out test set", "Șapte modele, un singur set de test păstrat deoparte"), table(
        "@{}p{0.30\\textwidth}p{0.62\\textwidth}@{}",
        [t("Model", "Model"), t("What is trained", "Ce se antrenează")],
        [[t("TF-IDF + logistic", "TF-IDF + logistic"), t("logistic regression on word 1--2-grams (@{tfidf.n_features|,d} features), $C$ chosen on validation",
                                                          "regresie logistică pe secvențe de 1--2 cuvinte (@{tfidf.n_features|,d} de variabile), $C$ ales pe validare")],
         [t("Embedding bag", "Medie de embedding-uri"), t("mean of learned token embeddings, no attention", "media reprezentărilor tokenilor, fără atenție")],
         [t("Small transformer", "Transformer mic"), t("2 blocks, 4 heads, $d=64$, from scratch; also without positions", "2 blocuri, 4 capete, $d=64$, de la zero; și fără poziții")],
         [t("FinBERT zero-shot", "FinBERT fără antrenare"), t("BERT further trained on financial text (FinBERT, Araci, 2019), used as is", "BERT antrenat suplimentar pe texte financiare (FinBERT, Araci, 2019), folosit ca atare")],
         [t("FinBERT frozen + logistic", "FinBERT înghețat + logistic"), t("logistic regression on the mean last-layer vector (@{probe.dim})", "regresie logistică pe vectorul mediu din ultimul strat (@{probe.dim})")],
         [t("FinBERT fine-tuned", "FinBERT ajustat"), t("all @{models.finbert_ft.params_m|.1f} million weights, 3 epochs", "toate cele @{models.finbert_ft.params_m|.1f} milioane de ponderi, 3 epoci")]],
        size="scriptsize") + "\n\\vspace{1mm}\n" + it(
        t("\\textbf{Split}: published training tweets = @{data.n_train|,d} training + @{data.n_val|,d} validation (10\\,\\%, stratified); the published validation set = our test (@{data.n_test|,d}), used once.",
          "\\textbf{Împărțirea}: tweet-urile de antrenare publicate = @{data.n_train|,d} de antrenare + @{data.n_val|,d} de validare (10\\,\\%, stratificat); setul de validare publicat = testul nostru (@{data.n_test|,d}), folosit o singură dată"),
        t("\\textbf{Why not Financial PhraseBank}: FinBERT was fine-tuned on it, so testing there would leak; tweets are new to every model.",
          "\\textbf{De ce nu Financial PhraseBank}: FinBERT a fost ajustat pe aceste date, deci un test acolo ar fi contaminat; tweet-urile sînt noi pentru toate modelele")) +
        D.bottom(None, NB + "5"), "footnotesize", cite="(Araci, 2019; Malo et al., 2014)")

    D.concept(t("Our small transformer, trained from scratch", "Transformerul nostru mic, antrenat de la zero"),
              t("Two blocks on top of token and position embeddings, mean-pooled over the real tokens, then a linear layer to the three classes.",
                "Două blocuri peste reprezentările tokenilor și ale pozițiilor, media pe tokenii reali, apoi un strat liniar spre cele trei clase."),
              "\\[ \\vh=\\frac{1}{n}\\sum_{t=1}^{n}\\vx_t^{(2)},\\qquad \\loss=-\\frac1N\\sum_{i=1}^{N}w_{y_i}\\log\\softmax(W\\vh_i+\\vb)_{y_i} \\]"
              "\\where{" + t("$\\vx_t^{(2)}$ = output of the second block for token $t$; $w_c$ = class weight, inversely proportional to the class frequency; "
                             "Adam with decoupled weight decay (AdamW), learning rate $5\\cdot10^{-4}$, batch 64, early stopping on validation macro-F1.",
                             "$\\vx_t^{(2)}$ = ieșirea celui de-al doilea bloc pentru tokenul $t$; $w_c$ = ponderea clasei, invers proporțională cu frecvența ei; "
                             "Adam cu penalizare decuplată a ponderilor (AdamW), rata de învățare $5\\cdot10^{-4}$, lot 64, oprire timpurie după macro-F1 pe validare.") + "}",
              it((t("\\textbf{Size}", "\\textbf{Dimensiunea}"),
                  t("@{models.tiny.params|,d} parameters, mostly the embedding table", "@{models.tiny.params|,d} de parametri, mai ales tabelul reprezentărilor")),
                 (t("\\textbf{Cost}", "\\textbf{Costul}"),
                  t("@{models.tiny.sec_per_seed|.0f} s per seed; median @{models.tiny.epochs_median} epochs before stopping", "@{models.tiny.sec_per_seed|.0f} s pe sămînță; în mediană @{models.tiny.epochs_median} epoci pînă la oprire")),
                 (t("\\textbf{Macro-F1}", "\\textbf{Macro-F1}"),
                  t("mean of the three per-class F1 scores: bearish counts as much as neutral", "media celor trei scoruri F1 pe clase: clasa negativă contează cît cea neutră")), sep="3pt"),
              t("macro-F1 is the right yardstick when the rare class (bearish news) is the one that costs money.",
                "macro-F1 este măsura potrivită cînd clasa rară (știrile negative) este cea care costă bani."),
              tryit=NB + "5", cite="(Loshchilov \\& Hutter, 2019)")

    D.chart(t("The small transformer overfits: the validation loss turns up after a few epochs", "Transformerul mic supraînvață: pierderea pe validare crește după cîteva epoci"),
            "tiny_learning_curve", [
                t("\\textbf{Seed 1}: the validation loss is lowest at epoch @{models.tiny.minloss_epoch_seed1} (@{models.tiny.minloss_seed1|.3f}) and rises to @{models.tiny.lastloss_seed1|.3f} by epoch @{models.tiny.epochs_seed1}; the macro-F1 used for stopping still peaks at epoch @{models.tiny.best_epoch_seed1}.",
                  "\\textbf{Sămînța 1}: pierderea pe validare este minimă la epoca @{models.tiny.minloss_epoch_seed1} (@{models.tiny.minloss_seed1|.3f}) și urcă la @{models.tiny.lastloss_seed1|.3f} pînă la epoca @{models.tiny.epochs_seed1}; macro-F1 folosit pentru oprire atinge totuși maximul la epoca @{models.tiny.best_epoch_seed1}"),
                t("\\textbf{Data-hungry}: @{data.n_train|,d} tweets are few for @{models.tiny.params|,d} parameters learned from nothing; the model grows overconfident before it stops improving.",
                  "\\textbf{Are nevoie de multe date}: @{data.n_train|,d} de tweet-uri sînt puține pentru @{models.tiny.params|,d} de parametri învățați de la zero; modelul devine prea sigur pe el înainte să se oprească din progres")],
            height="0.50\\textheight", tryit=NB + "5")

    D.chart(t("Pretraining wins: fine-tuned FinBERT leads, the small transformer trails TF-IDF", "Preantrenarea cîștigă: FinBERT ajustat conduce, transformerul mic rămîne sub TF-IDF"), "model_comparison", [
        t("\\textbf{Macro-F1 on @{data.n_test|,d} test tweets}: FinBERT fine-tuned @{models.finbert_ft.macro_f1|.1f}\\,\\%, TF-IDF @{models.tfidf.macro_f1|.1f}\\,\\%, small transformer @{models.tiny.macro_f1|.1f}\\,\\% (sd @{models.tiny.macro_f1_sd|.1f}, 3 seeds).",
          "\\textbf{Macro-F1 pe @{data.n_test|,d} de tweet-uri de test}: FinBERT ajustat @{models.finbert_ft.macro_f1|.1f}\\,\\%, TF-IDF @{models.tfidf.macro_f1|.1f}\\,\\%, transformerul mic @{models.tiny.macro_f1|.1f}\\,\\% (abatere standard @{models.tiny.macro_f1_sd|.1f}, 3 semințe)"),
        t("\\textbf{FinBERT without training on tweets}: @{models.finbert_zs.macro_f1|.1f}\\,\\%; frozen FinBERT + logistic: @{models.finbert_probe.macro_f1|.1f}\\,\\%.",
          "\\textbf{FinBERT fără antrenare pe tweet-uri}: @{models.finbert_zs.macro_f1|.1f}\\,\\%; FinBERT înghețat + logistic: @{models.finbert_probe.macro_f1|.1f}\\,\\%")],
        height="0.56\\textheight", tryit=NB + "5")

    rows = []
    names = {"tfidf": "TF-IDF + logistic", "embbag": t("Embedding bag", "Medie embedding-uri"), "tiny_nopos": t("Transformer, no positions", "Transformer fără poziții"),
             "tiny": t("Small transformer", "Transformer mic"), "finbert_zs": t("FinBERT zero-shot", "FinBERT fără antrenare"),
             "finbert_probe": t("FinBERT frozen + logistic", "FinBERT înghețat + logistic"), "finbert_ft": t("FinBERT fine-tuned", "FinBERT ajustat")}
    keys = list(names)
    rng = {c: (min(M[k][c] for k in keys), max(M[k][c] for k in keys)) for c in ("macro_f1", "acc", "rec_bearish", "prec_bearish")}
    for k in keys:
        rows.append([names[k]] + [heat(M[k][c], *rng[c], good_low=False, shown=n_(M[k][c], ".1f")) for c in ("macro_f1", "acc", "rec_bearish", "prec_bearish")] +
                    [str(M[k]["n_runs"])])
    D.frame(t("The bearish class is where models differ most", "Clasa negativă este cea în care modelele diferă cel mai mult"), table(
        "@{}lrrrrr@{}", [t("Model", "Model"), "Macro-F1", t("Accuracy", "Acuratețe"), t("Bearish recall", "Recall negativ"), t("Bearish precision", "Precizie negativ"), t("Runs", "Rulări")],
        rows, size="scriptsize", stretch="1.15") + "\n\\vspace{1.5mm}\n" + it(
        t("\\textbf{All numbers in \\%}, test tweets; green = best, red = worst in the column; means over the runs.",
          "\\textbf{Toate valorile în \\%}, pe tweet-urile de test; verde = cel mai bun, roșu = cel mai slab din coloană; medii pe rulări"),
        t("\\textbf{Bearish news}: fine-tuned FinBERT finds @{models.finbert_ft.rec_bearish|.1f}\\,\\% of them with @{models.finbert_ft.prec_bearish|.1f}\\,\\% precision; TF-IDF @{models.tfidf.rec_bearish|.1f}\\,\\% and @{models.tfidf.prec_bearish|.1f}\\,\\%.",
          "\\textbf{Știrile negative}: FinBERT ajustat găsește @{models.finbert_ft.rec_bearish|.1f}\\,\\% dintre ele, cu o precizie de @{models.finbert_ft.prec_bearish|.1f}\\,\\%; TF-IDF @{models.tfidf.rec_bearish|.1f}\\,\\% și @{models.tfidf.prec_bearish|.1f}\\,\\%")) +
        D.bottom(t("report per-class recall and precision: an average can hide a model that misses the news that matter.",
                   "raportați recall-ul și precizia pe clase: o medie poate ascunde un model care ratează știrile importante."), NB + "5"),
        "footnotesize", cite=t("(own computation)", "(calcule proprii)"))

    B = N["boot"]
    D.concept(t("Are the differences real? Paired bootstrap of the test tweets", "Sînt diferențele reale? Bootstrap pereche pe tweet-urile de test"),
              t("Resampling the test tweets, with the same draw for both models, shows how much a gap in macro-F1 could move by chance.",
                "Reeșantionînd tweet-urile de test, cu aceeași extragere pentru ambele modele, vedem cît s-ar putea mișca întîmplător o diferență de macro-F1."),
              "\\[ \\Delta^{*b}=F_1^{A}(\\mathcal T^{*b})-F_1^{B}(\\mathcal T^{*b}),\\quad b=1,\\dots,1000 \\]\\[ \\text{CI}_{95\\%}=[q_{0.025},q_{0.975}] \\]"
              "\\where{" + t("$\\mathcal T^{*b}$ = test tweets drawn with replacement; $F_1$ = macro-F1 averaged over the seeds of the model; differences in percentage points.",
                             "$\\mathcal T^{*b}$ = tweet-uri de test extrase cu întoarcere; $F_1$ = macro-F1 mediat pe semințele modelului; diferențe în puncte procentuale.") + "}",
              it((t("\\textbf{FinBERT fine-tuned $-$ TF-IDF}", "\\textbf{FinBERT ajustat $-$ TF-IDF}"),
                  "@{boot.ft_tfidf.mean|+.1f} [@{boot.ft_tfidf.lo|+.1f}; @{boot.ft_tfidf.hi|+.1f}]"),
                 (t("\\textbf{Small transformer $-$ TF-IDF}", "\\textbf{Transformer mic $-$ TF-IDF}"),
                  "@{boot.tiny_tfidf.mean|+.1f} [@{boot.tiny_tfidf.lo|+.1f}; @{boot.tiny_tfidf.hi|+.1f}]"),
                 (t("\\textbf{With $-$ without positions}", "\\textbf{Cu $-$ fără poziții}"),
                  "@{boot.tiny_nopos.mean|+.1f} [@{boot.tiny_nopos.lo|+.1f}; @{boot.tiny_nopos.hi|+.1f}]"),
                 (t("\\textbf{Frozen FinBERT $-$ TF-IDF}", "\\textbf{FinBERT înghețat $-$ TF-IDF}"),
                  "@{boot.probe_tfidf.mean|+.1f} [@{boot.probe_tfidf.lo|+.1f}; @{boot.probe_tfidf.hi|+.1f}]"), sep="2pt"),
              t("an interval that excludes zero is the minimum evidence before replacing a production model.",
                "un interval care nu conține zero este dovada minimă înainte de a înlocui un model aflat în producție."),
              tryit=NB + "5", cite="(Efron \\& Tibshirani, 1993)")

    D.chart(t("With little labelled data, pretrained representations matter most", "Cu puține date etichetate, reprezentările preantrenate contează cel mai mult"), "data_scaling", [
        t("\\textbf{With @{scaling.n_small} training tweets}: frozen FinBERT + logistic @{scaling.probe_small|.1f}\\,\\%, TF-IDF @{scaling.tfidf_small|.1f}\\,\\%, small transformer @{scaling.tiny_small|.1f}\\,\\% macro-F1.",
          "\\textbf{Cu @{scaling.n_small} de tweet-uri de antrenare}: FinBERT înghețat + logistic @{scaling.probe_small|.1f}\\,\\%, TF-IDF @{scaling.tfidf_small|.1f}\\,\\%, transformerul mic @{scaling.tiny_small|.1f}\\,\\% macro-F1"),
        t("\\textbf{Pretraining = data you did not label}: billions of words already shaped the FinBERT vectors; a model from scratch must learn language and task at once.",
          "\\textbf{Preantrenarea = date pe care nu le-ați etichetat}: miliarde de cuvinte au format deja vectorii FinBERT; un model de la zero trebuie să învețe deodată limba și sarcina")],
        height="0.52\\textheight", tryit=NB + "6", cite=t("(own computation, seed 1)", "(calcule proprii, sămînța 1)"))

    D.chart(t("Fine-tuning fixes what the zero-shot model gets wrong", "Ajustarea fină corectează ce greșește modelul fără antrenare"), "confusion", [
        t("\\textbf{Zero-shot}: FinBERT was trained on news sentences, not tweets; it finds @{cm_zs.bear_ok} of the @{data.n_bearish_test} bearish tweets; after fine-tuning: @{cm.bear_ok}.",
          "\\textbf{Fără antrenare}: FinBERT a fost antrenat pe propoziții din știri, nu pe tweet-uri; găsește @{cm_zs.bear_ok} dintre cele @{data.n_bearish_test} de tweet-uri negative; după ajustare: @{cm.bear_ok}"),
        t("\\textbf{Remaining errors}: @{cm.bear_as_neutral} bearish and @{cm.bull_as_neutral} bullish tweets read as neutral; only @{cm.bear_as_bull} bearish read as bullish.",
          "\\textbf{Erorile rămase}: @{cm.bear_as_neutral} tweet-uri negative și @{cm.bull_as_neutral} pozitive citite ca neutre; doar @{cm.bear_as_bull} negative citite ca pozitive")],
        height="0.52\\textheight", tryit=NB + "5", cite=t("(own computation, fine-tuned model of seed 1)", "(calcule proprii, modelul ajustat cu sămînța 1)"))

    D.concept(t("Fine-tuning: start from pretrained weights, take small steps", "Ajustarea fină: pornim de la ponderile preantrenate și facem pași mici"),
              t("All weights move, but only a little: the language knowledge stays, the decision layer adapts to tweets.",
                "Toate ponderile se mișcă, dar puțin: cunoștințele de limbă rămîn, stratul de decizie se adaptează la tweet-uri."),
              "\\[ \\btheta_0=\\btheta_{\\text{FinBERT}},\\qquad \\btheta_{s+1}=\\btheta_s-\\eta_s\\,\\widehat{\\nabla\\loss}(\\btheta_s),\\quad \\eta_s=\\eta_0\\Big(1-\\frac{s}{S}\\Big) \\]"
              "\\where{" + t("$\\eta_0=2\\cdot10^{-5}$, linear decay over $S$ = 3 epochs of batches of 32; AdamW with weight decay 0.01; the epoch with the best validation macro-F1 is kept.",
                             "$\\eta_0=2\\cdot10^{-5}$, scădere liniară pe $S$ = 3 epoci cu loturi de 32; AdamW cu penalizarea ponderilor 0,01; se păstrează epoca cu cel mai bun macro-F1 pe validare.") + "}",
              it((t("\\textbf{Gain}", "\\textbf{Cîștigul}"),
                  t("@{gain_ft_vs_zs|+.1f} points over zero-shot, @{gain_ft_vs_tfidf|+.1f} over TF-IDF", "@{gain_ft_vs_zs|+.1f} puncte față de varianta fără antrenare, @{gain_ft_vs_tfidf|+.1f} față de TF-IDF")),
                 (t("\\textbf{Cost}", "\\textbf{Costul}"),
                  t("@{models.finbert_ft.params_m|.1f} million weights, @{models.finbert_ft.min_per_seed|.1f} minutes per seed on a laptop graphics processor", "@{models.finbert_ft.params_m|.1f} milioane de ponderi, @{models.finbert_ft.min_per_seed|.1f} minute pe sămînță pe procesorul grafic al unui laptop")),
                 (t("\\textbf{Risk}", "\\textbf{Riscul}"),
                  t("a large learning rate erases pretraining (catastrophic forgetting)", "o rată de învățare mare șterge preantrenarea (uitare catastrofală)")), sep="3pt"),
              t("fine-tuning a pretrained encoder is today's default for any labelled text task with a few thousand examples.",
                "ajustarea fină a unui codificator preantrenat este azi alegerea implicită pentru orice sarcină pe text etichetat, cu cîteva mii de exemple."),
              tryit=NB + "5", cite="(Devlin et al., 2019; Araci, 2019)")

    D.frame(t("Does word order matter for tweets?", "Contează ordinea cuvintelor în tweet-uri?"), cols(
        it((t("\\textbf{Ablation}", "\\textbf{Experimentul de eliminare}"),
            t("same small transformer with and without positional encoding, 3 seeds each", "același transformer mic, cu și fără codificarea poziției, cîte 3 semințe"),
            t("macro-F1 @{models.tiny.macro_f1|.1f}\\,\\% with positions, @{models.tiny_nopos.macro_f1|.1f}\\,\\% without: @{gain_pos|+.1f} points",
              "macro-F1 @{models.tiny.macro_f1|.1f}\\,\\% cu poziții, @{models.tiny_nopos.macro_f1|.1f}\\,\\% fără: @{gain_pos|+.1f} puncte")),
           (t("\\textbf{Bootstrap}", "\\textbf{Bootstrap}"),
            t("95\\,\\% interval of the gain: [@{boot.tiny_nopos.lo|+.1f}; @{boot.tiny_nopos.hi|+.1f}] points", "intervalul de 95\\,\\% al cîștigului: [@{boot.tiny_nopos.lo|+.1f}; @{boot.tiny_nopos.hi|+.1f}] puncte")), sep="5pt"),
        it((t("\\textbf{Reading}", "\\textbf{Interpretare}"),
            t("short, formulaic tweets are mostly bags of signal words (``cut'', ``beats'', ``downgrade'')", "tweet-urile scurte și formulaice sînt mai ales colecții de cuvinte-semnal (``cut'', ``beats'', ``downgrade'')"),
            t("the embedding bag, with no attention, is close: @{models.embbag.macro_f1|.1f}\\,\\%", "media de embedding-uri, fără atenție, este aproape: @{models.embbag.macro_f1|.1f}\\,\\%")),
           (t("\\textbf{When order matters}", "\\textbf{Cînd contează ordinea}"),
            t("negation and contrast (``not'', ``but''), long documents, numbers attached to the right entity", "negația și contrastul (``not'', ``but''), documentele lungi, cifrele legate de entitatea corectă")), sep="5pt"),
        "0.49", "0.48") + D.bottom(t("test the simple version of your model first: on short texts much of the transformer's power is unused.",
                                     "testați mai întîi versiunea simplă a modelului: pe texte scurte o mare parte din puterea transformerului rămîne nefolosită."), NB + "5"),
        "footnotesize", cite=t("(own computation)", "(calcule proprii)"))

    # ------------------------------------------------------------------ 5 practice
    D.section(t("Transformers in practice", "Transformerele în practică"))
    D.frame(t("From sentiment scores to a business signal", "De la scorurile de sentiment la un semnal de afaceri"), cols(
        it((t("\\textbf{Pipeline}", "\\textbf{Fluxul de lucru}"),
            t("stream of news and posts $\\to$ tokenizer $\\to$ fine-tuned encoder $\\to$ class probabilities", "fluxul de știri și postări $\\to$ tokenizator $\\to$ codificator ajustat $\\to$ probabilitățile claselor"),
            t("aggregate by firm and day: share of bearish minus bullish, weighted by reach", "agregare pe firmă și zi: ponderea mesajelor negative minus pozitive, ponderată cu audiența")),
           (t("\\textbf{Uses}", "\\textbf{Utilizări}"),
            t("risk monitoring of a credit portfolio, early warning for issuers, an input to Value-at-Risk (VaR) overlays",
              "monitorizarea riscului unui portofoliu de credite, avertizare timpurie pentru emitenți, o intrare pentru ajustarea valorii la risc (VaR)")), sep="5pt"),
        it((t("\\textbf{Pitfalls}", "\\textbf{Capcane}"),
            t("look-ahead: use only texts time-stamped before the decision", "informația din viitor: folosiți doar texte cu marcaj de timp anterior deciziei"),
            t("domain shift: a model of news sentences misreads tweets (zero-shot gap)", "schimbarea domeniului: un model pentru propoziții din știri citește greșit tweet-urile (diferența față de varianta fără antrenare)"),
            t("label noise and sarcasm; drift when the market vocabulary changes", "zgomotul din etichete și sarcasmul; derivă cînd se schimbă vocabularul pieței")), sep="5pt"), "0.49", "0.48") +
        D.bottom(t("a sentiment score is an input to a model, not a trading decision: backtest it before use.",
                   "un scor de sentiment este o intrare într-un model, nu o decizie de tranzacționare: testați-l retrospectiv înainte de folosire."), None),
        "footnotesize", cite="(Tetlock, 2007)")

    D.frame(t("Long documents: central-bank speeches and annual reports", "Documentele lungi: discursurile băncilor centrale și rapoartele anuale"), cols(
        it((t("\\textbf{Our speeches}", "\\textbf{Discursurile noastre}"),
            t("@{len.n_speeches} speeches of the " + ECB + ", @{len.sp_first|date} to @{len.sp_last|date}", "@{len.n_speeches} de discursuri ale BCE, @{len.sp_first|date} -- @{len.sp_last|date}"),
            t("median @{len.sp_median|,d} tokens, longest @{len.sp_max|,d}", "mediana @{len.sp_median|,d} de tokeni, cel mai lung @{len.sp_max|,d}")),
           (t("\\textbf{Quadratic cost}", "\\textbf{Costul pătratic}"),
            t("attention on a median speech costs about @{len.cost_ratio|,d} times that on a median tweet", "atenția pe un discurs median costă de circa @{len.cost_ratio|,d} de ori mai mult decît pe un tweet median")), sep="5pt"),
        it((t("\\textbf{Options}", "\\textbf{Variante}"),
            t("chunk into windows of 512 and pool the chunk scores", "împărțire în ferestre de 512 și agregarea scorurilor pe bucăți"),
            t("hierarchical models: sentence vectors, then attention over sentences", "modele ierarhice: vectori de propoziții, apoi atenție pe propoziții"),
            t("long-context models: sparse or memory-efficient attention", "modele cu context lung: atenție rară sau eficientă ca memorie")), sep="5pt"), "0.49", "0.48") +
        D.bottom(t("for a monetary-policy desk, chunking plus a well-tested encoder usually beats an untested long-context model.",
                   "pentru o echipă de politică monetară, împărțirea în bucăți plus un codificator bine testat bate de obicei un model cu context lung netestat."), NB + "1"),
        "footnotesize", cite="(Vaswani et al., 2017)")

    D.frame(t("Transformers beyond text", "Transformerele dincolo de text"), cols(
        it((t("\\textbf{Time series}", "\\textbf{Serii de timp}"),
            t("patches of a series as tokens; pretrained forecasting models (Lecture 13)", "fragmente ale seriei ca tokeni; modele de prognoză preantrenate (cursul 13)")),
           (t("\\textbf{Tables and transactions}", "\\textbf{Tabele și tranzacții}"),
            t("each column or each past payment as a token; attention across them", "fiecare coloană sau fiecare plată anterioară ca token; atenție între ele")), sep="5pt"),
        it((t("\\textbf{Graphs}", "\\textbf{Grafuri}"),
            t("attention restricted to the edges = graph attention network (Lecture 9)", "atenția restrînsă la muchii = rețeaua cu atenție pe graf (cursul 9)")),
           (t("\\textbf{Images and documents}", "\\textbf{Imagini și documente}"),
            t("image patches as tokens; vision-language models read charts and reports (Lecture 14)", "fragmente de imagine ca tokeni; modelele viziune-limbaj citesc grafice și rapoarte (cursul 14)")), sep="5pt"), "0.49", "0.48") +
        D.bottom(t("whatever the data, the question is the same: what are the tokens, and which pairs may attend to each other?",
                   "oricare ar fi datele, întrebarea este aceeași: care sînt tokenii și ce perechi au voie să-și acorde atenție?"), None),
        "footnotesize", cite="(Bishop \\& Bishop, 2024, ch.\\ 12)")

    # ------------------------------------------------------------------ wrap-up
    D.section(t("Wrap-up", "Concluzii"))
    D.frame(t("What to remember", "Ce reținem"), it(
        t("\\textbf{Attention} = softmax of scaled query-key scores times values; one step between any two words, $O(n^2)$ cost; $\\sqrt{d_k}$ keeps the softmax trainable.",
          "\\textbf{Atenția} = softmax al scorurilor întrebare-cheie scalate, înmulțit cu valorile; un singur pas între oricare două cuvinte, cost $O(n^2)$; $\\sqrt{d_k}$ păstrează softmax-ul antrenabil"),
        t("\\textbf{Transformer block} = multi-head attention + feed-forward, each with a residual link and layer normalisation; positions are added, because attention ignores order.",
          "\\textbf{Blocul transformer} = atenție cu mai multe capete + rețea directă, fiecare cu conexiune reziduală și normalizare; pozițiile se adaugă, pentru că atenția ignoră ordinea"),
        t("\\textbf{On finance tweets} pretraining decides: fine-tuned FinBERT @{models.finbert_ft.macro_f1|.1f}\\,\\% macro-F1, TF-IDF @{models.tfidf.macro_f1|.1f}\\,\\%, a transformer from scratch @{models.tiny.macro_f1|.1f}\\,\\%.",
          "\\textbf{Pe tweet-urile financiare} preantrenarea decide: FinBERT ajustat @{models.finbert_ft.macro_f1|.1f}\\,\\% macro-F1, TF-IDF @{models.tfidf.macro_f1|.1f}\\,\\%, un transformer de la zero @{models.tiny.macro_f1|.1f}\\,\\%"),
        t("\\textbf{Attention maps} are clues, not explanations; check per-class errors and bootstrap intervals before trusting a model.",
          "\\textbf{Hărțile de atenție} sînt indicii, nu explicații; verificați erorile pe clase și intervalele bootstrap înainte de a avea încredere într-un model"),
        sep="6pt"))

    D.frame(t("Research from our group: transformers as risk forecasters", "Cercetările grupului nostru: transformerele ca instrumente de prognoză a riscului"), it(
        t("\\textbf{Large language models (LLM) as tail-risk forecasters} (Pele, Bolovăneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann \\& Härdle, 2026): uses decoder transformers, "
          "prompted with return histories, to forecast Value-at-Risk and the expected shortfall (ES) --- LLM-VaR and LLM-ES --- and evaluates them with standard risk backtests.",
          "\\textbf{Modele mari de limbaj (LLM) pentru prognoza riscului extrem} (Pele, Bolovăneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann \\& Härdle, 2026); lucrarea folosește transformere decodor, "
          "cărora li se dau istoricele randamentelor, pentru prognoza valorii la risc și a pierderii așteptate în coadă (ES) --- LLM-VaR și LLM-ES --- și le evaluează cu testele retrospective standard"),
        t("\\textbf{Foundation models for tail risk} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026): uses pretrained transformer forecasting models, without retraining and with conformal calibration, "
          "to forecast VaR and ES in Central and Eastern European markets.",
          "\\textbf{Modele fundaționale pentru riscul extrem} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026); lucrarea folosește modele transformer de prognoză preantrenate, fără reantrenare și cu calibrare conformală, "
          "pentru prognoza VaR și ES pe piețele din Europa Centrală și de Est"),
        t("\\textbf{Link with this lecture}: the same idea as FinBERT, a model pretrained on large data and used with little or no task-specific training, applied to risk numbers instead of words.",
          "\\textbf{Legătura cu acest curs}: aceeași idee ca la FinBERT, un model preantrenat pe date multe și folosit cu puțină sau fără antrenare specifică, aplicată măsurilor de risc în loc de cuvinte"),
        sep="5pt") + "\n\\vspace{1mm}{\\scriptsize " + t("Findings as stated in the papers; the numbers on the other slides come from our own computations.",
                                                     "Rezultatele de pe acest slide sînt cele din lucrări; cifrele de pe celelalte slide-uri provin din calculele noastre.") + "}",
        "footnotesize", cite=t("(Bucharest University of Economic Studies; Romanian Academy, Institute for Economic Forecasting)",
                               "(Academia de Studii Economice din București; Academia Română, Institutul de Prognoză Economică)"))

    D.frame(t("Want to learn more?", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Vaswani et al.\\ (2017)}, ``Attention is all you need'': the original transformer.", "\\textbf{Vaswani et al.\\ (2017)}, ``Attention is all you need'': transformerul original"),
        t("\\textbf{Bishop \\& Bishop (2024)}, chapter 12: transformers, from attention to large language models.", "\\textbf{Bishop \\& Bishop (2024)}, capitolul 12: transformerele, de la atenție la modelele mari de limbaj"),
        t("\\textbf{The Illustrated Transformer} (J.\\ Alammar): ", "\\textbf{The Illustrated Transformer} (J.\\ Alammar): ") + "\\href{https://jalammar.github.io/illustrated-transformer/}{jalammar.github.io}.", sep="4pt"),
        it(t("\\textbf{Hugging Face course}, fine-tuning encoders: ", "\\textbf{Cursul Hugging Face}, ajustarea codificatoarelor: ") + "\\href{https://huggingface.co/learn/llm-course}{huggingface.co/learn}.",
           t("\\textbf{This course}: code, data, notebooks --- ", "\\textbf{Acest curs}: cod, date, notebook-uri --- ") + "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Experiment yourself}: the notebook reproduces every chart and number of this lecture.",
             "\\textbf{Exercițiu practic}: notebook-ul reproduce toate graficele și cifrele acestui curs"), sep="4pt"), "0.52", "0.45"))

    # ------------------------------------------------------------------ quiz
    D.quiz_frame(t("scaling", "scalarea"), t("Why are attention scores divided by $\\sqrt{d_k}$?", "De ce se împart scorurile de atenție la $\\sqrt{d_k}$?"),
                 t("With unit-variance entries the dot product $\\mathbf q^{\\top}\\mathbf k$ has variance $d_k$; dividing by $\\sqrt{d_k}$ brings it back to 1, so the softmax does not saturate and its gradient does not vanish.",
                   "Cu elemente de varianță unu, produsul scalar $\\mathbf q^{\\top}\\mathbf k$ are varianța $d_k$; împărțirea la $\\sqrt{d_k}$ o readuce la 1, deci softmax-ul nu se saturează și gradientul nu dispare"))
    D.quiz_frame(t("positions", "pozițiile"), t("What happens to a transformer without positional encoding?", "Ce se întîmplă cu un transformer fără codificarea poziției?"),
                 t("Self-attention is permutation-equivariant, so the model sees a bag of tokens; on our short tweets this cost only @{gain_pos|.1f} points of macro-F1, but order matters for negation and long texts.",
                   "Auto-atenția este echivariantă la permutări, deci modelul vede o colecție de tokeni; pe tweet-urile noastre scurte costul a fost doar @{gain_pos|.1f} puncte de macro-F1, dar ordinea contează pentru negație și texte lungi"))
    D.quiz_frame(t("pretraining", "preantrenarea"), t("Why does a transformer trained from scratch on @{data.n_train|,d} tweets lose to TF-IDF?", "De ce pierde în fața TF-IDF un transformer antrenat de la zero pe @{data.n_train|,d} de tweet-uri?"),
                 t("It must learn language and task from few examples with @{models.tiny.params|,d} parameters; pretrained FinBERT brings that knowledge and reaches @{models.finbert_ft.macro_f1|.1f}\\,\\% after fine-tuning.",
                   "Trebuie să învețe limba și sarcina din puține exemple, cu @{models.tiny.params|,d} de parametri; FinBERT preantrenat aduce aceste cunoștințe și ajunge la @{models.finbert_ft.macro_f1|.1f}\\,\\% după ajustare"))
    D.quiz_frame(t("encoder or decoder", "codificator sau decodor"), t("A bank wants to label 10,000 complaints a day by topic. Encoder or decoder transformer?", "O bancă vrea să eticheteze după subiect 10.000 de reclamații pe zi. Transformer codificator sau decodor?"),
                 t("An encoder (BERT-type), fine-tuned on labelled complaints: it reads the whole text in both directions and outputs class probabilities, cheaply and consistently.",
                   "Un codificator (de tip BERT), ajustat pe reclamații etichetate: citește tot textul în ambele direcții și produce probabilitățile claselor, ieftin și consecvent"))
    D.quiz_frame(t("attention maps", "hărțile de atenție"), t("Can an attention map justify a decision to a regulator?", "Poate o hartă de atenție să justifice o decizie în fața unui supraveghetor?"),
                 t("No: different maps can give the same prediction, and much attention goes to [SEP] (up to @{attn.sep_max_pct|.1f}\\,\\% here); use attribution methods and tests instead.",
                   "Nu: hărți diferite pot da aceeași predicție, iar multă atenție merge spre [SEP] (pînă la @{attn.sep_max_pct|.1f}\\,\\% aici); se folosesc metode de atribuire și teste"))

    reference_frames(D, REFS, per_frame=8)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.parts.append("\\section*{" + t("Appendix", "Anexă") + "}\n")
    D.frame(t("Appendix A: gradient of the softmax and saturation", "Anexa A: gradientul softmax-ului și saturația"), cols(
        "\\[ a_j=\\frac{e^{s_j}}{\\sum_l e^{s_l}},\\qquad \\frac{\\partial a_j}{\\partial s_i}=a_j(\\delta_{ij}-a_i) \\]"
        "\\[ J=\\operatorname{diag}(\\mathbf a)-\\mathbf a\\mathbf a^{\\top},\\qquad \\|J\\|\\to0\\ \\text{" + t("when", "cînd") + "}\\ \\mathbf a\\to\\mathbf e_k \\]",
        it(t("If one score dominates, $\\mathbf a$ is almost one-hot and every entry of $J$ is close to zero: the queries and keys stop learning.",
             "Dacă un scor domină, $\\mathbf a$ este aproape one-hot și toate elementele lui $J$ sînt aproape zero: întrebările și cheile nu mai învață"),
           t("Scores with standard deviation $\\sigma$ make domination likely when $\\sigma$ is large; scaling by $\\sqrt{d_k}$ fixes $\\sigma=1$ at initialisation.",
             "Scorurile cu abaterea standard $\\sigma$ fac dominarea probabilă cînd $\\sigma$ este mare; scalarea cu $\\sqrt{d_k}$ fixează $\\sigma=1$ la inițializare"), sep="3pt"), "0.50", "0.47"),
        "small", cite="(Vaswani et al., 2017)")
    D.frame(t("Appendix B: parameters of a transformer block", "Anexa B: parametrii unui bloc transformer"), cols(
        "\\[ \\underbrace{4d^2+4d}_{\\text{" + t("attention", "atenție") + "}}+\\underbrace{8d^2+5d}_{\\text{FFN}}+\\underbrace{4d}_{\\text{LN}}\\ \\approx\\ 12d^2 \\]",
        it(t("Attention: $W_Q,W_K,W_V,W_O\\in\\R^{d\\times d}$ with biases; FFN: $W_1\\in\\R^{4d\\times d}$, $W_2\\in\\R^{d\\times4d}$; two layer norms with $\\boldsymbol\\gamma,\\boldsymbol\\beta$.",
             "Atenția: $W_Q,W_K,W_V,W_O\\in\\R^{d\\times d}$, cu termeni liberi; FFN: $W_1\\in\\R^{4d\\times d}$, $W_2\\in\\R^{d\\times4d}$; două normalizări cu $\\boldsymbol\\gamma,\\boldsymbol\\beta$"),
           t("BERT-base, $d=768$: about 7 million per block, 85 million in 12 blocks, plus 23 million in the embedding table (as published).",
             "BERT-base, $d=768$: circa 7 milioane pe bloc, 85 de milioane în 12 blocuri, plus 23 de milioane în tabelul reprezentărilor (după datele publicate)"), sep="3pt"), "0.50", "0.47"),
        "small", cite="(Devlin et al., 2019)")
    D.frame(t("Appendix C: settings of the experiments", "Anexa C: setările experimentelor"), table(
        "@{}p{0.27\\textwidth}p{0.66\\textwidth}@{}",
        [t("Model", "Model"), t("Settings", "Setări")],
        [["TF-IDF", t("1--2-grams, minimum 2 documents, sublinear term frequency; logistic regression with balanced class weights, $C$ = @{tfidf.C} chosen on validation",
                      "secvențe de 1--2 cuvinte, minimum 2 documente, frecvență sublineară; regresie logistică cu ponderi de clasă echilibrate, $C$ = @{tfidf.C} ales pe validare")],
         [t("Small transformer", "Transformer mic"), t("WordPiece tokens of FinBERT, 64 tokens; $d=64$, 4 heads, 2 pre-norm blocks, FFN 128, dropout 0.1; AdamW $5\\cdot10^{-4}$, batch 64, at most 20 epochs, patience 4",
                                                       "tokenii WordPiece ai FinBERT, 64 de tokeni; $d=64$, 4 capete, 2 blocuri cu normalizare înainte, FFN 128, dropout 0,1; AdamW $5\\cdot10^{-4}$, lot 64, cel mult 20 de epoci, răbdare 4")],
         [t("FinBERT fine-tuned", "FinBERT ajustat"), t("ProsusAI checkpoint, its own 3-class head re-mapped to bearish/bullish/neutral; AdamW $2\\cdot10^{-5}$, linear decay, batch 32, 3 epochs; seeds 1--3",
                                                         "versiunea ProsusAI, cu stratul de 3 clase reasociat claselor negativ/pozitiv/neutru; AdamW $2\\cdot10^{-5}$, scădere liniară, lot 32, 3 epoci; semințele 1--3")],
         [t("Frozen FinBERT", "FinBERT înghețat"), t("mean of the last hidden layer over the real tokens; logistic regression, $C$ = @{probe.C}", "media ultimului strat ascuns pe tokenii reali; regresie logistică, $C$ = @{probe.C}")],
         [t("Hardware", "Echipament"), t("Apple-silicon laptop, graphics processor, Metal Performance Shaders (MPS); total run @{runtime_min|.1f} minutes", "laptop cu procesor Apple, procesorul grafic, Metal Performance Shaders (MPS); rularea completă @{runtime_min|.1f} minute")]],
        size="scriptsize"), "small")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 7, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
