r"""
build_L12.py -- Lecture 12 "Large language models (LLM) in finance and prompt engineering" (NN course), EN + RO.
Writes lecture12.tex / lecture12_ro.tex from the slide text below and L12_numbers.json (analysis_L12.py), compiles both and
runs every check of lectures/STYLE.md (nn_build.run_build).
Run:  python3 lectures/L12/analysis_L12.py && python3 lectures/L12/make_quantlets_L12.py && python3 lectures/L12/build_L12.py
Hand-typed numbers are definitions only (quantile levels, window of 250 days, temperatures, chunk sizes fixed a priori).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, block, cols, fmt, heat, hier, items,  # noqa: E402
                      reference_frames, run_build, table)

ABBR = {
    "AI": ("artificial intelligence", "inteligență artificială"),
    "API": ("application programming interface", "interfață de programare a aplicațiilor"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "BCE": (None, "Banca Centrală Europeană"),
    "BERT": ("bidirectional encoder representations from transformers", "reprezentări bidirecționale ale codificatorului transformer"),
    "BM25": ("Best Matching 25 ranking function", "funcția de ordonare Best Matching 25"),
    "BPE": ("byte-pair encoding", "codificare prin perechi de octeți (byte-pair encoding)"),
    "DPO": ("direct preference optimisation", "optimizarea directă a preferințelor"),
    "ECB": ("European Central Bank", None),
    "ECE": ("expected calibration error", "eroarea de calibrare așteptată"),
    "ES": ("expected shortfall", "pierderea așteptată în coada distribuției (expected shortfall)"),
    "EU": ("European Union", None),
    "UE": (None, "Uniunea Europeană"),
    "FinBERT": ("BERT model adapted to financial text", "model BERT adaptat textelor financiare"),
    "GARCH": ("generalised autoregressive conditional heteroskedasticity", "heteroscedasticitate condiționată autoregresivă generalizată"),
    "GPT": ("generative pre-trained transformer", "transformer generativ preantrenat"),
    "HS": ("historical simulation", "simulare istorică"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "KL": ("Kullback--Leibler divergence", "divergența Kullback--Leibler"),
    "LLM": ("large language model", "model lingvistic mare"),
    "LoRA": ("low-rank adaptation", "adaptare de rang mic (low-rank adaptation)"),
    "LR": ("likelihood ratio", "raportul de verosimilitate"),
    "MRR": ("mean reciprocal rank", "rangul reciproc mediu"),
    "RAG": ("retrieval-augmented generation", "generare augmentată prin regăsire (retrieval-augmented generation)"),
    "RLHF": ("reinforcement learning from human feedback", "învățare prin recompensă din feedback uman"),
    "SFT": ("supervised fine-tuning", "ajustare fină supervizată"),
    "TF-IDF": ("term frequency--inverse document frequency", "frecvența termenului--frecvența inversă în documente"),
    "VaR": ("Value-at-Risk", "valoarea expusă la risc (Value-at-Risk)"),
}
UNITS = set()
IGNORE = {"BAAI", "RRF", "BIZ", "NEURAL", "EUR", "LL", "LLM-ES", "LLaMA", "NLP", "PPL", "RRF", "MPS", "PICBE", "ESWA", "NAJEF", "BM", "II", "OK", "NB", "ICLR", "NeurIPS", "ACL", "EMNLP", "JASIST", "TMLR", "DE", "LU", "QA"}

REFS = [
    "Amza, A., Bolovăneanu, V.\\ \\& Pele, D.\\,T.\\ (2026). Inflation forecasting with monetary policy narratives: Evidence from an "
    "emerging economy. \\textit{Proceedings of the International Conference on Business Excellence}, 20(1), 75--83. "
    "\\href{https://doi.org/10.2478/picbe-2026-0012}{doi:10.2478/picbe-2026-0012}",
    "Brown, T.\\,B.\\ et al.\\ (2020). Language models are few-shot learners. \\textit{Advances in Neural Information Processing Systems}, 33. "
    "\\href{https://arxiv.org/abs/2005.14165}{arXiv:2005.14165}",
    "Guo, C., Pleiss, G., Sun, Y.\\ \\& Weinberger, K.\\,Q.\\ (2017). On calibration of modern neural networks. \\textit{Proceedings of the 34th "
    "International Conference on Machine Learning}, 1321--1330. \\href{https://arxiv.org/abs/1706.04599}{arXiv:1706.04599}",
    "Hoffmann, J.\\ et al.\\ (2022). Training compute-optimal large language models. \\textit{Advances in Neural Information Processing "
    "Systems}, 35. \\href{https://arxiv.org/abs/2203.15556}{arXiv:2203.15556}",
    "Hu, E.\\,J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.\\ \\& Chen, W.\\ (2022). LoRA: Low-rank adaptation of large "
    "language models. \\textit{International Conference on Learning Representations}. \\href{https://arxiv.org/abs/2106.09685}{arXiv:2106.09685}",
    "Kaplan, J.\\ et al.\\ (2020). Scaling laws for neural language models. \\href{https://arxiv.org/abs/2001.08361}{arXiv:2001.08361}",
    "Kupiec, P.\\,H.\\ (1995). Techniques for verifying the accuracy of risk measurement models. \\textit{The Journal of Derivatives}, 3(2), "
    "73--84. \\href{https://doi.org/10.3905/jod.1995.407942}{doi:10.3905/jod.1995.407942}",
    "Lewis, P.\\ et al.\\ (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. \\textit{Advances in Neural "
    "Information Processing Systems}, 33. \\href{https://arxiv.org/abs/2005.11401}{arXiv:2005.11401}",
    "Malo, P., Sinha, A., Korhonen, P., Wallenius, J.\\ \\& Takala, P.\\ (2014). Good debt or bad debt: Detecting semantic orientations in "
    "economic texts. \\textit{Journal of the Association for Information Science and Technology}, 65(4), 782--796. "
    "\\href{https://doi.org/10.1002/asi.23062}{doi:10.1002/asi.23062}",
    "Ouyang, L.\\ et al.\\ (2022). Training language models to follow instructions with human feedback. \\textit{Advances in Neural "
    "Information Processing Systems}, 35. \\href{https://arxiv.org/abs/2203.02155}{arXiv:2203.02155}",
    "Pele, D.\\,T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A.\\,T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S.\\ \\& "
    "Härdle, W.\\,K.\\ (2026). In the beginning was the Word: LLM-VaR and LLM-ES. \\textit{Expert Systems with Applications}, 295, 128676. "
    "\\href{https://doi.org/10.1016/j.eswa.2025.128676}{doi:10.1016/j.eswa.2025.128676}",
    "Qwen Team (2024). Qwen2.5 technical report. \\href{https://arxiv.org/abs/2412.15115}{arXiv:2412.15115}",
    "Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.\\,D.\\ \\& Finn, C.\\ (2023). Direct preference optimization: Your language "
    "model is secretly a reward model. \\textit{Advances in Neural Information Processing Systems}, 36. \\href{https://arxiv.org/abs/2305.18290}{arXiv:2305.18290}",
    "Robertson, S.\\ \\& Zaragoza, H.\\ (2009). The probabilistic relevance framework: BM25 and beyond. \\textit{Foundations and Trends in "
    "Information Retrieval}, 3(4), 333--389. \\href{https://doi.org/10.1561/1500000019}{doi:10.1561/1500000019}",
    "Sennrich, R., Haddow, B.\\ \\& Birch, A.\\ (2016). Neural machine translation of rare words with subword units. \\textit{Proceedings of "
    "the 54th Annual Meeting of the ACL}, 1715--1725. \\href{https://doi.org/10.18653/v1/P16-1162}{doi:10.18653/v1/P16-1162}",
    "Tak, R.\\ \\& Pele, D.\\,T.\\ (2025). LLM-driven stock prediction: Capturing market trends with LLaMA. \\textit{Proceedings of the "
    "International Conference on Business Excellence}, 19(1), 529--540. \\href{https://doi.org/10.2478/picbe-2025-0043}{doi:10.2478/picbe-2025-0043}",
    "Wei, J.\\ et al.\\ (2022). Chain-of-thought prompting elicits reasoning in large language models. \\textit{Advances in Neural "
    "Information Processing Systems}, 35. \\href{https://arxiv.org/abs/2201.11903}{arXiv:2201.11903}",
    "Xiao, S., Liu, Z., Zhang, P.\\ \\& Muennighoff, N.\\ (2023). C-Pack: Packaged resources to advance general Chinese embedding "
    "(BAAI general embedding). \\href{https://arxiv.org/abs/2309.07597}{arXiv:2309.07597}",
    "Zhao, Z., Wallace, E., Feng, S., Klein, D.\\ \\& Singh, S.\\ (2021). Calibrate before use: Improving few-shot performance of language "
    "models. \\textit{Proceedings of the 38th International Conference on Machine Learning}, 12697--12706. \\href{https://arxiv.org/abs/2102.09690}{arXiv:2102.09690}",
]


def esc(s):
    """Plain text from the numbers file (model outputs, sentences) made safe for LaTeX."""
    rep = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
           "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "’": "'", "“": "``", "”": "''", "\u2423": "\\textvisiblespace{}"}
    return "".join(rep.get(c, c) for c in str(s))


def it(*xs, sep="2pt"):
    def one(x):
        if isinstance(x, tuple):
            return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                    "\n".join("      \\item " + (s[0].upper() + s[1:] if s[0].islower() else s) for s in x[1:] if s) + "\n    \\end{itemize}")
        return hier(x)
    return (f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{{sep}}}\n" + "\n".join("  \\item " + one(x) for x in xs if x) + "\n\\end{itemize}")


def note(*xs):
    return "\n\\par\\vspace{0.5mm}{\\scriptsize " + items(*xs, opt="\\setlength{\\itemsep}{0pt}", nest=False) + "}"


def pipeline_tikz(steps, w="2.55cm"):
    nodes = []
    for i, (txt, col) in enumerate(steps):
        nodes.append(f"\\node[draw, rounded corners, fill={col}!12, text width={w}, align=center, minimum height=1.05cm, font=\\scriptsize] "
                     f"(n{i}) at ({i * 3.05},0) {{{txt}}};")
    arrows = "".join(f"\\draw[-{{Stealth}}, thick] (n{i}.east) -- (n{i + 1}.west);" for i in range(len(steps) - 1))
    return "\\begin{center}\\begin{tikzpicture}" + "".join(nodes) + arrows + "\\end{tikzpicture}\\end{center}"


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L12_numbers.json"), lang)
    N = V.N
    D = Deck(12, "Large language models in finance and prompt engineering" if lang == "en" else
             "Modele lingvistice mari în finanțe și formularea instrucțiunilor (prompt engineering)", V, lang, heights=H)
    t = D.t
    ro = lang == "ro"
    NB = "notebook"
    C = N["cls"]
    MN = {"majority": t("majority class", "clasa majoritară"), "tfidf": t("TF-IDF + logistic", "TF-IDF + logistică"),
          "zs_05B": t("zero-shot 0.5B", "zero-shot 0,5B"), "zs_15B": t("zero-shot 1.5B", "zero-shot 1,5B"), "zs_3B": "zero-shot 3B",
          "fs_15B": t("few-shot 1.5B", "few-shot 1,5B"), "fs_3B": "few-shot 3B", "cot_3B": "chain-of-thought 3B",
          "probe": t("embeddings 1.5B + logistic", "embedding 1,5B + logistică")}

    def pv(p):
        return ("$<$0{,}001" if ro else "$<$0.001") if p < 0.001 else fmt(p, ".3f", lang)

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Write the language-model objective, perplexity and the temperature of sampling, and measure them on central-bank speeches.",
          "Să scrieți obiectivul unui model de limbaj, perplexitatea și temperatura de eșantionare și să le măsurați pe discursuri ale unei bănci centrale."),
        t("Explain pre-training, instruction tuning and alignment from human preferences, with their loss functions.",
          "Să explicați preantrenarea, ajustarea pe instrucțiuni și alinierea la preferințele umane, cu funcțiile lor de pierdere."),
        t("Turn a prompt into a classifier and measure zero-shot, few-shot and chain-of-thought prompts on labelled financial text, with confidence intervals.",
          "Să transformați un prompt într-un clasificator și să evaluați prompturile zero-shot, few-shot și chain-of-thought pe texte financiare etichetate, cu intervale de încredere."),
        t("Build and evaluate retrieval-augmented generation (RAG) over real speeches, and measure hallucination.",
          "Să construiți și să evaluați generarea augmentată prin regăsire (RAG) pe discursuri reale și să măsurați halucinațiile."),
        t("Backtest a language model used for the Value-at-Risk (VaR), and know the risks: leakage, cost, calibration.",
          "Să testați retrospectiv un model de limbaj folosit pentru valoarea expusă la risc (VaR) și să cunoașteți riscurile: scurgerea de date, costul, calibrarea."),
        sep="4pt"))
    abbreviation_frames(D, ABBR)

    # ------------------------------------------------------------------ 1. language models
    D.section(t("From next-token prediction to large language models", "De la prezicerea următorului token la modelele lingvistice mari"))
    D.frame(t("Text data in financial institutions", "Datele de tip text în instituțiile financiare"), it(
        (t("\\textbf{Markets}", "\\textbf{Piețe}"), t("news and social media move prices within minutes; sentiment is a signal",
                                                     "știrile și rețelele sociale mișcă prețurile în cîteva minute; sentimentul este un semnal")),
        (t("\\textbf{Central banks}", "\\textbf{Bănci centrale}"),
         t("speeches and minutes guide rate expectations; the European Central Bank (ECB) published @{rag.n_speeches} speeches since 2022 in our data",
           "discursurile și minutele ghidează așteptările privind dobînzile; în datele noastre, Banca Centrală Europeană (BCE) are @{rag.n_speeches} de discursuri din 2022")),
        (t("\\textbf{Credit and risk}", "\\textbf{Credit și risc}"),
         t("loan files, annual reports, risk reports: reading them costs analyst hours", "dosare de credit, rapoarte anuale, rapoarte de risc: citirea lor costă ore de analiză")),
        (t("\\textbf{This lecture}", "\\textbf{Acest curs}"),
         t("open large language models (LLM) run on a laptop, measured on labelled data, never trusted on faith",
           "modele lingvistice mari (LLM) deschise, rulate pe un laptop, evaluate pe date etichetate, niciodată acceptate fără verificare")),
        sep="3pt"), "small")
    D.concept(t("A language model: next-token probabilities, trained by cross-entropy", "Modelul de limbaj: probabilitatea următorului token, prin entropie încrucișată"),
              t("everything an LLM does, from answering questions to writing code, is repeated prediction of the next token.",
                "tot ce face un LLM, de la răspunsuri la cod, este prezicerea repetată a următorului token."),
              "\\[ p_{\\btheta}(x_1,\\dots,x_T)=\\prod_{t=1}^{T}p_{\\btheta}(x_t\\mid x_{<t}) \\]"
              "\\[ \\loss(\\btheta)=-\\frac{1}{T}\\sum_{t=1}^{T}\\log p_{\\btheta}(x_t\\mid x_{<t}),\\quad \\mathrm{PPL}=e^{\\loss} \\]" +
              "\\where{" + t("$x_t$ = token $t$; $p_{\\btheta}$ = softmax over the vocabulary at the last layer of a causal transformer (Lecture 7); "
                             "perplexity = the effective number of equally likely next tokens.",
                             "$x_t$ = tokenul $t$; $p_{\\btheta}$ = softmax peste vocabular (transformer cauzal, cursul 7); "
                             "perplexitatea = numărul efectiv de tokeni următori echiprobabili.") + "}",
              it(t("\\textbf{On ECB speeches} after July 2025, first 1,024 tokens: perplexity @{ppl.05B.new|.2f} (0.5B), @{ppl.15B.new|.2f} (1.5B), @{ppl.3B.new|.2f} (3B).",
                   "\\textbf{Discursuri BCE} după iulie 2025: perplexitatea @{ppl.05B.new|.2f} (0,5B), @{ppl.15B.new|.2f} (1,5B), @{ppl.3B.new|.2f} (3B)."),
                 None),
              t("a lower perplexity on your own documents means the model 'speaks' your domain better.",
                "o perplexitate mai mică pe documentele proprii înseamnă că modelul „vorbește” limbajul domeniului."),
              cite="(Bishop \\& Bishop, 2024, ch.\\ 12)", tryit=NB + " \\S1", wl="0.62", wr="0.35")
    tk = N["tok"]
    D.concept(t("Tokens are sub-words: byte-pair encoding", "Tokenii sînt fragmente de cuvinte: codificarea byte-pair"),
              t("frequent words are one token, rare words and numbers are split; the model never sees letters, only token indices.",
                "cuvintele frecvente sînt un singur token, iar cele rare și numerele se împart; modelul nu vede litere, doar indici de tokeni."),
              "\\[ (a,b)^{*}=\\argmax_{(a,b)}\\ \\mathrm{count}(a\\,b)\\ \\Rightarrow\\ \\text{" + t("merge", "unește") + "}\\ a\\,b\\to ab \\]" +
              "\\where{" + t("repeated until the vocabulary is full: byte-pair encoding (BPE), @{tok.vocab_qwen|,d} tokens in Qwen2.5; WordPiece, @{tok.vocab_bert|,d} tokens in a model of "
                             "bidirectional encoder representations from transformers (BERT) adapted to finance (FinBERT).",
                             "se repetă pînă cînd vocabularul este complet: codificarea byte-pair (BPE), @{tok.vocab_qwen|,d} de tokeni în Qwen2.5; WordPiece, @{tok.vocab_bert|,d} de tokeni într-un model de "
                             "reprezentări bidirecționale ale codificatorului transformer (BERT) adaptat finanțelor (FinBERT).") + "}",
              it(t("\\textbf{Example} (@{tok.example_n_qwen} tokens): " + esc(tk["example_qwen"]) + ".",
                   "\\textbf{Exemplu} (@{tok.example_n_qwen} tokeni): " + esc(tk["example_qwen"]) + "."),
                 t("\\textbf{Financial PhraseBank}: @{tok.qwen_per_word|.2f} Qwen tokens and @{tok.bert_per_word|.2f} WordPiece tokens per word.",
                   "\\textbf{Financial PhraseBank}: @{tok.qwen_per_word|.2f} tokeni Qwen și @{tok.bert_per_word|.2f} tokeni WordPiece pe cuvînt.")),
              t("cost and context limits are counted in tokens, not words: budget about 1.3 tokens per English word.",
                "costul și limita contextului se măsoară în tokeni, nu în cuvinte: socotiți aproximativ 1,3 tokeni pe cuvînt în engleză."),
              cite="(Sennrich, Haddow \\& Birch, 2016)", tryit=NB + " \\S1")
    D.figframe(t("Sentence length in words and in tokens: two tokenisers", "Lungimea propozițiilor în cuvinte și în tokeni: două metode de tokenizare"),
               t("the same @{data.pb_n|,d} sentences are a little longer in tokens than in words.",
                 "aceleași @{data.pb_n|,d} propoziții sînt puțin mai lungi în tokeni decît în cuvinte."),
               None, "tokens_per_sentence",
               bullets=[t("Mean: @{tok.words_mean|.1f} words, @{tok.qwen_mean|.1f} Qwen tokens, @{tok.bert_mean|.1f} WordPiece tokens; an ECB speech has on average "
                          "@{tok.ecb_tokens_mean|,d} tokens (longest @{tok.ecb_tokens_max|,d}).",
                          "Media: @{tok.words_mean|.1f} cuvinte, @{tok.qwen_mean|.1f} tokeni Qwen, @{tok.bert_mean|.1f} tokeni WordPiece; un discurs BCE are în medie "
                          "@{tok.ecb_tokens_mean|,d} de tokeni (cel mai lung: @{tok.ecb_tokens_max|,d}).")],
               height="0.55\\textheight", tryit=NB + " \\S1")
    nx = N["next"]
    D.concept(t("Temperature reshapes the next-token distribution", "Temperatura schimbă forma distribuției următorului token"),
              t("the model gives scores (logits); temperature decides how boldly we sample from them.",
                "modelul produce scoruri (logits); temperatura decide cît de îndrăzneț eșantionăm din ele."),
              "\\[ p_T(v\\mid x_{<t})=\\frac{\\exp(z_v/T)}{\\sum_{u}\\exp(z_u/T)} \\]\\[ H_T=-\\sum_v p_T(v)\\log p_T(v) \\]" +
              "\\where{" + t("$z_v$ = logit of token $v$; $T\\to 0$: always the most likely token (greedy); $T>1$: flatter, more diverse, more errors; $H_T$ = entropy in nats.",
                             "$z_v$ = scorul (logit) tokenului $v$; $T\\to 0$: mereu tokenul cel mai probabil (lacom); $T>1$: distribuție mai plată, mai diversă, cu mai multe erori; $H_T$ = entropia, în nats.") + "}",
              it(t("\\textbf{Prompt} from the ECB speech of @{next.speech_date|date}: ``" + esc(nx["prompt"]) + " \\ldots''",
                   "\\textbf{Prompt} din discursul BCE din @{next.speech_date|date}: „" + esc(nx["prompt"]) + " \\ldots”"),
                 t("\\textbf{Top token} ``" + esc(nx["top1"]) + "'': @{next.top1_p05|.1f}\\,\\% at $T$ = 0.5, @{next.top1_p|.1f}\\,\\% at 1, @{next.top1_p2|.1f}\\,\\% at 2; "
                   "entropy @{next.entropy.t05|.2f}, @{next.entropy.t10|.2f}, @{next.entropy.t20|.2f}.",
                   "\\textbf{Primul token} „" + esc(nx["top1"]) + "”: @{next.top1_p05|.1f}\\,\\% la $T$ = 0,5, @{next.top1_p|.1f}\\,\\% la 1, @{next.top1_p2|.1f}\\,\\% la 2; "
                   "entropia @{next.entropy.t05|.2f}, @{next.entropy.t10|.2f}, @{next.entropy.t20|.2f}.")),
              t("for classification and extraction use temperature 0; reserve high temperatures for brainstorming.",
                "pentru clasificare și extragere folosiți temperatura 0; temperaturile mari sînt pentru generarea de idei."),
              tryit=NB + " \\S2", wl="0.5", wr="0.47")
    D.figframe(t("Next-token probabilities of a real model on a real sentence", "Probabilitățile următorului token: un model real, o frază reală"),
               t("at low temperature almost all mass sits on one continuation; at high temperature unlikely tokens become plausible.",
                 "la temperatură mică aproape toată masa de probabilitate stă pe o continuare; la temperatură mare, tokenii improbabili devin plauzibili."),
               None, "next_token", height="0.62\\textheight", tryit=NB + " \\S2")
    D.concept(t("Scaling: bigger models predict text better, on a power law", "Scalarea: modelele mai mari prezic mai bine textul, după o lege de putere"),
              t("the loss falls as a power of the number of parameters and of training tokens.", "pierderea scade ca o putere a numărului de parametri și de tokeni de antrenare."),
              "\\[ \\loss(P,D)=E+\\frac{A}{P^{\\alpha}}+\\frac{B}{D^{\\beta}} \\]"
              "\\[ \\text{" + t("own estimate", "estimare proprie") + ":}\\ \\log\\loss=a+b\\log P,\\quad \\hat b=@{ppl_slope|.3f} \\]" +
              "\\where{" + t("$P$ = parameters; $D$ = training tokens; $E$ = irreducible loss; $\\alpha,\\beta\\approx 0.3$ (Hoffmann et al., 2022); "
                             "$\\hat b$ from the three Qwen2.5 sizes on ECB speeches after July 2025.",
                             "$P$ = numărul de parametri; $D$ = tokenii de antrenare; $E$ = pierderea ireductibilă; $\\alpha,\\beta\\approx 0{,}3$ (Hoffmann et al., 2022); "
                             "$\\hat b$ din cele trei dimensiuni Qwen2.5, pe discursurile BCE de după iulie 2025.") + "}",
              it(t("\\textbf{Three points are not a law}: the slope is a description of our sample, not a forecast.",
                   "\\textbf{Trei puncte nu fac o lege}: panta descrie eșantionul nostru, nu este o prognoză."),
                 t("\\textbf{Compute-optimal}: about 20 training tokens per parameter.", "\\textbf{Optimul de calcul}: aproximativ 20 de tokeni de antrenare pe parametru.")),
              t("a 3B model on a laptop costs nothing per call; a frontier model costs per token: measure the gain first.",
                "un model de 3B pe laptop nu costă nimic pe apel; un model de vîrf costă pe token: măsurați întîi cîștigul."),
              cite="(Kaplan et al., 2020; Hoffmann et al., 2022)", tryit=NB + " \\S1")
    pp = N["ppl"]
    lower_old = all(pp[s]["old"] < pp[s]["new"] for s in pp)
    D.figframe(t("Older speeches look easier: a warning about training cut-offs", "Discursurile mai vechi par mai ușoare: un avertisment despre limita datelor de antrenare"),
               t("text the model may have seen in pre-training scores a lower perplexity than text written after it.",
                 "textul pe care modelul l-ar fi putut vedea la preantrenare are o perplexitate mai mică decît textul scris după aceea."),
               None, "perplexity_size",
               bullets=[t("3B: @{ppl.3B.old|.2f} on 2022--2023 speeches vs @{ppl.3B.new|.2f} after July 2025; " +
                          ("lower for the older speeches at every size." if lower_old else "the gap is not uniform across sizes."),
                          "3B: @{ppl.3B.old|.2f} pe discursurile din 2022--2023, față de @{ppl.3B.new|.2f} după iulie 2025; " +
                          ("mai mică pentru discursurile vechi la toate dimensiunile." if lower_old else "diferența nu este aceeași la toate dimensiunile.")),
                        t("Topics also changed; the gap is a symptom, not a proof, of memorisation.",
                          "Și temele s-au schimbat; diferența este un simptom, nu o dovadă, a memorării.")],
               job=t("evaluate an LLM on documents written after its training cut-off, or the score may be inflated by leakage.",
                     "evaluați un LLM pe documente scrise după limita datelor sale de antrenare; altfel scorul poate fi umflat prin scurgerea datelor."),
               height="0.48\\textheight", tryit=NB + " \\S1")

    # ------------------------------------------------------------------ 2. pre-training, instruction tuning, alignment
    D.section(t("Pre-training, instruction tuning and alignment", "Preantrenare, ajustare pe instrucțiuni și aliniere"))
    D.frame(t("Three training stages turn a text predictor into an assistant", "Trei etape de antrenare transformă un predictor de text într-un asistent"),
            pipeline_tikz([(t("\\textbf{Pre-training}\\\\trillions of tokens\\\\next-token loss", "\\textbf{Preantrenare}\\\\trilioane de tokeni\\\\pierderea next-token"), "MainBlue"),
                           (t("\\textbf{Supervised fine-tuning}\\\\(prompt, answer) pairs", "\\textbf{Ajustare fină supervizată}\\\\perechi (prompt, răspuns)"), "Ser2"),
                           (t("\\textbf{Preferences}\\\\answer A better than B", "\\textbf{Preferințe}\\\\răspunsul A mai bun decît B"), "Ser3"),
                           (t("\\textbf{Assistant}\\\\follows instructions", "\\textbf{Asistent}\\\\urmează instrucțiunile"), "Ser7")]) +
            it(t("\\textbf{Generative pre-trained transformer (GPT)}: stage 1 alone gives a model that continues text, not one that answers.",
                 "\\textbf{Transformerul generativ preantrenat (GPT)}: doar etapa 1 dă un model care continuă textul, nu unul care răspunde."),
               t("\\textbf{Supervised fine-tuning (SFT)} teaches the format; \\textbf{reinforcement learning from human feedback (RLHF)} or "
                 "\\textbf{direct preference optimisation (DPO)} teaches which answers people prefer.",
                 "\\textbf{Ajustarea fină supervizată (SFT)} învață formatul; \\textbf{învățarea prin recompensă din feedback uman (RLHF)} sau "
                 "\\textbf{optimizarea directă a preferințelor (DPO)} învață ce răspunsuri preferă oamenii."),
               t("\\textbf{Qwen2.5-Instruct} (Qwen Team, 2024): 18 trillion pre-training tokens, then SFT and preference optimisation.",
                 "\\textbf{Qwen2.5-Instruct} (Qwen Team, 2024): 18 trilioane de tokeni la preantrenare, apoi SFT și optimizarea preferințelor."), sep="3pt"),
            "small", cite="(Ouyang et al., 2022; Qwen Team, 2024)")
    D.concept(t("Instruction tuning is next-token loss on the answer only", "Ajustarea pe instrucțiuni este pierderea next-token calculată doar pe răspuns"),
              t("the model sees prompt and answer, but is graded only on the answer tokens.", "modelul vede promptul și răspunsul, dar este evaluat doar pe tokenii răspunsului."),
              "\\[ \\loss_{\\text{SFT}}(\\btheta)=-\\sum_{(\\mathbf x,\\mathbf y)\\in\\dataset}\\ \\sum_{t=1}^{|\\mathbf y|}\\log p_{\\btheta}\\big(y_t\\mid \\mathbf x,\\mathbf y_{<t}\\big) \\]" +
              "\\where{" + t("$\\mathbf x$ = instruction (with a chat template: system, user, assistant roles); $\\mathbf y$ = reference answer; $\\dataset$ = tens of thousands to millions of pairs.",
                             "$\\mathbf x$ = instrucțiunea (cu un șablon de conversație: rolurile sistem, utilizator, asistent); $\\mathbf y$ = răspunsul de referință; $\\dataset$ = de la zeci de mii la milioane de perechi.") + "}",
              it(t("\\textbf{Low-rank adaptation (LoRA)}: train $\\mathbf W=\\mathbf W_0+\\mathbf B\\mathbf A$ with $\\mathbf B\\in\\R^{d\\times r}$, $\\mathbf A\\in\\R^{r\\times k}$, $r\\ll d$; "
                   "$\\mathbf W_0$ frozen.",
                   "\\textbf{Adaptarea de rang mic (LoRA)}: se antrenează $\\mathbf W=\\mathbf W_0+\\mathbf B\\mathbf A$, cu $\\mathbf B\\in\\R^{d\\times r}$, $\\mathbf A\\in\\R^{r\\times k}$, $r\\ll d$; "
                   "$\\mathbf W_0$ rămîne fix."),
                 t("\\textbf{A $4096\\times4096$ layer, $r=8$}: 65,536 trainable weights instead of 16.8 million.",
                   "\\textbf{Un strat $4096\\times4096$, $r=8$}: 65.536 de ponderi antrenabile în loc de 16,8 milioane.")),
              t("a bank adapts an open model to its own documents with LoRA on one graphics card, keeping data in-house.",
                "o bancă își adaptează un model deschis la propriile documente cu LoRA, pe o singură placă grafică, fără ca datele să iasă din instituție."),
              cite="(Hu et al., 2022)")
    D.concept(t("Alignment: from pairwise preferences to a reward, or directly to the policy", "Alinierea: din preferințe, prin recompensă sau direct"),
              t("people say which of two answers is better; the model is pushed towards preferred answers without drifting too far.",
                "oamenii aleg răspunsul mai bun dintr-o pereche; modelul este împins spre răspunsurile preferate."),
              "\\[ \\Pr(y_w\\succ y_l\\mid x)=\\sigma\\big(r_\\phi(x,y_w)-r_\\phi(x,y_l)\\big) \\]"
              "\\[ \\max_{\\btheta}\\ \\E\\big[r_\\phi(x,y)\\big]-\\beta\\,\\mathrm{KL}\\big(p_{\\btheta}\\,\\|\\,p_{\\text{SFT}}\\big) \\]"
              "\\[ \\loss_{\\text{DPO}}=-\\log\\sigma\\Big(\\beta\\log\\tfrac{p_{\\btheta}(y_w|x)}{p_{\\text{SFT}}(y_w|x)}-\\beta\\log\\tfrac{p_{\\btheta}(y_l|x)}{p_{\\text{SFT}}(y_l|x)}\\Big) \\]" +
              "\\where{" + t("$y_w,y_l$ = preferred and rejected answers; $r_\\phi$ = reward model (Bradley--Terry); $\\sigma$ = logistic function; "
                             "Kullback--Leibler (KL) penalty with weight $\\beta$ keeps the model close to the SFT model.",
                             "$y_w,y_l$ = răspunsul preferat și cel respins; $r_\\phi$ = modelul de recompensă (Bradley--Terry); $\\sigma$ = funcția logistică; "
                             "penalizarea Kullback--Leibler (KL), cu ponderea $\\beta$, ține modelul aproape de modelul SFT.") + "}",
              it(t("\\textbf{RLHF} fits $r_\\phi$, then optimises the policy; \\textbf{DPO} skips the reward model.",
                   "\\textbf{RLHF} estimează $r_\\phi$, apoi optimizează politica; \\textbf{DPO} renunță la modelul de recompensă."),
                 t("\\textbf{Side effect}: aligned models sound confident; confidence is not accuracy (calibration, below).",
                   "\\textbf{Efect secundar}: modelele aliniate sună sigure; siguranța nu înseamnă precizie.")),
              t("alignment sets tone and safety, not financial knowledge: evaluate on your task.",
                "alinierea stabilește tonul și siguranța, nu cunoștințele financiare: evaluați pe sarcina proprie."),
              cite="(Ouyang et al., 2022; Rafailov et al., 2023)", wl="0.56", wr="0.41")
    md = N["models"]
    D.frame(t("The open models used in this lecture", "Modelele deschise folosite în acest curs"),
            table("@{}p{0.33\\textwidth}rp{0.45\\textwidth}@{}", [t("Model", "Model"), t("Parameters", "Parametri"), t("Role here", "Rolul în acest curs")],
                  [["Qwen2.5-0.5B-Instruct", fmt(md["05B"]["params_b"], ".2f", lang) + t(" bn", " mld."), t("perplexity, zero-shot classification", "perplexitate, clasificare zero-shot")],
                   ["Qwen2.5-1.5B-Instruct", fmt(md["15B"]["params_b"], ".2f", lang) + t(" bn", " mld."), t("next-token probabilities, few-shot, embeddings", "probabilitățile următorului token, few-shot, embedding")],
                   ["Qwen2.5-3B-Instruct", fmt(md["3B"]["params_b"], ".2f", lang) + t(" bn", " mld."), t("best prompts, chain-of-thought, question answering, LLM-VaR",
                                                                                                       "cele mai bune prompturi, chain-of-thought, răspunsuri la întrebări, LLM-VaR")],
                   ["BAAI bge-small-en-v1.5", "33 " + t("m", "mil."), t("dense retrieval embeddings (384 dimensions)", "embedding pentru regăsire densă (384 de dimensiuni)")],
                   [t("Term frequency--inverse document frequency (TF-IDF) + logistic regression", "Frecvența termenului--frecvența inversă în documente (TF-IDF) + regresie logistică"), "--", t("classical benchmark trained on the labelled data", "reperul clasic, antrenat pe datele etichetate")]]) +
            note(t("All models run locally on a laptop; no text leaves the machine and no application programming interface (API) is paid.",
                   "Toate modelele rulează local, pe un laptop; niciun text nu părăsește calculatorul și nu se plătește nicio interfață de programare a aplicațiilor (API)."),
                 t("TF-IDF: word and word-pair counts weighted by their rarity across documents.",
                   "TF-IDF: numărul de apariții ale cuvintelor și perechilor de cuvinte, ponderat cu raritatea lor în documente.")),
            "small", cite="(Qwen Team, 2024; Xiao et al., 2023)")

    # ------------------------------------------------------------------ 3. prompt engineering, measured
    D.section(t("Prompt engineering, measured on labelled data", "Formularea instrucțiunilor, evaluată pe date etichetate"))
    D.frame(t("Structure of a prompt", "Structura unui prompt"), cols(it(
        (t("\\textbf{System}", "\\textbf{Sistem}"), t("role and rules: ``You are a careful financial analyst.''", "rolul și regulile: „You are a careful financial analyst.”")),
        (t("\\textbf{Instruction}", "\\textbf{Instrucțiune}"), t("task, label set, output format: ``Answer with one word: positive, negative or neutral.''",
                                                             "sarcina, mulțimea etichetelor, formatul: „Answer with one word: positive, negative or neutral.”")),
        (t("\\textbf{Examples} (optional)", "\\textbf{Exemple} (opțional)"), t("$k$ labelled input--answer pairs: few-shot", "$k$ perechi etichetate intrare--răspuns: few-shot")),
        (t("\\textbf{Input}", "\\textbf{Intrare}"), t("the sentence, tweet or document to process", "propoziția, tweet-ul sau documentul de prelucrat")),
        sep="3pt"),
        it((t("\\textbf{Variants we test}", "\\textbf{Variantele testate}"),
            t("zero-shot: instruction + input", "zero-shot: instrucțiune + intrare"),
            t("few-shot: 3 to 24 examples", "few-shot: între 3 și 24 de exemple"),
            t("chain-of-thought: ``explain, then answer''", "chain-of-thought: „explică, apoi răspunde”"),
            t("five paraphrases of the instruction", "cinci reformulări ale instrucțiunii")),
           (t("\\textbf{Rule of this lecture}", "\\textbf{Regula acestui curs}"),
            t("every prompt is a model: it is scored on a held-out test set", "fiecare prompt este un model: se evaluează pe un set de test separat")), sep="3pt"), "0.5", "0.47"),
            "small", cite="(Brown et al., 2020; Wei et al., 2022)")
    D.concept(t("A prompt becomes a classifier through the probabilities of the label words", "Un prompt devine clasificator prin probabilitățile cuvintelor-etichetă"),
              t("read the probability of each label at the answer position: no text generation, no parsing, and a calibrated-looking score.",
                "citim probabilitatea fiecărei etichete în poziția răspunsului: fără generare de text, fără interpretarea lui și cu un scor de tip probabilitate."),
              "\\[ \\hat p(c\\mid x)=\\frac{\\sum_{v\\in\\mathcal V_c}p_{\\btheta}(v\\mid \\mathrm{prompt}(x))}{\\sum_{c'}\\sum_{v\\in\\mathcal V_{c'}}p_{\\btheta}(v\\mid \\mathrm{prompt}(x))} \\]\\[ \\hat c=\\argmax_c\\hat p(c\\mid x) \\]" +
              "\\where{" + t("$\\mathcal V_c$ = first tokens of the label word $c$ in its spellings (``positive'', `` Positive'', \\ldots); the prompt ends where the assistant's answer starts.",
                             "$\\mathcal V_c$ = primii tokeni ai cuvîntului-etichetă $c$ în toate formele lui („positive”, „ Positive”, \\ldots); promptul se termină acolo unde începe răspunsul asistentului.") + "}",
              it(t("\\textbf{Fast}: one forward pass per item; the shared instruction and examples are encoded once (key--value cache).",
                   "\\textbf{Rapid}: o singură trecere înainte pe element; instrucțiunea și exemplele comune se codifică o singură dată (memoria cheie--valoare)."),
                 t("\\textbf{Comparable}: the same macro-F1 and confidence intervals as any classifier.",
                   "\\textbf{Comparabil}: același macro-F1 și aceleași intervale de încredere ca pentru orice clasificator.")),
              t("constrain the output space; free text answers need parsing and fail silently.",
                "restrîngeți spațiul răspunsurilor; răspunsurile libere trebuie interpretate și pot eșua fără să observați."),
              cite="(Zhao et al., 2021)", tryit=NB + " \\S3")
    dd = N["data"]
    D.frame(t("Two labelled data sets: financial news and finance tweets", "Două seturi de date etichetate: știri financiare și tweet-uri financiare"),
            table("@{}p{0.27\\textwidth}p{0.3\\textwidth}rrp{0.17\\textwidth}@{}",
                  [t("Data set", "Set de date"), t("Labels", "Etichete"), t("Train", "Antrenare"), t("Test", "Test"), t("Neutral share (test)", "Ponderea neutrelor (test)")],
                  [["Financial PhraseBank, 100\\,\\% " + t("agreement", "acord"), t("positive, negative, neutral", "pozitiv, negativ, neutru"),
                    f"{dd['pb_tr']}", f"{dd['pb_te']}", fmt(dd["pb_share_neutral"], ".1f", lang) + "\\,\\%"],
                   ["Twitter financial news", t("bullish, bearish, neutral", "optimist (bullish), pesimist (bearish), neutru"),
                    fmt(dd["tw_tr"], ",d", lang), f"{dd['tw_te']}", fmt(dd["tw_share_neutral"], ".1f", lang) + "\\,\\%"]]) +
            note(t("PhraseBank: @{data.pb_n|,d} sentences from news about Finnish listed companies, labelled by 5--8 annotators; we use the "
                   "@{data.pb100_n|,d} with unanimous labels, split 70/30 stratified (seed @{seed}).",
                   "PhraseBank: @{data.pb_n|,d} de propoziții din știri despre companii listate în Finlanda, etichetate de 5--8 adnotatori; folosim cele "
                   "@{data.pb100_n|,d} cu etichete unanime, împărțite 70/30 stratificat (valoarea generatorului aleator: @{seed})."),
                 t("Twitter: published split; test = stratified sample of @{data.tw_te} of the @{data.tw_va|,d} validation tweets.",
                   "Twitter: împărțirea publicată; testul = eșantion stratificat de @{data.tw_te} din cele @{data.tw_va|,d} de tweet-uri de validare."),
                 t("The examples for few-shot prompts are drawn from the training part only.", "Exemplele pentru prompturile few-shot se extrag doar din partea de antrenare.")),
            "small", cite="(Malo et al., 2014)")
    bpb, btw = C["pb"]["best"], C["tw"]["best"]
    D.figframe(t("Prompted LLMs against a trained benchmark", "LLM-uri cu prompt comparate cu un reper antrenat"),
               t("no single strategy wins everywhere; the trained classifiers use the labelled data, the prompts do not.",
                 "nicio strategie nu cîștigă peste tot; clasificatorii antrenați folosesc datele etichetate, prompturile nu."),
               None, "prompt_results",
               bullets=[t(f"Best on PhraseBank: {MN[bpb]} (@{{cls.pb.{bpb}.f1|.1f}}\\,\\%); on Twitter: {MN[btw]} (@{{cls.tw.{btw}.f1|.1f}}\\,\\%); "
                          "zero-shot 3B: @{cls.pb.zs_3B.f1|.1f}\\,\\% and @{cls.tw.zs_3B.f1|.1f}\\,\\%.",
                          f"Cel mai bun pe PhraseBank: {MN[bpb]} (@{{cls.pb.{bpb}.f1|.1f}}\\,\\%); pe Twitter: {MN[btw]} (@{{cls.tw.{btw}.f1|.1f}}\\,\\%); "
                          "zero-shot 3B: @{cls.pb.zs_3B.f1|.1f}\\,\\% și @{cls.tw.zs_3B.f1|.1f}\\,\\%.")],
               height="0.62\\textheight", tryit=NB + " \\S3")
    meths = ["majority", "tfidf", "zs_05B", "zs_15B", "zs_3B", "fs_15B", "fs_3B", "cot_3B", "probe"]
    rows = []
    for m in meths:
        row = [MN[m]]
        for ds in ("pb", "tw"):
            r = C[ds][m]
            row += [fmt(r["acc"], ".1f", lang), heat(r["f1"], 30, max(C[ds][x]["f1"] for x in meths), good_low=False, shown=fmt(r["f1"], ".1f", lang)),
                    "[" + fmt(r["lo"], ".1f", lang) + "; " + fmt(r["hi"], ".1f", lang) + "]"]
        rows.append(row)
    D.frame(t("Accuracy and macro-F1 with 95\\,\\% bootstrap intervals", "Acuratețea și macro-F1, cu intervale bootstrap de 95\\,\\%"),
            table("@{}lrrlrrl@{}", [t("Method", "Metoda"), t("Acc.", "Acur."), "F1", t("95\\,\\% int.", "Int.\\ 95\\,\\%"), t("Acc.", "Acur."), "F1", t("95\\,\\% int.", "Int.\\ 95\\,\\%")],
                  [["", "\\multicolumn{3}{c}{PhraseBank}", "", "", "\\multicolumn{3}{c}{Twitter}"][:1] + ["\\multicolumn{3}{c}{PhraseBank}", "\\multicolumn{3}{c}{Twitter}"]] + rows,
                  size="scriptsize") +
            note(t("Macro-F1 = mean of the three per-class F1 scores, so the small negative class counts as much as neutral; "
                   "intervals from 1,000 bootstrap resamples of the test set; few-shot = mean of 3 seeds.",
                   "Macro-F1 = media celor trei scoruri F1 pe clase, astfel încît clasa mică negativă contează cît cea neutră; "
                   "intervalele provin din 1.000 de reeșantionări bootstrap ale setului de test; few-shot = media a 3 rulări."),
                 t("Int. = 95\\,\\% bootstrap interval of the macro-F1.", "Int. = intervalul bootstrap de 95\\,\\% al macro-F1.")),
            "small", cite=t("(own computation)", "(calcul propriu)"), tryit=NB + " \\S3")
    fs = N["fewshot"]
    D.figframe(t("A few examples help the small model; more examples do not", "Cîteva exemple ajută modelul mic; mai multe exemple nu mai ajută"),
               t("in-context learning: the examples change the output without changing a single weight.",
                 "învățarea în context: exemplele schimbă răspunsul fără să modifice nicio pondere."),
               "\\[ \\hat c=\\argmax_c\\,p_{\\btheta}\\big(c\\mid (x_1,c_1),\\dots,(x_k,c_k),x\\big) \\]", "fewshot_curve",
               bullets=[t("1.5B: @{fewshot.15B.0.mean|.1f}\\,\\% with no example, @{fewshot.15B.8.mean|.1f}\\,\\% with 24; "
                          "3B: @{fewshot.3B.0.mean|.1f}\\,\\% and @{fewshot.3B.8.mean|.1f}\\,\\%; seeds move 3B with 12 examples between @{fewshot.3B.4.min|.1f} and @{fewshot.3B.4.max|.1f}\\,\\%.",
                          "1,5B: @{fewshot.15B.0.mean|.1f}\\,\\% fără exemple, @{fewshot.15B.8.mean|.1f}\\,\\% cu 24; "
                          "3B: @{fewshot.3B.0.mean|.1f}\\,\\% și @{fewshot.3B.8.mean|.1f}\\,\\%; cu 12 exemple, rulările mută 3B între @{fewshot.3B.4.min|.1f} și @{fewshot.3B.4.max|.1f}\\,\\%.")],
               height="0.5\\textheight", tryit=NB + " \\S4")
    D.figframe(t("Rewording the instruction moves the score: prompt sensitivity", "Reformularea instrucțiunii schimbă scorul: sensibilitatea la prompt"),
               t("five instructions with the same meaning, the same model and data: the spread is a source of model risk.",
                 "cinci instrucțiuni cu același sens, același model și aceleași date: diferențele sînt o sursă de risc de model."),
               None, "prompt_sensitivity",
               bullets=[t("Range of macro-F1 over the five wordings: PhraseBank 3B @{sens.pb_3B.range|.1f} points, Twitter 3B @{sens.tw_3B.range|.1f} points, "
                          "Twitter 1.5B @{sens.tw_15B.range|.1f} points.",
                          "Amplitudinea macro-F1 pe cele cinci formulări: PhraseBank 3B @{sens.pb_3B.range|.1f} puncte, Twitter 3B @{sens.tw_3B.range|.1f} puncte, "
                          "Twitter 1,5B @{sens.tw_15B.range|.1f} puncte.")],
               job=t("choose the prompt on a validation set and freeze it, like any hyper-parameter; never on the test set.",
                     "alegeți promptul pe un set de validare și fixați-l, ca pe orice hiperparametru; niciodată pe setul de test."),
               height="0.5\\textheight", tryit=NB + " \\S4")
    cot = N["cot"]
    D.concept(t("Chain-of-thought: let the model write its reasoning before the answer", "Chain-of-thought: modelul își scrie raționamentul înainte de răspuns"),
              t("generated reasoning tokens become extra context for the final label.", "tokenii de raționament generați devin context suplimentar pentru eticheta finală."),
              "\\[ p(c\\mid x)=\\sum_{r}p(r\\mid x)\\,p(c\\mid x,r)\\ \\approx\\ p(c\\mid x,\\hat r) \\]" +
              "\\where{" + t("$r$ = reasoning text, $\\hat r$ = the greedy one (at most two sentences, on average @{cot.pb.mean_words|.1f} words); the label is parsed from ``Label: \\ldots''; "
                             "parsed for @{cot.pb.parsed_pct|.1f}\\,\\% of PhraseBank answers.",
                             "$r$ = textul raționamentului, $\\hat r$ = cel lacom (cel mult două propoziții, în medie @{cot.pb.mean_words|.1f} cuvinte); eticheta se extrage din „Label: \\ldots”; "
                             "extragerea reușește pentru @{cot.pb.parsed_pct|.1f}\\,\\% din răspunsurile PhraseBank.") + "}",
              it(t("\\textbf{Macro-F1}: PhraseBank @{cls.pb.cot_3B.f1|.1f}\\,\\% (zero-shot 3B @{cls.pb.zs_3B.f1|.1f}\\,\\%); Twitter @{cls.tw.cot_3B.f1|.1f}\\,\\% (@{cls.tw.zs_3B.f1|.1f}\\,\\%).",
                   "\\textbf{Macro-F1}: PhraseBank @{cls.pb.cot_3B.f1|.1f}\\,\\% (zero-shot 3B @{cls.pb.zs_3B.f1|.1f}\\,\\%); Twitter @{cls.tw.cot_3B.f1|.1f}\\,\\% (@{cls.tw.zs_3B.f1|.1f}\\,\\%)."),
                 t("\\textbf{Cost}: @{cls.pb.cot_3B.ms|,.0f} ms per sentence against @{cls.pb.zs_3B.ms|,.1f} ms for zero-shot.",
                   "\\textbf{Costul}: @{cls.pb.cot_3B.ms|,.0f} ms pe propoziție, față de @{cls.pb.zs_3B.ms|,.1f} ms pentru zero-shot.")),
              t("chain-of-thought helps on multi-step reasoning; for one-word labels on short text, test before paying for it.",
                "chain-of-thought ajută la raționamente în mai mulți pași; pentru etichete de un cuvînt pe texte scurte, testați înainte să plătiți pentru el."),
              cite="(Wei et al., 2022)", tryit=NB + " \\S5")
    D.frame(t("What the model wrote: one chain-of-thought answer", "Ce a scris modelul: un răspuns chain-of-thought"), it(
        (t("\\textbf{Input} (PhraseBank test set)", "\\textbf{Intrare} (setul de test PhraseBank)"), "``" + esc(cot["pb"]["example_text"]) + "''"),
        (t("\\textbf{Output} of Qwen2.5-3B-Instruct, temperature 0", "\\textbf{Ieșirea} modelului Qwen2.5-3B-Instruct, temperatura 0"), "``" + esc(cot["pb"]["example_out"]) + "''"),
        (t("\\textbf{How to read it}", "\\textbf{Cum se citește}"),
         t("the reasoning is a plausible story, not a faithful trace of the computation", "raționamentul este o poveste plauzibilă, nu o urmă fidelă a calculului"),
         t("audit the label against the data, not the explanation against your intuition", "verificați eticheta pe date, nu explicația pe intuiție")),
        sep="4pt"), "small", cite=t("(own computation)", "(calcul propriu)"))
    cal = N["calib"]
    D.figframe(t("Are the label probabilities calibrated? Reliability on real data", "Sînt calibrate probabilitățile etichetelor? Fiabilitatea pe date reale"),
               t("a probability of 0.9 should be right nine times in ten; points below the diagonal mean over-confidence.",
                 "o probabilitate de 0,9 trebuie să fie corectă de nouă ori din zece; punctele sub diagonală înseamnă încredere excesivă."),
               "\\[ \\mathrm{ECE}=\\sum_{b=1}^{B}\\frac{|S_b|}{n}\\,\\big|\\mathrm{acc}(S_b)-\\mathrm{conf}(S_b)\\big| \\]", "calibration",
               bullets=[t("Expected calibration error (ECE): zero-shot 3B @{calib.zs_3B|.1f}\\,\\%, few-shot 3B @{calib.fs_3B_s0|.1f}\\,\\%, TF-IDF + logistic @{calib.tfidf|.1f}\\,\\%.",
                          "Eroarea de calibrare așteptată (ECE): zero-shot 3B @{calib.zs_3B|.1f}\\,\\%, few-shot 3B @{calib.fs_3B_s0|.1f}\\,\\%, TF-IDF + logistică @{calib.tfidf|.1f}\\,\\%.")],
               job=t("recalibrate on a validation set (temperature scaling) before using LLM probabilities as risk scores.",
                     "recalibrați pe un set de validare (scalarea temperaturii) înainte de a folosi probabilitățile LLM ca scoruri de risc."),
               height="0.5\\textheight", cite="(Guo et al., 2017)", tryit=NB + " \\S6")
    ag = N["agree"]
    D.figframe(t("Accuracy by level of annotator agreement", "Acuratețea în funcție de gradul de acord al adnotatorilor"),
               t("sentences that split the annotators are genuinely ambiguous; no model reaches the unanimous-case accuracy on them.",
                 "propozițiile care îi împart pe adnotatori sînt cu adevărat ambigue; niciun model nu atinge pe ele acuratețea din cazurile unanime."),
               None, "agreement",
               bullets=[t("Zero-shot 3B: @{agree.zs_3B.100|.1f}\\,\\% at 100\\,\\% agreement, @{agree.zs_3B.50|.1f}\\,\\% at 50\\,\\%; TF-IDF: @{agree.tfidf.100|.1f}\\,\\% and @{agree.tfidf.50|.1f}\\,\\%.",
                          "Zero-shot 3B: @{agree.zs_3B.100|.1f}\\,\\% la acord de 100\\,\\%, @{agree.zs_3B.50|.1f}\\,\\% la 50\\,\\%; TF-IDF: @{agree.tfidf.100|.1f}\\,\\% și @{agree.tfidf.50|.1f}\\,\\%.")],
               job=t("report the label quality of your test set; a model cannot be more certain than the people who labelled it.",
                     "raportați calitatea etichetelor din setul de test; un model nu poate fi mai sigur decît oamenii care l-au etichetat."),
               height="0.5\\textheight", tryit=NB + " \\S6")
    cm = N["confusion"]["m"]
    labs = [t("positive", "pozitiv"), t("negative", "negativ"), t("neutral", "neutru")]
    hi = max(max(r) for r in cm)
    from nn_build import seq
    D.frame(t("Where the best prompt errs: the confusion matrix", "Unde greșește cel mai bun prompt: matricea de confuzie"),
            cols(table("@{}lrrr@{}", [t("True \\textbackslash{} predicted", "Real \\textbackslash{} prezis")] + labs,
                       [[labs[i]] + [seq(cm[i][j], hi, shown=str(cm[i][j])) for j in range(3)] for i in range(3)]),
                 it(t("\\textbf{Model}: " + MN[N["confusion"]["model"]] + ", PhraseBank test set.", "\\textbf{Modelul}: " + MN[N["confusion"]["model"]] + ", setul de test PhraseBank."),
                    t(f"\\textbf{{Neutral read as positive}}: {cm[2][0]} sentences; \\textbf{{positive read as neutral}}: {cm[0][2]}.",
                      f"\\textbf{{Neutru citit ca pozitiv}}: {cm[2][0]} propoziții; \\textbf{{pozitiv citit ca neutru}}: {cm[0][2]}."),
                    t(f"\\textbf{{Negative and positive}} are rarely swapped: {cm[1][0]} + {cm[0][1]}.",
                      f"\\textbf{{Negativ și pozitiv}} se confundă rar: {cm[1][0]} + {cm[0][1]}."), sep="4pt"), "0.5", "0.47") +
            note(t("Errors sit on the neutral border: ``profit rose but less than expected'' is the hard case, not good versus bad news.",
                   "Erorile stau la granița cu neutrul: „profitul a crescut, dar mai puțin decît se aștepta” este cazul dificil, nu știrile bune față de cele rele.")),
            "small", cite=t("(own computation)", "(calcul propriu)"), tryit=NB + " \\S6")
    D.figframe(t("Cost against accuracy: milliseconds per sentence on a laptop", "Costul față de precizie: milisecunde pe propoziție, pe un laptop"),
               t("the cheapest model is thousands of times faster than the slowest; the accuracy gap is small.",
                 "cel mai ieftin model este de mii de ori mai rapid decît cel mai lent; diferența de precizie este mică."),
               None, "cost_accuracy",
               bullets=[t("TF-IDF: @{cls.pb.tfidf.ms|.2f} ms per sentence; zero-shot 3B: @{cls.pb.zs_3B.ms|.1f} ms; chain-of-thought 3B: @{cls.pb.cot_3B.ms|,.0f} ms.",
                          "TF-IDF: @{cls.pb.tfidf.ms|.2f} ms pe propoziție; zero-shot 3B: @{cls.pb.zs_3B.ms|.1f} ms; chain-of-thought 3B: @{cls.pb.cot_3B.ms|,.0f} ms.")],
               job=t("with labels, train a small model; without labels, prompt an LLM; for millions of documents, the cost decides.",
                     "dacă aveți etichete, antrenați un model mic; fără etichete, folosiți un prompt; pentru milioane de documente, costul decide."),
               height="0.5\\textheight", tryit=NB + " \\S6")
    D.concept(t("LLM embeddings plus a small classifier: the cheapest form of fine-tuning", "Embedding LLM plus un clasificator mic: cea mai ieftină formă de ajustare fină"),
              t("freeze the language model, average its last hidden states, and fit a logistic regression on top.",
                "fixăm modelul de limbaj, facem media ultimelor stări ascunse și estimăm deasupra o regresie logistică."),
              "\\[ \\mathbf e(x)=\\frac{1}{|x|}\\sum_{t=1}^{|x|}\\mathbf h^{(L)}_t,\\qquad \\hat p(c\\mid x)=\\softmax\\big(\\mathbf W\\mathbf e(x)+\\mathbf b\\big)_c \\]" +
              "\\where{" + t("$\\mathbf h^{(L)}_t\\in\\R^{@{probe.pb.dim}}$ = last hidden state of Qwen2.5-1.5B; $\\mathbf W$ fitted with an L2 penalty chosen by 5-fold cross-validation.",
                             "$\\mathbf h^{(L)}_t\\in\\R^{@{probe.pb.dim}}$ = ultima stare ascunsă a Qwen2.5-1.5B; $\\mathbf W$ estimat cu penalizare L2 aleasă prin validare încrucișată în 5 pași.") + "}",
              it(t("\\textbf{Macro-F1}: PhraseBank @{cls.pb.probe.f1|.1f}\\,\\%, Twitter @{cls.tw.probe.f1|.1f}\\,\\%.",
                   "\\textbf{Macro-F1}: PhraseBank @{cls.pb.probe.f1|.1f}\\,\\%, Twitter @{cls.tw.probe.f1|.1f}\\,\\%."),
                 t("\\textbf{Uses the labels} like TF-IDF, but with the language knowledge of the LLM.",
                   "\\textbf{Folosește etichetele} ca TF-IDF, dar cu cunoștințele lingvistice ale LLM.")),
              t("before full fine-tuning, try a probe: minutes on a laptop, no gradient through the LLM.",
                "înainte de o ajustare fină completă, încercați un clasificator pe embedding: cîteva minute pe laptop, fără gradient prin LLM."),
              tryit=NB + " \\S6")

    # ------------------------------------------------------------------ 4. RAG
    rg = N["rag"]
    D.section(t("Retrieval-augmented generation on ECB speeches", "Generare augmentată prin regăsire pe discursurile BCE"))
    D.frame(t("RAG: retrieve the right documents, then let the model answer from them", "RAG: regăsim documentele potrivite, apoi modelul răspunde pe baza lor"),
            pipeline_tikz([(t("\\textbf{Question}", "\\textbf{Întrebare}"), "MainBlue"),
                           (t("\\textbf{Retriever}\\\\BM25 + dense\\\\top-$k$ chunks", "\\textbf{Regăsire}\\\\BM25 + dense\\\\primele $k$ fragmente"), "Ser3"),
                           (t("\\textbf{Prompt}\\\\documents + question", "\\textbf{Prompt}\\\\documente + întrebare"), "Ser2"),
                           (t("\\textbf{LLM answer}\\\\with sources", "\\textbf{Răspunsul LLM}\\\\cu surse"), "Ser7")]) +
            "\\vspace{-2mm}\\[ p(y\\mid q)\\approx\\sum_{z\\in\\mathrm{top}_k(q)}p_\\eta(z\\mid q)\\,p_{\\btheta}(y\\mid q,z) \\]" +
            it(t("\\textbf{Retrieval-augmented generation (RAG)}: $z$ = retrieved chunk, $p_\\eta$ = retriever, $p_{\\btheta}$ = generator.",
                 "\\textbf{Generarea augmentată prin regăsire (RAG)}: $z$ = fragmentul regăsit, $p_\\eta$ = modulul de regăsire, $p_{\\btheta}$ = generatorul."),
               t("\\textbf{Our corpus}: @{rag.n_speeches} ECB speeches, @{rag.n_chunks|,d} chunks of @{rag.chunk_words} words (overlap @{rag.overlap}).",
                 "\\textbf{Corpusul nostru}: @{rag.n_speeches} de discursuri BCE, @{rag.n_chunks|,d} de fragmente de @{rag.chunk_words} de cuvinte (suprapunere @{rag.overlap})."), sep="2pt"),
            "small", cite="(Lewis et al., 2020)")
    D.concept(t("Three retrievers: shared words, weighted terms, dense embeddings", "Trei metode de regăsire: cuvinte comune, termeni ponderați, embedding"),
              t("lexical retrievers match words, dense ones match meaning, a hybrid combines both; the query here is a speech title.",
                "metodele lexicale potrivesc cuvinte, cele dense sensul, iar metoda hibridă le combină; interogarea este titlul unui discurs."),
              "\\[ \\mathrm{BM25}(q,z)=\\sum_{w\\in q}\\mathrm{idf}(w)\\frac{f(w,z)(k_1+1)}{f(w,z)+k_1\\big(1-b+b\\frac{|z|}{\\overline{|z|}}\\big)} \\]"
              "\\[ s_{\\text{dense}}=\\cos\\big(\\mathbf e(q),\\mathbf e(z)\\big),\\quad s_{\\text{RRF}}(z)=\\sum_{m}\\frac{1}{60+\\mathrm{rank}_m(z)} \\]" +
              "\\where{" + t("$f(w,z)$ = count of word $w$ in chunk $z$; $k_1=1.5$, $b=0.75$; Best Matching 25 (BM25); $\\mathbf e$ = bge-small embedding; RRF = rank fusion.",
                             "$f(w,z)$ = aparițiile cuvîntului $w$ în fragmentul $z$; $k_1=1{,}5$, $b=0{,}75$; Best Matching 25 (BM25); $\\mathbf e$ = embedding bge-small; RRF = fuziunea rangurilor.") + "}",
              it(t("\\textbf{Mean reciprocal rank (MRR)}: BM25 @{rag.bm25.mrr|.3f}, dense @{rag.dense.mrr|.3f}, hybrid @{rag.hybrid.mrr|.3f}.",
                   "\\textbf{Rangul reciproc mediu (MRR)}: BM25 @{rag.bm25.mrr|.3f}, dense @{rag.dense.mrr|.3f}, hibrid @{rag.hybrid.mrr|.3f}.")),
              t("the answer cannot be better than what is retrieved: evaluate the retriever on its own first.",
                "răspunsul nu poate fi mai bun decît ce s-a regăsit: evaluați întîi separat modulul de regăsire."),
              cite="(Robertson \\& Zaragoza, 2009; Xiao et al., 2023)", tryit=NB + " \\S7", wl="0.64", wr="0.33")
    D.figframe(t("Finding the right speech from its title", "Găsirea discursului corect după titlu"),
               t("recall@k = share of the @{rag.n_speeches} titles whose speech is among the top $k$ results.",
                 "recall@k = ponderea celor @{rag.n_speeches} de titluri pentru care discursul se află printre primele $k$ rezultate."),
               None, "rag_recall",
               bullets=[t("Recall@1: TF-IDF @{rag.tfidf.r1|.1f}\\,\\%, BM25 @{rag.bm25.r1|.1f}\\,\\%, dense @{rag.dense.r1|.1f}\\,\\%, hybrid @{rag.hybrid.r1|.1f}\\,\\%; "
                          "recall@10 hybrid @{rag.hybrid.r10|.1f}\\,\\%.",
                          "Recall@1: TF-IDF @{rag.tfidf.r1|.1f}\\,\\%, BM25 @{rag.bm25.r1|.1f}\\,\\%, dense @{rag.dense.r1|.1f}\\,\\%, hibrid @{rag.hybrid.r1|.1f}\\,\\%; "
                          "recall@10 hibrid @{rag.hybrid.r10|.1f}\\,\\%.")],
               height="0.5\\textheight", tryit=NB + " \\S7")
    D.figframe(t("Answers with and without retrieved documents", "Răspunsurile cu și fără documente regăsite"),
               t("question: which board member gave the speech with this title? @{rag.qa_n} speeches since @{rag.qa_first|date}, after the model's training data.",
                 "întrebarea: ce membru al conducerii a ținut discursul cu acest titlu? @{rag.qa_n} de discursuri din @{rag.qa_first|date}, ulterioare datelor de antrenare ale modelului."),
               None, "rag_qa",
               bullets=[t("Forced answer without documents: @{rag.qa_closed.wrong|.1f}\\,\\% wrong names, " + esc(rg["closed_top_wrong"]) + " @{rag.closed_top_wrong_n} times; "
                          "RAG @{rag.qa_rag.correct|.1f}\\,\\% correct, oracle @{rag.qa_oracle.correct|.1f}\\,\\%; the most frequent speaker alone: @{rag.qa_majority.pct|.1f}\\,\\%.",
                          "Răspuns obligatoriu fără documente: @{rag.qa_closed.wrong|.1f}\\,\\% nume greșite, " + esc(rg["closed_top_wrong"]) + " de @{rag.closed_top_wrong_n} de ori; "
                          "RAG @{rag.qa_rag.correct|.1f}\\,\\% corecte, oracol @{rag.qa_oracle.correct|.1f}\\,\\%; doar vorbitorul cel mai frecvent: @{rag.qa_majority.pct|.1f}\\,\\%.")],
               job=t("a confident wrong name is the costly failure: require sources in every answer and check them.",
                     "un nume greșit spus cu siguranță este eșecul costisitor: cereți surse în fiecare răspuns și verificați-le."),
               height="0.46\\textheight", tryit=NB + " \\S8")
    D.frame(t("Designing a RAG system for a bank: the choices that matter", "Proiectarea unui sistem RAG pentru o bancă: deciziile care contează"), it(
        (t("\\textbf{Chunking}", "\\textbf{Fragmentarea}"), t("200 words with 50 overlap here; tables and footnotes need their own rules",
                                                            "aici 200 de cuvinte cu suprapunere de 50; tabelele și notele de subsol au nevoie de reguli proprii")),
        (t("\\textbf{Metadata}", "\\textbf{Metadatele}"), t("speaker, date and title travel with each chunk; the answer cites them",
                                                          "vorbitorul, data și titlul însoțesc fiecare fragment; răspunsul le citează")),
        (t("\\textbf{Retriever}", "\\textbf{Regăsirea}"), t("hybrid BM25 + dense: recall@3 = @{rag.qa_r3_hybrid|.1f}\\,\\% on the question-answering speeches",
                                                          "hibrid BM25 + dense: recall@3 = @{rag.qa_r3_hybrid|.1f}\\,\\% pe discursurile folosite la întrebări")),
        (t("\\textbf{Abstention}", "\\textbf{Abținerea}"), t("allow ``I don't know'': without documents the model then abstains on @{rag.qa_closed_abstain.abstain|.1f}\\,\\% of questions",
                                                           "permiteți „nu știu”: fără documente, modelul se abține atunci la @{rag.qa_closed_abstain.abstain|.1f}\\,\\% din întrebări")),
        (t("\\textbf{Evaluation}", "\\textbf{Evaluarea}"), t("a fixed set of questions with known answers, rerun after every change", "un set fix de întrebări cu răspunsuri cunoscute, rulat din nou după fiecare modificare")),
        sep="3pt"), "small")

    # ------------------------------------------------------------------ 5. LLM-VaR
    lv = N["llmvar"]
    D.section(t("Language models as risk models: LLM-VaR", "Modelele de limbaj ca modele de risc: LLM-VaR"))
    D.concept(t("LLM-VaR: the return history in the prompt, a quantile in the answer", "LLM-VaR: istoricul randamentelor în prompt, o cuantilă în răspuns"),
              t("no estimation and no fine-tuning: the model reads 250 returns and writes the Value-at-Risk (VaR).",
                "fără estimare și fără ajustare fină: modelul citește 250 de randamente și scrie valoarea expusă la risc (VaR)."),
              "\\[ \\widehat{\\mathrm{VaR}}^{\\alpha}_{t+1}=\\mathrm{LLM}\\big(\\mathrm{prompt}(r_{t-249},\\dots,r_t;\\alpha)\\big) \\]"
              "\\[ \\mathrm{LR}_{uc}=-2\\ln\\frac{(1-\\alpha)^{n_0}\\alpha^{n_1}}{(1-\\hat\\pi)^{n_0}\\hat\\pi^{n_1}}\\sim\\chi^2_1 \\]" +
              "\\where{" + t("$r_t$ = daily log return of the S\\&P 500 (\\%); $\\alpha\\in\\{1\\,\\%,5\\,\\%\\}$; temperature 0; $n_1$ = violations; Kupiec likelihood ratio (LR) test.",
                             "$r_t$ = randamentul logaritmic zilnic al S\\&P 500 (\\%); $\\alpha\\in\\{1\\,\\%,5\\,\\%\\}$; temperatura 0; $n_1$ = depășirile; testul raportului de verosimilitate (LR) Kupiec.") + "}",
              it(t("\\textbf{Our replication}: @{llmvar.model}, @{llmvar.n} days from @{llmvar.first|date} to @{llmvar.last|date}.",
                   "\\textbf{Replicarea noastră}: @{llmvar.model}, @{llmvar.n} de zile, din @{llmvar.first|date} pînă la @{llmvar.last|date}."),
                 t("\\textbf{Benchmarks}: historical simulation (HS), 250 days; generalised autoregressive conditional heteroskedasticity (GARCH), GARCH(1,1)-$t$.",
                   "\\textbf{Repere}: simularea istorică (HS), 250 de zile; heteroscedasticitate condiționată autoregresivă generalizată (GARCH), GARCH(1,1)-$t$.")),
              t("a 3B open model on a laptop is the cheapest test of the idea; the paper used large commercial models.",
                "un model deschis de 3B pe laptop este cel mai ieftin test al ideii; lucrarea a folosit modele comerciale mari."),
              cite="(Pele et al., 2026; Kupiec, 1995)", tryit=NB + " \\S9")
    D.figframe(t("LLM-VaR with a 3B open model on the S\\&P 500", "LLM-VaR cu un model deschis de 3B pe S\\&P 500"),
               t("the language model's 5\\,\\% VaR against GARCH; red dots mark the days the LLM-VaR was breached.",
                 "VaR de 5\\,\\% a modelului de limbaj comparată cu GARCH; punctele roșii marchează zilele în care LLM-VaR a fost depășită."),
               None, "llm_var",
               bullets=[t("5\\,\\%: LLM @{llmvar.llm_05.viol} violations, GARCH @{llmvar.garch_05.viol}, HS @{llmvar.hs_05.viol}, expected @{llmvar.exp_05|.1f}.",
                          "5\\,\\%: LLM @{llmvar.llm_05.viol} depășiri, GARCH @{llmvar.garch_05.viol}, HS @{llmvar.hs_05.viol}, așteptate @{llmvar.exp_05|.1f}."),
                        t("Mean 5\\,\\% VaR: LLM @{llmvar.llm_05.mean|.2f}\\,\\%, GARCH @{llmvar.garch_05.mean|.2f}\\,\\%; correlation of the two @{llmvar.corr_llm_garch_05|.2f}: the small model covers by being cautious, not by tracking volatility.",
                          "VaR medie de 5\\,\\%: LLM @{llmvar.llm_05.mean|.2f}\\,\\%, GARCH @{llmvar.garch_05.mean|.2f}\\,\\%; corelația lor @{llmvar.corr_llm_garch_05|.2f}: modelul mic acoperă riscul fiind prudent, nu urmărind volatilitatea.")],
               height="0.64\\textheight", tryit=NB + " \\S9")
    vr = []
    for m, nm in (("llm", "LLM-VaR (Qwen2.5-3B)"), ("hs", t("HS (250 days)", "HS (250 de zile)")), ("garch", "GARCH(1,1)-$t$")):
        a, b = lv[f"{m}_01"], lv[f"{m}_05"]
        vr.append([nm, f"{a['viol']}", fmt(a["rate"], ".2f", lang), heat(a["kupiec_p"], 0, 0.1, good_low=False, shown=pv(a["kupiec_p"])),
                   f"{b['viol']}", fmt(b["rate"], ".2f", lang), heat(b["kupiec_p"], 0, 0.1, good_low=False, shown=pv(b["kupiec_p"])), fmt(b["mean"], ".2f", lang)])
    D.frame(t("Backtest of LLM-VaR against the classical benchmarks", "Testarea retrospectivă a LLM-VaR față de reperele clasice"),
            table("@{}lrrrrrrr@{}", [t("Model", "Model"), t("Viol.\\ 1\\,\\%", "Dep.\\ 1\\,\\%"), t("Rate (\\%)", "Rata (\\%)"), "$p_{uc}$",
                                     t("Viol.\\ 5\\,\\%", "Dep.\\ 5\\,\\%"), t("Rate (\\%)", "Rata (\\%)"), "$p_{uc}$", t("Mean VaR 5\\,\\%", "VaR 5\\,\\% medie")], vr) +
            note(t("Expected violations: @{llmvar.exp_01|.1f} at 1\\,\\% and @{llmvar.exp_05|.1f} at 5\\,\\%; green $p$-value: coverage not rejected at 10\\,\\%.",
                   "Depășiri așteptate: @{llmvar.exp_01|.1f} la 1\\,\\% și @{llmvar.exp_05|.1f} la 5\\,\\%; valoare $p$ verde: acoperirea nu este respinsă la 10\\,\\%."),
                 t("Answers that could not be read as a number: @{llmvar.parsed_05|.1f}\\,\\% parsed at 5\\,\\%, @{llmvar.parsed_01|.1f}\\,\\% at 1\\,\\%; an unreadable answer keeps the previous day's value.",
                   "Răspunsuri citite ca număr: @{llmvar.parsed_05|.1f}\\,\\% la 5\\,\\%, @{llmvar.parsed_01|.1f}\\,\\% la 1\\,\\%; un răspuns ilizibil păstrează valoarea din ziua precedentă."),
                 t("Model size matters: the published study evaluated much larger models; results here describe a 3B model only.",
                   "Dimensiunea modelului contează: studiul publicat a evaluat modele mult mai mari; rezultatele de aici descriu doar un model de 3B.")),
            "small", cite=t("(own computation; Pele et al., 2026)", "(calcul propriu; Pele et al., 2026)"), tryit=NB + " \\S9")

    # ------------------------------------------------------------------ 6. risks
    D.section(t("Risks and governance", "Riscuri și guvernanță"))
    D.frame(t("Six risks of LLMs in a financial institution, and the test for each", "Șase riscuri ale LLM într-o instituție financiară și testul pentru fiecare"),
            table("@{}p{0.2\\textwidth}p{0.38\\textwidth}p{0.36\\textwidth}@{}", [t("Risk", "Risc"), t("Seen in this lecture", "Observat în acest curs"), t("Control", "Control")],
                  [[t("Hallucination", "Halucinația"), t("@{rag.qa_closed.wrong|.1f}\\,\\% wrong speaker names without documents", "@{rag.qa_closed.wrong|.1f}\\,\\% nume greșite fără documente"),
                    t("RAG with sources; abstention allowed", "RAG cu surse; abținerea permisă")],
                   [t("Leakage / contamination", "Scurgerea datelor"), t("lower perplexity on older speeches", "perplexitate mai mică pe discursurile vechi"),
                    t("test on text after the cut-off", "testare pe texte ulterioare limitei datelor")],
                   [t("Prompt sensitivity", "Sensibilitatea la prompt"), t("@{sens.tw_3B.range|.1f} macro-F1 points across wordings", "@{sens.tw_3B.range|.1f} puncte macro-F1 între formulări"),
                    t("select on validation, freeze, version", "selecție pe validare, fixare, versionare")],
                   [t("Miscalibration", "Calibrarea"), t("ECE @{calib.zs_3B|.1f}\\,\\% zero-shot, @{calib.tfidf|.1f}\\,\\% TF-IDF", "ECE @{calib.zs_3B|.1f}\\,\\% zero-shot, @{calib.tfidf|.1f}\\,\\% TF-IDF"),
                    t("temperature scaling, thresholds on validation", "scalarea temperaturii, praguri stabilite pe validare")],
                   [t("Cost and latency", "Costul și latența"), t("@{cls.pb.cot_3B.ms|,.0f} ms vs @{cls.pb.tfidf.ms|.2f} ms per sentence", "@{cls.pb.cot_3B.ms|,.0f} ms față de @{cls.pb.tfidf.ms|.2f} ms pe propoziție"),
                    t("smallest model that passes the test", "cel mai mic model care trece testul")],
                   [t("Data protection", "Protecția datelor"), t("client documents in prompts", "documente ale clienților în prompturi"),
                    t("local models; duties of the European Union (EU) Artificial Intelligence (AI) Act for high-risk uses such as credit scoring", "modele locale; obligațiile din Regulamentul Uniunii Europene (UE) privind inteligența artificială (AI) pentru utilizările cu risc ridicat, precum scoringul de credit")]],
                  size="scriptsize") +
            note(t("The EU AI Act is Regulation (EU) 2024/1689.", "Regulamentul (UE) 2024/1689 privind inteligența artificială.")),
            "small")

    # ------------------------------------------------------------------ wrap-up
    D.parts.append("\\section*{" + t("Wrap-up", "Recapitulare") + "}\n")
    D.quiz_frame(t("temperature", "temperatura"),
                 t("You classify 10,000 news items with an LLM and get different labels on two runs of the same item; what do you change first?",
                   "Clasificați 10.000 de știri cu un LLM și obțineți etichete diferite la două rulări pentru aceeași știre; ce schimbați mai întîi?"),
                 t("Set temperature to 0 (or read label probabilities): sampling at $T>0$ is random by construction.",
                   "Setați temperatura la 0 (sau citiți probabilitățile etichetelor): eșantionarea la $T>0$ este aleatoare prin construcție."))
    D.quiz_frame(t("evaluation", "evaluarea"),
                 t("A vendor reports 95\\,\\% accuracy of its LLM on Financial PhraseBank; what two questions do you ask?",
                   "Un furnizor raportează o acuratețe de 95\\,\\% a LLM-ului său pe Financial PhraseBank; ce două întrebări puneți?"),
                 t("Which agreement subset (the unanimous one is easiest) and was PhraseBank in the training data (published 2014, widely copied)? Ask for macro-F1 too: "
                   "@{data.pb_share_neutral|.1f}\\,\\% of the sentences are neutral.",
                   "Pe ce subset de acord (cel unanim este cel mai ușor) și a fost PhraseBank în datele de antrenare (publicat în 2014, copiat pe scară largă)? Cereți și macro-F1: "
                   "@{data.pb_share_neutral|.1f}\\,\\% dintre propoziții sînt neutre."))
    D.quiz_frame(t("RAG", "RAG"),
                 t("Your RAG assistant answers wrongly although the right document exists in the corpus; where do you look first?",
                   "Asistentul RAG răspunde greșit, deși documentul corect există în corpus; unde căutați mai întîi?"),
                 t("At the retriever: check whether the document is in the top $k$ (recall@k); only then at the prompt and the model.",
                   "La modulul de regăsire: verificați dacă documentul este printre primele $k$ (recall@k); abia apoi la prompt și la model."))
    D.frame(t("Wrap-up", "Recapitulare"), it(
        t("An LLM is a next-token model; perplexity, temperature and tokens are its basic quantities.",
          "Un LLM este un model al următorului token; perplexitatea, temperatura și tokenii sînt mărimile sale de bază."),
        t("A prompt is a model: score it on held-out labels, with intervals, seeds and paraphrases.",
          "Un prompt este un model: evaluați-l pe etichete separate, cu intervale, rulări repetate și reformulări."),
        t("With labels, small trained models and LLM embeddings are hard to beat; without labels, prompting gives a strong start.",
          "Cu etichete, modelele mici antrenate și embedding-urile LLM sînt greu de depășit; fără etichete, prompturile oferă un început solid."),
        t("RAG turns hallucinated answers into sourced ones; evaluate the retriever first.",
          "RAG transformă răspunsurile inventate în răspunsuri cu surse; evaluați întîi modulul de regăsire."),
        t("Next: time-series foundation models for energy and finance (Lecture 13).", "Urmează: modelele fundație pentru serii de timp din energie și finanțe (cursul 13)."),
        sep="4pt"))
    D.frame(t("Research from our group", "Cercetările grupului nostru"), it(
        t("\\textbf{LLM-VaR and, for the expected shortfall (ES), LLM-ES}: general-purpose LLMs estimate both risk measures zero-shot from a structured prompt with past "
          "returns; backtested against classical risk models (Pele, Bolovăneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann \\& Härdle, 2026, "
          "Expert Systems with Applications).",
          "\\textbf{LLM-VaR și, pentru pierderea așteptată în coada distribuției (ES), LLM-ES}: LLM-uri generale estimează, fără reantrenare, ambele măsuri de risc dintr-un prompt structurat "
          "cu randamentele trecute; testate retrospectiv față de modelele clasice de risc (Pele, Bolovăneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann \\& Härdle, 2026, "
          "Expert Systems with Applications)."),
        t("\\textbf{LLM-driven stock prediction}: LLaMA 3.3 combines price history with news sentiment in the prompt and outperforms LLaMA 3.1 and ARIMA "
          "(Tak \\& Pele, 2025, PICBE).".replace("ARIMA", "an autoregressive integrated moving-average benchmark"),
          "\\textbf{Prognoza acțiunilor cu LLM}: LLaMA 3.3 combină în prompt istoricul prețurilor cu sentimentul știrilor și depășește LLaMA 3.1 și un model autoregresiv integrat "
          "cu medie mobilă (Tak \\& Pele, 2025, PICBE)."),
        t("\\textbf{Central-bank narratives}: a transformer-based sentiment index of the National Bank of Romania's minutes improves Romanian inflation forecasts "
          "(Amza, Bolovăneanu \\& Pele, 2026, PICBE) --- the same kind of text as our ECB speeches.",
          "\\textbf{Narațiunile băncii centrale}: un indice de sentiment construit cu un model transformer din minutele Băncii Naționale a României îmbunătățește prognoza inflației "
          "(Amza, Bolovăneanu \\& Pele, 2026, PICBE) --- același tip de text ca discursurile BCE din acest curs."),
        sep="4pt") + note(t("Findings as stated in the papers' abstracts; numbers on the other slides come from our own data.",
                            "Rezultatele sînt preluate din rezumatele lucrărilor; cifrele de pe celelalte slide-uri provin din datele noastre.")),
        "footnotesize", cite=t("(Bucharest University of Economic Studies, ASE; Romanian Academy, Institute for Economic Forecasting, IPE)",
                               "(Academia de Studii Economice din București, ASE; Academia Română, Institutul de Prognoză Economică, IPE)"))
    D.frame(t("Want to learn more?", "Vreți să aflați mai mult?"), cols(it(
        t("\\textbf{Hugging Face} course on LLMs: ", "\\textbf{Hugging Face}, curs despre LLM: ") + "\\href{https://huggingface.co/learn/llm-course}{huggingface.co/learn/llm-course}.",
        t("\\textbf{Qwen2.5} models and report: ", "\\textbf{Qwen2.5}, modele și raport: ") + "\\href{https://huggingface.co/Qwen}{huggingface.co/Qwen}.",
        t("\\textbf{Prompting guide}: ", "\\textbf{Ghid de formulare a prompturilor}: ") + "\\href{https://www.promptingguide.ai}{promptingguide.ai}.", sep="4pt"),
        it(t("\\textbf{ECB speeches} data set: ", "\\textbf{Setul de discursuri BCE}: ") + "\\href{https://www.ecb.europa.eu/press/key/html/downloads.en.html}{ecb.europa.eu}.",
           t("\\textbf{Quantinar}: The Next Word Problem: ", "\\textbf{Quantinar}: The Next Word Problem: ") + "\\href{https://quantinar.com/course/100100/the-next-word-problem-full-course}{quantinar.com}.",
           t("\\textbf{This lecture}: notebook, seminar and Quantlets --- ", "\\textbf{Acest curs}: notebook, seminar și Quantlets --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.", sep="4pt"), "0.5", "0.47"))
    reference_frames(D, REFS, per_frame=10)
    D.mark_main_end()

    D.parts.append("\\section*{" + t("Appendix", "Anexă") + "}\n")
    D.frame(t("Appendix A: the prompt used for zero-shot classification", "Anexa A: promptul folosit pentru clasificarea zero-shot"), it(
        (t("\\textbf{System}", "\\textbf{Sistem}"), "``You are a careful financial analyst.''"),
        (t("\\textbf{User}", "\\textbf{Utilizator}"), "``Classify the sentiment of this sentence from financial news, from the point of view of an investor in the company. "
                                                       "Answer with one word: positive, negative or neutral. Sentence: \\ldots{} Answer:''"),
        (t("\\textbf{Few-shot}", "\\textbf{Few-shot}"), t("the same, with ``Sentence: \\ldots{} Answer: label'' pairs before the input; examples balanced across classes and shuffled",
                                                          "la fel, cu perechi „Sentence: \\ldots{} Answer: etichetă” înaintea intrării; exemple echilibrate pe clase și amestecate")),
        (t("\\textbf{Chain-of-thought}", "\\textbf{Chain-of-thought}"), "``\\ldots{} First explain your reasoning in at most two short sentences, then give the final answer on a new line as `Label: <positive|negative|neutral>'.''"),
        sep="4pt"), "small")
    D.frame(t("Appendix B: implementation details", "Anexa B: detalii de implementare"), it(
        t("Models in float16 on the laptop's graphics processor (PyTorch); Hugging Face transformers; scikit-learn for TF-IDF, logistic regression and bootstrap.",
          "Modele în float16, pe procesorul grafic al laptopului (PyTorch); biblioteca Hugging Face transformers; scikit-learn pentru TF-IDF, regresia logistică și bootstrap."),
        t("Seeds: @{seed} for splits and samples; few-shot examples drawn with seeds @{seed}, +1 and +2.",
          "Valorile generatorului aleator: @{seed} pentru împărțiri și eșantioane; exemplele few-shot cu valorile @{seed}, +1 și +2."),
        t("The whole analysis took @{runtime_min|.1f} minutes on its last run (model outputs cached); first run about 40 minutes.",
          "Întreaga analiză a durat @{runtime_min|.1f} minute la ultima rulare (ieșirile modelelor stocate); prima rulare, aproximativ 40 de minute."),
        t("Per-item predictions are published with the Quantlets, so every number can be recomputed without a graphics card.",
          "Predicțiile pentru fiecare element sînt publicate împreună cu Quantlet-urile, astfel încît fiecare cifră poate fi recalculată fără placă grafică."),
        sep="4pt"))
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 12, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
