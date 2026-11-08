r"""
build_L1.py -- Lecture 1 "Deep learning for business: from linear models to the perceptron and the MLP" (EN + RO).
Writes lecture1.tex / lecture1_ro.tex from the text below and the numbers in L1_numbers.json (analysis_L1.py),
compiles both and runs every check of lectures/STYLE.md (nn_build.run_build).
Run:  python3 lectures/L1/build_L1.py   [en|ro]
Rules: no computed number is typed by hand (results are @{key} placeholders); hand-typed numbers are definitions
(0.5 threshold, the UCI 5:1 cost matrix, layer sizes chosen a priori, seeds).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_build  # noqa: E402
from nn_build import Deck, Numbers, abbreviation_frames, block, cols, items, reference_frames, run_build, table, hier  # noqa: E402

_bw = nn_build.bulletize_where


def _compact_where(tex, per_line=3):
    """Keep the course's 'where' bullet, but put up to three symbol definitions on one level-2 line (shorter frames)."""
    out, i = [], 0
    while True:
        k = tex.find("\\where{", i)
        if k < 0:
            out.append(tex[i:])
            break
        e = nn_build._balanced(tex, k + 6)
        parts = nn_build.split_clauses(tex[k + 7:e - 1])
        groups = [", ".join(parts[j:j + per_line]) for j in range(0, len(parts), per_line)]
        out.append(tex[i:k] + "\\where{" + "; ".join(groups) + "}")
        i = e
    return _bw("".join(out))


nn_build.bulletize_where = _compact_where
sys.path.insert(0, os.path.join(HERE, "..", "L1"))
import fitgrow  # noqa: E402

fitgrow.install(nn_build)   # shrink-then-grow figure fit (keeps charts as large as each frame allows)

ABBR = {
    "AI": ("artificial intelligence", "inteligență artificială"),
    "AUC": ("area under the receiver operating characteristic curve", "aria de sub curba caracteristică de operare a receptorului"),
    "CPU": ("central processing unit", "procesor central"),
    "DM": ("Deutsche Mark", "marca germană"),
    "ES": ("expected shortfall", "pierderea așteptată în coada distribuției (expected shortfall)"),
    "GELU": ("Gaussian error linear unit", "unitate liniară cu eroare gaussiană"),
    "GNN": ("graph neural network", "rețea neuronală pe grafuri"),
    "GPU": ("graphics processing unit", "procesor grafic"),
    "GraphSAGE": ("graph sample and aggregate (a GNN architecture)", "eșantionare și agregare pe graf (o arhitectură GNN)"),
    "HMEQ": ("home equity loans dataset", "setul de date despre creditele garantate cu locuința (home equity)"),
    "ICBE": ("International Conference on Business Excellence", "International Conference on Business Excellence (conferința internațională de excelență în afaceri)"),
    "IEEE-CIS": ("Institute of Electrical and Electronics Engineers, Computational Intelligence Society",
                 "Institute of Electrical and Electronics Engineers, Computational Intelligence Society"),
    "LLM": ("large language model", "model lingvistic mare"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "MSE": ("mean squared error", "eroarea pătratică medie"),
    "PD": ("probability of default", "probabilitatea de neplată"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată"),
    "ROC": ("receiver operating characteristic", "caracteristica de operare a receptorului"),
    "SGD": ("stochastic gradient descent", "coborîrea stohastică pe gradient"),
    "SHAP": ("Shapley additive explanations", "explicații aditive Shapley"),
    "UCI": ("University of California, Irvine (Machine Learning Repository)", "Universitatea California din Irvine (Machine Learning Repository)"),
    "USD": ("US dollar", "dolar american"),
    "VaR": ("value at risk", "valoarea la risc"),
    "VLM": ("vision-language model", "model vizual-lingvistic"),
    "XAI": ("explainable artificial intelligence", "inteligență artificială explicabilă"),
    "XGBoost": ("extreme gradient boosting", "extreme gradient boosting (arbori de decizie cu amplificare pe gradient)"),
    "XOR": ("exclusive or", "sau exclusiv"),
}
UNITS = {"DM", "USD"}
IGNORE = {"BF", "BIZ", "C5NC", "NEURAL", "US", "XG", "TPR", "FPR", "FN", "FP", "KDD", "MIT", "LLM-ES", "CEE", "BCE"}

NAME = {"LOAN": ("loan amount", "valoarea creditului"), "MORTDUE": ("mortgage due", "ipoteca datorată"),
        "VALUE": ("property value", "valoarea proprietății"), "YOJ": ("years on job", "ani la locul de muncă"),
        "DEROG": ("derogatory reports", "rapoarte negative"), "DELINQ": ("delinquent credit lines", "linii de credit cu restanțe"),
        "CLAGE": ("age of the oldest line", "vechimea celei mai vechi linii"), "NINQ": ("recent credit inquiries", "interogări recente"),
        "CLNO": ("number of credit lines", "numărul de linii de credit"), "DEBTINC": ("debt-to-income ratio", "raportul datorii/venit"),
        "miss_DEBTINC": ("debt-to-income missing", "raport datorii/venit lipsă")}

REFS = [
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}. Springer. "
    "\\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Baesens, B., Rösch, D.\\ \\& Scheule, H.\\ (2016). \\textit{Credit Risk Analytics}. Wiley. "
    "\\href{https://doi.org/10.1002/9781119449560}{doi:10.1002/9781119449560}",
    "Będowska-Sójka, B., Wójcik, P.\\ \\& Pele, D.\\,T.\\ (2026). Early warning systems for cryptocurrency markets: Predicting `zombie' "
    "assets using machine learning. \\textit{The North American Journal of Economics and Finance}, 81, 102543. "
    "\\href{https://doi.org/10.1016/j.najef.2025.102543}{doi:10.1016/j.najef.2025.102543}",
    "Chen, T.\\ \\& Guestrin, C.\\ (2016). XGBoost: A scalable tree boosting system. \\textit{Proceedings of KDD 2016}, 785--794. "
    "\\href{https://doi.org/10.1145/2939672.2939785}{doi:10.1145/2939672.2939785}",
    "Cybenko, G.\\ (1989). Approximation by superpositions of a sigmoidal function. \\textit{Mathematics of Control, Signals and "
    "Systems}, 2(4), 303--314. \\href{https://doi.org/10.1007/BF02551274}{doi:10.1007/BF02551274}",
    "Diaconu, D.\\ \\& Pele, D.\\,T.\\ (2026). Explainable AI in financial fraud detection: Evidence from the IEEE-CIS fraud dataset. "
    "\\textit{Proceedings of the International Conference on Business Excellence}, 20(1), 118--132. "
    "\\href{https://doi.org/10.2478/picbe-2026-0015}{doi:10.2478/picbe-2026-0015}",
    "Grinsztajn, L., Oyallon, E.\\ \\& Varoquaux, G.\\ (2022). Why do tree-based models still outperform deep learning on typical "
    "tabular data? \\textit{Advances in Neural Information Processing Systems}, 35. \\href{https://arxiv.org/abs/2207.08815}{arXiv:2207.08815}",
    "Hofmann, H.\\ (1994). Statlog (German Credit Data). UCI Machine Learning Repository. "
    "\\href{https://doi.org/10.24432/C5NC77}{doi:10.24432/C5NC77}",
    "Hornik, K.\\ (1991). Approximation capabilities of multilayer feedforward networks. \\textit{Neural Networks}, 4(2), 251--257. "
    "\\href{https://doi.org/10.1016/0893-6080(91)90009-T}{doi:10.1016/0893-6080(91)90009-T}",
    "Jheng, S.-L., Pele, D.\\,T., Tak, R.\\ \\& Găman, Ș.\\ (2026). A multimodal vision-language framework for financial anomaly "
    "detection. \\textit{Proceedings of the International Conference on Business Excellence}, 20(1), 191--201. "
    "\\href{https://doi.org/10.2478/picbe-2026-0020}{doi:10.2478/picbe-2026-0020}",
    "Jheng, S.-L., Tak, R., Găman, Ș., Mazurencu-Marinescu-Pele, M.\\ \\& Pele, D.\\,T.\\ (2026). Can foundation models manage risk? "
    "Zero-shot VaR and ES forecasting with conformal calibration in CEE markets. \\textit{Economic Computation and Economic "
    "Cybernetics Studies and Research}, 60(2), 267--290. \\href{https://doi.org/10.24818/18423264/60.2.26.14}{doi:10.24818/18423264/60.2.26.14}",
    "LeCun, Y., Bengio, Y.\\ \\& Hinton, G.\\ (2015). Deep learning. \\textit{Nature}, 521, 436--444. "
    "\\href{https://doi.org/10.1038/nature14539}{doi:10.1038/nature14539}",
    "Lessmann, S., Baesens, B., Seow, H.-V.\\ \\& Thomas, L.\\,C.\\ (2015). Benchmarking state-of-the-art classification algorithms "
    "for credit scoring: An update of research. \\textit{European Journal of Operational Research}, 247(1), 124--136. "
    "\\href{https://doi.org/10.1016/j.ejor.2015.05.030}{doi:10.1016/j.ejor.2015.05.030}",
    "McCulloch, W.\\,S.\\ \\& Pitts, W.\\ (1943). A logical calculus of the ideas immanent in nervous activity. \\textit{The Bulletin "
    "of Mathematical Biophysics}, 5(4), 115--133. \\href{https://doi.org/10.1007/BF02478259}{doi:10.1007/BF02478259}",
    "Minsky, M.\\ \\& Papert, S.\\ (1969). \\textit{Perceptrons: An Introduction to Computational Geometry}. MIT Press.",
    "Nadeau, C.\\ \\& Bengio, Y.\\ (2003). Inference for the generalization error. \\textit{Machine Learning}, 52(3), 239--281. "
    "\\href{https://doi.org/10.1023/A:1024068626366}{doi:10.1023/A:1024068626366}",
    "Novikoff, A.\\,B.\\,J.\\ (1962). On convergence proofs on perceptrons. \\textit{Proceedings of the Symposium on the Mathematical "
    "Theory of Automata}, 12, 615--622. Polytechnic Institute of Brooklyn.",
    "Pace, R.\\,K.\\ \\& Barry, R.\\ (1997). Sparse spatial autoregressions. \\textit{Statistics \\& Probability Letters}, 33(3), "
    "291--297. \\href{https://doi.org/10.1016/S0167-7152(96)00140-X}{doi:10.1016/S0167-7152(96)00140-X}",
    "Montúfar, G., Pascanu, R., Cho, K.\\ \\& Bengio, Y.\\ (2014). On the number of linear regions of deep neural networks. "
    "\\textit{Advances in Neural Information Processing Systems}, 27. \\href{https://arxiv.org/abs/1402.1869}{arXiv:1402.1869}",
    "Paszke, A.\\ et al.\\ (2019). PyTorch: An imperative style, high-performance deep learning library. \\textit{Advances in Neural "
    "Information Processing Systems}, 32. \\href{https://arxiv.org/abs/1912.01703}{arXiv:1912.01703}",
    "Pele, D.\\,T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A.\\,T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S.\\ \\& "
    "Härdle, W.\\,K.\\ (2026). In the beginning was the Word: LLM-VaR and LLM-ES. \\textit{Expert Systems with Applications}, 295, "
    "128676. \\href{https://doi.org/10.1016/j.eswa.2025.128676}{doi:10.1016/j.eswa.2025.128676}",
    "Rosenblatt, F.\\ (1958). The perceptron: A probabilistic model for information storage and organization in the brain. "
    "\\textit{Psychological Review}, 65(6), 386--408. \\href{https://doi.org/10.1037/h0042519}{doi:10.1037/h0042519}",
    "Rumelhart, D.\\,E., Hinton, G.\\,E.\\ \\& Williams, R.\\,J.\\ (1986). Learning representations by back-propagating errors. "
    "\\textit{Nature}, 323, 533--536. \\href{https://doi.org/10.1038/323533a0}{doi:10.1038/323533a0}",
]


def it(*xs, sep="2pt"):
    return items(*xs, opt=f"\\setlength{{\\itemsep}}{{{sep}}}")


def note(*xs):
    return "\n\\par\\vspace{0.5mm}{\\scriptsize " + items(*xs, opt="\\setlength{\\itemsep}{0pt}", nest=False) + "}"


def code(*lines):
    esc = lambda s: (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("{", "\\{").replace("}", "\\}")  # noqa: E731
                     .replace("#", "\\#").replace("%", "\\%").replace("&", "\\&").replace("~", "\\textasciitilde{}")
                     .replace("^", "\\textasciicircum{}").replace("  ", "~~").replace("\\textbackslash\\{\\}", "\\textbackslash{}"))
    return ("{\\ttfamily\\scriptsize\\setlength{\\parindent}{0pt}" + "\\\\\n".join(esc(l) if l else "~" for l in lines) + "\\par}")


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L1_numbers.json"), lang)
    N = V.N
    ro = lang == "ro"
    D = Deck(1, ("Deep learning for business: from linear models to the perceptron and the multilayer perceptron (MLP)" if not ro else
                 "Deep learning pentru afaceri: de la modelele liniare la perceptron și la perceptronul multistrat (MLP)"),
             V, lang, heights=H)
    t = D.t
    NB = t("notebook", "notebook")
    nm = lambda v: NAME[v][1 if ro else 0]  # noqa: E731
    OWN = t("(own computation)", "(calcul propriu)")

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Place deep learning among the tools of a business analyst: where it creates value and where a linear model is enough.",
          "Să situați deep learning printre instrumentele unui analist de afaceri: unde creează valoare și unde un model liniar este suficient."),
        t("Write linear and logistic regression as one-neuron networks, with their loss as a likelihood.",
          "Să scrieți regresia liniară și regresia logistică drept rețele cu un singur neuron, cu funcția de pierdere ca verosimilitate."),
        t("Explain Rosenblatt's perceptron, its learning rule, its convergence theorem and its limit (the exclusive or (XOR) problem).",
          "Să explicați perceptronul lui Rosenblatt, regula lui de învățare, teorema de convergență și limita lui (problema sau-ului exclusiv (XOR))."),
        t("Build a multilayer perceptron (MLP): forward pass, activation functions, number of parameters, universal approximation.",
          "Să construiți un perceptron multistrat (MLP): propagarea înainte, funcțiile de activare, numărul de parametri, aproximarea universală."),
        t("Compare an MLP with logistic regression and gradient boosting for credit scoring on real data, with costs and uncertainty.",
          "Să comparați un MLP cu regresia logistică și cu gradient boosting pentru scoringul de credit pe date reale, cu costuri și incertitudine."),
        sep="4pt"))
    abbreviation_frames(D, ABBR)

    # ================================================================== 1. overview
    D.section(t("Deep learning for business", "Deep learning pentru afaceri"))
    D.frame(t("Deep learning learns the features that analysts used to build by hand",
              "Deep learning învață singur variabilele pe care analiștii le construiau manual"),
            "\\intuition{" + t("deep learning, the branch of artificial intelligence (AI) behind today's image, text and speech systems, fits a chain of simple functions to data.",
                               "deep learning, ramura inteligenței artificiale (AI) din spatele sistemelor actuale pentru imagini, text și vorbire, estimează din date un lanț de funcții simple.") + "}\n" +
            "\\[ f(\\vx;\\btheta)=f^{(L)}\\big(f^{(L-1)}(\\cdots f^{(1)}(\\vx))\\big),\\qquad "
            "\\hat\\btheta=\\argmin_{\\btheta}\\ \\frac1n\\sum_{i=1}^{n}\\ell\\big(y_i,f(\\vx_i;\\btheta)\\big) \\]\n" +
            "\\where{" + t("$\\vx$ the input (applicant, transaction, text); $f^{(l)}$ layer $l$ of $L$; $\\btheta$ all weights; $\\ell$ the loss; $n$ the sample size",
                           "$\\vx$ intrarea (solicitant, tranzacție, text); $f^{(l)}$ stratul $l$ din $L$; $\\btheta$ toate ponderile; $\\ell$ pierderea; $n$ volumul eșantionului") + "}\n" + it(
                t("Why now: data, hardware and software arrived together; large labelled datasets, graphics processing units (GPU) and "
                  "automatic differentiation in PyTorch (Paszke et al., 2019).",
                  "De ce acum: datele, hardware-ul și software-ul au apărut în același timp; seturi mari de date etichetate, procesoare "
                  "grafice (GPU) și diferențiere automată în PyTorch (Paszke et al., 2019)."),
                t("What changed: representation learning; the network builds its own features from raw inputs (LeCun, Bengio \\& Hinton, 2015).",
                  "Ce s-a schimbat: învățarea reprezentărilor; rețeaua își construiește singură variabilele din datele brute (LeCun, Bengio \\& Hinton, 2015).")) +
            D.bottom(t("ask what the network would learn that a scorecard analyst cannot write down by hand.",
                       "întrebați ce ar învăța rețeaua în plus față de ce poate scrie manual un analist de scoring.")),
            cite="(Bishop \\& Bishop, 2024, ch.~1, 6)")
    D.frame(t("Where deep learning creates value in business and finance", "Unde creează valoare deep learning în afaceri și finanțe"),
            table("@{}p{0.20\\textwidth}p{0.30\\textwidth}p{0.20\\textwidth}p{0.22\\textwidth}@{}",
                  [t("Domain", "Domeniu"), t("Decision", "Decizie"), t("Network", "Rețea"), t("Course data", "Datele cursului")],
                  [[t("Credit", "Credit"), t("accept or reject a loan; price the risk", "acceptarea sau respingerea unui credit; prețul riscului"),
                    t("MLP (Lectures 1, 3)", "MLP (cursurile 1, 3)"), t("German credit, home equity (HMEQ)", "German credit, credite garantate cu locuința (HMEQ)")],
                   [t("Payments", "Plăți"), t("block or let a card transaction pass", "blocarea sau aprobarea unei tranzacții cu cardul"),
                    t("MLP, autoencoder", "MLP, autoencoder"), t("card fraud (Lectures 2, 3, 8)", "fraude cu carduri (cursurile 2, 3, 8)")],
                   [t("Markets", "Piețe"), t("size a position; set a risk limit", "dimensionarea unei poziții; o limită de risc"),
                    t("recurrent, transformer", "recurentă, transformer"), t("indices, Bucharest shares", "indici, acțiuni de la Bursa de Valori București")],
                   [t("Energy", "Energie"), t("bid on the day-ahead market", "oferta pe piața pentru ziua următoare"),
                    t("recurrent, foundation models", "recurentă, modele fundație"), t("Romanian power prices", "prețurile energiei din România")],
                   [t("Text", "Text"), t("read news, filings, central-bank speeches", "citirea știrilor, rapoartelor, discursurilor"),
                    t("embeddings, transformers", "embedding, transformer"), t("PhraseBank, tweets, speeches", "PhraseBank, tweet-uri, discursuri")],
                   [t("Networks", "Rețele"), t("flag money laundering in a payment graph", "semnalarea spălării banilor într-un graf de plăți"),
                    t("graph networks", "rețele pe grafuri"), "Elliptic Bitcoin"]], size="scriptsize") +
            note(t("The course follows this table: Lectures 1--3 tabular data, 4 images, 5 and 13 time series, 6, 7 and 12 text, "
                   "9 graphs, 10 decisions, 11 explanations.",
                   "Cursul urmează acest tabel: cursurile 1--3 date tabelare, 4 imagini, 5 și 13 serii de timp, 6, 7 și 12 text, "
                   "9 grafuri, 10 decizii, 11 explicații.")) +
            D.bottom(t("start from the decision and its cost, then pick the network that fits the data type.",
                       "porniți de la decizie și de la costul ei, apoi alegeți rețeaua potrivită tipului de date.")))
    D.frame(t("For small tabular data, a linear model or boosting is often enough", "Pentru date tabelare puține, un model liniar sau boosting este adesea suficient"),
            "\\intuition{" + t("deep learning wins on raw, high-dimensional inputs; on a few dozen engineered columns it must earn its place.",
                               "deep learning cîștigă pe date brute, de dimensiune mare; pe cîteva zeci de coloane construite trebuie să-și justifice locul.") + "}\n" +
            cols(it(t("Credit scoring benchmark: across 41 classifiers on 8 credit datasets, logistic regression stays a strong reference; "
                      "ensembles lead (Lessmann et al., 2015).",
                      "Comparația metodelor de scoring: din 41 de clasificatori pe 8 seturi de date de credit, regresia logistică rămîne o "
                      "referință solidă; ansamblurile de modele conduc (Lessmann et al., 2015)."),
                    t("Tabular benchmark: tree ensembles still beat deep networks on typical medium-sized tabular data "
                      "(Grinsztajn, Oyallon \\& Varoquaux, 2022).",
                      "Comparația pe date tabelare: ansamblurile de arbori încă depășesc rețelele adînci pe date tabelare obișnuite, "
                      "de mărime medie (Grinsztajn, Oyallon \\& Varoquaux, 2022)."),
                    t("Regulation: a credit decision must be explained to the applicant.",
                      "Reglementare: decizia de creditare trebuie explicată solicitantului.")),
                 it(t("Use a network when: inputs are images, text, sequences or graphs; data are large; the relation is strongly nonlinear.",
                      "Folosiți o rețea cînd: intrările sînt imagini, text, secvențe sau grafuri; datele sînt multe; relația este puternic neliniară."),
                    t("Keep the benchmark when: a few thousand rows, engineered features, a need for a scorecard.",
                      "Păstrați modelul de referință cînd: avem cîteva mii de rînduri, variabile construite, nevoie de un scorecard."),
                    t("This lecture tests both cases on real loans, with a fixed protocol.",
                      "Acest curs testează ambele situații pe credite reale, după un protocol fix.")), "0.52", "0.45") +
            D.bottom(t("every network in this course is reported next to a linear or boosting benchmark.",
                       "fiecare rețea din acest curs este raportată alături de un model de referință liniar sau de boosting.")))
    D.frame(t("Three real credit datasets, used without any simulated record", "Trei seturi reale de date de credit, fără nicio înregistrare simulată"),
            table("@{}p{0.27\\textwidth}rrp{0.40\\textwidth}@{}",
                  [t("Dataset", "Setul de date"), t("Rows", "Rînduri"), t("Events", "Evenimente"), t("Source", "Sursa")],
                  [["German credit", "@{data.german_n|,d}", "@{data.german_bad|,d} " + t("bad risks", "riscuri proaste"),
                    t("University of California, Irvine (UCI) Machine Learning Repository (Hofmann, 1994)",
                      "Universitatea California din Irvine (UCI), Machine Learning Repository (Hofmann, 1994)")],
                   [t("Home equity loans (HMEQ)", "Credite garantate cu locuința (HMEQ)"), "@{data.hmeq_n|,d}",
                    "@{data.hmeq_bad|,d} " + t("defaults", "neplăți"), t("Baesens, Rösch \\& Scheule (2016)", "Baesens, Rösch \\& Scheule (2016)")],
                   ["California housing", "@{data.california_n|,d}", t("(regression)", "(regresie)"),
                    t("1990 census block groups (Pace \\& Barry, 1997)", "grupuri de locuințe, recensămîntul din 1990 (Pace \\& Barry, 1997)")]]) +
            note(t("German credit: @{data.german_vars} attributes, amounts in Deutsche Mark (DM); default rate @{data.german_bad_pct|.1f}\\,\\%.",
                   "German credit: @{data.german_vars} de atribute, sume în mărci germane (DM); rata de neplată @{data.german_bad_pct|.1f}\\,\\%."),
                 t("HMEQ: @{data.hmeq_vars} variables, default rate @{data.hmeq_bad_pct|.1f}\\,\\%; missing values are kept and modelled.",
                   "HMEQ: @{data.hmeq_vars} variabile, rata de neplată @{data.hmeq_bad_pct|.1f}\\,\\%; valorile lipsă sînt păstrate și modelate."),
                 t("California: median house value in 100,000 US dollars (USD), capped by the source at 5.",
                   "California: valoarea mediană a locuințelor în sute de mii de dolari americani (USD), plafonată de sursă la 5.")) +
            D.bottom(t("check the source, the unit of observation and the event rate before any model.",
                       "verificați sursa, unitatea de observație și frecvența evenimentului înaintea oricărui model."), NB + " \\S1"))
    D.figframe(t("Default rates differ strongly across simple applicant groups", "Ratele de neplată diferă puternic între grupuri simple de solicitanți"),
               t("before any model, raw default rates already show which variables carry risk.",
                 "înaintea oricărui model, ratele brute de neplată arată ce variabile poartă risc."),
               None, "default_rates",
               bullets=[t("Probability of default (PD): @{eda.chk.A11.pct|.1f}\\,\\% with a negative checking balance, @{eda.chk.A14.pct|.1f}\\,\\% with no "
                          "account; @{eda.delinq.0.pct|.1f}\\,\\% with no delinquent line, @{eda.delinq.3plus.pct|.1f}\\,\\% with three or more.",
                          "Probabilitatea de neplată (PD): @{eda.chk.A11.pct|.1f}\\,\\% cu sold negativ în contul curent, @{eda.chk.A14.pct|.1f}\\,\\% fără cont; "
                          "@{eda.delinq.0.pct|.1f}\\,\\% fără linii cu restanțe, @{eda.delinq.3plus.pct|.1f}\\,\\% cu trei sau mai multe.")],
               job=t("a model must at least reproduce these group rates on the validation set.",
                     "un model trebuie să reproducă cel puțin aceste rate pe setul de validare."),
               tryit=NB + " \\S1", cite=OWN, height="0.56\\textheight")
    D.frame(t("Learning = minimising the average loss on training data, judged on new data",
              "Învățarea = minimizarea pierderii medii pe datele de antrenare, judecată pe date noi"),
            "\\intuition{" + t("we fit on one sample and pay for the errors on the next applicants.",
                               "estimăm pe un eșantion și plătim pentru erorile făcute cu următorii solicitanți.") + "}\n" +
            "\\[ \\underbrace{\\hat R_n(\\btheta)=\\frac1n\\sum_{i=1}^n \\ell\\big(y_i,f(\\vx_i;\\btheta)\\big)}_{\\text{" +
            t("empirical risk (training)", "riscul empiric (antrenare)") + "}}\\qquad\\text{vs.}\\qquad "
            "\\underbrace{R(\\btheta)=\\E_{(\\vx,y)\\sim P}\\,\\ell\\big(y,f(\\vx;\\btheta)\\big)}_{\\text{" +
            t("risk on new applicants", "riscul pe solicitanți noi") + "}} \\]\n" +
            "\\where{$P$ the unknown distribution of future applicants; $\\ell$ the loss; $\\hat R_n$ is minimised, $R$ is what the bank pays}\n".replace(
                "the unknown distribution of future applicants", t("the unknown distribution of future applicants", "distribuția necunoscută a viitorilor solicitanți")).replace(
                "the loss; $\\hat R_n$ is minimised, $R$ is what the bank pays", t("the loss; $\\hat R_n$ is minimised, $R$ is what the bank pays",
                                                                                "pierderea; minimizăm $\\hat R_n$, banca plătește $R$")) +
            it(t("Three sets: train (fit $\\btheta$), validation (choose the model), test (report once).",
                 "Trei seturi: antrenare (estimăm $\\btheta$), validare (alegem modelul), testare (raportăm o dată)."),
               t("HMEQ, stratified: @{hmeq.n_train|,d} train, @{hmeq.n_val|,d} validation, @{hmeq.n_test|,d} test loans.",
                 "HMEQ, stratificat: @{hmeq.n_train|,d} de credite de antrenare, @{hmeq.n_val|,d} de validare, @{hmeq.n_test|,d} de testare."),
               t("No leakage: medians and scaling are fitted on the training part only.",
                 "Fără scurgere de informație: medianele și standardizarea se estimează doar pe partea de antrenare.")) +
            D.bottom(t("the test set is used once, at the end (validation design: Lecture 3).",
                       "setul de testare se folosește o singură dată, la final (schema de validare: cursul 3)."), NB + " \\S2"),
            cite="(Bishop \\& Bishop, 2024, ch.~1)")

    # ================================================================== 2. linear models
    D.section(t("Linear models are one-neuron networks", "Modelele liniare sînt rețele cu un singur neuron"))
    D.concept(t("Linear regression is one neuron with no activation", "Regresia liniară este un neuron fără funcție de activare"),
              t("a weighted sum of the inputs plus a bias; the squared loss is the Gaussian log-likelihood.",
                "o sumă ponderată a intrărilor plus un termen liber; pierderea pătratică este log-verosimilitatea gaussiană."),
              "\\[ \\hat y=\\vw^\\top\\vx+b,\\qquad \\mathrm{MSE}(\\vw,b)=\\frac1n\\sum_{i=1}^n\\big(y_i-\\vw^\\top\\vx_i-b\\big)^2 \\]\n"
              "\\where{$\\vx\\in\\R^d$ the inputs; $\\vw$ the weights; $b$ the bias; mean squared error (MSE) the loss}".replace(
                  "the inputs; $\\vw$ the weights; $b$ the bias; mean squared error (MSE) the loss",
                  t("the inputs; $\\vw$ the weights; $b$ the bias; mean squared error (MSE) the loss",
                    "intrările; $\\vw$ ponderile; $b$ termenul liber; eroarea pătratică medie (MSE) pierderea")) +
              "\\[ \\varepsilon_i\\iid\\mathcal N(0,\\sigma^2):\\ \\ -\\log L=\\tfrac{n}{2}\\log 2\\pi\\sigma^2+\\tfrac{n}{2\\sigma^2}\\mathrm{MSE} \\]",
              it(t("Maximum likelihood = least squares: the same $\\hat\\vw=(\\mX^\\top\\mX)^{-1}\\mX^\\top\\vy$.",
                   "Verosimilitate maximă = cele mai mici pătrate: același $\\hat\\vw=(\\mX^\\top\\mX)^{-1}\\mX^\\top\\vy$."),
                 t("A network has no closed form: it is trained by gradient descent (Lecture 2).",
                   "O rețea nu are o formulă explicită: se antrenează prin coborîre pe gradient (cursul 2)."),
                 t("California housing, latitude only: held-out MSE @{uat.mse_lin|.3f} for the line.",
                   "California, doar latitudinea: MSE în afara eșantionului @{uat.mse_lin|.3f} pentru dreaptă.")),
              t("a linear fit is the benchmark every regression network must beat out of sample.",
                "dreapta de regresie este reperul pe care orice rețea de regresie trebuie să-l depășească în afara eșantionului."),
              cite="(Bishop \\& Bishop, 2024, ch.~4)")
    D.concept(t("Logistic regression is one neuron with a sigmoid", "Regresia logistică este un neuron cu funcție sigmoidă"),
              t("the neuron outputs a probability, and the loss is the Bernoulli negative log-likelihood (cross-entropy).",
                "neuronul produce o probabilitate, iar pierderea este minus log-verosimilitatea Bernoulli (entropia încrucișată)."),
              "\\[ p_i=\\sigma(\\vw^\\top\\vx_i+b)=\\frac{1}{1+e^{-z_i}},\\qquad \\nabla_{\\vw}\\loss=\\frac1n\\sum_{i=1}^n (p_i-y_i)\\,\\vx_i \\]\n"
              "\\[ \\loss(\\vw,b)=-\\frac1n\\sum_{i=1}^n\\big[y_i\\log p_i+(1-y_i)\\log(1-p_i)\\big] \\]\n" +
              "\\where{" + t("$z_i$ the logit (log-odds); $\\sigma$ the sigmoid; $y_i\\in\\{0,1\\}$ default; derivation in the appendix",
                             "$z_i$ logitul (logaritmul șansei); $\\sigma$ funcția sigmoidă; $y_i\\in\\{0,1\\}$ neplata; derivarea în anexă") + "}",
              it(t("The gradient is error times input: the same form will appear in every layer of backpropagation.",
                   "Gradientul este eroarea înmulțită cu intrarea: aceeași formă apare în fiecare strat la propagarea înapoi."),
                 t("$e^{w_j}$ is the odds ratio for a one-unit change of $x_j$.",
                   "$e^{w_j}$ este raportul șanselor pentru o creștere cu o unitate a lui $x_j$."),
                 t("Convex loss: one global minimum.", "Pierdere convexă: un singur minim global.")),
              t("the logistic score is the industry standard for PD models.",
                "scorul logistic este standardul industriei pentru modelele PD."),
              cite="(Bishop \\& Bishop, 2024, ch.~5)")
    D.figframe(t("Delinquencies and a missing debt ratio raise the odds of default most",
                 "Restanțele și lipsa raportului datorii/venit cresc cel mai mult șansa de neplată"),
               t("HMEQ, standardised inputs: each coefficient is the change in log-odds per one standard deviation.",
                 "HMEQ, intrări standardizate: fiecare coeficient este variația log-șansei la o abatere standard."),
               None, "logit_coef",
               bullets=[t("Odds ratios $e^{w_j}$: @{logit.or.miss_DEBTINC|.2f} for a missing debt ratio, @{logit.or.DELINQ|.2f} for delinquent lines, "
                          "@{logit.or.CLAGE|.2f} for an older credit history (McFadden pseudo-$R^2$ @{logit.pseudo_r2|.3f}).",
                          "Rapoartele șanselor $e^{w_j}$: @{logit.or.miss_DEBTINC|.2f} pentru raportul datorii/venit lipsă, @{logit.or.DELINQ|.2f} pentru "
                          "liniile cu restanțe, @{logit.or.CLAGE|.2f} pentru un istoric mai vechi (pseudo-$R^2$ McFadden @{logit.pseudo_r2|.3f}).")],
               job=t("a sign against credit sense (property value) is a reason to inspect the data.",
                     "un semn contrar intuiției (valoarea proprietății) este un motiv să analizați datele."),
               tryit=NB + " \\S2", cite=OWN, height="0.56\\textheight")
    top = N["one"]["top"]
    D.frame(t("The forward pass of one real applicant, step by step", "Propagarea înainte pentru un solicitant real, pas cu pas"),
            "\\intuition{" + t("the score is a sum of contributions $w_jx_j$; the sigmoid turns it into a probability.",
                               "scorul este o sumă de contribuții $w_jx_j$; funcția sigmoidă îl transformă într-o probabilitate.") + "}\n" +
            cols(table("@{}lrrr@{}", [t("Input", "Intrare"), "$x_j$", "$w_j$", "$w_jx_j$"],
                       [[nm(top[k]["var"]), f"@{{one.top.{k}.x|.2f}}", f"@{{one.top.{k}.b|.3f}}", f"@{{one.top.{k}.c|+.2f}}"] for k in range(3)] +
                       [[t("bias $b$", "termenul liber $b$"), "", "", "@{logit.intercept|+.2f}"],
                        [t("8 other inputs", "celelalte 8 intrări"), "", "", "@{one.rest|+.2f}"],
                        ["\\textbf{" + t("score", "scor") + " $z$}", "", "", "\\textbf{@{one.z|+.2f}}"]]),
                 it(t("Applicant from the test set who later defaulted: loan @{one.loan|,d} USD, debt-to-income @{one.debtinc|.1f}\\,\\%, "
                      "@{one.delinq} delinquent lines.",
                      "Solicitant din setul de testare care a intrat ulterior în neplată: credit de @{one.loan|,d} USD, raport datorii/venit "
                      "@{one.debtinc|.1f}\\,\\%, @{one.delinq} linii cu restanțe."),
                    t("Output: $p=\\sigma(z)=$ @{one.p_pct|.1f}\\,\\%, below the 50\\,\\% default cut-off.",
                      "Ieșirea: $p=\\sigma(z)=$ @{one.p_pct|.1f}\\,\\%, sub pragul implicit de 50\\,\\%."),
                    t("Inputs $x_j$ are standardised: $x_j=(\\text{raw}_j-\\bar x_j)/s_j$ with training means and standard deviations.",
                      "Intrările $x_j$ sînt standardizate: $x_j=(\\text{brut}_j-\\bar x_j)/s_j$, cu mediile și abaterile standard de antrenare.")),
                 "0.50", "0.47") +
            D.bottom(t("the cut-off is a business choice: Section 5 sets it from the cost of a missed default.",
                       "pragul este o decizie de afaceri: secțiunea 5 îl stabilește din costul unei neplăți neidentificate."), NB + " \\S2"),
            cite=OWN)
    D.frame(t("A missing value can be the strongest signal", "O valoare lipsă poate fi cel mai puternic semnal"),
            "\\intuition{" + t("applicants who do not report their debt ratio default far more often: missingness is information.",
                               "solicitanții care nu își declară raportul datorii/venit intră mult mai des în neplată: lipsa datelor este o informație.") + "}\n" +
            "\\[ \\tilde x_{ij}=\\begin{cases}x_{ij}&\\text{" + t("observed", "observat") + "}\\\\ \\mathrm{med}_{\\text{train}}(x_j)&\\text{" +
            t("missing", "lipsă") + "}\\end{cases},\\qquad m_{ij}=\\mathbb 1\\{x_{ij}\\ \\text{" + t("missing", "lipsă") + "}\\} \\]\n" +
            it(t("Debt-to-income is missing for @{data.hmeq_debtinc_missing_pct|.1f}\\,\\% of HMEQ loans.",
                 "Raportul datorii/venit lipsește pentru @{data.hmeq_debtinc_missing_pct|.1f}\\,\\% dintre creditele HMEQ."),
               t("Default rate @{data.hmeq_bad_debtinc_missing_pct|.1f}\\,\\% when it is missing against @{data.hmeq_bad_debtinc_present_pct|.1f}\\,\\% "
                 "when it is reported.",
                 "Rata de neplată este @{data.hmeq_bad_debtinc_missing_pct|.1f}\\,\\% cînd lipsește, față de @{data.hmeq_bad_debtinc_present_pct|.1f}\\,\\% "
                 "cînd este declarat."),
               t("Each of the 10 numeric inputs gets a median fill and an indicator $m_{ij}$; with 8 dummies for reason and job, the network sees "
                 "$d=$ @{hmeq.d} inputs.",
                 "Fiecare dintre cele 10 intrări numerice primește o completare cu mediana și un indicator $m_{ij}$; cu 8 variabile binare pentru "
                 "scop și ocupație, rețeaua vede $d=$ @{hmeq.d} intrări.")) +
            D.bottom(t("never drop rows with missing values in credit data: you would drop the riskiest applicants.",
                       "nu eliminați rîndurile cu valori lipsă din datele de credit: ați elimina solicitanții cei mai riscanți."), NB + " \\S2"),
            cite="(Baesens, Rösch \\& Scheule, 2016)")

    # ================================================================== 3. perceptron
    D.section(t("The perceptron", "Perceptronul"))
    D.concept(t("From a logical neuron to a learning machine", "De la neuronul logic la mașina care învață"),
              t("McCulloch and Pitts wrote the neuron as a threshold unit; Rosenblatt made its weights learn from examples.",
                "McCulloch și Pitts au scris neuronul ca unitate cu prag; Rosenblatt a făcut ca ponderile lui să învețe din exemple."),
              "\\[ \\hat y=\\operatorname{step}(\\vw^\\top\\vx+b)=\\begin{cases}+1&\\vw^\\top\\vx+b>0\\\\-1&\\text{" +
              t("otherwise", "altfel") + "}\\end{cases} \\]\n" +
              "\\where{" + t("$\\vx$ the inputs (dendrites); $\\vw$ the synaptic weights; $-b$ the firing threshold; $\\hat y$ the class (axon)",
                             "$\\vx$ intrările (dendritele); $\\vw$ ponderile sinaptice; $-b$ pragul de activare; $\\hat y$ clasa (axonul)") + "}",
              it(t("1943: logical calculus of threshold neurons (McCulloch \\& Pitts, 1943).",
                   "1943: calculul logic al neuronilor cu prag (McCulloch \\& Pitts, 1943)."),
                 t("1958: the perceptron learns $\\vw$ from labelled examples (Rosenblatt, 1958).",
                   "1958: perceptronul învață $\\vw$ din exemple etichetate (Rosenblatt, 1958)."),
                 t("1969: a single layer cannot represent XOR (Minsky \\& Papert, 1969).",
                   "1969: un singur strat nu poate reprezenta XOR (Minsky \\& Papert, 1969)."),
                 t("1986: backpropagation trains several layers (Rumelhart, Hinton \\& Williams, 1986).",
                   "1986: propagarea înapoi a erorii antrenează mai multe straturi (Rumelhart, Hinton \\& Williams, 1986).")),
              t("the perceptron is a linear classifier: its boundary $\\vw^\\top\\vx+b=0$ is a hyperplane.",
                "perceptronul este un clasificator liniar: frontiera lui $\\vw^\\top\\vx+b=0$ este un hiperplan."),
              cite="(Bishop \\& Bishop, 2024, ch.~5)")
    D.concept(t("The perceptron learns only from its mistakes", "Perceptronul învață numai din greșelile sale"),
              t("when an example is misclassified, the weights move towards it; otherwise nothing changes.",
                "cînd un exemplu este clasificat greșit, ponderile se mută spre el; altfel nu se schimbă nimic."),
              "\\[ \\text{" + t("if", "dacă") + "}\\ y_i(\\vw^\\top\\vx_i+b)\\le 0:\\qquad \\vw\\leftarrow\\vw+\\eta\\,y_i\\vx_i,\\quad b\\leftarrow b+\\eta\\,y_i \\]\n"
              "\\[ \\loss_P(\\vw,b)=-\\!\\!\\sum_{i\\in\\mathcal M}\\! y_i(\\vw^\\top\\vx_i+b) \\]\n" +
              "\\where{" + t("$y_i\\in\\{-1,+1\\}$; $\\eta>0$ the learning rate; $\\mathcal M$ the set of misclassified examples; "
                             "$\\loss_P$ the perceptron criterion",
                             "$y_i\\in\\{-1,+1\\}$; $\\eta>0$ rata de învățare; $\\mathcal M$ mulțimea exemplelor clasificate greșit; "
                             "$\\loss_P$ criteriul perceptronului") + "}",
              it(t("The update is one step of stochastic gradient descent (SGD) on $\\loss_P$ for one misclassified example.",
                   "Actualizarea este un pas de coborîre stohastică pe gradient (SGD) pe $\\loss_P$, pentru un exemplu clasificat greșit."),
                 t("Unlike logistic regression, correct examples give no gradient: the perceptron ignores how confident it is.",
                   "Spre deosebire de regresia logistică, exemplele corecte nu dau gradient: perceptronul ignoră cît de sigur este."),
                 t("No probability comes out: only a class.", "Nu rezultă nicio probabilitate, ci doar o clasă.")),
              t("for credit, a class without a probability cannot be priced or set against a cost.",
                "pentru credit, o clasă fără probabilitate nu poate fi evaluată și nici comparată cu un cost."),
              cite="(Rosenblatt, 1958; Bishop \\& Bishop, 2024, ch.~5)")
    D.frame(t("If the classes can be separated, the perceptron stops after finitely many mistakes",
              "Dacă clasele pot fi separate, perceptronul se oprește după un număr finit de greșeli"),
            "\\intuition{" + t("a wide margin between the classes means few corrections are needed.",
                               "o margine largă între clase înseamnă că sînt necesare puține corecții.") + "}\n" +
            "\\[ \\exists\\,\\vw^*,\\ \\|\\vw^*\\|=1:\\ y_i\\,\\vw^{*\\top}\\vx_i\\ge\\gamma>0\\ \\ \\forall i,\\quad \\|\\vx_i\\|\\le R\\ "
            "\\ \\Longrightarrow\\ \\ \\#\\text{" + t("mistakes", "greșeli") + "}\\le\\Big(\\frac{R}{\\gamma}\\Big)^2 \\]\n" +
            "\\where{" + t("$\\gamma$ the margin of the best separating hyperplane; $R$ the radius of the data; the bias is absorbed in $\\vx$ (a constant 1)",
                           "$\\gamma$ marginea celui mai bun hiperplan separator; $R$ raza datelor; termenul liber este inclus în $\\vx$ (o constantă 1)") + "}\n" +
            it(t("Novikoff's theorem (1962): the bound does not depend on the number of examples or on the dimension.",
                 "Teorema lui Novikoff (1962): limita nu depinde nici de numărul de exemple, nici de dimensiune."),
               t("Proof idea: $\\vw^\\top\\vw^*$ grows by at least $\\gamma$ per mistake, $\\|\\vw\\|^2$ by at most $R^2$ (appendix).",
                 "Ideea demonstrației: $\\vw^\\top\\vw^*$ crește cu cel puțin $\\gamma$ la fiecare greșeală, iar $\\|\\vw\\|^2$ cu cel mult $R^2$ (anexa)."),
               t("If no separating hyperplane exists, the theorem says nothing and the weights cycle for ever.",
                 "Dacă nu există un hiperplan separator, teorema nu spune nimic, iar ponderile se rotesc la nesfîrșit.")) +
            D.bottom(t("real default data are never separable: applicants with the same file can repay or default.",
                       "datele reale de neplată nu sînt niciodată separabile: solicitanți cu același dosar pot rambursa sau nu."), NB + " \\S3"),
            cite="(Novikoff, 1962)")
    D.figframe(t("On real loans the perceptron never settles", "Pe credite reale, perceptronul nu se stabilizează niciodată"),
               t("the classes overlap, so every epoch corrects some loans and breaks others.",
                 "clasele se suprapun, așa că fiecare epocă corectează unele credite și le strică pe altele."),
               None, "perceptron_errors",
               bullets=[t("Error after each of @{perceptron.epochs} epochs: @{perceptron.err_min_pct|.1f}--@{perceptron.err_max_pct|.1f}\\,\\%, "
                          "against @{perceptron.logit_err_pct|.1f}\\,\\% for logistic regression and @{perceptron.majority_err_pct|.1f}\\,\\% for `always no default'.",
                          "Eroarea după fiecare dintre cele @{perceptron.epochs} de epoci: @{perceptron.err_min_pct|.1f}--@{perceptron.err_max_pct|.1f}\\,\\%, "
                          "față de @{perceptron.logit_err_pct|.1f}\\,\\% pentru regresia logistică și @{perceptron.majority_err_pct|.1f}\\,\\% pentru „mereu fără neplată”.")],
               job=t("a smooth loss and a probability output fix this: networks use sigmoids, not steps.",
                     "o pierdere netedă și o ieșire probabilistă rezolvă problema: rețelele folosesc sigmoide, nu trepte."),
               tryit=NB + " \\S3", cite=OWN, height="0.58\\textheight")
    D.figframe(t("No straight line separates XOR: one hidden layer does", "Nicio dreaptă nu separă XOR: un strat ascuns reușește"),
               t("XOR is true when exactly one input is on; the two positive points sit on a diagonal.",
                 "XOR este adevărat cînd exact o intrare este activă; cele două puncte pozitive stau pe o diagonală."),
               "\\[ h_k=\\tanh(\\vw_k^\\top\\vx+b_k),\\ k=1,2;\\qquad p=\\sigma(v_1h_1+v_2h_2+c) \\]",
               "xor",
               bullets=[t("Accuracy on the four points: @{xor.logit_acc_pct}\\,\\% for any linear boundary, @{xor.mlp_acc_pct}\\,\\% with two hidden units.",
                          "Acuratețea pe cele patru puncte: @{xor.logit_acc_pct}\\,\\% pentru orice frontieră liniară, @{xor.mlp_acc_pct}\\,\\% cu doi neuroni ascunși.")],
               job=t("interactions such as `high debt and short history' are XOR-like.",
                     "interacțiunile de tipul „datorii mari și istoric scurt” seamănă cu XOR."),
               cite="(Minsky \\& Papert, 1969) \\simlabel", height="0.52\\textheight")
    # ================================================================== 4. MLP
    D.section(t("The multilayer perceptron", "Perceptronul multistrat"))
    D.frame(t("An MLP alternates affine maps and nonlinear activations", "Un MLP alternează transformări afine și activări neliniare"),
            "\\intuition{" + t("each layer computes new features from the previous ones; the last layer is a logistic regression on them.",
                               "fiecare strat calculează variabile noi din cele anterioare; ultimul strat este o regresie logistică pe ele.") + "}\n" +
            "\\[ \\vh^{(0)}=\\vx,\\quad \\vh^{(l)}=a\\big(\\mW^{(l)}\\vh^{(l-1)}+\\vb^{(l)}\\big),\\ l=1,\\dots,L-1,\\quad "
            "p(\\vx)=\\sigma\\big(\\vw^{(L)\\top}\\vh^{(L-1)}+b^{(L)}\\big) \\]\n" +
            "\\where{" + t("$\\mW^{(l)}\\in\\R^{H_l\\times H_{l-1}}$, $\\vb^{(l)}$ weights and biases of layer $l$, of width $H_l$ ($H_0=d$); "
                           "$a$ the activation, element by element",
                           "$\\mW^{(l)}\\in\\R^{H_l\\times H_{l-1}}$, $\\vb^{(l)}$ ponderile și termenii liberi ai stratului $l$, de lățime $H_l$ ($H_0=d$); "
                           "$a$ funcția de activare, element cu element") + "}\n" +
            it(t("Without $a$, the layers collapse: $\\mW^{(2)}\\mW^{(1)}\\vx$ is again linear (appendix).",
                 "Fără $a$, straturile se reduc la unul singur: $\\mW^{(2)}\\mW^{(1)}\\vx$ este tot liniar (anexa)."),
               t("Our credit network: $d=$ @{hmeq.d} inputs, two hidden layers of 64 rectified linear units (ReLU), one sigmoid output (`MLP 64-64').",
                 "Rețeaua noastră de credit: $d=$ @{hmeq.d} intrări, două straturi ascunse a cîte 64 de unități liniare rectificate (ReLU), o ieșire sigmoidă („MLP 64-64”).")) +
            D.bottom(t("width and depth are design choices; Section 5 chooses them on the validation set.",
                       "lățimea și adîncimea sînt alegeri de proiectare; secțiunea 5 le alege pe setul de validare."), NB + " \\S4"),
            cite="(Bishop \\& Bishop, 2024, ch.~6)")
    D.figframe(t("Activation functions: smooth, cheap, and with a usable gradient", "Funcțiile de activare: netede, ieftine și cu un gradient utilizabil"),
               None,
               "\\[ \\sigma(z)=\\tfrac{1}{1+e^{-z}},\\ \\ \\tanh(z)=\\tfrac{e^z-e^{-z}}{e^z+e^{-z}},\\ \\ \\mathrm{ReLU}(z)=\\max(0,z),\\ \\ "
               "\\mathrm{GELU}(z)=z\\,\\Phi(z) \\]",
               "activations",
               bullets=[t("Sigmoid and tanh saturate ($\\sigma'\\le$ @{act.sigmoid_max_grad|.2f}); the rectified linear unit (ReLU) passes gradient 1 for $z>0$; "
                          "the Gaussian error linear unit (GELU, $\\Phi$ the normal distribution function) is used in transformers.",
                          "Sigmoida și tanh se saturează ($\\sigma'\\le$ @{act.sigmoid_max_grad|.2f}); unitatea liniară rectificată (ReLU) transmite gradientul 1 pentru "
                          "$z>0$; unitatea liniară cu eroare gaussiană (GELU, $\\Phi$ funcția de repartiție normală) apare în arhitecturile transformer.")],
               job=t("use ReLU in hidden layers; the output activation follows the target.",
                     "folosiți ReLU în straturile ascunse; activarea de ieșire decurge din variabila țintă."),
               cite="(Bishop \\& Bishop, 2024, ch.~6) \\simlabel", height="0.52\\textheight")
    D.frame(t("The output layer and the loss come from the type of the target", "Stratul de ieșire și pierderea decurg din tipul variabilei țintă"),
            "\\intuition{" + t("each pair (output activation, loss) is a maximum-likelihood estimator for one distribution of $y$.",
                               "fiecare pereche (activare de ieșire, pierdere) este un estimator de verosimilitate maximă pentru o distribuție a lui $y$.") + "}\n" +
            table("@{}p{0.20\\textwidth}p{0.18\\textwidth}p{0.30\\textwidth}p{0.24\\textwidth}@{}",
                  [t("Target", "Ținta"), t("Output", "Ieșirea"), t("Loss", "Pierderea"), t("Example", "Exemplu")],
                  [[t("real number", "număr real"), "$\\hat y=z$", "MSE: $\\frac1n\\sum(y_i-\\hat y_i)^2$", t("house value, a price", "valoarea locuinței, un preț")],
                   [t("binary", "binară"), "$p=\\sigma(z)$", t("cross-entropy", "entropia încrucișată") + ": $-\\frac1n\\sum[y\\log p+(1-y)\\log(1-p)]$",
                    t("default, fraud", "neplată, fraudă")],
                   [t("$K$ classes", "$K$ clase"), "$p_k=\\frac{e^{z_k}}{\\sum_j e^{z_j}}$", "$-\\frac1n\\sum_i\\log p_{i,y_i}$",
                    t("rating grade, news sentiment", "clasa de rating, sentimentul știrilor")],
                   [t("quantile $\\tau$", "cuantila $\\tau$"), "$\\hat q=z$", t("pinball", "pinball") + ": $\\frac1n\\sum\\rho_\\tau(y_i-\\hat q_i)$",
                    t("value at risk (VaR), Lecture 2", "valoarea la risc (VaR), cursul 2")]], size="scriptsize") +
            note(t("$\\rho_\\tau(u)=u(\\tau-\\mathbb 1\\{u<0\\})$; softmax with $K=2$ reduces to the sigmoid.",
                   "$\\rho_\\tau(u)=u(\\tau-\\mathbb 1\\{u<0\\})$; funcția softmax cu $K=2$ se reduce la sigmoidă.")) +
            D.bottom(t("a PD model needs a sigmoid output and cross-entropy, so that its scores are probabilities.",
                       "un model PD are nevoie de o ieșire sigmoidă și de entropia încrucișată, ca scorurile să fie probabilități.")),
            cite="(Bishop \\& Bishop, 2024, ch.~5--6)")
    D.frame(t("A small MLP already has hundreds of times more parameters than the logistic score",
              "Un MLP mic are deja de sute de ori mai mulți parametri decît scorul logistic"),
            "\\intuition{" + t("every connection is a weight, and every unit has a bias.",
                               "fiecare conexiune este o pondere, iar fiecare neuron are un termen liber.") + "}\n" +
            "\\[ |\\btheta|=\\sum_{l=1}^{L}(H_{l-1}+1)\\,H_l,\\qquad H_0=d,\\ H_L=1 \\]\n" +
            it(t("Logistic regression on HMEQ: $(d+1)\\cdot1=$ @{hmeq.n_params_logit} parameters.",
                 "Regresia logistică pe HMEQ: $(d+1)\\cdot1=$ @{hmeq.n_params_logit} de parametri."),
               t("MLP 64-64: $(d+1)\\cdot64+65\\cdot64+65\\cdot1=$ @{hmeq.n_params_6464|,d} parameters for @{hmeq.n_train|,d} training loans.",
                 "MLP 64-64: $(d+1)\\cdot64+65\\cdot64+65\\cdot1=$ @{hmeq.n_params_6464|,d} de parametri pentru @{hmeq.n_train|,d} de credite de antrenare."),
               t("More parameters than training defaults: without validation-based stopping the network would memorise them (Lecture 3).",
                 "Mai mulți parametri decît neplăți în datele de antrenare: fără oprire pe baza setului de validare, rețeaua le-ar memora (cursul 3)."),
               t("Here every MLP is trained with Adam (Lecture 2) and keeps the weights of its best validation epoch.",
                 "Aici fiecare MLP este antrenat cu Adam (cursul 2) și păstrează ponderile din epoca cu cea mai bună pierdere de validare.")) +
            D.bottom(t("count parameters before training: it tells you how much data and regularisation you will need.",
                       "numărați parametrii înainte de antrenare: aflați astfel de cîte date și de cîtă regularizare veți avea nevoie."), NB + " \\S4"))
    D.frame(t("One hidden layer can approximate any continuous function", "Un singur strat ascuns poate aproxima orice funcție continuă"),
            "\\intuition{" + t("enough sigmoid `bumps', added together, can trace any smooth curve on a bounded domain.",
                               "suficiente „cocoașe” sigmoide, adunate, pot desena orice curbă netedă pe un domeniu mărginit.") + "}\n" +
            "\\[ \\forall\\,g\\in C(K),\\ \\forall\\,\\varepsilon>0\\ \\ \\exists\\,H,\\ v_k,\\vw_k,b_k:\\quad "
            "\\sup_{\\vx\\in K}\\Big|g(\\vx)-\\sum_{k=1}^{H}v_k\\,\\sigma(\\vw_k^\\top\\vx+b_k)\\Big|<\\varepsilon \\]\n" +
            "\\where{" + t("$K\\subset\\R^d$ compact; $g$ continuous; $\\sigma$ sigmoidal (Cybenko, 1989) or non-polynomial (Hornik, 1991)",
                           "$K\\subset\\R^d$ compact; $g$ continuă; $\\sigma$ sigmoidală (Cybenko, 1989) sau nepolinomială (Hornik, 1991)") + "}\n" +
            it(t("What it says: the MLP family is rich enough.", "Ce spune: familia MLP este suficient de bogată."),
               t("What it does not say: how large $H$ must be, whether gradient descent finds the weights, or whether they generalise.",
                 "Ce nu spune: cît de mare trebuie să fie $H$, dacă coborîrea pe gradient găsește ponderile sau dacă ele generalizează.")) +
            D.bottom(t("the theorem justifies trying an MLP; only out-of-sample tests justify keeping it.",
                       "teorema justifică încercarea unui MLP; doar testele în afara eșantionului justifică păstrarea lui.")),
            cite="(Cybenko, 1989; Hornik, 1991)")
    D.figframe(t("More hidden units trace the price geography of California", "Mai mulți neuroni ascunși urmăresc geografia prețurilor din California"),
               t("values peak near Los Angeles and San Francisco; $\\hat y(x)=c+\\sum_{k=1}^{H}v_k\\sigma(w_kx+b_k)$ with $H=1,3,50$.",
                 "valorile au vîrfuri lîngă Los Angeles și San Francisco; $\\hat y(x)=c+\\sum_{k=1}^{H}v_k\\sigma(w_kx+b_k)$, cu $H=1,3,50$."),
               None, "uat_california",
               bullets=[t("Held-out MSE: linear @{uat.mse_lin|.3f}, $H=1$ @{uat.mse_w1|.3f}, $H=3$ @{uat.mse_w3|.3f}, $H=50$ @{uat.mse_w50|.3f} "
                          "(@{uat.n_capped} block groups at the 500,000 USD cap left out).",
                          "MSE pe datele păstrate deoparte: liniar @{uat.mse_lin|.3f}, $H=1$ @{uat.mse_w1|.3f}, $H=3$ @{uat.mse_w3|.3f}, $H=50$ @{uat.mse_w50|.3f} "
                          "(fără cele @{uat.n_capped} de grupuri de la plafonul de 500\\,000 USD).")],
               job=t("location effects are nonlinear: a linear term in latitude cannot price them.",
                     "efectele de localizare sînt neliniare: un termen liniar în latitudine nu le poate surprinde."),
               tryit=NB + " \\S4", cite=OWN, height="0.58\\textheight")
    D.frame(t("Depth reuses features: deep and narrow can beat shallow and wide", "Adîncimea refolosește variabilele: o rețea adîncă și îngustă poate întrece una lată"),
            "\\intuition{" + t("each layer composes the features of the previous one, so the number of linear pieces can grow with depth.",
                               "fiecare strat compune variabilele stratului anterior, astfel încît numărul de bucăți liniare poate crește cu adîncimea.") + "}\n" +
            "\\[ \\text{ReLU MLP:}\\ \\ f(\\vx)\\ \\text{" + t("is piecewise linear; number of linear regions up to", "este liniară pe porțiuni; numărul de regiuni liniare pînă la") +
            "}\\ \\ O\\big(H^{dL}\\big) \\]\n" +
            "\\where{" + t("$H$ the width of each layer; $L$ the number of hidden layers; $d$ the input dimension (bound of Montúfar et al., 2014)",
                           "$H$ lățimea fiecărui strat; $L$ numărul de straturi ascunse; $d$ dimensiunea intrării (limita lui Montúfar et al., 2014)") + "}\n" +
            it(t("For some functions a shallow net needs exponentially more units than a deep one.",
                 "Pentru unele funcții, o rețea puțin adîncă are nevoie de exponențial mai mulți neuroni decît una adîncă."),
               t("On tabular credit data, two or three hidden layers are usually enough; depth pays off on images, text and sequences.",
                 "Pe datele tabelare de credit, două sau trei straturi ascunse sînt de obicei suficiente; adîncimea contează la imagini, text și secvențe.")) +
            D.bottom(t("start with 2 hidden layers on tabular data; go deeper only if validation loss keeps falling.",
                       "începeți cu 2 straturi ascunse pe date tabelare; adăugați straturi doar dacă pierderea de validare continuă să scadă.")),
            cite="(Bishop \\& Bishop, 2024, ch.~6)")
    D.figframe(t("An MLP bends the decision boundary, but two inputs carry little extra signal",
                 "Un MLP curbează frontiera de decizie, dar două intrări aduc puțin semnal în plus"),
               t("predicted PD for two HMEQ variables: straight bands for the logistic model, curved regions for the MLP.",
                 "PD prezisă pentru două variabile HMEQ: benzi drepte pentru modelul logistic, regiuni curbe pentru MLP."),
               None, "boundary_hmeq",
               bullets=[t("Area under the receiver operating characteristic (ROC) curve (AUC) = $\\Pr(p_{\\text{default}}>p_{\\text{good}})$; with these two inputs: "
                          "logistic @{two.auc_logit|.3f}, MLP 64-64 @{two.auc_mlp|.3f}.",
                          "Aria de sub curbă (AUC) a caracteristicii de operare a receptorului (ROC) = $\\Pr(p_{\\text{neplată}}>p_{\\text{bun}})$; cu aceste două intrări: "
                          "logistic @{two.auc_logit|.3f}, MLP 64-64 @{two.auc_mlp|.3f}.")],
               job=t("a flexible boundary pays only if the inputs carry the signal.",
                     "o frontieră flexibilă merită doar dacă intrările conțin semnalul."),
               tryit=NB + " \\S4", cite=OWN, height="0.58\\textheight")
    # ================================================================== 5. case study
    D.section(t("Case study: credit scoring with real loans", "Studiu de caz: scoringul de credit pe credite reale"))
    D.frame(t("The protocol: same split, same inputs, three models", "Protocolul: aceeași împărțire, aceleași intrări, trei modele"),
            "\\intuition{" + t("a fair comparison changes only the model; data, split and metric stay fixed before anything is trained.",
                               "o comparație corectă schimbă doar modelul; datele, împărțirea și metrica sînt fixate înainte de orice antrenare.") + "}\n" +
            cols(it(t("Logistic regression: maximum likelihood, no penalty on HMEQ; a mild ridge penalty ($C=1$) on German credit.",
                      "Regresia logistică: verosimilitate maximă, fără penalizare pe HMEQ; o penalizare ridge ușoară ($C=1$) pe German credit."),
                    t("MLP: ReLU layers, Adam with learning rate $10^{-3}$, batches of 128 (64 on German credit), best validation epoch, 5 seeds.",
                      "MLP: straturi ReLU, Adam cu rata de învățare $10^{-3}$, loturi de 128 (64 pe German credit), epoca cea mai bună pe validare, 5 inițializări."),
                    t("Extreme gradient boosting (XGBoost): depth 4, learning rate 0.03, early stopping on validation (Chen \\& Guestrin, 2016).",
                      "Extreme gradient boosting (XGBoost): adîncime 4, rata de învățare 0,03, oprire timpurie pe validare (Chen \\& Guestrin, 2016).")),
                 it(t("HMEQ: one stratified 60/20/20 split; AUC, ROC curve, calibration, bootstrap interval of the AUC gain.",
                      "HMEQ: o împărțire stratificată 60/20/20; AUC, curba ROC, calibrarea, intervalul bootstrap al cîștigului de AUC."),
                    t("German credit: 5 times repeated 5-fold cross-validation (@{german.n_folds} test folds), corrected $t$-test.",
                      "German credit: validare încrucișată cu 5 pliuri, repetată de 5 ori (@{german.n_folds} de pliuri de test), test $t$ corectat."),
                    t("Seed @{seed} (+ run index); trained on the central processing unit (CPU), under a minute per model family.",
                      "Valoarea inițială a generatorului @{seed} (+ indicele rulării); antrenare pe procesorul central (CPU), sub un minut pe familie de modele.")),
                 "0.50", "0.47") +
            D.bottom(t("write the protocol down before training: it is what a model validator will ask for first.",
                       "scrieți protocolul înainte de antrenare: este primul lucru pe care îl cere cel care validează modelul."), NB + " \\S5"))
    D.figframe(t("On HMEQ, a wider hidden layer raises the test AUC above the logistic score",
                 "Pe HMEQ, un strat ascuns mai lat crește AUC pe test peste scorul logistic"),
               t("one hidden layer of width $H$; each point is the mean of 5 seeds, the band their range.",
                 "un strat ascuns de lățime $H$; fiecare punct este media a 5 inițializări, banda arată intervalul lor."),
               None, "width_auc",
               bullets=[t("AUC @{hmeq.auc_w1|.3f} at $H=1$, @{hmeq.auc_best_width|.3f} at $H=$ @{hmeq.best_width}; logistic @{hmeq.auc_logit|.3f}, "
                          "XGBoost @{hmeq.auc_xgb|.3f}; the gain flattens after about 16 units.",
                          "AUC @{hmeq.auc_w1|.3f} la $H=1$, @{hmeq.auc_best_width|.3f} la $H=$ @{hmeq.best_width}; logistic @{hmeq.auc_logit|.3f}, "
                          "XGBoost @{hmeq.auc_xgb|.3f}; cîștigul se aplatizează după circa 16 neuroni.")],
               job=t("interactions matter here: the network beats the linear score, boosting beats both.",
                     "aici interacțiunile contează: rețeaua depășește scorul liniar, iar boosting le depășește pe amîndouă."),
               tryit=NB + " \\S5", cite=OWN, height="0.58\\textheight")
    D.figframe(t("At a 10\\,\\% false-alarm rate, the MLP catches more defaults than the logistic score",
                 "La 10\\,\\% alarme false, MLP detectează mai multe neplăți decît scorul logistic"),
               t("the ROC curve plots $\\Pr(p\\ge c\\mid\\text{default})$ against $\\Pr(p\\ge c\\mid\\text{good})$ for every cut-off $c$.",
                 "curba ROC reprezintă $\\Pr(p\\ge c\\mid\\text{neplată})$ în funcție de $\\Pr(p\\ge c\\mid\\text{bun})$, pentru fiecare prag $c$."),
               None, "roc_hmeq",
               bullets=[t("Defaults caught when 10\\,\\% of good loans are rejected: logistic @{hmeq.tpr10_logit|.1f}\\,\\%, MLP @{hmeq.tpr10_mlp|.1f}\\,\\%, "
                          "XGBoost @{hmeq.tpr10_xgb|.1f}\\,\\% (@{hmeq.test_bad} test defaults).",
                          "Neplăți detectate cînd respingem 10\\,\\% dintre creditele bune: logistic @{hmeq.tpr10_logit|.1f}\\,\\%, MLP @{hmeq.tpr10_mlp|.1f}\\,\\%, "
                          "XGBoost @{hmeq.tpr10_xgb|.1f}\\,\\% (@{hmeq.test_bad} de neplăți în test).")],
               job=t("read the curve at the rejection rate the business accepts, not only its area.",
                     "citiți curba la rata de respingere acceptată de afacere, nu doar aria de sub ea."),
               tryit=NB + " \\S5", cite=OWN, height="0.58\\textheight")
    D.frame(t("The MLP gain on HMEQ is larger than seed noise and test-set noise", "Cîștigul MLP pe HMEQ depășește zgomotul inițializării și al setului de testare"),
            "\\intuition{" + t("one run proves nothing: we repeat the training and resample the test set.",
                               "o singură rulare nu dovedește nimic: repetăm antrenarea și reeșantionăm setul de testare.") + "}\n" +
            "\\[ \\Delta=\\mathrm{AUC}_{\\mathrm{MLP}}-\\mathrm{AUC}_{\\text{logit}},\\qquad "
            "\\Delta^{*}_b=\\mathrm{AUC}_{\\mathrm{MLP}}(\\mathcal T^*_b)-\\mathrm{AUC}_{\\text{logit}}(\\mathcal T^*_b),\\ b=1,\\dots,2000 \\]\n" +
            "\\where{" + t("$\\mathcal T^*_b$ a bootstrap resample of the @{hmeq.n_test|,d} test loans; the same resample for both models (paired)",
                           "$\\mathcal T^*_b$ o reeșantionare bootstrap a celor @{hmeq.n_test|,d} de credite de testare; aceeași pentru ambele modele (perechi)") + "}\n" +
            it(t("Seeds: MLP 64-64 test AUC @{hmeq.auc_mlp_mean|.3f} on average over 5 seeds (standard deviation @{hmeq.auc_mlp_sd|.3f}; "
                 "range @{hmeq.auc_mlp_min|.3f}--@{hmeq.auc_mlp_max|.3f}).",
                 "Inițializări: MLP 64-64 are în medie AUC pe test @{hmeq.auc_mlp_mean|.3f} pe 5 inițializări (abaterea standard @{hmeq.auc_mlp_sd|.3f}; "
                 "interval @{hmeq.auc_mlp_min|.3f}--@{hmeq.auc_mlp_max|.3f})."),
               t("Test noise: $\\Delta=$ @{hmeq.dauc_mlp_lr|.3f}, 95\\,\\% bootstrap interval [@{hmeq.dauc_lo|.3f}; @{hmeq.dauc_hi|.3f}]: zero is excluded.",
                 "Zgomotul setului de testare: $\\Delta=$ @{hmeq.dauc_mlp_lr|.3f}, intervalul bootstrap de 95\\,\\% [@{hmeq.dauc_lo|.3f}; @{hmeq.dauc_hi|.3f}]: zero este exclus."),
               t("The seed spread of the logistic model is zero: its loss is convex.", "Variația între inițializări a modelului logistic este zero: pierderea lui este convexă.")) +
            D.bottom(t("report a model gain with an interval; a gain inside the seed spread is not a gain.",
                       "raportați cîștigul unui model cu un interval; un cîștig mai mic decît variația dintre inițializări nu este un cîștig."), NB + " \\S5"),
            cite=OWN)
    D.figframe(t("The scores are close to calibrated probabilities", "Scorurile sînt apropiate de probabilități calibrate"),
               t("in each decile of predicted PD, the observed default rate should equal the predicted one.",
                 "în fiecare decilă a PD prezise, rata observată de neplată ar trebui să fie egală cu cea prezisă."),
               None, "calibration",
               bullets=[t("Brier score $\\frac1n\\sum(p_i-y_i)^2$: logistic @{calib.brier_logit|.4f}, MLP @{calib.brier_mlp|.4f}, XGBoost @{calib.brier_xgb|.4f}; "
                          "constant rate @{calib.brier_base|.4f}.",
                          "Scorul Brier $\\frac1n\\sum(p_i-y_i)^2$: logistic @{calib.brier_logit|.4f}, MLP @{calib.brier_mlp|.4f}, XGBoost @{calib.brier_xgb|.4f}; "
                          "rata constantă @{calib.brier_base|.4f}.")],
               job=t("provisions and prices use the PD itself: check calibration, not only ranking.",
                     "provizioanele și prețurile folosesc PD ca atare: verificați calibrarea, nu doar ordonarea."),
               tryit=NB + " \\S5", cite=OWN, height="0.58\\textheight")
    D.figframe(t("On 1,000 German applicants, the MLP does not beat logistic regression", "Pe 1\\,000 de solicitanți germani, MLP nu depășește regresia logistică"),
               t("each dot is the AUC of one held-out fold; the black bar is the mean of the 25 folds.",
                 "fiecare punct este AUC pe un pliu de test; bara neagră este media celor 25 de pliuri."),
               None, "german_cv",
               bullets=[t("Mean AUC: logistic @{german.auc_logit|.3f}, MLP @{german.auc_mlp|.3f}, XGBoost @{german.auc_xgb|.3f}; "
                          "MLP minus logistic: $t=$ @{german.t_mlp|.2f}, $p=$ @{german.p_mlp|.3f} (next frame).",
                          "AUC medie: logistic @{german.auc_logit|.3f}, MLP @{german.auc_mlp|.3f}, XGBoost @{german.auc_xgb|.3f}; "
                          "MLP minus logistic: $t=$ @{german.t_mlp|.2f}, $p=$ @{german.p_mlp|.3f} (slide-ul următor).")],
               job=t("with 1,000 rows, keep the logistic scorecard: the network adds risk, not accuracy.",
                     "cu 1\\,000 de rînduri, păstrați scorecard-ul logistic: rețeaua adaugă risc, nu acuratețe."),
               tryit=NB + " \\S6", cite=OWN, height="0.58\\textheight")
    D.frame(t("Why the usual $t$-test overstates differences in cross-validation", "De ce testul $t$ obișnuit exagerează diferențele în validarea încrucișată"),
            "\\intuition{" + t("the training sets of different folds overlap, so the fold scores are positively correlated.",
                               "seturile de antrenare ale pliurilor diferite se suprapun, deci scorurile pliurilor sînt corelate pozitiv.") + "}\n" +
            "\\[ t=\\frac{\\bar d}{\\sqrt{\\big(\\frac{1}{J}+\\frac{n_{\\text{test}}}{n_{\\text{train}}}\\big)\\,s_d^2}}\\ \\sim\\ t_{J-1},\\qquad "
            "d_j=\\mathrm{AUC}^{\\mathrm{MLP}}_j-\\mathrm{AUC}^{\\text{logit}}_j \\]\n" +
            "\\where{" + t("$J=$ @{german.n_folds} folds; $s_d^2$ the sample variance of $d_j$; $n_{\\text{test}}/n_{\\text{train}}=1/4$ for 5 folds",
                           "$J=$ @{german.n_folds} de pliuri; $s_d^2$ varianța de selecție a lui $d_j$; $n_{\\text{test}}/n_{\\text{train}}=1/4$ pentru 5 pliuri") + "}\n" +
            it(t("The term $n_{\\text{test}}/n_{\\text{train}}$ is the Nadeau--Bengio correction: without it, $t$ is far too large.",
                 "Termenul $n_{\\text{test}}/n_{\\text{train}}$ este corecția Nadeau--Bengio: fără ea, $t$ este mult prea mare."),
               t("MLP vs logistic: $t=$ @{german.t_mlp|.2f} ($p=$ @{german.p_mlp|.3f}); XGBoost vs logistic: $t=$ @{german.t_xgb|.2f} ($p=$ @{german.p_xgb|.3f}).",
                 "MLP față de logistic: $t=$ @{german.t_mlp|.2f} ($p=$ @{german.p_mlp|.3f}); XGBoost față de logistic: $t=$ @{german.t_xgb|.2f} ($p=$ @{german.p_xgb|.3f}).")) +
            D.bottom(t("a model change in a bank needs evidence that survives this correction.",
                       "schimbarea unui model într-o bancă are nevoie de dovezi care rezistă acestei corecții."), NB + " \\S6"),
            cite="(Nadeau \\& Bengio, 2003)")
    D.figframe(t("A missed default costs five times a rejected good applicant: cut at 1/6, not 1/2",
                 "O neplată ratată costă de cinci ori cît un client bun respins: pragul este 1/6, nu 1/2"),
               None,
               "\\[ \\text{" + t("reject if", "respingem dacă") + "}\\ \\ c_{\\mathrm{FN}}\\,p\\ \\ge\\ c_{\\mathrm{FP}}(1-p)\\iff p\\ \\ge\\ "
               "\\frac{c_{\\mathrm{FP}}}{c_{\\mathrm{FP}}+c_{\\mathrm{FN}}}=\\frac16 \\]",
               "cost_threshold",
               bullets=[t("Cost per applicant (UCI cost matrix 5:1): at 1/6 logistic @{german.cost_logit_bayes|.3f}, MLP @{german.cost_mlp_bayes|.3f}; "
                          "at 1/2 @{german.cost_logit_05|.3f} and @{german.cost_mlp_05|.3f}; accept all @{german.cost_all_accept|.2f}.",
                          "Costul pe solicitant (matricea de cost UCI 5:1): la 1/6 logistic @{german.cost_logit_bayes|.3f}, MLP @{german.cost_mlp_bayes|.3f}; "
                          "la 1/2 @{german.cost_logit_05|.3f} și @{german.cost_mlp_05|.3f}; acceptarea tuturor @{german.cost_all_accept|.2f}.")],
               job=t("here the choice of cut-off matters more than the choice of model.",
                     "aici alegerea pragului contează mai mult decît alegerea modelului."),
               tryit=NB + " \\S6", cite=OWN, height="0.56\\textheight")
    D.frame(t("A minimal MLP in PyTorch: about fifteen lines", "Un MLP minimal în PyTorch: circa cincisprezece rînduri"),
            cols(code("import torch, torch.nn as nn",
                      "model = nn.Sequential(",
                      "    nn.Linear(d, 64), nn.ReLU(),",
                      "    nn.Linear(64, 64), nn.ReLU(),",
                      "    nn.Linear(64, 1))            # logit z",
                      "loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + cross-entropy",
                      "opt = torch.optim.Adam(model.parameters(), lr=1e-3)",
                      "for epoch in range(300):",
                      "    for xb, yb in train_batches:",
                      "        opt.zero_grad()",
                      "        loss = loss_fn(model(xb), yb)",
                      "        loss.backward()           # backpropagation",
                      "        opt.step()                # Adam update",
                      "    # evaluate on validation, keep the best weights",
                      "p = torch.sigmoid(model(x_test))   # predicted PD"),
                 it(t("\\texttt{BCEWithLogitsLoss} combines the sigmoid and the cross-entropy in one numerically stable function.",
                      "\\texttt{BCEWithLogitsLoss} combină sigmoida și entropia încrucișată într-o singură funcție, stabilă numeric."),
                    t("\\texttt{backward()} computes all gradients by backpropagation; \\texttt{step()} applies the optimiser: both are Lecture 2.",
                      "\\texttt{backward()} calculează toate gradienții prin propagare înapoi; \\texttt{step()} aplică optimizatorul: ambele sînt tema cursului 2."),
                    t("The notebook adds the data pipeline, the validation stopping and the seeds.",
                      "Notebook-ul adaugă pregătirea datelor, oprirea pe baza validării și valorile inițiale ale generatorului.")), "0.55", "0.42") +
            D.bottom(t("the same fifteen lines train every tabular network of this course; only data and layers change.",
                       "aceleași cincisprezece rînduri antrenează toate rețelele tabelare din curs; se schimbă doar datele și straturile."), NB + " \\S7"),
            cite="(Paszke et al., 2019)")
    D.frame(t("When to use an MLP for tabular business data", "Cînd folosim un MLP pentru date tabelare de afaceri"),
            cols(block(t("Evidence from this lecture", "Ce arată acest curs"), it(
                t("HMEQ, @{hmeq.n_train|,d} loans: MLP AUC @{hmeq.auc_mlp_mean|.3f} vs logistic @{hmeq.auc_logit|.3f}; XGBoost @{hmeq.auc_xgb|.3f}.",
                  "HMEQ, @{hmeq.n_train|,d} de credite: AUC MLP @{hmeq.auc_mlp_mean|.3f} față de logistic @{hmeq.auc_logit|.3f}; XGBoost @{hmeq.auc_xgb|.3f}."),
                t("German credit, 1,000 applicants: no significant difference ($p=$ @{german.p_mlp|.3f}).",
                  "German credit, 1\\,000 de solicitanți: nicio diferență semnificativă ($p=$ @{german.p_mlp|.3f})."),
                t("Both: the cut-off from the cost matrix matters more than the model.",
                  "În ambele cazuri: pragul din matricea de cost contează mai mult decît modelul."))),
                 block(t("Rule of thumb", "Regulă practică"), it(
                     t("Few rows, engineered features, explanation required: logistic scorecard.",
                       "Puține rînduri, variabile construite, explicație obligatorie: scorecard logistic."),
                     t("Thousands of rows, interactions: test boosting and an MLP against it.",
                       "Mii de rînduri, interacțiuni: testați boosting și un MLP față de el."),
                     t("Raw text, images, sequences, graphs, or a network shared across tasks: deep learning.",
                       "Text brut, imagini, secvențe, grafuri sau o rețea folosită pentru mai multe sarcini: deep learning.")), "exampleblock"),
                 "0.49", "0.49") +
            D.bottom(t("choose the simplest model whose out-of-sample gain survives seeds, resampling and the cost matrix.",
                       "alegeți cel mai simplu model al cărui cîștig în afara eșantionului rezistă la inițializări, reeșantionare și matricea de cost.")))

    # ================================================================== wrap-up
    D.section(t("Wrap-up", "Sinteză"))
    D.frame(t("Key takeaways", "Idei principale"), it(
        t("Linear and logistic regression are one-neuron networks; their losses are likelihoods, their gradients are error times input.",
          "Regresia liniară și regresia logistică sînt rețele cu un neuron; pierderile lor sînt verosimilități, gradienții lor sînt eroarea înmulțită cu intrarea."),
        t("The perceptron converges only on separable data; real credit data are not separable, and XOR shows the limit of one layer.",
          "Perceptronul converge doar pe date separabile; datele reale de credit nu sînt separabile, iar XOR arată limita unui singur strat."),
        t("An MLP stacks affine maps and activations; one hidden layer approximates any continuous function, depth makes it efficient.",
          "Un MLP suprapune transformări afine și activări; un strat ascuns aproximează orice funcție continuă, adîncimea îl face eficient."),
        t("On real loans: the MLP beats logistic regression on HMEQ (AUC @{hmeq.auc_mlp_mean|.3f} vs @{hmeq.auc_logit|.3f}), not on German credit; boosting is the bar to clear.",
          "Pe credite reale: MLP depășește regresia logistică pe HMEQ (AUC @{hmeq.auc_mlp_mean|.3f} față de @{hmeq.auc_logit|.3f}), dar nu pe German credit; boosting este ștacheta."),
        t("Evaluate with seeds, intervals, calibration and the cost of errors, never with one accuracy number.",
          "Evaluați cu mai multe inițializări, intervale, calibrare și costul erorilor, niciodată cu o singură cifră de acuratețe."), sep="4pt"))
    D.frame(t("Key equations", "Ecuațiile principale"), cols(
        "\\[ p=\\sigma(\\vw^\\top\\vx+b) \\]\n\\[ \\loss=-\\tfrac1n\\textstyle\\sum[y\\log p+(1-y)\\log(1-p)] \\]\n"
        "\\[ \\nabla_{\\vw}\\loss=\\tfrac1n\\textstyle\\sum(p_i-y_i)\\vx_i \\]\n"
        "\\[ \\vw\\leftarrow\\vw+\\eta\\,y_i\\vx_i\\ \\ \\text{" + t("if misclassified", "dacă e greșit") + "} \\]",
        "\\[ \\vh^{(l)}=a\\big(\\mW^{(l)}\\vh^{(l-1)}+\\vb^{(l)}\\big) \\]\n\\[ |\\btheta|=\\textstyle\\sum_l(H_{l-1}+1)H_l \\]\n"
        "\\[ \\#\\text{" + t("mistakes", "greșeli") + "}\\le(R/\\gamma)^2 \\]\n"
        "\\[ \\text{" + t("reject if", "respingem dacă") + "}\\ p\\ge\\frac{c_{\\mathrm{FP}}}{c_{\\mathrm{FP}}+c_{\\mathrm{FN}}} \\]"))
    D.frame(t("Research from our group: deep learning for business decisions", "Cercetările grupului nostru: deep learning pentru decizii de afaceri"), it(
        t("\\textbf{Explainable fraud detection} (Diaconu \\& Pele, 2026): compared XGBoost and CatBoost with a graph neural network (GNN), "
          "graph sample and aggregate (GraphSAGE), on the Institute of Electrical and Electronics Engineers, Computational Intelligence Society "
          "(IEEE-CIS) fraud data; boosting had the higher AUC, the graph model captured relational fraud patterns; a multi-level explainable "
          "artificial intelligence (XAI) framework joins Shapley additive explanations (SHAP) with node and edge ablation --- Lectures 9, 11.",
          "\\textbf{Detectarea explicabilă a fraudei} (Diaconu \\& Pele, 2026): a comparat XGBoost și CatBoost cu o rețea neuronală pe grafuri (GNN), "
          "eșantionare și agregare pe graf (GraphSAGE), pe datele de fraudă Institute of Electrical and Electronics Engineers, Computational "
          "Intelligence Society (IEEE-CIS); boosting a avut AUC mai mare, modelul pe graf a surprins tiparele relaționale ale fraudei; un cadru "
          "multinivel de inteligență artificială explicabilă (XAI) combină explicațiile aditive Shapley (SHAP) cu eliminarea de noduri și muchii --- cursurile 9, 11."),
        t("\\textbf{Early warning of crypto `zombies'} (Będowska-Sójka, Wójcik \\& Pele, 2026): machine-learning early warning of crypto-assets "
          "that become `zombie' assets --- Lecture 3.",
          "\\textbf{Criptoactivele „zombie”} (Będowska-Sójka, Wójcik \\& Pele, 2026): avertizare timpurie, prin învățare automată, "
          "pentru criptoactivele care devin „zombie” --- cursul 3."),
        t("\\textbf{Risk measures from language models} (Pele et al., 2026): value at risk (VaR) and expected shortfall (ES) obtained by prompting "
          "large language models (LLM) --- Lecture 12.",
          "\\textbf{Măsuri de risc obținute din modele lingvistice} (Pele et al., 2026): valoarea la risc (VaR) și pierderea așteptată în coada "
          "distribuției (ES), obținute prin interogarea modelelor lingvistice mari (LLM) --- cursul 12."),
        t("\\textbf{Foundation models for tail risk} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026): zero-shot VaR and ES with conformal "
          "calibration in Central and Eastern European markets --- Lecture 13.",
          "\\textbf{Modele fundație pentru riscul extrem} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026): VaR și ES fără antrenare "
          "(zero-shot), cu calibrare conformală, pe piețele din Europa Centrală și de Est --- cursul 13."),
        t("\\textbf{Seeing market anomalies} (Jheng, Pele, Tak \\& Găman, 2026; best paper award at the International Conference on Business Excellence (ICBE) 2026): market indicators drawn as images "
          "and classified by a vision-language model (VLM), with natural-language rationales, tested out of sample on the S\\&P 500 --- Lecture 14.",
          "\\textbf{Anomaliile pieței văzute ca imagini} (Jheng, Pele, Tak \\& Găman, 2026; premiul pentru cea mai bună lucrare la International Conference on Business Excellence (ICBE) 2026): "
          "indicatorii pieței, desenați ca imagini, sînt clasificați de un model vizual-lingvistic (VLM), cu justificări în limbaj natural, "
          "pe S\\&P 500 --- cursul 14."), sep="2pt") +
        note(t("Findings as stated in the papers; numbers on the other slides come from our own computations.",
               "Rezultatele sînt cele declarate în lucrări; cifrele de pe celelalte slide-uri provin din calculele noastre.")), size="footnotesize")
    D.frame(t("Want to learn more?", "Vreți să aflați mai mult?"), cols(it(
        t("\\textbf{Bishop \\& Bishop (2024)}, chapters 4--6: linear models, classification, deep networks: ",
          "\\textbf{Bishop \\& Bishop (2024)}, capitolele 4--6: modele liniare, clasificare, rețele adînci: ") +
        "\\href{https://www.bishopbook.com}{bishopbook.com}.",
        t("\\textbf{Dive into Deep Learning}, chapters 3--5, with PyTorch code: ", "\\textbf{Dive into Deep Learning}, capitolele 3--5, cu cod PyTorch: ") +
        "\\href{https://d2l.ai}{d2l.ai}.",
        t("\\textbf{Goodfellow, Bengio \\& Courville}, chapter 6: ", "\\textbf{Goodfellow, Bengio \\& Courville}, capitolul 6: ") +
        "\\href{https://www.deeplearningbook.org}{deeplearningbook.org}.", sep="4pt"),
        it(t("\\textbf{PyTorch tutorials}: ", "\\textbf{Tutorialele PyTorch}: ") + "\\href{https://pytorch.org/tutorials}{pytorch.org/tutorials}.",
           t("\\textbf{This course}: code, data, notebooks, Quantlets --- ", "\\textbf{Acest curs}: cod, date, notebook-uri, Quantlets --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Credit scoring in depth}: Baesens, Rösch \\& Scheule (2016).", "\\textbf{Scoringul de credit în detaliu}: Baesens, Rösch \\& Scheule (2016)."),
           sep="4pt"), "0.52", "0.45"))

    # ------------------------------------------------------------------ quiz
    D.quiz_frame(t("logistic regression as a neuron", "regresia logistică ca neuron"),
                 t("Which loss makes a one-neuron network with a sigmoid output equal to logistic regression?",
                   "Ce pierdere face ca o rețea cu un neuron și ieșire sigmoidă să fie echivalentă cu regresia logistică?"),
                 t("The cross-entropy, i.e.\\ the Bernoulli negative log-likelihood; its gradient is $\\frac1n\\sum(p_i-y_i)\\vx_i$.",
                   "Entropia încrucișată, adică minus log-verosimilitatea Bernoulli; gradientul ei este $\\frac1n\\sum(p_i-y_i)\\vx_i$."))
    D.quiz_frame(t("the perceptron on real data", "perceptronul pe date reale"),
                 t("Why does the training error of the perceptron on HMEQ jump between @{perceptron.err_min_pct|.1f}\\,\\% and @{perceptron.err_max_pct|.1f}\\,\\%?",
                   "De ce eroarea de antrenare a perceptronului pe HMEQ sare între @{perceptron.err_min_pct|.1f}\\,\\% și @{perceptron.err_max_pct|.1f}\\,\\%?"),
                 t("The classes are not linearly separable, so Novikoff's bound does not apply: each correction of one loan breaks others.",
                   "Clasele nu sînt separabile liniar, deci limita lui Novikoff nu se aplică: fiecare corecție a unui credit le strică pe altele."))
    D.quiz_frame(t("counting parameters", "numărarea parametrilor"),
                 t("How many parameters has an MLP on the HMEQ inputs ($d=$ @{hmeq.d}) with one hidden layer of 32 units and one output?",
                   "Cîți parametri are un MLP pe intrările HMEQ ($d=$ @{hmeq.d}) cu un strat ascuns de 32 de neuroni și o ieșire?"),
                 t("$(d+1)\\cdot32+(32+1)\\cdot1=$ @{hmeq.n_params_32|,d}.", "$(d+1)\\cdot32+(32+1)\\cdot1=$ @{hmeq.n_params_32|,d}."))
    D.quiz_frame(t("the cost-optimal cut-off", "pragul optim pentru cost"),
                 t("A missed default costs 5, a rejected good applicant costs 1. Above which PD should the bank reject?",
                   "O neplată ratată costă 5, un client bun respins costă 1. Peste ce PD ar trebui banca să respingă?"),
                 t("$p\\ge c_{\\mathrm{FP}}/(c_{\\mathrm{FP}}+c_{\\mathrm{FN}})=1/6$; on German credit this cut-off lowers the logistic cost per applicant "
                   "from @{german.cost_logit_05|.3f} (cut-off 1/2) to @{german.cost_logit_bayes|.3f}.",
                   "$p\\ge c_{\\mathrm{FP}}/(c_{\\mathrm{FP}}+c_{\\mathrm{FN}})=1/6$; pe German credit, acest prag reduce costul pe solicitant al modelului logistic "
                   "de la @{german.cost_logit_05|.3f} (pragul 1/2) la @{german.cost_logit_bayes|.3f}."))
    D.quiz_frame(t("when is an MLP worth it?", "cînd merită un MLP?"),
                 t("On German credit the MLP has mean AUC @{german.auc_mlp|.3f} and logistic regression @{german.auc_logit|.3f}. What do you recommend?",
                   "Pe German credit, MLP are AUC medie @{german.auc_mlp|.3f}, iar regresia logistică @{german.auc_logit|.3f}. Ce recomandați?"),
                 t("Keep the logistic scorecard: the difference is not significant ($p=$ @{german.p_mlp|.3f}) and the simpler model is easier to explain and validate.",
                   "Păstrați scorecard-ul logistic: diferența nu este semnificativă ($p=$ @{german.p_mlp|.3f}), iar modelul mai simplu este mai ușor de explicat și de validat."))
    reference_frames(D, REFS, per_frame=12)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.frame(t("Appendix: the gradient of the logistic loss", "Anexă: gradientul pierderii logistice"),
            "\\[ \\sigma'(z)=\\sigma(z)\\big(1-\\sigma(z)\\big),\\qquad \\ell_i=-y_i\\log p_i-(1-y_i)\\log(1-p_i),\\quad p_i=\\sigma(z_i) \\]\n"
            "\\[ \\frac{\\partial\\ell_i}{\\partial z_i}=-\\frac{y_i}{p_i}p_i(1-p_i)+\\frac{1-y_i}{1-p_i}p_i(1-p_i)=p_i-y_i \\]\n"
            "\\[ \\frac{\\partial\\ell_i}{\\partial\\vw}=\\frac{\\partial\\ell_i}{\\partial z_i}\\,\\frac{\\partial z_i}{\\partial\\vw}=(p_i-y_i)\\,\\vx_i,\\qquad "
            "\\frac{\\partial^2\\loss}{\\partial\\vw\\,\\partial\\vw^\\top}=\\frac1n\\sum_i p_i(1-p_i)\\vx_i\\vx_i^\\top\\succeq0 \\]\n" +
            it(t("The Hessian is positive semi-definite: the loss is convex, so gradient descent reaches the global minimum.",
                 "Hessiana este pozitiv semidefinită: pierderea este convexă, deci coborîrea pe gradient atinge minimul global."),
               t("The same chain rule, applied layer by layer, is backpropagation (Lecture 2).",
                 "Aceeași regulă de derivare a funcțiilor compuse, aplicată strat cu strat, este propagarea înapoi a erorii (cursul 2).")))
    D.frame(t("Appendix: proof of the perceptron convergence theorem", "Anexă: demonstrația teoremei de convergență a perceptronului"),
            it(t("Start at $\\vw_0=\\mathbf 0$ and let $\\vw_k$ be the weights after the $k$-th mistake, on example $(\\vx,y)$: $\\vw_k=\\vw_{k-1}+y\\vx$ ($\\eta=1$).",
                 "Pornim de la $\\vw_0=\\mathbf 0$ și notăm cu $\\vw_k$ ponderile după a $k$-a greșeală, pe exemplul $(\\vx,y)$: $\\vw_k=\\vw_{k-1}+y\\vx$ ($\\eta=1$)."),
               t("Progress: $\\vw_k^\\top\\vw^*=\\vw_{k-1}^\\top\\vw^*+y\\vx^\\top\\vw^*\\ge\\vw_{k-1}^\\top\\vw^*+\\gamma$, so $\\vw_k^\\top\\vw^*\\ge k\\gamma$.",
                 "Progresul: $\\vw_k^\\top\\vw^*=\\vw_{k-1}^\\top\\vw^*+y\\vx^\\top\\vw^*\\ge\\vw_{k-1}^\\top\\vw^*+\\gamma$, deci $\\vw_k^\\top\\vw^*\\ge k\\gamma$."),
               t("Growth: $\\|\\vw_k\\|^2=\\|\\vw_{k-1}\\|^2+2y\\vw_{k-1}^\\top\\vx+\\|\\vx\\|^2\\le\\|\\vw_{k-1}\\|^2+R^2$, since the example was misclassified; so $\\|\\vw_k\\|^2\\le kR^2$.",
                 "Creșterea: $\\|\\vw_k\\|^2=\\|\\vw_{k-1}\\|^2+2y\\vw_{k-1}^\\top\\vx+\\|\\vx\\|^2\\le\\|\\vw_{k-1}\\|^2+R^2$, pentru că exemplul era greșit; deci $\\|\\vw_k\\|^2\\le kR^2$."),
               t("Cauchy--Schwarz: $k\\gamma\\le\\vw_k^\\top\\vw^*\\le\\|\\vw_k\\|\\le\\sqrt k\\,R$, hence $k\\le(R/\\gamma)^2$.",
                 "Cauchy--Schwarz: $k\\gamma\\le\\vw_k^\\top\\vw^*\\le\\|\\vw_k\\|\\le\\sqrt k\\,R$, de unde $k\\le(R/\\gamma)^2$."), sep="5pt"),
            cite="(Novikoff, 1962)")
    D.frame(t("Appendix: stacked linear layers collapse to one", "Anexă: straturile liniare suprapuse se reduc la unul singur"),
            "\\[ \\mW^{(2)}\\big(\\mW^{(1)}\\vx+\\vb^{(1)}\\big)+\\vb^{(2)}=\\underbrace{\\mW^{(2)}\\mW^{(1)}}_{\\mW}\\vx+\\underbrace{\\mW^{(2)}\\vb^{(1)}+\\vb^{(2)}}_{\\vb} \\]\n" +
            it(t("Any depth of linear layers is a single linear map: depth without activation adds parameters, not expressiveness.",
                 "Orice număr de straturi liniare formează o singură transformare liniară: adîncimea fără activare adaugă parametri, nu expresivitate."),
               t("With a ReLU between them, $f$ becomes piecewise linear with a number of pieces that grows with depth.",
                 "Cu o funcție ReLU între ele, $f$ devine liniară pe porțiuni, cu un număr de porțiuni care crește cu adîncimea.")))
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 1, ABBR, IGNORE, UNITS, langs=langs) else 1)
