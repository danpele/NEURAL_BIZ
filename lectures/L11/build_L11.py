r"""
build_L11.py -- Lecture 11 "Explainable AI (XAI): SHAP, LIME, gradient methods and the stability of explanations"
(EN + RO) from one source. Numbers come from L11_numbers.json (analysis_L11.py) as @{key} placeholders.
Hand-typed numbers are definitions: 0.5 decision threshold, 100 background applicants, 0.75 sqrt(d) kernel width,
95 % levels, the number of seeds and samples chosen a priori.
Run:  python3 lectures/L11/build_L11.py      (exit 0 only if every check of lectures/STYLE.md passes)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import Deck, Numbers, abbreviation_frames, cols, hier, items, reference_frames, run_build, table  # noqa: E402

ABBR = {
    "AI": ("artificial intelligence", "inteligență artificială"),
    "AOPC": ("area over the perturbation curve", "aria de deasupra curbei de perturbare"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "AUC": ("area under the ROC curve", "aria de sub curba ROC"),
    "EBA": ("European Banking Authority", "Autoritatea Bancară Europeană"),
    "EU": ("European Union", "Uniunea Europeană (abrevierea engleză, în citări)"),
    "UE": (None, "Uniunea Europeană"),
    "GDPR": ("General Data Protection Regulation", "Regulamentul general privind protecția datelor"),
    "GNN": ("graph neural network", "rețea neuronală pe grafuri"),
    "HMEQ": ("home-equity loan data set", "setul de date privind creditele garantate cu locuința"),
    "ICE": ("individual conditional expectation", "așteptarea condiționată individuală"),
    "IG": ("integrated gradients", "gradienți integrați"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "KL": ("Kullback-Leibler divergence", "divergența Kullback-Leibler"),
    "ULB": ("Université Libre de Bruxelles", "Université Libre de Bruxelles"),
    "JS": ("Jensen-Shannon divergence", "divergența Jensen-Shannon"),
    "LIME": ("local interpretable model-agnostic explanations", "explicații locale interpretabile, independente de model"),
    "MAD": ("median absolute deviation", "abaterea absolută mediană"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "PCA": ("principal component analysis", "analiza componentelor principale"),
    "PD": ("probability of default", "probabilitatea de nerambursare"),
    "PDP": ("partial dependence plot", "graficul dependenței parțiale"),
    "PICBE": ("Proceedings of the International Conference on Business Excellence", "Proceedings of the International Conference on Business Excellence"),
    "PR-AUC": ("area under the precision-recall curve", "aria de sub curba precizie-sensibilitate"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată"),
    "ROC": ("receiver operating characteristic", "caracteristica de operare a receptorului"),
    "SHAP": ("Shapley additive explanations", "explicații aditive Shapley"),
    "SR": ("Supervisory Guidance on Model Risk Management (Federal Reserve SR 11-7)",
           "ghidul de supraveghere privind riscul de model (Federal Reserve SR 11-7)"),
    "TCAV": ("testing with concept activation vectors", "testarea cu vectori de activare a conceptelor"),
    "XAI": ("explainable artificial intelligence", "inteligență artificială explicabilă"),
    "XGBoost": ("extreme gradient boosting", "extreme gradient boosting (arbori amplificați prin gradient)"),
    "USD": ("US dollar", "dolar american"), "EUR": ("euro", "euro"),
}
UNITS = {"USD", "EUR"}
IGNORE = {"ESWA", "IEEE", "JMLR", "NIPS", "ICML", "KDD", "SSCI", "OK", "II", "III", "MIT", "XG", "TreeSHAP", "CLAGE", "NINQ", "CLNO",
          "LOAN", "MORTDUE", "VALUE", "YOJ", "REASON", "JOB", "MedInc", "HomeImp", "PNAS", "NBER", "AAAI", "ACM", "SIGKDD", "FAccT",
          "DebtCon", "ProfExe", "AveRooms", "CIS", "DEBTINC", "DELINQ", "DEROG", "BIZ", "NEURAL", "ICLR", "IEEE-CIS", "SAS", "SG", "TNNLS", "US", "OpenML"}

REFS = [
    "Alvarez-Melis, D.\\ \\& Jaakkola, T.\\,S.\\ (2018). On the robustness of interpretability methods. ICML Workshop on Human "
    "Interpretability in Machine Learning. \\href{https://arxiv.org/abs/1806.08049}{arXiv:1806.08049}",
    "Andrei, A.-V.\\ \\& Pele, D.\\,T.\\ (2025). Explainable AI for electricity price anomaly detection: A SHAP-driven approach in "
    "Romania's energy market. \\textit{Preprints}. \\href{https://doi.org/10.20944/preprints202508.0410.v1}{doi:10.20944/preprints202508.0410.v1}",
    "Andrei, A.-V., Pele, D.-T.\\ \\& Cramer, A.-A.\\ (2026). Multi-horizon explainability in energy price forecasting: Disentangling "
    "market dynamics and renewable intermittency in Romania's liberalized electricity market. \\textit{Proceedings of the "
    "International Conference on Business Excellence}, 20(1), 104--117. \\href{https://doi.org/10.2478/picbe-2026-0014}{doi:10.2478/picbe-2026-0014}",
    "Andrei, A.-V., Pele, D.-T., Cramer, A.-A., Amza, A.\\ \\& Diaconu, D.\\ (2026). Explainable AI-based early warning system for "
    "electricity price spikes: Evidence from the Romanian market. \\textit{Proceedings of the International Conference on Business "
    "Excellence}, 20(1), 4820--4836. \\href{https://doi.org/10.2478/picbe-2026-0352}{doi:10.2478/picbe-2026-0352}",
    "Baesens, B., Roesch, D.\\ \\& Scheule, H.\\ (2016). \\textit{Credit Risk Analytics: Measurement Techniques, Applications, and "
    "Examples in SAS}. Wiley.",
    "Breiman, L.\\ (2001). Random forests. \\textit{Machine Learning}, 45, 5--32. \\href{https://doi.org/10.1023/A:1010933404324}{doi:10.1023/A:1010933404324}",
    "Dal Pozzolo, A., Caelen, O., Johnson, R.\\,A.\\ \\& Bontempi, G.\\ (2015). Calibrating probability with undersampling for unbalanced "
    "classification. IEEE SSCI, 159--166. \\href{https://doi.org/10.1109/SSCI.2015.33}{doi:10.1109/SSCI.2015.33}",
    "Diaconu, D.\\ \\& Pele, D.\\,T.\\ (2026). SHAP stability across random and forward-time validation in graph-augmented fraud detection. "
    "Submitted to \\textit{Expert Systems with Applications}. Code: "
    "\\href{https://github.com/QuantLet/GNN_XAI_Stability}{github.com/QuantLet/GNN\\_XAI\\_Stability}",
    "Friedman, J.\\,H.\\ (2001). Greedy function approximation: A gradient boosting machine. \\textit{Annals of Statistics}, 29(5), "
    "1189--1232. \\href{https://doi.org/10.1214/aos/1013203451}{doi:10.1214/aos/1013203451}",
    "Goldstein, A., Kapelner, A., Bleich, J.\\ \\& Pitkin, E.\\ (2015). Peeking inside the black box: Visualizing statistical learning "
    "with plots of individual conditional expectation. \\textit{Journal of Computational and Graphical Statistics}, 24(1), 44--65. "
    "\\href{https://doi.org/10.1080/10618600.2014.907095}{doi:10.1080/10618600.2014.907095}",
    "Kim, B.\\ et al.\\ (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors "
    "(TCAV). ICML. \\href{https://arxiv.org/abs/1711.11279}{arXiv:1711.11279}",
    "Lundberg, S.\\,M.\\ \\& Lee, S.-I.\\ (2017). A unified approach to interpreting model predictions. \\textit{Advances in Neural "
    "Information Processing Systems}, 30. \\href{https://arxiv.org/abs/1705.07874}{arXiv:1705.07874}",
    "Lundberg, S.\\,M.\\ et al.\\ (2020). From local explanations to global understanding with explainable AI for trees. "
    "\\textit{Nature Machine Intelligence}, 2, 56--67. \\href{https://doi.org/10.1038/s42256-019-0138-9}{doi:10.1038/s42256-019-0138-9}",
    "Ribeiro, M.\\,T., Singh, S.\\ \\& Guestrin, C.\\ (2016). ``Why should I trust you?'' Explaining the predictions of any classifier. "
    "KDD, 1135--1144. \\href{https://doi.org/10.1145/2939672.2939778}{doi:10.1145/2939672.2939778}",
    "Rudin, C.\\ (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models "
    "instead. \\textit{Nature Machine Intelligence}, 1, 206--215. \\href{https://doi.org/10.1038/s42256-019-0048-x}{doi:10.1038/s42256-019-0048-x}",
    "Samek, W., Binder, A., Montavon, G., Lapuschkin, S.\\ \\& Müller, K.-R.\\ (2017). Evaluating the visualization of what a deep "
    "neural network has learned. \\textit{IEEE Transactions on Neural Networks and Learning Systems}, 28(11), 2660--2673. "
    "\\href{https://doi.org/10.1109/TNNLS.2016.2599820}{doi:10.1109/TNNLS.2016.2599820}",
    "Shapley, L.\\,S.\\ (1953). A value for n-person games. In \\textit{Contributions to the Theory of Games II}, 307--317. Princeton "
    "University Press.",
    "Simonyan, K., Vedaldi, A.\\ \\& Zisserman, A.\\ (2014). Deep inside convolutional networks: Visualising image classification "
    "models and saliency maps. ICLR Workshop. \\href{https://arxiv.org/abs/1312.6034}{arXiv:1312.6034}",
    "Smilkov, D., Thorat, N., Kim, B., Viégas, F.\\ \\& Wattenberg, M.\\ (2017). SmoothGrad: Removing noise by adding noise. "
    "\\href{https://arxiv.org/abs/1706.03825}{arXiv:1706.03825}",
    "Sundararajan, M., Taly, A.\\ \\& Yan, Q.\\ (2017). Axiomatic attribution for deep networks. ICML, 3319--3328. "
    "\\href{https://arxiv.org/abs/1703.01365}{arXiv:1703.01365}",
    "Wachter, S., Mittelstadt, B.\\ \\& Russell, C.\\ (2017). Counterfactual explanations without opening the black box: Automated "
    "decisions and the GDPR. \\textit{Harvard Journal of Law \\& Technology}, 31(2), 841--887. \\href{https://doi.org/10.2139/ssrn.3063289}{doi:10.2139/ssrn.3063289}",
    "Regulation (EU) 2024/1689 (Artificial Intelligence Act), \\textit{Official Journal of the European Union}, L, 12 July 2024. "
    "\\href{http://data.europa.eu/eli/reg/2024/1689/oj}{data.europa.eu/eli/reg/2024/1689/oj}",
    "Board of Governors of the Federal Reserve System (2011). SR 11-7: Supervisory guidance on model risk management. "
    "\\href{https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm}{federalreserve.gov}",
]


def it(*xs, sep="1pt"):
    def one(x):
        if isinstance(x, tuple):
            return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                    "\n".join("      \\item " + (s[0].upper() + s[1:] if s[0].islower() else s) for s in x[1:] if s) +
                    "\n    \\end{itemize}")
        return hier(x)
    return (f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{{sep}}}\n" + "\n".join("  \\item " + one(x) for x in xs if x) +
            "\n\\end{itemize}")


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L11_numbers.json"), lang)
    D = Deck(11, "Explainable AI: SHAP, LIME, gradient methods and the stability of explanations" if lang == "en" else
             "Inteligența artificială explicabilă: SHAP, LIME, metode bazate pe gradient și stabilitatea explicațiilor", V, lang, heights=H)
    t = D.t
    ro = lang == "ro"
    NB = "notebook"
    OWN = t("(own computation)", "(calcul propriu)")
    cf = V.N["cf"]

    def tt(f):
        return f.replace("_", "\\_")

    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Explain why banks, insurers and regulators require explanations of a neural network, and classify explanation methods "
          "of explainable artificial intelligence (XAI).",
          "Să explicați de ce băncile, asigurătorii și autoritățile de reglementare cer explicații pentru o rețea neuronală și să clasificați "
          "metodele inteligenței artificiale explicabile (XAI)."),
        t("Compute Shapley additive explanations (SHAP) and local interpretable model-agnostic explanations (LIME) with full formulas, "
          "and read them for one customer and for a portfolio.",
          "Să calculați explicațiile aditive Shapley (SHAP) și explicațiile locale interpretabile, independente de model (LIME), cu formulele "
          "complete, și să le interpretați pentru un client și pentru un portofoliu."),
        t("Use gradient methods (saliency, integrated gradients, SmoothGrad) and check their axioms on a real fraud model.",
          "Să folosiți metodele bazate pe gradient (saliența, gradienții integrați, SmoothGrad) și să verificați axiomele lor pe un model real de detectare a fraudei."),
        t("Test whether explanations agree, are faithful to the model and stay stable across seeds, samples and time.",
          "Să testați dacă explicațiile sînt concordante, fidele modelului și stabile la schimbarea valorii inițiale, a eșantionului și a perioadei."),
        t("Write a counterfactual explanation that a rejected applicant can act on.",
          "Să formulați o explicație contrafactuală pe care un solicitant respins o poate pune în practică."), sep="4pt"))
    abbreviation_frames(D, ABBR, per_frame=36)

    D.frame(t("Real data in this lecture", "Datele reale ale cursului"), table(
        "@{}p{0.17\\textwidth}p{0.42\\textwidth}p{0.34\\textwidth}@{}",
        [t("Case", "Studiul de caz"), t("Data", "Date"), t("Models and split", "Modele și împărțire")],
        [[t("Credit", "Credit"), t("Home-equity loan data set (HMEQ), @{hmeq.n|,d} loans, @{hmeq.bad_pct|.1f}\\,\\% defaults (Baesens et al., 2016)",
                                   "Setul de date privind creditele garantate cu locuința (HMEQ), @{hmeq.n|,d} de credite, @{hmeq.bad_pct|.1f}\\,\\% neplăți (Baesens et al., 2016)"),
          t("multilayer perceptron (MLP), logistic regression, extreme gradient boosting (XGBoost); 60/20/20 stratified",
            "perceptron multistrat (MLP), regresie logistică, extreme gradient boosting (XGBoost); 60/20/20 stratificat")],
         [t("Fraud", "Fraudă"), t("Credit-card transactions of the Université Libre de Bruxelles (ULB) data set, two days of September 2013; @{fraud.fr_day1} + @{fraud.fr_day2} frauds in the sample "
                                  "(Dal Pozzolo et al., 2015)",
                                  "Tranzacții cu carduri din setul Université Libre de Bruxelles (ULB), două zile din septembrie 2013; @{fraud.fr_day1} + @{fraud.fr_day2} fraude în eșantion "
                                  "(Dal Pozzolo et al., 2015)"),
          t("MLP and logistic regression; train on day 1, test on day 2", "MLP și regresie logistică; antrenare pe ziua 1, test pe ziua 2")],
         [t("Housing", "Locuințe"), t("California housing, @{cal.n|,d} block groups of the 1990 census",
                                     "Locuințele din California, @{cal.n|,d} de grupuri de locuințe din recensămîntul din 1990"),
          t("MLP regression (partial dependence)", "MLP de regresie (dependență parțială)")]]) + it(
        t("Every number is computed by analysis\\_L11.py; seeds from @{seed}; explanations computed on an Apple M5 Max processor.",
          "Fiecare cifră este calculată de analysis\\_L11.py; valorile inițiale încep de la @{seed}; explicațiile sînt calculate pe un procesor Apple M5 Max."),
        t("Fraud variables V1--V28 are principal component analysis (PCA) components released by the data owner: explanations name components, not raw fields.",
          "Variabilele V1--V28 de fraudă sînt componente ale analizei componentelor principale (PCA) publicate de proprietarul datelor: explicațiile numesc componente, nu cîmpuri brute."),
        sep="1pt"), "footnotesize")

    # ================================================================== 1. why explain
    D.section(t("Reasons for explaining neural networks", "Motivele explicării rețelelor neuronale"))
    D.frame(t("Regulators require explanations of automated decisions", "Reglementările cer explicarea deciziilor automate"), cols(it(
        t("\\textbf{European Union (EU) Artificial Intelligence (AI) Act} (Regulation 2024/1689): creditworthiness scoring of natural persons is high-risk (Annex III); "
          "transparency to deployers (art.\\ 13) and a right to an explanation of individual decisions (art.\\ 86).",
          "\\textbf{Regulamentul Uniunii Europene (UE) privind inteligența artificială (AI)} (Regulamentul 2024/1689): evaluarea bonității persoanelor fizice este cu risc ridicat (anexa III); "
          "transparență față de utilizatori (art.\\ 13) și dreptul la explicarea deciziilor individuale (art.\\ 86)."),
        t("\\textbf{General Data Protection Regulation (GDPR)}, art.\\ 22 and recital 71: safeguards against decisions based solely on automated processing.",
          "\\textbf{Regulamentul general privind protecția datelor (GDPR)}, art.\\ 22 și considerentul 71: garanții față de deciziile bazate exclusiv pe prelucrarea automată."),
        t("\\textbf{European Banking Authority (EBA)}, guidelines on loan origination and monitoring (2020): models used for credit decisions must be explainable.",
          "\\textbf{Autoritatea Bancară Europeană (EBA)}, ghidul privind acordarea și monitorizarea creditelor (2020): modelele folosite la decizia de creditare trebuie să fie explicabile."),
        sep="3pt"), it(
        t("\\textbf{Model risk}: the Federal Reserve Supervisory Guidance on Model Risk Management (SR 11-7) asks for conceptual soundness, "
          "independent validation and ongoing monitoring.",
          "\\textbf{Riscul de model}: ghidul de supraveghere al Federal Reserve privind riscul de model (SR 11-7) cere fundamentare conceptuală, "
          "validare independentă și monitorizare continuă."),
        t("\\textbf{Business reasons}: finding leakage, trust of credit officers, explaining a rejection.",
          "\\textbf{Motive de afaceri}: depistarea scurgerilor de informație, încrederea analiștilor de credit, explicarea unei respingeri pentru client."),
        t("\\textbf{But}: an explanation is a model of the model, and it can be wrong.",
          "\\textbf{Totuși}: o explicație este un model al modelului și poate fi greșită."), sep="3pt")) +
        D.bottom(t("an explanation that changes when nothing important changed is a compliance risk, not a safeguard.",
                   "o explicație care se schimbă fără ca ceva important să se schimbe este un risc de conformitate, nu o garanție.")),
        "footnotesize", cite="(Regulation (EU) 2024/1689; SR 11-7)")

    D.frame(t("A taxonomy of explanation methods", "O clasificare a metodelor de explicare"), table(
        "@{}p{0.22\\textwidth}p{0.2\\textwidth}p{0.2\\textwidth}p{0.3\\textwidth}@{}",
        [t("Method", "Metoda"), t("Scope", "Domeniu"), t("Needs", "Necesită"), t("This lecture", "În acest curs")],
        [[t("Coefficients, scorecards", "Coeficienți, grile de scor"), t("global, intrinsic", "global, intrinsec"), t("a linear model", "un model liniar"), t("logistic benchmark", "reperul logistic")],
         ["SHAP", t("local $\\to$ global", "local $\\to$ global"), t("model calls (Kernel); trees (TreeSHAP)", "apeluri ale modelului (Kernel); arbori (TreeSHAP)"), t("credit MLP, XGBoost", "MLP de credit, XGBoost")],
         ["LIME", t("local", "local"), t("model calls", "apeluri ale modelului"), t("same applicant", "același solicitant")],
         [t("Saliency, integrated gradients (IG), SmoothGrad", "Saliență, gradienți integrați (IG), SmoothGrad"), t("local", "local"), t("gradients", "gradienți"), t("fraud MLP", "MLP de fraudă")],
         [t("Partial dependence plot (PDP), individual conditional expectation (ICE)", "Graficul dependenței parțiale (PDP), așteptarea condiționată individuală (ICE)"),
          t("global effect", "efect global"), t("model calls", "apeluri ale modelului"), t("housing MLP", "MLP pentru locuințe")],
         [t("Counterfactuals", "Contrafactuale"), t("local, actionable", "local, acționabil"), t("gradients or search", "gradienți sau căutare"), t("rejected applicant", "solicitantul respins")],
         [t("Permutation importance", "Importanța prin permutare"), t("global", "global"), t("labels + model calls", "etichete + apeluri ale modelului"), t("fraud MLP", "MLP de fraudă")]], size="scriptsize") +
        it(t("Post-hoc methods explain a trained black box; interpretable models are transparent by design (Rudin, 2019).",
             "Metodele post-hoc explică o cutie neagră deja antrenată; modelele interpretabile sînt transparente prin construcție (Rudin, 2019)."), sep="1pt"),
        "footnotesize", cite="(Rudin, 2019)")

    D.concept(t("Credit case study: comparison of three scoring models", "Studiul de caz pentru credit: compararea a trei modele de scoring"),
              t("the network ranks defaulters better than the logistic scorecard (higher area under the receiver operating characteristic (ROC) curve (AUC) and "
                "area under the precision-recall curve (PR-AUC)); can we explain it as well?",
                "rețeaua ordonează rău-platnicii mai bine decît grila logistică (aria de sub curba de operare (ROC) (AUC) și "
                "aria de sub curba precizie-sensibilitate (PR-AUC) sînt mai mari); o putem explica la fel de bine?"),
              "\\[ \\vh_1=\\ReLU(\\mW_1\\vx+\\vb_1),\\quad \\vh_2=\\ReLU(\\mW_2\\vh_1+\\vb_2) \\]"
              "\\[ f(\\vx)=\\vw_3^{\\top}\\vh_2+b_3,\\qquad \\mathrm{PD}(\\vx)=\\sigma\\big(f(\\vx)\\big) \\]" +
              "\\where{" + t("$\\vx$ = @{hmeq.n_feat} standardised features, median-imputed, with a flag for a missing debt-to-income ratio (DEBTINC); $f$ = log-odds of default; "
                             "64 and 32 rectified linear units (ReLU), dropout 0.1; $\\mathrm{PD}$ = probability of default (PD).",
                             "$\\vx$ = @{hmeq.n_feat} variabile standardizate, cu indicator pentru raportul datorie/venit lipsă (DEBTINC); $f$ = log-șansele de neplată; "
                             "64 și 32 de unități liniare rectificate (ReLU), dropout 0{,}1; $\\mathrm{PD}$ = probabilitatea de nerambursare (PD).") + "}",
              table("@{}lrrr@{}", [t("Model (test)", "Model (test)"), "AUC", "PR-AUC", "Brier"],
                    [[t("Logistic", "Logistic"), "@{perf.lr_auc|.3f}", "@{perf.lr_pr|.3f}", "@{perf.lr_brier|.3f}"],
                     [t("MLP, 10 seeds", "MLP, 10 valori"), "@{perf.mlp_auc|.3f}", "@{perf.mlp_pr|.3f}", "@{perf.mlp_brier|.3f}"],
                     ["XGBoost, 10", "@{perf.xgb_auc|.3f}", "@{perf.xgb_pr|.3f}", "@{perf.xgb_brier|.3f}"]], size="footnotesize") + it(
                  t("Base rate (PR-AUC of a random score) @{perf.base_pr|.3f}.", "Rata de bază (PR-AUC al unui scor aleator) @{perf.base_pr|.3f}."), sep="0pt"),
              t("the logistic coefficients are the explanation; for the MLP the explanation must be computed.",
                "la modelul logistic coeficienții sînt explicația; pentru MLP explicația trebuie calculată."),
              tryit=NB + " \\S1", cite="(Baesens, Roesch \\& Scheule, 2016; " + t("own computation", "calcul propriu") + ")", wl="0.53", wr="0.44")

    # ================================================================== 2. SHAP
    D.section(t("Shapley values and SHAP", "Valorile Shapley și SHAP"))
    D.concept(t("Shapley values: a fair split of the prediction among features", "Valorile Shapley: o împărțire echitabilă a prognozei între variabile"),
              t("each feature receives its average marginal contribution over all orders in which features could join the prediction.",
                "fiecare variabilă primește contribuția sa marginală medie pe toate ordinile în care variabilele s-ar putea alătura prognozei."),
              "\\[ \\phi_j=\\sum_{S\\subseteq\\{1,\\dots,d\\}\\setminus\\{j\\}}\\frac{|S|!\\,(d-|S|-1)!}{d!}\\big[v(S\\cup\\{j\\})-v(S)\\big] \\]" +
              "\\where{" + t("$d$ = number of features; $v(S)$ = value of the coalition $S$ (what the model predicts when only the features in $S$ are known); "
                             "the weight is the share of orderings in which $j$ joins right after $S$.",
                             "$d$ = numărul de variabile; $v(S)$ = valoarea coaliției $S$ (ce prognozează modelul cînd sînt cunoscute doar variabilele din $S$); "
                             "ponderea este ponderea ordinilor în care $j$ intră imediat după $S$.") + "}",
              it((t("\\textbf{The four axioms} (Shapley, 1953) that make $\\phi$ unique", "\\textbf{Cele patru axiome} (Shapley, 1953) care fac $\\phi$ unic"),
                  t("efficiency: $\\sum_j\\phi_j=v(\\{1..d\\})-v(\\emptyset)$", "eficiența: $\\sum_j\\phi_j=v(\\{1..d\\})-v(\\emptyset)$"),
                  t("symmetry: equal contributions, equal shares", "simetria: contribuții egale, cote egale"),
                  t("dummy: a feature that never changes $v$ gets 0", "variabila nulă: o variabilă care nu schimbă niciodată $v$ primește 0"),
                  t("additivity: $\\phi(v_1+v_2)=\\phi(v_1)+\\phi(v_2)$", "aditivitatea: $\\phi(v_1+v_2)=\\phi(v_1)+\\phi(v_2)$")),
                 t("\\textbf{Cost}: $2^{d}$ coalitions; with @{hmeq.n_feat} features, $2^{18}$ model values per applicant: we need approximations.",
                   "\\textbf{Cost}: $2^{d}$ coaliții; cu @{hmeq.n_feat} variabile, $2^{18}$ evaluări ale modelului pentru fiecare solicitant: avem nevoie de aproximări."), sep="3pt"),
              t("efficiency is what a credit officer needs: the parts add up exactly to the score.",
                "eficiența este ceea ce îi trebuie analistului de credit: părțile se adună exact la scor."),
              cite="(Shapley, 1953)", wl="0.53", wr="0.44")

    D.concept(t("SHAP: the value of a coalition of features", "SHAP: valoarea unei coaliții de variabile"),
              t("SHAP sets $v(S)$ = expected model output when the features outside $S$ are drawn from a background sample.",
                "SHAP definește $v(S)$ = ieșirea așteptată a modelului cînd variabilele din afara lui $S$ sînt extrase dintr-un eșantion de referință."),
              "\\[ v(S)=\\E_{\\vx_{\\bar S}\\sim\\mathcal{B}}\\big[f(\\vx_S,\\vx_{\\bar S})\\big] \\]"
              "\\[ f(\\vx)=\\phi_0+\\sum_{j=1}^{d}\\phi_j(\\vx),\\qquad \\phi_0=\\E_{\\mathcal{B}}[f] \\]" +
              "\\where{" + t("$\\mathcal{B}$ = 100 training applicants (background); interventional expectation: features outside $S$ are replaced independently of $\\vx_S$; "
                             "explanations are on the log-odds scale, where effects add.",
                             "$\\mathcal{B}$ = 100 de solicitanți de antrenare (referința); variabilele din afara lui $S$ sînt înlocuite independent de $\\vx_S$; "
                             "explicațiile sînt pe scara log-șanselor, unde efectele se adună.") + "}",
              it(t("\\textbf{Additivity on our data}: largest error $|\\phi_0+\\sum_j\\phi_j-f(\\vx)|$ = @{shap.add_err|.1e} over @{shap.n_explained} test applicants.",
                   "\\textbf{Aditivitatea pe datele noastre}: eroarea maximă $|\\phi_0+\\sum_j\\phi_j-f(\\vx)|$ = @{shap.add_err|.1e} pe @{shap.n_explained} de solicitanți din test."),
                 t("\\textbf{Linear model}: $\\phi_j=\\beta_j(x_j-\\E x_j)$ exactly (linear SHAP).", "\\textbf{Model liniar}: $\\phi_j=\\beta_j(x_j-\\E x_j)$ exact (SHAP liniar)."),
                 t("\\textbf{Off-manifold risk}: with correlated features, independent replacement creates unrealistic applicants.",
                   "\\textbf{Risc}: cu variabile corelate, înlocuirea independentă creează solicitanți nerealiști."), sep="3pt"),
              t("the background sample defines the ``average applicant''; document it like any model input.",
                "eșantionul de referință definește ``solicitantul mediu''; documentați-l ca pe orice intrare a modelului."),
              tryit=NB + " \\S2", cite="(Lundberg \\& Lee, 2017)", wl="0.55", wr="0.42")

    D.concept(t("Kernel SHAP and TreeSHAP: how SHAP is computed", "Kernel SHAP și TreeSHAP: cum se calculează SHAP"),
              t("Kernel SHAP fits a weighted linear regression on sampled coalitions; for trees the exact values come in polynomial time.",
                "Kernel SHAP estimează o regresie liniară ponderată pe coaliții eșantionate; pentru arbori valorile exacte se obțin în timp polinomial."),
              "\\[ \\min_{\\phi}\\sum_{\\vz\\in\\{0,1\\}^{d}}\\pi(\\vz)\\Big[v(\\vz)-\\phi_0-\\sum_j\\phi_j z_j\\Big]^2 \\]"
              "\\[ \\pi(\\vz)=\\frac{d-1}{\\binom{d}{|\\vz|}\\,|\\vz|\\,(d-|\\vz|)} \\]" +
              "\\where{" + t("$\\vz$ = coalition indicator; the Shapley kernel $\\pi$ makes the weighted least-squares solution equal the Shapley values (Appendix A); "
                             "we sample @{shap.nsamples} coalitions per applicant.",
                             "$\\vz$ = indicatorul coaliției; nucleul Shapley $\\pi$ face ca soluția celor mai mici pătrate ponderate să fie egală cu valorile Shapley (Anexa A); "
                             "eșantionăm @{shap.nsamples} de coaliții pentru fiecare solicitant.") + "}",
              it(t("\\textbf{Kernel SHAP}: model-agnostic; @{shap.seconds|.0f} s for @{shap.n_explained} applicants.",
                   "\\textbf{Kernel SHAP}: independent de model; @{shap.seconds|.0f} s pentru @{shap.n_explained} de solicitanți."),
                 t("\\textbf{TreeSHAP}: exact for tree ensembles such as XGBoost (Lundberg et al., 2020).",
                   "\\textbf{TreeSHAP}: exact pentru ansambluri de arbori (Lundberg et al., 2020)."),
                 t("\\textbf{Deep networks}: Deep/Gradient SHAP combine Shapley values with gradients (Section 3).",
                   "\\textbf{Rețele adînci}: Deep/Gradient SHAP combină valorile Shapley cu gradienții (secțiunea 3)."), sep="3pt"),
              t("Kernel SHAP has sampling noise: fix the seed and the number of samples in the model documentation.",
                "Kernel SHAP are zgomot de eșantionare: documentați valoarea inițială și numărul de eșantioane."),
              tryit=NB + " \\S2", cite="(Lundberg \\& Lee, 2017; Lundberg et al., 2020)", wl="0.56", wr="0.41")

    D.figframe(t("Global importance: the missing debt-to-income ratio dominates", "Importanța globală: lipsa raportului datorie/venit domină"),
               t("mean absolute SHAP value per feature, for the network and the two benchmarks.",
                 "valoarea SHAP absolută medie pe variabilă, pentru rețea și cele două repere."),
               None, "shap_global", bullets=[
                   t("Top for all three models: " + tt(V.N["shap"]["top1"]) + "; MLP: then " + tt(V.N["shap"]["top2"]) + " and " + tt(V.N["shap"]["top3"]) +
                     "; Spearman correlation of importances MLP--XGBoost @{shap.rho_mlp_xgb|.2f}, MLP--logistic @{shap.rho_mlp_lr|.2f}.",
                     "Pe primul loc la toate trei modelele: " + tt(V.N["shap"]["top1"]) + "; la MLP urmează " + tt(V.N["shap"]["top2"]) + " și " + tt(V.N["shap"]["top3"]) +
                     "; corelația Spearman a importanțelor MLP--XGBoost @{shap.rho_mlp_xgb|.2f}, MLP--logistic @{shap.rho_mlp_lr|.2f}."),
                   t("Debt-to-income ratio (DEBTINC) missing for @{hmeq.debtinc_missing_pct|.1f}\\,\\% of loans; default rate @{hmeq.bad_if_debtinc_missing|.1f}\\,\\% when missing vs "
                     "@{hmeq.bad_if_debtinc_present|.1f}\\,\\% when reported.",
                     "Raportul datorie/venit (DEBTINC) lipsește la @{hmeq.debtinc_missing_pct|.1f}\\,\\% dintre credite; rata de neplată este @{hmeq.bad_if_debtinc_missing|.1f}\\,\\% cînd lipsește și "
                     "@{hmeq.bad_if_debtinc_present|.1f}\\,\\% cînd este raportat.")],
               job=t("an explanation can reveal that a model scores the paperwork rather than the borrower; decide whether that is acceptable.",
                     "o explicație poate arăta că modelul notează dosarul, nu debitorul; decideți dacă acest lucru este acceptabil."),
               tryit=NB + " \\S2", height="0.50\\textheight", cite=OWN)

    D.figframe(t("The beeswarm: direction and spread of every effect", "Diagrama roi: sensul și dispersia fiecărui efect"),
               t("each dot is one applicant; colour = rank of the feature value; position = its SHAP value.",
                 "fiecare punct este un solicitant; culoarea = rangul valorii variabilei; poziția = valoarea SHAP."),
               None, "shap_beeswarm", bullets=[
                   t("Delinquent credit lines (DELINQ) and derogatory reports (DEROG): high values push the risk up, as credit theory expects.",
                     "Liniile de credit cu restanțe (DELINQ) și rapoartele negative (DEROG): valorile mari cresc riscul, așa cum se așteaptă în teoria creditului.")],
               tryit=NB + " \\S2", height="0.56\\textheight", cite=OWN)

    D.figframe(t("Local explanation of a rejected applicant: SHAP waterfall", "Explicația locală pentru un solicitant respins: diagrama cascadă SHAP"),
               t("from the average log-odds $\\phi_0$ to this applicant's score, one feature at a time.",
                 "de la log-șansele medii $\\phi_0$ la scorul acestui solicitant, variabilă cu variabilă."),
               None, "shap_waterfall", bullets=[
                   t("Test applicant (row @{app.row}): PD @{app.pd|.1f}\\,\\%, log-odds @{app.logit|.3f} = @{app.base|.3f} + @{app.sum_phi|.3f}; largest pushes: "
                     + tt(V.N["app"]["f1"]) + " (@{app.f1_phi|+.2f}), " + tt(V.N["app"]["f2"]) + " (@{app.f2_phi|+.2f}), " + tt(V.N["app"]["f3"]) + " (@{app.f3_phi|+.2f}).",
                     "Solicitant din test (rîndul @{app.row}): PD @{app.pd|.1f}\\,\\%, log-șanse @{app.logit|.3f} = @{app.base|.3f} + @{app.sum_phi|.3f}; cele mai mari contribuții: "
                     + tt(V.N["app"]["f1"]) + " (@{app.f1_phi|+.2f}), " + tt(V.N["app"]["f2"]) + " (@{app.f2_phi|+.2f}), " + tt(V.N["app"]["f3"]) + " (@{app.f3_phi|+.2f}).")],
               job=t("the adverse-action notice lists the top reasons in this order, in words the customer understands.",
                     "notificarea de respingere enumeră principalele motive în această ordine, în cuvinte pe care clientul le înțelege."),
               tryit=NB + " \\S2", height="0.52\\textheight", cite=OWN)

    D.figframe(t("Dependence plot: the network learned the missing-value signal", "Graficul de dependență: rețeaua a învățat semnalul valorii lipsă"),
               t("SHAP value of DEBTINC against its value; red points are imputed at the training median.",
                 "valoarea SHAP a DEBTINC în funcție de valoarea sa; punctele roșii sînt imputate cu mediana din antrenare."),
               None, "shap_dependence", bullets=[
                   t("Imputed applicants sit at one DEBTINC value; their risk is carried by the missing flag (mean SHAP @{dep.shap_missing_flag|+.2f}), not by DEBTINC itself.",
                     "Solicitanții cu valoare imputată au aceeași valoare DEBTINC; riscul lor este preluat de indicatorul de lipsă (SHAP mediu @{dep.shap_missing_flag|+.2f}), nu de DEBTINC.")],
               job=t("imputation choices change explanations; keep a missing-value flag so that the reason can be named.",
                     "alegerea metodei de imputare schimbă explicațiile; păstrați un indicator de lipsă pentru ca motivul să poată fi numit."),
               tryit=NB + " \\S2", height="0.50\\textheight", cite=OWN)

    # ================================================================== 3. LIME, counterfactuals, PDP
    D.section(t("LIME, counterfactuals and global effects", "LIME, contrafactuale și efecte globale"))
    D.concept(t("LIME: a simple model that is faithful near one point", "LIME: un model simplu, fidel în jurul unui punct"),
              t("perturb the applicant, weight the perturbations by closeness, fit a sparse linear model to the network's answers.",
                "solicitantul este perturbat, perturbările sînt ponderate după apropiere și se estimează un model liniar rar pe răspunsurile rețelei."),
              "\\[ \\xi(\\vx)=\\argmin_{g\\in\\mathcal{G}}\\ \\sum_{\\vz}\\pi_{\\vx}(\\vz)\\big[f(\\vz)-g(\\vz)\\big]^2+\\Omega(g) \\]"
              "\\[ \\pi_{\\vx}(\\vz)=\\exp\\!\\Big(-\\frac{\\|\\vz-\\vx\\|^2}{\\sigma^2}\\Big) \\]" +
              "\\where{" + t("$\\mathcal{G}$ = linear models $g(\\vz)=\\beta_0+\\boldsymbol\\beta^{\\top}\\vz$; $\\Omega$ = complexity penalty; $\\vz\\sim\\mathcal{N}(\\vx,\\mathbf{I})$ in standardised units; "
                             "kernel width $\\sigma=0.75\\sqrt{d}$ = @{lime.kw_default|.2f} by default.",
                             "$\\mathcal{G}$ = modelele liniare $g(\\vz)=\\beta_0+\\boldsymbol\\beta^{\\top}\\vz$; $\\Omega$ = penalizarea complexității; $\\vz\\sim\\mathcal{N}(\\vx,\\mathbf{I})$ în unități standardizate; "
                             "lățimea nucleului $\\sigma=0{,}75\\sqrt{d}$ = @{lime.kw_default|.2f} implicit.") + "}",
              it(t("\\textbf{Local slope}, not a split of the prediction: the coefficients do not add up to $f(\\vx)$.",
                   "\\textbf{Pantă locală}, nu o împărțire a prognozei: coeficienții nu se adună la $f(\\vx)$."),
                 t("\\textbf{Three choices} decide the answer: neighbourhood, kernel width, number of samples.",
                   "\\textbf{Trei alegeri} decid rezultatul: vecinătatea, lățimea nucleului, numărul de eșantioane."),
                 t("\\textbf{Our applicant}: LIME's top feature is " + tt(V.N["lime"]["top1"]) + ", SHAP's is " + tt(V.N["app"]["f1"]) + ".",
                   "\\textbf{Solicitantul nostru}: variabila de top la LIME este " + tt(V.N["lime"]["top1"]) + ", la SHAP este " + tt(V.N["app"]["f1"]) + "."), sep="3pt"),
              t("LIME answers ``what moves the score near this customer?'', SHAP answers ``why is this customer far from average?''.",
                "LIME răspunde la ``ce mișcă scorul în jurul acestui client?'', iar SHAP la ``de ce este acest client departe de medie?''."),
              tryit=NB + " \\S3", cite="(Ribeiro, Singh \\& Guestrin, 2016)", wl="0.56", wr="0.41")

    D.figframe(t("LIME and SHAP for the same applicant", "LIME și SHAP pentru același solicitant"),
               t("LIME slope times the standardised value, next to the SHAP contribution, for the eight largest SHAP features.",
                 "panta LIME înmulțită cu valoarea standardizată, alături de contribuția SHAP, pentru cele mai importante opt variabile SHAP."),
               None, "lime_vs_shap", bullets=[
                   t("Spearman correlation of the absolute contributions: @{lime.rho_shap|.2f}; a binary flag is far from its local linear approximation.",
                     "Corelația Spearman a contribuțiilor absolute: @{lime.rho_shap|.2f}; un indicator binar este departe de aproximarea sa liniară locală.")],
               tryit=NB + " \\S3", height="0.50\\textheight", cite=OWN)

    D.figframe(t("Sensitivity of LIME to its settings", "Sensibilitatea LIME la propriile setări"),
               t("left: 30 runs with different random seeds; right: the five slopes as the kernel width changes.",
                 "stînga: 30 de rulări cu valori inițiale diferite; dreapta: cele cinci pante cînd se schimbă lățimea nucleului."),
               None, "lime_instability", bullets=[
                   t("Same top feature: @{lime.top1_same_500|.0f}\\,\\% of runs with 500 samples, @{lime.top1_same_5000|.0f}\\,\\% with 5,000; Jaccard overlap of the top-5 sets "
                     "@{lime.jacc_500|.2f} vs @{lime.jacc_5000|.2f}; coefficient of variation @{lime.cv_500|.0f}\\,\\% vs @{lime.cv_5000|.0f}\\,\\%.",
                     "Aceeași variabilă de top: @{lime.top1_same_500|.0f}\\,\\% din rulări cu 500 de eșantioane, @{lime.top1_same_5000|.0f}\\,\\% cu 5 000; suprapunerea Jaccard a primelor 5 variabile "
                     "@{lime.jacc_500|.2f}, respectiv @{lime.jacc_5000|.2f}; coeficientul de variație @{lime.cv_500|.0f}\\,\\%, respectiv @{lime.cv_5000|.0f}\\,\\%.")],
               job=t("fix and report the seed, sample size and kernel width of every LIME explanation that reaches a customer or an auditor.",
                     "fixați și raportați valoarea inițială, mărimea eșantionului și lățimea nucleului pentru orice explicație LIME care ajunge la un client sau la un auditor."),
               tryit=NB + " \\S3", height="0.50\\textheight", cite=OWN)

    rows = [[tt(cf[f"f{k}"]), f"@{{cf.b{k}|,.1f}}", f"@{{cf.a{k}|,.1f}}"] for k in range(cf["n_changed"])]
    D.concept(t("Counterfactual explanations: minimal changes that reverse the decision", "Explicații contrafactuale: modificările minime care schimbă decizia"),
              t("find the closest applicant, in actionable features only, whom the model would accept.",
                "se caută cel mai apropiat solicitant, schimbînd doar variabile acționabile, pe care modelul l-ar accepta."),
              "\\[ \\vx'=\\argmin_{\\vx'}\\ \\lambda\\,\\big(f(\\vx')-t\\big)_+^{2}+\\sum_{j\\in\\mathcal{A}}\\frac{|x'_j-x_j|}{\\mathrm{MAD}_j} \\]" +
              "\\where{" + t("$t$ = target log-odds ($-0.5$, below the 0.5 PD threshold); $\\mathcal{A}$ = actionable features (" + tt(cf["actionable"]) + "); "
                             "median absolute deviation (MAD) scales the distance; raw values kept $\\ge0$, counts rounded.",
                             "$t$ = log-șansele-țintă ($-0{,}5$, sub pragul PD de 0{,}5); $\\mathcal{A}$ = variabilele acționabile (" + tt(cf["actionable"]) + "); "
                             "abaterea absolută mediană (MAD) scalează distanța; valorile brute rămîn $\\ge0$, numărătorile sînt rotunjite.") + "}",
              table("@{}lrr@{}", [t("Feature", "Variabila"), t("now", "acum"), t("needed", "necesar")], rows, size="footnotesize") + it(
                  t("PD @{cf.pd_before|.1f}\\,\\% $\\to$ @{cf.pd_after|.1f}\\,\\% after these changes.",
                    "PD @{cf.pd_before|.1f}\\,\\% $\\to$ @{cf.pd_after|.1f}\\,\\% după aceste schimbări."),
                  t("Documentary counterfactual: reporting DEBTINC (at the median @{cf.debtinc_median|.1f}) gives PD @{cf.pd_doc|.1f}\\,\\%.",
                    "Contrafactualul documentar: raportarea DEBTINC (la mediana @{cf.debtinc_median|.1f}) dă PD @{cf.pd_doc|.1f}\\,\\%."), sep="1pt"),
              t("``bring the missing income document'' is a better answer to the customer than any list of SHAP values.",
                "``aduceți documentul de venit care lipsește'' este un răspuns mai bun pentru client decît orice listă de valori SHAP."),
              tryit=NB + " \\S4", cite="(Wachter, Mittelstadt \\& Russell, 2017)", wl="0.55", wr="0.42")

    D.figframe(t("Global effects: partial dependence and ICE", "Efecte globale: dependența parțială și ICE"),
               t("the PDP averages the prediction over the data while one feature is set to a grid value; ICE keeps one curve per observation.",
                 "PDP mediază prognoza pe date cînd o variabilă este fixată la o valoare din grilă; ICE păstrează cîte o curbă pentru fiecare observație."),
               "\\[ \\mathrm{PD}_j(z)=\\frac1n\\sum_{i=1}^{n}f(z,\\vx_{i,-j}),\\qquad \\mathrm{ICE}_{i,j}(z)=f(z,\\vx_{i,-j}) \\]",
               "pdp_ice", bullets=[
                   t("California housing MLP (test $R^2$ @{cal.r2_mlp|.3f}, linear @{cal.r2_lin|.3f}): the predicted value rises from @{pdp.pdp_lo|.2f} to @{pdp.pdp_hi|.2f} "
                     "(100,000 USD) over the income grid; spread of the ICE curves at the top @{pdp.ice_spread_hi|.2f}.",
                     "MLP pentru locuințele din California ($R^2$ de test @{cal.r2_mlp|.3f}, liniar @{cal.r2_lin|.3f}): valoarea prognozată crește de la @{pdp.pdp_lo|.2f} la @{pdp.pdp_hi|.2f} "
                     "(100 000 USD) pe grila de venit; dispersia curbelor ICE la capăt @{pdp.ice_spread_hi|.2f}.")],
               tryit=NB + " \\S8", height="0.42\\textheight", cite="(Friedman, 2001; Goldstein et al., 2015)")

    # ================================================================== 4. gradients
    D.section(t("Gradient methods for neural networks", "Metode bazate pe gradient pentru rețele neuronale"))
    D.concept(t("Fraud case study: a network trained on day 1 and explained on day 2", "Studiul de caz pentru fraudă: o rețea antrenată pe ziua 1 și explicată pe ziua 2"),
              t("gradients are free for a network: one backward pass gives the sensitivity of the fraud score to every input.",
                "pentru o rețea gradienții sînt gratuiți: o singură trecere înapoi dă sensibilitatea scorului de fraudă la fiecare intrare."),
              "\\[ \\mathrm{saliency}_j(\\vx)=\\Big|\\frac{\\partial f(\\vx)}{\\partial x_j}\\Big|,\\qquad \\mathrm{G\\!\\times\\!I}_j(\\vx)=(x_j-x'_j)\\,\\frac{\\partial f(\\vx)}{\\partial x_j} \\]" +
              "\\where{" + t("$f$ = fraud log-odds of an MLP on V1--V28 and log amount; $\\vx'$ = baseline = mean legitimate transaction of day 1; class weight $\\sqrt{\\text{imbalance}}$, "
                             "early stopping on weighted validation PR-AUC.",
                             "$f$ = log-șansele de fraudă ale unui MLP pe V1--V28 și logaritmul sumei; $\\vx'$ = referința = tranzacția legitimă medie din ziua 1; ponderea clasei $\\sqrt{\\text{dezechilibru}}$, "
                             "oprire timpurie după PR-AUC ponderat de validare.") + "}",
              it(t("\\textbf{Test, day 2} (full-data weights): MLP PR-AUC @{fraud.mlp_prauc|.3f}, logistic @{fraud.lr_prauc|.3f}; AUC @{fraud.mlp_auc|.3f} vs @{fraud.lr_auc|.3f}.",
                   "\\textbf{Test, ziua 2} (ponderi pentru datele complete): PR-AUC MLP @{fraud.mlp_prauc|.3f}, logistic @{fraud.lr_prauc|.3f}; AUC @{fraud.mlp_auc|.3f}, respectiv @{fraud.lr_auc|.3f}."),
                 t("\\textbf{Saturation}: in a flat region of $f$ the gradient is near zero even for a decisive feature.",
                   "\\textbf{Saturarea}: într-o zonă plată a lui $f$ gradientul este aproape zero chiar și pentru o variabilă decisivă."),
                 t("\\textbf{ReLU networks}: the gradient is piecewise constant, so saliency jumps between linear regions.",
                   "\\textbf{Rețelele ReLU}: gradientul este constant pe porțiuni, deci saliența sare între regiunile liniare."), sep="3pt"),
              t("a fraud analyst gets a ranked list of the inputs that drove the alert, in milliseconds per transaction.",
                "un analist de fraudă primește lista ordonată a intrărilor care au declanșat alerta, în milisecunde pe tranzacție."),
              tryit=NB + " \\S5", cite="(Simonyan, Vedaldi \\& Zisserman, 2014; Dal Pozzolo et al., 2015)", wl="0.55", wr="0.42")

    D.concept(t("Integrated gradients and the completeness of attributions", "Gradienții integrați și completitudinea atribuirilor"),
              t("integrate the gradient along the straight path from a baseline to the input; the attributions sum to $f(\\vx)-f(\\vx')$.",
                "se integrează gradientul pe drumul drept de la referință la intrare; atribuirile se adună la $f(\\vx)-f(\\vx')$."),
              "\\[ \\mathrm{IG}_j(\\vx)=(x_j-x'_j)\\int_0^1\\frac{\\partial f\\big(\\vx'+\\alpha(\\vx-\\vx')\\big)}{\\partial x_j}\\,d\\alpha \\]"
              "\\[ \\approx(x_j-x'_j)\\frac1m\\sum_{k=1}^{m}\\frac{\\partial f\\big(\\vx'+\\tfrac{k-1/2}{m}(\\vx-\\vx')\\big)}{\\partial x_j} \\]" +
              "\\where{" + t("$m$ = Riemann steps (midpoint rule, 64 here); completeness $\\sum_j\\mathrm{IG}_j=f(\\vx)-f(\\vx')$ follows from the fundamental theorem of calculus for paths.",
                             "$m$ = pașii Riemann (regula punctului de mijloc, 64 aici); completitudinea $\\sum_j\\mathrm{IG}_j=f(\\vx)-f(\\vx')$ rezultă din teorema fundamentală a calculului integral pe drumuri.") + "}",
              it((t("\\textbf{Axioms} (Sundararajan et al., 2017)", "\\textbf{Axiome} (Sundararajan et al., 2017)"),
                  t("completeness; sensitivity: a feature that changes $f$ gets non-zero credit", "completitudinea; sensibilitatea: o variabilă care schimbă $f$ primește o atribuire nenulă"),
                  t("implementation invariance: equal functions, equal attributions", "invarianța la implementare: funcții egale, atribuiri egale")),
                 t("\\textbf{Link to Shapley}: IG is the Aumann-Shapley value of the continuous game along the path.",
                   "\\textbf{Legătura cu Shapley}: IG este valoarea Aumann-Shapley a jocului continuu de-a lungul drumului."),
                 t("\\textbf{Baseline matters}: ``mean legitimate transaction'' answers ``why is this not a normal payment?''.",
                   "\\textbf{Referința contează}: ``tranzacția legitimă medie'' răspunde la ``de ce nu este aceasta o plată normală?''."), sep="3pt"),
              t("write the baseline into the model documentation: a different baseline is a different question.",
                "scrieți referința în documentația modelului: o altă referință înseamnă o altă întrebare."),
              tryit=NB + " \\S5", cite="(Sundararajan, Taly \\& Yan, 2017)", wl="0.6", wr="0.37")

    D.figframe(t("Completeness holds once the integral is computed accurately", "Completitudinea se verifică atunci cînd integrala este calculată precis"),
               t("median of $|\\sum_j\\mathrm{IG}_j-(f(\\vx)-f(\\vx'))|/|f(\\vx)-f(\\vx')|$ over the frauds of day 2.",
                 "mediana lui $|\\sum_j\\mathrm{IG}_j-(f(\\vx)-f(\\vx'))|/|f(\\vx)-f(\\vx')|$ pe fraudele din ziua 2."),
               None, "ig_completeness", bullets=[
                   t("@{ig.comp4_pct|.1f}\\,\\% error with 4 steps, @{ig.comp64_pct|.2f}\\,\\% with 64, over @{ig.n_frauds} frauds; the error falls roughly as $1/m^{2}$, as for the midpoint rule.",
                     "Eroare de @{ig.comp4_pct|.1f}\\,\\% cu 4 pași și de @{ig.comp64_pct|.2f}\\,\\% cu 64, pe @{ig.n_frauds} de fraude; eroarea scade aproximativ ca $1/m^{2}$, ca la regula punctului de mijloc.")],
               tryit=NB + " \\S5", height="0.50\\textheight", cite=OWN)

    D.concept(t("SmoothGrad and concept-level explanations", "SmoothGrad și explicațiile la nivel de concept"),
              t("average gradients over noisy copies of the input to remove ReLU jitter; or ask whether a human concept moves the output.",
                "se mediază gradienții pe copii perturbate ale intrării pentru a elimina oscilațiile ReLU; sau se testează dacă un concept uman mișcă ieșirea."),
              "\\[ \\mathrm{SG}_j(\\vx)=\\frac1n\\sum_{k=1}^{n}\\frac{\\partial f(\\vx+\\boldsymbol\\varepsilon_k)}{\\partial x_j},\\quad \\boldsymbol\\varepsilon_k\\sim\\mathcal{N}(\\mathbf 0,s^{2}\\mathbf I) \\]"
              "\\[ S_{C}(\\vx)=\\nabla h(\\vx)^{\\top}\\mathbf{v}_C \\]" +
              "\\where{" + t("$n=50$, $s=0.15$ standard deviations; testing with concept activation vectors (TCAV): $\\mathbf{v}_C$ = normal of a linear classifier separating examples of concept $C$ "
                             "in a hidden layer $h$; the TCAV score is the share of inputs with $S_C>0$.",
                             "$n=50$, $s=0{,}15$ abateri standard; testarea cu vectori de activare a conceptelor (TCAV): $\\mathbf{v}_C$ = normala unui clasificator liniar care separă exemplele conceptului $C$ "
                             "într-un strat ascuns $h$; scorul TCAV este ponderea intrărilor cu $S_C>0$.") + "}",
              it(t("\\textbf{SmoothGrad} sharpens image saliency maps and stabilises tabular gradients (next section).",
                   "\\textbf{SmoothGrad} clarifică hărțile de saliență ale imaginilor și stabilizează gradienții tabelari (secțiunea următoare)."),
                 t("\\textbf{TCAV} speaks the language of the business: ``night-time transaction'', ``new merchant'' --- if such concepts are labelled.",
                   "\\textbf{TCAV} vorbește limbajul afacerii: ``tranzacție de noapte'', ``comerciant nou'' --- dacă asemenea concepte sînt etichetate."),
                 t("\\textbf{Our data}: V1--V28 are anonymised components, so concept tests are not possible here.",
                   "\\textbf{Datele noastre}: V1--V28 sînt componente anonimizate, deci aici nu se pot face teste pe concepte."), sep="3pt"),
              t("concept explanations need concepts labelled when the data are collected.",
                "explicațiile pe concepte cer concepte etichetate încă de la colectarea datelor."),
              cite="(Smilkov et al., 2017; Kim et al., 2018)", wl="0.58", wr="0.39")

    D.figframe(t("Four explanation methods for one fraudulent transaction", "Patru metode de explicare aplicate unei tranzacții frauduloase"),
               t("signed attributions of one day-2 fraud (median fraud score), as shares of the total absolute attribution.",
                 "atribuirile cu semn pentru o fraudă din ziua 2 (scorul median al fraudelor), ca ponderi în atribuirea absolută totală."),
               None, "fraud_local", bullets=[
                   t("Top feature: Kernel SHAP " + V.N["onefraud"]["top_shap"] + ", integrated gradients " + V.N["onefraud"]["top_ig"] + ", saliency "
                     + V.N["onefraud"]["top_sal"] + "; fraud score @{onefraud.p|.1f}\\,\\%: saliency is blind to the size of the deviation from the baseline.",
                     "Variabila de top: Kernel SHAP " + V.N["onefraud"]["top_shap"] + ", gradienți integrați " + V.N["onefraud"]["top_ig"] + ", saliență "
                     + V.N["onefraud"]["top_sal"] + "; scorul de fraudă @{onefraud.p|.1f}\\,\\%: saliența nu ține cont de mărimea abaterii față de referință.")],
               tryit=NB + " \\S5", height="0.52\\textheight", cite=OWN)

    # ================================================================== 5. evaluation and stability
    D.section(t("Agreement, faithfulness and stability of explanations", "Concordanța, fidelitatea și stabilitatea explicațiilor"))
    D.figframe(t("Agreement between explanation methods", "Concordanța metodelor de explicare"),
               t("Spearman correlation of the global importances (mean $|$attribution$|$) of seven methods on the same frauds.",
                 "corelația Spearman a importanțelor globale (media $|$atribuirii$|$) pentru șapte metode, pe aceleași fraude."),
               None, "method_agreement", bullets=[
                   t("IG--SHAP @{agree.ig_shap|.2f}, saliency--SHAP @{agree.sal_shap|.2f}, LIME--SHAP @{agree.lime_shap|.2f}, permutation--SHAP @{agree.perm_shap|.2f}; all methods put "
                     + V.N["agree"]["top_ig"] + " first.",
                     "IG--SHAP @{agree.ig_shap|.2f}, saliență--SHAP @{agree.sal_shap|.2f}, LIME--SHAP @{agree.lime_shap|.2f}, permutare--SHAP @{agree.perm_shap|.2f}; toate metodele pun "
                     + V.N["agree"]["top_ig"] + " pe primul loc.")],
               job=t("disagreement is information: permutation importance measures what the model needs, attributions what it uses.",
                     "dezacordul este informație: importanța prin permutare arată de ce are nevoie modelul, atribuirile arată ce folosește."),
               tryit=NB + " \\S6", height="0.56\\textheight", cite="(Breiman, 2001; " + t("own computation", "calcul propriu") + ")")

    D.figframe(t("Faithfulness of explanations: deletion test", "Fidelitatea explicațiilor: testul eliminării variabilelor"),
               t("reset the top-$k$ features of each fraud to the baseline; a faithful explanation makes the score fall fast.",
                 "variabilele de top $k$ ale fiecărei fraude sînt readuse la referință; o explicație fidelă face scorul să scadă repede."),
               "\\[ \\mathrm{AOPC}=\\frac{1}{K}\\sum_{k=1}^{K}\\big[f(\\vx)-f(\\vx^{(k)})\\big] \\]",
               "deletion_curve", bullets=[
                   t("Area over the perturbation curve (AOPC) in points of fraud score: IG @{del.aopc_ig|.1f}, SHAP @{del.aopc_shap|.1f}, LIME @{del.aopc_lime|.1f}, "
                     "saliency @{del.aopc_sal|.1f}, random order @{del.aopc_rand|.1f}.",
                     "Aria de deasupra curbei de perturbare (AOPC), în puncte de scor de fraudă: IG @{del.aopc_ig|.1f}, SHAP @{del.aopc_shap|.1f}, LIME @{del.aopc_lime|.1f}, "
                     "saliență @{del.aopc_sal|.1f}, ordine aleatoare @{del.aopc_rand|.1f}.")],
               tryit=NB + " \\S6", height="0.44\\textheight", cite="(Samek et al., 2017)")

    D.figframe(t("Local robustness of explanations to small input changes", "Robustețea locală a explicațiilor la mici modificări ale intrării"),
               t("relative change of the attribution vector per unit of a small perturbation (Lipschitz-type ratio).",
                 "variația relativă a vectorului de atribuiri pe unitatea unei perturbări mici (raport de tip Lipschitz)."),
               "\\[ \\hat L(\\vx)=\\frac{\\|\\mathbf a(\\vx+\\boldsymbol\\varepsilon)-\\mathbf a(\\vx)\\|/\\|\\mathbf a(\\vx)\\|}{\\|\\boldsymbol\\varepsilon\\|},\\quad \\boldsymbol\\varepsilon\\sim\\mathcal N(\\mathbf 0,0.05^{2}\\mathbf I) \\]",
               "local_robustness", bullets=[
                   t("Medians: saliency @{rob.saliency|.3f}, gradient $\\times$ input @{rob.gradxinput|.3f}, IG @{rob.ig|.3f}, SmoothGrad @{rob.smoothgrad|.3f}; "
                     "within a linear region of a ReLU network the gradient does not move.",
                     "Mediane: saliență @{rob.saliency|.3f}, gradient $\\times$ intrare @{rob.gradxinput|.3f}, IG @{rob.ig|.3f}, SmoothGrad @{rob.smoothgrad|.3f}; "
                     "în interiorul unei regiuni liniare a unei rețele ReLU gradientul nu se schimbă.")],
               tryit=NB + " \\S6", height="0.42\\textheight", cite="(Alvarez-Melis \\& Jaakkola, 2018)")

    D.figframe(t("The Rashomon effect: equally accurate models with different explanations", "Efectul Rashomon: modele la fel de precise, cu explicații diferite"),
               t("rank of each feature by mean $|$SHAP$|$ for 10 retrained networks and 10 boosted ensembles (different seeds only).",
                 "rangul fiecărei variabile după media $|$SHAP$|$ pentru 10 rețele și 10 ansambluri de arbori reantrenate (doar valorile inițiale diferă)."),
               None,
               "seed_stability", bullets=[
                   t("MLP test AUC between @{stab.auc_range_mlp_lo|.3f} and @{stab.auc_range_mlp_hi|.3f}; mean pairwise Kendall $\\tau$ (rank agreement of importances, 1 = identical order): MLP @{stab.tau_mlp|.2f}, XGBoost @{stab.tau_xgb|.2f}; "
                     "same top feature for one applicant in @{stab.local_mlp|.0f}\\,\\% (MLP) and @{stab.local_xgb|.0f}\\,\\% (XGBoost) of seed pairs.",
                     "AUC de test MLP între @{stab.auc_range_mlp_lo|.3f} și @{stab.auc_range_mlp_hi|.3f}; tau Kendall mediu pe perechi (acordul rangurilor importanțelor, 1 = aceeași ordine): MLP @{stab.tau_mlp|.2f}, XGBoost @{stab.tau_xgb|.2f}; "
                     "aceeași variabilă de top pentru un solicitant în @{stab.local_mlp|.0f}\\,\\% (MLP) și @{stab.local_xgb|.0f}\\,\\% (XGBoost) dintre perechile de valori inițiale.")],
               tryit=NB + " \\S7", height="0.42\\textheight", cite=OWN)

    D.figframe(t("Explanations over time: day 1 against day 2", "Explicațiile în timp: ziua 1 față de ziua 2"),
               t("share of the total $|$IG$|$ by feature, for held-out frauds of the training day and for the frauds of the next day.",
                 "ponderea în $|$IG$|$ total pe variabile, pentru fraudele rezervate din ziua de antrenare și pentru fraudele din ziua următoare."),
               None,
               "time_shift", bullets=[
                   t("Spearman @{shift.rho|.2f}, Jensen-Shannon (JS) divergence (built on the Kullback-Leibler (KL) divergence) @{shift.js|.3f}, @{shift.top5_overlap} of the top 5 shared; one day later the explanation is stable here.",
                     "Spearman @{shift.rho|.2f}, divergența Jensen-Shannon (JS) (construită pe divergența Kullback-Leibler (KL)) @{shift.js|.3f}, @{shift.top5_overlap} variabile comune în primele 5; după o zi, explicația este stabilă aici.")],
               job=t("monitor the distribution of attributions monthly, next to the score distribution; a drift in reasons can precede a drift in accuracy.",
                     "monitorizați lunar distribuția atribuirilor, alături de distribuția scorurilor; o deplasare a motivelor poate preceda o scădere a acurateței."),
               tryit=NB + " \\S7", height="0.40\\textheight", cite=OWN)

    D.frame(t("Stability scorecard of this lecture", "Fișa de stabilitate a acestui curs"), table(
        "@{}p{0.3\\textwidth}p{0.32\\textwidth}p{0.3\\textwidth}@{}",
        [t("Source of change", "Sursa schimbării"), t("Measure", "Măsura"), t("Result", "Rezultat")],
        [[t("Explainer randomness (LIME)", "Aleatorismul explicației (LIME)"), t("same top feature over 30 runs", "aceeași variabilă de top în 30 de rulări"),
          t("@{lime.top1_same_500|.0f}\\,\\% (500), @{lime.top1_same_5000|.0f}\\,\\% (5,000 samples)", "@{lime.top1_same_500|.0f}\\,\\% (500), @{lime.top1_same_5000|.0f}\\,\\% (5 000 de eșantioane)")],
         [t("Training seed (Rashomon)", "Valoarea inițială (Rashomon)"), t("Kendall $\\tau$ of global importances", "tau Kendall al importanțelor globale"),
          t("MLP @{stab.tau_mlp|.2f}, XGBoost @{stab.tau_xgb|.2f}", "MLP @{stab.tau_mlp|.2f}, XGBoost @{stab.tau_xgb|.2f}")],
         [t("Training seed, one applicant", "Valoarea inițială, un solicitant"), t("same top feature", "aceeași variabilă de top"),
          t("MLP @{stab.local_mlp|.0f}\\,\\%, XGBoost @{stab.local_xgb|.0f}\\,\\%", "MLP @{stab.local_mlp|.0f}\\,\\%, XGBoost @{stab.local_xgb|.0f}\\,\\%")],
         [t("Method", "Metoda"), t("Spearman of importances", "Spearman al importanțelor"), t("@{agree.min|.2f} -- @{agree.max|.2f}", "@{agree.min|.2f} -- @{agree.max|.2f}")],
         [t("Input noise", "Zgomotul la intrare"), t("median relative change", "variația relativă mediană"), t("IG @{rob.ig|.3f}, SmoothGrad @{rob.smoothgrad|.3f}", "IG @{rob.ig|.3f}, SmoothGrad @{rob.smoothgrad|.3f}")],
         [t("Time (day 1 $\\to$ day 2)", "Timpul (ziua 1 $\\to$ ziua 2)"), t("Spearman, JS divergence", "Spearman, divergența JS"), "@{shift.rho|.2f}; @{shift.js|.3f}"]], size="footnotesize") +
        D.bottom(t("report every explanation with its stability: which seed, which sample, which period, and how much it moves.",
                   "raportați orice explicație împreună cu stabilitatea ei: ce valoare inițială, ce eșantion, ce perioadă și cît se modifică.")),
        "small", cite=OWN)

    # ================================================================== 6. governance
    D.section(t("Explainability in model governance", "Explicabilitatea în guvernanța modelelor"))
    D.frame(t("An explainability audit checklist", "Lista de verificare pentru un audit al explicabilității"), cols(it(
        t("\\textbf{Purpose}: who reads the explanation (customer, analyst, validator, regulator) and what decision it supports.",
          "\\textbf{Scopul}: cine citește explicația (client, analist, validator, autoritate) și ce decizie susține."),
        t("\\textbf{Settings}: background sample, baseline, seed, number of samples, kernel width --- versioned with the model.",
          "\\textbf{Setările}: eșantionul de referință, referința, valoarea inițială, numărul de eșantioane, lățimea nucleului --- versionate odată cu modelul."),
        t("\\textbf{Checks}: additivity/completeness, faithfulness (deletion), stability (seeds, time), agreement of two methods.",
          "\\textbf{Verificările}: aditivitate/completitudine, fidelitate (eliminare), stabilitate (valori inițiale, timp), concordanța a două metode."), sep="4pt"), it(
        t("\\textbf{Fairness}: attributions of protected attributes and of their proxies (age, sex, postcode); an explanation shows use, not causation.",
          "\\textbf{Echitatea}: atribuirile variabilelor protejate și ale substitutelor lor (vîrstă, sex, cod poștal); o explicație arată utilizarea, nu cauzalitatea."),
        t("\\textbf{Customer output}: few reasons, actionable counterfactuals, no raw component names (V14 means nothing to a customer).",
          "\\textbf{Pentru client}: puține motive, contrafactuale acționabile, fără nume brute de componente (V14 nu înseamnă nimic pentru un client)."),
        t("\\textbf{Alternative}: if an interpretable model is almost as accurate (here logistic AUC @{perf.lr_auc|.3f} vs MLP @{perf.mlp_auc|.3f}), consider using it.",
          "\\textbf{Alternativa}: dacă un model interpretabil este aproape la fel de precis (aici AUC logistic @{perf.lr_auc|.3f} față de MLP @{perf.mlp_auc|.3f}), luați-l în considerare."), sep="4pt")) +
        D.bottom(t("the validation report should contain the explanations and their stability tests, not only the accuracy.",
                   "raportul de validare trebuie să conțină explicațiile și testele lor de stabilitate, nu doar acuratețea.")),
        "small", cite="(SR 11-7; Rudin, 2019)")

    # ------------------------------------------------------------------ wrap-up
    D.parts.append("\\section*{" + t("Wrap-up", "Recapitulare") + "}\n")
    D.quiz_frame(t("adding up SHAP values", "adunarea valorilor SHAP"),
                 t("For our applicant the base value is @{app.base|.3f} and the SHAP values sum to @{app.sum_phi|.3f}. What is the model's log-odds of default, and why can you be sure?",
                   "Pentru solicitantul nostru valoarea de bază este @{app.base|.3f}, iar valorile SHAP se adună la @{app.sum_phi|.3f}. Care sînt log-șansele de neplată date de model și de ce sînteți siguri?"),
                 t("@{app.logit|.3f}: by the efficiency axiom $f(\\vx)=\\phi_0+\\sum_j\\phi_j$; the PD is $\\sigma$(@{app.logit|.3f}) = @{app.pd|.1f}\\,\\%.",
                   "@{app.logit|.3f}: din axioma eficienței, $f(\\vx)=\\phi_0+\\sum_j\\phi_j$; PD este $\\sigma$(@{app.logit|.3f}) = @{app.pd|.1f}\\,\\%."))
    D.quiz_frame(t("an unstable LIME", "un LIME instabil"),
                 t("Two analysts explain the same rejected applicant with LIME and report different top reasons. Is the model inconsistent?",
                   "Doi analiști explică același solicitant respins cu LIME și raportează motive principale diferite. Este modelul inconsecvent?"),
                 t("Not necessarily: LIME samples perturbations; with 500 samples the top feature matched in @{lime.top1_same_500|.0f}\\,\\% of our runs; fix the seed, raise the samples, check against SHAP.",
                   "Nu neapărat: LIME eșantionează perturbări; cu 500 de eșantioane variabila de top a coincis în @{lime.top1_same_500|.0f}\\,\\% din rulările noastre; fixați valoarea inițială, creșteți eșantionul, comparați cu SHAP."))
    D.quiz_frame(t("choosing an IG baseline", "alegerea referinței IG"),
                 t("A colleague uses the all-zero vector as the integrated-gradients baseline for standardised fraud features. What question does that explanation answer?",
                   "Un coleg folosește vectorul nul ca referință pentru gradienții integrați, pe variabile de fraudă standardizate. La ce întrebare răspunde explicația?"),
                 t("``Why does this transaction differ from the average transaction (frauds included)?''; our baseline, the mean legitimate transaction, asks why it is not a normal payment.",
                   "``De ce diferă această tranzacție de tranzacția medie (inclusiv fraudele)?''; referința noastră, tranzacția legitimă medie, întreabă de ce nu este o plată normală."))

    D.frame(t("Wrap-up", "Recapitulare"), it(
        t("\\textbf{SHAP}: Shapley values with an exact additivity check; the missing debt-to-income ratio drives the credit network.",
          "\\textbf{SHAP}: valori Shapley cu verificarea exactă a aditivității; lipsa raportului datorie/venit determină rețeaua de credit."),
        t("\\textbf{LIME}: a local slope that depends on seed, sample size and kernel width; useful, but report its settings.",
          "\\textbf{LIME}: o pantă locală care depinde de valoarea inițială, de eșantion și de lățimea nucleului; utilă, dar raportați setările."),
        t("\\textbf{Gradients}: integrated gradients satisfy completeness (@{ig.comp64_pct|.2f}\\,\\% error at 64 steps) and are as faithful as SHAP at a fraction of the cost.",
          "\\textbf{Gradienți}: gradienții integrați satisfac completitudinea (eroare de @{ig.comp64_pct|.2f}\\,\\% la 64 de pași) și sînt la fel de fideli ca SHAP, la un cost mult mai mic."),
        t("\\textbf{Stability}: retraining with another seed reorders features (Kendall $\\tau$ @{stab.tau_mlp|.2f} for the MLP); explanations must be tested like models.",
          "\\textbf{Stabilitatea}: reantrenarea cu altă valoare inițială reordonează variabilele (tau Kendall @{stab.tau_mlp|.2f} pentru MLP); explicațiile trebuie testate ca modelele."),
        t("\\textbf{Next}: Lecture 12, large language models, where explanations become text.",
          "\\textbf{Urmează}: cursul 12, modelele lingvistice mari, unde explicațiile devin text."), sep="4pt"), "small")

    D.frame(t("Research from our group: XAI in fraud and energy markets", "Cercetările grupului nostru: XAI în fraudă și piețele de energie"), it(
        t("\\textbf{SHAP stability in graph-augmented fraud detection}: on the IEEE-CIS fraud data, SHAP contributions of an XGBoost model with "
          "graph neural network (GNN) embeddings are compared under random and forward-time validation, with a Gaussian control of the same dimension and covariance; "
          "whether the graph model looks more stable depends on how the change is measured, a careful protocol for explanation stability "
          "(Diaconu \\& Pele, 2026, submitted to Expert Systems with Applications; code: QuantLet GNN\\_XAI\\_Stability).",
          "\\textbf{Stabilitatea SHAP în detectarea fraudei cu informație din graf}: pe datele de fraudă IEEE-CIS se compară contribuțiile SHAP ale unui model XGBoost cu "
          "reprezentări dintr-o rețea neuronală pe grafuri (GNN), sub validare aleatoare și validare înainte în timp, cu un control gaussian de aceeași dimensiune și covarianță; "
          "dacă modelul cu graf pare mai stabil depinde de modul în care se măsoară schimbarea, un protocol riguros pentru stabilitatea explicațiilor "
          "(Diaconu \\& Pele, 2026, trimis la Expert Systems with Applications; cod: QuantLet GNN\\_XAI\\_Stability)."),
        t("\\textbf{Explainable early warning of electricity price spikes} in Romania (Andrei, Pele, Cramer, Amza \\& Diaconu, 2026, Proceedings of the International Conference "
          "on Business Excellence (PICBE)) and \\textbf{SHAP-driven anomaly detection} of electricity prices (Andrei \\& Pele, 2025, Preprints).",
          "\\textbf{Avertizarea timpurie explicabilă a vîrfurilor de preț} la energia electrică din România (Andrei, Pele, Cramer, Amza \\& Diaconu, 2026, Proceedings of the International Conference "
          "on Business Excellence (PICBE)) și \\textbf{detectarea anomaliilor de preț cu SHAP} (Andrei \\& Pele, 2025, Preprints)."),
        t("\\textbf{Multi-horizon explainability} of price forecasts: which drivers matter at which horizon in Romania's liberalised market (Andrei, Pele \\& Cramer, 2026, PICBE).",
          "\\textbf{Explicabilitatea pe mai multe orizonturi} a prognozelor de preț: ce factori contează la fiecare orizont pe piața liberalizată din România (Andrei, Pele \\& Cramer, 2026, PICBE)."),
        sep="4pt") + "\\par\\vspace{1mm}{\\scriptsize " + items(t("Findings as stated in the papers; numbers on the other slides come from our own data.",
                                                                  "Rezultatele sînt cele din lucrări; cifrele de pe celelalte slide-uri provin din datele noastre."), nest=False) + "}",
        "footnotesize", cite=t("(Bucharest University of Economic Studies (ASE); Romanian Academy, Institute for Economic Forecasting (IPE))",
                               "(Academia de Studii Economice din București (ASE); Academia Română, Institutul de Prognoză Economică (IPE))"))

    D.frame(t("Further reading and resources", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{C.\\ Molnar, Interpretable Machine Learning}, free online: ", "\\textbf{C.\\ Molnar, Interpretable Machine Learning}, gratuit online: ") +
        "\\href{https://christophm.github.io/interpretable-ml-book/}{christophm.github.io/interpretable-ml-book}.",
        t("\\textbf{SHAP library} and examples: ", "\\textbf{Biblioteca SHAP} și exemple: ") + "\\href{https://shap.readthedocs.io}{shap.readthedocs.io}.",
        t("\\textbf{Rudin (2019)}: the case for interpretable models in high-stakes decisions.", "\\textbf{Rudin (2019)}: argumentele pentru modele interpretabile în deciziile cu miză mare."), sep="4pt"),
        it(t("\\textbf{This course}: code, data, notebooks --- ", "\\textbf{Acest curs}: cod, date, notebook-uri --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Experiment yourself}: notebooks/L11\\_notebook.ipynb; the seminar explains a German credit model, with a fairness check.",
             "\\textbf{Încercați singuri}: notebooks/L11\\_notebook\\_ro.ipynb; seminarul explică un model pentru German credit, cu o verificare a echității."),
           t("\\textbf{Replication of our stability study}: ", "\\textbf{Replicarea studiului nostru de stabilitate}: ") +
           "\\href{https://github.com/QuantLet/GNN_XAI_Stability}{github.com/QuantLet/GNN\\_XAI\\_Stability}.", sep="4pt"), "0.5", "0.47"), "small")

    reference_frames(D, REFS, per_frame=8)
    D.mark_main_end()
    D.parts.append("\\section*{" + t("Appendix", "Anexă") + "}\n")
    D.frame(t("Appendix A: recovery of Shapley values by the Shapley kernel", "Anexa A: obținerea valorilor Shapley prin nucleul Shapley"), it(
        t("Restrict $g(\\vz)=\\phi_0+\\sum_j\\phi_jz_j$ to satisfy $g(\\mathbf 0)=v(\\emptyset)$ and $g(\\mathbf 1)=v(\\{1..d\\})$ (infinite weight on the empty and full coalitions).",
          "Se impune ca $g(\\vz)=\\phi_0+\\sum_j\\phi_jz_j$ să satisfacă $g(\\mathbf 0)=v(\\emptyset)$ și $g(\\mathbf 1)=v(\\{1..d\\})$ (pondere infinită pe coaliția vidă și pe cea completă)."),
        t("The weighted least-squares normal equations with $\\pi(\\vz)\\propto(d-1)/[\\binom{d}{|\\vz|}|\\vz|(d-|\\vz|)]$ have the unique solution $\\phi_j$ = Shapley value (Lundberg \\& Lee, 2017, Theorem 2).",
          "Ecuațiile normale ale celor mai mici pătrate ponderate cu $\\pi(\\vz)\\propto(d-1)/[\\binom{d}{|\\vz|}|\\vz|(d-|\\vz|)]$ au soluția unică $\\phi_j$ = valoarea Shapley (Lundberg \\& Lee, 2017, teorema 2)."),
        t("Kernel SHAP samples coalitions from $\\pi$ (pairing $\\vz$ with $\\mathbf 1-\\vz$) and solves the constrained regression; the error falls as $1/\\sqrt{\\text{samples}}$.",
          "Kernel SHAP eșantionează coaliții din $\\pi$ (împerecheind $\\vz$ cu $\\mathbf 1-\\vz$) și rezolvă regresia cu restricții; eroarea scade ca $1/\\sqrt{\\text{eșantioane}}$."),
        t("For a linear $f$ and independent features, $v(S)=\\beta_0+\\sum_{j\\in S}\\beta_jx_j+\\sum_{j\\notin S}\\beta_j\\E x_j$, so $\\phi_j=\\beta_j(x_j-\\E x_j)$.",
          "Pentru $f$ liniar și variabile independente, $v(S)=\\beta_0+\\sum_{j\\in S}\\beta_jx_j+\\sum_{j\\notin S}\\beta_j\\E x_j$, deci $\\phi_j=\\beta_j(x_j-\\E x_j)$."), sep="3pt"),
        "small", cite="(Lundberg \\& Lee, 2017)")
    D.frame(t("Appendix B: integrated gradients as a path method", "Anexa B: gradienții integrați ca metodă de integrare pe traiectorie"), it(
        t("For a path $\\gamma(\\alpha)$ from $\\vx'$ to $\\vx$: $f(\\vx)-f(\\vx')=\\int_0^1\\nabla f(\\gamma(\\alpha))^{\\top}\\gamma'(\\alpha)\\,d\\alpha=\\sum_j\\int_0^1\\partial_jf(\\gamma)\\,\\gamma_j'\\,d\\alpha$.",
          "Pentru un drum $\\gamma(\\alpha)$ de la $\\vx'$ la $\\vx$: $f(\\vx)-f(\\vx')=\\int_0^1\\nabla f(\\gamma(\\alpha))^{\\top}\\gamma'(\\alpha)\\,d\\alpha=\\sum_j\\int_0^1\\partial_jf(\\gamma)\\,\\gamma_j'\\,d\\alpha$."),
        t("The straight line $\\gamma(\\alpha)=\\vx'+\\alpha(\\vx-\\vx')$ gives $\\gamma_j'=x_j-x'_j$: each term is $\\mathrm{IG}_j$, hence completeness.",
          "Dreapta $\\gamma(\\alpha)=\\vx'+\\alpha(\\vx-\\vx')$ dă $\\gamma_j'=x_j-x'_j$: fiecare termen este $\\mathrm{IG}_j$, de unde completitudinea."),
        t("Among path methods, the straight line is the only one that is symmetric in features (Sundararajan et al., 2017).",
          "Dintre metodele pe drum, dreapta este singura simetrică în raport cu variabilele (Sundararajan et al., 2017)."),
        t("Midpoint Riemann sum error is $O(m^{-2})$ for smooth $f$; ReLU kinks make it slower, as our completeness chart shows at small $m$.",
          "Eroarea sumei Riemann cu punctul de mijloc este $O(m^{-2})$ pentru $f$ neted; punctele de frîngere ReLU o încetinesc, cum arată graficul de completitudine pentru $m$ mic."), sep="3pt"),
        "small", cite="(Sundararajan, Taly \\& Yan, 2017)")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 11, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
