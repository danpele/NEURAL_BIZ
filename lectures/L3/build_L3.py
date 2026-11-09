r"""
build_L3.py -- Lecture 3 "Regularisation and generalisation" (EN + RO).
Writes lecture3.tex / lecture3_ro.tex from the text below and the numbers in L3_numbers.json (analysis_L3.py),
compiles both and runs every check of lectures/STYLE.md (nn_build.run_build).
Run:  python3 lectures/L3/build_L3.py   [en|ro]
Rules: no computed number is typed by hand; hand-typed numbers are definitions and a-priori choices (grids of lambda and p,
patience 20, review cost 5 EUR per alert, 60/20/20 splits, seeds).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_build  # noqa: E402
from nn_build import Deck, Numbers, abbreviation_frames, block, cols, items, reference_frames, run_build, table  # noqa: E402

_bw = nn_build.bulletize_where


def _compact_where(tex, per_line=3):
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
    "AdamW": ("Adam with decoupled weight decay", "Adam cu penalizarea ponderilor decuplată"),
    "AUC": ("area under the receiver operating characteristic curve", "aria de sub curba caracteristică de operare a receptorului"),
    "BN": ("batch normalisation", "normalizarea pe loturi (batch normalisation)"),
    "EUR": ("euro", "euro"),
    "GNN": ("graph neural network", "rețea neuronală pe grafuri"),
    "GraphSAGE": ("graph sample and aggregate (a GNN architecture)", "eșantionare și agregare pe graf (o arhitectură GNN)"),
    "IEEE-CIS": ("Institute of Electrical and Electronics Engineers, Computational Intelligence Society",
                 "Institute of Electrical and Electronics Engineers, Computational Intelligence Society"),
    "MAP": ("maximum a posteriori", "maximum a posteriori"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "MSE": ("mean squared error", "eroarea pătratică medie"),
    "OpenML": ("Open Machine Learning platform", "platforma Open Machine Learning"),
    "PCA": ("principal component analysis", "analiza componentelor principale"),
    "PR-AUC": ("area under the precision-recall curve", "aria de sub curba precizie-sensibilitate"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată"),
    "ROC": ("receiver operating characteristic", "caracteristica de operare a receptorului"),
    "SGD": ("stochastic gradient descent", "coborîrea stohastică pe gradient"),
    "SHAP": ("Shapley additive explanations", "explicații aditive Shapley"),
    "UCI": ("University of California, Irvine (Machine Learning Repository)", "Universitatea California din Irvine (Machine Learning Repository)"),
    "ULB": ("Université libre de Bruxelles", "Université libre de Bruxelles"),
    "XAI": ("explainable artificial intelligence", "inteligență artificială explicabilă"),
    "XGBoost": ("extreme gradient boosting", "extreme gradient boosting (arbori de decizie cu amplificare pe gradient)"),
}
UNITS = {"EUR"}
IGNORE = {"BF", "BIZ", "C5NC", "NEURAL", "US", "XG", "ACM", "PLOS", "ONE", "PNAS", "JMLR", "ICML", "LNCS", "NAJEF", "PICBE", "FN", "FP",
          "TP", "TN", "IEEE", "SSCI", "KDD", "ESL", "II", "PR", "PLOS ONE", "CV"}

REFS = [
    "Będowska-Sójka, B., Wójcik, P.\\ \\& Pele, D.\\,T.\\ (2026). Early warning systems for cryptocurrency markets: Predicting `zombie' "
    "assets using machine learning. \\textit{The North American Journal of Economics and Finance}, 81, 102543. "
    "\\href{https://doi.org/10.1016/j.najef.2025.102543}{doi:10.1016/j.najef.2025.102543}",
    "Belkin, M., Hsu, D., Ma, S.\\ \\& Mandal, S.\\ (2019). Reconciling modern machine-learning practice and the classical bias--variance "
    "trade-off. \\textit{Proceedings of the National Academy of Sciences}, 116(32), 15849--15854. \\href{https://doi.org/10.1073/pnas.1903070116}{doi:10.1073/pnas.1903070116}",
    "Bergmeir, C.\\ \\& Benítez, J.\\,M.\\ (2012). On the use of cross-validation for time series predictor evaluation. \\textit{Information "
    "Sciences}, 191, 192--213. \\href{https://doi.org/10.1016/j.ins.2011.12.028}{doi:10.1016/j.ins.2011.12.028}",
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}. Springer. "
    "\\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Chen, T.\\ \\& Guestrin, C.\\ (2016). XGBoost: A scalable tree boosting system. \\textit{Proceedings of KDD 2016}, 785--794. "
    "\\href{https://doi.org/10.1145/2939672.2939785}{doi:10.1145/2939672.2939785}",
    "Dal Pozzolo, A., Caelen, O., Johnson, R.\\,A.\\ \\& Bontempi, G.\\ (2015). Calibrating probability with undersampling for unbalanced "
    "classification. \\textit{IEEE Symposium Series on Computational Intelligence}, 159--166. \\href{https://doi.org/10.1109/SSCI.2015.33}{doi:10.1109/SSCI.2015.33}",
    "Diaconu, D.\\ \\& Pele, D.\\,T.\\ (2026). Explainable AI in financial fraud detection: Evidence from the IEEE-CIS fraud dataset. "
    "\\textit{Proceedings of the International Conference on Business Excellence}, 20(1), 118--132. "
    "\\href{https://doi.org/10.2478/picbe-2026-0015}{doi:10.2478/picbe-2026-0015}",
    "Hastie, T., Tibshirani, R.\\ \\& Friedman, J.\\ (2009). \\textit{The Elements of Statistical Learning} (2nd ed.). Springer. "
    "\\href{https://doi.org/10.1007/978-0-387-84858-7}{doi:10.1007/978-0-387-84858-7}",
    "Hoerl, A.\\,E.\\ \\& Kennard, R.\\,W.\\ (1970). Ridge regression: Biased estimation for nonorthogonal problems. \\textit{Technometrics}, "
    "12(1), 55--67. \\href{https://doi.org/10.1080/00401706.1970.10488634}{doi:10.1080/00401706.1970.10488634}",
    "Hofmann, H.\\ (1994). Statlog (German Credit Data). UCI Machine Learning Repository. \\href{https://doi.org/10.24432/C5NC77}{doi:10.24432/C5NC77}",
    "Ioffe, S.\\ \\& Szegedy, C.\\ (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. "
    "\\textit{Proceedings of ICML}, 448--456. \\href{https://arxiv.org/abs/1502.03167}{arXiv:1502.03167}",
    "Kaufman, S., Rosset, S., Perlich, C.\\ \\& Stitelman, O.\\ (2012). Leakage in data mining: Formulation, detection, and avoidance. "
    "\\textit{ACM Transactions on Knowledge Discovery from Data}, 6(4), 1--21. \\href{https://doi.org/10.1145/2382577.2382579}{doi:10.1145/2382577.2382579}",
    "Loshchilov, I.\\ \\& Hutter, F.\\ (2019). Decoupled weight decay regularization. \\textit{International Conference on Learning "
    "Representations}. \\href{https://arxiv.org/abs/1711.05101}{arXiv:1711.05101}",
    "Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B.\\ \\& Sutskever, I.\\ (2021). Deep double descent: where bigger models and more "
    "data hurt. \\textit{Journal of Statistical Mechanics}, 124003. \\href{https://doi.org/10.1088/1742-5468/ac3a74}{doi:10.1088/1742-5468/ac3a74}",
    "Pace, R.\\,K.\\ \\& Barry, R.\\ (1997). Sparse spatial autoregressions. \\textit{Statistics \\& Probability Letters}, 33(3), "
    "291--297. \\href{https://doi.org/10.1016/S0167-7152(96)00140-X}{doi:10.1016/S0167-7152(96)00140-X}",
    "Prechelt, L.\\ (1998). Early stopping --- but when? In \\textit{Neural Networks: Tricks of the Trade}, LNCS 1524, 55--69. Springer. "
    "\\href{https://doi.org/10.1007/3-540-49430-8_3}{doi:10.1007/3-540-49430-8\\_3}",
    "Saito, T.\\ \\& Rehmsmeier, M.\\ (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary "
    "classifiers on imbalanced datasets. \\textit{PLOS ONE}, 10(3), e0118432. \\href{https://doi.org/10.1371/journal.pone.0118432}{doi:10.1371/journal.pone.0118432}",
    "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I.\\ \\& Salakhutdinov, R.\\ (2014). Dropout: A simple way to prevent neural "
    "networks from overfitting. \\textit{Journal of Machine Learning Research}, 15, 1929--1958.",
    "Tibshirani, R.\\ (1996). Regression shrinkage and selection via the lasso. \\textit{Journal of the Royal Statistical Society B}, "
    "58(1), 267--288. \\href{https://doi.org/10.1111/j.2517-6161.1996.tb02080.x}{doi:10.1111/j.2517-6161.1996.tb02080.x}",
]


def it(*xs, sep="2pt"):
    return items(*xs, opt=f"\\setlength{{\\itemsep}}{{{sep}}}")


def note(*xs):
    return "\n\\par\\vspace{0.5mm}{\\scriptsize " + items(*xs, opt="\\setlength{\\itemsep}{0pt}", nest=False) + "}"


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L3_numbers.json"), lang)
    N = V.N
    ro = lang == "ro"
    D = Deck(3, "Regularisation and generalisation" if not ro else "Regularizare și generalizare", V, lang, heights=H)
    t = D.t
    NB = "notebook"
    OWN = t("(own computation)", "(calcul propriu)")
    BL, BP = N["cv"]["best_lam_key"], N["cv"]["best_p_key"]

    def sci(key, digits=1):
        v = float(V.get(key))
        m, e = f"{v:.{digits}e}".split("e")
        m = m.replace(".", "{,}") if ro else m
        return f"${m}\\times10^{{{int(e)}}}$"

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Diagnose overfitting from training and validation curves, and explain the bias-variance trade-off and double descent.",
          "Să diagnosticați supraînvățarea (overfitting) din curbele de antrenare și validare și să explicați compromisul deplasare-varianță și dubla coborîre."),
        t("Write weight decay, L1 penalties, dropout, early stopping and batch normalisation with their formulas, and say what each one controls.",
          "Să scrieți penalizarea ponderilor, penalizarea L1, dropout, oprirea timpurie și normalizarea pe loturi cu formulele lor și să spuneți ce controlează fiecare."),
        t("Design a validation scheme without leakage: hold-out, cross-validation, time-ordered splits, and resampling only inside the training set.",
          "Să proiectați o schemă de validare fără scurgere de informație: setul păstrat deoparte, validarea încrucișată, împărțirea cronologică și reeșantionarea doar în setul de antrenare."),
        t("Evaluate a fraud detector on imbalanced data with precision, recall, the area under the receiver operating characteristic (ROC) curve (AUC), "
          "the area under the precision-recall curve (PR-AUC) and the cost of errors.",
          "Să evaluați un detector de fraudă pe date dezechilibrate cu precizia, sensibilitatea, aria de sub curbă (AUC) a caracteristicii de operare a "
          "receptorului (ROC), aria de sub curba precizie-sensibilitate (PR-AUC) și costul erorilor."),
        sep="4pt"))
    abbreviation_frames(D, ABBR)

    # ================================================================== 1. generalisation
    D.section(t("Generalisation and overfitting", "Generalizarea și supraînvățarea"))
    D.frame(t("A model is judged on applicants it has never seen", "Un model este judecat pe solicitanți pe care nu i-a văzut niciodată"),
            "\\intuition{" + t("the training error is optimistic by construction; the generalisation gap is what regularisation tries to close.",
                               "eroarea de antrenare este optimistă prin construcție; diferența de generalizare este ceea ce regularizarea încearcă să reducă.") + "}\n" +
            "\\[ \\underbrace{R(\\hat f)-\\hat R_n(\\hat f)}_{\\text{" + t("generalisation gap", "diferența de generalizare") + "}},\\qquad "
            "\\hat f=\\argmin_{f\\in\\mathcal F}\\hat R_n(f)+\\lambda\\,\\Omega(f) \\]\n" +
            "\\where{" + t("$R$ the expected loss on new data; $\\hat R_n$ the training loss; $\\mathcal F$ the model class; $\\Omega$ a penalty on complexity, $\\lambda\\ge0$ its weight",
                           "$R$ pierderea așteptată pe date noi; $\\hat R_n$ pierderea de antrenare; $\\mathcal F$ clasa de modele; $\\Omega$ o penalizare a complexității, $\\lambda\\ge0$ ponderea ei") + "}\n" +
            it(t("Regularisation = anything that lowers $R$ at the price of a higher $\\hat R_n$: penalties, noise, early stopping, more data.",
                 "Regularizarea = orice reduce $R$ cu prețul unei $\\hat R_n$ mai mari: penalizări, zgomot, oprire timpurie, mai multe date."),
               t("Data: German credit from the University of California, Irvine (UCI) repository, California housing, and @{fraud.n|,d} card payments "
                 "of the Université libre de Bruxelles (ULB) with @{fraud.frauds} frauds; models: multilayer perceptrons (MLP).",
                 "Date: German credit, din colecția Universității California din Irvine (UCI), California housing și @{fraud.n|,d} de plăți cu cardul de la "
                 "Université libre de Bruxelles (ULB), cu @{fraud.frauds} de fraude; modele: perceptroni multistrat (MLP).")) +
            D.bottom(t("always report the validation loss next to the training loss.", "raportați întotdeauna pierderea de validare alături de cea de antrenare.")),
            cite="(Bishop \\& Bishop, 2024, ch.~9)")
    D.figframe(t("Without regularisation, a large MLP memorises 600 applicants in a few epochs",
                 "Fără regularizare, un MLP mare memorează 600 de solicitanți în cîteva epoci"),
               t("MLP 256-256 (@{overfit.params|,d} parameters) on German credit; dotted line: best validation epoch.",
                 "MLP 256-256 (@{overfit.params|,d} de parametri) pe German credit; linia punctată: epoca cea mai bună pe validare."),
               None, "overfit_german",
               bullets=[t("Validation loss @{overfit.val_min|.3f} at epoch @{overfit.best_ep}, @{overfit.val_last|.3f} after @{overfit.epochs}; with dropout and decay "
                          "@{overfit.reg_val_last|.3f}; logistic @{overfit.logit_val|.3f}.",
                          "Pierderea de validare @{overfit.val_min|.3f} la epoca @{overfit.best_ep}, @{overfit.val_last|.3f} după @{overfit.epochs}; cu dropout și penalizare "
                          "@{overfit.reg_val_last|.3f}; logistic @{overfit.logit_val|.3f}.")],
               job=t("a training loss near zero on credit data is a warning, not a success.",
                     "o pierdere de antrenare aproape de zero pe date de credit este un semnal de alarmă."),
               tryit=NB + " \\S1", cite=OWN, height="0.60\\textheight")
    D.frame(t("The test error splits into bias, variance and noise", "Eroarea pe test se descompune în deplasare, varianță și zgomot"),
            "\\intuition{" + t("a rigid model misses the pattern (bias), a flexible one chases the noise of its sample (variance).",
                               "un model rigid ratează tiparul (deplasare), unul flexibil urmărește zgomotul eșantionului său (varianță).") + "}\n" +
            "\\[ \\E_{\\mathcal D,\\varepsilon}\\big[(y-\\hat f_{\\mathcal D}(x))^2\\big]=\\underbrace{\\big(f(x)-\\E_{\\mathcal D}\\hat f_{\\mathcal D}(x)\\big)^2}_{\\text{" +
            t("bias", "deplasare") + "}^2}+\\underbrace{\\Var_{\\mathcal D}\\big(\\hat f_{\\mathcal D}(x)\\big)}_{\\text{" + t("variance", "varianță") +
            "}}+\\underbrace{\\sigma^2}_{\\text{" + t("noise", "zgomot") + "}} \\]\n" +
            "\\where{" + t("$y=f(x)+\\varepsilon$, $\\Var(\\varepsilon)=\\sigma^2$; $\\hat f_{\\mathcal D}$ the model fitted on a random training sample $\\mathcal D$; derivation in the appendix",
                           "$y=f(x)+\\varepsilon$, $\\Var(\\varepsilon)=\\sigma^2$; $\\hat f_{\\mathcal D}$ modelul estimat pe un eșantion aleator de antrenare $\\mathcal D$; derivarea în anexă") + "}\n" +
            it(t("Regularisation trades a little bias for a large cut in variance; the noise $\\sigma^2$ is a floor no model can beat.",
                 "Regularizarea acceptă puțină deplasare în schimbul unei reduceri mari a varianței; zgomotul $\\sigma^2$ este o limită pe care niciun model nu o poate depăși."),
               t("Classical U-curve: test error first falls, then rises with capacity; very large networks can show a second descent (next frame).",
                 "Curba clasică în U: eroarea pe test întîi scade, apoi crește cu capacitatea; rețelele foarte mari pot avea o a doua coborîre (slide-ul următor).")) +
            D.bottom(t("with few observations variance dominates: prefer simple models or strong regularisation.",
                       "cu puține observații domină varianța: preferați modele simple sau regularizare puternică.")),
            cite="(Hastie, Tibshirani \\& Friedman, 2009, ch.~7)")
    D.figframe(t("Past the interpolation threshold, more hidden units lower the test error again",
                 "După pragul de interpolare, mai mulți neuroni ascunși reduc din nou eroarea"),
               t("$p$ random, frozen rectified linear units (ReLU), minimum-norm output, $n=$ @{dd.n} block groups.",
                 "$p$ unități liniare rectificate (ReLU) aleatoare și fixe, ieșire de normă minimă, $n=$ @{dd.n} de grupuri."),
               None, "double_descent",
               bullets=[t("Test mean squared error (MSE): @{dd.best_under_mse|.3f} at $p=$ @{dd.best_under_p}, " + sci("dd.peak_mse") + " at $p=n$, "
                          "@{dd.big_mse|.3f} at $p=$ @{dd.big_p|,d} (median of @{dd.seeds} draws).",
                          "Eroarea pătratică medie (MSE) pe test: @{dd.best_under_mse|.3f} la $p=$ @{dd.best_under_p}, " + sci("dd.peak_mse") + " la $p=n$, "
                          "@{dd.big_mse|.3f} la $p=$ @{dd.big_p|,d} (mediana a @{dd.seeds} extrageri).")],
               job=t("near $p=n$ a bigger model can be far worse, far past it size helps again.",
                     "lîngă $p=n$, un model mai mare poate fi mult mai slab, mult după, mărimea ajută din nou."),
               tryit=NB + " \\S2", cite="(Belkin et al., 2019; Nakkiran et al., 2021)", height="0.60\\textheight")
    # ================================================================== 2. penalties
    D.section(t("Penalties on the weights", "Penalizarea ponderilor"))
    D.frame(t("Weight decay as an L2 penalty and as a Gaussian prior on the weights", "Penalizarea ponderilor ca penalizare L2 și ca distribuție a priori gaussiană a ponderilor"),
            "\\intuition{" + t("large weights make the function steep and sensitive to noise; shrinking them makes it smoother.",
                               "ponderile mari fac funcția abruptă și sensibilă la zgomot; micșorarea lor o face mai netedă.") + "}\n" +
            "\\[ \\tilde\\loss(\\vw)=\\loss(\\vw)+\\frac\\lambda2\\|\\vw\\|_2^2\\ \\Rightarrow\\ \\vw_{t+1}=(1-\\eta\\lambda)\\,\\vw_t-\\eta\\nabla\\loss(\\vw_t);\\qquad "
            "\\vw\\sim\\mathcal N(\\mathbf 0,\\lambda^{-1}\\mathbf I)\\Rightarrow\\hat\\vw_{\\mathrm{MAP}}=\\argmin\\tilde\\loss \\]\n" +
            "\\where{" + t("$\\lambda>0$ the penalty strength; $(1-\\eta\\lambda)$ the decay per stochastic gradient descent (SGD) step; maximum a posteriori (MAP) estimate",
                           "$\\lambda>0$ intensitatea penalizării; $(1-\\eta\\lambda)$ micșorarea la fiecare pas de coborîre stohastică pe gradient (SGD); estimarea maximum a posteriori (MAP)") + "}\n" +
            it(t("In a linear model this is ridge regression: along eigen-direction $k$, coefficients shrink by $\\frac{d_k}{d_k+\\lambda}$ (Hoerl \\& Kennard, 1970).",
                 "În modelul liniar este regresia ridge: pe direcția proprie $k$, coeficienții sînt micșorați cu $\\frac{d_k}{d_k+\\lambda}$ (Hoerl \\& Kennard, 1970)."),
               t("With Adam, an L2 term in the loss is rescaled per weight; Adam with decoupled weight decay (AdamW) shrinks every weight by the same factor (Loshchilov \\& Hutter, 2019).",
                 "Cu Adam, termenul L2 este rescalat pe fiecare pondere; Adam cu penalizare decuplată (AdamW) micșorează toate ponderile la fel (Loshchilov \\& Hutter, 2019)."),
               "") +
            D.bottom(t("weight decay is the first regulariser to tune; search $\\lambda$ on a log grid with validation.",
                       "penalizarea ponderilor este primul regularizator de ajustat; căutați $\\lambda$ pe o grilă logaritmică, cu validare."), NB + " \\S3"),
            cite="(Bishop \\& Bishop, 2024, ch.~9)")
    D.frame(t("An L1 penalty sets weights exactly to zero", "O penalizare L1 aduce ponderile exact la zero"),
            "\\intuition{" + t("the L1 ball has corners on the axes, so the penalised optimum often lands on one: sparsity.",
                               "bila L1 are colțuri pe axe, deci optimul penalizat cade adesea într-unul: rezultă un model rar (sparse).") + "}\n" +
            "\\[ \\tilde\\loss(\\vw)=\\loss(\\vw)+\\lambda\\|\\vw\\|_1,\\qquad \\text{" + t("proximal step", "pasul proximal") + ": }\\ "
            "w_j\\leftarrow\\operatorname{sign}(u_j)\\max\\big(|u_j|-\\eta\\lambda,0\\big),\\ u_j=w_j-\\eta\\,\\partial_j\\loss \\]\n" +
            "\\where{" + t("$\\|\\vw\\|_1=\\sum_j|w_j|$; the soft-thresholding step sets small weights exactly to zero (the lasso of Tibshirani, 1996)",
                           "$\\|\\vw\\|_1=\\sum_j|w_j|$; pasul de prag moale aduce ponderile mici exact la zero (lasso, Tibshirani, 1996)") + "}\n" +
            it(t("L1 on the first layer of an MLP acts as variable selection: an input whose outgoing weights are all zero is switched off.",
                 "L1 pe primul strat al unui MLP acționează ca selecție de variabile: o intrare ale cărei ponderi spre stratul ascuns sînt toate zero este oprită."),
               t("Elastic net mixes both: $\\lambda_1\\|\\vw\\|_1+\\tfrac{\\lambda_2}2\\|\\vw\\|_2^2$.", "Elastic net le combină: $\\lambda_1\\|\\vw\\|_1+\\tfrac{\\lambda_2}2\\|\\vw\\|_2^2$.")) +
            D.bottom(t("use L1 when a short list of inputs is needed for the model documentation.",
                       "folosiți L1 cînd documentația modelului cere o listă scurtă de intrări."), NB + " \\S3"),
            cite="(Tibshirani, 1996)")
    D.figframe(t("First-layer weights under L1 and L2 penalties", "Ponderile primului strat cu penalizare L1 și cu penalizare L2"),
               t("MLP with 64 hidden units on German credit, full-batch Adam, penalty on the first layer only.",
                 "MLP cu 64 de neuroni ascunși pe German credit, Adam pe tot setul, penalizare doar pe primul strat."),
               None, "l1_l2_weights",
               bullets=[t("Weights with $|w|<$ @{l1l2.thr|g}: @{l1l2.share_small_l1_pct|.1f}\\,\\% under L1 against @{l1l2.share_small_l2_pct|.1f}\\,\\% under L2 "
                          "(@{l1l2.n_weights|,d} weights); test AUC @{l1l2.auc_l1|.3f} with L1 against @{l1l2.auc_l2|.3f} with L2.",
                          "Ponderi cu $|w|<$ @{l1l2.thr|g}: @{l1l2.share_small_l1_pct|.1f}\\,\\% cu L1 față de @{l1l2.share_small_l2_pct|.1f}\\,\\% cu L2 "
                          "(@{l1l2.n_weights|,d} de ponderi); AUC pe test @{l1l2.auc_l1|.3f} cu L1, față de @{l1l2.auc_l2|.3f} cu L2.")],
               job=t("a stronger L1 penalty would zero whole input columns, which can then leave the data pipeline.",
                     "o penalizare L1 mai puternică ar anula coloane întregi de intrări, care pot ieși apoi din fluxul de date."),
               tryit=NB + " \\S3", cite=OWN, height="0.56\\textheight")

    # ================================================================== 3. dropout, early stopping, batch norm
    D.section(t("Dropout, early stopping and batch normalisation", "Dropout, oprirea timpurie și normalizarea pe loturi"))
    D.frame(t("Dropout trains an ensemble of thinned networks that share their weights", "Dropout antrenează un ansamblu de rețele subțiate care își împart ponderile"),
            "\\intuition{" + t("randomly silencing units forbids any unit to rely on a few others: features must be useful on their own.",
                               "oprirea aleatoare a unor neuroni împiedică orice neuron să depindă de cîțiva alții: fiecare variabilă trebuie să fie utilă singură.") + "}\n" +
            "\\[ \\text{" + t("training", "antrenare") + ": }\\ \\tilde h_j=\\frac{m_j}{1-p}\\,h_j,\\ \\ m_j\\iid\\text{Bernoulli}(1-p);\\qquad \\text{" +
            t("test", "test") + ": }\\ \\tilde h_j=h_j \\]\n" +
            "\\where{" + t("$p$ the dropout rate; $m_j$ the random mask; dividing by $1-p$ (inverted dropout) keeps $\\E[\\tilde h_j]=h_j$, so nothing changes at test time",
                           "$p$ rata de dropout; $m_j$ masca aleatoare; împărțirea la $1-p$ (dropout inversat) păstrează $\\E[\\tilde h_j]=h_j$, deci nimic nu se schimbă la testare") + "}\n" +
            it(t("A network with $H$ droppable units samples from $2^H$ sub-networks; at test time the full network approximates their average (Srivastava et al., 2014).",
                 "Cu $H$ neuroni care pot fi opriți, eșantionăm din $2^H$ subrețele; la testare, rețeaua completă aproximează media lor (Srivastava et al., 2014)."),
               t("Typical rates: 0.1--0.3 in a tabular MLP, 0.5 in large fully connected layers; in PyTorch \\texttt{model.eval()} switches it off.",
                 "Rate uzuale: 0,1--0,3 în MLP tabelare, 0,5 în straturile mari; în PyTorch, \\texttt{model.eval()} îl dezactivează.")) +
            D.bottom(t("forgetting \\texttt{model.eval()} before scoring makes every prediction random.",
                       "dacă uitați \\texttt{model.eval()} înainte de scorare, fiecare predicție devine aleatoare."), NB + " \\S4"),
            cite="(Srivastava et al., 2014)")
    D.figframe(t("On 600 applicants, regularisation does not close the gap to logistic regression",
                 "Pe 600 de solicitanți, regularizarea nu egalează regresia logistică"),
               t("MLP 256-256, 150 epochs without early stopping; held-out AUC over 5 folds for each $\\lambda$ and $p$.",
                 "MLP 256-256, 150 de epoci fără oprire timpurie; AUC pe pliul de test, 5 pliuri, pentru fiecare $\\lambda$ și $p$."),
               None, "wd_dropout",
               bullets=[t("AUC @{cv.auc_none|.3f} unregularised, @{cv.auc_best_lam|.3f} at $\\lambda=$ @{cv.best_lam|g}, @{cv.auc_best_p|.3f} at $p=$ @{cv.best_p|g}; "
                          "logistic @{cv.auc_logit|.3f}; weight norm @{wd.0.norm|.1f} vs @{wd.3.norm|.1f} at $\\lambda=3$.",
                          "AUC @{cv.auc_none|.3f} fără regularizare, @{cv.auc_best_lam|.3f} la $\\lambda=$ @{cv.best_lam|g}, @{cv.auc_best_p|.3f} la $p=$ @{cv.best_p|g}; "
                          "logistic @{cv.auc_logit|.3f}; norma ponderilor @{wd.0.norm|.1f} față de @{wd.3.norm|.1f} la $\\lambda=3$.")],
               job=t("on very small data, a smaller model beats a large regularised one.",
                     "pe date foarte puține, un model mai mic depășește unul mare regularizat."),
               tryit=NB + " \\S4", cite=OWN, height="0.60\\textheight")
    D.frame(t("Early stopping: keep the weights of the best validation epoch", "Oprirea timpurie: păstrăm ponderile epocii celei mai bune pe validare"),
            "\\intuition{" + t("training longer fits more noise; stop when the validation loss has not improved for a while.",
                               "o antrenare mai lungă potrivește mai mult zgomot; ne oprim cînd pierderea de validare nu s-a mai îmbunătățit de ceva timp.") + "}\n" +
            "\\[ t^*=\\argmin_{t\\le T}\\ \\loss_{\\text{val}}(\\vw_t),\\quad \\text{" + t("stop when", "oprire cînd") + "}\\ t-t^*\\ge P;\\qquad "
            "\\text{" + t("quadratic loss", "pierdere pătratică") + ":}\\ \\ \\eta\\,t\\ \\approx\\ \\frac1\\lambda \\]\n" +
            "\\where{" + t("$P$ the patience in epochs; the last relation says that $t$ steps of size $\\eta$ act like an L2 penalty of strength $\\lambda\\approx1/(\\eta t)$ (appendix)",
                           "$P$ răbdarea, în epoci; ultima relație spune că $t$ pași de mărime $\\eta$ acționează ca o penalizare L2 de intensitate $\\lambda\\approx1/(\\eta t)$ (anexa)") + "}\n" +
            it(t("The validation set is used for the stopping decision: it is no longer an unbiased estimate, so a separate test set is needed.",
                 "Setul de validare este folosit pentru decizia de oprire: nu mai este o estimare nedeplasată, deci este nevoie de un set de test separat."),
               t("Free, cheap and compatible with every other regulariser (Prechelt, 1998).", "Este gratuită, ieftină și compatibilă cu orice alt regularizator (Prechelt, 1998).")) +
            D.bottom(t("save the best checkpoint, not the last one.", "salvați cea mai bună stare a modelului, nu ultima."), NB + " \\S5"),
            cite="(Prechelt, 1998; Bishop \\& Bishop, 2024, ch.~9)")
    D.figframe(t("On card fraud, validation loss bottoms out early while training loss keeps falling",
                 "La fraudele cu carduri, pierderea pe validare atinge minimul devreme, în timp ce pierderea pe antrenare continuă să scadă"),
               t("MLP 256-256, time-ordered split; dashed: best validation epoch; grey: the patience window.",
                 "MLP 256-256, împărțire cronologică; punctat: epoca cea mai bună pe validare; gri: fereastra de răbdare."),
               None, "early_stopping",
               bullets=[t("Best epoch @{es.best_ep}, stop at @{es.stop_ep} (patience @{es.patience}); validation loss @{es.val_min|.4f} there, @{es.val_last|.4f} after @{es.epochs} epochs.",
                          "Epoca cea mai bună @{es.best_ep}, oprire la @{es.stop_ep} (răbdare @{es.patience}); pierderea de validare @{es.val_min|.4f} atunci, @{es.val_last|.4f} după @{es.epochs} de epoci.")],
               job=t("with rare events the network starts memorising the few training frauds within a handful of epochs.",
                     "la evenimente rare, rețeaua începe să memoreze puținele fraude de antrenare după cîteva epoci."),
               tryit=NB + " \\S5", cite=OWN, height="0.60\\textheight")
    D.frame(t("Batch normalisation standardises each unit over the mini-batch", "Normalizarea pe loturi standardizează fiecare neuron pe mini-lot"),
            "\\intuition{" + t("keeping pre-activations centred and scaled lets deep networks train with larger steps.",
                               "păstrarea preactivărilor centrate și scalate permite rețelelor adînci să se antreneze cu pași mai mari.") + "}\n" +
            "\\[ \\hat z_{ij}=\\frac{z_{ij}-\\mu_{\\mathcal B,j}}{\\sqrt{\\sigma^2_{\\mathcal B,j}+\\epsilon}},\\qquad y_{ij}=\\gamma_j\\hat z_{ij}+\\beta_j,\\qquad "
            "\\mu_{\\mathcal B,j}=\\frac1B\\sum_{i\\in\\mathcal B}z_{ij} \\]\n" +
            "\\where{" + t("$\\sigma^2_{\\mathcal B,j}$ the batch variance of unit $j$; $\\gamma_j,\\beta_j$ learned scale and shift; at test time running averages replace the batch statistics",
                           "$\\sigma^2_{\\mathcal B,j}$ varianța pe lot a neuronului $j$; $\\gamma_j,\\beta_j$ scara și deplasarea învățate; la testare, mediile mobile înlocuiesc statisticile lotului") + "}\n" +
            it(t("Batch normalisation (BN) smooths the loss landscape, which tolerates larger learning rates (Ioffe \\& Szegedy, 2015).",
                 "Normalizarea pe loturi (BN) netezește suprafața pierderii, ceea ce permite rate de învățare mai mari (Ioffe \\& Szegedy, 2015)."),
               t("The batch statistics add noise to each example's output: a mild regulariser; layer normalisation (Lecture 7) avoids the batch dependence.",
                 "Statisticile lotului adaugă zgomot la ieșirea fiecărui exemplu: un regularizator slab; normalizarea pe strat (cursul 7) evită dependența de lot.")) +
            D.bottom(t("with batch norm, \\texttt{model.eval()} is mandatory and very small batches should be avoided.",
                       "cu normalizare pe loturi, \\texttt{model.eval()} este obligatoriu, iar loturile foarte mici trebuie evitate."), NB + " \\S6"),
            cite="(Ioffe \\& Szegedy, 2015)")
    D.figframe(t("Batch norm lets a deep MLP train at a rate at which it otherwise diverges",
                 "Normalizarea pe loturi permite antrenarea la o rată la care altfel rețeaua diverge"),
               t("6 hidden layers of 64 units on California housing, SGD with momentum; medians of 3 seeds.",
                 "6 straturi ascunse de 64 de neuroni pe California housing, SGD cu moment; mediane a 3 inițializări."),
               None, "batchnorm",
               bullets=[t("At $\\eta=0.1$: @{bn.plain_01.diverged} of 3 seeds diverge without BN, best validation MSE @{bn.bn_01.val_best|.3f} with BN; at $\\eta=0.01$: "
                          "@{bn.plain_001.val_best|.3f} without, @{bn.bn_001.val_best|.3f} with BN.",
                          "La $\\eta=0{,}1$: @{bn.plain_01.diverged} din 3 inițializări diverg fără BN, cea mai bună MSE pe validare @{bn.bn_01.val_best|.3f} cu BN; la $\\eta=0{,}01$: "
                          "@{bn.plain_001.val_best|.3f} fără, @{bn.bn_001.val_best|.3f} cu BN.")],
               job=t("batch norm is mainly an optimisation aid; keep early stopping, its validation error fluctuates.",
                     "normalizarea pe loturi ajută mai ales optimizarea; păstrați oprirea timpurie, eroarea ei fluctuează."),
               tryit=NB + " \\S6", cite=OWN, height="0.60\\textheight")
    D.frame(t("Other regularisers: more data, noise, ensembles", "Alți regularizatori: mai multe date, zgomot, ansambluri"),
            cols(it(t("\\textbf{Data augmentation}: label-preserving transformations (crops and flips for images, Lecture 4); for tabular data, rarely safe.",
                      "\\textbf{Augmentarea datelor}: transformări care păstrează eticheta (decupări și oglindiri pentru imagini, cursul 4); pentru date tabelare, rareori sigură."),
                    t("\\textbf{Noise injection}: Gaussian noise on the inputs is equivalent, to first order, to an L2 penalty on the input gradient.",
                      "\\textbf{Injectarea de zgomot}: zgomotul gaussian pe intrări este echivalent, în prima aproximare, cu o penalizare L2 a gradientului în raport cu intrările."),
                    t("\\textbf{Label smoothing}: train on $(1-\\epsilon)y+\\epsilon/2$ instead of 0/1 targets; curbs over-confident probabilities.",
                      "\\textbf{Netezirea etichetelor}: antrenăm pe $(1-\\epsilon)y+\\epsilon/2$ în loc de ținte 0/1; limitează probabilitățile prea sigure.")),
                 it(t("\\textbf{Ensembles}: average several seeds or models; variance falls roughly as $1/M$ for $M$ weakly correlated members.",
                      "\\textbf{Ansamblurile}: media mai multor inițializări sau modele; varianța scade aproximativ ca $1/M$ pentru $M$ membri slab corelați."),
                    t("\\textbf{Smaller model or fewer inputs}: the oldest regulariser.", "\\textbf{Un model mai mic sau mai puține intrări}: cel mai vechi regularizator."),
                    t("\\textbf{More data}: the only one with no side effects; one more year of transactions beats any penalty.",
                      "\\textbf{Mai multe date}: singurul fără efecte secundare; încă un an de tranzacții valorează mai mult decît orice penalizare.")), "0.50", "0.47") +
            D.bottom(t("our fraud case averages 5 seeds: a cheap ensemble that also reports the uncertainty.",
                       "studiul de caz despre fraudă face media a 5 inițializări: un ansamblu ieftin care raportează și incertitudinea.")),
            cite="(Bishop \\& Bishop, 2024, ch.~9)")

    # ================================================================== 4. validation design
    D.section(t("Validation design", "Schema de validare"))
    D.frame(t("Hold-out, cross-validation and nested cross-validation", "Setul păstrat deoparte, validarea încrucișată și validarea încrucișată imbricată"),
            "\\intuition{" + t("every choice made by looking at data must be paid for with data the choice has not seen.",
                               "orice alegere făcută privind datele trebuie plătită cu date pe care alegerea nu le-a văzut.") + "}\n" +
            "\\[ \\widehat{\\mathrm{CV}}_K=\\frac1K\\sum_{k=1}^{K}\\frac1{|\\mathcal D_k|}\\sum_{i\\in\\mathcal D_k}\\ell\\big(y_i,\\hat f^{(-k)}(\\vx_i)\\big) \\]\n" +
            "\\where{" + t("$\\mathcal D_k$ fold $k$; $\\hat f^{(-k)}$ the model fitted on the other $K-1$ folds; usually $K=5$ or 10, stratified on the label",
                           "$\\mathcal D_k$ pliul $k$; $\\hat f^{(-k)}$ modelul estimat pe celelalte $K-1$ pliuri; de obicei $K=5$ sau 10, stratificat după etichetă") + "}\n" +
            it(t("Hold-out: one train/validation/test split; cheap, but noisy on small data (next frame).",
                 "Setul păstrat deoparte: o singură împărțire antrenare/validare/test; ieftin, dar zgomotos pe date puține (slide-ul următor)."),
               t("Nested cross-validation: an inner loop tunes hyperparameters, an outer loop measures the whole tuning procedure.",
                 "Validarea încrucișată imbricată: o buclă interioară ajustează hiperparametrii, una exterioară măsoară întreaga procedură de ajustare."),
               t("Report the spread across folds, not only the mean; compare models with the corrected $t$-test of Lecture 1.",
                 "Raportați variația între pliuri, nu doar media; comparați modelele cu testul $t$ corectat din cursul 1.")) +
            D.bottom(t("the reported performance must come from data used neither for fitting nor for choosing.",
                       "performanța raportată trebuie să provină din date folosite nici la estimare, nici la alegere."), NB + " \\S7"),
            cite="(Hastie, Tibshirani \\& Friedman, 2009, ch.~7)")
    D.figframe(t("One random hold-out split can move the AUC by several points", "O singură împărțire aleatoare poate muta AUC cu cîteva puncte"),
               t("logistic regression on German credit, 200 random stratified 70/30 splits of the same applicants.",
                 "regresia logistică pe German credit, 200 de împărțiri aleatoare stratificate 70/30 ale acelorași solicitanți."),
               None, "holdout_noise",
               bullets=[t("Test AUC from @{holdout.min|.3f} to @{holdout.max|.3f}; 90\\,\\% of splits in [@{holdout.p05|.3f}; @{holdout.p95|.3f}], standard deviation @{holdout.sd|.3f}.",
                          "AUC pe test între @{holdout.min|.3f} și @{holdout.max|.3f}; 90\\,\\% dintre împărțiri în [@{holdout.p05|.3f}; @{holdout.p95|.3f}], abaterea standard @{holdout.sd|.3f}.")],
               job=t("on a small dataset, one split cannot separate two models: use repeated cross-validation.",
                     "pe un set mic, o singură împărțire nu poate separa două modele: folosiți validarea încrucișată repetată."),
               tryit=NB + " \\S7", cite=OWN, height="0.60\\textheight")
    D.frame(t("Information leakage into the validation set", "Scurgerea de informație în setul de validare"),
            "\\intuition{" + t("any information from the evaluation rows that reaches training inflates the score and disappears in production.",
                               "orice informație din rîndurile de evaluare care ajunge în antrenare umflă scorul și dispare în producție.") + "}\n" +
            table("@{}p{0.27\\textwidth}p{0.36\\textwidth}p{0.30\\textwidth}@{}",
                  [t("Leak", "Scurgerea"), t("Example", "Exemplu"), t("Fix", "Remediu")],
                  [[t("Preprocessing on all rows", "Preprocesare pe toate rîndurile"), t("scaler or imputer fitted before the split", "standardizare sau completare estimate înainte de împărțire"),
                    t("fit on the training part only", "estimare doar pe antrenare")],
                   [t("Resampling before the split", "Reeșantionare înainte de împărțire"), t("copies of a fraud in both train and test", "copii ale unei fraude în antrenare și în test"),
                    t("oversample inside the training fold", "supraeșantionare doar în pliul de antrenare")],
                   [t("Ignoring time", "Ignorarea timpului"), t("random split of a time series or of payments", "împărțire aleatoare a unei serii de timp sau a plăților"),
                    t("train on the past, test on the future", "antrenare pe trecut, testare pe viitor")],
                   [t("Target in the inputs", "Ținta în intrări"), t("`days past due' used to predict default", "„zile de întîrziere” folosite pentru a prezice neplata"),
                    t("keep only information known at decision time", "doar informații cunoscute la momentul deciziei")]], size="scriptsize") +
            note(t("Time series: expanding-window or blocked cross-validation; never shuffle (Bergmeir \\& Benítez, 2012).",
                   "Serii de timp: validare cu fereastră extinsă sau pe blocuri; nu amestecați niciodată ordinea (Bergmeir \\& Benítez, 2012).")) +
            D.bottom(t("write down, for every input, the moment it becomes known.", "notați, pentru fiecare intrare, momentul în care devine cunoscută."), NB + " \\S8"),
            cite="(Kaufman et al., 2012)")
    D.figframe(t("Oversampling before the split overstates performance",
                 "Supraeșantionarea înainte de împărțire supraestimează performanța"),
               t("the same MLP 128-128 and the same transactions, evaluated three ways; 5 seeds each.",
                 "același MLP 128-128 și aceleași tranzacții, evaluate în trei moduri; cîte 5 inițializări."),
               None, "split_leakage",
               bullets=[t("PR-AUC @{design.time.ap|.3f} with a time-ordered split, @{design.random.ap|.3f} with a random split, @{design.oversample_first.ap|.3f} with frauds copied 20 times before splitting.",
                          "PR-AUC @{design.time.ap|.3f} cu împărțire cronologică, @{design.random.ap|.3f} cu împărțire aleatoare, @{design.oversample_first.ap|.3f} cu fraudele copiate de 20 de ori înainte de împărțire.")],
               job=t("only the time-ordered estimate describes what the bank will see after deployment.",
                     "doar estimarea cronologică descrie ce va vedea banca după punerea în producție."),
               tryit=NB + " \\S8", cite=OWN, height="0.60\\textheight")
    # ================================================================== 5. case study fraud
    D.section(t("Case study: card fraud detection", "Studiu de caz: detectarea fraudei cu carduri"))
    D.figframe(t("Fraud is rare and clustered in time", "Frauda este rară și grupată în timp"),
               t("Université libre de Bruxelles (ULB) and Worldline data, Open Machine Learning platform (OpenML): two days of card payments.",
                 "datele Université libre de Bruxelles (ULB) și Worldline, platforma Open Machine Learning (OpenML): două zile de plăți cu cardul."),
               None, "fraud_by_hour",
               bullets=[t("@{fraud.n_full|,d} payments, fraud rate @{fraud.prev_full_pct|.3f}\\,\\%; we keep all @{fraud.frauds} frauds and 10\\,\\% of legitimate payments (weight "
                          "@{fraud.legit_weight|.0f}); inputs: 28 principal component analysis (PCA) scores and the log amount.",
                          "@{fraud.n_full|,d} de plăți, rata fraudei @{fraud.prev_full_pct|.3f}\\,\\%; păstrăm toate cele @{fraud.frauds} de fraude și 10\\,\\% din plățile legitime (ponderea "
                          "@{fraud.legit_weight|.0f}); intrări: 28 de scoruri de analiză a componentelor principale (PCA) și logaritmul sumei.")],
               job=t("the hourly rate swings strongly: a random split would mix calm and attack hours.",
                     "rata orară variază puternic: o împărțire aleatoare ar amesteca orele liniștite și cele de atac."),
               tryit=NB + " \\S9", cite="(Dal Pozzolo et al., 2015)", height="0.60\\textheight")
    D.frame(t("Performance measures for rare events: precision, recall and PR-AUC", "Măsuri de performanță pentru evenimente rare: precizia, sensibilitatea și PR-AUC"),
            "\\intuition{" + t("flagging nothing is 99.8\\,\\% accurate; the bank cares how many alerts are frauds and how many frauds raise an alert.",
                               "dacă nu semnalăm nimic avem o acuratețe de 99,8\\,\\%; banca vrea să știe cîte alerte sînt fraude și cîte fraude declanșează o alertă.") + "}\n" +
            "\\[ \\text{" + t("precision", "precizia") + "}=\\frac{TP}{TP+FP},\\quad \\text{" + t("recall", "sensibilitatea") + "}=\\frac{TP}{TP+FN},\\quad "
            "\\text{PR-AUC}=\\sum_k(R_k-R_{k-1})\\,P_k \\]\n" +
            "\\where{" + t("$TP$, $FP$, $FN$ true positives, false positives, false negatives at a cut-off; $P_k,R_k$ precision and recall at the $k$-th threshold",
                           "$TP$, $FP$, $FN$ adevărat pozitive, fals pozitive, fals negative la un prag; $P_k,R_k$ precizia și sensibilitatea la al $k$-lea prag") + "}\n" +
            it(t("PR-AUC depends on the fraud rate (a random score gets PR-AUC = fraud rate); AUC does not, which hides the false alarms (Saito \\& Rehmsmeier, 2015).",
                 "PR-AUC depinde de rata fraudei (un scor aleator are PR-AUC = rata fraudei); AUC nu depinde, ceea ce ascunde alarmele false (Saito \\& Rehmsmeier, 2015)."),
               t("Undersampled data: every legitimate payment counts with weight @{fraud.legit_weight|.2f} in $FP$, so precision is that of the full population.",
                 "Date subeșantionate: fiecare plată legitimă contează cu ponderea @{fraud.legit_weight|.2f} în $FP$, deci precizia este cea a populației complete.")) +
            D.bottom(t("report precision at the recall the investigation team can handle.", "raportați precizia la sensibilitatea pe care o poate gestiona echipa de investigații."), NB + " \\S9"),
            cite="(Saito \\& Rehmsmeier, 2015)")
    D.figframe(t("On the test hours, early stopping is the regulariser with the clearest gain", "Pe orele de test, oprirea timpurie este regularizatorul cu cel mai clar cîștig"),
               t("MLP 256-256, time-ordered split, 5 seeds per configuration; precision weighted to the population.",
                 "MLP 256-256, împărțire cronologică, 5 inițializări pe configurație; precizia ponderată la populație."),
               None, "reg_ablation",
               bullets=[t("Test PR-AUC: none @{abl.none.ap|.3f}, early stopping @{abl.es.ap|.3f}, decay @{abl.wd.ap|.3f}, dropout @{abl.dropout.ap|.3f}, "
                          "batch norm @{abl.bn.ap|.3f}, all @{abl.all.ap|.3f}.",
                          "PR-AUC pe test: nimic @{abl.none.ap|.3f}, oprire timpurie @{abl.es.ap|.3f}, penalizare @{abl.wd.ap|.3f}, dropout @{abl.dropout.ap|.3f}, "
                          "normalizare @{abl.bn.ap|.3f}, toate @{abl.all.ap|.3f}.")],
               job=t("with 60 fixed epochs, the other regularisers stay within the seed noise.",
                     "cu 60 de epoci fixe, ceilalți regularizatori rămîn în zgomotul inițializărilor."),
               tryit=NB + " \\S10", cite=OWN, height="0.60\\textheight")
    D.figframe(t("The regularised MLP compared with logistic regression and gradient boosting",
                 "MLP regularizat comparat cu regresia logistică și cu gradient boosting"),
               t("test hours (last 20\\,\\%); the MLP averages 5 seeds with early stopping, decay and dropout.",
                 "orele de test (ultimele 20\\,\\%); MLP mediază 5 inițializări cu oprire timpurie, penalizare și dropout."),
               None, "fraud_pr",
               bullets=[t("PR-AUC: logistic @{final.logit.ap|.3f}, MLP @{final.mlp.ap|.3f}, extreme gradient boosting (XGBoost) @{final.xgb.ap|.3f}; "
                          "recall at 80\\,\\% precision @{final.logit.recall_at_p80_pct|.1f}, @{final.mlp.recall_at_p80_pct|.1f}, @{final.xgb.recall_at_p80_pct|.1f}\\,\\%.",
                          "PR-AUC: logistic @{final.logit.ap|.3f}, MLP @{final.mlp.ap|.3f}, extreme gradient boosting (XGBoost) @{final.xgb.ap|.3f}; "
                          "sensibilitatea la precizie de 80\\,\\%: @{final.logit.recall_at_p80_pct|.1f}, @{final.mlp.recall_at_p80_pct|.1f}, @{final.xgb.recall_at_p80_pct|.1f}\\,\\%.")],
               job=t("with @{fraud.fr_test} test frauds, a few points of difference are within the noise.",
                     "cu @{fraud.fr_test} de fraude în test, cîteva puncte de diferență sînt în interiorul zgomotului."),
               tryit=NB + " \\S10", cite=OWN, height="0.60\\textheight")
    D.frame(t("Class weights and the prior correction for undersampled data", "Ponderile claselor și corecția pentru datele subeșantionate"),
            "\\intuition{" + t("weighting the rare class changes the scores' level, not only their order; after undersampling, probabilities must be corrected.",
                               "ponderarea clasei rare schimbă nivelul scorurilor, nu doar ordinea lor; după subeșantionare, probabilitățile trebuie corectate.") + "}\n" +
            "\\[ \\loss_w=-\\frac1n\\sum_i\\big[w_+\\,y_i\\log p_i+w_-\\,(1-y_i)\\log(1-p_i)\\big],\\qquad "
            "p=\\frac{\\beta\\,p_s}{\\beta\\,p_s-p_s+1} \\]\n" +
            "\\where{" + t("$w_\\pm$ class weights; $p_s$ the probability learned on data where legitimate payments were kept with rate $\\beta$ (here 0.1); $p$ the corrected probability",
                           "$w_\\pm$ ponderile claselor; $p_s$ probabilitatea învățată pe date în care plățile legitime au fost păstrate cu rata $\\beta$ (aici 0,1); $p$ probabilitatea corectată") + "}\n" +
            it(t("Class weights or resampling change the effective cut-off; the ranking, and thus PR-AUC, changes little.",
                 "Ponderile claselor sau reeșantionarea schimbă pragul efectiv; ordonarea, deci și PR-AUC, se schimbă puțin."),
               t("Whatever is done, the cut-off should come from costs (next frame), and probabilities must be corrected before they are used as such.",
                 "Orice am face, pragul trebuie să provină din costuri (slide-ul următor), iar probabilitățile trebuie corectate înainte de a fi folosite ca atare.")) +
            D.bottom(t("never report probabilities from a resampled model to the business without the prior correction.",
                       "nu raportați niciodată probabilități dintr-un model reeșantionat fără corecția pentru distribuția a priori."), NB + " \\S10"),
            cite="(Dal Pozzolo et al., 2015)")
    D.figframe(t("Number of alerts and the share of fraud losses avoided", "Numărul de alerte și ponderea pierderilor din fraudă evitate"),
               t("a missed fraud costs its amount; each alert costs a review of @{cost.review|.0f} euro (EUR), a constant chosen a priori.",
                 "o fraudă ratată costă suma ei; fiecare alertă costă o verificare de @{cost.review|.0f} euro (EUR), o constantă aleasă a priori."),
               None, "fraud_cost",
               bullets=[t("Test losses with no alerts @{cost.no_model|,.0f} EUR; at the best alert count: logistic @{cost.logit.min|,.0f}, MLP @{cost.mlp.min|,.0f}, XGBoost @{cost.xgb.min|,.0f} EUR.",
                          "Pierderi fără alerte @{cost.no_model|,.0f} EUR; la numărul optim de alerte: logistic @{cost.logit.min|,.0f}, MLP @{cost.mlp.min|,.0f}, XGBoost @{cost.xgb.min|,.0f} EUR.")],
               job=t("the alert budget is the business decision, the model only ranks.", "bugetul de alerte este decizia de afaceri, modelul doar ordonează."),
               tryit=NB + " \\S10", cite=OWN, height="0.60\\textheight")
    D.frame(t("Summary of regularisation methods", "Sinteza metodelor de regularizare"),
            table("@{}p{0.17\\textwidth}p{0.30\\textwidth}p{0.22\\textwidth}p{0.25\\textwidth}@{}",
                  [t("Method", "Metoda"), t("What it does", "Ce face"), t("Main knob", "Parametrul principal"), t("On our data", "Pe datele noastre")],
                  [[t("Weight decay", "Penalizarea ponderilor"), t("shrinks all weights", "micșorează toate ponderile"), "$\\lambda$",
                    t("smaller weights, same AUC (German)", "ponderi mai mici, aceeași AUC (German)")],
                   ["L1", t("sets weights to zero", "aduce ponderi la zero"), "$\\lambda$", t("most first-layer weights near zero", "majoritatea ponderilor primului strat aproape de zero")],
                   ["Dropout", t("trains an ensemble of sub-networks", "antrenează un ansamblu de subrețele"), "$p$", t("within seed noise", "în zgomotul inițializărilor")],
                   [t("Early stopping", "Oprirea timpurie"), t("stops before memorising", "se oprește înainte de memorare"), t("patience $P$", "răbdarea $P$"),
                    t("clear gain on fraud", "cîștig clar la fraudă")],
                   [t("Batch norm", "Normalizarea pe loturi"), t("stabilises training", "stabilizează antrenarea"), t("batch size", "mărimea lotului"),
                    t("trains at $\\eta$ where plain SGD diverges", "învață la $\\eta$ la care SGD simplu diverge")],
                   [t("Smaller model", "Model mai mic"), t("fewer parameters", "mai puțini parametri"), t("width, depth", "lățime, adîncime"),
                    t("logistic beats MLP on German credit", "regresia logistică depășește MLP pe German credit")]], size="scriptsize") +
            D.bottom(t("start with early stopping and a modest model, then tune weight decay; add dropout and batch norm for deep networks.",
                       "porniți cu oprirea timpurie și un model modest, apoi ajustați penalizarea; adăugați dropout și normalizare pe loturi la rețelele adînci.")))

    # ================================================================== wrap-up
    D.section(t("Wrap-up", "Sinteză"))
    D.frame(t("Key takeaways", "Idei principale"), it(
        t("Overfitting shows as a falling training loss with a rising validation loss; on small data it starts within a few epochs.",
          "Supraînvățarea apare ca o pierdere de antrenare care scade și o pierdere de validare care crește; pe date puține începe după cîteva epoci."),
        t("Weight decay (Gaussian prior), L1 (sparsity), dropout (ensemble), early stopping (implicit L2) and batch norm (optimisation aid) control capacity in different ways.",
          "Penalizarea ponderilor (distribuție a priori gaussiană), L1 (raritate), dropout (ansamblu), oprirea timpurie (L2 implicit) și normalizarea pe loturi (ajutor la optimizare) controlează capacitatea în moduri diferite."),
        t("Very large models can show double descent: the classical U-curve is not the whole story.",
          "Modelele foarte mari pot prezenta dubla coborîre: curba clasică în U nu spune totul."),
        t("Validation must mimic deployment: time-ordered splits, preprocessing and resampling inside the training part only (PR-AUC @{design.time.ap|.3f} honest vs @{design.oversample_first.ap|.3f} leaked).",
          "Validarea trebuie să imite producția: împărțire cronologică, preprocesare și reeșantionare doar în partea de antrenare (PR-AUC @{design.time.ap|.3f} corect față de @{design.oversample_first.ap|.3f} cu scurgere)."),
        t("For rare events, use PR-AUC, precision at a workable recall and the cost of errors, with several seeds.",
          "Pentru evenimente rare, folosiți PR-AUC, precizia la o sensibilitate gestionabilă și costul erorilor, cu mai multe inițializări."), sep="4pt"))
    D.frame(t("Key equations", "Ecuațiile principale"), cols(
        "\\[ \\E(y-\\hat f)^2=\\text{bias}^2+\\Var(\\hat f)+\\sigma^2 \\]\n"
        "\\[ \\vw_{t+1}=(1-\\eta\\lambda)\\vw_t-\\eta\\nabla\\loss \\]\n"
        "\\[ w_j\\leftarrow\\operatorname{sign}(u_j)(|u_j|-\\eta\\lambda)_+ \\]\n"
        "\\[ \\tilde h_j=m_jh_j/(1-p) \\]",
        "\\[ \\hat z=(z-\\mu_{\\mathcal B})/\\sqrt{\\sigma^2_{\\mathcal B}+\\epsilon} \\]\n"
        "\\[ \\eta\\,t^*\\approx1/\\lambda \\]\n"
        "\\[ P=\\tfrac{TP}{TP+FP},\\ R=\\tfrac{TP}{TP+FN} \\]\n"
        "\\[ p=\\beta p_s/(\\beta p_s-p_s+1) \\]"))
    D.frame(t("Research from our group: fraud detection and early warning", "Cercetările grupului nostru: detectarea fraudei și avertizarea timpurie"), it(
        t("\\textbf{Explainable artificial intelligence (AI) for fraud} (Diaconu \\& Pele, 2026): on the Institute of Electrical and Electronics Engineers, Computational Intelligence Society "
          "(IEEE-CIS) fraud data, XGBoost and CatBoost had the higher AUC, while a graph neural network (GNN), graph sample and aggregate (GraphSAGE), captured relational "
          "fraud patterns that tabular models cannot describe.",
          "\\textbf{Inteligența artificială (AI) explicabilă pentru fraudă} (Diaconu \\& Pele, 2026): pe datele de fraudă Institute of Electrical and Electronics Engineers, Computational Intelligence "
          "Society (IEEE-CIS), XGBoost și CatBoost au avut AUC mai mare, în timp ce o rețea neuronală pe grafuri (GNN), eșantionare și agregare pe graf (GraphSAGE), a surprins "
          "tipare relaționale ale fraudei pe care modelele tabelare nu le pot descrie."),
        t("\\textbf{Explaining at several levels}: their explainable AI (XAI) framework combines Shapley additive explanations (SHAP) for "
          "variables with node and edge ablation for the graph; the paper concludes that fraud explanations should combine several model types --- replicated in Case study 1 (course site).",
          "\\textbf{Explicații pe mai multe niveluri}: cadrul lor de AI explicabilă (XAI) combină explicațiile aditive Shapley (SHAP) pentru variabile "
          "cu eliminarea de noduri și muchii pentru graf; lucrarea conchide că explicarea fraudei trebuie să combine mai multe tipuri de modele --- replicat în studiul de caz 1 (site-ul cursului)."),
        t("\\textbf{Early warning of crypto `zombies'} (Będowska-Sójka, Wójcik \\& Pele, 2026): machine-learning early warning systems that predict which "
          "crypto-assets become `zombie' assets; an early-warning model must be validated forward in time, as in Section 4.",
          "\\textbf{Avertizarea timpurie pentru criptoactivele „zombie”} (Będowska-Sójka, Wójcik \\& Pele, 2026): sisteme de avertizare timpurie bazate pe învățare "
          "automată, care prezic ce criptoactive devin active „zombie”; un model de avertizare timpurie trebuie validat înainte în timp, ca în secțiunea 4."), sep="5pt") +
        note(t("Findings as stated in the papers; numbers on the other slides come from our own computations.",
               "Rezultatele sînt cele declarate în lucrări; cifrele de pe celelalte slide-uri provin din calculele noastre.")))
    D.frame(t("Further reading and resources", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Bishop \\& Bishop (2024)}, chapter 9: regularisation: ", "\\textbf{Bishop \\& Bishop (2024)}, capitolul 9: regularizarea: ") +
        "\\href{https://www.bishopbook.com}{bishopbook.com}.",
        t("\\textbf{Hastie, Tibshirani \\& Friedman (2009)}, chapter 7: model assessment and selection: ",
          "\\textbf{Hastie, Tibshirani \\& Friedman (2009)}, capitolul 7: evaluarea și selecția modelelor: ") +
        "\\href{https://hastie.su.domains/ElemStatLearn/}{hastie.su.domains}.",
        t("\\textbf{Dive into Deep Learning}, sections 3.7, 5.6, 8.5: ", "\\textbf{Dive into Deep Learning}, secțiunile 3.7, 5.6, 8.5: ") + "\\href{https://d2l.ai}{d2l.ai}.", sep="4pt"),
        it(t("\\textbf{Fraud detection handbook} (Le Borgne et al., ULB): ", "\\textbf{Manualul de detectare a fraudei} (Le Borgne et al., ULB): ") +
           "\\href{https://fraud-detection-handbook.github.io/fraud-detection-handbook/}{fraud-detection-handbook.github.io}.",
           t("\\textbf{This course}: code, data, notebooks, Quantlets --- ", "\\textbf{Acest curs}: cod, date, notebook-uri, Quantlets --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Leakage}: Kaufman et al.\\ (2012).", "\\textbf{Scurgerea de informație}: Kaufman et al.\\ (2012)."), sep="4pt"), "0.52", "0.45"))

    # ------------------------------------------------------------------ quiz
    D.quiz_frame(t("reading the curves", "citirea curbelor"),
                 t("Training loss keeps falling, validation loss rises after epoch @{overfit.best_ep}. What do you do?",
                   "Pierderea de antrenare continuă să scadă, cea de validare crește după epoca @{overfit.best_ep}. Ce faceți?"),
                 t("The network overfits: keep the weights of epoch @{overfit.best_ep} (early stopping) and add weight decay or dropout; do not train longer.",
                   "Rețeaua supra-ajustează: păstrați ponderile din epoca @{overfit.best_ep} (oprire timpurie) și adăugați penalizare sau dropout; nu antrenați mai mult."))
    D.quiz_frame(t("L1 or L2", "L1 sau L2"),
                 t("Which penalty can switch an input off completely, and why?", "Ce penalizare poate opri complet o intrare și de ce?"),
                 t("L1: its soft-thresholding step sets small weights exactly to zero; here @{l1l2.share_small_l1_pct|.1f}\\,\\% of the first-layer weights are below @{l1l2.thr|g} (L2: @{l1l2.share_small_l2_pct|.1f}\\,\\%).",
                   "L1: pasul de prag moale aduce ponderile mici exact la zero; aici @{l1l2.share_small_l1_pct|.1f}\\,\\% dintre ponderile primului strat sînt sub @{l1l2.thr|g} (L2: @{l1l2.share_small_l2_pct|.1f}\\,\\%)."))
    D.quiz_frame(t("dropout at test time", "dropout la testare"),
                 t("With inverted dropout and $p=0.3$, what must be done to the activations at test time?",
                   "Cu dropout inversat și $p=0{,}3$, ce trebuie făcut cu activările la testare?"),
                 t("Nothing: dividing by $1-p$ during training already keeps the expectation; just call \\texttt{model.eval()}.",
                   "Nimic: împărțirea la $1-p$ în antrenare păstrează deja media; apelați doar \\texttt{model.eval()}."))
    D.quiz_frame(t("oversampling", "supraeșantionarea"),
                 t("Why does oversampling frauds before the split raise PR-AUC from @{design.time.ap|.3f} to @{design.oversample_first.ap|.3f}?",
                   "De ce supraeșantionarea fraudelor înainte de împărțire crește PR-AUC de la @{design.time.ap|.3f} la @{design.oversample_first.ap|.3f}?"),
                 t("Copies of the same fraud land in training and test, so the model is tested on frauds it has memorised; part of the gap is also the ignored time order.",
                   "Copii ale aceleiași fraude ajung în antrenare și în test, deci modelul este testat pe fraude memorate; o parte din diferență vine și din ignorarea ordinii în timp."))
    D.quiz_frame(t("the right metric", "metrica potrivită"),
                 t("A fraud model has AUC @{final.mlp.auc|.3f}. Is that enough to deploy it?", "Un model de fraudă are AUC @{final.mlp.auc|.3f}. Este suficient pentru a-l pune în producție?"),
                 t("No: with a fraud rate of @{fraud.prev_full_pct|.3f}\\,\\% a high AUC can hide many false alarms; check PR-AUC (@{final.mlp.ap|.3f}), precision at the target recall and the cost of alerts.",
                   "Nu: la o rată a fraudei de @{fraud.prev_full_pct|.3f}\\,\\%, o AUC mare poate ascunde multe alarme false; verificați PR-AUC (@{final.mlp.ap|.3f}), precizia la sensibilitatea țintă și costul alertelor."))
    reference_frames(D, REFS, per_frame=10)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.frame(t("Appendix: derivation of the bias-variance decomposition", "Anexă: derivarea descompunerii deplasare-varianță"),
            "\\[ \\E\\big[(y-\\hat f)^2\\big]=\\E\\big[(f+\\varepsilon-\\hat f)^2\\big]=\\sigma^2+\\E\\big[(f-\\hat f)^2\\big] \\]\n"
            "\\[ \\E\\big[(f-\\hat f)^2\\big]=\\E\\big[(f-\\bar f+\\bar f-\\hat f)^2\\big]=(f-\\bar f)^2+\\E\\big[(\\hat f-\\bar f)^2\\big],\\qquad \\bar f=\\E_{\\mathcal D}\\hat f \\]\n" +
            it(t("The cross terms vanish because $\\E[\\varepsilon]=0$, $\\varepsilon$ is independent of $\\mathcal D$, and $\\E[\\hat f-\\bar f]=0$.",
                 "Termenii micști dispar pentru că $\\E[\\varepsilon]=0$, $\\varepsilon$ este independent de $\\mathcal D$, iar $\\E[\\hat f-\\bar f]=0$."),
               t("Averaging over $x$ gives the decomposition of the expected test MSE.", "Media după $x$ dă descompunerea MSE așteptate pe test.")))
    D.frame(t("Appendix: why early stopping acts like an L2 penalty", "Anexă: de ce oprirea timpurie acționează ca o penalizare L2"),
            "\\[ \\loss(\\vw)\\approx\\tfrac12(\\vw-\\vw^*)^\\top\\mathbf H(\\vw-\\vw^*),\\ \\vw_0=\\mathbf 0:\\quad "
            "\\vw_t=\\big[\\mathbf I-(\\mathbf I-\\eta\\mathbf H)^t\\big]\\vw^* \\]\n"
            "\\[ \\text{L2:}\\ \\ \\tilde\\vw=(\\mathbf H+\\lambda\\mathbf I)^{-1}\\mathbf H\\vw^*;\\qquad \\text{" + t("on eigen-direction $k$", "pe direcția proprie $k$") +
            ":}\\ 1-(1-\\eta h_k)^t\\ \\approx\\ \\frac{h_k}{h_k+\\lambda}\\ \\ \\text{" + t("when", "cînd") + "}\\ \\lambda\\approx\\frac1{\\eta t} \\]\n" +
            it(t("Both shrink directions of small curvature $h_k$ most; stopping after $t$ steps is a penalty of strength about $1/(\\eta t)$.",
                 "Ambele micșorează cel mai mult direcțiile cu curbură $h_k$ mică; oprirea după $t$ pași este o penalizare de intensitate aproximativ $1/(\\eta t)$.")),
            cite="(Bishop \\& Bishop, 2024, ch.~9)")
    D.frame(t("Appendix: weight decay and the L2 penalty under SGD and under Adam", "Anexă: penalizarea ponderilor și penalizarea L2 pentru SGD și pentru Adam"),
            "\\[ \\text{SGD:}\\ \\vw-\\eta\\nabla\\big(\\loss+\\tfrac\\lambda2\\|\\vw\\|^2\\big)=(1-\\eta\\lambda)\\vw-\\eta\\nabla\\loss \\]\n"
            "\\[ \\text{Adam+L2:}\\ \\ \\vw-\\eta\\,\\frac{\\hat m(\\nabla\\loss+\\lambda\\vw)}{\\sqrt{\\hat v}+\\epsilon}\\qquad\\text{vs.}\\qquad "
            "\\text{AdamW:}\\ \\ (1-\\eta\\lambda)\\vw-\\eta\\,\\frac{\\hat m(\\nabla\\loss)}{\\sqrt{\\hat v}+\\epsilon} \\]\n" +
            it(t("With Adam, the L2 gradient is divided by $\\sqrt{\\hat v}$: weights with large gradients are barely decayed.",
                 "Cu Adam, gradientul L2 este împărțit la $\\sqrt{\\hat v}$: ponderile cu gradienți mari aproape nu sînt micșorate."),
               t("AdamW restores uniform decay; it is the optimiser used in this lecture's experiments.",
                 "AdamW refacе micșorarea uniformă; este optimizatorul folosit în experimentele acestui curs.").replace("refacе", "reface")),
            cite="(Loshchilov \\& Hutter, 2019)")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 3, ABBR, IGNORE, UNITS, langs=langs) else 1)
