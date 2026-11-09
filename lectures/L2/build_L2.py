r"""
build_L2.py -- Lecture 2 "Training: backpropagation and optimisation" (EN + RO).
Writes lecture2.tex / lecture2_ro.tex from the text below and the numbers in L2_numbers.json (analysis_L2.py),
compiles both and runs every check of lectures/STYLE.md (nn_build.run_build).
Run:  python3 lectures/L2/build_L2.py   [en|ro]
Rules: no computed number is typed by hand (results are @{key} placeholders); hand-typed numbers are definitions and
choices made a priori (learning-rate grids, momentum 0.9, Adam's 0.9/0.999, tau = 1 %, layer sizes, seeds).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_build  # noqa: E402
from nn_build import Deck, Numbers, abbreviation_frames, block, cols, items, reference_frames, run_build, table  # noqa: E402

_bw = nn_build.bulletize_where


def _compact_where(tex, per_line=3):
    """Keep the course's 'where' bullet, but put up to three symbol definitions on one level-2 line."""
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
    "AdamW": ("Adam with decoupled weight decay", "Adam cu penalizarea ponderilor decuplată"),
    "CPU": ("central processing unit", "procesor central"),
    "ES": ("expected shortfall", "pierderea așteptată în coada distribuției (expected shortfall)"),
    "EWMA": ("exponentially weighted moving average", "medie mobilă ponderată exponențial"),
    "GD": ("gradient descent", "coborîrea pe gradient"),
    "GPU": ("graphics processing unit", "procesor grafic"),
    "LLM": ("large language model", "model lingvistic mare"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "MPS": ("Metal Performance Shaders (Apple GPU back-end of PyTorch)", "Metal Performance Shaders (interfața PyTorch pentru procesorul grafic Apple)"),
    "MSE": ("mean squared error", "eroarea pătratică medie"),
    "OpenML": ("Open Machine Learning platform", "platforma Open Machine Learning"),
    "PCA": ("principal component analysis", "analiza componentelor principale"),
    "PR-AUC": ("area under the precision-recall curve", "aria de sub curba precizie-sensibilitate"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată"),
    "RMSProp": ("root mean square propagation", "propagarea cu medie pătratică (root mean square propagation)"),
    "SGD": ("stochastic gradient descent", "coborîrea stohastică pe gradient"),
    "ULB": ("Université libre de Bruxelles", "Université libre de Bruxelles"),
    "VaR": ("value at risk", "valoarea la risc"),
}
UNITS = set()
IGNORE = {"BF", "BIZ", "C5NC", "NEURAL", "US", "RMS", "LR", "POF", "KDD", "MIT", "LLM-ES", "CEE", "ICCV", "IEEE", "SSCI", "CS", "BCE", "ICLR",
          "JMLR", "PMLR", "SP", "NVIDIA", "AISTATS", "SPIE", "COURSERA", "SGDR", "USSR"}

REFS = [
    "Baydin, A.\\,G., Pearlmutter, B.\\,A., Radul, A.\\,A.\\ \\& Siskind, J.\\,M.\\ (2018). Automatic differentiation in machine learning: "
    "a survey. \\textit{Journal of Machine Learning Research}, 18(153), 1--43. \\href{https://arxiv.org/abs/1502.05767}{arXiv:1502.05767}",
    "Bengio, Y., Simard, P.\\ \\& Frasconi, P.\\ (1994). Learning long-term dependencies with gradient descent is difficult. "
    "\\textit{IEEE Transactions on Neural Networks}, 5(2), 157--166. \\href{https://doi.org/10.1109/72.279181}{doi:10.1109/72.279181}",
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}. Springer. "
    "\\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Dal Pozzolo, A., Caelen, O., Johnson, R.\\,A.\\ \\& Bontempi, G.\\ (2015). Calibrating probability with undersampling for unbalanced "
    "classification. \\textit{IEEE Symposium Series on Computational Intelligence}, 159--166. \\href{https://doi.org/10.1109/SSCI.2015.33}{doi:10.1109/SSCI.2015.33}",
    "Duchi, J., Hazan, E.\\ \\& Singer, Y.\\ (2011). Adaptive subgradient methods for online learning and stochastic optimization. "
    "\\textit{Journal of Machine Learning Research}, 12, 2121--2159.",
    "Glorot, X.\\ \\& Bengio, Y.\\ (2010). Understanding the difficulty of training deep feedforward neural networks. \\textit{Proceedings "
    "of AISTATS}, PMLR 9, 249--256.",
    "He, K., Zhang, X., Ren, S.\\ \\& Sun, J.\\ (2015). Delving deep into rectifiers. \\textit{IEEE International Conference on Computer "
    "Vision}, 1026--1034. \\href{https://doi.org/10.1109/ICCV.2015.123}{doi:10.1109/ICCV.2015.123}",
    "Jheng, S.-L., Tak, R., Găman, Ș., Mazurencu-Marinescu-Pele, M.\\ \\& Pele, D.\\,T.\\ (2026). Can foundation models manage risk? "
    "Zero-shot VaR and ES forecasting with conformal calibration in CEE markets. \\textit{Economic Computation and Economic "
    "Cybernetics Studies and Research}, 60(2), 267--290. \\href{https://doi.org/10.24818/18423264/60.2.26.14}{doi:10.24818/18423264/60.2.26.14}",
    "Keskar, N.\\,S., Mudigere, D., Nocedal, J., Smelyanskiy, M.\\ \\& Tang, P.\\,T.\\,P.\\ (2017). On large-batch training for deep learning: "
    "Generalization gap and sharp minima. \\textit{International Conference on Learning Representations}. \\href{https://arxiv.org/abs/1609.04836}{arXiv:1609.04836}",
    "Kingma, D.\\,P.\\ \\& Ba, J.\\ (2015). Adam: A method for stochastic optimization. \\textit{International Conference on Learning "
    "Representations}. \\href{https://arxiv.org/abs/1412.6980}{arXiv:1412.6980}",
    "Koenker, R.\\ \\& Bassett, G.\\ (1978). Regression quantiles. \\textit{Econometrica}, 46(1), 33--50. \\href{https://doi.org/10.2307/1913643}{doi:10.2307/1913643}",
    "Kupiec, P.\\,H.\\ (1995). Techniques for verifying the accuracy of risk measurement models. \\textit{The Journal of Derivatives}, 3(2), "
    "73--84. \\href{https://doi.org/10.3905/jod.1995.407942}{doi:10.3905/jod.1995.407942}",
    "Loshchilov, I.\\ \\& Hutter, F.\\ (2017). SGDR: Stochastic gradient descent with warm restarts. \\textit{International Conference on "
    "Learning Representations}. \\href{https://arxiv.org/abs/1608.03983}{arXiv:1608.03983}",
    "Loshchilov, I.\\ \\& Hutter, F.\\ (2019). Decoupled weight decay regularization. \\textit{International Conference on Learning "
    "Representations}. \\href{https://arxiv.org/abs/1711.05101}{arXiv:1711.05101}",
    "Pace, R.\\,K.\\ \\& Barry, R.\\ (1997). Sparse spatial autoregressions. \\textit{Statistics \\& Probability Letters}, 33(3), "
    "291--297. \\href{https://doi.org/10.1016/S0167-7152(96)00140-X}{doi:10.1016/S0167-7152(96)00140-X}",
    "Paszke, A.\\ et al.\\ (2019). PyTorch: An imperative style, high-performance deep learning library. \\textit{Advances in Neural "
    "Information Processing Systems}, 32. \\href{https://arxiv.org/abs/1912.01703}{arXiv:1912.01703}",
    "Pele, D.\\,T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A.\\,T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S.\\ \\& "
    "Härdle, W.\\,K.\\ (2026). In the beginning was the Word: LLM-VaR and LLM-ES. \\textit{Expert Systems with Applications}, 295, "
    "128676. \\href{https://doi.org/10.1016/j.eswa.2025.128676}{doi:10.1016/j.eswa.2025.128676}",
    "Polyak, B.\\,T.\\ (1964). Some methods of speeding up the convergence of iteration methods. \\textit{USSR Computational Mathematics "
    "and Mathematical Physics}, 4(5), 1--17. \\href{https://doi.org/10.1016/0041-5553(64)90137-5}{doi:10.1016/0041-5553(64)90137-5}",
    "Robbins, H.\\ \\& Monro, S.\\ (1951). A stochastic approximation method. \\textit{The Annals of Mathematical Statistics}, 22(3), "
    "400--407. \\href{https://doi.org/10.1214/aoms/1177729586}{doi:10.1214/aoms/1177729586}",
    "Rumelhart, D.\\,E., Hinton, G.\\,E.\\ \\& Williams, R.\\,J.\\ (1986). Learning representations by back-propagating errors. "
    "\\textit{Nature}, 323, 533--536. \\href{https://doi.org/10.1038/323533a0}{doi:10.1038/323533a0}",
    "Smith, L.\\,N.\\ \\& Topin, N.\\ (2019). Super-convergence: Very fast training of neural networks using large learning rates. "
    "\\textit{Proceedings of SPIE}, 11006. \\href{https://arxiv.org/abs/1708.07120}{arXiv:1708.07120}",
    "Tieleman, T.\\ \\& Hinton, G.\\ (2012). Lecture 6.5 --- RMSProp: Divide the gradient by a running average of its recent magnitude. "
    "\\textit{COURSERA: Neural Networks for Machine Learning}.",
]


def it(*xs, sep="2pt"):
    return items(*xs, opt=f"\\setlength{{\\itemsep}}{{{sep}}}")


def note(*xs):
    return "\n\\par\\vspace{0.5mm}{\\scriptsize " + items(*xs, opt="\\setlength{\\itemsep}{0pt}", nest=False) + "}"


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L2_numbers.json"), lang)
    N = V.N
    ro = lang == "ro"
    D = Deck(2, "Training: backpropagation and optimisation" if not ro else "Antrenarea rețelelor: propagarea înapoi a erorii și optimizarea",
             V, lang, heights=H)
    t = D.t
    NB = "notebook"
    OWN = t("(own computation)", "(calcul propriu)")

    def sci(key, digits=1):
        v = float(V.get(key))
        m, e = f"{v:.{digits}e}".split("e")
        m = m.replace(".", "{,}") if ro else m
        return f"${m}\\times10^{{{int(e)}}}$"

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Write gradient descent (GD) as the solution method of empirical risk minimisation and explain why conditioning sets its speed.",
          "Să scrieți coborîrea pe gradient (GD) ca metodă de rezolvare a minimizării riscului empiric și să explicați de ce condiționarea îi stabilește viteza."),
        t("Derive backpropagation for a multilayer perceptron (MLP), compute it by hand on real data, and check it against automatic differentiation.",
          "Să derivați propagarea înapoi a erorii pentru un perceptron multistrat (MLP), să o calculați manual pe date reale și să o verificați cu diferențierea automată."),
        t("Explain vanishing and exploding gradients and choose an initialisation that keeps the signal alive.",
          "Să explicați dispariția și explozia gradienților și să alegeți o inițializare care păstrează semnalul."),
        t("Compare stochastic gradient descent (SGD), momentum, root mean square propagation (RMSProp), Adam and learning-rate schedules on real data, with several seeds.",
          "Să comparați coborîrea stohastică pe gradient (SGD), metoda momentului, propagarea cu medie pătratică (RMSProp), Adam și programele ratei de învățare pe date reale, cu mai multe inițializări."),
        t("Train a quantile network for the 1-day value at risk (VaR) of the S\\&P 500 and backtest it.",
          "Să antrenați o rețea cuantilică pentru valoarea la risc (VaR) pe o zi a indicelui S\\&P 500 și să o testați retroactiv."), sep="4pt"))
    abbreviation_frames(D, ABBR)

    # ================================================================== 1. from loss to gradient
    D.section(t("From the loss to the gradient", "De la pierdere la gradient"))
    D.frame(t("Training a network is a high-dimensional, non-convex minimisation", "Antrenarea unei rețele este o minimizare nonconvexă, în dimensiune mare"),
            "\\intuition{" + t("we cannot solve for the weights as in least squares; we walk downhill on the loss, one step at a time.",
                               "nu putem afla ponderile printr-o formulă, ca la cele mai mici pătrate; coborîm pe suprafața pierderii, pas cu pas.") + "}\n" +
            "\\[ \\hat\\btheta=\\argmin_{\\btheta\\in\\R^{P}}\\ \\loss(\\btheta),\\qquad \\loss(\\btheta)=\\frac1n\\sum_{i=1}^n\\ell\\big(y_i,f(\\vx_i;\\btheta)\\big),\\qquad "
            "\\btheta_{t+1}=\\btheta_t-\\eta\\,\\nabla\\loss(\\btheta_t) \\]\n" +
            "\\where{" + t("$P$ the number of parameters; $\\eta>0$ the learning rate; $\\nabla\\loss$ the gradient, computed by backpropagation (Section 2)",
                           "$P$ numărul de parametri; $\\eta>0$ rata de învățare; $\\nabla\\loss$ gradientul, calculat prin propagare înapoi (secțiunea 2)") + "}\n" +
            it(t("Why it works: $\\loss(\\btheta-\\eta\\mathbf g)=\\loss(\\btheta)-\\eta\\|\\mathbf g\\|^2+O(\\eta^2)$, so a small step against the gradient lowers the loss.",
                 "De ce funcționează: $\\loss(\\btheta-\\eta\\mathbf g)=\\loss(\\btheta)-\\eta\\|\\mathbf g\\|^2+O(\\eta^2)$, deci un pas mic contra gradientului scade pierderea."),
               t("The MLP 64-64 of this lecture has @{gradcheck.n_params|,d} parameters on @{cal.n_train|,d} training block groups; models in production have millions.",
                 "MLP 64-64 din acest curs are @{gradcheck.n_params|,d} de parametri pe @{cal.n_train|,d} de grupuri de antrenare; modelele din producție au milioane."),
               t("Two questions: how to get $\\nabla\\loss$ cheaply and how to step well.",
                 "Două întrebări: cum obținem ieftin $\\nabla\\loss$ și cum facem pași buni.")) +
            D.bottom(t("every model of this course is trained by some variant of this one-line update.",
                       "fiecare model din acest curs este antrenat cu o variantă a acestei actualizări de un rînd.")),
            cite="(Bishop \\& Bishop, 2024, ch.~7)")
    D.frame(t("The data of this lecture: houses, card payments and the S\\&P 500", "Datele acestui curs: locuințe, plăți cu cardul și indicele S\\&P 500"),
            table("@{}p{0.25\\textwidth}rp{0.27\\textwidth}p{0.34\\textwidth}@{}",
                  [t("Dataset", "Setul de date"), t("Rows", "Rînduri"), t("Task", "Sarcina"), t("Split", "Împărțirea")],
                  [["California housing", "@{cal.n|,d}", t("regression: house value", "regresie: valoarea locuinței"),
                    t("random 60/20/20: @{cal.n_train|,d} / @{cal.n_val|,d} / @{cal.n_test|,d}", "aleatoare 60/20/20: @{cal.n_train|,d} / @{cal.n_val|,d} / @{cal.n_test|,d}")],
                   [t("Card fraud (sample)", "Fraude cu carduri (eșantion)"), "@{fraud.n|,d}", t("classification: fraud", "clasificare: fraudă"),
                    t("by time: first 60\\,\\% train, next 20\\,\\% validation, last 20\\,\\% test", "cronologic: primele 60\\,\\% antrenare, următoarele 20\\,\\% validare, ultimele 20\\,\\% test")],
                   ["S\\&P 500", "@{var.n_test|,d}", t("1\\,\\% quantile of the next-day return", "cuantila de 1\\,\\% a randamentului de a doua zi"),
                    t("by time: train to 2016, validation 2017--2019, test from 2020", "cronologic: antrenare pînă în 2016, validare 2017--2019, test din 2020")]],
                  size="scriptsize") +
            note(t("Fraud: Université libre de Bruxelles (ULB) and Worldline data, on the Open Machine Learning platform (OpenML); all @{fraud.frauds} frauds and a "
                   "10\\,\\% sample of legitimate payments; inputs V1--V28 are principal component analysis (PCA) scores released by the bank.",
                   "Frauda: datele Université libre de Bruxelles (ULB) și Worldline, pe platforma Open Machine Learning (OpenML); toate cele @{fraud.frauds} de fraude și "
                   "un eșantion de 10\\,\\% din plățile legitime; intrările V1--V28 sînt scoruri de analiză a componentelor principale (PCA) publicate de bancă."),
                 t("Weights restore the full-data fraud rate of @{fraud.prev_full_pct|.3f}\\,\\% in every precision-type metric; S\\&P 500: daily log returns in \\%, "
                   "test rows = forecast days.",
                   "Ponderile refac rata fraudei din datele complete, @{fraud.prev_full_pct|.3f}\\,\\%, în orice metrică de tip precizie; S\\&P 500: randamente "
                   "logaritmice zilnice în \\%, rîndurile de test = zilele prognozate.")) +
            D.bottom(t("time series and fraud are split in time order: the model never sees the future.",
                       "seriile de timp și frauda sînt împărțite cronologic: modelul nu vede niciodată viitorul."), NB + " \\S1"),
            cite="(Pace \\& Barry, 1997; Dal Pozzolo et al., 2015)")
    D.figframe(t("Badly scaled inputs turn the loss into a narrow valley", "Intrările nescalate transformă pierderea într-o vale îngustă"),
               t("the real loss of $\\hat y=a+b\\cdot\\text{income}$ on California: gradient descent zig-zags on raw income and goes straight on standardised income.",
                 "pierderea reală a lui $\\hat y=a+b\\cdot\\text{venit}$ pe California: coborîrea pe gradient face zigzaguri pe venitul brut și merge direct pe venitul standardizat."),
               None, "gd_paths",
               bullets=[t("Condition number $\\kappa=\\lambda_{\\max}/\\lambda_{\\min}$ of the Hessian: @{surface.cond_raw|.1f} raw, @{surface.cond_std|.0f} standardised; "
                          "after @{surface.steps} steps: mean squared error (MSE) @{surface.mse_gd_raw_60|.3f} (GD, raw), @{surface.mse_mom_raw_60|.3f} (momentum, raw), "
                          "@{surface.mse_gd_std_60|.3f} (GD, standardised); minimum @{surface.mse_star|.3f}.",
                          "Numărul de condiționare $\\kappa=\\lambda_{\\max}/\\lambda_{\\min}$ al hessianei: @{surface.cond_raw|.1f} brut, @{surface.cond_std|.0f} standardizat; "
                          "după @{surface.steps} de pași: eroarea pătratică medie (MSE) @{surface.mse_gd_raw_60|.3f} (GD, brut), @{surface.mse_mom_raw_60|.3f} (moment, brut), "
                          "@{surface.mse_gd_std_60|.3f} (GD, standardizat); minimul @{surface.mse_star|.3f}.")],
               job=t("standardise every input on the training set before training any network.",
                     "standardizați fiecare intrare pe setul de antrenare înainte de a antrena orice rețea."),
               tryit=NB + " \\S1", cite=OWN, height="0.56\\textheight")
    D.frame(t("On a quadratic, the step size must stay below $2/\\lambda_{\\max}$", "Pe o funcție pătratică, pasul trebuie să rămînă sub $2/\\lambda_{\\max}$"),
            "\\intuition{" + t("along each eigen-direction of the Hessian, gradient descent multiplies the error by $(1-\\eta\\lambda_k)$.",
                               "pe fiecare direcție proprie a hessianei, coborîrea pe gradient înmulțește eroarea cu $(1-\\eta\\lambda_k)$.") + "}\n" +
            "\\[ \\loss(\\btheta)=\\loss^*+\\tfrac12(\\btheta-\\btheta^*)^\\top\\mathbf H(\\btheta-\\btheta^*)\\ \\Rightarrow\\ "
            "\\btheta_{t}-\\btheta^*=(\\mathbf I-\\eta\\mathbf H)^t(\\btheta_0-\\btheta^*),\\qquad |1-\\eta\\lambda_k|<1\\iff 0<\\eta<\\frac{2}{\\lambda_{\\max}} \\]\n" +
            "\\where{" + t("$\\mathbf H$ the Hessian with eigenvalues $\\lambda_{\\min}\\le\\dots\\le\\lambda_{\\max}$; $\\btheta^*$ the minimum",
                           "$\\mathbf H$ hessiana, cu valorile proprii $\\lambda_{\\min}\\le\\dots\\le\\lambda_{\\max}$; $\\btheta^*$ minimul") + "}\n" +
            it(t("With the best fixed step the error shrinks by $(\\kappa-1)/(\\kappa+1)$ per step: slow when $\\kappa$ is large.",
                 "Cu cel mai bun pas fix, eroarea scade cu factorul $(\\kappa-1)/(\\kappa+1)$ la fiecare pas: lent cînd $\\kappa$ este mare."),
               t("Raw income: $\\kappa=$ @{surface.cond_raw|.1f}, so the step $1/\\lambda_{\\max}=$ @{surface.lr_raw|.4f} barely moves the flat direction.",
                 "Venitul brut: $\\kappa=$ @{surface.cond_raw|.1f}, deci pasul $1/\\lambda_{\\max}=$ @{surface.lr_raw|.4f} abia mișcă direcția plată."),
               t("Networks are not quadratic, but near a minimum they behave like one: the same limit explains divergence at large $\\eta$.",
                 "Rețelele nu sînt pătratice, dar lîngă un minim se comportă ca una: aceeași limită explică divergența la $\\eta$ mare.")) +
            D.bottom(t("if the loss explodes in the first epochs, divide the learning rate by 3 before changing anything else.",
                       "dacă pierderea explodează în primele epoci, împărțiți rata de învățare la 3 înainte de a schimba altceva.")),
            cite="(Bishop \\& Bishop, 2024, ch.~7)")

    # ================================================================== 2. backpropagation
    D.section(t("Backpropagation", "Propagarea înapoi a erorii"))
    D.frame(t("Backpropagation is the chain rule, applied from the loss backwards", "Propagarea înapoi este regula de derivare a funcțiilor compuse, aplicată de la pierdere înapoi"),
            "\\intuition{" + t("each node of the computational graph multiplies the incoming gradient by its own local derivative and passes it on.",
                               "fiecare nod al grafului de calcul înmulțește gradientul primit cu propria derivată locală și îl transmite mai departe.") + "}\n" +
            "\\[ \\vz^{(l)}=\\mW^{(l)}\\vh^{(l-1)}+\\vb^{(l)},\\quad \\vh^{(l)}=a(\\vz^{(l)});\\qquad "
            "\\boldsymbol\\delta^{(L)}=\\frac{\\partial\\ell}{\\partial\\vz^{(L)}},\\quad "
            "\\boldsymbol\\delta^{(l)}=\\big(\\mW^{(l+1)\\top}\\boldsymbol\\delta^{(l+1)}\\big)\\odot a'(\\vz^{(l)}) \\]\n"
            "\\[ \\frac{\\partial\\ell}{\\partial\\mW^{(l)}}=\\boldsymbol\\delta^{(l)}\\,\\vh^{(l-1)\\top},\\qquad \\frac{\\partial\\ell}{\\partial\\vb^{(l)}}=\\boldsymbol\\delta^{(l)} \\]\n" +
            "\\where{" + t("$\\boldsymbol\\delta^{(l)}$ the error signal of layer $l$; $\\odot$ the element-wise product; $a'$ the derivative of the activation",
                           "$\\boldsymbol\\delta^{(l)}$ semnalul de eroare al stratului $l$; $\\odot$ produsul element cu element; $a'$ derivata funcției de activare") + "}\n" +
            it(t("Output error: $\\hat y-y$ (squared loss), $p-y$ (sigmoid with cross-entropy, Lecture 1).",
                 "Eroarea de ieșire: $\\hat y-y$ (pierdere pătratică), $p-y$ (sigmoidă cu entropie încrucișată, cursul 1)."),
               t("Weight gradient = error of the layer times its input, as in the logistic gradient.",
                 "Gradientul unei ponderi = eroarea stratului înmulțită cu intrarea lui, ca la gradientul logistic.")) +
            D.bottom(t("the forward pass must store $\\vz^{(l)}$ and $\\vh^{(l)}$: this memory, not arithmetic, limits batch size on a graphics processing unit (GPU).",
                       "propagarea înainte trebuie să rețină $\\vz^{(l)}$ și $\\vh^{(l)}$: memoria, nu calculul, limitează mărimea lotului pe un procesor grafic (GPU).")),
            cite="(Rumelhart, Hinton \\& Williams, 1986)")
    b = "bp"
    D.frame(t("Backpropagation by hand on one real block group", "Propagarea înapoi, de mînă, pentru un grup real de locuințe"),
            "\\intuition{" + t("two standardised inputs, two rectified linear units (ReLU), one linear output and the loss $\\ell=\\tfrac12(\\hat y-y)^2$.",
                               "două intrări standardizate, două unități liniare rectificate (ReLU), ieșire liniară, $\\ell=\\tfrac12(\\hat y-y)^2$.") + "}\n" +
            cols(table("@{}lrr@{}", [t("Forward", "Înainte"), t("unit 1", "neuronul 1"), t("unit 2", "neuronul 2")],
                       [[t("input $x$ (income, latitude)", "intrarea $x$ (venit, latitudine)"), f"@{{{b}.x1|.2f}}", f"@{{{b}.x2|.2f}}"],
                        [t("weights $\\mW^{(1)}$, row 1 / row 2", "ponderile $\\mW^{(1)}$, rîndul 1 / 2"), f"@{{{b}.W11|.2f}}, @{{{b}.W12|.2f}}", f"@{{{b}.W21|.2f}}, @{{{b}.W22|.2f}}"],
                        ["$z=\\mW^{(1)}x$", f"@{{{b}.z1|.3f}}", f"@{{{b}.z2|.3f}}"],
                        ["$h=\\max(0,z)$", f"@{{{b}.h1|.3f}}", f"@{{{b}.h2|.3f}}"],
                        [t("output weights $w^{(2)}$", "ponderile de ieșire $w^{(2)}$"), f"@{{{b}.w21|.2f}}", f"@{{{b}.w22|.2f}}"],
                        ["$\\hat y=w^{(2)\\top}h$; $y$", f"@{{{b}.yhat|.3f}}", f"@{{{b}.y|.3f}}"]], size="scriptsize"),
                 table("@{}lrr@{}", [t("Backward", "Înapoi"), t("unit 1", "neuronul 1"), t("unit 2", "neuronul 2")],
                       [["$\\delta^{(2)}=\\hat y-y$", f"@{{{b}.d2|.3f}}", ""],
                        ["$\\partial\\ell/\\partial w^{(2)}=\\delta^{(2)}h$", f"@{{{b}.gw21|.3f}}", f"@{{{b}.gw22|.3f}}"],
                        ["$\\delta^{(1)}=\\delta^{(2)}w^{(2)}\\odot\\mathbb 1\\{z>0\\}$", f"@{{{b}.d11|.3f}}", f"@{{{b}.d12|.3f}}"],
                        ["$\\partial\\ell/\\partial W^{(1)}_{k1}=\\delta^{(1)}_kx_1$", f"@{{{b}.gW11|.3f}}", f"@{{{b}.gW21|.3f}}"],
                        ["$\\partial\\ell/\\partial W^{(1)}_{k2}=\\delta^{(1)}_kx_2$", f"@{{{b}.gW12|.3f}}", f"@{{{b}.gW22|.3f}}"]], size="scriptsize"),
                 "0.48", "0.49") +
            it(t("Unit 2 is switched off ($z_2<0$): it passes no gradient, so its weights do not move on this example.",
                 "Neuronul 2 este oprit ($z_2<0$): nu transmite gradient, deci ponderile lui nu se modifică."),
               t("PyTorch autograd gives the same gradients (largest difference @{bp.max_abs_diff_autograd|.1f}).",
                 "Diferențierea automată din PyTorch dă aceiași gradienți (diferența maximă @{bp.max_abs_diff_autograd|.1f}).")) +
            D.bottom(t("a ReLU unit that is off for every customer never learns again: watch the share of dead units.",
                       "un neuron ReLU oprit pentru toți clienții nu mai învață niciodată: urmăriți ponderea neuronilor „morți”."), NB + " \\S2"),
            cite=OWN)
    D.frame(t("Reverse mode costs about one extra forward pass, whatever the number of weights", "Modul invers costă cam încă o propagare înainte, oricîte ponderi ar fi"),
            "\\intuition{" + t("one backward sweep gives all $P$ partial derivatives of a scalar loss; finite differences would need $P$ extra passes.",
                               "o singură trecere înapoi dă toate cele $P$ derivate parțiale ale unei pierderi scalare; diferențele finite ar cere $P$ treceri în plus.") + "}\n" +
            "\\[ \\text{cost}\\big(\\nabla_{\\btheta}\\loss\\big)\\ \\le\\ c\\cdot\\text{cost}\\big(\\loss\\big),\\ \\ c\\approx 2\\text{--}3;\\qquad "
            "\\frac{\\partial\\loss}{\\partial\\theta_j}\\approx\\frac{\\loss(\\btheta+\\epsilon\\mathbf e_j)-\\loss(\\btheta-\\epsilon\\mathbf e_j)}{2\\epsilon} \\]\n" +
            it(t("Forward mode (dual numbers) is cheap for few inputs and many outputs; reverse mode for many inputs (weights) and one output (loss).",
                 "Modul direct (numere duale) este ieftin pentru puține intrări și multe ieșiri; modul invers pentru multe intrări (ponderi) și o ieșire (pierderea)."),
               t("Gradient check on our MLP 64-64: @{gradcheck.n_checked} random weights, central differences with $\\epsilon=10^{-6}$ in double precision, largest relative error "
                 + sci("gradcheck.max_rel_err") + ".",
                 "Verificarea gradientului pe MLP 64-64: @{gradcheck.n_checked} de ponderi alese aleator, diferențe centrale cu $\\epsilon=10^{-6}$ în dublă precizie, eroarea relativă maximă "
                 + sci("gradcheck.max_rel_err") + "."),
               t("PyTorch records the graph during the forward pass; \\texttt{loss.backward()} walks it in reverse.",
                 "PyTorch înregistrează graful în timpul propagării înainte; \\texttt{loss.backward()} îl parcurge invers.")) +
            D.bottom(t("when you write a custom layer or loss, run a gradient check once before trusting it.",
                       "cînd scrieți un strat sau o pierdere proprie, rulați o dată verificarea gradientului înainte de a vă baza pe ele."), NB + " \\S2"),
            cite="(Baydin et al., 2018; Paszke et al., 2019)")
    D.figframe(t("Vanishing gradients in deep sigmoid networks",
                 "Dispariția gradientului în rețelele adînci cu activare sigmoidă"),
               t("the backward signal is a product of Jacobians; with $\\sigma'\\le1/4$ it shrinks at every layer.",
                 "semnalul înapoi este un produs de matrice jacobiene; cu $\\sigma'\\le1/4$, el scade la fiecare strat."),
               "\\[ \\frac{\\partial\\ell}{\\partial\\vh^{(l)}}=\\Big(\\prod_{k=l+1}^{L}\\mW^{(k)\\top}\\operatorname{diag}\\big(a'(\\vz^{(k)})\\big)\\Big)\\frac{\\partial\\ell}{\\partial\\vh^{(L)}} \\]",
               "vanishing",
               bullets=[t("@{vanish.depth} hidden layers, one real batch: first/last layer gradient norm " + sci("vanish.sig.first") + " vs " + sci("vanish.sig.last") +
                          " with sigmoids; " + sci("vanish.relu_he.first") + " vs " + sci("vanish.relu_he.last") + " with ReLU and He initialisation.",
                          "@{vanish.depth} straturi ascunse, un lot real: norma gradientului în primul/ultimul strat " + sci("vanish.sig.first") + " față de " +
                          sci("vanish.sig.last") + " cu sigmoide; " + sci("vanish.relu_he.first") + " față de " + sci("vanish.relu_he.last") + " cu ReLU și inițializare He.")],
               job=t("deep nets need ReLU-type activations, a scaled initialisation and, later, normalisation (Lecture 3).",
                     "rețelele adînci au nevoie de activări ReLU, de o inițializare scalată și de normalizare (cursul 3)."),
               tryit=NB + " \\S3", cite=OWN, height="0.46\\textheight")
    D.frame(t("Initialisation that keeps the variance of the signal constant across layers", "Inițializarea păstrează constantă varianța semnalului între straturi"),
            "\\intuition{" + t("too small and the signal vanishes, too large and it explodes; the right scale depends on the fan-in and on the activation.",
                               "prea mică și semnalul dispare, prea mare și explodează; scara depinde de numărul de intrări și de activare.") + "}\n" +
            "\\[ \\Var(z_j^{(l)})=n_{l-1}\\Var(W)\\,\\E\\big[(h^{(l-1)})^2\\big]\\ \\Rightarrow\\ "
            "\\underbrace{\\Var(W)=\\frac{2}{n_{l-1}+n_l}}_{\\text{Xavier (tanh, sigmoid)}},\\qquad \\underbrace{\\Var(W)=\\frac{2}{n_{l-1}}}_{\\text{He (ReLU)}} \\]\n" +
            "\\where{" + t("$n_{l-1}$ the fan-in, $n_l$ the fan-out of layer $l$; for ReLU, $\\E[h^2]=\\tfrac12\\Var(z)$, hence the factor 2",
                           "$n_{l-1}$ numărul de intrări, $n_l$ numărul de ieșiri ale stratului $l$; pentru ReLU, $\\E[h^2]=\\tfrac12\\Var(z)$, de unde factorul 2") + "}\n" +
            it(t("Weights $\\mathcal N(0,0.01^2)$ in our 10-layer ReLU net: first-layer gradient norm " + sci("vanish.relu_small.first") + ", i.e.\\ no learning at all.",
                 "Ponderi $\\mathcal N(0;0{,}01^2)$, rețea ReLU cu 10 straturi: norma gradientului în primul strat " + sci("vanish.relu_small.first") + ", deci nicio învățare."),
               t("Never initialise all weights equal: every unit would learn the same thing.",
                 "Nu inițializați toate ponderile la fel: toți neuronii ar învăța același lucru.")) +
            D.bottom(t("PyTorch's defaults are close to these rules; set He initialisation explicitly for deep ReLU nets.",
                       "valorile implicite din PyTorch sînt apropiate de aceste reguli; pentru rețele ReLU adînci, setați explicit inițializarea He."), NB + " \\S3"),
            cite="(Glorot \\& Bengio, 2010; He et al., 2015)")

    # ================================================================== 3. stochastic optimisation
    D.section(t("Stochastic optimisation", "Optimizarea stohastică"))
    D.frame(t("A mini-batch gives an unbiased, noisy gradient at a fraction of the cost", "Un mini-lot dă un gradient nedeplasat și zgomotos, la o fracțiune din cost"),
            "\\intuition{" + t("averaging the gradient over $B$ random examples is enough to know roughly which way is down.",
                               "media gradientului pe $B$ exemple aleatoare este suficientă pentru a ști aproximativ încotro este coborîrea.") + "}\n" +
            "\\[ \\mathbf g_t=\\frac1B\\sum_{i\\in\\mathcal B_t}\\nabla\\ell_i(\\btheta_t),\\qquad \\E[\\mathbf g_t]=\\nabla\\loss(\\btheta_t),\\qquad "
            "\\Var(\\mathbf g_t)\\approx\\frac{1}{B}\\Big(1-\\frac{B}{n}\\Big)\\Sigma(\\btheta_t) \\]\n" +
            "\\where{" + t("$\\mathcal B_t$ a random batch of size $B$; $\\Sigma$ the covariance of the per-example gradients; one epoch $=\\lceil n/B\\rceil$ steps",
                           "$\\mathcal B_t$ un lot aleator de mărime $B$; $\\Sigma$ covarianța gradienților pe exemple; o epocă $=\\lceil n/B\\rceil$ pași") + "}\n" +
            it(t("Robbins--Monro: with $\\sum_t\\eta_t=\\infty$ and $\\sum_t\\eta_t^2<\\infty$, SGD converges to a stationary point.",
                 "Robbins--Monro: cu $\\sum_t\\eta_t=\\infty$ și $\\sum_t\\eta_t^2<\\infty$, SGD converge către un punct staționar."),
               t("The noise is not only a cost: it helps to leave sharp minima, which tend to generalise worse (Keskar et al., 2017).",
                 "Zgomotul nu este doar un cost: ajută la ieșirea din minimele abrupte, care tind să generalizeze mai slab (Keskar et al., 2017).")) +
            D.bottom(t("batches of 32--256 are the usual compromise between noise, speed and memory.",
                       "loturile de 32--256 de exemple sînt compromisul obișnuit între zgomot, viteză și memorie."), NB + " \\S4"),
            cite="(Robbins \\& Monro, 1951)")
    D.figframe(t("Effect of the learning rate on training", "Efectul ratei de învățare asupra antrenării"),
               t("plain SGD on an MLP 64-64, 30 epochs, batches of 128, for nine learning rates and three seeds.",
                 "SGD simplu pe un MLP 64-64, 30 de epoci, loturi de 128, pentru nouă rate de învățare și trei inițializări."),
               None, "lr_sweep",
               bullets=[t("Validation MSE @{lrsweep.smallest_mse|.3f} at $\\eta=10^{-4}$ (worse than linear regression, @{cal.mse_lin_val|.3f}), @{lrsweep.best_mse|.3f} at "
                          "$\\eta=$ @{lrsweep.best_lr|g}; from $\\eta=$ @{lrsweep.first_diverged|g} at least one seed diverges.",
                          "MSE pe validare @{lrsweep.smallest_mse|.3f} la $\\eta=10^{-4}$ (mai slab decît regresia liniară, @{cal.mse_lin_val|.3f}), @{lrsweep.best_mse|.3f} la "
                          "$\\eta=$ @{lrsweep.best_lr|g}; de la $\\eta=$ @{lrsweep.first_diverged|g}, cel puțin o inițializare diverge.")],
               job=t("tune the learning rate first, on a log grid, before any other hyperparameter.",
                     "ajustați mai întîi rata de învățare, pe o grilă logaritmică, înaintea oricărui alt hiperparametru."),
               tryit=NB + " \\S4", cite=OWN, height="0.56\\textheight")
    D.figframe(t("Effect of batch size: progress per epoch and per second", "Efectul dimensiunii lotului: progresul pe epocă și pe secundă"),
               t("Adam with learning rate 0.001, 40 epochs, batch sizes 16, 128 and 1,024.",
                 "Adam cu rata de învățare 0,001, 40 de epoci, loturi de 16, 128 și 1\\,024 de exemple."),
               None, "batch_size",
               bullets=[t("After 5 epochs: validation MSE @{batch.16.val_ep5|.3f} (batch 16), @{batch.128.val_ep5|.3f} (128), @{batch.1024.val_ep5|.3f} (1,024); "
                          "wall time for 40 epochs on one processor thread: @{batch.16.sec|.1f}, @{batch.128.sec|.1f} and @{batch.1024.sec|.1f} seconds.",
                          "După 5 epoci: MSE pe validare @{batch.16.val_ep5|.3f} (lot 16), @{batch.128.val_ep5|.3f} (128), @{batch.1024.val_ep5|.3f} (1\\,024); "
                          "durata a 40 de epoci pe un fir al procesorului: @{batch.16.sec|.1f}, @{batch.128.sec|.1f} și @{batch.1024.sec|.1f} secunde.")],
               job=t("compare training budgets in gradient steps or seconds, not in epochs.",
                     "comparați bugetele de antrenare în pași de gradient sau în secunde, nu în epoci."),
               tryit=NB + " \\S4", cite=OWN, height="0.56\\textheight")
    D.frame(t("Momentum: a running average of past gradients", "Metoda momentului: media mobilă a gradienților trecuți"),
            "\\intuition{" + t("zig-zag components cancel in the average, the consistent downhill component accumulates.",
                               "componentele în zigzag se anulează în medie, iar componenta constantă spre vale se acumulează.") + "}\n" +
            "\\[ \\mathbf v_{t+1}=\\beta\\,\\mathbf v_t+\\mathbf g_t,\\qquad \\btheta_{t+1}=\\btheta_t-\\eta\\,\\mathbf v_{t+1};\\qquad "
            "\\text{Nesterov:}\\ \\mathbf g_t=\\nabla\\loss(\\btheta_t-\\eta\\beta\\mathbf v_t) \\]\n" +
            "\\where{" + t("$\\mathbf v$ the velocity; $\\beta\\in[0,1)$ the momentum, usually 0.9; the effective step on a constant slope is $\\eta/(1-\\beta)$, ten times $\\eta$",
                           "$\\mathbf v$ viteza; $\\beta\\in[0,1)$ coeficientul de moment, de obicei 0,9; pe o pantă constantă, pasul efectiv este $\\eta/(1-\\beta)$, de zece ori $\\eta$") + "}\n" +
            it(t("On a quadratic, the best momentum improves the contraction from $\\frac{\\kappa-1}{\\kappa+1}$ to $\\frac{\\sqrt\\kappa-1}{\\sqrt\\kappa+1}$ (Polyak, 1964).",
                 "Pe o funcție pătratică, momentul optim îmbunătățește contracția de la $\\frac{\\kappa-1}{\\kappa+1}$ la $\\frac{\\sqrt\\kappa-1}{\\sqrt\\kappa+1}$ (Polyak, 1964)."),
               t("Raw-income regression after @{surface.steps} steps: MSE @{surface.mse_mom_raw_60|.3f} with momentum against @{surface.mse_gd_raw_60|.3f} for plain GD.",
                 "Regresia pe venitul brut după @{surface.steps} de pași: MSE @{surface.mse_mom_raw_60|.3f} cu moment, față de @{surface.mse_gd_raw_60|.3f} pentru GD simplu.")) +
            D.bottom(t("SGD with momentum 0.9 is still the default for very large vision models.",
                       "SGD cu moment 0,9 rămîne alegerea implicită pentru modelele foarte mari de vedere artificială."), NB + " \\S5"),
            cite="(Polyak, 1964)")
    D.frame(t("Adaptive methods give each weight its own step size", "Metodele adaptive dau fiecărei ponderi propriul pas"),
            "\\intuition{" + t("divide the step by a running measure of how large that weight's gradients have been.",
                               "împărțiți pasul la o măsură curentă a mărimii gradienților acelei ponderi.") + "}\n" +
            "\\[ \\text{AdaGrad: } s_t=s_{t-1}+g_t^2;\\qquad \\text{RMSProp: } s_t=\\rho\\,s_{t-1}+(1-\\rho)\\,g_t^2;\\qquad "
            "\\theta_{t+1}=\\theta_t-\\frac{\\eta}{\\sqrt{s_t}+\\epsilon}\\,g_t \\]\n" +
            "\\where{" + t("all operations element by element; $\\rho\\approx0.99$; $\\epsilon\\approx10^{-8}$ avoids division by zero",
                           "toate operațiile element cu element; $\\rho\\approx0{,}99$; $\\epsilon\\approx10^{-8}$ evită împărțirea la zero") + "}\n" +
            it(t("AdaGrad's sum only grows, so its steps shrink to zero: good for sparse features, too slow for deep nets (Duchi, Hazan \\& Singer, 2011).",
                 "Suma din AdaGrad doar crește, deci pașii lui scad spre zero: bun pentru variabile rare, prea lent pentru rețele adînci (Duchi, Hazan \\& Singer, 2011)."),
               t("Root mean square propagation (RMSProp) forgets old gradients exponentially, so the step adapts to the current region (Tieleman \\& Hinton, 2012).",
                 "Propagarea cu medie pătratică (RMSProp) uită exponențial gradienții vechi, deci pasul se adaptează regiunii curente (Tieleman \\& Hinton, 2012)."),
               t("Rescaling per weight partly undoes bad conditioning: rarely updated or flat directions get larger steps.",
                 "Rescalarea pe fiecare pondere compensează parțial condiționarea proastă: direcțiile plate sau rar actualizate primesc pași mai mari.")) +
            D.bottom(t("adaptive methods are less sensitive to the learning rate, which makes them a safe first choice.",
                       "metodele adaptive sînt mai puțin sensibile la rata de învățare, ceea ce le face o primă alegere sigură."), NB + " \\S5"))
    D.frame(t("Adam: momentum, RMSProp and a bias correction", "Adam: moment, RMSProp și corecția deplasării"),
            "\\intuition{" + t("keep moving averages of the gradient and of its square, correct them for starting at zero, and step by their ratio.",
                               "păstrăm medii mobile ale gradientului și ale pătratului lui, le corectăm pentru pornirea de la zero și facem un pas egal cu raportul lor.") + "}\n" +
            "\\[ m_t=\\beta_1m_{t-1}+(1-\\beta_1)g_t,\\quad v_t=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2,\\quad "
            "\\hat m_t=\\frac{m_t}{1-\\beta_1^t},\\ \\hat v_t=\\frac{v_t}{1-\\beta_2^t},\\quad \\theta_{t+1}=\\theta_t-\\eta\\,\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon} \\]\n" +
            "\\where{" + t("defaults $\\beta_1=0.9$, $\\beta_2=0.999$, $\\epsilon=10^{-8}$, $\\eta=10^{-3}$; $m_0=v_0=0$, hence the correction (appendix)",
                           "valori implicite $\\beta_1=0{,}9$, $\\beta_2=0{,}999$, $\\epsilon=10^{-8}$, $\\eta=10^{-3}$; $m_0=v_0=0$, de unde corecția (anexa)") + "}\n" +
            it(t("The step is roughly bounded by $\\eta$ per weight, whatever the scale of the loss: easy to tune.",
                 "Pasul este aproximativ limitat la $\\eta$ pentru fiecare pondere, oricare ar fi scara pierderii: ușor de ajustat."),
               t("Adam with decoupled weight decay (AdamW) shrinks weights directly, $\\theta\\leftarrow\\theta-\\eta\\lambda\\theta$, instead of adding $\\lambda\\theta$ to $g_t$ "
                 "(Loshchilov \\& Hutter, 2019; Lecture 3).",
                 "Adam cu penalizarea ponderilor decuplată (AdamW) micșorează direct ponderile, $\\theta\\leftarrow\\theta-\\eta\\lambda\\theta$, în loc să adauge $\\lambda\\theta$ la $g_t$ "
                 "(Loshchilov \\& Hutter, 2019; cursul 3).")) +
            D.bottom(t("start with AdamW and $\\eta=10^{-3}$; move to tuned SGD with momentum only if it pays out of sample.",
                       "porniți de la AdamW cu $\\eta=10^{-3}$; treceți la SGD cu moment, ajustat, doar dacă cîștigați în afara eșantionului."), NB + " \\S5"),
            cite="(Kingma \\& Ba, 2015)")
    D.figframe(t("On California housing, tuned optimisers reach similar errors and adaptive methods converge more reliably",
                 "Pe datele California, optimizatorii ajustați ating erori apropiate, iar metodele adaptive converg mai sigur"),
               t("best of three learning rates per optimiser (on validation), mean of 5 new seeds.",
                 "cea mai bună dintre trei rate de învățare (pe validare), media a 5 inițializări noi."),
               None, "optimizers",
               bullets=[t("Validation MSE after 10 epochs: SGD @{opt.sgd.val_ep10|.3f}, momentum @{opt.momentum.val_ep10|.3f}, RMSProp @{opt.rmsprop.val_ep10|.3f}, "
                          "Adam @{opt.adam.val_ep10|.3f}, AdamW @{opt.adamw.val_ep10|.3f}; linear regression @{cal.mse_lin_val|.3f}.",
                          "MSE pe validare după 10 epoci: SGD @{opt.sgd.val_ep10|.3f}, moment @{opt.momentum.val_ep10|.3f}, RMSProp @{opt.rmsprop.val_ep10|.3f}, "
                          "Adam @{opt.adam.val_ep10|.3f}, AdamW @{opt.adamw.val_ep10|.3f}; regresia liniară @{cal.mse_lin_val|.3f}.")],
               job=t("the network halves the error of the linear benchmark whatever the optimiser.",
                     "rețeaua înjumătățește eroarea modelului liniar de referință, oricare ar fi optimizatorul."),
               tryit=NB + " \\S5", cite=OWN, height="0.56\\textheight")
    D.figframe(t("Across seeds, Adam and AdamW are far more stable than plain SGD", "De la o inițializare la alta, Adam și AdamW sînt mult mai stabile decît SGD simplu"),
               t("test MSE of 5 seeds per optimiser; a single run would have ranked them by luck.",
                 "MSE pe test pentru 5 inițializări la fiecare optimizator; o singură rulare le-ar fi ordonat la întîmplare."),
               None, "optimizer_seeds",
               bullets=[t("Test MSE mean (standard deviation): SGD @{opt.sgd.test_mean|.3f} (@{opt.sgd.test_sd|.3f}), Adam @{opt.adam.test_mean|.3f} "
                          "(@{opt.adam.test_sd|.3f}), AdamW @{opt.adamw.test_mean|.3f} (@{opt.adamw.test_sd|.3f}); paired $t$-test SGD vs Adam: "
                          "$t=$ @{opt_test.t|.2f}, $p=$ @{opt_test.p|.3f}.",
                          "MSE pe test, media (abaterea standard): SGD @{opt.sgd.test_mean|.3f} (@{opt.sgd.test_sd|.3f}), Adam @{opt.adam.test_mean|.3f} "
                          "(@{opt.adam.test_sd|.3f}), AdamW @{opt.adamw.test_mean|.3f} (@{opt.adamw.test_sd|.3f}); testul $t$ pe perechi SGD față de Adam: "
                          "$t=$ @{opt_test.t|.2f}, $p=$ @{opt_test.p|.3f}.")],
               job=t("the mean gain is not significant with 5 seeds, but the risk of a bad run is much lower with Adam.",
                     "cîștigul mediu nu este semnificativ cu 5 inițializări, dar riscul unei rulări proaste este mult mai mic cu Adam."),
               tryit=NB + " \\S5", cite=OWN, height="0.54\\textheight")
    D.frame(t("Learning-rate schedules: step, cosine and one-cycle", "Programele ratei de învățare: în trepte, cosinus și one-cycle"),
            "\\intuition{" + t("explore with a large step, then reduce it so that the noise of SGD settles into a minimum.",
                               "explorăm cu un pas mare, apoi îl reducem, ca zgomotul SGD să se așeze într-un minim.") + "}\n" +
            "\\[ \\text{step: }\\eta_t=\\eta_0\\gamma^{\\lfloor t/s\\rfloor};\\qquad \\text{cosine: }\\eta_t=\\tfrac{\\eta_0}{2}\\Big(1+\\cos\\frac{\\pi t}{T}\\Big) \\]\n"
            "\\[ \\text{one-cycle: linear warm-up to }\\eta_0,\\text{ then cosine to }\\approx0 \\]\n".replace(
                "one-cycle: linear warm-up to }\\eta_0,\\text{ then cosine to }\\approx0", t("one-cycle: linear warm-up to }\\eta_0,\\text{ then cosine to }\\approx0",
                                                                                          "un ciclu: creștere liniară pînă la }\\eta_0,\\text{ apoi cosinus spre }\\approx0")).replace(
                "\\text{step: }", "\\text{" + t("step", "trepte") + ": }").replace("\\text{cosine: }", "\\text{" + t("cosine", "cosinus") + ": }") +
            "\\where{" + t("$\\eta_0$ the peak rate; $\\gamma$ the decay factor every $s$ steps; $T$ the total number of steps",
                           "$\\eta_0$ rata maximă; $\\gamma$ factorul de scădere la fiecare $s$ pași; $T$ numărul total de pași") + "}\n" +
            it(t("Cosine annealing (Loshchilov \\& Hutter, 2017) and one-cycle (Smith \\& Topin, 2019) need only the peak rate and the budget.",
                 "Descreșterea cosinus (Loshchilov \\& Hutter, 2017) și programul cu un singur ciclu (Smith \\& Topin, 2019) au nevoie doar de rata maximă și de buget."),
               t("Warm-up protects the first steps, when Adam's second-moment estimates and batch statistics are still unreliable.",
                 "Încălzirea protejează primii pași, cînd estimările momentului al doilea din Adam și statisticile loturilor nu sînt încă fiabile.")) +
            D.bottom(t("a cosine schedule over the planned number of epochs is a safe default.",
                       "un program cosinus pe numărul planificat de epoci este o alegere implicită sigură."), NB + " \\S6"),
            cite="(Bishop \\& Bishop, 2024, ch.~7)")
    D.figframe(t("Decaying the learning rate lowers the final error", "Scăderea ratei de învățare reduce eroarea finală"),
               t("SGD with momentum 0.9 and peak rate 0.01; left the schedules, right the validation MSE (mean of 3 seeds).",
                 "SGD cu moment 0,9 și rata maximă 0,01; stînga: programele, dreapta: MSE pe validare (media a 3 inițializări)."),
               None, "schedules",
               bullets=[t("Final validation MSE: constant @{sched.const.val_final|.3f}, step @{sched.step.val_final|.3f}, cosine @{sched.cosine.val_final|.3f}, "
                          "one-cycle @{sched.onecycle.val_final|.3f}; test @{sched.const.test|.3f} vs @{sched.cosine.test|.3f} (constant vs cosine).",
                          "MSE final pe validare: constantă @{sched.const.val_final|.3f}, trepte @{sched.step.val_final|.3f}, cosinus @{sched.cosine.val_final|.3f}, "
                          "un ciclu @{sched.onecycle.val_final|.3f}; pe test @{sched.const.test|.3f} față de @{sched.cosine.test|.3f} (constantă față de cosinus).")],
               job=t("the schedule is a cheap gain: same model, same budget, lower error.",
                     "programul ratei este un cîștig ieftin: același model, același buget, eroare mai mică."),
               tryit=NB + " \\S6", cite=OWN, height="0.54\\textheight")
    D.figframe(t("On card fraud, the optimiser determines the speed of learning", "La fraudele cu carduri, optimizatorul determină viteza învățării"),
               t("MLP 64-64 trained on the first 60\\,\\% of the two days, scored on the next 20\\,\\%.",
                 "MLP 64-64 antrenat pe primele 60\\,\\% din cele două zile, evaluat pe următoarele 20\\,\\%."),
               None, "fraud_optimizers",
               bullets=[t("Area under the precision-recall curve (PR-AUC) after 1 epoch: SGD @{fraud.sgd.ap_ep1|.3f}, momentum @{fraud.momentum.ap_ep1|.3f}, "
                          "Adam @{fraud.adam.ap_ep1|.3f} (test hours: @{fraud.sgd.ap_test|.3f}, @{fraud.momentum.ap_test|.3f}, @{fraud.adam.ap_test|.3f}).",
                          "Aria de sub curba precizie-sensibilitate (PR-AUC) după o epocă: SGD @{fraud.sgd.ap_ep1|.3f}, moment @{fraud.momentum.ap_ep1|.3f}, "
                          "Adam @{fraud.adam.ap_ep1|.3f} (orele de test: @{fraud.sgd.ap_test|.3f}, @{fraud.momentum.ap_test|.3f}, @{fraud.adam.ap_test|.3f}).")],
               job=t("with a rare class, plain SGD needs many epochs, momentum or Adam only one.",
                     "cu o clasă rară, SGD simplu are nevoie de multe epoci, momentul sau Adam de una singură."),
               tryit=NB + " \\S7", cite=OWN, height="0.52\\textheight")

    D.frame(t("Comparison of optimisers", "Compararea optimizatorilor"),
            table("@{}p{0.15\\textwidth}p{0.37\\textwidth}p{0.17\\textwidth}p{0.24\\textwidth}@{}",
                  [t("Method", "Metoda"), t("Update", "Actualizarea"), t("Extra memory", "Memorie suplimentară"), t("Use when", "Cînd se folosește")],
                  [["SGD", "$\\theta-\\eta g$", t("none", "nimic"), t("large vision models, tuned schedule", "modele mari de imagini, program ajustat")],
                   [t("Momentum", "Moment"), "$v\\leftarrow\\beta v+g$, $\\theta-\\eta v$", t("1 vector", "1 vector"), t("ill-conditioned losses", "pierderi prost condiționate")],
                   ["RMSProp", "$\\theta-\\eta g/\\sqrt{s}$, $s$ = EMA$(g^2)$".replace("EMA", t("avg", "medie")), t("1 vector", "1 vector"), t("recurrent nets, non-stationary losses", "rețele recurente, pierderi nestaționare")],
                   ["Adam", "$\\theta-\\eta\\hat m/(\\sqrt{\\hat v}+\\epsilon)$", t("2 vectors", "2 vectori"), t("default first choice", "prima alegere implicită")],
                   ["AdamW", t("Adam $+$ decay $(1-\\eta\\lambda)\\theta$", "Adam $+$ micșorare $(1-\\eta\\lambda)\\theta$"), t("2 vectors", "2 vectori"),
                    t("whenever weight decay is used", "ori de cîte ori se folosește penalizarea")]], size="scriptsize") +
            note(t("Memory is per weight: Adam stores two extra numbers for every parameter, which matters for very large models.",
                   "Memoria este pe pondere: Adam reține două numere suplimentare pentru fiecare parametru, ceea ce contează la modelele foarte mari.")) +
            D.bottom(t("choose by budget and stability; on our data the tuned optimisers end close, Adam and AdamW are the most stable.",
                       "alegeți după buget și stabilitate; pe datele noastre, optimizatorii ajustați ajung aproape, iar Adam și AdamW sînt cei mai stabili.")),
            cite="(Bishop \\& Bishop, 2024, ch.~7)")

    # ================================================================== 4. case study: VaR
    D.section(t("Case study: value at risk with a quantile network", "Studiu de caz: valoarea la risc cu o rețea cuantilică"))
    D.frame(t("The pinball loss turns any network into a quantile forecaster", "Pierderea pinball transformă orice rețea într-un model de prognoză a cuantilelor"),
            "\\intuition{" + t("penalise returns below the forecast $1-\\tau$ times more than returns above it, and the minimiser is the $\\tau$-quantile.",
                               "penalizăm randamentele de sub prognoză de $(1-\\tau)/\\tau$ ori mai mult decît pe cele de deasupra, iar minimul este cuantila $\\tau$.") + "}\n" +
            "\\[ \\mathrm{VaR}^{\\tau}_{t+1}=-q_{\\tau}(r_{t+1}\\mid\\mathcal F_t),\\qquad \\rho_\\tau(u)=u\\big(\\tau-\\mathbb 1\\{u<0\\}\\big),\\qquad "
            "\\hat q=\\argmin_{q}\\ \\E\\,\\rho_\\tau(r-q) \\]\n" +
            "\\where{" + t("$r_{t+1}$ the next-day log return in \\%; $\\mathcal F_t$ the information up to day $t$; $\\tau=1\\,\\%$ for a 99\\,\\% VaR",
                           "$r_{t+1}$ randamentul logaritmic de a doua zi, în \\%; $\\mathcal F_t$ informația pînă în ziua $t$; $\\tau=1\\,\\%$ pentru un VaR de 99\\,\\%") + "}\n" +
            it(t("Network: inputs $r_t$, $|r_t|$, 5-day mean, 5-, 22- and 66-day volatility; MLP 32-32 trained with Adam on $\\frac1n\\sum\\rho_\\tau(r_{t+1}-\\hat q_t)$.",
                 "Rețeaua: intrări $r_t$, $|r_t|$, media pe 5 zile, volatilitatea pe 5, 22 și 66 de zile; MLP 32-32 antrenat cu Adam pe $\\frac1n\\sum\\rho_\\tau(r_{t+1}-\\hat q_t)$."),
               t("Train from @{var.train_start|date} to 2016, validation 2017--2019 for early stopping, test @{var.test_start|date} to @{var.test_end|date}; average of @{var.seeds} seeds.",
                 "Antrenare de la @{var.train_start|date} pînă în 2016, validare 2017--2019 pentru oprirea timpurie, test @{var.test_start|date} -- @{var.test_end|date}; media a @{var.seeds} inițializări."),
               t("Benchmarks: 250-day historical simulation and exponentially weighted moving average (EWMA) volatility with $\\lambda=0.94$ and normal quantile.",
                 "Modele de referință: simularea istorică pe 250 de zile și volatilitatea calculată cu media mobilă ponderată exponențial (EWMA), $\\lambda=0{,}94$, cu cuantila normală.")) +
            D.bottom(t("the same network with another loss gives another risk measure: the loss defines what is learned.",
                       "aceeași rețea cu altă pierdere dă altă măsură de risc: pierderea definește ce se învață."), NB + " \\S8"),
            cite="(Koenker \\& Bassett, 1978)")
    D.figframe(t("Breaches of the quantile-MLP VaR during and after the 2020 pandemic are close to the nominal rate",
                 "Depășirile VaR-ului MLP cuantilic în pandemia din 2020 și după aceasta sînt apropiate de rata nominală"),
               t("test period only; red dots are days when the S\\&P 500 lost more than the MLP's 99\\,\\% VaR.",
                 "doar perioada de test; punctele roșii sînt zilele în care S\\&P 500 a pierdut mai mult decît VaR-ul de 99\\,\\% al MLP."),
               None, "var_backtest",
               bullets=[t("Violations over @{var.n_test|,d} days (expected @{var.expected|.1f}): MLP @{var.nn.viol}, historical simulation @{var.hs.viol}, "
                          "EWMA @{var.ewma.viol}; mean VaR @{var.nn.mean_var|.2f}\\,\\%, @{var.hs.mean_var|.2f}\\,\\%, @{var.ewma.mean_var|.2f}\\,\\%.",
                          "Depășiri pe @{var.n_test|,d} de zile (așteptat @{var.expected|.1f}): MLP @{var.nn.viol}, simularea istorică @{var.hs.viol}, "
                          "EWMA @{var.ewma.viol}; VaR mediu @{var.nn.mean_var|.2f}\\,\\%, @{var.hs.mean_var|.2f}\\,\\%, @{var.ewma.mean_var|.2f}\\,\\%.")],
               job=t("a VaR that reacts to volatility within days avoids the clusters of breaches of historical simulation.",
                     "un VaR care reacționează în cîteva zile la volatilitate evită grupările de depășiri ale simulării istorice."),
               tryit=NB + " \\S8", cite=OWN, height="0.54\\textheight")
    D.frame(t("The Kupiec test accepts the MLP and rejects both benchmarks", "Testul Kupiec acceptă MLP și respinge ambele modele de referință"),
            "\\intuition{" + t("if the VaR is right, the number of breaches is binomial with probability $\\tau$; test the observed rate against it.",
                               "dacă VaR-ul este corect, numărul de depășiri este binomial, cu probabilitatea $\\tau$; testăm rata observată față de ea.") + "}\n" +
            "\\[ \\mathrm{LR}_{\\mathrm{POF}}=-2\\ln\\frac{(1-\\tau)^{n-x}\\tau^{x}}{(1-\\hat\\pi)^{n-x}\\hat\\pi^{x}}\\ \\sim\\ \\chi^2_1,\\quad \\hat\\pi=\\frac xn,\\ "
            "n\\ \\text{" + t("days", "zile") + "},\\ x\\ \\text{" + t("violations", "depășiri") + "} \\]\n" +
            table("@{}lrrrrr@{}", [t("Model", "Model"), t("Violations", "Depășiri"), t("Rate (\\%)", "Rata (\\%)"), "$\\mathrm{LR}_{\\mathrm{POF}}$", "$p$",
                                   t("Pinball loss", "Pierderea pinball")],
                  [[t("Quantile MLP (5 seeds)", "MLP cuantilic (5 inițializări)"), "@{var.nn.viol}", "@{var.nn.rate_pct|.2f}", "@{var.nn.lr|.2f}", "@{var.nn.p|.3f}", "@{var.nn.pinball|.4f}"],
                   [t("Historical simulation, 250 days", "Simulare istorică, 250 de zile"), "@{var.hs.viol}", "@{var.hs.rate_pct|.2f}", "@{var.hs.lr|.2f}", "@{var.hs.p|.3f}", "@{var.hs.pinball|.4f}"],
                   ["EWMA, $\\lambda=0.94$".replace("0.94", "0{,}94" if ro else "0.94"), "@{var.ewma.viol}", "@{var.ewma.rate_pct|.2f}", "@{var.ewma.lr|.2f}", "@{var.ewma.p|.3f}", "@{var.ewma.pinball|.4f}"]]) +
            note(t("Individual seeds give violation rates between @{var.nn_seed_rate_min|.2f}\\,\\% and @{var.nn_seed_rate_max|.2f}\\,\\%; the pinball loss is the proper "
                   "score for comparing quantile forecasts.",
                   "Inițializările individuale dau rate de depășire între @{var.nn_seed_rate_min|.2f}\\,\\% și @{var.nn_seed_rate_max|.2f}\\,\\%; pierderea pinball este "
                   "scorul potrivit pentru compararea prognozelor de cuantile.")) +
            D.bottom(t("a regulator looks at the breach count first; a risk manager should also compare pinball losses.",
                       "o autoritate de supraveghere se uită întîi la numărul de depășiri; un manager de risc ar trebui să compare și pierderile pinball."), NB + " \\S8"),
            cite="(Kupiec, 1995)")
    D.frame(t("Training hardware: a laptop GPU is ten times faster than its main processor on a large MLP",
              "Echipamentul de calcul: procesorul grafic al unui laptop este de zece ori mai rapid decît procesorul central pe un MLP mare"),
            "\\intuition{" + t("training is matrix multiplication; a graphics processing unit (GPU) runs thousands of them in parallel.",
                               "antrenarea înseamnă înmulțiri de matrice; un procesor grafic (GPU) le execută în paralel, cu miile.") + "}\n" +
            it(t("Measured on this course's laptop (@{timing.machine}): one Adam step of an MLP 2048-2048 (@{timing.params|,d} parameters) on a batch of 2,048 real "
                 "California rows takes @{timing.cpu_ms|.1f} ms on the central processing unit (CPU, 4 threads) and @{timing.mps_ms|.1f} ms on the GPU through "
                 "Metal Performance Shaders (MPS): @{timing.speedup|.1f} times faster.",
                 "Măsurat pe laptopul cursului (@{timing.machine}): un pas Adam pentru un MLP 2048-2048 (@{timing.params|,d} de parametri), pe un lot de 2\\,048 de rînduri "
                 "reale California, durează @{timing.cpu_ms|.1f} ms pe procesorul central (CPU, 4 fire) și @{timing.mps_ms|.1f} ms pe GPU prin Metal Performance "
                 "Shaders (MPS): de @{timing.speedup|.1f} ori mai rapid."),
               t("The small tabular nets of Lectures 1--3 train faster on one CPU thread: copying tiny batches to the GPU costs more than it saves.",
                 "Rețelele tabelare mici din cursurile 1--3 se antrenează mai repede pe un singur fir al procesorului: copierea loturilor mici pe GPU costă mai mult decît economisește."),
               t("On Google Colab the same code runs on an NVIDIA GPU with \\texttt{device='cuda'}; set it once and move model and data with \\texttt{.to(device)}.",
                 "Pe Google Colab, același cod rulează pe un GPU NVIDIA cu \\texttt{device='cuda'}; setați-l o dată și mutați modelul și datele cu \\texttt{.to(device)}."),
               sep="4pt") +
            D.bottom(t("measure before you buy: for tabular models a CPU is enough, for images and text a GPU is not optional.",
                       "măsurați înainte să cumpărați: pentru modele tabelare ajunge un CPU, pentru imagini și text un GPU este indispensabil."), NB + " \\S9"),
            cite=OWN)
    D.frame(t("A default training procedure", "O procedură standard de antrenare"),
            cols(block(t("Before training", "Înainte de antrenare"), it(
                t("Standardise inputs on the training set; split in time order when time matters.",
                  "Standardizați intrările pe setul de antrenare; împărțiți cronologic cînd timpul contează."),
                t("ReLU layers with He initialisation; output and loss from the target (MSE, cross-entropy, pinball).",
                  "Straturi ReLU cu inițializare He; ieșirea și pierderea după țintă (MSE, entropie încrucișată, pinball)."),
                t("Overfit one small batch first: if the loss does not go to zero, there is a bug.",
                  "Întîi supra-ajustați un lot mic: dacă pierderea nu ajunge la zero, există o eroare în cod."))),
                 block(t("During training", "În timpul antrenării"), it(
                     t("AdamW, $\\eta$ on a log grid around $10^{-3}$, batch 64--256, cosine schedule.",
                       "AdamW, $\\eta$ pe o grilă logaritmică în jurul lui $10^{-3}$, loturi de 64--256, program cosinus."),
                     t("Watch training and validation loss every epoch; keep the best validation weights (Lecture 3).",
                       "Urmăriți pierderea de antrenare și de validare la fiecare epocă; păstrați ponderile cele mai bune pe validare (cursul 3)."),
                     t("Repeat with at least 3--5 seeds before claiming a gain.",
                       "Repetați cu cel puțin 3--5 inițializări înainte de a anunța un cîștig.")), "exampleblock"), "0.49", "0.49") +
            D.bottom(t("write the recipe into the model documentation: it is what makes a result reproducible.",
                       "scrieți rețeta în documentația modelului: ea face rezultatul reproductibil.")))

    # ================================================================== wrap-up
    D.section(t("Wrap-up", "Sinteză"))
    D.frame(t("Key takeaways", "Idei principale"), it(
        t("Training is gradient descent on the empirical risk; conditioning and the learning rate set its speed and stability.",
          "Antrenarea este coborîre pe gradient pe riscul empiric; condiționarea și rata de învățare îi stabilesc viteza și stabilitatea."),
        t("Backpropagation is the chain rule in reverse: all gradients for about the cost of one more forward pass; check custom code numerically.",
          "Propagarea înapoi este regula de derivare a funcțiilor compuse aplicată invers: toți gradienții pentru costul a încă o propagare înainte; verificați numeric codul propriu."),
        t("Saturating activations and bad initial scales make gradients vanish; ReLU with He initialisation keeps them alive.",
          "Activările care se saturează și scările inițiale greșite fac gradienții să dispară; ReLU cu inițializare He îi păstrează."),
        t("Momentum and adaptive steps (Adam, AdamW) reach good solutions faster and more reliably across seeds; schedules lower the final error.",
          "Momentul și pașii adaptivi (Adam, AdamW) ajung la soluții bune mai repede și mai sigur între inițializări; programele ratei reduc eroarea finală."),
        t("The loss defines the product: with the pinball loss, an MLP gives a 99\\,\\% VaR that passes the Kupiec test (@{var.nn.viol} breaches, expected @{var.expected|.1f}).",
          "Pierderea definește produsul: cu pierderea pinball, un MLP dă un VaR de 99\\,\\% care trece testul Kupiec (@{var.nn.viol} depășiri, așteptat @{var.expected|.1f})."),
        sep="4pt"))
    D.frame(t("Key equations", "Ecuațiile principale"), cols(
        "\\[ \\btheta_{t+1}=\\btheta_t-\\eta\\,\\mathbf g_t,\\quad 0<\\eta<2/\\lambda_{\\max} \\]\n"
        "\\[ \\boldsymbol\\delta^{(l)}=(\\mW^{(l+1)\\top}\\boldsymbol\\delta^{(l+1)})\\odot a'(\\vz^{(l)}) \\]\n"
        "\\[ \\partial\\ell/\\partial\\mW^{(l)}=\\boldsymbol\\delta^{(l)}\\vh^{(l-1)\\top} \\]\n"
        "\\[ \\Var(W)=2/n_{\\text{in}}\\ \\ (\\text{He}) \\]",
        "\\[ \\mathbf v_{t+1}=\\beta\\mathbf v_t+\\mathbf g_t \\]\n"
        "\\[ \\theta_{t+1}=\\theta_t-\\eta\\,\\hat m_t/(\\sqrt{\\hat v_t}+\\epsilon) \\]\n"
        "\\[ \\eta_t=\\tfrac{\\eta_0}{2}(1+\\cos\\pi t/T) \\]\n"
        "\\[ \\rho_\\tau(u)=u(\\tau-\\mathbb 1\\{u<0\\}) \\]"))
    D.frame(t("Research from our group: risk forecasts without gradient training", "Cercetările grupului nostru: prognoze de risc fără antrenare prin gradient"), it(
        t("\\textbf{Risk from language models} (Pele et al., 2026): value at risk and expected shortfall (ES) are obtained by prompting general-purpose large "
          "language models (LLM) with the return history (LLM-VaR, LLM-ES), without any task-specific gradient training --- Lecture 12.",
          "\\textbf{Riscul estimat de modele lingvistice} (Pele et al., 2026): valoarea la risc și pierderea așteptată în coada distribuției (ES) sînt obținute prin "
          "interogarea unor modele lingvistice mari (LLM) generale cu istoricul randamentelor (LLM-VaR, LLM-ES), fără antrenare prin gradient specifică sarcinii --- cursul 12."),
        t("\\textbf{Foundation models for tail risk} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026): pretrained time-series foundation models forecast "
          "VaR and ES zero-shot, and conformal calibration corrects their coverage, in Central and Eastern European markets --- Lecture 13.",
          "\\textbf{Modele fundație pentru riscul extrem} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026): modele fundație preantrenate pentru serii de "
          "timp prognozează VaR și ES fără antrenare suplimentară (zero-shot), iar calibrarea conformală le corectează acoperirea, pe piețele din Europa Centrală și de Est --- cursul 13."),
        t("\\textbf{How they relate to this lecture}: the same backtests (violation counts, Kupiec test, pinball loss) compare our gradient-trained quantile MLP "
          "with these pretrained models.",
          "\\textbf{Legătura cu acest curs}: aceleași teste retroactive (numărul de depășiri, testul Kupiec, pierderea pinball) compară MLP-ul nostru cuantilic, "
          "antrenat prin gradient, cu aceste modele preantrenate."), sep="5pt") +
        note(t("Findings as stated in the papers; numbers on the other slides come from our own computations.",
               "Rezultatele sînt cele declarate în lucrări; cifrele de pe celelalte slide-uri provin din calculele noastre.")))
    D.frame(t("Further reading and resources", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Bishop \\& Bishop (2024)}, chapters 7--8: gradient descent and backpropagation: ",
          "\\textbf{Bishop \\& Bishop (2024)}, capitolele 7--8: coborîrea pe gradient și propagarea înapoi: ") + "\\href{https://www.bishopbook.com}{bishopbook.com}.",
        t("\\textbf{Dive into Deep Learning}, chapter 12 (optimisation algorithms): ", "\\textbf{Dive into Deep Learning}, capitolul 12 (algoritmi de optimizare): ") +
        "\\href{https://d2l.ai/chapter_optimization}{d2l.ai}.",
        t("\\textbf{A. Karpathy, micrograd}: backpropagation in 100 lines: ", "\\textbf{A. Karpathy, micrograd}: propagarea înapoi în 100 de rînduri: ") +
        "\\href{https://github.com/karpathy/micrograd}{github.com/karpathy/micrograd}.", sep="4pt"),
        it(t("\\textbf{PyTorch autograd tutorial}: ", "\\textbf{Tutorialul PyTorch despre autograd}: ") +
           "\\href{https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html}{pytorch.org/tutorials}.",
           t("\\textbf{This course}: code, data, notebooks, Quantlets --- ", "\\textbf{Acest curs}: cod, date, notebook-uri, Quantlets --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Original papers}: Kingma \\& Ba (2015), Loshchilov \\& Hutter (2019).", "\\textbf{Lucrările originale}: Kingma \\& Ba (2015), Loshchilov \\& Hutter (2019)."),
           sep="4pt"), "0.52", "0.45"))

    # ------------------------------------------------------------------ quiz
    D.quiz_frame(t("the stable step size", "pasul stabil"),
                 t("The raw-income regression has Hessian eigenvalues @{surface.lam_min_raw|.3f} and @{surface.lam_max_raw|.2f}. Up to which learning rate does gradient descent converge?",
                   "Regresia pe venitul brut are valorile proprii ale hessianei @{surface.lam_min_raw|.3f} și @{surface.lam_max_raw|.2f}. Pînă la ce rată de învățare converge coborîrea pe gradient?"),
                 t("$\\eta<2/\\lambda_{\\max}=$ @{surface.lr_max_raw|.4f}; with $\\kappa=$ @{surface.cond_raw|.1f} the flat direction contracts very slowly: standardise or use momentum.",
                   "$\\eta<2/\\lambda_{\\max}=$ @{surface.lr_max_raw|.4f}; cu $\\kappa=$ @{surface.cond_raw|.1f}, direcția plată se contractă foarte lent: standardizați sau folosiți momentul."))
    D.quiz_frame(t("a dead ReLU", "un neuron ReLU mort"),
                 t("In the hand-computed example, why is $\\partial\\ell/\\partial W^{(1)}_{2j}=0$ for both inputs?",
                   "În exemplul calculat de mînă, de ce $\\partial\\ell/\\partial W^{(1)}_{2j}=0$ pentru ambele intrări?"),
                 t("Unit 2 has $z_2=$ @{bp.z2|.3f} $<0$, so $\\mathbb 1\\{z_2>0\\}=0$ and $\\delta^{(1)}_2=0$: a ReLU that is off passes no gradient.",
                   "Neuronul 2 are $z_2=$ @{bp.z2|.3f} $<0$, deci $\\mathbb 1\\{z_2>0\\}=0$ și $\\delta^{(1)}_2=0$: un ReLU oprit nu transmite gradient."))
    D.quiz_frame(t("vanishing gradients", "dispariția gradienților"),
                 t("Why does a 10-layer sigmoid network learn its first layers so slowly?",
                   "De ce învață atît de lent primele straturi ale unei rețele cu sigmoide și 10 straturi?"),
                 t("The backward signal is multiplied by $\\sigma'\\le0.25$ at each layer; on our batch the first-layer gradient norm is " + sci("vanish.sig.first") + ".",
                   "Semnalul înapoi este înmulțit cu $\\sigma'\\le0{,}25$ la fiecare strat; pe lotul nostru, norma gradientului în primul strat este " + sci("vanish.sig.first") + "."))
    D.quiz_frame(t("Adam's bias correction", "corecția deplasării în Adam"),
                 t("At step $t=1$, with $m_0=0$ and $\\beta_1=0.9$, what are $m_1$ and $\\hat m_1$?",
                   "La pasul $t=1$, cu $m_0=0$ și $\\beta_1=0{,}9$, cît sînt $m_1$ și $\\hat m_1$?"),
                 t("$m_1=0.1\\,g_1$, biased towards zero; $\\hat m_1=m_1/(1-0.9)=g_1$, the correction removes the bias.",
                   "$m_1=0{,}1\\,g_1$, deplasat spre zero; $\\hat m_1=m_1/(1-0{,}9)=g_1$, corecția elimină deplasarea."))
    D.quiz_frame(t("backtesting a VaR", "testarea retroactivă a unui VaR"),
                 t("Over @{var.n_test|,d} test days, how many breaches of a correct 99\\,\\% VaR do we expect, and which model passes the Kupiec test?",
                   "Pe @{var.n_test|,d} de zile de test, cîte depășiri ale unui VaR corect de 99\\,\\% ne așteptăm și ce model trece testul Kupiec?"),
                 t("About @{var.expected|.1f}; the quantile MLP has @{var.nn.viol} ($p=$ @{var.nn.p|.3f}), historical simulation @{var.hs.viol} ($p=$ @{var.hs.p|.3f}): only the MLP passes.",
                   "Circa @{var.expected|.1f}; MLP cuantilic are @{var.nn.viol} ($p=$ @{var.nn.p|.3f}), simularea istorică @{var.hs.viol} ($p=$ @{var.hs.p|.3f}): doar MLP trece."))
    reference_frames(D, REFS, per_frame=11)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.frame(t("Appendix: backpropagation in matrix form for a mini-batch", "Anexă: propagarea înapoi în formă matriceală pentru un mini-lot"),
            "\\[ \\mathbf Z^{(l)}=\\mathbf H^{(l-1)}\\mW^{(l)\\top}+\\mathbf 1\\vb^{(l)\\top},\\quad \\mathbf H^{(l)}=a(\\mathbf Z^{(l)}),\\qquad "
            "\\boldsymbol\\Delta^{(l)}=\\big(\\boldsymbol\\Delta^{(l+1)}\\mW^{(l+1)}\\big)\\odot a'(\\mathbf Z^{(l)}) \\]\n"
            "\\[ \\frac{\\partial\\loss}{\\partial\\mW^{(l)}}=\\frac1B\\boldsymbol\\Delta^{(l)\\top}\\mathbf H^{(l-1)},\\qquad "
            "\\frac{\\partial\\loss}{\\partial\\vb^{(l)}}=\\frac1B\\boldsymbol\\Delta^{(l)\\top}\\mathbf 1 \\]\n" +
            it(t("Rows of $\\mathbf H$, $\\mathbf Z$, $\\boldsymbol\\Delta$ are the $B$ examples of the batch; every step is a matrix product, which is why graphics processors help.",
                 "Rîndurile lui $\\mathbf H$, $\\mathbf Z$, $\\boldsymbol\\Delta$ sînt cele $B$ exemple ale lotului; fiecare pas este un produs de matrice, de aceea ajută procesoarele grafice."),
               t("Cost: forward $\\sum_l B\\,n_{l-1}n_l$ multiply-adds; backward about twice that.",
                 "Costul: înainte $\\sum_l B\\,n_{l-1}n_l$ înmulțiri și adunări; înapoi, cam de două ori mai mult.")))
    D.frame(t("Appendix: why Adam needs a bias correction", "Anexă: de ce are Adam nevoie de corecția deplasării"),
            "\\[ m_t=(1-\\beta_1)\\sum_{k=1}^{t}\\beta_1^{t-k}g_k\\ \\Rightarrow\\ \\E[m_t]=\\E[g]\\,(1-\\beta_1)\\sum_{k=1}^{t}\\beta_1^{t-k}=\\E[g]\\,(1-\\beta_1^t) \\]\n" +
            it(t("If the gradient distribution is stationary, $m_t/(1-\\beta_1^t)$ is unbiased for $\\E[g]$; the same holds for $v_t$ and $\\E[g^2]$.",
                 "Dacă distribuția gradientului este staționară, $m_t/(1-\\beta_1^t)$ este nedeplasat pentru $\\E[g]$; la fel $v_t$ pentru $\\E[g^2]$."),
               t("With $\\beta_2=0.999$, $v_t$ without correction is 1000 times too small at $t=1$: the first steps would be huge.",
                 "Cu $\\beta_2=0{,}999$, $v_t$ fără corecție este de 1000 de ori prea mic la $t=1$: primii pași ar fi uriași.")),
            cite="(Kingma \\& Ba, 2015)")
    D.frame(t("Appendix: the Kupiec likelihood-ratio test", "Anexă: testul raportului de verosimilitate al lui Kupiec"),
            it(t("Violations $I_t=\\mathbb 1\\{r_t<-\\mathrm{VaR}_t\\}$; under a correct model, $I_t\\iid\\text{Bernoulli}(\\tau)$ and $x=\\sum I_t\\sim\\text{Bin}(n,\\tau)$.",
                 "Depășirile $I_t=\\mathbb 1\\{r_t<-\\mathrm{VaR}_t\\}$; dacă modelul este corect, $I_t\\iid\\text{Bernoulli}(\\tau)$ și $x=\\sum I_t\\sim\\text{Bin}(n,\\tau)$."),
               t("Likelihood under $H_0$: $(1-\\tau)^{n-x}\\tau^x$; under the alternative, maximised at $\\hat\\pi=x/n$.",
                 "Verosimilitatea sub $H_0$: $(1-\\tau)^{n-x}\\tau^x$; sub alternativă, maximizată în $\\hat\\pi=x/n$."),
               t("$\\mathrm{LR}=-2\\ln(L_0/L_1)\\to\\chi^2_1$; reject at 5\\,\\% when $\\mathrm{LR}>3.84$.",
                 "$\\mathrm{LR}=-2\\ln(L_0/L_1)\\to\\chi^2_1$; respingem la 5\\,\\% cînd $\\mathrm{LR}>3{,}84$."),
               t("The test checks only the count; independence of the violations needs Christoffersen's test (Lecture 13).",
                 "Testul verifică doar numărul; independența depășirilor cere testul Christoffersen (cursul 13)."), sep="5pt"),
            cite="(Kupiec, 1995)")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 2, ABBR, IGNORE, UNITS, langs=langs) else 1)
