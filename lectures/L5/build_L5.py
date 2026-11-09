r"""
build_L5.py -- Lecture 5 "Recurrent networks: RNN, LSTM and GRU for sequences and financial time series" (EN + RO).
Slide text only; every result is a placeholder @{key} filled from L5_numbers.json (analysis_L5.py).
Run:  python3 lectures/L5/build_L5.py   ("en" or "ro" builds one language; exit 0 only if every check passes)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, cols, hier, items, reference_frames, run_build, table)  # noqa: E402

ABBR = {
    "AR": ("autoregressive model", "model autoregresiv"),
    "AI": ("artificial intelligence", "inteligență artificială"),
    "CNN": ("convolutional neural network", "rețea neuronală convoluțională"),
    "ARX": ("autoregressive model with exogenous inputs", "model autoregresiv cu variabile exogene"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "BPTT": ("backpropagation through time", "propagarea înapoi în timp"),
    "CEE": ("Central and Eastern Europe", "Europa Centrală și de Est"),
    "CPU": ("central processing unit", "procesor central"),
    "DM": ("Diebold--Mariano test", "testul Diebold--Mariano"),
    "ES": ("expected shortfall", "pierderea medie așteptată peste VaR (expected shortfall)"),
    "GARCH": ("generalised autoregressive conditional heteroskedasticity", "model autoregresiv generalizat cu heteroscedasticitate condiționată"),
    "GK": ("Garman--Klass range estimator of variance", "estimatorul Garman--Klass al varianței, bazat pe amplitudine"),
    "GRU": ("gated recurrent unit", "unitate recurentă cu porți (gated recurrent unit)"),
    "HAR": ("heterogeneous autoregressive model", "model autoregresiv heterogen"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "LSTM": ("long short-term memory", "memorie pe termen scurt extinsă (long short-term memory)"),
    "MAE": ("mean absolute error", "eroarea absolută medie"),
    "MAPE": ("mean absolute percentage error", "eroarea procentuală absolută medie"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "MSE": ("mean squared error", "eroarea pătratică medie"),
    "RNN": ("recurrent neural network", "rețea neuronală recurentă"),
    "TLV": ("Banca Transilvania (Bucharest Stock Exchange ticker)", "Banca Transilvania (simbolul la Bursa de Valori București)"),
    "VaR": ("value-at-risk", "valoarea expusă la risc (value-at-risk)"),
    "BVB": ("Bucharest Stock Exchange", "Bursa de Valori București"),
    "ENTSO-E": ("European Network of Transmission System Operators for Electricity", "Rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică"),
    "EODHD": ("End of Day Historical Data (data provider)", "End of Day Historical Data (furnizor de date)"),
    "BiLSTM": ("bidirectional LSTM", "LSTM bidirecțional"),
    "MW": ("megawatt", "megawatt"), "MWh": ("megawatt-hour", "megawatt-oră"), "GW": ("gigawatt", "gigawatt"),
}
UNITS = {"MW", "MWh", "GW"}
IGNORE = {"BIZ", "NEURAL", "SOURCES", "NW", "OOS", "RN", "PICBE", "OK", "II", "IEEE", "JBES", "ICML", "PMLR", "NIPS", "EMNLP", "S&P", "SP", "ISE", "ESWA", "NN", "IJF", "JAE", "JFEC"}

REFS = [
    "Bengio, Y., Simard, P.\\ \\& Frasconi, P.\\ (1994). Learning long-term dependencies with gradient descent is difficult. \\textit{IEEE Transactions on Neural Networks}, 5(2), 157--166. \\href{https://doi.org/10.1109/72.279181}{doi:10.1109/72.279181}",
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}, ch.~12. Springer. \\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Bollerslev, T.\\ (1986). Generalized autoregressive conditional heteroskedasticity. \\textit{Journal of Econometrics}, 31(3), 307--327. \\href{https://doi.org/10.1016/0304-4076(86)90063-1}{doi:10.1016/0304-4076(86)90063-1}",
    "Będowska-Sójka, B., Wójcik, P.\\ \\& Pele, D.\\,T.\\ (2026). Early warning systems for cryptocurrency markets: Predicting `zombie' assets using machine learning. \\textit{North American Journal of Economics and Finance}, 81, 102543. \\href{https://doi.org/10.1016/j.najef.2025.102543}{doi:10.1016/j.najef.2025.102543}",
    "Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H.\\ \\& Bengio, Y.\\ (2014). Learning phrase representations using RNN encoder--decoder for statistical machine translation. \\textit{EMNLP}, 1724--1734. \\href{https://doi.org/10.3115/v1/D14-1179}{doi:10.3115/v1/D14-1179}",
    "Conda, A.\\,I., Găman, Ș., Bâg, R.\\,C., Mazurencu-Marinescu-Pele, M., Pele, D.\\,T.\\ \\& Härdle, W.\\,K.\\ (2026). BitMood: AI analysis of Bitcoin trends via Facebook emotions. \\textit{Digital Finance}, 8(3), 42. \\href{https://doi.org/10.1007/s42521-026-00205-3}{doi:10.1007/s42521-026-00205-3}",
    "Corsi, F.\\ (2009). A simple approximate long-memory model of realized volatility. \\textit{Journal of Financial Econometrics}, 7(2), 174--196. \\href{https://doi.org/10.1093/jjfinec/nbp001}{doi:10.1093/jjfinec/nbp001}",
    "Diebold, F.\\,X.\\ \\& Mariano, R.\\,S.\\ (1995). Comparing predictive accuracy. \\textit{Journal of Business \\& Economic Statistics}, 13(3), 253--263. \\href{https://doi.org/10.1080/07350015.1995.10524599}{doi:10.1080/07350015.1995.10524599}",
    "Garman, M.\\,B.\\ \\& Klass, M.\\,J.\\ (1980). On the estimation of security price volatilities from historical data. \\textit{Journal of Business}, 53(1), 67--78. \\href{https://doi.org/10.1086/296072}{doi:10.1086/296072}",
    "Hochreiter, S.\\ \\& Schmidhuber, J.\\ (1997). Long short-term memory. \\textit{Neural Computation}, 9(8), 1735--1780. \\href{https://doi.org/10.1162/neco.1997.9.8.1735}{doi:10.1162/neco.1997.9.8.1735}",
    "Jheng, S.-L., Tak, R., Găman, Ș., Mazurencu-Marinescu-Pele, M.\\ \\& Pele, D.\\,T.\\ (2026). Can foundation models manage risk? Zero-shot VaR and ES forecasting with conformal calibration in CEE markets. \\textit{Economic Computation and Economic Cybernetics Studies and Research}, 60(2), 267--290. \\href{https://doi.org/10.24818/18423264/60.2.26.14}{doi:10.24818/18423264/60.2.26.14}",
    "Pascanu, R., Mikolov, T.\\ \\& Bengio, Y.\\ (2013). On the difficulty of training recurrent neural networks. \\textit{ICML}, PMLR 28(3), 1310--1318. \\href{https://arxiv.org/abs/1211.5063}{arXiv:1211.5063}",
    "Rumelhart, D.\\,E., Hinton, G.\\,E.\\ \\& Williams, R.\\,J.\\ (1986). Learning representations by back-propagating errors. \\textit{Nature}, 323, 533--536. \\href{https://doi.org/10.1038/323533a0}{doi:10.1038/323533a0}",
    "Werbos, P.\\,J.\\ (1990). Backpropagation through time: What it does and how to do it. \\textit{Proceedings of the IEEE}, 78(10), 1550--1560. \\href{https://doi.org/10.1109/5.58337}{doi:10.1109/5.58337}",
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
    V = Numbers(os.path.join(HERE, "L5_numbers.json"), lang)
    D = Deck(5, "Recurrent networks: RNN, LSTM and GRU for sequences and financial time series" if lang == "en" else
             "Rețele recurente: RNN, LSTM și GRU pentru secvențe și serii de timp financiare", V, lang, heights=H)
    t = D.t
    ro = lang == "ro"
    OWN = t("(own computation)", "(calcul propriu)")
    NB = "\\S"

    def P(key):
        v = V.get(key)
        return ("<0{,}001" if ro else "<0.001") if v < 0.001 else "=" + V.fmt(key, ".3f")

    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Write a recurrent neural network (RNN) as equations, unroll it in time and count its parameters.",
          "Să scrieți o rețea neuronală recurentă (RNN) sub formă de ecuații, să o desfășurați în timp și să-i numărați parametrii."),
        t("Derive backpropagation through time (BPTT) and explain why gradients vanish or explode.",
          "Să derivați propagarea înapoi în timp (BPTT) și să explicați de ce gradienții dispar sau explodează."),
        t("Read the long short-term memory (LSTM) and gated recurrent unit (GRU) cells gate by gate.",
          "Să citiți celulele long short-term memory (LSTM) și gated recurrent unit (GRU) poartă cu poartă."),
        t("Forecast volatility, returns and electricity load with a leakage-free protocol, after the benchmarks: heterogeneous autoregressive (HAR), generalised autoregressive conditional heteroskedasticity (GARCH), autoregressive (AR) and autoregressive with exogenous inputs (ARX) models.",
          "Să prognozați volatilitatea, randamentele și consumul de energie electrică cu un protocol fără scurgeri de informație, după modelele de referință: autoregresiv heterogen (HAR), autoregresiv generalizat cu heteroscedasticitate condiționată (GARCH), autoregresiv (AR) și autoregresiv cu variabile exogene (ARX)."),
        t("Explain why a multilayer perceptron (MLP) on a window of past values is not enough.",
          "Să explicați de ce un perceptron multistrat (MLP) aplicat unei ferestre de valori trecute nu este suficient."),
        t("Decide with a forecast-comparison test when a recurrent network is worth its cost.",
          "Să decideți, cu un test de comparare a prognozelor, cînd merită costul unei rețele recurente."), sep="4pt"))
    abbreviation_frames(D, ABBR)

    D.frame(t("The data: three real sequences", "Datele: trei secvențe reale"),
            table("@{}p{0.25\\textwidth}p{0.29\\textwidth}p{0.22\\textwidth}p{0.17\\textwidth}@{}",
                  [t("Series", "Seria"), t("Source", "Sursa"), t("Period", "Perioada"), t("Task", "Sarcina")],
                  [[t("S\\&P 500 daily open, high, low, close", "S\\&P 500, prețuri zilnice de deschidere, maxim, minim, închidere"), "Yahoo Finance",
                    "@{data.first|date} -- @{data.last|date}", t("volatility, returns", "volatilitate, randamente")],
                   [t("Banca Transilvania (TLV), adjusted close", "Banca Transilvania (TLV), închidere ajustată"),
                    t("End of Day Historical Data (EODHD), Bucharest Stock Exchange (BVB)", "End of Day Historical Data (EODHD), Bursa de Valori București (BVB)"),
                    "@{ret.tlv_first|date} -- @{data.last|date}", t("returns", "randamente")],
                   [t("Romanian hourly load", "Consumul orar din România"),
                    t("European Network of Transmission System Operators for Electricity (ENTSO-E), via energy-charts.info",
                      "Rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică (ENTSO-E), prin energy-charts.info"),
                    "@{load.first|date} -- @{load.last|date}", t("day-ahead load", "consumul pentru ziua următoare")]]) +
            note(t("Time-ordered splits: train to @{data.train_end|date}, validation to @{data.valid_end|date}, test from @{data.test_start|date} (load: train to @{load.train_end|date}, validation @{load.valid_end|date}).",
                   "Împărțiri în ordine cronologică: antrenare pînă la @{data.train_end|date}, validare pînă la @{data.valid_end|date}, test de la @{data.test_start|date} (consum: antrenare pînă la @{load.train_end|date}, validare pînă la @{load.valid_end|date})."),
                 t("Scalers, early stopping and penalties are fitted on the training or validation years only; the test years are used once.",
                   "Scalările, oprirea timpurie și penalizările sînt estimate doar pe anii de antrenare sau de validare; anii de test sînt folosiți o singură dată."),
                 t("Networks trained on the central processing unit (CPU), deterministic for a fixed seed (@{seed} + seed index).",
                   "Rețelele sînt antrenate pe procesorul central (CPU), determinist pentru o sămînță fixă (@{seed} + indicele seminței).")),
            "footnotesize", cite=t("(data/SOURCES.md)", "(data/SOURCES.md)"))

    # ------------------------------------------------------------------ 1. sequences
    D.section(t("Sequences in finance and energy", "Secvențe în finanțe și energie"))
    D.figframe(t("Returns and volatility clustering", "Randamentele și gruparea volatilității"),
               t("the size of the move today says a lot about the size tomorrow; its sign says almost nothing.",
                 "mărimea mișcării de azi spune mult despre mărimea celei de mîine; semnul ei nu spune aproape nimic."),
               "\\[ r_t=100\\ln\\frac{P_t}{P_{t-1}},\\qquad \\hat\\sigma^2_{\\mathrm{GK},t}=\\tfrac12\\Big(\\ln\\frac{H_t}{L_t}\\Big)^2-(2\\ln2-1)\\Big(\\ln\\frac{C_t}{O_t}\\Big)^2 \\]",
               "sp500_vol",
               bullets=[t("Garman--Klass (GK) range estimator from open $O_t$, high $H_t$, low $L_t$, close $C_t$; annualised volatility median @{data.gk_ann_median|.2f}\\,\\%, peak @{data.gk_ann_max|.0f}\\,\\% on @{data.gk_ann_max_date|date}.",
                          "Estimatorul Garman--Klass (GK), din prețurile de deschidere $O_t$, maxim $H_t$, minim $L_t$ și închidere $C_t$; volatilitatea anualizată are mediana @{data.gk_ann_median|.2f}\\,\\% și vîrful @{data.gk_ann_max|.0f}\\,\\%, la @{data.gk_ann_max_date|date}.")],
               height="0.42\\textheight", cite=t("(Garman \\& Klass, 1980; own computation)", "(Garman \\& Klass, 1980; calcul propriu)"))

    D.figframe(t("Autocorrelation of returns and of absolute returns", "Autocorelația randamentelor și a randamentelor absolute"),
               t("autocorrelations of $|r_t|$ stay positive for months; those of $r_t$ are within the noise band.",
                 "autocorelațiile lui $|r_t|$ rămîn pozitive luni întregi; cele ale lui $r_t$ sînt în banda de zgomot."),
               "\\[ \\hat\\rho(k)=\\frac{\\sum_{t=k+1}^{T}(x_t-\\bar x)(x_{t-k}-\\bar x)}{\\sum_{t=1}^{T}(x_t-\\bar x)^2} \\]",
               "acf_abs",
               bullets=[t("$\\hat\\rho_r(1)=@{acf.r1|.3f}$ but $\\hat\\rho_{|r|}(1)=@{acf.a1|.3f}$ and $\\hat\\rho_{|r|}(50)=@{acf.a50|.3f}$; band $\\pm@{acf.band|.3f}$; significant lags out of 50: @{acf.n_sig_r} for $r_t$, @{acf.n_sig_a} for $|r_t|$.",
                          "$\\hat\\rho_r(1)=@{acf.r1|.3f}$, dar $\\hat\\rho_{|r|}(1)=@{acf.a1|.3f}$ și $\\hat\\rho_{|r|}(50)=@{acf.a50|.3f}$; banda $\\pm@{acf.band|.3f}$; decalaje semnificative din 50: @{acf.n_sig_r} pentru $r_t$, @{acf.n_sig_a} pentru $|r_t|$.")],
               job=t("a model with memory is worth trying for risk (volatility, value-at-risk (VaR)), much less for the direction of the price.",
                     "un model cu memorie merită încercat pentru risc (volatilitate, valoarea expusă la risc (VaR)), mult mai puțin pentru direcția prețului."),
               height="0.4\\textheight", cite=OWN)

    D.concept(t("Limited memory of a fixed-window MLP", "Memoria limitată a unui MLP pe fereastră fixă"),
              t("an MLP on the last $W$ observations has separate weights for every lag and a hard memory limit.",
                "un MLP aplicat ultimelor $W$ observații are ponderi separate pentru fiecare decalaj și o limită strictă de memorie."),
              "\\[ \\hat y_{t+1}=g\\big(\\mW[\\mathbf x_{t-W+1};\\dots;\\mathbf x_t]+\\vb\\big),\\qquad \\mW\\in\\R^{m\\times Wd} \\]"
              + t("\\where{$\\mathbf x_t\\in\\R^d$ the inputs of day $t$; $W$ window length; $m$ hidden units; parameters grow linearly with $W$}",
                  "\\where{$\\mathbf x_t\\in\\R^d$ intrările zilei $t$; $W$ lungimea ferestrei; $m$ unitățile ascunse; numărul de parametri crește liniar cu $W$}"),
              it(t(("\\textbf{What we want instead}", "a state $\\vh_t$ that summarises the past", "the same update rule at every step (weight sharing in time, like a convolutional neural network (CNN) in space)",
                    "any sequence length"),
                   ("\\textbf{Ce ne dorim în schimb}", "o stare $\\vh_t$ care rezumă trecutul", "aceeași regulă de actualizare la fiecare pas (ponderi partajate în timp, ca o rețea neuronală convoluțională (CNN) în spațiu)",
                    "orice lungime a secvenței"))),
              t("in production the state is updated with each new observation, without recomputing the whole history.",
                "în producție, starea se actualizează la fiecare observație nouă, fără a recalcula întregul istoric."),
              cite=t("(Bishop \\& Bishop, 2024, ch.~12)", "(Bishop \\& Bishop, 2024, cap.~12)"))

    # ------------------------------------------------------------------ 2. RNN and BPTT
    D.section(t("The recurrent network and backpropagation through time", "Rețeaua recurentă și propagarea înapoi în timp"))
    D.concept(t("The vanilla RNN: one state, updated by the same rule", "RNN simplă: o stare, actualizată după aceeași regulă"),
              t("the new state mixes the old state and the new input through a nonlinearity.",
                "noua stare combină starea veche cu intrarea nouă printr-o funcție neliniară."),
              "\\[ \\vh_t=\\tanh(\\mW_{xh}\\vx_t+\\mW_{hh}\\vh_{t-1}+\\vb_h),\\qquad \\hat y_t=\\mW_{hy}\\vh_t+b_y \\]"
              + t("\\where{$\\vh_t\\in\\R^{n_h}$ hidden state, $\\vh_0=\\mathbf 0$; $\\mW_{xh}\\in\\R^{n_h\\times d}$, $\\mW_{hh}\\in\\R^{n_h\\times n_h}$ shared across $t$; parameters $n_h(d+n_h+1)+n_h+1$, independent of the length $T$}",
                  "\\where{$\\vh_t\\in\\R^{n_h}$ starea ascunsă, $\\vh_0=\\mathbf 0$; $\\mW_{xh}\\in\\R^{n_h\\times d}$, $\\mW_{hh}\\in\\R^{n_h\\times n_h}$ comune pentru toți $t$; parametri $n_h(d+n_h+1)+n_h+1$, independenți de lungimea $T$}"),
              it(t("Unrolled in time it is a deep network of depth $T$ with tied weights.", "Desfășurată în timp, este o rețea adîncă de adîncime $T$, cu ponderi legate."),
                 t("Our volatility model: $d=3$ inputs (log variance, return, absolute return), $n_h=@{vol.hidden}$, window $T=@{vol.W}$ trading days.",
                   "Modelul nostru de volatilitate: $d=3$ intrări (log-varianța, randamentul, randamentul absolut), $n_h=@{vol.hidden}$, fereastră de $T=@{vol.W}$ de zile de tranzacționare."),
                 t("Many-to-one: only $\\hat y_T$ (tomorrow's log variance) enters the loss.", "Many-to-one: doar $\\hat y_T$ (log-varianța de mîine) intră în funcția de pierdere.")),
              t("the hidden state is a learned, nonlinear analogue of the conditional variance of a GARCH model.",
                "starea ascunsă este un analog învățat și neliniar al varianței condiționate dintr-un model GARCH."),
              cite=t("(Rumelhart et al., 1986; Bishop \\& Bishop, 2024, sec.~12.2)", "(Rumelhart et al., 1986; Bishop \\& Bishop, 2024, secț.~12.2)"))

    D.concept(t("Backpropagation through time: a sum over paths", "Propagarea înapoi în timp: o sumă peste drumuri"),
              t("the gradient of a shared weight adds its contribution at every step, each passing through a product of Jacobians.",
                "gradientul unei ponderi comune adună contribuția ei de la fiecare pas, fiecare trecînd printr-un produs de matrice jacobiene."),
              "\\[ \\frac{\\partial\\loss_T}{\\partial\\mW_{hh}}=\\sum_{t=1}^{T}\\frac{\\partial\\loss_T}{\\partial\\vh_T}\\Big(\\prod_{s=t+1}^{T}\\mathbf J_s\\Big)\\frac{\\partial^{+}\\vh_t}{\\partial\\mW_{hh}} \\]\\vspace{-2mm}\\[ \\mathbf J_s=\\frac{\\partial\\vh_s}{\\partial\\vh_{s-1}}=\\mathrm{diag}(1-\\vh_s^{2})\\mW_{hh} \\]"
              + t("\\where{$\\partial^{+}$ the immediate derivative, with $\\vh_{t-1}$ held fixed; $\\mathbf J_s$ Jacobian of one step; the derivation is in the appendix}",
                  "\\where{$\\partial^{+}$ derivata imediată, cu $\\vh_{t-1}$ fixat; $\\mathbf J_s$ matricea jacobiană a unui pas; derivarea completă este în anexă}"),
              it(t("Cost: one backward pass through all $T$ steps, memory $O(T n_h)$.", "Cost: o trecere înapoi prin toți cei $T$ pași, memorie $O(T n_h)$."),
                 t("Truncated BPTT: back-propagate only $K<T$ steps; cheaper, but no gradient signal beyond $K$.",
                   "BPTT trunchiată: propagare înapoi doar pe $K<T$ pași; mai ieftin, dar fără semnal de gradient dincolo de $K$.")),
              t("the product of $T-t$ Jacobians decides what the network can learn from far in the past.",
                "produsul celor $T-t$ matrice jacobiene decide ce poate învăța rețeaua din trecutul îndepărtat."),
              cite=t("(Werbos, 1990)", "(Werbos, 1990)"))

    D.concept(t("Vanishing and exploding gradients", "Gradienți care dispar și gradienți care explodează"),
              t("a product of many matrices shrinks or grows geometrically with the distance in time.",
                "un produs de multe matrice scade sau crește geometric cu distanța în timp."),
              "\\[ \\Big\\|\\prod_{s=t+1}^{T}\\mathbf J_s\\Big\\|\\le\\big(\\gamma\\,\\sigma_{\\max}(\\mW_{hh})\\big)^{T-t},\\qquad \\gamma=\\max|\\tanh'|\\le1 \\]"
              + t("\\where{$\\sigma_{\\max}$ largest singular value; if $\\gamma\\sigma_{\\max}<1$ the bound vanishes geometrically (sufficient condition); a largest eigenvalue above 1 is necessary for explosion}",
                  "\\where{$\\sigma_{\\max}$ cea mai mare valoare singulară; dacă $\\gamma\\sigma_{\\max}<1$, marginea dispare geometric (condiție suficientă); o valoare proprie maximă peste 1 este necesară pentru explozie}"),
              it(t(("\\textbf{Exploding}: clip the gradient norm", "$\\mathbf g\\leftarrow\\mathbf g\\cdot\\min(1,\\,c/\\|\\mathbf g\\|)$; we use $c=1$ (chosen a priori)"),
                   ("\\textbf{Explozia}: limitarea normei gradientului", "$\\mathbf g\\leftarrow\\mathbf g\\cdot\\min(1,\\,c/\\|\\mathbf g\\|)$; folosim $c=1$ (ales a priori)")),
                 t(("\\textbf{Vanishing}: change the architecture", "an additive path through time: the LSTM and GRU cells, next section"),
                   ("\\textbf{Dispariția}: schimbarea arhitecturii", "un drum aditiv prin timp: celulele LSTM și GRU, în secțiunea următoare"))),
              t("a vanilla RNN in practice remembers a few dozen steps: enough for daily volatility, not for weekly patterns in hourly data.",
                "în practică, o RNN simplă își amintește cîteva zeci de pași: suficient pentru volatilitatea zilnică, nu pentru tiparele săptămînale din datele orare."),
              cite=t("(Bengio et al., 1994; Pascanu et al., 2013)", "(Bengio et al., 1994; Pascanu et al., 2013)"))

    # ------------------------------------------------------------------ 3. LSTM and GRU
    D.section(t("Gated cells: LSTM and GRU", "Celule cu porți: LSTM și GRU"))
    D.concept(t("The LSTM cell and its gates", "Celula LSTM și porțile ei"),
              t("a separate cell state $\\mathbf c_t$ is updated additively; three gates decide what to forget, write and show.",
                "o stare separată a celulei, $\\mathbf c_t$, se actualizează aditiv; trei porți decid ce se uită, ce se scrie și ce se arată."),
              "\\[\\begin{aligned} \\mathbf f_t&=\\sigma(\\mW_f[\\vh_{t-1};\\vx_t]+\\vb_f), & \\mathbf i_t&=\\sigma(\\mW_i[\\vh_{t-1};\\vx_t]+\\vb_i),\\\\"
              " \\mathbf o_t&=\\sigma(\\mW_o[\\vh_{t-1};\\vx_t]+\\vb_o), & \\tilde{\\mathbf c}_t&=\\tanh(\\mW_c[\\vh_{t-1};\\vx_t]+\\vb_c),\\\\"
              " \\mathbf c_t&=\\mathbf f_t\\odot\\mathbf c_{t-1}+\\mathbf i_t\\odot\\tilde{\\mathbf c}_t, & \\vh_t&=\\mathbf o_t\\odot\\tanh(\\mathbf c_t) \\end{aligned}\\]"
              + t("\\where{$\\sigma$ logistic function, values in $(0,1)$; $\\odot$ element-wise product; $\\mathbf f,\\mathbf i,\\mathbf o$ forget, input, output gates; parameters $4n_h(n_h+d+1)$}",
                  "\\where{$\\sigma$ funcția logistică, cu valori în $(0,1)$; $\\odot$ produsul element cu element; $\\mathbf f,\\mathbf i,\\mathbf o$ porțile de uitare, intrare și ieșire; parametri $4n_h(n_h+d+1)$}"),
              it(t("Forget gate near 1: the cell keeps its memory; near 0: it resets.", "Poarta de uitare aproape de 1: celula își păstrează memoria; aproape de 0: o resetează."),
                 t("Input gate: how much of the candidate $\\tilde{\\mathbf c}_t$ to write.", "Poarta de intrare: cît din candidatul $\\tilde{\\mathbf c}_t$ se scrie."),
                 t("Output gate: how much of the memory is exposed in $\\vh_t$.", "Poarta de ieșire: cît din memorie apare în $\\vh_t$.")),
              t("the gates are learned switches; on market data they react to shocks (two frames below).",
                "porțile sînt comutatoare învățate; pe datele de piață ele reacționează la șocuri (două diapozitive mai încolo)."),
              cite=t("(Hochreiter \\& Schmidhuber, 1997)", "(Hochreiter \\& Schmidhuber, 1997)"), wl="0.6", wr="0.37")

    D.concept(t("Why the cell state carries gradients far back", "De ce starea celulei duce gradienții departe în trecut"),
              t("along the cell state the Jacobian is a diagonal of forget gates, not a weight matrix squashed by $\\tanh'$.",
                "de-a lungul stării celulei, jacobianul este o diagonală de porți de uitare, nu o matrice de ponderi comprimată de $\\tanh'$."),
              "\\[ \\frac{\\partial\\mathbf c_t}{\\partial\\mathbf c_{t-1}}\\Big|_{\\text{" + t("direct", "direct") + "}}=\\mathrm{diag}(\\mathbf f_t),\\ \\ \\prod_{s=t+1}^{T}\\mathrm{diag}(\\mathbf f_s)\\approx\\mathbf I\\ \\text{" + t("if", "dacă") + "}\\ \\mathbf f_s\\approx\\mathbf 1 \\]"
              + t("\\where{the other paths go through $\\vh_{t-1}$ and the gates; the direct path is what keeps long-range gradients alive}",
                  "\\where{celelalte drumuri trec prin $\\vh_{t-1}$ și prin porți; drumul direct este cel care menține gradienții pe distanțe lungi}"),
              it(t("Forget-gate bias initialised positive (e.g. 1) makes $\\mathbf f_t\\approx\\sigma(1)=0.73$ at the start: memory by default.",
                   "Inițializarea pozitivă a termenului liber al porții de uitare (de exemplu 1) face $\\mathbf f_t\\approx\\sigma(1)=0{,}73$ la început: memorie implicită."),
                 t("On our volatility test period the mean forget gate is @{gates.f_mean|.3f} and the mean input gate @{gates.i_mean|.3f}.",
                   "În perioada de test a volatilității, media porții de uitare este @{gates.f_mean|.3f}, iar a porții de intrare @{gates.i_mean|.3f}."),
                 t("The additive update is the same idea as the residual connection of Lecture 4.", "Actualizarea aditivă este aceeași idee ca a conexiunii reziduale din cursul 4.")),
              t("choose an LSTM or GRU when the useful past is long (a week of hours), a vanilla RNN only for short memories.",
                "alegeți LSTM sau GRU atunci cînd trecutul util este lung (o săptămînă de ore) și o RNN simplă doar pentru memorii scurte."),
              cite=t("(Hochreiter \\& Schmidhuber, 1997)", "(Hochreiter \\& Schmidhuber, 1997)"))

    D.concept(t("The GRU: two gates and no separate cell", "GRU: două porți și nicio stare separată"),
              t("an update gate interpolates between the old state and a candidate; a reset gate decides how much past enters the candidate.",
                "o poartă de actualizare interpolează între starea veche și un candidat; o poartă de resetare decide cît trecut intră în candidat."),
              "\\[\\begin{aligned} \\mathbf z_t&=\\sigma(\\mW_z[\\vh_{t-1};\\vx_t]+\\vb_z), & \\mathbf r_t&=\\sigma(\\mW_r[\\vh_{t-1};\\vx_t]+\\vb_r),\\\\"
              " \\tilde\\vh_t&=\\tanh(\\mW_h[\\mathbf r_t\\odot\\vh_{t-1};\\vx_t]+\\vb_h), & \\vh_t&=(1-\\mathbf z_t)\\odot\\vh_{t-1}+\\mathbf z_t\\odot\\tilde\\vh_t \\end{aligned}\\]"
              + t("\\where{$\\mathbf z_t$ update gate; $\\mathbf r_t$ reset gate; parameters $3n_h(n_h+d+1)$, three quarters of an LSTM with the same $n_h$}",
                  "\\where{$\\mathbf z_t$ poarta de actualizare; $\\mathbf r_t$ poarta de resetare; parametri $3n_h(n_h+d+1)$, trei sferturi dintr-un LSTM cu același $n_h$}"),
              it(t("Same additive path: $\\partial\\vh_t/\\partial\\vh_{t-1}\\supset\\mathrm{diag}(1-\\mathbf z_t)$.", "Același drum aditiv: $\\partial\\vh_t/\\partial\\vh_{t-1}\\supset\\mathrm{diag}(1-\\mathbf z_t)$."),
                 t("LSTM vs GRU: no general winner; compare both on validation data, as we do below.",
                   "LSTM sau GRU: nu există un cîștigător general; comparați-le pe datele de validare, așa cum facem mai jos.")),
              t("fewer parameters make the GRU slightly faster to train and less data-hungry.",
                "numărul mai mic de parametri face GRU puțin mai rapidă la antrenare și mai puțin pretențioasă la date."),
              cite=t("(Cho et al., 2014)", "(Cho et al., 2014)"), wl="0.6", wr="0.37")

    D.figframe(t("Input sensitivity by lag: the networks rely on the previous day and the previous week", "Sensibilitatea la intrări pe decalaje: rețelele se bazează pe ziua și pe săptămîna precedentă"),
               t("sensitivity of the output of the trained load networks to each of the 168 input hours, averaged over 200 test days.",
                 "sensibilitatea ieșirii rețelelor de consum antrenate la fiecare dintre cele 168 de ore de intrare, mediată pe 200 de zile de test."),
               "\\[ S(k)=\\frac1n\\sum_{i=1}^{n}\\Big|\\frac{\\partial\\hat y_{i,12}}{\\partial x_{i,\\,T+1-k}}\\Big| \\]",
               "grad_lag",
               bullets=[t("Reading 7 daily profiles instead of 168 hours shortens the gradient path to 7 steps; at a lag of one week the sensitivity, relative to the last day's maximum, is RNN @{grad.rnn.lag168_rel|.3f}, LSTM @{grad.lstm.lag168_rel|.3f}, GRU @{grad.gru.lag168_rel|.3f}.",
                          "Citirea a 7 profiluri zilnice în locul a 168 de ore scurtează drumul gradientului la 7 pași; la un decalaj de o săptămînă, sensibilitatea raportată la maximul ultimei zile este RNN @{grad.rnn.lag168_rel|.3f}, LSTM @{grad.lstm.lag168_rel|.3f}, GRU @{grad.gru.lag168_rel|.3f}.")],
               height="0.42\\textheight", cite=OWN, tryit=NB + "6")

    D.figframe(t("LSTM gates react to market stress", "Porțile LSTM reacționează la stresul din piață"),
               t("gate activations of the trained volatility LSTM on the test years 2020--2026 (5-day means).",
                 "activările porților LSTM-ului de volatilitate antrenat, în anii de test 2020--2026 (medii pe 5 zile)."),
               None, "lstm_gates",
               bullets=[t("Correlation with today's log variance: forget gate @{gates.f_corr_y|.3f}, input gate @{gates.i_corr_y|.3f}.",
                          "Corelația cu log-varianța de azi: poarta de uitare @{gates.f_corr_y|.3f}, poarta de intrare @{gates.i_corr_y|.3f}.")],
               job=t("gates are a diagnostic, not an explanation: use them to see when the model changes regime, then check with the data.",
                     "porțile sînt un instrument de diagnostic, nu o explicație: folosiți-le pentru a vedea cînd își schimbă modelul regimul, apoi verificați pe date."),
               height="0.42\\textheight", cite=OWN)

    D.frame(t("Sequence architectures and look-ahead leakage", "Arhitecturi pentru secvențe și scurgerea de informație din viitor"), cols(it(
        t(("\\textbf{Many-to-one}", "a window in, one value out: tomorrow's volatility"), ("\\textbf{Many-to-one}", "o fereastră la intrare, o valoare la ieșire: volatilitatea de mîine")),
        t(("\\textbf{Many-to-many (vector output)}", "a week of daily profiles in, 24 hours of tomorrow out: our load model"),
          ("\\textbf{Many-to-many (ieșire vectorială)}", "o săptămînă de profiluri zilnice la intrare, cele 24 de ore de mîine la ieșire: modelul nostru de consum")),
        t(("\\textbf{Encoder--decoder}", "one network reads, another writes a sequence step by step"),
          ("\\textbf{Codificator--decodificator}", "o rețea citește, alta scrie o secvență pas cu pas")), sep="4pt"),
        it(t(("\\textbf{Stacked}", "the states of layer 1 are the inputs of layer 2"), ("\\textbf{Straturi suprapuse}", "stările stratului 1 sînt intrările stratului 2")),
           t(("\\textbf{Bidirectional (BiLSTM)}", "reads the window forward and backward", "fine for classifying a finished text or sequence",
              "a leak in forecasting if the backward pass sees the future"),
             ("\\textbf{Bidirecțional (BiLSTM)}", "citește fereastra înainte și înapoi", "potrivit pentru clasificarea unui text sau a unei secvențe încheiate",
              "o scurgere de informație în prognoză dacă trecerea înapoi vede viitorul")), sep="4pt"), "0.49", "0.48")
        + D.bottom(t("in forecasting every input must be known at the forecast origin: check this before any architecture choice.",
                     "în prognoză, fiecare intrare trebuie să fie cunoscută în momentul prognozei: verificați acest lucru înaintea oricărei alegeri de arhitectură.")),
        "footnotesize", cite=t("(Bishop \\& Bishop, 2024, ch.~12)", "(Bishop \\& Bishop, 2024, cap.~12)"))

    # ------------------------------------------------------------------ 4. protocol and benchmarks
    D.section(t("A protocol for financial time series", "Un protocol pentru serii de timp financiare"))
    D.concept(t("Windows, scaling and splits without leakage", "Ferestre, scalare și împărțiri fără scurgeri de informație"),
              t("every number the model sees must be computable at the forecast origin $t$.",
                "fiecare număr pe care îl vede modelul trebuie să poată fi calculat în momentul prognozei $t$."),
              "\\[ \\mathcal D=\\{(\\mathbf x_{t-W+1:t},\\,y_{t+1})\\},\\qquad \\tilde x=\\frac{x-\\bar x_{\\mathrm{train}}}{s_{\\mathrm{train}}} \\]"
              + t("\\where{$W$ window; $\\bar x_{\\mathrm{train}},s_{\\mathrm{train}}$ mean and standard deviation of the training years only}",
                  "\\where{$W$ fereastra; $\\bar x_{\\mathrm{train}},s_{\\mathrm{train}}$ media și abaterea standard calculate numai pe anii de antrenare}"),
              it(t("S\\&P 500: @{data.n_train|,d} training, @{data.n_valid|,d} validation and @{data.n_test|,d} test days, in this order.",
                   "S\\&P 500: @{data.n_train|,d} de zile de antrenare, @{data.n_valid|,d} de validare și @{data.n_test|,d} de test, în această ordine."),
                 t("Early stopping: stop when the validation loss has not improved for 6 epochs; keep the best weights.",
                   "Oprirea timpurie: ne oprim cînd pierderea de validare nu s-a îmbunătățit timp de 6 epoci; păstrăm cele mai bune ponderi."),
                 t("Seeds: 5 per network (volatility, returns), 3 for load; we report the spread and the mean forecast.",
                   "Semințe: cîte 5 pentru fiecare rețea (volatilitate, randamente), 3 pentru consum; raportăm dispersia și prognoza medie.")),
              t("a random shuffle of days before splitting puts 2020 in the training set and makes any model look good on 2020.",
                "o amestecare aleatoare a zilelor înainte de împărțire pune anul 2020 în setul de antrenare și face orice model să pară bun pe 2020."),
              cite=OWN, tryit=NB + "2")

    D.concept(t("Benchmark models: naive, HAR and GARCH", "Modelele de referință: naiv, HAR și GARCH"),
              t("a network must beat models that econometrics has refined for decades.",
                "o rețea trebuie să depășească modele pe care econometria le-a rafinat de decenii."),
              "\\[\\begin{aligned} \\text{HAR: } y_{t+1}&=\\beta_0+\\beta_d y_t+\\beta_w\\bar y_{t-4:t}+\\beta_m\\bar y_{t-21:t}\\\\"
              " \\text{GARCH(1,1): } \\sigma^2_{t+1}&=\\omega+\\alpha\\,\\varepsilon_t^2+\\beta\\,\\sigma^2_t,\\quad \\varepsilon_t=r_t-\\mu \\end{aligned}\\]"
              + t("\\where{$y_t=\\ln\\hat\\sigma^2_{\\mathrm{GK},t}$; $\\bar y$ means over a week and a month; GARCH with Student-$t$ errors, mapped to $y$ by a linear regression on the training years; naive: $\\hat y_{t+1}=y_t$}",
                  "\\where{$y_t=\\ln\\hat\\sigma^2_{\\mathrm{GK},t}$; $\\bar y$ medii pe o săptămînă și pe o lună; GARCH cu erori Student-$t$, raportat la $y$ printr-o regresie liniară pe anii de antrenare; naiv: $\\hat y_{t+1}=y_t$}"),
              it(t("HAR estimates: $\\hat\\beta_d=@{har.bd|.3f}$, $\\hat\\beta_w=@{har.bw|.3f}$, $\\hat\\beta_m=@{har.bm|.3f}$.",
                   "Estimările modelului HAR: $\\hat\\beta_d=@{har.bd|.3f}$, $\\hat\\beta_w=@{har.bw|.3f}$, $\\hat\\beta_m=@{har.bm|.3f}$."),
                 t("GARCH: $\\hat\\alpha=@{garch.alpha|.3f}$, $\\hat\\beta=@{garch.beta|.3f}$, persistence @{garch.persistence|.3f}, $\\hat\\nu=@{garch.nu|.2f}$.",
                   "GARCH: $\\hat\\alpha=@{garch.alpha|.3f}$, $\\hat\\beta=@{garch.beta|.3f}$, persistență @{garch.persistence|.3f}, $\\hat\\nu=@{garch.nu|.2f}$.")),
              t("HAR is a linear model with three averages; if an LSTM cannot beat it, the LSTM is not worth maintaining.",
                "HAR este un model liniar cu trei medii; dacă un LSTM nu îl poate depăși, LSTM-ul nu merită întreținut."),
              cite=t("(Corsi, 2009; Bollerslev, 1986)", "(Corsi, 2009; Bollerslev, 1986)"))

    D.concept(t("The Diebold--Mariano test of equal predictive accuracy", "Testul Diebold--Mariano pentru acuratețe predictivă egală"),
              t("compare the losses of two forecasts day by day and test whether their mean difference is zero.",
                "comparăm zi de zi pierderile a două prognoze și testăm dacă diferența lor medie este zero."),
              "\\[ d_t=L(e_{1,t})-L(e_{2,t}),\\qquad \\mathrm{DM}=\\frac{\\bar d}{\\sqrt{\\hat\\sigma^2_{\\mathrm{NW}}/n}}\\;\\overset{H_0}{\\sim}\\;\\mathcal N(0,1) \\]"
              + t("\\where{$L$ squared error (volatility) or absolute error (load); $\\hat\\sigma^2_{\\mathrm{NW}}$ Newey--West long-run variance of $d_t$; $\\mathrm{DM}<0$: model 1 better}",
                  "\\where{$L$ eroarea pătratică (volatilitate) sau eroarea absolută (consum); $\\hat\\sigma^2_{\\mathrm{NW}}$ varianța de termen lung Newey--West a lui $d_t$; $\\mathrm{DM}<0$: modelul 1 este mai bun}"),
              it(t("The Diebold--Mariano (DM) test accounts for autocorrelated loss differences, which forecast errors always have.",
                   "Testul Diebold--Mariano (DM) ține seama de autocorelația diferențelor de pierdere, pe care erorile de prognoză o au întotdeauna."),
                 t("Seeds measure the training noise; DM measures the sampling noise of the test period.",
                   "Semințele măsoară zgomotul antrenării; DM măsoară zgomotul de eșantionare al perioadei de test.")),
              t("report a model as better only with a DM $p$-value, and say over which period.",
                "declarați un model mai bun doar cu valoarea $p$ a testului DM și precizați perioada."),
              cite=t("(Diebold \\& Mariano, 1995)", "(Diebold \\& Mariano, 1995)"))

    # ------------------------------------------------------------------ 5. Case A: volatility
    D.section(t("Case study A: S\\&P 500 volatility", "Studiul de caz A: volatilitatea S\\&P 500"))
    D.figframe(t("In the 2020 crash the LSTM adapted faster than HAR", "În criza din 2020, LSTM s-a adaptat mai repede decît HAR"),
               t("one-day-ahead forecasts of annualised volatility in the first test year.",
                 "prognoze pentru ziua următoare ale volatilității anualizate, în primul an de test."),
               "\\[ \\hat\\sigma_{t+1}^{\\mathrm{ann}}=100\\sqrt{252\\,\\exp(\\hat y_{t+1})} \\]",
               "vol_forecast",
               bullets=[t("Crash window (20 Feb -- 30 Apr 2020), mean squared error (MSE) of $\\hat y$: naive @{vol.crash.naive|.3f}, GARCH @{vol.crash.garch|.3f}, HAR @{vol.crash.har|.3f}, LSTM @{vol.crash.lstm|.3f}.",
                          "Fereastra crizei (20 februarie -- 30 aprilie 2020), eroarea pătratică medie (MSE) a lui $\\hat y$: naiv @{vol.crash.naive|.3f}, GARCH @{vol.crash.garch|.3f}, HAR @{vol.crash.har|.3f}, LSTM @{vol.crash.lstm|.3f}.")],
               height="0.42\\textheight", cite=OWN, tryit=NB + "3")

    D.figframe(t("Recurrent networks beat HAR and GARCH on volatility", "Rețelele recurente depășesc HAR și GARCH pe volatilitate"),
               t("test MSE of the log variance over 2020--2026; dots are single seeds.",
                 "MSE pe test a log-varianței, în perioada 2020--2026; punctele sînt semințe individuale."),
               None, "vol_mse",
               bullets=[t("MSE: naive @{vol.naive.mse|.3f}, GARCH @{vol.garch.mse|.3f}, HAR @{vol.har.mse|.3f}; RNN @{vol.rnn.mse_mean|.3f}, LSTM @{vol.lstm.mse_mean|.3f}, GRU @{vol.gru.mse_mean|.3f} (seed means).",
                          "MSE: naiv @{vol.naive.mse|.3f}, GARCH @{vol.garch.mse|.3f}, HAR @{vol.har.mse|.3f}; RNN @{vol.rnn.mse_mean|.3f}, LSTM @{vol.lstm.mse_mean|.3f}, GRU @{vol.gru.mse_mean|.3f} (medii pe semințe).")],
               height="0.42\\textheight", cite=OWN, tryit=NB + "3")

    D.frame(t("Volatility results with tests", "Rezultatele pentru volatilitate, cu teste"), cols(
        table("@{}lrrr@{}", [t("Model", "Modelul"), "MSE", "$R^2$", t("Seeds", "Semințe")],
              [[t("Naive", "Naiv"), "@{vol.naive.mse|.3f}", "@{vol.naive.r2|.3f}", "--"],
               ["GARCH(1,1)-$t$", "@{vol.garch.mse|.3f}", "@{vol.garch.r2|.3f}", "--"],
               ["HAR", "@{vol.har.mse|.3f}", "@{vol.har.r2|.3f}", "--"],
               [t("RNN (mean forecast)", "RNN (prognoza medie)"), "@{vol.rnn.mse_ens|.3f}", "@{vol.rnn.r2_ens|.3f}", "5"],
               [t("LSTM (mean forecast)", "LSTM (prognoza medie)"), "@{vol.lstm.mse_ens|.3f}", "@{vol.lstm.r2_ens|.3f}", "5"],
               [t("GRU (mean forecast)", "GRU (prognoza medie)"), "@{vol.gru.mse_ens|.3f}", "@{vol.gru.r2_ens|.3f}", "5"]]) +
        note(t("$R^2=1-\\mathrm{MSE}/\\widehat{\\Var}(y)$ on the @{data.n_test|,d} test days.", "$R^2=1-\\mathrm{MSE}/\\widehat{\\Var}(y)$ pe cele @{data.n_test|,d} de zile de test.")),
        it(t("LSTM vs HAR: MSE change @{vol.lstm_vs_har_pct|+.1f}\\,\\%; DM $=@{vol.dm_lstm_har.stat|.2f}$, $p" + P("vol.dm_lstm_har.p") + "$.",
             "LSTM față de HAR: variația MSE @{vol.lstm_vs_har_pct|+.1f}\\,\\%; DM $=@{vol.dm_lstm_har.stat|.2f}$, $p" + P("vol.dm_lstm_har.p") + "$."),
           t("LSTM vs GARCH: DM $=@{vol.dm_lstm_garch.stat|.2f}$, $p" + P("vol.dm_lstm_garch.p") + "$.", "LSTM față de GARCH: DM $=@{vol.dm_lstm_garch.stat|.2f}$, $p" + P("vol.dm_lstm_garch.p") + "$."),
           t("LSTM vs naive: MSE change @{vol.lstm_vs_naive_pct|+.1f}\\,\\%.", "LSTM față de modelul naiv: variația MSE @{vol.lstm_vs_naive_pct|+.1f}\\,\\%."),
           t("Best network by mean forecast: @{vol.best_net}; seed spread of the LSTM: @{vol.lstm.mse_min|.3f} to @{vol.lstm.mse_max|.3f}.",
             "Cea mai bună rețea după prognoza medie: @{vol.best_net}; dispersia LSTM pe semințe: între @{vol.lstm.mse_min|.3f} și @{vol.lstm.mse_max|.3f}."), sep="4pt"), "0.5", "0.47")
        + D.bottom(t("the gain is significant on this test period; before replacing HAR in a VaR model, check it on another market and sub-period and keep HAR as the fallback.",
                     "cîștigul este semnificativ pe această perioadă de test; înainte de a înlocui HAR într-un model VaR, verificați-l pe altă piață și pe subperioade și păstrați HAR ca soluție de rezervă.")),
        "footnotesize", cite=OWN)

    # ------------------------------------------------------------------ 6. Case B: returns
    D.section(t("Case study B: forecasting returns with neural networks", "Studiul de caz B: prognoza randamentelor cu rețele neuronale"))
    D.figframe(t("Return forecasts explain almost none of the out-of-sample variance", "Prognozele randamentelor explică foarte puțin din varianța în afara eșantionului"),
               t("next-day returns of the S\\&P 500 and of Banca Transilvania (TLV) from the last 22 returns.",
                 "randamentele de a doua zi ale S\\&P 500 și ale Banca Transilvania (TLV), din ultimele 22 de randamente."),
               "\\[ R^2_{\\mathrm{OOS}}=1-\\frac{\\sum_{t\\in\\mathrm{test}}(r_{t+1}-\\hat r_{t+1})^2}{\\sum_{t\\in\\mathrm{test}}r_{t+1}^2} \\]",
               "returns_r2",
               bullets=[t("S\\&P 500: AR(1) @{ret.SP500.r2_ar|.3f}\\,\\%, LSTM @{ret.SP500.r2_lstm_mean|.3f}\\,\\% (seeds @{ret.SP500.r2_lstm_min|.3f} to @{ret.SP500.r2_lstm_max|.3f}); TLV: AR(1) @{ret.TLV.r2_ar|.3f}\\,\\%, LSTM @{ret.TLV.r2_lstm_mean|.3f}\\,\\%.",
                          "S\\&P 500: AR(1) @{ret.SP500.r2_ar|.3f}\\,\\%, LSTM @{ret.SP500.r2_lstm_mean|.3f}\\,\\% (semințe între @{ret.SP500.r2_lstm_min|.3f} și @{ret.SP500.r2_lstm_max|.3f}); TLV: AR(1) @{ret.TLV.r2_ar|.3f}\\,\\%, LSTM @{ret.TLV.r2_lstm_mean|.3f}\\,\\%.")],
               height="0.42\\textheight", cite=OWN, tryit=NB + "4")

    D.concept(t("Low predictability of returns", "Predictibilitatea redusă a randamentelor"),
              t("in a liquid market the predictable part of returns is tiny compared with their noise.",
                "pe o piață lichidă, partea previzibilă a randamentelor este infimă față de zgomotul lor."),
              "\\[ r_{t+1}=\\mu_t+\\varepsilon_{t+1},\\qquad \\frac{\\Var(\\mu_t)}{\\Var(r_{t+1})}\\approx0 \\]"
              + t("\\where{$\\mu_t$ the conditional mean a model can learn; $\\varepsilon_{t+1}$ news that arrives after $t$}",
                  "\\where{$\\mu_t$ media condiționată pe care o poate învăța un model; $\\varepsilon_{t+1}$ știrile care apar după $t$}"),
              it(t("LSTM sign accuracy on the S\\&P 500 test: @{ret.SP500.hit_lstm|.1f}\\,\\%, against @{ret.SP500.up_share|.1f}\\,\\% up days (always ``up'').",
                   "Acuratețea semnului pentru LSTM pe testul S\\&P 500: @{ret.SP500.hit_lstm|.1f}\\,\\%, față de @{ret.SP500.up_share|.1f}\\,\\% zile de creștere (prognoza ``mereu în sus'')."),
                 t("AR(1) slope estimated on the training years: @{ret.SP500.ar_slope|.3f} (S\\&P 500), @{ret.TLV.ar_slope|.3f} (TLV).",
                   "Panta AR(1) estimată pe anii de antrenare: @{ret.SP500.ar_slope|.3f} (S\\&P 500), @{ret.TLV.ar_slope|.3f} (TLV)."),
                 t("Flexible models fit noise in training; early stopping limits the damage, it cannot create a signal.",
                   "Modelele flexibile învață zgomotul în antrenare; oprirea timpurie limitează paguba, dar nu poate crea un semnal.")),
              t("a vendor's back-test of an ``artificial intelligence (AI) return predictor'' must pass this test on data it has never seen, after costs.",
                "testul retrospectiv al unui furnizor de ``predicții de randamente bazate pe inteligență artificială (AI)'' trebuie să treacă acest test pe date nevăzute, după costuri."),
              cite=OWN)

    # ------------------------------------------------------------------ 7. Case C: load
    D.section(t("Case study C: Romanian electricity load", "Studiul de caz C: consumul de energie electrică din România"))
    D.figframe(t("Load has daily, weekly and yearly cycles", "Consumul are cicluri zilnice, săptămînale și anuale"),
               t("a sequence with long, regular memory: a natural case for recurrent networks.",
                 "o secvență cu memorie lungă și regulată: un candidat natural pentru rețelele recurente."),
               None, "load_series",
               bullets=[t("Hourly load from @{load.first|date} to @{load.last|date}, mean @{load.mean_mw|,d} megawatt (MW); @{load.missing_hours} missing hours stay missing (those days are skipped).",
                          "Consum orar de la @{load.first|date} la @{load.last|date}, în medie @{load.mean_mw|,d} megawatt (MW); cele @{load.missing_hours} de ore lipsă rămîn lipsă (zilele respective sînt omise).")],
               height="0.42\\textheight", cite=OWN)

    D.concept(t("The day-ahead task and its benchmarks", "Sarcina pentru ziua următoare și modelele ei de referință"),
              t("at the end of day $d$ forecast the 24 hourly loads of day $d+1$ from the last week.",
                "la sfîrșitul zilei $d$, prognozăm cele 24 de valori orare ale consumului din ziua $d+1$, pe baza ultimei săptămîni de ore."),
              "\\[ \\hat{\\mathbf y}_{d+1}=\\mathbf y_{d-6}+\\mW_{o}[\\vh_{7};\\mathbf s_{d+1}]+\\vb_o\\in\\R^{24} \\]\\vspace{-2mm}\\[ \\text{ARX: } \\hat{\\mathbf y}_{d+1}=\\mathbf y_{d-6}+\\mathbf B\\,\\mathbf x_{d,1:168}+\\mathbf\\Gamma\\mathbf s_{d+1} \\]"
              + t("\\where{$\\mathbf y_{d-6}$ seasonal naive forecast (the models learn its correction); $\\vh_{7}$ last state after the 7 past days, one step per day (24 hourly loads + weekday); $\\mathbf s_{d+1}$ weekday and season of day $d+1$; ARX by ridge regression (penalty @{load.ridge_alpha}, chosen on the validation year)}",
                  "\\where{$\\mathbf y_{d-6}$ prognoza naivă sezonieră (modelele învață corecția ei); $\\vh_{7}$ ultima stare după cele 7 zile anterioare, cîte un pas pe zi (24 de valori orare + ziua săptămînii); $\\mathbf s_{d+1}$ ziua săptămînii și sezonul zilei $d+1$; ARX prin regresie ridge (penalizarea @{load.ridge_alpha}, aleasă pe anul de validare)}"),
              it(                 t("@{load.n_train|,d} training, @{load.n_valid} validation, @{load.n_test} test days; 3 seeds per network.",
                   "@{load.n_train|,d} de zile de antrenare, @{load.n_valid} de validare, @{load.n_test} de test; 3 semințe pentru fiecare rețea.")),
              t("a supplier submits the 24 hourly volumes before noon; every MW of error is bought or sold at the imbalance price.",
                "un furnizor transmite cele 24 de volume orare înainte de prînz; fiecare MW de eroare se cumpără sau se vinde la prețul de dezechilibru."),
              cite=OWN, tryit=NB + "5", size="scriptsize", wl="0.64", wr="0.33")

    D.figframe(t("Two winter test weeks: forecasts against the observed load", "Două săptămîni de test din iarnă: prognozele față de consumul observat"),
               t("hourly load in two winter test weeks, observed against two forecasts.",
                 "consumul orar în două săptămîni de test din iarnă, observat față de două prognoze."),
               None, "load_fortnight",
               bullets=[t("The seasonal naive copies last week, including its level and its holidays; the networks learn only a correction of it.",
                          "Prognoza naivă sezonieră copiază săptămîna trecută, inclusiv nivelul și sărbătorile ei; rețelele învață doar o corecție a acesteia.")],
               height="0.45\\textheight", cite=OWN)

    D.figframe(t("On load, the linear ARX beats the recurrent networks", "Pe consum, modelul liniar ARX depășește rețelele recurente"),
               t("left: mean absolute percentage error (MAPE) over all test hours; right: mean absolute error (MAE) by hour of day.",
                 "stînga: eroarea procentuală absolută medie (MAPE) pe toate orele de test; dreapta: eroarea absolută medie (MAE) pe ore ale zilei."),
               "\\[ \\mathrm{MAPE}=\\frac{100}{24n}\\sum_{d}\\sum_{h=1}^{24}\\frac{|y_{d,h}-\\hat y_{d,h}|}{y_{d,h}} \\]",
               "load_errors",
               bullets=[t("MAPE (\\%): seasonal naive @{load.snaive|.2f}, naive @{load.yday|.2f}, ridge ARX @{load.ridge|.2f}, RNN @{load.rnn.ens|.2f}, LSTM @{load.lstm.ens|.2f}, GRU @{load.gru.ens|.2f}.",
                          "MAPE (\\%): naiv sezonier @{load.snaive|.2f}, naiv @{load.yday|.2f}, ridge ARX @{load.ridge|.2f}, RNN @{load.rnn.ens|.2f}, LSTM @{load.lstm.ens|.2f}, GRU @{load.gru.ens|.2f}.")],
               height="0.42\\textheight", cite=OWN, tryit=NB + "5")

    D.figframe(t("Longer windows help the LSTM, up to a week", "Ferestrele mai lungi ajută LSTM, pînă la o săptămînă"),
               t("LSTM test error as a function of the input window, same protocol otherwise.",
                 "eroarea LSTM pe test în funcție de fereastra de intrare, cu același protocol în rest."),
               None, "load_window",
               bullets=[t("MAPE (\\%): 24 hours @{load.window.24|.2f}, 72 hours @{load.window.72|.2f}, 168 hours @{load.window.168|.2f}, 336 hours @{load.window.336|.2f}.",
                          "MAPE (\\%): 24 de ore @{load.window.24|.2f}, 72 de ore @{load.window.72|.2f}, 168 de ore @{load.window.168|.2f}, 336 de ore @{load.window.336|.2f}.")],
               height="0.42\\textheight", cite=OWN)

    D.frame(t("Economic value of a more accurate load forecast", "Valoarea economică a unei prognoze mai precise a consumului"), it(
        t(("\\textbf{Significance}", "best network by mean forecast: @{load.best}, MAPE @{load.best_vs_ridge_pct|+.1f}\\,\\% relative to ridge ARX: with @{load.n_train|,d} training days the linear model wins",
           "DM on daily MAE, LSTM vs ridge ARX: @{load.dm_lstm_ridge.stat|.2f}, $p" + P("load.dm_lstm_ridge.p") + "$"),
          ("\\textbf{Semnificația}", "cea mai bună rețea după prognoza medie: @{load.best}, MAPE cu @{load.best_vs_ridge_pct|+.1f}\\,\\% față de ridge ARX: cu @{load.n_train|,d} de zile de antrenare, modelul liniar cîștigă",
           "DM pe MAE zilnic, LSTM față de ridge ARX: @{load.dm_lstm_ridge.stat|.2f}, $p" + P("load.dm_lstm_ridge.p") + "$")),
        t(("\\textbf{In megawatts}", "mean absolute error per hour: seasonal naive @{load.mae_snaive_mw|,d} MW, ridge ARX @{load.mae_ridge_mw|,d} MW, LSTM @{load.mae_lstm_mw|,d} MW",
           "LSTM instead of the seasonal naive: about @{load.lstm_gain_mwh_day|,d} megawatt-hours (MWh) less absolute error per day, at national scale"),
          ("\\textbf{În megawați}", "eroarea absolută medie pe oră: naiv sezonier @{load.mae_snaive_mw|,d} MW, ridge ARX @{load.mae_ridge_mw|,d} MW, LSTM @{load.mae_lstm_mw|,d} MW",
           "LSTM în locul prognozei naive sezoniere: cu aproximativ @{load.lstm_gain_mwh_day|,d} megawați-oră (MWh) mai puțină eroare absolută pe zi, la scară națională")),
        t(("\\textbf{Cost}", "training of all load models and the window study: @{load.seconds|.0f} seconds on a laptop CPU"),
          ("\\textbf{Costul}", "antrenarea tuturor modelelor de consum și studiul ferestrei: @{load.seconds|.0f} de secunde pe procesorul central al unui laptop")), sep="4pt")
        + D.bottom(t("multiply the MWh saved by the imbalance price spread and compare with the cost of the model: here the ridge ARX saves the most for the least; weather inputs would add more (Lecture 13).",
                     "înmulțiți MWh economisiți cu diferența de preț de dezechilibru și comparați cu costul modelului: aici ridge ARX economisește cel mai mult cu cel mai mic cost; variabilele meteo ar aduce un plus (cursul 13).")),
        "footnotesize", cite=OWN)

    # ------------------------------------------------------------------ wrap-up
    D.parts.append("\\section*{" + t("Wrap-up", "Recapitulare") + "}\n")
    D.quiz_frame(t("parameters", "parametrii"),
                 t("How many parameters has an LSTM layer with $d=3$ inputs and $n_h=32$ units?",
                   "Cîți parametri are un strat LSTM cu $d=3$ intrări și $n_h=32$ de unități?"),
                 t("$4n_h(n_h+d+1)=4\\cdot32\\cdot36=4608$ (PyTorch keeps two bias vectors, so it reports $4608+128=4736$).",
                   "$4n_h(n_h+d+1)=4\\cdot32\\cdot36=4608$ (PyTorch păstrează doi vectori de termeni liberi, deci raportează $4608+128=4736$)."))
    D.quiz_frame(t("gradients", "gradienții"),
                 t("Why does a vanilla RNN struggle with the weekly pattern of hourly load?",
                   "De ce are o RNN simplă dificultăți cu tiparul săptămînal al consumului orar?"),
                 t("Over 168 hourly steps the gradient is a product of 168 Jacobians and vanishes; reading 7 daily profiles shortens the path to 7 steps, which is how our load models are built.",
                   "Pe 168 de pași orari, gradientul este un produs de 168 de matrice jacobiene și dispare; citirea a 7 profiluri zilnice scurtează drumul la 7 pași, iar așa sînt construite modelele noastre de consum."))
    D.quiz_frame(t("a vendor's claim", "afirmația unui furnizor"),
                 t("A vendor shows an LSTM with 60\\,\\% sign accuracy on S\\&P 500 returns, trained on shuffled days; what do you ask?",
                   "Un furnizor prezintă un LSTM cu acuratețe de 60\\,\\% a semnului randamentelor S\\&P 500, antrenat pe zile amestecate; ce întrebați?"),
                 t("A time-ordered test on unseen years, against ``always up'' (@{ret.SP500.up_share|.1f}\\,\\%) and after costs; our honest LSTM reaches @{ret.SP500.hit_lstm|.1f}\\,\\%.",
                   "Un test în ordine cronologică, pe ani nevăzuți, față de prognoza ``mereu în sus'' (@{ret.SP500.up_share|.1f}\\,\\%) și după costuri; LSTM-ul nostru, evaluat corect, atinge @{ret.SP500.hit_lstm|.1f}\\,\\%."))
    D.quiz_frame(t("HAR or LSTM", "HAR sau LSTM"),
                 t("The LSTM changes the volatility MSE by @{vol.lstm_vs_har_pct|+.1f}\\,\\% against HAR, with DM $=@{vol.dm_lstm_har.stat|.2f}$; does it replace HAR in the VaR system?",
                   "LSTM schimbă MSE a volatilității cu @{vol.lstm_vs_har_pct|+.1f}\\,\\% față de HAR, cu DM $=@{vol.dm_lstm_har.stat|.2f}$; înlocuiește HAR în sistemul VaR?"),
                 t("The gain is significant here, so the LSTM is a serious candidate; switch only after it holds on another market and sub-period, with monitoring and HAR as the fallback.",
                   "Cîștigul este semnificativ aici, deci LSTM este un candidat serios; schimbați modelul doar după ce cîștigul se confirmă pe altă piață și pe subperioade, cu monitorizare și cu HAR ca soluție de rezervă."))
    D.quiz_frame(t("bidirectional", "bidirecțional"),
                 t("Can you use a bidirectional LSTM to forecast tomorrow's load from a window that ends today?",
                   "Puteți folosi un LSTM bidirecțional pentru a prognoza consumul de mîine dintr-o fereastră care se termină azi?"),
                 t("Yes, if the window ends at the forecast origin: the backward pass then sees only the past; it leaks only if the window contains hours after the origin.",
                   "Da, dacă fereastra se termină în momentul prognozei: trecerea înapoi vede atunci doar trecutul; apare o scurgere doar dacă fereastra conține ore de după acel moment."))

    D.frame(t("Wrap-up", "Recapitulare"), "\\vspace{-2mm}\\begin{multicols}{2}\\scriptsize\\tableofcontents\\end{multicols}\n" + it(
        t("An RNN shares weights in time; BPTT multiplies Jacobians, so gradients vanish or explode; LSTM and GRU add an additive memory path.",
          "O RNN partajează ponderile în timp; BPTT înmulțește matrice jacobiene, deci gradienții dispar sau explodează; LSTM și GRU adaugă un drum aditiv al memoriei."),
        t("On real data: the networks beat HAR on volatility, return forecasts explain almost nothing, and a linear ARX wins on hourly load. Next: text (Lecture 6).",
          "Pe date reale: rețelele depășesc HAR pe volatilitate, prognozele randamentelor nu explică aproape nimic, iar un ARX liniar cîștigă pe consumul orar; urmează textul (cursul 6)."), sep="1pt"))

    D.frame(t("Research from our group: sequences and risk", "Cercetări ale grupului nostru: secvențe și risc"), it(
        t(("\\textbf{Can foundation models manage risk?} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026, Economic Computation and Economic Cybernetics Studies and Research)",
           "pre-trained time-series models used without fine-tuning to forecast VaR and expected shortfall (ES) for Central and Eastern European (CEE) markets",
           "a conformal calibration step makes the tail forecasts reliable; the transformer successors of the LSTM come in Lectures 7 and 13"),
          ("\\textbf{Pot gestiona riscul modelele fundaționale?} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026, Economic Computation and Economic Cybernetics Studies and Research)",
           "modele preantrenate pentru serii de timp, folosite fără reantrenare pentru prognoza VaR și a pierderii medii peste VaR (ES) pe piețele din Europa Centrală și de Est (CEE)",
           "o calibrare conformală face fiabile prognozele de coadă; succesorii transformer ai LSTM sînt prezentați în cursurile 7 și 13")),
        t(("\\textbf{Early warning of ``zombie'' crypto-assets} (Będowska-Sójka, Wójcik \\& Pele, 2026, North American Journal of Economics and Finance)",
           "a machine-learning early-warning system for crypto-assets that lose their market activity"),
          ("\\textbf{Avertizarea timpurie pentru criptoactivele ``zombie''} (Będowska-Sójka, Wójcik \\& Pele, 2026, North American Journal of Economics and Finance)",
           "un sistem de avertizare timpurie bazat pe învățare automată pentru criptoactivele care își pierd activitatea de piață")),
        t(("\\textbf{BitMood} (Conda, Găman, Bâg, Mazurencu-Marinescu-Pele, Pele \\& Härdle, 2026, Digital Finance)",
           "Facebook sentiment and Bitcoin volume: linear regression and a bidirectional LSTM reach comparable test performance, so model complexity does not guarantee better forecasts"),
          ("\\textbf{BitMood} (Conda, Găman, Bâg, Mazurencu-Marinescu-Pele, Pele \\& Härdle, 2026, Digital Finance)",
           "sentimentul de pe Facebook și volumul Bitcoin: regresia liniară și un LSTM bidirecțional au performanțe comparabile pe test, deci complexitatea modelului nu garantează prognoze mai bune")), sep="3pt") +
        note(t("Findings as stated in the papers; numbers on the other slides come from our own computations.",
               "Rezultatele de pe acest diapozitiv sînt cele raportate în lucrări; celelalte cifre din curs provin din calculele noastre.")),
        "footnotesize", cite=t("(Bucharest University of Economic Studies; Romanian Academy, Institute for Economic Forecasting)",
                               "(Academia de Studii Economice din București; Academia Română, Institutul de Prognoză Economică)"))

    D.frame(t("Further reading and resources", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Bishop \\& Bishop (2024)}, ch.~12: sequence models, from RNNs to transformers.",
          "\\textbf{Bishop \\& Bishop (2024)}, cap.~12: modele de secvențe, de la RNN la transformere."),
        t("\\textbf{Pascanu, Mikolov \\& Bengio (2013)}: the clearest analysis of vanishing and exploding gradients.",
          "\\textbf{Pascanu, Mikolov \\& Bengio (2013)}: cea mai clară analiză a gradienților care dispar sau explodează."),
        t("\\textbf{Corsi (2009)}: the HAR model, the benchmark every volatility network must face.",
          "\\textbf{Corsi (2009)}: modelul HAR, reperul cu care trebuie să se măsoare orice rețea de volatilitate."), sep="4pt"),
        it(t("\\textbf{This course}: code, data, notebooks --- ", "\\textbf{Acest curs}: cod, date, notebook-uri Jupyter --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Experiment yourself}: the notebook reproduces the charts of this lecture --- ",
             "\\textbf{Exercițiu practic}: notebook-ul reproduce graficele acestui curs --- ") + "\\href{\\nnrepo/\\nnnb}{notebooks/}.",
           t("\\textbf{Seminar 5}: a GRU for the volatility of another index, against HAR, with a DM test.",
             "\\textbf{Seminarul 5}: un GRU pentru volatilitatea altui indice, față de HAR, cu testul DM."), sep="4pt"), "0.52", "0.45"))

    reference_frames(D, REFS, per_frame=8)
    D.mark_main_end()

    D.frame(t("Appendix: BPTT for the vanilla RNN, step by step", "Anexă: BPTT pentru RNN simplă, pas cu pas"), it(
        t("Pre-activation $\\mathbf a_t=\\mW_{xh}\\vx_t+\\mW_{hh}\\vh_{t-1}+\\vb_h$, state $\\vh_t=\\tanh(\\mathbf a_t)$, loss $\\loss=\\sum_t\\ell_t(\\hat y_t,y_t)$.",
          "Preactivarea $\\mathbf a_t=\\mW_{xh}\\vx_t+\\mW_{hh}\\vh_{t-1}+\\vb_h$, starea $\\vh_t=\\tanh(\\mathbf a_t)$, pierderea $\\loss=\\sum_t\\ell_t(\\hat y_t,y_t)$."),
        t("Backward recursion: $\\boldsymbol\\delta_t\\equiv\\partial\\loss/\\partial\\vh_t=\\mW_{hy}^{\\top}\\partial\\ell_t/\\partial\\hat y_t+\\mW_{hh}^{\\top}\\big[(1-\\vh_{t+1}^2)\\odot\\boldsymbol\\delta_{t+1}\\big]$, with $\\boldsymbol\\delta_{T+1}=\\mathbf 0$.",
          "Recurența înapoi: $\\boldsymbol\\delta_t\\equiv\\partial\\loss/\\partial\\vh_t=\\mW_{hy}^{\\top}\\partial\\ell_t/\\partial\\hat y_t+\\mW_{hh}^{\\top}\\big[(1-\\vh_{t+1}^2)\\odot\\boldsymbol\\delta_{t+1}\\big]$, cu $\\boldsymbol\\delta_{T+1}=\\mathbf 0$."),
        t("Weight gradients: $\\partial\\loss/\\partial\\mW_{hh}=\\sum_t\\big[(1-\\vh_t^2)\\odot\\boldsymbol\\delta_t\\big]\\vh_{t-1}^{\\top}$, $\\partial\\loss/\\partial\\mW_{xh}=\\sum_t\\big[(1-\\vh_t^2)\\odot\\boldsymbol\\delta_t\\big]\\vx_t^{\\top}$.",
          "Gradienții ponderilor: $\\partial\\loss/\\partial\\mW_{hh}=\\sum_t\\big[(1-\\vh_t^2)\\odot\\boldsymbol\\delta_t\\big]\\vh_{t-1}^{\\top}$, $\\partial\\loss/\\partial\\mW_{xh}=\\sum_t\\big[(1-\\vh_t^2)\\odot\\boldsymbol\\delta_t\\big]\\vx_t^{\\top}$."),
        t("Unrolling the recursion gives the product of Jacobians of the main text; many-to-one models have $\\partial\\ell_t/\\partial\\hat y_t=0$ for $t<T$.",
          "Desfășurarea recurenței dă produsul de matrice jacobiene din textul principal; la modelele many-to-one, $\\partial\\ell_t/\\partial\\hat y_t=0$ pentru $t<T$."), sep="3pt"), "footnotesize")
    D.frame(t("Appendix: the models in PyTorch", "Anexă: modelele în PyTorch"),
            code(
            "class RecNet(nn.Module):\n"
            "    def __init__(self, cell, n_in, hidden=32, n_out=1, n_static=0):\n"
            "        super().__init__()\n"
            "        self.rnn = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell](n_in, hidden, batch_first=True)\n"
            "        self.out = nn.Linear(hidden + n_static, n_out)\n"
            "    def forward(self, x, s=None):                 # x: batch x time x features\n"
            "        h, _ = self.rnn(x)\n"
            "        z = h[:, -1] if s is None else torch.cat([h[:, -1], s], 1)\n"
            "        return self.out(z)\n"
            "# training step: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()\n"
            ) + it(t("Full code: lectures/L5/analysis\\_L5.py and the notebook of this lecture.",
                                      "Codul complet: lectures/L5/analysis\\_L5.py și notebook-ul acestui curs.")), "footnotesize")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 5, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
