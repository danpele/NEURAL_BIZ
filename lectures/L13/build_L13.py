r"""
build_L13.py -- Lecture 13 "Deep learning for energy and financial time series; time-series foundation models"
(NN course), English and Romanian from one source.
Writes lecture13.tex / lecture13_ro.tex from the slide text below and the numbers in L13_numbers.json (analysis_L13.py),
compiles both and runs every check of lectures/STYLE.md (nn_build.run_build).
Run:  python3 lectures/L13/analysis_L13.py && python3 lectures/L13/make_quantlets_L13.py && python3 lectures/L13/build_L13.py
Hand-typed numbers are definitions only (24 hours, 0.6745, 252 days, quantile levels, window lengths fixed a priori).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, block, cols, fmt, heat, hier, items,  # noqa: E402
                      reference_frames, run_build, table)

ABBR = {
    "AIC": ("Akaike information criterion", "criteriul informațional Akaike"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "CET": ("Central European Time", "ora Europei Centrale"),
    "CQR": ("conformalised quantile regression", "regresie cuantilă conformală"),
    "DE-LU": ("Germany--Luxembourg bidding zone", "zona de ofertare Germania--Luxemburg"),
    "DeepAR": ("deep autoregressive recurrent network (Salinas et al., 2020)", "rețea recurentă autoregresivă profundă (Salinas et al., 2020)"),
    "DM": ("Diebold--Mariano test", "testul Diebold--Mariano"),
    "DNN": ("deep neural network (here: a multilayer perceptron with 24 outputs)", "rețea neuronală profundă (aici: perceptron multistrat cu 24 de ieșiri)"),
    "ENTSO-E": ("European Network of Transmission System Operators for Electricity", "Rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică"),
    "ES": ("expected shortfall", "pierderea așteptată în coada distribuției (expected shortfall)"),
    "GARCH": ("generalised autoregressive conditional heteroskedasticity", "heteroscedasticitate condiționată autoregresivă generalizată"),
    "HS": ("historical simulation", "simulare istorică"),
    "HU": ("Hungary (bidding zone)", "Ungaria (zonă de ofertare)"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "ISE": ("Fraunhofer Institute for Solar Energy Systems", "Institutul Fraunhofer pentru Sisteme de Energie Solară"),
    "LEAR": ("Lasso-estimated autoregressive model", "model autoregresiv estimat prin Lasso"),
    "LLM": ("large language model", "model lingvistic mare"),
    "LR": ("likelihood ratio", "raportul de verosimilitate"),
    "LSTM": ("long short-term memory network", "rețea cu memorie pe termen scurt și lung"),
    "MAD": ("median absolute deviation", "abaterea absolută mediană"),
    "MAE": ("mean absolute error", "eroarea absolută medie"),
    "MAPE": ("mean absolute percentage error", "eroarea procentuală absolută medie"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "N-BEATS": ("neural basis expansion analysis for time series", "analiza prin expansiune în baze neuronale pentru serii de timp"),
    "OPCOM": ("Romanian electricity and gas market operator", "operatorul pieței de energie electrică și gaze naturale din România"),
    "PatchTST": ("patch time-series transformer", "arhitectura transformer pe petice de serie de timp"),
    "ReLU": ("rectified linear unit", "funcția liniară rectificată"),
    "RMSE": ("root mean squared error", "rădăcina erorii pătratice medii"),
    "RO": ("Romania (bidding zone)", "România (zonă de ofertare)"),
    "TimesFM": ("Time-series Foundation Model (Google Research)", "modelul fundație pentru serii de timp al Google Research"),
    "VaR": ("Value-at-Risk", "valoarea expusă la risc (Value-at-Risk)"),
    "EUR": ("euro", "euro"), "MWh": ("megawatt-hour", "megawatt-oră"), "MW": ("megawatt", "megawatt"),
}
UNITS = {"EUR", "MWh", "MW"}
IGNORE = {"AI", "BIZ", "NEURAL", "BTC", "USD", "FC", "PL", "TPWRS", "LassoLarsIC", "LLM-ES", "BEATS", "PICBE", "NAJEF", "ESWA", "IJF", "OK", "II", "TS", "LU", "DE", "CRPS", "SSRN", "JSTARS", "IEEE", "ICML",
          "NeurIPS", "TMLR", "AAAI", "EUR"}

REFS = [
    "Andrei, A.-V., Pele, D.-T., Cramer, A.-A., Amza, A.\\ \\& Diaconu, D.\\ (2026). Explainable AI-based early warning system for "
    "electricity price spikes: Evidence from the Romanian market. \\textit{Proceedings of the International Conference on Business "
    "Excellence}, 20(1), 4820--4836. \\href{https://doi.org/10.2478/picbe-2026-0352}{doi:10.2478/picbe-2026-0352}",
    "Andrei, A.-V., Pele, D.-T.\\ \\& Cramer, A.-A.\\ (2026). Multi-horizon explainability in energy price forecasting: Disentangling "
    "market dynamics and renewable intermittency in Romania's liberalized electricity market. \\textit{Proceedings of the International "
    "Conference on Business Excellence}, 20(1), 104--117. \\href{https://doi.org/10.2478/picbe-2026-0014}{doi:10.2478/picbe-2026-0014}",
    "Andrei, A.-V., Velev, G., Toma, F.-M., Pele, D.\\,T.\\ \\& Lessmann, S.\\ (2024). Energy price modelling: A comparative evaluation "
    "of four generations of forecasting methods. \\href{https://arxiv.org/abs/2411.03372}{arXiv:2411.03372}",
    "Ansari, A.\\,F.\\ et al.\\ (2024). Chronos: Learning the language of time series. \\textit{Transactions on Machine Learning "
    "Research}. \\href{https://arxiv.org/abs/2403.07815}{arXiv:2403.07815}",
    "Ansari, A.\\,F.\\ et al.\\ (2025). Chronos-2: From univariate to universal forecasting. "
    "\\href{https://arxiv.org/abs/2510.15821}{arXiv:2510.15821}",
    "Będowska-Sójka, B., Wójcik, P.\\ \\& Pele, D.\\,T.\\ (2026). Early warning systems for cryptocurrency markets: Predicting "
    "`zombie' assets using machine learning. \\textit{The North American Journal of Economics and Finance}, 81, 102543. "
    "\\href{https://doi.org/10.1016/j.najef.2025.102543}{doi:10.1016/j.najef.2025.102543}",
    "Christoffersen, P.\\,F.\\ (1998). Evaluating interval forecasts. \\textit{International Economic Review}, 39(4), 841--862. "
    "\\href{https://doi.org/10.2307/2527341}{doi:10.2307/2527341}",
    "Das, A., Kong, W., Sen, R.\\ \\& Zhou, Y.\\ (2024). A decoder-only foundation model for time-series forecasting. "
    "\\textit{Proceedings of the 41st International Conference on Machine Learning}. \\href{https://arxiv.org/abs/2310.10688}{arXiv:2310.10688}",
    "Diebold, F.\\,X.\\ \\& Mariano, R.\\,S.\\ (1995). Comparing predictive accuracy. \\textit{Journal of Business \\& Economic "
    "Statistics}, 13(3), 253--263. \\href{https://doi.org/10.1080/07350015.1995.10524599}{doi:10.1080/07350015.1995.10524599}",
    "Kupiec, P.\\,H.\\ (1995). Techniques for verifying the accuracy of risk measurement models. \\textit{The Journal of Derivatives}, "
    "3(2), 73--84. \\href{https://doi.org/10.3905/jod.1995.407942}{doi:10.3905/jod.1995.407942}",
    "Lago, J., Marcjasz, G., De Schutter, B.\\ \\& Weron, R.\\ (2021). Forecasting day-ahead electricity prices: A review of "
    "state-of-the-art algorithms, best practices and an open-access benchmark. \\textit{Applied Energy}, 293, 116983. "
    "\\href{https://doi.org/10.1016/j.apenergy.2021.116983}{doi:10.1016/j.apenergy.2021.116983}",
    "Nie, Y., Nguyen, N.\\,H., Sinthong, P.\\ \\& Kalagnanam, J.\\ (2023). A time series is worth 64 words: Long-term forecasting "
    "with transformers. \\textit{International Conference on Learning Representations}. \\href{https://arxiv.org/abs/2211.14730}{arXiv:2211.14730}",
    "Oreshkin, B.\\,N., Carpov, D., Chapados, N.\\ \\& Bengio, Y.\\ (2020). N-BEATS: Neural basis expansion analysis for interpretable "
    "time series forecasting. \\textit{International Conference on Learning Representations}. \\href{https://arxiv.org/abs/1905.10437}{arXiv:1905.10437}",
    "Pele, D.\\,T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A.\\,T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S.\\ \\& "
    "Härdle, W.\\,K.\\ (2026). In the beginning was the Word: LLM-VaR and LLM-ES. \\textit{Expert Systems with Applications}, 295, "
    "128676. \\href{https://doi.org/10.1016/j.eswa.2025.128676}{doi:10.1016/j.eswa.2025.128676}",
    "Romano, Y., Patterson, E.\\ \\& Candès, E.\\ (2019). Conformalized quantile regression. \\textit{Advances in Neural Information "
    "Processing Systems}, 32. \\href{https://arxiv.org/abs/1905.03222}{arXiv:1905.03222}",
    "Salinas, D., Flunkert, V., Gasthaus, J.\\ \\& Januschowski, T.\\ (2020). DeepAR: Probabilistic forecasting with autoregressive "
    "recurrent networks. \\textit{International Journal of Forecasting}, 36(3), 1181--1191. "
    "\\href{https://doi.org/10.1016/j.ijforecast.2019.07.001}{doi:10.1016/j.ijforecast.2019.07.001}",
    "Uniejewski, B., Weron, R.\\ \\& Ziel, F.\\ (2018). Variance stabilizing transformations for electricity spot price forecasting. "
    "\\textit{IEEE Transactions on Power Systems}, 33(2), 2219--2229. \\href{https://doi.org/10.1109/TPWRS.2017.2734563}{doi:10.1109/TPWRS.2017.2734563}",
    "Weron, R.\\ (2014). Electricity price forecasting: A review of the state-of-the-art with a look into the future. "
    "\\textit{International Journal of Forecasting}, 30(4), 1030--1081. \\href{https://doi.org/10.1016/j.ijforecast.2014.08.008}{doi:10.1016/j.ijforecast.2014.08.008}",
    "Ziel, F.\\ \\& Weron, R.\\ (2018). Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs.\\ "
    "multivariate modeling frameworks. \\textit{Energy Economics}, 70, 396--420. \\href{https://doi.org/10.1016/j.eneco.2017.12.016}{doi:10.1016/j.eneco.2017.12.016}",
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


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L13_numbers.json"), lang)
    N = V.N
    D = Deck(13, "Deep learning for energy and financial time series; time-series foundation models" if lang == "en" else
             "Deep learning pentru serii de timp din energie și finanțe; modele fundație pentru serii de timp", V, lang, heights=H)
    t = D.t
    f = V.fmt
    ro = lang == "ro"
    NB = "notebook"
    A = N["acc"]
    mae = {k: A[k]["mae"] for k in A}
    best = N["acc_best"]
    NM = {"naive": t("the naive benchmark", "modelul naiv"), "lear": "LEAR", "dnn": "DNN", "nbeats": "N-BEATS", "chronos2": "Chronos-2",
          "chronos2_cov": t("Chronos-2 with covariates", "Chronos-2 cu covariabile"), "bolt": "Chronos-Bolt", "timesfm": "TimesFM",
          "ens": t("the ensemble", "ansamblul")}

    def better(a, b):
        return mae[a] < mae[b]

    def de(x):
        v = int(round(float(x)))
        return "de " if ro and (v % 100 >= 20 or (v % 100 == 0 and v > 0)) else ""

    def pv(key):
        p = V.get(key)
        return ("$<$0{,}001" if ro else "$<$0.001") if p < 0.001 else fmt(p, ".3f", lang)

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Set up a day-ahead forecasting problem without leakage: information set at gate closure, rolling origin, benchmarks.",
          "Să formulați o problemă de prognoză pentru ziua următoare fără scurgeri de informație: setul de informații la închiderea "
          "licitației, originea glisantă, modelele de referință."),
        t("Write and estimate the Lasso-estimated autoregressive model (LEAR), a deep neural network (DNN) with 24 outputs and the neural basis "
          "expansion analysis (N-BEATS) on Romanian prices.",
          "Să scrieți și să estimați modelul autoregresiv estimat prin Lasso (LEAR), o rețea neuronală profundă (DNN) cu 24 de ieșiri și analiza prin "
          "expansiune în baze neuronale (N-BEATS) pe prețurile din România."),
        t("Explain how time-series foundation models, Chronos and the Time-series Foundation Model (TimesFM), forecast without training on our data, and test them.",
          "Să explicați cum prognozează modelele fundație pentru serii de timp, Chronos și modelul fundație al Google Research (TimesFM), fără antrenare pe datele noastre și să le testați."),
        t("Compare forecasts with the Diebold--Mariano test and calibrate probabilistic forecasts with conformal prediction.",
          "Să comparați prognozele cu testul Diebold--Mariano și să calibrați prognozele probabilistice prin predicție conformală."),
        t("Use a foundation model for Value-at-Risk and backtest it on the S\\&P 500 and Bitcoin.",
          "Să folosiți un model fundație pentru valoarea expusă la risc și să o testați retrospectiv pe S\\&P 500 și Bitcoin."),
        sep="4pt"))
    abbreviation_frames(D, ABBR, per_frame=24)

    # ------------------------------------------------------------------ 1. the problem and the data
    D.section(t("The forecasting problem and the data", "Problema de prognoză și datele"))
    D.frame(t("Two business decisions that need a forecast tomorrow morning", "Două decizii de afaceri care au nevoie de o prognoză mîine dimineață"), it(
        (t("\\textbf{Energy: bidding on the day-ahead market}", "\\textbf{Energie: ofertarea pe piața pentru ziua următoare}"),
         t("a supplier, a hydro producer or a battery operator bids for the 24 hours of tomorrow before 12:00 Central European Time (CET) today",
           "un furnizor, un producător hidro sau operatorul unei baterii trimite oferte pentru cele 24 de ore de mîine înainte de ora 12:00, ora Europei Centrale (CET), de azi"),
         t("the price of every hour is set by one auction coupled across Europe; the Romanian electricity and gas market operator (OPCOM) runs it in Romania",
           "prețul fiecărei ore este stabilit de o singură licitație, cuplată la nivel european; în România o organizează operatorul pieței de energie electrică și gaze naturale (OPCOM)"),
         t("an error of 1 EUR/MWh on 100 MW for 24 hours costs 2,400 euro per day", "o eroare de 1 EUR/MWh pe 100 MW timp de 24 de ore costă 2.400 de euro pe zi")),
        (t("\\textbf{Finance: the Value-at-Risk (VaR) of a portfolio}", "\\textbf{Finanțe: valoarea expusă la risc (VaR) a unui portofoliu}"),
         t("a bank reports every day the loss that is exceeded with probability 1\\,\\%; the Basel rules count the exceptions",
           "o bancă raportează zilnic pierderea depășită cu probabilitatea de 1\\,\\%; regulile Basel numără depășirile"),
         t("a quantile forecast, not a point forecast", "este o prognoză de cuantilă, nu una punctuală")),
        (t("\\textbf{The question of this lecture}", "\\textbf{Întrebarea acestui curs}"),
         t("do networks trained on our data, or foundation models pre-trained on millions of other series, beat simple benchmarks?",
           "sînt rețelele antrenate pe datele noastre sau modelele fundație preantrenate pe milioane de alte serii mai bune decît reperele simple?")),
        sep="3pt"), "small", cite=t("(Weron, 2014; Lago et al., 2021)", "(Weron, 2014; Lago et al., 2021)"))
    D.frame(t("Real data only: Romanian power prices and load, the S\\&P 500 and Bitcoin", "Numai date reale: prețurile și consumul de energie din România, S\\&P 500 și Bitcoin"),
            table("@{}p{0.34\\textwidth}p{0.12\\textwidth}p{0.3\\textwidth}p{0.17\\textwidth}@{}",
                  [t("Series", "Serie"), t("Frequency", "Frecvență"), t("Source", "Sursă"), t("Used for", "Folosită pentru")],
                  [[t("Day-ahead prices: Romania (RO), Germany--Luxembourg (DE-LU), Hungary (HU)", "Prețurile pentru ziua următoare: România (RO), Germania--Luxemburg (DE-LU), Ungaria (HU)"),
                    t("hourly", "orară"), t("energy-charts.info, Fraunhofer Institute for Solar Energy Systems (ISE), from exchange and European Network of Transmission System Operators for Electricity (ENTSO-E) data",
                                           "energy-charts.info, Institutul Fraunhofer pentru Sisteme de Energie Solară (ISE), din datele burselor și ale Rețelei europene a operatorilor de transport și de sistem pentru energie electrică (ENTSO-E)"),
                    t("price forecasts", "prognoza prețului")],
                   [t("Romanian load", "Consumul României"), t("hourly", "orară"), "energy-charts.info, ENTSO-E", t("load forecasts", "prognoza consumului")],
                   [t("S\\&P 500, Bitcoin (BTC/USD)", "S\\&P 500, Bitcoin (BTC/USD)"), t("daily", "zilnică"), "Yahoo Finance", t("Value-at-Risk", "valoarea expusă la risc")]]) +
            note(t("Prices from @{data.first|date} to @{data.last|date}: @{data.n_hours|,d} hourly prices; the test year is "
                   "@{split.test_start|date} to @{split.test_end|date}: @{data.n_test_days} complete days.",
                   "Prețuri din @{data.first|date} pînă la @{data.last|date}: @{data.n_hours|,d} de prețuri orare; anul de test este "
                   "@{split.test_start|date} -- @{split.test_end|date}: @{data.n_test_days} de zile complete."),
                 t("Missing hours stay missing; the two clock-change days per year are kept only when all 24 hours exist.",
                   "Orele lipsă rămîn lipsă; cele două zile pe an cu schimbarea orei sînt păstrate doar dacă există toate cele 24 de ore.")),
            "small")
    D.figframe(t("Romanian prices: a crisis level, then a new regime", "Prețurile din România: nivelul de criză, apoi un regim nou"),
               t("a model trained before 2021 has never seen prices like those of 2022; a model trained on 2022 overestimates today.",
                 "un model antrenat înainte de 2021 nu a văzut niciodată prețuri ca în 2022; unul antrenat pe 2022 supraestimează prețurile de azi."),
               "\\[ \\bar P_d=\\frac{1}{24}\\sum_{h=0}^{23}P_{d,h} \\]", "price_history",
               bullets=[t("Mean 2019: @{facts.mean_2019|.2f} EUR/MWh; mean 2022: @{facts.mean_2022|.2f} EUR/MWh; highest day @{facts.max_daily|.2f} "
                          "on @{facts.max_daily_date|date}; correlation of daily means RO--DE-LU: @{facts.corr_ro_de_daily|.2f}.",
                          "Media 2019: @{facts.mean_2019|.2f} EUR/MWh; media 2022: @{facts.mean_2022|.2f} EUR/MWh; ziua maximă: @{facts.max_daily|.2f}, "
                          "pe @{facts.max_daily_date|date}; corelația mediilor zilnice RO--DE-LU: @{facts.corr_ro_de_daily|.2f}.")],
               job=t("choose the training window deliberately: three years, as in Lago et al. (2021), here.",
                     "alegeți fereastra de antrenare în mod deliberat: aici, trei ani, ca la Lago et al. (2021)."), height="0.55\\textheight")
    p26 = N["facts"]["profile"]["2026"]
    D.figframe(t("Solar power has turned the midday peak into a trough", "Producția solară a transformat vîrful de la prînz într-un minim"),
               t("the hourly shape is the strongest regularity a model can learn, and it has changed within six years.",
                 "forma pe ore este cea mai puternică regularitate pe care o poate învăța un model și s-a schimbat în doar șase ani."),
               "\\[ \\bar P_{\\cdot,h}^{(y)}=\\frac{1}{|D_y|}\\sum_{d\\in D_y}P_{d,h} \\]", "hourly_profile",
               bullets=[t(f"2026 (January--October): lowest hour {p26['min_hour']} at @{{facts.profile.2026.min|.2f}} EUR/MWh, highest hour "
                          f"{p26['max_hour']} at @{{facts.profile.2026.max|.2f}} EUR/MWh.",
                          f"2026 (ianuarie--octombrie): ora cea mai ieftină este {p26['min_hour']}, cu @{{facts.profile.2026.min|.2f}} EUR/MWh, iar cea mai scumpă "
                          f"este {p26['max_hour']}, cu @{{facts.profile.2026.max|.2f}} EUR/MWh.")],
               job=t("a forecaster must predict 24 different hours, not one daily number: hence 24 outputs.",
                     "prognoza trebuie să acopere 24 de ore diferite, nu un singur număr pe zi: de aici cele 24 de ieșiri ale modelelor."),
               height="0.55\\textheight")
    D.concept(t("Information set at gate closure: what the model may use", "Setul de informații la închiderea licitației: ce poate folosi modelul"),
              t("a forecast for day $d$ may use only what is known before 12:00 CET on day $d-1$.",
                "o prognoză pentru ziua $d$ poate folosi doar ce se știe înainte de ora 12:00 CET din ziua $d-1$."),
              "\\[ \\hat{\\mathbf y}_d=f_{\\btheta}(\\mathbf x_d),\\quad \\mathbf x_d\\in\\mathcal I_{d-1} \\]"
              "\\[ \\mathbf x_d=(\\mathbf y_{d-1},\\mathbf y_{d-2},\\mathbf y_{d-3},\\mathbf y_{d-7},\\mathbf y^{\\text{DE}}_{d-1},\\boldsymbol\\ell_{d-2},\\mathbf c_d) \\]" +
              "\\where{" + t("$\\mathbf y_d\\in\\R^{24}$ = Romanian prices of day $d$; $\\mathbf y^{\\text{DE}}$ = DE-LU prices; $\\boldsymbol\\ell$ = Romanian load "
                             "(complete two days back); $\\mathbf c_d$ = day-of-week dummies; $\\mathcal I_{d-1}$ = information at gate closure.",
                             "$\\mathbf y_d\\in\\R^{24}$ = prețurile din România în ziua $d$; $\\mathbf y^{\\text{DE}}$ = prețurile DE-LU; $\\boldsymbol\\ell$ = consumul "
                             "României (complet cu două zile în urmă); $\\mathbf c_d$ = variabilele pentru ziua săptămînii; $\\mathcal I_{d-1}$ = informația de la închiderea licitației.") + "}",
              it(t("\\textbf{@{features.n} inputs}, @{features.n_samples|,d} daily samples since 2016.",
                   "\\textbf{@{features.n} de variabile de intrare} și @{features.n_samples|,d} " + de(N["features"]["n_samples"]) + "eșantioane zilnice din 2016."),
                 t("\\textbf{Rolling origin}: re-estimate on the last @{split.win_days|,d} days, forecast the next day, move one day.",
                   "\\textbf{Origine glisantă}: reestimare pe ultimele @{split.win_days|,d} de zile, prognoză pentru ziua următoare, avans cu o zi."),
                 t("\\textbf{Test once}: @{data.n_test_days} days never used for fitting or tuning.",
                   "\\textbf{Testare o singură dată}: @{data.n_test_days} de zile nefolosite nici la estimare, nici la ajustare.")),
              t("load of day $d-1$ is not complete at noon; using it would leak the future into the forecast.",
                "consumul zilei $d-1$ nu este complet la prînz; folosirea lui ar introduce viitorul în prognoză."),
              tryit=NB + " \\S1")
    D.concept(t("Error measures and the benchmark every model must beat", "Măsurile erorii și reperul pe care orice model trebuie să-l depășească"),
              t("report errors in EUR/MWh and relative to the naive forecast, never as a percentage of a price that can be zero.",
                "erorile se raportează în EUR/MWh și relativ la prognoza naivă, niciodată ca procent dintr-un preț care poate fi zero."),
              "\\[ \\mathrm{MAE}=\\frac{1}{24|\\mathcal T|}\\sum_{d\\in\\mathcal T}\\sum_{h}|y_{d,h}-\\hat y_{d,h}|,\\quad "
              "\\mathrm{rMAE}=\\frac{\\mathrm{MAE}}{\\mathrm{MAE}^{\\text{naive}}} \\]"
              "\\[ \\hat y^{\\text{naive}}_{d,h}=\\begin{cases}y_{d-7,h}, & \\text{" + t("Mon, Sat, Sun", "lun., sîm., dum.") + "}\\\\ y_{d-1,h}, & \\text{" +
              t("Tue--Fri", "mar.--vin.") + "}\\end{cases} \\]" +
              "\\where{" + t("$\\mathcal T$ = test days; $\\mathrm{rMAE}<1$ = better than naive; the root mean squared error (RMSE) weights spikes more.",
                             "$\\mathcal T$ = zilele de test; $\\mathrm{rMAE}<1$ = mai bun decît modelul naiv; rădăcina erorii pătratice medii (RMSE) ponderează mai mult vîrfurile.") + "}",
              it(t("\\textbf{Test year}: mean @{data.test_mean|.2f} EUR/MWh, standard deviation @{data.test_std|.2f}, "
                   "range @{data.test_min|.2f} to @{data.test_max|.2f}; @{data.test_neg_hours} negative hours.",
                   "\\textbf{Anul de test}: media @{data.test_mean|.2f} EUR/MWh, abaterea standard @{data.test_std|.2f}, "
                   "între @{data.test_min|.2f} și @{data.test_max|.2f}; @{data.test_neg_hours} " + de(N["data"]["test_neg_hours"]) + "ore cu preț negativ."),
                 t("\\textbf{Naive}: mean absolute error (MAE) @{acc.naive.mae|.2f} EUR/MWh.",
                   "\\textbf{Modelul naiv}: eroarea absolută medie (MAE) @{acc.naive.mae|.2f} EUR/MWh.")),
              t("a model with rMAE above 1 should not be deployed, however deep.", "un model cu rMAE peste 1 nu trebuie folosit, oricît de profund ar fi."),
              tryit=NB + " \\S1")

    # ------------------------------------------------------------------ 2. LEAR and the DNN
    D.section(t("A strong linear benchmark and a deep network", "Un reper liniar puternic și o rețea profundă"))
    D.concept(t("Variance-stabilising transform: spikes and negative prices tamed", "Transformarea care stabilizează varianța: vîrfurile și prețurile negative sînt îmblînzite"),
              t("the area hyperbolic sine behaves like a log for large prices and like the identity near zero, so it accepts negative prices.",
                "sinusul hiperbolic invers se comportă ca un logaritm pentru prețuri mari și ca identitatea în jurul lui zero, deci acceptă prețuri negative."),
              "\\[ z=\\asinh\\!\\Big(\\frac{y-\\tilde m}{\\hat\\sigma}\\Big),\\quad \\hat\\sigma=\\frac{\\mathrm{MAD}}{0.6745},\\quad y=\\tilde m+\\hat\\sigma\\sinh(\\hat z) \\]" +
              "\\where{" + t("$\\tilde m$ = median of the training window; median absolute deviation (MAD) scaled to a standard deviation; $\\hat z$ = forecast on the transformed scale.",
                             "$\\tilde m$ = mediana ferestrei de antrenare; abaterea absolută mediană (MAD) scalată la o abatere standard; $\\hat z$ = prognoza pe scara transformată.") + "}",
              it(t("\\textbf{Fitted on the window only}: no test information enters $\\tilde m$, $\\hat\\sigma$.",
                   "\\textbf{Estimată doar pe fereastră}: nicio informație din test nu intră în $\\tilde m$ și $\\hat\\sigma$."),
                 t("\\textbf{Same transform for LEAR and the DNN}, so differences come from the model, not from the scale.",
                   "\\textbf{Aceeași transformare pentru LEAR și DNN}, astfel încît diferențele vin din model, nu din scară.")),
              t("transform before fitting, invert after forecasting, evaluate in EUR/MWh.",
                "transformați înainte de estimare, inversați după prognoză, evaluați în EUR/MWh."),
              cite="(Uniejewski, Weron \\& Ziel, 2018)", tryit=NB + " \\S2")
    D.concept(t("LEAR: 24 Lasso regressions, one per hour", "LEAR: 24 de regresii Lasso, cîte una pentru fiecare oră"),
              t("a linear model with many lags, where the Lasso keeps only the lags that matter for each hour.",
                "un model liniar cu multe decalaje, în care Lasso păstrează doar decalajele care contează pentru fiecare oră."),
              "\\[ \\hat{\\boldsymbol\\beta}_h=\\argmin_{\\boldsymbol\\beta}\\sum_{d\\in W}\\big(z_{d,h}-\\mathbf x_d^{\\top}\\boldsymbol\\beta\\big)^2+\\lambda_h\\lVert\\boldsymbol\\beta\\rVert_1 \\]" +
              "\\where{" + t("$W$ = last @{split.win_days|,d} days; $\\lambda_h$ chosen by the Akaike information criterion (AIC) on the Lasso path; "
                             "re-estimated every day: @{data.n_test_days} $\\times$ 24 fits.",
                             "$W$ = ultimele @{split.win_days|,d} de zile; $\\lambda_h$ ales prin criteriul informațional Akaike (AIC) pe traiectoria Lasso; "
                             "reestimare zilnică: @{data.n_test_days} $\\times$ 24 de estimări.") + "}",
              it(t("\\textbf{Lasso-estimated autoregressive model (LEAR)}: MAE @{acc.lear.mae|.2f} EUR/MWh, rMAE @{acc.lear.rmae|.3f}.",
                   "\\textbf{Modelul autoregresiv estimat prin Lasso (LEAR)}: MAE @{acc.lear.mae|.2f} EUR/MWh, rMAE @{acc.lear.rmae|.3f}."),
                 t("\\textbf{Why it is hard to beat}: estimated in seconds, interpretable, adapts every day.",
                   "\\textbf{De ce este greu de depășit}: se estimează în cîteva secunde, este interpretabil și se adaptează zilnic.")),
              t("always report LEAR next to any network: it is the open benchmark of the field.",
                "raportați întotdeauna LEAR lîngă orice rețea: este reperul deschis al domeniului."),
              cite="(Lago et al., 2021)", tryit=NB + " \\S2")
    D.concept(t("The DNN: one network, 24 outputs, trained on the same inputs", "Rețeaua DNN: o singură rețea, 24 de ieșiri, aceleași intrări"),
              t("a multilayer perceptron (MLP) forecasts the whole day at once, so hours share what they learn.",
                "un perceptron multistrat (MLP) prognozează întreaga zi deodată, iar orele împart ce învață."),
              "\\[ \\hat{\\mathbf z}_d=\\mathbf W_3\\,\\ReLU\\big(\\mathbf W_2\\,\\ReLU(\\mathbf W_1\\mathbf x_d+\\mathbf b_1)+\\mathbf b_2\\big)+\\mathbf b_3 \\]"
              "\\[ \\loss=\\frac{1}{24|W'|}\\sum_{d\\in W'}\\lVert\\mathbf z_d-\\hat{\\mathbf z}_d\\rVert_1,\\ \\ z_{d,h}=\\asinh\\frac{y_{d,h}-m_d}{s_d} \\]" +
              "\\where{" + t("$m_d$ = median price of day $d-1$; $s_d$ = MAD of the lagged prices + 1; rectified linear unit (ReLU); $W'$ = first 80\\,\\% of the window.",
                             "$m_d$ = mediana prețurilor din ziua $d-1$; $s_d$ = MAD a prețurilor decalate + 1; funcția liniară rectificată (ReLU); $W'$ = primele 80\\,\\% din fereastră.") + "}",
              it(t("\\textbf{Deep neural network (DNN)}: MAE @{acc.dnn.mae|.2f} EUR/MWh, mean of @{dnn.n_seeds} seeds (single seeds @{acc_dnn_seed_min|.2f}--@{acc_dnn_seed_max|.2f}).",
                   "\\textbf{Rețeaua neuronală profundă (DNN)}: MAE @{acc.dnn.mae|.2f} EUR/MWh, media a @{dnn.n_seeds} rulări (individual: @{acc_dnn_seed_min|.2f}--@{acc_dnn_seed_max|.2f})."),
                 t("\\textbf{Size}: @{dnn.hidden1} and @{dnn.hidden2} units, @{dnn.n_params|,d} parameters, dropout @{dnn.dropout}, Adam, early stopping on the last 20\\,\\%.",
                   "\\textbf{Dimensiune}: @{dnn.hidden1} și @{dnn.hidden2} de neuroni, @{dnn.n_params|,d} " + de(N["dnn"]["n_params"]) + "parametri, dropout @{dnn.dropout}, Adam, oprire timpurie pe ultimele 20\\,\\%."),
                 t("\\textbf{Per-day scaling} keeps the network usable when the price level jumps.",
                   "\\textbf{Scalarea pe fiecare zi} păstrează rețeaua utilizabilă cînd nivelul prețului sare.")),
              t("average several seeds; one seed is one draw of a random estimator.",
                "faceți media mai multor rulări; o singură rulare este o extragere dintr-un estimator aleator."),
              cite="(Lago et al., 2021)", tryit=NB + " \\S3", wl="0.55", wr="0.42")
    D.figframe(t("Early stopping picks the epoch on data the network did not train on", "Oprirea timpurie alege epoca pe date pe care rețeaua nu s-a antrenat"),
               t("training loss keeps falling; the validation loss tells when to stop.", "pierderea pe antrenare continuă să scadă; pierderea pe validare arată cînd trebuie oprită antrenarea."),
               "\\[ e^{*}=\\argmin_{e}\\loss_{\\text{val}}(\\btheta_e) \\]", "training_curves",
               bullets=[t("DNN: best epoch @{train.dnn_best_epoch} of @{train.dnn_epochs}; N-BEATS: best epoch @{train.nb_best_epoch} of @{train.nb_epochs} (patience rule).",
                          "DNN: epoca optimă @{train.dnn_best_epoch} din @{train.dnn_epochs}; N-BEATS: epoca optimă @{train.nb_best_epoch} din @{train.nb_epochs} (regula răbdării).")],
               height="0.5\\textheight", tryit=NB + " \\S3")

    # ------------------------------------------------------------------ 3. deep forecasting architectures
    D.section(t("Deep forecasting architectures", "Arhitecturi profunde de prognoză"))
    D.concept(t("A recurrent network that outputs a distribution, not a number", "O rețea recurentă care produce o distribuție, nu un număr"),
              t("the recurrent state summarises the past; the network outputs the parameters of tomorrow's distribution.",
                "starea recurentă rezumă trecutul; rețeaua produce parametrii distribuției de mîine."),
              "\\[ \\mathbf h_t=\\mathrm{LSTM}(\\mathbf h_{t-1},y_{t-1},\\mathbf c_t),\\quad y_t\\mid\\mathbf h_t\\sim\\mathcal N\\big(\\mu(\\mathbf h_t),\\sigma^2(\\mathbf h_t)\\big) \\]"
              "\\[ \\loss=-\\sum_{t}\\log p\\big(y_t\\mid\\mu(\\mathbf h_t),\\sigma(\\mathbf h_t)\\big) \\]" +
              "\\where{" + t("long short-term memory network (LSTM, Lecture 5); $\\mathbf c_t$ = covariates; trained on many related series at once; "
                             "multi-step forecasts by sampling $y_t$ and feeding it back.",
                             "rețea cu memorie pe termen scurt și lung (LSTM, cursul 5); $\\mathbf c_t$ = covariabile; antrenată simultan pe multe serii înrudite; "
                             "prognozele pe mai mulți pași se obțin eșantionînd $y_t$ și reintroducîndu-l în rețea.") + "}",
              it(t("\\textbf{Deep autoregressive recurrent network (DeepAR)}: one global model for many series.",
                   "\\textbf{Rețeaua recurentă autoregresivă profundă (DeepAR)}: un singur model global pentru multe serii."),
                 t("\\textbf{Its idea lives on} in the foundation models below: global training, probabilistic output.",
                   "\\textbf{Ideea ei continuă} în modelele fundație de mai jos: antrenare globală, prognoză probabilistică.")),
              t("useful when you have thousands of related series (stores, meters, assets), not one.",
                "este utilă cînd aveți mii de serii înrudite (magazine, contoare, active), nu una singură."),
              cite="(Salinas et al., 2020)")
    D.concept(t("N-BEATS: stacked blocks that explain the input and forecast the rest", "N-BEATS: blocuri suprapuse care explică intrarea și prognozează restul"),
              t("each block removes what it can explain from the input (backcast) and adds its part to the forecast.",
                "fiecare bloc elimină din intrare ce poate explica (reconstrucția) și adaugă contribuția sa la prognoză."),
              "\\[ \\mathbf h_\\ell=\\mathrm{FC}_\\ell(\\mathbf r_{\\ell-1}),\\ \\ \\hat{\\mathbf r}_\\ell=\\mathbf V^{b}_\\ell\\mathbf h_\\ell,\\ \\ "
              "\\hat{\\mathbf y}_\\ell=\\mathbf V^{f}_\\ell\\mathbf h_\\ell \\]"
              "\\[ \\mathbf r_\\ell=\\mathbf r_{\\ell-1}-\\hat{\\mathbf r}_\\ell,\\qquad \\hat{\\mathbf y}=\\sum_{\\ell=1}^{L}\\hat{\\mathbf y}_\\ell \\]" +
              "\\where{" + t("$\\mathbf r_0$ = last @{nbeats.input} hours, standardised per window; $\\mathrm{FC}_\\ell$ = @{nbeats.layers} layers of width @{nbeats.width}; "
                             "$L$ = @{nbeats.n_blocks} blocks; @{nbeats.n_params|,d} parameters; output 24 hours.",
                             "$\\mathbf r_0$ = ultimele @{nbeats.input} ore, standardizate pe fereastră; $\\mathrm{FC}_\\ell$ = @{nbeats.layers} straturi de lățime @{nbeats.width}; "
                             "$L$ = @{nbeats.n_blocks} blocuri; @{nbeats.n_params|,d} " + de(N["nbeats"]["n_params"]) + "parametri; ieșire de 24 de ore.") + "}",
              it(t("\\textbf{Neural basis expansion analysis (N-BEATS)}: prices only, trained once before the test year on every hourly origin.",
                   "\\textbf{Analiza prin expansiune în baze neuronale (N-BEATS)}: doar prețuri, antrenată o singură dată înainte de anul de test, pe toate originile orare."),
                 t("\\textbf{Test MAE} @{acc.nbeats.mae|.2f} EUR/MWh (@{nbeats.n_seeds} seeds: @{acc_nb_seed_min|.2f} to @{acc_nb_seed_max|.2f}).",
                   "\\textbf{MAE pe test}: @{acc.nbeats.mae|.2f} EUR/MWh (@{nbeats.n_seeds} rulări: între @{acc_nb_seed_min|.2f} și @{acc_nb_seed_max|.2f}).")),
              t("a univariate network does not see German prices or load: compare it with LEAR, which does.",
                "o rețea univariată nu vede prețurile din Germania sau consumul: comparați-o cu LEAR, care le folosește."),
              cite="(Oreshkin et al., 2020)", tryit=NB + " \\S3")
    D.concept(t("Transformers for time series: patches become tokens", "Arhitecturi transformer pentru serii de timp: peticele devin tokeni"),
              t("cut the series into short patches, embed each patch like a word and let attention compare them.",
                "seria se taie în petice scurte, fiecare petic este reprezentat ca un cuvînt, iar mecanismul de atenție le compară."),
              "\\[ \\mathbf p_j=(y_{(j-1)P+1},\\dots,y_{jP}),\\quad \\mathbf e_j=\\mathbf W_{e}\\mathbf p_j+\\mathbf q_j \\]"
              "\\[ \\mathrm{Attn}(\\mathbf Q,\\mathbf K,\\mathbf V)=\\softmax\\!\\Big(\\frac{\\mathbf Q\\mathbf K^{\\top}}{\\sqrt{d_k}}\\Big)\\mathbf V \\]" +
              "\\where{" + t("$P$ = patch length (e.g.\\ 16 or 32 hours); $\\mathbf q_j$ = position encoding; $\\mathbf Q,\\mathbf K,\\mathbf V$ = projections of $\\mathbf e_j$ (Lecture 7).",
                             "$P$ = lungimea peticului (de exemplu 16 sau 32 de ore); $\\mathbf q_j$ = codificarea poziției; $\\mathbf Q,\\mathbf K,\\mathbf V$ = proiecții ale lui $\\mathbf e_j$ (cursul 7).") + "}",
              it(t("\\textbf{Patch time-series transformer (PatchTST)}: 2,048 hours become 128 tokens of 16: attention cost falls 256 times.",
                   "\\textbf{Arhitectura transformer pe petice (PatchTST)}: 2.048 de ore devin 128 de tokeni de cîte 16: costul atenției scade de 256 de ori."),
                 t("\\textbf{Foundation models} below use exactly this design, pre-trained on huge collections of series.",
                   "\\textbf{Modelele fundație} de mai jos folosesc exact acest design, preantrenat pe colecții uriașe de serii.")),
              t("patching is what makes long contexts (weeks of hourly data) affordable.", "peticele fac accesibile contextele lungi (săptămîni de date orare)."),
              cite="(Nie et al., 2023)")

    # ------------------------------------------------------------------ 4. foundation models
    D.section(t("Time-series foundation models", "Modele fundație pentru serii de timp"))
    D.concept(t("A foundation model forecasts a series it has never seen", "Un model fundație prognozează o serie pe care nu a văzut-o niciodată"),
              t("one network is pre-trained on many series; at use time it only reads our context, with no gradient step.",
                "o singură rețea este preantrenată pe multe serii; la utilizare doar citește contextul nostru, fără niciun pas de gradient."),
              "\\[ \\hat q^{\\tau}_{T+k}=g_{\\btheta^{*}}\\big(y_{T-C+1:T}\\big)_{k,\\tau} \\]\\[ \\btheta^{*}=\\argmin_{\\btheta}\\sum_{s\\in\\mathcal S}\\sum_{t,\\tau}\\rho_\\tau\\big(y^{(s)}_t-\\hat q^{\\tau,(s)}_t\\big) \\]" +
              "\\where{" + t("$\\mathcal S$ = pre-training corpus (public and synthetic series); $C$ = context length (@{split.ctx_hours|,d} hours here); "
                             "$\\rho_\\tau$ = quantile (pinball) loss; $k=1,\\dots,24$ = horizon; $\\btheta^{*}$ is frozen.",
                             "$\\mathcal S$ = corpusul de preantrenare (serii publice și sintetice); $C$ = lungimea contextului (aici @{split.ctx_hours|,d} de ore); "
                             "$\\rho_\\tau$ = pierderea cuantilă (pinball); $k=1,\\dots,24$ = orizontul; $\\btheta^{*}$ rămîne fix.") + "}",
              it((t("\\textbf{Three models tested, zero-shot}", "\\textbf{Trei modele testate, fără reantrenare (zero-shot)}"),
                  t("Chronos-2 (Amazon, 2025): covariates and many series together", "Chronos-2 (Amazon, 2025): covariabile și mai multe serii împreună"),
                  t("Chronos-Bolt base (Amazon): direct quantile forecasts", "Chronos-Bolt base (Amazon): prognoze directe de cuantile"),
                  t("Time-series Foundation Model (TimesFM) 2.5 (Google): decoder-only", "modelul fundație pentru serii de timp al Google Research (TimesFM) 2.5: numai decodor"))),
              t("zero-shot means no training cost and no data pipeline, but also no adaptation to our market.",
                "zero-shot înseamnă fără cost de antrenare și fără flux de date, dar și fără adaptare la piața noastră."),
              cite="(Ansari et al., 2024, 2025; Das et al., 2024)", tryit=NB + " \\S4")
    D.concept(t("Chronos: a time series read as a sentence", "Chronos: o serie de timp citită ca o propoziție"),
              t("scale the series, cut its values into bins, and train a language model to predict the next bin.",
                "seria se scalează, valorile se împart în intervale, iar un model de limbaj prezice intervalul următor."),
              "\\[ \\tilde y_t=\\frac{y_t}{s},\\ \\ s=\\frac{1}{C}\\sum_{t}|y_t|,\\qquad v_t=\\mathrm{bin}(\\tilde y_t)\\in\\{1,\\dots,B\\} \\]"
              "\\[ \\loss=-\\sum_{t}\\log p_{\\btheta}(v_{t+1}\\mid v_{1:t}) \\]" +
              "\\where{" + t("$s$ = mean absolute scale of the context; $B$ = 4,096 bins in the original Chronos; the model is a T5 language model; "
                             "Chronos-Bolt and Chronos-2 replace the bins by patches and quantile heads.",
                             "$s$ = scara absolută medie a contextului; $B$ = 4.096 de intervale; modelul este de tip T5; "
                             "Chronos-Bolt și Chronos-2 folosesc petice și ieșiri de cuantile.") + "}",
              it(t("\\textbf{Scaling makes series comparable}: a price in EUR/MWh and a sales count become the same kind of input.",
                   "\\textbf{Scalarea face seriile comparabile}: un preț și un volum de vînzări devin intrări de același tip."),
                 t("\\textbf{Quantile loss} gives probabilistic forecasts directly (next section).",
                   "\\textbf{Pierderea cuantilă} dă direct prognoze probabilistice.")),
              t("the model knows shapes (seasonality, mean reversion), not the Romanian market.",
                "modelul cunoaște forme (sezonalitate, revenire la medie), nu piața românească."),
              cite="(Ansari et al., 2024)")
    rk = sorted(["naive", "lear", "dnn", "nbeats", "bolt", "timesfm", "chronos2", "chronos2_cov", "ens"], key=lambda k: mae[k])
    lo_, hi_ = min(mae.values()), mae["naive"]
    rows = []
    for k in rk:
        rows.append([("\\textbf{" + NM[k] + "}") if k == best else NM[k],
                     heat(A[k]["mae"], lo_, hi_, shown=fmt(A[k]["mae"], ".2f", lang)),
                     fmt(A[k]["rmse"], ".2f", lang), heat(A[k]["rmae"], lo_ / hi_, 1.0, shown=fmt(A[k]["rmae"], ".3f", lang))])
    fm_best = min(("chronos2", "chronos2_cov", "bolt", "timesfm"), key=lambda k: mae[k])
    D.frame(t("Test-year accuracy: all nine forecasts on the same days", "Precizia în anul de test: toate cele nouă prognoze, pe aceleași zile"),
            cols(table("@{}lrrr@{}", [t("Model", "Model"), "MAE", "RMSE", "rMAE"], rows),
                 it(t(f"\\textbf{{Lowest MAE}}: {NM[best]}, @{{acc.{best}.mae|.2f}} EUR/MWh.", f"\\textbf{{Cea mai mică MAE}}: {NM[best]}, @{{acc.{best}.mae|.2f}} EUR/MWh."),
                    t(f"\\textbf{{Best foundation model}}: {NM[fm_best]}, @{{acc.{fm_best}.mae|.2f}}, "
                      + ("better" if better(fm_best, "lear") else "worse") + " than LEAR (@{acc.lear.mae|.2f}) without any training on our data.",
                      f"\\textbf{{Cel mai bun model fundație}}: {NM[fm_best]}, @{{acc.{fm_best}.mae|.2f}}, "
                      + ("mai bun" if better(fm_best, "lear") else "mai slab") + " decît LEAR (@{acc.lear.mae|.2f}), fără nicio antrenare pe datele noastre."),
                    t("\\textbf{All models} beat the naive forecast: rMAE below 1." if all(A[k]["rmae"] < 1 for k in A if k != "naive")
                      else "\\textbf{Not every model} beats the naive forecast: see rMAE.",
                      "\\textbf{Toate modelele} sînt mai bune decît cel naiv: rMAE sub 1." if all(A[k]["rmae"] < 1 for k in A if k != "naive")
                      else "\\textbf{Nu toate modelele} sînt mai bune decît cel naiv: vedeți rMAE."),
                    t("Green = lower error; significance on the next slides.", "Verde = eroare mai mică; semnificația, pe slide-urile următoare."),
                    sep="3pt"), "0.52", "0.45"), "small", cite=t("(own computation, EUR/MWh)", "(calcul propriu, EUR/MWh)"), tryit=NB + " \\S4")
    D.figframe(t("Trained networks versus zero-shot foundation models", "Rețele antrenate față de modele fundație fără reantrenare"),
               t("the same @{data.n_test_hours|,d} hours, the same information set, nine forecasting strategies.",
                 "aceleași @{data.n_test_hours|,d} de ore, același set de informații, nouă strategii de prognoză."),
               None, "mae_models",
               bullets=[t("Ensemble = simple mean of LEAR, DNN and Chronos-2: MAE @{acc.ens.mae|.2f} EUR/MWh.",
                          "Ansamblul = media simplă a LEAR, DNN și Chronos-2: MAE @{acc.ens.mae|.2f} EUR/MWh.")],
               job=t("start with the benchmark and a foundation model: both cost minutes, not weeks.",
                     "începeți cu reperul și cu un model fundație: amîndouă costă minute, nu săptămîni."), height="0.6\\textheight", tryit=NB + " \\S4")
    bh = N["byhour"]
    D.figframe(t("Errors are largest in the evening and around solar hours", "Erorile sînt cele mai mari seara și în orele cu producție solară"),
               t("the average hides hours where all models struggle together.", "media ascunde orele în care toate modelele greșesc împreună."),
               "\\[ \\mathrm{MAE}_h=\\frac{1}{|\\mathcal T|}\\sum_{d\\in\\mathcal T}|y_{d,h}-\\hat y_{d,h}| \\]", "mae_by_hour",
               bullets=[t(f"LEAR: worst hour {bh['lear']['worst_hour']} (@{{byhour.lear.worst|.2f}}), best hour {bh['lear']['best_hour']} (@{{byhour.lear.best|.2f}}); "
                          f"Chronos-2: worst hour {bh['chronos2']['worst_hour']} (@{{byhour.chronos2.worst|.2f}}).",
                          f"LEAR: ora cea mai grea {bh['lear']['worst_hour']} (@{{byhour.lear.worst|.2f}}), cea mai ușoară {bh['lear']['best_hour']} (@{{byhour.lear.best|.2f}}); "
                          f"Chronos-2: ora cea mai grea {bh['chronos2']['worst_hour']} (@{{byhour.chronos2.worst|.2f}}).")],
               height="0.55\\textheight", tryit=NB + " \\S4")
    D.figframe(t("One week of forecasts: level right, spikes smoothed", "O săptămînă de prognoze: nivelul este corect, vîrfurile sînt netezite"),
               t("all three models follow the daily shape; none of them anticipates the largest evening spikes in full.",
                 "toate cele trei modele urmăresc forma zilnică; niciunul nu anticipează complet cele mai mari vîrfuri de seară."),
               None, "forecast_week",
               bullets=[t("Week @{week.start|date} to @{week.end|date}: MAE LEAR @{week.mae.lear|.2f}, DNN @{week.mae.dnn|.2f}, Chronos-2 @{week.mae.chronos2|.2f} EUR/MWh.",
                          "Săptămîna @{week.start|date} -- @{week.end|date}: MAE LEAR @{week.mae.lear|.2f}, DNN @{week.mae.dnn|.2f}, Chronos-2 @{week.mae.chronos2|.2f} EUR/MWh.")],
               height="0.58\\textheight", tryit=NB + " \\S4")
    D.figframe(t("Longer context helps the foundation model, up to a point", "Un context mai lung ajută modelul fundație, pînă la un punct"),
               t("the model sees only the context: one week shows the daily shape, three months also the level.",
                 "modelul vede doar contextul: o săptămînă arată forma zilnică, trei luni arată și nivelul."),
               None, "context_length",
               bullets=[t("Chronos-2 MAE: 168 h @{ctx.168|.2f}, 512 h @{ctx.512|.2f}, 2,048 h @{ctx.2048|.2f}, 8,192 h @{ctx.8192|.2f} EUR/MWh.",
                          "MAE Chronos-2: 168 h @{ctx.168|.2f}, 512 h @{ctx.512|.2f}, 2.048 h @{ctx.2048|.2f}, 8.192 h @{ctx.8192|.2f} EUR/MWh.")],
               job=t("the context length is the main tuning knob of a zero-shot model: choose it on past data.",
                     "lungimea contextului este principalul parametru de ajustare al unui model zero-shot: alegeți-o pe date din trecut."),
               height="0.5\\textheight", tryit=NB + " \\S5")
    cov_better = better("chronos2_cov", "chronos2")
    D.concept(t("Covariates in a foundation model: German prices and Romanian load", "Covariabile într-un model fundație: prețurile din Germania și consumul României"),
              t("Chronos-2 reads several aligned series and lets attention share information across them.",
                "Chronos-2 citește mai multe serii aliniate și lasă mecanismul de atenție să transfere informație între ele."),
              "\\[ \\hat{\\mathbf q}_{T+1:T+24}=g_{\\btheta^{*}}\\big(y_{1:T},\\ x^{\\text{DE}}_{1:T},\\ x^{\\text{HU}}_{1:T},\\ \\ell_{1:T}\\big) \\]" +
              "\\where{" + t("past-only covariates: DE-LU and HU prices and Romanian load up to the forecast origin (their future values are not known at gate closure).",
                             "covariabile cunoscute doar în trecut: prețurile DE-LU și HU și consumul României pînă la originea prognozei (valorile lor viitoare nu se cunosc la închiderea licitației).") + "}",
              it(t("\\textbf{Without covariates}: MAE @{acc.chronos2.mae|.2f}; \\textbf{with covariates}: @{acc.chronos2_cov.mae|.2f} EUR/MWh.",
                   "\\textbf{Fără covariabile}: MAE @{acc.chronos2.mae|.2f}; \\textbf{cu covariabile}: @{acc.chronos2_cov.mae|.2f} EUR/MWh."),
                 t("\\textbf{Covariates " + ("helped" if cov_better else "did not help") + "} here; test, do not assume.",
                   "\\textbf{Covariabilele " + ("au ajutat" if cov_better else "nu au ajutat") + "} aici; testați, nu presupuneți.")),
              t("a covariate known in advance (a weather or load forecast) is worth more than one known only in the past.",
                "o covariabilă cunoscută dinainte (prognoza meteo sau a consumului) valorează mai mult decît una cunoscută doar în trecut."),
              cite="(Ansari et al., 2025)", tryit=NB + " \\S5")
    D.concept(t("Is the difference real? The Diebold--Mariano test on daily losses", "Este diferența reală? Testul Diebold--Mariano pe pierderile zilnice"),
              t("a lower average error can be luck; test whether the daily loss differential has mean zero.",
                "o eroare medie mai mică poate fi noroc; testăm dacă diferența zilnică a pierderilor are media zero."),
              "\\[ \\Delta_d=\\lVert\\mathbf e^{A}_d\\rVert_1-\\lVert\\mathbf e^{B}_d\\rVert_1,\\qquad \\mathrm{DM}=\\frac{\\bar\\Delta}{\\hat\\sigma_\\Delta/\\sqrt{|\\mathcal T|}}\\ \\overset{a}{\\sim}\\ \\mathcal N(0,1) \\]" +
              "\\where{" + t("$\\mathbf e^{A}_d\\in\\R^{24}$ = errors of model A on day $d$; one statistic per pair (multivariate version); "
                             "one-sided $H_1$: B is more accurate than A; $|\\mathcal T|$ = @{dm_n_days} days.",
                             "$\\mathbf e^{A}_d\\in\\R^{24}$ = erorile modelului A în ziua $d$; o statistică pentru fiecare pereche (varianta multivariată); "
                             "$H_1$ unilaterală: B este mai precis decît A; $|\\mathcal T|$ = @{dm_n_days} de zile.") + "}",
              it(t("\\textbf{Diebold--Mariano (DM)}: LEAR vs DNN, $p$ = " + pv("dm.lear__dnn") + "; LEAR vs Chronos-2, $p$ = " + pv("dm.lear__chronos2") + ".",
                   "\\textbf{Diebold--Mariano (DM)}: LEAR față de DNN, $p$ = " + pv("dm.lear__dnn") + "; LEAR față de Chronos-2, $p$ = " + pv("dm.lear__chronos2") + "."),
                 t("\\textbf{Daily, not hourly}: hours of one day share the same information, so they are not independent.",
                   "\\textbf{Pe zile, nu pe ore}: orele aceleiași zile folosesc aceeași informație, deci nu sînt independente.")),
              t("claim that a model is better only with a test on the test period.", "afirmați că un model este mai bun doar pe baza unui test pe perioada de test."),
              cite="(Diebold \\& Mariano, 1995; Ziel \\& Weron, 2018)", tryit=NB + " \\S6")
    D.figframe(t("Which forecast is significantly better? The full chessboard", "Care prognoză este semnificativ mai bună? Tabla de șah completă"),
               t("green cells: the column model is significantly more accurate than the row model.",
                 "celulele verzi: modelul din coloană este semnificativ mai precis decît cel din rînd."),
               None, "dm_pvalues",
               bullets=[t("Ensemble vs LEAR: $p$ = " + pv("dm.lear__ens") + "; ensemble vs Chronos-2: $p$ = " + pv("dm.chronos2__ens") + ".",
                          "Ansamblul față de LEAR: $p$ = " + pv("dm.lear__ens") + "; ansamblul față de Chronos-2: $p$ = " + pv("dm.chronos2__ens") + ".")],
               height="0.62\\textheight", tryit=NB + " \\S6")

    # ------------------------------------------------------------------ 5. probabilistic forecasts
    D.section(t("Probabilistic forecasts and conformal calibration", "Prognoze probabilistice și calibrare conformală"))
    D.concept(t("Quantile forecasts are scored with the pinball loss", "Prognozele de cuantile se evaluează cu pierderea pinball"),
              t("a quantile forecast is good when the right share of outcomes falls below it and the band is narrow.",
                "o prognoză de cuantilă este bună cînd sub ea cade proporția corectă de valori, iar banda este îngustă."),
              "\\[ \\rho_\\tau(u)=u\\,\\big(\\tau-\\mathbb 1\\{u<0\\}\\big),\\quad u=y-\\hat q^{\\tau} \\]"
              "\\[ \\mathrm{PL}=\\frac{1}{9}\\sum_{\\tau\\in\\{0.1,\\dots,0.9\\}}\\frac{1}{24|\\mathcal T|}\\sum_{d,h}\\rho_\\tau\\big(y_{d,h}-\\hat q^{\\tau}_{d,h}\\big) \\]" +
              "\\where{" + t("$\\tau$ = quantile level; the average over the nine deciles approximates the continuous ranked probability score.",
                             "$\\tau$ = nivelul cuantilei; media pe cele nouă decile aproximează scorul probabilistic continuu (continuous ranked probability score).") + "}",
              it((t("\\textbf{Average pinball loss, test year} (EUR/MWh)", "\\textbf{Pierderea pinball medie, anul de test} (EUR/MWh)"),
                  t("Chronos-2 @{pinball.chronos2|.2f}; with covariates @{pinball.chronos2_cov|.2f}", "Chronos-2 @{pinball.chronos2|.2f}; cu covariabile @{pinball.chronos2_cov|.2f}"),
                  t("Chronos-Bolt @{pinball.bolt|.2f}; TimesFM @{pinball.timesfm|.2f}", "Chronos-Bolt @{pinball.bolt|.2f}; TimesFM @{pinball.timesfm|.2f}"),
                  t("LEAR + empirical errors of the last @{split.cal_days} days @{pinball.lear_hs|.2f}", "LEAR + erorile empirice din ultimele @{split.cal_days} de zile @{pinball.lear_hs|.2f}"))),
              t("a battery or a risk desk needs quantiles; score them with pinball, not with MAE.",
                "o baterie sau un departament de risc are nevoie de cuantile; evaluați-le cu pinball, nu cu MAE."),
              tryit=NB + " \\S7")
    D.figframe(t("A week of Chronos-2 quantiles: the band widens where prices jump", "O săptămînă de cuantile Chronos-2: banda se lărgește acolo unde prețurile sar"),
               t("the model is less sure in the evening hours, exactly where the errors are largest.",
                 "modelul este mai puțin sigur în orele de seară, exact acolo unde erorile sînt cele mai mari."),
               None, "chronos_fan", height="0.6\\textheight", tryit=NB + " \\S7")
    c2 = N["cov"]["chronos2"]
    D.concept(t("Split-conformal correction: coverage you can trust", "Corecția conformală: o acoperire pe care vă puteți baza"),
              t("measure how far the outcomes fell outside the raw band in the recent past, and widen (or shrink) the band by that amount.",
                "măsurăm cît de departe de banda brută au căzut valorile în trecutul recent și lărgim (sau îngustăm) banda cu acea valoare."),
              "\\[ s_i=\\max\\big(\\hat q^{\\alpha/2}_i-y_i,\\ y_i-\\hat q^{1-\\alpha/2}_i\\big),\\quad \\hat Q=s_{(\\lceil (n+1)(1-\\alpha)\\rceil)} \\]"
              "\\[ \\hat C_{d,h}=\\big[\\hat q^{\\alpha/2}_{d,h}-\\hat Q,\\ \\hat q^{1-\\alpha/2}_{d,h}+\\hat Q\\big],\\qquad \\Pr\\big(y\\in\\hat C\\big)\\ge 1-\\alpha \\]" +
              "\\where{" + t("$s_i$ = conformity scores on the last @{split.cal_days} days ($n$ = 24 $\\times$ days); $s_{(k)}$ = $k$-th smallest; "
                             "the guarantee holds under exchangeability, only approximately for time series.",
                             "$s_i$ = scorurile de conformitate pe ultimele @{split.cal_days} de zile ($n$ = 24 $\\times$ zile); $s_{(k)}$ = a $k$-a cea mai mică valoare; "
                             "garanția este exactă pentru date interschimbabile și doar aproximativă pentru serii de timp.") + "}",
              it(t("\\textbf{Conformalised quantile regression (CQR)} applied to Chronos-2, 80\\,\\% band: raw coverage @{cov.chronos2.80.cov|.1f}\\,\\%, "
                   "corrected @{cov.chronos2_conf.80.cov|.1f}\\,\\%.",
                   "\\textbf{Regresia cuantilă conformală (CQR)} aplicată lui Chronos-2, banda de 80\\,\\%: acoperire brută @{cov.chronos2.80.cov|.1f}\\,\\%, "
                   "corectată @{cov.chronos2_conf.80.cov|.1f}\\,\\%."),
                 t("\\textbf{Width}: @{cov.chronos2.80.width|.2f} $\\rightarrow$ @{cov.chronos2_conf.80.width|.2f} EUR/MWh.",
                   "\\textbf{Lățimea}: @{cov.chronos2.80.width|.2f} $\\rightarrow$ @{cov.chronos2_conf.80.width|.2f} EUR/MWh.")),
              t("a foundation model plus conformal calibration is a cheap, honest probabilistic forecaster.",
                "un model fundație plus calibrarea conformală formează o prognoză probabilistică ieftină și onestă."),
              cite="(Romano, Patterson \\& Candès, 2019)", tryit=NB + " \\S7")
    D.figframe(t("Conformal calibration moves coverage towards the nominal level", "Calibrarea conformală aduce acoperirea spre nivelul nominal"),
               t("black lines: the nominal level; bars: the share of test hours inside the band.",
                 "liniile negre: nivelul nominal; barele: ponderea orelor de test din interiorul benzii."),
               None, "coverage",
               bullets=[t("90\\,\\% band: raw @{cov.chronos2.90.cov|.1f}\\,\\%, conformal @{cov.chronos2_conf.90.cov|.1f}\\,\\%, LEAR + empirical errors @{cov.lear_hs.90.cov|.1f}\\,\\%; "
                          "TimesFM 80\\,\\%: raw @{cov.timesfm.80.cov|.1f}\\,\\%, conformal @{cov.timesfm_conf.80.cov|.1f}\\,\\%.",
                          "Banda de 90\\,\\%: brut @{cov.chronos2.90.cov|.1f}\\,\\%, conformal @{cov.chronos2_conf.90.cov|.1f}\\,\\%, LEAR + erori empirice @{cov.lear_hs.90.cov|.1f}\\,\\%; "
                          "TimesFM 80\\,\\%: brut @{cov.timesfm.80.cov|.1f}\\,\\%, conformal @{cov.timesfm_conf.80.cov|.1f}\\,\\%.")],
               height="0.52\\textheight", tryit=NB + " \\S7")
    D.figframe(t("Load is easier than price: foundation models forecast it within a few percent", "Consumul este mai ușor de prognozat decît prețul: modelele fundație greșesc cu cîteva procente"),
               t("load is smooth and seasonal, the shapes a pre-trained model knows best.",
                 "consumul este neted și sezonier, exact formele pe care un model preantrenat le cunoaște cel mai bine."),
               "\\[ \\mathrm{MAPE}=\\frac{100}{24|\\mathcal T|}\\sum_{d,h}\\Big|\\frac{\\ell_{d,h}-\\hat\\ell_{d,h}}{\\ell_{d,h}}\\Big| \\]", "load_mape",
               bullets=[t("Mean absolute percentage error (MAPE): last week @{load.week.mape|.2f}\\,\\%, TimesFM @{load.timesfm.mape|.2f}\\,\\%, Chronos-2 "
                          "@{load.chronos2.mape|.2f}\\,\\% (MAE @{load.chronos2.mae|.1f} MW, mean load @{load.mean_mw|,.0f} MW).",
                          "Eroarea procentuală absolută medie (MAPE): săptămîna trecută @{load.week.mape|.2f}\\,\\%, TimesFM @{load.timesfm.mape|.2f}\\,\\%, Chronos-2 "
                          "@{load.chronos2.mape|.2f}\\,\\% (MAE @{load.chronos2.mae|.1f} MW, consum mediu @{load.mean_mw|,.0f} MW).")],
               job=t("percentage errors are fine for load, which is never close to zero.", "erorile procentuale sînt potrivite pentru consum, care nu este niciodată aproape de zero."),
               height="0.48\\textheight", tryit=NB + " \\S8")

    # ------------------------------------------------------------------ 6. VaR
    D.section(t("Foundation models for financial risk", "Modele fundație pentru riscul financiar"))
    D.concept(t("Value-at-Risk is a conditional quantile, backtested by counting violations", "Valoarea expusă la risc este o cuantilă condiționată, testată prin numărarea depășirilor"),
              t("a 1\\,\\% VaR should be breached on about 1\\,\\% of days, and the breaches should not cluster.",
                "o VaR de 1\\,\\% trebuie depășită în aproximativ 1\\,\\% din zile, iar depășirile nu trebuie să vină grupat."),
              "\\[ \\Pr\\big(r_{t+1}<\\mathrm{VaR}^{\\alpha}_{t+1}\\mid\\mathcal F_t\\big)=\\alpha,\\qquad I_{t+1}=\\mathbb 1\\{r_{t+1}<\\mathrm{VaR}^{\\alpha}_{t+1}\\} \\]"
              "\\[ \\mathrm{LR}_{uc}=-2\\ln\\frac{(1-\\alpha)^{n_0}\\alpha^{n_1}}{(1-\\hat\\pi)^{n_0}\\hat\\pi^{n_1}}\\sim\\chi^2_1 \\]" +
              "\\where{" + t("$r_t$ = daily log return (\\%); $n_1$ = violations, $n_0$ = other days, $\\hat\\pi=n_1/(n_0+n_1)$; likelihood ratio (LR) test of Kupiec; "
                             "Christoffersen adds a test of independence of $I_t$.",
                             "$r_t$ = randamentul logaritmic zilnic (\\%); $n_1$ = depășirile, $n_0$ = celelalte zile, $\\hat\\pi=n_1/(n_0+n_1)$; testul raportului de verosimilitate (LR) al lui Kupiec; "
                             "Christoffersen adaugă un test de independență a lui $I_t$.") + "}",
              it((t("\\textbf{Three forecasters}", "\\textbf{Trei metode}"),
                  t("historical simulation (HS): empirical quantile of the last 250 days", "simularea istorică (HS): cuantila empirică a ultimelor 250 de zile"),
                  t("generalised autoregressive conditional heteroskedasticity (GARCH): GARCH(1,1) with Student-$t$ errors, re-estimated every 21 days",
                    "heteroscedasticitate condiționată autoregresivă generalizată (GARCH): GARCH(1,1) cu erori Student-$t$, reestimat la fiecare 21 de zile"),
                  t("Chronos-2 quantiles from the last 1,024 returns, zero-shot", "cuantilele Chronos-2 din ultimele 1.024 de randamente, fără reantrenare"))),
              t("the Basel traffic light turns on the number of 1\\,\\% exceptions in 250 days.",
                "semaforul Basel depinde de numărul de depășiri ale VaR de 1\\,\\% în 250 de zile."),
              cite="(Kupiec, 1995; Christoffersen, 1998)", tryit=NB + " \\S9")
    D.concept(t("A one-sided conformal shift turns a raw quantile into a calibrated VaR", "O deplasare conformală unilaterală transformă cuantila brută într-o VaR calibrată"),
              t("if the model's quantile was breached too often in the last year, lower it by exactly the amount that would have fixed it.",
                "dacă cuantila modelului a fost depășită prea des în ultimul an, o coborîm exact cu valoarea care ar fi corectat-o."),
              "\\[ s_i=\\hat q^{\\alpha}_i-r_i,\\qquad c_t=s_{(\\lceil (n+1)(1-\\alpha)\\rceil)}\\ \\text{" + t("over the last", "pe ultimele") + " } n=250 \\]"
              "\\[ \\widehat{\\mathrm{VaR}}^{\\alpha}_{t+1}=\\hat q^{\\alpha}_{t+1}-c_t \\]" +
              "\\where{" + t("$\\hat q^{\\alpha}$ = raw Chronos-2 quantile; $c_t>0$ widens the VaR, $c_t<0$ narrows it; the shift uses only past forecasts and returns.",
                             "$\\hat q^{\\alpha}$ = cuantila brută Chronos-2; $c_t>0$ mărește VaR, $c_t<0$ o micșorează; deplasarea folosește doar prognoze și randamente din trecut.") + "}",
              it(t("\\textbf{S\\&P 500, 1\\,\\% level}: raw Chronos-2 @{var.SP500.ch_01.viol} violations, conformal @{var.SP500.chc_01.viol}, expected @{var.SP500.exp_01|.1f}.",
                   "\\textbf{S\\&P 500, nivelul de 1\\,\\%}: Chronos-2 brut @{var.SP500.ch_01.viol} depășiri, conformal @{var.SP500.chc_01.viol}, așteptate @{var.SP500.exp_01|.1f}."),
                 t("\\textbf{Same idea} as the energy bands, applied to one tail.", "\\textbf{Aceeași idee} ca la benzile din energie, aplicată unei singure cozi.")),
              t("recalibration is a model change: document it and backtest it out of sample.",
                "recalibrarea este o schimbare de model: documentați-o și testați-o pe date noi."),
              cite="(Romano et al., 2019)", tryit=NB + " \\S9")
    D.figframe(t("S\\&P 500: the calibrated foundation-model VaR follows volatility regimes", "S\\&P 500: VaR calibrată a modelului fundație urmărește regimurile de volatilitate"),
               t("both lines widen in March 2020 and in April 2025; red dots are the days the conformal Chronos-2 VaR was breached.",
                 "ambele linii se lărgesc în martie 2020 și în aprilie 2025; punctele roșii sînt zilele în care VaR conformală Chronos-2 a fost depășită."),
               None, "var_sp500", height="0.6\\textheight", tryit=NB + " \\S9")
    vrows = []
    for tic, lab in (("SP500", "S\\&P 500"), ("BTCUSD", "Bitcoin")):
        for m_, mn in (("hs", t("HS (250 days)", "HS (250 de zile)")), ("garch", "GARCH(1,1)-$t$"), ("ch", t("Chronos-2, raw", "Chronos-2, brut")),
                       ("chc", t("Chronos-2 + conformal", "Chronos-2 + conformal"))):
            r1, r5 = N["var"][tic][f"{m_}_01"], N["var"][tic][f"{m_}_05"]
            vrows.append([lab if m_ == "hs" else "", mn, f"{r1['viol']}", heat(r1["kupiec_p"], 0, 0.1, good_low=False, shown=fmt(r1["kupiec_p"], ".3f", lang)),
                          fmt(r1["chris_p"], ".3f", lang), f"{r5['viol']}", heat(r5["kupiec_p"], 0, 0.1, good_low=False, shown=fmt(r5["kupiec_p"], ".3f", lang)),
                          fmt(r5["tick"], ".4f", lang)])
    D.frame(t("Backtests, January 2020 to October 2026: S\\&P 500 and Bitcoin", "Testarea retrospectivă, ianuarie 2020 -- octombrie 2026: S\\&P 500 și Bitcoin"),
            table("@{}llrrrrrr@{}", [t("Asset", "Activ"), t("Model", "Model"), t("Viol.\\ 1\\,\\%", "Dep.\\ 1\\,\\%"), "$p_{uc}$", "$p_{ind}$",
                                     t("Viol.\\ 5\\,\\%", "Dep.\\ 5\\,\\%"), "$p_{uc}$", t("Tick loss 5\\,\\%", "Pierdere tick 5\\,\\%")], vrows, size="scriptsize") +
            note(t("Expected violations: S\\&P 500 @{var.SP500.exp_01|.1f} (1\\,\\%) and @{var.SP500.exp_05|.1f} (5\\,\\%) over @{var.SP500.n|,d} days; "
                   "Bitcoin @{var.BTCUSD.exp_01|.1f} and @{var.BTCUSD.exp_05|.1f} over @{var.BTCUSD.n|,d} days (traded every day).",
                   "Depășiri așteptate: S\\&P 500 @{var.SP500.exp_01|.1f} (1\\,\\%) și @{var.SP500.exp_05|.1f} (5\\,\\%) în @{var.SP500.n|,d} " + de(N["var"]["SP500"]["n"]) + "zile; "
                   "Bitcoin @{var.BTCUSD.exp_01|.1f} și @{var.BTCUSD.exp_05|.1f} în @{var.BTCUSD.n|,d} " + de(N["var"]["BTCUSD"]["n"]) + "zile (tranzacționat zilnic)."),
                 t("Green $p$-value: coverage not rejected at 10\\,\\%; red: rejected. Tick loss = pinball loss at $\\tau=0.05$ (lower is better).",
                   "Valoare $p$ verde: acoperirea nu este respinsă la 10\\,\\%; roșie: este respinsă. Pierderea tick = pierderea pinball la $\\tau=0{,}05$ (mai mică este mai bine).")),
            "small", cite=t("(own computation; Kupiec, 1995; Christoffersen, 1998)", "(calcul propriu; Kupiec, 1995; Christoffersen, 1998)"), tryit=NB + " \\S9")
    D.frame(t("Language models and time-series models: two routes to the same quantile", "Modele de limbaj și modele de serii de timp: două căi spre aceeași cuantilă"), it(
        (t("\\textbf{Large language model (LLM) route} (Lecture 12)", "\\textbf{Calea modelului lingvistic mare (LLM)} (cursul 12)"),
         t("the returns are written as text in a prompt; a general model answers with a number", "randamentele sînt scrise ca text într-un prompt; un model general răspunde cu un număr"),
         t("LLM-VaR and, for the expected shortfall (ES), LLM-ES, of our group: zero-shot, no fine-tuning (Pele et al., 2026)",
           "LLM-VaR și, pentru pierderea așteptată în coada distribuției (ES), LLM-ES, ale grupului nostru: zero-shot, fără ajustare fină (Pele et al., 2026)")),
        (t("\\textbf{Time-series foundation model route} (this lecture)", "\\textbf{Calea modelului fundație pentru serii de timp} (acest curs)"),
         t("numbers enter as scaled patches; the output is a set of quantiles", "numerele intră ca petice scalate; ieșirea este un set de cuantile"),
         t("cheaper per forecast, can be run locally, calibrated here with a conformal shift", "mai ieftin pe prognoză, rulează local, calibrat aici printr-o deplasare conformală")),
        (t("\\textbf{Common rules}", "\\textbf{Reguli comune}"),
         t("backtest out of sample, count violations, test independence, compare with GARCH and HS", "testați pe date noi, numărați depășirile, testați independența, comparați cu GARCH și HS")),
        sep="3pt"), "small")

    # ------------------------------------------------------------------ wrap-up
    D.parts.append("\\section*{" + t("Wrap-up", "Recapitulare") + "}\n")
    D.quiz_frame(t("leakage", "scurgerea de informație"),
                 t("A colleague adds the actual Romanian load of day $d$ to the inputs of the price forecast for day $d$ and the MAE falls; should you deploy the model?",
                   "Un coleg adaugă consumul efectiv al României din ziua $d$ la intrările prognozei de preț pentru ziua $d$, iar MAE scade; puneți modelul în producție?"),
                 t("No: the actual load of day $d$ is unknown at 12:00 on $d-1$; use the operator's load forecast or lags available at gate closure.",
                   "Nu: consumul efectiv din ziua $d$ nu se cunoaște la ora 12:00 din ziua $d-1$; folosiți prognoza operatorului sau decalaje disponibile la închiderea licitației."))
    D.quiz_frame(t("conformal width", "lățimea conformală"),
                 t("The raw 80\\,\\% Chronos-2 band covered @{cov.chronos2.80.cov|.1f}\\,\\% of test hours; what does the conformal step do to its width, and why?",
                   "Banda brută de 80\\,\\% a lui Chronos-2 a acoperit @{cov.chronos2.80.cov|.1f}\\,\\% din orele de test; ce face pasul conformal cu lățimea ei și de ce?"),
                 t("It changes it by $\\hat Q$, the $(1-\\alpha)$ quantile of recent out-of-band distances: wider if the band was too narrow, narrower if too wide "
                   "(here @{cov.chronos2.80.width|.2f} $\\rightarrow$ @{cov.chronos2_conf.80.width|.2f} EUR/MWh).",
                   "O modifică cu $\\hat Q$, cuantila de nivel $(1-\\alpha)$ a distanțelor recente în afara benzii: o lărgește dacă banda era prea îngustă și o îngustează dacă era prea largă "
                   "(aici @{cov.chronos2.80.width|.2f} $\\rightarrow$ @{cov.chronos2_conf.80.width|.2f} EUR/MWh)."))
    D.quiz_frame(t("Kupiec", "testul Kupiec"),
                 t("Over @{var.SP500.n|,d} days a 1\\,\\% VaR had 3 times the expected number of violations; is the Kupiec $p$-value large or small?",
                   "În @{var.SP500.n|,d} " + de(N["var"]["SP500"]["n"]) + "zile, o VaR de 1\\,\\% a avut de 3 ori numărul așteptat de depășiri; valoarea $p$ a testului Kupiec este mare sau mică?"),
                 t("Small: $\\hat\\pi$ is far from $\\alpha$, so $\\mathrm{LR}_{uc}$ is large and coverage is rejected; the model underestimates risk.",
                   "Mică: $\\hat\\pi$ este departe de $\\alpha$, deci $\\mathrm{LR}_{uc}$ este mare și acoperirea se respinge; modelul subestimează riscul."))
    D.frame(t("Wrap-up", "Recapitulare"), it(
        t(f"Benchmarks first: naive @{{acc.naive.mae|.2f}}, LEAR @{{acc.lear.mae|.2f}} EUR/MWh; lowest MAE: {NM[best]} (@{{acc.{best}.mae|.2f}}).",
          f"Mai întîi reperele: naiv @{{acc.naive.mae|.2f}}, LEAR @{{acc.lear.mae|.2f}} EUR/MWh; cea mai mică MAE: {NM[best]} (@{{acc.{best}.mae|.2f}})."),
        t("Foundation models forecast prices and load zero-shot; the context length and covariates are their tuning knobs.",
          "Modelele fundație prognozează prețul și consumul fără reantrenare; lungimea contextului și covariabilele sînt parametrii lor de ajustare."),
        t("Significance by Diebold--Mariano on daily losses; several seeds for every network.",
          "Semnificația, prin testul Diebold--Mariano pe pierderile zilnice; mai multe rulări pentru fiecare rețea."),
        t("Conformal calibration turns raw quantiles into honest bands and VaR; backtest with Kupiec and Christoffersen.",
          "Calibrarea conformală transformă cuantilele brute în benzi și VaR oneste; testați retrospectiv cu Kupiec și Christoffersen."),
        t("Next: multimodal models and the synthesis of the course (Lecture 14).", "Urmează: modelele multimodale și sinteza cursului (cursul 14)."),
        sep="4pt"))
    D.frame(t("Research from our group", "Cercetările grupului nostru"), it(
        t("\\textbf{Four generations of energy price forecasting}: econometric models, machine learning, LSTM and transformers compared on European "
          "energy markets, with pre-training and transfer learning (Andrei, Velev, Toma, Pele \\& Lessmann, 2024, arXiv 2411.03372).",
          "\\textbf{Patru generații de metode de prognoză a prețurilor energiei}: modele econometrice, învățare automată, LSTM și arhitecturi transformer, comparate pe "
          "piețele europene de energie, inclusiv preantrenarea și transferul învățării (Andrei, Velev, Toma, Pele \\& Lessmann, 2024, arXiv 2411.03372)."),
        t("\\textbf{Early warning of price spikes}: explainable gradient boosting on Romanian hourly data; the residual load forecast is the main trigger "
          "of spike warnings (Andrei, Pele, Cramer, Amza \\& Diaconu, 2026, PICBE).",
          "\\textbf{Avertizarea timpurie a vîrfurilor de preț}: modele explicabile de tip gradient boosting pe date orare din România; prognoza sarcinii reziduale este principalul "
          "declanșator al avertizărilor (Andrei, Pele, Cramer, Amza \\& Diaconu, 2026, PICBE)."),
        t("\\textbf{Multi-horizon explainability}: which drivers matter from 0--6 hours to two weeks ahead and in which price regime, Romania 2018--2024 "
          "(Andrei, Pele \\& Cramer, 2026, PICBE).",
          "\\textbf{Explicabilitate pe mai multe orizonturi}: ce factori contează de la 0--6 ore pînă la două săptămîni și în ce regim de preț, România 2018--2024 "
          "(Andrei, Pele \\& Cramer, 2026, PICBE)."),
        t("\\textbf{LLM-VaR and LLM-ES}: general-purpose language models produce Value-at-Risk and expected shortfall (ES) forecasts zero-shot from a prompt "
          "with past returns (Pele et al., 2026, Expert Systems with Applications).",
          "\\textbf{LLM-VaR și LLM-ES}: modele de limbaj generale produc, fără reantrenare, prognoze ale valorii expuse la risc și ale pierderii așteptate în coada distribuției (ES) "
          "pe baza unui prompt cu randamentele trecute (Pele et al., 2026, Expert Systems with Applications)."),
        t("\\textbf{Early warning for crypto `zombies'}: machine learning predicts which crypto-assets will become inactive "
          "(Będowska-Sójka, Wójcik \\& Pele, 2026, North American Journal of Economics and Finance).",
          "\\textbf{Avertizare timpurie pentru criptoactivele „zombie”}: învățarea automată prezice care criptoactive vor deveni inactive "
          "(Będowska-Sójka, Wójcik \\& Pele, 2026, North American Journal of Economics and Finance)."),
        sep="3pt") + note(t("Findings as stated in the papers' abstracts; numbers on the other slides come from our own data.",
                            "Rezultatele sînt preluate din rezumatele lucrărilor; cifrele de pe celelalte slide-uri provin din datele noastre.")),
        "footnotesize", cite=t("(Bucharest University of Economic Studies, ASE; Romanian Academy, Institute for Economic Forecasting, IPE)",
                               "(Academia de Studii Economice din București, ASE; Academia Română, Institutul de Prognoză Economică, IPE)"))
    D.frame(t("Want to learn more?", "Vreți să aflați mai mult?"), cols(it(
        t("\\textbf{Chronos} code and models: ", "\\textbf{Chronos}, cod și modele: ") + "\\href{https://github.com/amazon-science/chronos-forecasting}{github.com/amazon-science/chronos-forecasting}.",
        t("\\textbf{TimesFM}: ", "\\textbf{TimesFM}: ") + "\\href{https://github.com/google-research/timesfm}{github.com/google-research/timesfm}.",
        t("\\textbf{epftoolbox}, the open benchmark of Lago et al.\\ (2021): ", "\\textbf{epftoolbox}, reperul deschis al lui Lago et al.\\ (2021): ") +
        "\\href{https://github.com/jeslago/epftoolbox}{github.com/jeslago/epftoolbox}.", sep="4pt"),
        it(t("\\textbf{Conformal prediction}, a gentle introduction: ", "\\textbf{Predicția conformală}, o introducere accesibilă: ") +
           "\\href{https://arxiv.org/abs/2107.07511}{arXiv:2107.07511}.",
           t("\\textbf{Forecasting: Principles and Practice}, 3rd ed.: ", "\\textbf{Forecasting: Principles and Practice}, ed.\\ a 3-a: ") + "\\href{https://otexts.com/fpp3/}{otexts.com/fpp3}.",
           t("\\textbf{This lecture}: notebook, seminar and Quantlets --- ", "\\textbf{Acest curs}: notebook, seminar și Quantlets --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.", sep="4pt"), "0.5", "0.47"))
    reference_frames(D, REFS, per_frame=10)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.parts.append("\\section*{" + t("Appendix", "Anexă") + "}\n")
    D.frame(t("Appendix A: why the conformal band covers", "Anexa A: de ce acoperă banda conformală"), it(
        t("Let $s_1,\\dots,s_n,s_{n+1}$ be exchangeable conformity scores; the rank of $s_{n+1}$ among them is uniform on $\\{1,\\dots,n+1\\}$.",
          "Fie $s_1,\\dots,s_n,s_{n+1}$ scoruri de conformitate interschimbabile; rangul lui $s_{n+1}$ printre ele este uniform pe $\\{1,\\dots,n+1\\}$."),
        t("Hence $\\Pr(s_{n+1}\\le s_{(k)})\\ge k/(n+1)$; choosing $k=\\lceil (n+1)(1-\\alpha)\\rceil$ gives $\\Pr(s_{n+1}\\le\\hat Q)\\ge 1-\\alpha$.",
          "De aici $\\Pr(s_{n+1}\\le s_{(k)})\\ge k/(n+1)$; alegînd $k=\\lceil (n+1)(1-\\alpha)\\rceil$ obținem $\\Pr(s_{n+1}\\le\\hat Q)\\ge 1-\\alpha$."),
        t("$s_{n+1}\\le\\hat Q$ is equivalent to $y_{n+1}\\in[\\hat q^{\\alpha/2}-\\hat Q,\\hat q^{1-\\alpha/2}+\\hat Q]$, which is the band.",
          "$s_{n+1}\\le\\hat Q$ este echivalent cu $y_{n+1}\\in[\\hat q^{\\alpha/2}-\\hat Q,\\hat q^{1-\\alpha/2}+\\hat Q]$, adică banda."),
        t("Hours of a price series are dependent and drift; the rolling window of @{split.cal_days} days trades exactness for adaptation.",
          "Orele unei serii de prețuri sînt dependente și au derivă; fereastra glisantă de @{split.cal_days} de zile renunță la exactitate în favoarea adaptării."),
        sep="4pt"), cite="(Romano, Patterson \\& Candès, 2019)")
    D.frame(t("Appendix B: Christoffersen's test of independence", "Anexa B: testul de independență al lui Christoffersen"), it(
        t("Count transitions of the hit sequence: $n_{ij}$ = number of days with $I_{t-1}=i$ and $I_t=j$; $\\hat\\pi_{01}=n_{01}/(n_{00}+n_{01})$, $\\hat\\pi_{11}=n_{11}/(n_{10}+n_{11})$.",
          "Numărăm tranzițiile șirului de depășiri: $n_{ij}$ = numărul de zile cu $I_{t-1}=i$ și $I_t=j$; $\\hat\\pi_{01}=n_{01}/(n_{00}+n_{01})$, $\\hat\\pi_{11}=n_{11}/(n_{10}+n_{11})$."),
        t("$\\mathrm{LR}_{ind}=-2\\ln\\frac{(1-\\hat\\pi)^{n_{00}+n_{10}}\\hat\\pi^{n_{01}+n_{11}}}{(1-\\hat\\pi_{01})^{n_{00}}\\hat\\pi_{01}^{n_{01}}(1-\\hat\\pi_{11})^{n_{10}}\\hat\\pi_{11}^{n_{11}}}\\sim\\chi^2_1$.",
          "$\\mathrm{LR}_{ind}=-2\\ln\\frac{(1-\\hat\\pi)^{n_{00}+n_{10}}\\hat\\pi^{n_{01}+n_{11}}}{(1-\\hat\\pi_{01})^{n_{00}}\\hat\\pi_{01}^{n_{01}}(1-\\hat\\pi_{11})^{n_{10}}\\hat\\pi_{11}^{n_{11}}}\\sim\\chi^2_1$."),
        t("A small $p$-value means violations cluster: the model reacts too slowly to a volatility shock.",
          "O valoare $p$ mică înseamnă că depășirile vin grupat: modelul reacționează prea încet la un șoc de volatilitate."),
        sep="4pt"), cite="(Christoffersen, 1998)")
    D.frame(t("Appendix C: implementation details", "Anexa C: detalii de implementare"), it(
        t("LEAR: scikit-learn LassoLarsIC (criterion AIC), standardised inputs, daily re-estimation on @{split.win_days|,d} days.",
          "LEAR: LassoLarsIC din scikit-learn (criteriul AIC), intrări standardizate, reestimare zilnică pe @{split.win_days|,d} de zile."),
        t("DNN and N-BEATS: PyTorch, L1 loss, Adam, early stopping; seeds @{seed} and the next ones.",
          "DNN și N-BEATS: PyTorch, pierdere L1, Adam, oprire timpurie; valorile generatorului aleator: @{seed} și următoarele."),
        t("Foundation models: amazon/chronos-2, amazon/chronos-bolt-base, google/timesfm-2.5-200m-pytorch from Hugging Face, inference only; "
          "the full run took @{runtime_min|.1f} minutes on a laptop.",
          "Modelele fundație: amazon/chronos-2, amazon/chronos-bolt-base, google/timesfm-2.5-200m-pytorch de pe Hugging Face, doar inferență; "
          "rularea completă a durat @{runtime_min|.1f} minute pe un laptop."),
        t("Hours on the CET calendar; the doubled autumn hour is averaged; days with a missing hour are excluded from training targets and from the test.",
          "Orele urmează calendarul CET; ora dublă din toamnă este mediată; zilele cu o oră lipsă sînt excluse din țintele de antrenare și din test."),
        sep="4pt"))
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 13, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
