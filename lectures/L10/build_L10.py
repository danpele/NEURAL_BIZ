r"""
build_L10.py -- Lecture 10 "Reinforcement learning: trading, hedging and battery dispatch" (EN + RO) from one source.
Numbers come from L10_numbers.json (analysis_L10.py) as @{key} placeholders; nothing computed is typed by hand.
Hand-typed numbers are definitions: battery size and efficiency, cost levels, gamma, epsilon, the PPO clip 0.2,
the 95 % level of the expected shortfall, 252 trading days.
Run:  python3 lectures/L10/build_L10.py      (exit 0 only if every check of lectures/STYLE.md passes)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import Deck, Numbers, abbreviation_frames, block, cols, hier, items, reference_frames, run_build, table  # noqa: E402

ABBR = {
    "A2C": ("advantage actor-critic", "actor-critic cu avantaj (advantage actor-critic)"),
    "AI": ("artificial intelligence", "inteligență artificială"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "ATM": ("at the money (strike = current price)", "la bani (prețul de exercitare = prețul curent)"),
    "BG": ("Bulgaria (bidding zone)", "Bulgaria (zonă de ofertare)"),
    "CET": ("Central European Time", "ora Europei Centrale"),
    "CPU": ("central processing unit", "procesor central"),
    "DE-LU": ("Germany-Luxembourg bidding zone", "zona de ofertare Germania-Luxemburg"),
    "DM": ("Diebold-Mariano test", "testul Diebold-Mariano"),
    "DP": ("dynamic programming", "programare dinamică"),
    "DQN": ("deep Q-network", "rețea Q profundă (deep Q-network)"),
    "ENTSO-E": ("European Network of Transmission System Operators for Electricity",
                "Rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică"),
    "ES": ("expected shortfall", "pierderea așteptată în coada distribuției (expected shortfall)"),
    "EU": ("European Union", None),
    "UE": (None, "Uniunea Europeană"),
    "GAE": ("generalised advantage estimation", "estimarea generalizată a avantajului"),
    "HU": ("Hungary (bidding zone)", "Ungaria (zonă de ofertare)"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "ISE": ("Fraunhofer Institute for Solar Energy Systems", "Institutul Fraunhofer pentru Sisteme de Energie Solară"),
    "KL": ("Kullback-Leibler divergence", "divergența Kullback-Leibler"),
    "MDP": ("Markov decision process", "proces decizional Markov"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "PPO": ("proximal policy optimisation", "optimizarea proximală a politicii (proximal policy optimisation)"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată"),
    "RL": ("reinforcement learning", "învățare prin întărire (reinforcement learning)"),
    "RO": ("Romania (bidding zone)", "România (zonă de ofertare)"),
    "SSRN": ("Social Science Research Network", "Social Science Research Network"),
    "TD": ("temporal difference", "diferență temporală"),
    "TRPO": ("trust region policy optimisation", "optimizarea politicii în regiune de încredere"),
    "VIX": ("Cboe Volatility Index (30-day implied volatility of the S\\&P 500)",
            "indicele de volatilitate Cboe (volatilitatea implicită pe 30 de zile a indicelui S\\&P 500)"),
    "VaR": ("value at risk", "valoarea la risc (value at risk)"),
    "EUR": ("euro", "euro"), "USD": ("US dollar", "dolar american"), "MW": ("megawatt", "megawatt"),
    "MWh": ("megawatt-hour", "megawatt-oră"),
}
UNITS = {"EUR", "USD", "MW", "MWh"}
IGNORE = {"AAAI", "BF", "BIZ", "NEURAL", "JOR", "PL", "US", "OpenAI", "CLIP", "BS", "REINFORCE", "PICBE", "ESR", "JF", "JMLR", "IEEE", "NIPS", "ICML", "ICLR", "MIT", "QF", "II", "III", "IV", "OK", "SR", "PNAS"}

REFS = [
    "Andrei, A.-V., Cepoi, C.\\,O., Dumitrescu, B.\\,A., Lessmann, S., Chiriac (Luchian), A.-C., Pele, D.\\,T.\\ \\& Petukhina, A.\\ "
    "(2026). Joint probabilistic forecasting for battery arbitrage in day-ahead power markets. SSRN Working Paper 6872381. "
    "\\href{https://doi.org/10.2139/ssrn.6872381}{doi:10.2139/ssrn.6872381}",
    "Anagnoste, S., Andrei, A.-V., Bolovăneanu, V., Cepoi, C.-O., Clodnitchi, R., Cramer, A.-A., Grecu, R.-A., Lessmann, S., "
    "Pele, D.\\,T., Petukhina, A.\\ \\& Strat, V.\\,A.\\ (2025). The role of AI in (re)shaping energy finance: A systematic literature "
    "review. \\textit{Energy Strategy Reviews}, 61, 101833. \\href{https://doi.org/10.1016/j.esr.2025.101833}{doi:10.1016/j.esr.2025.101833}",
    "Andrei, A.-V., Velev, G., Toma, F.-M., Pele, D.\\,T.\\ \\& Lessmann, S.\\ (2024). Energy price modelling: A comparative "
    "evaluation of four generations of forecasting methods. \\href{https://arxiv.org/abs/2411.03372}{arXiv:2411.03372}",
    "Bellman, R.\\ (1957). \\textit{Dynamic Programming}. Princeton University Press.",
    "Buehler, H., Gonon, L., Teichmann, J.\\ \\& Wood, B.\\ (2019). Deep hedging. \\textit{Quantitative Finance}, 19(8), "
    "1271--1291. \\href{https://doi.org/10.1080/14697688.2019.1571683}{doi:10.1080/14697688.2019.1571683}",
    "Diebold, F.\\,X.\\ \\& Mariano, R.\\,S.\\ (1995). Comparing predictive accuracy. \\textit{Journal of Business \\& Economic "
    "Statistics}, 13(3), 253--263. \\href{https://doi.org/10.1080/07350015.1995.10524599}{doi:10.1080/07350015.1995.10524599}",
    "Hambly, B., Xu, R.\\ \\& Yang, H.\\ (2023). Recent advances in reinforcement learning in finance. \\textit{Mathematical "
    "Finance}, 33(3), 437--503. \\href{https://doi.org/10.1111/mafi.12382}{doi:10.1111/mafi.12382}",
    "Mnih, V.\\ et al.\\ (2015). Human-level control through deep reinforcement learning. \\textit{Nature}, 518, 529--533. "
    "\\href{https://doi.org/10.1038/nature14236}{doi:10.1038/nature14236}",
    "Moody, J.\\ \\& Saffell, M.\\ (2001). Learning to trade via direct reinforcement. \\textit{IEEE Transactions on Neural "
    "Networks}, 12(4), 875--889. \\href{https://doi.org/10.1109/72.935097}{doi:10.1109/72.935097}",
    "Politis, D.\\,N.\\ \\& Romano, J.\\,P.\\ (1994). The stationary bootstrap. \\textit{Journal of the American Statistical "
    "Association}, 89(428), 1303--1313. \\href{https://doi.org/10.1080/01621459.1994.10476870}{doi:10.1080/01621459.1994.10476870}",
    "Rockafellar, R.\\,T.\\ \\& Uryasev, S.\\ (2000). Optimization of conditional value-at-risk. \\textit{Journal of Risk}, 2(3), "
    "21--41. \\href{https://doi.org/10.21314/JOR.2000.038}{doi:10.21314/JOR.2000.038}",
    "Schulman, J., Moritz, P., Levine, S., Jordan, M.\\ \\& Abbeel, P.\\ (2016). High-dimensional continuous control using "
    "generalized advantage estimation. ICLR. \\href{https://arxiv.org/abs/1506.02438}{arXiv:1506.02438}",
    "Schulman, J., Wolski, F., Dhariwal, P., Radford, A.\\ \\& Klimov, O.\\ (2017). Proximal policy optimization algorithms. "
    "\\href{https://arxiv.org/abs/1707.06347}{arXiv:1707.06347}",
    "Sutton, R.\\,S.\\ \\& Barto, A.\\,G.\\ (2018). \\textit{Reinforcement Learning: An Introduction}, 2nd ed. MIT Press. "
    "\\href{http://incompleteideas.net/book/the-book-2nd.html}{incompleteideas.net/book}",
    "Sutton, R.\\,S., McAllester, D., Singh, S.\\ \\& Mansour, Y.\\ (2000). Policy gradient methods for reinforcement learning "
    "with function approximation. \\textit{Advances in Neural Information Processing Systems}, 12, 1057--1063.",
    "van Hasselt, H., Guez, A.\\ \\& Silver, D.\\ (2016). Deep reinforcement learning with double Q-learning. "
    "\\textit{Proceedings of the AAAI Conference on Artificial Intelligence}, 30(1). \\href{https://doi.org/10.1609/aaai.v30i1.10295}{doi:10.1609/aaai.v30i1.10295}",
    "Watkins, C.\\,J.\\,C.\\,H.\\ \\& Dayan, P.\\ (1992). Q-learning. \\textit{Machine Learning}, 8, 279--292. "
    "\\href{https://doi.org/10.1007/BF00992698}{doi:10.1007/BF00992698}",
    "Williams, R.\\,J.\\ (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. "
    "\\textit{Machine Learning}, 8, 229--256. \\href{https://doi.org/10.1007/BF00992696}{doi:10.1007/BF00992696}",
]


def it(*xs, sep="1pt"):
    """Bullets: a string is split by hier() (first clause = level 1, the rest = level 2); a tuple (head, sub1, ...)
    gives an explicit level-1 head with its level-2 bullets."""
    def one(x):
        if isinstance(x, tuple):
            return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                    "\n".join("      \\item " + (s[0].upper() + s[1:] if s[0].islower() else s) for s in x[1:] if s) +
                    "\n    \\end{itemize}")
        return hier(x)
    return (f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{{sep}}}\n" + "\n".join("  \\item " + one(x) for x in xs if x) +
            "\n\\end{itemize}")


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L10_numbers.json"), lang)
    N = V.N
    D = Deck(10, "Reinforcement learning: trading, hedging and battery dispatch" if lang == "en" else
             "Învățarea prin întărire: tranzacționare, acoperirea riscului și dispecerizarea bateriilor", V, lang, heights=H)
    t = D.t
    ro = lang == "ro"
    NB = t("notebook", "notebook")
    OWN = t("(own computation)", "(calcul propriu)")

    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Write a business decision as a Markov decision process (MDP): state, action, reward, transition and discount.",
          "Să formulați o decizie economică drept proces decizional Markov (MDP): stare, acțiune, recompensă, tranziție și factor de actualizare."),
        t("Solve a small MDP exactly with the Bellman equation and dynamic programming (DP), and read the value function.",
          "Să rezolvați exact un MDP de mici dimensiuni cu ecuația Bellman și programarea dinamică (DP) și să interpretați funcția de valoare."),
        t("Train value-based agents (tabular Q-learning, a deep Q-network (DQN)) and a policy-gradient agent (proximal policy "
          "optimisation (PPO)) on real prices, with several seeds and an honest benchmark.",
          "Să antrenați agenți bazați pe valoare (Q-learning tabelar, o rețea Q profundă (DQN)) și un agent cu gradient de "
          "politică (optimizarea proximală a politicii (PPO)) pe prețuri reale, cu mai multe valori inițiale și un reper corect."),
        t("Hedge an option with a neural policy that minimises the expected shortfall (ES) of the loss, and compare it with the "
          "Black-Scholes delta.",
          "Să acoperiți riscul unei opțiuni cu o politică neuronală care minimizează pierderea așteptată în coada distribuției "
          "(ES) și să o comparați cu delta Black-Scholes."),
        t("Judge when reinforcement learning (RL) adds value in business and when a simple rule is enough.",
          "Să judecați cînd învățarea prin întărire (RL) aduce valoare într-o afacere și cînd ajunge o regulă simplă."),
        sep="4pt"))
    abbreviation_frames(D, ABBR, per_frame=34)

    D.frame(t("Real data in this lecture", "Datele reale ale cursului"), it(
        t("Three problems, three real datasets; every number on the slides is computed by analysis\\_L10.py and stored in "
          "L10\\_numbers.json.",
          "Trei probleme, trei seturi de date reale; fiecare cifră de pe slide-uri este calculată de analysis\\_L10.py și "
          "salvată în L10\\_numbers.json."), sep="2pt") + "\\vspace{1mm}" + table(
        "@{}p{0.2\\textwidth}p{0.4\\textwidth}p{0.33\\textwidth}@{}",
        [t("Problem", "Problema"), t("Data and source", "Date și sursă"), t("Split (time order)", "Împărțire (ordine cronologică)")],
        [[t("Battery dispatch", "Dispecerizarea bateriei"),
          t("Hourly day-ahead prices, Romania (RO), Hungary (HU), Germany-Luxembourg (DE-LU), Bulgaria (BG); energy-charts.info of "
            "the Fraunhofer Institute for Solar Energy Systems (ISE), from data of the European Network of Transmission System "
            "Operators for Electricity (ENTSO-E)",
            "Prețuri orare pentru ziua următoare: România (RO), Ungaria (HU), Germania-Luxemburg (DE-LU), Bulgaria (BG); "
            "energy-charts.info, al Institutului Fraunhofer pentru Sisteme de Energie Solară (ISE), din datele Rețelei europene a "
            "operatorilor de transport și de sistem pentru energie electrică (ENTSO-E)"),
          t("train 2019--2023, validation 2024, test @{split.test_first|date} -- @{split.test_last|date}",
            "antrenare 2019--2023, validare 2024, test @{split.test_first|date} -- @{split.test_last|date}")],
         [t("Trading", "Tranzacționare"), t("S\\&P 500 daily closes and the Cboe Volatility Index (VIX); Yahoo Finance",
                                           "Închiderile zilnice S\\&P 500 și indicele de volatilitate Cboe (VIX); Yahoo Finance"),
          t("train 2005--2016, validation 2017--2019, test 2020 -- @{trade.test_last|date}",
            "antrenare 2005--2016, validare 2017--2019, test 2020 -- @{trade.test_last|date}")],
         [t("Option hedging", "Acoperirea opțiunii"), t("21-day S\\&P 500 paths, VIX as implied volatility",
                                                        "Traiectorii S\\&P 500 de 21 de zile, VIX ca volatilitate implicită"),
          t("paths starting 2005--2016 / 2017--2019 / 2020 -- @{hedge.test_last|date}",
            "traiectorii începute în 2005--2016 / 2017--2019 / 2020 -- @{hedge.test_last|date}")]]) + it(
        t("Data cut at @{cutoff.end|date}, the last complete month; seeds @{seed} to @{seed}+4; training on an Apple M5 Max "
          "central processing unit (CPU), trained results cached.",
          "Datele se opresc la @{cutoff.end|date}, ultima lună completă; valorile inițiale @{seed} ... @{seed}+4; antrenare pe "
          "procesorul central (CPU) Apple M5 Max, cu rezultatele salvate."), sep="1pt"), "footnotesize")

    # ================================================================== 1. sequential decisions
    D.section(t("Sequential decisions in business", "Decizii secvențiale în afaceri"))
    D.concept(t("Reinforcement learning and supervised learning", "Învățarea prin întărire și învățarea supervizată"),
              t("supervised learning predicts a label; RL chooses actions whose consequences arrive later and depend on what it did before.",
                "învățarea supervizată prognozează o etichetă; RL alege acțiuni cu efecte întîrziate, care depind de deciziile anterioare."),
              it((t("\\textbf{Supervised}", "\\textbf{Supervizată}"),
                  t("data $(\\vx_i, y_i)$, loss $\\loss(\\hat y_i, y_i)$ known for each example", "date $(\\vx_i, y_i)$, pierderea $\\loss(\\hat y_i, y_i)$ cunoscută pentru fiecare exemplu"),
                  t("the prediction does not change the next example", "prognoza nu schimbă exemplul următor")),
                 (t("\\textbf{Reinforcement}", "\\textbf{Prin întărire}"),
                  t("the agent sees a state $s_t$, acts $a_t$, receives a reward $r_{t+1}$", "agentul vede o stare $s_t$, acționează $a_t$, primește o recompensă $r_{t+1}$"),
                  t("the action changes the next state (a full battery cannot charge)", "acțiunea schimbă starea următoare (o baterie plină nu se mai încarcă)"),
                  t("no teacher gives the right action: only a delayed, noisy reward", "nimeni nu indică acțiunea corectă: doar o recompensă întîrziată și zgomotoasă")), sep="2pt"),
              it((t("\\textbf{Business decisions of this kind}", "\\textbf{Decizii economice de acest tip}"),
                  t("storage: when to charge and discharge", "stocare: cînd se încarcă și cînd se descarcă"),
                  t("trading: when to be in or out of the market", "tranzacționare: cînd se intră și cînd se iese din piață"),
                  t("hedging: how much of the underlying to hold each day", "acoperire: cît din activul suport se deține în fiecare zi"),
                  t("inventory, pricing, credit limits", "stocuri, prețuri, limite de credit")), sep="2pt"),
              t("use RL only when today's action changes tomorrow's options.",
                "folosiți RL doar cînd acțiunea de azi schimbă opțiunile de mîine."),
              cite="(Sutton \\& Barto, 2018, ch.\\ 1)")

    loop = ("\\begin{center}\\begin{tikzpicture}[node distance=2.6cm, every node/.style={font=\\footnotesize}]\n"
            "\\node[draw, rounded corners, fill=Ser2!20, minimum width=2.6cm, minimum height=0.6cm] (ag) {" + t("Agent: policy $\\pi_{\\btheta}(a\\mid s)$", "Agent: politica $\\pi_{\\btheta}(a\\mid s)$") + "};\n"
            "\\node[draw, rounded corners, fill=Ser1!15, minimum width=2.6cm, minimum height=0.6cm, right=4.2cm of ag] (env) {" + t("Environment: market, battery", "Mediu: piața, bateria") + "};\n"
            "\\draw[-{Latex}, thick] (ag.north) to[bend left=12] node[above]{" + t("action $a_t$", "acțiunea $a_t$") + "} (env.north);\n"
            "\\draw[-{Latex}, thick] (env.south) to[bend left=12] node[below]{" + t("reward $r_{t+1}$, new state $s_{t+1}$", "recompensa $r_{t+1}$, starea nouă $s_{t+1}$") + "} (ag.south);\n"
            "\\end{tikzpicture}\\end{center}")
    D.frame(t("The agent-environment loop", "Bucla agent-mediu"),
            "\\intuition{" + t("the agent acts, the environment answers with a reward and a new state; learning means improving the policy from these answers.",
                               "agentul acționează, mediul răspunde cu o recompensă și o stare nouă; a învăța înseamnă a îmbunătăți politica pe baza acestor răspunsuri.") + "}\n" +
            loop + it(
                t("\\textbf{Episode}: a delivery day, a 126-day trading window, a 21-day option life.",
                  "\\textbf{Episod}: o zi de livrare, o fereastră de 126 de zile, durata de 21 de zile a unei opțiuni."),
                t("\\textbf{Model-free}: the agent learns from sampled transitions $(s_t,a_t,r_{t+1},s_{t+1})$, never from the transition law.",
                  "\\textbf{Fără model}: agentul învață din tranziții eșantionate $(s_t,a_t,r_{t+1},s_{t+1})$, nu din legea de tranziție."),
                t("\\textbf{Offline here}: all three agents learn from historical data.",
                  "\\textbf{Aici, offline}: toți cei trei agenți învață din date istorice."), sep="0pt") +
            D.bottom(t("write down state, action and reward before training anything.",
                       "scrieți starea, acțiunea și recompensa înainte de orice antrenare.")),
            "small", cite="(Sutton \\& Barto, 2018, ch.\\ 3)")

    D.frame(t("Three business applications of reinforcement learning", "Trei aplicații economice ale învățării prin întărire"), table(
        "@{}p{0.14\\textwidth}p{0.27\\textwidth}p{0.2\\textwidth}p{0.33\\textwidth}@{}",
        [t("Problem", "Problema"), t("State $s_t$", "Starea $s_t$"), t("Action $a_t$", "Acțiunea $a_t$"), t("Reward $r_{t+1}$", "Recompensa $r_{t+1}$")],
        [[t("Battery", "Baterie"), t("hour, stored energy, yesterday's prices", "ora, energia stocată, prețurile de ieri"),
          t("charge / idle / discharge", "încărcare / inactiv / descărcare"),
          t("cash flow of the hour (EUR)", "fluxul de numerar al orei (EUR)")],
         [t("Trading", "Tranzacționare"), t("past returns, volatility, VIX, trend, current position", "randamente trecute, volatilitate, VIX, trend, poziția curentă"),
          t("flat / long", "în afara pieței / long"), t("position $\\times$ next return $-$ cost", "poziția $\\times$ randamentul următor $-$ cost")],
         [t("Hedging", "Acoperire"), t("index level, time to maturity, VIX, previous hedge", "nivelul indicelui, timpul pînă la scadență, VIX, acoperirea anterioară"),
          t("hedge ratio $\\delta_t\\in[0,1]$", "raportul de acoperire $\\delta_t\\in[0,1]$"),
          t("terminal profit and loss, judged by its ES", "rezultatul final, evaluat prin ES")]], size="footnotesize") +
        "\\vspace{1mm}" + it(
            t("Same mathematics: maximise expected discounted reward; different information and different risk attitudes.",
              "Aceeași matematică: maximizarea recompensei actualizate așteptate; informație și atitudini față de risc diferite."),
            t("Benchmarks are rules a manager already uses: yesterday's plan, buy-and-hold, the 200-day moving average, Black-Scholes delta.",
              "Reperele sînt reguli pe care un manager le folosește deja: planul de ieri, cumpără și păstrează, media mobilă de 200 de zile, delta Black-Scholes."), sep="2pt") +
        D.bottom(t("an RL agent is worth deploying only if it beats the rule the desk already uses, after costs and over several seeds.",
                   "un agent RL merită folosit doar dacă bate regula deja folosită, după costuri și pe mai multe valori inițiale.")),
        "footnotesize", cite=OWN)

    D.figframe(t("Batteries earn the daily price spread, and the spread has grown", "Bateriile cîștigă diferența zilnică de preț, iar aceasta a crescut"),
               t("a battery buys in the cheapest hours and sells in the dearest; its income scales with the daily max-min spread.",
                 "o baterie cumpără în orele cele mai ieftine și vinde în cele mai scumpe; venitul ei crește cu diferența zilnică dintre maxim și minim."),
               None, "spread_monthly", bullets=[
                   t("Romania: mean daily spread @{spread.ro_2019|.1f} EUR/MWh in 2019, @{spread.ro_2025|.1f} in 2025, @{spread.ro_ytd|.1f} in @{cutoff.year} so far.",
                     "România: diferența zilnică medie a fost @{spread.ro_2019|.1f} EUR/MWh în 2019, @{spread.ro_2025|.1f} în 2025 și @{spread.ro_ytd|.1f} în @{cutoff.year} pînă acum."),
                   t("Peak month @{spread.ro_max_month|ym}: @{spread.ro_max_value|.1f} EUR/MWh; solar output now pushes midday prices down.",
                     "Luna de vîrf, @{spread.ro_max_month|ym}: @{spread.ro_max_value|.1f} EUR/MWh; producția solară coboară acum prețurile de la prînz.")],
               job=t("the business case of a battery is a bet on future spreads, not on the price level.",
                     "rentabilitatea unei baterii depinde de diferențele viitoare de preț, nu de nivelul prețului."),
               tryit=NB + " \\S1", height="0.52\\textheight", cite=OWN)

    # ================================================================== 2. MDP and Bellman
    D.section(t("Markov decision processes and the Bellman equation", "Procese decizionale Markov și ecuația Bellman"))
    D.concept(t("The five elements of a Markov decision process", "Cele cinci elemente ale unui proces decizional Markov"),
              t("the future depends on the past only through the current state and action.",
                "viitorul depinde de trecut doar prin starea și acțiunea curente."),
              "\\[ \\mathcal{M}=(\\mathcal{S},\\mathcal{A},P,R,\\gamma) \\]"
              "\\[ P(s'\\mid s,a)=\\Pr(s_{t+1}=s'\\mid s_t=s,a_t=a) \\]"
              "\\[ G_t=\\sum_{k=0}^{T-t-1}\\gamma^{k}r_{t+k+1},\\quad \\pi^{*}=\\argmax_{\\pi}\\E_{\\pi}[G_0] \\]" +
              "\\where{" + t("$\\mathcal{S},\\mathcal{A}$ = states and actions, $P$ = transition law, $R(s,a)=\\E[r_{t+1}\\mid s,a]$; "
                             "$\\gamma\\in[0,1]$ = discount factor, $G_t$ = return, $\\pi(a\\mid s)$ = policy.",
                             "$\\mathcal{S},\\mathcal{A}$ = stări și acțiuni, $P$ = legea de tranziție, $R(s,a)=\\E[r_{t+1}\\mid s,a]$; "
                             "$\\gamma\\in[0,1]$ = factorul de actualizare, $G_t$ = cîștigul cumulat, $\\pi(a\\mid s)$ = politica.") + "}",
              it(t("\\textbf{Markov property}: the state must contain everything relevant; for the battery: hour and stored energy; "
                   "for trading: the current position, because it determines the cost of the next trade.",
                   "\\textbf{Proprietatea Markov}: starea trebuie să conțină tot ce contează; pentru baterie: ora și energia stocată; "
                   "pentru tranzacționare: poziția curentă, deoarece ea determină costul tranzacției următoare."),
                 t("\\textbf{Discounting}: $\\gamma=1$ for one-day battery episodes (24 steps), $\\gamma=0.99$ for trading.",
                   "\\textbf{Actualizarea}: $\\gamma=1$ pentru episoadele de o zi ale bateriei (24 de pași), $\\gamma=0{,}99$ pentru tranzacționare."),
                 t("\\textbf{Finite horizon}: the battery starts each day empty and its value at midnight is zero.",
                   "\\textbf{Orizont finit}: bateria pornește goală în fiecare zi, iar valoarea ei la miezul nopții este zero."), sep="3pt"),
              t("a missing state variable, such as the current position, makes the agent erratic.",
                "o variabilă de stare omisă, de exemplu poziția curentă, face agentul instabil."),
              cite="(Sutton \\& Barto, 2018, ch.\\ 3)", wl="0.52", wr="0.45")

    D.concept(t("Value functions measure how good a state is", "Funcțiile de valoare măsoară cît de bună este o stare"),
              t("$V$ is the expected future reward from a state; $Q$ is the same after a given first action.",
                "$V$ este recompensa viitoare așteptată dintr-o stare; $Q$ este același lucru după o primă acțiune dată."),
              "\\[ V^{\\pi}(s)=\\E_{\\pi}[G_t\\mid s_t=s] \\]"
              "\\[ Q^{\\pi}(s,a)=\\E_{\\pi}[G_t\\mid s_t=s,a_t=a] \\]"
              "\\[ A^{\\pi}(s,a)=Q^{\\pi}(s,a)-V^{\\pi}(s) \\]" +
              "\\where{" + t("$A^{\\pi}$ = advantage of action $a$ over the policy's average action; the optimal policy is greedy in $Q^{*}$: "
                             "$\\pi^{*}(s)=\\argmax_a Q^{*}(s,a)$.",
                             "$A^{\\pi}$ = avantajul acțiunii $a$ față de acțiunea medie a politicii; politica optimă este lacomă în raport cu $Q^{*}$: "
                             "$\\pi^{*}(s)=\\argmax_a Q^{*}(s,a)$.") + "}",
              it(t("\\textbf{Value-based methods} (Q-learning, DQN) estimate $Q^{*}$ and act greedily.",
                   "\\textbf{Metodele bazate pe valoare} (Q-learning, DQN) estimează $Q^{*}$ și acționează lacom."),
                 t("\\textbf{Policy-based methods} (REINFORCE, PPO) parameterise $\\pi_{\\btheta}$ directly and use $V$ only as a baseline.",
                   "\\textbf{Metodele bazate pe politică} (REINFORCE, PPO) parametrizează direct $\\pi_{\\btheta}$ și folosesc $V$ doar ca reper."),
                 t("\\textbf{Battery example}: $V_h(s)$ = the euros still to be earned today from hour $h$ with $s$ MWh stored.",
                   "\\textbf{Exemplul bateriei}: $V_h(s)$ = numărul de euro care mai pot fi cîștigați azi de la ora $h$, cu $s$ MWh stocați."), sep="3pt"),
              t("$V$ prices an asset (a charged battery, an open position); $Q$ compares decisions.",
                "$V$ evaluează un activ (o baterie încărcată, o poziție deschisă); $Q$ compară decizii."),
              cite="(Sutton \\& Barto, 2018, ch.\\ 3--4)", wl="0.52", wr="0.45")

    D.concept(t("The Bellman equation: recursive decomposition of the value function", "Ecuația Bellman: descompunerea recursivă a funcției de valoare"),
              t("the best value now = best immediate reward + discounted best value of where you land.",
                "valoarea optimă acum = cea mai bună recompensă imediată + valoarea optimă actualizată a stării în care ajungeți."),
              "\\[ V^{*}(s)=\\max_{a}\\Big\\{R(s,a)+\\gamma\\sum_{s'}P(s'\\mid s,a)\\,V^{*}(s')\\Big\\} \\]"
              "\\[ Q^{*}(s,a)=R(s,a)+\\gamma\\,\\E_{s'}\\big[\\max_{a'}Q^{*}(s',a')\\big] \\]" +
              "\\where{" + t("the right-hand side is a contraction for $\\gamma<1$, so value iteration $V_{k+1}=\\mathcal{T}V_k$ converges to the unique "
                             "fixed point $V^{*}$; with a finite horizon, one backward pass is exact.",
                             "pentru $\\gamma<1$ membrul drept este o contracție, deci iterarea valorii $V_{k+1}=\\mathcal{T}V_k$ converge la punctul fix "
                             "unic $V^{*}$; pe un orizont finit, o singură trecere înapoi dă soluția exactă.") + "}",
              it(t("\\textbf{Principle of optimality} (Bellman, 1957): an optimal plan stays optimal from any intermediate state.",
                   "\\textbf{Principiul optimalității} (Bellman, 1957): un plan optim rămîne optim din orice stare intermediară."),
                 t("\\textbf{Curse of dimensionality}: the table has $|\\mathcal{S}|\\times|\\mathcal{A}|$ cells; neural networks replace the table when the state is large.",
                   "\\textbf{Blestemul dimensionalității}: tabelul are $|\\mathcal{S}|\\times|\\mathcal{A}|$ celule; rețelele neuronale înlocuiesc tabelul cînd starea este mare."),
                 t("\\textbf{Model needed}: DP uses $P$ and $R$; Q-learning and DQN replace the expectation by samples.",
                   "\\textbf{Modelul este necesar}: DP folosește $P$ și $R$; Q-learning și DQN înlocuiesc speranța matematică prin eșantioane."), sep="3pt"),
              t("every RL algorithm in this lecture is an approximate way to solve this equation.",
                "fiecare algoritm RL din acest curs este o metodă aproximativă de rezolvare a acestei ecuații."),
              cite="(Bellman, 1957; Sutton \\& Barto, 2018, ch.\\ 4)", wl="0.53", wr="0.44")

    D.concept(t("The battery as an MDP", "Bateria ca proces decizional Markov"),
              t("state = (hour, stored energy); action = charge, idle or discharge; reward = the cash flow of the hour.",
                "starea = (ora, energia stocată); acțiunea = încărcare, inactivitate sau descărcare; recompensa = fluxul de numerar al orei."),
              "\\[ s_{h+1}=s_h-a_h,\\quad s_h\\in\\{0,1,2\\}\\ \\text{MWh},\\quad a_h\\in\\{-1,0,+1\\} \\]"
              "\\[ r_h(a_h)=\\begin{cases}\\eta_d\\,p_h & a_h=+1\\ (" + t("sell", "vînzare") + ")\\\\ -p_h/\\eta_c & a_h=-1\\ (" + t("buy", "cumpărare") + ")\\\\ 0 & a_h=0\\end{cases} \\]" +
              "\\where{" + t("$p_h$ = day-ahead price of hour $h$ (EUR/MWh), $\\eta_c=\\eta_d=\\sqrt{0.85}$ (85\\,\\% round trip); "
                             "2 MWh, 1 MWh moved per hour, empty at 0:00.",
                             "$p_h$ = prețul orei $h$ (EUR/MWh), $\\eta_c=\\eta_d=\\sqrt{0{,}85}$ (85\\,\\% pe ciclu); "
                             "2 MWh, 1 MWh mutat pe oră, goală la 0:00.") + "}",
              it(t("\\textbf{Constraint}: no charging when full, no discharging when empty (action masking).",
                   "\\textbf{Restricție}: nu se încarcă atunci cînd este plină și nu se descarcă atunci cînd este goală (mascarea acțiunilor)."),
                 t("\\textbf{Data}: @{data.ro_days|,d} Romanian delivery days with 24 complete hours, @{data.ro_first|date} -- @{data.ro_last|date}; "
                   "since Oct 2025 the hour is the mean of four 15-minute prices.",
                   "\\textbf{Date}: @{data.ro_days|,d} de zile de livrare din România cu 24 de ore complete, @{data.ro_first|date} -- @{data.ro_last|date}; "
                   "din oct.\\ 2025, ora este media a patru prețuri de 15 minute."),
                 t("\\textbf{Simplifications}: price taker, no degradation cost, no grid fees, no ancillary services.",
                   "\\textbf{Simplificări}: bateria nu influențează prețul; fără cost de degradare, tarife de rețea sau servicii de sistem."), sep="3pt"),
              t("the stored energy is what makes the decision sequential.",
                "energia stocată este cea care face decizia secvențială."),
              tryit=NB + " \\S2", cite=OWN, wl="0.5", wr="0.47")

    D.concept(t("Dynamic programming solves the battery day exactly", "Programarea dinamică rezolvă exact ziua bateriei"),
              t("go backwards from midnight: at each hour, keep the best action for each level of stored energy.",
                "se pornește înapoi de la miezul nopții: la fiecare oră se păstrează cea mai bună acțiune pentru fiecare nivel de energie stocată."),
              "\\[ V_{24}(s)=0 \\]"
              "\\[ V_h(s)=\\max_{a:\\,0\\le s-a\\le 2}\\big\\{r_h(a)+V_{h+1}(s-a)\\big\\},\\quad h=23,\\dots,0 \\]" +
              "\\where{" + t("$V_h(s)$ = maximal cash flow from hour $h$ to midnight with $s$ MWh stored; the revenue of the day is $V_0(0)$; "
                             "the argmax at each $(h,s)$ is the optimal action.",
                             "$V_h(s)$ = fluxul de numerar maxim de la ora $h$ la miezul nopții, cu $s$ MWh stocați; venitul zilei este $V_0(0)$; "
                             "argmax în fiecare $(h,s)$ dă acțiunea optimă.") + "}",
              it(t("\\textbf{Cost}: $24\\times3\\times3$ evaluations per day; all @{data.ro_days|,d} days in milliseconds, vectorised over days.",
                   "\\textbf{Cost}: $24\\times3\\times3$ evaluări pe zi; toate cele @{data.ro_days|,d} de zile în cîteva milisecunde, vectorizat pe zile."),
                 t("\\textbf{Perfect foresight}: DP uses the day's 24 prices before they are known, so its revenue is an \\textbf{upper bound} for any real strategy.",
                   "\\textbf{Previziune perfectă}: DP folosește cele 24 de prețuri ale zilei înainte ca ele să fie cunoscute, deci venitul său este o "
                   "\\textbf{limită superioară} pentru orice strategie reală."),
                 t("\\textbf{Naive benchmark}: compute yesterday's optimal plan and apply it today.",
                   "\\textbf{Reperul naiv}: se calculează planul optim de ieri și se aplică azi."), sep="3pt"),
              t("DP gives the ceiling against which every learning agent is scored (capture ratio = revenue / perfect-foresight revenue).",
                "DP dă plafonul față de care se evaluează orice agent care învață (rata de captare = venit / venitul cu previziune perfectă)."),
              tryit=NB + " \\S2", cite="(Bellman, 1957)", wl="0.53", wr="0.44")

    D.figframe(t("Optimal battery schedule for one day", "Programul optim al bateriei pentru o zi"),
               t("on the day with the largest spread of @{cutoff.year}, DP charges in the cheapest hours and discharges in the dearest.",
                 "în ziua cu cea mai mare diferență de preț din @{cutoff.year}, DP încarcă în orele cele mai ieftine și descarcă în cele mai scumpe."),
               None, "dp_day", bullets=[
                   t("@{day.date|date}: prices from @{day.pmin|.2f} (hour @{day.h_min}) to @{day.pmax|.2f} EUR/MWh (hour @{day.h_max}); "
                     "@{day.n_charge} charging and @{day.n_dis} discharging hours.",
                     "@{day.date|date}: prețuri de la @{day.pmin|.2f} (ora @{day.h_min}) la @{day.pmax|.2f} EUR/MWh (ora @{day.h_max}); "
                     "@{day.n_charge} ore de încărcare și @{day.n_dis} ore de descărcare."),
                   t("Revenue of the day for a 2 MWh battery: @{day.rev|,.0f} EUR.",
                     "Venitul zilei pentru o baterie de 2 MWh: @{day.rev|,.0f} EUR.")],
               tryit=NB + " \\S2", height="0.52\\textheight", cite=OWN)

    D.figframe(t("The value function prices stored energy hour by hour", "Funcția de valoare evaluează energia stocată oră de oră"),
               t("$V_h(2)-V_h(0)$ is what two stored MWh are worth at hour $h$: high before the evening peak, zero after it.",
                 "$V_h(2)-V_h(0)$ arată cît valorează doi MWh stocați la ora $h$: mult înaintea vîrfului de seară, zero după el."),
               None, "dp_value", bullets=[
                   t("At 12:00 on @{day.date|date}: $V_{12}(0)$ = @{day.v12_s0|,.0f} EUR, $V_{12}(2)$ = @{day.v12_s2|,.0f} EUR, "
                     "so one stored MWh is worth @{day.v12_gain_per_mwh|.2f} EUR, the shadow price of energy in the battery.",
                     "La ora 12:00, pe @{day.date|date}: $V_{12}(0)$ = @{day.v12_s0|,.0f} EUR, $V_{12}(2)$ = @{day.v12_s2|,.0f} EUR, "
                     "deci un MWh stocat valorează @{day.v12_gain_per_mwh|.2f} EUR, prețul-umbră al energiei din baterie."),
                   t("A good learned $Q$ must reproduce this shape without seeing the day's prices.",
                     "Un $Q$ bine învățat trebuie să reproducă această formă fără să vadă prețurile zilei.")],
               tryit=NB + " \\S2", height="0.50\\textheight", cite=OWN)

    D.figframe(t("Perfect foresight: what a 2 MWh battery could have earned", "Previziune perfectă: cît ar fi putut cîștiga o baterie de 2 MWh"),
               t("the DP revenue is an upper bound; it follows the spreads, not the price level.",
                 "venitul DP este o limită superioară; el urmează diferențele de preț, nu nivelul prețului."),
               None, "dp_revenue_year", bullets=[
                   t("Romania: @{dp.ro_2019_k|.1f} thousand EUR in 2019, @{dp.ro_2022_k|.1f} in 2022, @{dp.ro_2025_k|.1f} in 2025, "
                     "@{dp.ro_ytd_k|.1f} in @{cutoff.year} over @{dp.ro_ytd_days} days.",
                     "România: @{dp.ro_2019_k|.1f} mii EUR în 2019, @{dp.ro_2022_k|.1f} în 2022, @{dp.ro_2025_k|.1f} în 2025, "
                     "@{dp.ro_ytd_k|.1f} în @{cutoff.year}, pe @{dp.ro_ytd_days} de zile."),
                   t("2025: Hungary @{dp.hu_2025_k|.1f}, Bulgaria @{dp.bg_2025_k|.1f}, Germany-Luxembourg @{dp.de_2025_k|.1f} thousand EUR.",
                     "2025: Ungaria @{dp.hu_2025_k|.1f}, Bulgaria @{dp.bg_2025_k|.1f}, Germania-Luxemburg @{dp.de_2025_k|.1f} mii EUR.")],
               job=t("investors size storage on these numbers; a real operator captures only part of them, as the next section measures.",
                     "investitorii dimensionează stocarea pe baza acestor cifre; un operator real obține doar o parte, cum se măsoară în secțiunea următoare."),
               tryit=NB + " \\S2", height="0.48\\textheight", cite=OWN)

    # ================================================================== 3. value-based learning
    D.section(t("Value-based learning: Q-learning and DQN", "Învățarea bazată pe valoare: Q-learning și DQN"))
    D.concept(t("Q-learning: learn $Q^{*}$ from samples, without a model", "Q-learning: $Q^{*}$ învățat din eșantioane, fără model"),
              t("move $Q(s,a)$ towards the reward plus the best value of the next state; the gap is the temporal-difference (TD) error.",
                "$Q(s,a)$ se deplasează spre recompensă plus cea mai bună valoare a stării următoare; diferența este eroarea de diferență temporală (TD)."),
              "\\[ Q(s_t,a_t)\\leftarrow Q(s_t,a_t)+\\alpha\\,\\underbrace{\\big[r_{t+1}+\\gamma\\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\\big]}_{\\text{TD error }\\delta_t} \\]" +
              "\\where{" + t("$\\alpha$ = step size (0.05 here); the target uses the greedy action in $s_{t+1}$ whatever the agent does next, "
                             "so Q-learning is \\textbf{off-policy}.",
                             "$\\alpha$ = pasul de învățare (0{,}05 aici); ținta folosește acțiunea lacomă din $s_{t+1}$, indiferent ce face agentul în continuare, "
                             "deci Q-learning este \\textbf{off-policy}.") + "}",
              it(t("\\textbf{Convergence} (Watkins \\& Dayan, 1992): every pair $(s,a)$ visited infinitely often and $\\sum\\alpha_t=\\infty$, $\\sum\\alpha_t^2<\\infty$ (Appendix A).",
                   "\\textbf{Convergența} (Watkins \\& Dayan, 1992): fiecare pereche $(s,a)$ vizitată de o infinitate de ori, $\\sum\\alpha_t=\\infty$, $\\sum\\alpha_t^2<\\infty$ (Anexa A)."),
                 t("\\textbf{Exploration}: $\\varepsilon$-greedy; random action with probability $\\varepsilon$, decayed from 1 to 0.05.",
                   "\\textbf{Explorarea}: $\\varepsilon$-lacom; acțiune aleatoare cu probabilitatea $\\varepsilon$, redusă de la 1 la 0{,}05."),
                 t("\\textbf{Exploitation risk}: without exploration, a bad early estimate is never corrected.",
                   "\\textbf{Riscul exploatării}: fără explorare, o estimare proastă de la început nu mai este corectată."), sep="3pt"),
              t("the TD error is a learning signal available after every hour, not only at the end of the day.",
                "eroarea TD este un semnal de învățare disponibil după fiecare oră, nu doar la sfîrșitul zilei."),
              tryit=NB + " \\S3", cite="(Watkins \\& Dayan, 1992)", wl="0.56", wr="0.41")

    D.concept(t("Definition of the state: information available to the agent", "Definirea stării: informația disponibilă agentului"),
              t("the operator plans before the day-ahead prices are known; the agent conditions on how cheap each hour was yesterday.",
                "operatorul planifică înainte să cunoască prețurile pentru ziua următoare; agentul ține cont de cît de ieftină a fost fiecare oră ieri."),
              "\\[ s_h=\\big(h,\\ \\text{stored}_h,\\ q_{h}\\big),\\quad q_h=\\Big\\lfloor\\tfrac{\\mathrm{rank}(p^{\\,\\text{" + t("yesterday", "ieri") + "}}_h)}{6}\\Big\\rfloor\\in\\{0,1,2,3\\} \\]" +
              "\\where{" + t("rank of hour $h$ among yesterday's 24 prices, so $q_h=0$ = one of yesterday's six cheapest hours; "
                             "table size $24\\times4\\times3$ states $\\times$ 3 actions.",
                             "rangul orei $h$ între cele 24 de prețuri de ieri, deci $q_h=0$ = una dintre cele mai ieftine șase ore de ieri; "
                             "tabelul are $24\\times4\\times3$ stări $\\times$ 3 acțiuni.") + "}",
              it(t("\\textbf{Training}: @{split.train_days|,d} days of 2019--2023, 40 passes, the epoch with the best 2024 validation revenue is kept.",
                   "\\textbf{Antrenare}: @{split.train_days|,d} de zile din 2019--2023, 40 de treceri; se păstrează epoca cu cel mai mare venit de validare în 2024."),
                 t("\\textbf{Test}: @{split.test_days} days, @{split.test_first|date} -- @{split.test_last|date}, used once.",
                   "\\textbf{Test}: @{split.test_days} de zile, @{split.test_first|date} -- @{split.test_last|date}, folosite o singură dată."),
                 t("\\textbf{Reward scale}: EUR/100, $\\gamma=1$ within the day.", "\\textbf{Scara recompensei}: EUR/100, $\\gamma=1$ în cursul zilei."), sep="3pt"),
              t("a state that leaks today's prices would give a perfect but useless backtest.",
                "o stare care include prețurile de azi ar da un test retrospectiv perfect, dar inutil."),
              tryit=NB + " \\S3", cite=OWN, wl="0.55", wr="0.42")

    D.figframe(t("What tabular Q-learning learned", "Ce a învățat Q-learning tabelar"),
               t("with 1 MWh stored, the agent charges in hours that were cheap yesterday and sells in the evening.",
                 "cu 1 MWh stocat, agentul încarcă în orele care au fost ieftine ieri și vinde seara."),
               None, "q_policy", bullets=[
                   t("A readable policy: the table can be printed and audited hour by hour (@{bat.q_states} states).",
                     "O politică ușor de citit: tabelul poate fi tipărit și verificat oră cu oră (@{bat.q_states} de stări)."),
                   t("The hour of the day dominates; yesterday's ranking shifts the timing of charging.",
                     "Ora din zi domină; clasamentul de ieri mută momentul încărcării.")],
               tryit=NB + " \\S3", height="0.46\\textheight", cite=OWN)

    D.concept(t("From a table to a network: the deep Q-network", "De la tabel la rețea: rețeaua Q profundă"),
              t("replace the table by a network $Q_{\\btheta}(s,a)$ and regress it on the TD target.",
                "tabelul este înlocuit de o rețea $Q_{\\btheta}(s,a)$, ajustată prin regresie pe ținta TD."),
              "\\[ y=r+\\gamma\\,Q_{\\btheta^{-}}\\big(s',\\argmax_{a'}Q_{\\btheta}(s',a')\\big) \\]"
              "\\[ \\loss(\\btheta)=\\E_{(s,a,r,s')\\sim\\mathcal{D}}\\big[\\ell_{\\text{Huber}}\\big(y-Q_{\\btheta}(s,a)\\big)\\big] \\]" +
              "\\where{" + t("$\\mathcal{D}$ = replay memory (the last four epochs); $\\btheta^{-}$ = target network, a slow copy "
                             "$\\btheta^{-}\\leftarrow(1-\\tau)\\btheta^{-}+\\tau\\btheta$, $\\tau=0.005$; double DQN: the online network chooses $a'$, the target network evaluates it.",
                             "$\\mathcal{D}$ = memoria de reluare (ultimele patru epoci); $\\btheta^{-}$ = rețeaua-țintă, o copie lentă "
                             "$\\btheta^{-}\\leftarrow(1-\\tau)\\btheta^{-}+\\tau\\btheta$, $\\tau=0{,}005$; DQN dublu: rețeaua online alege $a'$, rețeaua-țintă o evaluează.") + "}",
              it(t("\\textbf{Replay} breaks the correlation of consecutive hours (Mnih et al., 2015).",
                   "\\textbf{Reluarea} elimină corelația dintre orele consecutive (Mnih et al., 2015)."),
                 t("\\textbf{Double DQN} removes the upward bias of $\\max_{a'}$ over noisy estimates (van Hasselt et al., 2016).",
                   "\\textbf{DQN dublu} elimină deplasarea în sus a lui $\\max_{a'}$ aplicat unor estimări zgomotoase (van Hasselt et al., 2016)."),
                 t("\\textbf{Our network}: @{bat.dqn_inputs} inputs (yesterday's and last week's 24 standardised prices, weekday, hour, stored energy), "
                   "two layers of 128 rectified linear units (ReLU), 3 outputs.",
                   "\\textbf{Rețeaua noastră}: @{bat.dqn_inputs} de intrări (cele 24 de prețuri standardizate de ieri și din aceeași zi a săptămînii trecute, "
                   "ziua săptămînii, ora, energia stocată), două straturi de cîte 128 de unități liniare rectificate (ReLU), 3 ieșiri."), sep="3pt"),
              t("function approximation + bootstrapping + off-policy data can diverge; replay and target networks keep it stable.",
                "aproximarea funcției, autoreferința țintei și datele off-policy pot produce divergență; reluarea și rețeaua-țintă mențin stabilitatea."),
              tryit=NB + " \\S4", cite="(Mnih et al., 2015; van Hasselt, Guez \\& Silver, 2016)", wl="0.56", wr="0.41")

    D.figframe(t("Learning curves: validation revenue by training epoch", "Curbele de învățare: venitul de validare pe epoci"),
               t("both agents improve quickly but stay below yesterday's plan; the epoch kept is the best on 2024, never on the test years.",
                 "ambii agenți progresează repede, dar rămîn sub planul de ieri; epoca păstrată este cea mai bună pe 2024, niciodată pe anii de test."),
               None, "battery_learning", bullets=[
                   t("Best validation revenue per day, mean of 5 seeds: DQN @{bat.dqn_valid_day|,.0f} EUR, Q-learning @{bat.q_valid_day|,.0f} EUR, "
                     "yesterday's plan @{bat.naive_valid_day|,.0f} EUR.",
                     "Cel mai bun venit de validare pe zi, media a 5 valori inițiale: DQN @{bat.dqn_valid_day|,.0f} EUR, Q-learning @{bat.q_valid_day|,.0f} EUR, "
                     "planul de ieri @{bat.naive_valid_day|,.0f} EUR.")],
               tryit=NB + " \\S4", height="0.50\\textheight", cite=OWN)

    D.figframe(t("Test years: how much of the perfect-foresight revenue is captured", "Anii de test: cît din venitul cu previziune perfectă se obține"),
               t("cumulative revenue from @{split.test_first|date} to @{split.test_last|date}; the gap to the top line is the value of perfect information.",
                 "venitul cumulat de la @{split.test_first|date} la @{split.test_last|date}; distanța față de linia de sus este valoarea informației perfecte."),
               None, "battery_test_cum", bullets=[
                   t("Perfect foresight @{bat.pf_k|.1f}; DQN @{bat.dqn_k|.1f}; Q-learning @{bat.q_k|.1f}; yesterday's plan @{bat.naive_k|.1f} thousand EUR.",
                     "Previziune perfectă @{bat.pf_k|.1f}; DQN @{bat.dqn_k|.1f}; Q-learning @{bat.q_k|.1f}; planul de ieri @{bat.naive_k|.1f} mii EUR.")],
               tryit=NB + " \\S4", height="0.52\\textheight", cite=OWN)

    D.frame(t("Battery results on the test years, five seeds", "Rezultatele bateriei pe anii de test, cinci valori inițiale"), table(
        "@{}p{0.36\\textwidth}rrr@{}",
        [t("Strategy", "Strategia"), t("Revenue (k EUR)", "Venit (mii EUR)"), t("Capture (\\%)", "Captare (\\%)"), t("Seed range (\\%)", "Interval (\\%)")],
        [[t("Perfect foresight (DP, upper bound)", "Previziune perfectă (DP, limita superioară)"), "@{bat.pf_k|.1f}", "100", "--"],
         [t("Double DQN", "DQN dublu"), "@{bat.dqn_k|.1f}", "@{bat.dqn_cap|.1f}", "@{bat.dqn_cap_min|.1f} -- @{bat.dqn_cap_max|.1f}"],
         [t("Tabular Q-learning", "Q-learning tabelar"), "@{bat.q_k|.1f}", "@{bat.q_cap|.1f}", "@{bat.q_cap_min|.1f} -- @{bat.q_cap_max|.1f}"],
         [t("Yesterday's optimal plan (naive)", "Planul optim de ieri (naiv)"), "@{bat.naive_k|.1f}", "@{bat.naive_cap|.1f}", "--"]], size="small") +
        "\\vspace{2mm}" + it(
            t("\\textbf{Test of DQN against the naive plan}: daily revenue differences, Diebold-Mariano (DM) statistic with Newey-West variance (7 lags): "
              "$t$ = @{bat.dm_t|.2f}, $p$ = @{bat.dm_p|.3f}; mean difference @{bat.dqn_minus_naive_day|.1f} EUR per day.",
              "\\textbf{DQN față de planul naiv}: statistica Diebold-Mariano (DM) pe diferențele zilnice, varianță Newey-West (7 decalaje): "
              "$t$ = @{bat.dm_t|.2f}, $p$ = @{bat.dm_p|.3f}; diferența medie @{bat.dqn_minus_naive_day|.1f} EUR pe zi."),
            t("DQN beats the naive plan in @{bat.dqn_beats_naive_seeds} of 5 seeds, Q-learning in @{bat.q_beats_naive_seeds} of 5: here the simple rule wins, because it optimises on "
              "all 24 of yesterday's prices, while the agents learn one average policy from a compressed state.",
              "DQN bate planul naiv pentru @{bat.dqn_beats_naive_seeds} din 5 valori inițiale, Q-learning pentru @{bat.q_beats_naive_seeds} din 5: regula simplă cîștigă, "
              "pentru că optimizează pe toate cele 24 de prețuri de ieri."), sep="2pt") +
        D.bottom(t("report the capture ratio with its range over seeds and a test against the naive plan, never a single best run.",
                   "raportați rata de captare cu intervalul pe valori inițiale și un test față de planul naiv, niciodată doar cea mai bună rulare."),
                 NB + " \\S4"),
        "small", cite="(Diebold \\& Mariano, 1995; " + t("own computation", "calcul propriu") + ")")

    # ================================================================== 4. policy gradients
    D.section(t("Policy gradients and PPO: trading the S\\&P 500", "Gradienți de politică și PPO: tranzacționarea indicelui S\\&P 500"))
    D.concept(t("Policy-gradient methods", "Metodele de gradient al politicii"),
              t("parameterise the policy directly and climb the gradient of the expected return.",
                "politica se parametrizează direct și se urcă pe gradientul cîștigului așteptat."),
              "\\[ \\nabla_{\\btheta}J(\\btheta)=\\E_{\\pi_{\\btheta}}\\Big[\\sum_{t}\\nabla_{\\btheta}\\log\\pi_{\\btheta}(a_t\\mid s_t)\\,\\big(G_t-b(s_t)\\big)\\Big] \\]" +
              "\\where{" + t("$J(\\btheta)=\\E_{\\pi_{\\btheta}}[G_0]$; $\\nabla\\log\\pi$ = score function; $b(s_t)$ = baseline, any function of the state, "
                             "that leaves the gradient unbiased and lowers its variance; $b=V_{\\boldsymbol{\\phi}}(s_t)$ gives an actor-critic.",
                             "$J(\\btheta)=\\E_{\\pi_{\\btheta}}[G_0]$; $\\nabla\\log\\pi$ = funcția scor; $b(s_t)$ = reperul, orice funcție de stare, "
                             "care lasă gradientul nedeplasat și îi reduce varianța; $b=V_{\\boldsymbol{\\phi}}(s_t)$ dă o metodă actor-critic.") + "}",
              it(t("\\textbf{REINFORCE} (Williams, 1992): Monte Carlo $G_t$; unbiased, high variance.",
                   "\\textbf{REINFORCE} (Williams, 1992): $G_t$ estimat Monte Carlo; nedeplasat, varianță mare."),
                 t("\\textbf{Policy gradient theorem} (Sutton et al., 2000): the gradient needs no derivative of the transition law, so the market can stay a black box.",
                   "\\textbf{Teorema gradientului politicii} (Sutton et al., 2000): gradientul nu cere derivata legii de tranziție, deci piața poate rămîne o cutie neagră."),
                 t("\\textbf{Stochastic policies} explore by themselves and handle continuous actions.",
                   "\\textbf{Politicile stocastice} explorează singure și tratează acțiuni continue."), sep="3pt"),
              t("the log-likelihood trick is the same one used to train classifiers, weighted by how good the outcome was.",
                "trucul log-verosimilității este același folosit la antrenarea clasificatorilor, ponderat cu cît de bun a fost rezultatul."),
              cite="(Williams, 1992; Sutton et al., 2000)", wl="0.55", wr="0.42")

    D.concept(t("Actor-critic and the advantage estimate", "Actor-critic și estimarea avantajului"),
              t("a critic $V_{\\boldsymbol{\\phi}}$ predicts the return, and the actor learns from the surprise $\\hat A_t$.",
                "un critic $V_{\\boldsymbol{\\phi}}$ prognozează cîștigul, iar actorul învață din surpriza $\\hat A_t$."),
              "\\[ \\delta_t=r_{t+1}+\\gamma V_{\\boldsymbol{\\phi}}(s_{t+1})-V_{\\boldsymbol{\\phi}}(s_t),\\qquad \\hat A_t=\\sum_{l\\ge0}(\\gamma\\lambda)^{l}\\,\\delta_{t+l} \\]"
              "\\[ \\loss_V(\\boldsymbol{\\phi})=\\E\\big[(\\hat A_t+V_{\\boldsymbol{\\phi}}^{\\text{old}}(s_t)-V_{\\boldsymbol{\\phi}}(s_t))^2\\big] \\]" +
              "\\where{" + t("generalised advantage estimation (GAE): $\\lambda=0$ gives the one-step TD advantage (low variance, biased), $\\lambda=1$ the Monte Carlo advantage; "
                             "we use $\\gamma=0.99$, $\\lambda=0.95$.",
                             "estimarea generalizată a avantajului (GAE): $\\lambda=0$ dă avantajul TD de un pas (varianță mică, deplasat), $\\lambda=1$ avantajul Monte Carlo; "
                             "folosim $\\gamma=0{,}99$, $\\lambda=0{,}95$.") + "}",
              it(t("\\textbf{Shared body}: actor and critic share two layers of 64 tanh units; heads: 2 action logits and 1 value.",
                   "\\textbf{Trunchi comun}: actorul și criticul împart două straturi de cîte 64 de unități tanh; ieșiri: 2 logiți pentru acțiuni și o valoare."),
                 t("\\textbf{Advantage normalisation} in each batch keeps the step size comparable across market regimes.",
                   "\\textbf{Normalizarea avantajului} în fiecare lot păstrează pasul comparabil între regimuri de piață."),
                 t("\\textbf{Entropy bonus} 0.01 keeps the policy from collapsing too early.",
                   "\\textbf{Bonusul de entropie} 0{,}01 împiedică politica să devină deterministă prea devreme."), sep="3pt"),
              t("the critic is a forecast of future reward; a poor critic gives noisy advantages and slow learning.",
                "criticul este o prognoză a recompensei viitoare; un critic slab dă avantaje zgomotoase și o învățare lentă."),
              cite="(Schulman et al., 2016)", wl="0.56", wr="0.41")

    D.figframe(t("PPO: clipped policy updates", "PPO: actualizarea plafonată a politicii"),
               t("the clipped objective removes the incentive to move the probability ratio outside $[1-\\epsilon,1+\\epsilon]$.",
                 "obiectivul plafonat elimină stimulentul de a muta raportul probabilităților în afara intervalului $[1-\\epsilon,1+\\epsilon]$."),
               "\\[ \\loss^{\\text{CLIP}}(\\btheta)=\\E_t\\big[\\min\\big(\\rho_t\\hat A_t,\\ \\mathrm{clip}(\\rho_t,1-\\epsilon,1+\\epsilon)\\hat A_t\\big)\\big],\\quad "
               "\\rho_t=\\frac{\\pi_{\\btheta}(a_t\\mid s_t)}{\\pi_{\\btheta_{\\text{old}}}(a_t\\mid s_t)} \\]",
               "ppo_clip", bullets=[
                   t("\\simlabel\\ Analytic function, no data; $\\epsilon=0.2$; a cheap substitute for the Kullback-Leibler (KL) trust region of trust region policy optimisation (TRPO).",
                     "\\simlabel\\ Funcție analitică, fără date; $\\epsilon=0{,}2$; un înlocuitor ieftin al regiunii de încredere bazate pe divergența Kullback-Leibler (KL) din optimizarea politicii în regiune de încredere (TRPO).")],
               height="0.40\\textheight", cite="(Schulman et al., 2017)")

    D.concept(t("Trading as an MDP: flat or long the S\\&P 500", "Tranzacționarea ca MDP: în afara pieței sau long pe S\\&P 500"),
              t("each evening the agent chooses tomorrow's position; it earns the next day's return and pays a cost when it changes position.",
                "în fiecare seară agentul alege poziția de mîine; cîștigă randamentul zilei următoare și plătește un cost cînd își schimbă poziția."),
              "\\[ r_{t+1}=a_t\\,\\ln\\frac{P_{t+1}}{P_t}-c\\,|a_t-a_{t-1}|,\\qquad a_t\\in\\{0,1\\},\\ c=5\\ \\text{bp} \\]" +
              "\\where{" + t("$s_t$ = 7 features standardised on 2005--2016 (log returns over 1, 5, 21, 63 days, 21-day volatility, VIX, distance to the 200-day "
                             "average) + $a_{t-1}$.",
                             "$s_t$ = 7 variabile standardizate pe 2005--2016 (randamente pe 1, 5, 21, 63 de zile, volatilitatea pe 21 de zile, VIX, distanța "
                             "față de media de 200 de zile) + $a_{t-1}$.") + "}",
              it(t("\\textbf{Training}: 150 PPO iterations of 64 random 126-day windows from 2005--2016, clip 0.2.",
                   "\\textbf{Antrenare}: 150 de iterații PPO cu cîte 64 de ferestre aleatoare de 126 de zile din 2005--2016, plafon 0{,}2."),
                 t("\\textbf{Selection}: the iteration with the best 2017--2019 Sharpe ratio.",
                   "\\textbf{Selecție}: iterația cu cel mai bun raport Sharpe în 2017--2019."),
                 t("\\textbf{Test}: @{trade.n_test|,d} days from @{trade.test_first|date}, five seeds, against buy-and-hold and the 200-day moving average.",
                   "\\textbf{Test}: @{trade.n_test|,d} de zile de la @{trade.test_first|date}, cinci valori inițiale, față de cumpără și păstrează și media mobilă de 200 de zile."), sep="2pt"),
              t("put the cost in the reward, or the agent trades far too often.",
                "includeți costul în recompensă, altfel agentul tranzacționează mult prea des."),
              tryit=NB + " \\S5", cite="(Moody \\& Saffell, 2001; " + t("own computation", "calcul propriu") + ")", wl="0.53", wr="0.44")

    D.figframe(t("Training and validation Sharpe ratios during PPO training", "Raportul Sharpe la antrenare și validare în timpul antrenării PPO"),
               t("the training Sharpe ratio rises; the validation ratio is noisy, and the best iteration is kept.",
                 "raportul Sharpe de antrenare crește; cel de validare este zgomotos, iar cea mai bună iterație este păstrată."),
               None, "trading_learning", bullets=[
                   t("Buy-and-hold Sharpe on validation (2017--2019): @{trade.bh_valid_sharpe|.2f}; the selected iteration is on average @{trade.best_it_mean|.0f}.",
                     "Raportul Sharpe al strategiei cumpără și păstrează pe validare (2017--2019): @{trade.bh_valid_sharpe|.2f}; iterația selectată este în medie @{trade.best_it_mean|.0f}.")],
               tryit=NB + " \\S5", height="0.50\\textheight", cite=OWN)

    D.figframe(t("Test period: the PPO agent against two simple rules", "Perioada de test: agentul PPO față de două reguli simple"),
               t("value of 1 USD invested on @{trade.test_first|date}, after a 5 basis-point cost per change of position.",
                 "valoarea unui USD investit la @{trade.test_first|date}, după un cost de 5 puncte de bază la fiecare schimbare de poziție."),
               None, "trading_equity", bullets=[
                   t("Sharpe ratio: buy-and-hold @{trade.bh_sharpe|.2f}; moving-average rule @{trade.ma_sharpe|.2f}; PPO @{trade.ppo_sharpe|.2f} "
                     "(seeds from @{trade.ppo_sharpe_min|.2f} to @{trade.ppo_sharpe_max|.2f}).",
                     "Raportul Sharpe: cumpără și păstrează @{trade.bh_sharpe|.2f}; regula mediei mobile @{trade.ma_sharpe|.2f}; PPO @{trade.ppo_sharpe|.2f} "
                     "(între @{trade.ppo_sharpe_min|.2f} și @{trade.ppo_sharpe_max|.2f} pe valori inițiale).")],
               tryit=NB + " \\S5", height="0.52\\textheight", cite=OWN)

    D.frame(t("Trading results on the test period", "Rezultatele tranzacționării pe perioada de test"), table(
        "@{}p{0.25\\textwidth}rrrrrr@{}",
        [t("Strategy", "Strategia"), t("Return \\%/yr", "Rand.\\ \\%/an"), t("Vol.\\ \\%/yr", "Vol.\\ \\%/an"), "Sharpe",
         t("Drawdown \\%", "Scădere max.\\ \\%"), t("Invested \\%", "Investit \\%"), t("Trades", "Tranz.")],
        [[t("Buy and hold", "Cumpără și păstrează"), "@{trade.bh_ret|.1f}", "@{trade.bh_vol|.1f}", "@{trade.bh_sharpe|.2f}", "@{trade.bh_dd|.1f}", "100", "1"],
         [t("200-day moving average", "Media mobilă de 200 de zile"), "@{trade.ma_ret|.1f}", "@{trade.ma_vol|.1f}", "@{trade.ma_sharpe|.2f}", "@{trade.ma_dd|.1f}",
          "@{trade.ma_inv|.0f}", "@{trade.ma_trades}"],
         [t("PPO (mean of 5 seeds)", "PPO (media a 5 valori inițiale)"), "@{trade.ppo_ret|.1f}", "@{trade.ppo_vol|.1f}", "@{trade.ppo_sharpe|.2f}",
          "@{trade.ppo_dd|.1f}", "@{trade.ppo_inv|.0f}", "@{trade.ppo_trades|.0f}"]], size="small") +
        "\\vspace{2mm}" + it(
            t("\\textbf{Uncertainty}: Sharpe difference of the seed-averaged PPO strategy minus buy-and-hold @{trade.dsh|+.2f}; 95\\,\\% stationary block-bootstrap "
              "interval [@{trade.dsh_lo|.2f}; @{trade.dsh_hi|.2f}] (2,000 resamples, mean block 21 days).",
              "\\textbf{Incertitudinea}: diferența dintre raportul Sharpe al strategiei PPO mediate pe valori inițiale și cel al strategiei cumpără și păstrează: @{trade.dsh|+.2f}; "
              "intervalul de 95\\,\\% din bootstrap staționar pe blocuri: [@{trade.dsh_lo|.2f}; @{trade.dsh_hi|.2f}] (2\\,000 de reeșantionări, bloc mediu de 21 de zile)."),
            t("PPO beats buy-and-hold on the Sharpe ratio in @{trade.ppo_beats_bh} of 5 seeds.",
              "PPO bate strategia cumpără și păstrează la raportul Sharpe pentru @{trade.ppo_beats_bh} din 5 valori inițiale."), sep="2pt") +
        D.bottom(t("an interval that contains zero means: no evidence that the agent adds value, whatever the backtest chart suggests.",
                   "un interval care conține zero înseamnă că nu există dovezi că agentul aduce valoare, orice ar sugera graficul retrospectiv."), NB + " \\S5"),
        "small", cite="(Politis \\& Romano, 1994; " + t("own computation", "calcul propriu") + ")")

    D.figframe(t("When is the agent out of the market?", "Cînd iese agentul din piață?"),
               t("exposure of the PPO agent (share of seeds and days invested) against the moving-average rule.",
                 "expunerea agentului PPO (ponderea valorilor inițiale și a zilelor investite) comparată cu regula mediei mobile."),
               None, "trading_exposure", bullets=[
                   t("Invested: PPO @{trade.ppo_inv|.0f}\\,\\% of test days; 20 Feb -- 30 Apr 2020 @{trade.ppo_inv_covid|.0f}\\,\\% (moving average @{trade.ma_inv_covid|.0f}\\,\\%); "
                     "2022 @{trade.ppo_inv_2022|.0f}\\,\\% (moving average @{trade.ma_inv_2022|.0f}\\,\\%).",
                     "Investit: PPO @{trade.ppo_inv|.0f}\\,\\% din zilele de test; 20 feb.\\ -- 30 apr.\\ 2020 @{trade.ppo_inv_covid|.0f}\\,\\% (media mobilă @{trade.ma_inv_covid|.0f}\\,\\%); "
                     "2022 @{trade.ppo_inv_2022|.0f}\\,\\% (media mobilă @{trade.ma_inv_2022|.0f}\\,\\%).")],
               job=t("read the policy, not only its profit: an agent that is simply always long is buy-and-hold with extra model risk.",
                     "citiți politica, nu doar profitul: un agent care este pur și simplu mereu investit este o strategie cumpără și păstrează cu risc de model suplimentar."),
               tryit=NB + " \\S5", height="0.46\\textheight", cite=OWN)

    D.frame(t("Why RL trading backtests mislead", "De ce induc în eroare testele retrospective ale agenților de tranzacționare"), it(
        t("\\textbf{Low signal-to-noise}: daily equity returns are close to unpredictable; the agent mostly learns the noise of its training windows.",
          "\\textbf{Raport semnal-zgomot redus}: randamentele zilnice ale acțiunilor sînt aproape imprevizibile; agentul învață mai ales zgomotul ferestrelor de antrenare."),
        t("\\textbf{Non-stationarity}: 2005--2016 contains 2008; 2020--2026 contains a pandemic crash, an inflation shock and a long rally.",
          "\\textbf{Nestaționaritatea}: 2005--2016 conține anul 2008; 2020--2026 conține prăbușirea din pandemie, șocul inflației și o creștere lungă."),
        t("\\textbf{Seed and selection risk}: five seeds already give a range of Sharpe ratios; reporting the best seed is backtest overfitting.",
          "\\textbf{Riscul valorii inițiale și al selecției}: cinci valori inițiale dau deja un interval de rapoarte Sharpe; raportarea celei mai bune este supraajustare retrospectivă."),
        t("\\textbf{Price impact and costs}: a price-taking simulator ignores market impact; costs must be in the reward, not added afterwards.",
          "\\textbf{Impactul asupra prețului și costurile}: un simulator în care agentul nu influențează prețul ignoră impactul; costurile trebuie incluse în recompensă, nu adăugate ulterior."),
        t("\\textbf{Where RL helps more}: execution (splitting a large order), market making, hedging, storage: problems with a real state and a known reward.",
          "\\textbf{Unde ajută mai mult RL}: execuția (împărțirea unui ordin mare), crearea de piață, acoperirea riscului, stocarea: probleme cu o stare reală și o recompensă cunoscută."),
        sep="4pt") + D.bottom(t("ask for the seed distribution, the cost assumption and a bootstrap interval before trusting any RL trading result.",
                                "cereți distribuția pe valori inițiale, ipoteza de cost și un interval bootstrap înainte de a avea încredere într-un rezultat RL de tranzacționare.")),
        "small", cite="(Hambly, Xu \\& Yang, 2023)")

    # ================================================================== 5. deep hedging
    D.section(t("Deep hedging: a policy that minimises tail risk", "Acoperirea neuronală: o politică ce minimizează riscul din coadă"))
    D.concept(t("Hedging a short call: the profit and loss of the desk", "Acoperirea unei opțiuni call vîndute: rezultatul desk-ului"),
              t("the desk sells a one-month at-the-money (ATM) call and trades the index every day to offset the risk.",
                "desk-ul vinde o opțiune call la bani (ATM) pe o lună și tranzacționează zilnic indicele pentru a compensa riscul."),
              "\\[ \\mathrm{PL}(\\delta)=C_0+\\sum_{k=0}^{20}\\delta_k\\,(S_{k+1}-S_k)-(S_{21}-K)^{+}-c\\sum_{k=0}^{21}S_k\\,|\\delta_k-\\delta_{k-1}| \\]" +
              "\\where{" + t("$S_k$ = S\\&P 500 close on day $k$ of a real 21-day path, $S_0=K=1$; $C_0$ = Black-Scholes premium (VIX of day 0, rate 0), "
                             "$\\delta_k$ = units held over $(k,k+1]$, $c$ = 10 bp.",
                             "$S_k$ = închiderea S\\&P 500 în ziua $k$ a unei traiectorii reale de 21 de zile, $S_0=K=1$; $C_0$ = prima Black-Scholes (VIX din ziua 0, rata 0), "
                             "$\\delta_k$ = unitățile deținute pe $(k,k+1]$, $c$ = 10 pb.") + "}",
              it(t("\\textbf{Paths}: one per start day; training @{hedge.n_train|,d} (2005--2016), validation @{hedge.n_valid|,d}, test @{hedge.n_test|,d} paths from 2020.",
                   "\\textbf{Traiectorii}: una pentru fiecare zi de început; antrenare @{hedge.n_train|,d} (2005--2016), validare @{hedge.n_valid|,d}, test @{hedge.n_test|,d} traiectorii din 2020."),
                 t("\\textbf{Test}: mean premium @{hedge.prem_mean|.2f}\\,\\%, mean payoff @{hedge.payoff_mean|.2f}\\,\\% of the index (mean VIX @{hedge.vix_mean_test|.1f}).",
                   "\\textbf{Test}: prima medie @{hedge.prem_mean|.2f}\\,\\%, plata medie @{hedge.payoff_mean|.2f}\\,\\% din indice (VIX mediu @{hedge.vix_mean_test|.1f})."),
                 t("\\textbf{Real, not simulated}: the paths keep the jumps and volatility clusters of 2008, 2020 and 2022.",
                   "\\textbf{Real, nu simulat}: traiectoriile păstrează salturile și grupările de volatilitate din 2008, 2020 și 2022."), sep="2pt"),
              t("the premium barely covers the mean payoff; the risk desk manages the tail.",
                "prima abia acoperă plata medie; departamentul de risc gestionează coada."),
              tryit=NB + " \\S6", cite="(Buehler et al., 2019)", wl="0.56", wr="0.41")

    D.concept(t("Black-Scholes delta and the deep-hedging objective", "Delta Black-Scholes și obiectivul acoperirii neuronale"),
              t("the benchmark hedges the model sensitivity; the network instead minimises the expected shortfall of the real loss, costs included.",
                "reperul acoperă sensibilitatea din model; rețeaua minimizează în schimb pierderea așteptată din coadă a pierderii reale, cu tot cu costuri."),
              "\\[ \\delta_k^{\\text{BS}}=\\Phi(d_1),\\quad d_1=\\frac{\\ln(S_k/K)+\\tfrac12\\sigma_k^{2}\\tau_k}{\\sigma_k\\sqrt{\\tau_k}} \\]"
              "\\[ \\min_{\\btheta,w}\\ w+\\frac{1}{1-\\alpha}\\E\\big[(-\\mathrm{PL}(\\delta_{\\btheta})-w)^{+}\\big]=\\min_{\\btheta}\\mathrm{ES}_{\\alpha}\\big(-\\mathrm{PL}(\\delta_{\\btheta})\\big) \\]" +
              "\\where{" + t("$\\sigma_k$ = VIX/100, $\\tau_k=(21-k)/252$, $\\alpha=0.95$; at the optimum $w$ is the value at risk (VaR); "
                             "$\\delta_{\\btheta}(\\ln S_k,\\tau_k,\\sigma_k,\\delta_{k-1})\\in[0,1]$: 2 layers of 32 ReLU units, sigmoid output.",
                             "$\\sigma_k$ = VIX/100, $\\tau_k=(21-k)/252$, $\\alpha=0{,}95$; la optim, $w$ este valoarea la risc (VaR); "
                             "$\\delta_{\\btheta}(\\ln S_k,\\tau_k,\\sigma_k,\\delta_{k-1})\\in[0,1]$: 2 straturi de cîte 32 de unități ReLU, ieșire sigmoidă.") + "}",
              it(t("\\textbf{Policy optimisation with a known reward}: the profit and loss is differentiable in $\\btheta$, so we back-propagate through all 21 days.",
                   "\\textbf{Optimizarea politicii cu recompensă cunoscută}: rezultatul este diferențiabil în $\\btheta$, deci propagăm gradientul înapoi prin toate cele 21 de zile."),
                 t("\\textbf{Previous hedge as input}: lets the network learn a no-trade band when costs are high.",
                   "\\textbf{Acoperirea anterioară ca intrare}: permite rețelei să învețe o bandă fără tranzacții atunci cînd costurile sînt mari."),
                 t("\\textbf{Training}: 600 full-batch Adam steps, five seeds.", "\\textbf{Antrenare}: 600 de pași Adam pe tot setul, cinci valori inițiale."), sep="3pt"),
              t("the objective is the risk measure the desk reports.",
                "obiectivul este chiar măsura de risc raportată de desk."),
              tryit=NB + " \\S6", cite="(Buehler et al., 2019; Rockafellar \\& Uryasev, 2000)", wl="0.56", wr="0.41")

    D.figframe(t("Hedging profit and loss on the test paths", "Rezultatul acoperirii pe traiectoriile de test"),
               t("hedging removes most of the left tail; the deep hedge cuts the remaining tail further, at lower turnover.",
                 "acoperirea elimină cea mai mare parte a cozii din stînga; acoperirea neuronală reduce și mai mult coada rămasă, cu un volum tranzacționat mai mic."),
               None, "hedge_pnl", bullets=[
                   t("Standard deviation: no hedge @{hedge.nh_std|.2f}, delta @{hedge.bs_std|.2f}, deep hedge @{hedge.dh_std|.2f} (\\% of the index).",
                     "Abaterea standard: fără acoperire @{hedge.nh_std|.2f}, delta @{hedge.bs_std|.2f}, acoperire neuronală @{hedge.dh_std|.2f} (\\% din indice).")],
               tryit=NB + " \\S6", height="0.52\\textheight", cite=OWN)

    D.frame(t("Hedging results on the test paths, 10 bp cost", "Rezultatele acoperirii pe traiectoriile de test, cost de 10 pb"), table(
        "@{}p{0.34\\textwidth}rrrr@{}",
        [t("Strategy", "Strategia"), t("Mean PL (\\%)", "PL mediu (\\%)"), t("Std.\\ dev.\\ (\\%)", "Abatere std.\\ (\\%)"),
         t("ES$_{95}$ of loss (\\%)", "ES$_{95}$ pierdere (\\%)"), t("Turnover", "Volum tranz.")],
        [[t("No hedge", "Fără acoperire"), "@{hedge.nh_mean|.2f}", "@{hedge.nh_std|.2f}", "@{hedge.nh_es|.2f}", "0"],
         [t("Black-Scholes delta (VIX)", "Delta Black-Scholes (VIX)"), "@{hedge.bs_mean|.2f}", "@{hedge.bs_std|.2f}", "@{hedge.bs_es|.2f}", "@{hedge.bs_turnover|.2f}"],
         [t("Deep hedge (mean of 5 seeds)", "Acoperire neuronală (media a 5 valori inițiale)"), "@{hedge.dh_mean|.2f}", "@{hedge.dh_std|.2f}", "@{hedge.dh_es|.2f}", "@{hedge.dh_turnover|.2f}"]],
        size="small") + "\\vspace{2mm}" + it(
            t("ES$_{95}$ of the deep hedge across seeds: @{hedge.dh_es_min|.2f} -- @{hedge.dh_es_max|.2f}; lower than the delta hedge in @{hedge.dh_beats_bs} of 5 seeds.",
              "ES$_{95}$ al acoperirii neuronale pe valori inițiale: @{hedge.dh_es_min|.2f} -- @{hedge.dh_es_max|.2f}; mai mic decît al acoperirii delta pentru @{hedge.dh_beats_bs} din 5 valori inițiale."),
            t("PL = profit and loss in \\% of the initial index level; turnover = mean $\\sum_k|\\delta_k-\\delta_{k-1}|$ per path.",
              "PL = rezultatul (profit sau pierdere) în \\% din nivelul inițial al indicelui; volumul tranzacționat = media $\\sum_k|\\delta_k-\\delta_{k-1}|$ pe traiectorie."), sep="2pt") +
        D.bottom(t("judge a hedge by the tail of the loss and by its turnover, not by its mean profit.",
                   "judecați o acoperire după coada pierderii și după volumul tranzacționat, nu după profitul mediu."), NB + " \\S6"),
        "small", cite=OWN)

    D.figframe(t("What the network learned: a flatter, cost-aware hedge ratio", "Ce a învățat rețeaua: un raport de acoperire mai plat, atent la costuri"),
               t("hedge ratio as a function of moneyness, 10 days before maturity, with the VIX at 20.",
                 "raportul de acoperire în funcție de raportul $S_t/K$, cu 10 zile înainte de scadență și VIX la 20."),
               None, "hedge_delta", bullets=[
                   t("At the money: deep hedge @{hedge.atm_dh|.2f} against Black-Scholes @{hedge.atm_bs|.2f}; the band shows the spread across five seeds.",
                     "La bani: acoperirea neuronală @{hedge.atm_dh|.2f}, față de Black-Scholes @{hedge.atm_bs|.2f}; banda arată dispersia pe cinci valori inițiale.")],
               tryit=NB + " \\S6", height="0.50\\textheight", cite=OWN)

    D.figframe(t("Transaction costs: where the learned hedge helps", "Costurile de tranzacționare: unde ajută acoperirea învățată"),
               t("ES$_{95}$ of the hedging loss for cost levels from 0 to 50 bp; one network trained per cost level.",
                 "ES$_{95}$ al pierderii din acoperire pentru costuri de la 0 la 50 pb; o rețea antrenată pentru fiecare nivel de cost."),
               None, "hedge_costs", bullets=[
                   t("No cost: delta @{hedge.bs_es_0|.2f}, deep hedge @{hedge.dh_es_0|.2f}; 50 bp: delta @{hedge.bs_es_50|.2f}, deep hedge @{hedge.dh_es_50|.2f} (\\% of the index).",
                     "Fără cost: delta @{hedge.bs_es_0|.2f}, acoperire neuronală @{hedge.dh_es_0|.2f}; 50 pb: delta @{hedge.bs_es_50|.2f}, acoperire neuronală @{hedge.dh_es_50|.2f} (\\% din indice).")],
               job=t("the value of a learned hedge grows with frictions: illiquid underlyings, wide spreads, energy and commodity hedges.",
                     "valoarea unei acoperiri învățate crește cu fricțiunile: active suport nelichide, diferențe mari între cotații, acoperiri pe energie și mărfuri."),
               tryit=NB + " \\S6", height="0.46\\textheight", cite=OWN)

    D.frame(t("Reinforcement learning in production: model risk", "Învățarea prin întărire în producție: riscul de model"), cols(it(
        t("\\textbf{Reward design}: the agent optimises exactly what you write; omit costs, risk or constraints and it will exploit the gap.",
          "\\textbf{Proiectarea recompensei}: agentul optimizează exact ce scrieți; dacă omiteți costurile, riscul sau restricțiile, va exploata lipsa."),
        t("\\textbf{Simulator gap}: a historical replay ignores the agent's own price impact and future regimes.",
          "\\textbf{Diferența față de simulator}: reluarea istorică ignoră impactul propriu asupra prețului și regimurile viitoare."),
        t("\\textbf{Offline evaluation}: one test period used once; several seeds; a test against the rule in use.",
          "\\textbf{Evaluarea offline}: o singură perioadă de test, folosită o dată; mai multe valori inițiale; un test față de regula folosită deja."), sep="4pt"),
        it(t("\\textbf{Guardrails}: action masks (state of charge, position limits), kill switches, human sign-off.",
             "\\textbf{Limite de siguranță}: măștile de acțiune (nivelul de încărcare, limitele de poziție), oprirea de urgență, aprobarea umană."),
           t("\\textbf{Explainability}: a table or a hedge-ratio curve can be audited; Lecture 11 explains neural policies.",
             "\\textbf{Explicabilitate}: un tabel sau o curbă a raportului de acoperire pot fi auditate; cursul 11 explică politicile neuronale."),
           t("\\textbf{Monitoring}: compare live capture ratio and turnover with the backtest every month.",
             "\\textbf{Monitorizare}: comparați lunar rata de captare și volumul tranzacționat cu testul retrospectiv."), sep="4pt")) +
        D.bottom(t("treat an RL policy like any model of the bank: documented, validated independently, monitored, with a fallback rule.",
                   "tratați o politică RL ca pe orice model al băncii: documentată, validată independent, monitorizată și cu o regulă de rezervă.")),
        "small", cite="(Hambly, Xu \\& Yang, 2023)")

    # ------------------------------------------------------------------ wrap-up
    D.parts.append("\\section*{" + t("Wrap-up", "Recapitulare") + "}\n")
    D.quiz_frame(t("perfect foresight", "previziunea perfectă"),
                 t("A vendor shows that its battery agent earned 95\\,\\% of the dynamic-programming revenue in a backtest in which the DP "
                   "used the day's own prices. What do you ask?",
                   "Un furnizor arată că agentul său pentru baterie a obținut 95\\,\\% din venitul programării dinamice într-un test retrospectiv "
                   "în care DP a folosit chiar prețurile zilei. Ce întrebați?"),
                 t("Whether the agent's state contained the day's prices: if so, the 95\\,\\% is look-ahead bias; a fair agent sees only yesterday "
                   "and is compared with yesterday's optimal plan, as in our test (DQN @{bat.dqn_cap|.1f}\\,\\%, naive @{bat.naive_cap|.1f}\\,\\%).",
                   "Dacă starea agentului conținea prețurile zilei: dacă da, cei 95\\,\\% provin din informație din viitor; un agent corect vede doar ziua de "
                   "ieri și este comparat cu planul optim de ieri, ca în testul nostru (DQN @{bat.dqn_cap|.1f}\\,\\%, planul naiv @{bat.naive_cap|.1f}\\,\\%)."))
    D.quiz_frame(t("PPO clipping", "plafonarea PPO"),
                 t("In a PPO update an action has advantage $\\hat A_t>0$ and the new policy already gives it 1.5 times its old probability. "
                   "What gradient does the clipped objective pass to this sample with $\\epsilon=0.2$?",
                   "Într-o actualizare PPO, o acțiune are avantajul $\\hat A_t>0$, iar noua politică îi dă deja de 1{,}5 ori probabilitatea veche. "
                   "Ce gradient transmite obiectivul plafonat pentru acest eșantion, cu $\\epsilon=0{,}2$?"),
                 t("Zero: $\\rho_t=1.5>1+\\epsilon$, the minimum picks the clipped term $1.2\\,\\hat A_t$, which does not depend on $\\btheta$; "
                   "the policy is not pushed further in this step.",
                   "Zero: $\\rho_t=1{,}5>1+\\epsilon$, minimul alege termenul plafonat $1{,}2\\,\\hat A_t$, care nu depinde de $\\btheta$; "
                   "politica nu mai este împinsă în acest pas."))
    D.quiz_frame(t("reading a trading result", "interpretarea unui rezultat de tranzacționare"),
                 t("Your PPO agent shows a test Sharpe ratio above buy-and-hold for its best seed. The bootstrap interval of the Sharpe "
                   "difference of the seed-averaged strategy is [@{trade.dsh_lo|.2f}; @{trade.dsh_hi|.2f}]. Do you deploy it?",
                   "Agentul PPO are, pentru cea mai bună valoare inițială, un raport Sharpe de test peste strategia cumpără și păstrează. Intervalul "
                   "bootstrap al diferenței de raport Sharpe pentru strategia mediată pe valori inițiale este [@{trade.dsh_lo|.2f}; @{trade.dsh_hi|.2f}]. Îl folosiți?"),
                 t("No: the best seed is a selected outcome, and an interval that contains zero gives no evidence of added value; keep the simple rule.",
                   "Nu: cea mai bună valoare inițială este un rezultat selectat, iar un interval care conține zero nu arată o valoare adăugată; păstrați regula simplă."))

    D.frame(t("Wrap-up", "Recapitulare"), it(
        t("\\textbf{MDP + Bellman}: write state, action, reward; DP solves small problems exactly and gives the ceiling "
          "(battery: @{bat.pf_k|.1f} thousand EUR on the test years).",
          "\\textbf{MDP + Bellman}: scrieți starea, acțiunea, recompensa; DP rezolvă exact problemele mici și dă plafonul "
          "(bateria: @{bat.pf_k|.1f} mii EUR pe anii de test)."),
        t("\\textbf{Value-based}: Q-learning and DQN learn from yesterday's prices; capture ratios @{bat.q_cap|.1f}\\,\\% and @{bat.dqn_cap|.1f}\\,\\%, "
          "naive plan @{bat.naive_cap|.1f}\\,\\%.",
          "\\textbf{Metode bazate pe valoare}: Q-learning și DQN învață din prețurile de ieri; rate de captare @{bat.q_cap|.1f}\\,\\% și @{bat.dqn_cap|.1f}\\,\\%, "
          "planul naiv @{bat.naive_cap|.1f}\\,\\%."),
        t("\\textbf{Policy gradient}: PPO on the S\\&P 500 gives a Sharpe ratio of @{trade.ppo_sharpe|.2f} against @{trade.bh_sharpe|.2f} for buy-and-hold, "
          "with a bootstrap interval that contains zero.",
          "\\textbf{Gradientul politicii}: PPO pe S\\&P 500 dă un raport Sharpe de @{trade.ppo_sharpe|.2f}, față de @{trade.bh_sharpe|.2f} pentru strategia cumpără și păstrează, "
          "cu un interval bootstrap care conține zero."),
        t("\\textbf{Deep hedging}: ES$_{95}$ of the loss @{hedge.dh_es|.2f}\\,\\% against @{hedge.bs_es|.2f}\\,\\% for the delta hedge at 10 bp.",
          "\\textbf{Acoperirea neuronală}: ES$_{95}$ al pierderii @{hedge.dh_es|.2f}\\,\\%, față de @{hedge.bs_es|.2f}\\,\\% pentru acoperirea delta, la 10 pb."),
        t("\\textbf{Verdict on real data}: battery --- the simple rule wins; trading --- higher Sharpe, but not significant; hedging --- the learned policy wins in every seed.",
          "\\textbf{Verdictul pe date reale}: bateria --- regula simplă cîștigă; tranzacționarea --- Sharpe mai mare, dar nesemnificativ; acoperirea --- politica învățată cîștigă pentru fiecare valoare inițială."),
        t("\\textbf{Next}: Lecture 11 opens the black box: explaining what a neural network has learned.",
          "\\textbf{Urmează}: cursul 11 deschide cutia neagră: explicarea a ceea ce a învățat o rețea neuronală."), sep="4pt"), "small")

    D.frame(t("Research from our group: storage, prices and learning in energy finance", "Cercetările grupului nostru: stocare, prețuri și învățare automată în finanțele energiei"), it(
        t("\\textbf{Battery arbitrage on day-ahead markets}: joint probabilistic forecasts of the hourly price vector feed the charge-discharge "
          "decision of a battery in Central and Eastern European day-ahead markets; forecasting the joint distribution, not only point prices, "
          "is what the dispatch needs (Andrei, Cepoi, Dumitrescu, Lessmann, Chiriac (Luchian), Pele \\& Petukhina, 2026, Social Science Research Network (SSRN) 6872381).",
          "\\textbf{Arbitrajul cu baterii pe piețele pentru ziua următoare}: prognozele probabilistice comune ale vectorului de prețuri orare alimentează "
          "decizia de încărcare și descărcare a unei baterii pe piețele din Europa Centrală și de Est; dispecerizarea are nevoie de distribuția comună, "
          "nu doar de prețuri punctuale (Andrei, Cepoi, Dumitrescu, Lessmann, Chiriac (Luchian), Pele \\& Petukhina, 2026, Social Science Research Network (SSRN) 6872381)."),
        t("\\textbf{Four generations of price models}: a comparative evaluation from statistical models to machine learning, deep learning and "
          "transformer models for energy prices (Andrei, Velev, Toma, Pele \\& Lessmann, 2024, arXiv 2411.03372) --- the forecasts a dispatch agent consumes.",
          "\\textbf{Patru generații de modele de preț}: o evaluare comparativă de la modelele statistice la învățarea automată, deep learning și modelele "
          "transformer pentru prețurile energiei (Andrei, Velev, Toma, Pele \\& Lessmann, 2024, arXiv 2411.03372) --- prognozele de care are nevoie un agent de dispecerizare."),
        t("\\textbf{Artificial intelligence (AI) in energy finance}: a systematic literature review of how machine learning and reinforcement learning reshape "
          "trading, risk and storage decisions in energy markets (Anagnoste et al., 2025, Energy Strategy Reviews).",
          "\\textbf{Inteligența artificială (AI) în finanțele energiei}: o analiză sistematică a literaturii despre modul în care învățarea automată și învățarea prin întărire "
          "schimbă deciziile de tranzacționare, de risc și de stocare pe piețele de energie (Anagnoste et al., 2025, Energy Strategy Reviews)."),
        sep="4pt") + "\\par\\vspace{1mm}{\\scriptsize " + items(t("Findings as stated in the papers; numbers on the other slides come from our own data.",
                                                                  "Rezultatele sînt cele din lucrări; cifrele de pe celelalte slide-uri provin din datele noastre."), nest=False) + "}",
        "footnotesize", cite=t("(Bucharest University of Economic Studies (ASE); Romanian Academy, Institute for Economic Forecasting (IPE))",
                               "(Academia de Studii Economice din București (ASE); Academia Română, Institutul de Prognoză Economică (IPE))"))

    D.frame(t("Further reading and resources", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Sutton \\& Barto (2018)}, the reference text, free online: ", "\\textbf{Sutton \\& Barto (2018)}, manualul de referință, gratuit online: ") +
        "\\href{http://incompleteideas.net/book/the-book-2nd.html}{incompleteideas.net/book}.",
        t("\\textbf{Spinning Up in Deep RL} (OpenAI), clear derivations of policy gradients and PPO: ",
          "\\textbf{Spinning Up in Deep RL} (OpenAI), derivări clare pentru gradienții de politică și PPO: ") +
        "\\href{https://spinningup.openai.com}{spinningup.openai.com}.",
        t("\\textbf{Hambly, Xu \\& Yang (2023)}: survey of RL in finance.", "\\textbf{Hambly, Xu \\& Yang (2023)}: o sinteză a aplicațiilor RL în finanțe."), sep="4pt"),
        it(t("\\textbf{This course}: code, data, notebooks --- ", "\\textbf{Acest curs}: cod, date, notebook-uri --- ") +
           "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Experiment yourself}: notebooks/L10\\_notebook.ipynb reproduces every method of this lecture; the seminar adds efficiency, "
             "battery size and cost experiments.",
             "\\textbf{Încercați singuri}: notebooks/L10\\_notebook\\_ro.ipynb reproduce fiecare metodă din acest curs; seminarul adaugă experimente "
             "cu randamentul, dimensiunea bateriei și costurile."),
           t("\\textbf{Deep hedging}: Buehler et al.\\ (2019), the paper behind the last section.",
             "\\textbf{Acoperirea neuronală}: Buehler et al.\\ (2019), lucrarea din spatele ultimei secțiuni."), sep="4pt"), "0.5", "0.47"), "small")

    reference_frames(D, REFS, per_frame=9)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.parts.append("\\section*{" + t("Appendix", "Anexă") + "}\n")
    D.frame(t("Appendix A: why Q-learning converges", "Anexa A: de ce converge Q-learning"), it(
        t("Bellman optimality operator $(\\mathcal{T}Q)(s,a)=R(s,a)+\\gamma\\E_{s'}\\max_{a'}Q(s',a')$ is a $\\gamma$-contraction in the sup norm: "
          "$\\|\\mathcal{T}Q_1-\\mathcal{T}Q_2\\|_\\infty\\le\\gamma\\|Q_1-Q_2\\|_\\infty$.",
          "Operatorul de optimalitate Bellman $(\\mathcal{T}Q)(s,a)=R(s,a)+\\gamma\\E_{s'}\\max_{a'}Q(s',a')$ este o $\\gamma$-contracție în norma sup: "
          "$\\|\\mathcal{T}Q_1-\\mathcal{T}Q_2\\|_\\infty\\le\\gamma\\|Q_1-Q_2\\|_\\infty$."),
        t("Q-learning is a stochastic approximation of the fixed point: $Q_{t+1}=(1-\\alpha_t)Q_t+\\alpha_t(\\mathcal{T}Q_t+\\varepsilon_t)$ with zero-mean noise $\\varepsilon_t$.",
          "Q-learning este o aproximare stocastică a punctului fix: $Q_{t+1}=(1-\\alpha_t)Q_t+\\alpha_t(\\mathcal{T}Q_t+\\varepsilon_t)$, cu zgomot $\\varepsilon_t$ de medie zero."),
        t("Robbins-Monro conditions $\\sum_t\\alpha_t(s,a)=\\infty$, $\\sum_t\\alpha_t^2(s,a)<\\infty$ and infinite visits give $Q_t\\to Q^{*}$ with probability one (Watkins \\& Dayan, 1992).",
          "Condițiile Robbins-Monro $\\sum_t\\alpha_t(s,a)=\\infty$, $\\sum_t\\alpha_t^2(s,a)<\\infty$ și vizitele infinite dau $Q_t\\to Q^{*}$ aproape sigur (Watkins \\& Dayan, 1992)."),
        t("With $\\gamma=1$ on a finite horizon the contraction argument is replaced by backward induction over the 24 steps.",
          "Cu $\\gamma=1$ pe un orizont finit, argumentul contracției este înlocuit de inducția înapoi pe cei 24 de pași."),
        t("A constant $\\alpha=0.05$ (as here) converges only to a neighbourhood of $Q^{*}$; the validation set picks the best epoch.",
          "Un $\\alpha=0{,}05$ constant (ca aici) converge doar într-o vecinătate a lui $Q^{*}$; setul de validare alege cea mai bună epocă."), sep="3pt"), "small",
        cite="(Watkins \\& Dayan, 1992; Sutton \\& Barto, 2018, ch.\\ 6)")
    D.frame(t("Appendix B: settings of the three agents", "Anexa B: setările celor trei agenți"), table(
        "@{}p{0.24\\textwidth}p{0.22\\textwidth}p{0.24\\textwidth}p{0.24\\textwidth}@{}",
        [t("Setting", "Setare"), t("Q-learning / DQN", "Q-learning / DQN"), "PPO", t("Deep hedge", "Acoperire neuronală")],
        [[t("Network", "Rețea"), t("table; 2$\\times$128 ReLU", "tabel; 2$\\times$128 ReLU"), t("2$\\times$64 tanh, shared", "2$\\times$64 tanh, comun"), "2$\\times$32 ReLU, sigmoid"],
         [t("Optimiser", "Optimizator"), "$\\alpha=0.05$; Adam $3\\cdot10^{-4}$" if not ro else "$\\alpha=0{,}05$; Adam $3\\cdot10^{-4}$",
          "Adam $3\\cdot10^{-4}$", "Adam $3\\cdot10^{-3}$"],
         [t("Discount / horizon", "Actualizare / orizont"), "$\\gamma=1$, 24 h", "$\\gamma=0.99$, $\\lambda=0.95$, 126 d" if not ro else "$\\gamma=0{,}99$, $\\lambda=0{,}95$, 126 z",
          t("21 days", "21 de zile")],
         [t("Exploration", "Explorare"), t("$\\varepsilon$: 1 $\\to$ 0.05", "$\\varepsilon$: 1 $\\to$ 0{,}05"), t("entropy 0.01", "entropie 0{,}01"), "--"],
         [t("Training budget", "Bugetul de antrenare"), t("40 epochs; 40 updates of 512 per hour", "40 de epoci; 40 de actualizări de 512 pe oră"),
          t("150 iterations $\\times$ 64 episodes", "150 de iterații $\\times$ 64 de episoade"), t("600 full-batch steps", "600 de pași pe tot setul")],
         [t("Selection", "Selecție"), t("best 2024 revenue", "cel mai bun venit în 2024"), t("best 2017--2019 Sharpe", "cel mai bun Sharpe 2017--2019"), t("none (fixed budget)", "fără (buget fix)")],
         [t("Run time (5 seeds)", "Timp de rulare (5 valori)"), "@{bat.seconds|,.0f} s", "@{trade.seconds|,.0f} s", "@{hedge.seconds|,.0f} s"]], size="footnotesize") +
        D.bottom(None, NB + " \\S3--\\S6"), "footnotesize", cite=OWN)
    D.frame(t("Appendix C: the generalised advantage estimator", "Anexa C: estimatorul generalizat al avantajului"), it(
        t("$k$-step advantage: $\\hat A_t^{(k)}=\\sum_{l=0}^{k-1}\\gamma^{l}r_{t+l+1}+\\gamma^{k}V(s_{t+k})-V(s_t)=\\sum_{l=0}^{k-1}\\gamma^{l}\\delta_{t+l}$.",
          "Avantajul pe $k$ pași: $\\hat A_t^{(k)}=\\sum_{l=0}^{k-1}\\gamma^{l}r_{t+l+1}+\\gamma^{k}V(s_{t+k})-V(s_t)=\\sum_{l=0}^{k-1}\\gamma^{l}\\delta_{t+l}$."),
        t("GAE is the exponentially weighted average $(1-\\lambda)\\sum_{k\\ge1}\\lambda^{k-1}\\hat A_t^{(k)}=\\sum_{l\\ge0}(\\gamma\\lambda)^{l}\\delta_{t+l}$.",
          "GAE este media ponderată exponențial $(1-\\lambda)\\sum_{k\\ge1}\\lambda^{k-1}\\hat A_t^{(k)}=\\sum_{l\\ge0}(\\gamma\\lambda)^{l}\\delta_{t+l}$."),
        t("Computed backwards in one pass: $\\hat A_t=\\delta_t+\\gamma\\lambda\\hat A_{t+1}$, $\\hat A_T=0$ (the loop of analysis\\_L10.py).",
          "Se calculează înapoi într-o singură trecere: $\\hat A_t=\\delta_t+\\gamma\\lambda\\hat A_{t+1}$, $\\hat A_T=0$ (bucla din analysis\\_L10.py)."),
        t("Bias-variance trade-off: small $\\lambda$ trusts the critic (biased if $V$ is wrong), large $\\lambda$ trusts the sampled rewards (noisy).",
          "Compromisul deplasare-varianță: un $\\lambda$ mic are încredere în critic (deplasat dacă $V$ este greșit), un $\\lambda$ mare în recompensele eșantionate (zgomotos)."), sep="3pt"),
        "small", cite="(Schulman et al., 2016)")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 10, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
