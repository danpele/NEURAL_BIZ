"""
make_quiz_L10.py -- quiz bank of Lecture 10 (EN + RO): assets/quizzes/l10.js.
Every number comes from L10_numbers.json; correct = index (0-3) of the right option; incorrectExplanation never names a letter.
Run:  python3 lectures/L10/make_quiz_L10.py
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = json.load(open(os.path.join(HERE, "L10_numbers.json")))


def g(key):
    v = N
    for p in key.split("."):
        v = v[p]
    return v


def E(key, d=2):
    return f"{g(key):,.{d}f}"


def R(key, d=2):
    return f"{g(key):,.{d}f}".replace(",", " ").replace(".", ",").replace("-", "−")


MEN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MRO = ["ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie"]


def D(key, lang):
    y, m, d = g(key).split("-")
    return f"{int(d)} {(MEN if lang == 'en' else MRO)[int(m) - 1]} {y}"


Q = []


def q(correct, en, ro):
    keys = ("title", "text", "options", "correctExplanation", "incorrectExplanation")
    assert len(en[2]) == 4 and len(ro[2]) == 4
    Q.append({"correct": correct, "en": dict(zip(keys, en)), "ro": dict(zip(keys, ro))})


q(1, ("Supervised or reinforcement", "Which business problem needs reinforcement learning rather than a forecast plus a fixed rule?",
       ["Predicting tomorrow's electricity price", "Deciding hour by hour when a battery charges, because today's action changes tomorrow's state of charge", "Classifying a transaction as fraud", "Estimating the default probability of a loan"],
       "RL is needed when actions change future states and options; the stored energy of a battery links today's decision to tomorrow's.",
       "Forecasting and classification have labels and no feedback from the decision to the next state; the battery decision is sequential."),
  ("Învățare supervizată sau reinforcement learning", "Ce problemă economică cere reinforcement learning, nu o prognoză urmată de o regulă fixă?",
       ["Prognoza prețului energiei de mîine", "Decizia, oră de oră, privind încărcarea unei baterii, deoarece acțiunea de azi schimbă nivelul de încărcare de mîine", "Clasificarea unei tranzacții ca fraudă", "Estimarea probabilității de nerambursare a unui credit"],
       "RL este necesară cînd acțiunile schimbă stările și opțiunile viitoare; energia stocată leagă decizia de azi de cea de mîine.",
       "Prognoza și clasificarea au etichete și nicio legătură între decizie și starea următoare; decizia bateriei este secvențială."))
q(0, ("Markov property", "A trading agent's state contains past returns and the VIX but not its current position, although each change of position costs 5 bp. What is wrong?",
       ["The state is not Markov: the cost of the next action depends on the current position, which is missing", "Nothing, positions are irrelevant", "The discount factor is too high", "The reward should not include costs"],
       "The reward depends on $|a_t-a_{t-1}|$, so $a_{t-1}$ must be in the state; otherwise the problem is not Markov in the observed state.",
       "Costs belong in the reward; the issue is that the state lacks the variable the reward depends on."),
  ("Proprietatea Markov", "Starea unui agent de tranzacționare conține randamentele trecute și VIX, dar nu și poziția curentă, deși fiecare schimbare de poziție costă 5 pb. Ce este greșit?",
       ["Starea nu este Markov: costul acțiunii următoare depinde de poziția curentă, care lipsește", "Nimic, poziția nu contează", "Factorul de actualizare este prea mare", "Recompensa nu ar trebui să includă costuri"],
       "Recompensa depinde de $|a_t-a_{t-1}|$, deci $a_{t-1}$ trebuie să fie în stare; altfel problema nu este Markov în starea observată.",
       "Costurile trebuie incluse în recompensă; problema este că starea nu conține variabila de care depinde recompensa."))
q(2, ("Value and action value", "What is the advantage $A^{\\pi}(s,a)$?",
       ["The reward of the next step", "The discounted sum of all future rewards", "$Q^{\\pi}(s,a)-V^{\\pi}(s)$, how much better action $a$ is than the policy's average action", "The probability of action $a$"],
       "The advantage compares one action with the policy's average in the same state; actor-critic methods learn from it.",
       "Rewards and returns are inputs to value functions; the advantage is a difference of two value functions."),
  ("Valoare și valoarea acțiunii", "Ce este avantajul $A^{\\pi}(s,a)$?",
       ["Recompensa pasului următor", "Suma actualizată a tuturor recompenselor viitoare", "$Q^{\\pi}(s,a)-V^{\\pi}(s)$, cu cît este acțiunea $a$ mai bună decît acțiunea medie a politicii", "Probabilitatea acțiunii $a$"],
       "Avantajul compară o acțiune cu media politicii în aceeași stare; metodele actor-critic învață din el.",
       "Recompensele și cîștigurile cumulate intră în funcțiile de valoare; avantajul este diferența a două funcții de valoare."))
q(3, ("Bellman optimality", "Which equation does dynamic programming solve backwards for the battery?",
       ["$V_h(s)=r_h$", "$V_h(s)=\\mathbb E[p_h]$", "$Q(s,a)\\leftarrow Q(s,a)+\\alpha\\delta$", "$V_h(s)=\\max_a\\{r_h(a)+V_{h+1}(s-a)\\}$ with $V_{24}=0$"],
       "On a finite horizon the Bellman optimality equation is solved exactly by one backward pass from midnight.",
       "The Q-learning update approximates the same equation from samples; DP uses the known prices and transitions directly."),
  ("Optimalitatea Bellman", "Ce ecuație rezolvă înapoi programarea dinamică pentru baterie?",
       ["$V_h(s)=r_h$", "$V_h(s)=\\mathbb E[p_h]$", "$Q(s,a)\\leftarrow Q(s,a)+\\alpha\\delta$", "$V_h(s)=\\max_a\\{r_h(a)+V_{h+1}(s-a)\\}$, cu $V_{24}=0$"],
       "Pe un orizont finit, ecuația de optimalitate Bellman se rezolvă exact printr-o singură trecere înapoi de la miezul nopții.",
       "Actualizarea Q-learning aproximează aceeași ecuație din eșantioane; DP folosește direct prețurile și tranzițiile cunoscute."))
q(1, ("Perfect foresight", f"The dynamic programme earns {E('bat.pf_k',1)} thousand EUR on the test days. Why can no real operator reach this number?",
       ["Because the battery is too small", "Because it uses each day's prices before they are known: it is an upper bound", "Because the efficiency is 100 %", "Because it ignores weekends"],
       "DP optimises with the day's 24 actual prices; any strategy that decides before prices are known earns at most this amount.",
       "The battery parameters are the same for every strategy; the difference is the information used."),
  ("Previziunea perfectă", f"Programarea dinamică obține {R('bat.pf_k',1)} mii EUR în zilele de test. De ce niciun operator real nu poate atinge această cifră?",
       ["Pentru că bateria este prea mică", "Pentru că folosește prețurile fiecărei zile înainte ca ele să fie cunoscute: este o limită superioară", "Pentru că randamentul este 100 %", "Pentru că ignoră sfîrșiturile de săptămînă"],
       "DP optimizează cu cele 24 de prețuri reale ale zilei; orice strategie care decide înainte de a cunoaște prețurile obține cel mult această sumă.",
       "Parametrii bateriei sînt aceiași pentru toate strategiile; diferența este informația folosită."))
q(2, ("Shadow price of energy", f"On {D('day.date', 'en')} at noon, $V_{{12}}(0)={E('day.v12_s0',0)}$ EUR and $V_{{12}}(2)={E('day.v12_s2',0)}$ EUR. What is one stored MWh worth at that hour?",
       [f"{E('day.v12_s0',0)} EUR", f"{E('day.v12_s2',0)} EUR", f"{E('day.v12_gain_per_mwh')} EUR, half of the difference", "Nothing, energy has no value at noon"],
       "The value of two extra MWh is $V_{12}(2)-V_{12}(0)$; per MWh this is half of it, the shadow price of stored energy.",
       "The value function at one level alone is the value of the whole day's remaining trading, not of the energy in the battery."),
  ("Prețul-umbră al energiei", f"Pe {D('day.date', 'ro')}, la prînz, $V_{{12}}(0)={R('day.v12_s0',0)}$ EUR și $V_{{12}}(2)={R('day.v12_s2',0)}$ EUR. Cît valorează un MWh stocat la acea oră?",
       [f"{R('day.v12_s0',0)} EUR", f"{R('day.v12_s2',0)} EUR", f"{R('day.v12_gain_per_mwh')} EUR, jumătate din diferență", "Nimic, energia nu are valoare la prînz"],
       "Valoarea a doi MWh în plus este $V_{12}(2)-V_{12}(0)$; pe MWh este jumătate din aceasta, prețul-umbră al energiei stocate.",
       "Funcția de valoare la un singur nivel este valoarea întregii tranzacționări rămase din zi, nu a energiei din baterie."))
q(0, ("Spreads", f"The mean daily max-min spread in Romania was {E('spread.ro_2019',1)} EUR/MWh in 2019 and {E('spread.ro_2025',1)} in 2025. What does this mean for a battery?",
       ["Its income opportunity grew, because it earns the spread, not the price level", "Its income fell, because prices rose", "Nothing, batteries earn the average price", "It should discharge only at night"],
       "A battery buys low and sells high within the day; its revenue scales with the spread.",
       "A higher price level alone does not help a battery; what matters is the gap between cheap and dear hours."),
  ("Diferențele de preț", f"Diferența zilnică medie maxim-minim din România a fost {R('spread.ro_2019',1)} EUR/MWh în 2019 și {R('spread.ro_2025',1)} în 2025. Ce înseamnă aceasta pentru o baterie?",
       ["Oportunitatea de venit a crescut, pentru că bateria cîștigă diferența, nu nivelul prețului", "Venitul a scăzut, pentru că prețurile au crescut", "Nimic, bateriile cîștigă prețul mediu", "Ar trebui să se descarce doar noaptea"],
       "O baterie cumpără ieftin și vinde scump în aceeași zi; venitul ei crește cu diferența de preț.",
       "Un nivel mai mare al prețului nu ajută singur o baterie; contează distanța dintre orele ieftine și cele scumpe."))
q(3, ("Q-learning update", "In the update $Q(s,a)\\leftarrow Q(s,a)+\\alpha[r+\\gamma\\max_{a'}Q(s',a')-Q(s,a)]$, why is Q-learning called off-policy?",
       ["Because it uses no policy", "Because $\\alpha$ is constant", "Because it needs a model of the market", "Because the target uses the greedy action in $s'$, whatever action the exploring agent takes next"],
       "The target evaluates the greedy policy while the data come from an exploring ($\\varepsilon$-greedy) policy.",
       "Q-learning is model-free and the step size is unrelated; the target's greedy maximum is what makes it off-policy."),
  ("Actualizarea Q-learning", "În actualizarea $Q(s,a)\\leftarrow Q(s,a)+\\alpha[r+\\gamma\\max_{a'}Q(s',a')-Q(s,a)]$, de ce se spune că Q-learning este off-policy?",
       ["Pentru că nu folosește nicio politică", "Pentru că $\\alpha$ este constant", "Pentru că are nevoie de un model al pieței", "Pentru că ținta folosește acțiunea lacomă din $s'$, indiferent ce acțiune alege apoi agentul care explorează"],
       "Ținta evaluează politica lacomă, în timp ce datele provin dintr-o politică ce explorează ($\\varepsilon$-lacomă).",
       "Q-learning nu are nevoie de model, iar pasul nu are legătură cu aceasta; maximul lacom din țintă îl face off-policy."))
q(1, ("Exploration", "Why does Q-learning start with $\\varepsilon=1$ and decay it to 0.05?",
       ["To make training deterministic", "To visit many state-action pairs early, so that bad initial estimates can be corrected", "To reduce the reward", "To avoid using a discount factor"],
       "Without exploration a poor early estimate is never revisited; decaying $\\varepsilon$ shifts from exploring to exploiting.",
       "Exploration is about which actions are tried; it does not change rewards or discounting."),
  ("Explorarea", "De ce pornește Q-learning cu $\\varepsilon=1$ și îl reduce la 0,05?",
       ["Pentru ca antrenarea să fie deterministă", "Pentru a vizita multe perechi stare-acțiune la început, astfel încît estimările inițiale proaste să poată fi corectate", "Pentru a reduce recompensa", "Pentru a evita un factor de actualizare"],
       "Fără explorare, o estimare proastă de la început nu mai este revizuită; reducerea lui $\\varepsilon$ trece de la explorare la exploatare.",
       "Explorarea privește acțiunile încercate; nu schimbă recompensele sau actualizarea."))
q(2, ("Look-ahead bias", "Why does the battery agent's state contain yesterday's prices and not today's?",
       ["Because yesterday's prices are cheaper to store", "Because today's prices are not on energy-charts.info", "Because the operator must plan before today's prices are known; using them would be look-ahead bias", "Because Q-learning cannot read 24 numbers"],
       "A state that leaks the day's prices would give a perfect but useless backtest; the information set must match the decision time.",
       "The data exist and the network can read them; the issue is what is known when the decision is taken."),
  ("Informație din viitor", "De ce conține starea agentului de baterie prețurile de ieri, nu pe cele de azi?",
       ["Pentru că prețurile de ieri sînt mai ușor de stocat", "Pentru că prețurile de azi nu sînt pe energy-charts.info", "Pentru că operatorul planifică înainte să cunoască prețurile de azi; folosirea lor ar introduce informație din viitor", "Pentru că Q-learning nu poate citi 24 de numere"],
       "O stare care conține prețurile zilei ar da un test retrospectiv perfect, dar inutil; informația trebuie să corespundă momentului deciziei.",
       "Datele există, iar rețeaua le poate citi; problema este ce se știe în momentul deciziei."))
q(0, ("Battery result", f"On the test days yesterday's optimal plan captured {E('bat.naive_cap',1)} % of the perfect-foresight revenue, tabular Q-learning {E('bat.q_cap',1)} % and the double DQN {E('bat.dqn_cap',1)} %. What is the honest conclusion?",
       ["On this problem the simple rule beats both learning agents; RL must be benchmarked against it", "The DQN is the best strategy", "Q-learning beats the rule in every seed", "Perfect foresight is achievable"],
       f"The naive plan optimises on all 24 of yesterday's prices; the agents beat it in {g('bat.dqn_beats_naive_seeds')} of 5 seeds (DQN) and {g('bat.q_beats_naive_seeds')} of 5 (Q-learning).",
       "The ranking is clear in every seed: the rule first, then the two agents; perfect foresight is only an upper bound."),
  ("Rezultatul bateriei", f"În zilele de test, planul optim de ieri a obținut {R('bat.naive_cap',1)} % din venitul cu previziune perfectă, Q-learning tabelar {R('bat.q_cap',1)} %, iar DQN dublu {R('bat.dqn_cap',1)} %. Care este concluzia onestă?",
       ["Pe această problemă regula simplă bate ambii agenți; RL trebuie comparat cu ea", "DQN este cea mai bună strategie", "Q-learning bate regula pentru fiecare valoare inițială", "Previziunea perfectă poate fi atinsă"],
       f"Planul naiv optimizează pe toate cele 24 de prețuri de ieri; agenții îl bat pentru {g('bat.dqn_beats_naive_seeds')} din 5 valori inițiale (DQN) și {g('bat.q_beats_naive_seeds')} din 5 (Q-learning).",
       "Clasamentul este clar pentru fiecare valoare inițială: regula prima, apoi cei doi agenți; previziunea perfectă este doar o limită superioară."))
q(3, ("Diebold-Mariano", f"The DM statistic of the daily revenue difference DQN minus naive plan is {E('bat.dm_t')}. How is it read?",
       ["DQN is significantly better", "The difference is not significant", "The test cannot be applied to revenues", "DQN is significantly worse than the naive plan"],
       f"A large negative $t$ with Newey-West variance means the mean daily difference ({E('bat.dqn_minus_naive_day',1)} EUR) is significantly below zero.",
       "The sign and size of the statistic decide: a large negative value is strong evidence against the agent."),
  ("Diebold-Mariano", f"Statistica DM a diferenței zilnice de venit DQN minus planul naiv este {R('bat.dm_t')}. Cum se interpretează?",
       ["DQN este semnificativ mai bun", "Diferența nu este semnificativă", "Testul nu se poate aplica veniturilor", "DQN este semnificativ mai slab decît planul naiv"],
       f"Un $t$ mare și negativ, cu varianță Newey-West, înseamnă că diferența zilnică medie ({R('bat.dqn_minus_naive_day',1)} EUR) este semnificativ sub zero.",
       "Semnul și mărimea statisticii decid: o valoare mare și negativă este o dovadă puternică împotriva agentului."))
q(1, ("Experience replay", "What does the replay memory of a DQN achieve?",
       ["It stores the test data", "It breaks the correlation between consecutive transitions and reuses each experience many times", "It makes the network linear", "It removes the need for exploration"],
       "Sampling minibatches from a memory of past transitions decorrelates the updates and improves data efficiency (Mnih et al., 2015).",
       "Replay does not touch test data or exploration; it changes which transitions each gradient step uses."),
  ("Memoria de reluare", "Ce obține memoria de reluare a unui DQN?",
       ["Stochează datele de test", "Elimină corelația dintre tranzițiile consecutive și refolosește fiecare experiență de mai multe ori", "Face rețeaua liniară", "Elimină nevoia de explorare"],
       "Eșantionarea loturilor dintr-o memorie a tranzițiilor trecute decorelează actualizările și folosește mai bine datele (Mnih et al., 2015).",
       "Reluarea nu atinge datele de test sau explorarea; ea schimbă tranzițiile folosite la fiecare pas de gradient."))
q(2, ("Double DQN", "Why does double DQN let the online network choose the next action and the target network evaluate it?",
       ["To halve the training time", "To make the policy stochastic", "To remove the upward bias of taking a maximum over noisy Q estimates", "To avoid a discount factor"],
       "Using the same noisy estimates to choose and to evaluate the maximum overestimates values; splitting the roles reduces the bias (van Hasselt et al., 2016).",
       "The change concerns the target value, not speed or stochasticity of the policy."),
  ("DQN dublu", "De ce lasă DQN dublu rețeaua online să aleagă acțiunea următoare, iar rețeaua-țintă să o evalueze?",
       ["Pentru a înjumătăți timpul de antrenare", "Pentru a face politica stocastică", "Pentru a elimina deplasarea în sus produsă de maximul unor estimări Q zgomotoase", "Pentru a evita factorul de actualizare"],
       "Folosirea acelorași estimări zgomotoase pentru alegerea și evaluarea maximului supraestimează valorile; separarea rolurilor reduce deplasarea (van Hasselt et al., 2016).",
       "Schimbarea privește valoarea-țintă, nu viteza sau caracterul stocastic al politicii."))
q(0, ("Policy gradient", "What does the term $\\nabla_{\\theta}\\log\\pi_{\\theta}(a_t\\mid s_t)\\,(G_t-b(s_t))$ do in REINFORCE?",
       ["It raises the probability of actions whose return beat the baseline and lowers the others", "It computes the transition law of the market", "It minimises the squared error of the critic", "It clips the probability ratio"],
       "The score function points towards making $a_t$ more likely; the sign of $G_t-b$ decides whether to follow it.",
       "Clipping belongs to PPO and the squared error to the critic; the policy-gradient term needs no transition law."),
  ("Gradientul politicii", "Ce face termenul $\\nabla_{\\theta}\\log\\pi_{\\theta}(a_t\\mid s_t)\\,(G_t-b(s_t))$ în REINFORCE?",
       ["Crește probabilitatea acțiunilor al căror cîștig a depășit reperul și o scade pentru celelalte", "Calculează legea de tranziție a pieței", "Minimizează eroarea pătratică a criticului", "Plafonează raportul probabilităților"],
       "Funcția scor indică direcția în care $a_t$ devine mai probabilă; semnul lui $G_t-b$ decide dacă o urmăm.",
       "Plafonarea aparține PPO, iar eroarea pătratică criticului; termenul gradientului politicii nu cere legea de tranziție."))
q(3, ("Baseline", "Why may any function of the state be subtracted from the return in a policy gradient?",
       ["Because it changes the optimal policy", "Because it makes the reward positive", "Because it removes the need for a critic", "Because its expected contribution to the gradient is zero, so the gradient stays unbiased while its variance falls"],
       "$\\mathbb E[\\nabla\\log\\pi(a\\mid s)\\,b(s)]=0$, so the baseline lowers variance without bias; $V(s)$ is the usual choice.",
       "A baseline does not change the optimum; the critic is one way to provide it."),
  ("Reperul", "De ce se poate scădea din cîștig orice funcție de stare într-un gradient de politică?",
       ["Pentru că schimbă politica optimă", "Pentru că face recompensa pozitivă", "Pentru că elimină nevoia unui critic", "Pentru că contribuția ei așteptată la gradient este zero, deci gradientul rămîne nedeplasat, iar varianța scade"],
       "$\\mathbb E[\\nabla\\log\\pi(a\\mid s)\\,b(s)]=0$, deci reperul reduce varianța fără deplasare; $V(s)$ este alegerea obișnuită.",
       "Reperul nu schimbă optimul; criticul este o modalitate de a-l furniza."))
q(1, ("GAE", "In generalised advantage estimation, what does $\\lambda=0$ give?",
       ["The Monte Carlo advantage", "The one-step TD advantage $\\delta_t=r_{t+1}+\\gamma V(s_{t+1})-V(s_t)$", "A zero advantage", "The PPO ratio"],
       "$\\hat A_t=\\sum_l(\\gamma\\lambda)^l\\delta_{t+l}$ reduces to $\\delta_t$ when $\\lambda=0$: low variance, biased if $V$ is wrong.",
       "$\\lambda=1$ gives the Monte Carlo advantage; $\\lambda=0$ keeps only the first TD error."),
  ("GAE", "În estimarea generalizată a avantajului, ce se obține pentru $\\lambda=0$?",
       ["Avantajul Monte Carlo", "Avantajul TD de un pas, $\\delta_t=r_{t+1}+\\gamma V(s_{t+1})-V(s_t)$", "Un avantaj nul", "Raportul PPO"],
       "$\\hat A_t=\\sum_l(\\gamma\\lambda)^l\\delta_{t+l}$ se reduce la $\\delta_t$ cînd $\\lambda=0$: varianță mică, deplasat dacă $V$ este greșit.",
       "$\\lambda=1$ dă avantajul Monte Carlo; $\\lambda=0$ păstrează doar prima eroare TD."))
q(2, ("PPO clipping", "With $\\epsilon=0.2$ and a positive advantage, what gradient does the clipped PPO objective give to a sample whose probability ratio is already 1.5?",
       ["A large positive gradient", "A negative gradient", "Zero: the clipped term $1.2\\hat A_t$ is selected and does not depend on $\\theta$", "The same gradient as REINFORCE"],
       "$\\min(\\rho\\hat A,\\mathrm{clip}(\\rho,0.8,1.2)\\hat A)$ picks the constant clipped term for $\\rho>1.2$, so the update stops pushing.",
       "Clipping removes the incentive to move the policy further once the ratio leaves $[1-\\epsilon,1+\\epsilon]$ in the favourable direction."),
  ("Plafonarea PPO", "Cu $\\epsilon=0{,}2$ și un avantaj pozitiv, ce gradient dă obiectivul PPO plafonat unui eșantion al cărui raport de probabilitate este deja 1,5?",
       ["Un gradient pozitiv mare", "Un gradient negativ", "Zero: se alege termenul plafonat $1{,}2\\hat A_t$, care nu depinde de $\\theta$", "Același gradient ca REINFORCE"],
       "$\\min(\\rho\\hat A,\\mathrm{clip}(\\rho;0{,}8;1{,}2)\\hat A)$ alege termenul plafonat constant pentru $\\rho>1{,}2$, deci actualizarea nu mai împinge.",
       "Plafonarea elimină stimulentul de a muta politica mai departe odată ce raportul iese din $[1-\\epsilon,1+\\epsilon]$ în direcția favorabilă."))
q(0, ("Trading reward", "Why does the trading reward include $-c\\,|a_t-a_{t-1}|$?",
       ["So that the agent learns to trade only when the expected gain exceeds the cost", "To make the reward always negative", "To penalise holding the index", "Because the Sharpe ratio requires it"],
       "Costs inside the reward shape the policy; an agent trained without them churns its position.",
       "The term only charges changes of position, not holding; it represents the real trading cost."),
  ("Recompensa tranzacționării", "De ce include recompensa tranzacționării termenul $-c\\,|a_t-a_{t-1}|$?",
       ["Pentru ca agentul să învețe să tranzacționeze doar cînd cîștigul așteptat depășește costul", "Pentru ca recompensa să fie mereu negativă", "Pentru a penaliza deținerea indicelui", "Pentru că o cere raportul Sharpe"],
       "Costurile din recompensă modelează politica; un agent antrenat fără ele își schimbă poziția mult prea des.",
       "Termenul taxează doar schimbările de poziție, nu deținerea; el reprezintă costul real al tranzacțiilor."))
q(3, ("Reading a trading backtest", f"PPO has a test Sharpe ratio of {E('trade.ppo_sharpe')} against {E('trade.bh_sharpe')} for buy-and-hold, and the 95 % bootstrap interval of the difference is [{E('trade.dsh_lo')}; {E('trade.dsh_hi')}]. What do you conclude?",
       ["PPO is proven superior", "PPO is proven inferior", "The bootstrap is invalid for Sharpe ratios", "The higher Sharpe ratio is not statistically established: the interval contains zero"],
       "An interval that contains zero means the data do not establish an advantage, even though all five seeds had a higher point estimate.",
       "Point estimates are not evidence by themselves; the block bootstrap accounts for the dependence of daily returns."),
  ("Interpretarea unui test retrospectiv", f"PPO are un raport Sharpe de test de {R('trade.ppo_sharpe')}, față de {R('trade.bh_sharpe')} pentru cumpără și păstrează, iar intervalul bootstrap de 95 % al diferenței este [{R('trade.dsh_lo')}; {R('trade.dsh_hi')}]. Ce concluzionați?",
       ["PPO este dovedit superior", "PPO este dovedit inferior", "Bootstrap-ul nu este valid pentru rapoartele Sharpe", "Raportul Sharpe mai mare nu este stabilit statistic: intervalul conține zero"],
       "Un interval care conține zero înseamnă că datele nu stabilesc un avantaj, deși toate cele cinci valori inițiale au avut o estimare punctuală mai mare.",
       "Estimările punctuale nu sînt dovezi prin ele însele; bootstrap-ul pe blocuri ține cont de dependența randamentelor zilnice."))
q(1, ("Agent exposure", f"Between 20 February and 30 April 2020 the PPO agent was invested {E('trade.ppo_inv_covid',0)} % of the time and the moving-average rule {E('trade.ma_inv_covid',0)} %. What does reading the exposure add?",
       ["Nothing beyond the Sharpe ratio", "It shows how the policy behaves in a crisis, which the summary statistics hide", "It proves that the agent predicts crashes", "It replaces the bootstrap"],
       "Reading positions tells you what the agent actually does; an always-long agent would be buy-and-hold with extra model risk.",
       "Exposure is descriptive; it neither proves forecasting skill nor replaces a test of significance."),
  ("Expunerea agentului", f"Între 20 februarie și 30 aprilie 2020, agentul PPO a fost investit {R('trade.ppo_inv_covid',0)} % din timp, iar regula mediei mobile {R('trade.ma_inv_covid',0)} %. Ce aduce citirea expunerii?",
       ["Nimic în plus față de raportul Sharpe", "Arată cum se comportă politica într-o criză, ceea ce statisticile sintetice ascund", "Dovedește că agentul prognozează prăbușirile", "Înlocuiește bootstrap-ul"],
       "Citirea pozițiilor arată ce face efectiv agentul; un agent mereu investit ar fi o strategie cumpără și păstrează cu risc de model în plus.",
       "Expunerea este descriptivă; nu dovedește o capacitate de prognoză și nu înlocuiește un test de semnificație."))
q(2, ("Seeds", "Why are all RL results in this lecture reported over five seeds?",
       ["Because five is required by law", "To make training faster", "Because RL training is noisy: one seed can be lucky, and the best seed is a selected result", "Because the data change with the seed"],
       "The spread over seeds shows how much of a result is luck; reporting only the best seed is backtest overfitting.",
       "The data are the same for every seed; only the random initialisation and sampling change."),
  ("Valori inițiale", "De ce sînt raportate toate rezultatele RL din acest curs pe cinci valori inițiale?",
       ["Pentru că legea cere cinci", "Pentru a accelera antrenarea", "Pentru că antrenarea RL este zgomotoasă: o valoare inițială poate fi norocoasă, iar cea mai bună este un rezultat selectat", "Pentru că datele se schimbă cu valoarea inițială"],
       "Dispersia pe valori inițiale arată cît dintr-un rezultat este noroc; raportarea doar a celei mai bune este supraajustare retrospectivă.",
       "Datele sînt aceleași pentru fiecare valoare inițială; se schimbă doar inițializarea aleatoare și eșantionarea."))
q(0, ("Hedging P&L", "In the hedging profit and loss, which term pays for the option?",
       ["$C_0$, the premium received when the call is sold", "$\\sum_k\\delta_k(S_{k+1}-S_k)$", "$(S_{21}-K)^+$", "$c\\sum_kS_k|\\delta_k-\\delta_{k-1}|$"],
       "The desk receives the premium $C_0$ up front, pays the payoff $(S_{21}-K)^+$ at maturity and gains or loses on the hedge in between.",
       "The sum of $\\delta_k$ terms is the hedge gain, the payoff is what the desk owes, and the last term is the cost of trading."),
  ("Rezultatul acoperirii", "În rezultatul acoperirii, ce termen plătește opțiunea?",
       ["$C_0$, prima încasată la vînzarea opțiunii call", "$\\sum_k\\delta_k(S_{k+1}-S_k)$", "$(S_{21}-K)^+$", "$c\\sum_kS_k|\\delta_k-\\delta_{k-1}|$"],
       "Desk-ul încasează prima $C_0$ la început, plătește la scadență $(S_{21}-K)^+$ și cîștigă sau pierde din acoperire între timp.",
       "Suma termenilor cu $\\delta_k$ este cîștigul din acoperire, plata finală este ce datorează desk-ul, iar ultimul termen este costul tranzacțiilor."))
q(3, ("Expected shortfall objective", "Why is the deep hedge trained by minimising $w+\\frac{1}{1-\\alpha}\\mathbb E[(L-w)^+]$ over $w$ and the network?",
       ["Because it is the mean of the loss", "Because it equals the variance of the loss", "Because it makes the hedge ratio equal to the Black-Scholes delta", "Because its minimum over $w$ is the expected shortfall of the loss (Rockafellar and Uryasev, 2000)"],
       "The Rockafellar-Uryasev form turns ES minimisation into a smooth joint problem; at the optimum $w$ is the VaR.",
       "The objective targets the tail of the loss, not its mean or variance, and it does not force the Black-Scholes delta."),
  ("Obiectivul ES", "De ce se antrenează acoperirea neuronală minimizînd $w+\\frac{1}{1-\\alpha}\\mathbb E[(L-w)^+]$ în raport cu $w$ și rețeaua?",
       ["Pentru că este media pierderii", "Pentru că este egal cu varianța pierderii", "Pentru că face raportul de acoperire egal cu delta Black-Scholes", "Pentru că minimul său în raport cu $w$ este pierderea așteptată în coadă a pierderii (Rockafellar și Uryasev, 2000)"],
       "Forma Rockafellar-Uryasev transformă minimizarea ES într-o problemă comună netedă; la optim, $w$ este VaR.",
       "Obiectivul vizează coada pierderii, nu media sau varianța ei, și nu impune delta Black-Scholes."))
q(1, ("Hedging result", f"At a 10 bp cost the ES$_{{95}}$ of the loss is {E('hedge.dh_es')} % for the deep hedge and {E('hedge.bs_es')} % for the Black-Scholes delta. What else should be compared?",
       ["Only the mean profit", f"The turnover: {E('hedge.dh_turnover')} for the deep hedge against {E('hedge.bs_turnover')} for the delta hedge, and the spread across seeds", "The colour of the histogram", "Nothing; ES is enough"],
       f"The deep hedge also trades less, and its ES ranges from {E('hedge.dh_es_min')} to {E('hedge.dh_es_max')} across seeds, below the delta hedge in {g('hedge.dh_beats_bs')} of 5.",
       "A hedge is judged by its tail risk, its cost of trading and its stability across training runs."),
  ("Rezultatul acoperirii", f"La un cost de 10 pb, ES$_{{95}}$ al pierderii este {R('hedge.dh_es')} % pentru acoperirea neuronală și {R('hedge.bs_es')} % pentru delta Black-Scholes. Ce altceva trebuie comparat?",
       ["Doar profitul mediu", f"Volumul tranzacționat: {R('hedge.dh_turnover')} pentru acoperirea neuronală, față de {R('hedge.bs_turnover')} pentru acoperirea delta, și dispersia pe valori inițiale", "Culoarea histogramei", "Nimic; ES este suficient"],
       f"Acoperirea neuronală tranzacționează și mai puțin, iar ES-ul ei variază între {R('hedge.dh_es_min')} și {R('hedge.dh_es_max')} pe valori inițiale, sub acoperirea delta pentru {g('hedge.dh_beats_bs')} din 5.",
       "O acoperire se judecă după riscul din coadă, după costul tranzacțiilor și după stabilitatea între antrenări."))
q(2, ("Hedge ratio shape", f"Ten days before maturity, at the money, the deep hedge holds {E('hedge.atm_dh')} against a Black-Scholes delta of {E('hedge.atm_bs')}. Why can a learned hedge differ from the delta?",
       ["Because the network ignores the index level", "Because the delta is wrong by construction", "Because it optimises the tail loss with costs on real paths, not a frictionless model", "Because the VIX is not used"],
       "Black-Scholes delta is optimal only without costs and under its model; the network optimises the realised ES with costs.",
       "Both use the index level and the VIX; the objectives and assumptions differ."),
  ("Forma raportului de acoperire", f"Cu zece zile înainte de scadență, la bani, acoperirea neuronală deține {R('hedge.atm_dh')}, față de delta Black-Scholes {R('hedge.atm_bs')}. De ce poate diferi o acoperire învățată de delta?",
       ["Pentru că rețeaua ignoră nivelul indicelui", "Pentru că delta este greșit prin construcție", "Pentru că optimizează pierderea din coadă, cu costuri, pe traiectorii reale, nu un model fără fricțiuni", "Pentru că VIX nu este folosit"],
       "Delta Black-Scholes este optim doar fără costuri și în ipotezele modelului; rețeaua optimizează ES-ul realizat, cu costuri.",
       "Ambele folosesc nivelul indicelui și VIX; diferă obiectivele și ipotezele."))
q(0, ("Costs and hedging", f"Without costs, ES$_{{95}}$ is {E('hedge.bs_es_0')} % for delta and {E('hedge.dh_es_0')} % for the deep hedge; at 50 bp, {E('hedge.bs_es_50')} % and {E('hedge.dh_es_50')} %. Where does a learned hedge help most?",
       ["In markets with real frictions and fat tails, where the model assumptions of the delta fail", "Only when costs are zero", "Never; the delta is always optimal", "Only for European options on bonds"],
       "The learned policy adapts to costs and to the real distribution of paths; the delta hedge does not.",
       "The numbers show a gap at every cost level; the policy is not restricted to one asset class."),
  ("Costuri și acoperire", f"Fără costuri, ES$_{{95}}$ este {R('hedge.bs_es_0')} % pentru delta și {R('hedge.dh_es_0')} % pentru acoperirea neuronală; la 50 pb, {R('hedge.bs_es_50')} % și {R('hedge.dh_es_50')} %. Unde ajută cel mai mult o acoperire învățată?",
       ["Pe piețe cu fricțiuni reale și cozi groase, unde ipotezele modelului delta nu se verifică", "Doar cînd costurile sînt zero", "Niciodată; delta este mereu optim", "Doar pentru opțiuni europene pe obligațiuni"],
       "Politica învățată se adaptează la costuri și la distribuția reală a traiectoriilor; acoperirea delta nu.",
       "Cifrele arată o diferență la fiecare nivel de cost; politica nu este limitată la o clasă de active."))
q(3, ("Simulator gap", "A trading agent is trained on a historical replay of prices. Which effect does the replay ignore?",
       ["Transaction costs", "Volatility clustering", "Weekends", "The agent's own price impact"],
       "Replaying history assumes that the agent's orders do not move prices; large orders in illiquid markets do.",
       "Costs can be put in the reward and volatility clustering is in the data; price impact is what a replay cannot show."),
  ("Diferența față de simulator", "Un agent de tranzacționare este antrenat pe o reluare istorică a prețurilor. Ce efect ignoră reluarea?",
       ["Costurile de tranzacționare", "Grupările de volatilitate", "Sfîrșiturile de săptămînă", "Impactul propriu al agentului asupra prețului"],
       "Reluarea istoriei presupune că ordinele agentului nu mișcă prețurile; ordinele mari pe piețe nelichide le mișcă.",
       "Costurile pot fi puse în recompensă, iar grupările de volatilitate sînt în date; impactul asupra prețului este ce nu poate arăta reluarea."))
q(1, ("Discount factor", "Why is $\\gamma=1$ acceptable for the battery but $\\gamma<1$ used for trading?",
       ["Because batteries do not lose money", "Because the battery episode has a fixed horizon of 24 steps, while trading windows are long and the critic needs a bounded return", "Because $\\gamma$ must be 1 for Q-learning", "Because prices are discounted by the exchange"],
       "With a finite horizon the return is finite for $\\gamma=1$; for long or open-ended episodes $\\gamma<1$ keeps returns and value estimates stable.",
       "Q-learning works with any $\\gamma\\le1$ on a finite horizon; the choice follows the length of the episode."),
  ("Factorul de actualizare", "De ce este acceptabil $\\gamma=1$ pentru baterie, dar pentru tranzacționare se folosește $\\gamma<1$?",
       ["Pentru că bateriile nu pierd bani", "Pentru că episodul bateriei are un orizont fix de 24 de pași, în timp ce ferestrele de tranzacționare sînt lungi, iar criticul are nevoie de un cîștig mărginit", "Pentru că $\\gamma$ trebuie să fie 1 la Q-learning", "Pentru că bursa actualizează prețurile"],
       "Pe un orizont finit, cîștigul este finit pentru $\\gamma=1$; pentru episoade lungi sau deschise, $\\gamma<1$ stabilizează cîștigurile și estimările de valoare.",
       "Q-learning funcționează cu orice $\\gamma\\le1$ pe un orizont finit; alegerea urmează lungimea episodului."))
q(2, ("Action masking", "How does the battery agent avoid charging a full battery?",
       ["It receives a large negative reward and learns to avoid it", "The battery explodes", "Infeasible actions are masked: their Q-values are set to minus infinity before the argmax", "It never reaches a full state"],
       "Masking enforces the physical constraint exactly, during exploration and in the target, so the agent never wastes samples on impossible actions.",
       "Penalties would teach the constraint slowly and imperfectly; the mask removes infeasible actions outright."),
  ("Mascarea acțiunilor", "Cum evită agentul bateriei încărcarea unei baterii pline?",
       ["Primește o recompensă foarte negativă și învață să evite acțiunea", "Bateria explodează", "Acțiunile imposibile sînt mascate: valorile lor Q sînt puse la minus infinit înainte de argmax", "Nu ajunge niciodată într-o stare plină"],
       "Mascarea impune exact restricția fizică, la explorare și în țintă, astfel încît agentul nu irosește eșantioane pe acțiuni imposibile.",
       "Penalizările ar învăța restricția încet și imperfect; masca elimină direct acțiunile imposibile."))
q(0, ("Group research", "What does the battery-arbitrage study of our group (Andrei, Cepoi et al., 2026) forecast to support the charge-discharge decision?",
       ["The joint probability distribution of the hourly prices of the next day", "Only the daily mean price", "The weather", "The battery's degradation"],
       "Dispatch depends on how prices of different hours move together; the paper forecasts their joint distribution.",
       "A daily mean says nothing about the spread between hours, which is what a battery earns."),
  ("Cercetarea grupului", "Ce prognozează studiul grupului nostru despre arbitrajul cu baterii (Andrei, Cepoi et al., 2026) pentru a susține decizia de încărcare și descărcare?",
       ["Distribuția comună de probabilitate a prețurilor orare din ziua următoare", "Doar prețul mediu zilnic", "Vremea", "Degradarea bateriei"],
       "Dispecerizarea depinde de felul în care se mișcă împreună prețurile diferitelor ore; lucrarea prognozează distribuția lor comună.",
       "O medie zilnică nu spune nimic despre diferența dintre ore, din care cîștigă o baterie."))
q(3, ("Annual revenue", f"The perfect-foresight revenue of the 2 MWh battery in Romania was {E('dp.ro_2019_k',1)} thousand EUR in 2019 and {E('dp.ro_2025_k',1)} in 2025. How should an investor use these numbers?",
       ["As the expected revenue of the project", "As the minimum revenue", "As irrelevant, because they are in euro", "As an upper bound: a real operator captures only part of it"],
       f"Perfect foresight is a ceiling; on the test days even yesterday's optimal plan captured {E('bat.naive_cap',1)} %.",
       "These are optimal values with hindsight, not forecasts; a business case needs a realistic capture ratio."),
  ("Venitul anual", f"Venitul cu previziune perfectă al bateriei de 2 MWh în România a fost {R('dp.ro_2019_k',1)} mii EUR în 2019 și {R('dp.ro_2025_k',1)} în 2025. Cum trebuie să folosească un investitor aceste cifre?",
       ["Ca venitul așteptat al proiectului", "Ca venitul minim", "Ca fiind irelevante, pentru că sînt în euro", "Ca limită superioară: un operator real obține doar o parte"],
       f"Previziunea perfectă este un plafon; în zilele de test, chiar și planul optim de ieri a obținut {R('bat.naive_cap',1)} %.",
       "Acestea sînt valori optime cu informația ulterioară, nu prognoze; un plan de afaceri are nevoie de o rată de captare realistă."))
q(1, ("Model risk", "Which guardrail limits the damage if an RL trading policy misbehaves in production?",
       ["A larger network", "Position limits, a kill switch and a fallback rule, with human sign-off", "More training epochs", "A higher discount factor"],
       "Guardrails sit outside the model: limits and fallbacks act even when the policy is wrong.",
       "Changes to the network or its training do not protect against failures that were not seen in training."),
  ("Riscul de model", "Ce limită de siguranță reduce pagubele dacă o politică RL de tranzacționare se comportă greșit în producție?",
       ["O rețea mai mare", "Limite de poziție, oprire de urgență și o regulă de rezervă, cu aprobare umană", "Mai multe epoci de antrenare", "Un factor de actualizare mai mare"],
       "Limitele de siguranță stau în afara modelului: limitele și regulile de rezervă acționează chiar și cînd politica greșește.",
       "Schimbările rețelei sau ale antrenării nu protejează împotriva eșecurilor nevăzute la antrenare."))
q(2, ("Model selection", "How were the epochs and iterations of the agents chosen without touching the test data?",
       ["By the best test result", "By training for a fixed time", "By the best result on a separate validation period (2024 for the battery, 2017-2019 for trading)", "By the lowest training loss"],
       "The validation period selects the model; the test period is used once, which keeps the test result honest.",
       "Selecting on the test period would make the test result optimistic; the training loss says nothing about generalisation."),
  ("Selecția modelului", "Cum au fost alese epocile și iterațiile agenților fără a folosi datele de test?",
       ["După cel mai bun rezultat de test", "Prin antrenare pe un timp fix", "După cel mai bun rezultat pe o perioadă de validare separată (2024 pentru baterie, 2017-2019 pentru tranzacționare)", "După cea mai mică pierdere de antrenare"],
       "Perioada de validare alege modelul; perioada de test este folosită o singură dată, ceea ce păstrează rezultatul de test corect.",
       "Selecția pe perioada de test ar face rezultatul optimist; pierderea de antrenare nu spune nimic despre generalizare."))

assert len(Q) >= 30, len(Q)
out = ("// ============================================================\n"
       "// Quiz bank for lecture id 'l10': Reinforcement learning: trading, hedging and battery dispatch (EN + RO)\n"
       "// GENERATED by lectures/L10/make_quiz_L10.py from L10_numbers.json; do not edit by hand.\n"
       "// correct = index (0-3) of the right option in the original order; incorrectExplanation never names a letter.\n"
       "// ============================================================\n"
       "window.NN_DATA.quizzes['l10'] = " + json.dumps({"draw": 20, "questions": Q}, ensure_ascii=False, indent=1) + ";\n")
with open(os.path.join(ROOT, "assets", "quizzes", "l10.js"), "w", encoding="utf-8") as fh:
    fh.write(out)
print("l10.js:", len(Q), "questions; correct index counts", dict(Counter(x["correct"] for x in Q)))
