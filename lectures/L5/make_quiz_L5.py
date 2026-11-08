"""
make_quiz_L5.py -- writes assets/quizzes/l5.js (quiz bank of Lecture 5, EN + RO) with every number from L5_numbers.json.
Format: NN_DATA.quizzes['l5'] = {draw: 20, questions: [{correct: 0-3, en: {...}, ro: {...}}]}; explanations never name a letter.
Run:  python3 lectures/L5/make_quiz_L5.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = json.load(open(os.path.join(HERE, "L5_numbers.json")))


def get(key):
    v = N
    for p in key.split("."):
        v = v[int(p)] if isinstance(v, list) else v[p]
    return v


def fmt(v, spec, lang):
    if isinstance(v, str):
        return v
    s = format(v, spec) if spec else str(v)
    if lang == "ro":
        s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s.replace("-", "−") if s.startswith("-") else s


def R(text, lang):
    return re.sub(r"\{([A-Za-z][\w.]*)(?:\|([^{}]*))?\}", lambda m: fmt(get(m.group(1)), m.group(2), lang), text)


Q = []


def q(correct, en, ro):
    """en/ro = (title, text, [4 options], correctExplanation, incorrectExplanation)"""
    Q.append({"correct": correct,
              "en": dict(zip(["title", "text", "options", "correctExplanation", "incorrectExplanation"], en)),
              "ro": dict(zip(["title", "text", "options", "correctExplanation", "incorrectExplanation"], ro))})


q(0, ("Parameters of an RNN", "A vanilla RNN has d = 3 inputs and n_h = 32 hidden units. How does its number of recurrent parameters change if the input window grows from 22 to 220 days?",
      ["It does not change: the same matrices are used at every step", "It grows ten times", "It grows with the square of the window", "It halves"],
      "The weights W_xh, W_hh and b_h are shared across time; the count n_h(d + n_h + 1) depends on d and n_h only.",
      "Weight sharing in time means the window length changes the computation and the memory, not the number of parameters."),
  ("Parametrii unei RNN", "O RNN simplă are d = 3 intrări și n_h = 32 de unități ascunse. Cum se schimbă numărul de parametri recurenți dacă fereastra crește de la 22 la 220 de zile?",
   ["Nu se schimbă: aceleași matrice sînt folosite la fiecare pas", "Crește de zece ori", "Crește cu pătratul ferestrei", "Se înjumătățește"],
   "Ponderile W_xh, W_hh și b_h sînt comune tuturor pașilor de timp; numărul n_h(d + n_h + 1) depinde doar de d și n_h.",
   "Partajarea ponderilor în timp înseamnă că lungimea ferestrei schimbă calculul și memoria, nu numărul de parametri."))
q(1, ("LSTM parameters", "How many parameters has an LSTM layer with d = 3 inputs and n_h = 32 units, by the formula 4n_h(n_h + d + 1)?",
      ["3,456", "4,608", "1,152", "9,216"],
      "4 · 32 · (32 + 3 + 1) = 4,608: four gates (forget, input, output, candidate), each with a weight matrix and a bias. PyTorch keeps two bias vectors and reports 4,736.",
      "Each of the four blocks has n_h(n_h + d) weights and n_h biases; 3,456 would be the GRU count, with three blocks."),
  ("Parametrii LSTM", "Cîți parametri are un strat LSTM cu d = 3 intrări și n_h = 32 de unități, după formula 4n_h(n_h + d + 1)?",
   ["3.456", "4.608", "1.152", "9.216"],
   "4 · 32 · (32 + 3 + 1) = 4.608: patru blocuri (uitare, intrare, ieșire, candidat), fiecare cu o matrice de ponderi și un termen liber. PyTorch păstrează doi vectori de termeni liberi și raportează 4.736.",
   "Fiecare dintre cele patru blocuri are n_h(n_h + d) ponderi și n_h termeni liberi; 3.456 ar fi numărul pentru GRU, cu trei blocuri."))
q(2, ("Vanishing gradients", "Why does the gradient of a vanilla RNN vanish over long horizons?",
      ["Because the learning rate is too small", "Because the loss is computed only at the last step", "Because it is a product of many Jacobians diag(1 − h²)W_hh whose norm is often below one", "Because tanh is not differentiable"],
      "‖∏ J_s‖ ≤ (γ σ_max(W_hh))^(T−t): when γσ_max < 1 the bound shrinks geometrically with the distance in time.",
      "The learning rate scales all gradients equally and tanh is smooth; the decay with distance comes from multiplying Jacobians step after step."),
  ("Gradienții care dispar", "De ce dispare gradientul unei RNN simple pe orizonturi lungi?",
   ["Pentru că rata de învățare este prea mică", "Pentru că pierderea se calculează doar la ultimul pas", "Pentru că este un produs de multe matrice jacobiene diag(1 − h²)W_hh, a căror normă este adesea sub unu", "Pentru că tanh nu este derivabilă"],
   "‖∏ J_s‖ ≤ (γ σ_max(W_hh))^(T−t): cînd γσ_max < 1, marginea scade geometric cu distanța în timp.",
   "Rata de învățare scalează toți gradienții la fel, iar tanh este netedă; scăderea cu distanța vine din înmulțirea matricelor jacobiene pas cu pas."))
q(3, ("Gradient clipping", "What does clipping the gradient norm at c = 1 do?",
      ["It sets every gradient component above 1 to zero", "It prevents vanishing gradients", "It removes the need for early stopping", "It rescales g to g·min(1, c/‖g‖), keeping its direction when ‖g‖ > c"],
      "Clipping caps the step size when gradients explode, without changing their direction.",
      "Clipping acts only on large gradients, so it cannot help vanishing ones; it does not zero components and it is unrelated to the choice of the stopping epoch."),
  ("Plafonarea gradientului", "Ce face plafonarea normei gradientului la c = 1?",
   ["Pune la zero fiecare componentă a gradientului peste 1", "Previne dispariția gradienților", "Elimină nevoia de oprire timpurie", "Rescalează g la g·min(1, c/‖g‖), păstrîndu-i direcția cînd ‖g‖ > c"],
   "Plafonarea limitează mărimea pasului cînd gradienții explodează, fără să le schimbe direcția.",
   "Plafonarea acționează doar asupra gradienților mari, deci nu ajută la cei care dispar; nu anulează componente și nu are legătură cu alegerea epocii de oprire."))
q(0, ("The cell state", "Along the LSTM cell state, what is ∂c_t/∂c_(t−1) on the direct path?",
      ["diag(f_t), the forget gate", "W_hh", "diag(1 − h_t²)", "The output gate o_t"],
      "c_t = f_t ⊙ c_(t−1) + i_t ⊙ c̃_t, so the direct derivative is diag(f_t); with f_t near 1 the gradient passes almost unchanged.",
      "The additive update makes the forget gate, not a weight matrix squashed by tanh′, the factor along the memory path."),
  ("Starea celulei", "De-a lungul stării celulei LSTM, cît este ∂c_t/∂c_(t−1) pe drumul direct?",
   ["diag(f_t), poarta de uitare", "W_hh", "diag(1 − h_t²)", "Poarta de ieșire o_t"],
   "c_t = f_t ⊙ c_(t−1) + i_t ⊙ c̃_t, deci derivata directă este diag(f_t); cu f_t aproape de 1, gradientul trece aproape neschimbat.",
   "Actualizarea aditivă face ca factorul de pe drumul memoriei să fie poarta de uitare, nu o matrice de ponderi comprimată de tanh′."))
q(1, ("Forget-gate bias", "Why is the forget-gate bias often initialised at a positive value such as 1?",
      ["To make the network forget faster at the start", "So that f_t ≈ σ(1) = 0.73 at the start and memory is kept by default", "Because negative biases are not allowed", "To reduce the number of parameters"],
      "A positive bias opens the forget gate at initialisation, so gradients flow through the cell state before the network has learned when to forget.",
      "The bias value changes neither the parameter count nor any constraint; a positive value keeps memory, not the opposite."),
  ("Termenul liber al porții de uitare", "De ce se inițializează adesea termenul liber al porții de uitare cu o valoare pozitivă, de exemplu 1?",
   ["Pentru ca rețeaua să uite mai repede la început", "Pentru ca f_t ≈ σ(1) = 0,73 la început, iar memoria să fie păstrată implicit", "Pentru că termenii liberi negativi nu sînt permiși", "Pentru a reduce numărul de parametri"],
   "Un termen liber pozitiv deschide poarta de uitare la inițializare, astfel încît gradienții circulă prin starea celulei înainte ca rețeaua să fi învățat cînd să uite.",
   "Valoarea termenului liber nu schimbă nici numărul de parametri, nici vreo restricție; o valoare pozitivă păstrează memoria, nu invers."))
q(2, ("GRU versus LSTM", "With the same n_h, how many parameters has a GRU compared with an LSTM?",
      ["The same", "Twice as many", "Three quarters: 3n_h(n_h + d + 1) instead of 4n_h(n_h + d + 1)", "Half"],
      "The GRU has two gates and a candidate, three blocks, against four in the LSTM, and no separate cell state.",
      "Count the blocks: update gate, reset gate and candidate for the GRU; forget, input, output and candidate for the LSTM."),
  ("GRU față de LSTM", "Cu același n_h, cîți parametri are un GRU față de un LSTM?",
   ["Același număr", "De două ori mai mulți", "Trei sferturi: 3n_h(n_h + d + 1) în loc de 4n_h(n_h + d + 1)", "Jumătate"],
   "GRU are două porți și un candidat, adică trei blocuri, față de patru la LSTM, și nu are o stare separată a celulei.",
   "Numărați blocurile: poarta de actualizare, poarta de resetare și candidatul la GRU; uitare, intrare, ieșire și candidat la LSTM."))
q(3, ("Memory of returns", "Over 1–50 trading days, which series of the S&P 500 shows strong, persistent autocorrelation?",
      ["Daily returns r_t, with ρ(1) = {acf.r1|.3f}", "Neither", "Both, equally", "Absolute returns |r_t|, with ρ(1) = {acf.a1|.3f} and ρ(50) = {acf.a50|.3f}"],
      "{acf.n_sig_a} of 50 lags of |r_t| are significant against {acf.n_sig_r} for r_t: the memory is in the size of the moves.",
      "Returns themselves are close to uncorrelated (band ±{acf.band|.3f}); volatility clustering is what makes a model with memory useful for risk."),
  ("Memoria randamentelor", "Pe 1–50 de zile de tranzacționare, ce serie a S&P 500 are o autocorelație puternică și persistentă?",
   ["Randamentele zilnice r_t, cu ρ(1) = {acf.r1|.3f}", "Niciuna", "Amîndouă, la fel", "Randamentele absolute |r_t|, cu ρ(1) = {acf.a1|.3f} și ρ(50) = {acf.a50|.3f}"],
   "{acf.n_sig_a} din 50 de decalaje ale lui |r_t| sînt semnificative, față de {acf.n_sig_r} pentru r_t: memoria este în mărimea mișcărilor.",
   "Randamentele propriu-zise sînt aproape necorelate (banda ±{acf.band|.3f}); gruparea volatilității face util un model cu memorie pentru risc."))
q(0, ("Garman–Klass", "What does the Garman–Klass estimator use to measure the variance of one day?",
      ["The open, high, low and close of that day", "Only the closing prices of the last 22 days", "The implied volatility of options", "The trading volume"],
      "σ̂²_GK = ½ ln²(H/L) − (2 ln 2 − 1) ln²(C/O): the intraday range is far more informative than one squared return.",
      "A range estimator works with one day of prices; it needs no options and no volume."),
  ("Garman–Klass", "Ce folosește estimatorul Garman–Klass pentru a măsura varianța unei zile?",
   ["Prețurile de deschidere, maxim, minim și închidere ale acelei zile", "Doar prețurile de închidere din ultimele 22 de zile", "Volatilitatea implicită a opțiunilor", "Volumul tranzacțiilor"],
   "σ̂²_GK = ½ ln²(H/L) − (2 ln 2 − 1) ln²(C/O): amplitudinea intrazilnică este mult mai informativă decît un singur randament la pătrat.",
   "Un estimator bazat pe amplitudine lucrează cu prețurile unei singure zile; nu are nevoie de opțiuni sau de volum."))
q(1, ("Time-ordered split", "Why did we split the S&P 500 data in time order (train to 2016, validation 2017–2019, test from 2020) instead of at random?",
      ["Because random splits are slower", "A random split would put days of 2020 into training and make every model look good on 2020: leakage", "Because recurrent networks cannot read shuffled data", "To have more test days"],
      "Forecasts must be made with information available at the time; a random split mixes the future into the training set.",
      "Speed and the number of test days are not the reason; the issue is that neighbouring days share information, so shuffling leaks the future."),
  ("Împărțirea în ordine cronologică", "De ce am împărțit datele S&P 500 în ordine cronologică (antrenare pînă în 2016, validare 2017–2019, test din 2020) și nu aleator?",
   ["Pentru că împărțirile aleatoare sînt mai lente", "O împărțire aleatoare ar pune zile din 2020 în antrenare și ar face orice model să pară bun pe 2020: scurgere de informație", "Pentru că rețelele recurente nu pot citi date amestecate", "Pentru a avea mai multe zile de test"],
   "Prognozele trebuie făcute cu informația disponibilă la momentul respectiv; o împărțire aleatoare amestecă viitorul în setul de antrenare.",
   "Viteza și numărul zilelor de test nu sînt motivul; problema este că zilele vecine au informație comună, deci amestecarea scurge viitorul."))
q(2, ("Scaling", "Where do the mean and standard deviation used to scale the inputs come from?",
      ["From the whole sample", "From the test years", "From the training years only", "From the last 22 days of each window"],
      "Scaling statistics are part of the model and must be estimated without the validation and test years.",
      "Using the whole sample or the test years would leak information about the future into the inputs."),
  ("Scalarea", "De unde provin media și abaterea standard folosite pentru scalarea intrărilor?",
   ["Din întregul eșantion", "Din anii de test", "Numai din anii de antrenare", "Din ultimele 22 de zile ale fiecărei ferestre"],
   "Statisticile de scalare fac parte din model și trebuie estimate fără anii de validare și de test.",
   "Folosirea întregului eșantion sau a anilor de test ar scurge în intrări informație despre viitor."))
q(3, ("HAR", "What is the HAR model of volatility?",
      ["A recurrent network with three gates", "A GARCH model with Student-t errors", "A model of the sign of returns", "A linear regression of tomorrow's log variance on today's value and on its weekly and monthly means"],
      "y_(t+1) = β0 + β_d y_t + β_w ȳ_(t−4:t) + β_m ȳ_(t−21:t); our estimates are {har.bd|.3f}, {har.bw|.3f} and {har.bm|.3f}.",
      "HAR is linear and has no gates or conditional-variance recursion; three averages over different horizons approximate long memory."),
  ("HAR", "Ce este modelul HAR pentru volatilitate?",
   ["O rețea recurentă cu trei porți", "Un model GARCH cu erori Student-t", "Un model pentru semnul randamentelor", "O regresie liniară a log-varianței de mîine pe valoarea de azi și pe mediile ei săptămînală și lunară"],
   "y_(t+1) = β0 + β_d y_t + β_w ȳ_(t−4:t) + β_m ȳ_(t−21:t); estimările noastre sînt {har.bd|.3f}, {har.bw|.3f} și {har.bm|.3f}.",
   "HAR este liniar și nu are porți sau o recurență a varianței condiționate; trei medii pe orizonturi diferite aproximează memoria lungă."))
q(0, ("GARCH persistence", "Our GARCH(1,1) estimates are α = {garch.alpha|.3f} and β = {garch.beta|.3f}. What does α + β = {garch.persistence|.3f} say?",
      ["Volatility shocks decay slowly: the conditional variance is highly persistent", "Returns are strongly autocorrelated", "The model is not estimable", "Volatility is constant"],
      "With α + β close to 1, a shock to the variance halves only after many days, the volatility clustering seen in the data.",
      "Persistence concerns the variance, not the returns; a value just below 1 is typical and fully estimable."),
  ("Persistența GARCH", "Estimările noastre GARCH(1,1) sînt α = {garch.alpha|.3f} și β = {garch.beta|.3f}. Ce arată α + β = {garch.persistence|.3f}?",
   ["Șocurile de volatilitate se sting încet: varianța condiționată este foarte persistentă", "Randamentele sînt puternic autocorelate", "Modelul nu poate fi estimat", "Volatilitatea este constantă"],
   "Cu α + β aproape de 1, un șoc al varianței se înjumătățește abia după multe zile, adică gruparea volatilității văzută în date.",
   "Persistența privește varianța, nu randamentele; o valoare puțin sub 1 este tipică și perfect estimabilă."))
q(1, ("Diebold–Mariano", "What does the Diebold–Mariano test compare?",
      ["The parameters of two models", "The mean of the daily loss differences of two forecasts, with a Newey–West variance", "The in-sample fit of two models", "The number of seeds of two networks"],
      "DM = d̄ / sqrt(σ̂²_NW / n), where d_t = L(e_1,t) − L(e_2,t); the Newey–West variance accounts for the autocorrelation of the loss differences.",
      "The test works on out-of-sample forecast losses, day by day; parameters and in-sample fit are not compared."),
  ("Diebold–Mariano", "Ce compară testul Diebold–Mariano?",
   ["Parametrii a două modele", "Media diferențelor zilnice de pierdere a două prognoze, cu o varianță Newey–West", "Potrivirea în eșantion a două modele", "Numărul de semințe a două rețele"],
   "DM = d̄ / sqrt(σ̂²_NW / n), unde d_t = L(e_1,t) − L(e_2,t); varianța Newey–West ține seama de autocorelația diferențelor de pierdere.",
   "Testul lucrează pe pierderile prognozelor în afara eșantionului, zi de zi; parametrii și potrivirea în eșantion nu sînt comparate."))
q(2, ("Volatility results", "On the S&P 500 test period, which statement matches our results?",
      ["The networks lost to the naive forecast", "The LSTM halved the error of HAR", "Naive {vol.naive.mse|.3f}, HAR {vol.har.mse|.3f}, LSTM {vol.lstm.mse_mean|.3f} (seed mean); DM LSTM vs HAR = {vol.dm_lstm_har.stat|.2f}", "GARCH could not be estimated"],
      "All models improve on the naive forecast; the LSTM also beats HAR, and the DM statistic of {vol.dm_lstm_har.stat|.2f} says the gap is significant.",
      "Read the numbers on the results slide: the naive forecast is the worst, and the LSTM lowers the MSE of HAR by about a tenth, not by half."),
  ("Rezultatele pentru volatilitate", "Pe perioada de test S&P 500, ce afirmație corespunde rezultatelor noastre?",
   ["Rețelele au pierdut în fața prognozei naive", "LSTM a înjumătățit eroarea HAR", "Naiv {vol.naive.mse|.3f}, HAR {vol.har.mse|.3f}, LSTM {vol.lstm.mse_mean|.3f} (media pe semințe); DM LSTM față de HAR = {vol.dm_lstm_har.stat|.2f}", "GARCH nu a putut fi estimat"],
   "Toate modelele îmbunătățesc prognoza naivă; LSTM depășește și HAR, iar statistica DM de {vol.dm_lstm_har.stat|.2f} arată că diferența este semnificativă.",
   "Citiți cifrele de pe slide-ul cu rezultate: prognoza naivă este cea mai slabă, iar LSTM scade MSE a HAR cu aproximativ o zecime, nu la jumătate."))
q(3, ("Choosing a VaR input", "The LSTM changes the volatility MSE of HAR by {vol.lstm_vs_har_pct|+.1f}%, with a DM statistic of {vol.dm_lstm_har.stat|.2f}. What is the sensible decision for a VaR system?",
      ["Always use the model with the lower point estimate", "Use the LSTM because it is more modern", "Average all six models", "Treat the LSTM as a serious candidate: confirm the gain on another market and sub-period, then switch with monitoring and HAR as the fallback"],
      "The gain is significant on this test period; a risk system also needs it to be stable across markets and periods, and a simple fallback.",
      "A point estimate from one test period carries sampling noise; modernity is not a criterion, and averaging is a separate modelling decision to be tested."),
  ("Alegerea intrării pentru VaR", "LSTM schimbă MSE a volatilității față de HAR cu {vol.lstm_vs_har_pct|+.1f}%, cu o statistică DM de {vol.dm_lstm_har.stat|.2f}. Care este decizia rezonabilă pentru un sistem VaR?",
   ["Folosiți întotdeauna modelul cu estimarea punctuală mai mică", "Folosiți LSTM pentru că este mai modern", "Mediați toate cele șase modele", "Tratați LSTM ca pe un candidat serios: confirmați cîștigul pe altă piață și pe subperioade, apoi schimbați modelul, cu monitorizare și cu HAR ca soluție de rezervă"],
   "Cîștigul este semnificativ pe această perioadă de test; un sistem de risc cere în plus ca el să fie stabil pe piețe și perioade diferite și o soluție simplă de rezervă.",
   "O estimare punctuală dintr-o singură perioadă de test conține zgomot de eșantionare; modernitatea nu este un criteriu, iar medierea este o altă decizie de modelare, care trebuie testată."))
q(0, ("Returns", "What out-of-sample R² did the LSTM reach on next-day S&P 500 returns (mean over seeds)?",
      ["{ret.SP500.r2_lstm_mean|.3f}%", "35%", "12%", "60%"],
      "Against the zero forecast the LSTM gives {ret.SP500.r2_lstm_mean|.3f}% (seeds from {ret.SP500.r2_lstm_min|.3f}% to {ret.SP500.r2_lstm_max|.3f}%): almost nothing to forecast in tomorrow's return.",
      "Double-digit out-of-sample R² values for daily returns are not credible in a liquid market; the honest test gives a value close to zero."),
  ("Randamentele", "Ce R² în afara eșantionului a obținut LSTM pentru randamentele S&P 500 de a doua zi (media pe semințe)?",
   ["{ret.SP500.r2_lstm_mean|.3f}%", "35%", "12%", "60%"],
   "Față de prognoza zero, LSTM dă {ret.SP500.r2_lstm_mean|.3f}% (semințele între {ret.SP500.r2_lstm_min|.3f}% și {ret.SP500.r2_lstm_max|.3f}%): aproape nimic de prognozat în randamentul de mîine.",
   "Valorile de două cifre ale R² în afara eșantionului pentru randamente zilnice nu sînt credibile pe o piață lichidă; testul corect dă o valoare apropiată de zero."))
q(1, ("Sign accuracy", "The LSTM got the sign of S&P 500 returns right on {ret.SP500.hit_lstm|.1f}% of test days. What benchmark must this beat?",
      ["50%", "The always-up forecast, right on {ret.SP500.up_share|.1f}% of the days", "100%", "The training accuracy"],
      "In a rising market, ‘always up’ is right more than half the time; a direction model must beat that, after costs.",
      "50% is the coin, not the relevant benchmark when up days dominate; training accuracy says nothing about new days."),
  ("Acuratețea semnului", "LSTM a ghicit semnul randamentelor S&P 500 în {ret.SP500.hit_lstm|.1f}% din zilele de test. Ce reper trebuie să depășească?",
   ["50%", "Prognoza „mereu în sus”, corectă în {ret.SP500.up_share|.1f}% din zile", "100%", "Acuratețea pe datele de antrenare"],
   "Pe o piață în creștere, „mereu în sus” are dreptate în mai mult de jumătate din zile; un model de direcție trebuie să depășească acest reper, după costuri.",
   "50% este aruncarea monedei, nu reperul relevant cînd zilele de creștere domină; acuratețea la antrenare nu spune nimic despre zilele noi."))
q(2, ("Out-of-sample R²", "How is the out-of-sample R² of returns defined in the lecture?",
      ["The squared correlation on the training data", "1 − MSE/variance on the training data", "1 − Σ(r − r̂)²/Σr² on the test days, against the zero forecast", "The share of days with the right sign"],
      "A negative value means the model does worse than simply forecasting zero.",
      "The definition is on the test days and against the zero forecast; training-data measures and sign accuracy are different quantities."),
  ("R² în afara eșantionului", "Cum este definit în curs R² în afara eșantionului pentru randamente?",
   ["Pătratul corelației pe datele de antrenare", "1 − MSE/varianță pe datele de antrenare", "1 − Σ(r − r̂)²/Σr² pe zilele de test, față de prognoza zero", "Ponderea zilelor cu semnul corect"],
   "O valoare negativă înseamnă că modelul este mai slab decît simpla prognoză zero.",
   "Definiția este pe zilele de test și față de prognoza zero; măsurile pe datele de antrenare și acuratețea semnului sînt alte mărimi."))
q(3, ("Load task", "In the Romanian load case, what exactly is forecast?",
      ["The daily mean load of next month", "Tomorrow's price", "The next hour only", "The 24 hourly loads of day d+1, from the 168 hours up to the end of day d and the calendar of day d+1"],
      "A many-to-many forecast with a vector output of 24 values, as a supplier needs for the day-ahead schedule.",
      "The model forecasts load, not prices, for a full day of hours, not for a single hour or a monthly mean."),
  ("Sarcina de consum", "În studiul de caz despre consumul din România, ce se prognozează exact?",
   ["Consumul mediu zilnic de luna viitoare", "Prețul de mîine", "Doar ora următoare", "Cele 24 de valori orare ale zilei d+1, din cele 168 de ore pînă la sfîrșitul zilei d și din calendarul zilei d+1"],
   "O prognoză many-to-many cu o ieșire vectorială de 24 de valori, așa cum are nevoie un furnizor pentru programul pe ziua următoare.",
   "Modelul prognozează consumul, nu prețul, pentru toate orele unei zile, nu pentru o singură oră sau pentru o medie lunară."))
q(0, ("Load benchmarks", "What MAPE did the seasonal naive forecast (same weekday a week earlier) reach on the load test days?",
      ["{load.snaive|.2f}%", "0.50%", "25%", "{load.window.24|.2f}% with a 24-hour window"],
      "The seasonal naive forecast reached {load.snaive|.2f}%; ridge ARX {load.ridge|.2f}% and the LSTM {load.lstm.ens|.2f}%.",
      "The last option is a result of the LSTM with a 24-hour window; the other values do not appear in the results."),
  ("Reperele pentru consum", "Ce MAPE a obținut prognoza naivă sezonieră (aceeași zi a săptămînii de acum o săptămînă) în zilele de test?",
   ["{load.snaive|.2f}%", "0,50%", "25%", "{load.window.24|.2f}% cu o fereastră de 24 de ore"],
   "Prognoza naivă sezonieră a obținut {load.snaive|.2f}%; ridge ARX {load.ridge|.2f}%, iar LSTM {load.lstm.ens|.2f}%.",
   "Ultima variantă este un rezultat al LSTM cu o fereastră de 24 de ore; celelalte valori nu apar în rezultate."))
q(1, ("Ridge ARX", "What is the ridge ARX benchmark of the load case?",
      ["A recurrent network without gates", "A linear model of the 24 hours of d+1 on the last 168 hours and the calendar, with a ridge penalty chosen on the validation year", "The seasonal naive forecast", "A GARCH model of load"],
      "It uses exactly the information set of the networks, so the comparison isolates the value of nonlinearity and memory.",
      "Ridge ARX is linear and non-recurrent; the penalty ({load.ridge_alpha}) was chosen on 2023, not on the test years."),
  ("Ridge ARX", "Ce este modelul de referință ridge ARX din studiul de caz despre consum?",
   ["O rețea recurentă fără porți", "Un model liniar al celor 24 de ore din d+1 pe ultimele 168 de ore și pe calendar, cu o penalizare ridge aleasă pe anul de validare", "Prognoza naivă sezonieră", "Un model GARCH pentru consum"],
   "Folosește exact setul de informații al rețelelor, deci comparația izolează valoarea neliniarității și a memoriei.",
   "Ridge ARX este liniar și nerecurent; penalizarea ({load.ridge_alpha}) a fost aleasă pe 2023, nu pe anii de test."))
q(2, ("Measured memory", "What did the gradient sensitivity at a lag of 168 hours show for the trained load models?",
      ["All three models ignore the last week", "Only the RNN uses the last week", "RNN {grad.rnn.lag168_rel|.3f}, LSTM {grad.lstm.lag168_rel|.3f}, GRU {grad.gru.lag168_rel|.3f}, relative to the maximum of the last day", "The sensitivity cannot be computed for an LSTM"],
      "S(k) = mean |∂ŷ_12/∂x_(T+1−k)| measures how much the forecast reacts to an input k hours back; autograd computes it for any architecture.",
      "The values on the slide compare the three trained networks directly; gradients with respect to inputs exist for every differentiable model."),
  ("Memoria măsurată", "Ce a arătat sensibilitatea gradientului la decalajul de 168 de ore pentru modelele de consum antrenate?",
   ["Toate trei modelele ignoră ultima săptămînă", "Doar RNN folosește ultima săptămînă", "RNN {grad.rnn.lag168_rel|.3f}, LSTM {grad.lstm.lag168_rel|.3f}, GRU {grad.gru.lag168_rel|.3f}, raportat la maximul ultimei zile", "Sensibilitatea nu poate fi calculată pentru un LSTM"],
   "S(k) = media |∂ŷ_12/∂x_(T+1−k)| măsoară cît reacționează prognoza la o intrare de acum k ore; diferențierea automată o calculează pentru orice arhitectură.",
   "Valorile de pe slide compară direct cele trei rețele antrenate; gradienții în raport cu intrările există pentru orice model derivabil."))
q(3, ("Window length", "How did the LSTM test MAPE change with the input window?",
      ["It did not depend on the window", "Longer windows always made it worse", "Only 24 hours could be trained", "24 h {load.window.24|.2f}%, 72 h {load.window.72|.2f}%, 168 h {load.window.168|.2f}%, 336 h {load.window.336|.2f}%"],
      "The window decides which daily and weekly patterns the network can see; compare the values on the slide to choose it on data.",
      "All four windows were trained with the same protocol; the window is a modelling choice, checked on validation and test data."),
  ("Lungimea ferestrei", "Cum s-a schimbat MAPE pe test a LSTM cu fereastra de intrare?",
   ["Nu a depins de fereastră", "Ferestrele mai lungi au înrăutățit-o mereu", "Doar fereastra de 24 de ore a putut fi antrenată", "24 h {load.window.24|.2f}%, 72 h {load.window.72|.2f}%, 168 h {load.window.168|.2f}%, 336 h {load.window.336|.2f}%"],
   "Fereastra decide ce tipare zilnice și săptămînale poate vedea rețeaua; comparați valorile de pe slide pentru a o alege pe date.",
   "Toate cele patru ferestre au fost antrenate cu același protocol; fereastra este o alegere de modelare, verificată pe datele de validare și de test."))
q(0, ("Bidirectional networks", "When does a bidirectional LSTM leak information in forecasting?",
      ["When its input window contains observations after the forecast origin", "Always, by construction", "Never", "Only with GRU cells"],
      "Reading a window backwards is harmless if the window ends at the origin; the leak comes from inputs that are not yet known.",
      "Direction of reading is not the problem; the content of the window is. The cell type is irrelevant."),
  ("Rețelele bidirecționale", "Cînd produce un LSTM bidirecțional o scurgere de informație în prognoză?",
   ["Cînd fereastra lui de intrare conține observații de după momentul prognozei", "Întotdeauna, prin construcție", "Niciodată", "Doar cu celule GRU"],
   "Citirea înapoi a unei ferestre este inofensivă dacă fereastra se termină în momentul prognozei; scurgerea vine din intrări care nu sînt încă cunoscute.",
   "Problema nu este direcția citirii, ci conținutul ferestrei; tipul celulei nu contează."))
q(1, ("Early stopping", "How was early stopping applied to the recurrent networks?",
      ["On the test years", "On the validation years: stop after 6 (volatility) or 5 (load) epochs without improvement and keep the best weights", "After a fixed 100 epochs", "When the training loss reached zero"],
      "The validation years act as unseen data during training; the test years are used once, at the end.",
      "Stopping on the test years would make the test optimistic; a fixed number of epochs or a zero training loss would ignore overfitting."),
  ("Oprirea timpurie", "Cum a fost aplicată oprirea timpurie rețelelor recurente?",
   ["Pe anii de test", "Pe anii de validare: oprire după 6 (volatilitate) sau 5 (consum) epoci fără îmbunătățire, păstrînd cele mai bune ponderi", "După 100 de epoci fixe", "Cînd pierderea la antrenare a ajuns la zero"],
   "Anii de validare joacă rolul datelor nevăzute în timpul antrenării; anii de test sînt folosiți o singură dată, la final.",
   "Oprirea pe anii de test ar face testul optimist; un număr fix de epoci sau o pierdere zero la antrenare ar ignora supraînvățarea."))
q(2, ("Seeds", "Why were the volatility networks trained with five seeds and their forecasts averaged?",
      ["To increase the size of the test set", "Because PyTorch requires it", "Seeds change initialisation and batch order; the spread measures training noise and the mean forecast is more stable", "To make the DM test unnecessary"],
      "A claim that one network beats another must survive the training noise; the DM test then handles the sampling noise of the test period.",
      "The test set is fixed; seeds and the DM test address two different sources of uncertainty, and neither replaces the other."),
  ("Semințele", "De ce au fost antrenate rețelele de volatilitate cu cinci semințe, iar prognozele lor mediate?",
   ["Pentru a mări setul de test", "Pentru că PyTorch o cere", "Semințele schimbă inițializarea și ordinea loturilor; dispersia măsoară zgomotul antrenării, iar prognoza medie este mai stabilă", "Pentru a face inutil testul DM"],
   "O afirmație că o rețea o depășește pe alta trebuie să reziste zgomotului antrenării; testul DM tratează apoi zgomotul de eșantionare al perioadei de test.",
   "Setul de test este fix; semințele și testul DM tratează două surse diferite de incertitudine, iar niciuna nu o înlocuiește pe cealaltă."))
q(3, ("Gates as diagnostics", "What can the gate activations of the trained volatility LSTM tell a risk manager?",
      ["The causal effect of each input", "The exact VaR", "Whether the model is overfitted", "When the model changes regime, as a diagnostic to be checked against the data"],
      "Correlations of the gates with log variance ({gates.f_corr_y|.3f} for the forget gate) describe the model's behaviour, not causes.",
      "Gates are internal states; they neither identify causal effects nor measure overfitting or tail risk directly."),
  ("Porțile ca diagnostic", "Ce îi pot spune activările porților LSTM-ului de volatilitate unui manager de risc?",
   ["Efectul cauzal al fiecărei intrări", "VaR exactă", "Dacă modelul este supraînvățat", "Cînd își schimbă modelul regimul, ca diagnostic de verificat pe date"],
   "Corelațiile porților cu log-varianța ({gates.f_corr_y|.3f} pentru poarta de uitare) descriu comportamentul modelului, nu cauze.",
   "Porțile sînt stări interne; nu identifică efecte cauzale și nu măsoară direct supraînvățarea sau riscul de coadă."))
q(0, ("Truncated BPTT", "What is the trade-off of truncated backpropagation through time with K steps?",
      ["Cheaper and less memory, but no gradient signal from more than K steps back", "More accurate gradients at higher cost", "It removes vanishing gradients", "It changes the forward pass"],
      "Truncation cuts the sum over time at K: dependencies longer than K cannot be learned from the gradient.",
      "The forward pass is unchanged; truncation trades long-range learning for cost and does not fix vanishing gradients."),
  ("BPTT trunchiată", "Care este compromisul propagării înapoi în timp trunchiate la K pași?",
   ["Mai ieftină și cu mai puțină memorie, dar fără semnal de gradient de la mai mult de K pași în urmă", "Gradienți mai exacți, la un cost mai mare", "Elimină dispariția gradienților", "Schimbă trecerea înainte"],
   "Trunchierea taie suma în timp la K: dependențele mai lungi de K nu pot fi învățate din gradient.",
   "Trecerea înainte rămîne aceeași; trunchierea schimbă învățarea pe distanțe lungi pe cost și nu rezolvă dispariția gradienților."))
q(1, ("Group research: foundation models", "What did Jheng, Tak, Găman, Mazurencu-Marinescu-Pele and Pele (2026) study?",
      ["LSTM networks for electricity load in Romania", "Pre-trained time-series foundation models, without fine-tuning, for VaR and expected shortfall in Central and Eastern European markets, with conformal calibration", "Image classification of satellite tiles", "Sentiment of central-bank speeches"],
      "The paper uses zero-shot foundation models for tail risk and a conformal calibration step that makes the forecasts reliable.",
      "The study is about risk forecasting with pre-trained time-series models, the successors of recurrent networks discussed in Lectures 7 and 13."),
  ("Cercetarea grupului: modele fundaționale", "Ce au studiat Jheng, Tak, Găman, Mazurencu-Marinescu-Pele și Pele (2026)?",
   ["Rețele LSTM pentru consumul de energie electrică din România", "Modele fundaționale preantrenate pentru serii de timp, fără reantrenare, pentru VaR și expected shortfall pe piețele din Europa Centrală și de Est, cu calibrare conformală", "Clasificarea imaginilor satelitare", "Sentimentul discursurilor băncilor centrale"],
   "Lucrarea folosește modele fundaționale fără antrenare suplimentară pentru riscul de coadă și o etapă de calibrare conformală care face prognozele fiabile.",
   "Studiul privește prognoza riscului cu modele preantrenate pentru serii de timp, succesorii rețelelor recurente discutați în cursurile 7 și 13."))
q(2, ("Group research: BitMood", "What did BitMood (Conda et al., 2026) find when forecasting Bitcoin volume?",
      ["The LSTM was far better than any linear model", "Sentiment had no relation with volume", "Linear regression and a bidirectional LSTM reached comparable test performance: complexity does not guarantee better forecasts", "Only Twitter data were useful"],
      "The paper reports comparable test performance of linear regression and a bidirectional LSTM for volume prediction, from Facebook sentiment.",
      "The study used Facebook posts and found predictive power of sentiment for volume; the comparison of models is the same lesson as in our volatility case."),
  ("Cercetarea grupului: BitMood", "Ce a constatat BitMood (Conda et al., 2026) în prognoza volumului Bitcoin?",
   ["LSTM a fost mult mai bun decît orice model liniar", "Sentimentul nu a avut nicio legătură cu volumul", "Regresia liniară și un LSTM bidirecțional au avut performanțe comparabile pe test: complexitatea nu garantează prognoze mai bune", "Doar datele de pe Twitter au fost utile"],
   "Lucrarea raportează performanțe comparabile pe test ale regresiei liniare și ale unui LSTM bidirecțional în prognoza volumului, pe baza sentimentului de pe Facebook.",
   "Studiul a folosit postări de pe Facebook și a găsit putere predictivă a sentimentului pentru volum; comparația modelelor este aceeași lecție ca în studiul nostru de caz despre volatilitate."))
q(3, ("Many-to-one", "Which pattern does our volatility model follow?",
      ["Encoder–decoder", "Many-to-many with one output per day of the window", "One-to-many", "Many-to-one: a window of {vol.W} days in, one value (tomorrow's log variance) out"],
      "Only the last hidden state feeds the output layer; the loss is computed on one value per window.",
      "The model reads a sequence and emits a single forecast; it does not generate a sequence step by step."),
  ("Many-to-one", "Ce tipar urmează modelul nostru de volatilitate?",
   ["Codificator–decodificator", "Many-to-many, cu o ieșire pentru fiecare zi a ferestrei", "One-to-many", "Many-to-one: o fereastră de {vol.W} de zile la intrare, o valoare (log-varianța de mîine) la ieșire"],
   "Doar ultima stare ascunsă alimentează stratul de ieșire; pierderea se calculează pe o singură valoare pentru fiecare fereastră.",
   "Modelul citește o secvență și emite o singură prognoză; nu generează o secvență pas cu pas."))


if __name__ == "__main__":
    out = []
    for x in Q:
        y_ = {"correct": x["correct"]}
        for lang in ("en", "ro"):
            d = dict(x[lang])
            y_[lang] = {k: ([R(o, lang) for o in v] if isinstance(v, list) else R(v, lang)) for k, v in d.items()}
        out.append(y_)
    assert len(out) >= 30 and all(len(o["en"]["options"]) == 4 and len(o["ro"]["options"]) == 4 for o in out)
    js = ("// ============================================================\n"
          "// Quiz bank for lecture id 'l5': Recurrent networks: RNN, LSTM and GRU (EN + RO)\n"
          "// Generated by lectures/L5/make_quiz_L5.py from L5_numbers.json; correct = index (0-3) in the original order.\n"
          "// incorrectExplanation never names a letter: the engine prepends the correct answer after shuffling.\n"
          "// ============================================================\n"
          "window.NN_DATA.quizzes['l5'] = " + json.dumps({"draw": 20, "questions": out}, ensure_ascii=False, indent=4) + ";\n")
    p = os.path.join(ROOT, "assets", "quizzes", "l5.js")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(js)
    print(f"wrote {p}: {len(out)} questions")
