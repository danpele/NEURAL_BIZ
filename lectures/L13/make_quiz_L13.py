"""
make_quiz_L13.py -- quiz bank of Lecture 13 (EN + RO) -> assets/quizzes/l13.js.
Every number in a question comes from L13_numbers.json; correct = index (0-3) of the right option in the original order.
Run:  python3 lectures/L13/make_quiz_L13.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "L13_numbers.json")))
OUT = os.path.join(HERE, "..", "..", "assets", "quizzes", "l13.js")


def en(x, d=2):
    return f"{x:,.{d}f}"


def ro(x, d=2):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


A = N["acc"]
Q = []


def q(correct, en_, ro_):
    Q.append({"correct": correct, "en": dict(zip(("title", "text", "options", "correctExplanation", "incorrectExplanation"), en_)),
              "ro": dict(zip(("title", "text", "options", "correctExplanation", "incorrectExplanation"), ro_))})


q(1, ("Gate closure", "A day-ahead price forecast for day d is made at 11:00 CET on day d−1. Which input may it use?",
      ["The actual Romanian load of day d", "The Romanian prices of day d−1, cleared by the auction held on day d−2",
       "The German prices of day d", "The realised imbalance prices of day d"],
      "Prices of day d−1 were set by the auction of day d−2, so they are known at 11:00 on d−1; everything about day d itself is still in the future.",
      "Load, German prices and imbalance prices of day d are all realised after the auction closes: using them leaks the future into the forecast."),
  ("Închiderea licitației", "O prognoză de preț pentru ziua d se face la ora 11:00 CET în ziua d−1. Ce variabilă de intrare poate folosi?",
   ["Consumul efectiv al României din ziua d", "Prețurile din România pentru ziua d−1, stabilite de licitația din ziua d−2",
    "Prețurile din Germania pentru ziua d", "Prețurile de dezechilibru realizate în ziua d"],
   "Prețurile zilei d−1 au fost stabilite de licitația din ziua d−2, deci se cunosc la ora 11:00 din d−1; tot ce ține de ziua d este încă în viitor.",
   "Consumul, prețurile din Germania și prețurile de dezechilibru din ziua d se realizează după închiderea licitației: folosirea lor introduce viitorul în prognoză."))
q(0, ("Percentage errors and negative prices", f"In the test year the Romanian price was negative in {N['data']['test_neg_hours']} hours. Why does the lecture report MAE in EUR/MWh and rMAE instead of MAPE?",
      ["Percentage errors explode or change sign when the actual price is near zero or negative", "MAPE cannot be computed for 24 hours at once",
       "MAE is always smaller than MAPE", "MAPE requires a normal distribution of errors"],
      "MAPE divides by the actual price; prices close to zero or negative make the ratio meaningless, while MAE stays in EUR/MWh.",
      "The problem is the division by the actual value: near zero or below zero the percentage error is unbounded or has the wrong sign."),
  ("Erorile procentuale și prețurile negative", f"În anul de test, prețul din România a fost negativ în {N['data']['test_neg_hours']} de ore. De ce raportează cursul MAE în EUR/MWh și rMAE, nu MAPE?",
   ["Erorile procentuale explodează sau își schimbă semnul cînd prețul realizat este aproape de zero sau negativ", "MAPE nu se poate calcula pentru 24 de ore deodată",
    "MAE este întotdeauna mai mică decît MAPE", "MAPE presupune o distribuție normală a erorilor"],
   "MAPE împarte la prețul realizat; prețurile apropiate de zero sau negative fac raportul lipsit de sens, în timp ce MAE rămîne în EUR/MWh.",
   "Problema este împărțirea la valoarea realizată: în jurul lui zero sau sub zero eroarea procentuală este nemărginită sau are semnul greșit."))
q(2, ("rMAE", f"LEAR has MAE {en(A['lear']['mae'])} EUR/MWh and the naive forecast {en(A['naive']['mae'])} EUR/MWh. What is the rMAE of LEAR?",
      [en(A['naive']['mae'] / A['lear']['mae'], 3), en(A['naive']['mae'] - A['lear']['mae']), en(A['lear']['rmae'], 3), "1.000"],
      f"rMAE = MAE of the model / MAE of the naive forecast = {en(A['lear']['mae'])} / {en(A['naive']['mae'])} = {en(A['lear']['rmae'], 3)}; below 1 means better than naive.",
      "The relative MAE divides the model's error by the naive error, not the other way round, and it is a ratio, not a difference."),
  ("rMAE", f"LEAR are MAE {ro(A['lear']['mae'])} EUR/MWh, iar prognoza naivă {ro(A['naive']['mae'])} EUR/MWh. Cît este rMAE pentru LEAR?",
   [ro(A['naive']['mae'] / A['lear']['mae'], 3), ro(A['naive']['mae'] - A['lear']['mae']), ro(A['lear']['rmae'], 3), "1,000"],
   f"rMAE = MAE a modelului / MAE a prognozei naive = {ro(A['lear']['mae'])} / {ro(A['naive']['mae'])} = {ro(A['lear']['rmae'], 3)}; sub 1 înseamnă mai bun decît modelul naiv.",
   "MAE relativă împarte eroarea modelului la eroarea naivă, nu invers, și este un raport, nu o diferență."))
q(3, ("asinh transform", "Why do LEAR and the DNN transform prices with asinh((y − m)/σ) rather than with log(y)?",
      ["asinh is faster to compute", "log removes the daily seasonality", "asinh makes the errors normally distributed",
       "asinh is defined for zero and negative prices and behaves like a log for large prices"],
      "The area hyperbolic sine is defined on the whole real line, is close to the identity near zero and grows like a logarithm for large values, which tames spikes.",
      "The logarithm is not defined for zero or negative prices, which occur on the Romanian market; asinh keeps the log-like damping of spikes without that problem."),
  ("Transformarea asinh", "De ce transformă LEAR și DNN prețurile cu asinh((y − m)/σ), și nu cu log(y)?",
   ["asinh se calculează mai repede", "logaritmul elimină sezonalitatea zilnică", "asinh face erorile normal distribuite",
    "asinh este definit pentru prețuri nule și negative și se comportă ca un logaritm pentru prețuri mari"],
   "Sinusul hiperbolic invers este definit pe toată axa reală, este aproape de identitate în jurul lui zero și crește ca un logaritm pentru valori mari, ceea ce îmblînzește vîrfurile.",
   "Logaritmul nu este definit pentru prețuri nule sau negative, care apar pe piața din România; asinh păstrează atenuarea de tip logaritmic a vîrfurilor fără această problemă."))
q(0, ("LEAR penalty", "In LEAR, what does the L1 penalty λ‖β‖₁ do with the roughly 150 lagged inputs of each hour?",
      ["It sets many coefficients exactly to zero, keeping only the lags that matter for that hour", "It forces all coefficients to be equal",
       "It removes the intercept", "It makes the model non-linear"],
      "The Lasso shrinks coefficients and sets many of them exactly to zero; λ is chosen per hour by AIC, so each hour keeps its own small set of lags.",
      "An L1 penalty selects variables: it does not make coefficients equal, does not touch the intercept and keeps the model linear."),
  ("Penalizarea LEAR", "În LEAR, ce face penalizarea L1 λ‖β‖₁ cu cele circa 150 de decalaje ale fiecărei ore?",
   ["Anulează exact mulți coeficienți și păstrează doar decalajele care contează pentru ora respectivă", "Forțează toți coeficienții să fie egali",
    "Elimină termenul liber", "Face modelul neliniar"],
   "Lasso micșorează coeficienții și îi anulează exact pe mulți dintre ei; λ se alege pentru fiecare oră prin AIC, astfel încît fiecare oră își păstrează propriul set mic de decalaje.",
   "O penalizare L1 selectează variabile: nu egalizează coeficienții, nu atinge termenul liber și păstrează modelul liniar."))
q(1, ("24 outputs", "Why does the DNN have 24 outputs instead of 24 separate networks?",
      ["Because a network cannot have a single output", "Hours share hidden features, so what is learned for one hour helps the others",
       "Because the loss function requires 24 outputs", "To make the network linear"],
      "A single network with 24 outputs learns common hidden features (level, daily shape) and specialises only in the last layer.",
      "The design choice is about sharing information across hours; a network can have one output, and the L1 loss works for any number of outputs."),
  ("24 de ieșiri", "De ce are rețeaua DNN 24 de ieșiri, în loc de 24 de rețele separate?",
   ["Pentru că o rețea nu poate avea o singură ieșire", "Orele împart trăsăturile ascunse, astfel încît ce se învață pentru o oră le ajută pe celelalte",
    "Pentru că funcția de pierdere cere 24 de ieșiri", "Pentru ca rețeaua să fie liniară"],
   "O singură rețea cu 24 de ieșiri învață trăsături ascunse comune (nivelul, forma zilnică) și se specializează doar în ultimul strat.",
   "Alegerea ține de împărțirea informației între ore; o rețea poate avea o singură ieșire, iar pierderea L1 funcționează pentru orice număr de ieșiri."))
q(2, ("Seeds", f"The four DNN seeds gave test MAEs between {en(N['acc_dnn_seed_min'])} and {en(N['acc_dnn_seed_max'])} EUR/MWh. What should you report?",
      ["The best seed only", "The worst seed only", "The mean of the seeds' forecasts (or of their errors) together with the range", "Nothing, because the seeds disagree"],
      "Each seed is one draw of a random estimator; averaging the forecasts and reporting the spread is honest and usually more accurate.",
      "Picking the best seed after seeing the test set is selection on the test data; the spread is information, not a reason to report nothing."),
  ("Rulări repetate", f"Cele patru rulări ale DNN au dat MAE pe test între {ro(N['acc_dnn_seed_min'])} și {ro(N['acc_dnn_seed_max'])} EUR/MWh. Ce raportați?",
   ["Doar cea mai bună rulare", "Doar cea mai slabă rulare", "Media prognozelor (sau a erorilor) rulărilor, împreună cu intervalul lor", "Nimic, pentru că rulările nu concordă"],
   "Fiecare rulare este o extragere dintr-un estimator aleator; media prognozelor și raportarea dispersiei sînt corecte și de obicei mai precise.",
   "Alegerea celei mai bune rulări după ce ați văzut testul înseamnă selecție pe datele de test; dispersia este o informație, nu un motiv de a nu raporta."))
q(3, ("Early stopping", f"The DNN validation loss was lowest at epoch {N['train']['dnn_best_epoch']} of {N['train']['dnn_epochs']}. Which weights are used for forecasting?",
      ["Those of the last epoch", "Those of the first epoch", "An average of all epochs", f"Those of epoch {N['train']['dnn_best_epoch']}, restored after training stops"],
      "Early stopping keeps the weights with the lowest validation loss; training continues for a patience period only to make sure the minimum is not improved.",
      "The last epoch has kept fitting the training data after the validation loss stopped improving; the stored best weights are restored."),
  ("Oprirea timpurie", f"Pierderea pe validare a DNN a fost minimă la epoca {N['train']['dnn_best_epoch']} din {N['train']['dnn_epochs']}. Ce ponderi se folosesc la prognoză?",
   ["Cele din ultima epocă", "Cele din prima epocă", "Media tuturor epocilor", f"Cele din epoca {N['train']['dnn_best_epoch']}, restaurate după oprirea antrenării"],
   "Oprirea timpurie păstrează ponderile cu cea mai mică pierdere pe validare; antrenarea continuă o perioadă de răbdare doar pentru a verifica că minimul nu mai este îmbunătățit.",
   "Ultima epocă a continuat să se potrivească datelor de antrenare după ce pierderea pe validare nu mai scădea; ponderile optime salvate sînt restaurate."))
q(1, ("Instance normalisation", "In October 2026 the price level jumped above the training range. What helped the DNN cope with this?",
      ["A larger learning rate", "Scaling each day by its own recent level and spread before the network sees it", "Removing the German prices", "Training on fewer days"],
      "With per-day (instance) normalisation the network predicts the shape around yesterday's level, so a new level does not push the inputs outside the range seen in training.",
      "The issue is a level shift; per-sample scaling removes it, while a different learning rate or fewer inputs do not."),
  ("Normalizarea pe eșantion", "În octombrie 2026 nivelul prețului a sărit peste intervalul de antrenare. Ce a ajutat rețeaua DNN să facă față?",
   ["O rată de învățare mai mare", "Scalarea fiecărei zile după propriul nivel și propria dispersie recentă înainte ca rețeaua să o vadă", "Eliminarea prețurilor din Germania", "Antrenarea pe mai puține zile"],
   "Cu normalizarea pe zi (pe eșantion), rețeaua prezice forma în jurul nivelului de ieri, astfel încît un nivel nou nu împinge intrările în afara intervalului văzut la antrenare.",
   "Problema este o schimbare de nivel; scalarea pe eșantion o elimină, în timp ce o altă rată de învățare sau mai puține intrări nu o fac."))
q(0, ("DeepAR output", "What does DeepAR output at each time step?",
      ["The parameters of a probability distribution for the next value", "A single point forecast", "A class label", "The attention weights of the series"],
      "DeepAR is trained by maximum likelihood: the LSTM state is mapped to, for example, the mean and standard deviation of the next value.",
      "DeepAR is probabilistic by design: it outputs distribution parameters, from which point forecasts and quantiles follow."),
  ("Ieșirea DeepAR", "Ce produce DeepAR la fiecare pas de timp?",
   ["Parametrii unei distribuții de probabilitate pentru valoarea următoare", "O singură prognoză punctuală", "O etichetă de clasă", "Ponderile de atenție ale seriei"],
   "DeepAR se antrenează prin verosimilitate maximă: starea LSTM este transformată, de exemplu, în media și abaterea standard ale valorii următoare.",
   "DeepAR este probabilistic prin construcție: produce parametrii unei distribuții, din care rezultă prognoze punctuale și cuantile."))
q(2, ("N-BEATS backcast", "In N-BEATS, what is the role of the backcast of each block?",
      ["It is the final forecast", "It is the loss function", "It removes from the input what the block has explained, so the next block works on the residual",
       "It adds noise for regularisation"],
      "Doubly residual stacking: each block subtracts its backcast from the input and adds its forecast to the total, like boosting inside one network.",
      "The forecast is the sum of the blocks' forecast outputs; the backcast is the part of the input explained by a block, which is subtracted."),
  ("Reconstrucția în N-BEATS", "În N-BEATS, care este rolul reconstrucției (backcast) fiecărui bloc?",
   ["Este prognoza finală", "Este funcția de pierdere", "Elimină din intrare ce a explicat blocul, astfel încît blocul următor lucrează pe reziduu",
    "Adaugă zgomot pentru regularizare"],
   "Suprapunerea dublu reziduală: fiecare bloc scade reconstrucția sa din intrare și adaugă prognoza sa la total, ca un boosting în interiorul unei rețele.",
   "Prognoza este suma ieșirilor de prognoză ale blocurilor; reconstrucția este partea din intrare explicată de un bloc, care se scade."))
q(1, ("Patches", "A context of 2,048 hours is cut into patches of 16 hours. How many tokens does the transformer see?",
      ["2,048", "128", "16", "32,768"],
      "2,048 / 16 = 128 tokens; since attention cost grows with the square of the number of tokens, it falls 16² = 256 times.",
      "Patching groups 16 consecutive hours into one token, so the number of tokens is the context length divided by the patch length."),
  ("Petice", "Un context de 2.048 de ore este tăiat în petice de 16 ore. Cîți tokeni vede arhitectura transformer?",
   ["2.048", "128", "16", "32.768"],
   "2.048 / 16 = 128 de tokeni; deoarece costul atenției crește cu pătratul numărului de tokeni, scade de 16² = 256 de ori.",
   "Peticele grupează 16 ore consecutive într-un token, deci numărul de tokeni este lungimea contextului împărțită la lungimea peticului."))
q(3, ("Zero-shot", "What does 'zero-shot' mean for Chronos-2 in this lecture?",
      ["It was trained on Romanian prices for zero epochs and then fine-tuned", "It forecasts zero values when data are missing",
       "It needs no context", "Its pre-trained weights are used as they are, with no gradient step on our data"],
      "The model only reads the context window of our series and returns quantiles; the weights learned in pre-training stay frozen.",
      "Zero-shot refers to the absence of training or fine-tuning on the target data, not to missing data or to an empty context."),
  ("Zero-shot", "Ce înseamnă „zero-shot” pentru Chronos-2 în acest curs?",
   ["A fost antrenat zero epoci pe prețurile din România și apoi ajustat fin", "Prognozează zero cînd lipsesc date",
    "Nu are nevoie de context", "Ponderile preantrenate sînt folosite ca atare, fără niciun pas de gradient pe datele noastre"],
   "Modelul doar citește fereastra de context a seriei noastre și întoarce cuantile; ponderile învățate la preantrenare rămîn fixe.",
   "Zero-shot se referă la lipsa antrenării sau a ajustării fine pe datele-țintă, nu la date lipsă sau la un context gol."))
q(0, ("Chronos scaling", "Before cutting values into bins, the original Chronos divides the series by its mean absolute value. Why?",
      ["So that series in EUR/MWh, megawatts or sales counts become comparable inputs for one model", "To remove negative values",
       "To make the series stationary", "To reduce the number of tokens"],
      "Mean scaling puts all series on a common scale, so the same bins and the same network serve prices, loads and sales.",
      "Scaling by a positive constant does not remove negative values, does not change stationarity and does not change the number of time steps."),
  ("Scalarea Chronos", "Înainte de a împărți valorile în intervale, Chronos original împarte seria la valoarea ei absolută medie. De ce?",
   ["Pentru ca seriile în EUR/MWh, megawați sau număr de vînzări să devină intrări comparabile pentru același model", "Pentru a elimina valorile negative",
    "Pentru a face seria staționară", "Pentru a reduce numărul de tokeni"],
   "Scalarea după medie aduce toate seriile la o scară comună, astfel încît aceleași intervale și aceeași rețea servesc prețuri, consumuri și vînzări.",
   "Împărțirea la o constantă pozitivă nu elimină valorile negative, nu schimbă staționaritatea și nu schimbă numărul de pași de timp."))
best_fm = min(("chronos2", "chronos2_cov", "bolt", "timesfm"), key=lambda k: A[k]["mae"])
q(2, ("Best model", "Which forecast had the lowest MAE on the Romanian test year?",
      [f"LEAR ({en(A['lear']['mae'])} EUR/MWh)", f"The DNN ({en(A['dnn']['mae'])} EUR/MWh)",
       f"Chronos-2 with past covariates ({en(A['chronos2_cov']['mae'])} EUR/MWh)", f"N-BEATS ({en(A['nbeats']['mae'])} EUR/MWh)"],
      f"Chronos-2 with German and Hungarian prices and Romanian load as past covariates reached {en(A['chronos2_cov']['mae'])} EUR/MWh, zero-shot.",
      f"The trained models were all above {en(min(A['lear']['mae'], A['dnn']['mae'], A['nbeats']['mae']))} EUR/MWh; the foundation model with covariates was lower."),
  ("Cel mai bun model", "Ce prognoză a avut cea mai mică MAE în anul de test din România?",
   [f"LEAR ({ro(A['lear']['mae'])} EUR/MWh)", f"DNN ({ro(A['dnn']['mae'])} EUR/MWh)",
    f"Chronos-2 cu covariabile din trecut ({ro(A['chronos2_cov']['mae'])} EUR/MWh)", f"N-BEATS ({ro(A['nbeats']['mae'])} EUR/MWh)"],
   f"Chronos-2 cu prețurile din Germania și Ungaria și consumul României drept covariabile din trecut a ajuns la {ro(A['chronos2_cov']['mae'])} EUR/MWh, fără reantrenare.",
   f"Modelele antrenate au rămas toate peste {ro(min(A['lear']['mae'], A['dnn']['mae'], A['nbeats']['mae']))} EUR/MWh; modelul fundație cu covariabile a fost mai jos."))
q(1, ("Context length", f"Chronos-2 had MAE {en(N['ctx']['168'])} with 168 hours of context and {en(N['ctx']['2048'])} with 2,048 hours. What is the most plausible reason?",
      ["Longer contexts make the model larger", "One week shows the daily shape, but a longer context also reveals the recent price level and weekly pattern",
       "The model was retrained on the longer context", "Shorter contexts contain more missing values"],
      "The model only knows what is in the context; with about three months it sees the current level and several weekly cycles.",
      "The weights are the same in both runs; only the information in the context changes."),
  ("Lungimea contextului", f"Chronos-2 a avut MAE {ro(N['ctx']['168'])} cu un context de 168 de ore și {ro(N['ctx']['2048'])} cu 2.048 de ore. Care este explicația cea mai plauzibilă?",
   ["Contextele mai lungi fac modelul mai mare", "O săptămînă arată forma zilnică, dar un context mai lung arată și nivelul recent al prețului și tiparul săptămînal",
    "Modelul a fost reantrenat pe contextul mai lung", "Contextele scurte conțin mai multe valori lipsă"],
   "Modelul știe doar ce se află în context; cu circa trei luni, vede nivelul actual și mai multe cicluri săptămînale.",
   "Ponderile sînt aceleași în ambele rulări; se schimbă doar informația din context."))
q(3, ("Past covariates", "Why were the German prices and the Romanian load given to Chronos-2 as past-only covariates?",
      ["Because Chronos-2 cannot read future covariates", "Because they are irrelevant for the future", "To reduce the context length",
       "Because their values for the delivery day are not known when the auction closes"],
      "German prices of day d are set by the same coupled auction, and the load of day d is realised later; only their past is in the information set.",
      "Chronos-2 can read known future covariates (such as calendar or a load forecast); these particular series are simply unknown at gate closure."),
  ("Covariabile din trecut", "De ce au fost date lui Chronos-2 prețurile din Germania și consumul României doar ca covariabile din trecut?",
   ["Pentru că Chronos-2 nu poate citi covariabile viitoare", "Pentru că sînt irelevante pentru viitor", "Pentru a reduce lungimea contextului",
    "Pentru că valorile lor pentru ziua livrării nu se cunosc la închiderea licitației"],
   "Prețurile din Germania pentru ziua d sînt stabilite de aceeași licitație cuplată, iar consumul zilei d se realizează mai tîrziu; doar trecutul lor face parte din setul de informații.",
   "Chronos-2 poate citi covariabile viitoare cunoscute (calendar, o prognoză a consumului); aceste serii anume sînt pur și simplu necunoscute la închiderea licitației."))
q(0, ("Diebold–Mariano unit", "Why is the Diebold–Mariano test applied to daily losses summed over 24 hours rather than to 8,640 hourly errors?",
      ["The 24 hours of a day share the same information set, so hourly errors are not independent", "Hourly errors cannot be negative",
       "The test needs exactly 365 observations", "Daily sums always give smaller p-values"],
      "Treating hours as independent would overstate the sample size; the multivariate version uses one loss differential per forecast day.",
      "The issue is dependence: all hours of a day are forecast from the same information, so the effective sample is the number of days."),
  ("Unitatea testului Diebold–Mariano", "De ce se aplică testul Diebold–Mariano pierderilor zilnice însumate pe 24 de ore, și nu celor 8.640 de erori orare?",
   ["Cele 24 de ore ale unei zile folosesc același set de informații, deci erorile orare nu sînt independente", "Erorile orare nu pot fi negative",
    "Testul cere exact 365 de observații", "Sumele zilnice dau întotdeauna valori p mai mici"],
   "Tratarea orelor ca independente ar supraevalua dimensiunea eșantionului; varianta multivariată folosește o singură diferență a pierderilor pentru fiecare zi de prognoză.",
   "Problema este dependența: toate orele unei zile sînt prognozate din aceeași informație, deci eșantionul efectiv este numărul de zile."))
q(2, ("Reading a p-value", f"The one-sided DM test of 'DNN more accurate than LEAR' gave p = {en(N['dm']['lear__dnn'], 3)}. What do you conclude?",
      ["The DNN is significantly better", "LEAR is significantly better at the 5 % level by this test", "There is no evidence that the DNN is more accurate than LEAR",
       "The test is invalid"],
      "A large one-sided p-value means the data give no evidence for the alternative 'the DNN is more accurate'; it is not, by itself, a test of the opposite direction.",
      "To claim that LEAR is better you would run the test in the other direction; a large p-value only fails to support the DNN."),
  ("Citirea unei valori p", f"Testul DM unilateral pentru „DNN mai precisă decît LEAR” a dat p = {ro(N['dm']['lear__dnn'], 3)}. Ce concluzionați?",
   ["DNN este semnificativ mai bună", "LEAR este semnificativ mai bun la 5 % după acest test", "Nu există dovezi că DNN este mai precisă decît LEAR",
    "Testul nu este valid"],
   "O valoare p unilaterală mare înseamnă că datele nu susțin alternativa „DNN este mai precisă”; nu este, singură, un test al direcției opuse.",
   "Pentru a afirma că LEAR este mai bun ar trebui rulat testul în direcția opusă; o valoare p mare doar nu susține DNN."))
q(1, ("Pinball loss", "For the 0.9-quantile forecast, the pinball loss is ρ(u) = u(τ − 1{u < 0}) with u = y − q. Which miss costs more per EUR/MWh?",
      ["A price below the forecast", "A price above the forecast: weight 0.9 against 0.1 below", "Both cost the same", "Neither: the loss is zero for τ = 0.9"],
      "For u > 0 (price above the quantile) the loss is 0.9·u; for u < 0 it is 0.1·|u|; a high quantile must rarely be exceeded.",
      "The asymmetry of the weights τ and 1 − τ is what makes the minimiser a quantile rather than the median."),
  ("Pierderea pinball", "Pentru prognoza cuantilei 0,9, pierderea pinball este ρ(u) = u(τ − 1{u < 0}), cu u = y − q. Care ratare costă mai mult pe EUR/MWh?",
   ["Un preț sub prognoză", "Un preț peste prognoză: pondere 0,9, față de 0,1 sub ea", "Ambele costă la fel", "Niciuna: pierderea este zero pentru τ = 0,9"],
   "Pentru u > 0 (prețul peste cuantilă) pierderea este 0,9·u; pentru u < 0 este 0,1·|u|; o cuantilă înaltă trebuie depășită rar.",
   "Asimetria ponderilor τ și 1 − τ face ca minimul să fie o cuantilă, nu mediana."))
c80, k80 = N["cov"]["chronos2"]["80"], N["cov"]["chronos2_conf"]["80"]
q(3, ("Conformal step", f"The raw 80 % Chronos-2 band covered {en(c80['cov'], 1)} % of test hours; after the conformal step {en(k80['cov'], 1)} %. What did the conformal step do?",
      ["It retrained Chronos-2 on Romanian data", "It replaced the quantiles by LEAR forecasts", "It narrowed the band",
       f"It widened the band by the recent out-of-band distance (mean width {en(c80['width'])} → {en(k80['width'])} EUR/MWh)"],
      "The correction adds Q̂, a high quantile of recent conformity scores, to both sides of the band; when the raw band was too narrow, Q̂ is positive.",
      "The conformal step changes only the band edges, by a quantity estimated on the previous 60 days; it does not retrain or replace the model."),
  ("Pasul conformal", f"Banda brută de 80 % a lui Chronos-2 a acoperit {ro(c80['cov'], 1)} % din orele de test; după pasul conformal, {ro(k80['cov'], 1)} %. Ce a făcut pasul conformal?",
   ["A reantrenat Chronos-2 pe date din România", "A înlocuit cuantilele cu prognozele LEAR", "A îngustat banda",
    f"A lărgit banda cu distanța recentă în afara benzii (lățimea medie {ro(c80['width'])} → {ro(k80['width'])} EUR/MWh)"],
   "Corecția adaugă Q̂, o cuantilă mare a scorurilor de conformitate recente, pe ambele părți ale benzii; cînd banda brută era prea îngustă, Q̂ este pozitiv.",
   "Pasul conformal modifică doar marginile benzii, cu o mărime estimată pe ultimele 60 de zile; nu reantrenează și nu înlocuiește modelul."))
q(0, ("Conformal guarantee", "Under which assumption is the split-conformal coverage guarantee exact?",
      ["Exchangeability of the calibration and test scores", "Normality of the errors", "Stationarity of the mean only", "A correctly specified forecasting model"],
      "The guarantee follows from the uniform rank of a new score among exchangeable scores; it needs no model or distribution assumption.",
      "Conformal prediction is distribution-free and model-free; for time series exchangeability fails, so the coverage is approximate."),
  ("Garanția conformală", "Sub ce ipoteză este exactă garanția de acoperire a predicției conformale?",
   ["Interschimbabilitatea scorurilor de calibrare și de test", "Normalitatea erorilor", "Doar staționaritatea mediei", "Un model de prognoză corect specificat"],
   "Garanția rezultă din rangul uniform al unui scor nou printre scoruri interschimbabile; nu are nevoie de ipoteze asupra modelului sau distribuției.",
   "Predicția conformală nu depinde de distribuție sau de model; pentru serii de timp interschimbabilitatea nu este îndeplinită, deci acoperirea este aproximativă."))
L_ = N["load"]
q(2, ("Load versus price", f"Chronos-2 forecast Romanian load with MAPE {en(L_['chronos2']['mape'])} %. Why is MAPE acceptable for load but not for price?",
      ["Load is measured in MW", "Load forecasts are always more accurate", "Load is never close to zero, so dividing by it is safe", "MAPE is required by the regulator for load"],
      f"The mean load is about {L_['mean_mw']:,.0f} MW and never near zero, so the percentage error is well defined and easy to communicate.",
      "The unit is irrelevant; the problem with MAPE is division by values near zero or below, which happens for prices but not for load."),
  ("Consum față de preț", f"Chronos-2 a prognozat consumul României cu MAPE {ro(L_['chronos2']['mape'])} %. De ce este MAPE acceptabilă pentru consum, dar nu pentru preț?",
   ["Consumul se măsoară în MW", "Prognozele consumului sînt întotdeauna mai precise", "Consumul nu este niciodată aproape de zero, deci împărțirea la el este sigură", "MAPE este cerută de autoritatea de reglementare pentru consum"],
   f"Consumul mediu este de circa {L_['mean_mw']:,.0f} MW și nu se apropie niciodată de zero, deci eroarea procentuală este bine definită și ușor de comunicat.".replace(",", " "),
   "Unitatea de măsură nu contează; problema MAPE este împărțirea la valori apropiate de zero sau negative, care apar la prețuri, nu la consum."))
V = N["var"]["SP500"]
q(1, ("VaR violations", f"Over {V['n']:,} trading days, how many violations does a correct 1 % VaR produce on average?",
      [f"{V['n'] // 10}", en(V['exp_01'], 1), "1", "250"],
      f"Expected violations = α × T = 0.01 × {V['n']:,} = {en(V['exp_01'], 1)}.",
      "Under correct coverage the hit indicator is Bernoulli(α), so the expected count is α times the number of days."),
  ("Depășiri VaR", f"Pe {V['n']:,} zile de tranzacționare, cîte depășiri produce în medie o VaR de 1 % corectă?".replace(",", " "),
   [f"{V['n'] // 10}", ro(V['exp_01'], 1), "1", "250"],
   f"Numărul așteptat de depășiri = α × T = 0,01 × {V['n']:,} = {ro(V['exp_01'], 1)}.".replace(",", " ", 1),
   "Cu o acoperire corectă, indicatorul de depășire este Bernoulli(α), deci numărul așteptat este α înmulțit cu numărul de zile."))
q(3, ("GARCH backtest", f"GARCH(1,1)-t had {V['garch_01']['viol']} violations of the 1 % VaR on the S&P 500 (Kupiec p = {en(V['garch_01']['kupiec_p'], 3)}). What does this say?",
      ["GARCH overestimates risk", "The violations are clustered", "The test has no power", "GARCH underestimated the 1 % tail: too many violations, coverage rejected"],
      f"{V['garch_01']['viol']} violations against {en(V['exp_01'], 1)} expected is a hit rate of {en(V['garch_01']['rate'])} %, and the small Kupiec p-value rejects correct coverage.",
      "Too many violations mean the VaR was not low enough, i.e. risk was underestimated; clustering is tested separately by Christoffersen."),
  ("Testarea GARCH", f"GARCH(1,1)-t a avut {V['garch_01']['viol']} de depășiri ale VaR de 1 % pentru S&P 500 (Kupiec p = {ro(V['garch_01']['kupiec_p'], 3)}). Ce înseamnă?",
   ["GARCH supraestimează riscul", "Depășirile vin grupat", "Testul nu are putere", "GARCH a subestimat coada de 1 %: prea multe depășiri, acoperirea este respinsă"],
   f"{V['garch_01']['viol']} de depășiri față de {ro(V['exp_01'], 1)} așteptate înseamnă o rată de {ro(V['garch_01']['rate'])} %, iar valoarea p mică a testului Kupiec respinge acoperirea corectă.",
   "Prea multe depășiri înseamnă că VaR nu a fost suficient de jos, adică riscul a fost subestimat; gruparea se testează separat, cu testul Christoffersen."))
q(0, ("Christoffersen", "What does the Christoffersen independence test add to the Kupiec test?",
      ["It checks whether violations cluster in time", "It checks the number of violations again", "It tests normality of returns", "It compares two VaR models"],
      "It compares the probability of a violation after a violation with that after a normal day; a VaR that reacts slowly to shocks produces clusters.",
      "Kupiec counts violations; Christoffersen looks at their timing through the transition probabilities of the hit sequence."),
  ("Christoffersen", "Ce adaugă testul de independență Christoffersen față de testul Kupiec?",
   ["Verifică dacă depășirile vin grupat în timp", "Numără din nou depășirile", "Testează normalitatea randamentelor", "Compară două modele VaR"],
   "Compară probabilitatea unei depășiri după o depășire cu cea după o zi obișnuită; o VaR care reacționează încet la șocuri produce grupuri de depășiri.",
   "Testul Kupiec numără depășirile; testul Christoffersen se uită la momentul lor, prin probabilitățile de tranziție ale șirului de depășiri."))
q(2, ("One-sided shift", "The conformal VaR is q̂ − c, where c is a high quantile of past scores s = q̂ − r. When does c become positive?",
      ["When the raw VaR was breached too rarely", "Never", "When the raw VaR was breached too often in the calibration window", "When returns are positive"],
      "If r fell below q̂ too often, many scores s are positive and their high quantile c is positive; subtracting c lowers (widens) the VaR.",
      "A too conservative raw VaR gives a negative c and the correction raises it; the sign of c follows the past breach frequency, not the sign of returns."),
  ("Deplasarea unilaterală", "VaR conformală este q̂ − c, unde c este o cuantilă mare a scorurilor trecute s = q̂ − r. Cînd devine c pozitiv?",
   ["Cînd VaR brută a fost depășită prea rar", "Niciodată", "Cînd VaR brută a fost depășită prea des în fereastra de calibrare", "Cînd randamentele sînt pozitive"],
   "Dacă r a căzut prea des sub q̂, multe scoruri s sînt pozitive, iar cuantila lor mare c este pozitivă; scăderea lui c coboară (lărgește) VaR.",
   "O VaR brută prea prudentă dă un c negativ, iar corecția o ridică; semnul lui c urmează frecvența depășirilor din trecut, nu semnul randamentelor."))
q(1, ("Ensemble", f"The ensemble (mean of LEAR, DNN and Chronos-2) had MAE {en(A['ens']['mae'])} EUR/MWh. Why can an average beat some of its members?",
      ["Averaging removes all bias", "Errors of different models are not perfectly correlated, so they partly cancel", "The ensemble uses more data", "It is trained on the test set"],
      "When errors are imperfectly correlated, the mean forecast has a lower expected loss than the average of the members' losses.",
      "The ensemble uses the same information as its members and no test data; its gain comes from diversification of errors."),
  ("Ansamblul", f"Ansamblul (media LEAR, DNN și Chronos-2) a avut MAE {ro(A['ens']['mae'])} EUR/MWh. De ce poate o medie să depășească unii dintre membrii ei?",
   ["Media elimină orice abatere sistematică", "Erorile modelelor diferite nu sînt perfect corelate, deci se compensează parțial", "Ansamblul folosește mai multe date", "Este antrenat pe setul de test"],
   "Cînd erorile sînt imperfect corelate, prognoza medie are o pierdere așteptată mai mică decît media pierderilor membrilor.",
   "Ansamblul folosește aceeași informație ca membrii lui și nicio dată de test; cîștigul vine din diversificarea erorilor."))
q(3, ("Training window", "Why does the lecture fix the LEAR and DNN training window at three years before seeing the test results?",
      ["Because three years is always optimal", "To include the 2008 crisis", "Because foundation models need it",
       "Choosing the window on the test year would leak test information into the model"],
      "Every choice made after looking at the test set (window, prompt, threshold) turns the test set into a validation set and inflates the result.",
      "The three-year window follows Lago et al. (2021) and was fixed a priori; its value is not claimed to be optimal."),
  ("Fereastra de antrenare", "De ce fixează cursul fereastra de antrenare LEAR și DNN la trei ani înainte de a vedea rezultatele pe test?",
   ["Pentru că trei ani sînt întotdeauna optimi", "Pentru a include criza din 2008", "Pentru că modelele fundație au nevoie de ea",
    "Alegerea ferestrei pe anul de test ar introduce informație din test în model"],
   "Orice alegere făcută după ce ați văzut testul (fereastra, promptul, pragul) transformă setul de test în set de validare și umflă rezultatul.",
   "Fereastra de trei ani urmează Lago et al. (2021) și a fost fixată a priori; nu se afirmă că valoarea ei este optimă."))
q(0, ("Evening errors", f"The worst hour of LEAR was {N['byhour']['lear']['worst_hour']}:00. Which explanation fits the Romanian price profile?",
      ["The evening ramp, when solar output disappears and demand is high, makes prices volatile", "Night hours have no trading",
       "The auction closes in the evening", "Evening prices are always negative"],
      "After sunset, solar generation drops while demand stays high; the steep part of the supply curve is reached and spikes appear.",
      "The evening hours are the most expensive and the most volatile in the profile, so they carry the largest errors."),
  ("Erorile de seară", f"Ora cea mai grea pentru LEAR a fost {N['byhour']['lear']['worst_hour']}:00. Ce explicație se potrivește profilului prețurilor din România?",
   ["Rampa de seară, cînd producția solară dispare și consumul este mare, face prețurile volatile", "Orele de noapte nu au tranzacții",
    "Licitația se închide seara", "Prețurile de seară sînt întotdeauna negative"],
   "După apusul soarelui, producția solară scade în timp ce consumul rămîne ridicat; se ajunge în partea abruptă a curbei ofertei și apar vîrfuri.",
   "Orele de seară sînt cele mai scumpe și cele mai volatile din profil, deci au cele mai mari erori."))
q(2, ("Foundation-model limit", "Which of the following is a real limitation of zero-shot foundation models for price forecasting?",
      ["They cannot output quantiles", "They need a graphics card for training on our data", "They do not know market rules or future events unless given as covariates",
       "They only accept daily data"],
      "The model knows patterns from pre-training; the coupling rules, outages or a new price cap are invisible unless encoded in the inputs.",
      "These models output quantiles directly, are not trained on our data and accept hourly series; their knowledge of the market is limited to the context."),
  ("Limita modelelor fundație", "Care dintre următoarele este o limită reală a modelelor fundație folosite fără reantrenare pentru prognoza prețurilor?",
   ["Nu pot produce cuantile", "Au nevoie de o placă grafică pentru antrenarea pe datele noastre", "Nu cunosc regulile pieței sau evenimentele viitoare decît dacă le primesc drept covariabile",
    "Acceptă doar date zilnice"],
   "Modelul cunoaște tipare din preantrenare; regulile de cuplare, opririle de centrale sau un nou plafon de preț sînt invizibile dacă nu sînt codificate în intrări.",
   "Aceste modele produc direct cuantile, nu se antrenează pe datele noastre și acceptă serii orare; cunoștințele lor despre piață se limitează la context."))
q(1, ("LLM-VaR versus Chronos", "What is the main difference between LLM-VaR (Lecture 12) and the Chronos-2 VaR of this lecture?",
      ["Only Chronos-2 is zero-shot", "LLM-VaR reads the returns as text in a prompt; Chronos-2 reads them as scaled numbers and outputs quantiles directly",
       "LLM-VaR does not need backtesting", "Chronos-2 needs fine-tuning on returns"],
      "Both are zero-shot; the language-model route goes through text, the time-series route through numeric patches and quantile heads.",
      "Both routes are zero-shot and both must be backtested with the same tests; what differs is how the numbers enter the model."),
  ("LLM-VaR față de Chronos", "Care este diferența principală dintre LLM-VaR (cursul 12) și VaR Chronos-2 din acest curs?",
   ["Doar Chronos-2 este zero-shot", "LLM-VaR citește randamentele ca text într-un prompt; Chronos-2 le citește ca numere scalate și produce direct cuantile",
    "LLM-VaR nu trebuie testată retrospectiv", "Chronos-2 trebuie ajustat fin pe randamente"],
   "Ambele sînt zero-shot; calea modelului de limbaj trece prin text, calea modelului de serii de timp prin petice numerice și ieșiri de cuantile.",
   "Ambele căi sînt zero-shot și ambele trebuie testate retrospectiv cu aceleași teste; diferă doar modul în care numerele intră în model."))
q(3, ("Solar trough", f"In 2026 the cheapest hour of the Romanian day-ahead market was {N['facts']['profile']['2026']['min_hour']}:00. What changed compared with 2019, when the cheapest hours were at night?",
      ["Night demand increased sharply", "The auction moved to midday", "Gas became cheaper at midday", "Solar generation now floods the market around noon"],
      "Large solar capacity produces at near-zero marginal cost around midday, shifting the supply curve and creating a midday price trough.",
      "The change is on the supply side: solar output, not the auction design or night demand, explains the midday minimum."),
  ("Minimul solar", f"În 2026, ora cea mai ieftină a pieței pentru ziua următoare din România a fost {N['facts']['profile']['2026']['min_hour']}:00. Ce s-a schimbat față de 2019, cînd orele cele mai ieftine erau noaptea?",
   ["Consumul de noapte a crescut puternic", "Licitația s-a mutat la prînz", "Gazul a devenit mai ieftin la prînz", "Producția solară inundă acum piața în jurul prînzului"],
   "Capacitatea solară mare produce la un cost marginal aproape nul în jurul prînzului, deplasează curba ofertei și creează un minim de preț la prînz.",
   "Schimbarea este de partea ofertei: producția solară, nu designul licitației sau consumul de noapte, explică minimul de la prînz."))
q(0, ("Test set used once", "After seeing the test-year results, a colleague wants to try ten more DNN architectures and keep the best. What is the problem?",
      ["The test year becomes a validation set and the reported error becomes optimistic", "Ten architectures are too few", "The DNN cannot be retrained",
       "Nothing, as long as the seeds are fixed"],
      "Selecting on the test set inflates performance; architectures must be chosen on a validation period before the test year.",
      "Fixed seeds do not prevent selection bias; the issue is reusing the test data for choices."),
  ("Setul de test folosit o singură dată", "După ce a văzut rezultatele din anul de test, un coleg vrea să încerce încă zece arhitecturi DNN și să o păstreze pe cea mai bună. Care este problema?",
   ["Anul de test devine set de validare, iar eroarea raportată devine optimistă", "Zece arhitecturi sînt prea puține", "DNN nu poate fi reantrenată",
    "Niciuna, dacă valorile generatorului aleator sînt fixe"],
   "Selecția pe setul de test umflă performanța; arhitecturile trebuie alese pe o perioadă de validare dinaintea anului de test.",
   "Valorile fixe ale generatorului aleator nu împiedică abaterea de selecție; problema este refolosirea datelor de test pentru decizii."))
q(2, ("Bitcoin coverage", f"For Bitcoin, the raw Chronos-2 5 % VaR had {N['var']['BTCUSD']['ch_05']['viol']} violations and a Christoffersen p-value of {en(N['var']['BTCUSD']['ch_05']['chris_p'], 3)}. What does the second number indicate?",
      ["Too many violations", "Too few violations", "Some clustering of violations at the 5 % significance level", "Normal returns"],
      "A Christoffersen p-value below 0.05 rejects independence: a violation makes another violation on the next day more likely.",
      "The count is judged by Kupiec; the Christoffersen test is about the timing of violations."),
  ("Acoperirea pentru Bitcoin", f"Pentru Bitcoin, VaR brută Chronos-2 de 5 % a avut {N['var']['BTCUSD']['ch_05']['viol']} de depășiri și o valoare p Christoffersen de {ro(N['var']['BTCUSD']['ch_05']['chris_p'], 3)}. Ce indică al doilea număr?",
   ["Prea multe depășiri", "Prea puține depășiri", "O grupare a depășirilor, la pragul de semnificație de 5 %", "Randamente normale"],
   "O valoare p Christoffersen sub 0,05 respinge independența: o depășire face mai probabilă o altă depășire în ziua următoare.",
   "Numărul depășirilor se judecă prin testul Kupiec; testul Christoffersen privește momentul depășirilor."))

assert len(Q) >= 30, len(Q)
js = ("// ============================================================\n"
      "// Quiz bank for lecture id 'l13': Deep learning for energy and financial time series; time-series foundation models (EN + RO)\n"
      "// GENERATED by lectures/L13/make_quiz_L13.py from L13_numbers.json; correct = index (0-3) of the right option.\n"
      "// incorrectExplanation never names a letter: the engine prepends the correct answer after shuffling.\n"
      "// ============================================================\n"
      "window.NN_DATA.quizzes['l13'] = " + json.dumps({"draw": 20, "questions": Q}, ensure_ascii=False, indent=4) + ";\n")
with open(OUT, "w", encoding="utf-8") as f:
    f.write(js)
print(f"{len(Q)} questions written to assets/quizzes/l13.js")
