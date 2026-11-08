"""
make_quiz_L11.py -- quiz bank of Lecture 11 (EN + RO): assets/quizzes/l11.js.
Every number comes from L11_numbers.json; correct = index (0-3) of the right option; incorrectExplanation never names a letter.
Run:  python3 lectures/L11/make_quiz_L11.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = json.load(open(os.path.join(HERE, "L11_numbers.json")))


def g(key):
    v = N
    for p in key.split("."):
        v = v[p]
    return v


def f(key, d=2, lang="en"):
    s = f"{g(key):.{d}f}"
    if lang == "ro":
        s = s.replace(".", ",").replace("-", "−")
    return s


Q = []


def q(correct, en, ro):
    keys = ("title", "text", "options", "correctExplanation", "incorrectExplanation")
    assert len(en[2]) == 4 and len(ro[2]) == 4
    Q.append({"correct": correct, "en": dict(zip(keys, en)), "ro": dict(zip(keys, ro))})


E, R = (lambda k, d=2: f(k, d, "en")), (lambda k, d=2: f(k, d, "ro"))

q(2, ("Shapley efficiency", f"For a test applicant, the base value is $\\phi_0={E('app.base',3)}$ and the SHAP values sum to {E('app.sum_phi',3)}. What is the model's log-odds of default?",
       [f"{E('app.sum_phi',3)}, the sum of the SHAP values alone", f"{E('app.base',3)}, the base value", f"{E('app.logit',3)}, the base value plus the sum of the SHAP values", "It cannot be computed without the model weights"],
       f"Efficiency: $f(\\mathbf x)=\\phi_0+\\sum_j\\phi_j={E('app.base',3)}+{E('app.sum_phi',3)}={E('app.logit',3)}$.",
       "SHAP values split the difference between the prediction and the average prediction, so the base value must be added back."),
  ("Eficiența Shapley", f"Pentru un solicitant din test, valoarea de bază este $\\phi_0={R('app.base',3)}$, iar valorile SHAP se adună la {R('app.sum_phi',3)}. Care sînt log-șansele de neplată date de model?",
       [f"{R('app.sum_phi',3)}, doar suma valorilor SHAP", f"{R('app.base',3)}, valoarea de bază", f"{R('app.logit',3)}, valoarea de bază plus suma valorilor SHAP", "Nu se pot calcula fără ponderile modelului"],
       f"Eficiența: $f(\\mathbf x)=\\phi_0+\\sum_j\\phi_j={R('app.base',3)}+{R('app.sum_phi',3)}={R('app.logit',3)}$.",
       "Valorile SHAP împart diferența dintre prognoză și prognoza medie, deci valoarea de bază trebuie adăugată înapoi."))
q(0, ("Shapley axioms", "Which property guarantees that a feature that never changes the model output receives a SHAP value of zero?",
       ["The dummy (null player) axiom", "Efficiency", "Additivity", "Monotonicity of the network"],
       "The dummy axiom: if $v(S\\cup\\{j\\})=v(S)$ for every coalition, then $\\phi_j=0$.",
       "Efficiency makes the values add up and additivity concerns sums of games; only the dummy axiom deals with features that never matter."),
  ("Axiomele Shapley", "Ce proprietate garantează că o variabilă care nu schimbă niciodată ieșirea modelului primește valoarea SHAP zero?",
       ["Axioma variabilei nule (jucătorul nul)", "Eficiența", "Aditivitatea", "Monotonia rețelei"],
       "Axioma variabilei nule: dacă $v(S\\cup\\{j\\})=v(S)$ pentru orice coaliție, atunci $\\phi_j=0$.",
       "Eficiența face ca valorile să se adune, iar aditivitatea privește sumele de jocuri; doar axioma variabilei nule tratează variabilele care nu contează niciodată."))
q(3, ("Number of coalitions", f"The HMEQ credit model uses {g('hmeq.n_feat')} features. How many coalitions does an exact Shapley computation need for one applicant?",
       [f"{g('hmeq.n_feat')}", f"{g('hmeq.n_feat')**2}", "$18!$", "$2^{18}$"],
       "Every subset of the features is a coalition: $2^{d}=2^{18}$ model values per applicant, which is why Kernel SHAP samples them.",
       "Shapley values average over subsets of features; the number of subsets of $d$ features is $2^d$, far more than $d$ or $d^2$."),
  ("Numărul de coaliții", f"Modelul de credit HMEQ folosește {g('hmeq.n_feat')} variabile. De cîte coaliții are nevoie un calcul Shapley exact pentru un solicitant?",
       [f"{g('hmeq.n_feat')}", f"{g('hmeq.n_feat')**2}", "$18!$", "$2^{18}$"],
       "Orice submulțime de variabile este o coaliție: $2^{d}=2^{18}$ evaluări ale modelului pentru fiecare solicitant; de aceea Kernel SHAP le eșantionează.",
       "Valorile Shapley mediază pe submulțimi de variabile; numărul submulțimilor a $d$ variabile este $2^d$, mult mai mare decît $d$ sau $d^2$."))
q(1, ("Linear SHAP", "For a logistic regression with independent features, what is the SHAP value of feature $j$ on the log-odds scale?",
       ["$\\beta_j x_j$", "$\\beta_j(x_j-\\mathbb E x_j)$", "$\\beta_j$", "$\\beta_j^2\\,\\mathrm{Var}(x_j)$"],
       "With a linear logit and independent features, $\\phi_j=\\beta_j(x_j-\\mathbb E x_j)$: the coefficient times the deviation from the average applicant.",
       "SHAP values explain the difference from the average prediction, so the feature value must be centred at its background mean."),
  ("SHAP liniar", "Pentru o regresie logistică cu variabile independente, care este valoarea SHAP a variabilei $j$ pe scara log-șanselor?",
       ["$\\beta_j x_j$", "$\\beta_j(x_j-\\mathbb E x_j)$", "$\\beta_j$", "$\\beta_j^2\\,\\mathrm{Var}(x_j)$"],
       "Cu un logit liniar și variabile independente, $\\phi_j=\\beta_j(x_j-\\mathbb E x_j)$: coeficientul înmulțit cu abaterea față de solicitantul mediu.",
       "Valorile SHAP explică diferența față de prognoza medie, deci valoarea variabilei trebuie centrată pe media ei din eșantionul de referință."))
q(2, ("The dominant driver", f"In the HMEQ data the default rate is {E('hmeq.bad_if_debtinc_missing',1)} % when the debt-to-income ratio is missing and {E('hmeq.bad_if_debtinc_present',1)} % when it is reported. Which feature tops the SHAP ranking of all three models?",
       ["DELINQ, the number of delinquent lines", "LOAN, the loan amount", "DEBTINC_missing, the flag for a missing ratio", "CLAGE, the age of the oldest credit line"],
       "DEBTINC_missing is first for the MLP, XGBoost and the logistic regression: an incomplete file is a strong risk signal in these data.",
       "The gap in default rates between files with and without the ratio is very large, and the models pick it up through the missing-value flag."),
  ("Factorul dominant", f"În datele HMEQ, rata de neplată este {R('hmeq.bad_if_debtinc_missing',1)} % cînd raportul datorie/venit lipsește și {R('hmeq.bad_if_debtinc_present',1)} % cînd este declarat. Ce variabilă este pe primul loc în clasamentul SHAP al celor trei modele?",
       ["DELINQ, numărul liniilor cu restanțe", "LOAN, suma creditului", "DEBTINC_missing, indicatorul raportului lipsă", "CLAGE, vechimea celei mai vechi linii de credit"],
       "DEBTINC_missing este pe primul loc la MLP, XGBoost și regresia logistică: un dosar incomplet este un semnal puternic de risc în aceste date.",
       "Diferența de rată de neplată dintre dosarele cu și fără raport este foarte mare, iar modelele o preiau prin indicatorul de valoare lipsă."))
q(0, ("Additivity check", f"On {g('shap.n_explained')} test applicants, the largest gap between $\\phi_0+\\sum_j\\phi_j$ and $f(\\mathbf x)$ for Kernel SHAP was {g('shap.add_err'):.1e}. What does this show?",
       ["The explanation adds up to the model output, up to rounding", "The explanation is faithful to the true causes of default", "The model is well calibrated", "LIME would give the same values"],
       "The constrained regression of Kernel SHAP enforces efficiency, so the parts add up to the score; this says nothing about causality or calibration.",
       "Additivity is an internal property of the decomposition; it does not test calibration, causality or agreement with another method."),
  ("Verificarea aditivității", f"Pe {g('shap.n_explained')} de solicitanți din test, cea mai mare diferență dintre $\\phi_0+\\sum_j\\phi_j$ și $f(\\mathbf x)$ pentru Kernel SHAP a fost {f"{g('shap.add_err'):.1e}".replace(".", ",").replace("-", "−")}. Ce arată aceasta?",
       ["Explicația se adună la ieșirea modelului, pînă la erorile de rotunjire", "Explicația reflectă cauzele reale ale neplății", "Modelul este bine calibrat", "LIME ar da aceleași valori"],
       "Regresia cu restricții din Kernel SHAP impune eficiența, deci părțile se adună la scor; aceasta nu spune nimic despre cauzalitate sau calibrare.",
       "Aditivitatea este o proprietate internă a descompunerii; ea nu testează calibrarea, cauzalitatea sau concordanța cu altă metodă."))
q(3, ("Background sample", "In interventional SHAP, what does the background sample define?",
       ["The test set used for the AUC", "The features that can be changed by the customer", "The kernel width of the explainer", "The ``average applicant'' against which every prediction is explained"],
       "$\\phi_0=\\mathbb E_{\\mathcal B}[f]$ and the missing features are drawn from $\\mathcal B$: the background defines the reference point of every explanation.",
       "Changing the background changes $\\phi_0$ and every $\\phi_j$; it plays the role of the reference applicant and should be documented."),
  ("Eșantionul de referință", "În SHAP intervențional, ce definește eșantionul de referință?",
       ["Setul de test folosit pentru AUC", "Variabilele pe care clientul le poate schimba", "Lățimea nucleului explicației", "„Solicitantul mediu” față de care se explică fiecare prognoză"],
       "$\\phi_0=\\mathbb E_{\\mathcal B}[f]$, iar variabilele lipsă sînt extrase din $\\mathcal B$: referința definește punctul de plecare al fiecărei explicații.",
       "Schimbarea referinței schimbă $\\phi_0$ și fiecare $\\phi_j$; ea joacă rolul solicitantului de referință și trebuie documentată."))
q(1, ("Correlated features", "Why can interventional SHAP be misleading when features are strongly correlated?",
       ["Because the Shapley kernel becomes negative", "Because replacing features independently creates combinations that do not occur in the data", "Because TreeSHAP cannot handle more than ten features", "Because efficiency no longer holds"],
       "Missing features are drawn independently of the known ones, so the model is evaluated off the data manifold, on applicants that do not exist.",
       "Efficiency and the kernel are unaffected; the issue is that the value function averages the model over unrealistic inputs."),
  ("Variabile corelate", "De ce poate induce în eroare SHAP intervențional cînd variabilele sînt puternic corelate?",
       ["Pentru că nucleul Shapley devine negativ", "Pentru că înlocuirea independentă a variabilelor creează combinații care nu apar în date", "Pentru că TreeSHAP nu acceptă mai mult de zece variabile", "Pentru că eficiența nu mai este respectată"],
       "Variabilele lipsă sînt extrase independent de cele cunoscute, deci modelul este evaluat în afara distribuției datelor, pe solicitanți care nu există.",
       "Eficiența și nucleul nu sînt afectate; problema este că funcția de valoare mediază modelul pe intrări nerealiste."))
q(2, ("TreeSHAP", "Why can SHAP values of XGBoost be computed exactly while those of the MLP are approximated?",
       ["Because XGBoost has fewer features", "Because the MLP has no gradients", "Because the tree structure allows exact Shapley values in polynomial time", "Because XGBoost is a linear model"],
       "TreeSHAP exploits the tree structure to compute exact values in polynomial time; for the MLP we sample coalitions with Kernel SHAP.",
       "Both models use the same features; the difference is the algorithm that the tree structure makes possible."),
  ("TreeSHAP", "De ce pot fi calculate exact valorile SHAP ale XGBoost, în timp ce pentru MLP ele sînt aproximate?",
       ["Pentru că XGBoost are mai puține variabile", "Pentru că MLP-ul nu are gradienți", "Pentru că structura arborilor permite valori Shapley exacte în timp polinomial", "Pentru că XGBoost este un model liniar"],
       "TreeSHAP folosește structura arborilor pentru a calcula valori exacte în timp polinomial; pentru MLP eșantionăm coaliții cu Kernel SHAP.",
       "Ambele modele folosesc aceleași variabile; diferența este algoritmul pe care îl permite structura arborilor."))
q(0, ("Reading a waterfall", f"In the waterfall of the rejected applicant, DELINQ contributes {E('app.f3_phi')} while the applicant has no delinquent lines. How should this be read?",
       ["Having no delinquent lines lowers this applicant's risk compared with the average applicant", "The model treats delinquencies as a good sign", "The SHAP value is a sampling error", "DELINQ has no effect in the model"],
       "SHAP values are relative to the average applicant: a value of zero delinquencies is better than average, so it pulls the log-odds down.",
       "A negative contribution means the feature value moves the score below the average prediction, which is what a clean record should do."),
  ("Interpretarea unei diagrame cascadă", f"În diagrama cascadă a solicitantului respins, DELINQ contribuie cu {R('app.f3_phi')}, iar solicitantul nu are linii cu restanțe. Cum se interpretează?",
       ["Lipsa restanțelor scade riscul acestui solicitant față de solicitantul mediu", "Modelul tratează restanțele ca pe un semn bun", "Valoarea SHAP este o eroare de eșantionare", "DELINQ nu are efect în model"],
       "Valorile SHAP sînt relative la solicitantul mediu: zero restanțe înseamnă o situație mai bună decît media, deci log-șansele scad.",
       "O contribuție negativă înseamnă că valoarea variabilei coboară scorul sub prognoza medie, exact ce ar trebui să facă un istoric curat."))
q(1, ("Missing-value flag", "Why did the dependence plot of DEBTINC show small SHAP values for the files with a missing ratio?",
       ["Because those applicants have low debt", "Because their risk is carried by the missing flag, while DEBTINC sits at the imputed median", "Because Kernel SHAP ignores imputed values", "Because the network was not trained on those files"],
       f"All missing files share the median value of DEBTINC; the separate flag carries their risk, with a mean SHAP value of {E('dep.shap_missing_flag')}.",
       "Imputation gives these files one common DEBTINC value, so the information about the gap moves to the flag."),
  ("Indicatorul de valoare lipsă", "De ce a arătat graficul de dependență al DEBTINC valori SHAP mici pentru dosarele cu raport lipsă?",
       ["Pentru că acei solicitanți au datorii mici", "Pentru că riscul lor este preluat de indicatorul de lipsă, iar DEBTINC rămîne la mediana imputată", "Pentru că Kernel SHAP ignoră valorile imputate", "Pentru că rețeaua nu a fost antrenată pe acele dosare"],
       f"Toate dosarele cu valoare lipsă au aceeași valoare DEBTINC, mediana; indicatorul separat le preia riscul, cu o valoare SHAP medie de {R('dep.shap_missing_flag')}.",
       "Imputarea dă acestor dosare o valoare DEBTINC comună, deci informația despre lipsă trece la indicator."))
q(3, ("LIME objective", "What does LIME minimise to explain one prediction?",
       ["The squared error of the network on the training set", "The sum of the Shapley values", "The distance to the closest accepted applicant", "A proximity-weighted squared error between the network and a simple model, plus a complexity penalty"],
       "LIME solves $\\min_g\\sum_z\\pi_x(z)[f(z)-g(z)]^2+\\Omega(g)$ on perturbations around the instance.",
       "LIME fits a local surrogate; the distance to an accepted applicant is the counterfactual problem, and Shapley values come from SHAP."),
  ("Obiectivul LIME", "Ce minimizează LIME pentru a explica o prognoză?",
       ["Eroarea pătratică a rețelei pe setul de antrenare", "Suma valorilor Shapley", "Distanța pînă la cel mai apropiat solicitant acceptat", "O eroare pătratică ponderată după apropiere între rețea și un model simplu, plus o penalizare a complexității"],
       "LIME rezolvă $\\min_g\\sum_z\\pi_x(z)[f(z)-g(z)]^2+\\Omega(g)$ pe perturbări din jurul instanței.",
       "LIME estimează un substitut local; distanța pînă la un solicitant acceptat este problema contrafactuală, iar valorile Shapley vin din SHAP."))
q(2, ("LIME sample size", f"With 500 samples, LIME gave the same top feature in {E('lime.top1_same_500',0)} % of 30 runs; with 5,000 samples, in {E('lime.top1_same_5000',0)} %. What follows?",
       ["The network is unstable", "LIME should never be used", "The sampling noise of LIME must be controlled by fixing the seed and using enough samples", "SHAP has the same noise"],
       "The model did not change between runs; only LIME's random perturbations did. More samples and a fixed seed make the explanation reproducible.",
       "The differences come from the explainer, not from the model; they shrink with more samples."),
  ("Mărimea eșantionului LIME", f"Cu 500 de eșantioane, LIME a dat aceeași variabilă de top în {R('lime.top1_same_500',0)} % din 30 de rulări; cu 5 000 de eșantioane, în {R('lime.top1_same_5000',0)} %. Ce rezultă?",
       ["Rețeaua este instabilă", "LIME nu trebuie folosit niciodată", "Zgomotul de eșantionare al LIME trebuie controlat prin fixarea valorii inițiale și un eșantion suficient", "SHAP are același zgomot"],
       "Modelul nu s-a schimbat între rulări, ci doar perturbările aleatoare ale LIME. Mai multe eșantioane și o valoare inițială fixă fac explicația reproductibilă.",
       "Diferențele vin de la metoda de explicare, nu de la model, și se reduc cu mai multe eșantioane."))
q(1, ("Kernel width", f"The default LIME kernel width here is {E('lime.kw_default')}. What happens to the explanation when the width is changed from 0.5 to 10?",
       ["Nothing: LIME is invariant to the kernel width", "The slopes change, because the neighbourhood that defines ``local'' changes", "Only the intercept changes", "The explanation becomes a Shapley value"],
       f"The width sets which perturbations count; the slope of the top feature moved between {E('lime.kw_top1_min',3)} and {E('lime.kw_top1_max',3)} on our applicant.",
       "A wider kernel averages the model over a larger region, so the local slopes are different."),
  ("Lățimea nucleului", f"Lățimea implicită a nucleului LIME este aici {R('lime.kw_default')}. Ce se întîmplă cu explicația cînd lățimea trece de la 0,5 la 10?",
       ["Nimic: LIME nu depinde de lățimea nucleului", "Pantele se schimbă, pentru că se schimbă vecinătatea care definește caracterul „local”", "Se schimbă doar termenul liber", "Explicația devine o valoare Shapley"],
       f"Lățimea stabilește ce perturbări contează; panta variabilei de top s-a mutat între {R('lime.kw_top1_min',3)} și {R('lime.kw_top1_max',3)} pentru solicitantul nostru.",
       "Un nucleu mai larg mediază modelul pe o regiune mai mare, deci pantele locale sînt diferite."))
q(0, ("LIME vs SHAP", "For the rejected applicant, LIME's top feature is DELINQ and SHAP's is DEBTINC_missing. What is the best reading?",
       ["They answer different questions: a local slope near the applicant versus the distance from the average applicant", "One of the two methods has a bug", "The model must be retrained", "SHAP is always right and LIME always wrong"],
       "LIME measures how the score moves near the instance; SHAP splits the gap to the average prediction. A binary flag far from its mean dominates SHAP but has no local slope.",
       "Different definitions lead to different rankings; neither result signals an error in the model."),
  ("LIME față de SHAP", "Pentru solicitantul respins, variabila de top la LIME este DELINQ, iar la SHAP este DEBTINC_missing. Care este interpretarea cea mai bună?",
       ["Răspund la întrebări diferite: o pantă locală în jurul solicitantului față de distanța față de solicitantul mediu", "Una dintre metode are o eroare de programare", "Modelul trebuie reantrenat", "SHAP are mereu dreptate, iar LIME niciodată"],
       "LIME măsoară cum se mișcă scorul în jurul instanței; SHAP împarte distanța față de prognoza medie. Un indicator binar departe de medie domină SHAP, dar nu are pantă locală.",
       "Definiții diferite duc la clasamente diferite; niciunul dintre rezultate nu semnalează o eroare a modelului."))
q(3, ("Counterfactual", f"Reporting the missing debt-to-income ratio lowers the PD of the rejected applicant from {E('cf.pd_before',1)} % to {E('cf.pd_doc',1)} %. What is the most useful message to the customer?",
       ["Your job category is the problem", "Lower your mortgage by twenty per cent", "Nothing can be done", "Provide the missing income document"],
       "The documentary counterfactual changes one actionable item and brings the PD far below the threshold; it is the clearest advice.",
       "A good counterfactual is actionable and small; the missing document is both, unlike a change of job or a large debt reduction."),
  ("Contrafactual", f"Declararea raportului datorie/venit lipsă reduce PD-ul solicitantului respins de la {R('cf.pd_before',1)} % la {R('cf.pd_doc',1)} %. Care este mesajul cel mai util pentru client?",
       ["Categoria dumneavoastră de ocupație este problema", "Reduceți ipoteca cu douăzeci la sută", "Nu se poate face nimic", "Aduceți documentul de venit care lipsește"],
       "Contrafactualul documentar schimbă un singur element acționabil și aduce PD-ul mult sub prag; este sfatul cel mai clar.",
       "Un contrafactual bun este acționabil și mic; documentul lipsă le îndeplinește pe amîndouă, spre deosebire de o schimbare de ocupație sau o reducere mare a datoriei."))
q(1, ("Counterfactual objective", "In the Wachter counterfactual objective, why are the changes divided by the median absolute deviation of each feature?",
       ["To make the network convex", "To put changes in different units (currency, counts, ratios) on a comparable scale", "To guarantee that the PD falls to zero", "To remove the need for a threshold"],
       "Dividing by $\\mathrm{MAD}_j$ makes a change in euros and a change in counts comparable, so the optimiser does not favour features with small units.",
       "The scaling only affects how distance is measured; it does not change convexity or the decision threshold."),
  ("Obiectivul contrafactual", "În obiectivul contrafactual al lui Wachter, de ce se împart schimbările la abaterea absolută mediană a fiecărei variabile?",
       ["Pentru a face rețeaua convexă", "Pentru a aduce schimbări exprimate în unități diferite (monedă, numărători, rapoarte) pe o scară comparabilă", "Pentru a garanta că PD scade la zero", "Pentru a elimina nevoia unui prag"],
       "Împărțirea la $\\mathrm{MAD}_j$ face comparabile o schimbare în euro și una în numărători, astfel încît optimizarea nu favorizează variabilele cu unități mici.",
       "Scalarea schimbă doar modul de măsurare a distanței; nu afectează convexitatea sau pragul de decizie."))
q(2, ("Partial dependence", f"The partial dependence of the California MLP on median income rises from {E('pdp.pdp_lo')} to {E('pdp.pdp_hi')}, but the ICE curves fan out (standard deviation {E('pdp.ice_spread_hi')} at the top). What does the fan show?",
       ["The model is linear in income", "The PDP is wrong", "The income effect differs across block groups: there are interactions", "The data contain missing values"],
       "The PDP is an average; diverging ICE curves show that the effect of income depends on the other features of each block group.",
       "If the effect were the same everywhere, all ICE curves would be parallel to the PDP; their spread reveals heterogeneity."),
  ("Dependența parțială", f"Dependența parțială a MLP-ului pentru California de venitul median crește de la {R('pdp.pdp_lo')} la {R('pdp.pdp_hi')}, dar curbele ICE se răsfiră (abatere standard {R('pdp.ice_spread_hi')} la capăt). Ce arată răsfirarea?",
       ["Modelul este liniar în venit", "PDP este greșit", "Efectul venitului diferă între grupurile de locuințe: există interacțiuni", "Datele conțin valori lipsă"],
       "PDP este o medie; curbele ICE divergente arată că efectul venitului depinde de celelalte caracteristici ale fiecărui grup.",
       "Dacă efectul ar fi același peste tot, toate curbele ICE ar fi paralele cu PDP; dispersia lor arată eterogenitatea."))
q(0, ("Saliency", "Why can saliency $|\\partial f/\\partial x_j|$ be near zero for a feature that decides the prediction?",
       ["Because the network may be saturated (flat) around the input", "Because gradients are always zero in ReLU networks", "Because saliency is a Shapley value", "Because the feature is standardised"],
       "Saliency measures local sensitivity; in a flat region of $f$ a decisive feature can have a tiny gradient. Integrated gradients fix this by integrating along a path.",
       "ReLU gradients are piecewise constant, not zero, and standardisation does not remove an effect; the issue is saturation."),
  ("Saliența", "De ce poate fi saliența $|\\partial f/\\partial x_j|$ aproape zero pentru o variabilă care decide prognoza?",
       ["Pentru că rețeaua poate fi saturată (plată) în jurul intrării", "Pentru că gradienții sînt mereu zero în rețelele ReLU", "Pentru că saliența este o valoare Shapley", "Pentru că variabila este standardizată"],
       "Saliența măsoară sensibilitatea locală; într-o zonă plată a lui $f$, o variabilă decisivă poate avea un gradient foarte mic. Gradienții integrați corectează aceasta integrînd pe un drum.",
       "Gradienții ReLU sînt constanți pe porțiuni, nu zero, iar standardizarea nu elimină un efect; problema este saturarea."))
q(3, ("Integrated gradients", "Which property makes the integrated gradients of a transaction sum to $f(\\mathbf x)-f(\\mathbf x')$?",
       ["Sensitivity", "Implementation invariance", "Linearity of the network", "Completeness, from the fundamental theorem of calculus along the path"],
       "Integrating the gradient along the straight path from the baseline gives $\\sum_j\\mathrm{IG}_j=f(\\mathbf x)-f(\\mathbf x')$: completeness.",
       "Sensitivity and implementation invariance are other axioms of IG; the summing-up property is completeness and does not need a linear network."),
  ("Gradienții integrați", "Ce proprietate face ca gradienții integrați ai unei tranzacții să se adune la $f(\\mathbf x)-f(\\mathbf x')$?",
       ["Sensibilitatea", "Invarianța la implementare", "Liniaritatea rețelei", "Completitudinea, din teorema fundamentală a calculului integral pe drum"],
       "Integrarea gradientului pe drumul drept de la referință dă $\\sum_j\\mathrm{IG}_j=f(\\mathbf x)-f(\\mathbf x')$: completitudinea.",
       "Sensibilitatea și invarianța la implementare sînt alte axiome ale IG; proprietatea de însumare este completitudinea și nu cere o rețea liniară."))
q(1, ("Riemann steps", f"The median completeness error of IG was {E('ig.comp4_pct',1)} % with 4 steps and {E('ig.comp64_pct')} % with 64 steps. What should a production system do?",
       ["Use 4 steps: the error does not matter", "Choose the number of steps so that the completeness error is below a stated tolerance", "Abandon IG for saliency", "Use a random baseline"],
       "Completeness can be checked for every explanation; the number of steps should be set so that the error stays below a tolerance written in the documentation.",
       "The error falls quickly with more steps, so it can be controlled; saliency does not satisfy completeness at all."),
  ("Pașii Riemann", f"Eroarea mediană de completitudine a IG a fost {R('ig.comp4_pct',1)} % cu 4 pași și {R('ig.comp64_pct')} % cu 64 de pași. Ce ar trebui să facă un sistem în producție?",
       ["Să folosească 4 pași: eroarea nu contează", "Să aleagă numărul de pași astfel încît eroarea de completitudine să fie sub o toleranță stabilită", "Să renunțe la IG în favoarea salienței", "Să folosească o referință aleatoare"],
       "Completitudinea poate fi verificată pentru fiecare explicație; numărul de pași se alege astfel încît eroarea să rămînă sub o toleranță trecută în documentație.",
       "Eroarea scade repede cu numărul de pași, deci poate fi controlată; saliența nu respectă deloc completitudinea."))
q(2, ("Baseline", "What question does an integrated-gradients explanation answer when the baseline is the mean legitimate transaction?",
       ["Why is the model accurate?", "Which features does the model need to stay accurate?", "Why is this transaction not a normal payment?", "Which features are causes of fraud?"],
       "Attributions explain $f(\\mathbf x)-f(\\mathbf x')$; with the mean legitimate transaction as $\\mathbf x'$ they explain the departure from a normal payment.",
       "The baseline fixes the reference of the comparison; attributions are not causal statements and do not measure accuracy."),
  ("Referința", "La ce întrebare răspunde o explicație prin gradienți integrați cînd referința este tranzacția legitimă medie?",
       ["De ce este modelul precis?", "De ce variabile are nevoie modelul ca să rămînă precis?", "De ce nu este această tranzacție o plată normală?", "Ce variabile cauzează frauda?"],
       "Atribuirile explică $f(\\mathbf x)-f(\\mathbf x')$; cu tranzacția legitimă medie ca $\\mathbf x'$, ele explică abaterea de la o plată normală.",
       "Referința fixează termenul de comparație; atribuirile nu sînt afirmații cauzale și nu măsoară acuratețea."))
q(0, ("SmoothGrad", "What does SmoothGrad change compared with plain saliency?",
       ["It averages gradients over noisy copies of the input", "It integrates along a path from a baseline", "It fits a local linear model", "It retrains the network with noise"],
       "SmoothGrad computes $\\frac1n\\sum_k\\partial f(\\mathbf x+\\varepsilon_k)/\\partial x_j$, which removes the jitter of piecewise-linear networks.",
       "Path integration is IG and local linear models are LIME; SmoothGrad does not retrain anything."),
  ("SmoothGrad", "Ce schimbă SmoothGrad față de saliența simplă?",
       ["Mediază gradienții pe copii perturbate ale intrării", "Integrează pe un drum pornind de la o referință", "Estimează un model liniar local", "Reantrenează rețeaua cu zgomot"],
       "SmoothGrad calculează $\\frac1n\\sum_k\\partial f(\\mathbf x+\\varepsilon_k)/\\partial x_j$, ceea ce elimină oscilațiile rețelelor liniare pe porțiuni.",
       "Integrarea pe drum este IG, iar modelele liniare locale sînt LIME; SmoothGrad nu reantrenează nimic."))
q(3, ("Agreement", f"The Spearman correlation of global importances was {E('agree.ig_shap')} between IG and SHAP but {E('agree.perm_shap')} between permutation importance and SHAP. Why can permutation differ?",
       ["Permutation importance is computed on a different model", "Permutation importance is a local method", "SHAP is not defined for neural networks", "Permutation measures how much accuracy depends on a feature, attributions how much the score uses it"],
       "Permutation importance (Breiman, 2001) is the loss of accuracy when a feature is scrambled; attributions describe the score of the frauds. The two answer different questions.",
       "Both methods use the same model; one measures dependence of accuracy, the other contributions to the output."),
  ("Concordanța", f"Corelația Spearman a importanțelor globale a fost {R('agree.ig_shap')} între IG și SHAP, dar {R('agree.perm_shap')} între importanța prin permutare și SHAP. De ce poate diferi permutarea?",
       ["Importanța prin permutare se calculează pe alt model", "Importanța prin permutare este o metodă locală", "SHAP nu este definit pentru rețele neuronale", "Permutarea măsoară cît depinde acuratețea de o variabilă, atribuirile cît folosește scorul variabila"],
       "Importanța prin permutare (Breiman, 2001) este pierderea de acuratețe cînd o variabilă este amestecată; atribuirile descriu scorul fraudelor. Cele două răspund la întrebări diferite.",
       "Ambele metode folosesc același model; una măsoară dependența acurateței, cealaltă contribuțiile la ieșire."))
q(1, ("Faithfulness", f"In the deletion test the AOPC was {E('del.aopc_ig',1)} for IG, {E('del.aopc_sal',1)} for saliency and {E('del.aopc_rand',1)} for a random order. What does a high AOPC mean?",
       ["The explanation is stable across seeds", "Removing the features the explanation ranks first makes the score fall fast: the explanation is faithful", "The model is accurate", "The explanation agrees with LIME"],
       "AOPC averages the fall of the score as the top-ranked features are reset; a faithful ranking produces a fast fall, a random one almost none.",
       "Faithfulness is about the model's response to removing features, not about stability, accuracy or agreement."),
  ("Fidelitatea", f"În testul de eliminare, AOPC a fost {R('del.aopc_ig',1)} pentru IG, {R('del.aopc_sal',1)} pentru saliență și {R('del.aopc_rand',1)} pentru o ordine aleatoare. Ce înseamnă un AOPC mare?",
       ["Explicația este stabilă la schimbarea valorii inițiale", "Eliminarea variabilelor clasate primele de explicație face scorul să scadă repede: explicația este fidelă", "Modelul este precis", "Explicația este de acord cu LIME"],
       "AOPC mediază scăderea scorului pe măsură ce variabilele de top sînt readuse la referință; o ordine fidelă produce o scădere rapidă, una aleatoare aproape niciuna.",
       "Fidelitatea privește răspunsul modelului la eliminarea variabilelor, nu stabilitatea, acuratețea sau concordanța."))
q(2, ("Rashomon effect", f"Ten MLPs differing only in the seed have test AUC between {E('stab.auc_range_mlp_lo',3)} and {E('stab.auc_range_mlp_hi',3)}, but the mean Kendall $\\tau$ of their SHAP importances is {E('stab.tau_mlp')}. What is this called?",
       ["Overfitting", "Data leakage", "The Rashomon effect: equally accurate models with different explanations", "Concept drift"],
       "Many models fit the data almost equally well and tell different stories; explanation stability must be tested across seeds.",
       "Nothing leaked and the data did not change; only the seed did, and accuracy stayed the same."),
  ("Efectul Rashomon", f"Zece MLP-uri care diferă doar prin valoarea inițială au AUC de test între {R('stab.auc_range_mlp_lo',3)} și {R('stab.auc_range_mlp_hi',3)}, dar tau Kendall mediu al importanțelor SHAP este {R('stab.tau_mlp')}. Cum se numește fenomenul?",
       ["Supraînvățare", "Scurgere de informație", "Efectul Rashomon: modele la fel de precise, cu explicații diferite", "Deplasarea conceptului"],
       "Multe modele se potrivesc aproape la fel de bine datelor și spun povești diferite; stabilitatea explicațiilor trebuie testată pe mai multe valori inițiale.",
       "Nu a existat scurgere de informație și datele nu s-au schimbat; s-a schimbat doar valoarea inițială, iar acuratețea a rămas aceeași."))
q(0, ("Trees vs network", f"The mean Kendall $\\tau$ across seeds was {E('stab.tau_xgb')} for XGBoost and {E('stab.tau_mlp')} for the MLP. What is the practical conclusion?",
       ["Here the boosted trees give the more stable global explanation; the MLP's explanation should be reported with its seed spread", "Neural networks can never be explained", "XGBoost is always more accurate", "Kendall's $\\tau$ is not a valid measure"],
       "On these data the ranking of XGBoost barely moves across seeds while the MLP's moves more: report the spread, or average over seeds.",
       "The result concerns stability on these data, not a general law about accuracy or explainability."),
  ("Arbori față de rețea", f"Tau Kendall mediu între valorile inițiale a fost {R('stab.tau_xgb')} pentru XGBoost și {R('stab.tau_mlp')} pentru MLP. Care este concluzia practică?",
       ["Aici arborii amplificați dau explicația globală mai stabilă; explicația MLP trebuie raportată cu dispersia ei între valori inițiale", "Rețelele neuronale nu pot fi explicate niciodată", "XGBoost este mereu mai precis", "Tau Kendall nu este o măsură validă"],
       "Pe aceste date, clasamentul XGBoost abia se mișcă de la o valoare inițială la alta, iar cel al MLP se mișcă mai mult: raportați dispersia sau mediați pe valori inițiale.",
       "Rezultatul privește stabilitatea pe aceste date, nu o lege generală despre acuratețe sau explicabilitate."))
q(3, ("Drift of explanations", f"Between held-out frauds of day 1 and frauds of day 2, the Spearman correlation of IG importances was {E('shift.rho')} and the JS divergence {E('shift.js',3)}. What should be monitored in production?",
       ["Only the AUC", "Only the number of frauds", "Nothing: one day of stability proves stability forever", "The distribution of attributions over time, next to the score distribution"],
       "A shift in the reasons for alerts can precede a fall in accuracy; monitoring attributions monthly catches it early.",
       "Stability over one day does not guarantee stability over months; explanations are monitored like scores."),
  ("Deplasarea explicațiilor", f"Între fraudele rezervate din ziua 1 și fraudele din ziua 2, corelația Spearman a importanțelor IG a fost {R('shift.rho')}, iar divergența JS {R('shift.js',3)}. Ce trebuie monitorizat în producție?",
       ["Doar AUC", "Doar numărul de fraude", "Nimic: o zi de stabilitate dovedește stabilitatea pentru totdeauna", "Distribuția atribuirilor în timp, alături de distribuția scorurilor"],
       "O deplasare a motivelor alertelor poate preceda o scădere a acurateței; monitorizarea lunară a atribuirilor o surprinde devreme.",
       "Stabilitatea pe o zi nu garantează stabilitatea pe luni; explicațiile se monitorizează ca scorurile."))
q(1, ("Local robustness", f"The median local sensitivity of saliency was {E('rob.saliency',3)}, lower than IG's {E('rob.ig',3)}. Why is saliency so stable to small noise?",
       ["Because saliency is the most faithful method", "Because the gradient of a ReLU network is constant within a linear region", "Because the noise was zero", "Because saliency averages over many samples"],
       "Small perturbations usually stay in the same linear region, where the gradient does not change; across region boundaries saliency jumps.",
       "The deletion test showed saliency to be less faithful; its local stability comes from the piecewise-linear structure, not from averaging."),
  ("Robustețea locală", f"Sensibilitatea locală mediană a salienței a fost {R('rob.saliency',3)}, sub cea a IG, {R('rob.ig',3)}. De ce este saliența atît de stabilă la zgomot mic?",
       ["Pentru că saliența este cea mai fidelă metodă", "Pentru că gradientul unei rețele ReLU este constant într-o regiune liniară", "Pentru că zgomotul a fost zero", "Pentru că saliența mediază pe multe eșantioane"],
       "Perturbările mici rămîn de obicei în aceeași regiune liniară, unde gradientul nu se schimbă; la granița regiunilor saliența sare.",
       "Testul de eliminare a arătat că saliența este mai puțin fidelă; stabilitatea ei locală vine din structura liniară pe porțiuni, nu din mediere."))
q(2, ("AI Act", "Under the EU AI Act, how is creditworthiness scoring of natural persons classified?",
       ["As prohibited", "As minimal risk, without obligations", "As high-risk (Annex III), with transparency and documentation obligations", "As outside the scope of the act"],
       "Annex III lists credit scoring of natural persons among high-risk systems; article 86 gives affected persons a right to an explanation.",
       "Credit scoring is neither prohibited nor exempt; the act places it in the high-risk category."),
  ("Regulamentul AI", "Cum este încadrată evaluarea bonității persoanelor fizice în regulamentul UE privind inteligența artificială?",
       ["Ca practică interzisă", "Ca risc minim, fără obligații", "Ca risc ridicat (anexa III), cu obligații de transparență și documentare", "În afara domeniului de aplicare al regulamentului"],
       "Anexa III include evaluarea bonității persoanelor fizice printre sistemele cu risc ridicat; articolul 86 dă persoanelor afectate dreptul la o explicație.",
       "Evaluarea creditului nu este nici interzisă, nici exceptată; regulamentul o plasează în categoria cu risc ridicat."))
q(0, ("Explanation is not causation", "A SHAP analysis shows that the job category ``Other'' raises the predicted risk. What can the bank conclude?",
       ["That the model uses this category to raise the score, not that the job causes default", "That changing job would lower the applicant's true risk", "That the variable is protected by law", "That the model is unfair"],
       "Attributions describe how the model uses its inputs; causal claims need a causal design. Fairness needs a separate analysis of protected attributes and proxies.",
       "SHAP values are statements about the model, not about the world, and they do not by themselves establish unfairness."),
  ("Explicația nu este cauzalitate", "O analiză SHAP arată că ocupația „Other” crește riscul prognozat. Ce poate concluziona banca?",
       ["Că modelul folosește această categorie pentru a crește scorul, nu că ocupația cauzează neplata", "Că schimbarea locului de muncă ar reduce riscul real al solicitantului", "Că variabila este protejată prin lege", "Că modelul este inechitabil"],
       "Atribuirile descriu cum își folosește modelul intrările; afirmațiile cauzale cer un design cauzal. Echitatea cere o analiză separată a variabilelor protejate și a substitutelor lor.",
       "Valorile SHAP sînt afirmații despre model, nu despre lume, și nu dovedesc singure inechitatea."))
q(3, ("Interpretable alternative", f"The logistic scorecard reaches a test AUC of {E('perf.lr_auc',3)}, the MLP {E('perf.mlp_auc',3)}. What does Rudin (2019) recommend in such a high-stakes setting?",
       ["Always use the most accurate model", "Never use logistic regression", "Use LIME to replace the scorecard", "Weigh whether the interpretable model is accurate enough before explaining a black box"],
       "Rudin argues that interpretable models should be tried first in high-stakes decisions; the accuracy gap is the price of transparency to be judged.",
       "The point is a trade-off: transparency by design against a gain in accuracy, decided explicitly."),
  ("Alternativa interpretabilă", f"Grila logistică atinge un AUC de test de {R('perf.lr_auc',3)}, iar MLP-ul {R('perf.mlp_auc',3)}. Ce recomandă Rudin (2019) într-o asemenea decizie cu miză mare?",
       ["Folosiți mereu modelul cel mai precis", "Nu folosiți niciodată regresia logistică", "Folosiți LIME în locul grilei de scor", "Cîntăriți dacă modelul interpretabil este suficient de precis înainte de a explica o cutie neagră"],
       "Rudin susține că, în deciziile cu miză mare, modelele interpretabile trebuie încercate primele; diferența de acuratețe este prețul transparenței și trebuie judecată.",
       "Ideea este un compromis: transparență prin construcție față de un cîștig de acuratețe, decis explicit."))
q(1, ("PR-AUC baseline", f"In the HMEQ test set the default rate is {E('perf.base_pr',3)}. What PR-AUC does a random score obtain?",
       ["0.5", f"About {E('perf.base_pr',3)}, the default rate", "1.0", "Zero"],
       "The precision of a random score equals the share of positives at every recall, so its PR-AUC is the base rate.",
       "Unlike the ROC AUC, whose random value is 0.5, the PR-AUC of a random score depends on the share of positives."),
  ("Reperul PR-AUC", f"În setul de test HMEQ rata de neplată este {R('perf.base_pr',3)}. Ce PR-AUC obține un scor aleator?",
       ["0,5", f"Aproximativ {R('perf.base_pr',3)}, rata de neplată", "1,0", "Zero"],
       "Precizia unui scor aleator este egală cu ponderea pozitivelor la orice sensibilitate, deci PR-AUC-ul său este rata de bază.",
       "Spre deosebire de AUC ROC, a cărui valoare aleatoare este 0,5, PR-AUC-ul unui scor aleator depinde de ponderea pozitivelor."))
q(2, ("Fraud evaluation", f"Why is the fraud model evaluated with PR-AUC ({E('fraud.mlp_prauc',3)}) rather than accuracy?",
       ["Because accuracy cannot be computed for neural networks", "Because PR-AUC is always higher", f"Because frauds are about {E('fraud.rate_full_day2_pct',2)} % of transactions, so a model that never flags fraud is almost perfectly accurate", "Because the data are images"],
       "With such imbalance accuracy is dominated by the legitimate class; PR-AUC focuses on how well the frauds are ranked.",
       "Accuracy is computable, but it rewards predicting the majority class; the imbalance is the reason."),
  ("Evaluarea fraudei", f"De ce se evaluează modelul de fraudă prin PR-AUC ({R('fraud.mlp_prauc',3)}) și nu prin acuratețe?",
       ["Pentru că acuratețea nu se poate calcula la rețele neuronale", "Pentru că PR-AUC este întotdeauna mai mare", f"Pentru că fraudele sînt aproximativ {R('fraud.rate_full_day2_pct',2)} % din tranzacții, deci un model care nu semnalează nicio fraudă este aproape perfect precis", "Pentru că datele sînt imagini"],
       "Cu un asemenea dezechilibru, acuratețea este dominată de clasa legitimă; PR-AUC arată cît de bine sînt ordonate fraudele.",
       "Acuratețea se poate calcula, dar recompensează prognozarea clasei majoritare; dezechilibrul este motivul."))
q(0, ("Temporal split", "Why is the fraud network trained on day 1 and tested on day 2 rather than on a random split?",
       ["Because in production the model always scores future transactions, so the test must come after the training data", "Because random splits are not allowed by law", "Because day 2 has more frauds", "Because the PCA components change every hour"],
       "A forward-time split mimics deployment; a random split mixes past and future and can overstate performance and the stability of explanations.",
       "The choice follows how the model is used, not legal rules or the number of frauds."),
  ("Împărțirea în timp", "De ce este rețeaua de fraudă antrenată pe ziua 1 și testată pe ziua 2, nu pe o împărțire aleatoare?",
       ["Pentru că în producție modelul evaluează mereu tranzacții viitoare, deci testul trebuie să vină după datele de antrenare", "Pentru că împărțirile aleatoare sînt interzise prin lege", "Pentru că ziua 2 are mai multe fraude", "Pentru că componentele PCA se schimbă la fiecare oră"],
       "O împărțire înainte în timp imită utilizarea reală; o împărțire aleatoare amestecă trecutul cu viitorul și poate supraevalua performanța și stabilitatea explicațiilor.",
       "Alegerea urmează modul de folosire a modelului, nu reguli legale sau numărul de fraude."))
q(3, ("Group research", "In the study of Diaconu and Pele (2026) on graph-augmented fraud detection, under which two validation schemes are SHAP contributions compared?",
       ["Training and test accuracy", "Bootstrap and jackknife", "Five-fold and ten-fold cross-validation", "Random and forward-time validation"],
       "The paper compares SHAP stability under random and forward-time validation, with a Gaussian control matched to the graph features.",
       "The comparison concerns how the data are split in time, which is what changes between deployment and a random benchmark."),
  ("Cercetarea grupului", "În studiul lui Diaconu și Pele (2026) despre detectarea fraudei cu informație din graf, sub ce două scheme de validare se compară contribuțiile SHAP?",
       ["Acuratețea de antrenare și cea de test", "Bootstrap și jackknife", "Validarea încrucișată cu cinci și cu zece părți", "Validarea aleatoare și validarea înainte în timp"],
       "Lucrarea compară stabilitatea SHAP sub validare aleatoare și validare înainte în timp, cu un control gaussian potrivit variabilelor din graf.",
       "Comparația privește modul de împărțire a datelor în timp, ceea ce diferă între utilizarea reală și un reper aleator."))
q(1, ("Taxonomy", "Which method is both model-specific and exact for its model class?",
       ["Kernel SHAP", "TreeSHAP for tree ensembles", "LIME", "Partial dependence"],
       "TreeSHAP uses the structure of trees to compute exact Shapley values; the other three only call the model and are approximate or averaged.",
       "Kernel SHAP, LIME and partial dependence are model-agnostic; only TreeSHAP relies on a specific model class."),
  ("Clasificarea metodelor", "Care metodă este specifică unui tip de model și exactă pentru acesta?",
       ["Kernel SHAP", "TreeSHAP pentru ansambluri de arbori", "LIME", "Dependența parțială"],
       "TreeSHAP folosește structura arborilor pentru valori Shapley exacte; celelalte trei doar apelează modelul și sînt aproximative sau mediate.",
       "Kernel SHAP, LIME și dependența parțială sînt independente de model; doar TreeSHAP se bazează pe un anumit tip de model."))
q(2, ("Documenting explanations", "Which settings must be versioned with a model so that its explanations can be reproduced?",
       ["Only the network weights", "Only the test AUC", "Background sample, baseline, seed, number of samples and kernel width", "The colour map of the charts"],
       "Each of these settings changes the explanation; without them an auditor cannot reproduce what the customer was told.",
       "The weights are needed but not sufficient: the explainer has its own settings, and each one moves the result."),
  ("Documentarea explicațiilor", "Ce setări trebuie versionate împreună cu un model pentru ca explicațiile lui să poată fi reproduse?",
       ["Doar ponderile rețelei", "Doar AUC-ul de test", "Eșantionul de referință, referința, valoarea inițială, numărul de eșantioane și lățimea nucleului", "Paleta de culori a graficelor"],
       "Fiecare dintre aceste setări schimbă explicația; fără ele, un auditor nu poate reproduce ce i s-a spus clientului.",
       "Ponderile sînt necesare, dar nu suficiente: metoda de explicare are propriile setări, iar fiecare mută rezultatul."))
q(0, ("Concept explanations", "What does TCAV require that the anonymised fraud data cannot provide?",
       ["Labelled examples of a human concept, such as ``night-time transaction''", "A differentiable network", "A background sample", "A decision threshold"],
       "TCAV learns a concept direction from labelled examples; the PCA components V1–V28 carry no such labels.",
       "The network is differentiable and the other items are available; the missing ingredient is the concept labels."),
  ("Explicații pe concepte", "De ce are nevoie TCAV și nu pot oferi datele anonimizate de fraudă?",
       ["De exemple etichetate ale unui concept uman, de exemplu „tranzacție de noapte”", "De o rețea diferențiabilă", "De un eșantion de referință", "De un prag de decizie"],
       "TCAV învață direcția unui concept din exemple etichetate; componentele PCA V1–V28 nu au asemenea etichete.",
       "Rețeaua este diferențiabilă, iar celelalte elemente sînt disponibile; ingredientul care lipsește sînt etichetele conceptelor."))
q(1, ("Fraud explanation for one case", "For the median-score fraud, SHAP and IG name V14 first while saliency names V12. Why?",
       ["Because saliency uses a different model", "Because saliency ignores how far the transaction is from the baseline, while SHAP and IG weight by that distance", "Because V12 and V14 are the same variable", "Because SHAP is random"],
       "Saliency is a pure gradient; SHAP and IG multiply sensitivity by the deviation from the reference, so a large deviation can outrank a large gradient.",
       "All methods explain the same network; their definitions differ in whether the size of the deviation counts."),
  ("Explicarea unei fraude", "Pentru frauda cu scorul median, SHAP și IG pun V14 pe primul loc, iar saliența pune V12. De ce?",
       ["Pentru că saliența folosește alt model", "Pentru că saliența ignoră cît de departe este tranzacția de referință, în timp ce SHAP și IG ponderează cu această distanță", "Pentru că V12 și V14 sînt aceeași variabilă", "Pentru că SHAP este aleator"],
       "Saliența este un gradient pur; SHAP și IG înmulțesc sensibilitatea cu abaterea față de referință, deci o abatere mare poate depăși un gradient mare.",
       "Toate metodele explică aceeași rețea; definițiile diferă prin faptul că mărimea abaterii contează sau nu."))

assert len(Q) >= 30, len(Q)
out = ("// ============================================================\n"
       "// Quiz bank for lecture id 'l11': Explainable AI: SHAP, LIME, gradient methods, stability (EN + RO)\n"
       "// GENERATED by lectures/L11/make_quiz_L11.py from L11_numbers.json; do not edit by hand.\n"
       "// correct = index (0-3) of the right option in the original order; incorrectExplanation never names a letter.\n"
       "// ============================================================\n"
       "window.NN_DATA.quizzes['l11'] = " + json.dumps({"draw": 20, "questions": Q}, ensure_ascii=False, indent=1) + ";\n")
with open(os.path.join(ROOT, "assets", "quizzes", "l11.js"), "w", encoding="utf-8") as fh:
    fh.write(out)
from collections import Counter  # noqa: E402
print("l11.js:", len(Q), "questions; correct index counts", dict(Counter(x["correct"] for x in Q)))
