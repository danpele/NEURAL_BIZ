"""
make_web_L1.py -- web version of Lecture 1: writes lectures/L1/web_en.md and web_ro.md (numbers from L1_numbers.json, frame hashes
from the decks) and builds web/L1.html, web/L1_ro.html with lectures/common/build_web_lecture.py.
Run (after build_L1.py and make_quantlets_L1.py):  python3 lectures/L1/make_web_L1.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from webgen import write_pages  # noqa: E402

B = dict
PAGE = dict(
    intro=[
        ("A bank decides every day whom to lend to, a card issuer which payment to block, a trader how much risk to take. Deep learning, the "
         "branch of artificial intelligence (AI) that fits long chains of simple functions to data, now sits behind many of these decisions. "
         "This first lecture starts from the models every analyst already knows, linear and logistic regression, and shows that they are "
         "neural networks with a single neuron. From there we build Rosenblatt's perceptron, see where it fails, and arrive at the "
         "multilayer perceptron (MLP), the network behind every later lecture.",
         "O bancă decide în fiecare zi cui acordă credite, un emitent de carduri ce plăți blochează, un trader cît risc își asumă. Deep learning, "
         "ramura inteligenței artificiale (AI) care estimează din date lanțuri lungi de funcții simple, stă acum în spatele multora dintre aceste "
         "decizii. Primul curs pornește de la modelele pe care orice analist le cunoaște deja, regresia liniară și regresia logistică, și arată că "
         "ele sînt rețele neuronale cu un singur neuron. De aici construim perceptronul lui Rosenblatt, vedem unde eșuează și ajungem la "
         "perceptronul multistrat (MLP), rețeaua din spatele tuturor cursurilor următoare."),
        ("Every number on this page is computed on real, public credit data: @{data.german_n|,d} applicants of the German credit data, "
         "@{data.hmeq_n|,d} home equity loans (HMEQ) and, for one picture of function approximation, @{data.california_n|,d} Californian "
         "census block groups. The case study compares logistic regression, an MLP and gradient boosting with the same protocol, several random "
         "seeds, intervals and the cost of errors, because a model choice in a bank is a decision under uncertainty, not a contest of single numbers.",
         "Toate cifrele de pe această pagină sînt calculate pe date de credit reale și publice: @{data.german_n|,d} de solicitanți din setul "
         "German credit, @{data.hmeq_n|,d} de credite garantate cu locuința (HMEQ, home equity) și, pentru o imagine a aproximării funcțiilor, "
         "@{data.california_n|,d} de grupuri de locuințe din recensămîntul Californiei. Studiul de caz compară regresia logistică, un MLP și "
         "gradient boosting după același protocol, cu mai multe inițializări aleatoare, intervale și costul erorilor, pentru că alegerea unui "
         "model într-o bancă este o decizie în condiții de incertitudine, nu o competiție între cifre izolate."),
    ],
    goals=[
        ("**Place** deep learning among the tools of a business analyst: where it creates value and where a linear model is enough.",
         "**Situați** deep learning printre instrumentele unui analist de afaceri: unde creează valoare și unde un model liniar este suficient."),
        ("**Write** linear and logistic regression as one-neuron networks, with their losses as likelihoods.",
         "**Scrieți** regresia liniară și regresia logistică drept rețele cu un neuron, cu pierderile lor ca verosimilități."),
        ("**Explain** the perceptron, its learning rule, its convergence theorem and why it cannot represent the exclusive or (XOR).",
         "**Explicați** perceptronul, regula lui de învățare, teorema de convergență și de ce nu poate reprezenta sau-ul exclusiv (XOR)."),
        ("**Build** an MLP: forward pass, activation functions, number of parameters, universal approximation.",
         "**Construiți** un MLP: propagarea înainte, funcțiile de activare, numărul de parametri, aproximarea universală."),
        ("**Compare** an MLP with logistic regression and gradient boosting on real loans, with seeds, intervals, calibration and costs.",
         "**Comparați** un MLP cu regresia logistică și cu gradient boosting pe credite reale, cu inițializări multiple, intervale, calibrare și costuri."),
    ],
    sections=[
        B(title=("Deep learning for business", "Deep learning pentru afaceri"),
          intro=("Why deep learning matters now, where it creates value in finance and business, where a simpler model is enough, and the data we use.",
                 "De ce contează deep learning acum, unde creează valoare în finanțe și afaceri, unde un model mai simplu este suficient și ce date folosim."),
          blocks=[
              B(frame="Deep learning learns the features",
                text=("A deep network is a composition of layers, $f(\\mathbf x;\\boldsymbol\\theta)=f^{(L)}(f^{(L-1)}(\\cdots f^{(1)}(\\mathbf x)))$, whose "
                      "parameters $\\boldsymbol\\theta$ are all chosen to minimise the average loss on the training data. What changed in the last fifteen "
                      "years is not the idea but its scale: large labelled datasets, graphics processing units (GPU) and software that differentiates "
                      "automatically, such as PyTorch. The network no longer needs features designed by an analyst; it learns its own "
                      "representations from raw inputs (LeCun, Bengio and Hinton, 2015).",
                      "O rețea adîncă este o compunere de straturi, $f(\\mathbf x;\\boldsymbol\\theta)=f^{(L)}(f^{(L-1)}(\\cdots f^{(1)}(\\mathbf x)))$, ai "
                      "cărei parametri $\\boldsymbol\\theta$ sînt aleși astfel încît să minimizeze pierderea medie pe datele de antrenare. În ultimii cincisprezece "
                      "ani nu s-a schimbat ideea, ci scara ei: seturi mari de date etichetate, procesoare grafice (GPU) și software care derivează automat, "
                      "precum PyTorch. Rețeaua nu mai are nevoie de variabile construite de un analist; își învață singură reprezentările din datele "
                      "brute (LeCun, Bengio și Hinton, 2015)."),
                practice=("ask what the network would learn that a scorecard analyst cannot write down by hand.",
                          "întrebați ce ar învăța rețeaua în plus față de ce poate scrie manual un analist de scoring.")),
              B(frame="Where deep learning creates value",
                text=("The course is organised by decision and data type. Credit and payments are tabular problems (Lectures 1 to 3); market and "
                      "energy forecasting are sequences (Lectures 5 and 13); news, filings and central-bank speeches are text (Lectures 6, 7 and 12); "
                      "payment networks are graphs (Lecture 9). The rule is always the same: start from the decision and its cost, then choose the "
                      "network that matches the data.",
                      "Cursul este organizat după decizie și tipul datelor. Creditul și plățile sînt probleme tabelare (cursurile 1--3); prognoza "
                      "piețelor și a energiei lucrează cu secvențe (cursurile 5 și 13); știrile, rapoartele și discursurile băncilor centrale sînt text "
                      "(cursurile 6, 7 și 12); rețelele de plăți sînt grafuri (cursul 9). Regula este mereu aceeași: porniți de la decizie și de la "
                      "costul ei, apoi alegeți rețeaua potrivită datelor.")),
              B(frame="For small tabular data",
                text=("Deep learning is not automatically better. In the large credit-scoring benchmark of Lessmann et al. (2015), logistic "
                      "regression stays a solid reference and ensembles lead; on typical medium-sized tabular data, tree ensembles still beat deep "
                      "networks (Grinsztajn, Oyallon and Varoquaux, 2022). Credit decisions must also be explained to the applicant. This lecture "
                      "therefore reports every network next to a linear and a boosting benchmark.",
                      "Deep learning nu este automat mai bun. În comparația amplă a metodelor de scoring de credit realizată de Lessmann et al. (2015), "
                      "regresia logistică rămîne o referință solidă, iar ansamblurile de modele conduc; pe date tabelare obișnuite, de mărime medie, "
                      "ansamblurile de arbori încă depășesc rețelele adînci (Grinsztajn, Oyallon și Varoquaux, 2022). Deciziile de creditare trebuie, "
                      "în plus, explicate solicitantului. De aceea, în acest curs fiecare rețea este raportată alături de un model de referință liniar "
                      "și de unul de boosting.")),
              B(frame="Three real credit datasets",
                text=("German credit comes from the University of California, Irvine (UCI) Machine Learning Repository: @{data.german_n|,d} applicants, "
                      "@{data.german_vars} attributes, amounts in Deutsche Mark (DM) and a default rate of @{data.german_bad_pct|.1f} %. HMEQ "
                      "(Baesens, Rösch and Scheule, 2016) has @{data.hmeq_n|,d} loans, a default rate of @{data.hmeq_bad_pct|.1f} % and many missing "
                      "values, which we keep. California housing gives median house values in units of 100,000 US dollars (USD).",
                      "German credit provine din Machine Learning Repository al Universității California din Irvine (UCI): @{data.german_n|,d} de "
                      "solicitanți, @{data.german_vars} de atribute, sume în mărci germane (DM) și o rată de neplată de @{data.german_bad_pct|.1f} %. "
                      "HMEQ (Baesens, Rösch și Scheule, 2016) are @{data.hmeq_n|,d} de credite, o rată de neplată de @{data.hmeq_bad_pct|.1f} % și multe "
                      "valori lipsă, pe care le păstrăm. Datele California dau valoarea mediană a locuințelor în sute de mii de dolari americani (USD)."),
                practice=("check the source, the unit of observation and the event rate before any model.",
                          "verificați sursa, unitatea de observație și frecvența evenimentului înaintea oricărui model.")),
              B(frame="Default rates differ strongly",
                text=("The simplest risk model is the default rate per group, $\\widehat{\\mathrm{PD}}_g=\\frac1{n_g}\\sum_{i\\in g}y_i$, where the "
                      "probability of default (PD) of group $g$ is the share of its loans with $y_i=1$. In German credit, applicants with a negative "
                      "checking balance default in @{eda.chk.A11.pct|.1f} % of cases, those without a checking account in @{eda.chk.A14.pct|.1f} %. In "
                      "HMEQ, the rate climbs from @{eda.delinq.0.pct|.1f} % with no delinquent credit line to @{eda.delinq.3plus.pct|.1f} % with "
                      "three or more.",
                      "Cel mai simplu model de risc este rata de neplată pe grup, $\\widehat{\\mathrm{PD}}_g=\\frac1{n_g}\\sum_{i\\in g}y_i$, unde "
                      "probabilitatea de neplată (PD) a grupului $g$ este ponderea creditelor lui cu $y_i=1$. În German credit, solicitanții cu sold "
                      "negativ în contul curent intră în neplată în @{eda.chk.A11.pct|.1f} % din cazuri, cei fără cont curent în @{eda.chk.A14.pct|.1f} %. "
                      "În HMEQ, rata crește de la @{eda.delinq.0.pct|.1f} % fără linii de credit cu restanțe la @{eda.delinq.3plus.pct|.1f} % cu trei sau "
                      "mai multe."),
                chart="default_rates",
                cap=("Default rate by checking-account status (German credit) and by number of delinquent credit lines (HMEQ); dashed: sample average.",
                     "Rata de neplată după starea contului curent (German credit) și după numărul de linii de credit cu restanțe (HMEQ); linia punctată: media eșantionului."),
                practice=("a model must at least reproduce these group rates on the validation set.",
                          "un model trebuie să reproducă cel puțin aceste rate pe setul de validare.")),
              B(frame="Learning = minimising",
                text=("Training minimises the empirical risk $\\hat R_n(\\boldsymbol\\theta)=\\frac1n\\sum_i\\ell(y_i,f(\\mathbf x_i;\\boldsymbol\\theta))$, "
                      "but the bank pays the risk on future applicants, $R(\\boldsymbol\\theta)=\\mathbb E\\,\\ell(y,f(\\mathbf x;\\boldsymbol\\theta))$. We "
                      "therefore split the data in three: training ($\\boldsymbol\\theta$), validation (model choice and stopping) and test (reported "
                      "once). For HMEQ the stratified split gives @{hmeq.n_train|,d}, @{hmeq.n_val|,d} and @{hmeq.n_test|,d} loans; medians and "
                      "scaling are fitted on the training part only, so no information leaks from the test set.",
                      "Antrenarea minimizează riscul empiric $\\hat R_n(\\boldsymbol\\theta)=\\frac1n\\sum_i\\ell(y_i,f(\\mathbf x_i;\\boldsymbol\\theta))$, "
                      "dar banca plătește riscul pe solicitanții viitori, $R(\\boldsymbol\\theta)=\\mathbb E\\,\\ell(y,f(\\mathbf x;\\boldsymbol\\theta))$. "
                      "De aceea împărțim datele în trei: antrenare ($\\boldsymbol\\theta$), validare (alegerea modelului și oprirea) și testare (raportată "
                      "o singură dată). Pentru HMEQ, împărțirea stratificată dă @{hmeq.n_train|,d}, @{hmeq.n_val|,d} și @{hmeq.n_test|,d} de credite; "
                      "medianele și standardizarea se estimează doar pe partea de antrenare, astfel încît nicio informație nu scapă din setul de testare.")),
          ],
          end=("Deep learning creates most value on raw, high-dimensional inputs; on small engineered tables it must beat logistic regression and "
               "boosting under a fixed protocol, on data that stay real from the first table to the last test.",
               "Deep learning creează cea mai multă valoare pe date brute, de dimensiune mare; pe tabele mici, cu variabile construite, trebuie să "
               "depășească regresia logistică și boosting după un protocol fix, pe date care rămîn reale de la primul tabel pînă la ultimul test.")),
        B(title=("Linear models are one-neuron networks", "Modelele liniare sînt rețele cu un singur neuron"),
          intro=("Linear and logistic regression are a single neuron with an identity or a sigmoid output; their losses are likelihoods and their "
                 "gradients have the form that backpropagation will repeat in every layer.",
                 "Regresia liniară și regresia logistică sînt un singur neuron, cu ieșire identitate sau sigmoidă; pierderile lor sînt verosimilități, "
                 "iar gradienții lor au forma pe care propagarea înapoi o va repeta în fiecare strat."),
          blocks=[
              B(frame="Linear regression is one neuron",
                text=("A neuron without activation computes $\\hat y=\\mathbf w^\\top\\mathbf x+b$. With Gaussian errors, maximising the likelihood "
                      "is the same as minimising the mean squared error (MSE), $\\frac1n\\sum(y_i-\\mathbf w^\\top\\mathbf x_i-b)^2$, which has the "
                      "closed form $\\hat{\\mathbf w}=(\\mathbf X^\\top\\mathbf X)^{-1}\\mathbf X^\\top\\mathbf y$. Networks lose this closed form and are "
                      "trained by gradient descent (Lecture 2).",
                      "Un neuron fără activare calculează $\\hat y=\\mathbf w^\\top\\mathbf x+b$. Cu erori gaussiene, maximizarea verosimilității "
                      "este același lucru cu minimizarea erorii pătratice medii (MSE), $\\frac1n\\sum(y_i-\\mathbf w^\\top\\mathbf x_i-b)^2$, care are "
                      "soluția explicită $\\hat{\\mathbf w}=(\\mathbf X^\\top\\mathbf X)^{-1}\\mathbf X^\\top\\mathbf y$. Rețelele pierd această formulă "
                      "explicită și se antrenează prin coborîre pe gradient (cursul 2).")),
              B(frame="Logistic regression is one neuron",
                text=("With a sigmoid output, $p_i=\\sigma(\\mathbf w^\\top\\mathbf x_i+b)$, and the Bernoulli negative log-likelihood (cross-entropy) "
                      "as loss, the neuron is logistic regression:",
                      "Cu ieșire sigmoidă, $p_i=\\sigma(\\mathbf w^\\top\\mathbf x_i+b)$, și cu minus log-verosimilitatea Bernoulli (entropia "
                      "încrucișată) ca pierdere, neuronul este regresia logistică:"),
                after=("$$ \\mathcal L(\\mathbf w,b)=-\\frac1n\\sum_{i=1}^n\\big[y_i\\log p_i+(1-y_i)\\log(1-p_i)\\big],\\qquad "
                       "\\nabla_{\\mathbf w}\\mathcal L=\\frac1n\\sum_{i=1}^n(p_i-y_i)\\,\\mathbf x_i $$ {#eq:logit}\n\n"
                       "The gradient @eq:logit is the error times the input; the loss is convex, so there is a single minimum, and $e^{w_j}$ is the "
                       "odds ratio of input $j$.",
                       "$$ \\mathcal L(\\mathbf w,b)=-\\frac1n\\sum_{i=1}^n\\big[y_i\\log p_i+(1-y_i)\\log(1-p_i)\\big],\\qquad "
                       "\\nabla_{\\mathbf w}\\mathcal L=\\frac1n\\sum_{i=1}^n(p_i-y_i)\\,\\mathbf x_i $$ {#eq:logit}\n\n"
                       "Gradientul @eq:logit este eroarea înmulțită cu intrarea; pierderea este convexă, deci are un singur minim, iar $e^{w_j}$ este "
                       "raportul șanselor pentru intrarea $j$.")),
              B(frame="Delinquencies and a missing debt ratio",
                text=("With standardised inputs, the coefficients are changes in log-odds per standard deviation and can be compared. The odds ratios "
                      "are @{logit.or.miss_DEBTINC|.2f} for a missing debt-to-income ratio, @{logit.or.DELINQ|.2f} for more delinquent lines and "
                      "@{logit.or.CLAGE|.2f} for an older credit history; McFadden's pseudo-$R^2$ is @{logit.pseudo_r2|.3f}.",
                      "Cu intrări standardizate, coeficienții sînt variații ale log-șansei la o abatere standard și pot fi comparați. Rapoartele "
                      "șanselor sînt @{logit.or.miss_DEBTINC|.2f} pentru un raport datorii/venit lipsă, @{logit.or.DELINQ|.2f} pentru mai multe linii "
                      "cu restanțe și @{logit.or.CLAGE|.2f} pentru un istoric de credit mai vechi; pseudo-$R^2$ McFadden este @{logit.pseudo_r2|.3f}."),
                chart="logit_coef",
                cap=("Logistic regression on HMEQ: standardised coefficients with 95 % confidence intervals.",
                     "Regresia logistică pe HMEQ: coeficienți standardizați cu intervale de încredere de 95 %."),
                practice=("a sign against credit sense is a reason to inspect the data, not to drop the model.",
                          "un semn contrar intuiției de credit este un motiv să analizați datele, nu să renunțați la model.")),
              B(frame="The forward pass of one real applicant",
                text=("For one test applicant who later defaulted (a loan of @{one.loan|,d} USD, debt-to-income @{one.debtinc|.1f} %, "
                      "@{one.delinq} delinquent lines), the score is the sum of the contributions $w_jx_j$ plus the bias, $z=$ @{one.z|.2f}, and the "
                      "neuron outputs $p=\\sigma(z)=$ @{one.p_pct|.1f} %. The applicant would pass a 50 % cut-off; Section 5 shows why the cut-off "
                      "should be set from costs.",
                      "Pentru un solicitant din setul de testare care a intrat ulterior în neplată (un credit de @{one.loan|,d} USD, raport datorii/venit "
                      "@{one.debtinc|.1f} %, @{one.delinq} linii cu restanțe), scorul este suma contribuțiilor $w_jx_j$ plus termenul liber, "
                      "$z=$ @{one.z|.2f}, iar neuronul produce $p=\\sigma(z)=$ @{one.p_pct|.1f} %. Solicitantul ar trece de un prag de 50 %; secțiunea 5 "
                      "arată de ce pragul trebuie stabilit din costuri.")),
              B(frame="Missing values as predictors of default",
                text=("The debt-to-income ratio is missing for @{data.hmeq_debtinc_missing_pct|.1f} % of HMEQ loans, and these default in "
                      "@{data.hmeq_bad_debtinc_missing_pct|.1f} % of cases against @{data.hmeq_bad_debtinc_present_pct|.1f} % when it is reported. "
                      "Dropping incomplete rows would drop the riskiest applicants. We fill each numeric input with its training median and add an "
                      "indicator of missingness, which gives $d=$ @{hmeq.d} inputs.",
                      "Raportul datorii/venit lipsește pentru @{data.hmeq_debtinc_missing_pct|.1f} % dintre creditele HMEQ, iar acestea intră în neplată "
                      "în @{data.hmeq_bad_debtinc_missing_pct|.1f} % din cazuri, față de @{data.hmeq_bad_debtinc_present_pct|.1f} % cînd raportul este "
                      "declarat. Eliminarea rîndurilor incomplete ar elimina solicitanții cei mai riscanți. Completăm fiecare intrare numerică cu mediana "
                      "de antrenare și adăugăm un indicator de valoare lipsă, ceea ce dă $d=$ @{hmeq.d} de intrări."),
                practice=("never drop rows with missing values in credit data.", "nu eliminați rîndurile cu valori lipsă din datele de credit.")),
          ],
          end=("One neuron with the right output and loss is a classical statistical model; the network view adds nothing yet, but it fixes the "
               "notation and the gradient that every deeper model reuses.",
               "Un neuron cu ieșirea și pierderea potrivite este un model statistic clasic; perspectiva de rețea nu adaugă încă nimic, dar fixează "
               "notația și gradientul pe care le refolosește orice model mai adînc.")),
        B(title=("The perceptron", "Perceptronul"),
          intro=("Rosenblatt's perceptron learns a linear boundary from its mistakes. It converges when the classes are separable and fails "
                 "otherwise, which is the usual case in credit, and it cannot represent XOR.",
                 "Perceptronul lui Rosenblatt învață o frontieră liniară din propriile greșeli. Converge cînd clasele sînt separabile și eșuează în caz "
                 "contrar, care este situația obișnuită în credit, și nu poate reprezenta XOR."),
          blocks=[
              B(frame="From a logical neuron",
                text=("McCulloch and Pitts (1943) wrote the neuron as a threshold unit, $\\hat y=\\operatorname{step}(\\mathbf w^\\top\\mathbf x+b)$; "
                      "Rosenblatt (1958) made its weights learn from labelled examples. Minsky and Papert (1969) proved that one layer cannot represent "
                      "XOR, and the field recovered only when backpropagation trained several layers (Rumelhart, Hinton and Williams, 1986).",
                      "McCulloch și Pitts (1943) au scris neuronul ca unitate cu prag, $\\hat y=\\operatorname{step}(\\mathbf w^\\top\\mathbf x+b)$; "
                      "Rosenblatt (1958) a făcut ca ponderile lui să învețe din exemple etichetate. Minsky și Papert (1969) au demonstrat că un singur strat "
                      "nu poate reprezenta XOR, iar domeniul și-a revenit abia cînd propagarea înapoi a erorii a permis antrenarea mai multor straturi "
                      "(Rumelhart, Hinton și Williams, 1986).")),
              B(frame="The perceptron updates its weights only on misclassified examples",
                text=("If example $i$ is misclassified, $y_i(\\mathbf w^\\top\\mathbf x_i+b)\\le0$ with $y_i\\in\\{-1,+1\\}$, the weights move towards it: "
                      "$\\mathbf w\\leftarrow\\mathbf w+\\eta y_i\\mathbf x_i$. This is one step of stochastic gradient descent (SGD) on the perceptron "
                      "criterion. Correct examples give no gradient and the output is a class, not a probability, which a lender cannot price.",
                      "Dacă exemplul $i$ este clasificat greșit, $y_i(\\mathbf w^\\top\\mathbf x_i+b)\\le0$ cu $y_i\\in\\{-1,+1\\}$, ponderile se mută spre "
                      "el: $\\mathbf w\\leftarrow\\mathbf w+\\eta y_i\\mathbf x_i$. Este un pas de coborîre stohastică pe gradient (SGD) pe criteriul "
                      "perceptronului. Exemplele corecte nu dau gradient, iar ieșirea este o clasă, nu o probabilitate, pe care un creditor nu o poate evalua.")),
              B(frame="If the classes can be separated",
                text=("Novikoff (1962): if a unit vector $\\mathbf w^*$ separates the data with margin $\\gamma$ and all inputs lie within radius $R$, the "
                      "perceptron makes at most $(R/\\gamma)^2$ mistakes, whatever the number of examples or the dimension. Without a separating "
                      "hyperplane the theorem says nothing.",
                      "Novikoff (1962): dacă un vector unitar $\\mathbf w^*$ separă datele cu marginea $\\gamma$ și toate intrările se află în raza $R$, "
                      "perceptronul face cel mult $(R/\\gamma)^2$ greșeli, oricîte exemple și oricare ar fi dimensiunea. Fără un hiperplan separator, "
                      "teorema nu spune nimic.")),
              B(frame="The perceptron does not converge on non-separable credit data",
                text=("On the HMEQ training loans the classes overlap. Over @{perceptron.epochs} epochs the training error jumps between "
                      "@{perceptron.err_min_pct|.1f} % and @{perceptron.err_max_pct|.1f} % after @{perceptron.updates|,d} updates, while logistic "
                      "regression on the same inputs reaches @{perceptron.logit_err_pct|.1f} % and always predicting 'no default' "
                      "@{perceptron.majority_err_pct|.1f} %.",
                      "Pe creditele HMEQ de antrenare, clasele se suprapun. În @{perceptron.epochs} de epoci, eroarea de antrenare sare între "
                      "@{perceptron.err_min_pct|.1f} % și @{perceptron.err_max_pct|.1f} %, după @{perceptron.updates|,d} de actualizări, în timp ce "
                      "regresia logistică pe aceleași intrări ajunge la @{perceptron.logit_err_pct|.1f} %, iar regula „mereu fără neplată” la "
                      "@{perceptron.majority_err_pct|.1f} %."),
                chart="perceptron_errors",
                cap=("Training error of the perceptron after each epoch on HMEQ, against logistic regression and the majority rule.",
                     "Eroarea de antrenare a perceptronului după fiecare epocă pe HMEQ, comparată cu regresia logistică și cu regula majorității."),
                practice=("a smooth loss and a probability output fix both problems.", "o pierdere netedă și o ieșire probabilistă rezolvă ambele probleme.")),
              B(frame="The XOR problem and the hidden layer",
                text=("XOR is true when exactly one input is on. Any linear boundary classifies at most three of the four points correctly "
                      "(@{xor.logit_acc_pct} % here), while an MLP with two tanh hidden units reaches @{xor.mlp_acc_pct} %: the hidden layer maps the "
                      "inputs to a space where the classes become separable. Interactions such as 'high debt and short history' are of this kind. "
                      "This is a mechanism picture on the logical table, not a data result.",
                      "XOR este adevărat cînd exact o intrare este activă. Orice frontieră liniară clasifică corect cel mult trei dintre cele patru "
                      "puncte (aici @{xor.logit_acc_pct} %), în timp ce un MLP cu doi neuroni ascunși tanh ajunge la @{xor.mlp_acc_pct} %: stratul "
                      "ascuns duce intrările într-un spațiu în care clasele devin separabile. Interacțiunile de tipul „datorii mari și istoric scurt” "
                      "sînt de acest fel. Este o ilustrare a mecanismului pe tabelul logic, nu un rezultat pe date."),
                chart="xor", cap=("XOR: one linear boundary (left) and an MLP with two hidden units (right).",
                                  "XOR: o singură frontieră liniară (stînga) și un MLP cu doi neuroni ascunși (dreapta).")),
          ],
          end=("The perceptron is a linear classifier that learns only from mistakes; it converges on separable data and cycles on real, overlapping "
               "classes. Smooth losses, probability outputs and hidden layers remove these limits.",
               "Perceptronul este un clasificator liniar care învață doar din greșeli; converge pe date separabile și se rotește la nesfîrșit pe "
               "clase reale, care se suprapun. Pierderile netede, ieșirile probabiliste și straturile ascunse elimină aceste limite.")),
        B(title=("The multilayer perceptron", "Perceptronul multistrat"),
          intro=("Stacking affine maps and nonlinear activations gives the MLP: we write its forward pass, choose activations and outputs, count "
                 "parameters, and see what the universal approximation theorem does and does not promise.",
                 "Prin suprapunerea transformărilor afine și a activărilor neliniare obținem MLP: scriem propagarea înainte, alegem activările și "
                 "ieșirile, numărăm parametrii și vedem ce promite și ce nu promite teorema aproximării universale."),
          blocks=[
              B(frame="An MLP alternates affine maps",
                text=("Each hidden layer computes $\\mathbf h^{(l)}=a(\\mathbf W^{(l)}\\mathbf h^{(l-1)}+\\mathbf b^{(l)})$, starting from "
                      "$\\mathbf h^{(0)}=\\mathbf x$, and the output layer is a logistic regression on the last hidden features. Without the "
                      "activation $a$ the layers collapse into one linear map. Our credit network has $d=$ @{hmeq.d} inputs and two hidden layers of "
                      "64 rectified linear units (ReLU).",
                      "Fiecare strat ascuns calculează $\\mathbf h^{(l)}=a(\\mathbf W^{(l)}\\mathbf h^{(l-1)}+\\mathbf b^{(l)})$, pornind de la "
                      "$\\mathbf h^{(0)}=\\mathbf x$, iar stratul de ieșire este o regresie logistică pe ultimele variabile ascunse. Fără activarea $a$, "
                      "straturile se reduc la o singură transformare liniară. Rețeaua noastră de credit are $d=$ @{hmeq.d} de intrări și două straturi "
                      "ascunse a cîte 64 de unități liniare rectificate (ReLU).")),
              B(frame="Activation functions",
                text=("Sigmoid and tanh saturate: the derivative of the sigmoid is at most @{act.sigmoid_max_grad|.2f} and both derivatives vanish for "
                      "large inputs, which slows learning in deep networks (Lecture 2). ReLU, $\\max(0,z)$, has slope 1 for positive inputs and is the "
                      "default in hidden layers; the Gaussian error linear unit (GELU), $z\\Phi(z)$, is its smooth variant used in transformers.",
                      "Sigmoida și tanh se saturează: derivata sigmoidei este cel mult @{act.sigmoid_max_grad|.2f}, iar ambele derivate dispar pentru "
                      "intrări mari, ceea ce încetinește învățarea în rețelele adînci (cursul 2). ReLU, $\\max(0,z)$, are panta 1 pentru intrări pozitive "
                      "și este alegerea implicită în straturile ascunse; unitatea liniară cu eroare gaussiană (GELU), $z\\Phi(z)$, este varianta ei netedă, "
                      "folosită în arhitecturile transformer."),
                chart="activations", cap=("Activation functions (left) and their derivatives (right); analytic curves.",
                                          "Funcțiile de activare (stînga) și derivatele lor (dreapta); curbe analitice.")),
              B(frame="The output layer and the loss",
                text=("The output and the loss follow from the distribution of the target: identity and MSE for a real number, sigmoid and cross-entropy "
                      "for a binary event, softmax and categorical cross-entropy for $K$ classes, and the pinball loss for a quantile, such as the "
                      "value at risk (VaR) of Lecture 2. Each pair is a maximum-likelihood estimator.",
                      "Ieșirea și pierderea decurg din distribuția țintei: identitate și MSE pentru un număr real, sigmoidă și entropie încrucișată pentru "
                      "un eveniment binar, softmax și entropie încrucișată categorială pentru $K$ clase și pierderea pinball pentru o cuantilă, de "
                      "exemplu valoarea la risc (VaR) din cursul 2. Fiecare pereche este un estimator de verosimilitate maximă.")),
              B(frame="A small MLP already has hundreds",
                text=("The number of parameters is $\\sum_l(H_{l-1}+1)H_l$. On HMEQ the logistic model has @{hmeq.n_params_logit} parameters and "
                      "the MLP 64-64 has @{hmeq.n_params_6464|,d}, for @{hmeq.n_train|,d} training loans: enough to memorise the training defaults, "
                      "which is why every network keeps the weights of its best validation epoch.",
                      "Numărul de parametri este $\\sum_l(H_{l-1}+1)H_l$. Pe HMEQ, modelul logistic are @{hmeq.n_params_logit} de parametri, iar "
                      "MLP 64-64 are @{hmeq.n_params_6464|,d}, pentru @{hmeq.n_train|,d} de credite de antrenare: suficient pentru a memora neplățile din "
                      "antrenare, motiv pentru care fiecare rețea păstrează ponderile din epoca cea mai bună pe validare.")),
              B(frame="One hidden layer can approximate",
                text=("For any continuous function on a compact set and any tolerance, a one-hidden-layer network with enough units approximates it "
                      "(Cybenko, 1989; Hornik, 1991). The theorem does not say how many units are needed, whether gradient descent finds the weights, "
                      "or whether they generalise.",
                      "Pentru orice funcție continuă pe o mulțime compactă și orice toleranță, o rețea cu un singur strat ascuns și suficienți neuroni o "
                      "aproximează (Cybenko, 1989; Hornik, 1991). Teorema nu spune cîți neuroni sînt necesari, dacă coborîrea pe gradient găsește ponderile "
                      "sau dacă acestea generalizează.")),
              B(frame="More hidden units trace",
                text=("House values in California peak around Los Angeles and San Francisco. Fitting them on latitude alone, the held-out MSE falls "
                      "from @{uat.mse_lin|.3f} for a straight line to @{uat.mse_w1|.3f}, @{uat.mse_w3|.3f} and @{uat.mse_w50|.3f} with 1, 3 and 50 "
                      "sigmoid units. The @{uat.n_capped} block groups at the source cap of 500,000 USD are left out.",
                      "Valorile locuințelor din California au vîrfuri în jurul orașelor Los Angeles și San Francisco. Estimîndu-le doar din latitudine, "
                      "MSE pe datele păstrate deoparte scade de la @{uat.mse_lin|.3f} pentru o dreaptă la @{uat.mse_w1|.3f}, @{uat.mse_w3|.3f} și "
                      "@{uat.mse_w50|.3f} cu 1, 3 și 50 de neuroni sigmoizi. Cele @{uat.n_capped} de grupuri aflate la plafonul sursei, de 500 000 USD, "
                      "sînt excluse."),
                chart="uat_california", cap=("House values against latitude: bin means and fitted networks with 1, 3 and 50 hidden units.",
                                             "Valoarea locuințelor în funcție de latitudine: medii pe clase și rețele estimate cu 1, 3 și 50 de neuroni ascunși."),
                practice=("location effects are nonlinear: a linear term cannot price them.", "efectele de localizare sînt neliniare: un termen liniar nu le poate surprinde.")),
              B(frame="Depth and width of a network",
                text=("A ReLU network is piecewise linear, and the number of its linear regions can grow exponentially with depth for a fixed number "
                      "of units (Montúfar et al., 2014). On tabular credit data, two or three hidden layers are usually enough; depth pays off on "
                      "images, text and sequences.",
                      "O rețea ReLU este liniară pe porțiuni, iar numărul regiunilor ei liniare poate crește exponențial cu adîncimea, la un număr fix "
                      "de neuroni (Montúfar et al., 2014). Pe datele tabelare de credit, două sau trei straturi ascunse sînt de obicei suficiente; "
                      "adîncimea contează la imagini, text și secvențe.")),
              B(frame="An MLP bends the decision boundary",
                text=("With two HMEQ inputs, debt-to-income and the age of the oldest credit line, the logistic model draws straight probability "
                      "bands and the MLP curved regions. The area under the receiver operating characteristic (ROC) curve (AUC), the probability that "
                      "a random defaulter scores above a random good loan, is @{two.auc_logit|.3f} for the logistic model and @{two.auc_mlp|.3f} for "
                      "the MLP: flexibility does not help when the two inputs carry little extra signal.",
                      "Cu două intrări HMEQ, raportul datorii/venit și vechimea celei mai vechi linii de credit, modelul logistic desenează benzi drepte "
                      "de probabilitate, iar MLP regiuni curbe. Aria de sub curbă (AUC) a caracteristicii de operare a receptorului (ROC), adică "
                      "probabilitatea ca un client în neplată ales la întîmplare să aibă un scor mai mare decît un credit bun, este @{two.auc_logit|.3f} "
                      "pentru modelul logistic și @{two.auc_mlp|.3f} pentru MLP: flexibilitatea nu ajută cînd cele două intrări aduc puțin semnal."),
                chart="boundary_hmeq", cap=("Predicted PD on two HMEQ variables: logistic regression (left) and MLP 64-64 (right), with test loans.",
                                            "PD prezisă pe două variabile HMEQ: regresia logistică (stînga) și MLP 64-64 (dreapta), cu creditele de test.")),
          ],
          end=("An MLP is a stack of affine maps and activations; ReLU in the hidden layers, an output and loss that match the target, and a "
               "parameter count that tells how much data and regularisation are needed.",
               "Un MLP este o suprapunere de transformări afine și activări; ReLU în straturile ascunse, o ieșire și o pierdere potrivite țintei și "
               "un număr de parametri care arată de cîte date și de cîtă regularizare este nevoie.")),
        B(title=("Case study: credit scoring with real loans", "Studiu de caz: scoringul de credit pe credite reale"),
          intro=("Logistic regression, an MLP and extreme gradient boosting (XGBoost) are compared on HMEQ and German credit with the same splits, "
                 "inputs and metrics, several seeds, bootstrap intervals, a corrected test, calibration and the cost of errors.",
                 "Regresia logistică, un MLP și extreme gradient boosting (XGBoost) sînt comparate pe HMEQ și German credit cu aceleași împărțiri, "
                 "intrări și metrici, cu mai multe inițializări, intervale bootstrap, un test corectat, calibrare și costul erorilor."),
          blocks=[
              B(frame="The protocol",
                text=("The MLPs use ReLU layers, Adam with learning rate 0.001 and the weights of the best validation epoch, each with 5 seeds; "
                      "XGBoost uses depth 4, learning rate 0.03 and early stopping on validation. HMEQ has one stratified 60/20/20 split; German "
                      "credit uses 5 times repeated 5-fold cross-validation (@{german.n_folds} test folds). All models trained on the central "
                      "processing unit (CPU) with seed @{seed}.",
                      "MLP-urile folosesc straturi ReLU, Adam cu rata de învățare 0,001 și ponderile din epoca cea mai bună pe validare, fiecare cu 5 "
                      "inițializări; XGBoost folosește adîncimea 4, rata de învățare 0,03 și oprire timpurie pe validare. HMEQ are o împărțire "
                      "stratificată 60/20/20; German credit folosește validarea încrucișată cu 5 pliuri repetată de 5 ori (@{german.n_folds} de pliuri "
                      "de test). Toate modelele au fost antrenate pe procesorul central (CPU), cu valoarea inițială @{seed}.")),
              B(frame="On HMEQ, a wider hidden layer",
                text=("With one hidden layer, the test AUC rises from @{hmeq.auc_w1|.3f} with one unit, essentially a logistic model, to "
                      "@{hmeq.auc_best_width|.3f} with @{hmeq.best_width} units; logistic regression reaches @{hmeq.auc_logit|.3f} and XGBoost "
                      "@{hmeq.auc_xgb|.3f}. The gain flattens after about 16 units and the seed spread shrinks as the network widens.",
                      "Cu un strat ascuns, AUC pe test crește de la @{hmeq.auc_w1|.3f} cu un neuron, practic un model logistic, la @{hmeq.auc_best_width|.3f} "
                      "cu @{hmeq.best_width} de neuroni; regresia logistică ajunge la @{hmeq.auc_logit|.3f}, iar XGBoost la @{hmeq.auc_xgb|.3f}. Cîștigul "
                      "se aplatizează după circa 16 neuroni, iar variația dintre inițializări scade pe măsură ce rețeaua se lărgește."),
                chart="width_auc", cap=("Test AUC on HMEQ against the number of hidden units (mean and range of 5 seeds), with the two benchmarks.",
                                        "AUC pe test pe HMEQ în funcție de numărul de neuroni ascunși (media și intervalul a 5 inițializări), cu cele două repere.")),
              B(frame="At a 10",
                text=("The ROC curve plots the share of defaults caught against the share of good loans rejected for every cut-off. When 10 % of good "
                      "loans are rejected, logistic regression catches @{hmeq.tpr10_logit|.1f} % of the @{hmeq.test_bad} test defaults, the MLP "
                      "@{hmeq.tpr10_mlp|.1f} % and XGBoost @{hmeq.tpr10_xgb|.1f} %.",
                      "Curba ROC reprezintă ponderea neplăților detectate în funcție de ponderea creditelor bune respinse, pentru fiecare prag. Cînd "
                      "respingem 10 % dintre creditele bune, regresia logistică detectează @{hmeq.tpr10_logit|.1f} % dintre cele @{hmeq.test_bad} de "
                      "neplăți din test, MLP @{hmeq.tpr10_mlp|.1f} %, iar XGBoost @{hmeq.tpr10_xgb|.1f} %."),
                chart="roc_hmeq", cap=("ROC curves on the HMEQ test set.", "Curbele ROC pe setul de test HMEQ."),
                practice=("read the curve at the rejection rate the business accepts.", "citiți curba la rata de respingere acceptată de afacere.")),
              B(frame="The MLP gain on HMEQ",
                text=("The MLP 64-64 reaches a mean test AUC of @{hmeq.auc_mlp_mean|.3f} over 5 seeds (standard deviation @{hmeq.auc_mlp_sd|.3f}). "
                      "A paired bootstrap of the test loans gives the AUC gain over logistic regression, @{hmeq.dauc_mlp_lr|.3f}, a 95 % interval of "
                      "[@{hmeq.dauc_lo|.3f}; @{hmeq.dauc_hi|.3f}], which excludes zero.",
                      "MLP 64-64 atinge o AUC medie pe test de @{hmeq.auc_mlp_mean|.3f} pe 5 inițializări (abaterea standard @{hmeq.auc_mlp_sd|.3f}). "
                      "Un bootstrap pe perechi al creditelor de test dă pentru cîștigul de AUC față de regresia logistică, @{hmeq.dauc_mlp_lr|.3f}, "
                      "intervalul de 95 % [@{hmeq.dauc_lo|.3f}; @{hmeq.dauc_hi|.3f}], care exclude zero.")),
              B(frame="The scores are close to calibrated",
                text=("Provisions and loan prices use the PD itself, so the probabilities must be right in level, not only in order. In each decile of "
                      "predicted PD the observed default rate is close to the predicted one; the Brier scores are @{calib.brier_logit|.4f} for "
                      "logistic regression, @{calib.brier_mlp|.4f} for the MLP and @{calib.brier_xgb|.4f} for XGBoost, against @{calib.brier_base|.4f} "
                      "for a constant rate.",
                      "Provizioanele și prețurile creditelor folosesc PD ca atare, deci probabilitățile trebuie să fie corecte ca nivel, nu doar ca ordine. "
                      "În fiecare decilă a PD prezise, rata observată de neplată este apropiată de cea prezisă; scorurile Brier sînt @{calib.brier_logit|.4f} "
                      "pentru regresia logistică, @{calib.brier_mlp|.4f} pentru MLP și @{calib.brier_xgb|.4f} pentru XGBoost, față de "
                      "@{calib.brier_base|.4f} pentru o rată constantă."),
                chart="calibration", cap=("Reliability diagram on the HMEQ test set: predicted against observed default rate by decile.",
                                          "Diagrama de calibrare pe setul de test HMEQ: rata de neplată prezisă și observată, pe decile.")),
              B(frame="On 1,000 German applicants",
                text=("On German credit the picture changes: mean AUC @{german.auc_logit|.3f} for logistic regression, @{german.auc_mlp|.3f} for the "
                      "MLP and @{german.auc_xgb|.3f} for XGBoost across @{german.n_folds} held-out folds. With so few applicants the network has "
                      "nothing to add.",
                      "Pe German credit situația se schimbă: AUC medie @{german.auc_logit|.3f} pentru regresia logistică, @{german.auc_mlp|.3f} pentru "
                      "MLP și @{german.auc_xgb|.3f} pentru XGBoost, pe @{german.n_folds} de pliuri de test. Cu atît de puțini solicitanți, rețeaua nu "
                      "are ce să adauge."),
                chart="german_cv", cap=("AUC of each held-out fold in 5 x 5-fold cross-validation on German credit; bars are means.",
                                        "AUC pe fiecare pliu de test în validarea încrucișată 5 x 5 pe German credit; barele sînt mediile.")),
              B(frame="Why the usual",
                text=("The fold scores are correlated because the training sets overlap, so the naive t-test is too optimistic. The Nadeau-Bengio "
                      "correction inflates the variance by $1/J+n_{\\text{test}}/n_{\\text{train}}$; the MLP against logistic regression gives "
                      "$t=$ @{german.t_mlp|.2f} and $p=$ @{german.p_mlp|.3f}, so there is no evidence of a difference.",
                      "Scorurile pliurilor sînt corelate pentru că seturile de antrenare se suprapun, deci testul t naiv este prea optimist. Corecția "
                      "Nadeau-Bengio mărește varianța cu $1/J+n_{\\text{test}}/n_{\\text{train}}$; MLP față de regresia logistică dă $t=$ @{german.t_mlp|.2f} "
                      "și $p=$ @{german.p_mlp|.3f}, deci nu există dovezi ale unei diferențe.")),
              B(frame="A missed default costs five times",
                text=("The UCI cost matrix weighs a missed bad risk 5 times a rejected good applicant. Rejecting when $5p\\ge1-p$ gives the cut-off "
                      "$1/6$. With out-of-fold predictions, the logistic model costs @{german.cost_logit_bayes|.3f} per applicant at 1/6 and "
                      "@{german.cost_logit_05|.3f} at 1/2; the MLP @{german.cost_mlp_bayes|.3f} and @{german.cost_mlp_05|.3f}; accepting everyone "
                      "costs @{german.cost_all_accept|.2f}.",
                      "Matricea de cost UCI ponderează un risc prost acceptat de 5 ori mai mult decît un client bun respins. Respingerea cînd $5p\\ge1-p$ "
                      "dă pragul $1/6$. Cu predicții în afara pliului, modelul logistic costă @{german.cost_logit_bayes|.3f} pe solicitant la 1/6 și "
                      "@{german.cost_logit_05|.3f} la 1/2; MLP @{german.cost_mlp_bayes|.3f} și @{german.cost_mlp_05|.3f}; acceptarea tuturor costă "
                      "@{german.cost_all_accept|.2f}."),
                chart="cost_threshold", cap=("Cost per German credit applicant against the cut-off, out-of-fold predictions; dashed: cut-off 1/6.",
                                             "Costul pe solicitant German credit în funcție de prag, cu predicții în afara pliului; linia punctată: pragul 1/6."),
                practice=("the cut-off is where most money is lost or saved.", "pragul este locul unde se pierd sau se cîștigă cei mai mulți bani.")),
              B(frame="A minimal MLP in PyTorch",
                text=("Fifteen lines are enough: an `nn.Sequential` of linear layers and ReLUs, `BCEWithLogitsLoss` (sigmoid and cross-entropy in one "
                      "stable function), the Adam optimiser and a loop of `zero_grad`, `backward` and `step`. The notebook of this lecture adds the "
                      "data pipeline, the validation stopping and the seeds.",
                      "Cincisprezece rînduri sînt suficiente: un `nn.Sequential` de straturi liniare și ReLU, `BCEWithLogitsLoss` (sigmoida și entropia "
                      "încrucișată într-o singură funcție stabilă), optimizatorul Adam și o buclă cu `zero_grad`, `backward` și `step`. Notebook-ul "
                      "acestui curs adaugă pregătirea datelor, oprirea pe baza validării și inițializările.")),
          ],
          end=("On @{hmeq.n_train|,d} HMEQ loans the MLP beats the logistic score (AUC @{hmeq.auc_mlp_mean|.3f} against @{hmeq.auc_logit|.3f}) "
               "and boosting beats both; on 1,000 German applicants nothing beats logistic regression significantly. In both cases the cost-based "
               "cut-off matters more than the model.",
               "Pe @{hmeq.n_train|,d} de credite HMEQ, MLP depășește scorul logistic (AUC @{hmeq.auc_mlp_mean|.3f} față de @{hmeq.auc_logit|.3f}), "
               "iar boosting le depășește pe amîndouă; pe @{data.german_n|,d} de solicitanți germani, niciun model nu depășește semnificativ regresia logistică. În "
               "ambele cazuri, pragul stabilit din costuri contează mai mult decît modelul.")),
    ],
)

if __name__ == "__main__":
    sys.exit(write_pages(1, HERE, PAGE))
