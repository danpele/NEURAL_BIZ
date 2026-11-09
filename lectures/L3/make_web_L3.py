"""
make_web_L3.py -- web version of Lecture 3: writes lectures/L3/web_en.md and web_ro.md (numbers from L3_numbers.json, frame hashes
from the decks) and builds web/L3.html, web/L3_ro.html. Run after build_L3.py and make_quantlets_L3.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "L1"))
from webgen import write_pages  # noqa: E402

B = dict
PAGE = dict(
    intro=[
        ("A network that fits its training data perfectly can still fail on the next applicant or the next payment. This lecture is about the gap "
         "between the two: why it opens, how penalties, dropout, early stopping and batch normalisation close it, and how to design a validation "
         "scheme that measures it honestly. The case study is card fraud, where the event is rare, the data arrive in time order and a careless "
         "split can make a mediocre detector look excellent.",
         "O rețea care se potrivește perfect datelor de antrenare poate eșua totuși la următorul solicitant sau la următoarea plată. Acest curs "
         "tratează diferența dintre cele două situații: de ce apare, cum o reduc penalizările, dropout, oprirea timpurie și normalizarea pe loturi și "
         "cum se proiectează o schemă de validare care o măsoară corect. Studiul de caz este frauda cu carduri, unde evenimentul este rar, datele "
         "sosesc în ordine cronologică, iar o împărțire neatentă poate face ca un detector mediocru să pară excelent."),
        ("We use three real datasets: @{german.n_train} training applicants of German credit from the University of California, Irvine (UCI) "
         "repository, Californian census block groups, and @{fraud.n|,d} card payments of the Université libre de Bruxelles (ULB) and Worldline, "
         "with all @{fraud.frauds} frauds. Every model is a multilayer perceptron (MLP) trained in PyTorch, compared with logistic regression and, "
         "for fraud, with extreme gradient boosting (XGBoost), always with several seeds.",
         "Folosim trei seturi de date reale: @{german.n_train} de solicitanți de antrenare din German credit, din colecția Universității California din "
         "Irvine (UCI), grupurile de locuințe din recensămîntul Californiei și @{fraud.n|,d} de plăți cu cardul de la Université libre de Bruxelles (ULB) "
         "și Worldline, cu toate cele @{fraud.frauds} de fraude. Fiecare model este un perceptron multistrat (MLP) antrenat în PyTorch, comparat cu "
         "regresia logistică și, pentru fraudă, cu extreme gradient boosting (XGBoost), întotdeauna cu mai multe inițializări."),
    ],
    goals=[
        ("**Diagnose** overfitting from training and validation curves, and explain the bias-variance trade-off and double descent.",
         "**Diagnosticați** supraînvățarea (overfitting) din curbele de antrenare și validare și explicați compromisul deplasare-varianță și dubla coborîre."),
        ("**Write** weight decay, L1 penalties, dropout, early stopping and batch normalisation with their formulas.",
         "**Scrieți** penalizarea ponderilor, penalizarea L1, dropout, oprirea timpurie și normalizarea pe loturi cu formulele lor."),
        ("**Design** a validation scheme without leakage: cross-validation, time-ordered splits, resampling inside the training part only.",
         "**Proiectați** o schemă de validare fără scurgere de informație: validarea încrucișată, împărțirea cronologică, reeșantionarea doar în partea de antrenare."),
        ("**Evaluate** a fraud detector with precision, recall, the area under the precision-recall curve (PR-AUC) and the cost of errors.",
         "**Evaluați** un detector de fraudă cu precizia, sensibilitatea, aria de sub curba precizie-sensibilitate (PR-AUC) și costul erorilor."),
    ],
    sections=[
        B(title=("Generalisation and overfitting", "Generalizarea și supraînvățarea"),
          intro=("What the generalisation gap is, how it shows in training curves, how test error decomposes into bias, variance and noise, and why "
                 "very large models can generalise again.",
                 "Ce este diferența de generalizare, cum apare în curbele de antrenare, cum se descompune eroarea pe test în deplasare, varianță și zgomot "
                 "și de ce modelele foarte mari pot generaliza din nou."),
          blocks=[
              B(frame="A model is judged on applicants",
                text=("The generalisation gap $R(\\hat f)-\\hat R_n(\\hat f)$ compares the expected loss on new data with the training loss. Regularisation is "
                      "anything that lowers $R$ at the price of a higher $\\hat R_n$, usually written as a penalised objective $\\hat R_n(f)+\\lambda\\Omega(f)$.",
                      "Diferența de generalizare $R(\\hat f)-\\hat R_n(\\hat f)$ compară pierderea așteptată pe date noi cu pierderea de antrenare. Regularizarea "
                      "înseamnă orice reduce $R$ cu prețul unei $\\hat R_n$ mai mari, scrisă de obicei ca obiectiv penalizat $\\hat R_n(f)+\\lambda\\Omega(f)$.")),
              B(frame="Without regularisation, a large MLP",
                text=("An MLP 256-256 has @{overfit.params|,d} parameters for 600 applicants. Its validation loss is lowest at epoch @{overfit.best_ep} "
                      "(@{overfit.val_min|.3f}) and reaches @{overfit.val_last|.3f} after @{overfit.epochs} epochs, while the training loss goes to zero; "
                      "dropout 0.5 with weight decay only slows the rise (@{overfit.reg_val_last|.3f}). Logistic regression stays at @{overfit.logit_val|.3f}.",
                      "Un MLP 256-256 are @{overfit.params|,d} de parametri pentru 600 de solicitanți. Pierderea lui de validare este minimă la epoca "
                      "@{overfit.best_ep} (@{overfit.val_min|.3f}) și ajunge la @{overfit.val_last|.3f} după @{overfit.epochs} de epoci, în timp ce pierderea "
                      "de antrenare tinde la zero; dropout 0,5 cu penalizare doar încetinește creșterea (@{overfit.reg_val_last|.3f}). Regresia logistică "
                      "rămîne la @{overfit.logit_val|.3f}."),
                chart="overfit_german", cap=("Training and validation cross-entropy of an MLP 256-256 on German credit.",
                                             "Entropia încrucișată de antrenare și de validare pentru un MLP 256-256 pe German credit."),
                practice=("a training loss near zero on credit data is a warning, not a success.", "o pierdere de antrenare aproape de zero pe date de credit este un semnal de alarmă.")),
              B(frame="The test error splits",
                text=("For $y=f(x)+\\varepsilon$ the expected squared error of a model fitted on a random sample decomposes as",
                      "Pentru $y=f(x)+\\varepsilon$, eroarea pătratică așteptată a unui model estimat pe un eșantion aleator se descompune astfel:"),
                after=("$$ \\mathbb E\\big[(y-\\hat f_{\\mathcal D}(x))^2\\big]=\\big(f(x)-\\mathbb E\\hat f_{\\mathcal D}(x)\\big)^2+\\operatorname{Var}\\big(\\hat f_{\\mathcal D}(x)\\big)+\\sigma^2 $$ {#eq:biasvar}\n\n"
                       "Regularisation in @eq:biasvar trades a little squared bias for a large reduction in variance; the noise $\\sigma^2$ is a floor no model can beat.",
                       "$$ \\mathbb E\\big[(y-\\hat f_{\\mathcal D}(x))^2\\big]=\\big(f(x)-\\mathbb E\\hat f_{\\mathcal D}(x)\\big)^2+\\operatorname{Var}\\big(\\hat f_{\\mathcal D}(x)\\big)+\\sigma^2 $$ {#eq:biasvar}\n\n"
                       "În @eq:biasvar, regularizarea acceptă puțină deplasare în schimbul unei reduceri mari a varianței; zgomotul $\\sigma^2$ este o limită pe care niciun model nu o poate depăși.")),
              B(frame="Past the interpolation threshold",
                text=("With $p$ random, frozen rectified linear units (ReLU) and a minimum-norm output layer fitted on $n=$ @{dd.n} block groups, the test "
                      "mean squared error (MSE) is @{dd.best_under_mse|.3f} at $p=$ @{dd.best_under_p}, explodes at the interpolation threshold $p=n$ "
                      "(@{dd.peak_mse|sci}) and falls again to @{dd.big_mse|.3f} at $p=$ @{dd.big_p|,d}: the double descent of Belkin et al. (2019).",
                      "Cu $p$ unități liniare rectificate (ReLU) aleatoare și fixe și un strat de ieșire de normă minimă, estimat pe $n=$ @{dd.n} de grupuri, "
                      "eroarea pătratică medie (MSE) pe test este @{dd.best_under_mse|.3f} la $p=$ @{dd.best_under_p}, explodează la pragul de interpolare "
                      "$p=n$ (@{dd.peak_mse|sci}) și scade din nou la @{dd.big_mse|.3f} pentru $p=$ @{dd.big_p|,d}: dubla coborîre descrisă de Belkin et al. (2019)."),
                chart="double_descent", cap=("Training and test MSE against the number of random ReLU features, median of 10 draws.",
                                             "MSE de antrenare și de test în funcție de numărul de caracteristici ReLU aleatoare, mediana a 10 extrageri.")),
          ],
          end=("Overfitting shows as falling training loss and rising validation loss; the bias-variance trade-off explains the classical U-curve, and "
               "double descent shows that very large models can generalise again.",
               "Supraînvățarea apare ca o pierdere de antrenare care scade și una de validare care crește; compromisul deplasare-varianță explică curba "
               "clasică în U, iar dubla coborîre arată că modelele foarte mari pot generaliza din nou.")),
        B(title=("Penalties on the weights", "Penalizarea ponderilor"),
          intro=("Weight decay and the L1 penalty constrain the size of the weights; they correspond to Gaussian and Laplace priors.",
                 "Penalizarea ponderilor și penalizarea L1 limitează mărimea ponderilor; ele corespund unor distribuții a priori gaussiene și Laplace."),
          blocks=[
              B(frame="Weight decay = an L2 penalty",
                text=("Adding $\\frac\\lambda2\\|\\mathbf w\\|^2$ to the loss turns each stochastic gradient descent (SGD) step into "
                      "$\\mathbf w\\leftarrow(1-\\eta\\lambda)\\mathbf w-\\eta\\nabla\\mathcal L$, and the penalised minimum is the maximum a posteriori (MAP) "
                      "estimate under a Gaussian prior. With Adam the two differ; Adam with decoupled weight decay (AdamW) restores uniform shrinkage.",
                      "Adăugarea lui $\\frac\\lambda2\\|\\mathbf w\\|^2$ la pierdere transformă fiecare pas de coborîre stohastică pe gradient (SGD) în "
                      "$\\mathbf w\\leftarrow(1-\\eta\\lambda)\\mathbf w-\\eta\\nabla\\mathcal L$, iar minimul penalizat este estimarea maximum a posteriori (MAP) "
                      "cu o distribuție a priori gaussiană. Cu Adam cele două diferă; Adam cu penalizarea ponderilor decuplată (AdamW) reface micșorarea uniformă.")),
              B(frame="An L1 penalty sets weights",
                text=("The L1 penalty $\\lambda\\|\\mathbf w\\|_1$ has a corner at zero; its proximal (soft-thresholding) step "
                      "$w\\leftarrow\\operatorname{sign}(u)\\max(|u|-\\eta\\lambda,0)$ sets small weights exactly to zero, the lasso of Tibshirani (1996).",
                      "Penalizarea L1 $\\lambda\\|\\mathbf w\\|_1$ are un colț în zero; pasul ei proximal (de prag moale) "
                      "$w\\leftarrow\\operatorname{sign}(u)\\max(|u|-\\eta\\lambda,0)$ aduce ponderile mici exact la zero, ca în lasso-ul lui Tibshirani (1996).")),
              B(frame="L1 piles the first-layer weights",
                text=("On German credit, @{l1l2.share_small_l1_pct|.1f} % of the @{l1l2.n_weights|,d} first-layer weights end below @{l1l2.thr|g} in absolute "
                      "value with L1, against @{l1l2.share_small_l2_pct|.1f} % with L2; the test AUC is @{l1l2.auc_l1|.3f} and @{l1l2.auc_l2|.3f}.",
                      "Pe German credit, @{l1l2.share_small_l1_pct|.1f} % dintre cele @{l1l2.n_weights|,d} de ponderi ale primului strat ajung sub @{l1l2.thr|g} "
                      "în valoare absolută cu L1, față de @{l1l2.share_small_l2_pct|.1f} % cu L2; AUC pe test este @{l1l2.auc_l1|.3f}, respectiv @{l1l2.auc_l2|.3f}."),
                chart="l1_l2_weights", cap=("First-layer weights of an MLP with 64 hidden units under L1 and L2 penalties (log counts).",
                                            "Ponderile primului strat ale unui MLP cu 64 de neuroni ascunși, cu penalizare L1 și L2 (frecvențe pe scară logaritmică).")),
          ],
          end=("Weight decay shrinks all weights in proportion, the L1 penalty sets many exactly to zero; both are priors in disguise.",
               "Penalizarea ponderilor le micșorează proporțional, penalizarea L1 aduce multe dintre ele exact la zero; ambele sînt distribuții a priori deghizate.")),
        B(title=("Dropout, early stopping and batch normalisation", "Dropout, oprirea timpurie și normalizarea pe loturi"),
          intro=("Three regularisers that act through training rather than through the loss, measured on German credit, card fraud and California housing.",
                 "Trei regularizatori care acționează prin antrenare, nu prin pierdere, măsurați pe German credit, pe fraudele cu carduri și pe California housing."),
          blocks=[
              B(frame="Dropout trains an ensemble",
                text=("Inverted dropout multiplies each hidden unit by $m_j/(1-p)$ with $m_j\\sim$ Bernoulli$(1-p)$ during training and uses the full network "
                      "at test time. With $H$ droppable units the network samples from $2^H$ thinned sub-networks that share their weights (Srivastava et al., 2014).",
                      "Dropout-ul inversat înmulțește fiecare neuron ascuns cu $m_j/(1-p)$, cu $m_j\\sim$ Bernoulli$(1-p)$, în timpul antrenării și folosește "
                      "rețeaua completă la testare. Cu $H$ neuroni care pot fi opriți, rețeaua eșantionează din $2^H$ subrețele subțiate, cu ponderi comune "
                      "(Srivastava et al., 2014).")),
              B(frame="On 600 applicants, regularisation",
                text=("Over 5 folds, the MLP 256-256 reaches AUC @{cv.auc_none|.3f} without regularisation, @{cv.auc_best_lam|.3f} at the best weight decay "
                      "and @{cv.auc_best_p|.3f} at the best dropout rate, while logistic regression reaches @{cv.auc_logit|.3f}. Weight decay does shrink "
                      "the network (weight norm @{wd.0.norm|.1f} at $\\lambda=0$, @{wd.3.norm|.1f} at $\\lambda=3$), but on so little data a smaller model is the better cure.",
                      "Pe 5 pliuri, MLP 256-256 ajunge la AUC @{cv.auc_none|.3f} fără regularizare, @{cv.auc_best_lam|.3f} la cea mai bună penalizare și "
                      "@{cv.auc_best_p|.3f} la cea mai bună rată de dropout, în timp ce regresia logistică ajunge la @{cv.auc_logit|.3f}. Penalizarea "
                      "micșorează rețeaua (norma ponderilor @{wd.0.norm|.1f} la $\\lambda=0$, @{wd.3.norm|.1f} la $\\lambda=3$), dar pe atît de puține date "
                      "un model mai mic este remediul mai bun."),
                chart="wd_dropout", cap=("Held-out AUC of an MLP 256-256 on German credit against weight decay and dropout rate.",
                                         "AUC pe pliul de test pentru un MLP 256-256 pe German credit, în funcție de penalizare și de rata de dropout.")),
              B(frame="Early stopping: keep the weights",
                text=("Early stopping keeps the weights of the epoch with the lowest validation loss and stops after $P$ epochs without improvement. Near a "
                      "quadratic minimum, $t$ steps of size $\\eta$ act like an L2 penalty of strength about $1/(\\eta t)$ (Prechelt, 1998; Bishop and Bishop, 2024).",
                      "Oprirea timpurie păstrează ponderile epocii cu cea mai mică pierdere de validare și se oprește după $P$ epoci fără îmbunătățire. Lîngă un "
                      "minim pătratic, $t$ pași de mărime $\\eta$ acționează ca o penalizare L2 de intensitate aproximativ $1/(\\eta t)$ (Prechelt, 1998; Bishop și Bishop, 2024).")),
              B(frame="On card fraud, validation loss",
                text=("On the time-ordered fraud split, the validation loss of an MLP 256-256 is lowest at epoch @{es.best_ep} (@{es.val_min|.4f}); with a "
                      "patience of @{es.patience} epochs training stops at epoch @{es.stop_ep}, and after @{es.epochs} epochs the validation loss would be @{es.val_last|.4f}.",
                      "Pe împărțirea cronologică a datelor despre fraude, pierderea de validare a unui MLP 256-256 este minimă la epoca @{es.best_ep} "
                      "(@{es.val_min|.4f}); cu o răbdare de @{es.patience} de epoci, antrenarea se oprește la epoca @{es.stop_ep}, iar după @{es.epochs} de "
                      "epoci pierderea de validare ar fi @{es.val_last|.4f}."),
                chart="early_stopping", cap=("Training and validation loss on card fraud, with the best epoch and the patience window.",
                                             "Pierderea de antrenare și de validare la fraudele cu carduri, cu epoca cea mai bună și fereastra de răbdare.")),
              B(frame="Batch normalisation standardises",
                text=("Batch normalisation (BN) standardises each unit over the mini-batch and then rescales it with learned $\\gamma_j,\\beta_j$; at test time "
                      "running averages replace the batch statistics (Ioffe and Szegedy, 2015).",
                      "Normalizarea pe loturi (BN) standardizează fiecare neuron pe mini-lot și îl rescalează apoi cu $\\gamma_j,\\beta_j$ învățați; la testare, "
                      "mediile mobile înlocuiesc statisticile lotului (Ioffe și Szegedy, 2015).")),
              B(frame="Batch norm lets a deep MLP",
                text=("A 6-layer MLP on California housing trained with SGD and momentum diverges at $\\eta=0.1$ in @{bn.plain_01.diverged} of 3 seeds without "
                      "BN; with BN the same rate trains, with a best validation MSE of @{bn.bn_01.val_best|.3f}. At $\\eta=0.01$ the best values are "
                      "@{bn.plain_001.val_best|.3f} without and @{bn.bn_001.val_best|.3f} with BN, the BN curve being noisier.",
                      "Un MLP cu 6 straturi pe California housing, antrenat cu SGD și moment, diverge la $\\eta=0{,}1$ în @{bn.plain_01.diverged} din 3 inițializări "
                      "fără BN; cu BN, aceeași rată funcționează, cu cea mai bună MSE pe validare de @{bn.bn_01.val_best|.3f}. La $\\eta=0{,}01$, cele mai "
                      "bune valori sînt @{bn.plain_001.val_best|.3f} fără și @{bn.bn_001.val_best|.3f} cu BN, curba cu BN fiind mai zgomotoasă."),
                chart="batchnorm", cap=("Validation MSE of a 6-layer MLP with and without batch normalisation at two learning rates.",
                                        "MSE pe validare pentru un MLP cu 6 straturi, cu și fără normalizare pe loturi, la două rate de învățare.")),
          ],
          end=("Dropout and penalties shrink capacity, early stopping is the most effective cheap regulariser on our fraud data, and batch norm is first "
               "of all an optimisation aid.",
               "Dropout și penalizările reduc capacitatea, oprirea timpurie este cel mai eficient regularizator ieftin pe datele noastre despre fraude, iar "
               "normalizarea pe loturi este în primul rînd un ajutor pentru optimizare.")),
        B(title=("Validation design", "Schema de validare"),
          intro=("How to measure generalisation honestly: cross-validation, the noise of a single split, and the leaks that inflate scores.",
                 "Cum se măsoară corect generalizarea: validarea încrucișată, zgomotul unei singure împărțiri și scurgerile care umflă scorurile."),
          blocks=[
              B(frame="Hold-out, cross-validation and nested",
                text=("$K$-fold cross-validation averages the loss of $K$ models, each evaluated on the fold it did not see. Nested cross-validation adds an "
                      "inner loop for tuning, so the outer estimate measures the whole procedure.",
                      "Validarea încrucișată cu $K$ pliuri face media pierderilor a $K$ modele, fiecare evaluat pe pliul pe care nu l-a văzut. Validarea "
                      "încrucișată imbricată adaugă o buclă interioară pentru ajustare, astfel încît estimarea exterioară măsoară întreaga procedură.")),
              B(frame="One random hold-out split",
                text=("Over 200 random 70/30 splits of German credit, the test AUC of the same logistic model ranges from @{holdout.min|.3f} to "
                      "@{holdout.max|.3f}, with 90 % of splits between @{holdout.p05|.3f} and @{holdout.p95|.3f}.",
                      "Pe 200 de împărțiri aleatoare 70/30 ale datelor German credit, AUC pe test a aceluiași model logistic variază între @{holdout.min|.3f} "
                      "și @{holdout.max|.3f}, iar 90 % dintre împărțiri dau valori între @{holdout.p05|.3f} și @{holdout.p95|.3f}."),
                chart="holdout_noise", cap=("Test AUC of logistic regression over 200 random hold-out splits of German credit.",
                                            "AUC pe test al regresiei logistice pe 200 de împărțiri aleatoare ale datelor German credit.")),
              B(frame="Information leakage into the validation",
                text=("Four leaks recur in business data: preprocessing fitted on all rows, resampling before the split, ignoring time, and inputs that are "
                      "only known after the decision (Kaufman et al., 2012). For time series, use expanding-window or blocked validation (Bergmeir and Benítez, 2012).",
                      "Patru scurgeri apar frecvent în datele de afaceri: preprocesare estimată pe toate rîndurile, reeșantionare înainte de împărțire, ignorarea "
                      "timpului și intrări cunoscute abia după decizie (Kaufman et al., 2012). Pentru serii de timp, folosiți validarea cu fereastră extinsă sau "
                      "pe blocuri (Bergmeir și Benítez, 2012).")),
              B(frame="Oversampling before the split",
                text=("The same MLP on the same payments gets PR-AUC @{design.time.ap|.3f} with a time-ordered split, @{design.random.ap|.3f} with a random "
                      "split, and @{design.oversample_first.ap|.3f} when every fraud is copied 20 times before splitting: copies end up on both sides.",
                      "Același MLP, pe aceleași plăți, obține PR-AUC @{design.time.ap|.3f} cu împărțire cronologică, @{design.random.ap|.3f} cu împărțire "
                      "aleatoare și @{design.oversample_first.ap|.3f} cînd fiecare fraudă este copiată de 20 de ori înainte de împărțire: copiile ajung de ambele părți."),
                chart="split_leakage", cap=("Estimated PR-AUC under three evaluation designs, 5 seeds each.",
                                            "PR-AUC estimată în trei scheme de evaluare, cîte 5 inițializări.")),
          ],
          end=("Validation must mimic deployment: one small split is noisy, random splits ignore time, and resampling before the split leaks.",
               "Validarea trebuie să imite producția: o singură împărțire mică este zgomotoasă, împărțirile aleatoare ignoră timpul, iar reeșantionarea "
               "înainte de împărțire produce scurgeri.")),
        B(title=("Case study: card fraud detection", "Studiu de caz: detectarea fraudei cu carduri"),
          intro=("Rare events need their own metrics, weights for the sampling and a cost-based alert budget.",
                 "Evenimentele rare au nevoie de metrici proprii, de ponderi pentru eșantionare și de un buget de alerte stabilit din costuri."),
          blocks=[
              B(frame="Fraud is rare and clustered",
                text=("The full data hold @{fraud.n_full|,d} payments with a fraud rate of @{fraud.prev_full_pct|.3f} %; our file keeps every fraud and 10 % "
                      "of the legitimate payments, each weighted @{fraud.legit_weight|.0f}. The inputs are 28 principal component analysis (PCA) scores and "
                      "the log amount. The hourly fraud rate reaches @{fraud.rate_h_max_pct|.2f} %, against a median of @{fraud.rate_h_median_pct|.3f} %.",
                      "Datele complete au @{fraud.n_full|,d} de plăți, cu o rată a fraudei de @{fraud.prev_full_pct|.3f} %; fișierul nostru păstrează toate "
                      "fraudele și 10 % din plățile legitime, fiecare cu ponderea @{fraud.legit_weight|.0f}. Intrările sînt 28 de scoruri de analiză a "
                      "componentelor principale (PCA) și logaritmul sumei. Rata orară a fraudei ajunge la @{fraud.rate_h_max_pct|.2f} %, față de o mediană "
                      "de @{fraud.rate_h_median_pct|.3f} %."),
                chart="fraud_by_hour", cap=("Hourly fraud rate over the two days, with the start of validation and test.",
                                            "Rata orară a fraudei în cele două zile, cu începutul validării și al testării.")),
              B(frame="Performance measures for rare events",
                text=("Precision is $TP/(TP+FP)$, recall $TP/(TP+FN)$, and the PR-AUC averages precision over recall levels; a random score gets a PR-AUC "
                      "equal to the fraud rate (Saito and Rehmsmeier, 2015). Every legitimate payment counts with weight @{fraud.legit_weight|.0f} in $FP$, "
                      "so precision is that of the full population.",
                      "Precizia este $TP/(TP+FP)$, sensibilitatea $TP/(TP+FN)$, iar PR-AUC face media preciziei pe nivelurile de sensibilitate; un scor aleator "
                      "are PR-AUC egal cu rata fraudei (Saito și Rehmsmeier, 2015). Fiecare plată legitimă contează cu ponderea @{fraud.legit_weight|.0f} în "
                      "$FP$, deci precizia este cea a populației complete.")),
              B(frame="On the test hours, early stopping",
                text=("With 60 fixed epochs and 5 seeds, the test PR-AUC is @{abl.none.ap|.3f} with no regularisation, @{abl.es.ap|.3f} with early stopping, "
                      "@{abl.wd.ap|.3f} with weight decay, @{abl.dropout.ap|.3f} with dropout, @{abl.bn.ap|.3f} with batch norm and @{abl.all.ap|.3f} with all combined.",
                      "Cu 60 de epoci fixe și 5 inițializări, PR-AUC pe test este @{abl.none.ap|.3f} fără regularizare, @{abl.es.ap|.3f} cu oprire timpurie, "
                      "@{abl.wd.ap|.3f} cu penalizare, @{abl.dropout.ap|.3f} cu dropout, @{abl.bn.ap|.3f} cu normalizare pe loturi și @{abl.all.ap|.3f} cu toate."),
                chart="reg_ablation", cap=("Test PR-AUC of six regularisation settings, 5 seeds each.", "PR-AUC pe test pentru șase variante de regularizare, cîte 5 inițializări.")),
              B(frame="Against logistic regression and gradient",
                text=("On the last 20 % of the two days, PR-AUC is @{final.logit.ap|.3f} for logistic regression, @{final.mlp.ap|.3f} for the regularised MLP "
                      "(5-seed ensemble) and @{final.xgb.ap|.3f} for XGBoost; at 80 % precision they catch @{final.logit.recall_at_p80_pct|.1f} %, "
                      "@{final.mlp.recall_at_p80_pct|.1f} % and @{final.xgb.recall_at_p80_pct|.1f} % of the @{fraud.fr_test} test frauds.",
                      "Pe ultimele 20 % din cele două zile, PR-AUC este @{final.logit.ap|.3f} pentru regresia logistică, @{final.mlp.ap|.3f} pentru MLP "
                      "regularizat (ansamblu de 5 inițializări) și @{final.xgb.ap|.3f} pentru XGBoost; la o precizie de 80 %, ele detectează "
                      "@{final.logit.recall_at_p80_pct|.1f} %, @{final.mlp.recall_at_p80_pct|.1f} % și @{final.xgb.recall_at_p80_pct|.1f} % dintre cele "
                      "@{fraud.fr_test} de fraude din test."),
                chart="fraud_pr", cap=("Precision-recall curves on the test hours, population-weighted.", "Curbele precizie-sensibilitate pe orele de test, ponderate la populație.")),
              B(frame="Class weights and the prior correction",
                text=("A model trained on data where legitimate payments were kept with rate $\\beta$ overstates the odds of fraud by $1/\\beta$; the corrected "
                      "probability is $p=\\beta p_s/(\\beta p_s-p_s+1)$ (Dal Pozzolo et al., 2015). Class weights and resampling move the effective cut-off but "
                      "hardly change the ranking.",
                      "Un model antrenat pe date în care plățile legitime au fost păstrate cu rata $\\beta$ supraestimează șansa de fraudă de $1/\\beta$ ori; "
                      "probabilitatea corectată este $p=\\beta p_s/(\\beta p_s-p_s+1)$ (Dal Pozzolo et al., 2015). Ponderile claselor și reeșantionarea "
                      "mută pragul efectiv, dar aproape nu schimbă ordonarea.")),
              B(frame="The right number of alerts",
                text=("If a missed fraud loses its amount and each alert costs a review of @{cost.review|.0f} euro (EUR), the test-period losses fall from "
                      "@{cost.no_model|,.0f} EUR with no alerts to @{cost.logit.min|,.0f} EUR for logistic regression, @{cost.mlp.min|,.0f} EUR for the MLP "
                      "and @{cost.xgb.min|,.0f} EUR for XGBoost at their best alert counts.",
                      "Dacă o fraudă ratată își pierde suma și fiecare alertă costă o verificare de @{cost.review|.0f} euro (EUR), pierderile din perioada de "
                      "test scad de la @{cost.no_model|,.0f} EUR fără alerte la @{cost.logit.min|,.0f} EUR pentru regresia logistică, @{cost.mlp.min|,.0f} EUR "
                      "pentru MLP și @{cost.xgb.min|,.0f} EUR pentru XGBoost, la numărul optim de alerte."),
                chart="fraud_cost", cap=("Total test-period cost against the number of alerts sent to review.", "Costul total în perioada de test în funcție de numărul de alerte trimise la verificare."),
                practice=("the alert budget, set from amounts and review costs, is the business decision; the model only ranks.",
                          "bugetul de alerte, stabilit din sume și costuri de verificare, este decizia de afaceri; modelul doar ordonează.")),
          ],
          end=("For rare events, report PR-AUC, precision at a workable recall and the cost of alerts, on a time-ordered test set and with several seeds.",
               "Pentru evenimente rare, raportați PR-AUC, precizia la o sensibilitate gestionabilă și costul alertelor, pe un set de test cronologic și cu "
               "mai multe inițializări.")),
    ],
)

if __name__ == "__main__":
    sys.exit(write_pages(3, HERE, PAGE))
