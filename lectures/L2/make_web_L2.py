"""
make_web_L2.py -- web version of Lecture 2: writes lectures/L2/web_en.md and web_ro.md (numbers from L2_numbers.json, frame hashes
from the decks) and builds web/L2.html, web/L2_ro.html. Run after build_L2.py and make_quantlets_L2.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "L1"))
from webgen import write_pages  # noqa: E402

B = dict
PAGE = dict(
    intro=[
        ("A neural network is only as good as its training. This lecture opens the black box of `loss.backward()` and `optimizer.step()`: "
         "how the gradient of a loss with thousands or millions of weights is computed exactly by backpropagation, and how gradient descent "
         "(GD) and its stochastic, momentum and adaptive variants turn that gradient into good weights. Every claim is measured on real data: "
         "@{cal.n|,d} Californian census block groups for regression, @{fraud.n|,d} card payments for fraud detection and the daily S&P 500 for a "
         "1-day value at risk (VaR).",
         "O rețea neuronală este atît de bună cît este antrenarea ei. Acest curs deschide „cutia neagră” a instrucțiunilor `loss.backward()` și "
         "`optimizer.step()`: cum se calculează exact, prin propagarea înapoi a erorii, gradientul unei pierderi cu mii sau milioane de ponderi și "
         "cum coborîrea pe gradient (GD) și variantele ei stohastice, cu moment sau adaptive transformă acest gradient în ponderi bune. Fiecare "
         "afirmație este măsurată pe date reale: @{cal.n|,d} de grupuri de locuințe din recensămîntul Californiei pentru regresie, @{fraud.n|,d} de "
         "plăți cu cardul pentru detectarea fraudei și indicele S&P 500 zilnic pentru valoarea la risc (VaR) pe o zi."),
    ],
    goals=[
        ("**Write** gradient descent as the solution method of empirical risk minimisation and explain why conditioning sets its speed.",
         "**Scrieți** coborîrea pe gradient ca metodă de rezolvare a minimizării riscului empiric și explicați de ce condiționarea îi stabilește viteza."),
        ("**Derive** backpropagation for a multilayer perceptron (MLP), compute it by hand and check it against automatic differentiation.",
         "**Derivați** propagarea înapoi pentru un perceptron multistrat (MLP), calculați-o de mînă și verificați-o cu diferențierea automată."),
        ("**Explain** vanishing gradients and choose an initialisation that keeps the signal alive.",
         "**Explicați** dispariția gradienților și alegeți o inițializare care păstrează semnalul."),
        ("**Compare** stochastic gradient descent (SGD), momentum, root mean square propagation (RMSProp), Adam and learning-rate schedules with several seeds.",
         "**Comparați** coborîrea stohastică pe gradient (SGD), metoda momentului, propagarea cu medie pătratică (RMSProp), Adam și programele ratei de învățare, cu mai multe inițializări."),
        ("**Train and backtest** a quantile network for the 99 % VaR of the S&P 500.",
         "**Antrenați și testați retroactiv** o rețea cuantilică pentru VaR de 99 % al indicelui S&P 500."),
    ],
    sections=[
        B(title=("From the loss to the gradient", "De la pierdere la gradient"),
          intro=("Training is the minimisation of an average loss over millions of parameters; gradient descent solves it one small step at a time, "
                 "and the shape of the loss surface decides how fast.",
                 "Antrenarea este minimizarea unei pierderi medii după milioane de parametri; coborîrea pe gradient o rezolvă pas cu pas, iar forma "
                 "suprafeței pierderii decide cît de repede."),
          blocks=[
              B(frame="Training a network is a high-dimensional",
                text=("There is no formula for the weights of a network. We start somewhere and repeat $\\boldsymbol\\theta_{t+1}=\\boldsymbol\\theta_t-\\eta\\nabla\\mathcal L(\\boldsymbol\\theta_t)$: "
                      "to first order, $\\mathcal L(\\boldsymbol\\theta-\\eta\\mathbf g)=\\mathcal L(\\boldsymbol\\theta)-\\eta\\|\\mathbf g\\|^2$, so a small step "
                      "against the gradient lowers the loss. Our MLP 64-64 has @{gradcheck.n_params|,d} parameters for @{cal.n_train|,d} training block groups.",
                      "Nu există o formulă pentru ponderile unei rețele. Pornim de undeva și repetăm $\\boldsymbol\\theta_{t+1}=\\boldsymbol\\theta_t-\\eta\\nabla\\mathcal L(\\boldsymbol\\theta_t)$: "
                      "în prima aproximare, $\\mathcal L(\\boldsymbol\\theta-\\eta\\mathbf g)=\\mathcal L(\\boldsymbol\\theta)-\\eta\\|\\mathbf g\\|^2$, deci un pas mic contra "
                      "gradientului scade pierderea. MLP 64-64 are @{gradcheck.n_params|,d} de parametri pentru @{cal.n_train|,d} de grupuri de antrenare.")),
              B(frame="The data of this lecture",
                text=("California housing is split at random 60/20/20 into @{cal.n_train|,d}, @{cal.n_val|,d} and @{cal.n_test|,d} block groups. The card-fraud "
                      "data of the Université libre de Bruxelles (ULB) and Worldline, published on the Open Machine Learning platform (OpenML), keep all "
                      "@{fraud.frauds} frauds and a 10 % sample of legitimate payments; their inputs are principal component analysis (PCA) scores, and "
                      "weights restore the true fraud rate of @{fraud.prev_full_pct|.3f} %. Fraud and S&P 500 are split in time order.",
                      "Datele California sînt împărțite aleator 60/20/20 în @{cal.n_train|,d}, @{cal.n_val|,d} și @{cal.n_test|,d} de grupuri. Datele despre "
                      "fraude cu carduri ale Université libre de Bruxelles (ULB) și Worldline, publicate pe platforma Open Machine Learning (OpenML), păstrează "
                      "toate cele @{fraud.frauds} de fraude și un eșantion de 10 % din plățile legitime; intrările sînt scoruri de analiză a componentelor "
                      "principale (PCA), iar ponderile refac rata reală a fraudei, de @{fraud.prev_full_pct|.3f} %. Frauda și S&P 500 sînt împărțite cronologic.")),
              B(frame="Badly scaled inputs",
                text=("The mean squared error (MSE) of the one-variable regression $\\hat y=a+b\\cdot\\text{income}$ is a real quadratic surface. On raw "
                      "income its Hessian has condition number @{surface.cond_raw|.1f}: the contours are long ellipses, and after @{surface.steps} steps "
                      "plain GD is still at MSE @{surface.mse_gd_raw_60|.3f}, momentum at @{surface.mse_mom_raw_60|.3f}. On standardised income the "
                      "condition number is @{surface.cond_std|.0f} and GD reaches the minimum, @{surface.mse_star|.3f}, almost at once.",
                      "Eroarea pătratică medie (MSE) a regresiei cu o variabilă $\\hat y=a+b\\cdot\\text{venit}$ este o suprafață pătratică reală. Pe venitul "
                      "brut, hessiana are numărul de condiționare @{surface.cond_raw|.1f}: conturul este format din elipse alungite, iar după @{surface.steps} "
                      "de pași GD simplu are încă MSE @{surface.mse_gd_raw_60|.3f}, iar metoda momentului @{surface.mse_mom_raw_60|.3f}. Pe venitul standardizat, "
                      "numărul de condiționare este @{surface.cond_std|.0f}, iar GD ajunge aproape imediat la minim, @{surface.mse_star|.3f}."),
                chart="gd_paths", cap=("Gradient descent and momentum on the real loss surface, raw and standardised income.",
                                       "Coborîrea pe gradient și metoda momentului pe suprafața reală a pierderii, cu venitul brut și standardizat."),
                practice=("standardise every input on the training set before training any network.",
                          "standardizați fiecare intrare pe setul de antrenare înainte de a antrena orice rețea.")),
              B(frame="On a quadratic, the step size",
                text=("Near a minimum, $\\mathcal L\\approx\\mathcal L^*+\\tfrac12(\\boldsymbol\\theta-\\boldsymbol\\theta^*)^\\top\\mathbf H(\\boldsymbol\\theta-\\boldsymbol\\theta^*)$ "
                      "and each eigen-direction of the Hessian contracts by $1-\\eta\\lambda_k$ per step. Convergence needs $0<\\eta<2/\\lambda_{\\max}$, and with "
                      "the best fixed step the error shrinks by $(\\kappa-1)/(\\kappa+1)$, slow when the condition number $\\kappa$ is large.",
                      "Lîngă un minim, $\\mathcal L\\approx\\mathcal L^*+\\tfrac12(\\boldsymbol\\theta-\\boldsymbol\\theta^*)^\\top\\mathbf H(\\boldsymbol\\theta-\\boldsymbol\\theta^*)$, "
                      "iar fiecare direcție proprie a hessianei se contractă cu $1-\\eta\\lambda_k$ la fiecare pas. Convergența cere $0<\\eta<2/\\lambda_{\\max}$, iar "
                      "cu cel mai bun pas fix eroarea scade cu factorul $(\\kappa-1)/(\\kappa+1)$, lent cînd numărul de condiționare $\\kappa$ este mare.")),
          ],
          end=("Gradient descent moves against the gradient; the learning rate must respect the steepest curvature, and poorly scaled inputs make "
               "the flat directions painfully slow.",
               "Coborîrea pe gradient se deplasează contra gradientului; rata de învățare trebuie să respecte curbura cea mai mare, iar intrările "
               "nescalate fac direcțiile plate foarte lente.")),
        B(title=("Backpropagation", "Propagarea înapoi a erorii"),
          intro=("Backpropagation is the chain rule applied from the loss backwards; it gives every partial derivative for about the cost of one more "
                 "forward pass. We compute it by hand, check it numerically and see why deep sigmoid networks stop learning.",
                 "Propagarea înapoi este regula de derivare a funcțiilor compuse, aplicată de la pierdere înapoi; dă toate derivatele parțiale pentru "
                 "costul a încă o propagare înainte. O calculăm de mînă, o verificăm numeric și vedem de ce rețelele adînci cu sigmoide nu mai învață."),
          blocks=[
              B(frame="Backpropagation is the chain rule",
                text=("With $\\mathbf z^{(l)}=\\mathbf W^{(l)}\\mathbf h^{(l-1)}+\\mathbf b^{(l)}$ and $\\mathbf h^{(l)}=a(\\mathbf z^{(l)})$, the error signal "
                      "travels backwards as",
                      "Cu $\\mathbf z^{(l)}=\\mathbf W^{(l)}\\mathbf h^{(l-1)}+\\mathbf b^{(l)}$ și $\\mathbf h^{(l)}=a(\\mathbf z^{(l)})$, semnalul de eroare "
                      "circulă înapoi astfel:"),
                after=("$$ \\boldsymbol\\delta^{(l)}=\\big(\\mathbf W^{(l+1)\\top}\\boldsymbol\\delta^{(l+1)}\\big)\\odot a'(\\mathbf z^{(l)}),\\qquad "
                       "\\frac{\\partial\\ell}{\\partial\\mathbf W^{(l)}}=\\boldsymbol\\delta^{(l)}\\mathbf h^{(l-1)\\top} $$ {#eq:backprop}\n\n"
                       "Each weight gradient in @eq:backprop is the error of its layer times the layer's input, the same form as the logistic gradient of "
                       "Lecture 1. The forward pass must keep every $\\mathbf z^{(l)}$ and $\\mathbf h^{(l)}$, which is why memory limits the batch size "
                       "on a graphics processing unit (GPU).",
                       "$$ \\boldsymbol\\delta^{(l)}=\\big(\\mathbf W^{(l+1)\\top}\\boldsymbol\\delta^{(l+1)}\\big)\\odot a'(\\mathbf z^{(l)}),\\qquad "
                       "\\frac{\\partial\\ell}{\\partial\\mathbf W^{(l)}}=\\boldsymbol\\delta^{(l)}\\mathbf h^{(l-1)\\top} $$ {#eq:backprop}\n\n"
                       "Fiecare gradient de pondere din @eq:backprop este eroarea stratului înmulțită cu intrarea lui, aceeași formă ca gradientul logistic "
                       "din cursul 1. Propagarea înainte trebuie să păstreze toate $\\mathbf z^{(l)}$ și $\\mathbf h^{(l)}$, de aceea memoria limitează mărimea "
                       "lotului pe un procesor grafic (GPU).")),
              B(frame="Backpropagation by hand",
                text=("For one real test block group with standardised income @{bp.x1|.2f} and latitude @{bp.x2|.2f}, a net with two rectified linear "
                      "units (ReLU) computes $z_1=$ @{bp.z1|.3f} and $z_2=$ @{bp.z2|.3f}, outputs $\\hat y=$ @{bp.yhat|.3f} against $y=$ @{bp.y|.3f}, and has output error "
                      "$\\delta^{(2)}=$ @{bp.d2|.3f}. Unit 2 is off ($z_2<0$), so it passes no gradient and its weights do not move; PyTorch autograd gives "
                      "the same numbers.",
                      "Pentru un grup real de locuințe din setul de test, cu venitul standardizat @{bp.x1|.2f} și latitudinea @{bp.x2|.2f}, o rețea cu două "
                      "unități liniare rectificate (ReLU) calculează $z_1=$ @{bp.z1|.3f} și $z_2=$ @{bp.z2|.3f}, produce $\\hat y=$ @{bp.yhat|.3f} față de $y=$ @{bp.y|.3f} și are eroarea "
                      "de ieșire $\\delta^{(2)}=$ @{bp.d2|.3f}. Neuronul 2 este oprit ($z_2<0$), deci nu transmite gradient, iar ponderile lui nu se modifică; "
                      "diferențierea automată din PyTorch dă aceleași cifre.")),
              B(frame="Reverse mode costs",
                text=("One backward sweep delivers all $P$ partial derivatives of a scalar loss; finite differences would need $P$ extra passes. A "
                      "gradient check on the MLP 64-64, with central differences in double precision on @{gradcheck.n_checked} random weights, finds a "
                      "largest relative error of @{gradcheck.max_rel_err|sci}.",
                      "O singură trecere înapoi dă toate cele $P$ derivate parțiale ale unei pierderi scalare; diferențele finite ar cere $P$ treceri "
                      "suplimentare. Verificarea gradientului pe MLP 64-64, cu diferențe centrale în dublă precizie pe @{gradcheck.n_checked} de ponderi "
                      "alese aleator, găsește o eroare relativă maximă de @{gradcheck.max_rel_err|sci}.")),
              B(frame="In a deep sigmoid network",
                text=("The backward signal is a product of Jacobians $\\mathbf W^\\top\\operatorname{diag}(a')$. In a 10-layer net on one real batch, the "
                      "first-layer gradient norm is @{vanish.sig.first|sci} with sigmoids against @{vanish.sig.last|sci} in the last layer; with ReLU and "
                      "He initialisation the first layer still gets @{vanish.relu_he.first|sci}.",
                      "Semnalul înapoi este un produs de jacobiene $\\mathbf W^\\top\\operatorname{diag}(a')$. Într-o rețea cu 10 straturi, pe un lot real, "
                      "norma gradientului în primul strat este @{vanish.sig.first|sci} cu sigmoide, față de @{vanish.sig.last|sci} în ultimul strat; cu "
                      "ReLU și inițializare He, primul strat primește încă @{vanish.relu_he.first|sci}."),
                chart="vanishing", cap=("Gradient norm per layer at initialisation, 10 hidden layers, mean of 5 seeds.",
                                        "Norma gradientului pe straturi la inițializare, 10 straturi ascunse, media a 5 inițializări.")),
              B(frame="Initialise so that",
                text=("If $\\operatorname{Var}(z^{(l)})=n_{l-1}\\operatorname{Var}(W)\\,\\mathbb E[(h^{(l-1)})^2]$ is to stay constant, Xavier initialisation sets "
                      "$\\operatorname{Var}(W)=2/(n_{l-1}+n_l)$ for tanh and sigmoid, and He initialisation $\\operatorname{Var}(W)=2/n_{l-1}$ for ReLU. Weights drawn "
                      "from $N(0,0.01^2)$ give a first-layer gradient of only @{vanish.relu_small.first|sci}: the network does not learn.",
                      "Pentru ca $\\operatorname{Var}(z^{(l)})=n_{l-1}\\operatorname{Var}(W)\\,\\mathbb E[(h^{(l-1)})^2]$ să rămînă constantă, inițializarea Xavier "
                      "alege $\\operatorname{Var}(W)=2/(n_{l-1}+n_l)$ pentru tanh și sigmoidă, iar inițializarea He $\\operatorname{Var}(W)=2/n_{l-1}$ pentru ReLU. "
                      "Ponderile extrase din $N(0;\\,0{,}01^2)$ dau în primul strat un gradient de doar @{vanish.relu_small.first|sci}: rețeaua nu învață.")),
          ],
          end=("Backpropagation gives exact gradients cheaply; whether they reach the first layers depends on the activation and on the scale of the "
               "initial weights.",
               "Propagarea înapoi dă ieftin gradienți exacți; dacă aceștia ajung la primele straturi depinde de activare și de scara ponderilor inițiale.")),
        B(title=("Stochastic optimisation", "Optimizarea stohastică"),
          intro=("Mini-batches, the learning rate, momentum, adaptive steps and schedules decide how fast and how reliably the network learns; we "
                 "measure each on California housing and on card fraud, with several seeds.",
                 "Mini-loturile, rata de învățare, momentul, pașii adaptivi și programele ratei decid cît de repede și cît de sigur învață rețeaua; le "
                 "măsurăm pe fiecare pe datele California și pe fraudele cu carduri, cu mai multe inițializări."),
          blocks=[
              B(frame="A mini-batch gives",
                text=("The mini-batch gradient $\\mathbf g_t=\\frac1B\\sum_{i\\in\\mathcal B_t}\\nabla\\ell_i$ is unbiased for the full gradient and its variance "
                      "falls roughly as $1/B$. With step sizes satisfying $\\sum\\eta_t=\\infty$ and $\\sum\\eta_t^2<\\infty$, stochastic gradient descent "
                      "converges to a stationary point (Robbins and Monro, 1951).",
                      "Gradientul pe mini-lot $\\mathbf g_t=\\frac1B\\sum_{i\\in\\mathcal B_t}\\nabla\\ell_i$ este nedeplasat pentru gradientul complet, iar varianța "
                      "lui scade aproximativ ca $1/B$. Cu pași care satisfac $\\sum\\eta_t=\\infty$ și $\\sum\\eta_t^2<\\infty$, coborîrea stohastică pe gradient "
                      "converge către un punct staționar (Robbins și Monro, 1951).")),
              B(frame="The learning rate decides everything",
                text=("With plain SGD, 30 epochs and batches of 128, the validation MSE is @{lrsweep.smallest_mse|.3f} at $\\eta=10^{-4}$, worse than "
                      "linear regression (@{cal.mse_lin_val|.3f}), and @{lrsweep.best_mse|.3f} at $\\eta=$ @{lrsweep.best_lr|g}; from $\\eta=$ "
                      "@{lrsweep.first_diverged|g} at least one seed diverges.",
                      "Cu SGD simplu, 30 de epoci și loturi de 128, MSE pe validare este @{lrsweep.smallest_mse|.3f} la $\\eta=10^{-4}$, mai slab decît "
                      "regresia liniară (@{cal.mse_lin_val|.3f}), și @{lrsweep.best_mse|.3f} la $\\eta=$ @{lrsweep.best_lr|g}; de la $\\eta=$ "
                      "@{lrsweep.first_diverged|g}, cel puțin o inițializare diverge."),
                chart="lr_sweep", cap=("Validation MSE after 30 epochs of SGD for nine learning rates and three seeds.",
                                       "MSE pe validare după 30 de epoci de SGD, pentru nouă rate de învățare și trei inițializări."),
                practice=("tune the learning rate first, on a log grid.", "ajustați mai întîi rata de învățare, pe o grilă logaritmică.")),
              B(frame="Small batches learn more per epoch",
                text=("After 5 epochs of Adam the validation MSE is @{batch.16.val_ep5|.3f} with batches of 16, @{batch.128.val_ep5|.3f} with 128 and "
                      "@{batch.1024.val_ep5|.3f} with 1,024: small batches make @{batch.16.steps_per_epoch} steps per epoch, large ones "
                      "@{batch.1024.steps_per_epoch}. Per second the picture reverses: 40 epochs take @{batch.16.sec|.1f}, @{batch.128.sec|.1f} and "
                      "@{batch.1024.sec|.1f} seconds.",
                      "După 5 epoci cu Adam, MSE pe validare este @{batch.16.val_ep5|.3f} cu loturi de 16, @{batch.128.val_ep5|.3f} cu 128 și "
                      "@{batch.1024.val_ep5|.3f} cu 1 024: loturile mici fac @{batch.16.steps_per_epoch} de pași pe epocă, cele mari "
                      "@{batch.1024.steps_per_epoch}. Pe secundă situația se inversează: 40 de epoci durează @{batch.16.sec|.1f}, @{batch.128.sec|.1f} și "
                      "@{batch.1024.sec|.1f} secunde."),
                chart="batch_size", cap=("Validation MSE by epoch (left) and by gradient step (right) for three batch sizes.",
                                         "MSE pe validare pe epoci (stînga) și pe pași de gradient (dreapta), pentru trei mărimi ale lotului.")),
              B(frame="Momentum averages",
                text=("Momentum keeps a velocity $\\mathbf v_{t+1}=\\beta\\mathbf v_t+\\mathbf g_t$ and steps along it. Zig-zag components cancel, the "
                      "consistent direction accumulates, and on a quadratic the best momentum improves the contraction from $(\\kappa-1)/(\\kappa+1)$ to "
                      "$(\\sqrt\\kappa-1)/(\\sqrt\\kappa+1)$ (Polyak, 1964).",
                      "Metoda momentului păstrează o viteză $\\mathbf v_{t+1}=\\beta\\mathbf v_t+\\mathbf g_t$ și face pași în direcția ei. Componentele în "
                      "zigzag se anulează, direcția constantă se acumulează, iar pe o funcție pătratică momentul optim îmbunătățește contracția de la "
                      "$(\\kappa-1)/(\\kappa+1)$ la $(\\sqrt\\kappa-1)/(\\sqrt\\kappa+1)$ (Polyak, 1964).")),
              B(frame="Adaptive methods give",
                text=("AdaGrad divides each weight's step by the root of the sum of its squared gradients, which only grows; root mean square propagation "
                      "(RMSProp) replaces the sum by an exponential average, $s_t=\\rho s_{t-1}+(1-\\rho)g_t^2$, so the step adapts to the current region.",
                      "AdaGrad împarte pasul fiecărei ponderi la rădăcina sumei pătratelor gradienților ei, care doar crește; propagarea cu medie pătratică "
                      "(RMSProp) înlocuiește suma cu o medie exponențială, $s_t=\\rho s_{t-1}+(1-\\rho)g_t^2$, astfel încît pasul se adaptează regiunii curente.")),
              B(frame="Adam = momentum",
                text=("Adam keeps exponential averages of the gradient and of its square, corrects them for starting at zero and steps by their ratio:",
                      "Adam păstrează medii exponențiale ale gradientului și ale pătratului său, le corectează pentru pornirea de la zero și face un pas egal cu raportul lor:"),
                after=("$$ \\hat m_t=\\frac{m_t}{1-\\beta_1^t},\\quad \\hat v_t=\\frac{v_t}{1-\\beta_2^t},\\quad \\theta_{t+1}=\\theta_t-\\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon} $$ {#eq:adam}\n\n"
                       "With the defaults $\\beta_1=0.9$, $\\beta_2=0.999$, the step @eq:adam is roughly bounded by $\\eta$ per weight. Adam with decoupled weight "
                       "decay (AdamW) shrinks the weights directly instead of adding the penalty to the gradient (Loshchilov and Hutter, 2019).",
                       "$$ \\hat m_t=\\frac{m_t}{1-\\beta_1^t},\\quad \\hat v_t=\\frac{v_t}{1-\\beta_2^t},\\quad \\theta_{t+1}=\\theta_t-\\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon} $$ {#eq:adam}\n\n"
                       "Cu valorile implicite $\\beta_1=0{,}9$, $\\beta_2=0{,}999$, pasul @eq:adam este aproximativ limitat la $\\eta$ pentru fiecare pondere. "
                       "Adam cu penalizarea ponderilor decuplată (AdamW) micșorează direct ponderile, în loc să adauge penalizarea la gradient "
                       "(Loshchilov și Hutter, 2019).")),
              B(frame="On California housing, all tuned",
                text=("Each optimiser gets the best of three learning rates on validation, then 5 new seeds. After 10 epochs the validation MSE is "
                      "@{opt.sgd.val_ep10|.3f} for SGD, @{opt.momentum.val_ep10|.3f} with momentum, @{opt.rmsprop.val_ep10|.3f} for RMSProp, "
                      "@{opt.adam.val_ep10|.3f} for Adam and @{opt.adamw.val_ep10|.3f} for AdamW; linear regression stays at @{cal.mse_lin_val|.3f}.",
                      "Fiecare optimizator primește cea mai bună dintre trei rate de învățare pe validare, apoi 5 inițializări noi. După 10 epoci, MSE pe "
                      "validare este @{opt.sgd.val_ep10|.3f} pentru SGD, @{opt.momentum.val_ep10|.3f} cu moment, @{opt.rmsprop.val_ep10|.3f} pentru RMSProp, "
                      "@{opt.adam.val_ep10|.3f} pentru Adam și @{opt.adamw.val_ep10|.3f} pentru AdamW; regresia liniară rămîne la @{cal.mse_lin_val|.3f}."),
                chart="optimizers", cap=("Validation MSE per epoch of five optimisers at their tuned learning rates, mean of 5 seeds.",
                                         "MSE pe validare pe epoci pentru cinci optimizatori, la ratele de învățare ajustate, media a 5 inițializări.")),
              B(frame="Across seeds, Adam and AdamW",
                text=("The test MSE over 5 seeds is @{opt.sgd.test_mean|.3f} (standard deviation @{opt.sgd.test_sd|.3f}) for SGD, @{opt.adam.test_mean|.3f} "
                      "(@{opt.adam.test_sd|.3f}) for Adam and @{opt.adamw.test_mean|.3f} (@{opt.adamw.test_sd|.3f}) for AdamW. The paired t-test of SGD "
                      "against Adam gives $t=$ @{opt_test.t|.2f} and $p=$ @{opt_test.p|.3f}: the mean gain is not established with 5 seeds, the stability "
                      "gain is plain to see.",
                      "MSE pe test pe 5 inițializări este @{opt.sgd.test_mean|.3f} (abaterea standard @{opt.sgd.test_sd|.3f}) pentru SGD, @{opt.adam.test_mean|.3f} "
                      "(@{opt.adam.test_sd|.3f}) pentru Adam și @{opt.adamw.test_mean|.3f} (@{opt.adamw.test_sd|.3f}) pentru AdamW. Testul t pe perechi, SGD față "
                      "de Adam, dă $t=$ @{opt_test.t|.2f} și $p=$ @{opt_test.p|.3f}: cîștigul mediu nu este demonstrat cu 5 inițializări, dar cîștigul de "
                      "stabilitate este evident."),
                chart="optimizer_seeds", cap=("Test MSE of five seeds per optimiser; bars are means; dashed: linear regression.",
                                              "MSE pe test pentru cinci inițializări la fiecare optimizator; barele sînt mediile; linia punctată: regresia liniară.")),
              B(frame="Learning-rate schedules",
                text=("Step decay multiplies the rate by $\\gamma$ every $s$ steps, cosine annealing follows $\\eta_t=\\tfrac{\\eta_0}{2}(1+\\cos\\pi t/T)$ "
                      "(Loshchilov and Hutter, 2017), and the one-cycle schedule warms up linearly and then decays (Smith and Topin, 2019). Warm-up protects "
                      "the first steps, when the optimiser's statistics are still unreliable.",
                      "Scăderea în trepte înmulțește rata cu $\\gamma$ la fiecare $s$ pași, descreșterea cosinus urmează $\\eta_t=\\tfrac{\\eta_0}{2}(1+\\cos\\pi t/T)$ "
                      "(Loshchilov și Hutter, 2017), iar programul cu un singur ciclu crește liniar, apoi scade (Smith și Topin, 2019). Încălzirea protejează "
                      "primii pași, cînd statisticile optimizatorului nu sînt încă fiabile.")),
              B(frame="Decaying the learning rate",
                text=("With SGD and momentum at a peak rate of 0.01, the final validation MSE is @{sched.const.val_final|.3f} with a constant rate, "
                      "@{sched.step.val_final|.3f} with step decay, @{sched.cosine.val_final|.3f} with cosine annealing and @{sched.onecycle.val_final|.3f} "
                      "with one-cycle; on the test set @{sched.const.test|.3f} against @{sched.cosine.test|.3f} for constant and cosine.",
                      "Cu SGD și moment, la o rată maximă de 0,01, MSE final pe validare este @{sched.const.val_final|.3f} cu rată constantă, "
                      "@{sched.step.val_final|.3f} cu scădere în trepte, @{sched.cosine.val_final|.3f} cu descreștere cosinus și @{sched.onecycle.val_final|.3f} "
                      "cu un singur ciclu; pe setul de test, @{sched.const.test|.3f} față de @{sched.cosine.test|.3f} pentru rata constantă și cosinus."),
                chart="schedules", cap=("Learning-rate schedules (left) and validation MSE (right), mean of 3 seeds.",
                                        "Programele ratei de învățare (stînga) și MSE pe validare (dreapta), media a 3 inițializări.")),
              B(frame="On card fraud, the optimiser",
                text=("With only @{fraud.fr_train} frauds in the first 60 % of the two days, the area under the precision-recall curve (PR-AUC) on the "
                      "validation hours after one epoch is @{fraud.sgd.ap_ep1|.3f} for plain SGD, @{fraud.momentum.ap_ep1|.3f} with momentum and "
                      "@{fraud.adam.ap_ep1|.3f} with Adam; on the test hours the three reach @{fraud.sgd.ap_test|.3f}, @{fraud.momentum.ap_test|.3f} and "
                      "@{fraud.adam.ap_test|.3f}, against @{fraud.ap_logit_test|.3f} for logistic regression.",
                      "Cu doar @{fraud.fr_train} de fraude în primele 60 % din cele două zile, aria de sub curba precizie-sensibilitate (PR-AUC) pe orele de "
                      "validare, după o epocă, este @{fraud.sgd.ap_ep1|.3f} pentru SGD simplu, @{fraud.momentum.ap_ep1|.3f} cu moment și @{fraud.adam.ap_ep1|.3f} "
                      "cu Adam; pe orele de test, cele trei ajung la @{fraud.sgd.ap_test|.3f}, @{fraud.momentum.ap_test|.3f} și @{fraud.adam.ap_test|.3f}, "
                      "față de @{fraud.ap_logit_test|.3f} pentru regresia logistică."),
                chart="fraud_optimizers", cap=("Validation PR-AUC per epoch on card fraud, three optimisers, mean and range of 5 seeds.",
                                               "PR-AUC pe validare pe epoci la fraudele cu carduri, trei optimizatori, media și intervalul a 5 inițializări.")),
          ],
          end=("Tune the learning rate first; momentum and adaptive methods converge faster and more reliably; a decaying schedule lowers the final "
               "error; and every comparison needs several seeds.",
               "Ajustați mai întîi rata de învățare; momentul și metodele adaptive converg mai repede și mai sigur; un program descrescător reduce "
               "eroarea finală; iar orice comparație cere mai multe inițializări.")),
        B(title=("Case study: value at risk with a quantile network", "Studiu de caz: valoarea la risc cu o rețea cuantilică"),
          intro=("The same network, trained with another loss, becomes a risk model: with the pinball loss an MLP forecasts the 1 % quantile of the "
                 "next-day S&P 500 return, and we backtest it from 2020.",
                 "Aceeași rețea, antrenată cu altă pierdere, devine un model de risc: cu pierderea pinball, un MLP prognozează cuantila de 1 % a "
                 "randamentului S&P 500 din ziua următoare, iar noi îl testăm retroactiv începînd din 2020."),
          blocks=[
              B(frame="The pinball loss",
                text=("The VaR is minus the conditional $\\tau$-quantile of the next-day return, and the pinball loss $\\rho_\\tau(u)=u(\\tau-\\mathbb 1\\{u<0\\})$ "
                      "is minimised by that quantile (Koenker and Bassett, 1978). Our MLP 32-32 uses the last return, its absolute value, the 5-day mean "
                      "and the 5-, 22- and 66-day volatility; it is trained from @{var.train_start|date} to 2016, stopped on 2017-2019 and averaged over "
                      "@{var.seeds} seeds. The benchmarks are 250-day historical simulation and an exponentially weighted moving average (EWMA) volatility "
                      "with a normal quantile.",
                      "VaR este minus cuantila condiționată $\\tau$ a randamentului de a doua zi, iar pierderea pinball $\\rho_\\tau(u)=u(\\tau-\\mathbb 1\\{u<0\\})$ "
                      "are minimul în această cuantilă (Koenker și Bassett, 1978). MLP 32-32 folosește ultimul randament, valoarea lui absolută, media pe 5 "
                      "zile și volatilitatea pe 5, 22 și 66 de zile; este antrenat de la @{var.train_start|date} pînă în 2016, oprit pe baza perioadei "
                      "2017--2019 și mediat pe @{var.seeds} inițializări. Reperele sînt simularea istorică pe 250 de zile și volatilitatea calculată cu o "
                      "medie mobilă ponderată exponențial (EWMA), cu cuantila normală.")),
              B(frame="Through the 2020 pandemic",
                text=("Over @{var.n_test|,d} test days, from @{var.test_start|date} to @{var.test_end|date}, a correct 99 % VaR should be breached about "
                      "@{var.expected|.1f} times. The quantile MLP is breached @{var.nn.viol} times, historical simulation @{var.hs.viol} and EWMA "
                      "@{var.ewma.viol}; the MLP's VaR widens within days when volatility jumps, as in March 2020.",
                      "Pe @{var.n_test|,d} de zile de test, de la @{var.test_start|date} la @{var.test_end|date}, un VaR corect de 99 % ar trebui depășit de "
                      "circa @{var.expected|.1f} ori. MLP cuantilic este depășit de @{var.nn.viol} ori, simularea istorică de @{var.hs.viol} ori, iar EWMA de "
                      "@{var.ewma.viol} ori; VaR-ul MLP crește în cîteva zile cînd volatilitatea sare, ca în martie 2020."),
                chart="var_backtest", cap=("S&P 500 daily log returns in the test period with the 99 % VaR of the quantile MLP and of historical simulation; red: MLP violations.",
                                           "Randamentele logaritmice zilnice S&P 500 în perioada de test, cu VaR de 99 % al MLP cuantilic și al simulării istorice; roșu: depășirile VaR-ului MLP.")),
              B(frame="The Kupiec test",
                text=("Under a correct VaR the violations are Bernoulli with probability $\\tau$; the Kupiec likelihood-ratio statistic gives "
                      "$p=$ @{var.nn.p|.3f} for the MLP, @{var.hs.p|.3f} for historical simulation and @{var.ewma.p|.3f} for EWMA. The pinball losses, the "
                      "proper score for quantiles, are @{var.nn.pinball|.4f}, @{var.hs.pinball|.4f} and @{var.ewma.pinball|.4f}.",
                      "Pentru un VaR corect, depășirile sînt de tip Bernoulli cu probabilitatea $\\tau$; statistica raportului de verosimilitate a lui Kupiec dă "
                      "$p=$ @{var.nn.p|.3f} pentru MLP, @{var.hs.p|.3f} pentru simularea istorică și @{var.ewma.p|.3f} pentru EWMA. Pierderile pinball, scorul "
                      "potrivit pentru cuantile, sînt @{var.nn.pinball|.4f}, @{var.hs.pinball|.4f} și @{var.ewma.pinball|.4f}.")),
              B(frame="Where training runs",
                text=("On this course's laptop one Adam step of an MLP 2048-2048 takes @{timing.cpu_ms|.1f} ms on the central processing unit (CPU) and "
                      "@{timing.mps_ms|.1f} ms on its GPU through Metal Performance Shaders (MPS), @{timing.speedup|.1f} times faster. The small tabular "
                      "nets of Lectures 1 to 3 are faster on one CPU thread: copying tiny batches costs more than it saves.",
                      "Pe laptopul cursului, un pas Adam pentru un MLP 2048-2048 durează @{timing.cpu_ms|.1f} ms pe procesorul central (CPU) și "
                      "@{timing.mps_ms|.1f} ms pe GPU prin Metal Performance Shaders (MPS), de @{timing.speedup|.1f} ori mai repede. Rețelele tabelare mici "
                      "din cursurile 1--3 sînt mai rapide pe un singur fir al procesorului: copierea loturilor mici costă mai mult decît economisește.")),
          ],
          end=("The loss defines the product: with the pinball loss the MLP gives a 99 % VaR that passes the Kupiec test while both standard "
               "benchmarks fail it on the same days.",
               "Pierderea definește produsul: cu pierderea pinball, MLP dă un VaR de 99 % care trece testul Kupiec, în timp ce ambele repere standard "
               "îl pică pe aceleași zile.")),
    ],
)

if __name__ == "__main__":
    sys.exit(write_pages(2, HERE, PAGE))
