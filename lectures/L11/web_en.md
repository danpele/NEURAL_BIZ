<!--
Web version of Lecture 11 (EN). Built into web/L11.html by lectures/common/build_web_lecture.py (format: see its docstring).
Frame hashes: python3 build_web_lecture.py 11 --frames
-->

## @intro

A bank that rejects a loan applicant with a neural network must be able to say why. The European Union (EU) Artificial Intelligence (AI) Act classifies credit scoring of natural persons as high-risk, the General Data Protection Regulation (GDPR) protects people against decisions taken solely by machines, and model-risk supervisors ask for models that can be validated and monitored. This lecture shows how to open the black box with the methods of explainable artificial intelligence (XAI): Shapley additive explanations (SHAP), local interpretable model-agnostic explanations (LIME), gradient methods such as integrated gradients (IG), partial dependence and counterfactual explanations.

Every method is computed on real data. The credit case uses the home-equity loan data set (HMEQ): 5,960 loans, 19.9 % of them in default, scored by a multilayer perceptron (MLP), a logistic regression and extreme gradient boosting (XGBoost). The fraud case uses two days of credit-card transactions released by the Université Libre de Bruxelles (ULB): a network trained on the first day is explained on the second. The housing case uses the 1990 California census. The second half of the lecture asks a question that is often skipped: are the explanations faithful to the model, and do they stay the same when nothing important changes?

@goals
- **Explain** why banks, insurers and regulators require explanations, and classify explanation methods.
- **Compute and read** SHAP and LIME explanations with their full formulas, for one customer and for a portfolio.
- **Use** saliency, integrated gradients and SmoothGrad, and check the completeness axiom on a real fraud model.
- **Test** whether explanations agree, are faithful and stay stable across seeds, samples and time.
- **Write** a counterfactual explanation that a rejected applicant can act on.

# Reasons for explaining neural networks

Explanations are required by law and by good risk management, but an explanation is itself a model of the model. This section sets out the rules, a taxonomy of methods and the credit case that runs through the lecture.

## Regulators require explanations of automated decisions {h=c8616903}

Three texts frame the European practice. The AI Act (Regulation 2024/1689) lists creditworthiness assessment of natural persons among high-risk uses (Annex III), requires providers to make the system transparent to those who deploy it (article 13) and gives a person affected by a decision the right to an explanation of the role the system played (article 86). The GDPR, article 22 and recital 71, adds safeguards against decisions based solely on automated processing. The European Banking Authority (EBA) guidelines on loan origination and monitoring (2020) require that models used for credit decisions can be explained. Outside the EU, the Federal Reserve Supervisory Guidance on Model Risk Management (SR 11-7) asks for conceptual soundness, independent validation and ongoing monitoring of every model.

There are business reasons too: explanations reveal leakage (a feature that should not be there), help credit officers trust a score and let the bank tell a customer why the answer was no. The caveat runs through the whole lecture: an explanation that changes when nothing important has changed is a compliance risk, not a safeguard.

@practice an explanation that changes when nothing important changed is a compliance risk, not a safeguard.

## A taxonomy of explanation methods {h=1f4d68c4}

Methods differ in scope and in what they need. Global methods describe the model over the whole population (coefficients, mean absolute SHAP values, partial dependence, permutation importance); local methods explain one decision (a SHAP waterfall, LIME, gradients, a counterfactual). Model-agnostic methods only call the model (Kernel SHAP, LIME, partial dependence); model-specific methods use its structure (TreeSHAP for trees, gradients for networks). Finally, post-hoc methods explain a model that is already trained, while interpretable models such as a scorecard are transparent by design. Rudin (2019) argues that in high-stakes decisions the second route should be tried first; we will see how much accuracy it costs on our data.

## Credit case study: comparison of three scoring models {h=8401cd9c}

The network has two hidden layers of 64 and 32 rectified linear units (ReLU):

$$ \mathbf h_1=\mathrm{ReLU}(\mathbf W_1\mathbf x+\mathbf b_1),\quad \mathbf h_2=\mathrm{ReLU}(\mathbf W_2\mathbf h_1+\mathbf b_2),\quad f(\mathbf x)=\mathbf w_3^{\top}\mathbf h_2+b_3,\quad \mathrm{PD}(\mathbf x)=\sigma(f(\mathbf x)) $$ {#eq:mlp}

where $\mathbf x$ holds 18 standardised features (ten numeric variables imputed with the training median, a flag for a missing debt-to-income ratio, DEBTINC, the loan reason and the job), $f$ is the log-odds of default and PD the probability of default. The loans are split 60/20/20 with stratification; the scalers and medians are fitted on the training part only.

| Model (test) | AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic regression | 0.909 | 0.785 | 0.080 |
| MLP, mean of 10 seeds | 0.935 | 0.842 | 0.068 |
| XGBoost, mean of 10 seeds | 0.959 | 0.889 | 0.058 |

AUC is the area under the receiver operating characteristic (ROC) curve and PR-AUC the area under the precision-recall curve; a random score has a PR-AUC equal to the default rate, 0.200. The network is clearly better than the logistic scorecard, and the boosted trees are better still. For the logistic model the coefficients are the explanation; for the network the explanation has to be computed.

@practice the logistic coefficients are the explanation; for the MLP the explanation must be computed.

# Shapley values and SHAP

Shapley values come from cooperative game theory: they split a total payoff among players in the only way that satisfies four fairness axioms. SHAP applies them to the features of one prediction.

## Shapley values: a fair split of the prediction among features {h=7ac37604}

For $d$ features and a value function $v(S)$ that says what the model predicts when only the features in $S$ are known, the Shapley value of feature $j$ is its marginal contribution averaged over all orders in which the features could be added:

$$ \phi_j=\sum_{S\subseteq\{1,\dots,d\}\setminus\{j\}}\frac{|S|!\,(d-|S|-1)!}{d!}\big[v(S\cup\{j\})-v(S)\big] $$ {#eq:shapley}

The weights are the share of orderings in which $j$ arrives right after the members of $S$. Shapley (1953) proved that this is the only split with four properties: efficiency (the contributions add up to $v$ of all features minus $v$ of none), symmetry (two features that contribute equally get the same share), dummy (a feature that never changes $v$ gets zero) and additivity. The price is computational: with 18 features there are $2^{18}$ coalitions for every applicant.

@practice efficiency is what a credit officer needs: the parts add up exactly to the score.

## SHAP: the value of a coalition of features {h=7d6fc27f}

SHAP (Lundberg & Lee, 2017) defines the value of a coalition as the expected model output when the missing features are drawn from a background sample $\mathcal B$ of 100 training applicants:

$$ v(S)=\mathbb E_{\mathbf x_{\bar S}\sim\mathcal B}\big[f(\mathbf x_S,\mathbf x_{\bar S})\big],\qquad f(\mathbf x)=\phi_0+\sum_{j=1}^{d}\phi_j(\mathbf x),\quad \phi_0=\mathbb E_{\mathcal B}[f] $$ {#eq:shap}

We explain the log-odds, where effects add. On 400 test applicants the largest additivity error is $2.9\cdot10^{-6}$, rounding error only. For a linear model the formula has a closed form, $\phi_j=\beta_j(x_j-\mathbb E x_j)$ (linear SHAP). The replacement of missing features is interventional, that is, independent of the known ones; when features are correlated this can create applicants that do not exist, a limitation to state in the model documentation.

@practice the background sample defines the "average applicant"; document it like any model input.

## Kernel SHAP and TreeSHAP: how SHAP is computed {h=92529ab6}

Kernel SHAP fits a weighted linear regression of the coalition values on coalition indicators $\mathbf z\in\{0,1\}^d$, with the Shapley kernel

$$ \pi(\mathbf z)=\frac{d-1}{\binom{d}{|\mathbf z|}\,|\mathbf z|\,(d-|\mathbf z|)} $$ {#eq:kernel}

and the weighted least-squares solution equals the Shapley values (the proof is in Appendix A of the slides). We sample 600 coalitions per applicant; the 400 applicants take 16 seconds. For tree ensembles TreeSHAP computes the exact values in polynomial time (Lundberg et al., 2020), which is how we explain XGBoost. Kernel SHAP has sampling noise, so its seed and sample size belong in the documentation.

## Global importance: the missing debt-to-income ratio dominates {h=07f158a6}

The mean absolute SHAP value of each feature ranks the drivers of the portfolio. All three models put the flag DEBTINC_missing first; for the network it is followed by DELINQ, the number of delinquent credit lines, and by the job category "Other". The Spearman correlation of the importances is 0.55 between the MLP and XGBoost and 0.79 between the MLP and the logistic regression.

The reason is in the data: the debt-to-income ratio is missing for 21.3 % of the loans, and the default rate is 62.0 % when it is missing against 8.6 % when it is reported. The network has learned that an incomplete file is a strong signal of risk.

@chart shap_global
Mean absolute SHAP value of the ten most important features for the MLP (Kernel SHAP), XGBoost (TreeSHAP) and the logistic regression (linear SHAP), on 400 test applicants.

@practice an explanation can reveal that a model scores the paperwork rather than the borrower; decide whether that is acceptable.

## The beeswarm: direction and spread of every effect {h=e5a387e1}

The beeswarm shows each applicant as a dot, placed at its SHAP value and coloured by the rank of the feature value. It adds what the bar chart hides: the direction of the effect and its spread. High values of DELINQ and of DEROG, the number of major derogatory reports, push the risk up, as credit theory expects; a reported debt-to-income ratio lowers it for most applicants.

@chart shap_beeswarm
SHAP values of the ten most important features for 400 test applicants; colour from blue (low value) to red (high value).

## Local explanation of a rejected applicant: SHAP waterfall {h=58f4332a}

For one test applicant (row 2411 of the data) the network gives a PD of 65.6 %, log-odds 0.648. The waterfall starts from the average log-odds $\phi_0=-3.613$ and adds the contributions one by one until it reaches the applicant's score: $-3.613+4.261=0.648$. The largest pushes are DEBTINC_missing ($+3.13$), the job category "Other" ($+1.74$) and DELINQ ($-0.71$, this applicant has no delinquent lines).

@chart shap_waterfall
Contributions of the eight largest features (values in original units) and of the remaining ten, from the average log-odds to the applicant's log-odds.

@practice the adverse-action notice lists the top reasons in this order, in words the customer understands.

## Dependence plot: the network learned the missing-value signal {h=5fea9c87}

Plotting the SHAP value of DEBTINC against DEBTINC shows a second effect of imputation. Applicants whose ratio is missing all sit at the training median and their SHAP value for DEBTINC is small: the risk of these files is carried by the missing flag, whose mean SHAP value is $+3.13$. Without the flag, the network would have to guess the risk from the imputed value, and the explanation would blame a debt-to-income ratio the applicant never reported.

@chart shap_dependence
SHAP value of DEBTINC against the reported ratio (blue) and for files with a missing ratio, imputed at the median (red).

@practice imputation choices change explanations; keep a missing-value flag so that the reason can be named.

# LIME, counterfactuals and global effects

LIME explains a prediction by a simple model fitted around it; counterfactuals say what would have to change; partial dependence shows the average effect of one feature.

## LIME: a simple model that is faithful near one point {h=c8da6820}

LIME (Ribeiro, Singh & Guestrin, 2016) perturbs the applicant, weights the perturbations by their distance and fits a linear model to the network's answers:

$$ \xi(\mathbf x)=\arg\min_{g\in\mathcal G}\sum_{\mathbf z}\pi_{\mathbf x}(\mathbf z)\big[f(\mathbf z)-g(\mathbf z)\big]^2+\Omega(g),\qquad \pi_{\mathbf x}(\mathbf z)=\exp\!\Big(-\frac{\|\mathbf z-\mathbf x\|^2}{\sigma^2}\Big) $$ {#eq:lime}

where $\mathcal G$ is the set of linear models, $\Omega$ a complexity penalty and $\sigma$ the kernel width, $0.75\sqrt d=3.18$ by default. LIME gives a local slope, not a split of the prediction, so its coefficients do not add up to $f(\mathbf x)$. For our applicant LIME's top feature is DELINQ, while SHAP's is DEBTINC_missing: a binary flag is far from a linear function, and the two methods answer different questions.

@practice LIME answers "what moves the score near this customer?", SHAP answers "why is this customer far from average?".

## LIME and SHAP for the same applicant {h=85d8d0f4}

To compare the two on one scale we multiply each LIME slope by the standardised value of the feature, which gives a contribution relative to the average applicant. The Spearman correlation of the absolute contributions is 0.80: the two methods agree on most of the ranking but not on the top feature.

@chart lime_vs_shap
Kernel SHAP contribution and LIME slope × standardised value for the eight largest SHAP features of the same applicant (LIME with 5,000 samples).

## Sensitivity of LIME to its settings {h=42e424e9}

We repeated LIME 30 times with different seeds. With 500 samples the top feature was the same in 73 % of the runs and the Jaccard overlap of the top-5 sets was 0.72; with 5,000 samples, in 100 % of the runs, overlap 1.00. The coefficient of variation of the five largest slopes falls from 10 % to 2 %. The kernel width matters as much: the slope of the top feature moves between $0.001$ and $2.042$ as $\sigma$ goes from 0.5 to 10.

@chart lime_instability
Left: LIME slopes of five features over 30 runs with 500 and with 5,000 samples. Right: the same slopes as a function of the kernel width (dashed line: default).

@practice fix and report the seed, sample size and kernel width of every LIME explanation that reaches a customer or an auditor.

## Counterfactual explanations: minimal changes that reverse the decision {h=02180eed}

A counterfactual (Wachter, Mittelstadt & Russell, 2017) looks for the closest applicant whom the model would accept, changing only features the applicant can act on:

$$ \mathbf x'=\arg\min_{\mathbf x'}\ \lambda\,\big(f(\mathbf x')-t\big)_+^{2}+\sum_{j\in\mathcal A}\frac{|x'_j-x_j|}{\mathrm{MAD}_j} $$ {#eq:cf}

where $t=-0.5$ is a target log-odds below the 0.5 PD threshold, $\mathcal A$ the actionable features (LOAN, DEBTINC, NINQ, DELINQ, DEROG, CLNO, MORTDUE) and the median absolute deviation (MAD) puts all distances on a comparable scale. Raw values are kept non-negative and counts are rounded. For our applicant the optimisation lowers the outstanding mortgage from 69,115 to 57,456.5, the debt-to-income ratio from 34.7 to 33.4 and changes the loan amount from 14,500 to 15,700; the PD falls from 65.6 % to 39.2 %.

A second, simpler counterfactual is more useful: if the applicant reports the debt-to-income ratio (at the median value, 34.7), the PD becomes 5.1 %. The explanation points to the missing document, not to the applicant's finances.

@practice "bring the missing income document" is a better answer to the customer than any list of SHAP values.

## Global effects: partial dependence and ICE {h=98fd8d15}

The partial dependence of feature $j$ averages the prediction over the data while $j$ is set to a grid value; individual conditional expectation (ICE) keeps one curve per observation (Friedman, 2001; Goldstein et al., 2015):

$$ \mathrm{PD}_j(z)=\frac1n\sum_{i=1}^{n}f(z,\mathbf x_{i,-j}),\qquad \mathrm{ICE}_{i,j}(z)=f(z,\mathbf x_{i,-j}) $$ {#eq:pdp}

On California housing the MLP explains house values far better than a linear regression (test $R^2$ 0.819 against 0.620). The partial dependence on median income rises from 1.40 to 3.60 (hundreds of thousands of US dollars, USD) over the income grid, and flattens at the top, where the data source caps the value at 5. The ICE curves fan out: at the top of the grid their standard deviation is 0.95, so the same income effect is not shared by all neighbourhoods. A partial dependence plot assumes that income can change while everything else stays fixed; with correlated features (income and rooms per household, correlation 0.33) it averages over combinations that may be rare.

@chart pdp_ice
Partial dependence (orange, mean over 1,000 test block groups) and ICE curves (60 block groups) of the predicted house value on median income.

# Gradient methods for neural networks

A network gives its gradients for free: one backward pass yields the sensitivity of the output to every input. This section explains a fraud model with gradients.

## Fraud case study: a network trained on day 1 and explained on day 2 {h=79803053}

The credit-card data cover two days of September 2013. The variables V1–V28 are principal component analysis (PCA) components released by the data owner, plus the transaction amount; frauds are 0.15 % of the transactions of day 2 once the sampling weights restore the full data. We train an MLP on day 1, with early stopping on the weighted validation PR-AUC, and test it on day 2: PR-AUC 0.815 against 0.742 for the logistic regression.

The simplest explanations are the saliency $|\partial f/\partial x_j|$ (Simonyan, Vedaldi & Zisserman, 2014) and the gradient × input $(x_j-x'_j)\,\partial f/\partial x_j$, where $\mathbf x'$ is a baseline, here the mean legitimate transaction of day 1. Both suffer from saturation: in a flat part of $f$ the gradient is close to zero even for a decisive feature.

@practice a fraud analyst gets a ranked list of the inputs that drove the alert, in milliseconds per transaction.

## Integrated gradients and the completeness of attributions {h=36245f36}

Integrated gradients (Sundararajan, Taly & Yan, 2017) integrate the gradient along the straight path from the baseline to the input:

$$ \mathrm{IG}_j(\mathbf x)=(x_j-x'_j)\int_0^1\frac{\partial f\big(\mathbf x'+\alpha(\mathbf x-\mathbf x')\big)}{\partial x_j}\,d\alpha $$ {#eq:ig}

By the fundamental theorem of calculus for paths the attributions add up to $f(\mathbf x)-f(\mathbf x')$ (completeness). IG also satisfies sensitivity and implementation invariance, and it is the Aumann–Shapley value of the continuous game along the path. In practice the integral is a Riemann sum with $m$ steps. The baseline defines the question: with the mean legitimate transaction we ask "why is this not a normal payment?".

@practice write the baseline into the model documentation: a different baseline is a different question.

## Completeness holds once the integral is computed accurately {h=96dd603b}

On the 211 frauds of day 2 the median relative completeness error is 3.7 % with 4 steps and 0.18 % with 64 steps; it falls roughly as $1/m^2$, the rate of the midpoint rule.

@chart ig_completeness
Median relative completeness error of integrated gradients against the number of Riemann steps (log scales).

## SmoothGrad and concept-level explanations {h=bf442476}

SmoothGrad (Smilkov et al., 2017) averages gradients over noisy copies of the input, which removes the jitter of a ReLU network. Testing with concept activation vectors (TCAV) (Kim et al., 2018) asks a different question: does a human concept, learned as a direction $\mathbf v_C$ in a hidden layer, move the output? Concepts must be labelled, which is not possible with anonymised PCA components; plan them when the data are collected.

## Four explanation methods for one fraudulent transaction {h=5facdc1a}

For the median-score fraud of day 2 (fraud score 99.9 %), Kernel SHAP and integrated gradients both name V14 as the main feature, while saliency names V12: saliency measures how sensitive the score is, not how far the transaction is from normal.

@chart fraud_local
Signed attributions of saliency, gradient × input, integrated gradients and Kernel SHAP for one fraud, as shares of the total absolute attribution.

# Agreement, faithfulness and stability of explanations

Three tests should accompany every explanation: agreement between methods, faithfulness to the model and stability when nothing important changes.

## Agreement between explanation methods {h=2b4759c3}

On the same 211 frauds we compare the global importances of seven methods. The Spearman correlation is 0.85 between IG and SHAP, 0.83 between saliency and SHAP, 0.82 between LIME and SHAP, and only 0.58 between permutation importance and SHAP; all methods put V12 first. The disagreement is informative: permutation importance (Breiman, 2001) measures how much the model needs a feature to stay accurate, attributions measure how much it uses the feature for the frauds.

@chart method_agreement
Spearman correlation of the global importances (mean absolute attribution) of seven explanation methods.

@practice disagreement is information: permutation importance measures what the model needs, attributions what it uses.

## Faithfulness of explanations: deletion test {h=f2550be3}

A faithful explanation names the features whose removal changes the prediction. We reset the top-$k$ features of each fraud to the baseline and follow the mean fraud score; the area over the perturbation curve (AOPC) summarises the fall (Samek et al., 2017). In points of fraud score: IG 56.2, SHAP 58.4, LIME 56.1, saliency 48.0, random order 3.9. SHAP, IG and LIME are equally faithful here, saliency less so, and every method is far better than chance.

@chart deletion_curve
Mean fraud score of the day-2 frauds after resetting the $k$ top-ranked features to the baseline, for four methods and a random order.

## Local robustness of explanations to small input changes {h=76396327}

We add Gaussian noise with a standard deviation of 0.05 to 60 frauds, ten times, and measure the relative change of the attribution per unit of input change (Alvarez-Melis & Jaakkola, 2018). The medians are 0.000 for saliency, 0.028 for gradient × input, 0.030 for IG and 0.036 for SmoothGrad. Saliency does not move because inside one linear region of a ReLU network the gradient is constant: robustness to small noise comes with abrupt jumps between regions.

@chart local_robustness
Relative change of the explanation per unit of input change for four gradient methods (60 frauds × 10 perturbations).

## The Rashomon effect: equally accurate models with different explanations {h=f6c7f234}

We retrained the credit models with ten seeds. The test AUC of the network stays between 0.929 and 0.941, but the order of the features moves: the mean pairwise Kendall $\tau$ of the global importances is 0.83 for the MLP and 0.97 for XGBoost. For a single applicant the top feature is the same in 78 % of the seed pairs for the network and 83 % for XGBoost. Many models fit the data equally well, and they do not all tell the same story.

@chart seed_stability
Rank of each feature by mean absolute SHAP value for ten MLPs and ten XGBoost models that differ only in the training seed.

## Explanations over time: day 1 against day 2 {h=8690a192}

Distribution shift can change explanations even when the model is fixed. We compare the integrated gradients of the 70 held-out frauds of day 1 with those of the 211 frauds of day 2. The Spearman correlation of the importances is 0.98, the Jensen–Shannon (JS) divergence, built on the Kullback–Leibler (KL) divergence, is 0.003, and four of the top five features are shared: one day later, the explanation of this model is stable. Our group studies the same question over longer horizons and with graph features (see the research slide).

@chart time_shift
Share of the total absolute integrated gradients by feature, for held-out frauds of day 1 and for the frauds of day 2.

@practice monitor the distribution of attributions monthly, next to the score distribution; a drift in reasons can precede a drift in accuracy.

## Stability scorecard of this lecture {h=dc3d58ab}

| Source of change | Measure | Result |
|---|---|---|
| Explainer randomness (LIME) | same top feature over 30 runs | 73 % (500 samples), 100 % (5,000) |
| Training seed | Kendall $\tau$ of global importances | MLP 0.83, XGBoost 0.97 |
| Training seed, one applicant | same top feature | MLP 78 %, XGBoost 83 % |
| Method | Spearman of importances | 0.52 to 1.00 |
| Input noise | median relative change | IG 0.030, SmoothGrad 0.036 |
| Time (day 1 to day 2) | Spearman, JS divergence | 0.98; 0.003 |

@practice report every explanation with its stability: which seed, which sample, which period, and how much it moves.

## @end

Explanations are estimates. SHAP adds up exactly, LIME depends on its settings, gradients are cheap and, with integrated gradients, complete; but every explanation moves with the seed, the sample and the period, and should be tested like a model.

# Explainability in model governance

## An explainability audit checklist {h=490745bc}

An audit of explanations starts from the purpose: who reads the explanation (customer, analyst, validator, regulator) and which decision it supports. The settings (background sample, baseline, seed, number of samples, kernel width) are versioned with the model. The checks are those of this lecture: additivity or completeness, faithfulness, stability across seeds and time, agreement of two methods. Fairness needs a look at protected attributes and their proxies, remembering that an explanation shows how a model uses a feature, not that the feature causes default. For customers, a few reasons and an actionable counterfactual are better than raw component names. And if an interpretable model is almost as accurate (here a logistic AUC of 0.909 against 0.935 for the network), it deserves consideration.

@practice the validation report should contain the explanations and their stability tests, not only the accuracy.

## @end

Lecture 12 moves to large language models, where explanations themselves become text, with new questions about faithfulness.
