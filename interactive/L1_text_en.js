// Readouts of the Lecture 1 interactive page, written by lectures/L1/build_interactive_L1.py.
window.LK_TEXT = {
 models: { logit: 'Logistic regression', mlp: 'MLP 64-64', xgb: 'XGBoost' },
 dti: 'Debt-to-income (%)', clage: 'Age of the oldest credit line (months)', prob: 'Predicted probability of default',
 repaid: 'Repaid (test)', defaulted: 'Defaulted (test)', boundary: c => `Boundary: probability = ${c}`,
 nbTitle: m => m === 'n' ? 'One neuron: p = σ(b + w₁·z₁ + w₂·z₂), z = standardised inputs' : 'Trained MLP 64-64',
 nb1: (auc, w1, w2, b) => `Test AUC of this score: <b>${auc}</b> (w₁ = ${w1}, w₂ = ${w2}, b = ${b}); fitted logistic regression 0.744, MLP 64-64 0.736.`,
 nb2: (t, n, k, kk, f, ff) => `Cut-off ${t}: ${n} of 871 test loans flagged; ${k} of ${kk} defaults caught; ${f} of ${ff} repaid loans flagged by mistake.`,
 nbRev: 'The AUC is below 0.5: this neuron ranks the safe loans as risky (flip the signs).',
 latitude: 'Latitude (degrees north)', value: 'Value (100,000 USD)', data: 'Data: mean value in 40 latitude bins',
 lin: 'Linear regression', fit: h => `MLP, H = ${h}`, unitsTitle: 'Contribution of each hidden unit, vⱼ·σ(aⱼ + bⱼ·x)',
 fitTitle: 'Median house value by latitude', unit: j => `unit ${j}`,
 ua: (h, te, tr, lin) => `H = ${h}: test MSE <b>${te}</b> (training ${tr}); straight line ${lin}; 14,736 training and 4,912 test block groups.`,
 hidden: 'Hidden units H (log scale)', auc: 'Test AUC', mean5: 'MLP, one hidden layer (mean of 5 seeds)', range5: 'range of 5 seeds',
 seeds: 'MLP, each of the 5 seeds', logitLine: 'Logistic regression', xgbLine: 'Gradient boosting (XGBoost)',
 wd: (h, m, sd, np) => `H = ${h}: mean test AUC <b>${m}</b> (standard deviation over 5 seeds ${sd}); ${np} parameters (logistic regression: 29).`,
 fpr: 'False positive rate: good loans flagged (%)', tpr: 'Defaults caught (%)', rocTitle: 'ROC curves, 1,192 test loans',
 histTitle: 'Predicted PD by outcome', pd: 'Predicted probability of default (PD)', count: 'Loans', chance: 'Random ranking',
 rc: (m, auc, t, tp, k, fp, f, n) => `${m} (AUC ${auc}), cut-off ${t}: ${n} loans flagged; <b>${tp}</b> of defaults caught (${k} of 238); ${fp} of good loans flagged (${f} of 954).`,
 thr: 'Cut-off: reject if predicted PD ≥ threshold', costPA: 'Cost per applicant', bayes: c => `Cost-optimal cut-off 1/(1 + ${c})`,
 acceptAll: 'Accept everyone', rejectAll: 'Reject everyone', costTitle: c => `Expected cost per applicant, missed bad = ${c}, rejected good = 1`,
 ct: (m, t, c, best, bt) => `${m}: cost <b>${c}</b> per applicant at cut-off ${t}; minimum ${best} at ${bt}.`,
 ct0: (bt, all, none) => `Bayes cut-off 1/(1 + c) = ${bt}. Accept everyone: ${all}; reject everyone: ${none} (1,000 applicants, 300 bad).`
};
