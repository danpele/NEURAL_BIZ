// Readouts of the Lecture 3 interactive page, written by lectures/L3/build_interactive_L3.py.
window.LK_TEXT = {
 epoch: 'Epoch', ce: 'Cross-entropy', auc: 'Validation AUC', train: 'Training loss', valid: 'Validation loss', logit: 'Logistic regression (validation)',
 stop: 'Stop here', best: 'Lowest validation loss', vauc: 'Validation AUC', lossTitle: (p, w) => `MLP 256-256, dropout ${p}, weight decay ${w}`,
 aucTitle: 'Ranking quality on the validation applicants',
 ov: (e, tr, va, auc, be, bv) => `Epoch ${e}: training loss ${tr}, validation loss <b>${va}</b>, validation AUC <b>${auc}</b>. Lowest validation loss ${bv} at epoch ${be}.`,
 ov2: (l, a) => `Logistic regression on the same split: validation loss ${l}, AUC ${a}.`,
 p: 'Hidden units p (log scale)', mse: 'MSE (log scale)', ddTest: 'Test MSE (median of 10 draws)', ddTrain: 'Training MSE (median)',
 draws: 'test MSE of each draw', thr: 'interpolation threshold p = n = 400',
 dd: (p, te, tr, lo, hi) => `p = ${p}: median test MSE <b>${te}</b> (10 draws from ${lo} to ${hi}); median training MSE ${tr}.`,
 esLoss: 'Training and validation loss (log scale)', esAp: 'Validation PR-AUC (population-weighted)', ap: 'PR-AUC', kept: 'Weights kept', stopAt: 'Training stops',
 es: (p, s, k, v, a, aEnd) => `Patience ${p}: training stops at epoch <b>${s}</b> and keeps the weights of epoch <b>${k}</b> (validation loss ${v}, PR-AUC ${a}); at epoch 120 the PR-AUC would be ${aEnd}.`,
 esNo: p => `Patience ${p}: no stop within the 120 epochs recorded.`,
 hoTitle: s => `Test AUC of logistic regression on 200 random ${s} test splits`, hoX: 'Test AUC', hoY: 'Number of splits',
 mean: 'mean', p5: '5th and 95th percentiles',
 ho: (s, n, m, sd, lo, hi, p5, p95) => `Test share ${s} (${n} applicants): mean AUC ${m}, standard deviation <b>${sd}</b>; range ${lo} to ${hi}; 90% of the splits between ${p5} and ${p95}.`,
 models: { logit: 'Logistic regression', mlp: 'Regularised MLP (5 seeds)', xgb: 'XGBoost' },
 alerts: 'Alerts sent to review (population-weighted, log scale)', cost: 'Cost (EUR)', noModel: 'No alerts: all test frauds lost',
 fcTitle: r => `Cost = ${r} EUR × alerts + amount of the missed frauds`,
 fc: (m, a, c, k, mc, ma) => `${m}: at ${a} alerts the cost is <b>${c} EUR</b> (${k} of 75 test frauds caught); minimum ${mc} EUR at ${ma} alerts.`,
 fc0: t => `Without any alert all ${t} EUR of test frauds are lost.`
};
