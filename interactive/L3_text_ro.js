// Readouts of the Lecture 3 interactive page, written by lectures/L3/build_interactive_L3.py.
const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 epoch: 'Epoca', ce: 'Entropia încrucișată', auc: 'AUC pe validare', train: 'Pierderea de antrenare', valid: 'Pierderea de validare', logit: 'Regresia logistică (validare)',
 stop: 'Oprire aici', best: 'Pierderea minimă pe validare', vauc: 'AUC pe validare', lossTitle: (p, w) => `MLP 256-256, dropout ${p}, penalizare ${w}`,
 aucTitle: 'Calitatea ordonării pe solicitanții de validare',
 ov: (e, tr, va, auc, be, bv) => `Epoca ${e}: pierderea de antrenare ${tr}, pierderea de validare <b>${va}</b>, AUC pe validare <b>${auc}</b>. Pierderea minimă pe validare ${bv}, la epoca ${be}.`,
 ov2: (l, a) => `Regresia logistică pe aceeași împărțire: pierderea de validare ${l}, AUC ${a}.`,
 p: 'Neuroni ascunși p (scară logaritmică)', mse: 'MSE (scară logaritmică)', ddTest: 'MSE pe test (mediana a 10 extrageri)', ddTrain: 'MSE de antrenare (mediana)',
 draws: 'MSE pe test pentru fiecare extragere', thr: 'pragul de interpolare p = n = 400',
 dd: (p, te, tr, lo, hi) => `p = ${p}: MSE mediană pe test <b>${te}</b> (10 extrageri, între ${lo} și ${hi}); MSE mediană de antrenare ${tr}.`,
 esLoss: 'Pierderea de antrenare și de validare (scară logaritmică)', esAp: 'PR-AUC pe validare (ponderat la populație)', ap: 'PR-AUC', kept: 'Ponderile păstrate', stopAt: 'Antrenarea se oprește',
 es: (p, s, k, v, a, aEnd) => `Răbdarea ${p}: antrenarea se oprește la epoca <b>${s}</b> și păstrează ponderile epocii <b>${k}</b> (pierderea de validare ${v}, PR-AUC ${a}); la epoca 120, PR-AUC ar fi ${aEnd}.`,
 esNo: p => `Răbdarea ${p}: nicio oprire în cele 120 de epoci înregistrate.`,
 hoTitle: s => `AUC pe test al regresiei logistice, pe 200 de împărțiri aleatoare cu ${s} test`, hoX: 'AUC pe test', hoY: 'Numărul de împărțiri',
 mean: 'media', p5: 'percentilele 5 și 95',
 ho: (s, n, m, sd, lo, hi, p5, p95) => `Proporția de test ${s} (${deRO(n)} solicitanți): AUC mediu ${m}, abaterea standard <b>${sd}</b>; între ${lo} și ${hi}; 90% dintre împărțiri între ${p5} și ${p95}.`,
 models: { logit: 'Regresia logistică', mlp: 'MLP regularizat (5 semințe)', xgb: 'XGBoost' },
 alerts: 'Alerte trimise la verificare (ponderat, scară logaritmică)', cost: 'Costul (EUR)', noModel: 'Fără alerte: toate fraudele de test pierdute',
 fcTitle: r => `Costul = ${r} EUR × alertele + valoarea fraudelor ratate`,
 fc: (m, a, c, k, mc, ma) => `${m}: la ${deRO(a)} alerte costul este <b>${c} EUR</b> (${k} din 75 de fraude de test detectate); minimul ${mc} EUR la ${deRO(ma)} alerte.`,
 fc0: t => `Fără nicio alertă se pierd toți cei ${t} EUR ai fraudelor de test.`
};
