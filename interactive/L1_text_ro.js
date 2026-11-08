// Readouts of the Lecture 1 interactive page, written by lectures/L1/build_interactive_L1.py.
const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 models: { logit: 'Regresia logistică', mlp: 'MLP 64-64', xgb: 'XGBoost' },
 dti: 'Datorii/venit (%)', clage: 'Vechimea celei mai vechi linii de credit (luni)', prob: 'Probabilitatea estimată de neplată',
 repaid: 'Rambursat (test)', defaulted: 'Neplată (test)', boundary: c => `Frontiera: probabilitatea = ${c}`,
 nbTitle: m => m === 'n' ? 'Un neuron: p = σ(b + w₁·z₁ + w₂·z₂), z = intrări standardizate' : 'MLP 64-64 antrenat',
 nb1: (auc, w1, w2, b) => `AUC pe test al acestui scor: <b>${auc}</b> (w₁ = ${w1}, w₂ = ${w2}, b = ${b}); regresia logistică estimată 0,744, MLP 64-64 0,736.`,
 nb2: (t, n, k, kk, f, ff) => `Pragul ${t}: ${n} din 871 de credite de test semnalate; ${k} din ${deRO(kk)} neplăți detectate; ${f} din ${deRO(ff)} credite rambursate semnalate din greșeală.`,
 nbRev: 'AUC este sub 0,5: acest neuron consideră riscante creditele sigure (schimbați semnele).',
 latitude: 'Latitudinea (grade nord)', value: 'Valoarea (100.000 USD)', data: 'Date: valoarea medie pe 40 de clase de latitudine',
 lin: 'Regresia liniară', fit: h => `MLP, H = ${h}`, unitsTitle: 'Contribuția fiecărui neuron ascuns, vⱼ·σ(aⱼ + bⱼ·x)',
 fitTitle: 'Valoarea mediană a locuințelor după latitudine', unit: j => `neuronul ${j}`,
 ua: (h, te, tr, lin) => `H = ${h}: MSE pe test <b>${te}</b> (antrenare ${tr}); dreapta de regresie ${lin}; 14.736 de grupuri de locuințe de antrenare și 4.912 de test.`,
 hidden: 'Neuroni ascunși H (scară logaritmică)', auc: 'AUC pe test', mean5: 'MLP cu un strat ascuns (media a 5 inițializări)', range5: 'intervalul celor 5 inițializări',
 seeds: 'MLP, fiecare dintre cele 5 inițializări', logitLine: 'Regresia logistică', xgbLine: 'Gradient boosting (XGBoost)',
 wd: (h, m, sd, np) => `H = ${h}: AUC mediu pe test <b>${m}</b> (abaterea standard pe 5 inițializări ${sd}); ${deRO(np)} parametri (regresia logistică: 29).`,
 fpr: 'Rata alarmelor false: credite bune semnalate (%)', tpr: 'Neplăți detectate (%)', rocTitle: 'Curbele ROC, 1.192 de credite de test',
 histTitle: 'PD estimată, după rezultat', pd: 'Probabilitatea estimată de neplată (PD)', count: 'Credite', chance: 'Ordonare aleatoare',
 rc: (m, auc, t, tp, k, fp, f, n) => `${m} (AUC ${auc}), pragul ${t}: ${deRO(n)} credite semnalate; <b>${tp}</b> dintre neplăți detectate (${k} din 238); ${fp} dintre creditele bune semnalate (${f} din 954).`,
 thr: 'Pragul: respingem dacă PD estimată ≥ prag', costPA: 'Costul pe solicitant', bayes: c => `Pragul optim pentru cost 1/(1 + ${c})`,
 acceptAll: 'Acceptăm pe toată lumea', rejectAll: 'Respingem pe toată lumea', costTitle: c => `Costul așteptat pe solicitant, rău-platnic acceptat = ${c}, bun-platnic respins = 1`,
 ct: (m, t, c, best, bt) => `${m}: costul <b>${c}</b> pe solicitant la pragul ${t}; minimul ${best} la ${bt}.`,
 ct0: (bt, all, none) => `Pragul Bayes 1/(1 + c) = ${bt}. Acceptăm pe toată lumea: ${all}; respingem pe toată lumea: ${none} (1.000 de solicitanți, 300 rău-platnici).`
};
