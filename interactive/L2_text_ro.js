// Readouts of the Lecture 2 interactive page, written by lectures/L2/build_interactive_L2.py.
const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 a: 'Termenul liber a', b: 'Panta b', step: 'Pasul', mse: 'MSE (scară logaritmică)', path: 'Drumul coborîrii pe gradient', start: 'Start', min: 'Minimul celor mai mici pătrate',
 pathTitle: x => `Suprafața pierderii: valoarea = a + b · ${x === 'raw' ? 'venit' : 'venit standardizat'}`, lossTitle: 'Pierderea după fiecare pas',
 gd1: (e, l, k, m, m0, b) => `η = ${e} (stabil sub 2/λmax = ${l}); după ${deRO(k)} pași MSE = <b>${m}</b> (minimul ${m0}; momentul β = ${b}).`,
 gd2: (lmax, lmin, c) => `Curburile λmax = ${lmax}, λmin = ${lmin}; numărul de condiționare ${c}.`, div: 'Pasul este prea mare: drumul diverge.',
 epoch: 'Epoca', vmse: 'MSE pe validare', lrTitle: e => `MSE pe validare după fiecare epocă, SGD cu η = ${e}`, sweepTitle: 'MSE pe validare după 30 de epoci (3 inițializări)',
 lr: 'Rata de învățare η (scară logaritmică)', seed: 'o inițializare', median: 'mediana celor 3 inițializări', diverged: 'divergent', linear: 'Regresia liniară',
 lrOut: (e, m, a, b) => `η = ${e}: MSE pe validare după 30 de epoci <b>${m}</b> (inițializarea 1); 3 inițializări: între ${a} și ${b}.`, lrDiv: (e, n) => `η = ${e}: ${n} din 3 inițializări diverg (MSE peste 5).`,
 onames: { sgd: 'SGD', momentum: 'SGD + moment', rmsprop: 'RMSProp', adam: 'Adam', adamw: 'AdamW' },
 opTitle: (o, e) => `${o}, η = ${e}: MSE pe validare, 3 inițializări`, mean3: 'media celor 3 inițializări', best: e => `η optim = ${e}`,
 op: (o, e, f, m, e10) => `${o}, η = ${e}: MSE finală pe validare <b>${f}</b> (media a 3 inițializări; minimul ${m}); după 10 epoci ${e10}.`,
 opBest: (e) => `Rata cea mai bună a acestui optimizator pe grilă: η = ${e}.`,
 steps: 'Pași de gradient (scară logaritmică)', secs: 'Secunde de calcul (scară logaritmică)', bsName: b => `lot de ${b}`,
 bsTitle: 'Adam, η = 0,001: MSE pe validare',
 bs: (b, spe, s, v5, v40) => `Lotul de ${b}: ${deRO(spe)} pași pe epocă, ${s} s pentru 40 de epoci; MSE pe validare ${v5} după 5 epoci, <b>${v40}</b> după 40.`,
 layer: 'Stratul (1 = lîngă intrare)', gnorm: 'Norma gradientului ‖∂L/∂W‖ (scară logaritmică)',
 vgTitle: (a, i, d) => `${a}, inițializare ${i}, ${deRO(d)} straturi ascunse de 64 de neuroni`,
 vg: (f, l, r) => `Primul strat ${f}, ultimul strat ${l}: raportul primul / ultimul <b>${r}</b> (media a 5 inițializări, un lot de 512 grupuri de locuințe).`,
 vgInf: 'Semnalul propagat înainte depășește capacitatea numerică: gradientul este infinit (nu este desenat).', vgZero: n => `Gradientul devine exact 0 în ${n} strat(uri) (nu apare pe scara logaritmică).`,
 acts: { sig: 'Sigmoidă', tanh: 'Tanh', relu: 'ReLU' }, inits: { xavier: 'Xavier', he: 'He', small: 'N(0; 0,01²)', large: 'N(0; 1)' }
};
