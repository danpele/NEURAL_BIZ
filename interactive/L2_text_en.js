// Readouts of the Lecture 2 interactive page, written by lectures/L2/build_interactive_L2.py.
window.LK_TEXT = {
 a: 'Intercept a', b: 'Slope b', step: 'Step', mse: 'MSE (log scale)', path: 'Path of gradient descent', start: 'Start', min: 'Least-squares minimum',
 pathTitle: x => `Loss surface: value = a + b · ${x === 'raw' ? 'income' : 'standardised income'}`, lossTitle: 'Loss after each step',
 gd1: (e, l, k, m, m0, b) => `η = ${e} (stable below 2/λmax = ${l}); after ${k} steps MSE = <b>${m}</b> (minimum ${m0}; momentum β = ${b}).`,
 gd2: (lmax, lmin, c) => `Curvatures λmax = ${lmax}, λmin = ${lmin}; condition number ${c}.`, div: 'The step is too large: the path diverges.',
 epoch: 'Epoch', vmse: 'Validation MSE', lrTitle: e => `Validation MSE per epoch, SGD with η = ${e}`, sweepTitle: 'Validation MSE after 30 epochs (3 seeds)',
 lr: 'Learning rate η (log scale)', seed: 'one seed', median: 'median of 3 seeds', diverged: 'diverged', linear: 'Linear regression',
 lrOut: (e, m, a, b) => `η = ${e}: validation MSE after 30 epochs <b>${m}</b> (seed 1); 3 seeds: ${a} to ${b}.`, lrDiv: (e, n) => `η = ${e}: ${n} of 3 seeds diverge (MSE above 5).`,
 onames: { sgd: 'SGD', momentum: 'SGD + momentum', rmsprop: 'RMSProp', adam: 'Adam', adamw: 'AdamW' },
 opTitle: (o, e) => `${o}, η = ${e}: validation MSE, 3 seeds`, mean3: 'mean of 3 seeds', best: e => `best η = ${e}`,
 op: (o, e, f, m, e10) => `${o}, η = ${e}: final validation MSE <b>${f}</b> (mean of 3 seeds; lowest ${m}); after 10 epochs ${e10}.`,
 opBest: (e) => `Best rate of this optimiser on the grid: η = ${e}.`,
 steps: 'Gradient steps (log scale)', secs: 'Seconds of computing (log scale)', bsName: b => `batch ${b}`,
 bsTitle: 'Adam, η = 0.001: validation MSE',
 bs: (b, spe, s, v5, v40) => `Batch ${b}: ${spe} steps per epoch, ${s} s for 40 epochs; validation MSE ${v5} after 5 epochs, <b>${v40}</b> after 40.`,
 layer: 'Layer (1 = next to the input)', gnorm: 'Gradient norm ‖∂L/∂W‖ (log scale)',
 vgTitle: (a, i, d) => `${a}, ${i} initialisation, ${d} hidden layers of 64 units`,
 vg: (f, l, r) => `First layer ${f}, last layer ${l}: ratio first / last <b>${r}</b> (mean of 5 seeds, one batch of 512 block groups).`,
 vgInf: 'The forward signal overflows: the gradient is infinite (not drawn).', vgZero: n => `The gradient underflows to exactly 0 in ${n} layer(s) (not drawn on the log scale).`,
 acts: { sig: 'Sigmoid', tanh: 'Tanh', relu: 'ReLU' }, inits: { xavier: 'Xavier', he: 'He', small: 'N(0, 0.01²)', large: 'N(0, 1)' }
};
