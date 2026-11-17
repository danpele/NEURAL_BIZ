// Texts of the Lecture 2 lab (English); numbers are filled in by L2_lab.js from the course data.
window.LAB_T = {
    lang: 'en',
    nogl: 'This browser cannot draw 3D graphics (WebGL is off); the readout below still shows every step.',
    lock: 'Lock my prediction', run: '▶ Run', why: 'Show why',
    afterLock: 'Prediction locked. Press ▶ Run to test it on the surface; drag the surface to turn it.',
    right: 'Your prediction was right.',
    wrong: o => `Your prediction was different. What happens: <b>${o}</b>`,
    score: (ok, n) => `Your predictions so far: ${ok} of ${n} right. Nothing is stored.`,
    readout: (k, a, b, L, m, off) => off
        ? `Step ${k}: the ball has left the surface; the loss grows without bound.`
        : `Step ${k}: intercept a = ${a}, slope b = ${b}, MSE = <b>${L}</b> (minimum ${m}).`,
    diverged: 'diverged', r4x: 'Epoch', r4y: 'Validation MSE (log scale)',
    r4lin: v => `linear regression ${v}`,
    r4read: (lr, e, cur, f, div) => `η = ${lr}, epoch ${e}: validation MSE ${cur === null ? '<b>diverged</b>' : '<b>' + cur + '</b>'} (seed 0). ` +
        `After 30 epochs, the three seeds: ${f.join(', ')}${div ? ` (${div} of 3 diverged)` : ''}.`,
    rounds: {
        r1: {
            q: 'The ball starts at intercept a = 4, slope b = −0.5. It takes 60 gradient-descent steps with the safe step η = 1/λ<sub>max</sub>. Where will it be?',
            opts: ['At the minimum, or very close to it', 'Partway along the valley floor, still far from the minimum', 'It bounces out of the valley'],
            answer: 1,
            proof: (N, f) => `<p>The loss curves very differently in the two directions. The second-derivative matrix of the MSE has eigenvalues λ<sub>max</sub> = ${f(N.lmax, 2)} and λ<sub>min</sub> = ${f(N.lmin, 3)}, so the condition number is ${f(N.cond, 1)}.</p>` +
                `<p>Along each of these directions, one step multiplies the remaining error by (1 − ηλ). With η = 1/λ<sub>max</sub> = ${f(N.eta, 4)}, the steep direction is solved in one step. The flat one keeps a factor 1 − λ<sub>min</sub>/λ<sub>max</sub> = ${f(N.flat, 4)} per step: after 60 steps, ${f(N.flat60 * 100, 0)}% of its error is still there.</p>` +
                `<p>The MSE falls from ${f(N.start, 3)} to ${f(N.gd60, 3)}, against a minimum of ${f(N.mstar, 3)} at a = ${f(N.intercept, 4)}, b = ${f(N.slope, 4)}. Now move the number of steps to 400 and run again.</p>`,
        },
        r2: {
            q: 'Same start, 60 steps, but the step is 2.1 × 1/λ<sub>max</sub>, about twice the safe step. What happens?',
            opts: ['It reaches the minimum faster', 'It zig-zags across the valley, then settles', 'It bounces higher at every step and leaves the surface'],
            answer: 2,
            proof: (N, f) => `<p>Along the steep direction, the error is multiplied by 1 − ηλ<sub>max</sub> = 1 − 2.1 = −1.1 at each step: it changes sign and grows by 10% each time. After 60 steps that is a factor 1.1<sup>60</sup> ≈ ${f(Math.pow(1.1, 60), 0)}.</p>` +
                `<p>Gradient descent on a quadratic loss converges only if |1 − ηλ| &lt; 1 in every direction, that is η &lt; 2/λ<sub>max</sub> = ${f(N.etaMax, 4)}. Between 1 and 2 times 1/λ<sub>max</sub> the ball zig-zags and still settles: try 1.5. Above 2 it diverges: try 1.95 and 2.05.</p>` +
                `<p>The steepest direction sets the largest step you can take, and the flattest direction then sets how long you wait.</p>`,
        },
        r3: {
            q: 'Now the income is standardised (mean 0, standard deviation 1), and the step rule is the same, η = 1/λ<sub>max</sub>. How many steps does the ball need to reach the minimum?',
            opts: ['One step', 'About ten steps', 'It is still crawling after 60 steps'],
            answer: 0,
            proof: (N, f) => `<p>With a standardised input, the second-derivative matrix becomes 2 times the identity: λ = ${f(N.lStd, 2)} in every direction and the condition number is ${f(N.condStd, 1)}. The valley is round.</p>` +
                `<p>Then η = 1/λ = 0.5 and one step gives w<sub>1</sub> = w<sub>0</sub> − 0.5 (2w<sub>0</sub> − 2w<sup>*</sup>) = w<sup>*</sup>: the MSE is ${f(N.std1, 3)} after one step, the minimum.</p>` +
                `<p>Momentum is the other fix, on the raw scale: with β = 0.9 and half the step, the MSE after 60 steps is ${f(N.mom60, 3)}, against ${f(N.gd60, 3)} for plain gradient descent. Switch to “raw income with momentum” and run it. This is why every notebook of the course standardises the inputs before training.</p>`,
        },
        r4: {
            q: 'A real network: two hidden layers of 64 ReLU units, the 8 standardised features of California housing, plain stochastic gradient descent, batches of 128, 30 epochs. Which learning rate gives the lowest validation MSE?',
            opts: ['η = 0.001', 'η = 0.01', 'η = 0.1', 'η = 1'],
            answer: 2,
            proof: (N, f) => `<p>η = ${f(N.best.lr, 1)} gives the lowest validation MSE, ${f(N.best.med, 3)} (median of three seeds), against ${f(N.lin, 3)} for the linear regression on the same features.</p>` +
                `<p>One step higher, at η = ${f(N.firstDiv.lr, 1)}, ${N.firstDiv.div} of the 3 seeds diverge, and at η = 1 all three do. A rate that is too small is slow but not wrong: at η = 0.0001 the MSE is still ${f(N.slow.med, 3)} after 30 epochs.</p>` +
                `<p>As on the quadratic surface, where the limit was 2/λ<sub>max</sub>, the best rate sits just below the edge of divergence. Learning-rate range tests therefore sweep η on a log scale and keep the largest rate that still trains stably.</p>`,
        },
    },
};
