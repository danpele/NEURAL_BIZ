// Texts of the Lecture 3 lab (English); numbers are filled in by L3_lab.js from the course data.
window.LAB_T = {
    lang: 'en',
    nogl: 'This browser cannot draw 3D graphics (WebGL is off); the readout below still shows every epoch.',
    lock: 'Lock my prediction', run: '▶ Run', why: 'Show why',
    afterLock: 'Prediction locked. Press ▶ Run to test it on the course data.',
    right: 'Your prediction was right.',
    wrong: o => `Your prediction was different. What happens: <b>${o}</b>`,
    score: (ok, n) => `Your predictions so far: ${ok} of ${n} right. Nothing is stored.`,
    ep: 'Epoch', lossY: 'Log loss (log scale)', aucY: 'Validation AUC', legVal: 'validation', legTr: 'training',
    logitLine: v => `logistic regression ${v}`, bestLoss: e => `lowest validation loss (epoch ${e})`,
    wdTick: v => `weight decay ${v}`, dpTick: v => `dropout ${v}`,
    planeKey: v => `<span class="sw-gold"></span>gold plane: logistic regression ${v}; bar height: validation loss`,
    cell: (p, wd) => `dropout ${p}, weight decay ${wd}`,
    esLossY: 'Validation loss', esApY: 'Average precision', stopAt: e => `stop: epoch ${e}`,
    ddX: 'Number of random features p (log scale)', ddY: 'MSE (log scale)', pn: n => `p = n = ${n}`,
    ddBestLine: (v, p) => `best below n: ${v} (p = ${p})`, legTe: 'test (median)', legTrDd: 'training (median)',
    big: v => `${v} million`,
    r1read: (e, tr, va, lg) => `Epoch ${e}: training loss ${tr}, validation loss <b>${va}</b> (logistic regression ${lg}).`,
    r2read: (e, va, auc, lg) => `Epoch ${e}: validation loss ${va}, validation AUC <b>${auc}</b> (logistic regression ${lg}).`,
    r3read: (e, lo, loN, hi, hiN) => `Epoch ${e}: lowest validation loss <b>${lo}</b> (${loN}); highest ${hi} (${hiN}). Drag the bars to turn them.`,
    r4read: (e, va, ap) => `Epoch ${e}: validation loss ${va}, average precision <b>${ap}</b>.`,
    r5read: (p, m, lo, hi, tr) => `p = ${p} features: test MSE <b>${m}</b> (median of 10 draws; from ${lo} to ${hi}); training MSE ${tr}.`,
    rounds: {
        r1: {
            q: 'A network with two hidden layers of 256 units, about 79,000 weights, learns from 600 German credit applicants with no regularisation. Its training loss falls towards zero. Where is the validation loss (log loss on 200 other applicants) after 150 epochs?',
            opts: ['Below the logistic regression: more training, a better model', 'Flat at its best level, about 0.54', 'Far above its best level, worse than after the first epoch'],
            answer: 2,
            proof: (N, f) => `<p>The validation loss is lowest at epoch ${N.best}, ${f(N.vmin, 3)}, and then climbs to ${f(N.vend, 3)} at epoch ${N.epochs}: ${f(N.ratio, 1)} times its best and above the ${f(N.v1, 3)} of the first epoch. The training loss ends at ${f(N.tend, 4)}.</p>` +
                `<p>With about 79,000 weights for 600 rows, the network can fit every applicant of the training set, noise included. From epoch ${N.best} on, each further epoch fits that noise a little better and the new applicants a little worse. The logistic regression, with 49 coefficients, reaches ${f(N.logit, 3)} and cannot overfit in this way.</p>` +
                `<p>A falling training loss says only that the optimiser works. Whether the model generalises can be read only on data it has not seen.</p>`,
        },
        r2: {
            q: 'The same run. By epoch 150 the validation loss is about four times its best. What happens to the validation AUC, the ranking of bad borrowers above good ones, between epoch 3 and epoch 150?',
            opts: ['It collapses towards 0.5', 'It falls a little, in step with the loss', 'It does not fall: it ends higher than at epoch 3'],
            answer: 2,
            proof: (N, f) => `<p>The validation AUC is ${f(N.aucBest, 3)} at epoch ${N.best}, where the loss is lowest, and ${f(N.aucEnd, 3)} at epoch ${N.epochs}, with a peak of ${f(N.aucMax, 3)} at epoch ${N.aucMaxEp}. The logistic regression reaches ${f(N.logitAuc, 3)}.</p>` +
                `<p>The two metrics measure different things. The log loss punishes confident mistakes: an applicant who defaults but gets a predicted default probability of 0.001 costs −ln 0.001 ≈ 6.9 on their own. The overfitted network pushes its probabilities to 0 and 1, so its few mistakes become very expensive, while the order of the applicants, which is all the AUC sees, changes little.</p>` +
                `<p>With 200 applicants, one split is a noisy judge: over ${N.hoN} random splits of the same 1,000 applicants, the test AUC of the same logistic regression ranges from ${f(N.hoMin, 3)} to ${f(N.hoMax, 3)} (standard deviation ${f(N.hoSd, 3)}), so the rise is suggestive, not proof. What is safe to say: the ranking did not collapse, but the probabilities became overconfident. A bank that prices loans by the probability needs the loss, and that is the metric to stop on.</p>`,
        },
        r3: {
            q: 'Nine networks with the same setup: dropout 0, 0.3 or 0.5 and weight decay 0, 0.1 or 1 (AdamW, learning rate 0.001). After 150 epochs, which knob lowers the validation loss more?',
            opts: ['Dropout, from 0 to 0.5', 'Weight decay, from 0 to 1', 'Both about equally', 'Neither: all nine end near 2'],
            answer: 1,
            proof: (N, f) => `<p>After ${N.epochs} epochs: ${f(N.none, 3)} with neither, ${f(N.dp, 3)} with dropout 0.5 alone, ${f(N.wd, 3)} with weight decay 1 alone and ${f(N.both, 3)} with both. Weight decay removes about ${f(N.none - N.wd, 2)} of loss, dropout about ${f(N.none - N.dp, 2)}, and dropout adds nothing on top of weight decay.</p>` +
                `<p>In AdamW, every step multiplies each weight by (1 − ηλ) = 1 − 0.001 × 1 before the gradient update. Over ${N.steps.toLocaleString('en')} steps (10 batches of 64 per epoch) that factor alone would shrink a weight to ${f(N.shrink * 100, 0)}% of its value, so the weights stay small and the predicted probabilities stay away from 0 and 1, which is what the log loss rewards. Dropout makes each unit less reliant on the others, but the network can still grow its weights and become overconfident.</p>` +
                `<p>Even the best setting ends at ${f(N.gridBest, 3)}, about twice the ${f(N.gridMinEarly, 3)} to ${f(N.gridMaxEarly, 3)} that the nine settings reach in their first epochs. Regularisation slows overfitting; the next round shows what stops it.</p>`,
        },
        r4: {
            q: 'Card fraud: a 256-256 network without regularisation, 120 epochs on a time-ordered split. Early stopping keeps the epoch with the lowest validation loss. How good is that epoch on the metric the bank cares about, the average precision (area under the precision-recall curve)?',
            opts: ['Poor: average precision keeps rising long after the loss minimum', 'Within 0.01 of the best of all 120 epochs, while training to the end loses about 0.07', 'The same as at the end: average precision hardly moves after the first epochs'],
            answer: 1,
            proof: (N, f) => `<p>The validation loss is lowest at epoch ${N.esBest} (${f(N.esVmin, 4)}), where the average precision is ${f(N.esApAt, 3)}. The best average precision of all ${N.esEp} epochs is ${f(N.esApMax, 3)}, at epoch ${N.esApMaxEp}: ${f(N.esApMax - N.esApAt, 3)} more. After ${N.esEp} epochs, the loss is ${f(N.esVend, 4)} and the average precision ${f(N.esApEnd, 3)}, ${f(N.esApMax - N.esApEnd, 3)} below the best.</p>` +
                `<p>With a patience of 20 epochs, training stops at epoch ${N.esStop} and restores the weights of epoch ${N.esBest}, which saves ${N.esEp - N.esStop} of the ${N.esEp} epochs. The validation loss is far less noisy than the average precision and is available after every epoch, so it is a cheap stopping signal even when the decision is made on another metric.</p>` +
                `<p>Unlike the AUC of round 2, the average precision does fall here: it looks only at the top of the ranking, where the overconfident mistakes of an overfitted network sit. Move the slider to compare any stopping epoch.</p>`,
        },
        r5: {
            q: 'Random-feature regression on California housing: 400 training block groups, p random ReLU features, fitted by least squares (the minimum-norm solution when p > 400), 10 random draws. As p grows from 10 to 8,000, what does the test MSE do?',
            opts: ['It falls steadily: more features, lower error', 'A U-shape: it falls, then rises and stays high for large p', 'It falls, explodes near p = 400, then falls again'],
            answer: 2,
            proof: (N, f) => `<p>The median test MSE is ${f(N.ddSmall, 3)} at p = ${N.ddSmallP} and ${f(N.ddBest, 3)} at p = ${N.ddBestP}, the best below n. It already climbs to ${f(N.ddRise, 0)} at p = ${N.ddRiseP}; at p = ${N.ddPeakP} = n it reaches about ${f(N.ddPeak / 1e6, 1)} million, and at p = ${N.ddBigP.toLocaleString('en')} it is back to ${f(N.ddBig, 3)}.</p>` +
                `<p>At p = n there are exactly as many coefficients as training points: the fit passes almost exactly through every point (training MSE ${f(N.ddTrPeak, 3)}) and, since some directions of the feature matrix are nearly flat, it needs enormous coefficients to do so. Above n, infinitely many fits pass through every point, and least squares picks the one with the smallest coefficients. The more features, the smaller that norm can be, so the fit becomes smoother again: this is implicit regularisation.</p>` +
                `<p>Here the second descent, ${f(N.ddBig, 3)}, does not beat the classical best, ${f(N.ddBest, 3)}. Large models can generalise beyond the interpolation point, but larger is not automatically better: the comparison has to be made on held-out data.</p>`,
        },
    },
};
