// Readouts of the Lecture 14 interactive page, written by lectures/L14/build_interactive_L14.py.
window.LK_TEXT = {
 cls: ["Annual crop", "Forest", "Herbaceous vegetation", "Highway", "Industrial", "Pasture", "Permanent crop", "Residential", "River", "Sea or lake"],
 px: 'Pixels (224 × 224 input)', img: p => `Input image cut into ${p} × ${p} patches`, att: o => `Attention of the class token (rollout, ViT-B/16), opacity ${o}%`,
 vp: (n, g, p, rel) => `Patch ${p} × ${p} pixels: grid ${g} × ${g} = ${n} patch tokens + 1 class token = ${n + 1} tokens; attention compares ${window.NNI.nf((n + 1) ** 2, 0)} pairs of tokens, ${rel} times the cost with 16 × 16 patches.`,
 vp2: (c, top, share) => `${c}: the most attended patches (top 10% of the map) take ${share} of the attention; peak at row ${top[0]}, column ${top[1]} of the 14 × 14 grid.`,
 prob: 'Probability (%)', simTitle: (s) => `Softmax of ${s} × cosine similarity`,
 zs: (t, p, pr, ok) => `True class: ${t}. Predicted: ${p} with probability ${pr} (${ok ? 'correct' : 'wrong'}).`,
 zs2: (cmax, cmin, s, h) => `Cosine similarities from ${cmin} to ${cmax}; at scale ${s} the entropy of the probabilities is ${h} bits (maximum 3.32 for 10 equal classes).`,
 pred: 'Predicted class (zero-shot)', truth: 'True class',
 cm: (m, acc, best, bestv, worst, worstv, t, p, n, nt) => `${m}: accuracy ${acc} on 5,400 test images. Best class ${best} (${bestv}), worst ${worst} (${worstv}). Largest confusion: ${n} of ${nt} images of ${t} predicted as ${p}.`,
 k: 'Images returned, k', prec: 'Precision of the top k (%)', avg: 'Mean of the 10 queries', query: q => `Query: "a centered satellite photo of ${q}." (5 templates averaged)`,
 ts: (q, k, p, n, a) => `Query ${q}: ${n} of the top ${k} images belong to the class, precision ${p}; mean over the 10 queries ${a}.`,
 ts2: 'Top 8 images (green border: right class, red: wrong class).',
 shots: 'Labelled images per class (log scale)', acc: 'Test accuracy (%)', probe: 'CLIP + logistic regression (5 seeds)', cnn: 'CNN from scratch (3 seeds)',
 zsl: 'CLIP zero-shot', full: 'CNN on all 21,600 training images', fullp: 'CLIP probe on all 21,600 training images',
 fs: (k, n, p, pl, ph, c) => `${window.NNI.cnt(k, 'image', 'images')} per class (${n} labelled images): CLIP probe ${p} (range over seeds ${pl}–${ph})${c}.`,
 fsc: (c, cl, ch) => `; CNN ${c} (range ${cl}–${ch})`, fsn: '; the CNN was not trained at this size',
 fs2: (z, f, fp) => `Zero-shot (no labels) ${z}; with all training labels: CNN ${f}, CLIP probe ${fp}.`,
 fpr: 'False alarm rate (share of calm windows flagged)', tpr: 'Hit rate (share of stress windows flagged)', chance: 'Random guess',
 score: 'Stress score', stress: 'Stress window (actual)', thr: 'Alarm threshold', date: 'End of the 120-day window',
 cr: (m, auc, a, tp, nt, fp, nf, pr) => `${m}: AUC ${auc}. Flagging ${a} of the test windows catches ${tp} of ${nt} stress windows, with ${fp} false alarms among ${nf} calm windows; precision of the alarms ${pr}.`,
 cr2: (n, s) => `${n} test windows (7 January 2019 – 30 October 2026), of which ${s} are stress (20-day volatility above the rolling 90th percentile of the previous 3 years).`
};
