// Texts of the Lecture 7 lab (English); numbers are filled in by L7_lab.js from the course data.
window.LAB_T = {
    lang: 'en',
    nogl: 'This browser cannot draw 3D graphics (WebGL is off); the readout below still shows every step.',
    lock: 'Lock my prediction', run: '▶ Run', why: 'Show why',
    afterLock: 'Prediction locked. Press ▶ Run to test it on the course data.',
    right: 'Your prediction was right.',
    wrong: o => `Your prediction was different. What happens: <b>${o}</b>`,
    score: (ok, n) => `Your predictions so far: ${ok} of ${n} right. Nothing is stored.`,
    key3d: 'rows (left): the token that looks (query) · columns (front): the token looked at (key) · height: attention weight',
    layer: 'Layer', share: 'Share of all attention', onSep: 'on [SEP]', onCls: 'on [CLS]',
    f1: 'Macro-F1 on the 2,388 test tweets', nTrain: 'Labelled training tweets (log scale)',
    ftLine: v => `FinBERT fine-tuned on all tweets ${v}`,
    ds: { probe: 'FinBERT frozen + logistic', tfidf: 'TF-IDF + logistic', tiny: 'small transformer' },
    model: { tfidf: 'TF-IDF + logistic', embbag: 'Embedding bag (no attention)', tiny_nopos: 'Small transformer, no positions',
        tiny: 'Small transformer', finbert_zs: 'FinBERT zero-shot', finbert_probe: 'FinBERT frozen + logistic', finbert_ft: 'FinBERT fine-tuned' },
    r1read: (l, h, t, m, one, n) => `Layer ${l}, head ${h}, scores divided by ${t}: the largest weight in a row is <b>${m}</b> on average; ${one} of ${n} rows put more than 90% on one token.`,
    r2read: (l, e, e500) => `Layer ${l}: normalised entropy of the rows <b>${e}</b> for this tweet (1 = spread evenly, 0 = one token); ${e500} on average over 500 test tweets.`,
    r3read: (l, s, c, s7) => `Layer ${l}: <b>${s}%</b> of all attention on [SEP], ${c}% on [CLS] (500 test tweets); ${s7}% on [SEP] in the seven tweets of round 2.`,
    r4read: (m, v, seeds) => `${m}: macro-F1 <b>${v}</b> (runs: ${seeds}).`,
    r5read: (n, pct, p, t, s) => `${n} labelled tweets (${pct}% of the training set): FinBERT frozen + logistic <b>${p}</b>, TF-IDF + logistic ${t}, small transformer ${s}.`,
    rounds: {
        r1: {
            q: 'FinBERT reads the tweet “BTIG points to breakfast pressure for Dunkin’ Brands”. Each row of attention is softmax(q·k / √64): the raw scores divided by 8. What happens if we drop the division by 8?',
            opts: ['Almost nothing: softmax keeps the order of the scores, so the weights barely change', 'Most rows put more than 90% of their weight on a single token', 'Every row spreads its weight evenly over all 13 tokens'],
            answer: 1,
            proof: (N, f) => `<p>The raw scores q·k of this tweet range from ${f(N.sLo, 1)} to ${f(N.sHi, 1)}. Over all 144 heads (12 layers × 12 heads), the largest weight in a row is ${f(N.max8, 2)} on average with the division by 8 and ${f(N.max1, 2)} without it; the rows with more than 90% on one token go from ${f(100 * N.one8, 0)}% to ${f(100 * N.one1, 0)}%.</p>` +
                `<p>q·k is a sum of 64 products. If each product has variance about 1, the sum has variance 64 and standard deviation 8, so raw scores easily differ by tens or more. Softmax turns a gap of 20 into a ratio of e<sup>20</sup> ≈ 500 million: the row becomes one-hot. A saturated softmax has a gradient close to zero, so the head stops learning (Vaswani et al., 2017).</p>` +
                `<p>Dividing by √d brings the scores back to a variance of about 1, whatever the size d of the head. Turn the divisor to 16 to see the opposite: the rows flatten towards an even spread.</p>`,
        },
        r2: {
            q: 'Averaged over 500 test tweets, in which layers is the attention of FinBERT most concentrated, that is, where do the rows have the lowest entropy?',
            opts: ['Layer 1, at the input', 'The middle layers, around layers 6 to 8', 'The last layer, 12, next to the classifier'],
            answer: 1,
            proof: (N, f) => `<p>The normalised entropy of a row (entropy divided by log n: 1 for an even spread, 0 for a single token) is ${f(N.entL1, 2)} in layer 1, reaches its minimum of ${f(N.entMin, 2)} in layer ${N.entMinL} and rises again to ${f(N.entL12, 2)} in layer 12.</p>` +
                `<p>In layer 1, each token gathers context broadly. In the middle layers, many heads specialise and look at one token: the next or the previous word, or a special token (Clark et al., 2019). The last layers spread out again as they mix information for the output.</p>` +
                `<p>Concentrated does not mean informative. Round 3 shows where much of that concentrated attention goes.</p>`,
        },
        r3: {
            q: '[SEP] is the marker at the end of every input; it carries no meaning of the tweet. In layer 6, what share of all attention (all heads, all tokens, 500 test tweets) goes to it?',
            opts: ['Less than 10%', 'About a quarter', 'More than half'],
            answer: 2,
            proof: (N, f) => `<p>[SEP] receives ${f(100 * N.sepL1, 1)}% of the attention in layer 1 and ${f(100 * N.sepMax, 1)}% in layer ${N.sepMaxL}, the maximum; the seven tweets of round 2 give ${f(100 * N.sep7max, 1)}% in that layer. [CLS] peaks earlier, at ${f(100 * N.clsMax, 1)}% in layer ${N.clsMaxL}.</p>` +
                `<p>Clark et al. (2019) found the same in BERT and showed that changing these weights hardly changes the output: a head that has nothing to add parks its weight on [SEP], a “no-op”. Softmax forces each row to sum to 1, so the weight has to go somewhere.</p>` +
                `<p>A large attention weight is therefore not the same as importance. Attention maps are a view of the mechanism, not an explanation; Lecture 11 returns to this with SHAP and LIME.</p>`,
        },
        r4: {
            q: 'A small transformer trained from scratch on the 8,588 training tweets, with and without positional encoding (without it, the model sees a tweet as a bag of tokens). How much macro-F1 does it lose without positions?',
            opts: ['About 20 points: it collapses', 'About 5 points', 'About 1 point', 'Nothing: it scores higher'],
            answer: 2,
            proof: (N, f) => `<p>With positions, ${f(N.tiny, 1)} (mean of 3 seeds); without, ${f(N.tiny_nopos, 1)}: a gain of ${f(N.posGain, 1)} points, with a bootstrap interval from ${f(N.boot.lo, 1)} to ${f(N.boot.hi, 1)} (1,000 draws of the test tweets). The embedding bag, with no attention at all, reaches ${f(N.embbag, 1)}.</p>` +
                `<p>Finance headlines are short and their sentiment sits in words such as “downgraded”, “upgrades” or “miss”, so the order adds little. It matters for negation and for who did what to whom, which are rare here.</p>` +
                `<p>All the models trained from scratch stay well below TF-IDF with logistic regression (${f(N.tfidf, 1)}) and the fine-tuned FinBERT (${f(N.finbert_ft, 1)}). What the small transformer lacks is data, not word order: round 5.</p>`,
        },
        r5: {
            q: 'Only 429 labelled tweets, 5% of the training set. Which approach gives the highest macro-F1 on the 2,388 test tweets?',
            opts: ['The small transformer trained from scratch', 'TF-IDF with logistic regression', 'FinBERT frozen, with a logistic regression on its sentence vector (a linear probe)', 'All three within 2 points'],
            answer: 2,
            proof: (N, f) => `<p>With ${N.dsN0} tweets: FinBERT frozen + logistic ${f(N.ds0.probe, 1)}, TF-IDF ${f(N.ds0.tfidf, 1)}, small transformer ${f(N.ds0.tiny, 1)}. With all ${N.dsNfull.toLocaleString('en')}: ${f(N.dsF.probe, 1)}, ${f(N.dsF.tfidf, 1)} and ${f(N.dsF.tiny, 1)}.</p>` +
                `<p>FinBERT was pretrained on a large amount of text, so its sentence vector already separates good news from bad: a logistic regression needs few labels to read it. TF-IDF and the small transformer have to learn everything from the labelled tweets, and they catch up only as labels grow.</p>` +
                `<p>With all the labels, the frozen probe is matched by TF-IDF, and only fine-tuning the whole of FinBERT pulls ahead, to ${f(N.finbert_ft, 1)}. Pretraining is worth most when labels are scarce, which is the usual case in finance.</p>`,
        },
    },
};
