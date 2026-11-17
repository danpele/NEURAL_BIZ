// Texts of the Lecture 6 lab (English); numbers are filled in by L6_lab.js from the course data.
window.LAB_T = {
    lang: 'en',
    nogl: 'This browser cannot draw 3D graphics (WebGL is off); the readout below still lists the neighbours.',
    lock: 'Lock my prediction', run: '▶ Run', why: 'Show why',
    afterLock: 'Prediction locked. Press ▶ Run to test it on the course data.',
    right: 'Your prediction was right.',
    wrong: o => `Your prediction was different. What happens: <b>${o}</b>`,
    score: (ok, n) => `Your predictions so far: ${ok} of ${n} right. Nothing is stored.`,
    rank: 'Rank of the word (log scale)', freq: 'Number of occurrences (log scale)', fitLab: s => `slope ${s}`,
    anQ: 'Analogy', anExp: 'Expected', anGot: 'Our word2vec answers',
    expect: { 'germany:german::france': 'french', 'italy:italian::spain': 'spanish', 'banks:bank::households': 'household', 'rise:rising::fall': 'falling', 'higher:lower::increase': 'decrease' },
    f1: 'Macro-F1, Financial PhraseBank (5 random splits)', legPb: 'PhraseBank (± 1 sd)', legTw: 'finance tweets',
    cls: ['negative', 'neutral', 'positive'], trueLab: 'true', predLab: 'predicted',
    agName: k => (k === 'all' ? 'all sentences' : `${k}% agreement`),
    cmNote: (a, n, acc, maj) => `TF-IDF + logistic, sentences with ${a}: ${n} test predictions over the 5 splits; accuracy ${acc}% (majority class ${maj}%). Rows: the label of the annotators; each row in % of its total.`,
    r1read: (r, f, cum) => `Rank ${r}: ${f} occurrence${f === '1' ? '' : 's'}; the words up to this rank make up <b>${cum}%</b> of all tokens.`,
    r2read: (w, list, gl, cl) => `Nearest words to <b>${w}</b> in our word2vec (cosine): ${list}. GloVe: ${gl}.` + (cl ? ` Among the 32 words drawn: ${cl}.` : ''),
    r3read: (k, n) => `${k} of ${n} analogies computed as the word nearest to b − a + c.`,
    r4read: (m, f, sd, tw) => `${m}: macro-F1 <b>${f}</b> (± ${sd}) on PhraseBank, ${tw} on the tweets.`,
    r5read: (a, acc, maj, d) => `Sentences with ${a}: accuracy <b>${acc}%</b>, majority class ${maj}%, a margin of ${d} points.`,
    rounds: {
        r1: {
            q: 'The speeches of the European Central Bank (ECB), 2022 to 2026: about 1.29 million tokens and 20,130 different words. What share of all tokens do the 10 most frequent words (the, of, and, …) make up?',
            opts: ['About 2%', 'About 10%', 'About 25%', 'About 50%'],
            answer: 2,
            proof: (N, f) => `<p>The 10 most frequent words (${N.top.slice(0, 6).join(', ')}, …) make up ${f(100 * N.top10, 1)}% of the ${N.tot.toLocaleString('en')} tokens, and the 100 most frequent ${f(100 * N.top100, 1)}%. Half of all tokens are covered by the ${N.r50} most frequent words, 90% by the ${N.r90.toLocaleString('en')} most frequent.</p>` +
                `<p>At the other end, ${N.hapax.toLocaleString('en')} words, ${f(100 * N.hapaxPct, 1)}% of the ${N.types.toLocaleString('en')} distinct words, appear only once. On log-log axes, frequency falls almost on a straight line with slope ${f(N.slope, 2)}: Zipf's law.</p>` +
                `<p>For a model this means two things: a few function words dominate every count, which is why TF-IDF down-weights them, and most of the vocabulary is too rare to learn a good vector for. Our word2vec keeps only words seen at least 5 times.</p>`,
        },
        r2: {
            q: 'Our own word2vec, trained on the ECB speeches (100 dimensions). Which word is the nearest neighbour of “inflation” (highest cosine similarity)?',
            opts: ['prices', 'headline', 'deflation', 'interest'],
            answer: 1,
            proof: (N, f) => `<p>The nearest words to “inflation” are ${N.nbInf[0].slice(0, 6).map((w, i) => `${w} (${f(N.nbInf[1][i], 2)})`).join(', ')}. “prices” is not among the first 12: its cosine with “inflation” is ${f(N.cosIP, 2)}, and “interest” gets ${f(N.cosII, 2)}.</p>` +
                `<p>word2vec puts words close when they appear in the same contexts, not when they mean related things. In central-bank speeches, “headline inflation” and “underlying inflation” are fixed phrases, so “headline” and “underlying” share almost every context of “inflation”. GloVe, trained on the same text, gives ${N.nbInfG[0].slice(0, 4).join(', ')}.</p>` +
                `<p>The 3D picture keeps only ${f(100 * N.pc3, 0)}% of the variance of the 100 dimensions, so distances in it are only a rough guide: the cosine in the readout is the real measure. Pick another word to compare the two models.</p>`,
        },
        r3: {
            q: 'Analogies are solved as the word nearest to b − a + c, as in the famous king − man + woman ≈ queen. How many of these five does our word2vec, trained on 1.29 million tokens, get right: germany : german :: france : ?, and four similar ones?',
            opts: ['All five', 'About half', 'Only one of the five'],
            answer: 2,
            proof: (N, f) => `<p>Only “banks : bank :: households → ${N.analogy['banks:bank::households']}” is right. The others give ${['germany:german::france', 'italy:italian::spain', 'rise:rising::fall', 'higher:lower::increase'].map(k => `“${N.analogy[k]}”`).join(', ')} instead of french, spanish, falling and decrease.</p>` +
                `<p>The analogies of Mikolov et al. (2013) worked on vectors trained on billions of tokens, where each relation (country to adjective, verb to its -ing form) is seen thousands of times. The ECB speeches have ${N.corpusTok.toLocaleString('en')} tokens and ${N.V.toLocaleString('en')} words in the vocabulary of word2vec; the country–adjective pairs appear too rarely for the direction “country → adjective” to be learnt. The singular/plural relation is frequent enough, and it works.</p>` +
                `<p>Analogies are a property of large corpora, not of the method alone. The neighbours of round 2 are much more stable than the arithmetic of vectors.</p>`,
        },
        r4: {
            q: 'Sentiment of the Financial PhraseBank sentences. One model averages the word2vec vectors of the words of a sentence and feeds the mean to a logistic regression; another uses TF-IDF with a logistic regression. Which gets the higher macro-F1?',
            opts: ['The mean of word2vec vectors, by more than 5 points', 'About the same, within 2 points', 'TF-IDF, by more than 10 points'],
            answer: 2,
            proof: (N, f) => `<p>TF-IDF + logistic reaches ${f(N.pb.tfidf_lr.f1, 1)}, the mean of word2vec vectors + logistic ${f(N.pb.w2v_lr.f1, 1)}: a gap of ${f(N.gapPb, 1)} points (5 random splits). On the finance tweets the gap is ${f(N.gapTw, 1)} points.</p>` +
                `<p>Averaging blurs the few words that carry the sentiment (rose, fell, down, improved) into the many neutral ones, and our vectors were trained on ECB speeches, whose vocabulary covers ${f(N.oov.pb_cover, 1)}% of the PhraseBank tokens. TF-IDF keeps every word as its own feature and lets the regression weight “fell” and “rose” directly.</p>` +
                `<p>An embedding bag trained for the task (${f(N.pb.embbag.f1, 1)}) comes close to TF-IDF, and starting it from our word2vec adds only ${f(N.w2vGain, 2)} points. Pretrained vectors help when they come from a large, matching corpus; Lecture 7 shows the transformer version of that.</p>`,
        },
        r5: {
            q: 'Each PhraseBank sentence was labelled by several annotators, and the data record how many of them agreed. How accurate is TF-IDF + logistic on the sentences where all of them agreed, compared with those where only half agreed?',
            opts: ['About the same, near 78% for both', 'Much better on full agreement: about 88% against 58%', 'Better on the sentences with half agreement, because they are more varied'],
            answer: 1,
            proof: (N, f) => `<p>Accuracy is ${f(N.acc('100'), 1)}% on sentences with full agreement, ${f(N.acc('75'), 1)}% with 75%, ${f(N.acc('66'), 1)}% with 66% and ${f(N.acc('50'), 1)}% with 50% agreement. With 50% agreement, the majority class alone gets ${f(N.maj('50'), 1)}%, so the model is barely better than always answering “neutral”.</p>` +
                `<p>When half of the annotators read a sentence as positive and half as neutral, there is no single correct label: the model cannot be right more often than the label itself is. The errors are concentrated where people also disagree.</p>` +
                `<p>Before blaming the model, look at the quality of the labels. Reporting accuracy by agreement level tells the reader how much of the error is noise in the data.</p>`,
        },
    },
};
