// Texts of the Lecture 12 lab (English); numbers are filled in by L12_lab.js from the course data.
window.LAB_T = {
    lang: 'en',
    nogl: 'This browser cannot draw 3D graphics (WebGL is off); the readout below still shows every temperature.',
    lock: 'Lock my prediction', run: '▶ Run', why: 'Show why',
    afterLock: 'Prediction locked. Press ▶ Run to test it on the course data.',
    right: 'Your prediction was right.',
    wrong: o => `Your prediction was different. What happens: <b>${o}</b>`,
    score: (ok, n) => `Your predictions so far: ${ok} of ${n} right. Nothing is stored.`,
    key3d: 'the 8 most likely next tokens, from “the” to “in” · temperature from 0.25 to 2 · height: probability relative to “the” · red bars: the current temperature',
    model: { majority: 'Majority class', tfidf: 'TF-IDF + logistic (trained)', 'zs_0.5B': 'Qwen2.5-0.5B, zero-shot', 'zs_1.5B': 'Qwen2.5-1.5B, zero-shot',
        'zs_3B': 'Qwen2.5-3B, zero-shot', probe: 'Qwen2.5-1.5B states + logistic', bm25: 'BM25', dense: 'dense embeddings', hybrid: 'hybrid' },
    ret: { tfidf: 'TF-IDF', bm25: 'BM25', dense: 'dense embeddings', hybrid: 'hybrid' },
    f1pb: 'Macro-F1 on 678 PhraseBank test sentences (95% bootstrap interval)', f1short: 'Macro-F1, PhraseBank', shots: 'Labelled examples in the prompt',
    instr: 'Instruction', pb: 'PhraseBank', tw: 'finance tweets',
    psNote: d => `Macro-F1 on the ${d} test set, zero-shot; the same model and data, only the wording changes.`,
    k: 'Number of chunks retrieved (k)', recall: 'Right speech among the first k',
    r1read: (t, r, p, e) => `T = ${t}: “The” has <b>${r}</b> times the probability of “the”` + (p ? `; “the” itself has <b>${p}%</b>, entropy ${e} nats.` : '.'),
    r2read: (m, f, lo, hi) => `${m}: macro-F1 <b>${f}</b>` + (lo ? ` (95% interval ${lo} to ${hi}).` : '.'),
    r3read: (k, m, seeds, m15) => `${k} example${k === 1 ? '' : 's'}: Qwen2.5-3B <b>${m}</b> (three draws of examples: ${seeds}); Qwen2.5-1.5B ${m15}.`,
    r4read: (i, v) => `Wording ${i}: Qwen2.5-1.5B macro-F1 <b>${v}</b>.`,
    r4done: (a, b) => `Best minus worst wording: <b>${a} points</b> for Qwen2.5-1.5B, ${b} for Qwen2.5-3B.`,
    r5read: (k, list) => `First ${k} chunk${k === 1 ? '' : 's'}: ${list}.`,
    rounds: {
        r1: {
            q: 'Qwen2.5-1.5B reads the start of a sentence from a speech by Philip R. Lane (30 October 2026): “The inflation impact is limited:”. At temperature T = 1, the most likely next token, “the”, has probability 10.1%. What is its probability at T = 0.5?',
            opts: ['About 5%: a lower temperature halves it', 'Still about 10%: temperature changes only how random the draw is', 'About 20%', 'About 50%'],
            answer: 3,
            proof: (N, f) => `<p>Temperature divides the scores before the softmax, so each probability becomes proportional to p<sup>1/T</sup>. At T = 0.5 every probability is squared and then renormalised. The long tail of the ${(151665).toLocaleString('en')} tokens of the vocabulary, each with a tiny probability, shrinks far more than the top: “the” goes from ${f(N.next.top1_p, 1)}% to ${f(N.next.top1_p05, 1)}%, and the entropy from ${f(N.next.entropy.t10, 2)} to ${f(N.next.entropy.t05, 2)} nats.</p>` +
                `<p>At T = 2 the opposite happens: “the” drops to ${f(N.next.top1_p2, 1)}% and the entropy rises to ${f(N.next.entropy.t20, 2)} nats, close to the ${f(N.entMax, 2)} of a uniform draw over the whole vocabulary. The ratio between two tokens, which the 3D ridge shows, is (p<sub>2</sub>/p<sub>1</sub>)<sup>1/T</sup>: ${f(N.ratio2(1), 2)} at T = 1, ${f(N.ratio2(0.5), 2)} at T = 0.5.</p>` +
                `<p>Temperature never changes the order, so greedy decoding picks “the” at any T. The word Lane actually said next was “${N.next.true_next}”, which is not among the 8 most likely tokens. For classification or extraction, use T close to 0; a high T only adds variety.</p>`,
        },
        r2: {
            q: 'Sentiment of 678 PhraseBank test sentences, with no training example: Qwen2.5 with 0.5, 1.5 and 3 billion parameters, against TF-IDF + logistic regression trained on 1,581 sentences. Which is true?',
            opts: ['All three language models beat TF-IDF', 'Only the 3B model beats TF-IDF; the 0.5B model is close to the majority class', 'None of them beats TF-IDF'],
            answer: 1,
            proof: (N, f) => `<p>Macro-F1: ${f(N.f1['zs_0.5B'], 1)} for 0.5B, against ${f(N.cls.majority.f1, 1)} for always answering the majority class; ${f(N.f1['zs_1.5B'], 1)} for 1.5B; ${f(N.f1['zs_3B'], 1)} for 3B (95% bootstrap interval ${f(N.cls.zs_3B.lo, 1)} to ${f(N.cls.zs_3B.hi, 1)}). TF-IDF reaches ${f(N.f1.tfidf, 1)} (${f(N.cls.tfidf.lo, 1)} to ${f(N.cls.tfidf.hi, 1)}).</p>` +
                `<p>The ability to follow an instruction appears with size: the 0.5B model answers “neutral” for ${N.zsNeu} of the 678 sentences, the 1.5B model is at the level of TF-IDF, and the 3B model is above it without seeing a single labelled sentence. A logistic regression on the hidden states of the 1.5B model gets ${f(N.f1.probe, 1)}.</p>` +
                `<p>The price is speed: ${f(N.cls.zs_3B.ms, 1)} ms per sentence for the 3B model against ${f(N.cls.tfidf.ms, 1)} ms for TF-IDF on the same laptop. With enough labels, a cheap trained model remains a strong benchmark.</p>`,
        },
        r3: {
            q: 'The 3B model scores 92.2 with no example in the prompt. Add one labelled example (three different random draws of the example). What happens to its macro-F1?',
            opts: ['It improves, to about 95', 'It stays at about 92', 'It falls, to about 89.5 on average'],
            answer: 2,
            proof: (N, f) => `<p>For Qwen2.5-3B, the mean macro-F1 is ${N.fsMean('3B').map((v, i) => `${f(v, 1)} with ${[0, 1, 2, 4, 8][i]}`).join(', ')} examples. With one example, the three draws give ${N.fs['3B'][1].map(v => f(v, 1)).join(', ')}. For Qwen2.5-1.5B: ${N.fsMean('1.5B').map(v => f(v, 1)).join(', ')}.</p>` +
                `<p>A single example pulls the model towards its own label and wording; which example is drawn matters as much as how many. A weaker model gains a little from one or two examples, a stronger one can lose, and eight examples are not better than four for either.</p>` +
                `<p>Few-shot prompting is a hyperparameter like any other: try it on a validation set, with several draws of examples, before trusting it.</p>`,
        },
        r4: {
            q: 'Five wordings of the same instruction (answer positive, negative or neutral), Qwen2.5-1.5B, zero-shot on PhraseBank. How far apart are the best and the worst macro-F1?',
            opts: ['Less than 1 point', 'About 3 points', 'More than 10 points'],
            answer: 2,
            proof: (N, f) => `<p>On PhraseBank, Qwen2.5-1.5B goes from ${f(Math.min(...N.ps['pb_1.5B']), 1)} to ${f(Math.max(...N.ps['pb_1.5B']), 1)}, a range of ${f(N.range('pb_1.5B'), 1)} points; Qwen2.5-3B from ${f(Math.min(...N.ps['pb_3B']), 1)} to ${f(Math.max(...N.ps['pb_3B']), 1)} (${f(N.range('pb_3B'), 1)} points). On the tweets even the 3B model spans ${f(N.range('tw_3B'), 1)} points.</p>` +
                `<p>The wordings ask the same question; what changes is the order of the labels, the point of view (investor, shareholder, press release) and how strictly the answer is constrained. Smaller models follow such details more literally.</p>` +
                `<p>A result obtained with one prompt is a result for that prompt. Choose the wording on validation data, report it in full, and keep it fixed before looking at the test set.</p>`,
        },
        r5: {
            q: 'Retrieval for RAG: the query is the title of an ECB speech, and the system must find that speech among 400 (8,400 chunks of 200 words). Which retriever puts the right speech first most often?',
            opts: ['TF-IDF', 'BM25', 'Dense embeddings (bge-small)', 'A hybrid of BM25 and dense ranks'],
            answer: 3,
            proof: (N, f) => `<p>The right speech comes first in ${f(N.rag.tfidf.r1, 1)}% of the queries with TF-IDF, ${f(N.rag.bm25.r1, 1)}% with BM25, ${f(N.rag.dense.r1, 1)}% with dense embeddings and ${f(N.rag.hybrid.r1, 1)}% with the hybrid; within the first 10 chunks, ${f(N.rag.tfidf.r10, 1)}%, ${f(N.rag.bm25.r10, 1)}%, ${f(N.rag.dense.r10, 1)}% and ${f(N.rag.hybrid.r10, 1)}%.</p>` +
                `<p>Word matching (TF-IDF, BM25) finds rare exact terms; embeddings find paraphrases. They fail on different queries, so fusing their ranks beats both.</p>` +
                `<p>Retrieval is what makes the answer reliable. Asked which Executive Board member gave each of ${N.rag.qa_n} recent speeches, Qwen2.5-3B without documents is right ${f(N.rag.qa_closed.correct, 1)}% of the time and answers “${N.rag.closed_top_wrong}” ${N.rag.closed_top_wrong_n} times; with the 3 retrieved chunks it is right ${f(N.rag.qa_rag.correct, 1)}% of the time, wrong ${f(N.rag.qa_rag.wrong, 1)}%, and says “I don't know” in the rest.</p>`,
        },
    },
};
