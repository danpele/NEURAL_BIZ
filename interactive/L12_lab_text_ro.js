// Textele laboratorului cursului 12 (română); cifrele sînt completate de L12_lab.js din datele cursului.
window.LAB_T = {
    lang: 'ro',
    nogl: 'Acest browser nu poate desena grafică 3D (WebGL este oprit); rezultatele de dedesubt arată totuși fiecare temperatură.',
    lock: 'Îmi fixez predicția', run: '▶ Rulează', why: 'Arată de ce',
    afterLock: 'Predicția este fixată. Apăsați ▶ Rulează ca s-o testați pe datele cursului.',
    right: 'Predicția voastră a fost corectă.',
    wrong: o => `Predicția voastră a fost alta. Ce se întîmplă: <b>${o}</b>`,
    score: (ok, n) => `Predicțiile de pînă acum: ${ok} corecte din ${n}. Nu se salvează nimic.`,
    key3d: 'cei mai probabili 8 tokeni următori, de la „the” la „in” · temperatura de la 0,25 la 2 · înălțimea: probabilitatea relativă la „the” · barele roșii: temperatura curentă',
    model: { majority: 'Clasa majoritară', tfidf: 'TF-IDF + logistic (antrenat)', 'zs_0.5B': 'Qwen2.5-0.5B, fără exemple', 'zs_1.5B': 'Qwen2.5-1.5B, fără exemple',
        'zs_3B': 'Qwen2.5-3B, fără exemple', probe: 'Stările Qwen2.5-1.5B + logistic', bm25: 'BM25', dense: 'embedding-uri dense', hybrid: 'hibrid' },
    ret: { tfidf: 'TF-IDF', bm25: 'BM25', dense: 'embedding-uri dense', hybrid: 'hibrid' },
    f1pb: 'Macro-F1 pe 678 de fraze de test PhraseBank (interval bootstrap de 95%)', f1short: 'Macro-F1, PhraseBank', shots: 'Exemple etichetate în prompt',
    instr: 'Instrucțiunea', pb: 'PhraseBank', tw: 'tweet-uri financiare',
    psNote: d => `Macro-F1 pe setul de test ${d}, fără exemple; același model și aceleași date, se schimbă doar formularea.`,
    k: 'Numărul de fragmente regăsite (k)', recall: 'Discursul corect printre primele k',
    r1read: (t, r, p, e) => `T = ${t}: „The” are de <b>${r}</b> ori probabilitatea lui „the”` + (p ? `; „the” are <b>${p}%</b>, entropia ${e} naturali (nats).` : '.'),
    r2read: (m, f, lo, hi) => `${m}: macro-F1 <b>${f}</b>` + (lo ? ` (interval de 95% între ${lo} și ${hi}).` : '.'),
    r3read: (k, m, seeds, m15) => `${k === 1 ? 'Un exemplu' : k === 0 ? 'Niciun exemplu' : k + ' exemple'}: Qwen2.5-3B <b>${m}</b> (trei extrageri de exemple: ${seeds}); Qwen2.5-1.5B ${m15}.`,
    r4read: (i, v) => `Formularea ${i}: Qwen2.5-1.5B, macro-F1 <b>${v}</b>.`,
    r4done: (a, b) => `Diferența dintre cea mai bună și cea mai slabă formulare: <b>${a} puncte</b> la Qwen2.5-1.5B, ${b} la Qwen2.5-3B.`,
    r5read: (k, list) => `Primele ${k} fragmente: ${list}.`,
    rounds: {
        r1: {
            q: 'Qwen2.5-1.5B citește începutul unei fraze dintr-un discurs al lui Philip R. Lane (30 octombrie 2026): „The inflation impact is limited:”. La temperatura T = 1, cel mai probabil token următor, „the”, are probabilitatea 10,1%. Ce probabilitate are la T = 0,5?',
            opts: ['Aproximativ 5%: o temperatură mai mică o înjumătățește', 'Tot aproximativ 10%: temperatura schimbă doar cît de aleatoare este alegerea', 'Aproximativ 20%', 'Aproximativ 50%'],
            answer: 3,
            proof: (N, f) => `<p>Temperatura împarte scorurile înainte de softmax, așa că fiecare probabilitate devine proporțională cu p<sup>1/T</sup>. La T = 0,5, fiecare probabilitate este ridicată la pătrat și apoi renormalizată. Coada lungă a celor ${(151665).toLocaleString('ro')} de tokeni din vocabular, fiecare cu o probabilitate foarte mică, se micșorează mult mai mult decît vîrful: „the” urcă de la ${f(N.next.top1_p, 1)}% la ${f(N.next.top1_p05, 1)}%, iar entropia scade de la ${f(N.next.entropy.t10, 2)} la ${f(N.next.entropy.t05, 2)} nats.</p>` +
                `<p>La T = 2 se întîmplă invers: „the” coboară la ${f(N.next.top1_p2, 1)}%, iar entropia urcă la ${f(N.next.entropy.t20, 2)} nats, aproape de valoarea ${f(N.entMax, 2)} a unei alegeri uniforme din tot vocabularul. Raportul dintre doi tokeni, pe care îl arată relieful 3D, este (p<sub>2</sub>/p<sub>1</sub>)<sup>1/T</sup>: ${f(N.ratio2(1), 2)} la T = 1 și ${f(N.ratio2(0.5), 2)} la T = 0,5.</p>` +
                `<p>Temperatura nu schimbă niciodată ordinea, deci decodarea greedy alege „the” la orice T. Cuvîntul pe care Lane l-a spus de fapt a fost „${N.next.true_next}”, care nu se află printre cei mai probabili 8 tokeni. Pentru clasificare sau extragere de informații folosiți T aproape de 0; un T mare adaugă doar varietate.</p>`,
        },
        r2: {
            q: 'Sentimentul a 678 de fraze de test din PhraseBank, fără niciun exemplu de antrenare: Qwen2.5 cu 0,5, 1,5 și 3 miliarde de parametri, față de TF-IDF + regresie logistică antrenat pe 1.581 de fraze. Ce este adevărat?',
            opts: ['Toate trei modelele de limbaj depășesc TF-IDF', 'Doar modelul 3B depășește TF-IDF; modelul 0,5B este aproape de clasa majoritară', 'Niciunul nu depășește TF-IDF'],
            answer: 1,
            proof: (N, f) => `<p>Macro-F1: ${f(N.f1['zs_0.5B'], 1)} pentru 0,5B, față de ${f(N.cls.majority.f1, 1)} pentru răspunsul dat mereu cu clasa majoritară; ${f(N.f1['zs_1.5B'], 1)} pentru 1,5B; ${f(N.f1['zs_3B'], 1)} pentru 3B (interval bootstrap de 95% între ${f(N.cls.zs_3B.lo, 1)} și ${f(N.cls.zs_3B.hi, 1)}). TF-IDF ajunge la ${f(N.f1.tfidf, 1)} (între ${f(N.cls.tfidf.lo, 1)} și ${f(N.cls.tfidf.hi, 1)}).</p>` +
                `<p>Capacitatea de a urma o instrucțiune apare odată cu mărimea: modelul 0,5B răspunde „neutral” la ${N.zsNeu} din cele 678 de fraze, modelul 1,5B este la nivelul TF-IDF, iar modelul 3B îl depășește fără să fi văzut nicio frază etichetată. O regresie logistică pe stările ascunse ale modelului 1,5B obține ${f(N.f1.probe, 1)}.</p>` +
                `<p>Prețul este viteza: ${f(N.cls.zs_3B.ms, 1)} ms pe frază pentru modelul 3B, față de ${f(N.cls.tfidf.ms, 1)} ms pentru TF-IDF pe același laptop. Cînd există destule etichete, un model antrenat ieftin rămîne un reper puternic.</p>`,
        },
        r3: {
            q: 'Modelul 3B obține 92,2 fără niciun exemplu în prompt. Adăugăm un exemplu etichetat (trei extrageri aleatoare diferite ale exemplului). Ce se întîmplă cu macro-F1?',
            opts: ['Crește, la aproximativ 95', 'Rămîne în jur de 92', 'Scade, în medie la aproximativ 89,5'],
            answer: 2,
            proof: (N, f) => `<p>Pentru Qwen2.5-3B, macro-F1 mediu este ${N.fsMean('3B').map((v, i) => `${f(v, 1)} cu ${[0, 1, 2, 4, 8][i]}`).join(', ')} exemple. Cu un singur exemplu, cele trei extrageri dau ${N.fs['3B'][1].map(v => f(v, 1)).join('; ')}. Pentru Qwen2.5-1.5B: ${N.fsMean('1.5B').map(v => f(v, 1)).join('; ')}.</p>` +
                `<p>Un singur exemplu trage modelul spre propria lui etichetă și formulare; contează la fel de mult ce exemplu este extras cît și cîte sînt. Un model mai slab cîștigă puțin din unul sau două exemple, unul mai puternic poate pierde, iar opt exemple nu sînt mai bune decît patru pentru niciunul.</p>` +
                `<p>Promptarea cu exemple (few-shot) este un hiperparametru ca oricare altul: încercați-o pe un set de validare, cu mai multe extrageri de exemple, înainte de a vă baza pe ea.</p>`,
        },
        r4: {
            q: 'Cinci formulări ale aceleiași instrucțiuni (răspundeți positive, negative sau neutral), Qwen2.5-1.5B, fără exemple, pe PhraseBank. Cît de departe sînt cel mai bun și cel mai slab macro-F1?',
            opts: ['Sub 1 punct', 'Aproximativ 3 puncte', 'Peste 10 puncte'],
            answer: 2,
            proof: (N, f) => `<p>Pe PhraseBank, Qwen2.5-1.5B merge de la ${f(Math.min(...N.ps['pb_1.5B']), 1)} la ${f(Math.max(...N.ps['pb_1.5B']), 1)}, o diferență de ${f(N.range('pb_1.5B'), 1)} puncte; Qwen2.5-3B de la ${f(Math.min(...N.ps['pb_3B']), 1)} la ${f(Math.max(...N.ps['pb_3B']), 1)} (${f(N.range('pb_3B'), 1)} puncte). Pe tweet-uri, chiar și modelul 3B variază cu ${f(N.range('tw_3B'), 1)} puncte.</p>` +
                `<p>Formulările pun aceeași întrebare; se schimbă ordinea etichetelor, punctul de vedere (investitor, acționar, comunicat de presă) și cît de strict este constrîns răspunsul. Modelele mai mici urmează aceste detalii mai literal.</p>` +
                `<p>Un rezultat obținut cu un prompt este un rezultat pentru acel prompt. Alegeți formularea pe date de validare, raportați-o integral și fixați-o înainte de a vă uita la setul de test.</p>`,
        },
        r5: {
            q: 'Regăsirea pentru RAG: interogarea este titlul unui discurs al BCE, iar sistemul trebuie să găsească acel discurs printre 400 (8.400 de fragmente de 200 de cuvinte). Care metodă de regăsire pune cel mai des discursul corect pe primul loc?',
            opts: ['TF-IDF', 'BM25', 'Embedding-uri dense (bge-small)', 'Un hibrid între rangurile BM25 și cele dense'],
            answer: 3,
            proof: (N, f) => `<p>Discursul corect apare primul în ${f(N.rag.tfidf.r1, 1)}% din interogări cu TF-IDF, ${f(N.rag.bm25.r1, 1)}% cu BM25, ${f(N.rag.dense.r1, 1)}% cu embedding-uri dense și ${f(N.rag.hybrid.r1, 1)}% cu hibridul; printre primele 10 fragmente, ${f(N.rag.tfidf.r10, 1)}%, ${f(N.rag.bm25.r10, 1)}%, ${f(N.rag.dense.r10, 1)}% și ${f(N.rag.hybrid.r10, 1)}%.</p>` +
                `<p>Potrivirea pe cuvinte (TF-IDF, BM25) găsește termenii rari exacți; embedding-urile găsesc parafrazele. Ele greșesc pe interogări diferite, așa că îmbinarea rangurilor le bate pe amîndouă.</p>` +
                `<p>Regăsirea este cea care face răspunsul de încredere. Întrebat care membru al Comitetului executiv a ținut fiecare dintre cele ${N.rag.qa_n} discursuri recente, Qwen2.5-3B fără documente răspunde corect în ${f(N.rag.qa_closed.correct, 1)}% din cazuri și spune „${N.rag.closed_top_wrong}” de ${N.rag.closed_top_wrong_n} de ori; cu cele 3 fragmente regăsite răspunde corect în ${f(N.rag.qa_rag.correct, 1)}% din cazuri, greșit în ${f(N.rag.qa_rag.wrong, 1)}%, iar în rest spune „nu știu”.</p>`,
        },
    },
};
