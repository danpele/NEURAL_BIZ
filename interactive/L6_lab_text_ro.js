// Textele laboratorului cursului 6 (română); cifrele sînt completate de L6_lab.js din datele cursului.
window.LAB_T = {
    lang: 'ro',
    nogl: 'Acest browser nu poate desena grafică 3D (WebGL este oprit); rezultatele de dedesubt arată totuși vecinii.',
    lock: 'Îmi fixez predicția', run: '▶ Rulează', why: 'Arată de ce',
    afterLock: 'Predicția este fixată. Apăsați ▶ Rulează ca s-o testați pe datele cursului.',
    right: 'Predicția voastră a fost corectă.',
    wrong: o => `Predicția voastră a fost alta. Ce se întîmplă: <b>${o}</b>`,
    score: (ok, n) => `Predicțiile de pînă acum: ${ok} corecte din ${n}. Nu se salvează nimic.`,
    rank: 'Rangul cuvîntului (scară logaritmică)', freq: 'Numărul de apariții (scară logaritmică)', fitLab: s => `panta ${s}`,
    anQ: 'Analogia', anExp: 'Răspunsul așteptat', anGot: 'Răspunsul word2vec-ului nostru',
    expect: { 'germany:german::france': 'french', 'italy:italian::spain': 'spanish', 'banks:bank::households': 'household', 'rise:rising::fall': 'falling', 'higher:lower::increase': 'decrease' },
    f1: 'Macro-F1, Financial PhraseBank (5 împărțiri aleatoare)', legPb: 'PhraseBank (± 1 abatere standard)', legTw: 'tweet-uri financiare',
    cls: ['negativ', 'neutru', 'pozitiv'], trueLab: 'real', predLab: 'prezis',
    agName: k => (k === 'all' ? 'toate frazele' : `acord de ${k}%`),
    cmNote: (a, n, acc, maj) => `TF-IDF + logistic, frazele cu ${a}: ${n} predicții de test pe cele 5 împărțiri; acuratețe ${acc}% (clasa majoritară ${maj}%). Rînduri: eticheta adnotatorilor; fiecare rînd în % din totalul lui.`,
    r1read: (r, f, cum) => `Rangul ${r}: numărul de apariții ${f}; cuvintele pînă la acest rang reprezintă <b>${cum}%</b> din toți tokenii.`,
    r2read: (w, list, gl, cl) => `Cele mai apropiate cuvinte de <b>${w}</b> în word2vec-ul nostru (cosinus): ${list}. GloVe: ${gl}.` + (cl ? ` Dintre cele 32 de cuvinte desenate: ${cl}.` : ''),
    r3read: (k, n) => `${k} din ${n} analogii, calculate drept cuvîntul cel mai apropiat de b − a + c.`,
    r4read: (m, f, sd, tw) => `${m}: macro-F1 <b>${f}</b> (± ${sd}) pe PhraseBank, ${tw} pe tweet-uri.`,
    r5read: (a, acc, maj, d) => `Frazele cu ${a}: acuratețe <b>${acc}%</b>, clasa majoritară ${maj}%, o diferență de ${d} puncte.`,
    rounds: {
        r1: {
            q: 'Discursurile Băncii Centrale Europene (BCE), 2022–2026: aproximativ 1,29 milioane de tokeni și 20.130 de cuvinte diferite. Ce parte din toți tokenii reprezintă cele mai frecvente 10 cuvinte (the, of, and, …)?',
            opts: ['Aproximativ 2%', 'Aproximativ 10%', 'Aproximativ 25%', 'Aproximativ 50%'],
            answer: 2,
            proof: (N, f) => `<p>Cele mai frecvente 10 cuvinte (${N.top.slice(0, 6).join(', ')}, …) reprezintă ${f(100 * N.top10, 1)}% din cei ${N.tot.toLocaleString('ro')} de tokeni, iar cele mai frecvente 100, ${f(100 * N.top100, 1)}%. Jumătate din tokeni sînt acoperiți de cele mai frecvente ${N.r50} cuvinte, 90% de cele mai frecvente ${N.r90.toLocaleString('ro')}.</p>` +
                `<p>La celălalt capăt, ${N.hapax.toLocaleString('ro')} de cuvinte, adică ${f(100 * N.hapaxPct, 1)}% din cele ${N.types.toLocaleString('ro')} de cuvinte distincte, apar o singură dată. Pe axe logaritmice, frecvența scade aproape pe o linie dreaptă cu panta ${f(N.slope, 2)}: legea lui Zipf.</p>` +
                `<p>Pentru un model, asta înseamnă două lucruri: cîteva cuvinte funcționale domină orice numărătoare, motiv pentru care TF-IDF le dă o pondere mică, iar cea mai mare parte a vocabularului este prea rară ca să se poată învăța un vector bun pentru ea. Word2vec-ul nostru păstrează doar cuvintele văzute de cel puțin 5 ori.</p>`,
        },
        r2: {
            q: 'Propriul nostru word2vec, antrenat pe discursurile BCE (100 de dimensiuni). Care cuvînt este cel mai apropiat vecin al lui „inflation” (cea mai mare similaritate cosinus)?',
            opts: ['prices', 'headline', 'deflation', 'interest'],
            answer: 1,
            proof: (N, f) => `<p>Cele mai apropiate cuvinte de „inflation” sînt ${N.nbInf[0].slice(0, 6).map((w, i) => `${w} (${f(N.nbInf[1][i], 2)})`).join(', ')}. „prices” nu se află printre primele 12: cosinusul lui cu „inflation” este ${f(N.cosIP, 2)}, iar „interest” are ${f(N.cosII, 2)}.</p>` +
                `<p>Word2vec apropie cuvintele care apar în aceleași contexte, nu pe cele cu sensuri înrudite. În discursurile băncilor centrale, „headline inflation” și „underlying inflation” sînt expresii fixe, așa că „headline” și „underlying” au aproape toate contextele lui „inflation”. GloVe, antrenat pe același text, dă ${N.nbInfG[0].slice(0, 4).join(', ')}.</p>` +
                `<p>Imaginea 3D păstrează doar ${f(100 * N.pc3, 0)}% din varianța celor 100 de dimensiuni, deci distanțele din ea sînt doar orientative: măsura reală este cosinusul din rezultate. Alegeți alt cuvînt ca să comparați cele două modele.</p>`,
        },
        r3: {
            q: 'Analogiile se rezolvă drept cuvîntul cel mai apropiat de b − a + c, ca în celebrul king − man + woman ≈ queen. Cîte dintre aceste cinci rezolvă corect word2vec-ul nostru, antrenat pe 1,29 milioane de tokeni: germany : german :: france : ? și alte patru asemănătoare?',
            opts: ['Toate cinci', 'Cam jumătate', 'Doar una din cinci'],
            answer: 2,
            proof: (N, f) => `<p>Doar „banks : bank :: households → ${N.analogy['banks:bank::households']}” este corectă. Celelalte dau ${['germany:german::france', 'italy:italian::spain', 'rise:rising::fall', 'higher:lower::increase'].map(k => `„${N.analogy[k]}”`).join(', ')} în loc de french, spanish, falling și decrease.</p>` +
                `<p>Analogiile lui Mikolov et al. (2013) funcționau pe vectori antrenați pe miliarde de tokeni, unde fiecare relație (țară → adjectiv, verb → forma în -ing) apare de mii de ori. Discursurile BCE au ${N.corpusTok.toLocaleString('ro')} de tokeni și ${N.V.toLocaleString('ro')} de cuvinte în vocabularul word2vec; perechile țară–adjectiv apar prea rar ca să se învețe direcția „țară → adjectiv”. Relația singular/plural este destul de frecventă și funcționează.</p>` +
                `<p>Analogiile țin de corpusurile mari, nu doar de metodă. Vecinii din runda 2 sînt mult mai stabili decît aritmetica vectorilor.</p>`,
        },
        r4: {
            q: 'Sentimentul frazelor din Financial PhraseBank. Un model face media vectorilor word2vec ai cuvintelor din frază și o dă unei regresii logistice; altul folosește TF-IDF cu regresie logistică. Care obține un macro-F1 mai mare?',
            opts: ['Media vectorilor word2vec, cu peste 5 puncte', 'Cam la fel, la mai puțin de 2 puncte', 'TF-IDF, cu peste 10 puncte'],
            answer: 2,
            proof: (N, f) => `<p>TF-IDF + logistic ajunge la ${f(N.pb.tfidf_lr.f1, 1)}, media vectorilor word2vec + logistic la ${f(N.pb.w2v_lr.f1, 1)}: o diferență de ${f(N.gapPb, 1)} puncte (5 împărțiri aleatoare). Pe tweet-urile financiare, diferența este de ${f(N.gapTw, 1)} puncte.</p>` +
                `<p>Media amestecă puținele cuvinte care poartă sentimentul (rose, fell, down, improved) cu multele cuvinte neutre, iar vectorii noștri au fost antrenați pe discursurile BCE, al căror vocabular acoperă ${f(N.oov.pb_cover, 1)}% din tokenii PhraseBank. TF-IDF păstrează fiecare cuvînt ca variabilă separată și lasă regresia să pondereze direct „fell” și „rose”.</p>` +
                `<p>O medie de embedding-uri antrenată pentru această sarcină (${f(N.pb.embbag.f1, 1)}) se apropie de TF-IDF, iar pornirea ei din word2vec-ul nostru adaugă doar ${f(N.w2vGain, 2)} puncte. Vectorii preantrenați ajută cînd provin dintr-un corpus mare și potrivit; cursul 7 arată varianta cu transformere.</p>`,
        },
        r5: {
            q: 'Fiecare frază din PhraseBank a fost etichetată de mai mulți adnotatori, iar datele arată cîți dintre ei au fost de acord. Cît de precis este TF-IDF + logistic pe frazele asupra cărora toți au fost de acord, față de cele cu acordul doar al jumătății?',
            opts: ['Cam la fel, în jur de 78% pentru ambele', 'Mult mai bine la acordul complet: aproximativ 88% față de 58%', 'Mai bine pe frazele cu acord de jumătate, fiindcă sînt mai variate'],
            answer: 1,
            proof: (N, f) => `<p>Acuratețea este ${f(N.acc('100'), 1)}% pe frazele cu acord complet, ${f(N.acc('75'), 1)}% cu acord de 75%, ${f(N.acc('66'), 1)}% cu 66% și ${f(N.acc('50'), 1)}% cu 50%. La un acord de 50%, clasa majoritară singură obține ${f(N.maj('50'), 1)}%, deci modelul abia depășește răspunsul „neutru” dat mereu.</p>` +
                `<p>Cînd jumătate dintre adnotatori citesc o frază ca pozitivă și jumătate ca neutră, nu există o singură etichetă corectă: modelul nu poate avea dreptate mai des decît eticheta însăși. Erorile se concentrează acolo unde și oamenii sînt în dezacord.</p>` +
                `<p>Înainte de a da vina pe model, verificați calitatea etichetelor. Acuratețea raportată pe niveluri de acord îi arată cititorului cît din eroare este zgomot în date.</p>`,
        },
    },
};
