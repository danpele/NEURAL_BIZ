// Textele laboratorului cursului 3 (română); cifrele sînt completate de L3_lab.js din datele cursului.
window.LAB_T = {
    lang: 'ro',
    nogl: 'Acest browser nu poate desena grafică 3D (WebGL este oprit); rezultatele de dedesubt arată totuși fiecare epocă.',
    lock: 'Îmi fixez predicția', run: '▶ Rulează', why: 'Arată de ce',
    afterLock: 'Predicția este fixată. Apăsați ▶ Rulează ca s-o testați pe datele cursului.',
    right: 'Predicția voastră a fost corectă.',
    wrong: o => `Predicția voastră a fost alta. Ce se întîmplă: <b>${o}</b>`,
    score: (ok, n) => `Predicțiile de pînă acum: ${ok} corecte din ${n}. Nu se salvează nimic.`,
    ep: 'Epoca', lossY: 'Log loss (scară logaritmică)', aucY: 'AUC pe validare', legVal: 'validare', legTr: 'antrenare',
    logitLine: v => `regresia logistică ${v}`, bestLoss: e => `pierderea minimă pe validare (epoca ${e})`,
    wdTick: v => `penalizare ${v}`, dpTick: v => `dropout ${v}`,
    planeKey: v => `<span class="sw-gold"></span>planul auriu: regresia logistică ${v}; înălțimea barei: pierderea pe validare`,
    cell: (p, wd) => `dropout ${p}, penalizarea ponderilor ${wd}`,
    esLossY: 'Pierderea pe validare', esApY: 'Precizia medie', stopAt: e => `oprire: epoca ${e}`,
    ddX: 'Numărul de variabile aleatoare p (scară logaritmică)', ddY: 'MSE (scară logaritmică)', pn: n => `p = n = ${n}`,
    ddBestLine: (v, p) => `cel mai bun sub n: ${v} (p = ${p})`, legTe: 'test (mediana)', legTrDd: 'antrenare (mediana)',
    big: v => `${v} milioane`,
    r1read: (e, tr, va, lg) => `Epoca ${e}: pierderea pe antrenare ${tr}, pierderea pe validare <b>${va}</b> (regresia logistică ${lg}).`,
    r2read: (e, va, auc, lg) => `Epoca ${e}: pierderea pe validare ${va}, AUC pe validare <b>${auc}</b> (regresia logistică ${lg}).`,
    r3read: (e, lo, loN, hi, hiN) => `Epoca ${e}: cea mai mică pierdere pe validare <b>${lo}</b> (${loN}); cea mai mare ${hi} (${hiN}). Trageți de bare ca să le rotiți.`,
    r4read: (e, va, ap) => `Epoca ${e}: pierderea pe validare ${va}, precizia medie <b>${ap}</b>.`,
    r5read: (p, m, lo, hi, tr) => `p = ${p} variabile: MSE pe test <b>${m}</b> (mediana a 10 extrageri; între ${lo} și ${hi}); MSE pe antrenare ${tr}.`,
    rounds: {
        r1: {
            q: 'O rețea cu două straturi ascunse a cîte 256 de unități, adică aproximativ 79.000 de ponderi, învață din 600 de solicitanți de credit din Germania, fără regularizare. Pierderea pe antrenare scade spre zero. Unde este pierderea pe validare (log loss pe alți 200 de solicitanți) după 150 de epoci?',
            opts: ['Sub regresia logistică: mai multă antrenare, un model mai bun', 'Constantă la nivelul ei cel mai bun, în jur de 0,54', 'Mult peste nivelul ei cel mai bun, mai rea decît după prima epocă'],
            answer: 2,
            proof: (N, f) => `<p>Pierderea pe validare este minimă la epoca ${N.best}, ${f(N.vmin, 3)}, apoi urcă la ${f(N.vend, 3)} la epoca ${N.epochs}: de ${f(N.ratio, 1)} ori nivelul ei cel mai bun și peste valoarea ${f(N.v1, 3)} de după prima epocă. Pierderea pe antrenare ajunge la ${f(N.tend, 4)}.</p>` +
                `<p>Cu aproximativ 79.000 de ponderi pentru 600 de rînduri, rețeaua poate potrivi fiecare solicitant din setul de antrenare, cu tot cu zgomot. De la epoca ${N.best} încolo, fiecare epocă în plus potrivește puțin mai bine acel zgomot și puțin mai prost solicitanții noi. Regresia logistică, cu 49 de coeficienți, ajunge la ${f(N.logit, 3)} și nu poate supraajusta în acest fel.</p>` +
                `<p>O pierdere pe antrenare care scade arată doar că optimizatorul funcționează. Dacă modelul generalizează se poate citi doar pe date pe care nu le-a văzut.</p>`,
        },
        r2: {
            q: 'Aceeași rulare. La epoca 150, pierderea pe validare este de aproximativ patru ori nivelul ei cel mai bun. Ce se întîmplă cu AUC pe validare, adică ordonarea debitorilor răi deasupra celor buni, între epoca 3 și epoca 150?',
            opts: ['Se prăbușește spre 0,5', 'Scade puțin, odată cu pierderea', 'Nu scade: la final este mai mare decît la epoca 3'],
            answer: 2,
            proof: (N, f) => `<p>AUC pe validare este ${f(N.aucBest, 3)} la epoca ${N.best}, unde pierderea este minimă, și ${f(N.aucEnd, 3)} la epoca ${N.epochs}, cu un maxim de ${f(N.aucMax, 3)} la epoca ${N.aucMaxEp}. Regresia logistică ajunge la ${f(N.logitAuc, 3)}.</p>` +
                `<p>Cele două metrici măsoară lucruri diferite. Log loss penalizează greșelile făcute cu încredere: un solicitant care nu plătește, dar primește o probabilitate de neplată de 0,001, costă singur −ln 0,001 ≈ 6,9. Rețeaua supraajustată își împinge probabilitățile spre 0 și 1, așa că puținele ei greșeli devin foarte scumpe, în timp ce ordinea solicitanților, singurul lucru pe care îl vede AUC, se schimbă puțin.</p>` +
                `<p>Cu 200 de solicitanți, o singură împărțire este un judecător zgomotos: pe ${N.hoN} de împărțiri aleatoare ale acelorași 1.000 de solicitanți, AUC pe test al aceleiași regresii logistice variază între ${f(N.hoMin, 3)} și ${f(N.hoMax, 3)} (abaterea standard ${f(N.hoSd, 3)}), deci creșterea sugerează ceva, dar nu dovedește. Ce se poate spune sigur: ordonarea nu s-a prăbușit, dar probabilitățile au devenit prea sigure pe ele. O bancă ce stabilește prețul creditului după probabilitate are nevoie de pierdere, iar după ea trebuie oprită antrenarea.</p>`,
        },
        r3: {
            q: 'Nouă rețele cu aceeași configurație: dropout 0, 0,3 sau 0,5 și penalizarea ponderilor 0, 0,1 sau 1 (AdamW, rata de învățare 0,001). După 150 de epoci, care parametru scade mai mult pierderea pe validare?',
            opts: ['Dropout, de la 0 la 0,5', 'Penalizarea ponderilor, de la 0 la 1', 'Amîndoi cam la fel', 'Niciunul: toate nouă ajung în jur de 2'],
            answer: 1,
            proof: (N, f) => `<p>După ${N.epochs} de epoci: ${f(N.none, 3)} fără niciunul, ${f(N.dp, 3)} doar cu dropout 0,5, ${f(N.wd, 3)} doar cu penalizarea ponderilor 1 și ${f(N.both, 3)} cu amîndouă. Penalizarea ponderilor scade pierderea cu aproximativ ${f(N.none - N.wd, 2)}, dropout cu aproximativ ${f(N.none - N.dp, 2)}, iar dropout nu mai adaugă nimic peste penalizarea ponderilor.</p>` +
                `<p>În AdamW, fiecare pas înmulțește fiecare pondere cu (1 − ηλ) = 1 − 0,001 × 1 înainte de actualizarea cu gradientul. În ${N.steps.toLocaleString('ro')} de pași (10 loturi de 64 pe epocă), doar acest factor ar reduce o pondere la ${f(N.shrink * 100, 0)}% din valoarea ei, așa că ponderile rămîn mici, iar probabilitățile prezise rămîn departe de 0 și 1, ceea ce răsplătește log loss. Dropout face fiecare unitate mai puțin dependentă de celelalte, dar rețeaua își poate crește în continuare ponderile și poate deveni prea sigură pe ea.</p>` +
                `<p>Chiar și cea mai bună configurație ajunge la ${f(N.gridBest, 3)}, aproape dublul valorilor de ${f(N.gridMinEarly, 3)} pînă la ${f(N.gridMaxEarly, 3)} atinse de cele nouă configurații în primele epoci. Regularizarea încetinește supraajustarea; runda următoare arată ce o oprește.</p>`,
        },
        r4: {
            q: 'Fraudă cu carduri: o rețea 256-256 fără regularizare, 120 de epoci pe o împărțire în ordinea timpului. Oprirea timpurie păstrează epoca cu cea mai mică pierdere pe validare. Cît de bună este acea epocă pe metrica ce contează pentru bancă, precizia medie (PR-AUC, aria de sub curba precizie-recall)?',
            opts: ['Slabă: precizia medie continuă să crească mult după minimul pierderii', 'La mai puțin de 0,01 de cea mai bună dintre cele 120 de epoci, în timp ce antrenarea pînă la capăt pierde aproximativ 0,07', 'La fel ca la final: precizia medie aproape nu se mai mișcă după primele epoci'],
            answer: 1,
            proof: (N, f) => `<p>Pierderea pe validare este minimă la epoca ${N.esBest} (${f(N.esVmin, 4)}), unde precizia medie este ${f(N.esApAt, 3)}. Cea mai bună precizie medie din cele ${N.esEp} de epoci este ${f(N.esApMax, 3)}, la epoca ${N.esApMaxEp}: cu ${f(N.esApMax - N.esApAt, 3)} mai mult. După ${N.esEp} de epoci, pierderea este ${f(N.esVend, 4)}, iar precizia medie ${f(N.esApEnd, 3)}, cu ${f(N.esApMax - N.esApEnd, 3)} sub cea mai bună.</p>` +
                `<p>Cu o răbdare de 20 de epoci, antrenarea se oprește la epoca ${N.esStop} și reface ponderile de la epoca ${N.esBest}, ceea ce economisește ${N.esEp - N.esStop} din cele ${N.esEp} de epoci. Pierderea pe validare este mult mai puțin zgomotoasă decît precizia medie și este disponibilă după fiecare epocă, deci este un semnal de oprire ieftin chiar și cînd decizia se ia pe altă metrică.</p>` +
                `<p>Spre deosebire de AUC din runda 2, precizia medie chiar scade aici: ea se uită doar la vîrful ordonării, acolo unde stau greșelile prea sigure ale unei rețele supraajustate. Mutați cursorul ca să comparați orice epocă de oprire.</p>`,
        },
        r5: {
            q: 'Regresie cu variabile aleatoare pe California housing: 400 de grupuri de locuințe pentru antrenare, p variabile ReLU aleatoare, potrivite prin cele mai mici pătrate (soluția de normă minimă cînd p > 400), 10 extrageri aleatoare. Cînd p crește de la 10 la 8.000, ce face MSE pe test?',
            opts: ['Scade continuu: mai multe variabile, eroare mai mică', 'O formă de U: scade, apoi crește și rămîne mare pentru p mare', 'Scade, explodează în jur de p = 400, apoi scade din nou'],
            answer: 2,
            proof: (N, f) => `<p>MSE mediană pe test este ${f(N.ddSmall, 3)} la p = ${N.ddSmallP} și ${f(N.ddBest, 3)} la p = ${N.ddBestP}, cea mai bună sub n. Urcă deja la ${f(N.ddRise, 0)} la p = ${N.ddRiseP}; la p = ${N.ddPeakP} = n ajunge la aproximativ ${f(N.ddPeak / 1e6, 1)} milioane, iar la p = ${N.ddBigP.toLocaleString('ro')} revine la ${f(N.ddBig, 3)}.</p>` +
                `<p>La p = n există exact atîția coeficienți cîte puncte de antrenare: potrivirea trece aproape exact prin fiecare punct (MSE pe antrenare ${f(N.ddTrPeak, 3)}) și, fiindcă unele direcții ale matricei de variabile sînt aproape plate, are nevoie de coeficienți uriași pentru asta. Peste n, o infinitate de potriviri trec prin fiecare punct, iar cele mai mici pătrate o aleg pe cea cu coeficienții cei mai mici. Cu cît sînt mai multe variabile, cu atît această normă poate fi mai mică, deci potrivirea devine din nou mai netedă: aceasta este regularizarea implicită.</p>` +
                `<p>Aici a doua coborîre, ${f(N.ddBig, 3)}, nu bate cel mai bun rezultat clasic, ${f(N.ddBest, 3)}. Modelele mari pot generaliza dincolo de punctul de interpolare, dar mai mare nu înseamnă automat mai bun: comparația trebuie făcută pe date păstrate deoparte.</p>`,
        },
    },
};
