// Textele laboratorului cursului 2 (română); cifrele sînt completate de L2_lab.js din datele cursului.
window.LAB_T = {
    lang: 'ro',
    nogl: 'Acest browser nu poate desena grafică 3D (WebGL este oprit); rezultatele de dedesubt arată totuși fiecare pas.',
    lock: 'Îmi fixez predicția', run: '▶ Rulează', why: 'Arată de ce',
    afterLock: 'Predicția este fixată. Apăsați ▶ Rulează ca s-o testați pe suprafață; trageți de suprafață ca s-o rotiți.',
    right: 'Predicția voastră a fost corectă.',
    wrong: o => `Predicția voastră a fost alta. Ce se întîmplă: <b>${o}</b>`,
    score: (ok, n) => `Predicțiile de pînă acum: ${ok} corecte din ${n}. Nu se salvează nimic.`,
    readout: (k, a, b, L, m, off) => off
        ? `Pasul ${k}: bila a ieșit de pe suprafață; pierderea crește nelimitat.`
        : `Pasul ${k}: termenul liber a = ${a}, panta b = ${b}, MSE = <b>${L}</b> (minimul ${m}).`,
    diverged: 'divergent', r4x: 'Epoca', r4y: 'MSE pe validare (scară logaritmică)',
    r4lin: v => `regresia liniară ${v}`,
    r4read: (lr, e, cur, f, div) => `η = ${lr}, epoca ${e}: MSE pe validare ${cur === null ? '<b>divergent</b>' : '<b>' + cur + '</b>'} (sămînța 0). ` +
        `După 30 de epoci, cele trei semințe: ${f.join('; ')}${div ? ` (${div} din 3 divergente)` : ''}.`,
    rounds: {
        r1: {
            q: 'Bila pornește de la termenul liber a = 4 și panta b = −0,5. Face 60 de pași de coborîre pe gradient cu pasul sigur η = 1/λ<sub>max</sub>. Unde va ajunge?',
            opts: ['La minim sau foarte aproape de el', 'Undeva pe fundul văii, încă departe de minim', 'Sare afară din vale'],
            answer: 1,
            proof: (N, f) => `<p>Pierderea se curbează foarte diferit pe cele două direcții. Matricea derivatelor a doua ale MSE are valorile proprii λ<sub>max</sub> = ${f(N.lmax, 2)} și λ<sub>min</sub> = ${f(N.lmin, 3)}, deci numărul de condiționare este ${f(N.cond, 1)}.</p>` +
                `<p>Pe fiecare dintre aceste direcții, un pas înmulțește eroarea rămasă cu (1 − ηλ). Cu η = 1/λ<sub>max</sub> = ${f(N.eta, 4)}, direcția abruptă este rezolvată într-un singur pas. Cea plată păstrează un factor 1 − λ<sub>min</sub>/λ<sub>max</sub> = ${f(N.flat, 4)} la fiecare pas: după 60 de pași a rămas încă ${f(N.flat60 * 100, 0)}% din eroarea ei.</p>` +
                `<p>MSE scade de la ${f(N.start, 3)} la ${f(N.gd60, 3)}, față de un minim de ${f(N.mstar, 3)} în a = ${f(N.intercept, 4)}, b = ${f(N.slope, 4)}. Acum duceți numărul de pași la 400 și rulați din nou.</p>`,
        },
        r2: {
            q: 'Același punct de plecare, 60 de pași, dar pasul este 2,1 × 1/λ<sub>max</sub>, cam dublul pasului sigur. Ce se întîmplă?',
            opts: ['Ajunge mai repede la minim', 'Face zig-zag peste vale, apoi se așază', 'Sare tot mai sus la fiecare pas și iese de pe suprafață'],
            answer: 2,
            proof: (N, f) => `<p>Pe direcția abruptă, eroarea se înmulțește la fiecare pas cu 1 − ηλ<sub>max</sub> = 1 − 2,1 = −1,1: își schimbă semnul și crește cu 10% de fiecare dată. După 60 de pași, factorul este 1,1<sup>60</sup> ≈ ${f(Math.pow(1.1, 60), 0)}.</p>` +
                `<p>Pe o pierdere pătratică, coborîrea pe gradient converge doar dacă |1 − ηλ| &lt; 1 pe toate direcțiile, adică η &lt; 2/λ<sub>max</sub> = ${f(N.etaMax, 4)}. Între 1 și 2 ori 1/λ<sub>max</sub>, bila face zig-zag și tot se așază: încercați 1,5. Peste 2 diverge: încercați 1,95 și 2,05.</p>` +
                `<p>Direcția cea mai abruptă fixează pasul cel mai mare pe care îl puteți face, iar direcția cea mai plată fixează apoi cît așteptați.</p>`,
        },
        r3: {
            q: 'Acum venitul este standardizat (media 0, abaterea standard 1), iar regula pasului este aceeași, η = 1/λ<sub>max</sub>. De cîți pași are nevoie bila ca să ajungă la minim?',
            opts: ['Un singur pas', 'Cam zece pași', 'Încă se tîrăște după 60 de pași'],
            answer: 0,
            proof: (N, f) => `<p>Cu intrarea standardizată, matricea derivatelor a doua devine de 2 ori matricea unitate: λ = ${f(N.lStd, 2)} pe toate direcțiile, iar numărul de condiționare este ${f(N.condStd, 1)}. Valea devine rotundă.</p>` +
                `<p>Atunci η = 1/λ = 0,5, iar un pas dă w<sub>1</sub> = w<sub>0</sub> − 0,5 (2w<sub>0</sub> − 2w<sup>*</sup>) = w<sup>*</sup>: MSE este ${f(N.std1, 3)} după un pas, adică minimul.</p>` +
                `<p>A doua soluție, pe scara brută, este momentul: cu β = 0,9 și jumătate din pas, MSE după 60 de pași este ${f(N.mom60, 3)}, față de ${f(N.gd60, 3)} la coborîrea simplă pe gradient. Alegeți „venit brut, cu moment” și rulați. De aceea fiecare notebook al cursului standardizează intrările înainte de antrenare.</p>`,
        },
        r4: {
            q: 'O rețea reală: două straturi ascunse a cîte 64 de unități ReLU, cele 8 variabile standardizate din California housing, coborîre stocastică pe gradient simplă, loturi de 128, 30 de epoci. Ce rată de învățare dă cea mai mică MSE pe validare?',
            opts: ['η = 0,001', 'η = 0,01', 'η = 0,1', 'η = 1'],
            answer: 2,
            proof: (N, f) => `<p>η = ${f(N.best.lr, 1)} dă cea mai mică MSE pe validare, ${f(N.best.med, 3)} (mediana a trei semințe), față de ${f(N.lin, 3)} la regresia liniară pe aceleași variabile.</p>` +
                `<p>Un pas mai sus, la η = ${f(N.firstDiv.lr, 1)}, ${N.firstDiv.div} din cele 3 semințe diverg, iar la η = 1 diverg toate trei. O rată prea mică este lentă, dar nu greșită: la η = 0,0001, MSE este încă ${f(N.slow.med, 3)} după 30 de epoci.</p>` +
                `<p>Ca pe suprafața pătratică, unde limita era 2/λ<sub>max</sub>, rata cea mai bună stă chiar sub pragul de divergență. De aceea testele de alegere a ratei de învățare parcurg η pe scară logaritmică și păstrează cea mai mare rată care încă antrenează stabil.</p>`,
        },
    },
};
