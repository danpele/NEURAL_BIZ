// Textele laboratorului cursului 7 (română); cifrele sînt completate de L7_lab.js din datele cursului.
window.LAB_T = {
    lang: 'ro',
    nogl: 'Acest browser nu poate desena grafică 3D (WebGL este oprit); rezultatele de dedesubt arată totuși fiecare pas.',
    lock: 'Îmi fixez predicția', run: '▶ Rulează', why: 'Arată de ce',
    afterLock: 'Predicția este fixată. Apăsați ▶ Rulează ca s-o testați pe datele cursului.',
    right: 'Predicția voastră a fost corectă.',
    wrong: o => `Predicția voastră a fost alta. Ce se întîmplă: <b>${o}</b>`,
    score: (ok, n) => `Predicțiile de pînă acum: ${ok} corecte din ${n}. Nu se salvează nimic.`,
    key3d: 'rînduri (stînga): tokenul care privește (interogarea) · coloane (în față): tokenul privit (cheia) · înălțimea: ponderea atenției',
    layer: 'Stratul', share: 'Ponderea din toată atenția', onSep: 'pe [SEP]', onCls: 'pe [CLS]',
    f1: 'Macro-F1 pe cele 2.388 de tweet-uri de test', nTrain: 'Tweet-uri etichetate pentru antrenare (scară logaritmică)',
    ftLine: v => `FinBERT ajustat pe toate tweet-urile ${v}`,
    ds: { probe: 'FinBERT înghețat + logistic', tfidf: 'TF-IDF + logistic', tiny: 'transformer mic' },
    model: { tfidf: 'TF-IDF + logistic', embbag: 'Medie de embedding-uri (fără atenție)', tiny_nopos: 'Transformer mic fără poziții',
        tiny: 'Transformer mic', finbert_zs: 'FinBERT fără antrenare', finbert_probe: 'FinBERT înghețat + logistic', finbert_ft: 'FinBERT ajustat (fine-tuned)' },
    r1read: (l, h, t, m, one, n) => `Stratul ${l}, capul ${h}, scoruri împărțite la ${t}: cea mai mare pondere dintr-un rînd este în medie <b>${m}</b>; ${one} din ${n} rînduri pun peste 90% pe un singur token.`,
    r2read: (l, e, e500) => `Stratul ${l}: entropia normalizată a rîndurilor <b>${e}</b> pentru acest tweet (1 = distribuită uniform, 0 = un singur token); ${e500} în medie pe 500 de tweet-uri de test.`,
    r3read: (l, s, c, s7) => `Stratul ${l}: <b>${s}%</b> din toată atenția pe [SEP], ${c}% pe [CLS] (500 de tweet-uri de test); ${s7}% pe [SEP] în cele șapte tweet-uri din runda 2.`,
    r4read: (m, v, seeds) => `${m}: macro-F1 <b>${v}</b> (rulări: ${seeds}).`,
    r5read: (n, pct, p, t, s) => `${n} tweet-uri etichetate (${pct}% din setul de antrenare): FinBERT înghețat + logistic <b>${p}</b>, TF-IDF + logistic ${t}, transformer mic ${s}.`,
    rounds: {
        r1: {
            q: 'FinBERT citește tweet-ul „BTIG points to breakfast pressure for Dunkin’ Brands”. Fiecare rînd de atenție este softmax(q·k / √64): scorurile brute împărțite la 8. Ce se întîmplă dacă renunțăm la împărțirea la 8?',
            opts: ['Aproape nimic: softmax păstrează ordinea scorurilor, deci ponderile abia se schimbă', 'Majoritatea rîndurilor pun peste 90% din pondere pe un singur token', 'Fiecare rînd își împarte ponderea egal între toate cele 13 tokenuri'],
            answer: 1,
            proof: (N, f) => `<p>Scorurile brute q·k ale acestui tweet sînt între ${f(N.sLo, 1)} și ${f(N.sHi, 1)}. Pe toate cele 144 de capete (12 straturi × 12 capete), cea mai mare pondere dintr-un rînd este în medie ${f(N.max8, 2)} cu împărțirea la 8 și ${f(N.max1, 2)} fără ea; rîndurile cu peste 90% pe un singur token cresc de la ${f(100 * N.one8, 0)}% la ${f(100 * N.one1, 0)}%.</p>` +
                `<p>q·k este o sumă de 64 de produse. Dacă fiecare produs are varianța în jur de 1, suma are varianța 64 și abaterea standard 8, deci scorurile brute diferă ușor cu zeci de unități sau mai mult. Softmax transformă o diferență de 20 într-un raport de e<sup>20</sup> ≈ 500 de milioane: rîndul devine one-hot. Un softmax saturat are gradientul aproape zero, așa că acel cap nu mai învață (Vaswani et al., 2017).</p>` +
                `<p>Împărțirea la √d readuce scorurile la o varianță în jur de 1, oricare ar fi dimensiunea d a capului. Duceți împărțitorul la 16 ca să vedeți opusul: rîndurile se aplatizează spre o distribuție uniformă.</p>`,
        },
        r2: {
            q: 'În medie pe 500 de tweet-uri de test, în ce straturi este atenția lui FinBERT cea mai concentrată, adică unde au rîndurile entropia cea mai mică?',
            opts: ['Stratul 1, la intrare', 'Straturile din mijloc, în jurul straturilor 6–8', 'Ultimul strat, 12, lîngă clasificator'],
            answer: 1,
            proof: (N, f) => `<p>Entropia normalizată a unui rînd (entropia împărțită la log n: 1 pentru o distribuție uniformă, 0 pentru un singur token) este ${f(N.entL1, 2)} în stratul 1, atinge minimul de ${f(N.entMin, 2)} în stratul ${N.entMinL} și urcă din nou la ${f(N.entL12, 2)} în stratul 12.</p>` +
                `<p>În stratul 1, fiecare token adună context din toată propoziția. În straturile din mijloc, multe capete se specializează și privesc un singur token: cuvîntul următor sau cel anterior, ori un token special (Clark et al., 2019). Ultimele straturi se împrăștie din nou, fiindcă amestecă informația pentru ieșire.</p>` +
                `<p>Concentrat nu înseamnă informativ. Runda 3 arată unde ajunge o mare parte din această atenție concentrată.</p>`,
        },
        r3: {
            q: '[SEP] este marcajul de la sfîrșitul fiecărei intrări; nu poartă nimic din sensul tweet-ului. În stratul 6, ce parte din toată atenția (toate capetele, toate tokenurile, 500 de tweet-uri de test) merge la el?',
            opts: ['Sub 10%', 'Aproximativ un sfert', 'Peste jumătate'],
            answer: 2,
            proof: (N, f) => `<p>[SEP] primește ${f(100 * N.sepL1, 1)}% din atenție în stratul 1 și ${f(100 * N.sepMax, 1)}% în stratul ${N.sepMaxL}, maximul; cele șapte tweet-uri din runda 2 dau ${f(100 * N.sep7max, 1)}% în acel strat. [CLS] atinge maximul mai devreme, ${f(100 * N.clsMax, 1)}% în stratul ${N.clsMaxL}.</p>` +
                `<p>Clark et al. (2019) au găsit același lucru la BERT și au arătat că modificarea acestor ponderi aproape nu schimbă ieșirea: un cap care nu are nimic de adăugat își parchează ponderea pe [SEP], o operație fără efect („no-op”). Softmax obligă fiecare rînd să aibă suma 1, deci ponderea trebuie să meargă undeva.</p>` +
                `<p>O pondere mare de atenție nu înseamnă deci importanță. Hărțile de atenție arată mecanismul, nu îl explică; cursul 11 revine la asta cu SHAP și LIME.</p>`,
        },
        r4: {
            q: 'Un transformer mic antrenat de la zero pe cele 8.588 de tweet-uri de antrenare, cu și fără codificarea poziției (fără ea, modelul vede tweet-ul ca pe un sac de tokenuri). Cît macro-F1 pierde fără poziții?',
            opts: ['Aproximativ 20 de puncte: se prăbușește', 'Aproximativ 5 puncte', 'Aproximativ 1 punct', 'Nimic: obține un scor mai mare'],
            answer: 2,
            proof: (N, f) => `<p>Cu poziții, ${f(N.tiny, 1)} (media a 3 semințe); fără, ${f(N.tiny_nopos, 1)}: un cîștig de ${f(N.posGain, 1)} puncte, cu un interval bootstrap de la ${f(N.boot.lo, 1)} la ${f(N.boot.hi, 1)} (1.000 de extrageri ale tweet-urilor de test). Media de embedding-uri, fără nicio atenție, ajunge la ${f(N.embbag, 1)}.</p>` +
                `<p>Titlurile financiare sînt scurte, iar sentimentul lor stă în cuvinte precum „downgraded”, „upgrades” sau „miss”, deci ordinea adaugă puțin. Ea contează pentru negație și pentru cine ce i-a făcut cui, situații rare aici.</p>` +
                `<p>Toate modelele antrenate de la zero rămîn mult sub TF-IDF cu regresie logistică (${f(N.tfidf, 1)}) și sub FinBERT ajustat (${f(N.finbert_ft, 1)}). Transformerului mic îi lipsesc datele, nu ordinea cuvintelor: runda 5.</p>`,
        },
        r5: {
            q: 'Doar 429 de tweet-uri etichetate, 5% din setul de antrenare. Care abordare dă cel mai mare macro-F1 pe cele 2.388 de tweet-uri de test?',
            opts: ['Transformerul mic antrenat de la zero', 'TF-IDF cu regresie logistică', 'FinBERT înghețat, cu o regresie logistică pe vectorul propoziției (o sondă liniară)', 'Toate trei la mai puțin de 2 puncte una de alta'],
            answer: 2,
            proof: (N, f) => `<p>Cu ${N.dsN0} de tweet-uri: FinBERT înghețat + logistic ${f(N.ds0.probe, 1)}, TF-IDF ${f(N.ds0.tfidf, 1)}, transformerul mic ${f(N.ds0.tiny, 1)}. Cu toate cele ${N.dsNfull.toLocaleString('ro')}: ${f(N.dsF.probe, 1)}, ${f(N.dsF.tfidf, 1)} și ${f(N.dsF.tiny, 1)}.</p>` +
                `<p>FinBERT a fost preantrenat pe foarte mult text, așa că vectorul lui de propoziție separă deja veștile bune de cele rele: o regresie logistică are nevoie de puține etichete ca să-l citească. TF-IDF și transformerul mic trebuie să învețe totul din tweet-urile etichetate și recuperează doar pe măsură ce etichetele cresc.</p>` +
                `<p>Cu toate etichetele, sonda înghețată este egalată de TF-IDF și doar ajustarea întregului FinBERT o ia înainte, la ${f(N.finbert_ft, 1)}. Preantrenarea valorează cel mai mult cînd etichetele sînt puține, situația obișnuită în finanțe.</p>`,
        },
    },
};
