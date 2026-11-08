// Readouts of the Lecture 14 interactive page, written by lectures/L14/build_interactive_L14.py.
window.LK_TEXT = {
 cls: ["Culturi anuale", "Pădure", "Vegetație ierboasă", "Autostradă", "Zonă industrială", "Pășune", "Culturi permanente", "Zonă rezidențială", "Rîu", "Mare sau lac"],
 px: 'Pixeli (intrarea de 224 × 224)', img: p => `Imaginea de intrare tăiată în petice de ${p} × ${p}`, att: o => `Atenția tokenului de clasă (propagare, ViT-B/16), opacitate ${o}%`,
 vp: (n, g, p, rel) => `Petic de ${p} × ${p} pixeli: grila ${g} × ${g} = ${window.NNI.cnt(n, 'token-petic', 'tokeni-petic')} + 1 token de clasă = ${window.NNI.cnt(n + 1, 'token', 'tokeni')}; atenția compară ${window.NNI.cnt((n + 1) ** 2, 'pereche', 'perechi')} de tokeni, de ${rel} ori costul cu petice de 16 × 16.`,
 vp2: (c, top, share) => `${c}: peticele cele mai urmărite (primele 10% din hartă) primesc ${share} din atenție; maximul este pe rîndul ${top[0]}, coloana ${top[1]} a grilei de 14 × 14.`,
 prob: 'Probabilitatea (%)', simTitle: (s) => `Softmax de ${s} × similaritatea cosinus`,
 zs: (t, p, pr, ok) => `Clasa reală: ${t}. Prezisă: ${p}, cu probabilitatea ${pr} (${ok ? 'corect' : 'greșit'}).`,
 zs2: (cmax, cmin, s, h) => `Similaritățile cosinus sînt între ${cmin} și ${cmax}; la scala ${s}, entropia probabilităților este ${h} biți (maximul este 3,32 pentru 10 clase egale).`,
 pred: 'Clasa prezisă (zero-shot)', truth: 'Clasa reală',
 cm: (m, acc, best, bestv, worst, worstv, t, p, n, nt) => `${m}: acuratețea ${acc} pe 5.400 de imagini de test. Cea mai bună clasă: ${best} (${bestv}), cea mai slabă: ${worst} (${worstv}). Cea mai mare confuzie: ${n} din ${nt} imagini din clasa ${t} sînt prezise ca ${p}.`,
 k: 'Imagini returnate, k', prec: 'Precizia primelor k (%)', avg: 'Media celor 10 căutări', query: q => `Căutarea: „a centered satellite photo of ${q}.” (media a 5 șabloane)`,
 ts: (q, k, p, n, a) => `Căutarea ${q}: ${n} din primele ${window.NNI.cnt(k, 'imagine', 'imagini')} aparțin clasei, precizia ${p}; media celor 10 căutări ${a}.`,
 ts2: 'Primele 8 imagini (chenar verde: clasa corectă, roșu: altă clasă).',
 shots: 'Imagini etichetate pe clasă (scară logaritmică)', acc: 'Acuratețea pe test (%)', probe: 'CLIP + regresie logistică (5 valori aleatoare)', cnn: 'CNN de la zero (3 valori aleatoare)',
 zsl: 'CLIP zero-shot', full: 'CNN pe toate cele 21.600 de imagini de antrenare', fullp: 'Regresia pe CLIP cu toate cele 21.600 de imagini',
 fs: (k, n, p, pl, ph, c) => `${window.NNI.cnt(k, 'imagine', 'imagini')} pe clasă (${window.NNI.cnt(n, 'imagine etichetată', 'imagini etichetate')}): regresia pe CLIP ${p} (interval după valoarea aleatoare ${pl}–${ph})${c}.`,
 fsc: (c, cl, ch) => `; CNN ${c} (interval ${cl}–${ch})`, fsn: '; CNN nu a fost antrenată la această mărime',
 fs2: (z, f, fp) => `Zero-shot (fără etichete) ${z}; cu toate etichetele de antrenare: CNN ${f}, regresia pe CLIP ${fp}.`,
 fpr: 'Rata alarmelor false (ponderea ferestrelor calme semnalate)', tpr: 'Rata de detectare (ponderea ferestrelor de stres semnalate)', chance: 'Ghicire aleatoare',
 score: 'Scorul de stres', stress: 'Fereastră de stres (reală)', thr: 'Pragul de alarmă', date: 'Sfîrșitul ferestrei de 120 de zile',
 cr: (m, auc, a, tp, nt, fp, nf, pr) => `${m}: AUC ${auc}. Dacă semnalăm ${a} din ferestrele de test, prindem ${tp} din ${window.NNI.cnt(nt, 'fereastră', 'ferestre')} de stres, cu ${window.NNI.cnt(fp, 'alarmă falsă', 'alarme false')} din ${window.NNI.cnt(nf, 'fereastră calmă', 'ferestre calme')}; precizia alarmelor ${pr}.`,
 cr2: (n, s) => `${window.NNI.cnt(n, 'fereastră', 'ferestre')} de test (7 ianuarie 2019 – 30 octombrie 2026), dintre care ${s} de stres (volatilitatea pe 20 de zile peste percentila 90 glisantă a ultimilor 3 ani).`
};
