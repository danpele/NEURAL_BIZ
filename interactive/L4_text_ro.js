// Readouts of the Lecture 4 interactive page, written by lectures/L4/build_interactive_L4.py.
const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 es: ['Culturi anuale', 'Pădure', 'Vegetație ierboasă', 'Autostradă', 'Zonă industrială', 'Pășune', 'Culturi permanente', 'Zonă rezidențială', 'Rîu', 'Mare/lac'], fm: ['Tricou', 'Pantaloni', 'Pulover', 'Rochie', 'Palton', 'Sandală', 'Cămașă', 'Pantof sport', 'Geantă', 'Ghete'],
 input: c => `Intrarea: ${c} (64 × 64, niveluri de gri)`, output: (k, n) => `Ieșirea filtrului: ${n} × ${n}`,
 kname: { sx: 'muchii verticale', sy: 'muchii orizontale', lap: 'contururi', blur: 'media 3 × 3', sharp: 'accentuare', id: 'identitate' },
 cv: (n, p, s, o) => `Mărimea ieșirii ⌊(${n} + 2·${p} − 3) / ${s}⌋ + 1 = <b>${deRO(o)}</b> pixeli pe latură; filtrul are 9 ponderi + 1 termen liber, folosite în toate cele ${deRO(o * o)} poziții.`,
 cvK: k => `Nucleul: ${k}`,
 layers: 'Straturi convoluționale', rfY: 'Cîmpul receptiv (pixeli)', plain: 'convoluții 3 × 3, pas 1', this: 'arhitectura aleasă',
 rfTile: 'Cîmpul receptiv pe o imagine de 64 × 64 (10 m pe pixel)',
 rf: (l, r, m, w) => `După ${deRO(l)} straturi, un neuron vede <b>${r} × ${deRO(r)} pixeli</b> (${m} m × ${m} m pe teren); ponderi pe pereche de canale în fiecare strat: ${w}.`,
 rfFull: 'Cîmpul receptiv acoperă întreaga imagine.',
 img: c => `Imaginea de test: ${c}`, map: (b, c, n) => `Blocul ${b === 'b1' ? 1 : 3}, canalul de rang ${c} (${n} × ${n})`,
 fmOut: (c, p, t) => `Predicția CNN: <b>${c}</b> (${p}); următoarele: ${t}.`, fmMap: (ch, mx) => `Canalul nr. ${ch} al stratului; activarea lui maximă pe această imagine este ${mx}.`,
 pred: 'Clasa prezisă', truth: 'Clasa reală', cmTitle: (m, a) => `${m}: matricea de confuzie, 10.000 de imagini de test (acuratețea ${a})`,
 models: { softmax: 'Regresie softmax', mlp: 'MLP', cnn: 'CNN' },
 cm: (c, r, n, t, tn) => `${c}: sensibilitatea <b>${r}</b> (${n} din 1.000 de imagini de test); eroarea cea mai frecventă: ${t} (${deRO(tn)} imagini).`,
 cm2: (p) => `Parametrii modelului: ${p}.`,
 cfNames: { cnn: 'CNN mică, fără augmentare', cnn_aug: 'CNN mică, cu augmentare', plain32: 'Simplă, 32 de straturi', res32: 'ResNet, 32 de straturi' },
 cfY: { train_loss: 'Pierderea la antrenare', train_acc: 'Acuratețea la antrenare (%)', valid_acc: 'Acuratețea pe validare (%)' }, epoch: 'Epoca',
 cf: (m, t, p, tr, va, l) => `${m}: acuratețea pe test <b>${t}</b>, ${deRO(p)} parametri; după 12 epoci, acuratețea la antrenare ${tr}, pe validare ${va}, pierderea la antrenare ${l}.`,
 tlNames: { scratch: 'CNN de la zero', probe: 'ResNet-18 înghețat + logistic', finetune: 'ResNet-18 ajustat fin' },
 tlHist: 'Tot setul de antrenare: acuratețea pe validare (%)', tlBudget: 'Acuratețea pe test (%) după numărul de etichete', epochs: 'Epoci', perClass: 'Imagini etichetate pe clasă (scară logaritmică)',
 acc: 'Acuratețea (%)', seeds: 'o sămînță',
 tl: (m, n, a, lo, hi) => `${m}: <b>${a}</b> cu ${deRO(n)} imagini pe clasă (2 semințe: ${lo} și ${hi}).`, tl0: (n, t) => `${deRO(n)} imagini etichetate pe clasă = ${deRO(t)} imagini în total (tot setul de antrenare: 16.200).`
};
