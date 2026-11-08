// Readouts of the Lecture 5 interactive page, written by lectures/L5/build_interactive_L5.py.
const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 mk: { SP500: 'S&P 500', TLV: 'Banca Transilvania (TLV)' }, ser: { r: 'randamente rₜ', a: 'randamente absolute |rₜ|', s: 'randamente la pătrat rₜ²' },
 lag: 'Lagul (zile de tranzacționare)', acf: 'Autocorelația', band: 'banda ±1,96/√T (zgomot alb)',
 acTitle: (m, s, a, b) => `${m}: autocorelația pentru ${s} zilnice, ${a} - ${b}`,
 ac: (r1, r10, n, L, b, T) => `ρ(1) = <b>${r1}</b>, ρ(10) = ${r10}; ${n} din ${deRO(L)} laguri în afara benzii ±${b} (${deRO(T)} zile de tranzacționare).`,
 mod: { obs: 'Observat (amplitudine Garman–Klass)', naive: 'Naiv (ziua de ieri)', har: 'HAR', garch: 'GARCH(1,1)', rnn: 'RNN (media a 5 semințe)', lstm: 'LSTM (media a 5 semințe)', gru: 'GRU (media a 5 semințe)' },
 vol: 'Volatilitatea anualizată (%)', vfTitle: (a, b) => `S&P 500, prognoze ale volatilității pentru ziua următoare, ${a} - ${b}`,
 vf: (m, e, r) => `${m}: MSE a log-varianței <b>${e}</b> (locul ${r})`, vf0: (n, a, b) => `${deRO(n)} zile de tranzacționare, de la ${a} la ${b}; toată perioada de test: naiv 0,966, GARCH 0,786, HAR 0,706, LSTM 0,625.`,
 rtMod: { mean: 'Media de antrenare', ar: 'AR(1)', lstm: 'LSTM' }, cum: 'Σ r² − Σ (r − prognoza)², cumulat  (%²)',
 rtTitle: m => `${m}: cîștigul față de prognoza zero, cumulat pe zilele de test (creștere = mai bine)`,
 rt: (m, r2, h) => `${m}: R² în afara eșantionului <b>${r2}</b>${h ? `; semnul corect în ${h} din zile` : ''}.`, rt0: (n, up) => `${deRO(n)} zile de test; ${up} dintre ele sînt zile de creștere.`,
 lmod: { obs: 'Consumul observat', snaive: 'Naiv sezonier', ridge: 'Ridge ARX', lstm: 'LSTM (media a 3 semințe)', gru: 'GRU (media a 3 semințe)' },
 mw: 'Consumul (MW)', hourUTC: 'Ora din ziua d+1 (UTC)', ldTitle: (a, b) => a === b ? `Prognoze pentru ziua următoare ale consumului din România, ${a}` : `Prognoze pentru ziua următoare ale consumului din România, ${a} - ${b}`,
 ld: (m, w, t) => `${m}: MAPE <b>${w}</b> pe zilele afișate (toată perioada de test ${t}).`,
 mae: 'MAE (MW)', leTitle: n => `Eroarea absolută medie pe ore, ${deRO(n)} zile de test`,
 le: (m, a, p, h, hm) => `${m}: MAE <b>${a} MW</b>, MAPE ${p}; ora cea mai slabă ${h}:00 UTC (${hm} MW).`
};
