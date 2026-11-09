// Lecture 12 game "You or the language model?": real Financial PhraseBank sentences at four levels of annotator agreement,
// negative / neutral / positive, against the zero-shot prompt classifier of the lecture (Qwen2.5-3B-Instruct).
// Data: L12_game_data.js (lectures/L12/make_game_L12.py). Engine: game_common.js.
(function () {
  'use strict';
  const G = window.NN_GAME_L12, I = G.info, { RO, pct, shuffle } = window.NNGame;
  const L = RO ? {
    note: 'Propoziție dintr-o știre financiară (în limba engleză, ca în setul de date)',
    lab: ['negativă', 'neutră', 'pozitivă'],
    prob: p => `Probabilitatea dată de model propriei etichete: <b>${pct(p)}</b>`,
    fs: 'Același model, cu 12 exemple în prompt (few-shot)',
    ann: 'Eticheta majorității adnotatorilor',
    agr: { 100: 'toți adnotatorii au ales-o', 75: 'au ales-o 75–99 % dintre adnotatori', 66: 'au ales-o 66–74 % dintre adnotatori', 50: 'au ales-o 50–65 % dintre adnotatori' },
    una: (y, m, n) => `Propoziții asupra cărora toți adnotatorii au fost de acord: dumneavoastră ${y} din ${n}, modelul ${m} din ${n}.`,
    spl: (y, m, n) => `Propoziții asupra cărora adnotatorii au fost împărțiți: dumneavoastră ${y} din ${n}, modelul ${m} din ${n}.`,
    conf: (a, b) => `Probabilitatea medie dată de model propriei etichete: ${a} cînd a avut dreptate` + (b ? `, ${b} cînd a greșit.` : '; nu a greșit în acest joc.')
  } : {
    note: 'Sentence from a financial news item',
    lab: ['negative', 'neutral', 'positive'],
    prob: p => `Probability the model gave its own label: <b>${pct(p)}</b>`,
    fs: 'The same model with 12 examples in the prompt (few-shot)',
    ann: 'Label of the majority of the annotators',
    agr: { 100: 'all annotators chose it', 75: '75–99% of the annotators chose it', 66: '66–74% of the annotators chose it', 50: '50–65% of the annotators chose it' },
    una: (y, m, n) => `Sentences on which all annotators agreed: you ${y} of ${n}, the model ${m} of ${n}.`,
    spl: (y, m, n) => `Sentences that split the annotators: you ${y} of ${n}, the model ${m} of ${n}.`,
    conf: (a, b) => `Average probability the model gave its own label: ${a} when it was right` + (b ? `, ${b} when it was wrong.` : '; it made no error in this game.')
  };
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  window.NNGame.choice({
    game: 'l12',
    cardTitle: { en: 'Sentence', ro: 'Propoziția' },
    draw: () => shuffle([].concat(...Object.keys(I.levels).map(a => shuffle(G.pool.filter(c => c.a === +a)).slice(0, I.levels[a])))),
    render: c => `<p class="g-of">${L.note}</p><p class="g-quote" lang="en">${esc(c.s)}</p>`,
    choices: [{ label: { en: 'Negative', ro: 'Negativă' }, cls: 'g-no' }, { label: { en: 'Neutral', ro: 'Neutră' }, cls: '' },
              { label: { en: 'Positive', ro: 'Pozitivă' }, cls: 'g-ok' }],
    truth: c => c.y,
    model: c => c.m,
    modelName: { en: 'The language model (zero-shot)', ro: 'Modelul de limbaj (zero-shot)' },
    reveal: c => `<li>${L.prob(c.p)}</li><li>${L.fs}: <b>${L.lab[c.f]}</b></li>` +
      `<li>${L.ann}: <b>${L.lab[c.y]}</b> (${L.agr[c.a]})</li>`,
    summary: (res, cards) => {
      const part = f => { const ix = cards.map((c, i) => i).filter(i => f(cards[i]));
        return [ix.filter(i => res[i].you === res[i].truth).length, ix.filter(i => res[i].model === res[i].truth).length, ix.length]; };
      const ok = cards.filter((c, i) => res[i].model === res[i].truth).map(c => c.p), ko = cards.filter((c, i) => res[i].model !== res[i].truth).map(c => c.p);
      return `<li>${L.una(...part(c => c.a === 100))}</li><li>${L.spl(...part(c => c.a < 100))}</li>` +
        `<li>${L.conf(ok.length ? pct(mean(ok)) : '–', ko.length ? pct(mean(ko)) : '')}</li>`;
    }
  });
})();
