// Lecture 6 game "You or the network?": real Financial PhraseBank test sentences, negative / neutral / positive,
// then the label of the embedding-bag network of the lecture and the label of the annotators.
// Data: L6_game_data.js (lectures/L6/make_game_L6.py). Engine: game_common.js.
(function () {
  'use strict';
  const G = window.NN_GAME_L6, I = G.info, { RO, shuffle } = window.NNGame;
  const L = RO ? {
    note: 'Propoziție dintr-o știre financiară (în limba engleză, ca în setul de date)',
    ann: 'Eticheta adnotatorilor', lab: ['negativă', 'neutră', 'pozitivă'],
    toNeu: (a, b) => `Propoziții pozitive sau negative citite ca neutre: dumneavoastră ${a}, rețeaua ${b}.`,
    swap: (a, b) => `Pozitiv confundat cu negativ (sau invers): dumneavoastră ${a}, rețeaua ${b}.`
  } : {
    note: 'Sentence from a financial news item',
    ann: 'Label of the annotators', lab: ['negative', 'neutral', 'positive'],
    toNeu: (a, b) => `Positive or negative sentences read as neutral: you ${a}, the network ${b}.`,
    swap: (a, b) => `Positive and negative swapped: you ${a}, the network ${b}.`
  };
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  window.NNGame.choice({
    game: 'l6',
    cardTitle: { en: 'Sentence', ro: 'Propoziția' },
    draw: () => {   // 3 negative, 3 positive, 4 neutral cards per game, drawn from the pool
      const by = y => shuffle(G.pool.filter(c => c.y === y));
      return shuffle(by(0).slice(0, I.game.neg).concat(by(2).slice(0, I.game.pos), by(1).slice(0, I.game.neu)));
    },
    render: c => `<p class="g-of">${L.note}</p><p class="g-quote" lang="en">${esc(c.s)}</p>`,
    choices: [{ label: { en: 'Negative', ro: 'Negativă' }, cls: 'g-no' }, { label: { en: 'Neutral', ro: 'Neutră' }, cls: '' },
              { label: { en: 'Positive', ro: 'Pozitivă' }, cls: 'g-ok' }],
    truth: c => c.y,
    model: c => c.m,
    modelName: { en: 'The network of the lecture', ro: 'Rețeaua din curs' },
    reveal: c => `<li>${L.ann}: <b>${L.lab[c.y]}</b></li>`,
    summary: res => {
      const neu = k => res.filter(r => r.truth !== 1 && r[k] === 1).length;
      const sw = k => res.filter(r => r.truth !== 1 && r[k] !== 1 && r[k] !== r.truth).length;
      return `<li>${L.toNeu(neu('you'), neu('model'))}</li><li>${L.swap(sw('you'), sw('model'))}</li>`;
    }
  });
})();
