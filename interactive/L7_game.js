// Lecture 7 game "You or FinBERT?": real test tweets on which the two simple models agree and fine-tuned FinBERT
// disagrees; bearish / neutral / bullish, then FinBERT's label, the simple models' label and the true label.
// Data: L7_game_data.js (lectures/L7/make_game_L7.py). Engine: game_common.js.
(function () {
  'use strict';
  const G = window.NN_GAME_L7, I = G.info, { RO, shuffle } = window.NNGame;
  const L = RO ? {
    note: 'Tweet financiar (în limba engleză, ca în setul de date)',
    lab: ['bearish (negativ)', 'neutru', 'bullish (pozitiv)'],
    simple: 'Modelele simple (TF-IDF cu regresie logistică și embedding bag)', truth: 'Eticheta corectă',
    sum: (a, n) => `Pe aceste ${n} tweet-uri, modelele simple au avut dreptate în ${a} cazuri.`
  } : {
    note: 'Finance tweet',
    lab: ['bearish', 'neutral', 'bullish'],
    simple: 'The simple models (TF-IDF with logistic regression and the embedding bag)', truth: 'The true label',
    sum: (a, n) => `On these ${n} tweets the simple models were right in ${a}.`
  };
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  window.NNGame.choice({
    game: 'l7',
    cardTitle: { en: 'Tweet', ro: 'Tweet-ul' },
    draw: () => shuffle(G.pool).slice(0, I.per_game),
    render: c => `<p class="g-of">${L.note}</p><p class="g-quote" lang="en">${esc(c.t)}</p>`,
    choices: [{ label: { en: 'Bearish', ro: 'Bearish (negativ)' }, cls: 'g-no' }, { label: { en: 'Neutral', ro: 'Neutru' }, cls: '' },
              { label: { en: 'Bullish', ro: 'Bullish (pozitiv)' }, cls: 'g-ok' }],
    truth: c => c.y,
    model: c => c.m,
    modelName: { en: 'FinBERT (fine-tuned)', ro: 'FinBERT (ajustat)' },
    reveal: c => `<li>${L.simple}: <b>${L.lab[c.s]}</b> <span class="${c.s === c.y ? 'g-right' : 'g-wrong'}">${c.s === c.y ? '✓' : '✗'}</span></li>` +
      `<li>${L.truth}: <b>${L.lab[c.y]}</b></li>`,
    summary: (res, cards) => `<li>${L.sum(cards.filter(c => c.s === c.y).length, cards.length)}</li>`
  });
})();
