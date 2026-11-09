// Lecture 14 game "You or the model? Market stress from chart images": real S&P 500 chart images of the test years.
// 7 regime cards (calm or stress; model: logistic regression on the CLIP image embedding) and 3 trend cards (higher or
// lower; model: Qwen2-VL-2B zero-shot). Data: L14_game_data.js (lectures/L14/make_game_L14.py). Engine: game_common.js.
(function () {
  'use strict';
  const G = window.NN_GAME_L14, I = G.info, { RO, pct, shuffle } = window.NNGame;
  const L = RO ? {
    alt: 'Graficul S&P 500 pe 120 de zile de tranzacționare: prețul și scăderea de la maxim',
    qr: 'La sfîrșitul ferestrei (marginea din dreapta), piața este calmă sau în criză?',
    qt: 'Indicele a încheiat fereastra mai sus sau mai jos decît a început-o?',
    calm: 'Calmă', stress: 'În criză', higher: 'Mai sus', lower: 'Mai jos',
    mr: 'Modelul pe imagine (regresie logistică pe CLIP)', mt: 'Modelul limbaj–viziune Qwen2-VL',
    pimg: p => `probabilitatea de criză estimată din imagine: ${pct(p)}`,
    pnum: p => `pentru comparație, modelul pe cifre (randamente, volatilitate, scădere de la maxim): ${pct(p)}`,
    pup: p => `probabilitatea răspunsului „mai sus” la Qwen2-VL: ${pct(p)}`,
    win: d => `Fereastra se încheie pe ${d}`, real: 'Situația reală', isS: 'criză', isC: 'calm',
    ret: r => `variația prețului pe fereastră: ${r > 0 ? '+' : ''}${pct(r, Math.abs(r) < 0.001 ? 2 : 1)}`,
    caught: (a, b, n) => `Ferestre de criză recunoscute: dumneavoastră ${a} din ${n}, modelul pe imagine ${b} din ${n}.`,
    alarm: (a, b, n) => `Ferestre calme semnalate greșit drept criză: dumneavoastră ${a} din ${n}, modelul ${b} din ${n}.`,
    trend: (a, b, n) => `Întrebările despre tendință: dumneavoastră ${a} din ${n}, Qwen2-VL ${b} din ${n}.`
  } : {
    alt: 'S&P 500 chart over 120 trading days: price and drawdown',
    qr: 'At the end of the window (right edge), is the market calm or in stress?',
    qt: 'Did the index end the window higher or lower than it started?',
    calm: 'Calm', stress: 'Stress', higher: 'Higher', lower: 'Lower',
    mr: 'The image model (logistic regression on CLIP)', mt: 'The vision-language model Qwen2-VL',
    pimg: p => `probability of stress estimated from the image: ${pct(p)}`,
    pnum: p => `for comparison, the model on the numbers (returns, volatility, drawdown): ${pct(p)}`,
    pup: p => `probability of the answer “higher” for Qwen2-VL: ${pct(p)}`,
    win: d => `The window ends on ${d}`, real: 'What was true', isS: 'stress', isC: 'calm',
    ret: r => `price change over the window: ${r > 0 ? '+' : ''}${pct(r, Math.abs(r) < 0.001 ? 2 : 1)}`,
    caught: (a, b, n) => `Stress windows recognised: you ${a} of ${n}, the image model ${b} of ${n}.`,
    alarm: (a, b, n) => `Calm windows wrongly flagged as stress: you ${a} of ${n}, the model ${b} of ${n}.`,
    trend: (a, b, n) => `Trend questions: you ${a} of ${n}, Qwen2-VL ${b} of ${n}.`
  };
  const day = s => new Date(s + 'T12:00:00').toLocaleDateString(RO ? 'ro-RO' : 'en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
  let cur = null;   // card on screen: the engine reads modelName after render, so the name can follow the card
  const NAME = { get en() { return cur && cur.kind === 't' ? L.mt : L.mr; }, get ro() { return cur && cur.kind === 't' ? L.mt : L.mr; } };
  window.NNGame.choice({
    game: 'l14',
    cardTitle: { en: 'Chart', ro: 'Graficul' },
    draw: () => {   // 3 stress and 4 calm regime cards in random order, then 3 trend cards
      const pick = (f, n) => shuffle(G.pool.filter(f)).slice(0, n);
      return shuffle(pick(c => c.kind === 'r' && c.stress, I.stress_per_game).concat(pick(c => c.kind === 'r' && !c.stress, I.calm_per_game)))
        .concat(pick(c => c.kind === 't', I.trend_per_game));
    },
    render: c => {
      cur = c;
      return `<p class="g-quote">${c.kind === 't' ? L.qt : L.qr}</p><img class="g-fig" src="game_img/L14/${c.img}" width="224" height="224" alt="${L.alt}" ` +
        'style="max-width:448px;margin:0 auto">';
    },
    choices: c => (c.kind === 't'
      ? [{ label: L.higher, cls: 'g-ok' }, { label: L.lower, cls: 'g-no' }]
      : [{ label: L.calm, cls: 'g-ok' }, { label: L.stress, cls: 'g-no' }]),
    truth: c => (c.kind === 't' ? (c.up ? 0 : 1) : c.stress),
    model: c => (c.kind === 't' ? (c.p_up > 0.5 ? 0 : 1) : (c.p_img > 0.5 ? 1 : 0)),
    modelName: NAME,
    reveal: c => (c.kind === 't'
      ? `<li>${L.pup(c.p_up)}</li><li>${L.real}: <b>${L.ret(c.ret)}</b></li>`
      : `<li>${L.pimg(c.p_img)}</li><li>${L.pnum(c.p_num)}</li><li>${L.real}: <b>${c.stress ? L.isS : L.isC}</b></li>`) +
      `<li>${L.win(day(c.end))}</li>`,
    summary: (res, cards) => {
      const sel = f => res.filter((r, i) => f(cards[i]));
      const s = sel(c => c.kind === 'r' && c.stress), q = sel(c => c.kind === 'r' && !c.stress), t = sel(c => c.kind === 't');
      const n1 = (a, who) => a.filter(r => r[who] === 1).length, ok = (a, who) => a.filter(r => r[who] === r.truth).length;
      return `<li>${L.caught(n1(s, 'you'), n1(s, 'model'), s.length)}</li><li>${L.alarm(n1(q, 'you'), n1(q, 'model'), q.length)}</li>` +
        `<li>${L.trend(ok(t, 'you'), ok(t, 'model'), t.length)}</li>`;
    }
  });
})();
