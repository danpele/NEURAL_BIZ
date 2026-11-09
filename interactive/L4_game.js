// Lecture 4 game "You or the CNN?": real EuroSAT test tiles, pick the land-use class among four, then the fine-tuned
// ResNet-18's prediction and the true class. Data: L4_game_data.js (lectures/L4/make_game_L4.py). Engine: game_common.js.
(function () {
  'use strict';
  const G = window.NN_GAME_L4, I = G.info, { RO, pct, shuffle } = window.NNGame;
  const NAMES = RO
    ? ['Culturi anuale', 'Pădure', 'Vegetație ierboasă', 'Autostradă', 'Zonă industrială', 'Pășune', 'Culturi permanente', 'Zonă rezidențială', 'Rîu', 'Mare/lac']
    : ['Annual crop', 'Forest', 'Herbaceous vegetation', 'Highway', 'Industrial', 'Pasture', 'Permanent crop', 'Residential', 'River', 'Sea/lake'];
  const L = RO ? {
    alt: 'Imagine satelitară Sentinel-2 de 64 × 64 pixeli', q: 'Ce fel de teren arată imaginea?',
    conf: p => `Încrederea rețelei în răspunsul ei (probabilitatea softmax): ${pct(p)}`, truth: 'Clasa corectă',
    hard: (a, n) => `Pe imaginile greșite de rețea: dumneavoastră ați avut dreptate în ${a} din ${n}.`,
    easy: (a, b, n) => `Pe celelalte imagini: dumneavoastră ${a} din ${n}, rețeaua ${b} din ${n}.`
  } : {
    alt: 'Sentinel-2 satellite image, 64 × 64 pixels', q: 'What kind of land does the image show?',
    conf: p => `The network's confidence in its answer (softmax probability): ${pct(p)}`, truth: 'True class',
    hard: (a, n) => `On the images the network got wrong: you were right in ${a} of ${n}.`,
    easy: (a, b, n) => `On the other images: you ${a} of ${n}, the network ${b} of ${n}.`
  };
  // four options: the true class, the network's class if different, other classes at random; shuffled
  function withOptions(c) {
    const o = [c.y];
    if (c.pred !== c.y) o.push(c.pred);
    shuffle([0, 1, 2, 3, 4, 5, 6, 7, 8, 9].filter(k => !o.includes(k))).slice(0, 4 - o.length).forEach(k => o.push(k));
    return Object.assign({}, c, { opts: shuffle(o) });
  }
  window.NNGame.choice({
    game: 'l4',
    cardTitle: { en: 'Image', ro: 'Imaginea' },
    draw: () => {   // 8 tiles the network classifies correctly and 2 it gets wrong, in random order
      const nw = I.wrong_per_game;
      return shuffle(shuffle(G.pool.filter(c => !c.ok)).slice(0, nw).concat(shuffle(G.pool.filter(c => c.ok)).slice(0, I.per_game - nw))).map(withOptions);
    },
    render: c => `<p>${L.q}</p><img class="g-fig" src="game_img/L4/${c.img}" width="64" height="64" alt="${L.alt}" ` +
      'style="image-rendering:pixelated;max-width:256px;margin:0 auto">',
    choices: c => c.opts.map(k => ({ label: NAMES[k], cls: '' })),
    truth: c => c.opts.indexOf(c.y),
    model: c => c.opts.indexOf(c.pred),
    modelName: { en: 'The network of the lecture (fine-tuned ResNet-18)', ro: 'Rețeaua din curs (ResNet-18 ajustat fin)' },
    reveal: c => `<li>${L.conf(c.conf)}</li><li>${L.truth}: <b>${NAMES[c.y]}</b></li>`,
    summary: (res, cards) => {
      const h = res.filter((r, i) => !cards[i].ok), e = res.filter((r, i) => cards[i].ok);
      const ok = r => r.you === r.truth;
      return `<li>${L.hard(h.filter(ok).length, h.length)}</li><li>${L.easy(e.filter(ok).length, e.filter(r => r.model === r.truth).length, e.length)}</li>`;
    }
  });
})();
