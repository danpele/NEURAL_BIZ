// Lecture 9 game "Judge the transaction by its neighbours": a real Elliptic test transaction and its two-step neighbourhood,
// illicit or licit, then the GraphSAGE network of the lecture, the same network without the graph (MLP) and the truth.
// Data: L9_game_data.js (lectures/L9/make_game_L9.py). Engine: game_common.js. Graph drawn as inline SVG.
(function () {
  'use strict';
  const G = window.NN_GAME_L9, I = G.info, { RO, pct, shuffle } = window.NNGame;
  const COL = { 1: '#C62828', 0: '#2E7D32', '-1': '#B5B5B5' };
  const L = RO ? {
    step: 'Pasul de timp', d1: 'Vecini direcți', d2: 'Vecini la doi pași', ill: 'ilicite', lic: 'licite', unk: 'necunoscute',
    leg: ['tranzacția de judecat', 'ilicită', 'licită', 'necunoscută'], more: 'alți',
    sage: p => `probabilitatea de tranzacție ilicită după GraphSAGE: ${pct(p)}`,
    mlp: p => `aceeași rețea fără graf (MLP): ${pct(p)}`, out: 'Eticheta din datele Elliptic', isI: 'ilicită', isL: 'licită',
    caught: (a, b, n) => `Tranzacții ilicite prinse: dumneavoastră ${a} din ${n}, GraphSAGE ${b} din ${n}.`,
    flag: (a, b, n) => `Tranzacții licite marcate greșit: dumneavoastră ${a} din ${n}, GraphSAGE ${b} din ${n}.`,
    aria: 'Graful vecinilor tranzacției'
  } : {
    step: 'Time step', d1: 'Direct neighbours', d2: 'Neighbours two steps away', ill: 'illicit', lic: 'licit', unk: 'unknown',
    leg: ['transaction to judge', 'illicit', 'licit', 'unknown'], more: 'more',
    sage: p => `probability of an illicit transaction according to GraphSAGE: ${pct(p)}`,
    mlp: p => `the same network without the graph (MLP): ${pct(p)}`, out: 'Label in the Elliptic data', isI: 'illicit', isL: 'licit',
    caught: (a, b, n) => `Illicit transactions caught: you ${a} of ${n}, GraphSAGE ${b} of ${n}.`,
    flag: (a, b, n) => `Licit transactions flagged by mistake: you ${a} of ${n}, GraphSAGE ${b} of ${n}.`,
    aria: 'Graph of the neighbours of the transaction'
  };

  function svg(c) {
    const W = 340, H = 330, cx = W / 2, cy = H / 2, R1 = 72, R2 = 128, k = c.r1.length;
    let e = '', n = '';
    const pt = (r, a) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    const sector = Math.min(2 * Math.PI / k * 0.8, 1.5);
    c.r1.forEach((u, i) => {
      const a = -Math.PI / 2 + 2 * Math.PI * i / k, [x, y] = pt(R1, a);
      const labs = [].concat(Array(u.r2[0]).fill(1), Array(u.r2[1]).fill(0), Array(u.r2[2]).fill(-1));
      const shown = labs.slice(0, I.max_r2), rest = labs.length - shown.length;
      shown.forEach((l, j) => {
        const b = a + (shown.length === 1 ? 0 : sector * (j / (shown.length - 1) - 0.5)), [x2, y2] = pt(R2, b);
        e += `<line x1="${x}" y1="${y}" x2="${x2}" y2="${y2}" stroke="#9AA5B1" stroke-width="1"/>`;
        n += `<circle cx="${x2}" cy="${y2}" r="5" fill="${COL[l]}" stroke="#fff" stroke-width="1"/>`;
      });
      if (rest > 0) {
        const [tx, ty] = pt(R2 + 22, a);
        n += `<text x="${tx}" y="${ty}" font-size="11" text-anchor="middle" dominant-baseline="middle" fill="#555">+${rest}</text>`;
      }
      e += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#5A6672" stroke-width="2"/>`;
      n += `<circle cx="${x}" cy="${y}" r="9" fill="${COL[u.l]}" stroke="#fff" stroke-width="1.5"/>`;
    });
    n += `<circle cx="${cx}" cy="${cy}" r="13" fill="#1F4E79" stroke="#fff" stroke-width="2"/>` +
      `<text x="${cx}" y="${cy + 1}" font-size="15" font-weight="700" text-anchor="middle" dominant-baseline="middle" fill="#fff">?</text>`;
    return `<svg class="g-fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="${L.aria}" style="max-width:420px;margin:0 auto">${e}${n}</svg>`;
  }
  const legend = () => '<p class="g-note" style="text-align:center">' +
    [['#1F4E79', L.leg[0]], [COL[1], L.leg[1]], [COL[0], L.leg[2]], [COL[-1], L.leg[3]]]
      .map(([cl, t]) => `<span style="white-space:nowrap;margin:0 0.4rem"><span style="display:inline-block;width:0.8em;height:0.8em;border-radius:50%;background:${cl};vertical-align:-0.05em"></span> ${t}</span>`).join(' ') + '</p>';
  function counts(c) {
    const d1 = [0, 0, 0], d2 = c.d2;
    c.r1.forEach(u => { d1[u.l === 1 ? 0 : u.l === 0 ? 1 : 2]++; });
    const f = a => `<b>${a[0]}</b> ${L.ill}, <b>${a[1]}</b> ${L.lic}, <b>${a[2]}</b> ${L.unk}`;
    return `<table class="g-tab"><tbody><tr><th scope="row">${L.step}</th><td>${c.step}</td></tr>` +
      `<tr><th scope="row">${L.d1}</th><td>${f(d1)}</td></tr><tr><th scope="row">${L.d2}</th><td>${f(d2)}</td></tr></tbody></table>`;
  }

  window.NNGame.choice({
    game: 'l9',
    cardTitle: { en: 'Transaction', ro: 'Tranzacția' },
    draw: () => shuffle(shuffle(G.pool.filter(c => c.ill)).slice(0, I.per_ill)
      .concat(shuffle(G.pool.filter(c => !c.ill)).slice(0, I.per_game - I.per_ill))),
    render: c => svg(c) + legend() + counts(c),
    choices: [{ label: { en: 'Licit', ro: 'Licită' }, cls: 'g-ok' }, { label: { en: 'Illicit', ro: 'Ilicită' }, cls: 'g-no' }],
    truth: c => c.ill,
    model: c => (c.sage >= 0.5 ? 1 : 0),
    modelName: { en: 'GraphSAGE, the graph network of the lecture', ro: 'GraphSAGE, rețeaua pe graf din curs' },
    reveal: c => `<li>${L.sage(c.sage)}; ${L.mlp(c.mlp)}</li><li>${L.out}: <b>${c.ill ? L.isI : L.isL}</b></li>`,
    summary: res => {
      const ni = res.filter(r => r.truth).length, nl = res.length - ni;
      const cy = res.filter(r => r.truth && r.you).length, cm = res.filter(r => r.truth && r.model).length;
      const fy = res.filter(r => !r.truth && r.you).length, fm = res.filter(r => !r.truth && r.model).length;
      return `<li>${L.caught(cy, cm, ni)}</li><li>${L.flag(fy, fm, nl)}</li>`;
    }
  });
})();
