// Lecture 2 game "Fast or slow training?": real training curves of the lecture; the student says which learning rate,
// batch size or optimiser produced each curve. No model answers (the engine scores the student only).
// Data: L2_game_data.js (lectures/L2/make_game_L2.py). Engine: game_common.js. Charts: inline SVG (svgChart below).
(function () {
  'use strict';
  const G = window.NN_GAME_L2, I = G.info, { RO, nf, shuffle } = window.NNGame;
  const n3 = x => nf(x, 3), lrf = x => (RO ? String(x).replace('.', ',') : String(x));
  const L = RO ? {
    card: 'Graficul', ep: 'Epoca', off: 'eroarea crește nelimitat', lin: 'regresia liniară',
    qlr: 'SGD simplu, 30 de epoci. Ce rată de învățare η a produs această curbă?', ylr: 'MSE pe validare',
    qbs: 'Adam cu η = 0,001, 40 de epoci. Ce mărime a lotului a produs această curbă?', ybs: 'MSE pe validare',
    qop: 'Detectarea fraudelor cu carduri, 30 de epoci. Ce optimizator a produs această curbă?', yop: 'PR-AUC pe validare (mai mare = mai bine)',
    bs: b => `Loturi de ${nf(b)}`, op: { sgd: 'SGD simplu', momentum: 'SGD cu moment', adam: 'Adam' },
    rlr: (c, last) => last === null
      ? `η = ${lrf(c.lr)}: după prima epocă eroarea a crescut nelimitat; antrenarea a divergat.`
      : `η = ${lrf(c.lr)}: MSE pe validare după 30 de epoci ${n3(last)} (regresia liniară: ${n3(I.mse_lin_val)}).`,
    rbs: c => `Loturi de ${nf(c.bs)}: ${nf(c.spe)} pași de gradient pe epocă, ${nf(c.sec, 1)} s pentru 40 de epoci; MSE după 40 de epoci ${n3(c.val[c.val.length - 1])} pe validare și ${n3(c.test)} pe test.`,
    rop: c => `${({ sgd: 'SGD simplu', momentum: 'SGD cu moment', adam: 'Adam' })[c.opt]} (η = ${lrf(c.lr)}, inițializarea ${c.seed + 1} din 5): PR-AUC pe validare ${n3(c.ap[0])} după prima epocă, ${n3(c.ap[c.ap.length - 1])} după 30; PR-AUC pe test ${n3(c.test)}.`,
    sum: (a, b, c) => `<li>Rata de învățare: ${a[0]} din ${a[1]} · mărimea lotului: ${b[0]} din ${b[1]} · optimizatorul: ${c[0]} din ${c[1]}.</li>`
  } : {
    card: 'Chart', ep: 'Epoch', off: 'error grows without bound', lin: 'linear regression',
    qlr: 'Plain SGD, 30 epochs. Which learning rate η produced this curve?', ylr: 'Validation MSE',
    qbs: 'Adam with η = 0.001, 40 epochs. Which batch size produced this curve?', ybs: 'Validation MSE',
    qop: 'Card-fraud detection, 30 epochs. Which optimiser produced this curve?', yop: 'PR-AUC on validation (higher = better)',
    bs: b => `Batches of ${nf(b)}`, op: { sgd: 'Plain SGD', momentum: 'SGD with momentum', adam: 'Adam' },
    rlr: (c, last) => last === null
      ? `η = ${lrf(c.lr)}: after the first epoch the error grew without bound; training diverged.`
      : `η = ${lrf(c.lr)}: validation MSE after 30 epochs ${n3(last)} (linear regression: ${n3(I.mse_lin_val)}).`,
    rbs: c => `Batches of ${nf(c.bs)}: ${nf(c.spe)} gradient steps per epoch, ${nf(c.sec, 1)} s for 40 epochs; MSE after 40 epochs ${n3(c.val[c.val.length - 1])} on validation and ${n3(c.test)} on test.`,
    rop: c => `${({ sgd: 'Plain SGD', momentum: 'SGD with momentum', adam: 'Adam' })[c.opt]} (η = ${lrf(c.lr)}, seed ${c.seed + 1} of 5): validation PR-AUC ${n3(c.ap[0])} after the first epoch, ${n3(c.ap[c.ap.length - 1])} after 30; test PR-AUC ${n3(c.test)}.`,
    sum: (a, b, c) => `<li>Learning rate: ${a[0]} of ${a[1]} · batch size: ${b[0]} of ${b[1]} · optimiser: ${c[0]} of ${c[1]}.</li>`
  };

  // Small line chart: series [{ y: [...], color }]; values above ymax are clipped; null = diverged (line goes off the top).
  function svgChart(o) {
    const W = 340, H = 196, ml = 40, mr = 10, mt = 22, mb = 30, pw = W - ml - mr, ph = H - mt - mb;
    const n = Math.max(...o.series.map(s => s.y.length));
    const X = e => ml + (n === 1 ? 0 : (e - 1) / (n - 1)) * pw, Y = v => mt + ph - (v - o.ymin) / (o.ymax - o.ymin) * ph;
    const id = 'c' + Math.random().toString(36).slice(2, 8);
    let s = `<svg class="g-fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.ylab}" style="font:11px Inter,Arial,sans-serif">` +
      `<defs><clipPath id="${id}"><rect x="${ml}" y="${mt}" width="${pw}" height="${ph}"/></clipPath></defs>`;
    o.ticks.forEach(t => {
      s += `<line x1="${ml}" x2="${ml + pw}" y1="${Y(t)}" y2="${Y(t)}" stroke="#e6e6e6"/>` +
        `<text x="${ml - 5}" y="${Y(t) + 4}" text-anchor="end" fill="#555">${nf(t, o.td || 1)}</text>`;
    });
    o.xticks.forEach(e => { s += `<text x="${X(e)}" y="${mt + ph + 14}" text-anchor="middle" fill="#555">${e}</text>`; });
    s += `<text x="${ml + pw / 2}" y="${H - 3}" text-anchor="middle" fill="#333">${L.ep}</text>` +
      `<text x="${ml}" y="13" fill="#333">${o.ylab}</text>`;
    if (o.ref !== undefined) {
      s += `<line x1="${ml}" x2="${ml + pw}" y1="${Y(o.ref)}" y2="${Y(o.ref)}" stroke="#1baf7a" stroke-dasharray="5 3"/>` +
        `<text x="${ml + pw - 2}" y="${Y(o.ref) - 4}" text-anchor="end" fill="#138a60">${L.lin}</text>`;
    }
    let off = false;
    o.series.forEach(se => {
      let d = '', last = null;
      se.y.forEach((v, i) => {
        if (v === null) { if (last && !off) { d += `L${X(last[0])},${mt - 40}`; off = true; } return; }
        d += (d ? 'L' : 'M') + X(i + 1).toFixed(1) + ',' + Y(v).toFixed(1); last = [i + 1, v];
      });
      s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2" stroke-linejoin="round" clip-path="url(#${id})"/>`;
    });
    if (off) s += `<text x="${ml + 30}" y="${mt + 14}" fill="#e34948">↑ ${L.off}</text>`;
    s += `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="none" stroke="#bbb"/></svg>`;
    return s;
  }

  const BLUE = '#2a78d6', ORANGE = '#eb6834';
  const kinds = {
    lr: { q: L.qlr, chart: c => svgChart({ series: [{ y: c.val, color: BLUE }], ymin: 0, ymax: 1.6, ticks: [0, 0.5, 1, 1.5], xticks: [1, 10, 20, 30], ylab: L.ylr, ref: I.mse_lin_val }),
      choices: () => G.lr.map(c => ({ label: `η = ${lrf(c.lr)}`, cls: '' })), truth: c => G.lr.findIndex(x => x.lr === c.lr),
      reveal: c => L.rlr(c, c.val[c.val.length - 1]) },
    bs: { q: L.qbs, chart: c => svgChart({ series: [{ y: c.val, color: BLUE }], ymin: 0, ymax: 1.2, ticks: [0, 0.4, 0.8, 1.2], xticks: [1, 10, 20, 30, 40], ylab: L.ybs, ref: I.mse_lin_val }),
      choices: () => G.bs.map(c => ({ label: L.bs(c.bs), cls: '' })), truth: c => G.bs.findIndex(x => x.bs === c.bs), reveal: c => L.rbs(c) },
    op: { q: L.qop, chart: c => svgChart({ series: [{ y: c.ap, color: ORANGE }], ymin: 0, ymax: 1, ticks: [0, 0.25, 0.5, 0.75, 1], td: 2, xticks: [1, 10, 20, 30], ylab: L.yop }),
      choices: () => ['sgd', 'momentum', 'adam'].map(o => ({ label: L.op[o], cls: '' })), truth: c => ['sgd', 'momentum', 'adam'].indexOf(c.opt), reveal: c => L.rop(c) }
  };
  const K = c => kinds[c.kind];

  window.NNGame.choice({
    game: 'l2',
    cardTitle: { en: L.card, ro: L.card },
    draw: () => [].concat(
      shuffle(G.lr).slice(0, I.per_kind).map(c => Object.assign({ kind: 'lr' }, c)),
      shuffle(G.bs).slice(0, I.per_kind).map(c => Object.assign({ kind: 'bs' }, c)),
      shuffle(G.opt).slice(0, I.per_kind).map(c => Object.assign({ kind: 'op' }, c))),
    render: c => `<p><b>${K(c).q}</b></p>` + K(c).chart(c),
    choices: c => K(c).choices(),
    truth: c => K(c).truth(c),
    model: () => null,
    reveal: c => `<li>${K(c).reveal(c)}</li>`,
    summary: (res, cards) => {
      const cnt = k => { const ix = cards.map((c, i) => (c.kind === k ? i : -1)).filter(i => i >= 0); return [ix.filter(i => res[i].you === res[i].truth).length, ix.length]; };
      return L.sum(cnt('lr'), cnt('bs'), cnt('op'));
    }
  });
})();
