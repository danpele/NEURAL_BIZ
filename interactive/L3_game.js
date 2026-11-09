// Lecture 3 game "Stop early or train longer?": real training (and validation) curves of the lecture; the student picks one of
// three epochs at which to stop. Right answer: the epoch with the lowest TEST loss. Model: early stopping (lowest validation loss).
// Data: L3_game_data.js (lectures/L3/make_game_L3.py). Engine: game_common.js. Charts: inline SVG (svgChart below); after the
// answer the chart of the card is redrawn with the validation and test curves.
(function () {
  'use strict';
  const G = window.NN_GAME_L3, I = G.info, { RO, nf, shuffle } = window.NNGame;
  const n3 = x => nf(x, 3), nl = (x, f) => nf(x, f ? 4 : 3);
  const L = RO ? {
    card: 'Graficul', ep: 'Epoca (scară logaritmică)', loss: 'Pierderea (entropia încrucișată)',
    tr: 'antrenare', va: 'validare', te: 'test', stop: e => `Mă opresc la epoca ${e}`,
    blind: 'Vedeți doar pierderea de antrenare.', both: 'Vedeți pierderea de antrenare și pierderea de validare.',
    q: 'La ce epocă opriți antrenarea?',
    german: r => `Credite germane, rețea 256-256, ${r.p ? 'dropout ' + nf(r.p, 1) : 'fără dropout'}, ${r.wd ? 'penalizarea ponderilor ' + nf(r.wd, 1) : 'fără penalizarea ponderilor'}.`,
    fraud: 'Fraude cu carduri, rețea 256-256, fără regularizare.',
    tl: (es, xs, b) => `Pierderea pe test la epocile ${es.join(' / ')}: ${xs.join(' / ')}; cea mai mică la epoca ${b}.`,
    sc: (f, es, xs) => `${f ? 'PR-AUC (aria de sub curba precizie-sensibilitate)' : 'AUC (aria de sub curba ROC)'} pe test la aceleași epoci: ${xs.join(' / ')}.`,
    sum: (a, b, n) => `<li>${a}: dumneavoastră ${b[0]} din ${n}, oprirea timpurie ${b[1]} din ${n}.</li>`,
    s1: 'Doar pierderea de antrenare', s2: 'Pierderea de antrenare și de validare',
    model: 'Oprirea timpurie (pierderea de validare)'
  } : {
    card: 'Chart', ep: 'Epoch (log scale)', loss: 'Loss (cross-entropy)',
    tr: 'training', va: 'validation', te: 'test', stop: e => `Stop at epoch ${e}`,
    blind: 'You see only the training loss.', both: 'You see the training loss and the validation loss.',
    q: 'At which epoch do you stop training?',
    german: r => `German credit, network 256-256, ${r.p ? 'dropout ' + nf(r.p, 1) : 'no dropout'}, ${r.wd ? 'weight decay ' + nf(r.wd, 1) : 'no weight decay'}.`,
    fraud: 'Card fraud, network 256-256, no regularisation.',
    tl: (es, xs, b) => `Test loss at epochs ${es.join(' / ')}: ${xs.join(' / ')}; lowest at epoch ${b}.`,
    sc: (f, es, xs) => `Test ${f ? 'PR-AUC (area under the precision-recall curve)' : 'AUC (area under the ROC curve)'} at the same epochs: ${xs.join(' / ')}.`,
    sum: (a, b, n) => `<li>${a}: you ${b[0]} of ${n}, early stopping ${b[1]} of ${n}.</li>`,
    s1: 'Training loss only', s2: 'Training and validation loss',
    model: 'Early stopping (validation loss)'
  };
  const COL = { tr: '#2a78d6', va: '#eb6834', te: '#6250d6' };

  // Line chart with a log-scaled epoch axis, dashed markers at the candidate epochs, values above ymax clipped.
  function svgChart(c, keys) {
    const r = c.run, f = r.data === 'fraud', n = r.tr.length, W = 340, H = 206, ml = 44, mr = 10, mt = 30, mb = 30, pw = W - ml - mr, ph = H - mt - mb;
    const ymax = f ? 0.07 : 2.5, ticks = f ? [0, 0.02, 0.04, 0.06] : [0, 0.5, 1, 1.5, 2, 2.5];
    const X = e => ml + Math.log(e) / Math.log(n) * pw, Y = v => mt + ph - v / ymax * ph;
    const id = 'k' + Math.random().toString(36).slice(2, 8);
    let s = `<svg class="g-fig" viewBox="0 0 ${W} ${H}" role="img" aria-label="${L.loss}" style="font:11px Inter,Arial,sans-serif">` +
      `<defs><clipPath id="${id}"><rect x="${ml}" y="${mt}" width="${pw}" height="${ph}"/></clipPath></defs>`;
    ticks.forEach(t => {
      s += `<line x1="${ml}" x2="${ml + pw}" y1="${Y(t)}" y2="${Y(t)}" stroke="#e6e6e6"/><text x="${ml - 5}" y="${Y(t) + 4}" text-anchor="end" fill="#555">${nf(t, f ? 2 : 1)}</text>`;
    });
    [1, 3, 10, 30, 100].filter(e => e <= n).forEach(e => { s += `<text x="${X(e)}" y="${mt + ph + 14}" text-anchor="middle" fill="#555">${e}</text>`; });
    s += `<text x="${ml + pw / 2}" y="${H - 3}" text-anchor="middle" fill="#333">${L.ep}</text>`;
    let lx = ml;                                       // legend on top
    keys.forEach(k => {
      s += `<line x1="${lx}" x2="${lx + 16}" y1="11" y2="11" stroke="${COL[k]}" stroke-width="2.5"${k === 'te' ? ' stroke-dasharray="5 3"' : ''}/><text x="${lx + 20}" y="15" fill="#333">${L[k]}</text>`;
      lx += 30 + L[k].length * 6.2;
    });
    c.eps.forEach(e => {
      s += `<line x1="${X(e)}" x2="${X(e)}" y1="${mt}" y2="${mt + ph}" stroke="#333" stroke-dasharray="3 3"/><text x="${X(e)}" y="${mt - 4}" text-anchor="middle" font-weight="600" fill="#222">${e}</text>`;
    });
    keys.forEach(k => {
      const d = r[k].map((v, i) => (i ? 'L' : 'M') + X(i + 1).toFixed(1) + ',' + Y(v).toFixed(1)).join('');
      s += `<path d="${d}" fill="none" stroke="${COL[k]}" stroke-width="2" stroke-linejoin="round"${k === 'te' ? ' stroke-dasharray="5 3"' : ''} clip-path="url(#${id})"/>`;
    });
    return s + `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="none" stroke="#bbb"/></svg>`;
  }

  const at = (r, k, eps) => eps.map(e => r[k][e - 1]);
  const argmin = a => a.indexOf(Math.min(...a));
  function card(run, blind) {
    const cand = I[run.data].cand, eps = shuffle(cand).slice(0, 3).sort((a, b) => a - b);
    return { run, blind, eps };
  }

  window.NNGame.choice({
    game: 'l3',
    cardTitle: { en: L.card, ro: L.card },
    draw: () => {
      const ger = shuffle(G.runs.filter(r => r.data === 'german')), fr = G.runs.find(r => r.data === 'fraud');
      const nb = I.n_blind, ng = I.n_cards - 2;
      const blind = shuffle([card(fr, true)].concat(ger.slice(0, nb - 1).map(r => card(r, true))));
      const seen = shuffle([card(fr, false)].concat(ger.slice(nb - 1, ng).map(r => card(r, false))));
      return blind.concat(seen);
    },
    render: c => `<p>${c.run.data === 'fraud' ? L.fraud : L.german(c.run)} ${c.blind ? L.blind : L.both}</p><p><b>${L.q}</b></p>` +
      `<div class="g-l3c">${svgChart(c, c.blind ? ['tr'] : ['tr', 'va'])}</div>`,
    choices: c => c.eps.map(e => ({ label: L.stop(e), cls: '' })),
    truth: c => argmin(at(c.run, 'te', c.eps)),
    model: c => argmin(at(c.run, 'va', c.eps)),
    modelName: { en: L.model, ro: L.model },
    reveal: c => {
      const f = c.run.data === 'fraud', te = at(c.run, 'te', c.eps);
      setTimeout(() => { const el = document.querySelector('#g-card .g-l3c'); if (el) el.innerHTML = svgChart(c, ['tr', 'va', 'te']); }, 0);
      return `<li>${L.tl(c.eps, te.map(x => nl(x, f)), c.eps[argmin(te)])}</li><li>${L.sc(f, c.eps, at(c.run, 'sc', c.eps).map(n3))}</li>`;
    },
    summary: (res, cards) => {
      const part = b => { const ix = cards.map((c, i) => (c.blind === b ? i : -1)).filter(i => i >= 0);
        return [[ix.filter(i => res[i].you === res[i].truth).length, ix.filter(i => res[i].model === res[i].truth).length], ix.length]; };
      const a = part(true), b = part(false);
      return L.sum(L.s1, a[0], a[1]) + L.sum(L.s2, b[0], b[1]);
    }
  });
})();
