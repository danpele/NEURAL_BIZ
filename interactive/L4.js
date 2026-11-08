// Lecture 4 interactive page: convolutional neural networks and transfer learning. Inputs in L4_data.js (lectures/L4/build_interactive_L4.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, nf, pct, quantile, $, AX, layout, ttl, plot, out, vline, hline, bindSlider, bindSelect, bindToggles, bindRadios, Player, range, safe } = window.NNI;
const P = (id, input, vals, ms) => new Player(id, $(input), vals, ms);
const GREY = [[0, '#000000'], [1, '#ffffff']], GREY_R = [[0, '#ffffff'], [1, '#000000']];
const DIV = [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']];
const IMG = (title, extra) => layout(Object.assign({ title: ttl(title), margin: { l: 30, r: 10, t: 30, b: 20 },
  xaxis: { visible: false, constrain: 'domain' }, yaxis: { visible: false, autorange: 'reversed', scaleanchor: 'x', constrain: 'domain' } }, extra || {}));
const COLM = { scratch: '#1baf7a', probe: '#eda100', finetune: '#eb6834', cnn: '#1baf7a', cnn_aug: '#eb6834', plain32: '#eda100', res32: '#6250d6' };

// ---------------------------------------------------------------- 1. a convolution on a real Sentinel-2 tile
safe('convolution', () => {
  const KER = { sx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], sy: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], lap: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]], sharp: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], id: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] };
  const i = bindSelect('cv-i', draw), k = bindSelect('cv-k', draw), s = bindSlider('cv-s', v => v, draw), o = bindToggles('cv-o', draw);
  function draw() {
    const X = D.tiles[+i()].map(r => r.map(v => v / 255)), n = X.length, kk = k(), Kr = KER[kk], st = s(), on = o();
    const p = on.includes('pad') ? 1 : 0, relu = on.includes('relu'), no = Math.floor((n + 2 * p - 3) / st) + 1;
    const Y = [];
    for (let r = 0; r < no; r++) {
      const row = [];
      for (let c = 0; c < no; c++) {
        let v = 0;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
          const rr = r * st + a - p, cc = c * st + b - p;
          if (rr >= 0 && rr < n && cc >= 0 && cc < n) v += Kr[a][b] * X[rr][cc];
        }
        row.push(relu ? Math.max(v, 0) : v);
      }
      Y.push(row);
    }
    const signed = !relu && ['sx', 'sy', 'lap'].includes(kk);
    // colour limits: 99th percentile of |output| inside the image (the zero-padded border would otherwise dominate the scale)
    const inner = Y.slice(1, -1).flatMap(r => r.slice(1, -1).map(Math.abs));
    const m = quantile(inner.length ? inner : [1], 0.99) || 1;
    plot('cv-in', [{ type: 'heatmap', z: X, colorscale: GREY, zmin: 0, zmax: 1, showscale: false, hovertemplate: '%{z:.2f}<extra></extra>' }],
      IMG(T.input(T.es[+i()])));
    plot('cv-res', [{ type: 'heatmap', z: Y, colorscale: signed ? DIV : GREY, zmin: signed ? -m : Math.min(0, ...Y.map(r => Math.min(...r))), zmax: m,
      showscale: true, colorbar: { thickness: 8, len: 0.8 }, hovertemplate: '%{z:.2f}<extra></extra>' }], IMG(T.output(kk, no)));
    out('cv-out', [T.cv(n, p, st, no), T.cvK(T.kname[kk] + ' [' + Kr.map(r => r.map(v => nf(v, Math.abs(v) < 1 && v !== 0 ? 2 : 0)).join(' ')).join(' | ') + ']')]);
  }
  draw();
});

// ---------------------------------------------------------------- 2. receptive field
safe('receptive-field', () => {
  const l = bindSlider('rf-l', v => v, draw), k = bindSelect('rf-k', draw), pl = bindToggles('rf-p', draw);
  const rf = (L, kk, pool) => { let r = 1, j = 1; const res = []; for (let q = 1; q <= L; q++) { r += (kk - 1) * j; res.push(r); if (pool && q % 2 === 0) { r += j; j *= 2; } } return res; };
  function draw() {
    const L = l(), kk = +k(), pool = pl().includes('pool'), xs = range(1, 12, 1);
    const base = rf(12, 3, false), cur = rf(12, kk, pool), r = cur[L - 1];
    plot('rf-chart', [{ x: xs, y: base, mode: 'lines+markers', name: T.plain, line: { color: '#1baf7a', width: 1.5 }, marker: { size: 4 } },
      { x: xs, y: cur, mode: 'lines+markers', name: T.this, line: { color: '#eb6834', width: 2.5 }, marker: { size: 5 } },
      { x: [L], y: [r], mode: 'markers', showlegend: false, marker: { size: 14, color: 'rgba(0,0,0,0)', line: { color: C.ink, width: 2 } } }],
      layout({ xaxis: AX(T.layers, { dtick: 1 }), yaxis: AX(T.rfY, { range: [0, 1.08 * Math.max(70, ...cur)] }), shapes: [hline(64, C.grey, 'dot')] }));
    const h = Math.min(r, 64) / 2, c0 = 31.5;
    plot('rf-tile', [{ type: 'heatmap', z: D.tiles[3], colorscale: GREY, zmin: 0, zmax: 255, showscale: false, hoverinfo: 'skip' }],
      IMG(T.rfTile, { shapes: [{ type: 'rect', x0: c0 - h, x1: c0 + h, y0: c0 - h, y1: c0 + h, line: { color: '#eb6834', width: 3 } }] }));
    const lines = [T.rf(L, r, nf(10 * r, 0), kk * kk)];
    if (r >= 64) lines.push(T.rfFull);
    out('rf-out', lines);
  }
  draw();
  P('rf-play', 'rf-l', range(1, 12, 1), 600);
});

// ---------------------------------------------------------------- 3. feature maps of the trained CNN
safe('feature-maps', () => {
  const i = bindSelect('fm-i', draw), b = bindRadios('fm-b', () => { const el = $('fm-c'); el.max = b() === 'b1' ? 31 : 127; if (+el.value > +el.max) el.value = el.max; $('fm-c-v').textContent = +el.value + 1; draw(); });
  const c = bindSlider('fm-c', v => v + 1, draw);
  function draw() {
    const ii = +i(), bb = b(), e = D.fm[bb][ii], rank = c(), ch = e.order[rank], M = e.m[ch], n = M.length;
    plot('fm-img', [{ type: 'heatmap', z: D.fm.img[ii], colorscale: GREY_R, zmin: 0, zmax: 255, showscale: false, hoverinfo: 'skip' }], IMG(T.img(T.fm[ii])));
    plot('fm-map', [{ type: 'heatmap', z: M, colorscale: 'Viridis', zmin: 0, zmax: 99, showscale: false, hovertemplate: '%{z}<extra></extra>' }], IMG(T.map(bb, rank + 1, n)));
    const p = D.fm.p[ii], ord = p.map((v, q) => q).sort((x, y) => p[y] - p[x]);
    out('fm-out', [T.fmOut(T.fm[ord[0]], pct(p[ord[0]], 1), ord.slice(1, 3).map(q => `${T.fm[q]} ${pct(p[q], 1)}`).join(', ')),
      T.fmMap(ch + 1, nf(e.max * Math.max(...M.map(r => Math.max(...r))) / 99, 2))]);
  }
  draw();
  P('fm-play', 'fm-c', range(0, 31, 1), 400);
});

// ---------------------------------------------------------------- 4. confusion matrix, Fashion-MNIST test set
safe('confusion', () => {
  const m = bindSelect('cm-m', draw), c = bindSelect('cm-c', draw), s = bindToggles('cm-n', draw);
  function draw() {
    const mm = m(), cc = +c(), M = D.cm.m[mm], usePct = s().includes('pct');
    const Z = M.map(r => { const t = r.reduce((a, b) => a + b, 0); return r.map(v => (usePct ? 100 * v / t : v)); });
    const acc = M.reduce((a, r, q) => a + r[q], 0) / 10000;
    plot('cm-chart', [{ type: 'heatmap', z: Z, x: T.fm, y: T.fm, colorscale: [[0, '#ffffff'], [1, '#6fa8e6']], zmin: 0, zmax: usePct ? 100 : 1000, showscale: false,
      text: Z.map(r => r.map(v => (usePct ? (v >= 1 ? nf(v, 0) : '') : (v >= 5 ? String(Math.round(v)) : '')))), texttemplate: '%{text}',
      hovertemplate: '%{y} → %{x}: %{z:.0f}<extra></extra>' }],
      layout({ title: ttl(T.cmTitle(T.models[mm], pct(acc, 1))), margin: { l: 110, r: 10, t: 30, b: 100 },
        xaxis: AX(T.pred, { tickangle: -40 }), yaxis: AX(T.truth, { autorange: 'reversed' }),
        shapes: [{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: cc - 0.5, y1: cc + 0.5, line: { color: '#e34948', width: 2.5 } }] }));
    const row = M[cc], off = row.map((v, q) => (q === cc ? -1 : v)), t = off.indexOf(Math.max(...off));
    out('cm-out', [T.cm(T.fm[cc], pct(row[cc] / 1000, 1), row[cc], `${T.fm[cc]} → ${T.fm[t]}`, row[t]), T.cm2(nf(D.cm.params[mm], 0))]);
  }
  draw();
});

// ---------------------------------------------------------------- 5. plain vs residual networks on CIFAR-10
safe('residual', () => {
  const m = bindToggles('cf-m', draw), y = bindRadios('cf-y', draw);
  function draw() {
    const yy = y(), tr = [], lines = [];
    m().forEach(k => {
      const h = D.cf[k], v = h[yy].map(x => (yy === 'train_loss' ? x : 100 * x));
      tr.push({ x: v.map((_, q) => q + 1), y: v, mode: 'lines+markers', name: T.cfNames[k], line: { color: COLM[k], width: 2 }, marker: { size: 4 } });
      lines.push(T.cf(T.cfNames[k], pct(h.test, 1), nf(h.params, 0), pct(h.train_acc[11], 1), pct(h.valid_acc[11], 1), nf(h.train_loss[11], 3)));
    });
    plot('cf-chart', tr, layout({ xaxis: AX(T.epoch, { dtick: 1 }), yaxis: AX(T.cfY[yy]) }));
    out('cf-out', lines.length ? lines : ['–']);
  }
  draw();
});

// ---------------------------------------------------------------- 6. transfer learning and the label budget
safe('transfer', () => {
  const d = D.tl, B = d.budget;
  const n = bindSlider('tl-n', v => B[v], draw), m = bindToggles('tl-m', draw);
  function draw() {
    const j = n(), on = m(), h = [], tb = [], lines = [];
    on.forEach(k => {
      if (d.hist[k]) h.push({ x: d.hist[k].map(r => r[0]), y: d.hist[k].map(r => 100 * r[1]), mode: 'lines+markers', name: T.tlNames[k], line: { color: COLM[k], width: 2 }, marker: { size: 4 } });
      else h.push({ x: [0, 18], y: [100 * d.probe_va, 100 * d.probe_va], mode: 'lines', name: T.tlNames[k], line: { color: COLM[k], dash: 'dash', width: 2 } });
      const mu = B.map(b => 50 * (d.acc[k][b][0] + d.acc[k][b][1]));
      tb.push({ x: B, y: mu, mode: 'lines+markers', name: T.tlNames[k], line: { color: COLM[k], width: 2 }, marker: { size: 5 }, showlegend: false });
      tb.push({ x: B.flatMap(b => [b, b]), y: B.flatMap(b => d.acc[k][b].map(v => 100 * v)), mode: 'markers', showlegend: false, marker: { color: COLM[k], size: 5, opacity: 0.5 }, hoverinfo: 'skip' });
      tb.push({ x: [B[j]], y: [mu[j]], mode: 'markers', showlegend: false, marker: { size: 14, color: 'rgba(0,0,0,0)', line: { color: C.ink, width: 2 } } });
      lines.push(T.tl(T.tlNames[k], B[j], pct(mu[j] / 100, 1), pct(d.acc[k][B[j]][0], 1), pct(d.acc[k][B[j]][1], 1)));
    });
    plot('tl-hist', h, layout({ title: ttl(T.tlHist), xaxis: AX(T.epochs), yaxis: AX(T.acc, { range: [50, 100] }) }));
    plot('tl-budget', tb, layout({ title: ttl(T.tlBudget), xaxis: AX(T.perClass, { type: 'log', tickvals: B, ticktext: B.map(String) }), yaxis: AX(T.acc, { range: [60, 100] }),
      shapes: [vline(B[j], C.grey, 'dot')] }));
    lines.push(T.tl0(B[j], nf(10 * B[j], 0)));
    out('tl-out', lines);
  }
  draw();
  P('tl-play', 'tl-n', range(0, B.length - 1, 1), 900);
});
})();
