// Lecture 1 interactive page: from linear models to the multilayer perceptron. Inputs in L1_data.js (lectures/L1/build_interactive_L1.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, SERIES, nf, pct, $, AX, layout, ttl, plot, out, vline, hline, bindSlider, bindSelect, bindToggles, Player, range, safe } = window.NNI;
const P = (id, input, vals, ms) => new Player(id, $(input), vals, ms);
const COL = { logit: '#1baf7a', mlp: '#eb6834', xgb: '#eda100', pos: '#e34948', neg: '#2a78d6' };
const sig = z => 1 / (1 + Math.exp(-z));
// area under the ROC curve = probability that a random positive scores above a random negative (ties count one half)
function auc(s, y) {
  const idx = s.map((v, i) => i).sort((a, b) => s[a] - s[b]);
  let r = 0, npos = 0, i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++;
    const rk = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) if (y[idx[k]] === 1) { r += rk; npos++; }
    i = j + 1;
  }
  const nneg = y.length - npos;
  return (r - npos * (npos + 1) / 2) / (npos * nneg);
}
// ROC curve points (false positive rate, true positive rate) over all cut-offs
function roc(s, y) {
  const idx = s.map((v, i) => i).sort((a, b) => s[b] - s[a]);
  const P1 = y.filter(v => v === 1).length, N0 = y.length - P1;
  const fx = [0], ty = [0];
  let tp = 0, fp = 0;
  for (let k = 0; k < idx.length; k++) {
    y[idx[k]] === 1 ? tp++ : fp++;
    if (k === idx.length - 1 || s[idx[k + 1]] !== s[idx[k]]) { fx.push(100 * fp / N0); ty.push(100 * tp / P1); }
  }
  return { fx, ty };
}
const confusion = (s, y, t) => { let tp = 0, fp = 0, pos = 0; s.forEach((v, i) => { if (y[i] === 1) pos++; if (v >= t) { if (y[i] === 1) tp++; else fp++; } }); return { tp, fp, pos, neg: y.length - pos }; };

// ---------------------------------------------------------------- 1. one neuron on two HMEQ variables
safe('one-neuron', () => {
  const d = D.nb;
  const m = bindSelect('nb-m', draw), t = bindSlider('nb-t', v => nf(v, 2), draw);
  const w1 = bindSlider('nb-w1', v => nf(v, 2), draw), w2 = bindSlider('nb-w2', v => nf(v, 2), draw), b = bindSlider('nb-b', v => nf(v, 2), draw);
  $('nb-fit').addEventListener('click', () => {
    [['nb-w1', d.w[0]], ['nb-w2', d.w[1]], ['nb-b', d.b]].forEach(([id, v]) => { $(id).value = v; $(id).dispatchEvent(new Event('input')); });
  });
  const z1 = d.x1.map(v => (v - d.mu[0]) / d.sd[0]), z2 = d.x2.map(v => (v - d.mu[1]) / d.sd[1]);
  const pts = (cls, name, col, sym) => {
    const ii = d.y.map((v, i) => i).filter(i => d.y[i] === cls);
    return { x: ii.map(i => d.x1[i]), y: ii.map(i => d.x2[i]), mode: 'markers', type: 'scatter', name,
      marker: { color: col, size: cls ? 6 : 4, opacity: cls ? 0.9 : 0.45, symbol: sym }, hovertemplate: '%{x:.1f}% · %{y:.0f}<extra></extra>' };
  };
  function draw() {
    const W1 = w1(), W2 = w2(), B = b(), th = t(), mode = m();
    const neuron = d.gy.map(yv => d.gx.map(xv => sig(B + W1 * (xv - d.mu[0]) / d.sd[0] + W2 * (yv - d.mu[1]) / d.sd[1])));
    const Z = mode === 'n' ? neuron : d.mlp;
    const tr = [
      { type: 'heatmap', x: d.gx, y: d.gy, z: Z, zmin: 0, zmax: 0.6, colorscale: [[0, '#fff7ec'], [0.5, '#fdae6b'], [1, '#d94801']], opacity: 0.6,
        showscale: true, colorbar: { title: { text: T.prob, side: 'right', font: { size: 11 } }, thickness: 10, len: 0.8 }, hovertemplate: '%{z:.2f}<extra></extra>' },
      { type: 'contour', x: d.gx, y: d.gy, z: Z, showscale: false, contours: { start: th, end: th, size: 1, coloring: 'none' },
        line: { color: C.ink, width: 2 }, name: T.boundary(nf(th, 2)), showlegend: true, hoverinfo: 'skip' },
      pts(0, T.repaid, COL.neg, 'circle'), pts(1, T.defaulted, COL.pos, 'x')];
    if (mode !== 'n') tr.splice(2, 0, { type: 'contour', x: d.gx, y: d.gy, z: neuron, showscale: false, contours: { start: th, end: th, size: 1, coloring: 'none' },
      line: { color: C.ink, width: 1.5, dash: 'dash' }, name: T.boundary(nf(th, 2)) + ' · ' + T.nbTitle('n').split(':')[0], showlegend: true, hoverinfo: 'skip' });
    plot('nb-chart', tr, layout({ title: ttl(T.nbTitle(mode)), xaxis: AX(T.dti, { range: [0, 70] }), yaxis: AX(T.clage, { range: [0, 600] }) }));
    const s = z1.map((v, i) => B + W1 * v + W2 * z2[i]);
    const a = auc(s, d.y);
    const ps = mode === 'n' ? s.map(sig) : null;
    const lines = [T.nb1(nf(a, 3), nf(W1, 2), nf(W2, 2), nf(B, 2))];
    if (ps) {
      const cf = confusion(ps, d.y, th);
      lines.push(T.nb2(nf(th, 2), cf.tp + cf.fp, cf.tp, cf.pos, cf.fp, cf.neg));
    }
    if (a < 0.5) lines.push(T.nbRev);
    out('nb-out', lines);
  }
  draw();
});

// ---------------------------------------------------------------- 2. universal approximation on latitude
safe('universal-approximation', () => {
  const d = D.ua;
  const h = bindSlider('ua-h', v => d.H[v], draw), s = bindToggles('ua-s', draw);
  function draw() {
    const H = d.H[h()], f = d.fits[H], on = s();
    const tr = [{ x: d.bx, y: d.by, mode: 'markers', type: 'scatter', name: T.data, marker: { color: C.RO, size: 6 } }];
    if (on.includes('lin')) tr.push({ x: d.x, y: d.lin, mode: 'lines', name: T.lin, line: { color: COL.logit, width: 1.5 } });
    tr.push({ x: d.x, y: f.f, mode: 'lines', name: T.fit(H), line: { color: COL.mlp, width: 2.5 } });
    plot('ua-fit', tr, layout({ title: ttl(T.fitTitle), xaxis: AX(T.latitude, { range: [32.5, 42] }), yaxis: AX(T.value, { range: [0.5, 3.2] }) }));
    if (on.includes('units')) {
      const ut = f.u.map((u, j) => ({ x: d.x, y: u, mode: 'lines', name: T.unit(j + 1), showlegend: H <= 5,
        line: { color: SERIES[j % SERIES.length], width: 1.2 }, hovertemplate: T.unit(j + 1) + ': %{y:.2f}<extra></extra>' }));
      plot('ua-units', ut, layout({ title: ttl(T.unitsTitle), xaxis: AX(T.latitude, { range: [32.5, 42] }), yaxis: AX(T.value),
        shapes: [hline(0, C.grey, 'dot')] }));
    } else plot('ua-units', [], layout({ title: ttl(T.unitsTitle), xaxis: AX(T.latitude, { range: [32.5, 42] }), yaxis: AX(T.value) }));
    out('ua-out', [T.ua(H, nf(f.te, 3), nf(f.tr, 3), nf(d.mse_lin, 3))]);
  }
  draw();
  P('ua-play', 'ua-h', range(0, d.H.length - 1, 1), 900);
});

// ---------------------------------------------------------------- 3. width of the hidden layer, HMEQ test AUC
safe('width', () => {
  const d = D.wd;
  const h = bindSlider('wd-h', v => d.H[v], draw), s = bindToggles('wd-s', draw);
  const mean = d.H.map(w => d.auc[w].reduce((a, b) => a + b, 0) / 5);
  function draw() {
    const i = h(), H = d.H[i], on = s();
    const lo = d.H.map(w => Math.min(...d.auc[w])), hi = d.H.map(w => Math.max(...d.auc[w]));
    const tr = [
      { x: d.H, y: hi, mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: d.H, y: lo, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(235,104,52,0.18)', name: T.range5, hoverinfo: 'skip' },
      { x: d.H, y: mean, mode: 'lines+markers', name: T.mean5, line: { color: COL.mlp, width: 2 }, marker: { size: 5 }, hovertemplate: 'H = %{x}: %{y:.3f}<extra></extra>' }];
    if (on.includes('seeds')) for (let k = 0; k < 5; k++) tr.push({ x: d.H, y: d.H.map(w => d.auc[w][k]), mode: 'markers', name: T.seeds, showlegend: k === 0,
      marker: { color: C.grey, size: 4, opacity: 0.7 }, hovertemplate: 'H = %{x}: %{y:.3f}<extra></extra>' });
    tr.push({ x: [H], y: [mean[i]], mode: 'markers', showlegend: false, marker: { color: COL.mlp, size: 14, line: { color: C.ink, width: 1.5 } }, hoverinfo: 'skip' });
    const sh = on.includes('bench') ? [hline(d.logit, COL.logit), hline(d.xgb, COL.xgb, 'dash')] : [];
    if (on.includes('bench')) tr.push({ x: [null], y: [null], mode: 'lines', name: T.logitLine, line: { color: COL.logit } },
      { x: [null], y: [null], mode: 'lines', name: T.xgbLine, line: { color: COL.xgb, dash: 'dash' } });
    plot('wd-chart', tr, layout({ xaxis: AX(T.hidden, { type: 'log', tickvals: d.H, ticktext: d.H.map(String) }), yaxis: AX(T.auc, { range: [0.87, 0.97] }), shapes: sh }));
    const sd = Math.sqrt(d.auc[H].reduce((a, v) => a + (v - mean[i]) ** 2, 0) / 4);
    out('wd-out', [T.wd(H, nf(mean[i], 3), nf(sd, 3), nf((d.d + 1) * H + H + 1, 0))]);
  }
  draw();
  P('wd-play', 'wd-h', range(0, d.H.length - 1, 1), 800);
});

// ---------------------------------------------------------------- 4. ROC curve and the cut-off, HMEQ test loans
safe('roc-threshold', () => {
  const d = D.rc, K = ['logit', 'mlp', 'xgb'];
  const R = {}, A = {};
  K.forEach(k => { R[k] = roc(d[k], d.y); A[k] = auc(d[k], d.y); });
  const m = bindSelect('rc-m', draw), t = bindSlider('rc-t', v => nf(v, 2), draw);
  function draw() {
    const mk = m(), th = t(), cf = confusion(d[mk], d.y, th);
    const tr = K.map(k => ({ x: R[k].fx, y: R[k].ty, mode: 'lines', name: `${T.models[k]} (AUC ${nf(A[k], 3)})`,
      line: { color: COL[k], width: k === mk ? 3 : 1.2 }, opacity: k === mk ? 1 : 0.6, hovertemplate: '%{x:.1f}% · %{y:.1f}%<extra></extra>' }));
    tr.push({ x: [0, 100], y: [0, 100], mode: 'lines', name: T.chance, line: { color: C.grey, dash: 'dot', width: 1 } });
    tr.push({ x: [100 * cf.fp / cf.neg], y: [100 * cf.tp / cf.pos], mode: 'markers', showlegend: false, marker: { color: COL[mk], size: 13, line: { color: C.ink, width: 1.5 } } });
    plot('rc-roc', tr, layout({ title: ttl(T.rocTitle), xaxis: AX(T.fpr, { range: [0, 100] }), yaxis: AX(T.tpr, { range: [0, 101] }) }));
    const sc = d[mk];
    const h = cls => ({ x: sc.filter((v, i) => d.y[i] === cls), type: 'histogram', xbins: { start: 0, end: 1, size: 0.025 }, opacity: 0.7,
      name: cls ? T.defaulted.replace(/ \(.*\)/, '') : T.repaid.replace(/ \(.*\)/, ''), marker: { color: cls ? COL.pos : COL.neg } });
    plot('rc-hist', [h(0), h(1)], layout({ title: ttl(T.histTitle + ' · ' + T.models[mk]), barmode: 'overlay', xaxis: AX(T.pd, { range: [0, 1] }),
      yaxis: AX(T.count, { type: 'log' }), shapes: [vline(th, C.ink, 'dash')] }));
    out('rc-out', [T.rc(T.models[mk], nf(A[mk], 3), nf(th, 2), pct(cf.tp / cf.pos, 1), cf.tp, pct(cf.fp / cf.neg, 1), cf.fp, cf.tp + cf.fp)]);
  }
  draw();
  P('rc-play', 'rc-t', range(0.05, 0.95, 0.05), 600);
});

// ---------------------------------------------------------------- 5. cost of errors and the cut-off, German credit
safe('cost-threshold', () => {
  const d = D.ct, K = ['logit', 'mlp', 'xgb'], n = d.y.length, bad = d.y.filter(v => v === 1).length;
  const c = bindSlider('ct-c', v => nf(v, 1), draw), t = bindSlider('ct-t', v => nf(v, 2), draw), on = bindToggles('ct-m', draw);
  const grid = range(0.02, 0.9, 0.01);
  const cost = (p, th, cfn) => { let s = 0; p.forEach((v, i) => { if (v >= th) { if (d.y[i] === 0) s += 1; } else if (d.y[i] === 1) s += cfn; }); return s / n; };
  function draw() {
    const cfn = c(), th = t(), ks = on(), bt = 1 / (1 + cfn);
    const tr = [], lines = [];
    ks.forEach(k => {
      const cs = grid.map(g => cost(d[k], g, cfn));
      const j = cs.indexOf(Math.min(...cs));
      tr.push({ x: grid, y: cs, mode: 'lines', name: T.models[k], line: { color: COL[k], width: 2 }, hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' });
      tr.push({ x: [grid[j]], y: [cs[j]], mode: 'markers', showlegend: false, marker: { color: COL[k], size: 9 }, hoverinfo: 'skip' });
      lines.push(T.ct(T.models[k], nf(th, 2), nf(cost(d[k], th, cfn), 3), nf(cs[j], 3), nf(grid[j], 2)));
    });
    tr.push({ x: [null], y: [null], mode: 'lines', name: T.bayes(nf(cfn, 1)), line: { color: C.ink, dash: 'dash' } },
      { x: [null], y: [null], mode: 'lines', name: T.acceptAll, line: { color: C.grey, dash: 'dot' } },
      { x: [null], y: [null], mode: 'lines', name: T.rejectAll, line: { color: C.grey, dash: 'dashdot' } });
    lines.push(T.ct0(nf(bt, 3), nf(cfn * bad / n, 3), nf((n - bad) / n, 3)));
    plot('ct-chart', tr, layout({ title: ttl(T.costTitle(nf(cfn, 1))), xaxis: AX(T.thr, { range: [0, 0.9] }), yaxis: AX(T.costPA, { rangemode: 'tozero' }),
      shapes: [vline(bt, C.ink, 'dash'), vline(th, C.navy, 'solid'), hline(cfn * bad / n, C.grey, 'dot'), hline((n - bad) / n, C.grey, 'dashdot')] }));
    out('ct-out', lines);
  }
  draw();
  P('ct-play', 'ct-t', range(0.05, 0.9, 0.05), 600);
});
})();
