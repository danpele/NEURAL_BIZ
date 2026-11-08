// Lecture 2 interactive page: training, backpropagation and optimisation. Inputs in L2_data.js (lectures/L2/build_interactive_L2.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, SERIES, nf, $, AX, layout, ttl, plot, out, hline, vline, bindSlider, bindSelect, bindToggles, bindRadios, Player, range, safe } = window.NNI;
const P = (id, input, vals, ms) => new Player(id, $(input), vals, ms);
const OCOL = { sgd: '#2a78d6', momentum: '#6250d6', rmsprop: '#eda100', adam: '#eb6834', adamw: '#e87ba4' };
const sci = x => {                                   // 8.5e-7 -> "8.5 × 10⁻⁷" (decimal comma in Romanian)
  if (x === null || !Number.isFinite(x)) return '∞';
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e >= -2 && e <= 3) return nf(x, e >= 2 ? 0 : e >= 0 ? 2 : 3);
  const sup = String(e).replace('-', '⁻').replace(/\d/g, c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);
  return `${nf(x / 10 ** e, 1)} × 10${sup}`;
};
const lrTxt = v => nf(v, v < 0.001 ? 4 : v < 0.01 ? 3 : v < 0.1 ? 2 : v < 1 ? 1 : 0);

// ---------------------------------------------------------------- 1. gradient descent on the real loss surface
safe('gd-paths', () => {
  const x = bindRadios('gd-x', draw), e = bindSlider('gd-e', v => nf(v, 2), draw), m = bindSlider('gd-m', v => nf(v, 2), draw),
    k = bindSlider('gd-k', v => v, draw);
  function draw() {
    const xs = x(), s = D.gd[xs], mult = e(), beta = m(), K = k();
    const L = (a, b) => s.myy - 2 * (a * s.my + b * s.mxy) + a * a + 2 * a * b * s.mx + b * b * s.mxx;
    const tr2 = 2 * (1 + s.mxx), det = 4 * (s.mxx - s.mx * s.mx), disc = Math.sqrt(tr2 * tr2 - 4 * det);
    const lmax = (tr2 + disc) / 2, lmin = (tr2 - disc) / 2, eta = mult / lmax;
    const bs = (s.mxy - s.mx * s.my) / (s.mxx - s.mx * s.mx), as = s.my - bs * s.mx, Lmin = L(as, bs);
    let w = xs === 'raw' ? [4, -0.5] : [4, -1], v = [0, 0];
    const pa = [w[0]], pb = [w[1]], loss = [L(w[0], w[1])];
    let div = false;
    for (let i = 0; i < K; i++) {
      const g = [2 * (w[0] + w[1] * s.mx - s.my), 2 * (w[0] * s.mx + w[1] * s.mxx - s.mxy)];
      v = [beta * v[0] + g[0], beta * v[1] + g[1]];
      w = [w[0] - eta * v[0], w[1] - eta * v[1]];
      if (!Number.isFinite(w[0]) || Math.abs(w[0]) + Math.abs(w[1]) > 1e8) { div = true; break; }
      pa.push(w[0]); pb.push(w[1]); loss.push(L(w[0], w[1]));
    }
    const A = range(Math.min(as, -0.5) - 1, 5, 0.05), B = range(-1.2, 1.4, 0.02);
    const Z = B.map(b => A.map(a => Math.log(L(a, b))));
    plot('gd-path', [
      { type: 'contour', x: A, y: B, z: Z, showscale: false, ncontours: 16, contours: { coloring: 'lines' }, line: { width: 0.8 },
        colorscale: [[0, C.grey], [1, C.band]], hoverinfo: 'skip', showlegend: false },
      { x: pa, y: pb, mode: 'lines+markers', name: T.path, line: { color: beta > 0 ? '#eb6834' : '#2a78d6', width: 1.5 }, marker: { size: 4 },
        hovertemplate: 'a = %{x:.3f}, b = %{y:.3f}<extra></extra>' },
      { x: [pa[0]], y: [pb[0]], mode: 'markers', name: T.start, marker: { color: C.ink, size: 9, symbol: 'circle-open' } },
      { x: [as], y: [bs], mode: 'markers', name: T.min, marker: { color: C.ink, size: 14, symbol: 'star' } }],
      layout({ title: ttl(T.pathTitle(xs)), xaxis: AX(T.a, { range: [A[0], 5] }), yaxis: AX(T.b, { range: [-1.2, 1.4] }) }));
    plot('gd-loss', [{ x: loss.map((_, i) => i), y: loss, mode: 'lines', name: 'MSE', line: { color: beta > 0 ? '#eb6834' : '#2a78d6', width: 2 } }],
      layout({ title: ttl(T.lossTitle), xaxis: AX(T.step, { range: [0, K] }), yaxis: AX(T.mse, { type: 'log' }), showlegend: false,
        shapes: [hline(Lmin, C.ink, 'dash')] }));
    const lines = [T.gd1(nf(eta, 4), nf(2 / lmax, 4), pa.length - 1, nf(loss[loss.length - 1], 4), nf(Lmin, 4), nf(beta, 2)),
      T.gd2(nf(lmax, 2), nf(lmin, 3), nf(lmax / lmin, 1))];
    if (div || loss[loss.length - 1] > 1e3) lines.push(T.div);
    out('gd-out', lines);
  }
  draw();
  P('gd-play', 'gd-k', range(1, 200, 3), 120);
});

// ---------------------------------------------------------------- 2. learning-rate sweep, plain SGD
safe('lr-sweep', () => {
  const d = D.lr, lrs = d.lrs, keys = d.keys;
  const i = bindSlider('lr-i', v => lrTxt(lrs[v]), draw), s = bindToggles('lr-s', draw);
  const ok = v => v < 5;
  function draw() {
    const j = i(), key = keys[j], ep = range(1, 30, 1);
    const tr = [];
    if (s().includes('all')) keys.forEach((kk, q) => { if (q !== j) tr.push({ x: ep, y: d.curve[kk].map(v => ok(v) ? v : null), mode: 'lines', showlegend: false,
      line: { color: C.band, width: 1 }, hovertemplate: `η = ${lrTxt(lrs[q])}: %{y:.3f}<extra></extra>` }); });
    tr.push({ x: ep, y: d.curve[key].map(v => ok(v) ? v : null), mode: 'lines+markers', name: `η = ${lrTxt(lrs[j])}`, line: { color: '#eb6834', width: 2.5 }, marker: { size: 4 } });
    plot('lr-curve', tr, layout({ title: ttl(T.lrTitle(lrTxt(lrs[j]))), xaxis: AX(T.epoch), yaxis: AX(T.vmse, { range: [0, 1.65] }),
      shapes: [hline(D.mse_lin_val, '#1baf7a', 'dash')] }));
    const gx = [], gy = [], dx = [];
    lrs.forEach((l, q) => d.final[keys[q]].forEach(v => { if (ok(v)) { gx.push(l); gy.push(v); } else dx.push(l); }));
    const med = keys.map(kk => { const v = d.final[kk]; return v.every(ok) ? [...v].sort((a, b) => a - b)[1] : null; });
    plot('lr-all', [
      { x: gx, y: gy, mode: 'markers', name: T.seed, marker: { color: '#eb6834', size: 7, opacity: 0.6 } },
      { x: lrs, y: med, mode: 'lines', name: T.median, line: { color: '#eb6834', width: 1.5 } },
      { x: dx, y: dx.map(() => 1.55), mode: 'markers', name: T.diverged, marker: { color: '#e34948', size: 10, symbol: 'x' } },
      { x: [lrs[j]], y: [ok(d.final[key][0]) ? d.final[key][0] : 1.55], mode: 'markers', showlegend: false, marker: { size: 16, color: 'rgba(0,0,0,0)', line: { color: C.ink, width: 2 } } },
      { x: [null], y: [null], mode: 'lines', name: T.linear, line: { color: '#1baf7a', dash: 'dash' } }],
      layout({ title: ttl(T.sweepTitle), xaxis: AX(T.lr, { type: 'log' }), yaxis: AX(T.vmse, { range: [0, 1.65] }), shapes: [hline(D.mse_lin_val, '#1baf7a', 'dash')] }));
    const f = d.final[key], nd = f.filter(v => !ok(v)).length;
    const lines = [];
    if (ok(d.curve[key][29])) lines.push(T.lrOut(lrTxt(lrs[j]), nf(d.curve[key][29], 3), nf(Math.min(...f.filter(ok)), 3), nf(Math.max(...f.filter(ok)), 3)));
    if (nd) lines.push(T.lrDiv(lrTxt(lrs[j]), nd));
    out('lr-out', lines);
  }
  draw();
  P('lr-play', 'lr-i', range(0, 8, 1), 900);
});

// ---------------------------------------------------------------- 3. optimisers and their learning rates
safe('optimizers', () => {
  const ep = range(1, 60, 1);
  const o = bindSelect('op-o', draw), s = bindToggles('op-s', draw);
  const l = bindSlider('op-l', v => lrTxt(D.grid[$('op-o').value][v]), draw);
  const mean = runs => runs[0].map((_, t) => runs.reduce((a, r) => a + r[t], 0) / runs.length);
  function draw() {
    const oo = o(), lr = D.grid[oo][l()], runs = D.opt[`${oo}|${lr}`];
    $('op-l-v').textContent = lrTxt(lr);
    const tr = [], on = s();
    if (on.includes('best')) Object.keys(D.grid).filter(k => k !== oo).forEach(k => tr.push({ x: ep, y: mean(D.opt[`${k}|${D.best[k]}`]), mode: 'lines',
      name: `${T.onames[k]} (${T.best(lrTxt(D.best[k]))})`, line: { color: OCOL[k], width: 1.2, dash: 'dot' } }));
    if (on.includes('seeds')) runs.forEach((r, q) => tr.push({ x: ep, y: r, mode: 'lines', showlegend: false, line: { color: OCOL[oo], width: 0.8 }, opacity: 0.45,
      hovertemplate: `seed ${q + 1}: %{y:.3f}<extra></extra>` }));
    const mu = mean(runs);
    tr.push({ x: ep, y: mu, mode: 'lines', name: `${T.onames[oo]}, η = ${lrTxt(lr)} (${T.mean3})`, line: { color: OCOL[oo], width: 2.8 } });
    tr.push({ x: [null], y: [null], mode: 'lines', name: T.linear, line: { color: '#1baf7a', dash: 'dash' } });
    plot('op-chart', tr, layout({ title: ttl(T.opTitle(T.onames[oo], lrTxt(lr))), xaxis: AX(T.epoch), yaxis: AX(T.vmse, { range: [0.22, 0.62] }),
      shapes: [hline(D.mse_lin_val, '#1baf7a', 'dash')] }));
    out('op-out', [T.op(T.onames[oo], lrTxt(lr), nf(mu[59], 3), nf(Math.min(...mu), 3), nf(mu[9], 3)), T.opBest(lrTxt(D.best[oo]))]);
  }
  draw();
  P('op-play', 'op-l', [0, 1, 2], 1000);
});

// ---------------------------------------------------------------- 4. batch size
safe('batch-size', () => {
  const x = bindRadios('bs-x', draw), b = bindToggles('bs-b', draw);
  const COLB = { 16: '#2a78d6', 128: '#eb6834', 1024: '#6250d6' };
  function draw() {
    const xs = x(), on = b(), tr = [], lines = [];
    on.forEach(k => {
      const r = D.bs[k], ep = r.val.map((_, i) => i + 1);
      const xx = xs === 'ep' ? ep : xs === 'st' ? ep.map(e => e * r.spe) : ep.map(e => e * r.sec / 40);
      tr.push({ x: xx, y: r.val, mode: 'lines+markers', name: T.bsName(k), line: { color: COLB[k], width: 2 }, marker: { size: 3 } });
      lines.push(T.bs(k, r.spe, nf(r.sec, 1), nf(r.val[4], 3), nf(r.val[39], 3)));
    });
    plot('bs-chart', tr, layout({ title: ttl(T.bsTitle), xaxis: AX(xs === 'ep' ? T.epoch : xs === 'st' ? T.steps : T.secs, xs === 'ep' ? {} : { type: 'log' }),
      yaxis: AX(T.vmse, { range: [0.2, 0.9] }) }));
    out('bs-out', lines.length ? lines : ['–']);
  }
  draw();
});

// ---------------------------------------------------------------- 5. vanishing and exploding gradients
safe('vanishing', () => {
  const dep = D.vg.depths;
  const a = bindSelect('vg-a', draw), i = bindSelect('vg-i', draw), d = bindSlider('vg-d', v => dep[v], draw);
  function draw() {
    const nl = dep[d()], v = D.vg.v[`${a()}|${i()}|${nl}`];
    const lay = v.map((_, q) => q + 1), good = v.map(x => (x !== null && x > 0 ? x : null));
    plot('vg-chart', [{ x: lay, y: good, mode: 'lines+markers', line: { color: '#2a78d6', width: 2 }, marker: { size: 7 }, hovertemplate: '%{x}: %{y:.2e}<extra></extra>' }],
      layout({ title: ttl(T.vgTitle(T.acts[a()], T.inits[i()], nl)), showlegend: false, xaxis: AX(T.layer, { dtick: nl > 12 ? 2 : 1, range: [0.5, nl + 0.5] }),
        yaxis: AX(T.gnorm, { type: 'log', exponentformat: 'power' }) }));
    const lines = [];
    if (v.every(x => x !== null)) lines.push(T.vg(sci(v[0]), sci(v[nl - 1]), v[nl - 1] > 0 ? sci(v[0] / v[nl - 1]) : '–'));
    if (v.some(x => x === null)) lines.push(T.vgInf);
    const nz = v.filter(x => x === 0).length;
    if (nz) lines.push(T.vgZero(nz));
    out('vg-out', lines);
  }
  draw();
  P('vg-play', 'vg-d', range(0, dep.length - 1, 1), 800);
});
})();
