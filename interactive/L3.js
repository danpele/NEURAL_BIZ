// Lecture 3 interactive page: regularisation and generalisation. Inputs in L3_data.js (lectures/L3/build_interactive_L3.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, nf, pct, quantile, $, AX, layout, ttl, plot, out, hline, vline, bindSlider, bindToggles, Player, range, safe } = window.NNI;
const P = (id, input, vals, ms) => new Player(id, $(input), vals, ms);
const COL = { train: '#2a78d6', valid: '#eb6834', test: '#6250d6', logit: '#1baf7a', mlp: '#eb6834', xgb: '#eda100' };
const med = a => quantile(a, 0.5);
const big = x => (x < 1e-9 ? '0' : x >= 1000 ? nf(x, 0) : x >= 100 ? nf(x, 1) : x >= 1 ? nf(x, 3) : x >= 0.001 ? nf(x, 4) : x.toExponential(1).replace('.', LANG === 'ro' ? ',' : '.'));

// ---------------------------------------------------------------- 1. overfitting on 600 German credit applicants
safe('overfitting', () => {
  const d = D.ov;
  const p = bindSlider('ov-p', v => nf(d.dps[v], 1), draw), w = bindSlider('ov-w', v => nf(d.wds[v], 1), draw), e = bindSlider('ov-e', v => v, draw);
  function draw() {
    const pp = d.dps[p()], ww = d.wds[w()], c = d.c[`${pp.toFixed(1)}|${ww.toFixed(1)}`], E = e();
    const ep = c.va.map((_, i) => i + 1), be = c.va.indexOf(Math.min(...c.va)) + 1;
    plot('ov-loss', [
      { x: ep, y: c.tr, mode: 'lines', name: T.train, line: { color: COL.train, width: 2 } },
      { x: ep, y: c.va, mode: 'lines', name: T.valid, line: { color: COL.valid, width: 2 } },
      { x: [E], y: [c.va[E - 1]], mode: 'markers', name: T.stop, marker: { color: C.ink, size: 11 } },
      { x: [null], y: [null], mode: 'lines', name: T.logit, line: { color: COL.logit, dash: 'dash' } }],
      layout({ title: ttl(T.lossTitle(nf(pp, 1), nf(ww, 1))), xaxis: AX(T.epoch), yaxis: AX(T.ce, { range: [0, 2.3] }),
        shapes: [hline(d.logit_va, COL.logit, 'dash'), vline(E, C.ink, 'dot'), vline(be, C.grey, 'dash')] }));
    plot('ov-auc', [{ x: ep, y: c.auc, mode: 'lines', name: T.vauc, line: { color: COL.test, width: 2 } },
      { x: [E], y: [c.auc[E - 1]], mode: 'markers', showlegend: false, marker: { color: C.ink, size: 11 } }],
      layout({ title: ttl(T.aucTitle), xaxis: AX(T.epoch), yaxis: AX(T.auc, { range: [0.6, 0.85] }), shapes: [hline(d.logit_auc, COL.logit, 'dash'), vline(E, C.ink, 'dot')] }));
    out('ov-out', [T.ov(E, nf(c.tr[E - 1], 4), nf(c.va[E - 1], 3), nf(c.auc[E - 1], 3), be, nf(c.va[be - 1], 3)), T.ov2(nf(d.logit_va, 3), nf(d.logit_auc, 3))]);
  }
  draw();
  P('ov-play', 'ov-e', range(1, 150, 4), 150);
});

// ---------------------------------------------------------------- 2. double descent with random ReLU features
safe('double-descent', () => {
  const d = D.dd, mt = d.te.map(med), mr = d.tr.map(med);
  const i = bindSlider('dd-p', v => nf(d.p[v], 0), draw), s = bindToggles('dd-s', draw);
  function draw() {
    const j = i(), pp = d.p[j];
    const tr = [{ x: d.p, y: mt, mode: 'lines+markers', name: T.ddTest, line: { color: COL.test, width: 2 }, marker: { size: 5 } },
      { x: d.p, y: mr.map(v => Math.max(v, 1e-2)), mode: 'lines+markers', name: T.ddTrain, line: { color: COL.train, width: 2 }, marker: { size: 5 } }];
    if (s().includes('draws')) tr.push({ x: d.te[j].map(() => pp), y: d.te[j], mode: 'markers', name: T.draws, marker: { color: COL.test, size: 7, opacity: 0.5, symbol: 'diamond' } });
    tr.push({ x: [pp], y: [mt[j]], mode: 'markers', showlegend: false, marker: { size: 16, color: 'rgba(0,0,0,0)', line: { color: C.ink, width: 2 } } });
    tr.push({ x: [null], y: [null], mode: 'lines', name: T.thr, line: { color: C.grey, dash: 'dash' } });
    plot('dd-chart', tr, layout({ xaxis: AX(T.p, { type: 'log' }), yaxis: AX(T.mse, { type: 'log', range: [-2, 7], exponentformat: 'power' }),
      shapes: [vline(d.n, C.grey, 'dash')] }));
    out('dd-out', [T.dd(nf(pp, 0), big(mt[j]), big(mr[j]), big(Math.min(...d.te[j])), big(Math.max(...d.te[j])))]);
  }
  draw();
  P('dd-play', 'dd-p', range(0, d.p.length - 1, 1), 700);
});

// ---------------------------------------------------------------- 3. early stopping with patience, card fraud
safe('early-stopping', () => {
  const d = D.es, ep = d.va.map((_, i) => i + 1);
  const p = bindSlider('es-p', v => v, draw);
  function draw() {
    const pat = p();
    let best = Infinity, bad = 0, kept = 1, stop = null;
    for (let i = 0; i < d.va.length; i++) {
      if (d.va[i] < best - 1e-5) { best = d.va[i]; bad = 0; kept = i + 1; }
      else if (++bad >= pat) { stop = i + 1; break; }
    }
    const sh = [vline(kept, C.ink, 'dash')];
    if (stop) sh.push({ type: 'rect', xref: 'x', yref: 'paper', x0: kept, x1: stop, y0: 0, y1: 1, fillcolor: 'rgba(138,148,166,0.18)', line: { width: 0 } }, vline(stop, '#e34948', 'solid'));
    plot('es-loss', [{ x: ep, y: d.tr, mode: 'lines', name: T.train, line: { color: COL.train, width: 2 } },
      { x: ep, y: d.va, mode: 'lines', name: T.valid, line: { color: COL.valid, width: 2 } },
      { x: [null], y: [null], mode: 'lines', name: T.kept, line: { color: C.ink, dash: 'dash' } },
      { x: [null], y: [null], mode: 'lines', name: T.stopAt, line: { color: '#e34948' } }],
      layout({ title: ttl(T.esLoss), xaxis: AX(T.epoch), yaxis: AX(T.ce, { type: 'log', exponentformat: 'power' }), shapes: sh }));
    plot('es-ap', [{ x: ep, y: d.ap, mode: 'lines', name: T.ap, line: { color: COL.test, width: 2 } },
      { x: [kept], y: [d.ap[kept - 1]], mode: 'markers', showlegend: false, marker: { color: C.ink, size: 11 } }],
      layout({ title: ttl(T.esAp), xaxis: AX(T.epoch), yaxis: AX(T.ap, { range: [0.4, 0.85] }), showlegend: false, shapes: sh }));
    out('es-out', [stop ? T.es(pat, stop, kept, nf(d.va[kept - 1], 4), nf(d.ap[kept - 1], 3), nf(d.ap[d.ap.length - 1], 3)) : T.esNo(pat)]);
  }
  draw();
  P('es-play', 'es-p', range(1, 60, 1), 250);
});

// ---------------------------------------------------------------- 4. how noisy is one hold-out AUC
safe('holdout-noise', () => {
  const sh = D.shares;
  const s = bindSlider('ho-s', v => pct(sh[v], 0), draw);
  function draw() {
    const share = sh[s()], a = D.ho[String(share)], m = a.reduce((x, y) => x + y, 0) / a.length;
    const sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)), p5 = quantile(a, 0.05), p95 = quantile(a, 0.95);
    plot('ho-chart', [{ x: a, type: 'histogram', xbins: { start: 0.6, end: 0.92, size: 0.005 }, marker: { color: '#1baf7a' }, opacity: 0.85, name: 'AUC' },
      { x: [null], y: [null], mode: 'lines', name: T.mean, line: { color: C.ink } }, { x: [null], y: [null], mode: 'lines', name: T.p5, line: { color: C.ink, dash: 'dash' } }],
      layout({ title: ttl(T.hoTitle(pct(share, 0))), xaxis: AX(T.hoX, { range: [0.6, 0.92] }), yaxis: AX(T.hoY), bargap: 0.05,
        shapes: [vline(m, C.ink), vline(p5, C.ink, 'dash'), vline(p95, C.ink, 'dash')] }));
    out('ho-out', [T.ho(pct(share, 0), Math.round(1000 * share), nf(m, 3), nf(sd, 3), nf(Math.min(...a), 3), nf(Math.max(...a), 3), nf(p5, 3), nf(p95, 3))]);
  }
  draw();
  P('ho-play', 'ho-s', [0, 1, 2, 3], 1000);
});

// ---------------------------------------------------------------- 5. cost of fraud alerts
safe('fraud-cost', () => {
  const d = D.fc, K = ['logit', 'mlp', 'xgb'];
  const r = bindSlider('fc-r', v => nf(v, 0), draw), a = bindSlider('fc-a', v => nf(10 ** v, 0), draw), on = bindToggles('fc-m', draw);
  function draw() {
    const R = r(), A = 10 ** a(), tr = [], lines = [];
    on().forEach(k => {
      const rows = d[k], x = rows.map(q => q[0]), c = rows.map(q => R * q[0] + q[1]);
      const j = c.indexOf(Math.min(...c));
      // cut-off closest to the chosen number of alerts (on the log scale)
      let q = 0;
      x.forEach((v, i) => { if (Math.abs(Math.log(v) - Math.log(A)) < Math.abs(Math.log(x[q]) - Math.log(A))) q = i; });
      tr.push({ x, y: c, mode: 'lines', name: T.models[k], line: { color: COL[k], width: 2 }, hovertemplate: '%{x:.0f}: %{y:.0f} EUR<extra></extra>' });
      tr.push({ x: [x[j]], y: [c[j]], mode: 'markers', showlegend: false, marker: { color: COL[k], size: 9, symbol: 'diamond' }, hoverinfo: 'skip' });
      tr.push({ x: [x[q]], y: [c[q]], mode: 'markers', showlegend: false, marker: { color: COL[k], size: 12, line: { color: C.ink, width: 1.5 } }, hoverinfo: 'skip' });
      lines.push(T.fc(T.models[k], nf(x[q], 0), nf(c[q], 0), rows[q][2], nf(c[j], 0), nf(x[j], 0)));
    });
    tr.push({ x: [null], y: [null], mode: 'lines', name: T.noModel, line: { color: C.grey, dash: 'dot' } });
    lines.push(T.fc0(nf(d.total, 0)));
    plot('fc-chart', tr, layout({ title: ttl(T.fcTitle(nf(R, 0))), xaxis: AX(T.alerts, { type: 'log' }), yaxis: AX(T.cost, { rangemode: 'tozero' }),
      shapes: [hline(d.total, C.grey, 'dot'), vline(A, C.navy, 'dash')] }));
    out('fc-out', lines);
  }
  draw();
  P('fc-play', 'fc-r', range(1, 50, 1), 200);
});
})();
