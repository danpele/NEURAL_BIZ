// Lecture 9 interactive page: graph neural networks on the Elliptic Bitcoin graph. Inputs in L9_data.js (lectures/L9/build_interactive_L9.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { C, SERIES, mean, sdev, nf, pct, $, AX, layout, ttl, plot, out, vline, hline,
  bindSlider, bindSelect, bindToggles, bindRadios, Player, range, safe } = window.NNI;
const players = [];
const P = (id, input, vals, ms) => players.push(new Player(id, $(input), vals, ms));
const ORDER = ['lr_local', 'lr_all', 'rf', 'mlp', 'gcn', 'sage', 'gat', 'rf_emb'];
const MC = { lr_local: C.grey, lr_all: SERIES[3], rf: SERIES[0], mlp: SERIES[4], gcn: SERIES[1], sage: SERIES[2], gat: SERIES[6], rf_emb: SERIES[5] };
const ILL = C.red, LIC = C.green, UNK = C.grey;
const testRect = (x0, x1) => ({ type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, fillcolor: C.band, opacity: 0.25, line: { width: 0 }, layer: 'below' });

// ---------------------------------------------------------------- 1. homophily by time step
safe('homophily', () => {
  const H = D.hm, steps = Array.from({ length: 49 }, (_, i) => i + 1);
  const s = bindSlider('hm-s', v => v, draw), m = bindSelect('hm-m', draw);
  function draw() {
    const i = s() - 1, mode = m();
    const bar = mode === 'node'
      ? { x: T.hmCat, y: [H.h_ill[i], H.h_lic[i], H.base[i]], marker: { color: [ILL, LIC, C.grey] } }
      : { x: T.hmCatE, y: [H.edge[i], H.edge_rand[i]], marker: { color: [C.blue, C.grey] } };
    plot('hm-bar', [Object.assign(bar, { type: 'bar', hovertemplate: '%{x}: %{y:.1%}<extra></extra>' })],
      layout({ title: ttl(T.hmBarT(s())), yaxis: AX(T.share, { range: [0, 1], tickformat: '.0%' }), xaxis: AX(''), showlegend: false }));
    const tr = mode === 'node'
      ? [{ x: steps, y: H.h_ill, name: T.hmIll, mode: 'lines+markers', line: { color: ILL, width: 2 }, marker: { size: 4 } },
        { x: steps, y: H.h_lic, name: T.hmLic, mode: 'lines+markers', line: { color: LIC, width: 2 }, marker: { size: 4 } },
        { x: steps, y: H.base, name: T.hmBase, mode: 'lines', line: { color: C.grey, width: 1.5, dash: 'dot' } }]
      : [{ x: steps, y: H.edge, name: T.hmEdge, mode: 'lines+markers', line: { color: C.blue, width: 2 }, marker: { size: 4 } },
        { x: steps, y: H.edge_rand, name: T.hmRand, mode: 'lines', line: { color: C.grey, width: 1.5, dash: 'dot' } }];
    tr.forEach(t => { t.hovertemplate = '%{x}: %{y:.1%}<extra>' + t.name + '</extra>'; });
    plot('hm-line', tr, layout({ title: ttl(T.hmLineT(mode)), xaxis: AX(T.step, { range: [0.5, 49.5] }), yaxis: AX(T.share, { range: [0, 1.02], tickformat: '.0%' }),
      shapes: [testRect(34.5, 49.5), vline(s(), C.navy, 'dash')] }));
    const A = H.all, lines = [];
    if (mode === 'node') lines.push(H.h_ill[i] === null ? T.hmNone(s()) : T.hm(s(), pct(H.h_ill[i]), H.h_lic[i] === null ? '–' : pct(H.h_lic[i]), pct(H.base[i]), H.n_ill[i], H.n_lic[i]));
    else lines.push(T.hmE(s(), H.edge[i] === null ? '–' : pct(H.edge[i]), pct(H.edge_rand[i])));
    lines.push(T.hmAll(pct(A.h_ill), pct(A.h_lic), pct(A.edge), pct(A.edge_rand)));
    out('hm-out', lines);
  }
  draw();
  P('hm-play', 'hm-s', range(1, 49, 1), 450);
});

// ---------------------------------------------------------------- 2. message passing on real neighbourhoods
safe('message-passing', () => {
  const g = bindSelect('mp-g', draw), x = bindSelect('mp-x', draw), k = bindSlider('mp-k', v => v, draw);
  const R = D.rfield;
  function draw() {
    const G = D.mp[+g()], sig = x(), kk = k(), vals = G[sig][kk];
    const ex = [], ey = [];
    G.e.forEach(([a, b]) => { ex.push(G.x[a], G.x[b], null); ey.push(G.y[a], G.y[b], null); });
    const amp = sig === 'lab' ? 1 : Math.max(...G.f53[0].map(Math.abs), 0.5);
    const reached = G.hop.map(h => h <= kk);
    const node = (idx, isNull) => ({
      x: idx.map(i => G.x[i]), y: idx.map(i => G.y[i]), mode: 'markers', hoverinfo: 'text', showlegend: false,
      text: idx.map(i => `${T.hop(G.hop[i])} · ${T.cls[String(G.cls[i])]} · ${vals[i] === null ? T.nolab : nf(vals[i], 3)}`),
      marker: Object.assign({ size: idx.map(i => (G.hop[i] === 0 ? 22 : 14)), opacity: idx.map(i => (reached[i] ? 1 : 0.35)),
        line: { width: 3, color: idx.map(i => (G.cls[i] === 1 ? ILL : G.cls[i] === 0 ? LIC : UNK)) } },
      isNull ? { color: '#ffffff' } : { color: idx.map(i => vals[i]), cmin: sig === 'lab' ? 0 : -amp, cmax: amp,
        colorscale: sig === 'lab' ? [[0, '#EEF3FA'], [0.5, '#eda100'], [1, '#DC3545']] : 'RdBu', reversescale: sig !== 'lab',
        colorbar: { thickness: 10, len: 0.8, tickfont: { size: 10 } } })
    });
    const all = G.x.map((_, i) => i), nn = all.filter(i => vals[i] !== null), nul = all.filter(i => vals[i] === null);
    const leg = [[ILL, T.cls['1']], [LIC, T.cls['0']], [UNK, T.cls['-1']]].map(([c, nm]) => ({ x: [null], y: [null], mode: 'markers', name: nm,
      marker: { size: 11, color: '#fff', line: { width: 3, color: c } } }));
    plot('mp-graph', [{ x: ex, y: ey, mode: 'lines', line: { color: C.band, width: 1.2 }, hoverinfo: 'skip', showlegend: false },
      node(nul, true), node(nn, false), ...leg],
    layout({ title: ttl(T.mpT(kk, sig)), xaxis: { visible: false }, yaxis: { visible: false, scaleanchor: 'x' }, margin: { l: 10, r: 10, t: 34, b: 30 } }));
    const ks = [1, 2, 3, 4, 5, 6];
    plot('mp-hops', [
      { x: ks, y: G.khop, type: 'bar', name: T.thisTx, marker: { color: ks.map(q => (q <= kk ? C.navy : C.band)) }, hovertemplate: 'k = %{x}: %{y:,}<extra></extra>' },
      { x: R.k, y: R.median, mode: 'lines+markers', name: T.med, line: { color: SERIES[1], width: 2 } },
      { x: R.k, y: R.p90, mode: 'lines+markers', name: T.p90, line: { color: SERIES[1], width: 1.5, dash: 'dash' } }],
    layout({ title: ttl(T.mpHopsT), xaxis: AX(T.kk, { dtick: 1 }), yaxis: AX(T.nodes, { type: 'log' }) }));
    const nv = vals.filter(v => v !== null);
    out('mp-out', [kk === 0 ? T.mp0(nv.length > 1 ? nf(sdev(nv), 3) : '–') : T.mp(kk, G.khop[kk - 1], vals[0] === null ? T.nolab : nf(vals[0], 3), nv.length > 1 ? nf(sdev(nv), 3) : '–'), T.mp2]);
  }
  draw();
  P('mp-play', 'mp-k', range(0, 6, 1), 900);
});

// ---------------------------------------------------------------- 3. over-smoothing with depth
safe('over-smoothing', () => {
  const L = bindSlider('os-L', v => v, draw), r = bindRadios('os-s', draw);
  const lay = [1, 2, 3, 4, 5, 6];
  const run = key => (key === 'mean'
    ? ['ap', 'f1', 'cos'].reduce((o, c) => (o[c] = lay.map((_, i) => (D.os['1'][c][i] + D.os['2'][c][i]) / 2), o), {})
    : D.os[key]);
  function draw() {
    const l = L(), R = run(r()), i = l - 1, prop = D.os.prop, lp = prop.map((_, j) => j);
    plot('os-perf', [
      { x: lay, y: R.ap, mode: 'lines+markers', name: T.ap, line: { color: C.blue, width: 2 } },
      { x: lay, y: R.f1, mode: 'lines+markers', name: T.f1, line: { color: SERIES[1], width: 2, dash: 'dash' } },
      { x: [l, l], y: [R.ap[i], R.f1[i]], mode: 'markers', showlegend: false, marker: { size: 13, color: [C.blue, SERIES[1]], line: { color: C.navy, width: 2 } }, hoverinfo: 'skip' }],
    layout({ title: ttl(T.osPerfT), xaxis: AX(T.layers, { dtick: 1 }), yaxis: AX('', { range: [0.25, 0.6] }), shapes: [vline(l, C.grey, 'dot')] }));
    plot('os-cos', [
      { x: lay, y: R.cos, mode: 'lines+markers', name: T.cosG, line: { color: C.blue, width: 2 } },
      { x: lp, y: prop, mode: 'lines+markers', name: T.cosP, line: { color: C.grey, width: 2, dash: 'dot' } },
      { x: [l, l], y: [R.cos[i], prop[l]], mode: 'markers', showlegend: false, marker: { size: 13, color: [C.blue, C.grey], line: { color: C.navy, width: 2 } }, hoverinfo: 'skip' }],
    layout({ title: ttl(T.osCosT), xaxis: AX(T.layers, { dtick: 1 }), yaxis: AX('', { range: [0, 1] }), shapes: [vline(l, C.grey, 'dot')] }));
    out('os-out', [T.os(l, nf(R.ap[i], 3), nf(R.f1[i], 3), nf(R.cos[i], 2)), T.os2(l, nf(prop[0], 2), nf(prop[l], 2))]);
  }
  draw();
  P('os-play', 'os-L', range(1, 6, 1), 900);
});

// ---------------------------------------------------------------- 4. precision-recall curves and the alert threshold
safe('pr-curves', () => {
  const ms = bindToggles('pr-m', draw), t = bindSlider('pr-t', v => nf(v, 2), draw);
  const R = D.pr;
  function draw() {
    const th = t(), j = R.thr.findIndex(v => Math.abs(v - th) < 1e-6), tr = [], lines = [];
    ms().forEach(m => {
      const M = R.m[m], tp = M.tp[j], fp = M.fp[j], p = tp + fp ? tp / (tp + fp) : null, rc = tp / R.npos, f = tp ? 2 * tp / (2 * tp + fp + R.npos - tp) : 0;
      tr.push({ x: M.R, y: M.P, mode: 'lines', name: T.models[m], line: { color: MC[m], width: 2 }, hovertemplate: 'recall %{x:.2f}, precision %{y:.2f}<extra>' + T.models[m] + '</extra>' });
      if (p !== null) tr.push({ x: [rc], y: [p], mode: 'markers', showlegend: false, marker: { size: 11, color: MC[m], line: { color: C.navy, width: 1.5 } }, hoverinfo: 'skip' });
      lines.push(T.pr(T.models[m], nf(th, 2), p === null ? '–' : pct(p), pct(rc), nf(f, 3), nf(M.ap, 3), tp + fp));
    });
    plot('pr-chart', tr, layout({ title: ttl(T.prT(nf(th, 2))), xaxis: AX(T.recall, { range: [0, 1.01] }), yaxis: AX(T.precision, { range: [0, 1.02] }),
      shapes: [hline(R.npos / R.n, C.grey, 'dot')],
      annotations: [{ x: 0.99, y: R.npos / R.n, xref: 'x', yref: 'y', xanchor: 'right', yanchor: 'bottom', showarrow: false, text: T.base + ' ' + pct(R.npos / R.n), font: { size: 10, color: C.grey } }] }));
    out('pr-out', lines.length ? lines : ['–']);
  }
  draw();
  P('pr-play', 'pr-t', range(0.05, 0.95, 0.05), 500);
});

// ---------------------------------------------------------------- 5. F1 by test step: the dark-market shutdown
safe('drift', () => {
  const ms = bindToggles('fd-m', draw), t = bindSlider('fd-t', v => nf(v, 2), draw);
  const F = D.f1, S = F.steps;
  function draw() {
    const th = t(), j = F.thr.findIndex(v => Math.abs(v - th) < 1e-6), tr = [], lines = [];
    tr.push({ x: S, y: F.npos, type: 'bar', yaxis: 'y2', name: T.nIll, marker: { color: C.band }, opacity: 0.6, hovertemplate: '%{x}: %{y}<extra></extra>' });
    ms().forEach(m => {
      const tp = F.m[m].tp[j], fp = F.m[m].fp[j];
      const f1 = S.map((_, i) => { const d = 2 * tp[i] + fp[i] + F.npos[i] - tp[i]; return d ? 2 * tp[i] / d : 0; });
      tr.push({ x: S, y: f1, mode: 'lines+markers', name: T.models[m], line: { color: MC[m], width: 2 }, marker: { size: 5 }, hovertemplate: '%{x}: F1 %{y:.3f}<extra>' + T.models[m] + '</extra>' });
      lines.push(T.fd(T.models[m], nf(mean(f1.filter((_, i) => S[i] < 43)), 3), nf(mean(f1.filter((_, i) => S[i] >= 43)), 3)));
    });
    plot('fd-chart', tr, layout({ title: ttl(T.fdT(nf(th, 2))), xaxis: AX(T.step, { dtick: 1 }), yaxis: AX('F1', { range: [0, 1.02] }),
      yaxis2: AX(T.nIll, { overlaying: 'y', side: 'right', showgrid: false, rangemode: 'tozero' }), barmode: 'overlay',
      shapes: [vline(42.5, C.red, 'dash')],
      annotations: [{ x: 42.6, y: 1, xref: 'x', yref: 'paper', xanchor: 'left', yanchor: 'top', showarrow: false, text: T.shutdown, font: { size: 11, color: C.red } }],
      margin: { l: 56, r: 56, t: 30, b: 40 } }));
    out('fd-out', lines.length ? lines : ['–']);
  }
  draw();
  P('fd-play', 'fd-t', range(0.05, 0.95, 0.05), 600);
});

// ---------------------------------------------------------------- 6. alert budget
safe('alert-budget', () => {
  const kS = bindSlider('ab-k', v => nf(v, 0), draw), m = bindSelect('ab-m', draw);
  const A = D.ab, npos = D.pr.npos, S = D.f1.steps, base = ['lr_all', 'rf', 'gcn', 'rf_emb'];
  function draw() {
    const k = kS(), mm = m(), j = A.k.indexOf(k), show = ORDER.filter(x => base.includes(x) || x === mm);
    const tr = show.map(x => ({ x: A.k, y: A.m[x].tp.map(v => v / npos), mode: 'lines', name: T.models[x],
      line: { color: MC[x], width: x === mm ? 3.5 : 1.6 }, hovertemplate: '%{x}: %{y:.1%}<extra>' + T.models[x] + '</extra>' }));
    tr.push({ x: show.map(() => k), y: show.map(x => A.m[x].tp[j] / npos), mode: 'markers', showlegend: false, hoverinfo: 'skip',
      marker: { size: 9, color: show.map(x => MC[x]), line: { color: C.navy, width: 1.5 } } });
    plot('ab-curve', tr, layout({ title: ttl(T.abT), xaxis: AX(T.alerts), yaxis: AX(T.share, { range: [0, 1], tickformat: '.0%' }), shapes: [vline(k, C.navy, 'dash')] }));
    const st = A.m[mm].st[j];
    plot('ab-step', [{ x: S, y: st, type: 'bar', name: T.found, marker: { color: MC[mm] }, hovertemplate: '%{x}: %{y}<extra></extra>' },
      { x: S, y: S.map((_, i) => D.f1.npos[i] - st[i]), type: 'bar', name: T.missed, marker: { color: C.band }, hovertemplate: '%{x}: %{y}<extra></extra>' }],
    layout({ title: ttl(T.abStepT(T.models[mm], k)), barmode: 'stack', xaxis: AX(T.step, { dtick: 2 }), yaxis: AX(T.nIll),
      shapes: [vline(42.5, C.red, 'dash')] }));
    const tp = A.m[mm].tp[j], pre = st.filter((_, i) => S[i] < 43).reduce((a, b) => a + b, 0);
    out('ab-out', [T.ab(T.models[mm], k, pct(tp / npos), pct(tp / k), tp), T.ab2(nf(pre, 0), nf(tp - pre, 0))]);
  }
  draw();
  P('ab-play', 'ab-k', range(100, 3000, 100), 400);
});
})();
