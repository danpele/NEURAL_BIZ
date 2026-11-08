// Lecture 6 interactive page: NLP foundations, text representation and word embeddings. Inputs in L6_data.js
// (lectures/L6/build_interactive_L6.py), readouts in L6_text_*.js, helpers in common.js (window.NNI).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, SERIES, nf, pct, $, esc, AX, layout, ttl, plot, out, vline, bindSlider, bindSelect, bindToggles, bindRadios, safe } = window.NNI;
const CLS = [C.red, C.grey, C.green];            // negative, neutral, positive

// tokeniser of analysis_L6.py: lower case, links and @mentions dropped, every number -> <num>
function tok(s) {
  s = String(s).toLowerCase().replace(/https?:\/\/\S+|@\w+/g, ' ');
  return (s.match(/[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*/g) || []).map(t => (/^\d/.test(t) ? '<num>' : t));
}

// ---------------------------------------------------------------- 1. Zipf's law and the size of the vocabulary
safe('zipf', () => {
  const f = D.zf.freq, n = f.length, tot = f.reduce((a, b) => a + b, 0);
  const cum = new Float64Array(n); let s = 0;
  for (let i = 0; i < n; i++) { s += f[i]; cum[i] = s / tot; }
  // ranks shown: all of the first 300, then about 400 log-spaced ranks (the curve has 20,130 points)
  const idx = []; for (let i = 0; i < Math.min(300, n); i++) idx.push(i);
  for (let k = 0; k <= 400; k++) { const i = Math.round(Math.exp(Math.log(300) + k / 400 * (Math.log(n) - Math.log(300)))) - 1; if (i > idx[idx.length - 1] && i < n) idx.push(i); }
  const m = bindSlider('zf-m', v => v, draw), sc = bindRadios('zf-s', draw), fit = bindToggles('zf-f', draw);
  function draw() {
    const mm = m(), lg = sc() === 'log', V = f.filter(c => c >= mm).length;
    const inI = idx.filter(i => i < V), outI = idx.filter(i => i >= V - 1);
    const tr = [
      { x: inI.map(i => i + 1), y: inI.map(i => f[i]), mode: 'lines', name: T.inVoc, line: { color: C.RO, width: 2.2 },
        hovertemplate: '%{x}: %{y}<extra></extra>' },
      { x: outI.map(i => i + 1), y: outI.map(i => f[i]), mode: 'lines', name: T.outVoc, line: { color: C.grey, width: 2.2, dash: 'dot' },
        hovertemplate: '%{x}: %{y}<extra></extra>' }];
    if (fit().length) {
      const xs = [10, 10000];
      tr.push({ x: xs, y: xs.map(r => Math.exp(D.zf.icpt) * r ** D.zf.slope), mode: 'lines', name: T.fit(nf(D.zf.slope, 2)),
        line: { color: C.DE_LU, width: 1.5, dash: 'dash' }, hoverinfo: 'skip' });
    }
    plot('zf-chart', tr, layout({ title: ttl(T.zfTitle), xaxis: AX(T.rank, lg ? { type: 'log', exponentformat: 'none' } : { range: [0, n] }),
      yaxis: AX(T.freq, lg ? { type: 'log', exponentformat: 'none' } : {}), shapes: [vline(V, C.navy, 'dash')] }));
    plot('zf-cov', [{ x: idx.map(i => i + 1), y: idx.map(i => 100 * cum[i]), mode: 'lines', line: { color: C.RO, width: 2 }, showlegend: false,
      hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    { x: [V], y: [100 * cum[V - 1]], mode: 'markers', marker: { color: C.navy, size: 10 }, showlegend: false, hoverinfo: 'skip' }],
    layout({ title: ttl(T.covTitle), xaxis: AX(T.vocSize, { type: lg ? 'log' : 'linear', exponentformat: 'none' }), yaxis: AX(T.cover, { range: [0, 101] }) }));
    out('zf-out', [T.zf(mm, V, nf(n, 0), pct(cum[V - 1], 1), pct(1 - cum[V - 1], 1)), T.zf2(esc(D.zf.top.join(', ')), pct(cum[9], 1))]);
  }
  draw();
});

// ---------------------------------------------------------------- 2. bag-of-words logistic regression on one sentence
safe('word-weights', () => {
  const voc = new Map(D.sc.voc.map((w, i) => [w, i])), B = D.sc.b, B0 = D.sc.b0, STOP = new Set(D.sc.stop);
  const ex = bindSelect('sc-ex', pick), sw = bindToggles('sc-sw', draw), box = $('sc-txt');
  let label = null;
  function pick() { const e = D.sc.ex[+ex()]; box.value = e.s; label = e.y; draw(); }
  box.addEventListener('change', () => { label = null; draw(); });
  box.addEventListener('keydown', e => { if (e.key === 'Enter') { label = null; draw(); } });
  function draw() {
    const all = tok(box.value), rm = sw().length > 0;
    const removed = rm ? all.filter(w => STOP.has(w)) : [];
    const t = rm ? all.filter(w => !STOP.has(w)) : all;
    const z = B0.slice();
    t.forEach(w => { const j = voc.get(w); if (j !== undefined) for (let k = 0; k < 3; k++) z[k] += B[k][j]; });
    const mx = Math.max(...z), e = z.map(v => Math.exp(v - mx)), se = e[0] + e[1] + e[2], p = e.map(v => v / se);
    const c = p.indexOf(Math.max(...p));
    const d = t.map(w => { const j = voc.get(w); return j === undefined ? null : B[2][j] - B[0][j]; });
    const yy = t.map((_, i) => i);
    $('sc-chart').style.height = Math.max(360, 16 * t.length + 90) + 'px';
    plot('sc-chart', [{ type: 'bar', orientation: 'h', y: yy, x: d.map(v => (v === null ? 0 : v)),
      marker: { color: d.map(v => (v === null ? C.band : v >= 0 ? C.green : C.red)) },
      customdata: t.map((w, i) => (d[i] === null ? T.notVoc : nf(d[i], 2))), hovertemplate: '%{customdata}<extra></extra>' }],
    layout({ title: ttl(T.scTitle), showlegend: false,
      xaxis: AX(T.contrib), yaxis: AX('', { tickvals: yy, ticktext: t.map((w, i) => (d[i] === null ? w + ' ·' : w)), autorange: 'reversed', tickfont: { size: 10 } }),
      margin: { l: 90, r: 12, t: 30, b: 44 } }));
    plot('sc-prob', [{ type: 'bar', x: T.cls, y: p.map(v => 100 * v), marker: { color: CLS }, text: p.map(v => pct(v, 1)), textposition: 'outside', cliponaxis: false,
      hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }],
    layout({ title: ttl(T.probTitle), showlegend: false, yaxis: AX(T.prob, { range: [0, 110] }), xaxis: AX('') }));
    if (!t.length) { out('sc-out', [T.empty]); return; }
    const oov = [...new Set(t.filter(w => !voc.has(w)))];
    const lines = [T.sc(T.cls[c], pct(p[c], 1), label === null ? null : T.cls[label]), T.sc2(t.length, t.length - t.filter(w => !voc.has(w)).length, esc(oov.join(', ')))];
    if (rm) lines.push(T.sc3(removed.length ? esc([...new Set(removed)].join(', ')) : '–'));
    out('sc-out', lines);
  }
  pick();
});

// ---------------------------------------------------------------- 3. nearest neighbours, word2vec against GloVe
safe('neighbours', () => {
  const w = bindSelect('nb-w', draw), k = bindSlider('nb-k', v => v, draw);
  function one(id, model, other, title) {
    const [ws, ss] = D.nb.nb[model][w()], kk = k(), oth = new Set(D.nb.nb[other][w()][0].slice(0, kk));
    const W = ws.slice(0, kk), S = ss.slice(0, kk), col = W.map(x => (oth.has(x) ? C.RO : C.band));
    plot(id, [{ type: 'bar', orientation: 'h', y: W, x: S, marker: { color: col }, text: S.map(v => nf(v, 2)), textposition: 'outside',
      hovertemplate: '%{y}: %{x:.3f}<extra></extra>', showlegend: false },
    { type: 'bar', x: [null], y: [null], marker: { color: C.RO }, name: T.shared }, { type: 'bar', x: [null], y: [null], marker: { color: C.band }, name: T.own }],
    layout({ title: ttl(title), xaxis: AX(T.cos, { range: [0, 1.05] }), yaxis: AX('', { autorange: 'reversed', type: 'category' }), barmode: 'overlay',
      margin: { l: 96, r: 12, t: 30, b: 40 } }));
    return W;
  }
  function draw() {
    const a = one('nb-w2v', 'w2v', 'glove', 'word2vec (SGNS)'), b = one('nb-glove', 'glove', 'w2v', 'GloVe'), sb = new Set(b);
    const sh = a.filter(x => sb.has(x));
    out('nb-out', [T.nb(esc(w()), D.nb.freq[w()], k(), sh.length, sh.length ? esc(sh.join(', ')) : T.nbNone)]);
  }
  draw();
});

// ---------------------------------------------------------------- 4. 2-D map of the embedding space: PCA recomputed on the chosen groups
safe('word-map', () => {
  const g = bindToggles('pc-g', draw), m = bindRadios('pc-m', draw);
  const GC = { monetary: SERIES[0], energy: SERIES[1], banking: SERIES[2], digital: SERIES[6] };
  // leading eigenvectors of the n x n Gram matrix of the centred vectors (power iteration with deflation)
  function pca2(Z) {
    const n = Z.length, d = Z[0].length, mu = new Array(d).fill(0);
    Z.forEach(z => z.forEach((v, j) => { mu[j] += v / n; }));
    const Zc = Z.map(z => z.map((v, j) => v - mu[j]));
    const G = Zc.map(a => Zc.map(b => a.reduce((s, v, j) => s + v * b[j], 0)));
    const tr = G.reduce((s, r, i) => s + r[i], 0), vecs = [], lams = [];
    for (let c = 0; c < Math.min(2, n - 1); c++) {
      let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1 + c)), lam = 0;
      for (let it = 0; it < 500; it++) {
        let u = G.map(r => r.reduce((s, x, j) => s + x * v[j], 0));
        vecs.forEach((q, qi) => { const dt = u.reduce((s, x, j) => s + x * q[j], 0); u = u.map((x, j) => x - dt * q[j]); });
        const nn = Math.hypot(...u); if (nn < 1e-12) break;
        v = u.map(x => x / nn); lam = nn;
      }
      vecs.push(v); lams.push(lam);
    }
    return { P: Z.map((_, i) => vecs.map((v, c) => v[i] * Math.sqrt(lams[c]))), share: lams.reduce((a, b) => a + b, 0) / tr };
  }
  function draw() {
    const gs = g(), M = m(), words = [], grp = [];
    gs.forEach(k => D.pc.groups[k].forEach(w => { words.push(w); grp.push(k); }));
    if (words.length < 3) { Plotly.purge('pc-chart'); plot('pc-chart', [], layout({ title: ttl(T.pcFew) })); out('pc-out', [T.pcFew]); return; }
    const r = pca2(words.map(w => D.pc[M][w]));
    const tr = gs.map(k => {
      const ii = grp.map((x, i) => (x === k ? i : -1)).filter(i => i >= 0);
      return { x: ii.map(i => r.P[i][0]), y: ii.map(i => (r.P[i][1] === undefined ? 0 : r.P[i][1])), text: ii.map(i => words[i]), mode: 'markers+text',
        textposition: 'top right', textfont: { size: 11, color: GC[k] }, marker: { color: GC[k], size: 8 }, name: D.gname[LANG][k],
        hovertemplate: '%{text}<extra></extra>' };
    });
    plot('pc-chart', tr, layout({ xaxis: AX(T.pc1, { zeroline: false }), yaxis: AX(T.pc2, { zeroline: false }), margin: { l: 56, r: 30, t: 20, b: 40 } }));
    out('pc-out', [T.pc(words.length, pct(r.share, 1), M === 'w2v' ? 'word2vec' : 'GloVe')]);
  }
  draw();
});

// ---------------------------------------------------------------- 5. accuracy and confusion by annotator agreement
safe('agreement', () => {
  const md = bindSelect('ag-m', draw), a = bindRadios('ag-a', draw), LV = ['all', '50', '66', '75', '100'];
  function draw() {
    const k = md(), aa = a(), c = D.ag.cm[k][aa], M = c.M, n = M.flat().reduce((x, y) => x + y, 0);
    const rows = M.map(r => r.reduce((x, y) => x + y, 0));
    plot('ag-cm', [{ type: 'heatmap', z: M.map((r, i) => r.map(v => (rows[i] ? 100 * v / rows[i] : 0))), x: T.cls, y: T.cls,
      text: M.map((r, i) => r.map(v => `${v}<br>${rows[i] ? pct(v / rows[i], 0) : '–'}`)), texttemplate: '%{text}', showscale: false,
      colorscale: [[0, '#ffffff'], [1, C.blue]], zmin: 0, zmax: 100, hovertemplate: '%{y} → %{x}: %{text}<extra></extra>' }],
    layout({ title: ttl(T.agTitle), xaxis: AX(T.predC, { side: 'bottom' }), yaxis: AX(T.trueC, { autorange: 'reversed' }), margin: { l: 80, r: 12, t: 30, b: 44 } }));
    plot('ag-acc', [{ type: 'bar', x: LV.map(l => T.lvl[l]), y: LV.map(l => D.ag.cm[k][l].acc), name: D.ag.mname[LANG][k],
      marker: { color: LV.map(l => (l === aa ? C.RO : C.band)) }, texttemplate: '%{y:.1f}', textposition: 'inside', insidetextanchor: 'end',
      hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    { type: 'scatter', mode: 'markers', x: LV.map(l => T.lvl[l]), y: LV.map(l => D.ag.cm.majority[l].acc), name: T.majority,
      marker: { color: C.navy, symbol: 'line-ew-open', size: 26, line: { width: 2 } }, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }],
    layout({ title: ttl(T.accTitle), xaxis: AX(T.agX, { type: 'category' }), yaxis: AX(T.acc, { range: [0, 105] }) }));
    const rec = M.map((r, i) => (rows[i] ? pct(r[i] / rows[i], 1) : '–'));
    out('ag-out', [T.ag(D.ag.mname[LANG][k], aa, n, pct(c.acc / 100, 1), pct(D.ag.cm.majority[aa].acc / 100, 1)), T.ag2(...rec)]);
  }
  draw();
});

// ---------------------------------------------------------------- 6. topics of finance tweets: where each topic goes
safe('topics', () => {
  const tsel = bindSelect('tp-t', draw), md = bindRadios('tp-m', draw);
  const NM = LANG === 'ro' ? D.tp.ro : D.tp.names;
  $('tp-row').style.height = '480px'; $('tp-f1').style.height = '480px';
  function stats(M) {
    const K = M.length, rs = M.map(r => r.reduce((a, b) => a + b, 0)), cs = M[0].map((_, j) => M.reduce((s, r) => s + r[j], 0));
    const p = M.map((r, i) => (cs[i] ? r[i] / cs[i] : 0)), r = M.map((row, i) => (rs[i] ? row[i] / rs[i] : 0));
    const f = p.map((v, i) => (v + r[i] > 0 ? 2 * v * r[i] / (v + r[i]) : 0));
    const acc = M.reduce((s, row, i) => s + row[i], 0) / rs.reduce((a, b) => a + b, 0);
    return { p, r, f, rs, macro: f.reduce((a, b) => a + b, 0) / K, acc };
  }
  function draw() {
    const t = +tsel(), M = D.tp[md()], st = stats(M), row = M[t], n = st.rs[t];
    const order = row.map((v, j) => [v, j]).filter(x => x[0] > 0).sort((a, b) => b[0] - a[0]);
    plot('tp-row', [{ type: 'bar', orientation: 'h', y: order.map(x => NM[x[1]]), x: order.map(x => 100 * x[0] / n),
      marker: { color: order.map(x => (x[1] === t ? C.green : C.amber)) }, text: order.map(x => String(x[0])), textposition: 'outside',
      hovertemplate: '%{y}: %{x:.1f}%<extra></extra>', showlegend: false }],
    layout({ title: ttl(T.rowTitle(NM[t])), xaxis: AX(T.share, { range: [0, 110] }), yaxis: AX('', { autorange: 'reversed', type: 'category', tickfont: { size: 10 } }),
      margin: { l: 150, r: 12, t: 30, b: 44 } }));
    const ord = st.f.map((v, j) => [v, j]).sort((a, b) => a[0] - b[0]);
    plot('tp-f1', [{ type: 'bar', orientation: 'h', y: ord.map(x => NM[x[1]]), x: ord.map(x => 100 * x[0]),
      marker: { color: ord.map(x => (x[1] === t ? C.RO : C.band)) }, hovertemplate: '%{y}: %{x:.1f}%<extra></extra>', showlegend: false }],
    layout({ title: ttl(T.f1Title), xaxis: AX(T.f1, { range: [0, 100] }), yaxis: AX('', { type: 'category', tickfont: { size: 9 } }), margin: { l: 150, r: 12, t: 30, b: 44 } }));
    out('tp-out', [T.tp(esc(NM[t]), n, pct(st.p[t], 1), pct(st.r[t], 1), pct(st.f[t], 1)), T.tp2(pct(st.macro, 2), pct(st.acc, 2))]);
  }
  draw();
});
})();
