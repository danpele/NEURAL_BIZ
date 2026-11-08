// Lecture 7 interactive page: attention and transformers. Inputs in L7_data.js (lectures/L7/build_interactive_L7.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, SERIES, mean, median, quantile, nf, pct, $, AX, layout, ttl, plot, out, vline, hline,
  bindSlider, bindSelect, bindToggles, Player, range, safe } = window.NNI;
const players = [];
const P = (id, input, vals, ms) => players.push(new Player(id, $(input), vals, ms));
const HEAT = [[0, '#ffffff'], [0.15, '#C5D2E8'], [0.5, '#2a78d6'], [1, '#1F2A44']];
const softmax = (s, div) => { const m = Math.max(...s); const e = s.map(x => Math.exp((x - m) / div)); const z = e.reduce((a, b) => a + b, 0); return e.map(x => x / z); };
const entN = w => -w.reduce((a, p) => a + (p > 1e-12 ? p * Math.log(p) : 0), 0) / Math.log(w.length);
const isSpecial = t => t === '[CLS]' || t === '[SEP]';

// ---------------------------------------------------------------- 1. context lengths of the three corpora
safe('context-lengths', () => {
  const c = bindSelect('cl-c', draw), w = bindSlider('cl-w', v => nf(2 ** v, 0), draw);
  const share = (a, n) => { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; a[m] <= n ? (lo = m + 1) : (hi = m); } return lo / a.length; };
  const grid = range(0, 4.7, 0.05).map(x => 10 ** x);
  function draw() {
    const cc = c(), W = 2 ** w(), a = D.cl[cc];
    // histogram on log10 bins of width 0.1
    const bins = {};
    a.forEach(x => { const b = Math.floor(Math.log10(x) * 10) / 10; bins[b] = (bins[b] || 0) + 1; });
    const bx = Object.keys(bins).map(Number).sort((p, q) => p - q);
    plot('cl-hist', [{ x: bx.map(b => 10 ** (b + 0.05)), y: bx.map(b => 100 * bins[b] / a.length), type: 'bar', width: bx.map(b => 10 ** (b + 0.1) - 10 ** b),
      marker: { color: C.RO }, hovertemplate: '%{x:.0f}: %{y:.1f}%<extra></extra>', name: T.corpus[cc] }],
    layout({ title: ttl(T.clHist(T.corpus[cc])), xaxis: AX(T.tokLog, { type: 'log' }), yaxis: AX(T.pctTexts), showlegend: false,
      shapes: [vline(W, C.red, 'dash')], annotations: [{ x: Math.log10(W), y: 1, yref: 'paper', text: T.window(nf(W, 0)), showarrow: false, xanchor: 'left', font: { color: C.red, size: 11 } }] }));
    const tr = ['tweets', 'phrasebank', 'speeches'].map((k, i) => ({ x: grid, y: grid.map(g => 100 * share(D.cl[k], g)), mode: 'lines', name: T.corpus[k],
      line: { color: SERIES[i], width: k === cc ? 3 : 1.5 }, hovertemplate: '%{x:.0f}: %{y:.1f}%<extra>' + T.corpus[k] + '</extra>' }));
    tr.push({ x: [W, W, W], y: ['tweets', 'phrasebank', 'speeches'].map(k => 100 * share(D.cl[k], W)), mode: 'markers', marker: { color: C.red, size: 9 }, showlegend: false, hoverinfo: 'skip' });
    plot('cl-cdf', tr, layout({ title: ttl(T.clCdf), xaxis: AX(T.tokLog, { type: 'log' }), yaxis: AX(T.fits, { range: [0, 102] }), shapes: [vline(W, C.red, 'dash')] }));
    const ms = median(D.cl.speeches), mt = median(D.cl.tweets);
    out('cl-out', [T.cl(T.corpus[cc], a.length, nf(W, 0), pct(share(a, W), 1), nf(median(a), 0), nf(a[a.length - 1], 0)),
      T.cl2(nf(W, 0), nf(Math.ceil(ms / W), 0), nf((W / mt) ** 2, 0))]);
  }
  draw();
  P('cl-play', 'cl-w', range(5, 15, 1), 700);
});

// ---------------------------------------------------------------- 2. FinBERT attention map on a real tweet
safe('attention-map', () => {
  const qEl = $('am-q');
  const t = bindSelect('am-t', () => { fixQ(); draw(); }), l = bindSlider('am-l', v => v, draw), q = bindSlider('am-q', v => v, draw);
  function fixQ() { const n = D.att[+t()].toks.length; qEl.max = n - 1; if (+qEl.value > n - 1) qEl.value = n - 1; qEl.dispatchEvent(new Event('input')); }
  function draw() {
    const a = D.att[+t()], L = l(), qi = Math.min(q(), a.toks.length - 1), A = a.avg[L - 1], tk = a.toks.map((s, i) => `${i}:${s}`);
    $('am-q-v').textContent = `${qi} (${a.toks[qi]})`;
    plot('am-map', [{ z: A, x: tk, y: tk, type: 'heatmap', colorscale: HEAT, zmin: 0, zmax: Math.max(0.3, ...A.flat()), showscale: false,
      hovertemplate: '%{y} → %{x}: %{z:.3f}<extra></extra>' }],
    layout({ title: ttl(T.amMap(L)), xaxis: AX(T.keys, { tickangle: -60, tickfont: { size: 9 } }), yaxis: AX(T.queries, { autorange: 'reversed', tickfont: { size: 9 } }),
      margin: { l: 90, r: 8, t: 30, b: 90 },
      shapes: [{ type: 'rect', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: qi - 0.5, y1: qi + 0.5, line: { color: C.red, width: 2 } }] }));
    const row = A[qi];
    plot('am-row', [{ x: tk, y: row, type: 'bar', marker: { color: a.toks.map(s => (isSpecial(s) ? C.grey : C.RO)) }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }],
      layout({ title: ttl(T.amRow(a.toks[qi])), xaxis: AX(T.keys, { tickangle: -60, tickfont: { size: 9 } }), yaxis: AX(T.weight, { range: [0, 1] }), showlegend: false, margin: { l: 56, r: 8, t: 30, b: 90 } }));
    let bi = -1;
    row.forEach((v, i) => { if (!isSpecial(a.toks[i]) && (bi < 0 || v > row[bi])) bi = i; });
    out('am-out', [T.am(a.toks[qi], L, a.toks[bi], pct(row[bi], 1), pct(row[row.length - 1], 1), pct(row[0], 1)),
      T.am2(T.labels[a.label], pct(a.p[0], 1), pct(a.p[1], 1), pct(a.p[2], 1))]);
  }
  draw();
  P('am-play', 'am-l', range(1, 12, 1), 800);
});

// ---------------------------------------------------------------- 3. scaled dot-product: the divisor of the scores and the entropy
safe('softmax-temperature', () => {
  const tok = D.att[0].toks, n = tok.length, tk = tok.map((s, i) => `${i}:${s}`), layers = range(1, 12, 1);
  const cv = bindSlider('st-c', v => (v < 0 ? '1/' + nf(2 ** -v, 2) : nf(2 ** v, 2)), draw), l = bindSlider('st-l', v => v, draw), h = bindSelect('st-h', draw);
  const q = bindSlider('st-q', v => `${v} (${tok[v]})`, draw);
  // mean over heads and rows of the normalised entropy of the example tweet, layer li, divisor div
  const entLayer = (li, div) => mean(D.qk[li].map(H => mean(H.map(r => entN(softmax(r, div))))));
  function draw() {
    const c = 2 ** cv(), L = l(), hh = +h(), qi = q(), s = D.qk[L - 1][hh][qi];
    const w = softmax(s, 8 * c), w1 = softmax(s, 8);
    plot('st-row', [
      { x: tk, y: w1, type: 'scatter', mode: 'markers', name: T.ref, marker: { color: C.navy, size: 9, symbol: 'line-ew-open', line: { width: 3 } } },
      { x: tk, y: w, type: 'bar', name: T.now, marker: { color: tok.map(t2 => (isSpecial(t2) ? C.grey : C.DE_LU)) } }],
    layout({ title: ttl(T.stRow(tok[qi], hh + 1, L)), xaxis: AX(T.keys, { tickangle: -60, tickfont: { size: 9 } }), yaxis: AX(T.weight, { range: [0, 1.02] }),
      margin: { l: 56, r: 8, t: 30, b: 90 }, legend: { orientation: 'h', x: 0, y: -0.55, yanchor: 'top' } }));
    const eThis = layers.map(li => entLayer(li - 1, 8 * c));
    plot('st-ent', [
      { x: layers, y: D.ent, mode: 'lines+markers', name: T.ent500, line: { color: C.navy, width: 2 }, marker: { size: 6 } },
      { x: layers, y: eThis, mode: 'lines+markers', name: T.entThis, line: { color: C.DE_LU, width: 2, dash: 'dash' }, marker: { size: 6 } }],
    layout({ title: ttl(T.stEnt), xaxis: AX(T.layer, { dtick: 1 }), yaxis: AX(T.ent, { range: [0, 1.02] }), shapes: [vline(L, C.red, 'dot')] }));
    const H = entN(w), mx = Math.max(...w), keff = Math.exp(H * Math.log(n));
    out('st-out', [T.st($('st-c-v').textContent, nf(H, 2), pct(mx, 1), nf(keff, 1)), T.st2(L, nf(entLayer(L - 1, 8), 2), nf(eThis[L - 1], 2))]);
  }
  draw();
  P('st-play', 'st-c', range(-3, 3, 0.25), 350);
});

// ---------------------------------------------------------------- 4. sinusoidal positional encoding (formula of Vaswani et al., 2017)
safe('positional-encoding', () => {
  const d = 64, NP = 64, pos = range(0, NP - 1, 1), dims = range(0, d - 1, 1);
  const PE = pos.map(p => dims.map(j => { const i = Math.floor(j / 2), a = p / 10000 ** (2 * i / d); return j % 2 ? Math.cos(a) : Math.sin(a); }));
  const dot = (u, v) => u.reduce((s, x, k) => s + x * v[k], 0) / (d / 2);
  const t = bindSlider('pe-t', v => v, draw), iv = bindSlider('pe-i', v => v, draw);
  function draw() {
    const tt = t(), i = iv();
    plot('pe-map', [{ z: dims.map(j => pos.map(p => PE[p][j])), x: pos, y: dims, type: 'heatmap', zmin: -1, zmax: 1,
      colorscale: [[0, '#DC3545'], [0.5, '#ffffff'], [1, '#1A3A6E']], colorbar: { thickness: 10, len: 0.8 }, hovertemplate: 't=%{x}, dim %{y}: %{z:.2f}<extra></extra>' }],
    layout({ title: ttl(T.peMap), xaxis: AX(T.position), yaxis: AX(T.dim, { autorange: 'reversed' }), margin: { l: 56, r: 8, t: 30, b: 44 },
      shapes: [vline(tt, C.amber, 'solid'), { type: 'rect', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 2 * i - 0.5, y1: 2 * i + 1.5, line: { color: C.green, width: 2 } }] }));
    const sim = pos.map(p => dot(PE[tt], PE[p]));
    const wave = pos.map(p => PE[p][2 * i]);
    plot('pe-sim', [
      { x: pos, y: sim, mode: 'lines+markers', name: T.dot, line: { color: C.RO, width: 2 }, marker: { size: 4 } },
      { x: pos, y: wave, mode: 'lines', name: `sin, dim ${2 * i}`, line: { color: C.green, width: 1.5, dash: 'dot' } }],
    layout({ title: ttl(T.peSim(tt)), xaxis: AX(T.position), yaxis: AX('', { range: [-1.05, 1.05] }), shapes: [vline(tt, C.amber, 'solid')] }));
    const lam = 2 * Math.PI * 10000 ** (2 * i / d), a = tt / 10000 ** (2 * i / d);
    const s = k => (tt + k < NP ? nf(sim[tt + k], 2) : '–');
    out('pe-out', [T.pe(tt, i, nf(lam, lam < 100 ? 1 : 0), nf(Math.sin(a), 2), nf(Math.cos(a), 2)), T.pe2(tt, s(1), s(5), s(20))]);
  }
  draw();
  P('pe-play', 'pe-t', range(0, 63, 1), 200);
});

// ---------------------------------------------------------------- 5. model comparison and paired bootstrap of macro-F1
safe('model-comparison', () => {
  const M = Object.keys(D.runs), y = [...D.pred.y].map(Number), n = y.length;
  const PR = {}; M.forEach(m => { PR[m] = [...D.pred[m]].map(Number); });
  const a = bindSelect('mc-a', draw), b = bindSelect('mc-b', draw), r = bindSlider('mc-r', v => nf(v, 0), draw);
  function macroF1(p, idx) {
    const tp = [0, 0, 0], fp = [0, 0, 0], fn = [0, 0, 0];
    for (const i of idx) { const yy = y[i], pp = p[i]; if (yy === pp) tp[yy]++; else { fp[pp]++; fn[yy]++; } }
    return mean([0, 1, 2].map(k => (2 * tp[k] + fp[k] + fn[k] ? 2 * tp[k] / (2 * tp[k] + fp[k] + fn[k]) : 0)));
  }
  // seeded generator (mulberry32): the same replications for every reader
  const rng = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t2 = Math.imul(seed ^ seed >>> 15, 1 | seed); t2 = t2 + Math.imul(t2 ^ t2 >>> 7, 61 | t2) ^ t2; return ((t2 ^ t2 >>> 14) >>> 0) / 4294967296; };
  const all = Array.from({ length: n }, (_, i) => i);
  function draw() {
    const A = a(), B = b(), R = r();
    const means = M.map(m => mean(D.runs[m]));
    const col = M.map(m => (m === A ? C.DE_LU : m === B ? C.RO : C.band));
    const tr = [{ y: M.map(m => T.models[m]), x: means, type: 'bar', orientation: 'h', marker: { color: col }, hovertemplate: '%{y}: %{x:.1f}%<extra></extra>', showlegend: false }];
    tr.push({ y: M.flatMap(m => D.runs[m].map(() => T.models[m])), x: M.flatMap(m => D.runs[m]), mode: 'markers', marker: { color: C.navy, size: 6 }, showlegend: false, hovertemplate: '%{x:.1f}%<extra></extra>' });
    plot('mc-bar', tr, layout({ title: ttl(T.mcBar), xaxis: AX(T.f1, { range: [50, 90] }), yaxis: AX('', { autorange: 'reversed', tickfont: { size: 10 } }), margin: { l: 10, r: 10, t: 30, b: 44 } }));
    if (A === B) {
      plot('mc-boot', [], layout({ title: ttl(T.mcBoot(nf(R, 0))), xaxis: AX(T.diff), yaxis: AX(T.freq) }));
      out('mc-out', [T.same]); return;
    }
    const fa = macroF1(PR[A], all), fb = macroF1(PR[B], all), g = rng(20261108), d = [];
    const idx = new Array(n);
    for (let k = 0; k < R; k++) { for (let i = 0; i < n; i++) idx[i] = Math.floor(g() * n); d.push(100 * (macroF1(PR[A], idx) - macroF1(PR[B], idx))); }
    const lo = quantile(d, 0.025), hi = quantile(d, 0.975), win = d.filter(x => x > 0).length / R;
    plot('mc-boot', [{ x: d, type: 'histogram', nbinsx: 40, marker: { color: C.DE_LU, line: { color: '#fff', width: 0.5 } }, hovertemplate: '%{x}: %{y}<extra></extra>' }],
      layout({ title: ttl(T.mcBoot(nf(R, 0))), xaxis: AX(T.diff), yaxis: AX(T.freq), showlegend: false, bargap: 0.02,
        shapes: [vline(lo, C.navy, 'dash'), vline(hi, C.navy, 'dash'), vline(0, C.red, 'solid')] }));
    out('mc-out', [T.mc(T.models[A], T.models[B], nf(100 * fa, 1), nf(100 * fb, 1), (fa >= fb ? '+' : '') + nf(100 * (fa - fb), 1)),
      T.mc2(nf(lo, 1), nf(hi, 1), pct(win, 1))]);
  }
  draw();
  P('mc-play', 'mc-r', range(200, 2000, 200), 900);
});

// ---------------------------------------------------------------- 6. learning with less data
safe('data-scaling', () => {
  const S = D.ds, keys = { tfidf: [T.tf, C.amber, 'circle'], tiny: [T.tiny, C.RO, 'square'], probe: [T.probe, C.c7, 'triangle-up'] };
  const k = bindSlider('ds-n', v => nf(D.ds.n[v], 0), draw), ms = bindToggles('ds-m', draw);
  function draw() {
    const j = k(), on = ms();
    plot('ds-line', on.map(m => ({ x: S.n, y: S[m], mode: 'lines+markers', name: keys[m][0], line: { color: keys[m][1], width: 2 },
      marker: { symbol: keys[m][2], size: S.n.map((_, i) => (i === j ? 13 : 7)) }, hovertemplate: '%{x}: %{y:.1f}%<extra>' + keys[m][0] + '</extra>' })),
    layout({ title: ttl(T.dsLine), xaxis: AX(T.trainN, { type: 'log', tickvals: S.n, ticktext: S.n.map(v => nf(v, 0)) }), yaxis: AX(T.f1, { range: [40, 80] }),
      shapes: [vline(S.n[j], C.red, 'dot')] }));
    plot('ds-bar', [{ x: on.map(m => keys[m][0]), y: on.map(m => S[m][j]), type: 'bar', marker: { color: on.map(m => keys[m][1]) },
      text: on.map(m => nf(S[m][j], 1)), textposition: 'outside', hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }],
    layout({ title: ttl(T.dsBar(nf(S.n[j], 0))), yaxis: AX(T.f1, { range: [0, 90] }), xaxis: AX('', { tickfont: { size: 10 } }), showlegend: false }));
    out('ds-out', [T.ds(nf(S.n[j], 0), on.length ? on.map(m => `${keys[m][0]} ${nf(S[m][j], 1)}%`).join('; ') : '–')]);
  }
  draw();
  P('ds-play', 'ds-n', range(0, 4, 1), 1000);
});
})();
