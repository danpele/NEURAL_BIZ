// Lecture 8 interactive page: generative models (autoencoder, VAE, GAN, diffusion). Inputs in L8_data.js
// (lectures/L8/build_interactive_L8.py), from the results of lectures/L8/analysis_L8.py and data/clean.
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, SERIES, mean, nf, pct, $, AX, layout, ttl, plot, out, vline, hline, bindSlider, bindSelect, bindToggles, Player, range, safe } = window.NNI;
const P = (id, input, vals, ms) => new Player(id, $(input), vals, ms);
const COL10 = SERIES.concat([C.navy, C.grey]);

// density on fixed bins (optionally weighted); returns step-line coordinates at bin centres
function density(v, edges, w) {
  const k = edges.length - 1, h = new Array(k).fill(0), lo = edges[0], wd = edges[1] - edges[0];
  let tot = 0;
  v.forEach((x, i) => { const ww = w ? w[i] : 1; tot += ww; const j = Math.floor((x - lo) / wd); if (j >= 0 && j < k) h[j] += ww; });
  return { x: h.map((_, j) => lo + (j + 0.5) * wd), y: h.map(c => c / (tot * wd)) };
}
const edgesOf = (lo, hi, k) => Array.from({ length: k + 1 }, (_, i) => lo + (hi - lo) * i / k);
function acf(x, L) {
  const m = mean(x), a = x.map(v => v - m), d = a.reduce((s, v) => s + v * v, 0), r = [];
  for (let l = 1; l <= L; l++) { let s = 0; for (let i = l; i < a.length; i++) s += a[i] * a[i - l]; r.push(s / d); }
  return r;
}
const qnt = (sorted, q) => { const p = (sorted.length - 1) * q, i = Math.floor(p); return i + 1 < sorted.length ? sorted[i] + (p - i) * (sorted[i + 1] - sorted[i]) : sorted[i]; };
const kurt = x => { const m = mean(x), n = x.length; let s2 = 0, s4 = 0; x.forEach(v => { const d = (v - m) ** 2; s2 += d; s4 += d * d; }); return n * s4 / (s2 * s2) - 3; };

// ---------------------------------------------------------------- 1. fraud alerts: autoencoder, PCA and logistic scores
safe('fraud-alerts', () => {
  const F = D.fr, y = F.y, w = y.map(v => (v ? 1 : F.w)), W = w.reduce((a, b) => a + b, 0), NF = y.reduce((a, b) => a + b, 0);
  // for each score: order of the rows by descending score (ties keep the file order), cumulative weight and frauds
  const R = {};
  ['ae', 'pca', 'logit'].forEach(k => {
    const s = F[k], o = s.map((_, i) => i).sort((a, b) => s[b] - s[a] || a - b), cw = [], cf = [];
    let a = 0, b = 0;
    o.forEach(i => { a += w[i]; b += y[i]; cw.push(a); cf.push(b); });
    R[k] = { o, cw, cf };
  });
  // number of rows flagged at a budget (share of the weighted test set): as analysis_L8.py, searchsorted + 1
  const cut = (k, share) => { const cw = R[k].cw, t = share * W; let i = 0; while (i < cw.length && cw[i] < t) i++; return Math.min(i + 1, cw.length); };
  const sc = bindSelect('fa-s', draw), b = bindSlider('fa-b', v => pct(T.budgets[v] / 100, T.budgets[v] < 0.1 ? 2 : 1), draw);
  const COLS = { ae: C.RO, pca: C.amber, logit: C.green };
  function draw() {
    const k = sc(), share = T.budgets[b()] / 100, n = cut(k, share), r = R[k];
    const thr = F[k][r.o[n - 1]], lg = F[k].map(v => Math.log10(v));
    const lo = Math.min(...lg), hi = Math.max(...lg), E = edgesOf(lo, hi, 40);
    const i0 = [], i1 = [];
    y.forEach((v, i) => (v ? i1 : i0).push(i));
    const d0 = density(i0.map(i => lg[i]), E, i0.map(i => w[i])), d1 = density(i1.map(i => lg[i]), E);
    plot('fa-hist', [
      { x: d0.x, y: d0.y, mode: 'lines', line: { shape: 'hvh', color: C.grey, width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(138,148,166,0.25)', name: T.legit },
      { x: d1.x, y: d1.y, mode: 'lines', line: { shape: 'hvh', color: C.red, width: 2 }, name: T.fraud }
    ], layout({ title: ttl(T.faHist(T.score[k])), xaxis: AX(T.faX(k)), yaxis: AX(T.density, { rangemode: 'tozero' }),
      shapes: [vline(Math.log10(thr), C.navy, 'dash')],
      annotations: [{ x: Math.log10(thr), y: 1, yref: 'paper', text: T.thr, showarrow: false, xanchor: 'right', font: { size: 11, color: C.navy } }] }));
    // precision-recall curves of the three scores, thinned to <= 800 points each
    const tr = [];
    ['ae', 'pca', 'logit'].forEach(kk => {
      const q = R[kk], step = Math.max(1, Math.floor(q.cw.length / 800)), xs = [], ys = [];
      for (let i = 0; i < q.cw.length; i += step) { xs.push(q.cf[i] / NF); ys.push(q.cf[i] / q.cw[i]); }
      tr.push({ x: xs, y: ys, mode: 'lines', name: T.score[kk].replace(/^./, c => c.toUpperCase()), line: { color: COLS[kk], width: kk === k ? 3 : 1.3 } });
    });
    const nk = r.cf[n - 1], prec = nk / r.cw[n - 1];
    tr.push({ x: [nk / NF], y: [prec], mode: 'markers', name: T.point, marker: { color: C.navy, size: 12, symbol: 'x' } });
    plot('fa-pr', tr, layout({ title: ttl(T.prTitle), xaxis: AX(T.recall, { range: [0, 1] }), yaxis: AX(T.precision, { type: 'log', range: [-3.2, 0.05] }) }));
    const found = kk => R[kk].cf[cut(kk, share) - 1];
    out('fa-out', [T.fa(pct(share, share < 0.001 ? 2 : 1), Math.round(share * W), T.scoreN[k], nk, NF, pct(nk / NF, 1), pct(prec, 1)),
      T.fa2(found('ae'), found('pca'), found('logit'))]);
  }
  draw();
  P('fa-play', 'fa-b', range(0, T.budgets.length - 1, 1), 900);
});

// ---------------------------------------------------------------- 2. VAE latent space: move a point, see what the decoder draws
safe('vae-latent', () => {
  const V = D.vl, G = D.vd, raw = atob(G.img), n1 = G.g1.length, n2 = G.g2.length;
  const a = bindSlider('vl-a', v => nf(v, 1), draw), b = bindSlider('vl-b', v => nf(v, 1), draw), cs = bindToggles('vl-c', draw);
  const idx = (g, v) => { let j = 0; g.forEach((x, i) => { if (Math.abs(x - v) < Math.abs(g[j] - v)) j = i; }); return j; };
  function image(i1, i2) {
    const off = (i1 * n2 + i2) * 784, z = [];
    for (let r = 0; r < 28; r++) { const row = []; for (let c = 0; c < 28; c++) row.push(raw.charCodeAt(off + r * 28 + c) / 255); z.push(row); }
    return z;
  }
  function draw() {
    const z1 = a(), z2 = b(), on = cs().map(Number), tr = [];
    on.forEach(c => {
      const ii = []; V.y.forEach((yy, i) => { if (yy === c && i % 2 === 0) ii.push(i); });
      tr.push({ x: ii.map(i => V.z1[i]), y: ii.map(i => V.z2[i]), type: 'scatter', mode: 'markers', name: T.classes[c],
        marker: { size: 3, color: COL10[c], opacity: 0.55 }, hovertemplate: T.classes[c] + '<extra></extra>' });
    });
    tr.push({ x: [z1], y: [z2], mode: 'markers', name: T.chosen, marker: { size: 16, color: C.ink, symbol: 'x-thin', line: { width: 3, color: C.ink } } });
    plot('vl-map', tr, layout({ title: ttl(T.mapTitle), xaxis: AX(T.z1, { range: [-2.4, 2.8] }), yaxis: AX(T.z2, { range: [-3.6, 3.4] }),
      legend: { orientation: 'h', y: -0.22, font: { size: 10 } }, margin: { l: 50, r: 8, t: 30, b: 40 } }));
    const i1 = idx(G.g1, z1), i2 = idx(G.g2, z2);
    plot('vl-img', [{ z: image(i1, i2), type: 'heatmap', colorscale: [[0, '#ffffff'], [1, '#111111']], zmin: 0, zmax: 1, showscale: false, hoverinfo: 'skip' }],
      layout({ title: ttl(T.imgTitle(nf(G.g1[i1], 1), nf(G.g2[i2], 1))), xaxis: { visible: false, scaleanchor: 'y', constrain: 'domain' },
        yaxis: { visible: false, autorange: 'reversed', constrain: 'domain' }, margin: { l: 10, r: 10, t: 30, b: 10 } }));
    // composition of the 50 nearest test codes
    const d = V.z1.map((x, i) => [(x - z1) ** 2 + (V.z2[i] - z2) ** 2, V.y[i]]).sort((p, q) => p[0] - q[0]).slice(0, 50);
    const cnt = new Array(10).fill(0); d.forEach(p => cnt[p[1]]++);
    const o = cnt.map((c, i) => [c, i]).sort((p, q) => q[0] - p[0]);
    out('vl-out', [T.vl(nf(z1, 1), nf(z2, 1), T.classes[o[0][1]].toLowerCase(), pct(o[0][0] / 50, 0), T.classes[o[1][1]].toLowerCase(), pct(o[1][0] / 50, 0),
      nf(Math.sqrt(d[49][0]), 2)), T.vl2]);
  }
  draw();
  P('vl-play', 'vl-a', G.g1, 500);
});

// ---------------------------------------------------------------- 3. GAN tails and the VaR backtest
safe('gan-tails', () => {
  const Gd = D.gan, str = [...Gd.tr].sort((p, q) => p - q), sg = [...Gd.g].sort((p, q) => p - q);
  const q = bindSlider('gt-q', v => pct(v / 100, 1), draw), view = bindSelect('gt-v', draw), ss = bindToggles('gt-s', draw);
  const E = edgesOf(-10, 10, 160);
  const dTr = density(Gd.tr, E), dG = density(Gd.g, E);
  const xs = range(-10, 10, 0.05), dN = xs.map(x => Math.exp(-0.5 * ((x - Gd.mu) / Gd.sd) ** 2) / (Gd.sd * Math.sqrt(2 * Math.PI)));
  function draw() {
    const p = q() / 100, s = ss(), tail = view() === 'tail';
    const vh = qnt(str, p), vg = qnt(sg, p), vn = Gd.mu + Gd.sd * window.NNI.nqnt(p);
    const tr = [];
    if (s.includes('tr')) tr.push({ x: dTr.x, y: dTr.y, mode: 'lines', line: { shape: 'hvh', color: C.grey, width: 1 }, fill: 'tozeroy', fillcolor: 'rgba(138,148,166,0.3)', name: T.real });
    if (s.includes('g')) tr.push({ x: dG.x, y: dG.y, mode: 'lines', line: { shape: 'hvh', color: C.RO, width: 2 }, name: T.gan });
    if (s.includes('n')) tr.push({ x: xs, y: dN, mode: 'lines', line: { color: C.DE_LU, width: 1.8, dash: 'dash' }, name: T.norm });
    plot('gt-hist', tr, layout({ title: ttl(T.gtHist(pct(p, 1))), xaxis: AX(T.ret, { range: tail ? [-9, -1] : [-6, 6] }),
      yaxis: AX(T.density, tail ? { type: 'log', range: [-3.5, -0.5] } : { rangemode: 'tozero' }),
      shapes: [vline(vh, C.grey, 'solid'), vline(vg, C.RO, 'dot'), vline(vn, C.DE_LU, 'dash')] }));
    // backtest on 2020-2026
    const te = Gd.te, ted = Gd.ted, cnt = v => te.filter(x => x < v).length;
    const bad = te.map((x, i) => [x, ted[i]]).filter(a => a[0] < Math.max(vh, vg, vn));
    plot('gt-bt', [
      { x: ted, y: te, mode: 'markers', marker: { size: 2.5, color: C.band }, name: T.test, hoverinfo: 'skip' },
      { x: bad.map(a => a[1]), y: bad.map(a => a[0]), mode: 'markers', marker: { size: 5, color: C.red }, name: T.viol, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
      { x: [ted[0], ted[ted.length - 1]], y: [vh, vh], mode: 'lines', line: { color: C.grey, width: 2 }, name: T.hist },
      { x: [ted[0], ted[ted.length - 1]], y: [vg, vg], mode: 'lines', line: { color: C.RO, width: 2, dash: 'dot' }, name: T.gan },
      { x: [ted[0], ted[ted.length - 1]], y: [vn, vn], mode: 'lines', line: { color: C.DE_LU, width: 2, dash: 'dash' }, name: T.norm }
    ], layout({ title: ttl(T.btTitle), xaxis: AX('', { type: 'date' }), yaxis: AX(T.ret, { range: [-13, 10] }) }));
    const N = te.length, f = v => `${cnt(v)} (${pct(cnt(v) / N, 2)})`;
    out('gt-out', [T.gt(pct(p, 1), nf(vh, 2) + '%', nf(vg, 2) + '%', nf(vn, 2) + '%'),
      T.gt2(nf(p * N, 1), f(vh), f(vg), f(vn), N), T.gt3(nf(Gd.kr, 2), nf(Gd.kg, 2))]);
  }
  draw();
  P('gt-play', 'gt-q', range(0.5, 5, 0.5), 700);
});

// ---------------------------------------------------------------- 4. GAN: no volatility clustering (ACF of |r|)
safe('gan-acf', () => {
  const Gd = D.gan, n = Gd.tr.length;
  const S = { tr: Gd.tr, g: Gd.g.slice(0, n), te: Gd.te };
  const name = { tr: T.real, g: T.gan, te: T.test }, col = { tr: C.navy, g: C.RO, te: C.amber };
  const tf = { abs: Math.abs, sq: x => x * x, raw: x => x };
  const cache = {};
  const get = (k, f) => (cache[k + f] = cache[k + f] || acf(S[k].map(tf[f]), 60));
  const fs = bindSelect('ga-f', draw), L = bindSlider('ga-L', v => v, draw), ss = bindToggles('ga-s', draw);
  function draw() {
    const f = fs(), LL = L(), lags = range(1, LL, 1), tr = [], lines = [];
    ss().forEach(k => {
      const a = get(k, f).slice(0, LL);
      tr.push({ x: lags, y: a, type: 'bar', name: name[k], marker: { color: col[k] }, hovertemplate: '%{x}: %{y:.3f}<extra>' + name[k] + '</extra>' });
      lines.push(T.ga(name[k], T.f[f], nf(a[0], 2), nf(mean(a), 2), LL));
    });
    const bd = 1.96 / Math.sqrt(n);
    plot('ga-chart', tr, layout({ title: ttl(T.gaTitle(T.f[f])), barmode: 'group', bargap: 0.15, xaxis: AX(T.lag), yaxis: AX(T.acf, { range: [-0.2, 0.55] }),
      shapes: [hline(bd, C.grey, 'dash'), hline(-bd, C.grey, 'dash')] }));
    lines.push(T.band(nf(bd, 3)));
    out('ga-out', lines.length > 1 ? lines : ['–']);
  }
  draw();
});

// ---------------------------------------------------------------- 5. diffusion: the forward process on a real yield curve
safe('diffusion-forward', () => {
  const Df = D.df, TT = 200, ab = [];
  let a = 1;
  for (let t = 0; t < TT; t++) { a *= 1 - (1e-4 + (0.02 - 1e-4) * t / (TT - 1)); ab.push(a); }
  const cc = bindSelect('df-c', draw), ee = bindSelect('df-e', draw), tt = bindSlider('df-t', v => v, draw);
  const fmtD = s => { const [y, m, d] = s.split('-'); return LANG === 'ro' ? `${d}.${m}.${y}` : `${+d} ${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][+m - 1]} ${y}`; };
  const tv = [0.25, 1, 2, 5, 10, 30], tl = LANG === 'ro' ? ['3L', '1A', '2A', '5A', '10A', '30A'] : ['3M', '1Y', '2Y', '5Y', '10Y', '30Y'];
  function draw() {
    const c = Df.curves[cc()], eps = Df.eps[ee()], t = tt(), s = Math.sqrt(ab[t]), nz = Math.sqrt(1 - ab[t]);
    const x0 = c.y.map((v, i) => (v - Df.ym[i]) / Df.ys[i]);
    const xt = x0.map((v, i) => (s * v + nz * eps[i]) * Df.ys[i] + Df.ym[i]);
    plot('df-curve', [
      { x: Df.mat, y: c.y, mode: 'lines+markers', name: T.x0, line: { color: C.navy, width: 2 }, marker: { size: 5 } },
      { x: Df.mat, y: xt, mode: 'lines+markers', name: T.xt(t), line: { color: C.DE_LU, width: 2 }, marker: { size: 5 } }
    ], layout({ title: ttl(T.dfTitle(fmtD(c.date))), xaxis: AX(T.mat, { type: 'log', tickvals: tv, ticktext: tl }), yaxis: AX(T.yield, { range: [-2.5, 7] }) }));
    const ts = range(0, TT - 1, 1);
    plot('df-w', [
      { x: ts, y: ab.map(Math.sqrt), mode: 'lines', name: T.sig, line: { color: C.navy, width: 2 } },
      { x: ts, y: ab.map(v => Math.sqrt(1 - v)), mode: 'lines', name: T.noi, line: { color: C.DE_LU, width: 2 } },
      { x: [t, t], y: [s, nz], mode: 'markers', marker: { size: 9, color: [C.navy, C.DE_LU] }, showlegend: false, hoverinfo: 'skip' }
    ], layout({ title: ttl(T.wTitle), xaxis: AX(T.step), yaxis: AX('', { range: [0, 1.02] }), shapes: [vline(t, C.grey, 'dash')] }));
    const rm = Math.sqrt(mean(xt.map((v, i) => (v - c.y[i]) ** 2)));
    out('df-out', [T.df(t, nf(ab[t], 3), nf(s, 3), nf(nz, 3), nf(rm, 2)), T.df2]);
  }
  draw();
  P('df-play', 'df-t', range(0, 199, 3), 120);
});

// ---------------------------------------------------------------- 6. yield-curve scenarios: real vs DDPM, Gaussian, VAE
safe('yc-scenarios', () => {
  const Y = D.yc, col = { real: C.grey, ddpm: C.RO, gauss: C.DE_LU, vae: C.c7 }, dash = { real: 'solid', ddpm: 'solid', gauss: 'dash', vae: 'dot' };
  const vv = bindSelect('yc-v', draw), xx = bindSlider('yc-x', v => nf(v, 1), draw), mm = bindToggles('yc-m', draw);
  // fixed subsample of indices for the scatter (every k-th point, about 1,500 per series)
  const sub = n => { const k = Math.max(1, Math.round(n / 1200)), r = []; for (let i = 0; i < n; i += k) r.push(i); return r; };
  function draw() {
    const v = vv(), x = xx(), ms = mm(), E = v === 'y10' ? edgesOf(-2, 6, 80) : edgesOf(-2.5, 4.5, 70), tr = [], sc = [], lines = [];
    ms.forEach(m => {
      const a = Y[m][v], d = density(a, E);
      tr.push(m === 'real'
        ? { x: d.x, y: d.y, mode: 'lines', line: { shape: 'hvh', color: col[m], width: 1 }, fill: 'tozeroy', fillcolor: 'rgba(138,148,166,0.3)', name: T.m[m] }
        : { x: d.x, y: d.y, mode: 'lines', line: { shape: 'hvh', color: col[m], width: 1.8, dash: dash[m] }, name: T.m[m] });
      const ii = sub(a.length);
      sc.push({ x: ii.map(i => Y[m].y10[i]), y: ii.map(i => Y[m].sl[i]), type: 'scatter', mode: 'markers', name: T.m[m], showlegend: false,
        marker: { size: 3, color: col[m], opacity: m === 'real' ? 0.5 : 0.35 } });
      const k = a.filter(z => z < x).length;
      lines.push(T.yc(T.m[m], T.v[v], nf(x, 1), pct(k / a.length, 1), a.length));
    });
    plot('yc-hist', tr, layout({ title: ttl(T.ycHist(T.vax[v])), xaxis: AX(T.vax[v]), yaxis: AX(T.density, { rangemode: 'tozero' }), shapes: [vline(x, C.navy, 'dash')] }));
    plot('yc-sc', sc, layout({ title: ttl(T.scTitle), xaxis: AX(T.scX, { range: [-2, 6] }), yaxis: AX(T.scY, { range: [-2.5, 4.5] }),
      shapes: [v === 'y10' ? vline(x, C.navy, 'dash') : hline(x, C.navy, 'dash')] }));
    lines.push(T.yc2);
    out('yc-out', lines);
  }
  draw();
});
})();
