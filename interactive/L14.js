// Lecture 14 interactive page: multimodal and vision-language models. Inputs in L14_data.js (lectures/L14/build_interactive_L14.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, nf, pct, cnt, $, AX, layout, ttl, plot, out, vline, hline, bindSlider, bindSelect, bindToggles, bindRadios, safe } = window.NNI;
const OK = '#008300', BAD = C.red;
const imgAxes = (n) => ({ xaxis: { range: [0, n], visible: false, constrain: 'domain' },
  yaxis: { range: [n, 0], visible: false, scaleanchor: 'x', constrain: 'domain' } });
const softmax = (v, s) => { const m = Math.max(...v), e = v.map(x => Math.exp(s * (x - m))), z = e.reduce((a, b) => a + b, 0); return e.map(x => x / z); };

// ---------------------------------------------------------------- 1. a vision transformer cuts the image into patches
safe('vit-patches', () => {
  const PS = [8, 14, 16, 28, 32, 56];
  const pix = D.vit.px.map(b64 => { const s = atob(b64), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; });
  const rgb = (a, r, c) => { const k = 3 * (64 * r + c); return [a[k], a[k + 1], a[k + 2]]; };
  const full = a => Array.from({ length: 64 }, (_, r) => Array.from({ length: 64 }, (_, c) => rgb(a, r, c)));
  const im = bindSelect('vp-i', draw), p = bindSlider('vp-p', v => `${PS[v]} × ${PS[v]}`, draw), view = bindRadios('vp-v', draw),
    op = bindSlider('vp-a', v => v, draw);
  function means(a, ps) {
    const g = 224 / ps, z = [];
    for (let gi = 0; gi < g; gi++) {
      const row = [];
      for (let gj = 0; gj < g; gj++) {
        const s = [0, 0, 0]; let n = 0;
        for (let u = gi * ps; u < (gi + 1) * ps; u++) for (let v = gj * ps; v < (gj + 1) * ps; v++) {
          const q = rgb(a, Math.floor(u * 64 / 224), Math.floor(v * 64 / 224)); s[0] += q[0]; s[1] += q[1]; s[2] += q[2]; n++;
        }
        row.push(s.map(x => Math.round(x / n)));
      }
      z.push(row);
    }
    return z;
  }
  function draw() {
    const i = +im(), ps = PS[p()], g = 224 / ps, n = g * g, a = pix[i];
    const shapes = [];
    for (let k = 0; k <= g; k++) {
      shapes.push({ type: 'line', x0: k * ps, x1: k * ps, y0: 0, y1: 224, line: { color: 'rgba(255,255,255,0.85)', width: 1 } });
      shapes.push({ type: 'line', y0: k * ps, y1: k * ps, x0: 0, x1: 224, line: { color: 'rgba(255,255,255,0.85)', width: 1 } });
    }
    const tr = view() === 'mean' ? { type: 'image', z: means(a, ps), x0: ps / 2, dx: ps, y0: ps / 2, dy: ps, hoverinfo: 'skip' }
      : { type: 'image', z: full(a), x0: 1.75, dx: 3.5, y0: 1.75, dy: 3.5, hoverinfo: 'skip' };
    plot('vp-img', [tr], layout(Object.assign({ title: ttl(T.img(ps)), shapes, margin: { l: 10, r: 10, t: 34, b: 10 } }, imgAxes(224))));
    const att = D.vit.att[i], ctr = Array.from({ length: 14 }, (_, k) => 8 + 16 * k);
    plot('vp-att', [{ type: 'image', z: full(a), x0: 1.75, dx: 3.5, y0: 1.75, dy: 3.5, hoverinfo: 'skip' },
      { type: 'heatmap', z: att, x: ctr, y: ctr, opacity: op() / 100, colorscale: 'Hot', showscale: false, hovertemplate: '%{z:.4f}<extra></extra>' }],
      layout(Object.assign({ title: ttl(T.att(op())), margin: { l: 10, r: 10, t: 34, b: 10 } }, imgAxes(224))));
    const flat = att.flat(), srt = [...flat].sort((x, y) => y - x), share = srt.slice(0, 20).reduce((s, x) => s + x, 0);
    const k = flat.indexOf(srt[0]);
    out('vp-out', [T.vp(n, g, ps, nf(((n + 1) / 197) ** 2, 2)), T.vp2(T.cls[i], [Math.floor(k / 14) + 1, (k % 14) + 1], pct(share, 0))]);
  }
  draw();
});

// ---------------------------------------------------------------- 2. zero-shot classification of one image
safe('zero-shot', () => {
  const im = bindSelect('zs-i', draw), m = bindSelect('zs-m', draw), t = bindSelect('zs-t', draw), s = bindSlider('zs-s', v => v, draw);
  function draw() {
    const i = +im(), sim = D.ex.sim[m() + '_' + t()][i], y = D.ex.y[i], pr = softmax(sim, s());
    const k = pr.indexOf(Math.max(...pr));
    plot('zs-img', [{ type: 'image', source: D.ex.jpg[i], x0: 0, dx: 1, y0: 0, dy: 1, hoverinfo: 'skip' }],
      layout(Object.assign({ title: ttl(T.cls[y]), margin: { l: 10, r: 10, t: 34, b: 10 } }, imgAxes(64))));
    const col = pr.map((_, c) => (c === k ? (k === y ? OK : BAD) : c === y ? OK : C.grey));
    plot('zs-bar', [{ type: 'bar', orientation: 'h', x: pr.map(v => 100 * v), y: T.cls, marker: { color: col },
      text: pr.map(v => nf(100 * v, 1)), textposition: 'outside', cliponaxis: false, hovertemplate: '%{y}: %{x:.1f}%<extra></extra>' }],
      layout({ title: ttl(T.simTitle(s())), xaxis: AX(T.prob, { range: [0, 105] }), yaxis: AX('', { autorange: 'reversed' }), margin: { l: 150, r: 20, t: 34, b: 44 } }));
    const H = -pr.reduce((a, v) => a + (v > 0 ? v * Math.log2(v) : 0), 0);
    out('zs-out', [T.zs(T.cls[y], T.cls[k], pct(pr[k], 1), k === y), T.zs2(nf(Math.max(...sim), 3), nf(Math.min(...sim), 3), s(), nf(H, 2))]);
  }
  draw();
});

// ---------------------------------------------------------------- 3. confusion matrix of zero-shot CLIP
safe('confusion', () => {
  const m = bindSelect('cm-m', draw), t = bindSelect('cm-t', draw), nrm = bindRadios('cm-n', draw);
  function draw() {
    const M = D.cm[m() + '_' + t()], rows = M.map(r => r.reduce((a, b) => a + b, 0));
    const R = M.map((r, i) => r.map(v => 100 * v / rows[i]));
    const Z = nrm() === 'row' ? R : M;
    plot('cm-chart', [{ type: 'heatmap', z: R, x: T.cls, y: T.cls, zmin: 0, zmax: 100, colorscale: [[0, '#ffffff'], [1, '#1A3A6E']], showscale: false,
      text: Z.map(r => r.map(v => (v >= (nrm() === 'row' ? 1 : 1) ? nf(v, 0) : ''))), texttemplate: '%{text}', textfont: { size: 11 },
      hovertemplate: `${T.truth}: %{y}<br>${T.pred}: %{x}<br>%{text}<extra></extra>` }],
      layout({ title: ttl($('cm-m').selectedOptions[0].textContent + ' · ' + $('cm-t').selectedOptions[0].textContent),
        xaxis: AX(T.pred, { tickangle: -30 }), yaxis: AX(T.truth, { autorange: 'reversed' }), margin: { l: 150, r: 12, t: 34, b: 120 } }));
    const n = rows.reduce((a, b) => a + b, 0), acc = M.reduce((a, r, i) => a + r[i], 0) / n;
    const d = R.map((r, i) => r[i]), bi = d.indexOf(Math.max(...d)), wi = d.indexOf(Math.min(...d));
    let bt = 0, bp = 1;
    M.forEach((r, i) => r.forEach((v, j) => { if (i !== j && v > M[bt][bp]) { bt = i; bp = j; } }));
    out('cm-out', [T.cm($('cm-m').selectedOptions[0].textContent, pct(acc, 1), T.cls[bi], nf(d[bi], 1) + '%', T.cls[wi], nf(d[wi], 1) + '%',
      T.cls[bt], T.cls[bp], M[bt][bp], rows[bt])]);
  }
  draw();
});

// ---------------------------------------------------------------- 4. text-to-image search
safe('text-search', () => {
  const PN = ['annual crop land', 'forest', 'brushland or shrubland', 'highway or road', 'industrial buildings or commercial buildings',
    'pasture land', 'permanent crop land', 'residential buildings or homes or apartments', 'river', 'lake or sea'];
  const curve = c => { const L = D.ts.lab[c]; let h = 0; return Array.from(L, (ch, j) => { h += +ch === c; return 100 * h / (j + 1); }); };
  const CUR = D.ts.lab.map((_, c) => curve(c)), AVG = CUR[0].map((_, j) => CUR.reduce((a, r) => a + r[j], 0) / 10);
  const q = bindSelect('ts-c', draw), k = bindSlider('ts-k', v => v, draw), av = bindToggles('ts-a', draw);
  let last = -1;
  function draw() {
    const c = +q(), kk = k(), xs = CUR[c].map((_, j) => j + 1);
    const tr = [{ x: xs, y: CUR[c], mode: 'lines', name: T.cls[c], line: { color: C.RO, width: 2.2 } }];
    if (av().includes('avg')) tr.push({ x: xs, y: AVG, mode: 'lines', name: T.avg, line: { color: C.grey, width: 1.5, dash: 'dash' } });
    tr.push({ x: [kk], y: [CUR[c][kk - 1]], mode: 'markers', marker: { color: C.navy, size: 10 }, showlegend: false, hoverinfo: 'skip' });
    plot('ts-chart', tr, layout({ title: ttl(T.query(PN[c])), xaxis: AX(T.k, { type: 'log' }), yaxis: AX(T.prec, { range: [0, 103] }),
      shapes: [vline(kk, C.navy, 'dot'), vline(D.ts.n[c], C.grey, 'dot')], showlegend: true }));
    if (c !== last) {
      $('ts-thumbs').innerHTML = `<div style="width:100%;font-size:0.85rem">${T.ts2}</div>` + D.ts.jpg[c].map(([src, y]) =>
        `<img src="${src}" width="72" height="72" alt="${T.cls[y]}" title="${T.cls[y]}" style="border:3px solid ${y === c ? OK : BAD};border-radius:3px">`).join('');
      last = c;
    }
    const hits = Math.round(CUR[c][kk - 1] * kk / 100);
    out('ts-out', [T.ts(T.cls[c], kk, nf(CUR[c][kk - 1], 1) + '%', hits, nf(AVG[kk - 1], 1) + '%') + ` (${cnt(D.ts.n[c], LANG === 'ro' ? 'imagine' : 'image', LANG === 'ro' ? 'imagini' : 'images')} ${LANG === 'ro' ? 'de test în clasă' : 'of the class in the test set'})`]);
  }
  draw();
});

// ---------------------------------------------------------------- 5. few labels: CLIP probe vs CNN trained from scratch
safe('few-shot', () => {
  const KS = [1, 2, 4, 8, 16, 32, 64, 128], F = D.fs;
  const st = (m, k) => { const v = F[m][k]; if (!v) return null; return { m: v.reduce((a, b) => a + b, 0) / v.length, lo: Math.min(...v), hi: Math.max(...v) }; };
  const k = bindSlider('fs-k', v => KS[v], draw), ser = bindToggles('fs-s', draw);
  function band(m, ks, color, name) {
    const s = ks.map(x => st(m, x));
    return [{ x: ks, y: s.map(v => v.hi), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: ks, y: s.map(v => v.lo), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: color + '33', showlegend: false, hoverinfo: 'skip' },
      { x: ks, y: s.map(v => v.m), mode: 'lines+markers', name, line: { color, width: 2.2 }, marker: { size: 6 }, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }];
  }
  function draw() {
    const kk = KS[k()], on = ser(), tr = [], shapes = [vline(kk, C.navy, 'dot')];
    if (on.includes('probe')) tr.push(...band('probe', KS, '#eb6834', T.probe));
    if (on.includes('cnn')) tr.push(...band('cnn', [4, 16, 64, 128], '#1baf7a', T.cnn));
    if (on.includes('zs')) tr.push({ x: [1, 128], y: [F.zs, F.zs], mode: 'lines', name: T.zsl, line: { color: '#eda100', width: 2, dash: 'dash' } });
    tr.push({ x: [1, 128], y: [F.full, F.full], mode: 'lines', name: T.full, line: { color: C.grey, width: 1.2, dash: 'dot' } });
    const pk = on.includes('probe') ? st('probe', kk) : null, ck = on.includes('cnn') ? st('cnn', kk) : null;
    if (pk) tr.push({ x: [kk], y: [pk.m], mode: 'markers', marker: { color: '#eb6834', size: 12, line: { color: C.navy, width: 1.5 } }, showlegend: false, hoverinfo: 'skip' });
    if (ck) tr.push({ x: [kk], y: [ck.m], mode: 'markers', marker: { color: '#1baf7a', size: 12, line: { color: C.navy, width: 1.5 } }, showlegend: false, hoverinfo: 'skip' });
    plot('fs-chart', tr, layout({ xaxis: AX(T.shots, { type: 'log', tickvals: KS, ticktext: KS.map(String) }), yaxis: AX(T.acc, { range: [40, 100] }), shapes, showlegend: true }));
    const p = st('probe', kk), c = st('cnn', kk);
    out('fs-out', [T.fs(kk, 10 * kk, nf(p.m, 1) + '%', nf(p.lo, 1), nf(p.hi, 1) + '%', c ? T.fsc(nf(c.m, 1) + '%', nf(c.lo, 1), nf(c.hi, 1) + '%') : T.fsn),
      T.fs2(nf(F.zs, 1) + '%', nf(F.full, 1) + '%', nf(F.fullp, 1) + '%')]);
  }
  draw();
});

// ---------------------------------------------------------------- 6. market stress from the picture or from the numbers
safe('chart-regime', () => {
  const R = D.cr, n = R.y.length, nt = R.y.reduce((a, b) => a + b, 0), MS = ['numeric', 'clip_probe', 'clip_zs', 'vlm_stress'];
  const COL = { numeric: '#1baf7a', clip_probe: '#eb6834', clip_zs: '#eda100', vlm_stress: '#6250d6' };
  const NAMES = {};
  [...$('cr-m').options].forEach(o => { NAMES[o.value] = o.textContent; });
  function roc(m) {
    const o = R[m].map((s, i) => [s, R.y[i]]).sort((a, b) => b[0] - a[0]);
    let tp = 0, fp = 0; const x = [0], y = [0];
    o.forEach(([, yy]) => { yy ? tp++ : fp++; x.push(fp / (n - nt)); y.push(tp / nt); });
    return { x, y, o };
  }
  const ROC = Object.fromEntries(MS.map(m => [m, roc(m)]));
  const m = bindSelect('cr-m', draw), a = bindSlider('cr-a', v => v + '%', draw);
  function draw() {
    const mm = m(), r = ROC[mm], kf = Math.max(1, Math.round(a() / 100 * n)), thr = r.o[kf - 1][0];
    const tp = r.o.slice(0, kf).filter(v => v[1]).length, fp = kf - tp;
    const tr = MS.map(q => ({ x: ROC[q].x, y: ROC[q].y, mode: 'lines', name: NAMES[q] + ` (${nf(R.auc[q], 3)})`,
      line: { color: COL[q], width: q === mm ? 3 : 1.2 }, opacity: q === mm ? 1 : 0.45, hoverinfo: 'skip' }));
    tr.push({ x: [0, 1], y: [0, 1], mode: 'lines', name: T.chance, line: { color: C.grey, dash: 'dot', width: 1 }, hoverinfo: 'skip' });
    tr.push({ x: [fp / (n - nt)], y: [tp / nt], mode: 'markers', marker: { color: C.navy, size: 12 }, showlegend: false, hoverinfo: 'skip' });
    plot('cr-roc', tr, layout({ xaxis: AX(T.fpr, { range: [0, 1] }), yaxis: AX(T.tpr, { range: [0, 1.02] }), showlegend: true,
      legend: { orientation: 'h', x: 0, y: -0.3, yanchor: 'top', font: { size: 10 } }, margin: { l: 56, r: 12, t: 20, b: 40 } }));
    const st = R.end.map((d, i) => (R.y[i] ? d : null)).filter(Boolean);
    plot('cr-time', [{ x: R.end, y: R[mm], mode: 'lines', name: T.score, line: { color: COL[mm], width: 1.6 } },
      { x: st, y: st.map(d => R[mm][R.end.indexOf(d)]), mode: 'markers', name: T.stress, marker: { color: C.red, size: 6 } },
      { x: [R.end[0], R.end[n - 1]], y: [thr, thr], mode: 'lines', name: T.thr, line: { color: C.navy, dash: 'dash', width: 1.5 } }],
    layout({ xaxis: AX(T.date, { type: 'date', tickformat: '%Y' }), yaxis: AX(T.score), showlegend: true, margin: { l: 56, r: 12, t: 20, b: 40 } }));
    out('cr-out', [T.cr(NAMES[mm], nf(R.auc[mm], 3), pct(kf / n, 1), tp, nt, fp, n - nt, pct(tp / kf, 1)), T.cr2(n, nt)]);
  }
  draw();
});
})();
