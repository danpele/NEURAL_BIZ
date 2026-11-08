// Lecture 12 interactive page: large language models in finance. Inputs in L12_data.js (lectures/L12/build_interactive_L12.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, SERIES, nf, pct, $, AX, layout, ttl, plot, out, vline, ncdf, bindSlider, bindSelect, bindToggles, bindRadios, Player, range, safe } = window.NNI;
const RO = LANG === 'ro';
const P = (id, input, vals, ms) => new Player(id, $(input), vals, ms);
// play button that steps through the options of a radio group
function RP(btn, group, ms) {
  const b = $(btn), box = $(group), lab = RO ? ['Pornește', 'Pauză'] : ['Play', 'Pause'];
  let timer = null;
  const label = () => { b.textContent = timer ? '⏸ ' + lab[1] : '▶ ' + lab[0]; };
  b.addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; label(); return; }
    const rs = [...box.querySelectorAll('input')];
    let i = (rs.findIndex(r => r.checked) + 1) % rs.length, left = rs.length;
    timer = setInterval(() => { rs[i].checked = true; box.dispatchEvent(new Event('change')); i = (i + 1) % rs.length; if (--left <= 0) { clearInterval(timer); timer = null; label(); } }, ms);
    label();
  });
  label();
}
const PCT = x => pct(x, 1);

// macro-F1 and accuracy of two label strings ('0', '1', '2' per item)
function scores(y, p, n) {
  n = n || 3;
  const M = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < y.length; i++) M[+y[i]][+p[i]]++;
  let f = 0, ok = 0;
  for (let c = 0; c < n; c++) {
    const tp = M[c][c], fp = M.reduce((s, r, i) => s + (i === c ? 0 : r[c]), 0), fn = M[c].reduce((s, v, j) => s + (j === c ? 0 : v), 0);
    f += tp ? 2 * tp / (2 * tp + fp + fn) : 0; ok += tp;
  }
  return { f1: f / n, acc: ok / y.length, M };
}
function confusion(id, M, labs, title) {
  const tot = M.map(r => r.reduce((a, b) => a + b, 0));
  const z = M.map((r, i) => r.map(v => v / Math.max(1, tot[i])));
  plot(id, [{ type: 'heatmap', z, x: labs, y: labs, colorscale: [[0, '#ffffff'], [1, C.blue]], zmin: 0, zmax: 1, showscale: false,
    text: M.map(r => r.map(String)), texttemplate: '%{text}', hovertemplate: '%{y} → %{x}: %{text}<extra></extra>' }],
  layout({ title: ttl(title), xaxis: AX(T.pred, { side: 'bottom' }), yaxis: AX(T.truth, { autorange: 'reversed' }), margin: { l: 110, r: 12, t: 34, b: 50 } }));
}

// ---------------------------------------------------------------- 1. next token: temperature and top-k
safe('next-token', () => {
  const tT = bindSlider('nt-T', v => nf(v, 2), draw), tk = bindSlider('nt-k', v => v, draw);
  const p1 = D.nt.p1, toks = D.nt.tok.map(s => s.trim() === '' ? '␣' : '“' + s.replace(/^ /, '␣') + '”');
  function draw() {
    const Tv = tT(), k = tk();
    const w = p1.slice(0, k).map(p => Math.pow(p, 1 / Tv)), s = w.reduce((a, b) => a + b, 0), q = w.map(x => x / s);
    const h = -q.reduce((a, x) => a + (x > 0 ? x * Math.log(x) : 0), 0);
    const ref = p1.slice(0, k).map(x => x / p1.slice(0, k).reduce((a, b) => a + b, 0));
    plot('nt-chart', [
      { type: 'bar', x: toks.slice(0, k), y: q.map(x => 100 * x), name: `T = ${nf(Tv, 2)}`, marker: { color: SERIES[1] },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
      { type: 'scatter', mode: 'markers', x: toks.slice(0, k), y: ref.map(x => 100 * x), name: 'T = 1', marker: { color: C.navy, symbol: 'line-ew-open', size: 26, line: { width: 2 } },
        hovertemplate: 'T = 1: %{y:.1f}%<extra></extra>' }
    ], layout({ title: ttl(T.ntTitle(nf(Tv, 2), k)), xaxis: AX(T.token), yaxis: AX(T.prob, { range: [0, 100] }), bargap: 0.3 }));
    out('nt-out', [T.nt(nf(Tv, 2), k, D.nt.tok[0].trim(), PCT(q[0]), nf(h, 2)),
      T.nt3 + PCT(p1.reduce((a, b) => a + b, 0)) + '.', T.nt2]);
  }
  draw();
  P('nt-play', 'nt-T', range(0.1, 2.0, 0.05), 150);
});

// ---------------------------------------------------------------- 2. few-shot: number of examples and draw
safe('few-shot', () => {
  const KS = [0, 1, 2, 4, 8];
  const m = bindSelect('fs-m', draw), kk = bindSlider('fs-k', v => KS[v], draw), sd = bindRadios('fs-s', draw);
  const f1 = (mm, k, s) => scores(D.fs.y, D.fs[`${mm}_k${k}_s${s}`]);
  function draw() {
    const mm = m(), k = KS[kk()], s = +sd();
    const tr = [0, 1, 2].map(j => ({ type: 'scatter', mode: 'markers', x: KS.map(String), y: KS.map(x => 100 * f1(mm, x, j).f1), name: T.seed(j + 1),
      marker: { color: SERIES[j + 2], size: j === s ? 11 : 7, symbol: j === s ? 'diamond' : 'circle' } }));
    tr.push({ type: 'scatter', mode: 'lines', x: KS.map(String), y: KS.map(x => 100 * [0, 1, 2].reduce((a, j) => a + f1(mm, x, j).f1, 0) / 3),
      name: T.mean, line: { color: C.navy, width: 2 } });
    const r = f1(mm, k, s), all = [0, 1, 2].map(j => f1(mm, k, j).f1);
    plot('fs-curve', tr, layout({ title: ttl(`Qwen2.5-${mm}, Financial PhraseBank`), xaxis: AX(T.kAxis, { type: 'category' }), yaxis: AX(T.f1),
      shapes: [{ type: 'rect', xref: 'x', yref: 'paper', x0: KS.indexOf(k) - 0.4, x1: KS.indexOf(k) + 0.4, y0: 0, y1: 1, fillcolor: C.band, opacity: 0.35, line: { width: 0 } }] }));
    confusion('fs-conf', r.M, T.lab.pb, T.confT(`k = ${k}, ${T.seed(s + 1)}`));
    out('fs-out', [T.fs(mm, k, s + 1, PCT(r.f1), PCT(r.acc), PCT(Math.min(...all)), PCT(Math.max(...all))), T.fs0(PCT(f1(mm, 0, 0).f1))]);
  }
  draw();
  P('fs-play', 'fs-k', [0, 1, 2, 3, 4], 900);
});

// ---------------------------------------------------------------- 3. prompt sensitivity: five wordings of the instruction
safe('prompt-sensitivity', () => {
  const d = bindSelect('ps-d', draw), m = bindSelect('ps-m', draw), ins = bindRadios('ps-i', draw);
  function draw() {
    const ds = d(), mm = m(), i = +ins(), S = D.ps[ds];
    const f = [0, 1, 2, 3, 4].map(j => scores(S.y, S[`${mm}_i${j + 1}`]));
    const fv = f.map(x => 100 * x.f1);
    plot('ps-bars', [{ type: 'bar', x: [1, 2, 3, 4, 5].map(j => `${T.instr} ${j}`), y: fv, marker: { color: fv.map((_, j) => j === i ? SERIES[1] : C.band) },
      text: fv.map(v => nf(v, 1)), textposition: 'outside', hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }],
    layout({ title: ttl(T.psBars(T.ds[ds], mm)), xaxis: AX(''), yaxis: AX(T.f1, { range: [0, 100] }), showlegend: false }));
    confusion('ps-conf', f[i].M, T.lab[ds], T.confT(`${T.instr.toLowerCase()} ${i + 1}`));
    out('ps-out', [T.ps(i + 1, PCT(f[i].f1), PCT(Math.min(...fv) / 100), PCT(Math.max(...fv) / 100)), T.psText(D.ps.instr[ds][i])]);
  }
  draw();
  RP('ps-play', 'ps-i', 900);
});

// ---------------------------------------------------------------- 4. calibration and abstention
safe('calibration', () => {
  const d = bindSelect('cb-d', draw), m = bindSelect('cb-m', draw), th = bindSlider('cb-t', v => nf(v, 2), draw);
  function draw() {
    const ds = d(), mm = m(), t = th(), S = D.cb[ds], M = S[mm], y = S.y, n = y.length;
    const ok = M.conf.map((_, i) => M.pred[i] === y[i]);
    // reliability: 10 bins on [1/3, 1], as in the lecture; bins with at least 5 items are drawn
    const ed = Array.from({ length: 11 }, (_, i) => 1 / 3 + i * (2 / 3) / 10), bx = [], by = [], bn = [];
    let ece = 0;
    for (let b = 0; b < 10; b++) {
      const idx = M.conf.map((c, i) => i).filter(i => M.conf[i] >= ed[b] && (b === 9 ? M.conf[i] <= 1 : M.conf[i] < ed[b + 1]));
      if (!idx.length) continue;
      const mc = idx.reduce((a, i) => a + M.conf[i], 0) / idx.length, ma = idx.filter(i => ok[i]).length / idx.length;
      ece += idx.length / n * Math.abs(mc - ma);
      if (idx.length >= 5) { bx.push(100 * mc); by.push(100 * ma); bn.push(idx.length); }
    }
    plot('cb-rel', [
      { x: [100 / 3, 100], y: [100 / 3, 100], mode: 'lines', name: T.diag, line: { color: C.grey, dash: 'dot' } },
      { x: bx, y: by, mode: 'lines+markers', name: T.model, line: { color: SERIES[1] }, marker: { size: bn.map(v => 5 + 2 * Math.sqrt(v / 10)) },
        text: bn.map(String), hovertemplate: '%{x:.0f}% → %{y:.0f}% (n = %{text})<extra></extra>' }
    ], layout({ title: ttl(T.relT), xaxis: AX(T.conf + ' (%)', { range: [30, 101] }), yaxis: AX(T.acc, { range: [0, 101] }),
      shapes: [vline(100 * t, C.navy, 'dash')] }));
    // accuracy vs coverage along the threshold
    const ts = range(0.33, 0.99, 0.01), cx = [], cy = [];
    ts.forEach(u => { const a = M.conf.map((c, i) => i).filter(i => M.conf[i] >= u); if (a.length >= 10) { cx.push(100 * a.length / n); cy.push(100 * a.filter(i => ok[i]).length / a.length); } });
    const ans = M.conf.map((c, i) => i).filter(i => M.conf[i] >= t), accA = ans.length ? ans.filter(i => ok[i]).length / ans.length : NaN;
    plot('cb-cov', [
      { x: cx, y: cy, mode: 'lines', name: T.model, line: { color: SERIES[1], width: 2 }, hoverinfo: 'skip' },
      { x: [100 * ans.length / n], y: [100 * accA], mode: 'markers', name: `${nf(t, 2)}`, marker: { color: C.navy, size: 11 } }
    ], layout({ title: ttl(T.covT), xaxis: AX(T.cov, { range: [0, 101], autorange: 'reversed' }), yaxis: AX(T.acc, { range: [Math.min(50, ...cy) - 2, 101] }), showlegend: false }));
    const s = scores(y, M.pred);
    out('cb-out', [T.cb(PCT(ece), PCT(s.f1), PCT(s.acc)),
      T.cb2(nf(t, 2), RO ? `${ans.length} din ${n}` : `${ans.length} of ${n}`, PCT(ans.length / n), PCT(accA))]);
  }
  draw();
  P('cb-play', 'cb-t', range(0.33, 0.99, 0.02), 200);
});

// ---------------------------------------------------------------- 5. RAG: recall@k of four retrievers
safe('rag-recall', () => {
  const rs = bindToggles('rg-r', draw), kk = bindSlider('rg-k', v => v, draw), yr = bindSelect('rg-y', draw);
  const COL = { tfidf: C.green, bm25: SERIES[5], dense: SERIES[1], hybrid: SERIES[6] };
  function draw() {
    const on = rs(), k = kk(), y = yr();
    const idx = D.rg.year.map((v, i) => i).filter(i => y === 'all' || D.rg.year[i] === +y), n = idx.length;
    const ks = range(1, 20, 1);
    const rec = (m, kx) => idx.filter(i => D.rg[m][i] <= kx).length / n;
    plot('rg-chart', on.map(m => ({ x: ks, y: ks.map(kx => 100 * rec(m, kx)), mode: 'lines+markers', name: T.ret[m], line: { color: COL[m] }, marker: { size: 5 } })),
      layout({ title: ttl(T.rgT(y === 'all' ? T.allYears : y, n)), xaxis: AX(T.kRet, { dtick: 1 }), yaxis: AX(T.recall, { range: [0, 101] }), shapes: [vline(k, C.navy, 'dash')] }));
    const lines = on.map(m => T.rg(T.ret[m], PCT(rec(m, k)), nf(idx.reduce((a, i) => a + 1 / D.rg[m][i], 0) / n, 3)));
    out('rg-out', (lines.length ? lines : [T.ret0]).concat([T.ret1]));
  }
  draw();
  P('rg-play', 'rg-k', range(1, 20, 1), 400);
});

// ---------------------------------------------------------------- 6. LLM-VaR backtest
safe('llm-var', () => {
  const a = bindRadios('lv-a', draw), ms = bindToggles('lv-m', draw);
  const months = []; D.lv.date.forEach((d, i) => { const m = d.slice(0, 7); if (!months.length || months[months.length - 1].m !== m) months.push({ m, i }); });
  const fmtM = m => { const [y, mo] = m.split('-'); return `${window.NNI.MONTHS[+mo - 1]} ${y}`; };
  $('lv-s').max = months.length - 4;
  const st = bindSlider('lv-s', v => fmtM(months[v].m), draw);
  const COL = { llm: SERIES[1], hs: C.green, garch: SERIES[6] };
  function kupiec(x, n, al) {
    const p = x / n;
    const lr = x === 0 ? -2 * n * Math.log(1 - al) : -2 * ((n - x) * Math.log(1 - al) + x * Math.log(al) - (x === n ? 0 : (n - x) * Math.log(1 - p)) - x * Math.log(p));
    return 2 * (1 - ncdf(Math.sqrt(Math.max(lr, 0))));      // chi-square with 1 degree of freedom
  }
  function draw() {
    const lv = a(), al = lv === '05' ? 0.05 : 0.01, on = ms(), i0 = months[st()].i;
    const dt = D.lv.date.slice(i0), r = D.lv.r.slice(i0), n = r.length;
    const tr = [{ type: 'bar', x: dt, y: r, name: T.retS, marker: { color: C.band }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }];
    const lines = [];
    on.forEach(m => {
      const v = D.lv[`${m}_${lv}`].slice(i0), hit = r.map((x, i) => x < v[i]);
      tr.push({ x: dt, y: v, mode: 'lines', name: T.vm[m], line: { color: COL[m], width: 1.6, shape: 'hv' } });
      tr.push({ x: dt.filter((_, i) => hit[i]), y: r.filter((_, i) => hit[i]), mode: 'markers', name: `${T.viol}, ${T.vm[m]}`, showlegend: false,
        marker: { color: COL[m], size: 7, symbol: 'x' } });
      const x = hit.filter(Boolean).length, p = kupiec(x, n, al);
      lines.push(T.lv(T.vm[m], x, nf(al * n, 1), pct(x / n, 2), nf(p, 3), p >= 0.05));
    });
    plot('lv-chart', tr, layout({ title: ttl(T.lvT(lv === '05' ? '5%' : '1%', fmtM(months[st()].m), fmtM(months[months.length - 1].m))),
      xaxis: AX('', { type: 'date', tickformat: RO ? '%b %Y' : '%b %Y' }), yaxis: AX(T.ret_), bargap: 0 }));
    out('lv-out', lines.length ? lines : [T.lv0]);
  }
  draw();
  P('lv-play', 'lv-s', range(0, months.length - 4, 1), 500);
});
})();
