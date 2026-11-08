// Lecture 5 interactive page: recurrent networks for financial time series and electricity load. Inputs in L5_data.js
// (lectures/L5/build_interactive_L5.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, MONTHS, nf, pct, $, AX, layout, ttl, plot, out, hline, vline, bindSlider, bindSelect, bindToggles, bindRadios, Player, range, safe } = window.NNI;
const P = (id, input, vals, ms) => new Player(id, $(input), vals, ms);
const COL = { obs: '#2a78d6', naive: '#8A94A6', snaive: '#1baf7a', har: '#1baf7a', garch: '#eda100', ridge: '#eda100', rnn: '#e87ba4', lstm: '#eb6834', gru: '#6250d6', mean: '#8A94A6', ar: '#1baf7a' };
const fd = s => { const [y, m, d] = s.split('-').map(Number); return `${d} ${MONTHS[m - 1]} ${y}`; };      // 2020-01-02 -> 2 Jan 2020
const mon = (y, m) => `${MONTHS[m]} ${y}`;
const dow = s => new Date(s + 'T00:00:00Z').getUTCDay();

// ---------------------------------------------------------------- 1. autocorrelation of returns and absolute returns
safe('acf', () => {
  const m = bindSelect('ac-m', draw), t = bindRadios('ac-t', draw), l = bindSlider('ac-l', v => v, draw);
  function acf(x, L) {
    const n = x.length, mu = x.reduce((a, b) => a + b, 0) / n, z = x.map(v => v - mu), v0 = z.reduce((a, b) => a + b * b, 0), r = [];
    for (let k = 1; k <= L; k++) { let s = 0; for (let i = k; i < n; i++) s += z[i] * z[i - k]; r.push(s / v0); }
    return r;
  }
  function draw() {
    const mk = m(), tt = t(), L = l(), raw = D.ac[mk];
    const x = tt === 'r' ? raw : tt === 'a' ? raw.map(Math.abs) : raw.map(v => v * v);
    const r = acf(x, L), b = 1.96 / Math.sqrt(x.length), lags = r.map((_, i) => i + 1);
    plot('ac-chart', [{ x: lags, y: r, type: 'bar', marker: { color: r.map(v => (Math.abs(v) > b ? '#eb6834' : '#2a78d6')) }, name: T.acf, showlegend: false,
      hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }, { x: [null], y: [null], mode: 'lines', name: T.band, line: { color: C.grey, dash: 'dash' } }],
      layout({ title: ttl(T.acTitle(T.mk[mk], T.ser[tt], fd(D.ac.range[mk][0]), fd(D.ac.range[mk][1]))), xaxis: AX(T.lag), yaxis: AX(T.acf, { range: [Math.min(-0.15, Math.min(...r) - 0.03), Math.max(0.35, Math.max(...r) + 0.03)] }),
        bargap: 0.15, shapes: [hline(b, C.grey, 'dash'), hline(-b, C.grey, 'dash')] }));
    out('ac-out', [T.ac(nf(r[0], 3), nf(r[9], 3), r.filter(v => Math.abs(v) > b).length, L, nf(b, 3), nf(x.length, 0))]);
  }
  draw();
});

// ---------------------------------------------------------------- 2. volatility forecasts on the test period
safe('volatility', () => {
  const v = D.vf, K = ['naive', 'har', 'garch', 'rnn', 'lstm', 'gru'];
  const toVol = x => Math.sqrt(252 * Math.exp(x)) * 100;
  const s = bindSlider('vf-s', i => mon(2020 + Math.floor(i / 12), i % 12), draw), n = bindSlider('vf-n', i => i, draw), m = bindToggles('vf-m', draw);
  function draw() {
    const i0 = s(), i1 = Math.min(i0 + n(), 82);
    const a = `${2020 + Math.floor(i0 / 12)}-${String(i0 % 12 + 1).padStart(2, '0')}`, b = `${2020 + Math.floor(i1 / 12)}-${String(i1 % 12 + 1).padStart(2, '0')}`;
    const idx = v.d.map((d, i) => i).filter(i => v.d[i] >= a && v.d[i] < b);
    if (!idx.length) { out('vf-out', ['–']); return; }
    const x = idx.map(i => v.d[i]), tr = [{ x, y: idx.map(i => toVol(v.obs[i])), mode: 'lines', name: T.mod.obs, line: { color: COL.obs, width: 1 }, opacity: 0.75 }];
    const on = m(), mse = {};
    K.forEach(k => { mse[k] = idx.reduce((acc, i) => acc + (v[k][i] - v.obs[i]) ** 2, 0) / idx.length; });
    const rank = [...K].sort((p, q) => mse[p] - mse[q]);
    on.forEach(k => tr.push({ x, y: idx.map(i => toVol(v[k][i])), mode: 'lines', name: T.mod[k], line: { color: COL[k], width: 2 } }));
    plot('vf-chart', tr, layout({ title: ttl(T.vfTitle(fd(x[0]), fd(x[x.length - 1]))), xaxis: AX('', { type: 'date', tickformat: '%b %Y' }), yaxis: AX(T.vol, { type: 'log', tickvals: [2, 5, 10, 20, 50, 100, 200], ticktext: ['2', '5', '10', '20', '50', '100', '200'] }) }));
    out('vf-out', on.map(k => T.vf(T.mod[k], nf(mse[k], 3), rank.indexOf(k) + 1)).concat([T.vf0(nf(idx.length, 0), fd(x[0]), fd(x[x.length - 1]))]));
  }
  draw();
  P('vf-play', 'vf-s', range(0, 81, 3), 500);
});

// ---------------------------------------------------------------- 3. returns: out-of-sample R² and where it comes from
safe('returns', () => {
  const m = bindSelect('rt-m', draw), s = bindSelect('rt-s', draw), k = bindToggles('rt-k', draw);
  function draw() {
    const r = D.rt[m()], y = r.y, n = y.length, ss = s();
    const lstm = ss === 'ens' ? y.map((_, i) => r.lstm.reduce((a, p) => a + p[i], 0) / r.lstm.length) : r.lstm[+ss];
    const F = { mean: y.map(() => r.mean), ar: r.ar, lstm };
    const sse0 = y.reduce((a, v) => a + v * v, 0), tr = [], lines = [];
    k().forEach(kk => {
      const f = F[kk]; let c = 0; const cum = y.map((v, i) => (c += v * v - (v - f[i]) ** 2));
      tr.push({ x: r.d, y: cum, mode: 'lines', name: T.rtMod[kk], line: { color: COL[kk], width: 2 } });
      const sse = y.reduce((a, v, i) => a + (v - f[i]) ** 2, 0), hit = kk === 'mean' ? null : y.filter((v, i) => Math.sign(v) === Math.sign(f[i])).length / n;
      lines.push(T.rt(T.rtMod[kk], pct(1 - sse / sse0, 2), hit === null ? '' : pct(hit, 1)));
    });
    lines.push(T.rt0(nf(n, 0), pct(y.filter(v => v > 0).length / n, 1)));
    plot('rt-chart', tr, layout({ title: ttl(T.rtTitle(T.mk[m()])), xaxis: AX('', { type: 'date', tickformat: '%Y' }), yaxis: AX(T.cum), shapes: [hline(0, C.ink, 'dot')] }));
    out('rt-out', lines);
  }
  draw();
});

// ---------------------------------------------------------------- 4. day-ahead load forecasts, day by day
safe('load-days', () => {
  const d = D.ld, K = ['snaive', 'ridge', 'lstm', 'gru'], nd = d.d.length;
  const mape = (k, i0, i1) => { let s = 0, c = 0; for (let i = i0; i < i1; i++) for (let h = 0; h < 24; h++) { s += Math.abs(d[k][i][h] - d.obs[i][h]) / d.obs[i][h]; c++; } return s / c; };
  const full = {}; K.forEach(k => { full[k] = mape(k, 0, nd); });
  const s = bindSlider('ld-d', i => fd(d.d[i]), draw), w = bindSelect('ld-w', draw), m = bindToggles('ld-m', draw);
  function draw() {
    const W = +w(), i0 = Math.min(s(), nd - W), i1 = i0 + W, x = [];
    for (let i = i0; i < i1; i++) for (let h = 0; h < 24; h++) x.push(`${d.d[i]} ${String(h).padStart(2, '0')}:00`);
    const series = k => { const r = []; for (let i = i0; i < i1; i++) r.push(...d[k][i]); return r; };
    const tr = [{ x, y: series('obs'), mode: 'lines', name: T.lmod.obs, line: { color: COL.obs, width: 2.5 } }], lines = [];
    m().forEach(k => {
      tr.push({ x, y: series(k), mode: 'lines', name: T.lmod[k], line: { color: COL[k], width: 1.6, dash: k === 'snaive' ? 'dash' : 'solid' } });
      lines.push(T.ld(T.lmod[k], pct(mape(k, i0, i1), 2), pct(full[k], 2)));
    });
    plot('ld-chart', tr, layout({ title: ttl(T.ldTitle(fd(d.d[i0]), fd(d.d[i1 - 1]))), xaxis: AX(W === 1 ? T.hourUTC : '', { type: 'date', tickformat: W === 1 ? '%H:%M' : '%d %b %Y' }), yaxis: AX(T.mw) }));
    out('ld-out', lines.length ? lines : ['–']);
  }
  draw();
  P('ld-play', 'ld-d', range(0, nd - 14, 7), 500);
});

// ---------------------------------------------------------------- 5. error by hour of the day, weekdays / weekends, seasons
safe('load-errors', () => {
  const d = D.ld, K = ['snaive', 'ridge', 'lstm', 'gru'];
  const SEAS = { all: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], win: [12, 1, 2], spr: [3, 4, 5], sum: [6, 7, 8], aut: [9, 10, 11] };
  const f = bindRadios('le-f', draw), s = bindSelect('le-s', draw), m = bindToggles('le-m', draw);
  function draw() {
    const ff = f(), months = SEAS[s()];
    const idx = d.d.map((x, i) => i).filter(i => { const w = dow(d.d[i]), mo = +d.d[i].slice(5, 7); return months.includes(mo) && (ff === 'all' || (ff === 'wd' ? w >= 1 && w <= 5 : w === 0 || w === 6)); });
    const tr = [], lines = [], hrs = range(0, 23, 1);
    m().forEach(k => {
      const mae = hrs.map(h => idx.reduce((a, i) => a + Math.abs(d[k][i][h] - d.obs[i][h]), 0) / idx.length);
      const all = mae.reduce((a, b) => a + b, 0) / 24, mp = idx.reduce((a, i) => a + d[k][i].reduce((q, v, h) => q + Math.abs(v - d.obs[i][h]) / d.obs[i][h], 0), 0) / (24 * idx.length);
      const hw = mae.indexOf(Math.max(...mae));
      tr.push({ x: hrs, y: mae, mode: 'lines+markers', name: T.lmod[k], line: { color: COL[k], width: 2, dash: k === 'snaive' ? 'dash' : 'solid' }, marker: { size: 4 } });
      lines.push(T.le(T.lmod[k], nf(all, 0), pct(mp, 2), String(hw).padStart(2, '0'), nf(mae[hw], 0)));
    });
    plot('le-chart', tr, layout({ title: ttl(T.leTitle(nf(idx.length, 0))), xaxis: AX(T.hourUTC, { dtick: 2 }), yaxis: AX(T.mae, { rangemode: 'tozero' }) }));
    out('le-out', lines.length ? lines : ['–']);
  }
  draw();
});
})();
