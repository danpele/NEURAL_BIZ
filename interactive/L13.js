// Lecture 13 interactive page: deep learning and foundation models for energy and financial time series.
// Inputs in L13_data.js (lectures/L13/build_interactive_L13.py): test-year forecasts, Chronos-2 quantiles, Value-at-Risk series.
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, mean, sdev, quantile, ncdf, nf, pct, $, AX, layout, ttl, plot, out, hline, bindSlider, bindSelect, bindToggles,
  bindRadios, Player, range, safe } = window.NNI;
const players = [];
const P = (id, input, vals, ms) => players.push(new Player(id, $(input), vals, ms));
const COL = { naive: C.grey, lear: '#1baf7a', dnn: '#eb6834', nbeats: '#6250d6', chronos2: '#eda100', chronos2_cov: '#008300',
  bolt: '#e87ba4', timesfm: '#e34948', ens: C.navy, actual: '#2a78d6' };
const NM = T.names;
const ND = D.days.length, H = 24;
const DAY = {};
D.days.forEach((d, i) => { DAY[d] = i; });
const MS = 864e5;
const addDays = (s, k) => new Date(Date.parse(s + 'T00:00:00Z') + k * MS).toISOString().slice(0, 10);
const MON0 = '2025-11-03';                                 // first Monday of the test year
const weekDays = w => { const m = addDays(MON0, 7 * w); return range(0, 6, 1).map(k => addDays(m, k)).filter(d => d in DAY); };
const hx = (d, h) => `${d} ${String(h).padStart(2, '0')}:00`;
const fmtP = p => (p < 1e-4 ? (LANG === 'ro' ? '< 0,0001' : '< 0.0001') : nf(p, 4));

// ---------------------------------------------------------------- 1. one week of forecasts
safe('forecast-week', () => {
  const w = bindSlider('fw-w', v => T.date(addDays(MON0, 7 * v)), draw), ms = bindToggles('fw-m', draw);
  const yearMae = m => mean(D.f[m].map((v, i) => Math.abs(v - D.act[i])));
  function draw() {
    const days = weekDays(w()), sel = ms();
    const idx = days.flatMap(d => range(0, 23, 1).map(h => DAY[d] * H + h));
    const x = days.flatMap(d => range(0, 23, 1).map(h => hx(d, h)));
    const tr = [{ x, y: idx.map(i => D.act[i]), mode: 'lines', name: T.actual, line: { color: COL.actual, width: 2.4 } }];
    sel.forEach(m => tr.push({ x, y: idx.map(i => D.f[m][i]), mode: 'lines', name: NM[m], line: { color: COL[m], width: 1.4 } }));
    plot('fw-chart', tr, layout({ title: ttl(T.fwTitle(T.date(days[0]), T.date(days[days.length - 1]))), xaxis: AX('', { type: 'date' }),
      yaxis: AX(T.price), margin: { l: 60, r: 12, t: 34, b: 40 } }));
    const wm = {};
    D.models.forEach(m => { wm[m] = mean(idx.map(i => Math.abs(D.f[m][i] - D.act[i]))); });
    const rk = [...D.models].sort((a, b) => wm[a] - wm[b]);
    out('fw-out', sel.length ? sel.map(m => T.fw(NM[m], nf(wm[m], 2), rk.indexOf(m) + 1)).concat([T.fwYear(NM[sel[0]], nf(yearMae(sel[0]), 2))]) : [T.fwNone]);
  }
  draw();
  P('fw-play', 'fw-w', range(0, 51, 1), 700);
});

// ---------------------------------------------------------------- 2. MAE by hour of the day and season
safe('mae-by-hour', () => {
  const ms = bindToggles('mh-m', draw), se = bindSelect('mh-s', draw);
  const SEAS = { all: null, win: [12, 1, 2], spr: [3, 4, 5], sum: [6, 7, 8], aut: [9, 10] };
  function draw() {
    const s = se(), sel = ms(), mo = SEAS[s];
    const di = D.days.map((d, i) => i).filter(i => !mo || mo.includes(+D.days[i].slice(5, 7)));
    const byh = m => range(0, 23, 1).map(h => mean(di.map(i => Math.abs(D.f[m][i * H + h] - D.act[i * H + h]))));
    const all = m => mean(di.flatMap(i => range(0, 23, 1).map(h => Math.abs(D.f[m][i * H + h] - D.act[i * H + h]))));
    const tr = sel.map(m => ({ x: range(0, 23, 1), y: byh(m), mode: 'lines+markers', name: NM[m], line: { color: COL[m], width: 2 }, marker: { size: 4 } }));
    plot('mh-chart', tr, layout({ title: ttl(T.mhTitle(T.season[s])), xaxis: AX(T.hour, { dtick: 3 }), yaxis: AX(T.mae, { rangemode: 'tozero' }) }));
    const tot = D.models.map(all);
    plot('mh-bar', [{ type: 'bar', orientation: 'h', y: D.models.map(m => NM[m]), x: tot, marker: { color: D.models.map(m => sel.includes(m) ? COL[m] : '#d5dbe6') },
      text: tot.map(v => nf(v, 2)), textposition: 'auto', hoverinfo: 'x+y', showlegend: false }],
      layout({ title: ttl(T.mhBar(T.season[s])), xaxis: AX(T.mae), yaxis: AX('', { autorange: 'reversed' }), margin: { l: 170, r: 12, t: 30, b: 40 } }));
    out('mh-out', sel.map(m => {
      const v = byh(m), hw = v.indexOf(Math.max(...v)), hb = v.indexOf(Math.min(...v));
      return T.mh(NM[m], nf(all(m), 2), String(hw).padStart(2, '0'), nf(v[hw], 2), String(hb).padStart(2, '0'), nf(v[hb], 2), di.length);
    }));
  }
  draw();
});

// ---------------------------------------------------------------- 3. Diebold-Mariano test on daily losses
safe('dm-test', () => {
  const a = bindSelect('dm-a', draw), b = bindSelect('dm-b', draw), l = bindRadios('dm-l', draw);
  function draw() {
    const A = a(), B = b(), sq = l() === 'sq';
    const loss = (m, i) => { let s = 0; for (let h = 0; h < H; h++) { const e = D.f[m][i * H + h] - D.act[i * H + h]; s += sq ? e * e / 1000 : Math.abs(e); } return s; };
    const d = D.days.map((_, i) => loss(A, i) - loss(B, i));
    let c = 0;
    const cum = d.map(v => (c += v));
    plot('dm-d', [{ type: 'bar', x: D.days, y: d, marker: { color: d.map(v => v > 0 ? COL[B] : C.grey) }, showlegend: false }],
      layout({ title: ttl(T.dmD(NM[A], NM[B])), xaxis: AX(T.day, { type: 'date' }), yaxis: AX(T.dmY(l())), bargap: 0 }));
    plot('dm-c', [{ x: D.days, y: cum, mode: 'lines', line: { color: COL[B], width: 2 }, showlegend: false }],
      layout({ title: ttl(T.dmC), xaxis: AX(T.day, { type: 'date' }), yaxis: AX(T.dmY(l())), shapes: [hline(0, C.grey, 'dot')] }));
    if (A === B) { out('dm-out', [T.dmSame]); return; }
    const md = mean(d), s = md / (sdev(d) / Math.sqrt(d.length)), p = 1 - ncdf(s);
    out('dm-out', [T.dm(NM[A], NM[B], nf(md, 2), nf(s, 2), fmtP(p), d.length), p < 0.05 ? T.dmYes(NM[B], NM[A]) : T.dmNo(NM[B], NM[A])]);
  }
  draw();
});

// ---------------------------------------------------------------- 4. context length of Chronos-2
safe('context-length', () => {
  const ci = bindSlider('cl-c', v => T.ctx(D.ctxs[v]), draw), w = bindSlider('cl-w', v => v, draw);
  const ma = (a, k) => a.map((_, i) => (i + 1 >= k ? mean(a.slice(i + 1 - k, i + 1)) : null));
  function draw() {
    const c = D.ctxs[ci()], cs = String(c), m = D.ctxs.map(k => mean(D.ctx_day[k]));
    plot('cl-bar', [{ type: 'bar', x: D.ctxs.map(k => T.ctx(k)), y: m, marker: { color: D.ctxs.map(k => k === c ? COL.chronos2 : '#d5dbe6') },
      text: m.map(v => nf(v, 2)), textposition: 'auto', name: 'Chronos-2' },
      { x: D.ctxs.map(k => T.ctx(k)), y: D.ctxs.map(() => D.lear_mae), mode: 'lines', name: T.lear, line: { color: COL.lear, dash: 'dash', width: 1.5 } }],
      layout({ title: ttl(T.clBar), xaxis: AX(T.ctxAx), yaxis: AX(T.mae, { rangemode: 'tozero' }) }));
    const tr = [];
    if (c !== 2048) tr.push({ x: D.days, y: ma(D.ctx_day['2048'], w()), mode: 'lines', name: T.ctx(2048), line: { color: C.grey, width: 1.4, dash: 'dot' } });
    tr.push({ x: D.days, y: ma(D.ctx_day[cs], w()), mode: 'lines', name: T.ctx(c), line: { color: COL.chronos2, width: 2 } });
    plot('cl-roll', tr, layout({ title: ttl(T.clRoll(w())), xaxis: AX(T.day, { type: 'date' }), yaxis: AX(T.mae, { rangemode: 'tozero' }) }));
    const mm = mean(D.ctx_day[cs]), base = mean(D.ctx_day['2048']);
    const better = c === 2048 ? '–' : pct(mean(D.ctx_day[cs].map((v, i) => (v < D.ctx_day['2048'][i] ? 1 : 0))), 0);
    const hv = D.ctx_hour[cs], hw = hv.indexOf(Math.max(...hv));
    out('cl-out', [T.cl(T.ctx(c), nf(mm, 2), (mm - base >= 0 ? '+' : '') + nf(mm - base, 2), better), T.clH(String(hw).padStart(2, '0'), nf(hv[hw], 2))]);
  }
  draw();
  P('cl-play', 'cl-c', [0, 1, 2, 3], 1200);
});

// ---------------------------------------------------------------- 5. Chronos-2 quantiles, coverage and split-conformal correction
safe('quantile-fan', () => {
  const Q = D.q, QI = {};
  Q.days.forEach((d, i) => { QI[d] = i; });
  const KEYS = { 50: ['q25', 'q75'], 80: ['q10', 'q90'], 90: ['q05', 'q95'] };
  const w = bindSlider('qf-w', v => T.date(addDays(MON0, 7 * v)), draw), lv = bindRadios('qf-l', draw), cf = bindToggles('qf-c', draw),
    nw = bindSlider('qf-n', v => v, draw);
  const cache = {};
  // conformal margin of each test day (rolling window of the n previous calendar days) and test-year coverage / width
  function calib(L, n) {
    const key = L + '_' + n;
    if (cache[key]) return cache[key];
    const [kl, kh] = KEYS[L], a = 1 - L / 100, lo = Q[kl], hi = Q[kh], y = Q.act;
    const score = i => Math.max(lo[i] - y[i], y[i] - hi[i]);
    const marg = {};
    let inR = 0, inC = 0, wR = 0, wC = 0, cnt = 0;
    D.days.forEach(d => {
      const j = QI[d], dn = Q.dn[j], s = [];
      for (let k = j - 1; k >= 0 && Q.dn[k] >= dn - n; k--) for (let h = 0; h < H; h++) s.push(score(k * H + h));
      const m = s.length, Qc = quantile(s, Math.min(1, Math.ceil((m + 1) * (1 - a)) / m));
      marg[d] = Qc;
      for (let h = 0; h < H; h++) {
        const i = j * H + h;
        inR += y[i] >= lo[i] && y[i] <= hi[i]; wR += hi[i] - lo[i];
        inC += y[i] >= lo[i] - Qc && y[i] <= hi[i] + Qc; wC += hi[i] - lo[i] + 2 * Qc; cnt++;
      }
    });
    return (cache[key] = { marg, cR: inR / cnt, cC: inC / cnt, wR: wR / cnt, wC: wC / cnt });
  }
  function draw() {
    const L = +lv(), conf = cf().length > 0, n = nw(), r = calib(L, n), [kl, kh] = KEYS[L];
    const days = weekDays(w());
    const idx = days.flatMap(d => range(0, 23, 1).map(h => QI[d] * H + h));
    const x = days.flatMap(d => range(0, 23, 1).map(h => hx(d, h)));
    const mg = i => (conf ? r.marg[Q.days[Math.floor(i / H)]] : 0);
    const lo = idx.map(i => Q[kl][i] - mg(i)), hi = idx.map(i => Q[kh][i] + mg(i));
    plot('qf-fan', [
      { x, y: lo, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
      { x, y: hi, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(237,161,0,0.28)', name: T.band(L, conf) },
      { x, y: idx.map(i => Q.q50[i]), mode: 'lines', name: T.median, line: { color: COL.chronos2, width: 1.4 } },
      { x, y: idx.map(i => Q.act[i]), mode: 'lines', name: T.actual, line: { color: COL.actual, width: 2 } }],
    layout({ title: ttl(T.qfTitle(L, T.date(days[0]), T.date(days[days.length - 1]))), xaxis: AX('', { type: 'date' }), yaxis: AX(T.price) }));
    plot('qf-cov', [{ type: 'bar', x: [T.raw, T.conf(n)], y: [100 * r.cR, 100 * r.cC], marker: { color: [C.grey, COL.chronos2] },
      text: [pct(r.cR, 1), pct(r.cC, 1)], textposition: 'auto', showlegend: false }],
    layout({ title: ttl(T.covTitle(L)), yaxis: AX(T.cov, { range: [Math.min(40, L - 15), 100] }),
      shapes: [hline(L, C.navy, 'dash')], annotations: [{ xref: 'paper', x: 1, y: L, yref: 'y', text: T.nominal, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 11 } }] }));
    const wk = mean(idx.map((i, k) => (Q.act[i] >= lo[k] && Q.act[i] <= hi[k] ? 1 : 0)));
    out('qf-out', [T.qf(L, pct(r.cR, 1), nf(r.wR, 2), pct(r.cC, 1), nf(r.wC, 2)), T.qfWeek(pct(wk, 0), L)]);
  }
  draw();
  P('qf-play', 'qf-w', range(0, 51, 1), 700);
});

// ---------------------------------------------------------------- 6. Value-at-Risk backtest with a conformal shift
safe('var-backtest', () => {
  const as = bindSelect('va-a', draw), lq = bindRadios('va-q', draw), bm = bindSelect('va-m', draw), nw = bindSlider('va-n', v => v, draw);
  function kupiec(x, n, a) {
    const p = x / n;
    const lr = x === 0 ? -2 * n * Math.log(1 - a) : -2 * ((n - x) * Math.log(1 - a) + x * Math.log(a) - (n - x) * Math.log(1 - p) - x * Math.log(p));
    return 2 * (1 - ncdf(Math.sqrt(Math.max(lr, 0))));
  }
  function draw() {
    const A = as(), s = lq(), a = s === '01' ? 0.01 : 0.05, V = D.var[A], n = nw(), i0 = V.i0, N = V.r.length - i0;
    const ch = V['ch_' + s], r = V.r;
    const sc = ch.map((v, i) => v - r[i]);
    const chc = [];
    for (let i = i0; i < r.length; i++) {
      const wv = sc.slice(i - n, i), m = wv.length;
      chc.push(ch[i] - quantile(wv, Math.min(1, Math.ceil((m + 1) * (1 - a)) / m)));
    }
    const dates = V.dates.slice(i0), rr = r.slice(i0), bmv = V[bm() + '_' + s].slice(i0);
    const hits = rr.map((v, k) => v < chc[k]);
    const cntv = arr => rr.reduce((t, v, k) => t + (v < arr[k] ? 1 : 0), 0);
    const nm = A === 'SP500' ? 'S&P 500' : 'Bitcoin';
    plot('va-chart', [
      { x: dates, y: rr, mode: 'lines', name: T.ret, line: { color: '#b8c2d6', width: 0.8 } },
      { x: dates, y: bmv, mode: 'lines', name: T.mname[bm()], line: { color: bm() === 'ch' ? COL.chronos2 : bm() === 'garch' ? '#6250d6' : '#1baf7a', width: 1, dash: 'dot' } },
      { x: dates, y: chc, mode: 'lines', name: T.chc(n), line: { color: '#eb6834', width: 1.2 } },
      { x: dates.filter((_, k) => hits[k]), y: rr.filter((_, k) => hits[k]), mode: 'markers', name: T.viol, marker: { color: '#e34948', size: 6 } }],
    layout({ title: ttl(T.vaTitle(nm, a * 100)), xaxis: AX('', { type: 'date' }), yaxis: AX(T.ret) }));
    const keys = ['hs', 'garch', 'ch'], cnts = keys.map(k => cntv(V[k + '_' + s].slice(i0))).concat([hits.filter(Boolean).length]);
    const labs = keys.map(k => T.mname[k]).concat([T.chc(n)]);
    plot('va-bar', [{ type: 'bar', x: labs, y: cnts, marker: { color: ['#1baf7a', '#6250d6', COL.chronos2, '#eb6834'] }, text: cnts.map(String), textposition: 'auto', showlegend: false }],
      layout({ title: ttl(T.vaBar(a * 100, N)), yaxis: AX(T.viols, { rangemode: 'tozero' }), shapes: [hline(a * N, C.navy, 'dash')],
        annotations: [{ xref: 'paper', x: 1, y: a * N, yref: 'y', text: T.expected + ' ' + nf(a * N, 1), showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 11 } }] }));
    out('va-out', [T.vaExp(N, nf(a * N, 1), T.date(dates[0]), T.date(dates[dates.length - 1]))].concat(
      labs.map((lb, k) => T.va(lb, cnts[k], pct(cnts[k] / N, 2), nf(kupiec(cnts[k], N, a), 3)))));
  }
  draw();
  P('va-play', 'va-n', range(50, 250, 10), 500);
});
})();
