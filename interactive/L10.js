// Lecture 10 interactive page: reinforcement learning (battery dispatch, trading, hedging). Inputs in L10_data.js
// (lectures/L10/build_interactive_L10.py), readouts in L10_text_en.js / L10_text_ro.js.
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { C, mean, sdev, quantile, nf, $, AX, layout, ttl, plot, out, vline, bindSlider, bindSelect, bindToggles, bindRadios, Player, range, safe } = window.NNI;
const COL = { charge: '#1baf7a', idle: '#c8c8c8', dis: '#e34948', pf: '#2a78d6', model: '#eb6834', alt: '#eda100', bench: '#1baf7a', muted: '#8a8a8a', c7: '#6250d6' };
const fmtD = s => { const [y, m, d] = s.split('-').map(Number); return `${d} ${T.months[m - 1]} ${y}`; };
// legend entries of the action colours (the heatmaps have no colour bar)
const actLegend = () => [0, 1, 2].map(i => ({ x: [null], y: [null], type: 'scatter', mode: 'markers', name: T.act[i],
  marker: { symbol: 'square', size: 11, color: [COL.charge, COL.idle, COL.dis][i] }, hoverinfo: 'skip' }));

// ---------------------------------------------------------------- battery model (lecture: 1 MWh per hour, start empty at 0:00)
// cash flow of action a at price p: discharge sells sqrt(eta) MWh, charge buys 1/sqrt(eta) MWh
function reward(p, a, eta) { const e = Math.sqrt(eta); return a > 0 ? e * p : a < 0 ? -p / e : 0; }
// exact backward induction: V[h][s] (h = 0..24), A[h][s] optimal action; ties resolved as in analysis_L10.py (charge, idle, discharge)
function dp(p, E, eta) {
  const V = Array.from({ length: 25 }, () => new Array(E + 1).fill(0)), A = Array.from({ length: 24 }, () => new Array(E + 1).fill(0));
  for (let h = 23; h >= 0; h--) {
    for (let s = 0; s <= E; s++) {
      let best = -Infinity, arg = 0;
      for (const a of [-1, 0, 1]) {
        const s2 = s - a;
        if (s2 < 0 || s2 > E) continue;
        const q = reward(p[h], a, eta) + V[h + 1][s2];
        if (q > best) { best = q; arg = a; }
      }
      V[h][s] = best; A[h][s] = arg;
    }
  }
  return { V, A };
}
function schedule(A) { let s = 0; const acts = [], soc = [0]; for (let h = 0; h < 24; h++) { const a = A[h][s]; acts.push(a); s -= a; soc.push(s); } return { acts, soc }; }
const H24 = Array.from({ length: 24 }, (_, i) => i);

// state shared by the first two blocks (the second one follows the day chosen in the first)
const day = { z: 'RO', date: '2026-06-30', E: 2, eta: 0.85 };
const followers = [];

// ---------------------------------------------------------------- 1. one real day solved by dynamic programming
safe('dp-day', () => {
  const zEl = $('dd-z'), dEl = $('dd-d');
  let dates = Object.keys(D.pr.RO);
  const setRange = (want) => {
    dates = Object.keys(D.pr[zEl.value]);
    let i = dates.indexOf(want);
    if (i < 0) { i = dates.findIndex(x => x >= want); if (i < 0) i = dates.length - 1; }
    dEl.max = dates.length - 1; dEl.value = i;
  };
  setRange(day.date);
  const z = bindSelect('dd-z', () => { setRange(dates[+dEl.value]); $('dd-d-v').textContent = fmtD(dates[+dEl.value]); draw(); });
  const di = bindSlider('dd-d', v => fmtD(dates[v] || dates[0]), draw);
  const eta = bindSlider('dd-e', v => v + '%', draw), E = bindSlider('dd-c', v => nf(v, 0) + ' MWh', draw);
  function draw() {
    const zz = z(), dt = dates[di()], p = D.pr[zz][dt], e = eta() / 100, cap = E();
    Object.assign(day, { z: zz, date: dt, E: cap, eta: e });
    const { V, A } = dp(p, cap, e), { acts, soc } = schedule(A);
    const col = acts.map(a => (a < 0 ? COL.charge : a > 0 ? COL.dis : COL.idle));
    const tr = [
      { x: H24, y: p, type: 'bar', marker: { color: col }, name: T.price, showlegend: false, customdata: acts.map(a => T.act[a + 1]),
        hovertemplate: '%{x}:00 · %{y:.2f} EUR/MWh · %{customdata}<extra></extra>' },
      { x: [null], y: [null], type: 'bar', marker: { color: COL.charge }, name: T.act[0] },
      { x: [null], y: [null], type: 'bar', marker: { color: COL.dis }, name: T.act[2] },
      { x: [null], y: [null], type: 'bar', marker: { color: COL.idle }, name: T.act[1] },
      { x: range(0, 24, 1).map(h => h - 0.5), y: soc, yaxis: 'y2', mode: 'lines', line: { shape: 'hv', color: C.navy, width: 2 }, name: T.stored,
        hovertemplate: '%{y} MWh<extra></extra>' }
    ];
    plot('dd-chart', tr, layout({ title: ttl(T.ddTitle(T.zone[zz], fmtD(dt))), xaxis: AX(T.hourCET, { dtick: 3, range: [-0.7, 23.7] }),
      yaxis: AX(T.price), yaxis2: AX(T.stored, { overlaying: 'y', side: 'right', range: [0, Math.max(cap, 1) * 1.05], showgrid: false, dtick: 1 }),
      bargap: 0.15, margin: { l: 56, r: 56, t: 30, b: 40 } }));
    const lo = Math.min(...p), hi = Math.max(...p), nc = acts.filter(a => a < 0).length, nd = acts.filter(a => a > 0).length;
    out('dd-out', [T.dd(T.zone[zz], fmtD(dt), nf(V[0][0], 0), nc, nd),
      T.dd2(nf(lo, 2), p.indexOf(lo), nf(hi, 2), p.indexOf(hi), nf(hi - lo, 2), nf(cap, 0), nf(100 * e, 0), nf(V[0][0] / cap, 0))]);
    followers.forEach(f => f());
  }
  draw();
  new Player('dd-play', dEl, range(0, dates.length - 1, 1), 350);
});

// ---------------------------------------------------------------- 2. backward induction step by step (Bellman equation)
safe('dp-value', () => {
  const k = bindSlider('dv-k', v => T.step(v), draw), view = bindSelect('dv-v', draw);
  function draw() {
    const kk = k(), h0 = 24 - kk, { V, A } = dp(D.pr[day.z][day.date], day.E, day.eta), shadow = view() === 'mwh';
    const hrs = range(0, 24, 1), tr = [];
    const nS = shadow ? day.E : day.E + 1;
    for (let s = 0; s < nS; s++) {
      const y = hrs.map(h => (h < h0 ? null : shadow ? V[h][s + 1] - V[h][s] : V[h][s] / 1000));
      tr.push({ x: hrs, y, mode: 'lines+markers', marker: { size: 4 }, line: { color: window.NNI.SERIES[s % 8], width: 2 },
        name: shadow ? T.dvShadowName(s, s + 1) : T.dvName(s), hovertemplate: 'h = %{x}: %{y:.' + (shadow ? 1 : 3) + 'f}<extra></extra>' });
    }
    plot('dv-chart', tr, layout({ title: ttl(shadow ? T.dvShadow : T.dvTitle), xaxis: AX(T.hourH, { dtick: 3, range: [-0.5, 24.5] }),
      yaxis: AX(shadow ? T.eurMWh : T.thEur, { rangemode: 'tozero' }), shapes: kk < 24 ? [vline(h0, C.grey, 'dash')] : [] }));
    const z = [], txt = [];
    for (let s = 0; s <= day.E; s++) { z.push(H24.map(h => (h < h0 ? null : A[h][s]))); txt.push(H24.map(h => (h < h0 ? '' : T.act[A[h][s] + 1]))); }
    plot('dv-pol', [{ z, x: H24, y: range(0, day.E, 1), type: 'heatmap', zmin: -1, zmax: 1, showscale: false, text: txt, xgap: 1, ygap: 1,
      colorscale: [[0, COL.charge], [0.33, COL.charge], [0.33, COL.idle], [0.67, COL.idle], [0.67, COL.dis], [1, COL.dis]],
      hovertemplate: 'h = %{x}, s = %{y} MWh: %{text}<extra></extra>' }].concat(actLegend()),
    layout({ title: ttl(T.dvPol), xaxis: AX(T.hourH, { dtick: 3, range: [-0.5, 23.5] }), yaxis: AX(T.storedS, { dtick: 1 }) }));
    const h = Math.min(h0, 23), lines = [T.dvDay(T.zone[day.z], fmtD(day.date), nf(day.E, 0), nf(100 * day.eta, 0))];
    if (kk === 0) lines.push(T.dv0);
    else lines.push(T.dv(h0, nf(V[h0][0], 0), day.E, nf(V[h0][day.E], 0), nf((V[h0][day.E] - V[h0][0]) / day.E, 2), A[h][0] !== undefined ? T.act[A[h][0] + 1] : '–'));
    if (kk === 24) lines.push(T.dvDone(nf(V[0][0], 0)));
    out('dv-out', lines);
  }
  followers.push(draw);
  draw();
  new Player('dv-play', $('dv-k'), range(0, 24, 1), 450);
});

// ---------------------------------------------------------------- 3. the learned Q-table
safe('q-policy', () => {
  const s = bindRadios('qp-s', draw), h = bindSlider('qp-h', v => v + ':00', draw), b = bindSelect('qp-b', draw);
  const feas = (ss, a) => ss - a >= 0 && ss - a <= 2;
  const greedy = (hh, bb, ss) => { let best = -Infinity, arg = 0; [-1, 0, 1].forEach((a, k) => { if (feas(ss, a) && D.q[hh][bb][ss][k] > best) { best = D.q[hh][bb][ss][k]; arg = a; } }); return arg; };
  function draw() {
    const ss = +s(), hh = h(), bb = +b();
    const z = [0, 1, 2, 3].map(q => H24.map(x => greedy(x, q, ss)));
    const txt = z.map(row => row.map(a => T.act[a + 1]));
    plot('qp-map', [{ z, x: H24, y: T.quart, type: 'heatmap', zmin: -1, zmax: 1, showscale: false, text: txt, xgap: 1, ygap: 1,
      colorscale: [[0, COL.charge], [0.33, COL.charge], [0.33, COL.idle], [0.67, COL.idle], [0.67, COL.dis], [1, COL.dis]],
      hovertemplate: '%{x}:00, %{y}: %{text}<extra></extra>' }].concat(actLegend()),
    layout({ title: ttl(T.qpTitle(ss)), xaxis: AX(T.hourCET, { dtick: 3 }), yaxis: AX('', { automargin: true }),
      shapes: [{ type: 'rect', xref: 'x', yref: 'y', x0: hh - 0.5, x1: hh + 0.5, y0: bb - 0.5, y1: bb + 0.5, line: { color: C.navy, width: 3 } }],
      margin: { l: 120, r: 12, t: 30, b: 40 } }));
    const qv = D.q[hh][bb][ss], ok = [-1, 0, 1].map(a => feas(ss, a)), g = greedy(hh, bb, ss);
    plot('qp-bar', [{ x: T.act, y: qv.map((v, i) => (ok[i] ? v : null)), type: 'bar',
      marker: { color: [-1, 0, 1].map(a => (a === g ? [COL.charge, C.navy, COL.dis][a + 1] : C.band)) },
      text: qv.map((v, i) => (ok[i] ? nf(v, 1) : T.masked)), textposition: 'outside', cliponaxis: false, hovertemplate: '%{x}: %{y:.1f} EUR<extra></extra>' }],
    layout({ title: ttl(T.qpBar(hh, T.quart[bb], ss)), yaxis: AX(T.qEur), xaxis: AX(''), showlegend: false }));
    const row = z[bb];
    out('qp-out', [T.qp(hh, T.quart[bb], ss, ok[0] ? nf(qv[0], 1) : T.masked, nf(qv[1], 1), ok[2] ? nf(qv[2], 1) : T.masked, T.act[g + 1]),
      T.qp2(T.quart[bb], row.filter(a => a < 0).length, row.filter(a => a > 0).length)]);
  }
  draw();
  new Player('qp-play', $('qp-h'), range(0, 23, 1), 500);
});

// ---------------------------------------------------------------- 4. test years: cumulative revenue of the agents
safe('battery-test', () => {
  const B = D.bt, months = [...new Set(B.d.map(x => x.slice(0, 7)))];
  const mEl = $('bt-m'); mEl.max = months.length - 1;
  const k = bindToggles('bt-k', draw), seed = bindSelect('bt-seed', draw), v = bindSelect('bt-v', draw);
  const m = bindSlider('bt-m', i => { const [y, mo] = months[i].split('-').map(Number); return `${T.months[mo - 1]} ${y}`; }, draw);
  const avg = A => A[0].map((_, i) => mean(A.map(r => r[i])));
  function draw() {
    const sd = seed(), i0 = B.d.findIndex(x => x.slice(0, 7) >= months[m()]), cap = v() === 'cap';
    const ser = { pf: B.pf, nv: B.nv, q: sd === 'avg' ? avg(B.q) : B.q[+sd], dqn: sd === 'avg' ? avg(B.dqn) : B.dqn[+sd] };
    const x = B.d.slice(i0), cs = a => { let c = 0; return a.slice(i0).map(y => (c += y)); }, pfc = cs(ser.pf);
    const tr = [], lines = [], on = k();
    ['pf', 'dqn', 'q', 'nv'].forEach(key => {
      if (!on.includes(key)) return;
      const c = cs(ser[key]);
      tr.push({ x, y: cap ? c.map((y, i) => 100 * y / pfc[i]) : c.map(y => y / 1000), mode: 'lines', name: T.btName[key] + (key === 'q' || key === 'dqn' ? T.seedTag(sd) : ''),
        line: { color: { pf: COL.pf, dqn: COL.model, q: COL.alt, nv: COL.bench }[key], width: 2 } });
      const tot = c[c.length - 1];
      lines.push(T.bt(T.btName[key], nf(tot / 1000, 1), nf(100 * tot / pfc[pfc.length - 1], 1), nf(tot / c.length, 0)));
    });
    if (sd !== 'avg' || on.length) {
      const dq = ser.dqn.slice(i0), nv = ser.nv.slice(i0);
      lines.push(T.btWin(nf(100 * dq.filter((y, i) => y > nv[i]).length / dq.length, 0), x.length, fmtD(x[0]), fmtD(x[x.length - 1])));
    }
    plot('bt-chart', tr, layout({ title: ttl(cap ? T.btCap : T.btCum), xaxis: AX('', { type: 'date' }), yaxis: AX(cap ? T.pctPF : T.thEur, cap ? { range: [0, 105] } : { rangemode: 'tozero' }) }));
    out('bt-out', lines.length ? lines : ['–']);
  }
  draw();
  new Player('bt-play', mEl, range(0, months.length - 1, 1), 600);
});

// ---------------------------------------------------------------- 5. PPO trading on the S&P 500
safe('trading', () => {
  const R = D.tr, n = R.r.length;
  const k = bindToggles('tq-k', draw), seed = bindSelect('tq-seed', draw), c = bindSlider('tq-c', v => nf(v, 0) + ' bp', draw),
    y = bindSlider('tq-y', v => v, draw);
  // daily log return of a position path chosen at the close (cost c per unit of position changed; flat before the window)
  const perf = (pos, i0, cost) => { const r = []; let prev = 0; for (let i = i0; i < n; i++) { r.push(pos[i] * R.r[i] - cost * Math.abs(pos[i] - prev)); prev = pos[i]; } return r; };
  const stats = x => { let eq = 1, pk = 1, dd = 0; x.forEach(v => { eq *= Math.exp(v); pk = Math.max(pk, eq); dd = Math.min(dd, eq / pk - 1); });
    const m = mean(x), sdv = Math.sqrt(x.reduce((a, v) => a + (v - m) ** 2, 0) / x.length);
    return { ret: 100 * (Math.exp(m * 252) - 1), vol: 100 * sdv * Math.sqrt(252), sh: sdv > 0 ? m / sdv * Math.sqrt(252) : 0, dd: 100 * dd }; };
  const trades = (pos, i0) => { let t = 0, prev = 0; for (let i = i0; i < n; i++) { t += Math.abs(pos[i] - prev); prev = pos[i]; } return t; };
  function roll(a, w) { const r = []; let s = 0; for (let i = 0; i < a.length; i++) { s += a[i]; if (i >= w) s -= a[i - w]; r.push(i >= w - 1 ? s / w : null); } return r; }
  function draw() {
    const cost = c() / 1e4, i0 = R.d.findIndex(d => d >= String(y())), sd = seed(), x = R.d.slice(i0), on = k(), tr = [], lines = [];
    const ones = new Array(n).fill(1);
    const ppoR = sd === 'avg' ? (() => { const A = R.pos.map(p => perf(p, i0, cost)); return A[0].map((_, i) => mean(A.map(r => r[i]))); })() : perf(R.pos[+sd], i0, cost);
    const S = { bh: perf(ones, i0, cost), ma: perf(R.ma, i0, cost), ppo: ppoR };
    const col = { bh: COL.pf, ma: COL.bench, ppo: COL.model };
    ['bh', 'ma', 'ppo'].forEach(key => {
      if (!on.includes(key)) return;
      let eq = 1;
      tr.push({ x, y: S[key].map(v => (eq *= Math.exp(v))), mode: 'lines', name: T.tqName[key] + (key === 'ppo' ? T.seedTag(sd) : ''), line: { color: col[key], width: 2 } });
      const st = stats(S[key]), nt = key === 'bh' ? 1 : key === 'ma' ? trades(R.ma, i0) : sd === 'avg' ? mean(R.pos.map(p => trades(p, i0))) : trades(R.pos[+sd], i0);
      lines.push(T.tq(T.tqName[key], nf(st.ret, 1), nf(st.vol, 1), nf(st.sh, 2), nf(st.dd, 1), nf(nt, 0)));
    });
    if (sd === 'avg' && on.includes('ppo')) {
      const shs = R.pos.map(p => stats(perf(p, i0, cost)).sh);
      lines.push(T.tqSeeds(nf(mean(shs), 2), nf(Math.min(...shs), 2), nf(Math.max(...shs), 2)));
    }
    plot('tq-eq', tr, layout({ title: ttl(T.tqTitle(nf(c(), 0))), xaxis: AX('', { type: 'date' }), yaxis: AX(T.usd) }));
    const ex = sd === 'avg' ? R.pos[0].map((_, i) => mean(R.pos.map(p => p[i]))) : R.pos[+sd];
    const et = [];
    if (on.includes('ppo')) et.push({ x, y: roll(ex.slice(i0), 21).map(v => (v === null ? null : 100 * v)), mode: 'lines', name: T.tqName.ppo, line: { color: COL.model, width: 1.8 } });
    if (on.includes('ma')) et.push({ x, y: roll(R.ma.slice(i0), 21).map(v => (v === null ? null : 100 * v)), mode: 'lines', name: T.tqName.ma, line: { color: COL.bench, width: 1.2 } });
    plot('tq-ex', et, layout({ title: ttl(T.tqEx), xaxis: AX('', { type: 'date' }), yaxis: AX(T.pctInv, { range: [-3, 103] }) }));
    out('tq-out', lines.length ? lines : ['–']);
  }
  draw();
  new Player('tq-play', $('tq-c'), range(0, 30, 1), 300);
});

// ---------------------------------------------------------------- 6. deep hedging against the Black-Scholes delta
safe('hedging', () => {
  const Hd = D.hd;
  const k = bindToggles('hg-k', draw), seed = bindSelect('hg-seed', draw), reg = bindSelect('hg-r', draw), al = bindSlider('hg-a', v => v + '%', draw);
  // expected shortfall as in analysis_L10.py: mean of the losses at or above the alpha-quantile (linear interpolation)
  const es = (loss, a) => { const q = quantile(loss, a); const t = loss.filter(v => v >= q); return mean(t); };
  const edges = range(-8, 4, 0.2);
  function hist(vals, w) { const c = new Array(edges.length - 1).fill(0); vals.forEach(v => { const x = Math.min(Math.max(v, -8), 3.9999); c[Math.min(Math.floor((x + 8) / 0.2), c.length - 1)] += w; }); return c; }
  function draw() {
    const a = al() / 100, rg = reg(), sd = seed(), on = k();
    const m = Hd.vix.map(v => (rg === 'all' ? true : rg === 'lo' ? v < 20 : v >= 20));
    const sel = arr => arr.filter((_, i) => m[i]);
    const ser = { nh: [sel(Hd.nh)], bs: [sel(Hd.bs)], dh: sd === 'avg' ? Hd.dh.map(sel) : [sel(Hd.dh[+sd])] };
    const col = { nh: COL.muted, bs: COL.bench, dh: COL.model }, mids = edges.slice(0, -1).map(e => e + 0.1);
    const tr = [], shapes = [], lines = [], bars = { x: [], es: [], sd: [], col: [] };
    ['nh', 'bs', 'dh'].forEach(key => {
      if (!on.includes(key)) return;
      const S = ser[key], w = 1 / S.length;
      const cnt = S.reduce((acc, v) => { const h = hist(v, w); return acc.map((x, i) => x + h[i]); }, new Array(edges.length - 1).fill(0));
      tr.push(key === 'nh' ? { x: mids, y: cnt, type: 'bar', marker: { color: col[key], opacity: 0.45 }, name: T.hgName.nh, width: 0.2 }
        : { x: mids, y: cnt, mode: 'lines', line: { shape: 'hvh', color: col[key], width: 2 }, name: T.hgName[key] + (key === 'dh' ? T.seedTag(sd) : '') });
      const e = mean(S.map(v => es(v.map(x => -x), a))), s = mean(S.map(v => sdev(v) * Math.sqrt((v.length - 1) / v.length))), mu = mean(S.map(v => mean(v)));
      shapes.push(vline(Math.max(-e, -8), col[key], 'dot'));
      bars.x.push(T.hgName[key]); bars.es.push(e); bars.sd.push(s); bars.col.push(col[key]);
      lines.push(T.hg(T.hgName[key], nf(mu, 2), nf(s, 2), nf(100 * a, 0), nf(e, 2)));
    });
    plot('hg-hist', tr, layout({ title: ttl(T.hgTitle(nf(100 * a, 0))), xaxis: AX(T.pnl, { range: [-8.2, 4.2] }), yaxis: AX(T.days), shapes, barmode: 'overlay' }));
    const cats = [T.esName(nf(100 * a, 0)), T.sdName];
    plot('hg-es', bars.x.map((nm, i) => ({ x: cats, y: [bars.es[i], bars.sd[i]], type: 'bar', name: nm, marker: { color: bars.col[i] },
      text: [nf(bars.es[i], 2), nf(bars.sd[i], 2)], textposition: 'outside', cliponaxis: false, hovertemplate: '%{x}: %{y:.2f}%<extra>' + nm + '</extra>' })),
    layout({ title: ttl(T.hgEs), yaxis: AX(T.pctS0, { rangemode: 'tozero' }), barmode: 'group' }));
    lines.push(T.hgN(sel(Hd.nh).length, T.reg[rg]));
    out('hg-out', lines);
  }
  draw();
  new Player('hg-play', $('hg-a'), range(90, 99, 1), 600);
});
})();
