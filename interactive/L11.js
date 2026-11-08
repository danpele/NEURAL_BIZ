// Lecture 11 interactive page: explainable AI. Inputs in L11_data.js (lectures/L11/build_interactive_L11.py).
(function () {
'use strict';
const D = window.LK_DATA, T = window.LK_TEXT;
const { LANG, C, SERIES, sum, mean, nf, pct, $, AX, layout, ttl, plot, out, vline, bindSlider, bindSelect, bindToggles, bindRadios,
  Player, range, safe } = window.NNI;
const players = [];
['wf-chart', 'pd-chart', 'dl-chart'].forEach(id => { if ($(id)) $(id).style.height = '440px'; });
const P = (id, input, vals, ms) => players.push(new Player(id, $(input), vals, ms));
const PAL = { model: '#eb6834', alt: '#eda100', bench: '#1baf7a', pos: '#e34948', neg: '#2a78d6', data: '#2a78d6', c7: '#6250d6', c5: '#e87ba4', muted: '#8a8a8a', ink: '#222222' };
const MCOL = { mlp: PAL.model, xgb: PAL.alt, lr: PAL.bench };
const F = D.feats;
const nice = f => T.feat[f] || f.replace('JOB_', 'JOB = ');
const sig = z => 1 / (1 + Math.exp(-z));
const sgn = (x, d) => (x >= 0 ? '+' : '') + nf(x, d);
const argsortDesc = a => a.map((v, i) => i).sort((i, j) => a[j] - a[i]);
const absv = a => a.map(Math.abs);
const rowWord = LANG === 'ro' ? 'rîndul' : 'row';

// ---------------------------------------------------------------- 1. SHAP waterfall of one applicant
safe('waterfall', () => {
  const W = D.wf;
  const rk = bindSlider('wf-r', v => `${v} (${rowWord} ${W.row[v - 1]})`, draw), md = bindSelect('wf-m', draw), kk = bindSlider('wf-k', v => v, draw);
  const fmtRaw = v => (Math.abs(v) >= 1000 ? nf(v, 0) : Number.isInteger(v) ? nf(v, 0) : nf(v, 1));
  function draw() {
    const i = rk() - 1, m = md(), k = kk(), phi = W.sv[m][i], b = W.base[m];
    const ord = argsortDesc(absv(phi)), js = ord.slice(0, k);
    const rest = sum(phi) - sum(js.map(j => phi[j]));
    const labs = js.map(j => `${nice(F[j])} = ${fmtRaw(W.raw[i][j])}`).concat([T.others(F.length - k)]);
    const vals = js.map(j => phi[j]).concat([rest]);
    let st = b;
    const starts = vals.map(v => { const s0 = st; st += v; return s0; });
    const tr = [true, false].map(up => {
      const ix = vals.map((v, q) => q).filter(q => (vals[q] > 0) === up);
      return { type: 'bar', orientation: 'h', y: ix.map(q => labs[q]), x: ix.map(q => vals[q]), base: ix.map(q => starts[q]),
        marker: { color: up ? PAL.pos : PAL.neg }, name: up ? T.up : T.down,
        hovertemplate: ix.map(q => `${labs[q]}: ${sgn(vals[q], 3)}<extra></extra>`) };
    });
    tr.push({ x: [null], y: [null], mode: 'lines', line: { color: PAL.muted, dash: 'dash' }, name: T.base });
    tr.push({ x: [null], y: [null], mode: 'lines', line: { color: PAL.ink }, name: T.fx });
    plot('wf-chart', tr, layout({ title: ttl(T.wfTitle(W.row[i], T.model[m])), barmode: 'overlay',
      xaxis: AX(T.logodds), yaxis: AX('', { categoryorder: 'array', categoryarray: labs, autorange: 'reversed', tickfont: { size: 10.5, color: C.navy } }),
      shapes: [vline(b, PAL.muted, 'dash'), vline(W.fx[m][i], PAL.ink)], margin: { l: 10, r: 12, t: 34, b: 44 } }));
    const top3 = ord.slice(0, 3).map(j => `${nice(F[j])} ${sgn(phi[j], 2)}`).join(', ');
    out('wf-out', [T.wf(W.row[i], pct(sig(W.fx[m][i])), nf(b, 3), pct(sig(b)), (sum(phi) >= 0 ? '+ ' : '− ') + nf(Math.abs(sum(phi)), 3), W.bad[i]),
      T.wf2(pct(sig(W.fx.mlp[i])), pct(sig(W.fx.xgb[i])), pct(sig(W.fx.lr[i]))), T.wf3(top3)]);
  }
  draw();
  P('wf-play', 'wf-r', range(1, 400, 9).concat([400]), 450);
});

// ---------------------------------------------------------------- 2. LIME: sample size, random run, kernel width
safe('lime', () => {
  const Lm = D.lime, ref = Lm.s5000[0];
  const nn = bindRadios('lm-n', draw), run = bindSlider('lm-r', v => v, draw), kw = bindSlider('lm-w', v => nf(Lm.kw[v], 2), draw);
  const top = argsortDesc(absv(ref)), top8 = top.slice(0, 8), top5 = top.slice(0, 5);
  const topSet = w => new Set(argsortDesc(absv(w)).slice(0, 5));
  const jac = (a, b) => { let c = 0; a.forEach(x => { if (b.has(x)) c++; }); return c / (a.size + b.size - c); };
  function draw() {
    const n = nn(), R = n === '500' ? Lm.s500 : Lm.s5000, r = run() - 1, w = R[r];
    const names = top8.map(j => nice(F[j]));
    const tr = [
      { type: 'bar', x: names, y: top8.map(j => w[j]), name: T.thisRun, marker: { color: n === '500' ? PAL.alt : PAL.model, opacity: 0.85 } },
      { x: [].concat(...top8.map(j => R.map(() => nice(F[j])))), y: [].concat(...top8.map(j => R.map(v => v[j]))), mode: 'markers',
        name: T.allRuns, marker: { color: PAL.ink, size: 4, opacity: 0.35 }, hoverinfo: 'y' },
      { x: names, y: top8.map(j => ref[j]), mode: 'markers', name: T.ref, marker: { symbol: 'diamond-open', size: 10, color: PAL.c7, line: { width: 2 } } }];
    plot('lm-runs', tr, layout({ title: ttl(T.runTitle(n === '500' ? '500' : T.n5000, r + 1)),
      xaxis: AX('', { tickangle: -30, tickfont: { size: 10, color: C.navy } }), yaxis: AX(T.slope) }));
    const k = kw(), x = Lm.kw;
    const tr2 = top5.map((j, q) => ({ x, y: Lm.kw_w.map(row => row[j]), mode: 'lines+markers', name: nice(F[j]), line: { color: SERIES[q], width: 2 }, marker: { size: 5 } }));
    tr2.push(...top5.map((j, q) => ({ x: [x[k]], y: [Lm.kw_w[k][j]], mode: 'markers', showlegend: false, marker: { color: SERIES[q], size: 12, line: { color: PAL.ink, width: 1.5 } }, hoverinfo: 'skip' })));
    plot('lm-kw', tr2, layout({ title: ttl(T.kwTitle), xaxis: AX(T.kw, { type: 'log', tickvals: [0.5, 1, 2, 4, 10] }), yaxis: AX(T.slope),
      shapes: [vline(Lm.kw_default, PAL.muted, 'dash'), vline(x[k], PAL.ink, 'dot')] }));
    const tRef = top[0], tThis = argsortDesc(absv(w))[0];
    const same = R.filter(v => argsortDesc(absv(v))[0] === tRef).length;
    const S = R.map(topSet), refS = topSet(ref);
    let pj = 0, np_ = 0;
    for (let a = 0; a < S.length; a++) for (let b = a + 1; b < S.length; b++) { pj += jac(S[a], S[b]); np_++; }
    out('lm-out', [T.lm(n === '500' ? '500' : T.n5000, r + 1, nice(F[tThis]), same, pct(same / 30, 0)),
      T.lm2(nf(jac(topSet(w), refS), 2), nf(pj / np_, 2)),
      T.lm3(nf(x[k], 2), nf(Lm.kw_default, 2), nice(F[tRef]), nf(Lm.kw_w[k][tRef], 3))]);
  }
  draw();
  P('lm-play', 'lm-r', range(1, 30, 1), 500);
});

// ---------------------------------------------------------------- 3. partial dependence and ICE (California housing)
safe('pdp-ice', () => {
  const ff = bindSelect('pd-f', draw), nn = bindSlider('pd-n', v => v, draw), cc = bindToggles('pd-c', draw);
  const interp = (g, y, x) => { if (x <= g[0]) return y[0]; if (x >= g[g.length - 1]) return y[y.length - 1];
    let i = 1; while (g[i] < x) i++; const f = (x - g[i - 1]) / (g[i] - g[i - 1]); return y[i - 1] + f * (y[i] - y[i - 1]); };
  function draw() {
    const f = ff(), d = D.pdp[f], n = nn(), cen = cc().includes('c'), g = d.grid;
    const sh = y => (cen ? y.map(v => v - y[0]) : y);
    const tr = [];
    d.ice.slice(0, n).forEach((y, q) => tr.push({ x: g, y: sh(y), mode: 'lines', line: { color: PAL.data, width: 1 }, opacity: 0.35,
      name: T.ice, showlegend: q === 0, legendgroup: 'ice', hoverinfo: 'skip' }));
    if (n) {
      const ox = d.own.slice(0, n), oy = d.ice.slice(0, n).map((y, q) => interp(g, y, ox[q]) - (cen ? y[0] : 0));
      tr.push({ x: ox.map(v => Math.min(Math.max(v, g[0]), g[g.length - 1])), y: oy, mode: 'markers', name: T.own, marker: { color: PAL.data, size: 5 } });
    }
    tr.push({ x: g, y: sh(d.pdp), mode: 'lines', name: T.pdp, line: { color: PAL.model, width: 3.5 } });
    plot('pd-chart', tr, layout({ title: ttl(T.fx_[f]), xaxis: AX(T.fx_[f]), yaxis: AX(cen ? T.pdC : T.pdUnit) }));
    const last = d.ice.map(y => y[y.length - 1]), m = mean(last);
    const sd = Math.sqrt(sum(last.map(v => (v - m) ** 2)) / last.length);
    const up = d.ice.filter(y => y[y.length - 1] > y[0]).length;
    out('pd-out', [T.pd(nf(d.pdp[0], 2), nf(d.pdp[d.pdp.length - 1], 2), nf(g[0], 2), nf(g[g.length - 1], 2), nf(sd, 2)), T.pd2(n, up)]);
  }
  draw();
  P('pd-play', 'pd-n', range(0, 60, 5), 450);
});

// ---------------------------------------------------------------- 4. integrated gradients: completeness against the number of steps
safe('ig-steps', () => {
  const G = D.ig, cs = bindSelect('ig-c', draw), mi = bindSlider('ig-m', v => G.M[v], draw);
  function draw() {
    const c = G.cases[+cs()], q = mi(), m = G.M[q], a = c.ig[q], full = c.ig[G.M.length - 1];
    const top = argsortDesc(absv(full)).slice(0, 10), names = top.map(j => G.feats[j]);
    plot('ig-bar', [
      { type: 'bar', x: names, y: top.map(j => a[j]), name: `m = ${m}`, marker: { color: top.map(j => (a[j] >= 0 ? PAL.pos : PAL.neg)) } },
      { x: names, y: top.map(j => full[j]), mode: 'markers', name: 'm = 256', marker: { symbol: 'line-ew-open', size: 22, color: PAL.ink, line: { width: 2 } } }],
      layout({ title: ttl(T.igTitle(m)), showlegend: true, xaxis: AX(''), yaxis: AX(T.attr) }));
    const gap = c.fx - c.fb, own = c.ig.map(v => Math.abs(sum(v) - gap) / Math.abs(gap));
    plot('ig-err', [
      { x: G.M, y: G.rel, mode: 'lines+markers', name: T.med211, line: { color: PAL.model, width: 2.5 } },
      { x: G.M, y: own, mode: 'lines+markers', name: T.thisFraud, line: { color: PAL.data, width: 1.5, dash: 'dot' } },
      { x: [m, m], y: [G.rel[q], own[q]], mode: 'markers', showlegend: false, marker: { size: 12, color: [PAL.model, PAL.data], line: { color: PAL.ink, width: 1.5 } }, hoverinfo: 'skip' }],
      layout({ title: ttl(T.relErr), xaxis: AX(T.steps, { type: 'log', tickvals: G.M }), yaxis: AX('', { type: 'log', exponentformat: 'power' }) }));
    out('ig-out', [T.ig(m, nf(sum(a), 3), nf(gap, 3), pct(own[q], 2)), T.ig2(pct(c.p, 1), pct(G.rel[q], 2))]);
  }
  draw();
  P('ig-play', 'ig-m', range(0, 8, 1), 700);
});

// ---------------------------------------------------------------- 5. deletion curves (faithfulness)
safe('deletion', () => {
  const Dl = D.del, METH = ['ig', 'shap', 'lime', 'saliency', 'random'];
  const COL = { ig: PAL.model, shap: PAL.alt, lime: PAL.c5, saliency: PAL.data, random: PAL.muted };
  const ms = bindToggles('dl-m', draw), fr = bindSlider('dl-f', v => (v ? `${v} (${pct(Dl.p[v - 1], 1)})` : '0'), draw), kk = bindSlider('dl-k', v => v, draw);
  const curve = (m, f) => (f ? Dl[m][f - 1] : Dl[m][0].map((_, k) => mean(Dl[m].map(r => r[k]))));
  function draw() {
    const sel = ms(), f = fr(), k = kk(), ks = range(0, 12, 1);
    const tr = sel.map(m => ({ x: ks, y: curve(m, f).map(v => 100 * v), mode: 'lines+markers', name: T.meth[m],
      line: { color: COL[m], width: m === 'random' ? 1.5 : 2.5, dash: m === 'random' ? 'dash' : 'solid' }, marker: { size: 5 } }));
    plot('dl-chart', tr, layout({ title: ttl(T.dlTitle(f)), xaxis: AX(T.kAx, { dtick: 1 }), yaxis: AX(T.score, { range: [0, 101] }),
      shapes: [vline(k, PAL.ink, 'dot')] }));
    const at = sel.map(m => `${T.meth[m]} ${pct(curve(m, f)[k], 1)}`).join(', ');
    const aopc = sel.map(m => { const c = curve(m, f); return `${T.meth[m]} ${nf(100 * mean(c.slice(1).map(v => c[0] - v)), 1)}`; }).join(', ');
    out('dl-out', sel.length ? [T.dl(k, at), T.dl2(aopc)] : ['–']);
  }
  draw();
  P('dl-play', 'dl-k', range(0, 12, 1), 600);
});

// ---------------------------------------------------------------- 6. seeds: the Rashomon effect
safe('seeds', () => {
  const S = D.seed, md = bindSelect('ss-m', draw), ns = bindSlider('ss-s', v => v, draw);
  const ap = bindSlider('ss-a', v => `${v} (${rowWord} ${S.row[v - 1]})`, draw);
  const imp = {}, ranks = {}, tops = {};
  for (const m of ['mlp', 'xgb']) {
    imp[m] = S[m].map(sv => F.map((_, j) => mean(sv.map(r => Math.abs(r[j])))));
    ranks[m] = imp[m].map(v => { const o = argsortDesc(v), rr = new Array(v.length); o.forEach((j, q) => { rr[j] = q + 1; }); return rr; });
    tops[m] = S[m].map(sv => sv.map(r => argsortDesc(absv(r))[0]));
  }
  const tau = (a, b) => { let s = 0, n = 0; for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) { s += Math.sign(a[i] - a[j]) * Math.sign(b[i] - b[j]); n++; } return s / n; };
  function draw() {
    const m = md(), n = ns(), a = ap() - 1, R = ranks[m].slice(0, n), I = imp[m].slice(0, n);
    const mr = F.map((_, j) => mean(R.map(r => r[j]))), feats = mr.map((v, j) => j).sort((i, j) => mr[i] - mr[j]).slice(0, 8);
    const names = feats.map(j => nice(F[j]));
    const tr = R.map((r, s) => ({ x: feats.map((j, q) => q + (s - (n - 1) / 2) * 0.06), y: feats.map(j => r[j]), mode: 'markers', name: `${T.seed} ${s + 1}`,
      showlegend: false, marker: { color: MCOL[m], size: 7, opacity: 0.75, line: { color: PAL.ink, width: 0.5 } },
      hovertemplate: feats.map(j => `${nice(F[j])}: ${r[j]}<extra>${T.seed} ${s + 1}</extra>`) }));
    plot('ss-rank', tr, layout({ title: ttl(T.ssTitle(n)),
      xaxis: AX('', { tickvals: feats.map((_, q) => q), ticktext: names, tickangle: -30, tickfont: { size: 10, color: C.navy } }),
      yaxis: AX(T.rank, { autorange: 'reversed', dtick: 2 }) }));
    const sv = S[m].slice(0, n).map(s => s[a]);
    const ml = F.map((_, j) => mean(sv.map(r => Math.abs(r[j])))), f6 = argsortDesc(ml).slice(0, 6);
    const tr2 = sv.map((r, s) => ({ x: f6.map((j, q) => q + (s - (n - 1) / 2) * 0.06), y: f6.map(j => r[j]), mode: 'markers', showlegend: false,
      marker: { color: f6.map(j => (r[j] >= 0 ? PAL.pos : PAL.neg)), size: 7, opacity: 0.8 }, hovertemplate: `%{y:.2f}<extra>${T.seed} ${s + 1}</extra>` }));
    plot('ss-loc', tr2, layout({ title: ttl(T.ssLoc(S.row[a])),
      xaxis: AX('', { tickvals: f6.map((_, q) => q), ticktext: f6.map(j => nice(F[j])), tickangle: -30, tickfont: { size: 10, color: C.navy } }),
      yaxis: AX(T.shapv, { zeroline: true }) }));
    let t = 0, o = 0, loc = 0, np_ = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      t += tau(I[i], I[j]);
      const A = new Set(argsortDesc(I[i]).slice(0, 3)), B = argsortDesc(I[j]).slice(0, 3);
      o += B.filter(x => A.has(x)).length / 3;
      loc += mean(tops[m][i].map((x, q) => (x === tops[m][j][q] ? 1 : 0)));
      np_++;
    }
    const cnt = {};
    tops[m].slice(0, n).forEach(tt => { const f = nice(F[tt[a]]); cnt[f] = (cnt[f] || 0) + 1; });
    const lst = Object.entries(cnt).sort((x, y) => y[1] - x[1]).map(([f, c]) => `${f} ×${c}`).join(', ');
    out('ss-out', [T.ss(n, nf(t / np_, 2), pct(o / np_, 0)), T.ss2(pct(loc / np_, 0)), T.ss3(S.row[a], lst)]);
  }
  draw();
  P('ss-play', 'ss-s', range(2, 10, 1), 700);
});
})();
