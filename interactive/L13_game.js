// Lecture 13 game "Higher or lower than today?": three weeks of real Romanian day-ahead prices (daily averages, test year),
// the student calls tomorrow's direction, then sees the Chronos-2 and LEAR forecasts of the lecture and the real price.
// Data: L13_game_data.js (lectures/L13/make_game_L13.py). Engine: game_common.js. Chart: Plotly (loaded by the page).
(function () {
  'use strict';
  const G = window.NN_GAME_L13, I = G.info, { RO, nf, shuffle } = window.NNGame;
  const MON = RO ? ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie']
    : ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const MS = RO ? ['ian.', 'feb.', 'mar.', 'apr.', 'mai', 'iun.', 'iul.', 'aug.', 'sep.', 'oct.', 'nov.', 'dec.']
    : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const DAY = RO ? ['duminică', 'luni', 'marți', 'miercuri', 'joi', 'vineri', 'sîmbătă'] : ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const DS = RO ? ['dum.', 'lun.', 'mar.', 'mie.', 'joi', 'vin.', 'sîm.'] : ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const D = s => new Date(s + 'T12:00:00Z');
  const iso = d => d.toISOString().slice(0, 10);
  const add = (s, k) => { const d = D(s); d.setUTCDate(d.getUTCDate() + k); return iso(d); };
  const shift = (s, h) => new Date(D(s).getTime() - 12 * 3600e3 + h * 3600e3).toISOString().slice(0, 19).replace('T', ' ');
  const longDate = s => { const d = D(s); return `${DAY[d.getUTCDay()]} ${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}`; };
  const shortDate = s => { const d = D(s); return `${DS[d.getUTCDay()]} ${d.getUTCDate()} ${MS[d.getUTCMonth()]} ${d.getUTCFullYear()}`; };
  const eur = x => `${nf(x, 2)} EUR/MWh`;
  const chg = (x, ref) => { const p = 100 * (x - ref) / Math.abs(ref); return (p > 0 ? '+' : p < 0 ? '−' : '') + nf(Math.abs(p), 1) + (RO ? '\u00a0%' : '%'); };
  const C = { navy: '#1A3A6E', grid: '#e6e6e6', wk: '#B5853F', real: '#222222', ch: '#6A3D9A', lear: '#777777' };
  const L = RO ? {
    today: 'Azi', tom: 'Mîine', avg: 'prețul mediu', q: 'Prețul mediu de mîine va fi mai mare sau mai mic decît cel de azi?',
    yax: 'Prețul mediu zilnic (EUR/MWh)', series: 'Prețul mediu zilnic', wk: 'Sîmbătă și duminică', lvl: 'Nivelul de azi',
    real: 'Mîine, prețul real', ch: 'Prognoza Chronos-2', lear: 'Prognoza LEAR',
    fc: (m, x, r) => `Prognoza ${m}: <b>${eur(x)}</b> (${chg(x, r)} față de azi)`,
    out: (x, r) => `Prețul real de mîine: <b>${eur(x)}</b> (${chg(x, r)} față de azi)`,
    sum: (a, n) => `Prețul a crescut în ${a} din cele ${n} zile ale jocului.`
  } : {
    today: 'Today', tom: 'Tomorrow', avg: 'average price', q: 'Will tomorrow\'s average price be higher or lower than today\'s?',
    yax: 'Daily average price (EUR/MWh)', series: 'Daily average price', wk: 'Saturday and Sunday', lvl: 'Today\'s level',
    real: 'Tomorrow, real price', ch: 'Chronos-2 forecast', lear: 'LEAR forecast',
    fc: (m, x, r) => `${m} forecast: <b>${eur(x)}</b> (${chg(x, r)} on today)`,
    out: (x, r) => `Tomorrow's real price: <b>${eur(x)}</b> (${chg(x, r)} on today)`,
    sum: (a, n) => `The price rose on ${a} of the ${n} days of the game.`
  };
  const today = c => c.h[c.h.length - 1];

  function draw(c, reveal) {
    const el = document.getElementById('g-ch');
    if (!el || !window.Plotly) return;
    const n = c.h.length, xs = c.h.map((_, i) => add(c.d, i - n));
    const isWk = s => [0, 6].includes(D(s).getUTCDay());
    const hov = xs.map(shortDate);
    const tv = [xs[0], xs[7], xs[14], c.d], tt = tv.map(s => { const d = D(s); return `${d.getUTCDate()} ${MS[d.getUTCMonth()]}<br>${d.getUTCFullYear()}`; });
    const tr = [
      { x: xs, y: c.h, type: 'scatter', mode: 'lines+markers', name: L.series, line: { color: C.navy, width: 2 }, marker: { size: 6, color: C.navy },
        text: hov, hovertemplate: '%{text}<br>%{y:,.2f} EUR/MWh<extra></extra>', connectgaps: false },
      { x: xs.filter(isWk), y: c.h.filter((_, i) => isWk(xs[i])), type: 'scatter', mode: 'markers', name: L.wk,
        marker: { size: 9, color: C.wk, line: { color: '#fff', width: 1 } }, text: hov.filter((_, i) => isWk(xs[i])), hovertemplate: '%{text}<br>%{y:,.2f} EUR/MWh<extra></extra>' }
    ];
    if (reveal) {
      const col = C.real, t2 = shortDate(c.d);
      tr.push({ x: [xs[n - 1], c.d], y: [today(c), c.a], type: 'scatter', mode: 'lines+markers', name: L.real, line: { color: col, width: 2, dash: 'dot' },
        marker: { size: [0, 11], color: col }, text: ['', t2], hovertemplate: '%{text}<br>%{y:,.2f} EUR/MWh<extra></extra>', showlegend: true });
      tr.push({ x: [c.d], y: [c.c], type: 'scatter', mode: 'markers', name: L.ch, marker: { size: 11, symbol: 'diamond-open', color: C.ch, line: { width: 2 } },
        text: [t2], hovertemplate: 'Chronos-2<br>%{y:,.2f} EUR/MWh<extra></extra>' });
      tr.push({ x: [c.d], y: [c.l], type: 'scatter', mode: 'markers', name: L.lear, marker: { size: 10, symbol: 'x-thin-open', color: C.lear, line: { width: 2 } },
        hovertemplate: 'LEAR<br>%{y:,.2f} EUR/MWh<extra></extra>' });
    }
    const vals = c.h.filter(v => v !== null).concat([c.a, c.c, c.l]);
    const lo = Math.min(...vals), hi = Math.max(...vals), pad = 0.08 * (hi - lo || 1);
    const lay = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', separators: RO ? ',.' : '.,',
      font: { family: 'Inter, system-ui, sans-serif', color: C.navy, size: 11 }, margin: { l: 52, r: 14, t: 8, b: 8 },
      xaxis: { type: 'date', tickvals: tv, ticktext: tt, gridcolor: C.grid, range: [shift(xs[0], -14), shift(c.d, 14)], fixedrange: true, automargin: true },
      yaxis: { title: { text: L.yax, font: { size: 11 } }, gridcolor: C.grid, tickformat: ',.0f', range: [lo - pad, hi + pad], fixedrange: true },
      shapes: [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: today(c), y1: today(c), line: { color: C.navy, width: 1, dash: 'dash' } }],
      legend: { orientation: 'h', x: 0, xanchor: 'left', y: -0.22, yanchor: 'top', font: { size: 10 } },
      hovermode: 'closest', hoverlabel: { bgcolor: '#fff', bordercolor: C.navy, font: { color: C.navy } }
    };
    Plotly.react(el, tr, lay, { displayModeBar: false, responsive: true });
  }

  window.NNGame.choice({
    game: 'l13',
    cardTitle: { en: 'Delivery day', ro: 'Ziua de livrare' },
    draw: () => shuffle(G.pool).slice(0, I.per_game),
    render: c => `<p>${L.today}: ${longDate(add(c.d, -1))}, ${L.avg} <b>${eur(today(c))}</b><br>${L.tom}: <b>${longDate(c.d)}</b></p>` +
      `<div class="g-chart" id="g-ch" role="img" aria-label="${L.yax}" style="height:320px"></div><p class="g-big">${L.q}</p>`,
    after: c => draw(c, false),
    choices: [{ label: { en: 'Higher ▲', ro: 'Mai mare ▲' }, cls: '' }, { label: { en: 'Lower ▼', ro: 'Mai mic ▼' }, cls: '' }],
    truth: c => (c.a > today(c) ? 0 : 1),
    model: c => (c.c > today(c) ? 0 : 1),
    modelName: { en: 'Chronos-2, the foundation model of the lecture', ro: 'Chronos-2, modelul fundație din curs' },
    reveal: c => {
      setTimeout(() => draw(c, true), 0);   // the engine has no hook after the reveal: redraw the chart with tomorrow's prices
      return `<li>${L.fc('Chronos-2', c.c, today(c))}</li><li>${L.fc('LEAR', c.l, today(c))}</li><li>${L.out(c.a, today(c))}</li>`;
    },
    summary: (res, cards) => `<li>${L.sum(cards.filter(c => c.a > today(c)).length, cards.length)}</li>`
  });
})();
