// Lecture 5 game "Up or down tomorrow?": 60 real trading days of the S&P 500 or Banca Transilvania (test period),
// the student calls the direction of the next close, then sees the LSTM forecast of the lecture and the real close.
// Data: L5_game_data.js (lectures/L5/make_game_L5.py). Engine: game_common.js. Chart: Plotly (loaded by the page).
(function () {
  'use strict';
  const G = window.NN_GAME_L5, I = G.info, { RO, nf, shuffle } = window.NNGame;
  const MON = RO ? ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie']
    : ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const MS = RO ? ['ian.', 'feb.', 'mar.', 'apr.', 'mai', 'iun.', 'iul.', 'aug.', 'sep.', 'oct.', 'nov.', 'dec.']
    : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const DAY = RO ? ['duminică', 'luni', 'marți', 'miercuri', 'joi', 'vineri', 'sîmbătă'] : ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const D = s => new Date(s + 'T12:00:00Z');
  const longDate = s => { const d = D(s); return `${DAY[d.getUTCDay()]} ${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}`; };
  const sgn = (x, d) => (x > 0 ? '+' : x < 0 ? '−' : '') + nf(Math.abs(x), d) + (RO ? '\u00a0%' : '%');
  const C = { navy: '#1A3A6E', grid: '#e6e6e6', up: '#2E7D32', down: '#DC3545' };
  const L = RO ? {
    name: { SP500: 'Indicele S&amp;P 500', TLV: 'Acțiunea Banca Transilvania (TLV)' },
    unit: { SP500: 'puncte', TLV: 'lei' }, yax: { SP500: 'Nivelul indicelui (puncte)', TLV: 'Prețul de închidere ajustat (lei)' },
    last: 'Ultima închidere', q: d => `Închiderea de ${d} va fi mai mare sau mai mică?`,
    lstm: (f, n) => `Prognoza LSTM pentru randamentul de mîine: <b>${sgn(f, 2)}</b> (${n} din cele 5 rețele indică o creștere)`,
    out: (p, u, r) => `Ce s-a întîmplat: închidere la <b>${p} ${u}</b>, randament <b>${r}</b>`,
    hov: 'închidere', next: 'ziua de ghicit',
    sum: (a, b, c, n) => `Ați răspuns „crește” ${ori(a)} din ${n}; rețeaua, ${ori(b)}; prețul a crescut în ${c} din cele ${n} zile.`
  } : {
    name: { SP500: 'S&amp;P 500 index', TLV: 'Banca Transilvania share (TLV)' },
    unit: { SP500: 'points', TLV: 'RON' }, yax: { SP500: 'Index level (points)', TLV: 'Adjusted close (RON)' },
    last: 'Last close', q: d => `Will the close of ${d} be higher or lower?`,
    lstm: (f, n) => `LSTM forecast of tomorrow's return: <b>${sgn(f, 2)}</b> (${n} of the 5 networks ${n === 1 ? 'says' : 'say'} up)`,
    out: (p, u, r) => `What happened: close at <b>${p} ${u}</b>, return <b>${r}</b>`,
    hov: 'close', next: 'the day to call',
    sum: (a, b, c, n) => `You answered 'up' ${a} ${a === 1 ? 'time' : 'times'} out of ${n}, the network ${b} ${b === 1 ? 'time' : 'times'}; the price rose on ${c} of the ${n} days.`
  };
  const ori = a => (a === 1 ? 'o dată' : `de ${a} ori`);
  const dec = () => 2;
  const dfmt = s => { const d = D(s); return `${d.getUTCDate()} ${MS[d.getUTCMonth()]} ${d.getUTCFullYear()}`; };

  function draw(c, reveal) {
    const el = document.getElementById('g-ch');
    if (!el || !window.Plotly) return;
    const n = c.d.length - 1, xs = c.d.slice(0, n), ys = c.p.slice(0, n);
    // monthly ticks at the first trading day of each month, labelled with the month and the four-digit year
    const tv = [], tt = [];
    xs.forEach((s, i) => { if (i > 2 && s.slice(0, 7) !== xs[i - 1].slice(0, 7)) { tv.push(s); const d = D(s); tt.push(`${MS[d.getUTCMonth()]} ${d.getUTCFullYear()}`); } });
    const tr = [{ x: xs, y: ys, type: 'scatter', mode: 'lines', line: { color: C.navy, width: 2 }, text: xs.map(dfmt),
      hovertemplate: `%{text}<br>${L.hov}: %{y:,.${dec(c)}f}<extra></extra>` }];
    const yr = [Math.min(...c.p), Math.max(...c.p)], pad = 0.08 * (yr[1] - yr[0] || 1);
    if (reveal) {
      const up = c.y > 0, col = up ? C.up : C.down;
      tr.push({ x: [xs[n - 1], c.d[n]], y: [ys[n - 1], c.p[n]], type: 'scatter', mode: 'lines+markers', line: { color: col, width: 2.5 },
        marker: { size: [0, 11], color: col }, text: ['', dfmt(c.d[n])], hovertemplate: `%{text}<br>${L.hov}: %{y:,.${dec(c)}f}<extra></extra>` });
    }
    const lay = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', separators: RO ? ',.' : '.,', showlegend: false,
      font: { family: 'Inter, system-ui, sans-serif', color: C.navy, size: 11 }, margin: { l: 58, r: 14, t: 8, b: 34 },
      xaxis: { type: 'date', tickvals: tv, ticktext: tt, gridcolor: C.grid, range: [xs[0], new Date(D(c.d[n]).getTime() + 48 * 3600e3).toISOString().slice(0, 10)], fixedrange: true },
      shapes: [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: ys[n - 1], y1: ys[n - 1], line: { color: C.navy, width: 1, dash: 'dash' } }],
      yaxis: { title: { text: L.yax[c.s], font: { size: 11 } }, gridcolor: C.grid, tickformat: c.s === 'SP500' ? ',.0f' : ',.2~f',
        range: [yr[0] - pad, yr[1] + pad], fixedrange: true },
      hovermode: 'closest', hoverlabel: { bgcolor: '#fff', bordercolor: C.navy, font: { color: C.navy } }
    };
    Plotly.react(el, tr, lay, { displayModeBar: false, responsive: true });
  }

  window.NNGame.choice({
    game: 'l5',
    cardTitle: { en: 'Trading day', ro: 'Ziua de tranzacționare' },
    draw: () => {   // 5 days of each series, in random order
      const half = Math.round(I.per_game / 2);
      return shuffle(shuffle(G.pool.filter(c => c.s === 'SP500')).slice(0, half).concat(shuffle(G.pool.filter(c => c.s === 'TLV')).slice(0, I.per_game - half)));
    },
    render: c => {
      const n = c.d.length - 1;
      return `<p><b>${L.name[c.s]}</b><br>${L.last}: ${longDate(c.d[n - 1])}, <b>${nf(c.p[n - 1], 2)} ${L.unit[c.s]}</b></p>` +
        `<div class="g-chart" id="g-ch" role="img" aria-label="${L.yax[c.s]}"></div><p class="g-big">${L.q(longDate(c.d[n]))}</p>`;
    },
    after: c => draw(c, false),
    choices: [{ label: { en: 'Up ▲', ro: 'Crește ▲' }, cls: 'g-ok' }, { label: { en: 'Down ▼', ro: 'Scade ▼' }, cls: 'g-no' }],
    truth: c => (c.y > 0 ? 0 : 1),
    model: c => (c.f > 0 ? 0 : 1),
    modelName: { en: 'The LSTM of the lecture', ro: 'LSTM-ul din curs' },
    reveal: c => {
      setTimeout(() => draw(c, true), 0);   // the engine has no hook after the reveal: redraw the chart with the real close
      const n = c.d.length - 1;
      return `<li>${L.lstm(c.f, c.nup)}</li><li>${L.out(nf(c.p[n], 2), L.unit[c.s], sgn(c.y, 2))}</li>`;
    },
    summary: (res, cards) => `<li>${L.sum(res.filter(r => r.you === 0).length, res.filter(r => r.model === 0).length, cards.filter(c => c.y > 0).length, res.length)}</li>`
  });
})();
