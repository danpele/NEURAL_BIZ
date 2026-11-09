// Lecture 10 game "You or the agent?": a 2 MWh battery on real Romanian day-ahead prices. The student sees yesterday's
// prices (what the tabular Q-learning agent of the lecture sees), chooses one of four simple plans, then sees the
// realised revenue of every plan, of the agent and of perfect foresight on the day's real prices.
// Data: L10_game_data.js (lectures/L10/make_game_L10.py). Engine: game_common.js. Charts: Plotly (loaded by the page).
// Engine use: truth = best of the four plans on the realised prices; model = the plan closest, hour by hour, to what the
// agent did; the head-to-head with the agent in euros (a card is won when your plan earns at least as much as the agent)
// is shown in the reveal and in the summary, since the engine only scores answer = truth.
// The reveal draws its chart with setTimeout(0), after the engine has inserted the reveal html.
(function () {
  'use strict';
  const G = window.NN_GAME_L10, I = G.info, { RO, nf, shuffle } = window.NNGame;
  const COL = { charge: '#1baf7a', dis: '#e34948', yest: '#8a8a8a', today: '#2a78d6' };
  const MON = RO ? ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie']
    : ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const DOW = RO ? ['luni', 'marți', 'miercuri', 'joi', 'vineri', 'sîmbătă', 'duminică'] : ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const date = (s, w) => { const [y, m, d] = s.split('-').map(Number); return `${DOW[w]}, ${d} ${MON[m - 1]} ${y}`; };
  const eur = x => nf(x, 0).replace('-', '−') + ' EUR';
  const hh = h => String(h).padStart(2, '0') + ':00';
  const span = hs => `${hh(hs[0])}–${hh(hs[hs.length - 1] + 1)}`;
  const H = I.hours;
  const L = RO ? {
    day: 'Ziua de livrare', plan: 'O planificați în ajun; vedeți prețurile de ieri (EUR/MWh, pe ore, ora Europei Centrale):',
    x: 'ora', y: 'EUR/MWh', yest: 'prețurile de ieri', today: 'prețurile reale ale zilei', ych: 'ieri, optim: încărcare', ydis: 'ieri, optim: descărcare',
    ach: 'agentul: încărcare', adis: 'agentul: descărcare',
    plans: [`încărcați noaptea (${span(H.night)}), vindeți seara (${span(H.evening)})`,
      `încărcați la prînz (${span(H.midday)}), vindeți seara (${span(H.evening)})`,
      `noaptea → dimineața (vindeți ${span(H.morning)}), apoi la prînz → seara`,
      'repetați planul care ar fi fost optim ieri (punctele verzi și roșii de pe grafic)'],
    lab: ['Noaptea → seara', 'La prînz → seara', 'Două cicluri', 'Planul optim de ieri'],
    rev: 'Venitul net al zilei la prețurile reale', agent: 'Agentul Q-learning (ce a făcut efectiv)', pf: 'Previziune perfectă (maximul posibil)',
    mine: 'planul dumneavoastră', vs: 'Față de agent', won: '✓ cîștigat', lost: '✗ pierdut',
    sumYou: (a, b, n) => `Venitul planurilor dumneavoastră: <b>${eur(a)}</b>; agentul: <b>${eur(b)}</b> în aceleași ${n} zile.`,
    sumWin: (k, n) => `Ați egalat sau ați întrecut agentul în <b>${k} din ${n}</b> zile.`,
    sumPf: (a, b) => `Cel mai bun dintre cele patru planuri ar fi adus ${eur(a)}; previziunea perfectă, ${eur(b)}.`
  } : {
    day: 'Delivery day', plan: 'You plan it the day before; you see yesterday\'s prices (EUR/MWh, by hour, Central European Time):',
    x: 'hour', y: 'EUR/MWh', yest: 'yesterday\'s prices', today: 'the day\'s real prices', ych: 'yesterday, optimal: charge', ydis: 'yesterday, optimal: discharge',
    ach: 'agent: charge', adis: 'agent: discharge',
    plans: [`charge at night (${span(H.night)}), sell in the evening (${span(H.evening)})`,
      `charge at midday (${span(H.midday)}), sell in the evening (${span(H.evening)})`,
      `night → morning (sell ${span(H.morning)}), then midday → evening`,
      'repeat the plan that would have been optimal yesterday (the green and red dots on the chart)'],
    lab: ['Night → evening', 'Midday → evening', 'Two cycles', 'Yesterday\'s best plan'],
    rev: 'Net revenue of the day at the real prices', agent: 'The Q-learning agent (what it actually did)', pf: 'Perfect foresight (the maximum possible)',
    mine: 'your plan', vs: 'Against the agent', won: '✓ won', lost: '✗ lost',
    sumYou: (a, b, n) => `Revenue of your plans: <b>${eur(a)}</b>; the agent: <b>${eur(b)}</b> on the same ${n} days.`,
    sumWin: (k, n) => `You matched or beat the agent on <b>${k} of ${n}</b> days.`,
    sumPf: (a, b) => `The best of the four plans would have earned ${eur(a)}; perfect foresight, ${eur(b)}.`
  };
  if (RO && window.Plotly) {
    try { Plotly.register({ moduleType: 'locale', name: 'ro', dictionary: {}, format: { decimal: ',', thousands: '.' } }); } catch (e) { /* no locale */ }
  }
  const HRS = Array.from({ length: 24 }, (_, i) => i);
  const marks = (p, acts, sign, name, color, symbol) => {
    const xs = HRS.filter(h => acts[h] === sign);
    return { x: xs, y: xs.map(h => p[h]), type: 'scatter', mode: 'markers', name, hoverinfo: 'skip',
      marker: { color, size: 9, symbol, line: { color: '#fff', width: 1 } } };
  };
  function chart(id, traces, ymax) {
    const el = document.getElementById(id);
    if (!el || !window.Plotly) return;
    Plotly.newPlot(el, traces, {
      margin: { l: 44, r: 8, t: 6, b: 34 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, sans-serif', size: 11 }, showlegend: true,
      legend: { orientation: 'h', x: 0, y: -0.2, yanchor: 'top', font: { size: 10 } },
      xaxis: { title: { text: L.x, standoff: 4 }, range: [-0.5, 23.5], tickvals: [0, 6, 12, 18, 23], ticktext: [0, 6, 12, 18, 23].map(hh), fixedrange: true },
      yaxis: { title: { text: L.y, standoff: 2 }, range: [Math.min(0, ...traces[0].y) * 1.1, ymax * 1.08], zeroline: true, zerolinecolor: '#999', fixedrange: true }
    }, { displayModeBar: false, staticPlot: true, responsive: true, locale: RO ? 'ro' : 'en' });
  }
  const line = (p, name, color, dash) => ({ x: HRS, y: p, type: 'scatter', mode: 'lines', line: { color, width: 2, dash: dash || 'solid', shape: 'hvh' }, name, hoverinfo: 'skip' });
  const won = c => c.rev[c.you] >= c.agent - 1e-9;

  window.NNGame.choice({
    game: 'l10',
    cardTitle: { en: 'Day', ro: 'Ziua' },
    draw: () => shuffle(G.pool).slice(0, I.per_game).map(c => Object.assign({}, c)),
    render: c => `<p><b>${L.day}: ${date(c.date, c.dow)}</b><br>${L.plan}</p>` +
      '<div class="g-chart" id="g10-y" style="height:230px"></div>' +
      '<ol class="g-res" type="A">' + L.plans.map((t, i) => `<li><b>${L.lab[i]}</b>: ${t}</li>`).join('') + '</ol>',
    after: c => {
      const ymax = Math.max(...c.yp);
      chart('g10-y', [line(c.yp, L.yest, COL.yest), marks(c.yp, c.yplan, -1, L.ych, COL.charge, 'circle'), marks(c.yp, c.yplan, 1, L.ydis, COL.dis, 'circle')], ymax);
    },
    choices: L.lab.map(t => ({ label: t })),
    truth: c => c.best,
    model: c => c.closest,
    modelName: { en: 'The Q-learning agent (closest plan)', ro: 'Agentul Q-learning (planul cel mai apropiat)' },
    reveal: (c, you) => {
      c.you = you;
      setTimeout(() => {
        const ymax = Math.max(...c.tp, ...c.yp);
        chart('g10-t', [line(c.tp, L.today, COL.today), line(c.yp, L.yest, COL.yest, 'dot'),
          marks(c.tp, c.agent_act, -1, L.ach, COL.charge, 'triangle-down'), marks(c.tp, c.agent_act, 1, L.adis, COL.dis, 'triangle-up')], ymax);
      }, 0);
      const rows = c.rev.map((v, i) => `<tr${i === c.you ? ' style="background:#EEF3FA"' : ''}><th scope="row">${L.lab[i]}${i === c.you ? ` (${L.mine})` : ''}</th><td>${eur(v)}</td></tr>`).join('') +
        `<tr><th scope="row">${L.agent}</th><td>${eur(c.agent)}</td></tr><tr><th scope="row">${L.pf}</th><td>${eur(c.pf)}</td></tr>`;
      return `<li>${L.vs}: <span class="${won(c) ? 'g-right' : 'g-wrong'}">${won(c) ? L.won : L.lost}</span> (${eur(c.rev[c.you])} / ${eur(c.agent)})</li></ul>` +
        `<h4>${L.rev}</h4><table class="g-tab"><tbody>${rows}</tbody></table><div class="g-chart" id="g10-t" style="height:240px"></div><ul class="g-res">`;
    },
    summary: (res, cards) => {
      const n = res.length, you = res.reduce((s, r, i) => s + cards[i].rev[r.you], 0), ag = cards.reduce((s, c) => s + c.agent, 0);
      const k = res.filter((r, i) => cards[i].rev[r.you] >= cards[i].agent - 1e-9).length;
      const best = cards.reduce((s, c) => s + c.rev[c.best], 0), pf = cards.reduce((s, c) => s + c.pf, 0);
      return `<li>${L.sumYou(you, ag, n)}</li><li>${L.sumWin(k, n)}</li><li>${L.sumPf(best, pf)}</li>`;
    }
  });
})();
