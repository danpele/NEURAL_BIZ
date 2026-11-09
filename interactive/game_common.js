// Classroom games of the NN course (one per lecture): shared engine for "you or the model?" card games.
// A lecture game page loads ../assets/config.js, Lk_game_data.js, this file and Lk_game.js, which calls
//   NNGame.choice({
//     game: 'l1',                          // lecture id, also the id of the class board (game_board.html?game=l1)
//     cards: [...],                        // the cards of one game (or draw: () => [...] for a new draw every game)
//     render: c => html,                   // what the student sees before deciding
//     choices: [{ label, cls }] | c => [...],   // answer buttons; cls 'g-ok' (green), 'g-no' (red) or '' (blue)
//     truth: c => index,                   // correct answer
//     model: c => index | null,            // the model's answer (null: no model on this card)
//     reveal: c => html,                   // <li> lines shown after the decision (model output, what really happened)
//     modelName: { en, ro },               // e.g. 'The network of the lecture'
//     summary: (res, cards) => html,       // optional <li> lines on the result screen; res[i] = { you, model, truth }
//   });
// The page holds #g-intro (with button #g-s), #g-play (#g-score, #g-card) and <template id="g-lesson"> (shown at the end).
// The first finished game of the day on a device can be sent, under a nickname, to the class board (tools/quiz_scores.gs).
(function () {
  'use strict';
  const RO = window.NN_LANG === 'ro';
  const $ = id => document.getElementById(id);
  const tr = o => (o && typeof o === 'object' && ('en' in o || 'ro' in o)) ? (RO ? o.ro : o.en) : o;
  const TX = RO ? {
    start: 'Începe jocul', next: 'Următorul', again: 'Joacă din nou', result: 'Rezultatul', see: 'Vezi rezultatul',
    card: (k, n) => `${k} din ${n}`, you: 'Răspunsul dumneavoastră', right: 'corect', wrong: 'greșit', truth: 'Răspunsul corect',
    score: 'Scor', you2: 'Dumneavoastră', model2: 'Modelul',
    end: (y, m, n, has) => `Ați răspuns corect în ${y} din ${n} cazuri` + (has ? `; modelul, în ${m} din ${n}.` : '.'),
    h: 'Clasamentul sălii', nick: 'Pseudonim (apare pe ecranul din sală, nu folosiți numele real)', send: 'Trimite rezultatul',
    sent: 'Rezultatul a fost trimis. Doar primul joc al zilei intră în clasament.',
    done: 'Ați trimis deja rezultatul de azi la acest joc; jocurile următoare nu mai intră în clasament.',
    fail: 'Rezultatul nu a putut fi trimis.', sending: 'Se trimite…'
  } : {
    start: 'Start the game', next: 'Next', again: 'Play again', result: 'The result', see: 'See the result',
    card: (k, n) => `${k} of ${n}`, you: 'Your answer', right: 'right', wrong: 'wrong', truth: 'The right answer',
    score: 'Score', you2: 'You', model2: 'Model',
    end: (y, m, n, has) => `You answered correctly in ${y} of ${n} cases` + (has ? `; the model in ${m} of ${n}.` : '.'),
    h: 'Class board', nick: 'Nickname (shown on the screen in the room; do not use your real name)', send: 'Send my result',
    sent: 'Your result was sent. Only the first game of the day counts on the board.',
    done: 'You already sent today\'s result for this game; later games do not count on the board.',
    fail: 'The result could not be sent.', sending: 'Sending…'
  };
  const URL_ = (window.NN_CONFIG || {}).QUIZ_SCORES_URL || '';
  const today = () => { const d = new Date(); return '' + d.getFullYear() + String(d.getMonth() + 1).padStart(2, '0') + String(d.getDate()).padStart(2, '0'); };

  function shuffle(a) {
    a = a.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }

  function choice(cfg) {
    const SESSION = today(), KEY = 'nn_game_' + cfg.game + '_' + SESSION;
    const store = { get: () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } },
                    set: v => { try { localStorage.setItem(KEY, v); } catch (e) { /* private mode */ } } };
    let cards = [], k = 0, res = [];
    const hasModel = () => res.some(r => r.model !== null);
    const mark = ok => `<span class="${ok ? 'g-right' : 'g-wrong'}">${ok ? '✓ ' + TX.right : '✗ ' + TX.wrong}</span>`;
    const opts = c => (typeof cfg.choices === 'function' ? cfg.choices(c) : cfg.choices).map(o => ({ label: tr(o.label), cls: o.cls || '' }));

    function score() {
      const y = res.filter(r => r.you === r.truth).length, m = res.filter(r => r.model !== null && r.model === r.truth).length;
      $('g-score').innerHTML = `${TX.score}: <b>${TX.you2} ${y}</b>` + (hasModel() ? ` · <b>${TX.model2} ${m}</b>` : '') + ` <span class="g-of">/ ${res.length}</span>`;
    }
    function show() {
      const c = cards[k], o = opts(c);
      $('g-card').innerHTML = `<h3>${tr(cfg.cardTitle) || ''} ${TX.card(k + 1, cards.length)}</h3>` + cfg.render(c) +
        `<div class="g-act">${o.map((x, i) => `<button type="button" class="btn ${x.cls}" data-i="${i}">${x.label}</button>`).join('')}</div><div id="g-rev" aria-live="polite"></div>`;
      const bs = [...$('g-card').querySelectorAll('.g-act .btn')];
      bs.forEach(b => { b.onclick = () => decide(+b.dataset.i, bs); });
      if (bs[0]) bs[0].focus({ preventScroll: true });
      if (cfg.after) cfg.after(c);
    }
    function decide(you, bs) {
      const c = cards[k], o = opts(c), t = cfg.truth(c), mm = cfg.model ? cfg.model(c) : null;
      res.push({ you, model: (mm === null || mm === undefined) ? null : mm, truth: t });
      bs.forEach((b, i) => { b.disabled = true; if (i === t) b.classList.add('g-is'); if (i === you) b.classList.add('g-mine'); });
      $('g-rev').innerHTML = '<ul class="g-res">' +
        `<li>${TX.you}: <b>${o[you].label}</b> ${mark(you === t)}</li>` +
        (mm === null || mm === undefined ? '' : `<li>${tr(cfg.modelName)}: <b>${o[mm].label}</b> ${mark(mm === t)}</li>`) +
        (cfg.reveal ? cfg.reveal(c, you) : `<li>${TX.truth}: <b>${o[t].label}</b></li>`) + '</ul>' +
        `<button type="button" class="btn play" id="g-n">${k + 1 < cards.length ? TX.next : TX.see}</button>`;
      score();
      $('g-n').onclick = () => { k++; k < cards.length ? show() : finish(); };
      $('g-n').focus({ preventScroll: true });
    }
    function finish() {
      const n = res.length, y = res.filter(r => r.you === r.truth).length, m = res.filter(r => r.model !== null && r.model === r.truth).length;
      $('g-card').innerHTML = `<h3>${TX.result}</h3><p class="g-big">${TX.end(y, m, n, hasModel())}</p>` +
        (cfg.summary ? `<ul class="g-res">${cfg.summary(res, cards)}</ul>` : '') +
        sendForm() + ($('g-lesson') ? $('g-lesson').innerHTML : '') + `<button type="button" class="btn play" id="g-s2">${TX.again}</button>`;
      $('g-s2').onclick = start;
      const f = $('g-send');
      if (f) f.onsubmit = ev => { ev.preventDefault(); send(y, hasModel() ? m : y, n); };
      $('g-card').scrollIntoView({ block: 'start' });
    }
    function sendForm() {
      if (!/^https:\/\/script\.google\.com\//.test(URL_)) return '';
      if (store.get()) return `<p class="g-note">${TX.done}</p>`;
      return `<form id="g-send" class="g-send"><h4>${TX.h}</h4><label for="g-nick">${TX.nick}</label>` +
        `<div class="g-row"><input id="g-nick" maxlength="20" autocomplete="off" required><button type="submit" class="btn">${TX.send}</button></div><p id="g-st" class="g-note" aria-live="polite"></p></form>`;
    }
    function send(you, net, total) {
      const st = $('g-st'), b = $('g-send').querySelector('button');
      b.disabled = true; st.textContent = TX.sending;
      fetch(URL_, { method: 'POST', headers: { 'Content-Type': 'text/plain;charset=utf-8' },
        body: JSON.stringify({ type: 'game', game: cfg.game, session: SESSION, nick: $('g-nick').value, you, net, total, lang: RO ? 'ro' : 'en' }) })
        .then(x => x.json()).then(j => {
          if (j.ok) { store.set('1'); st.textContent = TX.sent; $('g-nick').disabled = true; } else { st.textContent = TX.fail; b.disabled = false; }
        }).catch(() => { st.textContent = TX.fail; b.disabled = false; });
    }
    function start() {
      cards = cfg.draw ? cfg.draw() : cfg.cards.slice();
      k = 0; res = [];
      $('g-score').innerHTML = '';
      show();
    }
    $('g-s').textContent = TX.start;
    $('g-s').onclick = () => { $('g-intro').hidden = true; $('g-play').hidden = false; start(); };
  }

  window.NNGame = { choice, shuffle, RO, tr,
    nf: (x, d) => Number(x).toLocaleString(RO ? 'ro-RO' : 'en-US', { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }),
    pct: (x, d) => Number(100 * x).toLocaleString(RO ? 'ro-RO' : 'en-US', { minimumFractionDigits: d === undefined ? 1 : d, maximumFractionDigits: d === undefined ? 1 : d }) + (RO ? ' %' : '%') };
})();
