// Lecture 11 game "Which feature drove the decision?": a real HMEQ test applicant and the decision of the lecture's
// network; the student picks, among four features, the one with the largest Kernel SHAP contribution, then sees the
// SHAP top features, LIME's answer and the stability of the SHAP top feature over the 10 training seeds.
// Data: L11_game_data.js (lectures/L11/make_game_L11.py). Engine: game_common.js.
// truth = option with the largest |Kernel SHAP| contribution; model = option with the largest |LIME slope x standardised value|.
(function () {
  'use strict';
  const G = window.NN_GAME_L11, I = G.info, { RO, nf, pct, shuffle } = window.NNGame;
  const num = (x, d) => nf(x, d).replace('-', '−');
  const sgn = x => (x > 0 ? '+' : '') + num(x, 2);
  const L = RO ? {
    missing: 'necunoscut',
    dec: p => `Decizia rețelei: <b>${p >= 0.5 ? 'RESPINGE creditul' : 'APROBĂ creditul'}</b> (probabilitatea de neplată estimată: ${pct(p)})`,
    ask: 'Ce variabilă a contat cel mai mult pentru această decizie?',
    reason: { DebtCon: 'consolidarea datoriilor', HomeImp: 'renovarea locuinței' },
    job: { Mgr: 'manager', Office: 'funcționar', Other: 'altă ocupație', ProfExe: 'profesie liberală sau director', Sales: 'vînzări', Self: 'pe cont propriu' },
    rows: [['loan', 'Suma cerută', 'USD'], ['mortdue', 'Datoria rămasă pe ipoteca existentă', 'USD'], ['value', 'Valoarea locuinței', 'USD'],
      ['reason', 'Scopul creditului'], ['job', 'Ocupația'], ['yoj', 'Ani la locul de muncă actual'], ['derog', 'Rapoarte negative majore în istoricul de credit'],
      ['delinq', 'Linii de credit cu restanțe'], ['clage', 'Vechimea celei mai vechi linii de credit', 'ani'], ['ninq', 'Cereri recente de credit'],
      ['clno', 'Număr de linii de credit'], ['debtinc', 'Datorii / venit', '%']],
    feat: { LOAN: 'Suma cerută', MORTDUE: 'Datoria pe ipotecă', VALUE: 'Valoarea locuinței', YOJ: 'Ani la locul de muncă', DEROG: 'Rapoarte negative majore',
      DELINQ: 'Linii de credit cu restanțe', CLAGE: 'Vechimea liniei de credit', NINQ: 'Cereri recente de credit', CLNO: 'Număr de linii de credit',
      DEBTINC: 'Datorii / venit', DEBTINC_missing: 'Raportul datorii/venit nedeclarat', REASON_HomeImp: 'Scopul: renovarea locuinței',
      JOB_Mgr: 'Ocupația: manager', JOB_Office: 'Ocupația: funcționar', JOB_Other: 'Ocupația: altă ocupație', JOB_ProfExe: 'Ocupația: profesie liberală',
      JOB_Sales: 'Ocupația: vînzări', JOB_Self: 'Ocupația: pe cont propriu' },
    th: ['Variabila (primele 5 după SHAP)', 'SHAP', 'LIME'],
    limeTop: (f, v) => `LIME pune pe primul loc „${f}” (${v}), care nu este între primele 5 variabile după SHAP.`,
    tabNote: 'Contribuții la log-șansele de neplată: + crește riscul, − îl reduce.',
    flip: (f, a, b) => `SHAP și LIME nu sînt de acord nici asupra direcției: pentru „${f}” SHAP spune că riscul crește (${a}), LIME că scade (${b}).`,
    seeds: (k, n) => `Variabila SHAP principală este aceeași pentru <b>${k} din ${n}</b> rețele antrenate cu valori inițiale diferite.`,
    out: b => `Ce s-a întîmplat: clientul <b>${b ? 'nu a plătit (neplată)' : 'a rambursat creditul'}</b>`,
    sumAgree: (k, n) => `LIME a indicat aceeași variabilă ca SHAP în <b>${k} din ${n}</b> cazuri.`,
    sumSeeds: (k, n, s) => `Variabila SHAP principală a rămas aceeași pentru toate cele ${s} rețele în ${k} din ${n} cazuri.`
  } : {
    missing: 'not reported',
    dec: p => `The network's decision: <b>${p >= 0.5 ? 'REFUSE the loan' : 'APPROVE the loan'}</b> (estimated probability of default: ${pct(p)})`,
    ask: 'Which feature mattered most for this decision?',
    reason: { DebtCon: 'debt consolidation', HomeImp: 'home improvement' },
    job: { Mgr: 'manager', Office: 'office worker', Other: 'other', ProfExe: 'professional or executive', Sales: 'sales', Self: 'self-employed' },
    rows: [['loan', 'Amount requested', 'USD'], ['mortdue', 'Amount still due on the existing mortgage', 'USD'], ['value', 'Value of the home', 'USD'],
      ['reason', 'Purpose of the loan'], ['job', 'Occupation'], ['yoj', 'Years at the present job'], ['derog', 'Major derogatory reports in the credit history'],
      ['delinq', 'Delinquent credit lines'], ['clage', 'Age of the oldest credit line', 'years'], ['ninq', 'Recent credit inquiries'],
      ['clno', 'Number of credit lines'], ['debtinc', 'Debt-to-income ratio', '%']],
    feat: { LOAN: 'Amount requested', MORTDUE: 'Mortgage still due', VALUE: 'Value of the home', YOJ: 'Years at the job', DEROG: 'Major derogatory reports',
      DELINQ: 'Delinquent credit lines', CLAGE: 'Age of the oldest credit line', NINQ: 'Recent credit inquiries', CLNO: 'Number of credit lines',
      DEBTINC: 'Debt-to-income ratio', DEBTINC_missing: 'Debt-to-income not reported', REASON_HomeImp: 'Purpose: home improvement',
      JOB_Mgr: 'Occupation: manager', JOB_Office: 'Occupation: office worker', JOB_Other: 'Occupation: other', JOB_ProfExe: 'Occupation: professional',
      JOB_Sales: 'Occupation: sales', JOB_Self: 'Occupation: self-employed' },
    th: ['Feature (top 5 by SHAP)', 'SHAP', 'LIME'],
    limeTop: (f, v) => `LIME puts first “${f}” (${v}), which is not among the top 5 features by SHAP.`,
    tabNote: 'Contributions to the log-odds of default: + raises the risk, − lowers it.',
    flip: (f, a, b) => `SHAP and LIME disagree even on the direction: for “${f}” SHAP says it ${a.startsWith('+') ? 'raises' : 'lowers'} the risk (${a}), LIME that it ${a.startsWith('+') ? 'lowers' : 'raises'} it (${b}).`,
    seeds: (k, n) => `The top SHAP feature is the same for <b>${k} of ${n}</b> networks trained with different seeds.`,
    out: b => `What happened: the client <b>${b ? 'defaulted' : 'repaid the loan'}</b>`,
    sumAgree: (k, n) => `LIME named the same feature as SHAP in <b>${k} of ${n}</b> cases.`,
    sumSeeds: (k, n, s) => `The top SHAP feature stayed the same for all ${s} networks in ${k} of ${n} cases.`
  };
  function val(c, key, unit) {
    const v = c[key];
    if (v === null || v === undefined) return `<span class="g-miss">${L.missing}</span>`;
    if (key === 'reason') return L.reason[v] || v;
    if (key === 'job') return L.job[v] || v;
    if (key === 'clage') return nf(v / 12, 1) + ' ' + unit;
    if (key === 'yoj') return nf(v, 1);
    if (key === 'debtinc') return nf(v, 1) + (RO ? ' %' : '%');
    return nf(v) + (unit ? ' ' + unit : '');
  }
  const F = f => L.feat[f] || f;
  window.NNGame.choice({
    game: 'l11',
    cardTitle: { en: 'Applicant', ro: 'Solicitantul' },
    draw: () => {   // the same share of refused applicants as in the pool (12 of 30)
      const nr = Math.round(I.per_game * I.pool_ref / I.pool);
      return shuffle(shuffle(G.pool.filter(c => c.p >= 0.5)).slice(0, nr).concat(shuffle(G.pool.filter(c => c.p < 0.5)).slice(0, I.per_game - nr)));
    },
    render: c => '<table class="g-tab"><tbody>' + L.rows.map(([key, lab, unit]) => `<tr><th scope="row">${lab}</th><td>${val(c, key, unit)}</td></tr>`).join('') +
      `</tbody></table><p class="g-quote">${L.dec(c.p)}</p><p><b>${L.ask}</b></p>`,
    choices: c => c.opts.map(f => ({ label: F(f) })),
    truth: c => c.truth,
    model: c => c.lime_pick,
    modelName: { en: 'LIME', ro: 'LIME' },
    reveal: c => {
      const t = c.truth, flip = Math.sign(c.lime_opt[t]) !== Math.sign(c.shap_opt[t]) && c.lime_opt[t] !== 0;
      const rows = c.top5.map((f, i) => `<tr><th scope="row">${F(f)}</th><td style="padding-left:0.6rem">${sgn(c.top5_shap[i])}</td><td style="padding-left:0.6rem">${sgn(c.top5_lime[i])}</td></tr>`).join('');
      return (c.top5.indexOf(c.lime_top) < 0 ? `<li>${L.limeTop(F(c.lime_top), sgn(c.lime_top_val))}</li>` : '') +
        (flip ? `<li>${L.flip(F(c.opts[t]), sgn(c.shap_opt[t]), sgn(c.lime_opt[t]))}</li>` : '') +
        `<li>${L.seeds(c.seeds_same, I.seeds)}</li><li>${L.out(c.bad)}</li></ul>` +
        `<table class="g-tab g11"><thead><tr>${L.th.map((h, i) => `<th${i ? ' style="text-align:right"' : ''}>${h}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>` +
        `<p class="g-note">${L.tabNote}</p><ul class="g-res">`;
    },
    summary: (res, cards) => {
      const n = res.length, a = cards.filter(c => c.lime_pick === c.truth).length, s = cards.filter(c => c.seeds_same === I.seeds).length;
      return `<li>${L.sumAgree(a, n)}</li><li>${L.sumSeeds(s, n, I.seeds)}</li>`;
    }
  });
})();
