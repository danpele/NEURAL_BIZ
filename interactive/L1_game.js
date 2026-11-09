// Lecture 1 game "You or the network?": real HMEQ loans, approve or refuse, then the network's decision and the outcome.
// Data: L1_game_data.js (lectures/L1/make_game_L1.py). Engine: game_common.js.
(function () {
  'use strict';
  const G = window.NN_GAME_L1, I = G.info, { RO, nf, pct, shuffle } = window.NNGame;
  const L = RO ? {
    missing: 'necunoscut', approved: 'aprob', refused: 'resping',
    reason: { DebtCon: 'consolidarea datoriilor', HomeImp: 'renovarea locuinței' },
    job: { Mgr: 'manager', Office: 'funcționar', Other: 'altă ocupație', ProfExe: 'profesie liberală sau director', Sales: 'vînzări', Self: 'pe cont propriu' },
    rows: [['loan', 'Suma cerută', 'USD'], ['mortdue', 'Datoria rămasă pe ipoteca existentă', 'USD'], ['value', 'Valoarea locuinței', 'USD'],
      ['reason', 'Scopul creditului'], ['job', 'Ocupația'], ['yoj', 'Ani la locul de muncă actual'], ['derog', 'Rapoarte negative majore în istoricul de credit'],
      ['delinq', 'Linii de credit cu restanțe'], ['clage', 'Vechimea celei mai vechi linii de credit', 'ani'], ['ninq', 'Cereri recente de credit'],
      ['clno', 'Număr de linii de credit'], ['debtinc', 'Datorii / venit', '%']],
    pd: p => `probabilitatea de neplată estimată de rețea: ${pct(p)}`, out: 'Ce s-a întîmplat', repaid: 'a rambursat creditul', defaulted: 'nu a plătit (neplată)',
    caught: (a, b) => `Neplăți oprite: dumneavoastră ${a}, rețeaua ${b}.`, lost: (a, b) => `Clienți buni respinși: dumneavoastră ${a}, rețeaua ${b}.`
  } : {
    missing: 'not reported', approved: 'approve', refused: 'refuse',
    reason: { DebtCon: 'debt consolidation', HomeImp: 'home improvement' },
    job: { Mgr: 'manager', Office: 'office worker', Other: 'other', ProfExe: 'professional or executive', Sales: 'sales', Self: 'self-employed' },
    rows: [['loan', 'Amount requested', 'USD'], ['mortdue', 'Amount still due on the existing mortgage', 'USD'], ['value', 'Value of the home', 'USD'],
      ['reason', 'Purpose of the loan'], ['job', 'Occupation'], ['yoj', 'Years at the present job'], ['derog', 'Major derogatory reports in the credit history'],
      ['delinq', 'Delinquent credit lines'], ['clage', 'Age of the oldest credit line', 'years'], ['ninq', 'Recent credit inquiries'],
      ['clno', 'Number of credit lines'], ['debtinc', 'Debt-to-income ratio', '%']],
    pd: p => `probability of default estimated by the network: ${pct(p)}`, out: 'What happened', repaid: 'repaid the loan', defaulted: 'defaulted',
    caught: (a, b) => `Defaults stopped: you ${a}, the network ${b}.`, lost: (a, b) => `Good clients refused: you ${a}, the network ${b}.`
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
  window.NNGame.choice({
    game: 'l1',
    cardTitle: { en: 'Application', ro: 'Dosarul' },
    draw: () => {   // a third of the cards are defaults, as in the pool (12 of 36)
      const nb = Math.round(I.per_game * I.pool_bad / (I.pool_bad + I.pool_good));
      return shuffle(shuffle(G.pool.filter(c => c.bad)).slice(0, nb).concat(shuffle(G.pool.filter(c => !c.bad)).slice(0, I.per_game - nb)));
    },
    render: c => '<table class="g-tab"><tbody>' + L.rows.map(([key, lab, unit]) => `<tr><th scope="row">${lab}</th><td>${val(c, key, unit)}</td></tr>`).join('') + '</tbody></table>',
    choices: [{ label: { en: 'Approve the loan', ro: 'Aprob creditul' }, cls: 'g-ok' }, { label: { en: 'Refuse the loan', ro: 'Resping creditul' }, cls: 'g-no' }],
    truth: c => c.bad,
    model: c => (c.p >= 0.5 ? 1 : 0),
    modelName: { en: 'The network of the lecture', ro: 'Rețeaua din curs' },
    reveal: c => `<li>${L.pd(c.p)}</li><li>${L.out}: <b>${c.bad ? L.defaulted : L.repaid}</b></li>`,
    summary: res => {
      const cy = res.filter(r => r.truth && r.you).length, cm = res.filter(r => r.truth && r.model).length;
      const ly = res.filter(r => !r.truth && r.you).length, lm = res.filter(r => !r.truth && r.model).length;
      return `<li>${L.caught(cy, cm)}</li><li>${L.lost(ly, lm)}</li>`;
    }
  });
})();
