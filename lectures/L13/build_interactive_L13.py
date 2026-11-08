"""Build the interactive page of Lecture 13 (deep learning for energy and financial time series; time-series foundation models):
interactive/L13.html (EN) and interactive/L13_ro.html (RO).

Inputs (interactive/L13_data.js) are the forecasts, quantiles and Value-at-Risk series computed by lectures/L13/analysis_L13.py
(lectures/L13/results/*.csv and lectures/L13/cache/var_*.parquet); nothing is re-estimated here. The page logic is in
interactive/L13.js. Run: python3 lectures/L13/build_interactive_L13.py [--no-data]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, play, charts, rnd  # noqa: E402

K = 13
RES = os.path.join(HERE, 'results')
CACHE = os.path.join(HERE, 'cache')
MODELS = ['naive', 'lear', 'dnn', 'nbeats', 'chronos2', 'chronos2_cov', 'bolt', 'timesfm', 'ens']
NAMES = {'en': dict(naive='Naive (weekly/daily)', lear='LEAR', dnn='DNN', nbeats='N-BEATS', chronos2='Chronos-2',
                    chronos2_cov='Chronos-2 + covariates', bolt='Chronos-Bolt', timesfm='TimesFM 2.5', ens='Ensemble LEAR+DNN+Chronos-2'),
         'ro': dict(naive='Naiv (săptămînal/zilnic)', lear='LEAR', dnn='DNN', nbeats='N-BEATS', chronos2='Chronos-2',
                    chronos2_cov='Chronos-2 + covariabile', bolt='Chronos-Bolt', timesfm='TimesFM 2.5', ens='Ansamblu LEAR+DNN+Chronos-2')}
CTXS = [168, 512, 2048, 8192]

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('forecast-week', 'forecast_week', 'epf', 'fw'),
          ('mae-by-hour', 'mae_by_hour', 'epf', 'mh'),
          ('dm-test', 'dm_pvalues', 'eval', 'dm'),
          ('context-length', 'context_length', 'fm', 'cl'),
          ('quantile-fan', 'chronos_fan', 'fm', 'qf'),
          ('var-backtest', 'var_sp500', 'risk', 'va')]

SECTIONS = {
 'en': dict(epf='Day-ahead price forecasts for Romania', eval='Is one forecast really better?',
            fm='Foundation models: context and quantiles', risk='Value-at-Risk with a foundation model'),
 'ro': dict(epf='Prognoza prețurilor pentru ziua următoare în România', eval='Este o prognoză cu adevărat mai bună?',
            fm='Modele fundație: contextul și cuantilele', risk='Valoarea expusă la risc cu un model fundație'),
}

L = {
 'en': dict(h1='Lecture 13 · Deep learning for energy and financial time series; time-series foundation models',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the forecasts of the '
        'lecture when you move a control. The forecasts are those of the test year, 1 November 2025 to 31 October 2026 (360 days with '
        'all 24 hours and every model available): naive rule, LEAR (Lasso-estimated autoregressive model), DNN (deep neural network '
        'with 24 outputs), N-BEATS (neural basis expansion analysis) and the foundation models Chronos-2, Chronos-Bolt and TimesFM '
        '(Time-series Foundation Model), used zero-shot, i.e. without any training on our data. The ▶ button moves the main parameter '
        'step by step. Under each figure: a readout, a question for the room and the Quantlet with the code of the chart.',
   src='Romanian day-ahead prices (hourly, EUR/MWh) and load: energy-charts.info (Fraunhofer Institute for Solar Energy Systems, ISE), '
       'from exchange and ENTSO-E (European Network of Transmission System Operators for Electricity) data. S&P 500 and Bitcoin '
       '(BTC/USD) daily closing prices: course market data (data/clean). All forecasts, quantiles and Value-at-Risk series were computed by '
       'lectures/L13/analysis_L13.py; the page only recombines them (errors, tests, conformal shifts, violation counts).'),
 'ro': dict(h1='Cursul 13 · Deep learning pentru serii de timp din energie și finanțe; modele fundație pentru serii de timp',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser din prognozele cursului cînd modificați '
        'un control. Prognozele sînt cele din anul de test, 1 noiembrie 2025 – 31 octombrie 2026 (360 de zile cu toate cele 24 de ore '
        'și toate modelele disponibile): regula naivă, LEAR (model autoregresiv estimat prin Lasso), DNN (rețea neuronală profundă cu '
        '24 de ieșiri), N-BEATS (analiza prin expansiune în baze neuronale) și modelele fundație Chronos-2, Chronos-Bolt și TimesFM '
        '(model fundație pentru serii de timp), folosite zero-shot, adică fără nicio antrenare pe datele noastre. Butonul ▶ modifică '
        'parametrul principal pas cu pas. Sub fiecare figură: o sinteză numerică, o întrebare pentru sală și Quantlet-ul cu codul figurii.',
   src='Prețurile pentru ziua următoare din România (orare, EUR/MWh) și consumul: energy-charts.info (Institutul Fraunhofer pentru '
       'Sisteme de Energie Solară, ISE), din datele burselor și ale ENTSO-E (Rețeaua europeană a operatorilor de transport și sistem '
       'pentru energie electrică). Prețurile zilnice de închidere ale S&P 500 și Bitcoin (BTC/USD): datele de piață ale cursului '
       '(data/clean). Toate prognozele, cuantilele și seriile de valoare expusă la risc au fost calculate de lectures/L13/analysis_L13.py; '
       'pagina doar le recombină (erori, teste, deplasări conformale, numărul depășirilor).'),
}


def widgets(lang):
    ro = lang == 'ro'
    N = NAMES[lang]
    mopts = [(m, N[m]) for m in MODELS]
    seas = [('all', 'Tot anul de test' if ro else 'Whole test year'), ('win', 'Iarna (dec–feb)' if ro else 'Winter (Dec–Feb)'),
            ('spr', 'Primăvara (mar–mai)' if ro else 'Spring (Mar–May)'), ('sum', 'Vara (iun–aug)' if ro else 'Summer (Jun–Aug)'),
            ('aut', 'Toamna (sep–oct)' if ro else 'Autumn (Sep–Oct)')]
    week = 'Săptămîna (începe lunea)' if ro else 'Week (starting on Monday)'
    return {
     'forecast-week': (slider('fw-w', week, 0, 51, 1, 50)
                       + toggles('fw-m', 'Modele' if ro else 'Models', mopts, ['lear', 'dnn', 'chronos2']) + play('fw'),
                       charts('fw-chart', cls='one')),
     'mae-by-hour': (toggles('mh-m', 'Modele' if ro else 'Models', mopts, ['naive', 'lear', 'dnn', 'chronos2'])
                     + select('mh-s', 'Perioada' if ro else 'Period', seas, 'all'),
                     charts('mh-chart', 'mh-bar')),
     'dm-test': (select('dm-a', 'Modelul A (referința)' if ro else 'Model A (reference)', mopts, 'lear')
                 + select('dm-b', 'Modelul B (este mai bun?)' if ro else 'Model B (is it better?)', mopts, 'chronos2')
                 + radios('dm-l', 'Pierderea' if ro else 'Loss',
                          [('abs', 'eroarea absolută' if ro else 'absolute error'), ('sq', 'eroarea pătratică' if ro else 'squared error')], 'abs'),
                 charts('dm-d', 'dm-c')),
     'context-length': (slider('cl-c', 'Lungimea contextului Chronos-2' if ro else 'Context length of Chronos-2', 0, 3, 1, 2)
                        + slider('cl-w', 'Media mobilă a MAE zilnice (zile)' if ro else 'Moving average of the daily MAE (days)', 1, 60, 1, 30)
                        + play('cl'), charts('cl-bar', 'cl-roll')),
     'quantile-fan': (slider('qf-w', week, 0, 51, 1, 50)
                      + radios('qf-l', 'Intervalul de prognoză' if ro else 'Prediction interval',
                               [('50', '50%'), ('80', '80%'), ('90', '90%')], '80')
                      + toggles('qf-c', 'Corecția' if ro else 'Correction', [('conf', 'conformală (split-conformal)' if ro else 'split-conformal')], [])
                      + slider('qf-n', 'Fereastra de calibrare (zile)' if ro else 'Calibration window (days)', 10, 60, 5, 60)
                      + play('qf'), charts('qf-fan', 'qf-cov')),
     'var-backtest': (select('va-a', 'Activul' if ro else 'Asset', [('SP500', 'S&P 500'), ('BTCUSD', 'Bitcoin (BTC/USD)')], 'SP500')
                      + radios('va-q', 'Nivelul VaR' if ro else 'VaR level',
                               [('01', '1% (99%)'), ('05', '5% (95%)')], '01')
                      + select('va-m', 'Modelul de comparație' if ro else 'Benchmark model',
                               [('hs', 'Simulare istorică, 250 de zile' if ro else 'Historical simulation, 250 days'),
                                ('garch', 'GARCH(1,1) cu erori Student-t' if ro else 'GARCH(1,1) with Student-t errors'),
                                ('ch', 'Chronos-2, cuantila brută' if ro else 'Chronos-2, raw quantile')], 'hs')
                      + slider('va-n', 'Fereastra de calibrare conformală (zile de tranzacționare)' if ro else
                               'Conformal calibration window (trading days)', 50, 250, 10, 250)
                      + play('va'), charts('va-chart', 'va-bar')),
    }


QA = {
 'en': {
  'forecast-week': ('In the lecture week of 19–25 October 2026, which forecast was closest to the actual prices: LEAR, the DNN or '
                    'Chronos-2, a foundation model that never saw Romanian prices during training?',
                    'Chronos-2, with a mean absolute error (MAE) of 25.41 EUR/MWh, against 32.33 EUR/MWh for the DNN and 43.34 EUR/MWh '
                    'for LEAR (lecture numbers). Over the whole test year the ranking is the same at the top: Chronos-2 with covariates '
                    '20.41, Chronos-2 20.72, LEAR 22.79, DNN 23.06 EUR/MWh. Move the week: every model gets the level right and smooths the '
                    'evening spikes; the weeks with the largest errors are those with price jumps that no model saw coming.'),
  'mae-by-hour': ('At which hours of the day are the forecasts worst, and why?',
                  'In the evening: for LEAR the worst hour is 19:00 (35.78 EUR/MWh) and the best 01:00 (10.75 EUR/MWh); Chronos-2 has '
                  'the same worst hour (34.05) and best hour (8.57 EUR/MWh). The evening peak is when solar output has gone and prices '
                  'jump; the night is calm and predictable. Choose spring or summer to see the second bump of errors around midday, '
                  'when solar output pushes prices down and sometimes below zero.'),
  'dm-test': ('Chronos-2 has a lower test-year MAE than LEAR (20.72 against 22.79 EUR/MWh). Is the difference statistically significant?',
              'Yes. The Diebold–Mariano (DM) test compares the daily sums of absolute errors of the two models over the 360 days; '
              'the lecture reports a p-value below 0.0001 for "Chronos-2 is more accurate than LEAR". For DNN against LEAR the p-value '
              'is 0.76 (no evidence that the DNN is better), for N-BEATS against the DNN 0.36. A lower MAE on one test year is not '
              'enough: the curve on the right must rise steadily, not because of a few days.'),
  'context-length': ('Chronos-2 forecasts the next day from the last hours of prices it is given. Does a longer history always help?',
                     'Up to a point. With 168 hours (one week) the MAE is 24.45 EUR/MWh, with 512 hours 21.34, with 2,048 hours '
                     '(about 85 days) 20.72 and with 8,192 hours (about 341 days) 20.95 EUR/MWh (lecture numbers). One week is too '
                     'short to see the weekly pattern repeat often enough; a full year of context brings back the old price regime '
                     'and does not improve on about three months.'),
  'quantile-fan': ('Chronos-2 also gives quantiles. Does its 80% interval contain the actual price 80% of the time?',
                   'Not quite: the raw 80% interval of Chronos-2 covers 76.5% of the hours of the test year (mean width 60.03 EUR/MWh). '
                   'The split-conformal correction widens or narrows the interval by the quantile of the errors of the last 60 days and '
                   'brings coverage to 79.3% (width 63.89 EUR/MWh); for 90%: 88.3% raw, 89.4% conformal. A shorter calibration window '
                   'reacts faster but is noisier.'),
  'var-backtest': ('A 1% Value-at-Risk (VaR) should be breached on about 1% of days. Over January 2020 – October 2026 (1,717 trading days '
                   'of the S&P 500, 17.2 violations expected), how many breaches does the raw Chronos-2 quantile have, and how many after '
                   'the conformal shift?',
                   '22 for the raw quantile (1.28%, Kupiec p = 0.262) and 16 with the conformal shift calibrated on 250 trading days '
                   '(0.93%, Kupiec p = 0.774); historical simulation has 31 (p = 0.003) and GARCH(1,1) 32 (p = 0.001). At 5% the raw '
                   'Chronos-2 quantile is too tight (122 violations against 85.9 expected, p < 0.001) and the conformal shift brings it '
                   'to 93 (p = 0.434). The Kupiec test asks only whether the share of violations equals the nominal level.'),
 },
 'ro': {
  'forecast-week': ('În săptămîna din curs, 19–25 octombrie 2026, care prognoză a fost cea mai apropiată de prețurile realizate: LEAR, '
                    'DNN sau Chronos-2, un model fundație care nu a văzut prețuri românești la antrenare?',
                    'Chronos-2, cu o eroare absolută medie (MAE) de 25,41 EUR/MWh, față de 32,33 EUR/MWh pentru DNN și 43,34 EUR/MWh '
                    'pentru LEAR (cifrele din curs). Pe întregul an de test, ordinea este aceeași în vîrf: Chronos-2 cu covariabile 20,41, '
                    'Chronos-2 20,72, LEAR 22,79, DNN 23,06 EUR/MWh. Mutați săptămîna: toate modelele prind nivelul și netezesc vîrfurile '
                    'de seară; săptămînile cu cele mai mari erori sînt cele cu salturi de preț pe care niciun model nu le-a anticipat.'),
  'mae-by-hour': ('La ce ore ale zilei sînt prognozele cele mai slabe și de ce?',
                  'Seara: pentru LEAR, ora cea mai slabă este 19:00 (35,78 EUR/MWh), iar cea mai bună 01:00 (10,75 EUR/MWh); Chronos-2 '
                  'are aceeași oră cea mai slabă (34,05) și cea mai bună (8,57 EUR/MWh). Vîrful de seară vine cînd producția solară a '
                  'dispărut și prețurile sar; noaptea este liniștită și previzibilă. Alegeți primăvara sau vara pentru a vedea a doua '
                  'cocoașă a erorilor, la amiază, cînd producția solară împinge prețurile în jos, uneori sub zero.'),
  'dm-test': ('Chronos-2 are o MAE mai mică decît LEAR pe anul de test (20,72 față de 22,79 EUR/MWh). Este diferența semnificativă statistic?',
              'Da. Testul Diebold–Mariano (DM) compară sumele zilnice ale erorilor absolute ale celor două modele pe cele 360 de zile; '
              'cursul raportează o valoare p sub 0,0001 pentru „Chronos-2 este mai precis decît LEAR”. Pentru DNN față de LEAR, '
              'valoarea p este 0,76 (nicio dovadă că DNN este mai bun), pentru N-BEATS față de DNN 0,36. O MAE mai mică pe un an de '
              'test nu ajunge: curba din dreapta trebuie să crească constant, nu din cauza cîtorva zile.'),
  'context-length': ('Chronos-2 prognozează ziua următoare din ultimele ore de prețuri pe care le primește. Ajută întotdeauna un istoric mai lung?',
                     'Pînă la un punct. Cu 168 de ore (o săptămînă), MAE este 24,45 EUR/MWh, cu 512 ore 21,34, cu 2.048 de ore (circa '
                     '85 de zile) 20,72, iar cu 8.192 de ore (circa 341 de zile) 20,95 EUR/MWh (cifrele din curs). O săptămînă este prea '
                     'scurtă pentru ca tiparul săptămînal să se repete suficient; un an întreg de context readuce vechiul regim al '
                     'prețurilor și nu aduce nimic peste circa trei luni.'),
  'quantile-fan': ('Chronos-2 dă și cuantile. Conține intervalul său de 80% prețul realizat în 80% din cazuri?',
                   'Nu întocmai: intervalul brut de 80% al Chronos-2 acoperă 76,5% din orele anului de test (lățimea medie 60,03 EUR/MWh). '
                   'Corecția conformală (split-conformal) lărgește sau îngustează intervalul cu cuantila erorilor din ultimele 60 de '
                   'zile și aduce acoperirea la 79,3% (lățimea 63,89 EUR/MWh); pentru 90%: 88,3% brut, 89,4% conformal. O fereastră de '
                   'calibrare mai scurtă reacționează mai repede, dar este mai zgomotoasă.'),
  'var-backtest': ('O valoare expusă la risc (VaR) de 1% ar trebui depășită în circa 1% din zile. În ianuarie 2020 – octombrie 2026 '
                   '(1.717 zile de tranzacționare pentru S&P 500, 17,2 depășiri așteptate), cîte depășiri are cuantila brută a '
                   'Chronos-2 și cîte după deplasarea conformală?',
                   '22 pentru cuantila brută (1,28%, testul Kupiec p = 0,262) și 16 cu deplasarea conformală calibrată pe 250 de zile '
                   'de tranzacționare (0,93%, p = 0,774); simularea istorică are 31 (p = 0,003), iar GARCH(1,1) 32 (p = 0,001). La 5%, '
                   'cuantila brută a Chronos-2 este prea strînsă (122 de depășiri față de 85,9 așteptate, p < 0,001), iar deplasarea '
                   'conformală o aduce la 93 (p = 0,434). Testul Kupiec întreabă doar dacă proporția depășirilor este egală cu nivelul nominal.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 eur: 'EUR/MWh', hour: 'Hour of the day (CET)', mae: 'MAE (EUR/MWh)', price: 'Price (EUR/MWh)', actual: 'Actual price',
 names: %(names_en)s,
 date: d => { const M = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
   const [y, m, dd] = d.split('-'); return `${+dd} ${M[+m - 1]} ${y}`; },
 week: (a, b) => `${a} to ${b}`,
 fwTitle: (a, b) => `Hourly day-ahead prices and forecasts, Romania, ${a} – ${b}`,
 fw: (m, mae, rank) => `${m}: MAE ${mae} EUR/MWh this week (rank ${rank} of the 9 models).`,
 fwNone: 'Tick at least one model.',
 fwYear: (m, mae) => `Whole test year, ${m}: ${mae} EUR/MWh.`,
 mhTitle: s => `MAE by hour of the day, ${s}`, mhBar: s => `MAE over all hours, ${s}`,
 season: { all: 'whole test year', win: 'winter (Dec–Feb)', spr: 'spring (Mar–May)', sum: 'summer (Jun–Aug)', aut: 'autumn (Sep–Oct)' },
 mh: (m, mae, hw, w, hb, b, n) => `${m}: MAE ${mae} EUR/MWh over ${n} days; worst hour ${hw}:00 (${w}), best ${hb}:00 (${b}).`,
 dmD: (a, b) => `Daily loss differential d(t) = loss(${a}) − loss(${b})`, dmC: 'Cumulative sum of d(t)',
 dmY: l => l === 'abs' ? 'Sum of |errors| over 24 h (EUR/MWh)' : 'Sum of squared errors over 24 h (thousands)',
 day: 'Delivery day',
 dm: (a, b, md, s, p, n) => `Mean d(t) = ${md} over ${n} days; DM statistic ${s}; p-value ${p} for H1: "${b} is more accurate than ${a}".`,
 dmYes: (b, a) => `At the 5% level, ${b} is significantly more accurate than ${a}.`,
 dmNo: (b, a) => `No evidence at the 5% level that ${b} is more accurate than ${a}.`,
 dmSame: 'Choose two different models.',
 ctx: c => `${c} h (${Math.round(c / 24)} days)`, ctxAx: 'Context length given to Chronos-2', lear: 'LEAR (3-year window)',
 clBar: 'MAE over the test year', clRoll: w => w === 1 ? 'Daily MAE' : `Daily MAE, ${w}-day moving average`,
 cl: (c, m, d, best) => `Context ${c}: MAE ${m} EUR/MWh (${d} vs 2048 h). The selected context beats 2048 h on ${best} of the days.`,
 clH: (hw, w) => `Worst hour with this context: ${hw}:00 (${w} EUR/MWh).`,
 qfTitle: (l, a, b) => `Chronos-2: ${l}% prediction interval, ${a} – ${b}`, median: 'Median', band: (l, c) => `${l}% interval${c ? ', conformal' : ''}`,
 covTitle: l => `Coverage of the ${l}% interval over the test year`, cov: 'Coverage (%%)', nominal: 'Nominal level',
 raw: 'Raw quantiles', conf: n => `Conformal, ${n}-day window`,
 qf: (l, c, w, cc, cw) => `${l}% interval over the test year: raw coverage ${c} (mean width ${w} EUR/MWh); conformal ${cc} (width ${cw} EUR/MWh).`,
 qfWeek: (c, l) => `This week: ${c} of the hours inside the ${l}% interval shown.`,
 ret: 'Daily log return (%%)', viol: 'Violation', viols: 'Violations', expected: 'Expected',
 vaTitle: (a, q) => `${a}: daily returns and ${q}% Value-at-Risk`,
 vaBar: (q, n) => `Violations of the ${q}% VaR, ${n} trading days`,
 chc: n => `Chronos-2 + conformal (${n} days)`,
 mname: { hs: 'Historical simulation', garch: 'GARCH(1,1)', ch: 'Chronos-2 raw' },
 va: (m, v, r, p) => `${m}: ${v} violations (${r}), Kupiec p = ${p}.`,
 vaExp: (n, e, a, b) => `${n} trading days from ${a} to ${b}; expected ${e} violations.`
};
""",
 'ro': r"""window.LK_TEXT = {
 eur: 'EUR/MWh', hour: 'Ora zilei (CET)', mae: 'MAE (EUR/MWh)', price: 'Prețul (EUR/MWh)', actual: 'Prețul realizat',
 names: %(names_ro)s,
 date: d => { const M = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
   const [y, m, dd] = d.split('-'); return `${+dd} ${M[+m - 1]} ${y}`; },
 week: (a, b) => `${a} – ${b}`,
 fwTitle: (a, b) => `Prețurile orare pentru ziua următoare și prognozele, România, ${a} – ${b}`,
 fw: (m, mae, rank) => `${m}: MAE ${mae} EUR/MWh în această săptămînă (locul ${rank} din cele 9 modele).`,
 fwNone: 'Bifați cel puțin un model.',
 fwYear: (m, mae) => `Întregul an de test, ${m}: ${mae} EUR/MWh.`,
 mhTitle: s => `MAE pe ora zilei, ${s}`, mhBar: s => `MAE pe toate orele, ${s}`,
 season: { all: 'tot anul de test', win: 'iarna (dec–feb)', spr: 'primăvara (mar–mai)', sum: 'vara (iun–aug)', aut: 'toamna (sep–oct)' },
 mh: (m, mae, hw, w, hb, b, n) => `${m}: MAE ${mae} EUR/MWh pe ${window.NNI.cnt(n, 'zi', 'zile')}; ora cea mai slabă ${hw}:00 (${w}), cea mai bună ${hb}:00 (${b}).`,
 dmD: (a, b) => `Diferența zilnică a pierderilor d(t) = pierdere(${a}) − pierdere(${b})`, dmC: 'Suma cumulată a d(t)',
 dmY: l => l === 'abs' ? 'Suma |erorilor| pe 24 h (EUR/MWh)' : 'Suma pătratelor erorilor pe 24 h (mii)',
 day: 'Ziua de livrare',
 dm: (a, b, md, s, p, n) => `Media d(t) = ${md} pe ${window.NNI.cnt(n, 'zi', 'zile')}; statistica DM ${s}; valoarea p ${p} pentru H1: „${b} este mai precis decît ${a}”.`,
 dmYes: (b, a) => `La pragul de 5%, ${b} este semnificativ mai precis decît ${a}.`,
 dmNo: (b, a) => `Nicio dovadă, la pragul de 5%, că ${b} este mai precis decît ${a}.`,
 dmSame: 'Alegeți două modele diferite.',
 ctx: c => `${c} h (${window.NNI.cnt(Math.round(c / 24), 'zi', 'zile')})`, ctxAx: 'Lungimea contextului primit de Chronos-2', lear: 'LEAR (fereastră de 3 ani)',
 clBar: 'MAE pe anul de test', clRoll: w => w === 1 ? 'MAE zilnică' : `MAE zilnică, media mobilă pe ${w % 100 >= 1 && w % 100 <= 19 ? w : w + ' de'} zile`,
 cl: (c, m, d, best) => `Contextul ${c}: MAE ${m} EUR/MWh (${d} față de 2048 h). Contextul ales este mai bun decît 2048 h în ${best} din zile.`,
 clH: (hw, w) => `Ora cea mai slabă cu acest context: ${hw}:00 (${w} EUR/MWh).`,
 qfTitle: (l, a, b) => `Chronos-2: intervalul de prognoză de ${l}%, ${a} – ${b}`, median: 'Mediana', band: (l, c) => `Intervalul de ${l}%${c ? ', conformal' : ''}`,
 covTitle: l => `Acoperirea intervalului de ${l}% pe anul de test`, cov: 'Acoperirea (%%)', nominal: 'Nivelul nominal',
 raw: 'Cuantile brute', conf: n => `Conformal, fereastră de ${n % 100 >= 1 && n % 100 <= 19 ? n : n + ' de'} zile`,
 qf: (l, c, w, cc, cw) => `Intervalul de ${l}% pe anul de test: acoperirea brută ${c} (lățimea medie ${w} EUR/MWh); conformal ${cc} (lățimea ${cw} EUR/MWh).`,
 qfWeek: (c, l) => `În această săptămînă: ${c} din ore în intervalul de ${l}% afișat.`,
 ret: 'Randamentul logaritmic zilnic (%%)', viol: 'Depășire', viols: 'Depășiri', expected: 'Așteptat',
 vaTitle: (a, q) => `${a}: randamentele zilnice și VaR de ${q}%`,
 vaBar: (q, n) => `Depășirile VaR de ${q}%, ${window.NNI.cnt(n, 'zi de tranzacționare', 'zile de tranzacționare')}`,
 chc: n => `Chronos-2 + conformal (${n} de zile)`,
 mname: { hs: 'Simulare istorică', garch: 'GARCH(1,1)', ch: 'Chronos-2 brut' },
 va: (m, v, r, p) => `${m}: ${window.NNI.cnt(v, 'depășire', 'depășiri')} (${r}), testul Kupiec p = ${p}.`,
 vaExp: (n, e, a, b) => `${window.NNI.cnt(n, 'zi de tranzacționare', 'zile de tranzacționare')}, de la ${a} la ${b}; ${e} depășiri așteptate.`
};
""",
}
import json as _json  # noqa: E402
TEXT_JS['en'] = TEXT_JS['en'].replace('%(names_en)s', _json.dumps(NAMES['en'], ensure_ascii=False)).replace('%%', '%')
TEXT_JS['ro'] = TEXT_JS['ro'].replace('%(names_ro)s', _json.dumps(NAMES['ro'], ensure_ascii=False)).replace('%%', '%')


# ---------------------------------------------------------------- data (lecture results only)
def data():
    import numpy as np
    import pandas as pd
    out = {'models': MODELS, 'ctxs': CTXS}
    F = pd.read_csv(os.path.join(RES, 'forecasts_hourly.csv'))
    days = sorted(F['date'].unique())
    assert len(F) == 24 * len(days)
    F = F.sort_values(['date', 'hour'])
    out['days'] = days
    out['act'] = rnd(F['actual'].values, 2)
    out['f'] = {m: rnd(F[m].values, 1) for m in MODELS}
    # context length: daily and hourly MAE of Chronos-2 with each context
    A = F['actual'].values.reshape(-1, 24)
    out['ctx_day'], out['ctx_hour'] = {}, {}
    for c in CTXS:
        E = np.abs(F[f'chronos2_ctx{c}'].values.reshape(-1, 24) - A)
        out['ctx_day'][c] = rnd(E.mean(1), 2)
        out['ctx_hour'][c] = rnd(E.mean(0), 2)
    out['lear_mae'] = round(float(np.abs(F['lear'].values - F['actual'].values).mean()), 2)
    # Chronos-2 quantiles: test days and the 60 days before (conformal calibration)
    Q = pd.read_csv(os.path.join(RES, 'quantiles_hourly.csv')).sort_values(['date', 'hour'])
    qd = sorted(Q['date'].unique())
    assert len(Q) == 24 * len(qd)
    out['q'] = {'days': qd, 'dn': [int((pd.Timestamp(d) - pd.Timestamp('2025-01-01')).days) for d in qd],
                'act': rnd(Q['actual'].values, 2),
                **{f'q{k}': rnd(Q[f'ch2_q{k}'].values, 1) for k in ('05', '10', '25', '50', '75', '90', '95')}}
    # Value-at-Risk: returns, raw quantiles of Chronos-2, historical simulation and GARCH, from 250 trading days before 2020
    out['var'] = {}
    for tic in ('SP500', 'BTCUSD'):
        V = pd.read_parquet(os.path.join(CACHE, f'var_{tic}.parquet')).loc[:'2026-10-31']
        i0 = int(np.searchsorted(V.index.values, np.datetime64('2020-01-01')))
        out['var'][tic] = {'dates': [str(d.date()) for d in V.index], 'i0': i0,
                           **{c: rnd(V[c].values, 3) for c in V.columns}}
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
