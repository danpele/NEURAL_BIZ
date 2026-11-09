"""Build the interactive page of Lecture 5 (recurrent networks: RNN, LSTM and GRU for sequences and financial time series):
interactive/L5.html (EN) and interactive/L5_ro.html (RO).

Inputs (interactive/L5_data.js) come from data/clean (S&P 500, Banca Transilvania, Romanian hourly load) and the saved runs of
analysis_L5.py, through lectures/L5/interactive_data_L5.py; the page logic is in interactive/L5.js.
Run: python3 lectures/L5/build_interactive_L5.py [--no-data]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, play, charts  # noqa: E402

K = 5
BLOCKS = [('acf', 'acf_abs', 'fin', 'ac'),
          ('volatility', 'vol_forecast', 'fin', 'vf'),
          ('returns', 'returns_r2', 'fin', 'rt'),
          ('load-days', 'load_fortnight', 'load', 'ld'),
          ('load-errors', 'load_errors', 'load', 'le')]

SECTIONS = {
 'en': dict(fin='Predictability of financial time series', load='Day-ahead electricity load in Romania'),
 'ro': dict(fin='Predictibilitatea seriilor de timp financiare', load='Consumul de energie electrică din România, pentru ziua următoare'),
}

L = {
 'en': dict(h1='Lecture 5 · Recurrent networks: RNN, LSTM and GRU for sequences and financial time series',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the course data '
        'and the saved forecasts of the lecture when you move a control. The ▶ button moves the main parameter step by step. '
        'Under each figure: a readout, a question for the room and the Quantlet with the code of the chart. Choose a period, a '
        'market or a day and compare the recurrent networks with their benchmarks on the same days.',
   src='Data: S&P 500 daily open, high, low and close (Yahoo Finance, January 2005 to October 2026), Banca Transilvania (TLV) '
       'adjusted closes (EODHD, Bucharest Stock Exchange, from 2008), Romanian hourly electricity load (energy-charts.info, from '
       'ENTSO-E data, 2016 to October 2026, hours in UTC = coordinated universal time). Test periods: 2020 to October 2026 '
       '(markets), 2024 to October 2026 (load). Forecasts of the recurrent neural networks (RNN), long short-term memory (LSTM) '
       'and gated recurrent unit (GRU) networks are the saved runs of the lecture (analysis_L5.py, seeds 20261108 + s); HAR = '
       'heterogeneous autoregressive model, GARCH = generalised autoregressive conditional heteroskedasticity, ARX = '
       'autoregressive model with exogenous inputs, MSE = mean squared error, MAE = mean absolute error, MAPE = mean absolute '
       'percentage error.'),
 'ro': dict(h1='Cursul 5 · Rețele recurente: RNN, LSTM și GRU pentru secvențe și serii de timp financiare',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser din datele cursului și din '
        'prognozele salvate în curs cînd modificați un control. Butonul ▶ modifică parametrul principal pas cu pas. Sub fiecare '
        'figură: o sinteză numerică, o întrebare pentru sală și Quantlet-ul cu codul figurii. Alegeți o perioadă, o piață sau o '
        'zi și comparați rețelele recurente cu modelele de referință, pe aceleași zile.',
   src='Date: S&P 500, prețurile zilnice de deschidere, maxim, minim și închidere (Yahoo Finance, ianuarie 2005 - octombrie 2026), '
       'Banca Transilvania (TLV), prețuri de închidere ajustate (EODHD, Bursa de Valori București, din 2008), consumul orar de '
       'energie electrică din România (energy-charts.info, din datele ENTSO-E, 2016 - octombrie 2026, ore în UTC = timpul universal '
       'coordonat). Perioadele de test: 2020 - octombrie 2026 (piețe), 2024 - octombrie 2026 (consum). Prognozele rețelelor '
       'neuronale recurente (RNN), ale rețelelor cu memorie lungă pe termen scurt (LSTM) și ale celor cu unități recurente cu porți '
       '(GRU) sînt rulările salvate din curs (analysis_L5.py, semințele 20261108 + s); HAR = model autoregresiv heterogen, GARCH = '
       'model autoregresiv generalizat cu heteroscedasticitate condiționată, ARX = model autoregresiv cu variabile exogene, MSE = '
       'eroarea pătratică medie, MAE = eroarea absolută medie, MAPE = eroarea procentuală absolută medie.'),
}


def widgets(lang):
    ro = lang == 'ro'
    mk = [('SP500', 'S&P 500'), ('TLV', 'Banca Transilvania (TLV)')]
    lmod = [('snaive', 'Naiv sezonier' if ro else 'Seasonal naive'),
            ('ridge', 'Ridge ARX'), ('lstm', 'LSTM'), ('gru', 'GRU')]
    return {
     'acf': (select('ac-m', 'Piața' if ro else 'Market', mk, 'SP500')
             + radios('ac-t', 'Seria' if ro else 'Series', [('r', 'randamente rₜ' if ro else 'returns rₜ'), ('a', 'randamente absolute |rₜ|' if ro else 'absolute returns |rₜ|'),
                                                            ('s', 'randamente la pătrat rₜ²' if ro else 'squared returns rₜ²')], 'a')
             + slider('ac-l', 'Laguri afișate (zile de tranzacționare)' if ro else 'Lags shown (trading days)', 10, 250, 10, 50),
             charts('ac-chart', cls='one')),
     'volatility': (slider('vf-s', 'Începutul ferestrei' if ro else 'Start of the window', 0, 81, 1, 0)
                    + slider('vf-n', 'Lungimea ferestrei (luni)' if ro else 'Length of the window (months)', 1, 82, 1, 12)
                    + toggles('vf-m', 'Modele' if ro else 'Models', [('naive', 'Naiv (ziua de ieri)' if ro else 'Naive (yesterday)'), ('har', 'HAR'), ('garch', 'GARCH(1,1)'),
                                                                     ('rnn', 'RNN'), ('lstm', 'LSTM'), ('gru', 'GRU')], ['har', 'lstm'])
                    + play('vf'), charts('vf-chart', cls='one')),
     'returns': (select('rt-m', 'Piața' if ro else 'Market', mk, 'SP500')
                 + select('rt-s', 'LSTM', [('ens', 'media celor 5 semințe' if ro else 'mean of the 5 seeds')]
                          + [(str(s), (f'sămînța {s + 1}' if ro else f'seed {s + 1}')) for s in range(5)], 'ens')
                 + toggles('rt-k', 'Modele' if ro else 'Models', [('mean', 'Media de antrenare' if ro else 'Training mean'), ('ar', 'AR(1)'), ('lstm', 'LSTM')],
                           ['mean', 'ar', 'lstm']), charts('rt-chart', cls='one')),
     'load-days': (slider('ld-d', 'Prima zi afișată' if ro else 'First day shown', 0, 1034, 1, 385)
                   + select('ld-w', 'Fereastra' if ro else 'Window', [(1, 'o zi' if ro else 'one day'), (7, 'o săptămînă' if ro else 'one week'),
                                                                      (14, 'două săptămîni' if ro else 'two weeks')], 14)
                   + toggles('ld-m', 'Modele' if ro else 'Models', lmod, ['snaive', 'lstm']) + play('ld'),
                   charts('ld-chart', cls='one')),
     'load-errors': (radios('le-f', 'Zilele' if ro else 'Days', [('all', 'toate' if ro else 'all'), ('wd', 'lucrătoare (luni–vineri)' if ro else 'weekdays (Mon–Fri)'),
                                                                ('we', 'weekend')], 'all')
                     + select('le-s', 'Sezonul' if ro else 'Season', [('all', 'tot anul' if ro else 'whole year'), ('win', 'iarna (dec–feb)' if ro else 'winter (Dec–Feb)'),
                                                                     ('spr', 'primăvara (mar–mai)' if ro else 'spring (Mar–May)'), ('sum', 'vara (iun–aug)' if ro else 'summer (Jun–Aug)'),
                                                                     ('aut', 'toamna (sep–nov)' if ro else 'autumn (Sep–Nov)')], 'all')
                     + toggles('le-m', 'Modele' if ro else 'Models', lmod, ['snaive', 'ridge', 'lstm', 'gru']),
                     charts('le-chart', cls='one')),
    }


QA = {
 'en': {
  'acf': ('S&P 500 daily returns are almost uncorrelated, yet a recurrent network is used to forecast volatility. What does the '
          'autocorrelation of absolute returns show?',
          'Returns themselves have a first-order autocorrelation of −0.119 and only 15 of the first 50 lags outside the ±0.026 band '
          '(1.96/√T for white noise). Absolute returns have ρ(1) = 0.315 and all 50 lags are significant, still 0.126 at lag 50: '
          'calm and turbulent periods cluster (volatility clustering). The size of tomorrow\'s move is predictable even if its sign '
          'is not, which is why task A of the lecture forecasts volatility. For Banca Transilvania the pattern is the same but '
          'weaker (|r|: ρ(1) = 0.227).'),
  'volatility': ('Over the test period 2020–2026, the LSTM reduces the MSE of the log variance by 11.6% against HAR. Does it win in '
                 'every sub-period?',
                 'Over the whole test the MSE is 0.966 for the naive forecast, 0.786 for GARCH, 0.706 for HAR and 0.625 for the LSTM '
                 '(mean of 5 seeds; GRU 0.624, RNN 0.643). The gain is largest in the 2020 crash (20 February to 30 April: LSTM 0.521, '
                 'HAR 1.014, GARCH 1.032): the network reacts faster to the jump in range volatility. In 2022 the ranking is the same '
                 'but the gaps are smaller (0.585 against 0.623). Move the window: short windows give noisy rankings, which is why the '
                 'lecture uses the Diebold–Mariano test on the whole test period (LSTM vs HAR: statistic −5.52).'),
  'returns': ('The AR(1) model has an out-of-sample R² of 2.4% on S&P 500 returns, more than the LSTM (0.8%). Where does this R² come from?',
              'Almost entirely from the crash of February–April 2020: the AR(1) slope is −0.10 (reversals), and on those violent days '
              'it pays off. The cumulative curve jumps in March 2020 and then drifts sideways or down; outside 20 February – 30 April '
              '2020 the AR(1) R² is −0.2%. The LSTM ensemble earns about two thirds of its gain in the same weeks. Its sign is right on 52.1% of the days '
              '(mean of the 5 seeds; 53.1% for their average forecast), below the share of up days (54.0%). On TLV the R² of all models is between 0.1% and 0.4%. Daily returns are close '
              'to unpredictable; a small positive R² is not a trading strategy.'),
  'load-days': ('In the fortnight from 20 January 2025, which forecast follows the load best, and where does the seasonal naive fail?',
                'Over these 14 days the MAPE is 6.0% for the seasonal naive forecast (same day of the previous week), 3.2% for the LSTM, '
                '2.9% for the GRU and 2.6% for the ridge ARX. The naive forecast fails when the level changes from one week to the next '
                '(cold spells, holidays): it repeats last week\'s level. Over the whole test period (1,035 days) the MAPE is 6.79% '
                '(naive), 4.86% (ridge ARX), 5.81% (LSTM) and 5.17% (GRU). Scroll through the days around Christmas or Easter: every '
                'model misses holidays it has no input for.'),
  'load-errors': ('At which hours of the day are the day-ahead forecasts worst, and are weekends harder than weekdays?',
                  'The errors peak in the late morning (around 10:00–11:00 UTC), when the ramp-up of demand and the output of rooftop '
                  'solar make the net load hardest to predict: 681 MW for the naive forecast, 537 MW for the ridge ARX, 581 MW for the '
                  'LSTM. Weekends are not harder: the MAE of the LSTM is 342 MW on weekdays and 303 MW on weekends. Over all test days '
                  'the ridge ARX has the lowest MAE (269 MW, against 293 MW for the GRU, 331 MW for the LSTM and 396 MW for the naive '
                  'forecast): a linear model with the right lags is a hard benchmark for a recurrent network.'),
 },
 'ro': {
  'acf': ('Randamentele zilnice ale S&P 500 sînt aproape necorelate, totuși folosim o rețea recurentă pentru a prognoza volatilitatea. '
          'Ce arată autocorelația randamentelor absolute?',
          'Randamentele însele au o autocorelație de ordinul 1 de −0,119 și doar 15 din primele 50 de laguri ies din banda de ±0,026 '
          '(1,96/√T pentru zgomot alb). Randamentele absolute au ρ(1) = 0,315, iar toate cele 50 de laguri sînt semnificative, '
          'încă 0,126 la lagul 50: perioadele calme și cele agitate se grupează (gruparea volatilității). Mărimea mișcării de mîine '
          'este previzibilă, chiar dacă semnul ei nu este; de aceea sarcina A din curs prognozează volatilitatea. Pentru Banca '
          'Transilvania tiparul este același, dar mai slab (|r|: ρ(1) = 0,227).'),
  'volatility': ('Pe perioada de test 2020–2026, LSTM reduce MSE a log-varianței cu 11,6% față de HAR. Cîștigă în fiecare subperioadă?',
                 'Pe tot setul de test, MSE este 0,966 pentru prognoza naivă, 0,786 pentru GARCH, 0,706 pentru HAR și 0,625 pentru LSTM '
                 '(media a 5 semințe; GRU 0,624, RNN 0,643). Cîștigul este cel mai mare în criza din 2020 (20 februarie - 30 aprilie: '
                 'LSTM 0,521, HAR 1,014, GARCH 1,032): rețeaua reacționează mai repede la saltul volatilității. În 2022, ordinea este '
                 'aceeași, dar diferențele sînt mai mici (0,585 față de 0,623). Mutați fereastra: ferestrele scurte dau clasamente '
                 'zgomotoase; de aceea cursul folosește testul Diebold–Mariano pe toată perioada de test (LSTM contra HAR: statistica −5,52).'),
  'returns': ('Modelul AR(1) are un R² în afara eșantionului de 2,4% pe randamentele S&P 500, mai mult decît LSTM (0,8%). De unde vine acest R²?',
              'Aproape în întregime din criza din februarie–aprilie 2020: panta AR(1) este −0,10 (reveniri), iar în acele zile violente '
              'ea aduce cîștig. Curba cumulată sare în martie 2020, apoi merge lateral sau în jos; în afara intervalului 20 februarie - '
              '30 aprilie 2020, R² al AR(1) este −0,2%. Ansamblul LSTM obține aproximativ două treimi din cîștigul său în aceleași '
              'săptămîni. Semnul este corect în 52,1% din zile (media celor 5 semințe; 53,1% pentru prognoza lor medie), sub proporția zilelor de creștere (54,0%). Pe TLV, R² al tuturor '
              'modelelor este între 0,1% și 0,4%. Randamentele zilnice sînt aproape imprevizibile; un R² mic și pozitiv nu este o '
              'strategie de tranzacționare.'),
  'load-days': ('În cele două săptămîni de la 20 ianuarie 2025, care prognoză urmărește cel mai bine consumul și unde greșește prognoza naivă sezonieră?',
                'În aceste 14 zile, MAPE este 6,0% pentru prognoza naivă sezonieră (aceeași zi din săptămîna anterioară), 3,2% pentru '
                'LSTM, 2,9% pentru GRU și 2,6% pentru ridge ARX. Prognoza naivă greșește cînd nivelul se schimbă de la o săptămînă la '
                'alta (valuri de frig, sărbători): repetă nivelul de săptămîna trecută. Pe toată perioada de test (1.035 de zile), MAPE '
                'este 6,79% (naiv), 4,86% (ridge ARX), 5,81% (LSTM) și 5,17% (GRU). Parcurgeți zilele din jurul Crăciunului sau al '
                'Paștelui: toate modelele ratează sărbătorile pentru care nu au o variabilă de intrare.'),
  'load-errors': ('La ce ore din zi sînt cele mai mari erorile prognozelor pentru ziua următoare și sînt weekendurile mai dificile decît '
                  'zilele lucrătoare?',
                  'Erorile au un vîrf la sfîrșitul dimineții (în jurul orelor 10:00–11:00 UTC), cînd creșterea cererii și producția '
                  'panourilor solare de pe acoperișuri fac consumul net cel mai greu de prognozat: 681 MW pentru prognoza naivă, 537 MW '
                  'pentru ridge ARX, 581 MW pentru LSTM. Weekendurile nu sînt mai dificile: MAE pentru LSTM este 342 MW în zilele '
                  'lucrătoare și 303 MW în weekend. Pe toate zilele de test, ridge ARX are cea mai mică MAE (269 MW, față de 293 MW pentru '
                  'GRU, 331 MW pentru LSTM și 396 MW pentru prognoza naivă): un model liniar cu lagurile potrivite este un reper greu de '
                  'depășit pentru o rețea recurentă.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 mk: { SP500: 'S&P 500', TLV: 'Banca Transilvania (TLV)' }, ser: { r: 'returns rₜ', a: 'absolute returns |rₜ|', s: 'squared returns rₜ²' },
 lag: 'Lag (trading days)', acf: 'Autocorrelation', band: '±1.96/√T band (white noise)',
 acTitle: (m, s, a, b) => `${m}: autocorrelation of daily ${s}, ${a} to ${b}`,
 ac: (r1, r10, n, L, b, T) => `ρ(1) = <b>${r1}</b>, ρ(10) = ${r10}; ${n} of ${L} lags outside ±${b} (${T} trading days).`,
 mod: { obs: 'Observed (Garman–Klass range)', naive: 'Naive (yesterday)', har: 'HAR', garch: 'GARCH(1,1)', rnn: 'RNN (mean of 5 seeds)', lstm: 'LSTM (mean of 5 seeds)', gru: 'GRU (mean of 5 seeds)' },
 vol: 'Annualised volatility (%)', vfTitle: (a, b) => `S&P 500, one-day-ahead volatility forecasts, ${a} to ${b}`,
 vf: (m, e, r) => `${m}: MSE of the log variance <b>${e}</b> (rank ${r})`, vf0: (n, a, b) => `${n} trading days from ${a} to ${b}; whole test period: naive 0.966, GARCH 0.786, HAR 0.706, LSTM 0.625.`,
 rtMod: { mean: 'Training mean', ar: 'AR(1)', lstm: 'LSTM' }, cum: 'Cumulative Σ r² − Σ (r − forecast)²  (%²)',
 rtTitle: m => `${m}: gain over the zero forecast, cumulated over the test days (rising = better)`,
 rt: (m, r2, h) => `${m}: out-of-sample R² <b>${r2}</b>${h ? `; correct sign on ${h} of the days` : ''}.`, rt0: (n, up) => `${n} test days; ${up} of them are up days.`,
 lmod: { obs: 'Observed load', snaive: 'Seasonal naive', ridge: 'Ridge ARX', lstm: 'LSTM (mean of 3 seeds)', gru: 'GRU (mean of 3 seeds)' },
 mw: 'Load (MW)', hourUTC: 'Hour of day d+1 (UTC)', ldTitle: (a, b) => a === b ? `Day-ahead forecasts of the Romanian load, ${a}` : `Day-ahead forecasts of the Romanian load, ${a} to ${b}`,
 ld: (m, w, t) => `${m}: MAPE <b>${w}</b> on the days shown (whole test period ${t}).`,
 mae: 'MAE (MW)', leTitle: n => `Mean absolute error by hour, ${n} test days`,
 le: (m, a, p, h, hm) => `${m}: MAE <b>${a} MW</b>, MAPE ${p}; worst hour ${h}:00 UTC (${hm} MW).`
};
""",
 'ro': r"""const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 mk: { SP500: 'S&P 500', TLV: 'Banca Transilvania (TLV)' }, ser: { r: 'randamente rₜ', a: 'randamente absolute |rₜ|', s: 'randamente la pătrat rₜ²' },
 lag: 'Lagul (zile de tranzacționare)', acf: 'Autocorelația', band: 'banda ±1,96/√T (zgomot alb)',
 acTitle: (m, s, a, b) => `${m}: autocorelația pentru ${s} zilnice, ${a} - ${b}`,
 ac: (r1, r10, n, L, b, T) => `ρ(1) = <b>${r1}</b>, ρ(10) = ${r10}; ${n} din ${deRO(L)} laguri în afara benzii ±${b} (${deRO(T)} zile de tranzacționare).`,
 mod: { obs: 'Observat (amplitudine Garman–Klass)', naive: 'Naiv (ziua de ieri)', har: 'HAR', garch: 'GARCH(1,1)', rnn: 'RNN (media a 5 semințe)', lstm: 'LSTM (media a 5 semințe)', gru: 'GRU (media a 5 semințe)' },
 vol: 'Volatilitatea anualizată (%)', vfTitle: (a, b) => `S&P 500, prognoze ale volatilității pentru ziua următoare, ${a} - ${b}`,
 vf: (m, e, r) => `${m}: MSE a log-varianței <b>${e}</b> (locul ${r})`, vf0: (n, a, b) => `${deRO(n)} zile de tranzacționare, de la ${a} la ${b}; toată perioada de test: naiv 0,966, GARCH 0,786, HAR 0,706, LSTM 0,625.`,
 rtMod: { mean: 'Media de antrenare', ar: 'AR(1)', lstm: 'LSTM' }, cum: 'Σ r² − Σ (r − prognoza)², cumulat  (%²)',
 rtTitle: m => `${m}: cîștigul față de prognoza zero, cumulat pe zilele de test (creștere = mai bine)`,
 rt: (m, r2, h) => `${m}: R² în afara eșantionului <b>${r2}</b>${h ? `; semnul corect în ${h} din zile` : ''}.`, rt0: (n, up) => `${deRO(n)} zile de test; ${up} dintre ele sînt zile de creștere.`,
 lmod: { obs: 'Consumul observat', snaive: 'Naiv sezonier', ridge: 'Ridge ARX', lstm: 'LSTM (media a 3 semințe)', gru: 'GRU (media a 3 semințe)' },
 mw: 'Consumul (MW)', hourUTC: 'Ora din ziua d+1 (UTC)', ldTitle: (a, b) => a === b ? `Prognoze pentru ziua următoare ale consumului din România, ${a}` : `Prognoze pentru ziua următoare ale consumului din România, ${a} - ${b}`,
 ld: (m, w, t) => `${m}: MAPE <b>${w}</b> pe zilele afișate (toată perioada de test ${t}).`,
 mae: 'MAE (MW)', leTitle: n => `Eroarea absolută medie pe ore, ${deRO(n)} zile de test`,
 le: (m, a, p, h, hm) => `${m}: MAE <b>${a} MW</b>, MAPE ${p}; ora cea mai slabă ${h}:00 UTC (${hm} MW).`
};
""",
}


def data():
    import interactive_data_L5 as idl
    return idl.load()


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
