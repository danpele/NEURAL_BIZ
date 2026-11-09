"""Build the interactive page of Lecture 10 (Reinforcement learning: trading, hedging and battery dispatch):
interactive/L10.html (EN) and interactive/L10_ro.html (RO).

Inputs (interactive/L10_data.js) are computed here from data/clean (hourly day-ahead prices of RO, HU, DE_LU, BG; S&P 500 and
VIX from markets_daily.csv) and from the trained results cached by analysis_L10.py in lectures/L10/cache (Q-table, test-day
revenues of the Q-learning and double DQN agents, PPO positions, deep-hedging P&L). The page logic is in interactive/L10.js.
Run: python3 lectures/L10/build_interactive_L10.py [--no-data]
(The loop in lectures/common/interactive_common.py rebuilds L1-L7 only; run this file directly.)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, play, charts, rnd  # noqa: E402

K = 10
ZONES = ['RO', 'HU', 'DE_LU', 'BG']

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('dp-day', 'dp_day', 'bat', 'dd'),
          ('dp-value', 'dp_value', 'bat', 'dv'),
          ('q-policy', 'q_policy', 'val', 'qp'),
          ('battery-test', 'battery_test_cum', 'val', 'bt'),
          ('trading', 'trading_equity', 'pol', 'tq'),
          ('hedging', 'hedge_pnl', 'hed', 'hg')]

SECTIONS = {
 'en': dict(bat='The battery as a Markov decision process (MDP): dynamic programming (DP)',
            val='Value-based learning: Q-learning and the deep Q-network (DQN)',
            pol='Policy gradients: proximal policy optimisation (PPO) for trading',
            hed='Deep hedging of a short call'),
 'ro': dict(bat='Bateria ca proces decizional Markov (MDP): programarea dinamică (DP)',
            val='Învățarea bazată pe valoare: Q-learning și rețeaua Q profundă (DQN)',
            pol='Gradienți de politică: optimizarea proximală a politicii (PPO) în tranzacționare',
            hed='Acoperirea neuronală a unei opțiuni call vîndute'),
}

L = {
 'en': dict(h1='Lecture 10 · Reinforcement learning: trading, hedging and battery dispatch',
   note='Reinforcement learning (RL) on three real problems: a 2 MWh (megawatt-hour) battery trading on the day-ahead power '
        'market, a position in the S&P 500 index that is either flat or long, and the daily hedge of a one-month call option. The '
        'slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the course data '
        '(data/clean) and from the decisions of the trained agents when you move a control. The ▶ button moves the main parameter '
        'step by step. Under each figure: a readout, a question for the room and the Quantlet with the code of the chart.',
   src='Hourly day-ahead prices: OPCOM (Romania), HUPX (Hungary), EPEX (Germany–Luxembourg), IBEX (Bulgaria), via energy-charts.info '
       '(Fraunhofer ISE); delivery hours in Central European Time (CET/CEST), days with 24 complete hours; since October 2025 each hour '
       'is the mean of four 15-minute prices. S&P 500 and Cboe Volatility Index (VIX): Yahoo Finance, daily closes. The agents were '
       'trained once by lectures/L10/analysis_L10.py (five seeds, 20261108–20261112); the page replays their test-period decisions and '
       'recomputes the dynamic programme, the costs and the risk measures. Data to the cut-off of the lecture, 31 October 2026.'),
 'ro': dict(h1='Cursul 10 · Reinforcement learning: tranzacționare, acoperirea riscului și dispecerizarea bateriilor',
   note='Reinforcement learning (RL) pe trei probleme reale: o baterie de 2 MWh (megawatt-oră) care '
        'tranzacționează pe piața pentru ziua următoare, o poziție în indicele S&P 500 care este fie în afara pieței, fie investită, și '
        'acoperirea zilnică a unei opțiuni call pe o lună. În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în '
        'browser din datele cursului (data/clean) și din deciziile agenților antrenați cînd modificați un control. Butonul ▶ modifică '
        'parametrul principal pas cu pas. Sub fiecare figură: o sinteză numerică, o întrebare pentru sală și Quantlet-ul cu codul figurii.',
   src='Prețuri orare pentru ziua următoare: OPCOM (România), HUPX (Ungaria), EPEX (Germania–Luxemburg), IBEX (Bulgaria), prin '
       'energy-charts.info (Fraunhofer ISE); orele de livrare în ora Europei Centrale (CET/CEST), zile cu 24 de ore complete; din '
       'octombrie 2025, fiecare oră este media a patru prețuri pe 15 minute. S&P 500 și indicele de volatilitate Cboe (VIX): Yahoo '
       'Finance, prețuri zilnice de închidere. Agenții au fost antrenați o singură dată de lectures/L10/analysis_L10.py (cinci valori '
       'inițiale, 20261108–20261112); pagina reia deciziile lor din perioada de test și recalculează programarea dinamică, costurile și '
       'măsurile de risc. Date pînă la data-limită a cursului, 31 octombrie 2026.'),
}


def widgets(lang):
    ro = lang == 'ro'
    Z = dict(RO='România', HU='Ungaria', DE_LU='Germania–Luxemburg', BG='Bulgaria') if ro else \
        dict(RO='Romania', HU='Hungary', DE_LU='Germany–Luxembourg', BG='Bulgaria')
    seeds = [('avg', 'Media celor 5 valori inițiale' if ro else 'Mean of the 5 seeds')] + \
            [(str(i), (f'Valoarea inițială {i + 1} ({20261108 + i})' if ro else f'Seed {i + 1} ({20261108 + i})')) for i in range(5)]
    quart = (['cele mai ieftine 6 ore ieri', 'quartila a 2-a', 'quartila a 3-a', 'cele mai scumpe 6 ore ieri'] if ro else
             ["yesterday's 6 cheapest hours", '2nd quartile', '3rd quartile', "yesterday's 6 dearest hours"])
    return {
     'dp-day': (select('dd-z', 'Zona de licitație' if ro else 'Bidding zone', [(z, Z[z]) for z in ZONES], 'RO')
                + slider('dd-d', 'Ziua de livrare' if ro else 'Delivery day', 0, 1, 1, 0)
                + slider('dd-e', 'Randamentul ciclului complet (încărcare și descărcare)' if ro else 'Round-trip efficiency (charge and discharge)', 50, 100, 5, 85)
                + slider('dd-c', 'Capacitatea bateriei (se mută 1 MWh pe oră)' if ro else 'Battery capacity (1 MWh moved per hour)', 1, 6, 1, 2)
                + play('dd'), charts('dd-chart', cls='one')),
     'dp-value': (slider('dv-k', 'Pașii inducției înapoi (ore rezolvate, de la miezul nopții spre 0:00)' if ro else
                         'Steps of backward induction (hours solved, from midnight back to 0:00)', 0, 24, 1, 24)
                  + select('dv-v', 'Ce se afișează' if ro else 'Show', [('v', 'valoarea V_h(s)' if ro else 'the value V_h(s)'),
                           ('mwh', 'valoarea unui MWh stocat în plus, V_h(s+1) − V_h(s)' if ro else 'the value of one more stored MWh, V_h(s+1) − V_h(s)')], 'v')
                  + play('dv'), charts('dv-chart', 'dv-pol')),
     'q-policy': (radios('qp-s', 'Energia stocată' if ro else 'Stored energy', [(0, '0 MWh'), (1, '1 MWh'), (2, '2 MWh')], 1)
                  + slider('qp-h', 'Ora (CET)' if ro else 'Hour (CET)', 0, 23, 1, 13)
                  + select('qp-b', 'Rangul orei în prețurile de ieri' if ro else "Rank of the hour in yesterday's prices", list(enumerate(quart)), 0)
                  + play('qp'), charts('qp-map', 'qp-bar', cls='one')),
     'battery-test': (toggles('bt-k', 'Strategii' if ro else 'Strategies',
                              [('pf', 'previziune perfectă' if ro else 'perfect foresight'), ('dqn', 'DQN dublu' if ro else 'double DQN'),
                               ('q', 'Q-learning tabelar' if ro else 'tabular Q-learning'), ('nv', 'planul optim de ieri' if ro else "yesterday's optimal plan")])
                      + select('bt-seed', 'Agenții care învață' if ro else 'Learning agents', seeds, 'avg')
                      + select('bt-v', 'Scala' if ro else 'Scale', [('cum', 'venit cumulat (mii EUR)' if ro else 'cumulative revenue (thousand EUR)'),
                                                                   ('cap', '% din venitul cu previziune perfectă' if ro else '% of the perfect-foresight revenue')], 'cum')
                      + slider('bt-m', 'Începutul ferestrei' if ro else 'Start of the window', 0, 21, 1, 0)
                      + play('bt'), charts('bt-chart', cls='one')),
     'trading': (toggles('tq-k', 'Strategii' if ro else 'Strategies',
                         [('bh', 'cumpără și păstrează' if ro else 'buy and hold'), ('ma', 'media mobilă de 200 de zile' if ro else '200-day moving average'),
                          ('ppo', 'agentul PPO' if ro else 'PPO agent')])
                 + select('tq-seed', 'Agentul PPO' if ro else 'PPO agent', seeds, 'avg')
                 + slider('tq-c', 'Costul unei tranzacții (puncte de bază, pb)' if ro else 'Cost per trade (basis points, bp)', 0, 30, 1, 5)
                 + slider('tq-y', 'Începutul ferestrei de test (1 ianuarie)' if ro else 'Start of the test window (1 January)', 2020, 2026, 1, 2020)
                 + play('tq'), charts('tq-eq', 'tq-ex', cls='one')),
     'hedging': (toggles('hg-k', 'Strategii' if ro else 'Strategies',
                         [('nh', 'fără acoperire' if ro else 'no hedge'), ('bs', 'delta Black-Scholes' if ro else 'Black-Scholes delta'),
                          ('dh', 'acoperire neuronală' if ro else 'deep hedge')])
                 + select('hg-seed', 'Acoperirea neuronală' if ro else 'Deep hedge', seeds, 'avg')
                 + select('hg-r', 'Volatilitatea la începutul lunii (VIX)' if ro else 'Volatility at the start of the month (VIX)',
                          [('all', 'toate zilele de început' if ro else 'all start days'), ('lo', 'VIX sub 20' if ro else 'VIX below 20'),
                           ('hi', 'VIX de cel puțin 20' if ro else 'VIX of 20 or more')], 'all')
                 + slider('hg-a', 'Nivelul pierderii așteptate în coadă (ES)' if ro else 'Level of the expected shortfall (ES)', 90, 99, 1, 95)
                 + play('hg'), charts('hg-hist', 'hg-es')),
    }


QA = {
 'en': {
  'dp-day': ('On 30 June 2026 the Romanian day-ahead price ranged from 100.65 EUR/MWh at 13:00 to 954.59 EUR/MWh at 20:00, and a 2 MWh '
             'battery earned 1,541 EUR with perfect foresight. Does a 4 MWh battery earn twice as much?',
             'Almost, not quite: 2,885 EUR, 87% more. The battery moves 1 MWh per hour, so the third and fourth MWh are bought in less '
             'cheap hours and sold in less dear ones; with 6 MWh the revenue is 3,282 EUR, only 14% above 4 MWh, because the evening peak '
             'lasts a few hours. A perfect battery (100% round trip) would earn 1,745 EUR at 2 MWh. On the same day Hungary gives 1,466 EUR, '
             'Germany–Luxembourg 708 EUR and Bulgaria 744 EUR: the revenue follows the daily spread, not the price level. All these numbers '
             'are upper bounds, because dynamic programming (DP) sees the 24 prices of the day before they are known.'),
  'dp-value': ('Go back hour by hour with ▶. On 30 June 2026, what is one stored MWh worth at 12:00, at 18:00 and at 23:00?',
               'At 12:00, V<sub>12</sub>(0) = 1,541 EUR and V<sub>12</sub>(2) = 1,760 EUR, so each stored MWh is worth 109.38 EUR: the '
               'shadow price of energy in the battery, about what it costs to buy it at midday. At 18:00 each MWh is worth about 880 EUR, '
               'because the two dearest hours (19:00 and 20:00) are still ahead. At 23:00 (switch to the value of one more stored MWh) the first MWh is worth 211 EUR (it can still be '
               'sold in the last hour) and the second nothing, since only 1 MWh can be moved per hour. The Bellman equation builds these '
               'values backwards from V<sub>24</sub>(s) = 0, and the optimal action at each (h, s) is the one that attains the maximum.'),
  'q-policy': ("The tabular agent does not see today's prices, only how each hour ranked yesterday. At 13:00, with 1 MWh stored, in an hour "
               "that was among yesterday's six cheapest, which action does it choose, and how sure is it?",
               'It charges: Q(charge) = 175.0 EUR, Q(idle) = 167.2 EUR, Q(discharge) = 139.6 EUR, the values of the rest of the day after '
               'each action, learned from 1,746 training days (2019–2023). The margin over idling is small, about 8 EUR. With 1 MWh stored, '
               "the row of yesterday's cheapest hours charges in 7 hours and discharges in 8, mostly from 18:00 on: the hour of the day "
               'dominates, as in the lecture. With 0 MWh the agent can only charge or wait, with 2 MWh only discharge or wait (action '
               'masking). The table has 24 × 4 × 3 = 288 states and can be printed and audited.'),
  'battery-test': ('On the 638 test days, 1 January 2025 to 24 October 2026, which captures more of the perfect-foresight revenue: the double '
                   "deep Q-network (DQN), tabular Q-learning, or simply repeating yesterday's optimal plan?",
                   "Yesterday's plan: 163.1 thousand EUR, 80.7% of the 202.1 thousand EUR upper bound. Tabular Q-learning captures 115.2 "
                   'thousand EUR (57.0%) and the double DQN 101.3 thousand EUR (50.1%, mean of 5 seeds; choose the seeds one by one to see '
                   'the spread between them). The DQN beats the naive plan on only 9% of the days. The learning agents see a coarse summary of '
                   "yesterday (price quartiles and a few features), the naive plan the full 24-hour profile of yesterday, and prices are "
                   'strongly autocorrelated from one day to the next: a simple benchmark is hard to beat.'),
  'trading': ('With a cost of 5 basis points (bp) per trade, the proximal policy optimisation (PPO) agent has a higher Sharpe ratio than buy '
              'and hold over 2020–2026 (0.83 against 0.61, mean of the five seeds). Does the advantage survive higher costs or a later start?',
              'Only partly. At 20 bp the mean Sharpe ratio of the seeds falls to 0.63, level with buy and hold (0.61); at 30 bp it is 0.50, '
              'and two seeds that trade often drop to 0.02 and 0.07. Starting in 2023, buy and hold has a Sharpe ratio of 1.23 against 1.04 '
              'for PPO at 5 bp: the advantage comes from 2020 and 2022, when the agent was partly out of the market (exposure chart). The '
              'seeds differ (0.72 to 0.95 at 5 bp) and the block-bootstrap interval of the Sharpe difference in the lecture, −0.08 to 0.56, '
              'contains zero. One backtest of an RL agent is not evidence of skill.'),
  'hedging': ('A desk sells a one-month at-the-money call on the S&P 500 and hedges it daily with 10 bp costs. By how much does the deep '
              'hedge cut the expected shortfall (ES) of the loss compared with the Black-Scholes delta, and is the gain the same in calm and '
              'in stressed markets?',
              'Over the 1,696 test paths (start days 2020–2026), the 95% ES of the loss is 6.82% of the index level without a hedge, 1.94% '
              'with the Black-Scholes delta and 1.15% with the deep hedge (mean of 5 seeds). The standard deviations are close (0.74% and '
              '0.69%): the network was trained to shrink the tail, not the variance. The gain is larger in stress: with VIX below 20 at the '
              'start (986 paths) the ES falls from 1.41% to 1.01%, with VIX of 20 or more (710 paths) from 2.60% to 1.31%. At 99% the ES '
              'is 4.09% against 2.43%.'),
 },
 'ro': {
  'dp-day': ('Pe 30 iunie 2026, prețul românesc pentru ziua următoare a variat între 100,65 EUR/MWh la ora 13:00 și 954,59 EUR/MWh la ora '
             '20:00, iar o baterie de 2 MWh a cîștigat 1.541 EUR cu previziune perfectă. Cîștigă o baterie de 4 MWh de două ori mai mult?',
             'Aproape, dar nu chiar: 2.885 EUR, cu 87% mai mult. Bateria mută 1 MWh pe oră, deci al treilea și al patrulea MWh sînt '
             'cumpărate în ore mai puțin ieftine și vîndute în ore mai puțin scumpe; cu 6 MWh venitul este de 3.282 EUR, doar cu 14% peste '
             'cel de la 4 MWh, pentru că vîrful de seară durează puține ore. O baterie fără pierderi (randament 100%) ar cîștiga 1.745 EUR '
             'la 2 MWh. În aceeași zi, Ungaria dă 1.466 EUR, Germania–Luxemburg 708 EUR și Bulgaria 744 EUR: venitul urmează diferența '
             'zilnică dintre prețuri, nu nivelul lor. Toate aceste cifre sînt limite superioare, deoarece programarea dinamică (DP) '
             'cunoaște cele 24 de prețuri ale zilei înainte ca ele să fie publicate.'),
  'dp-value': ('Mergeți înapoi oră cu oră cu ▶. Pe 30 iunie 2026, cît valorează un MWh stocat la ora 12:00, la ora 18:00 și la ora 23:00?',
               'La ora 12:00, V<sub>12</sub>(0) = 1.541 EUR și V<sub>12</sub>(2) = 1.760 EUR, deci fiecare MWh stocat valorează 109,38 EUR: '
               'prețul-umbră al energiei din baterie, aproximativ cît costă cumpărarea ei la amiază. La ora 18:00, fiecare MWh valorează '
               'circa 880 EUR, pentru că cele mai scumpe două ore (19:00 și 20:00) urmează. La ora 23:00 (alegeți valoarea unui MWh stocat în plus), primul MWh valorează 211 EUR (mai '
               'poate fi vîndut în ultima oră), iar al doilea nimic, deoarece se poate muta doar 1 MWh pe oră. Ecuația Bellman construiește '
               'aceste valori înapoi, pornind de la V<sub>24</sub>(s) = 0, iar acțiunea optimă în fiecare (h, s) este cea care atinge maximul.'),
  'q-policy': ('Agentul tabelar nu vede prețurile de azi, ci doar rangul fiecărei ore în prețurile de ieri. La ora 13:00, cu 1 MWh stocat, '
               'într-o oră care ieri a fost printre cele mai ieftine șase, ce acțiune alege și cît de sigur este?',
               'Încarcă: Q(încărcare) = 175,0 EUR, Q(inactiv) = 167,2 EUR, Q(descărcare) = 139,6 EUR, valorile restului zilei după fiecare '
               'acțiune, învățate din 1.746 de zile de antrenare (2019–2023). Avantajul față de așteptare este mic, circa 8 EUR. Cu 1 MWh '
               'stocat, rîndul orelor celor mai ieftine de ieri încarcă în 7 ore și descarcă în 8, mai ales de la ora 18:00 încolo: ora '
               'zilei domină, ca în curs. Cu 0 MWh agentul poate doar să încarce sau să aștepte, cu 2 MWh doar să descarce sau să aștepte '
               '(mascarea acțiunilor). Tabelul are 24 × 4 × 3 = 288 de stări și poate fi tipărit și auditat.'),
  'battery-test': ('În cele 638 de zile de test, 1 ianuarie 2025 – 24 octombrie 2026, cine obține o parte mai mare din venitul cu previziune '
                   'perfectă: rețeaua Q profundă (DQN) dublă, Q-learning tabelar sau simpla repetare a planului optim de ieri?',
                   'Planul de ieri: 163,1 mii EUR, adică 80,7% din limita superioară de 202,1 mii EUR. Q-learning tabelar obține 115,2 mii '
                   'EUR (57,0%), iar DQN dublu 101,3 mii EUR (50,1%, media a 5 valori inițiale; alegeți valorile inițiale pe rînd pentru a '
                   'vedea cît diferă). DQN depășește planul naiv doar în 9% din zile. Agenții care învață văd un rezumat grosier al zilei de '
                   'ieri (quartilele prețurilor și cîteva variabile), planul naiv vede întregul profil orar de ieri, iar prețurile sînt '
                   'puternic autocorelate de la o zi la alta: un reper simplu este greu de depășit.'),
  'trading': ('Cu un cost de 5 puncte de bază (pb) pe tranzacție, agentul de optimizare proximală a politicii (PPO) are un raport Sharpe mai '
              'mare decît strategia cumpără și păstrează în 2020–2026 (0,83 față de 0,61, media celor cinci valori inițiale). Rezistă '
              'avantajul la costuri mai mari sau la un început mai tîrziu?',
              'Doar parțial. La 20 pb, media rapoartelor Sharpe ale valorilor inițiale scade la 0,63, la nivelul strategiei cumpără și '
              'păstrează (0,61); la 30 pb este 0,50, iar două valori inițiale care tranzacționează des coboară la 0,02 și 0,07. Cu începere '
              'din 2023, cumpără și păstrează are raportul Sharpe 1,23, față de 1,04 pentru PPO la 5 pb: avantajul vine din 2020 și 2022, '
              'cînd agentul a stat parțial în afara pieței (graficul expunerii). Valorile inițiale diferă (de la 0,72 la 0,95 la 5 pb), iar '
              'intervalul bootstrap pe blocuri al diferenței de raport Sharpe din curs, de la −0,08 la 0,56, conține zero. Un singur test '
              'retrospectiv al unui agent RL nu dovedește abilitate.'),
  'hedging': ('Un dealer vinde o opțiune call la bani pe o lună pe S&P 500 și o acoperă zilnic, cu costuri de 10 pb. Cu cît reduce acoperirea '
              'neuronală pierderea așteptată în coadă (ES, expected shortfall) față de delta Black-Scholes și este cîștigul același pe piețe '
              'calme și pe piețe agitate?',
              'Pe cele 1.696 de traiectorii de test (zile de început 2020–2026), ES de 95% al pierderii este 6,82% din nivelul indicelui '
              'fără acoperire, 1,94% cu delta Black-Scholes și 1,15% cu acoperirea neuronală (media a 5 valori inițiale). Abaterile '
              'standard sînt apropiate (0,74% și 0,69%): rețeaua a fost antrenată să micșoreze coada, nu varianța. Cîștigul este mai mare '
              'în perioadele agitate: cu VIX sub 20 la început (986 de traiectorii), ES scade de la 1,41% la 1,01%, cu VIX de cel puțin 20 '
              '(710 traiectorii) de la 2,60% la 1,31%. La 99%, ES este 4,09% față de 2,43%.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 months: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
 zone: { RO: 'Romania', HU: 'Hungary', DE_LU: 'Germany–Luxembourg', BG: 'Bulgaria' },
 act: ['charge (buy)', 'idle', 'discharge (sell)'],
 price: 'Price (EUR/MWh)', stored: 'Stored energy (MWh)', storedS: 'Stored energy s (MWh)', hourCET: 'Delivery hour (CET)', hourH: 'Hour h (CET)',
 eurMWh: 'EUR per MWh', thEur: 'Thousand EUR',
 ddTitle: (z, d) => `${z}, ${d}: optimal schedule with perfect foresight`,
 dd: (z, d, rev, nc, nd) => `${z}, ${d}: revenue of the day V<sub>0</sub>(0) = ${rev} EUR, with ${window.NNI.cnt(nc, 'charging hour', 'charging hours')} and ${window.NNI.cnt(nd, 'discharging hour', 'discharging hours')}.`,
 dd2: (lo, hlo, hi, hhi, sp, E, eta, per) => `Prices from ${lo} EUR/MWh (${hlo}:00) to ${hi} EUR/MWh (${hhi}:00), daily spread ${sp} EUR/MWh; battery of ${E} MWh with ${eta}% round-trip efficiency: ${per} EUR per MWh of capacity.`,
 step: v => (v === 0 ? '0 (only V₂₄(s) = 0 is known)' : `${v} (back to ${24 - v}:00)`),
 dvTitle: 'Value function V_h(s), cash flow to midnight',
 dvShadow: 'One more stored MWh: V_h(s+1) − V_h(s)',
 dvName: s => `stored energy ${s} MWh`, dvShadowName: (a, b) => `from ${a} to ${b} MWh`,
 dvPol: 'Optimal action at (h, s), hours solved',
 dvDay: (z, d, E, eta) => `Day, zone and battery of the block above: ${z}, ${d}, ${E} MWh, ${eta}% round trip.`,
 dv0: 'Step 0: at midnight stored energy is worth nothing, V<sub>24</sub>(s) = 0; press ▶ to go back one hour at a time.',
 dv: (h, v0, E, vE, sh, a) => `Hour ${h}: V<sub>${h}</sub>(0) = ${v0} EUR, V<sub>${h}</sub>(${E}) = ${vE} EUR, so one stored MWh is worth ${sh} EUR on average; optimal action with an empty battery: ${a}.`,
 dvDone: rev => `Back at 0:00: V<sub>0</sub>(0) = ${rev} EUR is the revenue of the day.`,
 quart: ["yesterday's 6 cheapest hours", '2nd quartile', '3rd quartile', "yesterday's 6 dearest hours"],
 qpTitle: s => `Greedy action of the learned Q-table, stored energy ${s} MWh`,
 qpBar: (h, q, s) => `Q(state, action) at ${h}:00, ${q}, ${s} MWh stored`,
 qEur: 'Q value (EUR to midnight)', masked: 'not allowed',
 qp: (h, q, s, qc, qi, qd, g) => `${h}:00, ${q}, ${s} MWh stored: Q(charge) = ${qc}, Q(idle) = ${qi}, Q(discharge) = ${qd} EUR; greedy action: ${g}.`,
 qp2: (q, nc, nd) => `In the row “${q}” the table charges in ${window.NNI.cnt(nc, 'hour', 'hours')} and discharges in ${window.NNI.cnt(nd, 'hour', 'hours')}.`,
 btName: { pf: 'Perfect foresight (upper bound)', dqn: 'Double DQN', q: 'Tabular Q-learning', nv: "Yesterday's optimal plan" },
 seedTag: sd => (sd === 'avg' ? ' (mean of 5 seeds)' : ` (seed ${+sd + 1})`),
 bt: (n, k, cap, day) => `${n}: ${k} thousand EUR, ${cap}% of perfect foresight, ${day} EUR per day.`,
 btWin: (p, n, d0, d1) => `The double DQN earns more than yesterday's plan on ${p}% of the ${window.NNI.cnt(n, 'test day', 'test days')} from ${d0} to ${d1}.`,
 btCum: 'Cumulative revenue on the test days (thousand EUR)', btCap: 'Share of the perfect-foresight revenue captured to date (%)', pctPF: '% of perfect foresight',
 tqName: { bh: 'Buy and hold', ma: '200-day moving-average rule', ppo: 'PPO agent' },
 tq: (n, r, v, s, dd, nt) => `${n}: annual return ${r}%, volatility ${v}%, Sharpe ratio ${s}, maximum drawdown ${dd}%, ${window.NNI.cnt(+nt, 'position change', 'position changes')}.`,
 tqSeeds: (m, lo, hi) => `PPO, Sharpe ratio of the individual seeds: mean ${m}, from ${lo} to ${hi} (the curve averages the daily returns of the five seeds).`,
 tqTitle: c => `S&P 500, value of 1 USD invested, cost ${c} bp per trade`, tqEx: 'Exposure: % invested, 21-day mean', usd: 'Value of 1 USD', pctInv: '% invested',
 hgName: { nh: 'No hedge', bs: 'Black-Scholes delta', dh: 'Deep hedge' },
 hgTitle: a => `Hedging profit and loss (dotted: −ES ${a}%)`,
 pnl: 'Profit and loss (% of S₀; < −8 at −8)', days: 'Start days',
 hg: (n, mu, s, a, e) => `${n}: mean ${mu}%, standard deviation ${s}%, ES ${a}% of the loss ${e}%.`,
 esName: a => `ES ${a}% of the loss`, sdName: 'Standard deviation', hgEs: 'Risk of the hedged position', pctS0: '% of the index level',
 hgN: (n, r) => `${window.NNI.cnt(n, 'path', 'paths')} of 21 trading days, ${r}; cost 10 bp of the traded value; results in % of S₀, the index level on the start day.`,
 reg: { all: 'all test start days (2020–2026)', lo: 'start days with VIX below 20', hi: 'start days with VIX of 20 or more' }
};
""",
 'ro': r"""window.LK_TEXT = {
 months: ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
 zone: { RO: 'România', HU: 'Ungaria', DE_LU: 'Germania–Luxemburg', BG: 'Bulgaria' },
 act: ['încărcare (cumpărare)', 'inactiv', 'descărcare (vînzare)'],
 price: 'Preț (EUR/MWh)', stored: 'Energie stocată (MWh)', storedS: 'Energie stocată s (MWh)', hourCET: 'Ora de livrare (CET)', hourH: 'Ora h (CET)',
 eurMWh: 'EUR pe MWh', thEur: 'Mii EUR',
 ddTitle: (z, d) => `${z}, ${d}: programul optim cu previziune perfectă`,
 dd: (z, d, rev, nc, nd) => `${z}, ${d}: venitul zilei V<sub>0</sub>(0) = ${rev} EUR, cu ${window.NNI.cnt(nc, 'oră de încărcare', 'ore de încărcare')} și ${window.NNI.cnt(nd, 'oră de descărcare', 'ore de descărcare')}.`,
 dd2: (lo, hlo, hi, hhi, sp, E, eta, per) => `Prețuri de la ${lo} EUR/MWh (ora ${hlo}:00) la ${hi} EUR/MWh (ora ${hhi}:00), diferența zilnică ${sp} EUR/MWh; baterie de ${E} MWh cu randamentul ciclului complet ${eta}%: ${per} EUR pe MWh de capacitate.`,
 step: v => (v === 0 ? '0 (se cunoaște doar V₂₄(s) = 0)' : `${v} (înapoi pînă la ora ${24 - v}:00)`),
 dvTitle: 'Funcția valoare V_h(s), fluxul pînă la miezul nopții',
 dvShadow: 'Un MWh stocat în plus: V_h(s+1) − V_h(s)',
 dvName: s => `energie stocată ${s} MWh`, dvShadowName: (a, b) => `de la ${a} la ${b} MWh`,
 dvPol: 'Acțiunea optimă în (h, s), orele rezolvate',
 dvDay: (z, d, E, eta) => `Ziua, zona și bateria din blocul de mai sus: ${z}, ${d}, ${E} MWh, randament ${eta}%.`,
 dv0: 'Pasul 0: la miezul nopții energia stocată nu mai valorează nimic, V<sub>24</sub>(s) = 0; apăsați ▶ pentru a merge înapoi cîte o oră.',
 dv: (h, v0, E, vE, sh, a) => `Ora ${h}: V<sub>${h}</sub>(0) = ${v0} EUR, V<sub>${h}</sub>(${E}) = ${vE} EUR, deci un MWh stocat valorează în medie ${sh} EUR; acțiunea optimă cu bateria goală: ${a}.`,
 dvDone: rev => `Înapoi la ora 0:00: V<sub>0</sub>(0) = ${rev} EUR este venitul zilei.`,
 quart: ['cele mai ieftine 6 ore ieri', 'quartila a 2-a', 'quartila a 3-a', 'cele mai scumpe 6 ore ieri'],
 qpTitle: s => `Acțiunea lacomă a tabelului Q învățat, energie stocată ${s} MWh`,
 qpBar: (h, q, s) => `Q(stare, acțiune) la ora ${h}:00, ${q}, energie stocată ${s} MWh`,
 qEur: 'Valoarea Q (EUR pînă la miezul nopții)', masked: 'nepermisă',
 qp: (h, q, s, qc, qi, qd, g) => `Ora ${h}:00, ${q}, energie stocată ${s} MWh: Q(încărcare) = ${qc}, Q(inactiv) = ${qi}, Q(descărcare) = ${qd} EUR; acțiunea lacomă: ${g}.`,
 qp2: (q, nc, nd) => `Pe rîndul „${q}”, tabelul încarcă în ${window.NNI.cnt(nc, 'oră', 'ore')} și descarcă în ${window.NNI.cnt(nd, 'oră', 'ore')}.`,
 btName: { pf: 'Previziune perfectă (limita superioară)', dqn: 'DQN dublu', q: 'Q-learning tabelar', nv: 'Planul optim de ieri' },
 seedTag: sd => (sd === 'avg' ? ' (media a 5 valori inițiale)' : ` (valoarea inițială ${+sd + 1})`),
 bt: (n, k, cap, day) => `${n}: ${k} mii EUR, ${cap}% din previziunea perfectă, ${day} EUR pe zi.`,
 btWin: (p, n, d0, d1) => `DQN dublu cîștigă mai mult decît planul de ieri în ${p}% din cele ${window.NNI.cnt(n, 'zi de test', 'zile de test')} dintre ${d0} și ${d1}.`,
 btCum: 'Venitul cumulat în zilele de test (mii EUR)', btCap: 'Partea din venitul cu previziune perfectă obținută pînă la data respectivă (%)', pctPF: '% din previziunea perfectă',
 tqName: { bh: 'Cumpără și păstrează', ma: 'Regula mediei mobile de 200 de zile', ppo: 'Agentul PPO' },
 tq: (n, r, v, s, dd, nt) => `${n}: randament anual ${r}%, volatilitate ${v}%, raport Sharpe ${s}, scădere maximă ${dd}%, ${window.NNI.cnt(+nt, 'schimbare de poziție', 'schimbări de poziție')}.`,
 tqSeeds: (m, lo, hi) => `PPO, raportul Sharpe al fiecărei valori inițiale: media ${m}, de la ${lo} la ${hi} (curba mediază randamentele zilnice ale celor cinci valori inițiale).`,
 tqTitle: c => `S&P 500, valoarea a 1 USD investit, cost de ${c} pb pe tranzacție`, tqEx: 'Expunerea: % investit, medie pe 21 de zile', usd: 'Valoarea a 1 USD', pctInv: '% investit',
 hgName: { nh: 'Fără acoperire', bs: 'Delta Black-Scholes', dh: 'Acoperire neuronală' },
 hgTitle: a => `Rezultatul acoperirii (punctat: −ES ${a}%)`,
 pnl: 'Rezultat (% din S₀; < −8 apare la −8)', days: 'Zile de început',
 hg: (n, mu, s, a, e) => `${n}: media ${mu}%, abaterea standard ${s}%, ES ${a}% al pierderii ${e}%.`,
 esName: a => `ES ${a}% al pierderii`, sdName: 'Abaterea standard', hgEs: 'Riscul poziției acoperite', pctS0: '% din nivelul indicelui',
 hgN: (n, r) => `${window.NNI.cnt(n, 'traiectorie', 'traiectorii')} de 21 de zile de tranzacționare, ${r}; cost de 10 pb din valoarea tranzacționată; rezultatele în % din S₀, nivelul indicelui în ziua de început.`,
 reg: { all: 'toate zilele de început din test (2020–2026)', lo: 'zilele de început cu VIX sub 20', hi: 'zilele de început cu VIX de cel puțin 20' }
};
""",
}


# ---------------------------------------------------------------- data (real course data and the lecture's cached results)
E_MAX, ETA = 2, 0.85
ACTS = [-1, 0, 1]


def _reward(p, a):
    import numpy as np
    ec = ed = np.sqrt(ETA)
    return np.where(a > 0, ed * p, np.where(a < 0, -p / ec, 0.0))


def _dp_days(P):
    """Backward induction for every day (rows of P), as dp_days() in analysis_L10.py."""
    import numpy as np
    D = P.shape[0]
    V = np.zeros((D, 25, E_MAX + 1))
    A = np.zeros((D, 24, E_MAX + 1), dtype=int)
    for h in range(23, -1, -1):
        best = np.full((D, E_MAX + 1), -np.inf)
        arg = np.zeros((D, E_MAX + 1), dtype=int)
        for s in range(E_MAX + 1):
            for a in ACTS:
                s2 = s - a
                if s2 < 0 or s2 > E_MAX:
                    continue
                q = _reward(P[:, h], a) + V[:, h + 1, s2]
                better = q > best[:, s]
                best[better, s] = q[better]
                arg[better, s] = a
        V[:, h] = best
        A[:, h] = arg
    return V, A


def _run_schedule(P, acts):
    import numpy as np
    s = np.zeros(P.shape[0], dtype=int)
    tot = np.zeros(P.shape[0])
    for h in range(24):
        a = acts[:, h].copy()
        a[(a > 0) & (s == 0)] = 0
        a[(a < 0) & (s == E_MAX)] = 0
        tot += _reward(P[:, h], a)
        s = s - a
    return tot


def data():
    import numpy as np
    import pandas as pd
    import nn_plot as npl
    from scipy.stats import norm
    cache = os.path.join(HERE, 'cache')
    out = {}
    # cut-off of analysis_L10.py: last complete month of the S&P 500 / VIX quotes
    mk = npl.markets('close', ['SP500', 'VIX'])
    last = mk.dropna().index.max()
    END = (last + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)
    out['end'] = str(END.date())

    # ---- A. hourly day-ahead prices (CET delivery days with 24 hours), 2024 to the cut-off, four zones
    ph = npl.prices_hourly('Europe/Brussels').loc[:str(END.date()) + ' 23:59']
    ph['day'] = ph.index.tz_localize(None).normalize()
    ph['hour'] = ph.index.hour
    mats = {}
    for z in ZONES:
        x = ph[['day', 'hour', z]].dropna()
        c = x.groupby('day')[z].count()
        mats[z] = x[x.day.isin(c[c == 24].index)].pivot_table(index='day', columns='hour', values=z).sort_index()
    out['pr'] = {z: {str(d.date()): rnd(v, 2) for d, v in zip(mats[z].loc['2024-01-01':].index, mats[z].loc['2024-01-01':].values)}
                 for z in ZONES}

    # ---- battery benchmarks and learning agents on the test days (Romania), as in analysis_L10.py
    RO = mats['RO']
    P, days = RO.values, RO.index
    prev_ok = np.array([i >= 7 and (days[i] - days[i - 1]).days == 1 and (days[i] - days[i - 7]).days == 7 for i in range(len(days))])
    iTE = np.where(prev_ok & (days.year >= 2025))[0]
    V, Aall = _dp_days(P)
    plan = np.zeros((len(days), 24), dtype=int)
    s0 = np.zeros(len(days), dtype=int)
    for h in range(24):
        a = Aall[np.arange(len(days)), h, s0]
        plan[:, h] = a
        s0 = s0 - a
    naive = np.full(len(days), np.nan)
    naive[1:] = _run_schedule(P[1:], plan[:-1])
    BQ = json.load(open(os.path.join(cache, 'battery_v3.json')))
    BD = json.load(open(os.path.join(cache, 'battery_dqn_v4.json')))
    out['bt'] = {'d': [str(d.date()) for d in days[iTE]], 'pf': rnd(V[iTE, 0, 0], 1), 'nv': rnd(naive[iTE], 1),
                 'q': [rnd(x['test'], 1) for x in BQ['q']], 'dqn': [rnd(x['test'], 1) for x in BD['dqn']]}
    # tabular Q-table of the first seed: Q[h, quartile, stored energy, action], in EUR (rewards were scaled by 1/100 in training)
    out['q'] = rnd(np.load(os.path.join(cache, 'q_table.npy')) * 100, 1)

    # ---- B. PPO trading on the S&P 500: next-day log returns of the test period, positions of the five seeds, 200-day rule
    px = mk['SP500'].dropna().loc[:END]
    lr_ = np.log(px).diff()
    vix = mk['VIX'].reindex(px.index).ffill()
    F = pd.DataFrame({'r1': lr_, 'r5': np.log(px).diff(5), 'r21': np.log(px).diff(21), 'r63': np.log(px).diff(63),
                      'rv21': lr_.rolling(21).std() * np.sqrt(252), 'vix': vix / 100, 'ma200': px / px.rolling(200).mean() - 1}).dropna()
    fwd = lr_.shift(-1).reindex(F.index)
    F, fwd = F.iloc[:-1], fwd.iloc[:-1]
    te = F.index.year >= 2020
    TR = json.load(open(os.path.join(cache, 'trading_v2.json')))
    out['tr'] = {'d': [str(d.date()) for d in F.index[te]], 'r': rnd(fwd.values[te], 6),
                 'ma': [int(v) for v in (F['ma200'].values[te] > 0)], 'pos': [[int(v) for v in x['pos']] for x in TR['data']]}

    # ---- C. deep hedging: P&L of the 21-day test paths (% of the initial index level), 10 bp cost, VIX at the start
    S = mk['SP500'].dropna().loc[:END]
    VX = mk['VIX'].reindex(S.index).ffill() / 100
    NS = 21
    logS = np.log(S.values)
    starts = np.arange(0, len(S) - NS)
    paths = np.stack([np.exp(logS[i:i + NS + 1] - logS[i]) for i in starts]).astype(np.float32)
    vols = np.stack([VX.values[i:i + NS + 1] for i in starts]).astype(np.float32)
    sd0 = S.index[starts]
    ih = np.where(sd0.year >= 2020)[0]
    TAU = (NS - np.arange(NS + 1)) / 252.0

    def bs_call(Sx, sig, tau, K=1.0):
        tau = np.maximum(tau, 1e-8)
        d1 = (np.log(Sx / K) + 0.5 * sig ** 2 * tau) / (sig * np.sqrt(tau))
        return Sx * norm.cdf(d1) - K * norm.cdf(d1 - sig * np.sqrt(tau)), norm.cdf(d1)
    prem = bs_call(1.0, vols[:, 0], TAU[0])[0]
    payoff = np.maximum(paths[:, -1] - 1.0, 0)
    bsd = np.stack([bs_call(paths[:, k], vols[:, k], TAU[k])[1] for k in range(NS)], 1)

    def pnl_of(delta, c, idx):
        Pp = paths[idx]
        dS = np.diff(Pp, axis=1)
        z = np.zeros((len(delta), 1))
        trade = np.abs(np.diff(np.concatenate([z, delta, z], 1), axis=1))
        return prem[idx] + (delta * dS).sum(1) - payoff[idx] - c * (trade * Pp).sum(1)
    HR = json.load(open(os.path.join(cache, 'hedge_v1.json')))
    out['hd'] = {'nh': rnd(pnl_of(np.zeros_like(bsd[ih]), 0.001, ih) * 100, 3), 'bs': rnd(pnl_of(bsd[ih], 0.001, ih) * 100, 3),
                 'dh': [rnd(np.array(x['pnl']) * 100, 3) for x in HR['main']], 'vix': rnd(100 * vols[ih, 0], 1)}
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
