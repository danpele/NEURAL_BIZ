"""Build the interactive page of Lecture 11 (Explainable AI): interactive/L11.html (EN) and interactive/L11_ro.html (RO).

Inputs (interactive/L11_data.js) are recomputed by data() from the cached models of analysis_L11.py (lectures/L11/cache): the
data preparation repeats analysis_L11.py line by line, so that applicants, frauds and models are those of the slides, and
nothing is retrained. The page logic is in interactive/L11.js.
Run: OMP_NUM_THREADS=1 python3 lectures/L11/build_interactive_L11.py [--no-data]
"""
import os
import pickle
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch  # noqa: E402  (PyTorch before scikit-learn: OpenMP order on macOS)
import torch.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, play, charts  # noqa: E402

K = 11
CACHE = os.path.join(HERE, 'cache')
SEED = 20261108

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('waterfall', 'shap_waterfall', 'shap', 'wf'),
          ('lime', 'lime_instability', 'lime', 'lm'),
          ('pdp-ice', 'pdp_ice', 'lime', 'pd'),
          ('ig-steps', 'ig_completeness', 'grad', 'ig'),
          ('deletion', 'deletion_curve', 'stab', 'dl'),
          ('seeds', 'seed_stability', 'stab', 'ss')]

SECTIONS = {
 'en': dict(shap='Shapley values and SHAP', lime='LIME, counterfactuals and global effects', grad='Gradient methods for neural networks',
            stab='Do explanations agree, are they faithful, are they stable?'),
 'ro': dict(shap='Valorile Shapley și SHAP', lime='LIME, explicații contrafactuale și efecte globale',
            grad='Metode bazate pe gradient pentru rețele neuronale', stab='Concordă explicațiile, sînt fidele, sînt stabile?'),
}

L = {
 'en': dict(h1='Lecture 11 · Explainable AI: SHAP, LIME, gradient methods and the stability of explanations',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the models of the '
        'lecture when you move a control. The ▶ button moves the main parameter step by step. Under each figure: a readout, a question '
        'for the room and the Quantlet with the code of the chart. The explanations come from SHAP (SHapley Additive exPlanations), '
        'LIME (local interpretable model-agnostic explanations), partial dependence and ICE (individual conditional expectation) '
        'curves and IG (integrated gradients), computed for a multilayer perceptron (MLP), an XGBoost (extreme gradient boosting) '
        'ensemble and a logistic regression.',
   src='Credit: HMEQ (home equity loans) data of Baesens, Roesch and Scheule (2016), 5,960 loans, 400 test applicants explained, 100 '
       'training applicants as background. Fraud: ULB (Université Libre de Bruxelles) Machine Learning Group and Worldline, credit-card '
       'transactions (OpenML 1597); network trained on day 1, 211 frauds of day 2 explained. House values: StatLib, California '
       'housing, 1990 census (Pace and Barry, 1997). Models, SHAP, LIME and attributions: analysis_L11.py, seed 20261108.'),
 'ro': dict(h1='Cursul 11 · Inteligența artificială explicabilă: SHAP, LIME, metode bazate pe gradient și stabilitatea explicațiilor',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser din modelele cursului cînd modificați un '
        'control. Butonul ▶ modifică parametrul principal pas cu pas. Sub fiecare figură: o sinteză numerică, o întrebare pentru sală și '
        'Quantlet-ul cu codul figurii. Explicațiile provin din SHAP (explicații aditive Shapley), LIME (explicații locale interpretabile, '
        'independente de model), din curbele de dependență parțială și ICE (așteptarea condiționată individuală) și din IG (gradienți '
        'integrați), calculate pentru un perceptron multistrat (MLP), un ansamblu XGBoost (extreme gradient boosting, arbori '
        'amplificați prin gradient) și o regresie logistică.',
   src='Credit: datele HMEQ (credite garantate cu locuința) ale lui Baesens, Roesch și Scheule (2016), 5.960 de credite, 400 de '
       'solicitanți din test explicați, 100 de solicitanți din antrenare ca referință. Fraudă: ULB (Université Libre de Bruxelles) '
       'Machine Learning Group și Worldline, tranzacții cu carduri (OpenML 1597); rețea antrenată pe ziua 1, 211 fraude din ziua 2 '
       'explicate. Valoarea locuințelor: StatLib, California housing, recensămîntul din 1990 (Pace și Barry, 1997). Modele, SHAP, LIME '
       'și atribuiri: analysis_L11.py, valoarea inițială 20261108.'),
}


def widgets(lang):
    ro = lang == 'ro'
    T = (lambda en, r: r) if ro else (lambda en, r: en)
    models = [('mlp', T('MLP (Kernel SHAP)', 'MLP (Kernel SHAP)')), ('xgb', T('XGBoost (TreeSHAP)', 'XGBoost (TreeSHAP)')),
              ('lr', T('Logistic regression (linear SHAP)', 'Regresie logistică (SHAP liniar)'))]
    feats = [('MedInc', T('Median income of the block group', 'Venitul median al grupului de locuințe')),
             ('HouseAge', T('Median age of the houses', 'Vîrsta mediană a locuințelor')),
             ('AveRooms', T('Average rooms per household', 'Numărul mediu de camere pe gospodărie')),
             ('AveOccup', T('Average occupants per household', 'Numărul mediu de ocupanți pe gospodărie')),
             ('Latitude', T('Latitude (north)', 'Latitudinea (nord)'))]
    meth = [('ig', T('Integrated gradients', 'Gradienți integrați')), ('shap', 'Kernel SHAP'), ('lime', 'LIME'),
            ('saliency', T('Saliency', 'Saliență')), ('random', T('Random order', 'Ordine aleatoare'))]
    frauds = [(i, T(f'Fraud {i + 1}', f'Frauda {i + 1}') + (T(' (median score, slides)', ' (scorul median, slide-uri)') if i == 0 else ''))
              for i in range(8)]
    return {
     'waterfall': (slider('wf-r', T('Applicant, ranked by the probability of default (PD) of the network (1 = lowest)',
                                    'Solicitantul, ordonat după probabilitatea de nerambursare (PD) a rețelei (1 = cea mai mică)'), 1, 400, 1, 346)
                   + select('wf-m', T('Model and SHAP method', 'Modelul și metoda SHAP'), models, 'mlp')
                   + slider('wf-k', T('Features shown separately', 'Variabile afișate separat'), 3, 12, 1, 8) + play('wf'),
                   charts('wf-chart', cls='one')),
     'lime': (radios('lm-n', T('Samples drawn by LIME around the applicant', 'Eșantioane generate de LIME în jurul solicitantului'),
                     [('500', '500'), ('5000', T('5,000', '5.000'))], '500')
              + slider('lm-r', T('Random run (seed of LIME)', 'Rularea aleatoare (valoarea inițială a LIME)'), 1, 30, 1, 1)
              + slider('lm-w', T('Kernel width σ (right panel, 5,000 samples)', 'Lățimea nucleului σ (panoul din dreapta, 5.000 de eșantioane)'), 0, 6, 1, 4)
              + play('lm'), charts('lm-runs', 'lm-kw')),
     'pdp-ice': (select('pd-f', T('Feature moved', 'Variabila modificată'), feats, 'MedInc')
                 + slider('pd-n', T('ICE curves shown (block groups)', 'Curbe ICE afișate (grupuri de locuințe)'), 0, 60, 5, 30)
                 + toggles('pd-c', T('Display', 'Afișare'), [('c', T('Curves centred at the left end (c-ICE)', 'Curbe centrate la capătul din stînga (c-ICE)'))], [])
                 + play('pd'), charts('pd-chart', cls='one')),
     'ig-steps': (select('ig-c', T('Fraud of day 2', 'Frauda din ziua 2'), frauds, 0)
                  + slider('ig-m', T('Steps m of the integral (Riemann sum)', 'Pașii m ai integralei (suma Riemann)'), 0, 8, 1, 2) + play('ig'),
                  charts('ig-bar', 'ig-err')),
     'deletion': (toggles('dl-m', T('Order of deletion', 'Ordinea eliminării'), meth, ['ig', 'shap', 'saliency', 'random'])
                  + slider('dl-f', T('Fraud (0 = mean of the 211 frauds; 1 = highest score)', 'Frauda (0 = media celor 211 fraude; 1 = scorul cel mai mare)'), 0, 211, 1, 0)
                  + slider('dl-k', T('Features reset to the baseline, k', 'Variabile readuse la referință, k'), 0, 12, 1, 3) + play('dl'),
                  charts('dl-chart', cls='one')),
     'seeds': (select('ss-m', T('Model', 'Modelul'), [('mlp', 'MLP (Kernel SHAP)'), ('xgb', 'XGBoost (TreeSHAP)')], 'mlp')
               + slider('ss-s', T('Training seeds compared', 'Valori inițiale de antrenare comparate'), 2, 10, 1, 10)
               + slider('ss-a', T('Applicant (right panel)', 'Solicitantul (panoul din dreapta)'), 1, 200, 1, 1) + play('ss'),
               charts('ss-rank', 'ss-loc')),
    }


QA = {
 'en': {
  'waterfall': ('Applicant row 2411 has a predicted PD of 65.6% and did default. Which feature pushes the network most towards '
                'rejection, and does the logistic regression blame the same feature?',
                'For the MLP the missing debt-to-income ratio (DEBTINC missing, +3.13 log-odds) dominates, followed by JOB = Other (+1.74); '
                'DELINQ = 0 (no delinquent credit lines) lowers the risk (−0.71). The bars add up exactly: −3.613 (the average applicant, '
                'PD 2.6%) + 4.261 = 0.648 log-odds, i.e. a PD of 65.6%. Switch the model: the three models agree that a missing DEBTINC '
                'is the strongest signal (in the data, 62.0% of the applicants without DEBTINC default against 8.6% of the others), but '
                'they split the rest of the score differently. Move the applicant slider to the safest applicants: there, most bars are '
                'blue.'),
  'lime': ('LIME fits a small linear model on random samples around the applicant. Run it 30 times with 500 samples: is the top '
           'feature always the same?',
           'No: with 500 samples the top feature is the same as in the reference run in 73% of the 30 runs (Jaccard overlap of the '
           'top-5 sets 0.72); with 5,000 samples in 100% (overlap 1.00), and the coefficient of variation of the top-5 slopes falls '
           'from 10% to 2%. The kernel width σ matters as well: the slope of DELINQ, the top LIME feature, goes from −2.04 to almost '
           '0 as σ moves over the seven widths tried (default 0.75·√18 = 3.18). Report the seed, the sample size and the kernel '
           'width with every LIME explanation.'),
  'pdp-ice': ('The partial dependence curve of income rises from 1.40 to 3.60 (100,000 USD). Does every block group follow it?',
              'No. The PDP (partial dependence plot) is an average; the ICE curves show one block group each, with all its other '
              'features fixed. At the top of the income grid the curves spread by 0.95 (standard deviation, 100,000 USD): the network '
              'has learned interactions, e.g. with location. Centre the curves (c-ICE) to compare only their shapes. With latitude the '
              'curves fall from south to north for most block groups, but not by the same amount.'),
  'ig-steps': ('Integrated gradients promise that the attributions add up to f(x) − f(x′). Why does the sum miss with 2 steps and not '
               'with 64?',
               'IG is an integral of the gradient along the straight path from the baseline x′ (the mean legitimate transaction of '
               'day 1) to the fraud x, computed with a midpoint Riemann sum of m steps. With few steps the sum is a poor approximation: '
               'over the 211 frauds of day 2 the median relative error is 7.5% with 2 steps, 3.7% with 4 and 0.18% with 64; it falls '
               'roughly as 1/m², as for the midpoint rule. The ranking of the features is stable much earlier than the sum.'),
  'deletion': ('If an explanation is faithful, resetting the features it calls important should make the fraud score fall fast. '
               'Which method passes this test, and which one does not?',
               'Integrated gradients, Kernel SHAP and LIME take the mean fraud score from 85.4% to about 61% after three features '
               '(IG 61.3%, SHAP 61.1%, LIME 61.2%); saliency (the absolute gradient) only to 71.9%, and a random order to 83.9%. The '
               'AOPC (area over the perturbation curve, mean drop over k = 1–12) is 56.2, 58.4 and 56.1 points for IG, SHAP and LIME, '
               '48.0 for saliency and 3.9 for the random order. Move the fraud slider: for single frauds the curves can cross.'),
  'seeds': ('Ten networks differ only in the random seed of training and have almost the same accuracy (AUC between 0.929 and 0.941). '
            'Do they give the same explanation?',
            'Not quite: the ranking of the global importances agrees with a mean pairwise Kendall τ of 0.83 for the networks and '
            '0.97 for XGBoost, and the top feature of a single applicant is the same in 78% (MLP) and 83% (XGBoost) of the seed pairs. '
            'This is the Rashomon effect: several equally good models explain the data in different ways. An explanation sent to a '
            'customer should be checked across seeds, or produced by a model whose explanation is stable.'),
 },
 'ro': {
  'waterfall': ('Solicitantul din rîndul 2411 are o PD prognozată de 65,6% și nu a rambursat creditul. Ce variabilă împinge cel mai '
                'mult rețeaua spre respingere și dă vina regresia logistică pe aceeași variabilă?',
                'Pentru MLP domină lipsa raportului datorie/venit (DEBTINC lipsă, +3,13 log-șanse), urmată de JOB = Other (+1,74); '
                'DELINQ = 0 (nicio linie de credit cu restanțe) reduce riscul (−0,71). Barele se adună exact: −3,613 (solicitantul '
                'mediu, PD 2,6%) + 4,261 = 0,648 log-șanse, adică o PD de 65,6%. Schimbați modelul: cele trei modele sînt de acord că '
                'lipsa DEBTINC este cel mai puternic semnal (în date, 62,0% dintre solicitanții fără DEBTINC nu rambursează, față de '
                '8,6% dintre ceilalți), dar împart diferit restul scorului. Mutați cursorul spre solicitanții cei mai siguri: acolo, '
                'majoritatea barelor sînt albastre.'),
  'lime': ('LIME ajustează un model liniar mic pe eșantioane aleatoare din jurul solicitantului. Rulat de 30 de ori cu 500 de '
           'eșantioane, dă mereu aceeași variabilă pe primul loc?',
           'Nu: cu 500 de eșantioane, variabila de pe primul loc coincide cu cea din rularea de referință în 73% dintre cele 30 de '
           'rulări (suprapunerea Jaccard a primelor cinci variabile 0,72); cu 5.000 de eșantioane în 100% (suprapunere 1,00), iar '
           'coeficientul de variație al pantelor primelor cinci variabile scade de la 10% la 2%. Contează și lățimea nucleului σ: '
           'panta DELINQ, variabila de top pentru LIME, trece de la −2,04 la aproape 0 pe cele șapte lățimi încercate (implicit '
           '0,75·√18 = 3,18). Raportați valoarea inițială, mărimea eșantionului și lățimea nucleului pentru fiecare explicație LIME.'),
  'pdp-ice': ('Curba de dependență parțială a venitului crește de la 1,40 la 3,60 (100.000 USD). O urmează fiecare grup de locuințe?',
              'Nu. PDP (graficul dependenței parțiale) este o medie; curbele ICE arată cîte un grup de locuințe, cu toate celelalte '
              'variabile fixate. La capătul de sus al grilei de venit, curbele se împrăștie cu 0,95 (abaterea standard, 100.000 USD): '
              'rețeaua a învățat interacțiuni, de exemplu cu localizarea. Centrați curbele (c-ICE) pentru a compara numai formele lor. '
              'Pentru latitudine, curbele scad de la sud spre nord la majoritatea grupurilor, dar nu cu aceeași valoare.'),
  'ig-steps': ('Gradienții integrați promit că atribuirile se adună la f(x) − f(x′). De ce suma nu se potrivește cu 2 pași și se '
               'potrivește cu 64?',
               'IG este o integrală a gradientului pe drumul drept de la referința x′ (tranzacția legitimă medie din ziua 1) la frauda '
               'x, calculată cu o sumă Riemann cu punctul de mijloc, cu m pași. Cu puțini pași, suma aproximează slab integrala: pe '
               'cele 211 fraude din ziua 2, eroarea relativă mediană este 7,5% cu 2 pași, 3,7% cu 4 și 0,18% cu 64; scade aproximativ '
               'ca 1/m², ca pentru regula punctului de mijloc. Ordinea variabilelor se stabilizează mult înaintea sumei.'),
  'deletion': ('Dacă o explicație este fidelă, readucerea la referință a variabilelor pe care le declară importante ar trebui să '
               'scadă repede scorul de fraudă. Ce metodă trece acest test și care nu?',
               'Gradienții integrați, Kernel SHAP și LIME duc scorul mediu de fraudă de la 85,4% la aproximativ 61% după trei '
               'variabile (IG 61,3%, SHAP 61,1%, LIME 61,2%); saliența (valoarea absolută a gradientului) doar la 71,9%, iar o ordine '
               'aleatoare la 83,9%. AOPC (aria de deasupra curbei de perturbare, scăderea medie pentru k = 1–12) este 56,2, 58,4 și '
               '56,1 puncte pentru IG, SHAP și LIME, 48,0 pentru saliență și 3,9 pentru ordinea aleatoare. Mutați cursorul fraudei: '
               'pentru fraude individuale, curbele se pot intersecta.'),
  'seeds': ('Zece rețele diferă numai prin valoarea inițială a antrenării și au aproape aceeași acuratețe (AUC, aria de sub curba ROC, '
            'între 0,929 și 0,941). Dau aceeași explicație?',
            'Nu întru totul: ordinea importanțelor globale concordă cu un tau Kendall mediu pe perechi de 0,83 pentru rețele și 0,97 '
            'pentru XGBoost, iar variabila de top a unui solicitant este aceeași în 78% (MLP) și 83% (XGBoost) dintre perechile de '
            'valori inițiale. Acesta este efectul Rashomon: mai multe modele la fel de bune explică datele în moduri diferite. O '
            'explicație trimisă unui client trebuie verificată pe mai multe valori inițiale sau produsă de un model cu explicații '
            'stabile.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 feat: { DEBTINC_missing: 'DEBTINC missing', REASON_HomeImp: 'REASON = HomeImp' },
 others: n => `${n} other features`, logodds: 'Log-odds of default', base: 'E[f(X)], average applicant', fx: 'f(x), this applicant',
 up: 'raises the risk', down: 'lowers the risk', model: { mlp: 'MLP', xgb: 'XGBoost', lr: 'logistic regression' },
 wfTitle: (r, m) => `Applicant row ${r}: SHAP contributions, ${m}`,
 wf: (r, pd, b, bpd, s, bad) => `Row ${r}: PD ${pd} = σ(${b} ${s}); the average applicant has PD ${bpd}. Outcome: ${bad ? 'defaulted' : 'repaid'}.`,
 wf2: (a, c, d) => `PD of the same applicant: MLP ${a}, XGBoost ${c}, logistic regression ${d}.`,
 wf3: s => `Largest pushes: ${s}.`,
 n5000: '5,000', slope: 'LIME slope (log-odds)', runTitle: (n, r) => `LIME run ${r} with ${n} samples (dots: the 30 runs)`,
 kwTitle: 'LIME slopes against the kernel width', kw: 'Kernel width σ (log scale)',
 thisRun: 'this run', allRuns: '30 runs', ref: 'reference run (5,000 samples, seed 0)',
 lm: (n, r, top, same, sh) => `Run ${r} with ${n} samples: top feature ${top}; ${same} of the 30 runs give the reference top feature (${sh}).`,
 lm2: (j, cv) => `Jaccard overlap of the top-5 features with the reference run: ${j}; mean over the 30 runs ${cv}.`,
 lm3: (w, d, f, s) => `Kernel width σ = ${w} (default ${d}): slope of ${f} ${s}.`,
 pdUnit: 'Predicted median house value (100,000 USD)', pdC: 'Change from the left end (100,000 USD)', pdp: 'Partial dependence (PDP)', ice: 'ICE, one block group',
 own: 'actual value of the block group',
 fx_: { MedInc: 'Median income (10,000 USD)', HouseAge: 'Median house age (years)', AveRooms: 'Average rooms per household',
        AveOccup: 'Average occupants per household', Latitude: 'Latitude (degrees north)' },
 pd: (a, b, lo, hi, s) => `PDP from ${a} to ${b} (100,000 USD) as the feature moves from ${lo} to ${hi}; spread of the ICE curves at the right end ${s}.`,
 pd2: (n, up) => `${n} curves shown; ${up} of the 60 block groups end above their starting point.`,
 igTitle: m => `Integrated gradients with m = ${m} steps (outline: 256 steps)`, attr: 'Attribution (log-odds)', steps: 'Steps m (log scale)',
 relErr: 'Relative completeness error (log scale)', med211: 'median over the 211 frauds', thisFraud: 'this fraud',
 ig: (m, s, g, e) => `m = ${m}: sum of the attributions ${s}, f(x) − f(x′) = ${g}; relative error ${e}.`,
 ig2: (p, e) => `Fraud score of this transaction ${p}; median error over the 211 frauds at this m: ${e}.`,
 meth: { ig: 'integrated gradients', shap: 'Kernel SHAP', lime: 'LIME', saliency: 'saliency', random: 'random order' },
 kAx: 'Features reset to the baseline, k', score: 'Fraud score (%)',
 dlTitle: f => (f ? `Fraud ${f} of 211 (by score)` : 'Mean over the 211 frauds of day 2'),
 dl: (k, s) => `After k = ${k}: ${s}.`, dl2: s => `AOPC (mean drop over k = 1–12, points of fraud score): ${s}.`,
 rank: 'Rank by mean |SHAP| (1 = most important)', seed: 'seed', ssLoc: r => `Applicant row ${r}: SHAP values, one dot per seed`, shapv: 'SHAP value (log-odds)',
 ssTitle: n => `Rank of the eight leading features, ${n} seeds`,
 ss: (n, t, o) => `${n} seeds: mean pairwise Kendall τ of the importances ${t}; top-3 overlap ${o}.`,
 ss2: a => `Same top feature for an applicant in ${a} of the seed pairs (200 applicants).`,
 ss3: (r, s) => `Applicant row ${r}: top feature by seed — ${s}.`
};
""",
 'ro': r"""window.LK_TEXT = {
 feat: { DEBTINC_missing: 'DEBTINC lipsă', REASON_HomeImp: 'REASON = HomeImp' },
 others: n => `celelalte ${n} variabile`, logodds: 'Log-șansele de nerambursare', base: 'E[f(X)], solicitantul mediu', fx: 'f(x), acest solicitant',
 up: 'crește riscul', down: 'reduce riscul', model: { mlp: 'MLP', xgb: 'XGBoost', lr: 'regresia logistică' },
 wfTitle: (r, m) => `Solicitantul din rîndul ${r}: contribuțiile SHAP, ${m}`,
 wf: (r, pd, b, bpd, s, bad) => `Rîndul ${r}: PD ${pd} = σ(${b} ${s}); solicitantul mediu are PD ${bpd}. Rezultatul: ${bad ? 'nu a rambursat' : 'a rambursat'}.`,
 wf2: (a, c, d) => `PD a aceluiași solicitant: MLP ${a}, XGBoost ${c}, regresia logistică ${d}.`,
 wf3: s => `Cele mai mari contribuții: ${s}.`,
 n5000: '5.000', slope: 'Panta LIME (log-șanse)', runTitle: (n, r) => `Rularea LIME ${r} cu ${n} de eșantioane (puncte: cele 30 de rulări)`,
 kwTitle: 'Pantele LIME și lățimea nucleului', kw: 'Lățimea nucleului σ (scară logaritmică)',
 thisRun: 'această rulare', allRuns: '30 de rulări', ref: 'rularea de referință (5.000 de eșantioane, valoarea inițială 0)',
 lm: (n, r, top, same, sh) => `Rularea ${r} cu ${n} de eșantioane: variabila de top ${top}; ${same} dintre cele 30 de rulări dau variabila de top de referință (${sh}).`,
 lm2: (j, cv) => `Suprapunerea Jaccard a primelor cinci variabile cu rularea de referință: ${j}; media pe cele 30 de rulări ${cv}.`,
 lm3: (w, d, f, s) => `Lățimea nucleului σ = ${w} (implicit ${d}): panta variabilei ${f} ${s}.`,
 pdUnit: 'Valoarea mediană prognozată a locuinței (100.000 USD)', pdC: 'Variația față de capătul din stînga (100.000 USD)', pdp: 'Dependența parțială (PDP)', ice: 'ICE, un grup de locuințe',
 own: 'valoarea reală a grupului',
 fx_: { MedInc: 'Venitul median (10.000 USD)', HouseAge: 'Vîrsta mediană a locuințelor (ani)', AveRooms: 'Numărul mediu de camere pe gospodărie',
        AveOccup: 'Numărul mediu de ocupanți pe gospodărie', Latitude: 'Latitudinea (grade nord)' },
 pd: (a, b, lo, hi, s) => `PDP de la ${a} la ${b} (100.000 USD) cînd variabila trece de la ${lo} la ${hi}; împrăștierea curbelor ICE la capătul din dreapta ${s}.`,
 pd2: (n, up) => `${n} curbe afișate; ${up} dintre cele 60 de grupuri se termină deasupra punctului de plecare.`,
 igTitle: m => `Gradienți integrați cu m = ${m} pași (contur: 256 de pași)`, attr: 'Atribuirea (log-șanse)', steps: 'Pașii m (scară logaritmică)',
 relErr: 'Eroarea relativă de completitudine (scară logaritmică)', med211: 'mediana pe cele 211 fraude', thisFraud: 'această fraudă',
 ig: (m, s, g, e) => `m = ${m}: suma atribuirilor ${s}, f(x) − f(x′) = ${g}; eroarea relativă ${e}.`,
 ig2: (p, e) => `Scorul de fraudă al acestei tranzacții ${p}; eroarea mediană pe cele 211 fraude la acest m: ${e}.`,
 meth: { ig: 'gradienți integrați', shap: 'Kernel SHAP', lime: 'LIME', saliency: 'saliență', random: 'ordine aleatoare' },
 kAx: 'Variabile readuse la referință, k', score: 'Scorul de fraudă (%)',
 dlTitle: f => (f ? `Frauda ${f} din 211 (după scor)` : 'Media celor 211 fraude din ziua 2'),
 dl: (k, s) => `După k = ${k}: ${s}.`, dl2: s => `AOPC (scăderea medie pentru k = 1–12, puncte de scor de fraudă): ${s}.`,
 rank: 'Rangul după media |SHAP| (1 = cea mai importantă)', seed: 'valoarea inițială', ssLoc: r => `Solicitantul din rîndul ${r}: valorile SHAP, un punct pentru fiecare valoare inițială`, shapv: 'Valoarea SHAP (log-șanse)',
 ssTitle: n => `Rangul celor opt variabile principale, ${n} valori inițiale`,
 ss: (n, t, o) => `${n} valori inițiale: tau Kendall mediu pe perechi al importanțelor ${t}; suprapunerea primelor trei variabile ${o}.`,
 ss2: a => `Aceeași variabilă de top pentru un solicitant în ${a} dintre perechile de valori inițiale (200 de solicitanți).`,
 ss3: (r, s) => `Solicitantul din rîndul ${r}: variabila de top pe valori inițiale — ${s}.`
};
""",
}


# ---------------------------------------------------------------- data (real course data, cached models of analysis_L11.py)
class MLP(nn.Module):
    def __init__(self, d, h1=64, h2=32, p=0.1, out=1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(p), nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p),
                               nn.Linear(h2, out))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def _net(state, d):
    n = MLP(d)
    n.load_state_dict(state)
    n.eval()
    return n


def _fn(net):
    def f(X):
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy().astype(np.float64)
    return f


def sig(z):
    return 1 / (1 + np.exp(-z))


def load(name):
    return pickle.load(open(os.path.join(CACHE, name + '.pkl'), 'rb'))


def R(a, d):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def data():
    from sklearn.model_selection import train_test_split
    import nn_plot as npl
    out = {}

    # ------------------------------------------------------------ A. credit (HMEQ), as in analysis_L11.py
    h = npl.hmeq()
    NUM = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
    JOBS = ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]
    y_all = h["BAD"].values
    idx = np.arange(len(h))
    i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
    i_tr, i_va = train_test_split(i_tr, test_size=0.25, stratify=y_all[i_tr], random_state=SEED)
    med = h.iloc[i_tr][NUM].median()
    XR = pd.DataFrame(index=h.index)
    for c in NUM:
        XR[c] = h[c].fillna(med[c])
    XR["DEBTINC_missing"] = h["DEBTINC"].isna().astype(float)
    XR["REASON_HomeImp"] = (h["REASON"] == "HomeImp").astype(float)
    for j in JOBS:
        XR["JOB_" + j] = (h["JOB"] == j).astype(float)
    FEATS = list(XR.columns)
    mu, sdv = XR.iloc[i_tr].mean(), XR.iloc[i_tr].std()
    XS = ((XR - mu) / sdv).values.astype(np.float32)
    Xtr, Xte = XS[i_tr], XS[i_te]
    yte = y_all[i_te]
    rng = np.random.default_rng(SEED)
    BG = Xtr[rng.choice(len(Xtr), 100, replace=False)]
    i_sub = rng.choice(len(Xte), 400, replace=False)
    Xsub = Xte[i_sub]

    CM, SH, LM, SS = load('credit_models_v1'), load('shap_main_v1'), load('lime_v1'), load('seed_shap_v1')
    f0 = _fn(_net(CM['mlp'][0]['state'], len(FEATS)))
    lrm = CM['lr']
    sv = {'mlp': SH['mlp'], 'xgb': SH['xgb'], 'lr': (Xsub - BG.mean(0)) * lrm.coef_[0]}       # linear SHAP: beta_j (x_j - E x_j)
    base = {'mlp': SH['mlp_base'], 'xgb': SH['xgb_base'], 'lr': float(lrm.intercept_[0] + BG.mean(0) @ lrm.coef_[0])}
    fx = {'mlp': f0(Xsub), 'xgb': SH['xgb'].sum(1) + SH['xgb_base'], 'lr': lrm.decision_function(Xsub)}
    for k in ('mlp', 'lr'):                         # additivity: base + sum of SHAP = model output
        assert np.abs(base[k] + sv[k].sum(1) - fx[k]).max() < 1e-3, k
    p_sub = sig(f0(Xsub))
    cand = np.where((p_sub > 0.5) & (p_sub < 0.8) & (yte[i_sub] == 1))[0]
    k_app = int(cand[np.argmin(np.abs(p_sub[cand] - 0.65))])
    rows = XR.index[i_te[i_sub]]
    assert int(rows[k_app]) == 2411                 # the applicant of the slides
    order = np.argsort(p_sub)                       # applicants by PD of the network (rank 1 = lowest)
    raw = XR.iloc[i_te[i_sub]][FEATS].values
    out['feats'] = FEATS
    out['wf'] = {'row': [int(rows[i]) for i in order], 'bad': [int(yte[i_sub][i]) for i in order],
                 'raw': [R(raw[i], 1) for i in order], 'app': int(np.where(order == k_app)[0][0]),
                 'base': {k: round(float(v), 4) for k, v in base.items()},
                 'fx': {k: R(fx[k][order], 4) for k in sv}, 'sv': {k: R(sv[k][order], 4) for k in sv}}
    # LIME on the applicant of the slides: 30 runs with 500 and 5,000 samples, 7 kernel widths (5,000 samples, seed 0)
    out['lime'] = {'s500': R(LM['s500'], 4), 's5000': R(LM['s5000'], 4), 'kw': [float(k) for k in LM['kw']], 'kw_w': R(LM['kw_w'], 4),
                   'kw_default': round(0.75 * np.sqrt(len(FEATS)), 2)}
    # 10 training seeds: SHAP of the first 200 explained applicants, networks and boosted ensembles
    out['seed'] = {m: [R(s, 3) for s in SS[m]] for m in ('mlp', 'xgb')}
    out['seed']['row'] = [int(r) for r in rows[:200]]

    # ------------------------------------------------------------ B. fraud: IG completeness and deletion curves
    fr = npl.fraud()
    FF = [f"V{i}" for i in range(1, 29)] + ["logAmount"]
    fr["logAmount"] = np.log1p(fr["Amount"])
    day1 = fr["Time"] < 86400
    d1, d2 = fr[day1], fr[~day1]
    j_tr, _ = train_test_split(np.arange(len(d1)), test_size=0.25, stratify=d1["Class"], random_state=SEED)
    fmu, fsd = d1.iloc[j_tr][FF].mean(), d1.iloc[j_tr][FF].std()
    Z1 = ((d1[FF] - fmu) / fsd).values.astype(np.float32)
    Z2 = ((d2[FF] - fmu) / fsd).values.astype(np.float32)
    y1, y2 = d1["Class"].values, d2["Class"].values
    FX = load('fraud_xai_v1')
    fnet = _net(load('fraud_model_v1')['state'], len(FF))
    ff = _fn(fnet)
    base_f = Z1[j_tr][y1[j_tr] == 0].mean(0)        # baseline x' = mean legitimate training transaction of day 1
    F2 = Z2[y2 == 1]
    A = FX['A']

    def grad_of(X):
        x = torch.tensor(np.asarray(X, dtype=np.float32), requires_grad=True)
        g, = torch.autograd.grad(fnet(x).sum(), x)
        return g.numpy()

    def ig(X, m):                                    # midpoint Riemann sum, as in analysis_L11.py
        tot = np.zeros_like(X)
        for a in (np.arange(m) + 0.5) / m:
            tot += grad_of(base_f + a * (X - base_f))
        return (X - base_f) * tot / m

    score = sig(ff(F2))
    srt = np.argsort(score)
    kf = int(srt[len(F2) // 2])                      # the median-score fraud of the slides, then 7 more over the score range
    pick = [kf] + [int(i) for i in srt[np.linspace(5, len(F2) - 6, 7).astype(int)] if int(i) != kf][:7]
    M = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    gap = ff(F2) - ff(base_f[None])[0]
    rel = {m: float(np.median(np.abs(ig(F2, m).sum(1) - gap) / np.abs(gap))) for m in M}
    for m in (2, 4, 64):                             # same numbers as the slides
        assert abs(rel[m] - FX['comp'][m]) / FX['comp'][m] < 1e-3, (m, rel[m], FX['comp'][m])
    out['ig'] = {'feats': FF, 'M': M, 'rel': [float(f'{rel[m]:.4g}') for m in M],
                 'cases': [{'p': round(float(score[i]), 4), 'fx': round(float(ff(F2[i:i + 1])[0]), 4), 'fb': round(float(ff(base_f[None])[0]), 4),
                            'ig': [R(ig(F2[i:i + 1], m)[0], 4) for m in M]} for i in pick]}

    # deletion: fraud score after resetting the k most important features (k = 0..12) of each fraud to the baseline
    Kd = 12
    rg = np.random.default_rng(SEED)

    def deletion_rows(Amat, rand=False):
        res = np.zeros((len(F2), Kd + 1))
        for i, x in enumerate(F2):
            o = rg.permutation(len(x)) if rand else np.argsort(-np.abs(Amat[i]))
            xs = np.repeat(x[None], Kd + 1, 0)
            for k in range(1, Kd + 1):
                xs[k:, o[k - 1]] = base_f[o[k - 1]]
            res[i] = sig(ff(xs))
        return res

    DEL = {m: deletion_rows(A[m]) for m in ('saliency', 'ig', 'shap', 'lime')}
    DEL['random'] = deletion_rows(None, rand=True)
    assert abs(100 * DEL['ig'].mean(0)[3] - 61.3) < 0.06 and abs(100 * DEL['random'].mean(0)[3] - 83.9) < 0.06
    od = np.argsort(-score)
    out['del'] = {'p': R(score[od], 4), **{m: R(DEL[m][od], 3) for m in DEL}}

    # ------------------------------------------------------------ C. California housing: PDP and ICE of the network
    ca = npl.california()
    CF = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
    k_tr, k_te = train_test_split(np.arange(len(ca)), test_size=0.2, random_state=SEED)
    k_tr, _ = train_test_split(k_tr, test_size=0.2, random_state=SEED)
    cmu, csd = ca.iloc[k_tr][CF].mean(), ca.iloc[k_tr][CF].std()
    CZ = ((ca[CF] - cmu) / csd).values.astype(np.float32)
    cf_ = _fn(_net(load('cal_model_v1')['state'], len(CF)))
    rgc = np.random.default_rng(SEED)
    ice_rows = CZ[k_te][rgc.choice(len(k_te), 60, replace=False)]
    pdp_rows = CZ[k_te][rgc.choice(len(k_te), 1000, replace=False)]
    out['pdp'] = {}
    for f in ("MedInc", "HouseAge", "AveRooms", "AveOccup", "Latitude"):
        j = CF.index(f)
        g_raw = np.linspace(np.percentile(ca[f], 2), np.percentile(ca[f], 98), 30)
        g_z = (g_raw - cmu[f]) / csd[f]

        def curves(rows_):
            res = np.zeros((len(rows_), len(g_z)))
            for g, z in enumerate(g_z):
                Xg = rows_.copy()
                Xg[:, j] = z
                res[:, g] = cf_(Xg)
            return res
        out['pdp'][f] = {'grid': R(g_raw, 3), 'ice': R(curves(ice_rows), 3), 'pdp': R(curves(pdp_rows).mean(0), 3),
                         'own': R(ice_rows[:, j] * csd[f] + cmu[f], 3)}
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
