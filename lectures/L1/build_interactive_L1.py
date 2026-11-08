"""Build the interactive page of Lecture 1 (from linear models to the perceptron and the MLP): interactive/L1.html (EN) and
interactive/L1_ro.html (RO).

Inputs (interactive/L1_data.js) come from data/clean through lectures/L1/interactive_data_L1.py (same splits, models and
seeds as analysis_L1.py) and from the caches of analysis_L1.py; the page logic is in interactive/L1.js.
Run: python3 lectures/L1/build_interactive_L1.py [--no-data]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, play, charts, rnd  # noqa: E402

K = 1
WIDTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
UA_H = [1, 2, 3, 5, 10, 20, 50]

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('one-neuron', 'boundary_hmeq', 'neuron', 'nb'),
          ('universal-approximation', 'uat_california', 'neuron', 'ua'),
          ('width', 'width_auc', 'neuron', 'wd'),
          ('roc-threshold', 'roc_hmeq', 'eval', 'rc'),
          ('cost-threshold', 'cost_threshold', 'eval', 'ct')]

SECTIONS = {
 'en': dict(neuron='From one neuron to a multilayer perceptron', eval='From scores to credit decisions'),
 'ro': dict(neuron='De la un neuron la perceptronul multistrat', eval='De la scoruri la decizii de creditare'),
}

L = {
 'en': dict(h1='Lecture 1 · Deep learning for business: from linear models to the multilayer perceptron',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the course data '
        'when you move a control. The ▶ button moves the main parameter step by step. Under each figure: a readout, a question '
        'for the room and the Quantlet with the code of the chart. No formula is needed to play: move the weights, the number of '
        'hidden units or the cut-off and watch what changes.',
   src='Data: HMEQ home-equity loans (Baesens, Roesch and Scheule, 2016; 5,960 loans), UCI Statlog German credit (1,000 '
       'applicants), California housing (StatLib, 1990 census). Splits, models and seeds (20261108 + s) as in the lecture '
       '(analysis_L1.py): HMEQ 60/20/20 stratified split; German credit out-of-fold scores of a 5-fold cross-validation; '
       'MLP = multilayer perceptron, XGBoost = gradient boosting, AUC = area under the ROC (receiver operating characteristic) curve.'),
 'ro': dict(h1='Cursul 1 · Deep learning pentru afaceri: de la modelele liniare la perceptronul multistrat',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser din datele cursului cînd modificați '
        'un control. Butonul ▶ modifică parametrul principal pas cu pas. Sub fiecare figură: o sinteză numerică, o întrebare '
        'pentru sală și Quantlet-ul cu codul figurii. Nu este nevoie de formule: modificați ponderile, numărul de neuroni ascunși '
        'sau pragul și urmăriți ce se schimbă.',
   src='Date: creditele ipotecare HMEQ (Baesens, Roesch și Scheule, 2016; 5.960 de credite), UCI Statlog German credit (1.000 '
       'de solicitanți), locuințele din California (StatLib, recensămîntul din 1990). Împărțirile, modelele și semințele '
       '(20261108 + s) sînt cele din curs (analysis_L1.py): HMEQ împărțit stratificat 60/20/20; pentru German credit, scoruri '
       'în afara pliului dintr-o validare încrucișată cu 5 pliuri; MLP = perceptron multistrat, XGBoost = gradient boosting, '
       'AUC = aria de sub curba ROC (receiver operating characteristic).'),
}


def widgets(lang):
    ro = lang == 'ro'
    models = [('logit', 'Regresia logistică' if ro else 'Logistic regression'), ('mlp', 'MLP 64-64'), ('xgb', 'XGBoost')]
    return {
     'one-neuron': (select('nb-m', 'Modelul' if ro else 'Model', [('n', 'Un neuron, ponderile dvs.' if ro else 'One neuron, your weights'),
                                                                   ('mlp', 'MLP 64-64 antrenat' if ro else 'Trained MLP 64-64')], 'n')
                    + slider('nb-w1', 'Ponderea pentru datorii/venit, w₁' if ro else 'Weight on debt-to-income, w₁', -3, 3, 0.05, 0.6)
                    + slider('nb-w2', 'Ponderea pentru vechimea celei mai vechi linii de credit, w₂' if ro else 'Weight on age of the oldest credit line, w₂', -3, 3, 0.05, -0.5)
                    + slider('nb-b', 'Termenul liber (bias), b' if ro else 'Bias, b', -5, 1, 0.05, -2.45)
                    + slider('nb-t', 'Pragul: semnalăm creditul dacă probabilitatea ≥' if ro else 'Cut-off: flag the loan if probability ≥', 0.05, 0.6, 0.01, 0.2)
                    + '<div class="ctl"><button class="btn" id="nb-fit" type="button">'
                    + ('Ponderile estimate (regresia logistică)' if ro else 'Fitted weights (logistic regression)') + '</button></div>',
                    charts('nb-chart', cls='one')),
     'universal-approximation': (slider('ua-h', 'Neuroni ascunși sigmoizi, H' if ro else 'Sigmoid hidden units, H', 0, len(UA_H) - 1, 1, 2)
                                 + toggles('ua-s', 'Afișare' if ro else 'Show', [('lin', 'Dreapta de regresie' if ro else 'Linear regression line'),
                                                                                 ('units', 'Contribuția fiecărui neuron' if ro else 'Contribution of each unit')], ['lin', 'units'])
                                 + play('ua'), charts('ua-fit', 'ua-units')),
     'width': (slider('wd-h', 'Neuroni ascunși, H (un strat ascuns)' if ro else 'Hidden units, H (one hidden layer)', 0, len(WIDTHS) - 1, 1, 0)
               + toggles('wd-s', 'Afișare' if ro else 'Show', [('seeds', 'Fiecare dintre cele 5 inițializări' if ro else 'Each of the 5 seeds'),
                                                               ('bench', 'Regresia logistică și XGBoost' if ro else 'Logistic regression and XGBoost')], ['seeds', 'bench'])
               + play('wd'), charts('wd-chart', cls='one')),
     'roc-threshold': (select('rc-m', 'Modelul' if ro else 'Model', models, 'mlp')
                       + slider('rc-t', 'Pragul: semnalăm creditul dacă PD ≥' if ro else 'Cut-off: flag the loan if PD ≥', 0.01, 0.95, 0.01, 0.5)
                       + play('rc'), charts('rc-roc', 'rc-hist')),
     'cost-threshold': (slider('ct-c', 'Costul unui rău-platnic acceptat (costul unui bun-platnic respins = 1)' if ro else 'Cost of accepting a bad applicant (cost of rejecting a good one = 1)', 1, 10, 0.5, 5)
                        + slider('ct-t', 'Pragul: respingem dacă PD ≥' if ro else 'Cut-off: reject if PD ≥', 0.02, 0.9, 0.01, 0.5)
                        + toggles('ct-m', 'Modele' if ro else 'Models', models, ['logit', 'mlp', 'xgb']) + play('ct'),
                        charts('ct-chart', cls='one')),
    }


QA = {
 'en': {
  'one-neuron': ('A single neuron with two inputs, debt-to-income and the age of the oldest credit line, draws a straight boundary. '
                 'Which sign should each weight have, and what happens to the AUC (area under the ROC curve) if you flip both signs?',
                 'Debt-to-income raises the risk (w₁ > 0) and an older credit history lowers it (w₂ < 0): the fitted logistic regression gives '
                 'w₁ = 0.62 and w₂ = −0.48 per standard deviation, with a test AUC of 0.744 on the 871 test loans that report both variables. '
                 'Flip both signs and the ranking is reversed: the AUC drops to 0.256 = 1 − 0.744. Changing the bias moves the line without '
                 'rotating it, so it changes how many loans are flagged but not the AUC. The trained MLP 64-64 bends the boundary and reaches '
                 '0.736 on the same test loans: with two inputs there is little non-linear structure to find.'),
  'universal-approximation': ('With one hidden layer of sigmoid units the network adds up "steps". How many units does it take to follow '
                              'the north–south pattern of house values in California?',
                              'One unit gives a single smooth step (test MSE 0.855, against 0.896 for the straight line); three units already '
                              'capture the two coastal peaks around Los Angeles and San Francisco (0.757); fifty units bring only a little '
                              'more (0.731) on the 4,912 test block groups. The right panel shows each unit\'s contribution: the fitted curve '
                              'is their sum plus a constant. This is the universal approximation theorem at work, and also its limit: a '
                              'single variable explains only part of the variance (mean squared error, MSE, in squared 100,000 USD).'),
  'width': ('On HMEQ, how many hidden units does a one-layer MLP need before it beats logistic regression on test AUC?',
            'One hidden unit (AUC 0.901) is a logistic regression with an extra squashing step and does worse than the logistic model '
            '(0.920); from 4 to 8 units the MLP passes it (0.923, 0.931) and the gain flattens above 32 units (0.946 to 0.953 at 256). '
            'Gradient boosting (XGBoost, 0.961) stays ahead on this small tabular dataset, with far fewer tuning choices. The readout '
            'gives the number of parameters: 7,681 at H = 256, against 29 for the logistic regression.'),
  'roc-threshold': ('A bank flags a loan for manual review when its predicted probability of default (PD) is at least the cut-off. '
                    'What does moving the cut-off from 0.5 to 0.2 do for the MLP?',
                    'At 0.5 the MLP flags 220 of the 1,192 test loans and catches 76.5% of the 238 defaults, with 4.0% of the good loans '
                    'flagged by mistake; at 0.2 it flags 290 loans, catches 84.9% of the defaults and flags 9.2% of the good loans. '
                    'The cut-off moves the point along the ROC curve; the curve itself, and the AUC, do not change. Which point is best '
                    'depends on the cost of each error, the subject of the next block.'),
  'cost-threshold': ('The UCI cost matrix of German credit says that accepting a bad applicant costs 5 times as much as rejecting a good one. '
                     'Is 0.5 a sensible cut-off?',
                     'No. With costs 5 and 1 the cut-off that minimises the expected cost is 1/(1 + 5) = 1/6 ≈ 0.17 (Bayes decision rule). '
                     'At 0.5 the cost per applicant is 0.87 for the logistic regression, 1.13 for the MLP and 0.98 for XGBoost, worse than '
                     'rejecting everyone (0.70); at 1/6 it falls to 0.55, 0.57 and 0.59. Move the cost slider: the best cut-off follows '
                     '1/(1 + c), and at c = 1 it is back at 0.5. The differences between the three models are much smaller than the cost '
                     'of a badly chosen cut-off.'),
 },
 'ro': {
  'one-neuron': ('Un singur neuron cu două intrări, raportul datorii/venit și vechimea celei mai vechi linii de credit, trasează o frontieră '
                 'dreaptă. Ce semn ar trebui să aibă fiecare pondere și ce se întîmplă cu AUC (aria de sub curba ROC) dacă schimbați ambele semne?',
                 'Raportul datorii/venit crește riscul (w₁ > 0), iar un istoric de credit mai vechi îl reduce (w₂ < 0): regresia logistică '
                 'estimată dă w₁ = 0,62 și w₂ = −0,48 la o abatere standard, cu un AUC pe test de 0,744 pe cele 871 de credite de test care '
                 'au ambele variabile. Dacă schimbați ambele semne, ordonarea se inversează: AUC scade la 0,256 = 1 − 0,744. Termenul liber '
                 'mută dreapta fără să o rotească, deci schimbă numărul de credite semnalate, dar nu și AUC. MLP 64-64 antrenat curbează '
                 'frontiera și obține 0,736 pe aceleași credite: cu două intrări există puțină structură neliniară de găsit.'),
  'universal-approximation': ('Cu un strat ascuns de neuroni sigmoizi, rețeaua adună „trepte”. Cîți neuroni sînt necesari pentru a urmări '
                              'profilul nord–sud al valorii locuințelor din California?',
                              'Un neuron dă o singură treaptă netedă (MSE pe test 0,855, față de 0,896 pentru dreapta de regresie); trei '
                              'neuroni surprind deja cele două vîrfuri de pe coastă, în jurul orașelor Los Angeles și San Francisco (0,757); '
                              'cincizeci de neuroni aduc doar puțin în plus (0,731) pe cele 4.912 grupuri de locuințe de test. Panoul din '
                              'dreapta arată contribuția fiecărui neuron: curba estimată este suma lor plus o constantă. Este teorema de '
                              'aproximare universală la lucru, dar și limita ei: o singură variabilă explică doar o parte din varianță '
                              '(eroarea pătratică medie, MSE, în pătrate de 100.000 USD).'),
  'width': ('Pe HMEQ, de cîți neuroni ascunși are nevoie un MLP cu un strat ascuns ca să depășească regresia logistică la AUC pe test?',
            'Un singur neuron ascuns (AUC 0,901) este o regresie logistică cu o comprimare în plus și este mai slab decît modelul logistic '
            '(0,920); între 4 și 8 neuroni, MLP îl depășește (0,923, 0,931), iar cîștigul se aplatizează peste 32 de neuroni (de la 0,946 '
            'la 0,953 pentru 256). Gradient boosting (XGBoost, 0,961) rămîne înainte pe acest set tabular mic, cu mult mai puține alegeri '
            'de reglaj. Sinteza dă numărul de parametri: 7.681 pentru H = 256, față de 29 pentru regresia logistică.'),
  'roc-threshold': ('O bancă trimite un credit la analiză manuală cînd probabilitatea estimată de neplată (PD) este cel puțin egală cu pragul. '
                    'Ce se întîmplă pentru MLP dacă pragul scade de la 0,5 la 0,2?',
                    'La 0,5, MLP semnalează 220 din cele 1.192 de credite de test și detectează 76,5% din cele 238 de neplăți, cu 4,0% '
                    'dintre creditele bune semnalate din greșeală; la 0,2 semnalează 290 de credite, detectează 84,9% din neplăți și '
                    'semnalează 9,2% dintre creditele bune. Pragul mută punctul de-a lungul curbei ROC; curba însăși și AUC nu se schimbă. '
                    'Care punct este cel mai bun depinde de costul fiecărei erori, subiectul blocului următor.'),
  'cost-threshold': ('Matricea de costuri UCI pentru German credit spune că acceptarea unui rău-platnic costă de 5 ori mai mult decît '
                     'respingerea unui bun-platnic. Este 0,5 un prag rezonabil?',
                     'Nu. Cu costurile 5 și 1, pragul care minimizează costul așteptat este 1/(1 + 5) = 1/6 ≈ 0,17 (regula de decizie Bayes). '
                     'La 0,5, costul pe solicitant este 0,87 pentru regresia logistică, 1,13 pentru MLP și 0,98 pentru XGBoost, mai mare '
                     'decît dacă am respinge pe toată lumea (0,70); la 1/6 scade la 0,55, 0,57 și 0,59. Modificați costul: pragul optim '
                     'urmează 1/(1 + c), iar la c = 1 revine la 0,5. Diferențele dintre cele trei modele sînt mult mai mici decît costul '
                     'unui prag ales prost.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 models: { logit: 'Logistic regression', mlp: 'MLP 64-64', xgb: 'XGBoost' },
 dti: 'Debt-to-income (%)', clage: 'Age of the oldest credit line (months)', prob: 'Predicted probability of default',
 repaid: 'Repaid (test)', defaulted: 'Defaulted (test)', boundary: c => `Boundary: probability = ${c}`,
 nbTitle: m => m === 'n' ? 'One neuron: p = σ(b + w₁·z₁ + w₂·z₂), z = standardised inputs' : 'Trained MLP 64-64',
 nb1: (auc, w1, w2, b) => `Test AUC of this score: <b>${auc}</b> (w₁ = ${w1}, w₂ = ${w2}, b = ${b}); fitted logistic regression 0.744, MLP 64-64 0.736.`,
 nb2: (t, n, k, kk, f, ff) => `Cut-off ${t}: ${n} of 871 test loans flagged; ${k} of ${kk} defaults caught; ${f} of ${ff} repaid loans flagged by mistake.`,
 nbRev: 'The AUC is below 0.5: this neuron ranks the safe loans as risky (flip the signs).',
 latitude: 'Latitude (degrees north)', value: 'Value (100,000 USD)', data: 'Data: mean value in 40 latitude bins',
 lin: 'Linear regression', fit: h => `MLP, H = ${h}`, unitsTitle: 'Contribution of each hidden unit, vⱼ·σ(aⱼ + bⱼ·x)',
 fitTitle: 'Median house value by latitude', unit: j => `unit ${j}`,
 ua: (h, te, tr, lin) => `H = ${h}: test MSE <b>${te}</b> (training ${tr}); straight line ${lin}; 14,736 training and 4,912 test block groups.`,
 hidden: 'Hidden units H (log scale)', auc: 'Test AUC', mean5: 'MLP, one hidden layer (mean of 5 seeds)', range5: 'range of 5 seeds',
 seeds: 'MLP, each of the 5 seeds', logitLine: 'Logistic regression', xgbLine: 'Gradient boosting (XGBoost)',
 wd: (h, m, sd, np) => `H = ${h}: mean test AUC <b>${m}</b> (standard deviation over 5 seeds ${sd}); ${np} parameters (logistic regression: 29).`,
 fpr: 'False positive rate: good loans flagged (%)', tpr: 'Defaults caught (%)', rocTitle: 'ROC curves, 1,192 test loans',
 histTitle: 'Predicted PD by outcome', pd: 'Predicted probability of default (PD)', count: 'Loans', chance: 'Random ranking',
 rc: (m, auc, t, tp, k, fp, f, n) => `${m} (AUC ${auc}), cut-off ${t}: ${n} loans flagged; <b>${tp}</b> of defaults caught (${k} of 238); ${fp} of good loans flagged (${f} of 954).`,
 thr: 'Cut-off: reject if predicted PD ≥ threshold', costPA: 'Cost per applicant', bayes: c => `Cost-optimal cut-off 1/(1 + ${c})`,
 acceptAll: 'Accept everyone', rejectAll: 'Reject everyone', costTitle: c => `Expected cost per applicant, missed bad = ${c}, rejected good = 1`,
 ct: (m, t, c, best, bt) => `${m}: cost <b>${c}</b> per applicant at cut-off ${t}; minimum ${best} at ${bt}.`,
 ct0: (bt, all, none) => `Bayes cut-off 1/(1 + c) = ${bt}. Accept everyone: ${all}; reject everyone: ${none} (1,000 applicants, 300 bad).`
};
""",
 'ro': r"""const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 models: { logit: 'Regresia logistică', mlp: 'MLP 64-64', xgb: 'XGBoost' },
 dti: 'Datorii/venit (%)', clage: 'Vechimea celei mai vechi linii de credit (luni)', prob: 'Probabilitatea estimată de neplată',
 repaid: 'Rambursat (test)', defaulted: 'Neplată (test)', boundary: c => `Frontiera: probabilitatea = ${c}`,
 nbTitle: m => m === 'n' ? 'Un neuron: p = σ(b + w₁·z₁ + w₂·z₂), z = intrări standardizate' : 'MLP 64-64 antrenat',
 nb1: (auc, w1, w2, b) => `AUC pe test al acestui scor: <b>${auc}</b> (w₁ = ${w1}, w₂ = ${w2}, b = ${b}); regresia logistică estimată 0,744, MLP 64-64 0,736.`,
 nb2: (t, n, k, kk, f, ff) => `Pragul ${t}: ${n} din 871 de credite de test semnalate; ${k} din ${deRO(kk)} neplăți detectate; ${f} din ${deRO(ff)} credite rambursate semnalate din greșeală.`,
 nbRev: 'AUC este sub 0,5: acest neuron consideră riscante creditele sigure (schimbați semnele).',
 latitude: 'Latitudinea (grade nord)', value: 'Valoarea (100.000 USD)', data: 'Date: valoarea medie pe 40 de clase de latitudine',
 lin: 'Regresia liniară', fit: h => `MLP, H = ${h}`, unitsTitle: 'Contribuția fiecărui neuron ascuns, vⱼ·σ(aⱼ + bⱼ·x)',
 fitTitle: 'Valoarea mediană a locuințelor după latitudine', unit: j => `neuronul ${j}`,
 ua: (h, te, tr, lin) => `H = ${h}: MSE pe test <b>${te}</b> (antrenare ${tr}); dreapta de regresie ${lin}; 14.736 de grupuri de locuințe de antrenare și 4.912 de test.`,
 hidden: 'Neuroni ascunși H (scară logaritmică)', auc: 'AUC pe test', mean5: 'MLP cu un strat ascuns (media a 5 inițializări)', range5: 'intervalul celor 5 inițializări',
 seeds: 'MLP, fiecare dintre cele 5 inițializări', logitLine: 'Regresia logistică', xgbLine: 'Gradient boosting (XGBoost)',
 wd: (h, m, sd, np) => `H = ${h}: AUC mediu pe test <b>${m}</b> (abaterea standard pe 5 inițializări ${sd}); ${deRO(np)} parametri (regresia logistică: 29).`,
 fpr: 'Rata alarmelor false: credite bune semnalate (%)', tpr: 'Neplăți detectate (%)', rocTitle: 'Curbele ROC, 1.192 de credite de test',
 histTitle: 'PD estimată, după rezultat', pd: 'Probabilitatea estimată de neplată (PD)', count: 'Credite', chance: 'Ordonare aleatoare',
 rc: (m, auc, t, tp, k, fp, f, n) => `${m} (AUC ${auc}), pragul ${t}: ${deRO(n)} credite semnalate; <b>${tp}</b> dintre neplăți detectate (${k} din 238); ${fp} dintre creditele bune semnalate (${f} din 954).`,
 thr: 'Pragul: respingem dacă PD estimată ≥ prag', costPA: 'Costul pe solicitant', bayes: c => `Pragul optim pentru cost 1/(1 + ${c})`,
 acceptAll: 'Acceptăm pe toată lumea', rejectAll: 'Respingem pe toată lumea', costTitle: c => `Costul așteptat pe solicitant, rău-platnic acceptat = ${c}, bun-platnic respins = 1`,
 ct: (m, t, c, best, bt) => `${m}: costul <b>${c}</b> pe solicitant la pragul ${t}; minimul ${best} la ${bt}.`,
 ct0: (bt, all, none) => `Pragul Bayes 1/(1 + c) = ${bt}. Acceptăm pe toată lumea: ${all}; respingem pe toată lumea: ${none} (1.000 de solicitanți, 300 rău-platnici).`
};
""",
}


# ---------------------------------------------------------------- data (real course data only)
def data():
    import interactive_data_L1 as idl
    import pandas as pd
    r = idl.load()
    cdir = os.path.join(HERE, 'cache')
    N = json.load(open(os.path.join(HERE, 'L1_numbers.json')))
    out = {}
    nb = r['nb']
    out['nb'] = {'mu': rnd(nb['mu'], 4), 'sd': rnd(nb['sd'], 4), 'w': rnd(nb['w'], 4), 'b': rnd(nb['b'], 4),
                 'x1': nb['x1'], 'x2': nb['x2'], 'y': nb['y'], 'gx': nb['gx'], 'gy': nb['gy'], 'mlp': nb['mlp']}
    ua = r['ua']
    out['ua'] = {'H': UA_H, 'x': ua['x'], 'bx': ua['bx'], 'by': ua['by'], 'lin': ua['lin'], 'mse_lin': rnd(ua['mse_lin'], 4),
                 'fits': {h: {'f': f['f'], 'c': rnd(f['c'], 3), 'u': f['u'], 'te': rnd(f['mse_te'], 4), 'tr': rnd(f['mse_tr'], 4)}
                          for h, f in ua['fits'].items()}}
    ws = json.load(open(os.path.join(cdir, 'width_sweep_v1.json')))
    out['wd'] = {'H': WIDTHS, 'auc': {str(w): rnd(ws[str(w)], 4) for w in WIDTHS}, 'logit': N['hmeq']['auc_logit'],
                 'xgb': N['hmeq']['auc_xgb'], 'd': N['hmeq']['d']}
    out['rc'] = r['roc']
    g = pd.read_csv(os.path.join(interactive_common.DATA, 'german_credit.csv'))
    oof = json.load(open(os.path.join(cdir, 'german_cv_v1.json')))['oof']
    out['ct'] = {'y': g['default'].astype(int).tolist(), **{k: rnd(v, 4) for k, v in oof.items()}}
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
