"""Build the interactive page of Lecture 3 (regularisation and generalisation): interactive/L3.html (EN) and interactive/L3_ro.html (RO).

Inputs (interactive/L3_data.js) come from data/clean (German credit, card fraud, California housing) and the caches of
analysis_L3.py through lectures/L3/interactive_data_L3.py (same splits, models and seeds); the page logic is in interactive/L3.js.
Run: python3 lectures/L3/build_interactive_L3.py [--no-data]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import interactive_common  # noqa: E402
from interactive_common import slider, toggles, play, charts, rnd  # noqa: E402

K = 3
BLOCKS = [('overfitting', 'overfit_german', 'gen', 'ov'),
          ('double-descent', 'double_descent', 'gen', 'dd'),
          ('early-stopping', 'early_stopping', 'reg', 'es'),
          ('holdout-noise', 'holdout_noise', 'eval', 'ho'),
          ('fraud-cost', 'fraud_cost', 'eval', 'fc')]

SECTIONS = {
 'en': dict(gen='Overfitting and generalisation', reg='Early stopping', eval='Validation and the cost of errors'),
 'ro': dict(gen='Supraajustare și generalizare', reg='Oprirea timpurie', eval='Validarea și costul erorilor'),
}

L = {
 'en': dict(h1='Lecture 3 · Regularisation and generalisation',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the course data '
        'when you move a control. The ▶ button moves the main parameter step by step. Under each figure: a readout, a question '
        'for the room and the Quantlet with the code of the chart. The curves are the real training runs of the lecture, saved '
        'epoch by epoch; the controls choose which run you see and where you stop it.',
   src='Data: UCI Statlog German credit (1,000 applicants; 600 training, 200 validation, 200 test), ULB and Worldline credit-card '
       'transactions (OpenML 1597; all 492 frauds and a 10% sample of the legitimate transactions, weighted back to the 284,807 '
       'transactions of the two days; time-ordered split 60/20/20), California housing (StatLib, 1990 census). Models, splits and '
       'seeds (20261108 + s) as in the lecture (analysis_L3.py): multilayer perceptrons (MLP) with AdamW (Adam with decoupled weight '
       'decay); PR-AUC = area under the precision-recall curve.'),
 'ro': dict(h1='Cursul 3 · Regularizare și generalizare',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser din datele cursului cînd modificați '
        'un control. Butonul ▶ modifică parametrul principal pas cu pas. Sub fiecare figură: o sinteză numerică, o întrebare '
        'pentru sală și Quantlet-ul cu codul figurii. Curbele sînt antrenările reale din curs, salvate epocă cu epocă; controalele '
        'aleg antrenarea afișată și momentul în care o opriți.',
   src='Date: UCI Statlog German credit (1.000 de solicitanți; 600 de antrenare, 200 de validare, 200 de test), tranzacțiile cu '
       'carduri ULB și Worldline (OpenML 1597; toate cele 492 de fraude și un eșantion de 10% din tranzacțiile legitime, ponderate '
       'înapoi la cele 284.807 tranzacții din cele două zile; împărțire cronologică 60/20/20), locuințele din California (StatLib, '
       'recensămîntul din 1990). Modelele, împărțirile și semințele (20261108 + s) sînt cele din curs (analysis_L3.py): perceptroni '
       'multistrat (MLP) antrenați cu AdamW (Adam cu penalizarea ponderilor decuplată); PR-AUC = aria de sub curba '
       'precizie-sensibilitate.'),
}


def widgets(lang):
    ro = lang == 'ro'
    models = [('logit', 'Regresia logistică' if ro else 'Logistic regression'), ('mlp', 'MLP regularizat' if ro else 'Regularised MLP'),
              ('xgb', 'XGBoost')]
    return {
     'overfitting': (slider('ov-p', 'Dropout (proporția de neuroni opriți)' if ro else 'Dropout (share of units switched off)', 0, 2, 1, 0)
                     + slider('ov-w', 'Penalizarea ponderilor (weight decay)' if ro else 'Weight decay', 0, 2, 1, 0)
                     + slider('ov-e', 'Oprim antrenarea după epoca' if ro else 'Stop training after epoch', 1, 150, 1, 150) + play('ov'),
                     charts('ov-loss', 'ov-auc')),
     'double-descent': (slider('dd-p', 'Neuroni ascunși p (caracteristici ReLU aleatoare)' if ro else 'Hidden units p (random ReLU features)', 0, 17, 1, 2)
                        + toggles('dd-s', 'Afișare' if ro else 'Show', [('draws', 'Cele 10 extrageri pentru p ales' if ro else 'The 10 draws at the chosen p')], ['draws'])
                        + play('dd'), charts('dd-chart', cls='one')),
     'early-stopping': (slider('es-p', 'Răbdarea (epoci fără îmbunătățire înainte de oprire)' if ro else 'Patience (epochs without improvement before stopping)', 1, 60, 1, 20)
                        + play('es'), charts('es-loss', 'es-ap')),
     'holdout-noise': (slider('ho-s', 'Proporția setului de test' if ro else 'Share of the test set', 0, 3, 1, 2) + play('ho'),
                       charts('ho-chart', cls='one')),
     'fraud-cost': (slider('fc-r', 'Costul verificării unei alerte (EUR)' if ro else 'Cost of reviewing one alert (EUR)', 1, 50, 1, 5)
                    + slider('fc-a', 'Alerte trimise la verificare (ponderat, scară logaritmică)' if ro else 'Alerts sent to review (weighted, log scale)', 1, 3.6, 0.02, 1.8)
                    + toggles('fc-m', 'Modele' if ro else 'Models', models, ['logit', 'mlp', 'xgb']) + play('fc'),
                    charts('fc-chart', cls='one')),
    }


QA = {
 'en': {
  'overfitting': ('An MLP with 78,593 parameters is trained on 600 applicants for 150 epochs. The validation loss explodes; does the '
                  'model also get worse at ranking applicants?',
                  'Not much. Without regularisation the validation cross-entropy is lowest at epoch 3 (0.536, below the logistic '
                  'regression at 0.570) and rises to 2.187 by epoch 150, while the training loss falls to 0.0001: the network '
                  'memorises the training applicants and becomes over-confident. The validation AUC, which only looks at the ranking, '
                  'moves from 0.730 at epoch 3 to 0.780 at epoch 150 (logistic regression: 0.760). Weight decay 1 keeps the final '
                  'validation loss at 1.08; dropout alone helps little. Probabilities (for pricing, provisions) are hurt far more by '
                  'overfitting than rankings are, which is why the lecture reports both.'),
  'double-descent': ('Random ReLU features on 400 California block groups: what happens to the test MSE when the number of features '
                     'reaches the number of observations, p = n = 400?',
                     'It explodes: the median test MSE is about 1,019,219 at p = 400, because the least-squares fit must interpolate '
                     'every training point with an ill-conditioned system. Below the threshold the best is p = 50 (0.781); above it '
                     'the minimum-norm solution becomes smoother again and the test MSE comes back down to 0.932 at p = 8,000, the '
                     '"second descent". The training MSE is zero from p = 400 on. Classical U-shaped bias–variance and the modern '
                     'over-parameterised regime are two halves of the same curve.'),
  'early-stopping': ('With a patience of 20 epochs, at which epoch does training stop on the fraud data, and which weights are kept?',
                     'The validation loss is lowest at epoch 4 (0.0159); with patience 20 training stops at epoch 24 and restores '
                     'the weights of epoch 4. Their validation PR-AUC is 0.763, against 0.697 if we kept training to epoch 120. Patience '
                     '1 stops too early (epoch 3, keeping epoch 2, PR-AUC 0.741); any patience from 2 to 60 keeps epoch 4 here. '
                     'Patience trades computing time against the risk of stopping on a noisy dip.'),
  'holdout-noise': ('Two analysts evaluate the same logistic regression on German credit with different random 70/30 splits. How far '
                    'apart can their test AUCs be?',
                    'Far: over 200 random splits the AUC ranges from 0.702 to 0.842 (mean 0.780, standard deviation 0.024). With a '
                    '10% test set (100 applicants, about 30 defaults) the standard deviation doubles to 0.047; with 50% it falls to 0.019, '
                    'but the model is then trained on fewer applicants (mean AUC 0.767). A difference of 0.01–0.02 AUC between two '
                    'models on one small hold-out is noise; repeated cross-validation or a paired test is needed.'),
  'fraud-cost': ('A missed fraud costs its amount; each alert costs a manual review. With a review cost of 5 EUR, how many alerts should '
                 'the bank send, and which model is cheapest?',
                 'Without a model all test frauds are lost: 7,729 EUR. With a review cost of 5 EUR the regularised MLP reaches its '
                 'minimum, 2,694 EUR, with about 59 weighted alerts; XGBoost is practically equal (2,695 EUR) and logistic regression '
                 'costs 3,093 EUR at about 142 alerts. Raise the review cost: every model sends fewer alerts and lets more small frauds '
                 'through, and the gap between the models widens. The cut-off is a business parameter, set by the cost of each error, '
                 'not by the model.'),
 },
 'ro': {
  'overfitting': ('Un MLP cu 78.593 de parametri este antrenat pe 600 de solicitanți timp de 150 de epoci. Pierderea pe validare '
                  'explodează; devine modelul și mai slab la ordonarea solicitanților?',
                  'Nu prea. Fără regularizare, entropia încrucișată pe validare este minimă la epoca 3 (0,536, sub regresia logistică, '
                  '0,570) și crește la 2,187 pînă la epoca 150, în timp ce pierderea de antrenare scade la 0,0001: rețeaua memorează '
                  'solicitanții de antrenare și devine prea sigură pe ea. AUC pe validare, care privește doar ordonarea, trece de la '
                  '0,730 la epoca 3 la 0,780 la epoca 150 (regresia logistică: 0,760). O penalizare a ponderilor de 1 ține pierderea '
                  'finală pe validare la 1,08; dropout-ul singur ajută puțin. Probabilitățile (pentru stabilirea dobînzii, a '
                  'provizioanelor) sînt afectate de supraajustare mult mai mult decît ordonarea; de aceea cursul le raportează pe amîndouă.'),
  'double-descent': ('Caracteristici ReLU aleatoare pe 400 de grupuri de locuințe din California: ce se întîmplă cu MSE pe test cînd '
                     'numărul de caracteristici ajunge la numărul de observații, p = n = 400?',
                     'Explodează: MSE mediană pe test este de aproximativ 1.019.219 la p = 400, pentru că ajustarea prin cele mai mici '
                     'pătrate trebuie să interpoleze fiecare punct de antrenare cu un sistem prost condiționat. Sub prag, cel mai bun '
                     'este p = 50 (0,781); peste prag, soluția de normă minimă devine din nou mai netedă, iar MSE pe test coboară la '
                     '0,932 la p = 8.000: „a doua coborîre”. MSE de antrenare este zero de la p = 400 încolo. Curba clasică în U '
                     'deplasare–varianță și regimul modern supraparametrizat sînt două jumătăți ale aceleiași curbe.'),
  'early-stopping': ('Cu o răbdare de 20 de epoci, la ce epocă se oprește antrenarea pe datele de fraudă și ce ponderi se păstrează?',
                     'Pierderea pe validare este minimă la epoca 4 (0,0159); cu răbdarea 20, antrenarea se oprește la epoca 24 și '
                     'restabilește ponderile epocii 4. PR-AUC pe validare al acestora este 0,763, față de 0,697 dacă am continua pînă '
                     'la epoca 120. Răbdarea 1 oprește prea devreme (epoca 3, păstrînd epoca 2, PR-AUC 0,741); orice răbdare între 2 '
                     'și 60 păstrează aici epoca 4. Răbdarea schimbă timp de calcul contra riscului de a ne opri pe o scădere zgomotoasă.'),
  'holdout-noise': ('Doi analiști evaluează aceeași regresie logistică pe German credit, cu împărțiri aleatoare 70/30 diferite. Cît de '
                    'diferite pot fi AUC-urile lor pe test?',
                    'Foarte diferite: pe 200 de împărțiri aleatoare, AUC variază între 0,702 și 0,842 (media 0,780, abaterea standard '
                    '0,024). Cu un set de test de 10% (100 de solicitanți, circa 30 de neplăți), abaterea standard se dublează la '
                    '0,047; cu 50% scade la 0,019, dar modelul este antrenat pe mai puțini solicitanți (AUC mediu 0,767). O diferență '
                    'de 0,01–0,02 AUC între două modele, pe un singur set mic de test, este zgomot; avem nevoie de validare încrucișată '
                    'repetată sau de un test pe perechi.'),
  'fraud-cost': ('O fraudă ratată costă valoarea ei; fiecare alertă costă o verificare manuală. Cu un cost de verificare de 5 EUR, cîte '
                 'alerte ar trebui să trimită banca și care model este cel mai ieftin?',
                 'Fără model, toate fraudele din setul de test sînt pierdute: 7.729 EUR. Cu un cost de verificare de 5 EUR, MLP '
                 'regularizat ajunge la minimul de 2.694 EUR, cu aproximativ 59 de alerte ponderate; XGBoost este practic egal '
                 '(2.695 EUR), iar regresia logistică costă 3.093 EUR, la circa 142 de alerte. Creșteți costul verificării: fiecare '
                 'model trimite mai puține alerte și lasă să treacă mai multe fraude mici, iar diferența dintre modele crește. Pragul '
                 'este un parametru de afaceri, stabilit de costul fiecărei erori, nu de model.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 epoch: 'Epoch', ce: 'Cross-entropy', auc: 'Validation AUC', train: 'Training loss', valid: 'Validation loss', logit: 'Logistic regression (validation)',
 stop: 'Stop here', best: 'Lowest validation loss', vauc: 'Validation AUC', lossTitle: (p, w) => `MLP 256-256, dropout ${p}, weight decay ${w}`,
 aucTitle: 'Ranking quality on the validation applicants',
 ov: (e, tr, va, auc, be, bv) => `Epoch ${e}: training loss ${tr}, validation loss <b>${va}</b>, validation AUC <b>${auc}</b>. Lowest validation loss ${bv} at epoch ${be}.`,
 ov2: (l, a) => `Logistic regression on the same split: validation loss ${l}, AUC ${a}.`,
 p: 'Hidden units p (log scale)', mse: 'MSE (log scale)', ddTest: 'Test MSE (median of 10 draws)', ddTrain: 'Training MSE (median)',
 draws: 'test MSE of each draw', thr: 'interpolation threshold p = n = 400',
 dd: (p, te, tr, lo, hi) => `p = ${p}: median test MSE <b>${te}</b> (10 draws from ${lo} to ${hi}); median training MSE ${tr}.`,
 esLoss: 'Training and validation loss (log scale)', esAp: 'Validation PR-AUC (population-weighted)', ap: 'PR-AUC', kept: 'Weights kept', stopAt: 'Training stops',
 es: (p, s, k, v, a, aEnd) => `Patience ${p}: training stops at epoch <b>${s}</b> and keeps the weights of epoch <b>${k}</b> (validation loss ${v}, PR-AUC ${a}); at epoch 120 the PR-AUC would be ${aEnd}.`,
 esNo: p => `Patience ${p}: no stop within the 120 epochs recorded.`,
 hoTitle: s => `Test AUC of logistic regression on 200 random ${s} test splits`, hoX: 'Test AUC', hoY: 'Number of splits',
 mean: 'mean', p5: '5th and 95th percentiles',
 ho: (s, n, m, sd, lo, hi, p5, p95) => `Test share ${s} (${n} applicants): mean AUC ${m}, standard deviation <b>${sd}</b>; range ${lo} to ${hi}; 90% of the splits between ${p5} and ${p95}.`,
 models: { logit: 'Logistic regression', mlp: 'Regularised MLP (5 seeds)', xgb: 'XGBoost' },
 alerts: 'Alerts sent to review (population-weighted, log scale)', cost: 'Cost (EUR)', noModel: 'No alerts: all test frauds lost',
 fcTitle: r => `Cost = ${r} EUR × alerts + amount of the missed frauds`,
 fc: (m, a, c, k, mc, ma) => `${m}: at ${a} alerts the cost is <b>${c} EUR</b> (${k} of 75 test frauds caught); minimum ${mc} EUR at ${ma} alerts.`,
 fc0: t => `Without any alert all ${t} EUR of test frauds are lost.`
};
""",
 'ro': r"""const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 epoch: 'Epoca', ce: 'Entropia încrucișată', auc: 'AUC pe validare', train: 'Pierderea de antrenare', valid: 'Pierderea de validare', logit: 'Regresia logistică (validare)',
 stop: 'Oprire aici', best: 'Pierderea minimă pe validare', vauc: 'AUC pe validare', lossTitle: (p, w) => `MLP 256-256, dropout ${p}, penalizare ${w}`,
 aucTitle: 'Calitatea ordonării pe solicitanții de validare',
 ov: (e, tr, va, auc, be, bv) => `Epoca ${e}: pierderea de antrenare ${tr}, pierderea de validare <b>${va}</b>, AUC pe validare <b>${auc}</b>. Pierderea minimă pe validare ${bv}, la epoca ${be}.`,
 ov2: (l, a) => `Regresia logistică pe aceeași împărțire: pierderea de validare ${l}, AUC ${a}.`,
 p: 'Neuroni ascunși p (scară logaritmică)', mse: 'MSE (scară logaritmică)', ddTest: 'MSE pe test (mediana a 10 extrageri)', ddTrain: 'MSE de antrenare (mediana)',
 draws: 'MSE pe test pentru fiecare extragere', thr: 'pragul de interpolare p = n = 400',
 dd: (p, te, tr, lo, hi) => `p = ${p}: MSE mediană pe test <b>${te}</b> (10 extrageri, între ${lo} și ${hi}); MSE mediană de antrenare ${tr}.`,
 esLoss: 'Pierderea de antrenare și de validare (scară logaritmică)', esAp: 'PR-AUC pe validare (ponderat la populație)', ap: 'PR-AUC', kept: 'Ponderile păstrate', stopAt: 'Antrenarea se oprește',
 es: (p, s, k, v, a, aEnd) => `Răbdarea ${p}: antrenarea se oprește la epoca <b>${s}</b> și păstrează ponderile epocii <b>${k}</b> (pierderea de validare ${v}, PR-AUC ${a}); la epoca 120, PR-AUC ar fi ${aEnd}.`,
 esNo: p => `Răbdarea ${p}: nicio oprire în cele 120 de epoci înregistrate.`,
 hoTitle: s => `AUC pe test al regresiei logistice, pe 200 de împărțiri aleatoare cu ${s} test`, hoX: 'AUC pe test', hoY: 'Numărul de împărțiri',
 mean: 'media', p5: 'percentilele 5 și 95',
 ho: (s, n, m, sd, lo, hi, p5, p95) => `Proporția de test ${s} (${deRO(n)} solicitanți): AUC mediu ${m}, abaterea standard <b>${sd}</b>; între ${lo} și ${hi}; 90% dintre împărțiri între ${p5} și ${p95}.`,
 models: { logit: 'Regresia logistică', mlp: 'MLP regularizat (5 semințe)', xgb: 'XGBoost' },
 alerts: 'Alerte trimise la verificare (ponderat, scară logaritmică)', cost: 'Costul (EUR)', noModel: 'Fără alerte: toate fraudele de test pierdute',
 fcTitle: r => `Costul = ${r} EUR × alertele + valoarea fraudelor ratate`,
 fc: (m, a, c, k, mc, ma) => `${m}: la ${deRO(a)} alerte costul este <b>${c} EUR</b> (${k} din 75 de fraude de test detectate); minimul ${mc} EUR la ${deRO(ma)} alerte.`,
 fc0: t => `Fără nicio alertă se pierd toți cei ${t} EUR ai fraudelor de test.`
};
""",
}


def data():
    import numpy as np
    import interactive_data_L3 as idl
    r = idl.load()
    out = {'ov': rnd(r['ov'], 4), 'dd': r['dd'], 'es': {'tr': r['es']['tr'], 'va': r['es']['va'], 'ap': r['es']['ap']},
           'ho': {k: v for k, v in r['ho'].items()}, 'shares': idl.SHARES}
    # fraud cost: for each model, the cut-offs of analysis_L3.py (300 quantiles of the test scores between 0.90 and 0.99995);
    # at each cut-off: weighted alerts, amount of the missed frauds, frauds caught (the review cost is applied in the browser)
    fc = r['fc']
    y, w, amt = np.array(fc['y']), np.array(fc['w']), np.array(fc['amt'])
    out['fc'] = {'total': round(float(amt[y == 1].sum()), 2)}
    for k in ('logit', 'mlp', 'xgb'):
        p = np.array(fc[k])
        ths = np.unique(np.quantile(p, np.linspace(0.90, 0.99995, 300)))
        rows = []
        for t in ths:
            a = p >= t
            rows.append([round(float(w[a].sum()), 2), round(float(amt[(~a) & (y == 1)].sum()), 2), int(((a) & (y == 1)).sum())])
        out['fc'][k] = rows
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
