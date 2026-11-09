"""Build the interactive page of Lecture 9 (Graph neural networks): interactive/L9.html (EN) and interactive/L9_ro.html (RO).

Inputs (interactive/L9_data.js) are computed here from the Elliptic Bitcoin graph cached by analysis_L9.py (cache/elliptic.npz)
and from the lecture's own results (results/test_scores.csv, results/receptive_field.csv, cache/depth.csv); the page logic is in
interactive/L9.js. Run: python3 lectures/L9/build_interactive_L9.py [--no-data]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, play, charts, rnd  # noqa: E402

K = 9
T_TRAIN = 34
MODELS = ['lr_local', 'lr_all', 'rf', 'mlp', 'gcn', 'sage', 'gat', 'rf_emb']
# transactions at the centre of the three real neighbourhoods drawn in the message-passing block (test step 40)
EGO_TX = [96957326, 98079154, 97732178]

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('homophily', 'homophily', 'why', 'hm'),
          ('message-passing', 'receptive_field', 'mp', 'mp'),
          ('over-smoothing', 'oversmoothing', 'mp', 'os'),
          ('pr-curves', 'pr_curves', 'case', 'pr'),
          ('drift', 'f1_by_timestep', 'case', 'fd'),
          ('alert-budget', 'alert_budget', 'case', 'ab')]

SECTIONS = {
 'en': dict(why='Graphs in the detection of financial crime', mp='Message passing',
            case='Case study: illicit Bitcoin transactions'),
 'ro': dict(why='Grafurile în detectarea infracțiunilor financiare', mp='Transmiterea mesajelor',
            case='Studiu de caz: tranzacții Bitcoin ilicite'),
}

L = {
 'en': dict(h1='Lecture 9 · Graph neural networks',
   note='Graph neural networks (GNN) learn from the neighbours of each node. The slide of the lecture is on the left, the same '
        'figure on the right, recomputed in your browser from the Elliptic Bitcoin graph and from the scores of the lecture\'s models '
        'when you move a control: the graph convolutional network (GCN), graph sample and aggregate (GraphSAGE), the graph attention '
        'network (GAT), the multilayer perceptron (MLP), logistic regression and random forest, tested for anti-money laundering (AML). '
        'The ▶ button moves the main parameter step by step. Under each figure: a readout, a question for the room and the Quantlet '
        'with the code of the chart.',
   src='Data: Elliptic Bitcoin transaction graph (Weber et al., 2019): 203,769 transactions, 234,355 payment flows, 165 features, '
       '49 time steps of about two weeks. Models trained on steps 1–34 and tested on steps 35–49 (16,670 labelled test transactions, '
       '1,083 illicit); scores averaged over the seeds of each model (own estimates, lectures/L9/analysis_L9.py).'),
 'ro': dict(h1='Cursul 9 · Rețele neuronale pe grafuri',
   note='Rețelele neuronale pe grafuri (GNN, graph neural networks) învață din vecinii fiecărui nod. În stînga este slide-ul din '
        'curs, în dreapta aceeași figură, recalculată în browser din graful Bitcoin Elliptic și din scorurile modelelor din curs cînd '
        'modificați un control: rețeaua convoluțională pe graf (GCN, graph convolutional network), eșantionarea și agregarea pe graf '
        '(GraphSAGE, graph sample and aggregate), rețeaua cu atenție pe graf (GAT, graph attention network), perceptronul multistrat '
        '(MLP, multilayer perceptron), regresia logistică și pădurea aleatoare, testate pentru combaterea spălării banilor (AML, '
        'anti-money laundering). Butonul ▶ modifică parametrul principal pas cu pas. Sub fiecare figură: o sinteză numerică, o '
        'întrebare pentru sală și Quantlet-ul cu codul figurii.',
   src='Date: graful tranzacțiilor Bitcoin Elliptic (Weber et al., 2019): 203.769 de tranzacții, 234.355 de fluxuri de plăți, 165 de '
       'variabile, 49 de pași de timp de circa două săptămîni. Modele antrenate pe pașii 1–34 și testate pe pașii 35–49 (16.670 de '
       'tranzacții de test etichetate, dintre care 1.083 ilicite); scorurile sînt mediate pe semințele fiecărui model (estimări proprii, '
       'lectures/L9/analysis_L9.py).'),
}

# frame title of the alert-budget slide written with a LaTeX thin space (1\\,000), which the shared title cleaner keeps
TITLES = {'en': {'alert-budget': 'Illicit transactions found among the top 1,000 alerts'},
          'ro': {'alert-budget': 'Tranzacțiile ilicite identificate între primele 1.000 de alerte'}}

MNAME = {
 'en': dict(lr_local='Logistic regression, local features', lr_all='Logistic regression, all features', rf='Random forest',
            mlp='MLP (no graph)', gcn='GCN', sage='GraphSAGE', gat='GAT', rf_emb='Random forest + GCN representations'),
 'ro': dict(lr_local='Regresie logistică, variabile locale', lr_all='Regresie logistică, toate variabilele', rf='Pădure aleatoare',
            mlp='MLP (fără graf)', gcn='GCN', sage='GraphSAGE', gat='GAT', rf_emb='Pădure aleatoare + reprezentări GCN'),
}


def widgets(lang):
    ro = lang == 'ro'
    mo = [(m, MNAME[lang][m]) for m in MODELS]
    ego = [(0, 'Tranzacția 96957326 (ilicită, pasul 40)' if ro else 'Transaction 96957326 (illicit, step 40)'),
           (1, 'Tranzacția 98079154 (ilicită, pasul 40)' if ro else 'Transaction 98079154 (illicit, step 40)'),
           (2, 'Tranzacția 97732178 (licită, pasul 40)' if ro else 'Transaction 97732178 (licit, step 40)')]
    return {
     'homophily': (slider('hm-s', 'Pasul de timp' if ro else 'Time step', 1, 49, 1, 29)
                   + select('hm-m', 'Măsura' if ro else 'Measure',
                            [('node', 'Ponderea vecinilor ilicite, pe clase' if ro else 'Share of illicit neighbours, by class'),
                             ('edge', 'Muchii care leagă aceeași clasă' if ro else 'Edges joining the same class')], 'node')
                   + play('hm'), charts('hm-bar', 'hm-line')),
     'message-passing': (select('mp-g', 'Vecinătatea reală (2 pași în jurul tranzacției)' if ro else 'Real neighbourhood (2 hops around the transaction)', ego, 0)
                         + select('mp-x', 'Semnalul transmis' if ro else 'Signal passed on',
                                  [('lab', 'Etichetele cunoscute (1 = ilicit, 0 = licit)' if ro else 'Known labels (1 = illicit, 0 = licit)'),
                                   ('f53', 'Variabila f53, standardizată (cea mai importantă pentru pădure)' if ro else 'Feature f53, standardised (most important for the forest)')], 'lab')
                         + slider('mp-k', 'Runde de transmitere a mesajelor, k' if ro else 'Rounds of message passing, k', 0, 6, 1, 0)
                         + play('mp'), charts('mp-graph', 'mp-hops')),
     'over-smoothing': (slider('os-L', 'Numărul de straturi GCN' if ro else 'Number of GCN layers', 1, 6, 1, 1)
                        + radios('os-s', 'Rularea' if ro else 'Run',
                                 [('mean', 'Media celor 2 semințe' if ro else 'Mean of 2 seeds'), ('1', 'Sămînța 1' if ro else 'Seed 1'),
                                  ('2', 'Sămînța 2' if ro else 'Seed 2')], 'mean')
                        + play('os'), charts('os-perf', 'os-cos')),
     'pr-curves': (toggles('pr-m', 'Modele' if ro else 'Models', mo, ['lr_all', 'rf', 'gcn', 'sage'])
                   + slider('pr-t', 'Pragul de alertă pe scor' if ro else 'Alert threshold on the score', 0.01, 0.99, 0.01, 0.5)
                   + play('pr'), charts('pr-chart', cls='one')),
     'drift': (toggles('fd-m', 'Modele' if ro else 'Models', mo, ['lr_all', 'rf', 'gcn', 'rf_emb'])
               + slider('fd-t', 'Pragul de alertă pe scor' if ro else 'Alert threshold on the score', 0.05, 0.95, 0.05, 0.5)
               + play('fd'), charts('fd-chart', cls='one')),
     'alert-budget': (slider('ab-k', 'Alerte verificate (scorurile cele mai mari)' if ro else 'Alerts reviewed (highest scores first)', 25, 3000, 25, 1000)
                      + select('ab-m', 'Modelul pentru alertele pe pași' if ro else 'Model for the alerts by step', mo, 'rf')
                      + play('ab'), charts('ab-curve', 'ab-step')),
    }


QA = {
 'en': {
  'homophily': ('Over the whole graph, 51.0% of the labelled neighbours of an illicit transaction are illicit, against 1.8% for a licit one. '
                'Is this pattern equally strong in every time step?',
                'No. Move the time step: in step 29, the peak of the training period, the share is 86.2% (276 illicit transactions with '
                'labelled neighbours), while in step 9 it is 25.4% and in a few steps it is 0. Edge homophily over the whole graph is 95.4%, '
                'against 82.4% if the labels were assigned at random. Homophily is the signal a GNN exploits, and its strength varies over '
                'time: one more reason to test in time order.'),
  'message-passing': ('Around transaction 96957326 only 3 transactions are direct neighbours. How many transactions can influence its '
                      'representation after 4 rounds of message passing?',
                      '1,259 transactions (exact count in the full graph): 3 after one round, 27 after two, 283 after three and 3,081 after six. The lecture '
                      'sample of 2,000 test transactions has a median of 2, 11, 87 and 325 nodes after 1–4 rounds. With the known labels as '
                      'signal, the degree-weighted share of illicit labels at the centre falls from 1.00 (k = 0) to 0.68 (k = 4) and 0.56 '
                      '(k = 6): the further the messages travel, the more the centre resembles the average of a large neighbourhood.'),
  'over-smoothing': ('If two GCN layers see more of the graph than one, why is the test precision-recall area (PR-AUC) lower with more layers?',
                     'Each layer averages a node with its neighbours, so the representations become alike. Averaged over 2 seeds, the trained '
                     'GCN has a PR-AUC of 0.504 with one layer, 0.458 with two and 0.371 with six, while the mean cosine similarity of its '
                     'representations rises from 0.57 to 0.96. The right panel repeats only the averaging, with no trained weights: the '
                     'cosine similarity of the 165 propagated features rises from 0.29 (raw features) to 0.56 after 3 rounds and 0.69 after '
                     '10. Seed 1 and seed 2 differ, which is why the lecture reports means.'),
  'pr-curves': ('The random forest flags far fewer transactions than the logistic regression at threshold 0.5. Which one is the better '
                'alert system?',
                'At threshold 0.5 the logistic regression on all features finds 77.1% of the illicit test transactions (recall) but only '
                '22.0% of its alerts are illicit (precision); the random forest finds 72.5% with a precision of 92.7%. The curve shows the '
                'whole trade-off: with seed-averaged scores the area under it is 0.785 for the forest, 0.499 for the GCN, 0.596 for '
                'GraphSAGE and 0.311 for the logistic regression, against a base rate of 6.5%. Move the threshold to see that the forest '
                'stays above the others at almost every recall.'),
  'drift': ('Before step 43 the random forest has a mean F1 (harmonic mean of precision and recall) of 0.898 on the illicit class. Would '
            'a lower alert threshold rescue it after the dark-market shutdown?',
            'No. From step 43 the mean F1 of the forest is 0.048 at threshold 0.5 and 0.015 at threshold 0.2; the GCN goes from 0.473 '
            'to 0.000 at 0.5. Lowering the threshold adds false alarms but does not find the new kind of illicit flows, which no model '
            'trained on steps 1–34 had seen. Only the logistic regression keeps a little recall (mean F1 0.151 after step 43 at threshold '
            '0.2), at the price of many false alerts. Drift is fixed by retraining on recent labels, not by moving a threshold.'),
  'alert-budget': ('An investigation team can review 1,000 alerts over the test period. How many of the 1,083 illicit transactions does '
                   'it find, and where are they?',
                   'With the random forest it finds 72.9% of them (789 of the 1,000 alerts are illicit, a hit rate of 78.9%); with the GCN '
                   '44.8% (hit rate 48.5%); with the logistic regression on all features 36.6%. Choose the forest and look at the right '
                   'panel: almost all hits fall in steps 35–42, before the shutdown; the illicit transactions of steps 43–49 are barely '
                   'found at any budget. The alert budget, not the F1, sizes a compliance team.'),
 },
 'ro': {
  'homophily': ('Pe întregul graf, 51,0% dintre vecinii etichetați ai unei tranzacții ilicite sînt ilicite, față de 1,8% pentru o tranzacție '
                'licită. Este acest tipar la fel de puternic în fiecare pas de timp?',
                'Nu. Modificați pasul de timp: în pasul 29, vîrful perioadei de antrenare, ponderea este 86,2% (276 de tranzacții ilicite '
                'cu vecini etichetați), în pasul 9 este 25,4%, iar în cîțiva pași este 0. Homofilia muchiilor pe întregul graf este 95,4%, '
                'față de 82,4% dacă etichetele ar fi atribuite aleator. Homofilia este semnalul pe care îl exploatează o rețea GNN, iar '
                'intensitatea ei variază în timp: încă un motiv pentru a testa în ordine cronologică.'),
  'message-passing': ('În jurul tranzacției 96957326 există doar 3 vecini direcți. Cîte tranzacții pot influența reprezentarea ei după '
                      '4 runde de transmitere a mesajelor?',
                      '1.259 de tranzacții (numărare exactă în graful complet): 3 după o rundă, 27 după două, 283 după trei și 3.081 după șase. Eșantionul '
                      'din curs, de 2.000 de tranzacții de test, are o mediană de 2, 11, 87 și 325 de noduri după 1–4 runde. Cu etichetele '
                      'cunoscute ca semnal, ponderea etichetelor ilicite în centru, ponderată după grad, scade de la 1,00 (k = 0) la 0,68 '
                      '(k = 4) și 0,56 (k = 6): cu cît mesajele ajung mai departe, cu atît centrul seamănă mai mult cu media unei vecinătăți mari.'),
  'over-smoothing': ('Dacă două straturi GCN văd mai mult din graf decît unul, de ce aria de sub curba precizie-recall (PR-AUC) de test '
                     'scade cu mai multe straturi?',
                     'Fiecare strat face media unui nod cu vecinii săi, așa că reprezentările ajung să semene. În medie pe 2 semințe, rețeaua '
                     'GCN antrenată are PR-AUC 0,504 cu un strat, 0,458 cu două și 0,371 cu șase, iar similaritatea cosinus medie a '
                     'reprezentărilor crește de la 0,57 la 0,96. Panoul din dreapta repetă doar medierea, fără ponderi antrenate: '
                     'similaritatea cosinus a celor 165 de variabile propagate crește de la 0,29 (variabilele brute) la 0,56 după 3 runde și '
                     '0,69 după 10. Sămînța 1 și sămînța 2 diferă, de aceea cursul raportează medii.'),
  'pr-curves': ('La pragul 0,5, pădurea aleatoare semnalează mult mai puține tranzacții decît regresia logistică. Care dintre ele este '
                'un sistem de alertă mai bun?',
                'La pragul 0,5, regresia logistică pe toate variabilele găsește 77,1% dintre tranzacțiile ilicite de test (recall), dar '
                'numai 22,0% dintre alertele ei sînt ilicite (precizie); pădurea aleatoare găsește 72,5%, cu o precizie de 92,7%. Curba '
                'arată întregul compromis: cu scorurile mediate pe semințe, aria de sub ea este 0,785 pentru pădure, 0,499 pentru GCN, '
                '0,596 pentru GraphSAGE și 0,311 pentru regresia logistică, față de o rată de bază de 6,5%. Modificați pragul: pădurea '
                'rămîne deasupra celorlalte modele la aproape orice recall.'),
  'drift': ('Înainte de pasul 43, pădurea aleatoare are un F1 mediu (media armonică a preciziei și a recall-ului) de 0,898 pe clasa '
            'ilicită. Ar salva-o un prag de alertă mai mic după închiderea pieței ilegale?',
            'Nu. Din pasul 43, F1 mediu al pădurii este 0,048 la pragul 0,5 și 0,015 la pragul 0,2; rețeaua GCN scade de la 0,473 la '
            '0,000 la pragul 0,5. Un prag mai mic adaugă alarme false, dar nu găsește noul tip de fluxuri ilicite, pe care niciun model '
            'antrenat pe pașii 1–34 nu le văzuse. Doar regresia logistică păstrează puțin recall (F1 mediu 0,151 după pasul 43 la pragul '
            '0,2), cu prețul multor alerte false. Schimbarea conceptului se corectează prin reantrenare pe etichete recente, nu prin mutarea pragului.'),
  'alert-budget': ('O echipă de investigații poate verifica 1.000 de alerte în perioada de test. Cîte dintre cele 1.083 de tranzacții '
                   'ilicite găsește și unde se află ele?',
                   'Cu pădurea aleatoare găsește 72,9% dintre ele (789 dintre cele 1.000 de alerte sînt ilicite, o rată de succes de 78,9%); '
                   'cu GCN, 44,8% (rată de succes 48,5%); cu regresia logistică pe toate variabilele, 36,6%. Alegeți pădurea și priviți '
                   'panoul din dreapta: aproape toate tranzacțiile găsite sînt în pașii 35–42, înainte de închiderea pieței; tranzacțiile '
                   'ilicite din pașii 43–49 sînt rareori găsite, oricare ar fi bugetul. Bugetul de alerte, nu F1, dimensionează o echipă de conformitate.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 models: %(men)s,
 step: 'Time step', share: 'Share', train: 'training', test: 'test',
 hmIll: 'Illicit transactions', hmLic: 'Licit transactions', hmBase: 'Share of illicit among labelled (base rate)',
 hmEdge: 'Edges joining the same class', hmRand: 'Expected with random labels',
 hmBarT: s => `Time step ${s}`, hmLineT: m => m === 'node' ? 'Share of illicit among labelled neighbours, by step' : 'Edge homophily, by step',
 hmCat: ['Neighbours of illicit', 'Neighbours of licit', 'Base rate'], hmCatE: ['Same-class edges', 'Random labels'],
 hm: (s, a, b, p, ni, nl) => `Step ${s}: ${a} of the labelled neighbours of illicit transactions are illicit (${window.NNI.cnt(ni, 'transaction', 'transactions')}), ${b} for licit ones (${window.NNI.cnt(nl, 'transaction', 'transactions')}); base rate ${p}.`,
 hmE: (s, e, r) => `Step ${s}: ${e} of the labelled edges join two transactions of the same class; with random labels ${r}.`,
 hmAll: (a, b, e, r) => `Whole graph: ${a} against ${b}; edge homophily ${e} against ${r} with random labels.`,
 hmNone: s => `Step ${s}: no illicit transaction with a labelled neighbour.`,
 mpT: (k, x) => `After k = ${k} rounds: ${x === 'lab' ? 'share of illicit labels' : 'propagated value of f53'}`,
 cls: { '1': 'illicit', '0': 'licit', '-1': 'unknown' }, centre: 'centre', hop: h => h === 0 ? 'centre' : `${h} hop${h > 1 ? 's' : ''}`,
 nolab: 'no labelled node reached yet',
 mpHopsT: 'Nodes that can reach a node after k rounds', kk: 'Rounds k', nodes: 'Nodes (log scale)',
 thisTx: 'This transaction (full graph)', med: 'Median, 2,000 test transactions', p90: '90th percentile',
 mp: (k, n, c, sd) => `k = ${k}: ${window.NNI.cnt(n, 'transaction', 'transactions')} of the full graph can influence the centre; value at the centre ${c}; spread of the values drawn (standard deviation) ${sd}.`,
 mp0: sd => `k = 0: no message yet, each transaction keeps its own value; spread of the values drawn (standard deviation) ${sd}.`,
 mp2: 'From k = 3 on, transactions outside the picture also send messages: each round multiplies by the normalised adjacency Â = D̃^−1/2 (A + I) D̃^−1/2 of the whole graph.',
 osPerfT: 'Trained GCN on the test steps', osCosT: 'Mean cosine similarity of node representations', layers: 'Layers / rounds',
 ap: 'PR-AUC', f1: 'F1, illicit class', cosG: 'Trained GCN', cosP: 'Averaging only (Â^L X, no weights)',
 os: (L, ap, f1, c) => `${L} layer${L > 1 ? 's' : ''}: PR-AUC ${ap}, F1 ${f1}, cosine similarity of the representations ${c}.`,
 os2: (L, c0, c) => `Averaging only: cosine similarity ${c0} for the raw features, ${c} after ${L} round${L > 1 ? 's' : ''}.`,
 recall: 'Recall (share of illicit found)', precision: 'Precision (share of alerts that are illicit)', base: 'Base rate',
 prT: t => `Precision-recall curves; dots: threshold ${t}`,
 pr: (m, t, p, r, f, ap, a) => `${m}, threshold ${t}: precision ${p}, recall ${r}, F1 ${f} (${window.NNI.cnt(a, 'alert', 'alerts')}); area under the curve ${ap}.`,
 fdT: t => `F1 of the illicit class per test step, threshold ${t}`, shutdown: 'dark-market shutdown',
 fd: (m, a, b) => `${m}: mean F1 ${a} in steps 35–42, ${b} in steps 43–49.`,
 nIll: 'Illicit transactions in the step',
 abT: 'Share of the illicit test transactions found', alerts: 'Alerts reviewed', abStepT: (m, k) => `${m}, top ${window.NNI.cnt(k, 'alert', 'alerts')}: illicit found per step`,
 found: 'Found', missed: 'Not found',
 ab: (m, k, r, p, n) => `${m}, ${window.NNI.cnt(k, 'alert', 'alerts')}: ${r} of the 1,083 illicit transactions found; ${p} of the alerts are illicit (${window.NNI.cnt(n, 'hit', 'hits')}).`,
 ab2: (a, b) => `Found in steps 35–42: ${a}; in steps 43–49: ${b}.`
};
""",
 'ro': r"""window.LK_TEXT = {
 models: %(mro)s,
 step: 'Pasul de timp', share: 'Pondere', train: 'antrenare', test: 'test',
 hmIll: 'Tranzacții ilicite', hmLic: 'Tranzacții licite', hmBase: 'Ponderea ilicitelor între cele etichetate (rata de bază)',
 hmEdge: 'Muchii care leagă aceeași clasă', hmRand: 'Așteptat cu etichete aleatoare',
 hmBarT: s => `Pasul de timp ${s}`, hmLineT: m => m === 'node' ? 'Ponderea ilicitelor între vecinii etichetați, pe pași' : 'Homofilia muchiilor, pe pași',
 hmCat: ['Vecinii ilicitelor', 'Vecinii licitelor', 'Rata de bază'], hmCatE: ['Muchii în aceeași clasă', 'Etichete aleatoare'],
 hm: (s, a, b, p, ni, nl) => `Pasul ${s}: ${a} dintre vecinii etichetați ai tranzacțiilor ilicite sînt ilicite (${window.NNI.cnt(ni, 'tranzacție', 'tranzacții')}), ${b} pentru cele licite (${window.NNI.cnt(nl, 'tranzacție', 'tranzacții')}); rata de bază ${p}.`,
 hmE: (s, e, r) => `Pasul ${s}: ${e} dintre muchiile etichetate leagă două tranzacții din aceeași clasă; cu etichete aleatoare ${r}.`,
 hmAll: (a, b, e, r) => `Întregul graf: ${a} față de ${b}; homofilia muchiilor ${e} față de ${r} cu etichete aleatoare.`,
 hmNone: s => `Pasul ${s}: nicio tranzacție ilicită cu vecini etichetați.`,
 mpT: (k, x) => `După k = ${k} runde: ${x === 'lab' ? 'ponderea etichetelor ilicite' : 'valoarea propagată a lui f53'}`,
 cls: { '1': 'ilicită', '0': 'licită', '-1': 'necunoscută' }, centre: 'centrul', hop: h => h === 0 ? 'centrul' : (h === 1 ? 'la 1 pas' : `la ${h} pași`),
 nolab: 'încă niciun nod etichetat atins',
 mpHopsT: 'Noduri care pot ajunge la un nod după k runde', kk: 'Runde k', nodes: 'Noduri (scară logaritmică)',
 thisTx: 'Această tranzacție (graful complet)', med: 'Mediana, 2.000 de tranzacții de test', p90: 'Percentila 90',
 mp: (k, n, c, sd) => `k = ${k}: ${window.NNI.cnt(n, 'tranzacție', 'tranzacții')} din graful complet pot influența centrul; valoarea în centru ${c}; dispersia valorilor desenate (abaterea standard) ${sd}.`,
 mp0: sd => `k = 0: încă niciun mesaj, fiecare tranzacție își păstrează valoarea; dispersia valorilor desenate (abaterea standard) ${sd}.`,
 mp2: 'De la k = 3, trimit mesaje și tranzacții din afara imaginii: fiecare rundă înmulțește cu adiacența normalizată Â = D̃^−1/2 (A + I) D̃^−1/2 a întregului graf.',
 osPerfT: 'Rețeaua GCN antrenată, pe pașii de test', osCosT: 'Similaritatea cosinus medie a reprezentărilor nodurilor', layers: 'Straturi / runde',
 ap: 'PR-AUC', f1: 'F1, clasa ilicită', cosG: 'GCN antrenată', cosP: 'Doar mediere (Â^L X, fără ponderi)',
 os: (L, ap, f1, c) => `${L === 1 ? '1 strat' : L + ' straturi'}: PR-AUC ${ap}, F1 ${f1}, similaritatea cosinus a reprezentărilor ${c}.`,
 os2: (L, c0, c) => `Doar mediere: similaritatea cosinus ${c0} pentru variabilele brute, ${c} după ${L === 1 ? 'o rundă' : L + ' runde'}.`,
 recall: 'Recall (ponderea ilicitelor găsite)', precision: 'Precizia (ponderea alertelor ilicite)', base: 'Rata de bază',
 prT: t => `Curbele precizie-recall; punctele: pragul ${t}`,
 pr: (m, t, p, r, f, ap, a) => `${m}, pragul ${t}: precizia ${p}, recall ${r}, F1 ${f} (${window.NNI.cnt(a, 'alertă', 'alerte')}); aria de sub curbă ${ap}.`,
 fdT: t => `F1 pe clasa ilicită, pe pașii de test, pragul ${t}`, shutdown: 'închiderea pieței ilegale',
 fd: (m, a, b) => `${m}: F1 mediu ${a} în pașii 35–42, ${b} în pașii 43–49.`,
 nIll: 'Tranzacții ilicite în pas',
 abT: 'Ponderea tranzacțiilor ilicite de test găsite', alerts: 'Alerte verificate', abStepT: (m, k) => `${m}, primele ${window.NNI.cnt(k, 'alertă', 'alerte')}: ilicite găsite pe pași`,
 found: 'Găsite', missed: 'Negăsite',
 ab: (m, k, r, p, n) => `${m}, ${window.NNI.cnt(k, 'alertă', 'alerte')}: ${r} dintre cele 1.083 de tranzacții ilicite găsite; ${p} dintre alerte sînt ilicite (${window.NNI.cnt(n, 'tranzacție găsită', 'tranzacții găsite')}).`,
 ab2: (a, b) => `Găsite în pașii 35–42: ${a}; în pașii 43–49: ${b}.`
};
""",
}
import json as _json  # noqa: E402
TEXT_JS = {lg: s % dict(men=_json.dumps(MNAME['en'], ensure_ascii=False), mro=_json.dumps(MNAME['ro'], ensure_ascii=False))
           for lg, s in TEXT_JS.items()}


def data():
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import networkx as nx
    z = np.load(os.path.join(HERE, 'cache', 'elliptic.npz'))
    E, cls, step, X, txid = z['E'], z['cls'], z['step'], z['X'], z['txid']
    n = len(cls)
    A = sp.coo_matrix((np.ones(E.shape[1]), (E[0], E[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64).tocsr()                     # undirected, no self-loops
    At = A + sp.eye(n, format='csr')
    Dm = sp.diags(1 / np.sqrt(np.asarray(At.sum(1)).ravel()))
    Ah = (Dm @ At @ Dm).tocsr()                                        # normalised adjacency with self-loops (GCN)
    out = {}

    # 1. homophily by time step (same definitions as analysis_L9.py)
    lab = cls >= 0
    nb_ill, nb_lab = A @ (cls == 1).astype(float), A @ lab.astype(float)
    has = nb_lab > 0
    e0, e1 = E
    ee = (cls[e0] >= 0) & (cls[e1] >= 0)

    def H(mask):
        mi, ml = (cls == 1) & has & mask, (cls == 0) & has & mask
        p1 = ((cls == 1) & mask).sum() / max((lab & mask).sum(), 1)
        em = ee & mask[e0]
        return [float(np.mean(nb_ill[mi] / nb_lab[mi])) if mi.sum() else None,
                float(np.mean(nb_ill[ml] / nb_lab[ml])) if ml.sum() else None, float(p1),
                float(np.mean(cls[e0][em] == cls[e1][em])) if em.sum() else None, float(p1 ** 2 + (1 - p1) ** 2),
                int(mi.sum()), int(ml.sum())]
    keys = ['h_ill', 'h_lic', 'base', 'edge', 'edge_rand', 'n_ill', 'n_lic']
    hm = {k: [] for k in keys}
    for s in range(1, 50):
        for k, v in zip(keys, H(step == s)):
            hm[k].append(v)
    hm['all'] = dict(zip(keys, H(np.ones(n, bool))))
    out['hm'] = rnd(hm, 4)

    # 2. message passing on three real 2-hop neighbourhoods; values propagated on the full graph
    mu, sd = X[step <= T_TRAIN].mean(0), X[step <= T_TRAIN].std(0) + 1e-6
    Xs = np.clip((X - mu) / sd, -10, 10).astype(np.float64)
    KMAX = 6
    a, b, f = (cls == 1).astype(float), lab.astype(float), Xs[:, 52].copy()      # f53 = column 53 (1-based)
    LAB, MASS, F53 = [a], [b], [f]
    for _ in range(KMAX):
        a, b, f = Ah @ a, Ah @ b, Ah @ f
        LAB.append(a); MASS.append(b); F53.append(f)
    pos_of = {int(t): i for i, t in enumerate(txid)}
    egos = []
    for tx in EGO_TX:
        v = pos_of[tx]
        n1 = set(A[v].indices)
        n2 = set(A[list(n1)].indices)
        S = [v] + sorted(n1) + sorted(n2 - n1 - {v})
        sub = A[S][:, S].tocoo()
        edges = [(int(i), int(j)) for i, j in zip(sub.row, sub.col) if i < j]
        G = nx.Graph()
        G.add_nodes_from(range(len(S)))
        G.add_edges_from(edges)
        pos = nx.kamada_kawai_layout(G)
        seen, fr, cnts = {v}, {v}, []
        for _ in range(KMAX):
            nxt = set(A[list(fr)].indices) - seen
            seen |= nxt
            fr = nxt
            cnts.append(len(seen) - 1)
        egos.append(dict(tx=tx, step=int(step[v]), cls=[int(cls[u]) for u in S],
                         hop=[0] + [1] * len(n1) + [2] * (len(S) - 1 - len(n1)), deg=[int(A[u].nnz) for u in S],
                         x=rnd([float(pos[i][0]) for i in range(len(S))], 3), y=rnd([float(pos[i][1]) for i in range(len(S))], 3),
                         e=edges, khop=cnts,
                         lab=rnd([[(LAB[k][u] / MASS[k][u]) if MASS[k][u] > 1e-12 else None for u in S] for k in range(KMAX + 1)], 3),
                         f53=rnd([[float(F53[k][u]) for u in S] for k in range(KMAX + 1)], 3)))
    out['mp'] = egos
    rf = pd.read_csv(os.path.join(HERE, 'results', 'receptive_field.csv'))
    out['rfield'] = rnd({c: rf[c].tolist() for c in rf.columns}, 1)

    # 3. over-smoothing: trained GCN by depth (cache/depth.csv, 2 seeds) and parameter-free averaging of the 165 features
    dp = pd.read_csv(os.path.join(HERE, 'cache', 'depth.csv'))
    os_ = {str(s): {c: dp[dp.seed == s].sort_values('layers')[c].tolist() for c in ['ap', 'f1', 'cos']} for s in [1, 2]}
    test_m = lab & (step > T_TRAIN)
    ii = np.random.default_rng(1).choice(int(test_m.sum()), (5000, 2))
    Hm, cosP = Xs.copy(), []
    for Lk in range(0, 11):
        if Lk:
            Hm = Ah @ Hm
        e_ = Hm[test_m]
        e_ = e_ / (np.linalg.norm(e_, axis=1, keepdims=True) + 1e-9)
        cosP.append(float(np.mean(np.sum(e_[ii[:, 0]] * e_[ii[:, 1]], 1))))
    os_['prop'] = cosP
    out['os'] = rnd(os_, 4)

    # 4.-6. seed-averaged test scores of the eight models (results/test_scores.csv)
    from sklearn.metrics import average_precision_score
    ts = pd.read_csv(os.path.join(HERE, 'results', 'test_scores.csv'))
    yy, st = ts.y.values, ts.step.values
    steps = list(range(T_TRAIN + 1, 50))
    thr = np.round(np.arange(0.01, 0.995, 0.01), 2)
    pr = {'thr': thr.tolist(), 'npos': int(yy.sum()), 'n': int(len(yy)), 'm': {}}
    for m in MODELS:
        s = ts[m].values
        o = np.argsort(-s, kind='stable')
        ss, cy = s[o], np.cumsum(yy[o])
        cut = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
        P_, R_ = cy[cut] / (cut + 1), cy[cut] / yy.sum()
        if len(cut) > 300:
            j = np.unique(np.linspace(0, len(cut) - 1, 300).astype(int))
            P_, R_ = P_[j], R_[j]
        pr['m'][m] = dict(tp=[int(((s >= t) & (yy == 1)).sum()) for t in thr], fp=[int(((s >= t) & (yy == 0)).sum()) for t in thr],
                          P=rnd(P_, 4), R=rnd(R_, 4), ap=round(float(average_precision_score(yy, s)), 4))
    out['pr'] = pr
    th2 = np.round(np.arange(0.05, 0.951, 0.05), 2)
    f1 = {'thr': th2.tolist(), 'steps': steps, 'npos': [int(((st == s_) & (yy == 1)).sum()) for s_ in steps], 'm': {}}
    for m in MODELS:
        sc = ts[m].values
        f1['m'][m] = {'tp': [[int(((st == s_) & (sc >= t) & (yy == 1)).sum()) for s_ in steps] for t in th2],
                      'fp': [[int(((st == s_) & (sc >= t) & (yy == 0)).sum()) for s_ in steps] for t in th2]}
    out['f1'] = f1
    kk = list(range(25, 3001, 25))
    ab = {'k': kk, 'm': {}}
    for m in MODELS:
        o = np.argsort(-ts[m].values, kind='stable')
        cy = np.cumsum(yy[o])
        ab['m'][m] = dict(tp=[int(cy[k - 1]) for k in kk],
                          st=[[int(((st[o[:k]] == s_) & (yy[o[:k]] == 1)).sum()) for s_ in steps] for k in kk])
    out['ab'] = ab
    return out


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    interactive_common.build(sys.modules[__name__])
