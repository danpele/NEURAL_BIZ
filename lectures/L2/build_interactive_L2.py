"""Build the interactive page of Lecture 2 (training: backpropagation and optimisation): interactive/L2.html (EN) and
interactive/L2_ro.html (RO).

Inputs (interactive/L2_data.js) come from data/clean (California housing) and the caches of analysis_L2.py through
lectures/L2/interactive_data_L2.py (same splits, models and seeds); the page logic is in interactive/L2.js.
Run: python3 lectures/L2/build_interactive_L2.py [--no-data]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, play, charts, rnd  # noqa: E402

K = 2
BLOCKS = [('gd-paths', 'gd_paths', 'gd', 'gd'),
          ('lr-sweep', 'lr_sweep', 'gd', 'lr'),
          ('optimizers', 'optimizers', 'opt', 'op'),
          ('batch-size', 'batch_size', 'opt', 'bs'),
          ('vanishing', 'vanishing', 'deep', 'vg')]

SECTIONS = {
 'en': dict(gd='Gradient descent and the learning rate', opt='Optimisers and mini-batches', deep='Deep networks: vanishing gradients'),
 'ro': dict(gd='Coborîrea pe gradient și rata de învățare', opt='Optimizatori și mini-loturi', deep='Rețele adînci: gradienți care se sting'),
}

L = {
 'en': dict(h1='Lecture 2 · Training: backpropagation and optimisation',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the course data '
        'when you move a control. The ▶ button moves the main parameter step by step. Under each figure: a readout, a question '
        'for the room and the Quantlet with the code of the chart. The first block runs gradient descent live in your browser on '
        'a real regression; the others replay the training runs of the lecture.',
   src='Data: California housing (StatLib, 1990 census; 20,640 block groups), split 60/20/20 with seed 20261108 as in the lecture '
       '(analysis_L2.py): 12,384 training, 4,128 validation and 4,128 test block groups. Networks: multilayer perceptron (MLP) with '
       'two hidden layers of 64 ReLU (rectified linear unit) units, mean squared error (MSE) on the median house value in 100,000 USD; '
       'SGD = stochastic gradient descent, RMSProp = root mean square propagation, Adam = adaptive moment estimation.'),
 'ro': dict(h1='Cursul 2 · Antrenarea rețelelor: propagarea înapoi a erorii și optimizarea',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser din datele cursului cînd modificați '
        'un control. Butonul ▶ modifică parametrul principal pas cu pas. Sub fiecare figură: o sinteză numerică, o întrebare '
        'pentru sală și Quantlet-ul cu codul figurii. Primul bloc rulează coborîrea pe gradient chiar în browser, pe o regresie '
        'reală; celelalte reiau antrenările din curs.',
   src='Date: locuințele din California (StatLib, recensămîntul din 1990; 20.640 de grupuri de locuințe), împărțite 60/20/20 cu '
       'sămînța 20261108, ca în curs (analysis_L2.py): 12.384 de grupuri de antrenare, 4.128 de validare și 4.128 de test. Rețele: '
       'perceptron multistrat (MLP) cu două straturi ascunse de cîte 64 de neuroni ReLU (unitate liniară rectificată), eroarea '
       'pătratică medie (MSE) pentru valoarea mediană a locuințelor în 100.000 USD; SGD = coborîre stocastică pe gradient, '
       'RMSProp = propagarea rădăcinii mediei pătratice, Adam = estimarea adaptivă a momentelor.'),
}

OPTS = {'sgd': [0.01, 0.03, 0.1], 'momentum': [0.001, 0.003, 0.01], 'rmsprop': [3e-4, 1e-3, 3e-3], 'adam': [3e-4, 1e-3, 3e-3],
        'adamw': [3e-4, 1e-3, 3e-3]}


def widgets(lang):
    ro = lang == 'ro'
    onames = [('sgd', 'SGD'), ('momentum', 'SGD + moment' if ro else 'SGD + momentum'), ('rmsprop', 'RMSProp'), ('adam', 'Adam'),
              ('adamw', 'AdamW')]
    return {
     'gd-paths': (radios('gd-x', 'Venitul median' if ro else 'Median income', [('raw', 'brut, în 10.000 USD' if ro else 'raw, in 10,000 USD'),
                                                                              ('std', 'standardizat' if ro else 'standardised')], 'raw')
                  + slider('gd-e', 'Pasul η, ca multiplu al pasului sigur 1/λmax' if ro else 'Step η, as a multiple of the safe step 1/λmax', 0.1, 2.2, 0.05, 1)
                  + slider('gd-m', 'Momentul β (0 = coborîre simplă)' if ro else 'Momentum β (0 = plain gradient descent)', 0, 0.95, 0.05, 0)
                  + slider('gd-k', 'Numărul de pași' if ro else 'Number of steps', 1, 200, 1, 60) + play('gd'),
                  charts('gd-path', 'gd-loss')),
     'lr-sweep': (slider('lr-i', 'Rata de învățare η (SGD)' if ro else 'Learning rate η (SGD)', 0, 8, 1, 6)
                  + toggles('lr-s', 'Afișare' if ro else 'Show', [('all', 'Curbele celorlalte rate (gri)' if ro else 'Curves of the other rates (grey)')], ['all'])
                  + play('lr'), charts('lr-curve', 'lr-all')),
     'optimizers': (select('op-o', 'Optimizatorul' if ro else 'Optimiser', onames, 'adam')
                    + slider('op-l', 'Rata de învățare (grila de reglaj a cursului)' if ro else 'Learning rate (tuning grid of the lecture)', 0, 2, 1, 2)
                    + toggles('op-s', 'Afișare' if ro else 'Show', [('seeds', 'Fiecare inițializare' if ro else 'Each seed'),
                                                                    ('best', 'Ceilalți optimizatori la rata lor cea mai bună' if ro else 'Other optimisers at their best rate')], ['seeds'])
                    + play('op'), charts('op-chart', cls='one')),
     'batch-size': (radios('bs-x', 'Axa orizontală' if ro else 'Horizontal axis', [('ep', 'Epoci (treceri prin date)' if ro else 'Epochs (passes over the data)'),
                                                                                   ('st', 'Pași de gradient' if ro else 'Gradient steps'),
                                                                                   ('sec', 'Secunde de calcul' if ro else 'Seconds of computing')], 'ep')
                    + toggles('bs-b', 'Mărimea lotului' if ro else 'Batch size', [('16', '16'), ('128', '128'), ('1024', '1024')], ['16', '128', '1024']),
                    charts('bs-chart', cls='one')),
     'vanishing': (select('vg-a', 'Funcția de activare' if ro else 'Activation function', [('sig', 'sigmoidă' if ro else 'sigmoid'), ('tanh', 'tanh'), ('relu', 'ReLU')], 'sig')
                   + select('vg-i', 'Inițializarea ponderilor' if ro else 'Weight initialisation',
                            [('xavier', 'Xavier (Glorot)'), ('he', 'He (Kaiming)'), ('small', 'N(0; 0,01²)' if ro else 'N(0, 0.01²)'),
                             ('large', 'N(0; 1)' if ro else 'N(0, 1)')], 'xavier')
                   + slider('vg-d', 'Numărul de straturi ascunse' if ro else 'Number of hidden layers', 0, 7, 1, 4) + play('vg'),
                   charts('vg-chart', cls='one')),
    }


QA = {
 'en': {
  'gd-paths': ('On the raw income scale the loss surface is a long narrow valley. Why does gradient descent crawl along it, and what '
               'does standardising the income change?',
               'The two directions of the valley have curvatures λmax = 38.8 and λmin = 0.37 (condition number 104): a step must stay below '
               '2/λmax to avoid divergence (try 2.05 × 1/λmax), and at that step the flat direction is covered 104 times more slowly. After '
               '60 steps the MSE is still 1.486, against a minimum of 0.701. Momentum (β = 0.9 with half the step) accumulates the '
               'consistent part of the gradient and reaches 0.704. Standardising makes the valley round (condition number 1.0): one '
               'step of 1/λmax lands on the minimum.'),
  'lr-sweep': ('Plain SGD (stochastic gradient descent) for 30 epochs: which learning rate gives the lowest validation MSE, and '
               'what happens just above it?',
               'η = 0.1 gives the lowest validation MSE, 0.281 (the linear regression reaches 0.500). One step higher, η = 0.3, two '
               'of the three seeds diverge, and at η = 1 all three do. Too small a rate is slow rather than wrong: at η = 0.0001 the '
               'MSE is still 1.005 to 1.142 after 30 epochs. The best rate sits just below the edge of divergence, which is why '
               'learning-rate range tests sweep η on a log scale.'),
  'optimizers': ('Each optimiser was tuned on a grid of three learning rates (3 seeds, 60 epochs). Does the choice of optimiser matter '
                 'as much as the choice of its learning rate?',
                 'Within one optimiser, the learning rate matters more: Adam ends at 0.288 with η = 0.0003 and 0.259 with η = 0.003; '
                 'SGD ends at 0.307 with η = 0.01 and 0.274 with η = 0.1. At their best rates the five optimisers end close to each '
                 'other (validation MSE 0.259 to 0.274) and all beat the linear regression (0.500); Adam and AdamW get there faster '
                 '(0.299 after 10 epochs, against 0.350 for SGD) and with less spread across seeds. On the test set, Adam is not '
                 'significantly better than SGD (paired t = 1.01, p = 0.37, 5 seeds).'),
  'batch-size': ('Batch 16, 128 or 1,024 with the same learning rate (Adam, 0.001): which one learns fastest per epoch, per gradient '
                 'step and per second?',
                 'Per epoch, batch 16 wins (validation MSE 0.294 after 5 epochs, against 0.367 for 128 and 0.870 for 1,024), because it '
                 'takes 774 steps per epoch instead of 97 or 13. Per gradient step the large batch is at least as good: each of its '
                 'steps averages more examples. Per second the picture changes again: the 40 epochs took 25.8 s with batch 16, 3.7 s with '
                 '128 and 1.0 s with 1,024 on this laptop. With the same learning rate, the large batch has not converged after 40 epochs '
                 '(0.335); larger batches usually need a larger learning rate.'),
  'vanishing': ('Ten hidden layers of 64 sigmoid units with Xavier initialisation: how much smaller is the gradient of the first '
                'layer than that of the last one?',
                'About 10⁷ times smaller (8.5 × 10⁻⁷ against 7.5; mean of 5 seeds on one batch of 512 block groups): each sigmoid '
                'layer multiplies the backward signal by at most 0.25, so the first layers barely learn. With tanh and Xavier the ratio '
                'stays near 0.6, and with ReLU and He initialisation the gradient stays of the same order in every layer. Small weights '
                'N(0, 0.01²) shrink the signal in both directions (gradients of 10⁻¹² at depth 10), large weights N(0, 1) make it '
                'explode (above 10¹⁶ at depth 10, overflow beyond). Move the depth: the problem grows geometrically with the number of layers.'),
 },
 'ro': {
  'gd-paths': ('Pe scara brută a venitului, suprafața pierderii este o vale lungă și îngustă. De ce avansează coborîrea pe gradient atît '
               'de încet de-a lungul ei și ce schimbă standardizarea venitului?',
               'Cele două direcții ale văii au curburile λmax = 38,8 și λmin = 0,37 (numărul de condiționare 104): pasul trebuie să '
               'rămînă sub 2/λmax ca să nu diveargă (încercați 2,05 × 1/λmax), iar la acest pas direcția plată este parcursă de 104 ori '
               'mai încet. După 60 de pași, MSE este încă 1,486, față de un minim de 0,701. Momentul (β = 0,9 cu jumătate din pas) '
               'acumulează partea constantă a gradientului și ajunge la 0,704. Standardizarea face valea rotundă (numărul de '
               'condiționare 1,0): un singur pas de 1/λmax ajunge în minim.'),
  'lr-sweep': ('SGD simplu (coborîre stocastică pe gradient), 30 de epoci: ce rată de învățare dă cea mai mică MSE pe validare și ce se '
               'întîmplă imediat peste ea?',
               'η = 0,1 dă cea mai mică MSE pe validare, 0,281 (regresia liniară ajunge la 0,500). Un pas mai sus, la η = 0,3, două '
               'din cele trei inițializări diverg, iar la η = 1 diverg toate trei. O rată prea mică este lentă, nu greșită: la '
               'η = 0,0001, MSE este încă între 1,005 și 1,142 după 30 de epoci. Rata cea mai bună stă chiar sub limita de divergență; '
               'de aceea testele de rată parcurg η pe o scară logaritmică.'),
  'optimizers': ('Fiecare optimizator a fost reglat pe o grilă de trei rate de învățare (3 inițializări, 60 de epoci). Contează alegerea '
                 'optimizatorului la fel de mult ca alegerea ratei sale de învățare?',
                 'În cadrul aceluiași optimizator, rata contează mai mult: Adam termină la 0,288 cu η = 0,0003 și la 0,259 cu η = 0,003; '
                 'SGD termină la 0,307 cu η = 0,01 și la 0,274 cu η = 0,1. La ratele lor cele mai bune, cei cinci optimizatori termină '
                 'aproape unul de altul (MSE pe validare între 0,259 și 0,274) și toți depășesc regresia liniară (0,500); Adam și AdamW '
                 'ajung acolo mai repede (0,299 după 10 epoci, față de 0,350 pentru SGD) și cu o dispersie mai mică între inițializări. '
                 'Pe setul de test, Adam nu este semnificativ mai bun decît SGD (t pereche = 1,01, p = 0,37, 5 inițializări).'),
  'batch-size': ('Loturi de 16, 128 sau 1.024 de exemple, cu aceeași rată de învățare (Adam, 0,001): care învață cel mai repede pe '
                 'epocă, pe pas de gradient și pe secundă?',
                 'Pe epocă cîștigă lotul de 16 (MSE pe validare 0,294 după 5 epoci, față de 0,367 pentru 128 și 0,870 pentru 1.024), '
                 'pentru că face 774 de pași pe epocă în loc de 97 sau 13. Pe pas de gradient, lotul mare este cel puțin la fel de bun: '
                 'fiecare pas al său mediază mai multe exemple. Pe secundă, imaginea se schimbă din nou: cele 40 de epoci au durat '
                 '25,8 s cu lotul de 16, 3,7 s cu 128 și 1,0 s cu 1.024 pe acest laptop. Cu aceeași rată de învățare, lotul mare nu a '
                 'convers după 40 de epoci (0,335); loturile mari cer de obicei o rată de învățare mai mare.'),
  'vanishing': ('Zece straturi ascunse de cîte 64 de neuroni sigmoizi, cu inițializare Xavier: de cîte ori este mai mic gradientul primului '
                'strat decît al ultimului?',
                'De aproximativ 10⁷ ori (8,5 × 10⁻⁷ față de 7,5; media a 5 inițializări pe un lot de 512 grupuri de locuințe): fiecare '
                'strat sigmoid înmulțește semnalul propagat înapoi cu cel mult 0,25, deci primele straturi abia învață. Cu tanh și '
                'Xavier, raportul rămîne în jur de 0,6, iar cu ReLU și inițializarea He gradientul rămîne de același ordin în toate '
                'straturile. Ponderile mici N(0; 0,01²) micșorează semnalul în ambele sensuri (gradienți de 10⁻¹² la adîncimea 10), '
                'ponderile mari N(0; 1) îl fac să explodeze (peste 10¹⁶ la adîncimea 10, depășire numerică dincolo de ea). Modificați '
                'adîncimea: problema crește geometric cu numărul de straturi.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 a: 'Intercept a', b: 'Slope b', step: 'Step', mse: 'MSE (log scale)', path: 'Path of gradient descent', start: 'Start', min: 'Least-squares minimum',
 pathTitle: x => `Loss surface: value = a + b · ${x === 'raw' ? 'income' : 'standardised income'}`, lossTitle: 'Loss after each step',
 gd1: (e, l, k, m, m0, b) => `η = ${e} (stable below 2/λmax = ${l}); after ${k} steps MSE = <b>${m}</b> (minimum ${m0}; momentum β = ${b}).`,
 gd2: (lmax, lmin, c) => `Curvatures λmax = ${lmax}, λmin = ${lmin}; condition number ${c}.`, div: 'The step is too large: the path diverges.',
 epoch: 'Epoch', vmse: 'Validation MSE', lrTitle: e => `Validation MSE per epoch, SGD with η = ${e}`, sweepTitle: 'Validation MSE after 30 epochs (3 seeds)',
 lr: 'Learning rate η (log scale)', seed: 'one seed', median: 'median of 3 seeds', diverged: 'diverged', linear: 'Linear regression',
 lrOut: (e, m, a, b) => `η = ${e}: validation MSE after 30 epochs <b>${m}</b> (seed 1); 3 seeds: ${a} to ${b}.`, lrDiv: (e, n) => `η = ${e}: ${n} of 3 seeds diverge (MSE above 5).`,
 onames: { sgd: 'SGD', momentum: 'SGD + momentum', rmsprop: 'RMSProp', adam: 'Adam', adamw: 'AdamW' },
 opTitle: (o, e) => `${o}, η = ${e}: validation MSE, 3 seeds`, mean3: 'mean of 3 seeds', best: e => `best η = ${e}`,
 op: (o, e, f, m, e10) => `${o}, η = ${e}: final validation MSE <b>${f}</b> (mean of 3 seeds; lowest ${m}); after 10 epochs ${e10}.`,
 opBest: (e) => `Best rate of this optimiser on the grid: η = ${e}.`,
 steps: 'Gradient steps (log scale)', secs: 'Seconds of computing (log scale)', bsName: b => `batch ${b}`,
 bsTitle: 'Adam, η = 0.001: validation MSE',
 bs: (b, spe, s, v5, v40) => `Batch ${b}: ${spe} steps per epoch, ${s} s for 40 epochs; validation MSE ${v5} after 5 epochs, <b>${v40}</b> after 40.`,
 layer: 'Layer (1 = next to the input)', gnorm: 'Gradient norm ‖∂L/∂W‖ (log scale)',
 vgTitle: (a, i, d) => `${a}, ${i} initialisation, ${d} hidden layers of 64 units`,
 vg: (f, l, r) => `First layer ${f}, last layer ${l}: ratio first / last <b>${r}</b> (mean of 5 seeds, one batch of 512 block groups).`,
 vgInf: 'The forward signal overflows: the gradient is infinite (not drawn).', vgZero: n => `The gradient underflows to exactly 0 in ${n} layer(s) (not drawn on the log scale).`,
 acts: { sig: 'Sigmoid', tanh: 'Tanh', relu: 'ReLU' }, inits: { xavier: 'Xavier', he: 'He', small: 'N(0, 0.01²)', large: 'N(0, 1)' }
};
""",
 'ro': r"""const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 a: 'Termenul liber a', b: 'Panta b', step: 'Pasul', mse: 'MSE (scară logaritmică)', path: 'Drumul coborîrii pe gradient', start: 'Start', min: 'Minimul celor mai mici pătrate',
 pathTitle: x => `Suprafața pierderii: valoarea = a + b · ${x === 'raw' ? 'venit' : 'venit standardizat'}`, lossTitle: 'Pierderea după fiecare pas',
 gd1: (e, l, k, m, m0, b) => `η = ${e} (stabil sub 2/λmax = ${l}); după ${deRO(k)} pași MSE = <b>${m}</b> (minimul ${m0}; momentul β = ${b}).`,
 gd2: (lmax, lmin, c) => `Curburile λmax = ${lmax}, λmin = ${lmin}; numărul de condiționare ${c}.`, div: 'Pasul este prea mare: drumul diverge.',
 epoch: 'Epoca', vmse: 'MSE pe validare', lrTitle: e => `MSE pe validare după fiecare epocă, SGD cu η = ${e}`, sweepTitle: 'MSE pe validare după 30 de epoci (3 inițializări)',
 lr: 'Rata de învățare η (scară logaritmică)', seed: 'o inițializare', median: 'mediana celor 3 inițializări', diverged: 'divergent', linear: 'Regresia liniară',
 lrOut: (e, m, a, b) => `η = ${e}: MSE pe validare după 30 de epoci <b>${m}</b> (inițializarea 1); 3 inițializări: între ${a} și ${b}.`, lrDiv: (e, n) => `η = ${e}: ${n} din 3 inițializări diverg (MSE peste 5).`,
 onames: { sgd: 'SGD', momentum: 'SGD + moment', rmsprop: 'RMSProp', adam: 'Adam', adamw: 'AdamW' },
 opTitle: (o, e) => `${o}, η = ${e}: MSE pe validare, 3 inițializări`, mean3: 'media celor 3 inițializări', best: e => `η optim = ${e}`,
 op: (o, e, f, m, e10) => `${o}, η = ${e}: MSE finală pe validare <b>${f}</b> (media a 3 inițializări; minimul ${m}); după 10 epoci ${e10}.`,
 opBest: (e) => `Rata cea mai bună a acestui optimizator pe grilă: η = ${e}.`,
 steps: 'Pași de gradient (scară logaritmică)', secs: 'Secunde de calcul (scară logaritmică)', bsName: b => `lot de ${b}`,
 bsTitle: 'Adam, η = 0,001: MSE pe validare',
 bs: (b, spe, s, v5, v40) => `Lotul de ${b}: ${deRO(spe)} pași pe epocă, ${s} s pentru 40 de epoci; MSE pe validare ${v5} după 5 epoci, <b>${v40}</b> după 40.`,
 layer: 'Stratul (1 = lîngă intrare)', gnorm: 'Norma gradientului ‖∂L/∂W‖ (scară logaritmică)',
 vgTitle: (a, i, d) => `${a}, inițializare ${i}, ${deRO(d)} straturi ascunse de 64 de neuroni`,
 vg: (f, l, r) => `Primul strat ${f}, ultimul strat ${l}: raportul primul / ultimul <b>${r}</b> (media a 5 inițializări, un lot de 512 grupuri de locuințe).`,
 vgInf: 'Semnalul propagat înainte depășește capacitatea numerică: gradientul este infinit (nu este desenat).', vgZero: n => `Gradientul devine exact 0 în ${n} strat(uri) (nu apare pe scara logaritmică).`,
 acts: { sig: 'Sigmoidă', tanh: 'Tanh', relu: 'ReLU' }, inits: { xavier: 'Xavier', he: 'He', small: 'N(0; 0,01²)', large: 'N(0; 1)' }
};
""",
}


def data():
    import interactive_data_L2 as idl
    d = idl.load()
    import json
    N = json.load(open(os.path.join(HERE, 'L2_numbers.json')))
    d['opt'] = {k: rnd(v, 4) for k, v in d['opt'].items()}
    d['grid'] = OPTS
    d['best'] = {o: N['opt'][o]['lr'] for o in OPTS}
    d['lr'] = rnd(d['lr'], 4)
    d['lr']['keys'] = [str(x) for x in idl.LRS]
    vg = {}
    for k, v in d.pop('vg')['v'].items():                   # inf (overflow) -> null, kept 0 for underflow
        vg[k] = [None if (x is None or x != x or abs(x) == float('inf')) else float(f'{x:.3e}') for x in v]
    d = rnd(d, 6)                                           # (gradient norms span 1e-23 to 1e16: not rounded)
    d['vg'] = {'depths': idl.DEPTHS, 'v': vg}
    return d


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
