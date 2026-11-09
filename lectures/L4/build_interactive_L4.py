"""Build the interactive page of Lecture 4 (convolutional neural networks and transfer learning): interactive/L4.html (EN) and
interactive/L4_ro.html (RO).

Inputs (interactive/L4_data.js) come from EuroSAT tiles, Fashion-MNIST test images, the trained CNN and the saved results of
analysis_L4.py, through lectures/L4/interactive_data_L4.py; the page logic is in interactive/L4.js.
Run: python3 lectures/L4/build_interactive_L4.py [--no-data]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, play, charts  # noqa: E402

K = 4
FM_EN = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FM_RO = ["Tricou", "Pantaloni", "Pulover", "Rochie", "Palton", "Sandală", "Cămașă", "Pantof sport", "Geantă", "Ghete"]
ES_EN = ["Annual crop", "Forest", "Herbaceous vegetation", "Highway", "Industrial", "Pasture", "Permanent crop", "Residential",
         "River", "Sea/lake"]
ES_RO = ["Culturi anuale", "Pădure", "Vegetație ierboasă", "Autostradă", "Zonă industrială", "Pășune", "Culturi permanente",
         "Zonă rezidențială", "Rîu", "Mare/lac"]

BLOCKS = [('convolution', 'conv_edges', 'conv', 'cv'),
          ('receptive-field', 'receptive_field', 'conv', 'rf'),
          ('feature-maps', 'fmnist_featuremaps', 'fm', 'fm'),
          ('confusion', 'fmnist_confusion', 'fm', 'cm'),
          ('residual', 'cifar_curves', 'arch', 'cf'),
          ('transfer', 'transfer_curves', 'arch', 'tl')]

SECTIONS = {
 'en': dict(conv='The convolutional layer', fm='Product photos: features learned by a CNN', arch='Deeper networks and transfer learning'),
 'ro': dict(conv='Stratul convoluțional', fm='Fotografii de produse: trăsăturile învățate de o CNN', arch='Rețele mai adînci și transferul învățării'),
}

L = {
 'en': dict(h1='Lecture 4 · Convolutional neural networks (CNN) and transfer learning',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser when you move a '
        'control. The ▶ button moves the main parameter step by step. Under each figure: a readout, a question for the room and '
        'the Quantlet with the code of the chart. The first block runs a convolution live on a real satellite image; the feature '
        'maps come from the CNN trained in the lecture; the curves are its saved training runs.',
   src='Data: EuroSAT (Sentinel-2 satellite tiles of 64 × 64 pixels, 10 m per pixel, 10 land-use classes; Helber et al., 2019), '
       'Fashion-MNIST (Zalando, 28 × 28 grey product photos; 10,000 test images), CIFAR-10 (32 × 32 colour photographs). Networks '
       'and seeds (20261108) as in the lecture (analysis_L4.py); ResNet-18 = residual network with 18 layers, pre-trained on ImageNet; '
       'MLP = multilayer perceptron; ReLU = rectified linear unit.'),
 'ro': dict(h1='Cursul 4 · Rețele neuronale convoluționale (CNN) și transferul învățării',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser cînd modificați un control. Butonul '
        '▶ modifică parametrul principal pas cu pas. Sub fiecare figură: o sinteză numerică, o întrebare pentru sală și '
        'Quantlet-ul cu codul figurii. Primul bloc aplică o convoluție, chiar în browser, pe o imagine reală din satelit; hărțile '
        'de caracteristici provin din CNN antrenată în curs; curbele sînt antrenările ei salvate.',
   src='Date: EuroSAT (imagini Sentinel-2 de 64 × 64 de pixeli, 10 m pe pixel, 10 clase de utilizare a terenului; Helber et al., '
       '2019), Fashion-MNIST (Zalando, fotografii de produse de 28 × 28, în tonuri de gri; 10.000 de imagini de test), CIFAR-10 '
       '(fotografii color de 32 × 32). Rețelele și semințele (20261108) sînt cele din curs (analysis_L4.py); ResNet-18 = rețea '
       'reziduală cu 18 straturi, preantrenată pe ImageNet; MLP = perceptron multistrat; ReLU = unitate liniară rectificată.'),
}


def widgets(lang):
    ro = lang == 'ro'
    ES = ES_RO if ro else ES_EN
    FM = FM_RO if ro else FM_EN
    kern = [('sx', 'Muchii verticale (Sobel)' if ro else 'Vertical edges (Sobel)'), ('sy', 'Muchii orizontale (Sobel)' if ro else 'Horizontal edges (Sobel)'),
            ('lap', 'Contururi (Laplace)' if ro else 'Outlines (Laplacian)'), ('blur', 'Media 3 × 3 (estompare)' if ro else 'Mean 3 × 3 (blur)'),
            ('sharp', 'Accentuare' if ro else 'Sharpen'), ('id', 'Identitate' if ro else 'Identity')]
    return {
     'convolution': (select('cv-i', 'Imaginea EuroSAT' if ro else 'EuroSAT tile', [(c, ES[c]) for c in range(10)], 3)
                     + select('cv-k', 'Filtrul (nucleul 3 × 3)' if ro else 'Filter (3 × 3 kernel)', kern, 'sx')
                     + slider('cv-s', 'Pasul (stride)' if ro else 'Stride', 1, 3, 1, 1)
                     + toggles('cv-o', 'Opțiuni' if ro else 'Options', [('pad', 'Bordare cu zero („same”)' if ro else 'Zero padding ("same")'),
                                                                       ('relu', 'ReLU după convoluție' if ro else 'ReLU after the convolution')], ['pad']),
                     charts('cv-in', 'cv-res')),
     'receptive-field': (slider('rf-l', 'Numărul de straturi convoluționale' if ro else 'Number of convolutional layers', 1, 12, 1, 6)
                         + select('rf-k', 'Mărimea nucleului' if ro else 'Kernel size', [(3, '3 × 3'), (5, '5 × 5'), (7, '7 × 7')], 3)
                         + toggles('rf-p', 'Agregare' if ro else 'Pooling', [('pool', 'Agregare 2 × 2 la două straturi' if ro else '2 × 2 pooling every two layers')], [])
                         + play('rf'), charts('rf-chart', 'rf-tile')),
     'feature-maps': (select('fm-i', 'Imaginea de test' if ro else 'Test image', [(c, FM[c]) for c in range(10)], 9)
                      + radios('fm-b', 'Blocul' if ro else 'Block', [('b1', 'blocul 1 (32 de canale, 14 × 14)' if ro else 'block 1 (32 channels, 14 × 14)'),
                                                                     ('b3', 'blocul 3 (128 de canale, 7 × 7)' if ro else 'block 3 (128 channels, 7 × 7)')], 'b1')
                      + slider('fm-c', 'Canalul (1 = cel mai activ)' if ro else 'Channel (1 = most active)', 0, 31, 1, 0) + play('fm'),
                      charts('fm-img', 'fm-map')),
     'confusion': (select('cm-m', 'Modelul' if ro else 'Model', [('softmax', 'Regresie softmax' if ro else 'Softmax regression'), ('mlp', 'MLP'), ('cnn', 'CNN')], 'cnn')
                   + select('cm-c', 'Clasa reală evidențiată' if ro else 'True class highlighted', [(c, FM[c]) for c in range(10)], 6)
                   + toggles('cm-n', 'Afișare' if ro else 'Show', [('pct', 'Procente pe rînd' if ro else 'Row percentages')], ['pct']),
                   charts('cm-chart', cls='one')),
     'residual': (toggles('cf-m', 'Rețele' if ro else 'Networks', [('cnn', 'CNN mică, fără augmentare' if ro else 'Small CNN, no augmentation'),
                                                                   ('cnn_aug', 'CNN mică, cu augmentare' if ro else 'Small CNN, augmentation'),
                                                                   ('plain32', 'Rețea simplă, 32 de straturi' if ro else 'Plain network, 32 layers'),
                                                                   ('res32', 'ResNet, 32 de straturi' if ro else 'ResNet, 32 layers')], ['plain32', 'res32'])
                  + radios('cf-y', 'Măsura' if ro else 'Measure', [('train_loss', 'Pierderea la antrenare' if ro else 'Training loss'),
                                                                   ('train_acc', 'Acuratețea la antrenare' if ro else 'Training accuracy'),
                                                                   ('valid_acc', 'Acuratețea pe validare' if ro else 'Validation accuracy')], 'train_loss'),
                  charts('cf-chart', cls='one')),
     'transfer': (slider('tl-n', 'Imagini etichetate pe clasă' if ro else 'Labelled tiles per class', 0, 5, 1, 0)
                  + toggles('tl-m', 'Modele' if ro else 'Models', [('scratch', 'CNN de la zero' if ro else 'CNN from scratch'),
                                                                   ('probe', 'ResNet-18 înghețat + regresie logistică' if ro else 'Frozen ResNet-18 + logistic regression'),
                                                                   ('finetune', 'ResNet-18 ajustat fin' if ro else 'Fine-tuned ResNet-18')], ['scratch', 'probe', 'finetune'])
                  + play('tl'), charts('tl-hist', 'tl-budget')),
    }


QA = {
 'en': {
  'convolution': ('A 3 × 3 filter slides over a 64 × 64 Sentinel-2 tile. How many weights does it have, how large is the output, and '
                  'why does the Sobel filter light up the highway?',
                  'Nine weights and one bias, whatever the size of the image (a dense layer from 4,096 pixels to 4,096 outputs would need '
                  'about 16.8 million weights). With zero padding the output keeps 64 × 64 pixels; without it, (64 − 3)/1 + 1 = 62; with '
                  'stride 2, (64 + 2 − 3)/2 + 1 = 32 (integer part). The vertical Sobel filter computes a left-minus-right difference: '
                  'large positive or negative values where brightness changes from left to right, as on the edges of the highway, and '
                  'zero on uniform fields. Switch to the river or the residential tile: the same filter finds other edges.'),
  'receptive-field': ('How many 3 × 3 convolutions must be stacked before one neuron "sees" a whole 64 × 64 EuroSAT tile (640 m × 640 m)?',
                      'Without pooling the receptive field grows by 2 pixels per layer: 13 pixels after 6 layers, 25 after 12, so about 32 '
                      'layers would be needed. With 2 × 2 pooling after every two convolutions it grows geometrically: 68 pixels after 8 '
                      'convolutions, more than the whole tile. Larger kernels (5 × 5, 7 × 7) grow it faster but cost 25 or 49 weights per '
                      'channel pair instead of 9; stacking small kernels with pooling is the cheaper way to see far.'),
  'feature-maps': ('Feed the trained CNN the shirt of the test set. Which class does it predict, and what do the block 1 and block 3 '
                   'channels respond to?',
                   'The CNN gives "T-shirt/top" 69.7% for this shirt: the most frequent error of the network, 102 of the 1,000 test '
                   'shirts go to T-shirt/top and the recall of the class is 75.1%. Block 1 channels (14 × 14) still look like the '
                   'image: edges, collars, the background. Block 3 channels (7 × 7) are coarse and respond to parts of the object; '
                   'their average over the map is what the classifier reads. For the other nine test images the CNN is almost certain '
                   '(99.8% or more).'),
  'confusion': ('Which product class is hardest, and does the CNN fix the errors of the softmax regression or only reduce them?',
                'Shirt is the hardest class for all three models: recall 56.9% for the softmax regression, 68.2% for the MLP '
                '(multilayer perceptron) and 75.1% for the CNN; the largest confusion is always shirt → T-shirt/top (138, 124 and 102 '
                'images). Accuracy rises from 84.6% (7,850 parameters) to 89.0% (235,146) and 92.9% (140,778): the CNN has fewer '
                'parameters than the MLP and makes 37% fewer errors, because it shares weights across positions. The remaining errors '
                'are between visually similar classes (shirt, T-shirt, pullover, coat).'),
  'residual': ('A plain network of 32 layers and a ResNet of 32 layers have the same 466,906 parameters. Why does the plain one train worse?',
               'This is the degradation problem: after 12 epochs the plain network still has a training loss of 0.557 and 80.0% test '
               'accuracy, while the residual one reaches 0.291 and 87.8%. The deeper plain network is not overfitting (its training '
               'accuracy is low too); it is hard to optimise. Skip connections let each block learn a correction to the identity and '
               'let the gradient flow back unchanged. The small CNN shows the other problem: without augmentation the training '
               'accuracy reaches 96.1% against 87.2% on validation; with augmentation the gap falls to 0.9 points.'),
  'transfer': ('With only 10 labelled satellite tiles per class (100 in all), how far can each approach go?',
               'A CNN trained from scratch reaches 69.9% test accuracy (mean of 2 seeds), a frozen ResNet-18 with a logistic regression on '
               'its features 78.6%, and a fine-tuned ResNet-18 87.8%. The gap closes as labels grow: with 1,000 tiles per class the '
               'three reach 94.7%, 93.0% and 98.1%, and with the full 16,200 training tiles 97.8%, 93.5% and 98.8%. Features learned '
               'on ImageNet photographs carry over to satellite images; they matter most when labels are scarce. The dip of the '
               'scratch CNN at 250 tiles (86.6% after 89.5% at 100) shows the noise of two seeds.'),
 },
 'ro': {
  'convolution': ('Un filtru de 3 × 3 se deplasează pe o imagine Sentinel-2 de 64 × 64. Cîte ponderi are, cît de mare este ieșirea și de '
                  'ce filtrul Sobel evidențiază autostrada?',
                  'Nouă ponderi și un termen liber, oricare ar fi mărimea imaginii (un strat dens de la 4.096 de pixeli la 4.096 de '
                  'ieșiri ar avea nevoie de circa 16,8 milioane de ponderi). Cu bordare cu zero, ieșirea păstrează 64 × 64 de pixeli; '
                  'fără ea, (64 − 3)/1 + 1 = 62; cu pasul 2, (64 + 2 − 3)/2 + 1 = 32 (partea întreagă). Filtrul Sobel vertical calculează '
                  'o diferență stînga-minus-dreapta: valori mari, pozitive sau negative, acolo unde luminozitatea se schimbă de la '
                  'stînga la dreapta, ca pe marginile autostrăzii, și zero pe cîmpurile uniforme. Treceți la imaginea cu rîul sau cu '
                  'zona rezidențială: același filtru găsește alte muchii.'),
  'receptive-field': ('Cîte convoluții de 3 × 3 trebuie suprapuse pînă cînd un neuron „vede” o imagine EuroSAT întreagă, de 64 × 64 '
                      '(640 m × 640 m)?',
                      'Fără agregare, cîmpul receptiv crește cu 2 pixeli pe strat: 13 pixeli după 6 straturi, 25 după 12, deci ar fi '
                      'nevoie de aproximativ 32 de straturi. Cu o agregare 2 × 2 după fiecare două convoluții, el crește geometric: '
                      '68 de pixeli după 8 convoluții, mai mult decît întreaga imagine. Nucleele mai mari (5 × 5, 7 × 7) îl cresc mai '
                      'repede, dar costă 25 sau 49 de ponderi pe pereche de canale în loc de 9; suprapunerea nucleelor mici cu agregare '
                      'este calea mai ieftină de a vedea departe.'),
  'feature-maps': ('Dați CNN antrenate cămașa din setul de test. Ce clasă prezice și la ce răspund canalele din blocul 1 și din blocul 3?',
                   'CNN dă „Tricou” cu 69,7% pentru această cămașă: este eroarea cea mai frecventă a rețelei, 102 din cele 1.000 de '
                   'cămăși de test sînt clasificate drept tricouri, iar sensibilitatea clasei este 75,1%. Canalele din blocul 1 '
                   '(14 × 14) seamănă încă cu imaginea: muchii, guler, fundal. Canalele din blocul 3 (7 × 7) sînt grosiere și răspund '
                   'la părți ale obiectului; media lor pe hartă este ceea ce citește clasificatorul. Pentru celelalte nouă imagini de '
                   'test, CNN este aproape sigură (99,8% sau mai mult).'),
  'confusion': ('Care clasă de produse este cea mai dificilă și corectează CNN erorile regresiei softmax sau doar le reduce?',
                'Cămașa este clasa cea mai dificilă pentru toate cele trei modele: sensibilitatea este 56,9% pentru regresia softmax, '
                '68,2% pentru MLP (perceptronul multistrat) și 75,1% pentru CNN; cea mai mare confuzie este mereu cămașă → tricou '
                '(138, 124 și 102 imagini). Acuratețea crește de la 84,6% (7.850 de parametri) la 89,0% (235.146) și 92,9% (140.778): '
                'CNN are mai puțini parametri decît MLP și face cu 37% mai puține erori, pentru că folosește aceleași ponderi în toate '
                'pozițiile. Erorile rămase sînt între clase asemănătoare vizual (cămașă, tricou, pulover, palton).'),
  'residual': ('O rețea simplă cu 32 de straturi și o ResNet cu 32 de straturi au aceiași 466.906 parametri. De ce se antrenează mai prost '
               'cea simplă?',
               'Este problema degradării: după 12 epoci, rețeaua simplă are încă o pierdere la antrenare de 0,557 și o acuratețe pe '
               'test de 80,0%, în timp ce rețeaua reziduală ajunge la 0,291 și 87,8%. Rețeaua simplă mai adîncă nu se supraajustează '
               '(și acuratețea ei la antrenare este mică); este greu de optimizat. Conexiunile de ocolire permit fiecărui bloc să '
               'învețe o corecție a identității și lasă gradientul să se propage înapoi neschimbat. CNN mică arată cealaltă problemă: '
               'fără augmentare, acuratețea la antrenare ajunge la 96,1%, față de 87,2% pe validare; cu augmentare, diferența scade la '
               '0,9 puncte.'),
  'transfer': ('Cu doar 10 imagini din satelit etichetate pe clasă (100 în total), pînă unde poate ajunge fiecare abordare?',
               'O CNN antrenată de la zero ajunge la 69,9% acuratețe pe test (media a 2 semințe), un ResNet-18 înghețat cu o regresie '
               'logistică pe caracteristicile lui la 78,6%, iar un ResNet-18 ajustat fin la 87,8%. Diferența se închide pe măsură ce '
               'etichetele cresc: cu 1.000 de imagini pe clasă, cele trei ajung la 94,7%, 93,0% și 98,1%, iar cu toate cele 16.200 de '
               'imagini de antrenare la 97,8%, 93,5% și 98,8%. Caracteristicile învățate pe fotografiile ImageNet se transferă la '
               'imaginile din satelit; contează cel mai mult cînd etichetele sînt puține. Scăderea CNN de la zero la 250 de imagini '
               '(86,6% după 89,5% la 100) arată zgomotul a două semințe.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 es: """ + repr(ES_EN) + r""", fm: """ + repr(FM_EN) + r""",
 input: c => `Input: ${c} (64 × 64, grey levels)`, output: (k, n) => `Output of the filter: ${n} × ${n}`,
 kname: { sx: 'vertical edges', sy: 'horizontal edges', lap: 'outlines', blur: 'mean 3 × 3', sharp: 'sharpen', id: 'identity' },
 cv: (n, p, s, o) => `Output size ⌊(${n} + 2·${p} − 3) / ${s}⌋ + 1 = <b>${o}</b> pixels per side; the filter has 9 weights + 1 bias, used at all ${o * o} positions.`,
 cvK: k => `Kernel: ${k}`,
 layers: 'Convolutional layers', rfY: 'Receptive field (pixels)', plain: '3 × 3 convolutions, stride 1', this: 'chosen design',
 rfTile: 'Receptive field on a 64 × 64 tile (10 m per pixel)',
 rf: (l, r, m, w) => `After ${l} layers one neuron sees <b>${r} × ${r} pixels</b> (${m} m × ${m} m on the ground); weights per channel pair in each layer: ${w}.`,
 rfFull: 'The receptive field covers the whole tile.',
 img: c => `Test image: ${c}`, map: (b, c, n) => `Block ${b === 'b1' ? 1 : 3}, channel ranked ${c} (${n} × ${n})`,
 fmOut: (c, p, t) => `Prediction of the CNN: <b>${c}</b> (${p}); next: ${t}.`, fmMap: (ch, mx) => `Channel no. ${ch} of the layer; its largest activation on this image is ${mx}.`,
 pred: 'Predicted class', truth: 'True class', cmTitle: (m, a) => `${m}: confusion matrix, 10,000 test images (accuracy ${a})`,
 models: { softmax: 'Softmax regression', mlp: 'MLP', cnn: 'CNN' },
 cm: (c, r, n, t, tn) => `${c}: recall <b>${r}</b> (${n} of 1,000 test images); most frequent error: ${t} (${tn} images).`,
 cm2: (p) => `Parameters of the model: ${p}.`,
 cfNames: { cnn: 'Small CNN, no augmentation', cnn_aug: 'Small CNN, augmentation', plain32: 'Plain, 32 layers', res32: 'ResNet, 32 layers' },
 cfY: { train_loss: 'Training loss', train_acc: 'Training accuracy (%)', valid_acc: 'Validation accuracy (%)' }, epoch: 'Epoch',
 cf: (m, t, p, tr, va, l) => `${m}: test accuracy <b>${t}</b>, ${p} parameters; after 12 epochs training accuracy ${tr}, validation ${va}, training loss ${l}.`,
 tlNames: { scratch: 'CNN from scratch', probe: 'Frozen ResNet-18 + logistic', finetune: 'Fine-tuned ResNet-18' },
 tlHist: 'Full training set: validation accuracy (%)', tlBudget: 'Test accuracy (%) by label budget', epochs: 'Epochs', perClass: 'Labelled tiles per class (log scale)',
 acc: 'Accuracy (%)', seeds: 'one seed',
 tl: (m, n, a, lo, hi) => `${m}: <b>${a}</b> with ${n} tiles per class (2 seeds: ${lo} and ${hi}).`, tl0: (n, t) => `${n} labelled tiles per class = ${t} tiles in all (full training set: 16,200).`
};
""",
 'ro': r"""const deRO = s => { const n = parseInt(String(s).replace(/\D/g, ''), 10), r = n % 100; return n !== 1 && (r === 0 || r >= 20) ? `${s} de` : `${s}`; };
window.LK_TEXT = {
 es: """ + repr(ES_RO) + r""", fm: """ + repr(FM_RO) + r""",
 input: c => `Intrarea: ${c} (64 × 64, niveluri de gri)`, output: (k, n) => `Ieșirea filtrului: ${n} × ${n}`,
 kname: { sx: 'muchii verticale', sy: 'muchii orizontale', lap: 'contururi', blur: 'media 3 × 3', sharp: 'accentuare', id: 'identitate' },
 cv: (n, p, s, o) => `Mărimea ieșirii ⌊(${n} + 2·${p} − 3) / ${s}⌋ + 1 = <b>${deRO(o)}</b> pixeli pe latură; filtrul are 9 ponderi + 1 termen liber, folosite în toate cele ${deRO(o * o)} poziții.`,
 cvK: k => `Nucleul: ${k}`,
 layers: 'Straturi convoluționale', rfY: 'Cîmpul receptiv (pixeli)', plain: 'convoluții 3 × 3, pas 1', this: 'arhitectura aleasă',
 rfTile: 'Cîmpul receptiv pe o imagine de 64 × 64 (10 m pe pixel)',
 rf: (l, r, m, w) => `După ${deRO(l)} straturi, un neuron vede <b>${r} × ${deRO(r)} pixeli</b> (${m} m × ${m} m pe teren); ponderi pe pereche de canale în fiecare strat: ${w}.`,
 rfFull: 'Cîmpul receptiv acoperă întreaga imagine.',
 img: c => `Imaginea de test: ${c}`, map: (b, c, n) => `Blocul ${b === 'b1' ? 1 : 3}, canalul de rang ${c} (${n} × ${n})`,
 fmOut: (c, p, t) => `Predicția CNN: <b>${c}</b> (${p}); următoarele: ${t}.`, fmMap: (ch, mx) => `Canalul nr. ${ch} al stratului; activarea lui maximă pe această imagine este ${mx}.`,
 pred: 'Clasa prezisă', truth: 'Clasa reală', cmTitle: (m, a) => `${m}: matricea de confuzie, 10.000 de imagini de test (acuratețea ${a})`,
 models: { softmax: 'Regresie softmax', mlp: 'MLP', cnn: 'CNN' },
 cm: (c, r, n, t, tn) => `${c}: sensibilitatea <b>${r}</b> (${n} din 1.000 de imagini de test); eroarea cea mai frecventă: ${t} (${deRO(tn)} imagini).`,
 cm2: (p) => `Parametrii modelului: ${p}.`,
 cfNames: { cnn: 'CNN mică, fără augmentare', cnn_aug: 'CNN mică, cu augmentare', plain32: 'Simplă, 32 de straturi', res32: 'ResNet, 32 de straturi' },
 cfY: { train_loss: 'Pierderea la antrenare', train_acc: 'Acuratețea la antrenare (%)', valid_acc: 'Acuratețea pe validare (%)' }, epoch: 'Epoca',
 cf: (m, t, p, tr, va, l) => `${m}: acuratețea pe test <b>${t}</b>, ${deRO(p)} parametri; după 12 epoci, acuratețea la antrenare ${tr}, pe validare ${va}, pierderea la antrenare ${l}.`,
 tlNames: { scratch: 'CNN de la zero', probe: 'ResNet-18 înghețat + logistic', finetune: 'ResNet-18 ajustat fin' },
 tlHist: 'Tot setul de antrenare: acuratețea pe validare (%)', tlBudget: 'Acuratețea pe test (%) după numărul de etichete', epochs: 'Epoci', perClass: 'Imagini etichetate pe clasă (scară logaritmică)',
 acc: 'Acuratețea (%)', seeds: 'o sămînță',
 tl: (m, n, a, lo, hi) => `${m}: <b>${a}</b> cu ${deRO(n)} imagini pe clasă (2 semințe: ${lo} și ${hi}).`, tl0: (n, t) => `${deRO(n)} imagini etichetate pe clasă = ${deRO(t)} imagini în total (tot setul de antrenare: 16.200).`
};
""",
}


def data():
    import interactive_data_L4 as idl
    return idl.load()


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
