"""build_lab_L3.py -- the Lecture 3 lab (interactive/L3_lab.html, L3_lab_ro.html): predict, test, prove.

Five rounds on regularisation and generalisation: the overfitting of the 256-256 network on German credit (loss and
AUC), the dropout x weight-decay grid drawn in 3D with three.js, early stopping on card fraud and double descent of the
random-feature regression on California housing. Each round asks for a prediction first, then lets the student test it
on the course data, then shows the proof. The page reads interactive/L3_data.js (build_interactive_L3.py); logic in
interactive/L3_lab.js and lab_common.js, texts in interactive/L3_lab_text_{en,ro}.js, style in interactive/lab.css.

usage:  python3 lectures/L3/build_lab_L3.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'L2'))
from build_lab_L2 import rng, rnd, THREE  # noqa: E402  (the round markup is shared with the Lecture 2 lab)

OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'interactive'))
VER = '20261117b'

L = {
    'en': dict(
        course='Neural Networks and Deep Learning', sub='Lecture 3 · Lab: predict, test, prove',
        title='Regularisation lab', back='← Course site', site='../index.html#chapters', pdf='Lecture PDF',
        inter='Interactive charts', inter_href='L3.html',
        lead='Five rounds on why a network that learns its training data perfectly can still fail, and what stops it. In each '
             'round you first <b>predict</b> what will happen and lock your answer, then <b>test</b> it on the course data, and '
             'only then see the <b>proof</b>. The data are those of the lecture: 1,000 German credit applicants, the card fraud '
             'sample (time-ordered split) and California housing.',
        steps=('1 Predict', '2 Test', '3 Prove'),
        heads=('Round 1 · Memorising', 'Round 2 · Loss or ranking?', 'Round 3 · Dropout or weight decay?',
               'Round 4 · When to stop', 'Round 5 · Beyond the interpolation point'),
        ep='Epochs', r3k='Epoch', r4s='Stop at epoch', r5p='Up to p',
        hint='Charts stay empty until you lock a prediction. In round 3, drag the bars to turn them; scroll or pinch to zoom.',
        foot='German credit: 600 training, 200 validation and 200 test applicants, 48 standardised inputs; network 256-256 '
             'with ReLU, AdamW (learning rate 0.001), batch 64, 150 epochs, no early stopping; logistic regression with C = 1. '
             'Card fraud: 28,924 transactions in time order (60/20/20), network 256-256, batch 256, 120 epochs; average '
             'precision with the weights that restore the full-data class mix. Double descent: 400 training and 5,000 test '
             'block groups per draw, random ReLU features, 10 draws, median. Seeds 20261108 + s. '
             'Code: lectures/L3/analysis_L3.py and interactive_data_L3.py on GitHub.',
        lang_ro='L3_lab_ro.html', lang_en='L3_lab.html'),
    'ro': dict(
        course='Rețele neuronale și deep learning', sub='Cursul 3 · Laborator: preziceți, testați, demonstrați',
        title='Laborator de regularizare', back='← Site-ul cursului', site='../index_ro.html#chapters', pdf='PDF-ul cursului',
        inter='Grafice interactive', inter_href='L3_ro.html',
        lead='Cinci runde despre motivul pentru care o rețea care își învață perfect datele de antrenare poate totuși să '
             'greșească și despre ce o oprește. În fiecare rundă mai întîi <b>preziceți</b> ce se va întîmpla și vă fixați '
             'răspunsul, apoi îl <b>testați</b> pe datele cursului și abia apoi vedeți <b>demonstrația</b>. Datele sînt cele '
             'din curs: 1.000 de solicitanți de credit din Germania, eșantionul de fraude cu carduri (împărțit în ordinea timpului) '
             'și California housing.',
        steps=('1 Preziceți', '2 Testați', '3 Demonstrați'),
        heads=('Runda 1 · Memorarea', 'Runda 2 · Pierdere sau ordonare?', 'Runda 3 · Dropout sau penalizarea ponderilor?',
               'Runda 4 · Cînd ne oprim', 'Runda 5 · Dincolo de punctul de interpolare'),
        ep='Epoci', r3k='Epoca', r4s='Oprire la epoca', r5p='Pînă la p',
        hint='Graficele rămîn goale pînă vă fixați predicția. În runda 3, trageți de bare ca să le rotiți; derulați sau ciupiți ca să măriți.',
        foot='Credit Germania: 600 de solicitanți pentru antrenare, 200 pentru validare și 200 pentru test, 48 de intrări '
             'standardizate; rețeaua 256-256 cu ReLU, AdamW (rata de învățare 0,001), loturi de 64, 150 de epoci, fără oprire '
             'timpurie; regresia logistică cu C = 1. Fraude cu carduri: 28.924 de tranzacții în ordinea timpului (60/20/20), '
             'rețeaua 256-256, loturi de 256, 120 de epoci; precizia medie cu ponderile care refac proporția claselor din datele '
             'complete. Dubla coborîre: 400 de grupuri de locuințe pentru antrenare și 5.000 pentru test la fiecare extragere, '
             'variabile ReLU aleatoare, 10 extrageri, mediana. Semințele 20261108 + s. '
             'Codul: lectures/L3/analysis_L3.py și interactive_data_L3.py pe GitHub.',
        lang_ro='L3_lab_ro.html', lang_en='L3_lab.html'),
}


def page(lang):
    t = L[lang]
    ro = lang == 'ro'
    h, s = t['heads'], t['steps']
    body = '\n'.join([
        rnd('r1', h[0], s, rng('r1-k', t['ep'], 1, 150, 1, 150), three=False),
        rnd('r2', h[1], s, rng('r2-k', t['ep'], 1, 150, 1, 150), three=False),
        rnd('r3', h[2], s, rng('r3-k', t['r3k'], 1, 150, 1, 150)),
        rnd('r4', h[3], s, rng('r4-s', t['r4s'], 1, 120, 1, 120), three=False),
        rnd('r5', h[4], s, rng('r5-p', t['r5p'], 0, 17, 1, 17), three=False),
    ])
    return f'''<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{t["title"]}</title>
<meta name="description" content="{t["course"]}: {t["sub"]}">
<link rel="icon" href="../assets/logos/ase_logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css?v={VER}">
<link rel="stylesheet" href="interactive.css?v={VER}">
<link rel="stylesheet" href="lab.css?v={VER}">
<script type="importmap">{{"imports": {{"three": "{THREE}/build/three.module.js", "three/addons/": "{THREE}/examples/jsm/"}}}}</script>
<script src="L3_data.js?v={VER}"></script>
<script src="L3_lab_text_{lang}.js?v={VER}"></script>
<script type="module" src="L3_lab.js?v={VER}"></script>
</head>
<body>
<header>
  <div class="header-inner">
    <a href="https://www.ase.ro" target="_blank" rel="noopener"><img class="header-logo" src="../assets/logos/ase_logo.png" alt="ASE"></a>
    <div class="header-title"><h1><a href="{t["site"].split('#')[0]}">{t["course"]}</a></h1><p>{t["sub"]}</p></div>
    <a href="https://ipe.ro/new/" target="_blank" rel="noopener"><img class="header-logo" src="../assets/logos/acad_logo.png" alt="IPE"></a>
  </div>
</header>
<nav><ul>
  <li><a href="{t["site"]}">{t["back"]}</a></li>
  <li><a href="../lectures/L3/lecture3{'_ro' if ro else ''}.pdf" target="_blank" rel="noopener">{t["pdf"]}</a></li>
  <li><a href="{t["inter_href"]}">{t["inter"]}</a></li>
  <li class="lang-switcher"><a href="{t["lang_en"]}" class="lang-btn{'' if ro else ' active'}">EN</a><a href="{t["lang_ro"]}" class="lang-btn{' active' if ro else ''}">RO</a></li>
</ul></nav>
<main class="imain lab">
<p class="lead">{t["lead"]}</p>
<p class="lab-hint">{t["hint"]}</p>
<p class="lab-score" id="lab-score" aria-live="polite"></p>
{body}
<p class="src foot">{t["foot"]}</p>
</main>
</body>
</html>
'''


if __name__ == '__main__':
    for lang, name in (('en', 'L3_lab.html'), ('ro', 'L3_lab_ro.html')):
        with open(os.path.join(OUT, name), 'w') as f:
            f.write(page(lang))
        print('wrote', name)
