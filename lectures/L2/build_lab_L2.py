"""build_lab_L2.py -- the Lecture 2 lab (interactive/L2_lab.html, L2_lab_ro.html): predict, test, prove.

Four rounds on gradient descent: the narrow valley of the raw-income regression, a step above 2/lambda_max, the two fixes
(standardising, momentum) and the learning-rate sweep of the 64-64 network. Each round asks for a prediction first,
then lets the student test it (a 3D loss surface drawn with three.js, or the validation curves), then shows the proof.
The page reads the course data of interactive/L2_data.js (build_interactive_L2.py); logic in interactive/L2_lab.js,
texts in interactive/L2_lab_text_{en,ro}.js, style in interactive/lab.css.

usage:  python3 lectures/L2/build_lab_L2.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'interactive'))
VER = '20261117a'
THREE = 'https://cdn.jsdelivr.net/npm/three@0.160.0'

L = {
    'en': dict(
        course='Neural Networks and Deep Learning', sub='Lecture 2 · Lab: predict, test, prove',
        title='Gradient descent lab', back='← Course site', site='../index.html#chapters', pdf='Lecture PDF',
        inter='Interactive charts', inter_href='L2.html',
        lead='Four rounds on how a network learns. In each round you first <b>predict</b> what will happen and lock your answer, '
             'then <b>test</b> it on the real loss surface of the lecture, which you can turn with the mouse or a finger, '
             'and only then see the <b>proof</b>. The data are those of the lecture: the median house value of the 20,640 '
             'block groups of California (1990 census) against their median income, and the 64-64 network of slide 17.',
        steps=('1 Predict', '2 Test', '3 Prove'),
        heads=('Round 1 · The narrow valley', 'Round 2 · Just above the edge', 'Round 3 · Two fixes', 'Round 4 · A real network'),
        r1e='Step η', r1k='Number of steps', r2e='Step η', r3m='Surface',
        r3opts=('standardised income, plain gradient descent', 'raw income, with momentum β = 0.9 and half the step'),
        r4e='Learning rate η',
        hint='The gold diamond marks the least-squares minimum; the red ball is gradient descent. Drag to turn, scroll or pinch to zoom.',
        foot='Loss surface: mean squared error (MSE) of value = a + b × income, computed exactly from five sample moments of the '
             'data; gradient descent and momentum use the formulas of analysis_L2.py (start a = 4, b = −0.5, or b = −1 when '
             'standardised). Network: two hidden layers of 64 rectified linear units (ReLU), plain stochastic gradient descent, '
             'batch 128, 30 epochs; curves of seed 0, final MSE of three seeds (60/20/20 split, seed 20261108). '
             'Code: lectures/L2/analysis_L2.py and interactive_data_L2.py on GitHub.',
        lang_ro='L2_lab_ro.html', lang_en='L2_lab.html'),
    'ro': dict(
        course='Rețele neuronale și deep learning', sub='Cursul 2 · Laborator: preziceți, testați, demonstrați',
        title='Laborator de coborîre pe gradient', back='← Site-ul cursului', site='../index_ro.html#chapters', pdf='PDF-ul cursului',
        inter='Grafice interactive', inter_href='L2_ro.html',
        lead='Patru runde despre felul în care învață o rețea. În fiecare rundă mai întîi <b>preziceți</b> ce se va întîmpla și vă '
             'fixați răspunsul, apoi îl <b>testați</b> pe suprafața de pierdere reală din curs, pe care o puteți roti cu mouse-ul sau '
             'cu degetul, și abia apoi vedeți <b>demonstrația</b>. Datele sînt cele din curs: valoarea mediană a locuințelor din cele '
             '20.640 de grupuri de locuințe din California (recensămîntul din 1990) în funcție de venitul median, și rețeaua 64-64 de pe slide-ul 17.',
        steps=('1 Preziceți', '2 Testați', '3 Demonstrați'),
        heads=('Runda 1 · Valea îngustă', 'Runda 2 · Chiar peste prag', 'Runda 3 · Două soluții', 'Runda 4 · O rețea reală'),
        r1e='Pasul η', r1k='Numărul de pași', r2e='Pasul η', r3m='Suprafața',
        r3opts=('venit standardizat, coborîre simplă pe gradient', 'venit brut, cu moment β = 0,9 și jumătate din pas'),
        r4e='Rata de învățare η',
        hint='Rombul auriu marchează minimul celor mai mici pătrate; bila roșie este coborîrea pe gradient. Trageți ca să rotiți; derulați sau ciupiți ca să măriți.',
        foot='Suprafața de pierdere: eroarea medie pătratică (MSE) a modelului valoare = a + b × venit, calculată exact din cinci momente '
             'de selecție ale datelor; coborîrea pe gradient și momentul folosesc formulele din analysis_L2.py (start a = 4, b = −0,5, '
             'sau b = −1 pentru venitul standardizat). Rețeaua: două straturi ascunse a cîte 64 de unități liniare rectificate (ReLU), '
             'coborîre stocastică pe gradient simplă, loturi de 128, 30 de epoci; curbele seminței 0, MSE finală pentru trei semințe '
             '(împărțire 60/20/20, sămînța 20261108). Codul: lectures/L2/analysis_L2.py și interactive_data_L2.py pe GitHub.',
        lang_ro='L2_lab_ro.html', lang_en='L2_lab.html'),
}


def rng(i, lab, lo, hi, st, v):
    return (f'<div class="ctl"><label for="{i}">{lab}: <b id="{i}-v"></b></label>'
            f'<input type="range" id="{i}" min="{lo}" max="{hi}" step="{st}" value="{v}"></div>')


def rnd(rid, head, steps, ctls, three=True):
    return f'''<section class="lab-round" id="{rid}">
  <h2 class="sec">{head}</h2>
  <ol class="lab-steps"><li class="lab-step-predict on">{steps[0]}</li><li class="lab-step-test">{steps[1]}</li><li class="lab-step-prove">{steps[2]}</li></ol>
  <div class="lab-grid">
    <div class="lab-left">
      <p class="lab-q"></p>
      <div class="lab-opts"></div>
      <button class="btn lab-lock" type="button"></button>
      <div class="lab-ctls">{ctls}<button class="btn lab-run" type="button"></button></div>
      <p class="lab-out" aria-live="polite"></p>
      <button class="btn lab-why" type="button"></button>
      <div class="lab-proof" hidden></div>
    </div>
    <div class="lab-right"><div class="lab-stage{' three' if three else ''}"></div></div>
  </div>
</section>'''


def page(lang):
    t = L[lang]
    ro = lang == 'ro'
    r3 = (f'<div class="ctl ctl-tg"><span class="lbl">{t["r3m"]}</span><div class="tgs">'
          f'<label class="tg"><input type="radio" name="r3-m" value="std" checked> {t["r3opts"][0]}</label>'
          f'<label class="tg"><input type="radio" name="r3-m" value="mom"> {t["r3opts"][1]}</label></div></div>')
    body = '\n'.join([
        rnd('r1', t['heads'][0], t['steps'], rng('r1-e', t['r1e'], 0.1, 2.2, 0.05, 1) + rng('r1-k', t['r1k'], 1, 400, 1, 60)),
        rnd('r2', t['heads'][1], t['steps'], rng('r2-e', t['r2e'], 0.1, 2.4, 0.05, 2.1)),
        rnd('r3', t['heads'][2], t['steps'], r3),
        rnd('r4', t['heads'][3], t['steps'], rng('r4-e', t['r4e'], 0, 8, 1, 2), three=False),
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
<script src="L2_data.js?v={VER}"></script>
<script src="L2_lab_text_{lang}.js?v={VER}"></script>
<script type="module" src="L2_lab.js?v={VER}"></script>
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
  <li><a href="../lectures/L2/lecture2{'_ro' if ro else ''}.pdf" target="_blank" rel="noopener">{t["pdf"]}</a></li>
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
    for lang, name in (('en', 'L2_lab.html'), ('ro', 'L2_lab_ro.html')):
        with open(os.path.join(OUT, name), 'w') as f:
            f.write(page(lang))
        print('wrote', name)
