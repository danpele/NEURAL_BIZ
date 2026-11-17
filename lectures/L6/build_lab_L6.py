"""build_lab_L6.py -- the Lecture 6 lab (interactive/L6_lab.html, L6_lab_ro.html): predict, test, prove.

Five rounds on text representation: Zipf's law in the speeches of the European Central Bank, the nearest neighbours of our
own word2vec (the word space drawn in 3D with three.js), analogies, mean embeddings against TF-IDF, and the agreement of the
Financial PhraseBank annotators. Each round asks for a prediction first, then lets the student test it on the course data,
then shows the proof. The page reads interactive/L6_data.js (build_interactive_L6.py) and window.LAB_NUM, written here from
L6_numbers.json of analysis_L6.py; logic in interactive/L6_lab.js and lab_common.js, texts in
interactive/L6_lab_text_{en,ro}.js, style in interactive/lab.css.

usage:  python3 lectures/L6/build_lab_L6.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'L2'))
from build_lab_L2 import rng, rnd, THREE  # noqa: E402  (the round markup is shared with the Lecture 2 lab)

OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'interactive'))
VER = '20261118b'


def lab_num():
    n = json.load(open(os.path.join(HERE, 'L6_numbers.json'), encoding='utf-8'))
    return {k: n[k] for k in ('analogy', 'corpus', 'w2v', 'sent', 'cos', 'oov', 'probes')}


L = {
    'en': dict(
        course='Neural Networks and Deep Learning', sub='Lecture 6 · Lab: predict, test, prove',
        title='Word embeddings lab', back='← Course site', site='../index.html#chapters', pdf='Lecture PDF',
        inter='Interactive charts', inter_href='L6.html',
        lead='Five rounds on how text becomes numbers. In each round you first <b>predict</b> what will happen and lock your '
             'answer, then <b>test</b> it on the course data, and only then see the <b>proof</b>. The data are those of the '
             'lecture: 400 speeches of the European Central Bank (2022–2026), on which we trained our own word2vec and GloVe, '
             'and the Financial PhraseBank sentences.',
        steps=('1 Predict', '2 Test', '3 Prove'),
        heads=('Round 1 · A few words do most of the work', 'Round 2 · Who are the neighbours?', 'Round 3 · Arithmetic with words',
               'Round 4 · Averaging embeddings or counting words?', 'Round 5 · When the annotators disagree'),
        r1r='Up to rank', r2w='Word', r5a='Annotator agreement',
        hint='Charts stay empty until you lock a prediction. In round 2, drag the word space to turn it; scroll or pinch to zoom.',
        foot='ECB speeches: 400 speeches, 8 January 2022 to 30 October 2026, 1,289,896 tokens. word2vec: skip-gram with negative '
             'sampling, 100 dimensions, window 5, words seen at least 5 times; GloVe on the same text; both trained in PyTorch. '
             'The 3D view is a principal component projection of the 32 words of four topic groups. Financial PhraseBank '
             '(Malo et al., 2014): 4,838 sentences, 5 random splits; finance tweets: 3 seeds. '
             'Code: lectures/L6/analysis_L6.py and build_interactive_L6.py on GitHub.',
        lang_ro='L6_lab_ro.html', lang_en='L6_lab.html'),
    'ro': dict(
        course='Rețele neuronale și deep learning', sub='Cursul 6 · Laborator: preziceți, testați, demonstrați',
        title='Laborator de reprezentări ale cuvintelor', back='← Site-ul cursului', site='../index_ro.html#chapters', pdf='PDF-ul cursului',
        inter='Grafice interactive', inter_href='L6_ro.html',
        lead='Cinci runde despre felul în care textul devine numere. În fiecare rundă mai întîi <b>preziceți</b> ce se va întîmpla '
             'și vă fixați răspunsul, apoi îl <b>testați</b> pe datele cursului și abia apoi vedeți <b>demonstrația</b>. Datele '
             'sînt cele din curs: 400 de discursuri ale Băncii Centrale Europene (2022–2026), pe care ne-am antrenat propriile '
             'modele word2vec și GloVe, și frazele din Financial PhraseBank.',
        steps=('1 Preziceți', '2 Testați', '3 Demonstrați'),
        heads=('Runda 1 · Cîteva cuvinte fac cea mai mare parte a muncii', 'Runda 2 · Cine sînt vecinii?', 'Runda 3 · Aritmetică cu cuvinte',
               'Runda 4 · Media embedding-urilor sau numărarea cuvintelor?', 'Runda 5 · Cînd adnotatorii nu sînt de acord'),
        r1r='Pînă la rangul', r2w='Cuvîntul', r5a='Acordul adnotatorilor',
        hint='Graficele rămîn goale pînă vă fixați predicția. În runda 2, trageți de spațiul cuvintelor ca să-l rotiți; derulați sau ciupiți ca să măriți.',
        foot='Discursurile BCE: 400 de discursuri, 8 ianuarie 2022 – 30 octombrie 2026, 1.289.896 de tokeni. word2vec: skip-gram cu '
             'eșantionare negativă, 100 de dimensiuni, fereastra 5, cuvinte văzute de cel puțin 5 ori; GloVe pe același text; ambele '
             'antrenate în PyTorch. Vederea 3D este o proiecție pe componente principale a celor 32 de cuvinte din patru grupuri '
             'tematice. Financial PhraseBank (Malo et al., 2014): 4.838 de fraze, 5 împărțiri aleatoare; tweet-uri financiare: '
             '3 semințe. Codul: lectures/L6/analysis_L6.py și build_interactive_L6.py pe GitHub.',
        lang_ro='L6_lab_ro.html', lang_en='L6_lab.html'),
}


def page(lang):
    t = L[lang]
    ro = lang == 'ro'
    h, s = t['heads'], t['steps']
    probes = lab_num()['probes']
    sel = (f'<div class="ctl"><label for="r2-w">{t["r2w"]}</label><select id="r2-w">'
           + ''.join(f'<option value="{w}">{w}</option>' for w in probes) + '</select></div>')
    body = '\n'.join([
        rnd('r1', h[0], s, rng('r1-r', t['r1r'], 0, 13, 1, 13), three=False),
        rnd('r2', h[1], s, sel),
        rnd('r3', h[2], s, '', three=False),
        rnd('r4', h[3], s, '', three=False),
        rnd('r5', h[4], s, rng('r5-a', t['r5a'], 0, 4, 1, 3), three=False),
    ])
    num = json.dumps(lab_num(), separators=(',', ':'))
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
<script src="L6_data.js?v={VER}"></script>
<script>window.LAB_NUM = {num};</script>
<script src="L6_lab_text_{lang}.js?v={VER}"></script>
<script type="module" src="L6_lab.js?v={VER}"></script>
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
  <li><a href="../lectures/L6/lecture6{'_ro' if ro else ''}.pdf" target="_blank" rel="noopener">{t["pdf"]}</a></li>
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
    for lang, name in (('en', 'L6_lab.html'), ('ro', 'L6_lab_ro.html')):
        with open(os.path.join(OUT, name), 'w') as f:
            f.write(page(lang))
        print('wrote', name)
