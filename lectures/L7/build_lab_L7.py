"""build_lab_L7.py -- the Lecture 7 lab (interactive/L7_lab.html, L7_lab_ro.html): predict, test, prove.

Five rounds on attention: why the scores are divided by sqrt(d) and which layers focus (the attention of FinBERT on real
finance tweets, drawn in 3D with three.js), the [SEP] sink, positional encoding and pretraining with little labelled data.
Each round asks for a prediction first, then lets the student test it on the course data, then shows the proof. The page
reads interactive/L7_data.js (build_interactive_L7.py) and window.LAB_NUM, written here from L7_numbers.json and
results/attention_entropy.csv of analysis_L7.py; logic in interactive/L7_lab.js and lab_common.js, texts in
interactive/L7_lab_text_{en,ro}.js, style in interactive/lab.css.

usage:  python3 lectures/L7/build_lab_L7.py
"""
import csv
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'L2'))
from build_lab_L2 import rng, rnd, THREE  # noqa: E402  (the round markup is shared with the Lecture 2 lab)

OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'interactive'))
VER = '20261118a'
K = 7
LABEL = {'en': ['bearish', 'bullish', 'neutral'], 'ro': ['pesimist', 'optimist', 'neutru']}


def lab_num():
    n = json.load(open(os.path.join(HERE, 'L7_numbers.json'), encoding='utf-8'))
    rows = list(csv.DictReader(open(os.path.join(HERE, 'results', 'attention_entropy.csv'), encoding='utf-8')))
    return {'on_sep': [round(float(r['on_sep']), 4) for r in rows], 'on_cls': [round(float(r['on_cls']), 4) for r in rows],
            'n_tweets': n['attn']['n_tweets'], 'boot_pos': n['boot']['tiny_nopos'],
            'f1': {m: v['macro_f1'] for m, v in n['models'].items()}, 'gain_pos': n['gain_pos']}


def tweets():
    s = open(os.path.join(OUT, 'L7_data.js'), encoding='utf-8').read()
    d = json.loads(s[s.index('{'):s.rindex('}') + 1])
    return [(a['text'], int(a['label'])) for a in d['att']]


L = {
    'en': dict(
        course='Neural Networks and Deep Learning', sub='Lecture 7 · Lab: predict, test, prove',
        title='Attention lab', back='← Course site', site='../index.html#chapters', pdf='Lecture PDF',
        inter='Interactive charts', inter_href='L7.html',
        lead='Five rounds on how attention works and what it is worth. In each round you first <b>predict</b> what will happen '
             'and lock your answer, then <b>test</b> it on the course data, and only then see the <b>proof</b>. Rounds 1 and 2 '
             'draw the real attention of FinBERT on finance tweets as a 3D landscape you can turn; rounds 3 to 5 use the '
             '500 test tweets and the models of the lecture.',
        steps=('1 Predict', '2 Test', '3 Prove'),
        heads=('Round 1 · Why divide by √d?', 'Round 2 · Where does attention focus?', 'Round 3 · The [SEP] sink',
               'Round 4 · Does word order matter?', 'Round 5 · Little labelled data'),
        r1l='Layer', r1h='Head', r1t='Divide the scores by', r2w='Tweet', r2l='Layer', r5n='Up to n',
        hint='Charts stay empty until you lock a prediction. In rounds 1 and 2, drag the landscape to turn it; scroll or pinch to zoom.',
        foot='Attention: ProsusAI FinBERT (12 layers × 12 heads, d = 64 per head), raw scores q·k recomputed from the model '
             'and checked against its attention; round 2 shows the mean over the 12 heads of a layer. Tweets: Twitter financial '
             'news sentiment, published validation split used as the test set (2,388 tweets); 8,588 training tweets. '
             'Entropy and [SEP] share: 500 test tweets. Small transformer: 3 seeds; bootstrap with 1,000 draws of the test tweets. '
             'Code: lectures/L7/analysis_L7.py and build_interactive_L7.py on GitHub.',
        lang_ro='L7_lab_ro.html', lang_en='L7_lab.html'),
    'ro': dict(
        course='Rețele neuronale și deep learning', sub='Cursul 7 · Laborator: preziceți, testați, demonstrați',
        title='Laborator de atenție', back='← Site-ul cursului', site='../index_ro.html#chapters', pdf='PDF-ul cursului',
        inter='Grafice interactive', inter_href='L7_ro.html',
        lead='Cinci runde despre felul în care funcționează atenția și cît valorează. În fiecare rundă mai întîi <b>preziceți</b> '
             'ce se va întîmpla și vă fixați răspunsul, apoi îl <b>testați</b> pe datele cursului și abia apoi vedeți '
             '<b>demonstrația</b>. Rundele 1 și 2 desenează atenția reală a lui FinBERT pe tweet-uri financiare ca un relief 3D '
             'pe care îl puteți roti; rundele 3–5 folosesc cele 500 de tweet-uri de test și modelele din curs.',
        steps=('1 Preziceți', '2 Testați', '3 Demonstrați'),
        heads=('Runda 1 · De ce împărțim la √d?', 'Runda 2 · Unde se concentrează atenția?', 'Runda 3 · Chiuveta [SEP]',
               'Runda 4 · Contează ordinea cuvintelor?', 'Runda 5 · Puține date etichetate'),
        r1l='Stratul', r1h='Capul', r1t='Împărțim scorurile la', r2w='Tweet-ul', r2l='Stratul', r5n='Pînă la n',
        hint='Graficele rămîn goale pînă vă fixați predicția. În rundele 1 și 2, trageți de relief ca să-l rotiți; derulați sau ciupiți ca să măriți.',
        foot='Atenția: ProsusAI FinBERT (12 straturi × 12 capete, d = 64 pe cap), scorurile brute q·k recalculate din model și '
             'verificate cu atenția lui; runda 2 arată media celor 12 capete ale unui strat. Tweet-uri: Twitter financial news '
             'sentiment, setul de validare publicat folosit ca set de test (2.388 de tweet-uri); 8.588 de tweet-uri de antrenare. '
             'Entropia și ponderea [SEP]: 500 de tweet-uri de test. Transformerul mic: 3 semințe; bootstrap cu 1.000 de extrageri '
             'ale tweet-urilor de test. Codul: lectures/L7/analysis_L7.py și build_interactive_L7.py pe GitHub.',
        lang_ro='L7_lab_ro.html', lang_en='L7_lab.html'),
}


def page(lang):
    t = L[lang]
    ro = lang == 'ro'
    h, s = t['heads'], t['steps']
    opts = ''.join(f'<option value="{i}">{html.escape(tx[:46] + ("…" if len(tx) > 46 else ""))} ({LABEL[lang][lb]})</option>'
                   for i, (tx, lb) in enumerate(tweets()))
    sel = f'<div class="ctl"><label for="r2-w">{t["r2w"]}</label><select id="r2-w">{opts}</select></div>'
    body = '\n'.join([
        rnd('r1', h[0], s, rng('r1-l', t['r1l'], 1, 12, 1, 1) + rng('r1-h', t['r1h'], 1, 12, 1, 1) + rng('r1-t', t['r1t'], 0, 4, 1, 0)),
        rnd('r2', h[1], s, sel + rng('r2-l', t['r2l'], 1, 12, 1, 1)),
        rnd('r3', h[2], s, '', three=False),
        rnd('r4', h[3], s, '', three=False),
        rnd('r5', h[4], s, rng('r5-n', t['r5n'], 0, 4, 1, 4), three=False),
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
<script src="L7_data.js?v={VER}"></script>
<script>window.LAB_NUM = {num};</script>
<script src="L7_lab_text_{lang}.js?v={VER}"></script>
<script type="module" src="L7_lab.js?v={VER}"></script>
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
  <li><a href="../lectures/L7/lecture7{'_ro' if ro else ''}.pdf" target="_blank" rel="noopener">{t["pdf"]}</a></li>
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
    for lang, name in (('en', 'L7_lab.html'), ('ro', 'L7_lab_ro.html')):
        with open(os.path.join(OUT, name), 'w') as f:
            f.write(page(lang))
        print('wrote', name)
