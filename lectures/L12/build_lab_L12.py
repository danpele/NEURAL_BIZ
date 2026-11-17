"""build_lab_L12.py -- the Lecture 12 lab (interactive/L12_lab.html, L12_lab_ro.html): predict, test, prove.

Five rounds on large language models: temperature on the real next-token distribution of Qwen2.5-1.5B (drawn in 3D with
three.js), zero-shot classification by model size, few-shot examples, the wording of the prompt and retrieval for
retrieval-augmented generation (RAG) on the ECB speeches. Each round asks for a prediction first, then lets the student
test it on the course data, then shows the proof. The page reads interactive/L12_data.js (build_interactive_L12.py) and
window.LAB_NUM, written here from L12_numbers.json of analysis_L12.py; logic in interactive/L12_lab.js and lab_common.js,
texts in interactive/L12_lab_text_{en,ro}.js, style in interactive/lab.css.

usage:  python3 lectures/L12/build_lab_L12.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'L2'))
from build_lab_L2 import rng, rnd, THREE  # noqa: E402  (the round markup is shared with the Lecture 2 lab)

OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'interactive'))
VER = '20261118c'


def lab_num():
    n = json.load(open(os.path.join(HERE, 'L12_numbers.json'), encoding='utf-8'))
    return {'next': n['next'], 'cls': n['cls'], 'rag': n['rag'], 'vocab': n['tok']['vocab_qwen']}


L = {
    'en': dict(
        course='Neural Networks and Deep Learning', sub='Lecture 12 · Lab: predict, test, prove',
        title='Language model lab', back='← Course site', site='../index.html#chapters', pdf='Lecture PDF',
        inter='Interactive charts', inter_href='L12.html',
        lead='Five rounds on what open language models can and cannot do with financial text. In each round you first '
             '<b>predict</b> what will happen and lock your answer, then <b>test</b> it on the course data, and only then see '
             'the <b>proof</b>. Every answer comes from Qwen2.5 models run locally on the lecture data: Financial PhraseBank, '
             'finance tweets and the speeches of the European Central Bank (ECB).',
        steps=('1 Predict', '2 Test', '3 Prove'),
        heads=('Round 1 · Temperature', 'Round 2 · How big must the model be?', 'Round 3 · Do examples help?',
               'Round 4 · Does the wording matter?', 'Round 5 · Finding the right document'),
        r1t='Temperature T', r4d='Data', r4o=('PhraseBank', 'finance tweets'), r5k='Up to k',
        hint='Charts stay empty until you lock a prediction. In round 1, drag the ridge to turn it; scroll or pinch to zoom.',
        foot='Models: Qwen2.5-0.5B, 1.5B and 3B-Instruct (Qwen Team, 2024), run locally on an Apple-silicon laptop; no paid '
             'application programming interface. Financial PhraseBank (Malo et al., 2014), sentences with full agreement: '
             '1,581 for training, 678 for testing; finance tweets: 900 test tweets. Retrieval: 400 ECB speeches (2022–2026) in '
             '8,400 chunks of 200 words, dense embeddings bge-small-en-v1.5; question answering on the speeches from 2025 on. '
             'Code: lectures/L12/analysis_L12.py and build_interactive_L12.py on GitHub.',
        lang_ro='L12_lab_ro.html', lang_en='L12_lab.html'),
    'ro': dict(
        course='Rețele neuronale și deep learning', sub='Cursul 12 · Laborator: preziceți, testați, demonstrați',
        title='Laborator de modele de limbaj', back='← Site-ul cursului', site='../index_ro.html#chapters', pdf='PDF-ul cursului',
        inter='Grafice interactive', inter_href='L12_ro.html',
        lead='Cinci runde despre ce pot și ce nu pot face modelele de limbaj deschise cu textele financiare. În fiecare rundă mai '
             'întîi <b>preziceți</b> ce se va întîmpla și vă fixați răspunsul, apoi îl <b>testați</b> pe datele cursului și abia '
             'apoi vedeți <b>demonstrația</b>. Fiecare răspuns provine de la modele Qwen2.5 rulate local pe datele cursului: '
             'Financial PhraseBank, tweet-uri financiare și discursurile Băncii Centrale Europene (BCE).',
        steps=('1 Preziceți', '2 Testați', '3 Demonstrați'),
        heads=('Runda 1 · Temperatura', 'Runda 2 · Cît de mare trebuie să fie modelul?', 'Runda 3 · Ajută exemplele?',
               'Runda 4 · Contează formularea?', 'Runda 5 · Găsirea documentului potrivit'),
        r1t='Temperatura T', r4d='Datele', r4o=('PhraseBank', 'tweet-uri financiare'), r5k='Pînă la k',
        hint='Graficele rămîn goale pînă vă fixați predicția. În runda 1, trageți de relief ca să-l rotiți; derulați sau ciupiți ca să măriți.',
        foot='Modele: Qwen2.5-0.5B, 1.5B și 3B-Instruct (Qwen Team, 2024), rulate local pe un laptop Apple silicon; fără nicio '
             'interfață de programare plătită. Financial PhraseBank (Malo et al., 2014), frazele cu acord complet: 1.581 pentru '
             'antrenare, 678 pentru test; tweet-uri financiare: 900 de tweet-uri de test. Regăsirea: 400 de discursuri BCE '
             '(2022–2026) în 8.400 de fragmente de 200 de cuvinte, embedding-uri dense bge-small-en-v1.5; întrebările despre '
             'discursurile din 2025 încoace. Codul: lectures/L12/analysis_L12.py și build_interactive_L12.py pe GitHub.',
        lang_ro='L12_lab_ro.html', lang_en='L12_lab.html'),
}


def page(lang):
    t = L[lang]
    ro = lang == 'ro'
    h, s = t['heads'], t['steps']
    tg = (f'<div class="ctl ctl-tg"><span class="lbl">{t["r4d"]}</span><div class="tgs">'
          f'<label class="tg"><input type="radio" name="r4-d" value="pb" checked> {t["r4o"][0]}</label>'
          f'<label class="tg"><input type="radio" name="r4-d" value="tw"> {t["r4o"][1]}</label></div></div>')
    body = '\n'.join([
        rnd('r1', h[0], s, rng('r1-t', t['r1t'], 0, 5, 1, 1)),
        rnd('r2', h[1], s, '', three=False),
        rnd('r3', h[2], s, '', three=False),
        rnd('r4', h[3], s, tg, three=False),
        rnd('r5', h[4], s, rng('r5-k', t['r5k'], 1, 10, 1, 10), three=False),
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
<script src="L12_data.js?v={VER}"></script>
<script>window.LAB_NUM = {num};</script>
<script src="L12_lab_text_{lang}.js?v={VER}"></script>
<script type="module" src="L12_lab.js?v={VER}"></script>
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
  <li><a href="../lectures/L12/lecture12{'_ro' if ro else ''}.pdf" target="_blank" rel="noopener">{t["pdf"]}</a></li>
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
    for lang, name in (('en', 'L12_lab.html'), ('ro', 'L12_lab_ro.html')):
        with open(os.path.join(OUT, name), 'w') as f:
            f.write(page(lang))
        print('wrote', name)
