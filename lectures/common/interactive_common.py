"""Shared builder of the interactive lecture pages (NN/interactive/LK.html and LK_ro.html).

Same design as the interactive chapter pages of the MFM course (latex/interactive_common.py there): one block per key idea,
the slide of the lecture on the left, the same figure on the right, recomputed in the browser from small precomputed inputs
when the reader moves a slider, a dropdown or a toggle; under it a readout, a question for the room and its answer, and the
link to the Quantlet that draws the chart.

A lecture spec module (lectures/LK/build_interactive_LK.py) defines
    K          lecture number
    BLOCKS     [(anchor, chart, section key, prefix)]; chart = name of the chart file of the deck (charts/en/<chart>.pdf)
               and of its Quantlet (Quantlets/LK/NN_LK_<chart>); prefix = id prefix of the controls (readout id <prefix>-out)
    SECTIONS   {lang: {section key: title}}
    L          {lang: dict(h1=..., note=..., src=...)}
    widgets(lang) -> {anchor: (controls html, charts html)}
    QA         {lang: {anchor: (question, answer)}}
    TEXT_JS    {lang: JavaScript that defines window.LK_TEXT (readouts)}
    data()     -> dict of inputs, written to interactive/LK_data.js as window.LK_DATA (real course data only)
and calls build(module). Optional: TITLES {lang: {anchor: title}} (used when the slide of a chart is not found).

Rebuild one lecture:   python3 lectures/L3/build_interactive_L3.py            (data + pages + slide images)
                       python3 lectures/L3/build_interactive_L3.py --no-data  (pages and slides only, keeps LK_data.js)
Rebuild all:           python3 lectures/common/interactive_common.py
The page logic is in interactive/LK.js, the shared helpers in interactive/common.js, the style in interactive/interactive.css.
Bump V below after any change of these files (cache-busting query string of every script and stylesheet).
"""
import json
import math
import os
import re
import sys
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(ROOT, 'interactive')
DATA = os.path.join(ROOT, 'data', 'clean')
QLROOT = os.path.join(ROOT, 'Quantlets')
PLOTLY = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'
V = '20261108a'
GH = 'https://github.com/danpele/NEURAL_BIZ'
COLAB = 'https://colab.research.google.com/github/danpele/NEURAL_BIZ/blob/main'

COMMON = {
 'en': dict(html_lang='en', course='Neural Networks and Deep Learning', site='../index.html#chapters', back='Course site',
            pdf_btn='Lecture PDF', ql='Quantlets', slide='Slide', ask='Question for the room', answer='Answer',
            code='Code of this chart', colab='open in Colab', lecture='Lecture'),
 'ro': dict(html_lang='ro', course='Rețele neuronale și deep learning', site='../index_ro.html#chapters',
            back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets', slide='Slide-ul', ask='Întrebare pentru sală',
            answer='Răspuns', code='Codul acestei figuri', colab='deschide în Colab', lecture='Cursul'),
}


# ---------------------------------------------------------------- controls
def slider(i, label, lo, hi, step, val):
    return (f'<div class="ctl"><label for="{i}">{label}: <b id="{i}-v"></b></label>'
            f'<input type="range" id="{i}" min="{lo}" max="{hi}" step="{step}" value="{val}"></div>')


def select(i, label, opts, sel=None):
    o = ''.join(f'<option value="{v}"{" selected" if str(v) == str(sel) else ""}>{t}</option>' for v, t in opts)
    return f'<div class="ctl"><label for="{i}">{label}</label><select id="{i}">{o}</select></div>'


def toggles(i, label, opts, on=None):
    """Group of check boxes (several series on or off); value of each box = its key."""
    on = set(map(str, on if on is not None else [v for v, _ in opts]))
    o = ''.join(f'<label class="tg"><input type="checkbox" name="{i}" value="{v}"{" checked" if str(v) in on else ""}> {t}</label>'
                for v, t in opts)
    return f'<div class="ctl ctl-tg" id="{i}"><span class="lbl">{label}</span><div class="tgs">{o}</div></div>'


def radios(i, label, opts, sel=None):
    o = ''.join(f'<label class="tg"><input type="radio" name="{i}" value="{v}"{" checked" if str(v) == str(sel) else ""}> {t}</label>'
                for v, t in opts)
    return f'<div class="ctl ctl-tg" id="{i}"><span class="lbl">{label}</span><div class="tgs">{o}</div></div>'


def play(i):
    return f'<button class="btn play" id="{i}-play" type="button"></button>'


def charts(*ids, cls=''):
    return f'<div class="charts {cls}">' + ''.join(f'<div class="chart" id="{i}"></div>' for i in ids) + '</div>'


# ---------------------------------------------------------------- data helpers
def rnd(x, k=2):
    """Round numbers (also inside lists / dicts / numpy arrays) for a compact JSON; NaN and inf become null."""
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, np.generic):
            x = x.item()
    except ImportError:
        pass
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        v = round(x, k)
        return int(v) if k <= 0 else v
    if isinstance(x, dict):
        return {str(a): rnd(b, k) for a, b in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, k) for v in x]
    return x


def csv(name, **kw):
    import pandas as pd
    return pd.read_csv(os.path.join(DATA, name), **kw)


def cutoff():
    """Last day of the last complete month covered by the Romanian prices and generation (rule of every analysis_LK.py)."""
    import pandas as pd
    P = csv('power_da_prices_daily_cet.csv', parse_dates=['date_cet']).set_index('date_cet')
    G = csv('generation_ro_mw_daily_cet.csv', usecols=['date_cet', 'hydro_run_of_river'], parse_dates=['date_cet']).set_index('date_cet')
    raw = min(P['RO'].dropna().index.max(), G['hydro_run_of_river'].dropna().index.max())
    return pd.Timestamp(((raw + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)).date())


def ql_dir(k, chart):
    return os.path.join(QLROOT, f'L{k}', f'NN_L{k}_{chart}')


# ---------------------------------------------------------------- slides: page of the deck that shows each chart
def balanced(s, i):
    depth, j = 0, i
    while True:
        c = s[j]
        if c == '\\':
            j += 2
            continue
        depth += c == '{'
        depth -= c == '}'
        j += 1
        if depth == 0:
            return s[i + 1:j - 1], j


def latex_text(t):
    """Plain text of a frame title: link texts kept, formatting commands and $ removed."""
    GREEK = dict(alpha='α', beta='β', gamma='γ', delta='δ', epsilon='ε', varepsilon='ε', theta='θ', kappa='κ', lambda_='λ',
                 mu='μ', nu='ν', pi='π', rho='ρ', sigma='σ', tau='τ', phi='φ', chi='χ', omega='ω', Delta='Δ', Sigma='Σ',
                 times='×', le='≤', leq='≤', ge='≥', geq='≥', approx='≈', neq='≠', infty='∞', to='→', cdot='·', pm='±')
    t = re.sub(r'\\([a-zA-Z]+)', lambda m: GREEK.get(m.group(1), GREEK.get(m.group(1) + '_', m.group(0))), t)
    t = re.sub(r'\\href\{[^{}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', t)
    for _ in range(3):
        t = re.sub(r'\\(?:textbf|textit|emph|mathrm|mathbf|text|textsc|textsf|texttt|boldsymbol|mbox|alert|textcolor\{[^{}]*\})'
                   r'\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', t)
    t = t.replace('\\ ', ' ').replace('---', '—').replace('--', '–').replace(r'\%', '%').replace(r'\&', '&').replace(r'\_', '_')
    t = t.replace(r'\#', '#').replace('~', ' ').replace('``', '“').replace("''", '”').replace(r'\ldots', '…').replace('$', '')
    t = re.sub(r'\\[a-zA-Z]+\s*', '', t).replace('{', '').replace('}', '').replace('^', '').replace('_', '')
    return re.sub(r'\s+', ' ', t).strip()


def frames_of(s):
    """(title, body) of every beamer frame, in order."""
    out = []
    for m in re.finditer(r'\\begin\{frame\}', s):
        i = m.end()
        while s[i:i + 1] in ('[', '<'):
            i = s.index(']' if s[i] == '[' else '>', i) + 1
        title = ''
        if s[i:i + 1] == '{':
            title, i = balanced(s, i)
        out.append((title, s[i:s.index(r'\end{frame}', i)]))
    return out


def _norm(t):
    # PDF text of the decks writes ș / ț as "s," / "t,": compare letters without diacritics
    t = unicodedata.normalize('NFKD', unicodedata.normalize('NFKC', t).lower())
    t = ''.join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r'[^0-9a-zα-ω]+', '', t)


def deck(k, lang):
    return os.path.join(ROOT, 'lectures', f'L{k}', f'lecture{k}' + ('_ro' if lang == 'ro' else ''))


def slide_pages(sp, lang):
    """{chart: (pdf page, frame number shown in the footer, frame title)} for the charts of BLOCKS found in the deck.
    Frames are matched to PDF pages in order, by their title (section divider pages are skipped)."""
    import pymupdf
    base = deck(sp.K, lang)
    if not (os.path.exists(base + '.tex') and os.path.exists(base + '.pdf')):
        return {}
    frames = frames_of(open(base + '.tex', encoding='utf-8').read())
    doc = pymupdf.open(base + '.pdf')
    texts = [_norm(p.get_text()) for p in doc]
    want = {c for _, c, _, _ in sp.BLOCKS}
    res, pg = {}, 0
    for title, body in frames:
        tt = _norm(latex_text(re.sub(r'\s*\(\d/\d\)\s*$', '', title)))[:40]
        if not tt:
            continue
        j = next((j for j in range(pg, len(texts)) if tt in texts[j]), None)
        if j is None:
            continue
        pg = j + 1
        charts_here = set(re.findall(r'includegraphics(?:\[[^\]]*\])?\{charts/(?:en|ro)/([A-Za-z0-9_\-]+)(?:\.\w+)?\}', body))
        for c in charts_here & want:
            if c not in res:
                res[c] = (j + 1, _footer_number(doc[j]) or j + 1, latex_text(re.sub(r'\s*\(\d/\d\)\s*$', '', title)))
    return res


def _footer_number(page):
    """Frame number printed in the top-right corner of a slide (NN headline), if any."""
    h, w = page.rect.height, page.rect.width
    nums = [wd for wd in page.get_text('words') if wd[3] < 0.1 * h and wd[0] > 0.9 * w and wd[4].isdigit()]
    return int(nums[-1][4]) if nums else None


# ---------------------------------------------------------------- page
def script_loader(early, page_js):
    """Load the libraries `early` (Plotly, LK_data.js, LK_text_*.js, common.js) in parallel, then the lecture script once the
    document is parsed and every library has run (a failed request is retried up to 3 times; MFM loader)."""
    js = ("(function (early, page) {\n"
          "  'use strict';\n"
          "  let left = early.length, ok = true, parsed = document.readyState !== 'loading', started = false;\n"
          "  function add(src, k, done, ordered) {\n"
          "    const s = document.createElement('script');\n"
          "    s.src = k ? src + (src.indexOf('?') < 0 ? '?' : '&') + 'retry=' + k : src;\n"
          "    s.async = !ordered;\n"
          "    s.onload = () => done(true);\n"
          "    s.onerror = () => { s.remove(); if (k < 3) setTimeout(() => add(src, k + 1, done, ordered), 300 * (k + 1)); else done(false); };\n"
          "    document.head.appendChild(s);\n"
          "  }\n"
          "  function go() {\n"
          "    if (started || left || !parsed) return;\n"
          "    started = true;\n"
          "    if (!ok || !window.NNI) { console.error('NN: a script of the interactive charts did not load; the live charts are off.'); return; }\n"
          "    add(page, 0, () => {}, true);\n"
          "  }\n"
          "  early.forEach(src => add(src, 0, r => { ok = ok && r; left--; go(); }, false));\n"
          "  if (!parsed) document.addEventListener('DOMContentLoaded', () => { parsed = true; go(); });\n"
          "  go();\n"
          f"}})({json.dumps(list(early))}, {json.dumps(page_js)});")
    return f'<link rel="preload" as="script" href="{page_js}">\n<script>\n{js}\n</script>'


def page(sp, lang, P):
    c, t, W = COMMON[lang], sp.L[lang], sp.widgets(lang)
    k = sp.K
    pdf = f'../lectures/L{k}/lecture{k}' + ('_ro' if lang == 'ro' else '') + '.pdf'
    titles = getattr(sp, 'TITLES', {}).get(lang, {})
    body, nav, sec = [], [], None
    for anchor, chart, skey, pfx in sp.BLOCKS:
        if skey != sec:
            sec = skey
            body.append(f'<h2 class="sec" id="s-{skey}">{sp.SECTIONS[lang][skey]}</h2>')
            nav.append(f'<a href="#s-{skey}">{sp.SECTIONS[lang][skey]}</a>')
        ctl, ch = W[anchor]
        q, a = sp.QA[lang][anchor]
        qn = f'NN_L{k}_{chart}'
        code = (f'<p class="qlk">{c["code"]}: <a href="{GH}/tree/main/Quantlets/L{k}/{qn}" target="_blank" rel="noopener">{qn}</a>'
                f' · <a href="{COLAB}/Quantlets/L{k}/{qn}/{qn}.ipynb" target="_blank" rel="noopener">{c["colab"]}</a></p>')
        if chart in P:
            pg, num, title = P[chart]
            title = titles.get(anchor, title)
            fig = (f'<figure class="slide"><a href="{pdf}#page={pg}" target="_blank" rel="noopener"><img src="slides/L{k}_{chart}_{lang}.jpg" '
                   f'alt="{c["slide"]} {num}: {title}" loading="lazy"></a>\n      <figcaption>{c["slide"]} {num}</figcaption></figure>')
            pair = 'pair'
        else:
            title, fig, pair = titles.get(anchor, anchor), '', 'pair solo'
        body.append(f'''
<section class="blk" id="{anchor}">
  <h3>{title}</h3>
  <div class="{pair}">
    {fig}
    <div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{pfx}-out"></ul>{code}</div>
  </div>
  <div class="ask"><span class="ask-tag">{c["ask"]}</span> {q}</div>
  <details><summary>{c["answer"]}</summary><p>{a}</p></details>
</section>''')
    early = [PLOTLY, f'L{k}_data.js?v={V}', f'L{k}_text_{lang}.js?v={V}', f'common.js?v={V}']
    return f'''<!DOCTYPE html>
<html lang="{c["html_lang"]}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{t["h1"]}</title>
<meta name="description" content="{c["course"]}: {t["h1"]}">
<link rel="icon" href="../assets/logos/ase_logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css?v={V}">
<link rel="stylesheet" href="interactive.css?v={V}">
<script>window.NN_LANG = '{lang}';</script>
{script_loader(early, f"L{k}.js?v={V}")}
</head>
<body>
<header>
  <div class="header-inner">
    <a href="https://www.ase.ro" target="_blank" rel="noopener"><img class="header-logo" src="../assets/logos/ase_logo.png" alt="ASE"></a>
    <div class="header-title"><h1><a href="{c["site"].split('#')[0]}">{c["course"]}</a></h1><p>{t["h1"]}</p></div>
    <a href="https://ipe.ro/new/" target="_blank" rel="noopener"><img class="header-logo" src="../assets/logos/acad_logo.png" alt="{'Academia Română, Institutul de Prognoză Economică' if lang == 'ro' else 'Romanian Academy, Institute for Economic Forecasting'}"></a>
  </div>
</header>
<nav><ul>
  <li><a href="{c["site"]}">← {c["back"]}</a></li>
  <li><a href="{pdf}" target="_blank" rel="noopener">{c["pdf_btn"]}</a></li>
  <li><a href="{GH}/tree/main/Quantlets/L{k}" target="_blank" rel="noopener">{c["ql"]}</a></li>
  <li class="lang-switcher"><a href="L{k}.html" class="lang-btn{" active" if lang == "en" else ""}">EN</a><a href="L{k}_ro.html" class="lang-btn{" active" if lang == "ro" else ""}">RO</a></li>
</ul></nav>
<main class="imain">
<p class="lead">{t["note"]}</p>
<p class="toc">{" · ".join(nav)}</p>
{"".join(body)}
<p class="src foot">{t["src"]}</p>
</main>
</body>
</html>
'''


def write_data(sp):
    d = sp.data()
    s = json.dumps(d, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    with open(os.path.join(OUT, f'L{sp.K}_data.js'), 'w', encoding='utf-8') as f:
        f.write(f'// Inputs of the Lecture {sp.K} interactive page, computed from the course data by '
                f'lectures/L{sp.K}/build_interactive_L{sp.K}.py.\nwindow.LK_DATA = {s};\n')
    print(f'wrote L{sp.K}_data.js ({len(s) / 1024:.0f} KB)')


def build(sp, argv=None):
    import pymupdf
    argv = sys.argv[1:] if argv is None else argv
    os.makedirs(os.path.join(OUT, 'slides'), exist_ok=True)
    if '--no-data' not in argv or not os.path.exists(os.path.join(OUT, f'L{sp.K}_data.js')):
        write_data(sp)
    for lang in ('en', 'ro'):
        P = slide_pages(sp, lang)
        missing = [c for _, c, _, _ in sp.BLOCKS if c not in P]
        if missing:
            print(f'  L{sp.K} {lang}: no slide found for {missing} (block shown without slide)')
        name = f'L{sp.K}' + ('_ro' if lang == 'ro' else '') + '.html'
        with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
            f.write(page(sp, lang, P))
        with open(os.path.join(OUT, f'L{sp.K}_text_{lang}.js'), 'w', encoding='utf-8') as f:
            f.write(f'// Readouts of the Lecture {sp.K} interactive page, written by lectures/L{sp.K}/build_interactive_L{sp.K}.py.\n'
                    + sp.TEXT_JS[lang])
        if P:
            doc = pymupdf.open(deck(sp.K, lang) + '.pdf')
            for chart, (pg, _, _) in P.items():
                doc[pg - 1].get_pixmap(dpi=110).save(os.path.join(OUT, 'slides', f'L{sp.K}_{chart}_{lang}.jpg'), jpg_quality=80)
        print('wrote', name)


if __name__ == '__main__':
    import subprocess
    for kk in range(1, 8):
        f = os.path.join(ROOT, 'lectures', f'L{kk}', f'build_interactive_L{kk}.py')
        if os.path.exists(f):
            subprocess.run([sys.executable, f] + sys.argv[1:], check=True)
