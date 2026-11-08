#!/usr/bin/env python3
"""Build assets/showcase.js: the chart carousel of each lecture card on the course site.

Scans Quantlets/L1..L14/NN_LK_<chart>/<chart>.png (English labels). For every chart:
  - caption (EN and RO) = title of the slide that shows the chart in
    lectures/LK/lectureK.tex and lectures/LK/lectureK_ro.tex, so the site uses the
    wording of the slides; if the chart is not on a slide, the Quantlet title
    (first "**title**" of the notebook) or the Metainfo description, shortened;
  - order = order of the slides (charts not on a slide come last, alphabetically);
  - link = the Quantlet folder on GitHub.
Rerun after adding or renaming Quantlets:  python3 assets/make_showcase.py
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH = 'https://github.com/danpele/NEURAL_BIZ/tree/main/'
MAXLEN = 160


def read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except OSError:
        return ''


def brace_arg(s, i):
    """s[i] == '{' -> (content, index after the closing brace)."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def frame_titles(tex):
    """{chart stem: (position, frame title)} for every chart included in a frame."""
    out = {}
    for m in re.finditer(r'\\begin\{frame\}', tex):
        start = m.end()
        end = tex.find('\\end{frame}', start)
        body = tex[start:end if end > 0 else len(tex)]
        k = 0
        while k < len(body) and body[k] == '[':          # frame options
            k = body.find(']', k) + 1
        title = ''
        if k < len(body) and body[k] == '{':
            title, _ = brace_arg(body, k)
        else:
            ft = re.search(r'\\frametitle\{', body)
            if ft:
                title, _ = brace_arg(body, ft.end() - 1)
        for g in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', body):
            stem = os.path.splitext(os.path.basename(g.group(1)))[0]
            out.setdefault(stem, (m.start(), title))
    return out


MATH = {r'\le': '≤', r'\leq': '≤', r'\ge': '≥', r'\geq': '≥', r'\times': '×', r'\approx': '≈',
        r'\alpha': 'α', r'\beta': 'β', r'\sigma': 'σ', r'\Delta': 'Δ', r'\to': '→', r'\cdot': '·',
        r'\pm': '±', r'\neq': '≠', r'\ne': '≠', r'\infty': '∞', r'\tau': 'τ', r'\lambda': 'λ'}


def clean(t):
    t = t.replace('{,}', ',').replace('{.}', '.')
    t = re.sub(r'\\(?:textbf|textit|emph|text|mathrm|textsc|mbox)\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\(?:alert|structure|textcolor\{[^}]*\})\{([^{}]*)\}', r'\1', t)
    for k in sorted(MATH, key=len, reverse=True):
        t = re.sub(re.escape(k) + r'(?![A-Za-z])', MATH[k], t)
    t = t.replace(r'\%', '%').replace(r'\&', '&').replace(r'\$', '$').replace(r'\_', '_')
    t = t.replace('---', '—').replace('--', '–').replace('~', '\u00a0').replace(r'\,', '\u202f')
    t = t.replace(r'\ ', ' ').replace("``", '“').replace("''", '”')
    t = re.sub(r'\$([^$]*)\$', r'\1', t)
    t = re.sub(r'\^\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'_\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\[A-Za-z]+\*?', '', t)
    t = t.replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', t).strip()


def shorten(t):
    if len(t) <= MAXLEN:
        return t
    for sep in [': ', ' - ', '; ', ' (']:
        i = t.find(sep, 25)
        if 25 <= i <= MAXLEN:
            return t[:i].rstrip(' ,;:')
    m = re.search(r', (?:\d{4}|from|to|by|in|with)\b', t)
    if m and 25 <= m.start() <= MAXLEN:
        return t[:m.start()]
    cut = t[:MAXLEN].rsplit(' ', 1)[0].rstrip(' ,;:')
    return cut + '…'


def quantlet_title(folder):
    nbs = sorted(glob.glob(os.path.join(folder, '*.ipynb')))
    if nbs:
        try:
            src = ''.join(json.loads(read(nbs[0]))['cells'][0]['source'])
            m = re.search(r'\*\*(.+?)\*\*', src)
            if m:
                return m.group(1).strip()
        except (ValueError, KeyError, IndexError):
            pass
    m = re.search(r"Description:\s*'([^']*)'", read(os.path.join(folder, 'Metainfo.txt')))
    if m:
        return re.split(r'(?<=\.)\s', m.group(1).strip())[0].rstrip('.')
    return os.path.basename(folder)


def main():
    charts = {}
    for k in range(1, 15):
        lec = os.path.join(ROOT, 'lectures', f'L{k}')
        en_t = frame_titles(read(os.path.join(lec, f'lecture{k}.tex')))
        ro_t = frame_titles(read(os.path.join(lec, f'lecture{k}_ro.tex')))
        items = []
        for folder in sorted(glob.glob(os.path.join(ROOT, 'Quantlets', f'L{k}', f'NN_L{k}_*'))):
            if not os.path.isdir(folder):
                continue
            rel = os.path.relpath(folder, ROOT).replace(os.sep, '/')
            for png in sorted(glob.glob(os.path.join(folder, '*.png'))):
                stem = os.path.splitext(os.path.basename(png))[0]
                qt = shorten(clean(quantlet_title(folder)))
                en = en_t.get(stem)
                ro = ro_t.get(stem)
                items.append({
                    'pos': en[0] if en else (ro[0] if ro else 10 ** 9),
                    'src': rel + '/' + os.path.basename(png),
                    'url': GH + rel,
                    'q': os.path.basename(folder),
                    'en': shorten(clean(en[1])) if en and en[1] else qt,
                    'ro': shorten(clean(ro[1])) if ro and ro[1] else qt,
                    'onSlide': bool(en or ro),
                })
        items.sort(key=lambda d: (d['pos'], d['q']))
        for d in items:
            d.pop('pos')
        charts[f'l{k}'] = items
        missing = [d['q'] for d in items if not d['onSlide']]
        print(f'L{k}: {len(items)} charts' + (f'; not on a slide (Quantlet title used): {", ".join(missing)}' if missing else ''))
    for v in charts.values():
        for d in v:
            d.pop('onSlide')

    js = ('// ============================================================\n'
          '// NN course website - chart carousel of each lecture card.\n'
          '// GENERATED by assets/make_showcase.py from Quantlets/LK/NN_LK_*/ - do not edit by hand.\n'
          '// Captions are the titles of the slides that show each chart (EN and RO decks).\n'
          '// ============================================================\n'
          '(function () {\n'
          "    'use strict';\n"
          '    window.NN_DATA.lectureCharts = ' + json.dumps(charts, ensure_ascii=False, indent=1) + ';\n'
          '})();\n')
    with open(os.path.join(ROOT, 'assets', 'showcase.js'), 'w', encoding='utf-8') as f:
        f.write(js)
    print('wrote assets/showcase.js:', sum(len(v) for v in charts.values()), 'charts')


if __name__ == '__main__':
    main()
