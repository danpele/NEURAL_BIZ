"""Update assets/course-data.js from the lectures: titles, topics and "further" links of each card (lectures/Lk/CARD.json,
status 'ready'; the verified videos and Quantinar lists are kept), the bibliography (every reference of REFS in
lectures/Lk/build_Lk.py, listed once under the first lecture that cites it, sorted, LaTeX converted to HTML) and the
abbreviations (ABBR of every deck, most frequent definition, merged with the site-only entries).

usage (from the repository root):  python3 tools/update_course_data.py
"""
import collections
import html
import importlib.util
import json
import os
import re
import subprocess
import sys
import unicodedata

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'lectures', 'common'))
import build_web_lecture as b  # noqa: E402

DUMP = r'''
import sys, os, json, importlib.util
k = int(sys.argv[1]); root = sys.argv[2]
d = os.path.join(root, 'lectures', f'L{k}')
sys.path.insert(0, d); os.chdir(d)
spec = importlib.util.spec_from_file_location(f'build_L{k}', os.path.join(d, f'build_L{k}.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(json.dumps(dict(refs=list(m.REFS), abbr={a: list(v) for a, v in m.ABBR.items()}), ensure_ascii=False))
'''


def deck_data(k):
    """REFS and ABBR of lectures/Lk/build_Lk.py, evaluated in a separate process (the builders share module names)."""
    r = subprocess.run([sys.executable, '-c', DUMP, str(k), ROOT], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


DECKS = {k: deck_data(k) for k in range(1, 15)}

def conv(r):
    t = b.tex_text(r)
    t = re.sub(r"`([^`']*)'", r"‘\1’", t)
    t = re.sub(r"(?<=\w)'(?=\w)", "’", t)
    t = t.replace('<a href=', '<a href=')
    return t
def key(r):
    m = re.search(r'doi\.org/([^}\s]+)\}', r) or re.search(r'arxiv\.org/abs/([^}\s]+)\}', r)
    if m: return m.group(1).lower().rstrip('/')
    plain = re.sub(r'<[^>]+>', '', html.unescape(conv(r)))
    m = re.search(r'\(\d{4}[a-z]?\)\.\s*(.{0,60})', plain)
    return unicodedata.normalize('NFKD', (m.group(1) if m else plain)).lower()
seen = {}; groups = []
for k in range(1, 15):
    d = DECKS[k]
    items = []
    for r in d['refs']:
        kk = key(r)
        if kk in seen:
            k0, r0 = seen[kk]
            a = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', conv(r0))))
            c = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', conv(r))))
            if a != c: print(f'DIFF L{k0} vs L{k}:\n  {a}\n  {c}', file=sys.stderr)
            continue
        seen[kk] = (k, r)
        items.append(conv(r))
    items.sort(key=lambda x: unicodedata.normalize('NFKD', html.unescape(re.sub(r'<[^>]+>', '', x))).encode('ascii', 'ignore').decode().lower())
    groups.append((k, items))

P = 'assets/course-data.js'
s = open(P, encoding='utf-8').read()
J = lambda x: json.dumps(x, ensure_ascii=False)
amp = lambda t: re.sub(r'&(?!amp;|lt;|gt;|quot;|#\d+;|[a-z]+;)', '&amp;', t)
cards = {k: json.load(open(f'lectures/L{k}/CARD.json', encoding='utf-8')) for k in range(1, 15)}

# 1. titles
tl = ''.join(f"        {k}: {{ en: {J(cards[k]['title']['en'])}, ro: {J(cards[k]['title']['ro'])} }}{',' if k < 14 else ''}\n" for k in range(1, 15))
s, n = re.subn(r"(    const TITLES = \{\n)(.*?)(    \};)", lambda m: m.group(1) + tl + m.group(3), s, flags=re.S); assert n == 1

# 2. lecture objects
for k in range(1, 15):
    pat = re.compile(r"            \{\n                id: 'l%d', num: %d,.*?\n            \}(?=,?\n)" % (k, k), re.S)
    m = pat.search(s); assert m, k
    obj = m.group(0)
    vid = re.search(r"^                videos: .*$", obj, re.M).group(0)
    qn = re.search(r"^                quantinar: .*$", obj, re.M).group(0)
    c = cards[k]
    inter = ' interactive: true,' if os.path.exists(os.path.join(ROOT, 'interactive', f'L{k}.html')) else ''
    fur = []
    for f in c['further']:
        en = f.get('en') or f.get('label') or f.get('title')
        ro = f.get('ro') or en
        fur.append(f"{{ en: {J(amp(en))}, ro: {J(amp(ro))}, url: {J(f['url'])} }}")
    new = (f"            {{\n                id: 'l{k}', num: {k}, status: 'ready', files: files({k}),{inter}\n{vid}\n{qn}\n"
           f"                title: TITLES[{k}],\n                topics: {{\n"
           f"                    en: {J([amp(t) for t in c['topics']['en']])},\n"
           f"                    ro: {J([amp(t) for t in c['topics']['ro']])}\n                }},\n"
           f"                further: [\n" + ',\n'.join('                    ' + x for x in fur) + "\n                ]\n            }")
    s = s[:m.start()] + new + s[m.end():]

# 3. bibliography (+ one group with the papers replicated in the research case studies, from cases/common/build_web_case.py CASE)
bl = ',\n'.join(f"            {{ h: lec({k}), items: [\n" + ',\n'.join('                ' + J(x) for x in items) + "] }" for k, items in groups)
sys.path.insert(0, os.path.join(ROOT, 'cases', 'common'))
import build_web_case as bwc  # noqa: E402


def case_ref(c):
    v = c['venue']
    v = v if v.startswith(('submitted', 'SSRN')) else f'<em>{html.escape(v, quote=False)}</em>'
    v = v.replace('SSRN ', 'SSRN Working Paper ') if v.startswith('SSRN') else v[:1].upper() + v[1:]
    return (f"{html.escape(c['authors'], quote=False).replace(' & ', ' &amp; ')} ({c['year']}). {html.escape(c['title'], quote=False)}. {v}. "
            f"<a href=\"{c['link']}\" target=\"_blank\" rel=\"noopener\">{html.escape(c['linktext'], quote=False)}</a>")


cs = [case_ref(bwc.CASE[k]) for k in sorted(bwc.CASE) if bwc.CASE[k].get('paper', True)]
if cs:
    n_ = len(bwc.CASE)
    bl += (",\n            { h: { en: 'Case studies C1–C%d: the papers replicated', ro: 'Studiile de caz C1–C%d: lucrările replicate' }, items: [\n" % (n_, n_)
           + ',\n'.join('                ' + J(x) for x in cs) + "] }")
s, n = re.subn(r"        // Bibliography by lecture:[^\n]*\n(?:        //[^\n]*\n)*        bibliography: \[\n.*?\n        \],\n",
               lambda m: "        // Bibliography by lecture: every reference cited in the slides (REFS of lectures/Lk/build_Lk.py), listed once under\n"
                         "        // the first lecture that cites it, sorted alphabetically; converted from LaTeX by tools/update_course_data.py.\n"
                         "        bibliography: [\n" + bl + "\n        ],\n", s, flags=re.S); assert n == 1

# 4. abbreviations
old = re.search(r"        abbreviations: \[\n(.*?)\n        \],\n", s, re.S)
site = {}
for row in re.findall(r"\[(.*?)\](?=,?\n)", old.group(1)):
    v = json.loads('[' + re.sub(r"'((?:[^'\\]|\\.)*)'", lambda m: J(m.group(1).replace("\\'", "'")), row) + ']')
    site[v[0]] = v
defs = collections.defaultdict(list)
for k in range(1, 15):
    for a, (en, ro) in DECKS[k]['abbr'].items():
        if en is None or ro is None:
            continue
        defs[a].append((html.unescape(b.tex_text(en)), html.unescape(b.tex_text(ro))))
rows = {}
for a, lst in defs.items():
    cnt = collections.Counter(lst)
    best = max(cnt, key=lambda d: (cnt[d], -lst.index(d)))
    rows[a] = [a, best[0][:1].upper() + best[0][1:] if best[0][:1].islower() else best[0], best[1][:1].upper() + best[1][1:] if best[1][:1].islower() else best[1]]
if 'DM' in rows:
    rows['DM'] = ['DM', 'Diebold–Mariano test; also Deutsche Mark (currency of the German credit data)', 'Testul Diebold–Mariano; de asemenea, marca germană (moneda din datele German credit)']
for a in ('ASE', 'IPE'):
    if a in site: rows[a] = site[a]
for a, v in site.items():
    parts = [p.strip() for p in a.split('/')]
    if a in rows or (len(parts) > 1 and all(p in rows for p in parts)):
        continue
    rows[a] = v
rows['EU / UE'] = ['EU / UE', 'European Union', 'Uniunea Europeană']
rows['ECB / BCE'] = ['ECB / BCE', 'European Central Bank', 'Banca Centrală Europeană']
out = []
for a in sorted(rows, key=lambda x: unicodedata.normalize('NFKD', x).lower()):
    v = rows[a]
    v = v if len(v) == 3 and v[2] and v[2] != v[1] else v[:2]
    out.append('            ' + J([amp(x) for x in v]))
s = s[:old.start()] + "        abbreviations: [\n" + ',\n'.join(out) + "\n        ],\n" + s[old.end():]
open(P, 'w', encoding='utf-8').write(s)
print('lectures ok; refs', sum(len(i) for _, i in groups), '; abbreviations', len(out))
