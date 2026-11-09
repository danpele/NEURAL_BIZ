"""Web version of an NN lecture: one scrollable HTML page per language (same layout as the MFM web lectures).

usage:  cd lectures/common && python3 build_web_lecture.py 3          # lecture 3, EN + RO
        python3 build_web_lecture.py all                              # all lectures that have a text file
        python3 build_web_lecture.py 3 --frames                       # list the deck's frames: PDF page, hash, title, charts
writes  web/L{K}.html (EN), web/L{K}_ro.html (RO), web/img/L{K}/<chart>_ro.png (RO charts, rendered from charts/ro/<chart>.pdf)
        and a log web/L{K}_log.txt (missing charts, stale frames, numbers not found in the deck, Romanian spelling checks).

The page text is written by hand in lectures/L{K}/web_en.md and web_ro.md. The deck (lecture{K}.tex, lecture{K}_ro.tex) supplies
the title, the chart list with each chart's source line and Quantlet, the PDF page of every frame and chart, and the references.
Format of web_{en,ro}.md:
  <!-- ... -->                   comment
  ## @intro                      (before the first #) opener: paragraphs, @chart lines, and
  @goals                         a bulleted list of learning outcomes (lines "- ..."), ended by a blank line
  # <section title>              a section (h2, entry of the table of contents); text up to the first ## is its introduction
  ## <title> {h=<hash>}          a block (h3) that explains one slide or a group of slides; {h=...} is optional: the hash of the
                                 frame it follows (--frames prints them); when that frame changes, the log says "text out of date"
                                 and the block gets a "Slide n in the PDF" link from the hash
  ## @end                        summary box of the current section
Inside a block:
  paragraphs separated by blank lines; $...$ inline math, $$...$$ display math (may span lines), `$$ ... $$ {#eq:label}` numbered
  (K.n) and referenced with @eq:label; **bold**, *italic*, [text](url); lists: lines "- " (bullets) or "1. " (numbered),
  "  - " nested; a line that is not an item continues the item above.
  @chart <name>                  the chart of the deck (charts/en/<name>.pdf); the following lines up to a blank line are its
                                 caption. EN shows the Quantlet PNG, RO the RO chart rendered to PNG. Source line, Quantlet link
                                 and PDF page are added from the deck.
  @practice <text>               green "In practice" box (the deck's \\manager line), text up to the next blank line
  | a | b |                      a table (first row = header, second row |---|---|)
Numbers: every number in the text should exist in the deck or in L{K}_numbers.json; the log lists those that do not.
"""
import hashlib
import html
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'web')
REPO = 'https://github.com/danpele/NEURAL_BIZ'
COLAB = 'https://colab.research.google.com/github/danpele/NEURAL_BIZ/blob/main/'
MATHJAX = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js'
VER = '20261109m'

L = {
    'en': dict(course='Neural Networks and Deep Learning', lecture='Lecture', back='Course site', site='../index.html#chapters',
               pdf='Slides (PDF)', nb='Notebook', colab='Open in Colab', ql='Quantlets', toc='Contents', intro='Introduction',
               goals='What you will be able to do', plan='Lecture plan', synth='Section summary', practice='In practice',
               src='Source', abbr='Abbreviations', slide='Slide {n} in the PDF', bib='References',
               videos='Videos: the intuition behind the mathematics (in English)', author='Prof. dr. Daniel Traian PELE',
               inst=['Bucharest University of Economic Studies (ASE), Department of Statistics and Econometrics',
                     'Romanian Academy, Institute for Economic Forecasting (IPE)'],
               foot='Text written for the web version of the lecture; numbers, charts and sources come from the slides and their Quantlets.',
               chart_alt='Chart'),
    'ro': dict(course='Rețele neuronale și deep learning', lecture='Cursul', back='Site-ul cursului', site='../index_ro.html#chapters',
               pdf='Slide-uri (PDF)', nb='Notebook', colab='Deschide în Colab', ql='Quantlets', toc='Cuprins', intro='Introducere',
               goals='Ce veți ști să faceți', plan='Planul cursului', synth='Sinteza secțiunii', practice='În practică',
               src='Sursa', abbr='Abrevieri', slide='Slide-ul {n} în PDF', bib='Bibliografie',
               videos='Video: intuiția din spatele matematicii (în engleză)', author='Prof. dr. Daniel Traian PELE',
               inst=['Academia de Studii Economice din București (ASE), Departamentul de Statistică și Econometrie',
                     'Academia Română, Institutul de Prognoză Economică (IPE)'],
               foot='Text scris pentru versiunea web a cursului; cifrele, graficele și sursele provin din slide-uri și din Quantlet-urile lor.',
               chart_alt='Grafic'),
}


class Stop(Exception):
    pass


# ------------------------------------------------------------------ LaTeX helpers
def strip_comments(s):
    out = []
    for line in s.split('\n'):
        m = re.search(r'(?<!\\)%', line)
        out.append(line if not m else line[:m.start()])
    return '\n'.join(out)


def group(s, i):
    """Content of the {...} group starting at s[i] (after whitespace) and the index after it."""
    while i < len(s) and s[i] in ' \t\n':
        i += 1
    if i >= len(s) or s[i] != '{':
        return None, i
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def macro_args(s, name):
    """All first arguments of \\name{...} in s."""
    out, i = [], 0
    while True:
        m = re.compile(r'\\' + name + r'(?![a-zA-Z])').search(s, i)
        if not m:
            return out
        g, j = group(s, m.end())
        if g is not None:
            out.append(g)
        i = max(j, m.end())


def tex_text(t, links=True):
    """Plain/HTML text of a short LaTeX string (chart sources, references, titles)."""
    t = t.replace('\\&', '&amp;').replace('\\%', '%').replace('\\_', '_').replace('\\#', '#').replace('\\$', '$')
    t = re.sub(r'\\[,;:!]\s*|~', ' ', t)
    t = re.sub(r'\\ ', ' ', t)
    t = t.replace('---', '—').replace('--', '–').replace("``", '“').replace("''", '”')
    t = re.sub(r'\\ensuremath\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\textbullet\s*', '•', t)

    def href(m):
        g1, j = group(t, m.end())
        g2, k = group(t, j)
        return g1, g2, k
    while links and '\\href' in t:
        m = re.search(r'\\href', t)
        u, txt, k = href(m)
        if u is None or txt is None:
            break
        u = u.replace('\\', '')
        t = t[:m.start()] + f'<a href="{html.escape(u, quote=True)}" target="_blank" rel="noopener">{txt}</a>' + t[k:]
    for mac, tag in (('textit', 'em'), ('emph', 'em'), ('textbf', 'strong')):
        while re.search(r'\\' + mac + r'\{', t):
            m = re.search(r'\\' + mac + r'\{', t)
            g, k = group(t, m.end() - 1)
            t = t[:m.start()] + f'<{tag}>{g}</{tag}>' + t[k:]
    t = re.sub(r'\\(text|mathrm|textrm|textsc|texttt)\{([^{}]*)\}', r'\2', t)
    t = re.sub(r'\$([^$]*)\$', lambda m: m.group(1).replace('\\', '').replace('{', '').replace('}', ''), t)
    t = re.sub(r'\\[a-zA-Z]+\*?', '', t)
    t = t.replace('{', '').replace('}', '')
    return re.sub(r'\s+', ' ', t).strip()


def norm_title(t):
    # html.unescape: tex_text turns \\& into &amp;, which must match the "&" of the PDF text (e.g. "S&P 500")
    t = unicodedata.normalize('NFKD', html.unescape(tex_text(t, links=False)))
    return re.sub(r'[^a-z0-9]', '', t.lower())


# ------------------------------------------------------------------ deck
def lecture_dir(k):
    """lectures/L{k}, or lectures/L{k}_<suffix> (e.g. the pipeline smoke test L0_smoke)."""
    d = os.path.join(ROOT, 'lectures', f'L{k}')
    if os.path.isdir(d):
        return d
    import glob
    alt = sorted(glob.glob(os.path.join(ROOT, 'lectures', f'L{k}_*')))
    return alt[0] if alt else d


def lecture_rel(k):
    return os.path.relpath(lecture_dir(k), ROOT).replace(os.sep, '/')


def read_deck(k, lang):
    """Frames (title, section, hash, charts), lecture title, references and PDF pages of the deck in one language."""
    d = lecture_dir(k)
    path = os.path.join(d, f'lecture{k}{"_ro" if lang == "ro" else ""}.tex')
    src = open(path, encoding='utf-8').read()
    s = strip_comments(src)
    m = re.search(r'\\nnlecture\{\d+\}\{', s)
    title = tex_text(group(s, m.end() - 1)[0]) if m else ''
    frames, refs, abbr = [], [], []
    sec = None
    pos = 0
    tok = re.compile(r'\\section\*?\{|\\begin\{frame\}')
    while True:
        m = tok.search(s, pos)
        if not m:
            break
        if m.group(0).startswith('\\section'):
            g, pos = group(s, m.end() - 1)
            sec = tex_text(g)
            continue
        end = s.find('\\end{frame}', m.end())
        body = s[m.end():end]
        pos = end + 10
        b = body
        while b.startswith('['):
            b = b[b.index(']') + 1:]
        ft, _ = group(b, 0)
        ft = ft or ''
        h = hashlib.sha1(re.sub(r'\s+', ' ', body).strip().encode()).hexdigest()[:8]
        charts = []
        for mm in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{charts/(?:en|ro)/([^}]+)\.pdf\}', body):
            name = mm.group(1)
            rest = body[mm.end():]
            cs = macro_args(rest[:600], 'chartsource')
            charts.append(dict(name=name, source=tex_text(cs[0]) if cs else ''))
        f = dict(title=tex_text(ft), raw_title=ft, section=sec, hash=h, charts=charts, page=None, body=body)
        frames.append(f)
        if re.match(r'(Abbreviations|Abrevieri)', f['title']):
            for tab in re.findall(r'\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*\}(.*?)\\end\{tabular\}', body, re.S):
                for row in re.split(r'\\\\', tab):
                    if '&' in row:
                        a, b = row.split('&', 1)
                        abbr.append((tex_text(a), tex_text(b)))
        if re.match(r'(References|Bibliografie|Referințe)', f['title']):
            refs += [tex_text(x) for x in re.findall(r'\\item\s+(.*?)(?=\\item|\\end\{itemize\})', body, re.S)]
    # PDF pages: each frame's title searched in order in the page texts; charts by their Quantlet name
    pdf = path[:-4] + '.pdf'
    try:
        import pymupdf
        doc = pymupdf.open(pdf)
        texts = [re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', p.get_text()).lower()) for p in doc]
        cur = 0
        for f in frames:
            # the text pieces of the title around its math ($\\lambda_{\\max}$, $\\sqrt{d_k}$ are typeset as symbols in the PDF)
            keys = [x[:40] for x in (norm_title(x) for x in re.split(r'\$[^$]*\$', f['raw_title'])) if x]
            if not keys:
                continue
            for p in range(cur, len(texts)):
                if all(x in texts[p] for x in keys):
                    f['page'] = p + 1
                    cur = p
                    break
        for f in frames:
            for c in f['charts']:
                q = re.sub(r'[^a-z0-9]', '', f'nn_l{k}_{c["name"]}'.lower())
                c['page'] = next((p + 1 for p, t in enumerate(texts) if q in t), f['page'])
    except Exception as e:                                      # no PDF or no PyMuPDF: pages stay unknown
        print('  (no PDF pages:', e, ')')
    return dict(title=title, frames=frames, refs=refs, abbr=abbr, src=src)


VIDEOS_JSON = os.path.join(HERE, 'videos.json')


def lecture_videos(k):
    """Verified YouTube videos of lecture k ([{id, by, title}]), read from assets/course-data.js (lectures[].videos) with node;
    the list is cached in lectures/common/videos.json, which is used when node is not available."""
    import subprocess
    js = ("global.window={};require(process.argv[1]);const o={};"
          "window.NN_DATA.lectures.forEach(l=>{o[l.id]=l.videos||[]});console.log(JSON.stringify(o))")
    try:
        r = subprocess.run(['node', '-e', js, os.path.join(ROOT, 'assets', 'course-data.js')], capture_output=True, text=True, check=True)
        v = json.loads(r.stdout)
        old = open(VIDEOS_JSON, encoding='utf-8').read() if os.path.exists(VIDEOS_JSON) else None
        new = json.dumps(v, ensure_ascii=False, indent=1) + '\n'
        if new != old:
            with open(VIDEOS_JSON, 'w', encoding='utf-8') as fh:
                fh.write(new)
    except Exception:
        v = json.load(open(VIDEOS_JSON, encoding='utf-8')) if os.path.exists(VIDEOS_JSON) else {}
    return v.get(f'l{k}', [])


def numbers_json(k):
    p = os.path.join(lecture_dir(k), f'L{k}_numbers.json')
    return open(p, encoding='utf-8').read() if os.path.exists(p) else ''


# ------------------------------------------------------------------ number check
NUM = re.compile(r'(?<![\w.,])[-−]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3}(?!\d))+|\d+)(?:[.,]\d+)*')


def canon(tok):
    """Possible numeric values of a token like '28,3', '2 860', '2,860', '1.5' (as normalised strings)."""
    t = tok.replace('−', '-').replace('\u00a0', ' ').replace('\u202f', ' ')
    t = t.replace(' ', '').lstrip('-')
    out = set()
    if re.fullmatch(r'\d+', t):
        out.add(str(int(t)))
    for dec, th in (('.', ','), (',', '.')):
        u = t
        if u.count(dec) <= 1:
            u2 = u.replace(th, '')
            u2 = u2.replace(dec, '.')
            try:
                v = float(u2)
                out.add(repr(v))
                out.add(str(int(v)) if v == int(v) else repr(v))
            except ValueError:
                pass
    return out


def deck_numbers(texts):
    vals = set()
    for t in texts:
        t = re.sub(r'(\d)\\,(\d)', r'\1\2', t)
        t = re.sub(r'(\d)\{,\}(\d)', r'\1,\2', t)
        for m in re.finditer(r'\d+(?:[.,]\d+)*', t):
            vals |= canon(m.group(0))
            for part in re.split(r'[.,]', m.group(0)):          # dates 2019-01-01, 1.5 split pieces
                vals.add(str(int(part)))
        for m in re.finditer(r'-?\d+\.\d+(?:e-?\d+)?', t):      # JSON floats: accept their roundings
            try:
                v = float(m.group(0))
            except ValueError:
                continue
            for nd in range(0, 5):
                r = round(v, nd)
                for x in (r, abs(r), r * 100, abs(r) * 100):
                    x = round(x, nd)
                    vals.add(repr(float(x)))
                    if x == int(x):
                        vals.add(str(int(x)))
    return vals


def check_numbers(text, allowed):
    bad = []
    t = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    t = re.sub(r'\]\([^)]*\)', ']', t)                         # link targets
    t = re.sub(r'\{#eq:[^}]*\}|@eq:[\w-]+|\{h=[0-9a-f]+\}', '', t)
    t = re.sub(r'@chart\s+\S+', '', t)
    t = re.sub(r'\\(?:frac|tfrac|sum|mathbb|mathrm|text|operatorname|log|ln|sqrt|hat|bar|tilde|left|right|cdot)\b', ' ', t)
    t = re.sub(r'\b[A-Za-zĂÎȘȚÂăîșțâ][\w-]*\d[\w]*\b', ' ', t)  # identifiers like L3, Q1, R2, PM10
    for m in NUM.finditer(t):
        tok = m.group(0)
        c = canon(tok)
        if not c:
            continue
        if any(x in allowed for x in c):
            continue
        try:
            if float(tok.replace(' ', '').replace('\u00a0', '').replace('−', '-').replace(',', '.').split('.')[0] or 0) <= 12 and \
                    re.fullmatch(r'[-−]?\d+', tok.strip()):
                continue                                         # small counts: one to twelve
        except ValueError:
            pass
        bad.append(tok)
    return sorted(set(bad), key=lambda x: t.find(x))


# ------------------------------------------------------------------ Romanian checks
RO_A_OK = re.compile(r'^(?:ne)?r[oO]m[âÂ]n', re.I)
RO_A_NAMES = {'Bâg'}                                             # surnames of authors, spelled as they are


def check_ro(text):
    issues = []
    plain = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    plain = re.sub(r'\$\$.*?\$\$|\$[^$\n]*\$', ' ', plain, flags=re.S)
    plain = re.sub(r'\]\([^)]*\)|https?://\S+', ' ', plain)
    for w in set(re.findall(r'\w*[âÂ]\w*', plain)):
        if not RO_A_OK.match(w) and w not in RO_A_NAMES:
            issues.append(f'"â" outside the român- family: {w}')
    for ch in 'şţŞŢ':
        if ch in plain:
            issues.append(f'cedilla letter {ch} (use ș/ț with comma below)')
    for w in set(re.findall(r'\b(sunt|suntem|sunteți|Sunt)\b', plain)):
        issues.append(f'"{w}" -> sînt/sîntem/sînteți')
    for m in set(re.findall(r'(?<![\w./:-])\d+\.\d+(?![\w./-])', plain)):
        issues.append(f'decimal point in Romanian text: {m}')
    for w in set(re.findall(r'\b[Cc]az(?:ul|uri|urile)? de studiu\b|\b[Ss]tudiul de caz\b', plain)):
        pass
    for m in re.finditer(r'(?i)\bcazul (?:românesc|nostru|practic)\b', plain):
        issues.append(f'"{m.group(0)}": a case study is a "studiu de caz"')
    return issues


# ------------------------------------------------------------------ markdown
def esc(t):
    return html.escape(t, quote=False)


class Ctx:
    def __init__(self, k, lang, deck, eqs):
        self.k, self.lang, self.deck, self.eqs = k, lang, deck, eqs
        self.T = L[lang]
        self.used_charts = []
        self.log = []
        self.charts = {c['name']: dict(c, frame=f) for f in deck['frames'] for c in f['charts']}
        self.hashes = {f['hash']: f for f in deck['frames']}


def md_inline(s, cx):
    """Inline markdown -> HTML; math kept as \\( \\) for MathJax."""
    maths = []

    def keep(m):
        maths.append('<span class="im">\\(' + esc(m.group(0)[1:-1]) + '\\)</span>')
        return f'\x00{len(maths) - 1}\x00'
    out = []
    for p in [re.sub(r'\$[^$]+\$', keep, s)]:
        t = esc(p)
        t = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)',
                   lambda m: f'<a href="{m.group(2)}"' + (' target="_blank" rel="noopener"' if m.group(2).startswith('http') else '') + f'>{m.group(1)}</a>', t)
        t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        t = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<em>\1</em>', t)

        def eqref(m):
            lab = m.group(1)
            if lab not in cx.eqs:
                raise Stop(f'unknown equation label @eq:{lab}')
            return f'<a class="eqref" href="#eq-{lab}">({cx.k}.{cx.eqs[lab]})</a>'
        t = re.sub(r'@eq:([\w-]+)', eqref, t)
        out.append(t)
    return re.sub(r'\x00(\d+)\x00', lambda m: maths[int(m.group(1))], ''.join(out))


ITEM = re.compile(r'^(\s*)(?:(\d+)\.|-)\s+')


def md_list(lines, cx):
    """Nested list from lines '- ', '1. ' (two spaces of indentation per level)."""
    items = []                                                   # (depth, ordered, text)
    for ln in lines:
        m = ITEM.match(ln)
        if m:
            items.append([len(m.group(1)) // 2, bool(m.group(2)), ln[m.end():].strip()])
        elif items:
            items[-1][2] += ' ' + ln.strip()
    html_, stack = [], []
    for d, ordered, txt in items:
        while len(stack) > d + 1:
            html_.append('</li></' + stack.pop() + '>')
        if len(stack) < d + 1:
            tag = 'ol' if ordered else 'ul'
            html_.append(f'<{tag}>')
            stack.append(tag)
        else:
            html_.append('</li>')
        li = md_inline(txt, cx)
        li = re.sub(r'\s(Answer|Răspuns):', r'<br><span class="qa">\1:</span>', li)
        html_.append('<li>' + li)
    while stack:
        html_.append('</li></' + stack.pop() + '>')
    return ''.join(html_).replace('<ul></li>', '<ul>').replace('<ol></li>', '<ol>')


def md_table(lines, cx):
    rows = [[c.strip() for c in ln.strip().strip('|').split('|')] for ln in lines]
    head, body = rows[0], [r for r in rows[1:] if not all(re.fullmatch(r':?-{2,}:?', c) for c in r)]
    num = lambda c: re.fullmatch(r'[-−+]?[\d\s.,%]+(?:\s*[A-Za-z/%]+)?', c) is not None
    h = '<thead><tr>' + ''.join(f'<th>{md_inline(c, cx)}</th>' for c in head) + '</tr></thead>'
    b = ''.join('<tr>' + ''.join(f'<td{" class=r" if num(c) else ""}>{md_inline(c, cx)}</td>' for c in r) + '</tr>' for r in body)
    return f'<div class="tbl"><table>{h}<tbody>{b}</tbody></table></div>'


def chart_html(name, caption, cx):
    c = cx.charts.get(name)
    if not c:
        raise Stop(f'@chart {name}: no such chart in lecture{cx.k}{"_ro" if cx.lang == "ro" else ""}.tex')
    cx.used_charts.append(name)
    k, T = cx.k, cx.T
    if cx.lang == 'en':
        png = os.path.join(ROOT, 'Quantlets', f'L{k}', f'NN_L{k}_{name}', f'{name}.png')
        if os.path.exists(png):
            src = f'../Quantlets/L{k}/NN_L{k}_{name}/{name}.png'
        else:
            src = render_png(k, name, 'en')
            cx.log.append(f'chart {name}: no Quantlet PNG, rendered charts/en/{name}.pdf')
    else:
        src = render_png(k, name, 'ro')
    ql = f'{REPO}/tree/main/Quantlets/L{k}/NN_L{k}_{name}'
    pdf = f'../{lecture_rel(k)}/lecture{k}{"_ro" if cx.lang == "ro" else ""}.pdf'
    page = c.get('page')
    links = (f'<a class="ql" href="{ql}" target="_blank" rel="noopener"><img src="../assets/logos/ql_logo.png" alt="">NN_L{k}_{name}</a>')
    if page:
        links += f'<a class="pdfp" href="{pdf}#page={page}" target="_blank" rel="noopener">{T["slide"].format(n=page)}</a>'
    cap = f'<p class="fcap">{md_inline(caption, cx)}</p>' if caption.strip() else ''
    srcl = f'<p class="fsrc">{c["source"]}</p>' if c['source'] else ''
    return (f'<figure class="wfig" id="fig-{name}"><img src="{src}" alt="{esc(T["chart_alt"])}: {esc(tex_text(c["frame"]["raw_title"], False))}" '
            f'loading="lazy"><figcaption>{cap}{srcl}<div class="flinks">{links}</div></figcaption></figure>')


def render_png(k, name, lang):
    pdf = os.path.join(lecture_dir(k), 'charts', lang, f'{name}.pdf')
    outd = os.path.join(OUT, 'img', f'L{k}')
    os.makedirs(outd, exist_ok=True)
    fn = f'{name}_{lang}.png'
    dst = os.path.join(outd, fn)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(pdf):
        import pymupdf
        doc = pymupdf.open(pdf)
        page = doc[0]
        zoom = 1600 / max(page.rect.width, 1)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        pix.save(dst)
    return f'img/L{k}/{fn}'


def md_blocks(lines, cx):
    """Paragraphs, lists, display math, tables, @chart, @practice -> HTML."""
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        st = ln.strip()
        if st.startswith('$$'):                                  # display math, possibly multi-line
            buf = [st]
            while not re.search(r'\$\$\s*(\{#eq:[\w-]+\})?\s*$', buf[-1]) or len(''.join(buf)) < 4 or \
                    (len(buf) == 1 and buf[0].count('$$') < 2):
                i += 1
                if i >= n:
                    raise Stop('unclosed $$')
                buf.append(lines[i].strip())
            txt = ' '.join(buf)
            m = re.match(r'\$\$(.*)\$\$\s*(?:\{#eq:([\w-]+)\})?\s*$', txt, re.S)
            body, lab = m.group(1).strip(), m.group(2)
            if lab:
                out.append(f'<div class="dm eq" id="eq-{lab}"><div class="eqb">\\[{esc(body)}\\]</div><span class="eqn">({cx.k}.{cx.eqs[lab]})</span></div>')
            else:
                out.append(f'<div class="dm">\\[{esc(body)}\\]</div>')
            i += 1
            continue
        buf = []
        while i < n and lines[i].strip() and not lines[i].strip().startswith('$$'):
            buf.append(lines[i])
            i += 1
        first = buf[0].strip()
        if first.startswith('@chart'):
            name = first.split()[1]
            out.append(chart_html(name, ' '.join(x.strip() for x in buf[1:]), cx))
        elif first.startswith('@practice'):
            txt = ' '.join([first[len('@practice'):].strip()] + [x.strip() for x in buf[1:]])
            out.append(f'<div class="mgr"><span class="mgr-t">{cx.T["practice"]}:</span> {md_inline(txt, cx)}</div>')
        elif first.startswith('|'):
            out.append(md_table([x for x in buf if x.strip().startswith('|')], cx))
        elif ITEM.match(buf[0]):
            out.append(md_list(buf, cx))
        elif any(ITEM.match(x) for x in buf[1:]):              # lead line + list
            j = next(j for j, x in enumerate(buf) if ITEM.match(x))
            out.append('<p class="lead">' + md_inline(' '.join(x.strip() for x in buf[:j]), cx) + '</p>')
            out.append(md_list(buf[j:], cx))
        else:
            out.append('<p>' + md_inline(' '.join(x.strip() for x in buf), cx) + '</p>')
    return ''.join(out)


def parse_md(path):
    """-> intro lines, sections [(title, intro_lines, blocks[(title, hash, lines)], end_lines)]."""
    txt = open(path, encoding='utf-8').read()
    txt = re.sub(r'<!--.*?-->', '', txt, flags=re.S)
    intro, secs, cur = [], [], None
    mode = 'pre'
    for ln in txt.split('\n'):
        if ln.startswith('# '):
            secs.append(dict(title=ln[2:].strip(), intro=[], blocks=[], end=[]))
            mode = 'sec'
            continue
        if ln.startswith('## '):
            t = ln[3:].strip()
            if t == '@intro':
                mode = 'intro'
                continue
            if t == '@end':
                mode = 'end'
                continue
            m = re.search(r'\s*\{h=([0-9a-f]+)\}\s*$', t)
            if not secs:
                raise Stop(f'block "{t}" before the first # section')
            secs[-1]['blocks'].append(dict(title=t[:m.start()] if m else t, hash=m.group(1) if m else None, lines=[]))
            mode = 'blk'
            continue
        if mode == 'intro':
            intro.append(ln)
        elif mode == 'sec':
            secs[-1]['intro'].append(ln)
        elif mode == 'blk':
            secs[-1]['blocks'][-1]['lines'].append(ln)
        elif mode == 'end':
            secs[-1]['end'].append(ln)
    return intro, secs, txt


def eq_labels(txt):
    labs = re.findall(r'\{#eq:([\w-]+)\}', txt)
    if len(labs) != len(set(labs)):
        raise Stop('duplicate equation label')
    return {l: i + 1 for i, l in enumerate(labs)}


def split_intro(lines):
    """Opener: the @goals list is separated from the rest."""
    body, goals, mode = [], [], 'b'
    for ln in lines:
        if ln.strip() == '@goals':
            mode = 'g'
            continue
        if mode == 'g':
            if not ln.strip() and goals:
                mode = 'b'
                continue
            if ln.strip():
                goals.append(ln)
            continue
        body.append(ln)
    return body, goals


def first_sentence(lines):
    t = ' '.join(x.strip() for x in lines if x.strip() and not x.strip().startswith(('@', '|', '$$', '-')))
    m = re.match(r'(.+?[.!?])(\s|$)', t)
    return m.group(1) if m else t


# ------------------------------------------------------------------ page
def build_page(k, lang, deck):
    T = L[lang]
    md = os.path.join(lecture_dir(k), f'web_{lang}.md')
    intro, secs, raw = parse_md(md)
    eqs = eq_labels(raw)
    cx = Ctx(k, lang, deck, eqs)
    sfx = '_ro' if lang == 'ro' else ''
    pdf = f'../{lecture_rel(k)}/lecture{k}{sfx}.pdf'
    nb = f'notebooks/L{k}_notebook{sfx}.ipynb'
    title = deck['title']
    head_t = f'{T["lecture"]} {k}: {title}'

    toc = [f'<li><a href="#intro">{T["intro"]}</a></li>']
    body = []
    # title + opener
    ib, goals = split_intro(intro)
    body.append(f'<section class="wintro" id="intro"><div class="tp"><p class="tp-k">{T["lecture"]} {k}</p><h2 class="tp-sub">{esc(title)}</h2>'
                f'<p class="tp-au">{T["author"]}</p><p class="tp-in">{"<br>".join(T["inst"])}</p></div>')
    body.append('<div class="ctext imot">' + md_blocks(ib, cx) + '</div>')
    if goals:
        body.append(f'<h3 class="ih">{T["goals"]}</h3><div class="ctext">' + md_list(goals, cx) + '</div>')
    plan = ''.join(f'<li><a href="#sec-{j}">{md_inline(s["title"], cx)}</a><span class="plan-d">{md_inline(first_sentence(s["intro"]), cx)}</span></li>'
                   for j, s in enumerate(secs, 1))
    body.append(f'<h3 class="ih">{T["plan"]}</h3><ol class="plan">{plan}</ol></section>')
    stale = []
    for j, s in enumerate(secs, 1):
        toc.append(f'<li><a href="#sec-{j}">{md_inline(s["title"], cx)}</a></li>')
        body.append(f'<h2 class="sec" id="sec-{j}">{md_inline(s["title"], cx)}</h2>')
        if any(x.strip() for x in s['intro']):
            body.append('<div class="ctext sintro">' + md_blocks(s['intro'], cx) + '</div>')
        for b_, b in enumerate(s['blocks'], 1):
            link = ''
            if b['hash']:
                f = cx.hashes.get(b['hash'])
                if not f:
                    stale.append(f'{b["title"]} (h={b["hash"]})')
                elif f.get('page'):
                    link = f'<a class="pdfp" href="{pdf}#page={f["page"]}" target="_blank" rel="noopener">{T["slide"].format(n=f["page"])}</a>'
            body.append(f'<article class="frame" id="s{j}-{b_}"><div class="fhead"><h3>{md_inline(b["title"], cx)}</h3>{link}</div>'
                        f'<div class="ctext">{md_blocks(b["lines"], cx)}</div></article>')
        if any(x.strip() for x in s['end']):
            body.append(f'<div class="send"><p class="send-t">{T["synth"]}</p>{md_blocks(s["end"], cx)}</div>')
    vids = lecture_videos(k)
    if vids:
        toc.append(f'<li><a href="#videos">{T["videos"]}</a></li>')
        body.append(f'<section class="wbib" id="videos"><h2 class="sec">{T["videos"]}</h2><ul class="bib videos">' +
                    ''.join(f'<li><a href="https://www.youtube.com/watch?v={html.escape(v["id"], quote=True)}" target="_blank" rel="noopener">'
                            f'{esc(v["title"])}</a> <span class="vby">({esc(v["by"])}, YouTube)</span></li>' for v in vids) + '</ul></section>')
    if deck['abbr']:
        toc.append(f'<li><a href="#abbr">{T["abbr"]}</a></li>')
        body.append(f'<section class="wbib" id="abbr"><h2 class="sec">{T["abbr"]}</h2><div class="tbl"><table class="abbr"><tbody>' +
                    ''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in deck['abbr']) + '</tbody></table></div></section>')
    if deck['refs']:
        toc.append(f'<li><a href="#bib">{T["bib"]}</a></li>')
        body.append(f'<section class="wbib" id="bib"><h2 class="sec">{T["bib"]}</h2><ul class="bib">' +
                    ''.join(f'<li>{r}</li>' for r in deck['refs']) + '</ul></section>')
    body.append(f'<p class="src foot">{T["foot"]}</p>')

    other = f'L{k}.html' if lang == 'ro' else f'L{k}_ro.html'
    en_a, ro_a = ('active', '') if lang == 'en' else ('', 'active')
    nav = (f'<li><a href="{T["site"]}">← {T["back"]}</a></li>'
           f'<li><a href="{pdf}" target="_blank" rel="noopener">{T["pdf"]}</a></li>'
           f'<li><a href="{REPO}/blob/main/{nb}" target="_blank" rel="noopener">{T["nb"]}</a></li>'
           f'<li><a href="{COLAB}{nb}" target="_blank" rel="noopener">{T["colab"]}</a></li>'
           f'<li><a href="{REPO}/tree/main/Quantlets/L{k}" target="_blank" rel="noopener">{T["ql"]}</a></li>'
           f'<li class="lang-switcher"><a href="L{k}.html" class="lang-btn {en_a}">EN</a><a href="L{k}_ro.html" class="lang-btn {ro_a}">RO</a></li>')
    page = f'''<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(head_t)} | NN</title>
<meta name="description" content="{esc(T['course'])}: {esc(head_t)}">
<link rel="icon" href="../assets/logos/ase_logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css?v={VER}">
<link rel="stylesheet" href="web.css?v={VER}">
<script>
window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']], processEscapes: false,
  macros: {{ E: '\\\\mathbb{{E}}', Var: '\\\\operatorname{{Var}}', Cov: '\\\\operatorname{{Cov}}', Corr: '\\\\operatorname{{Corr}}',
            asinh: '\\\\operatorname{{asinh}}', iid: '\\\\overset{{\\\\text{{i.i.d.}}}}{{\\\\sim}}', EUR: '\\\\text{{EUR}}' }} }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }},
  startup: {{ pageReady: () => MathJax.startup.defaultPageReady().then(() => {{ document.body.setAttribute('data-mj', 'done'); window.dispatchEvent(new Event('nn-mathjax')); }}) }} }};
</script>
<script id="MathJax-script" async src="{MATHJAX}"></script>
<script defer src="web.js?v={VER}"></script>
</head>
<body class="web">
<header>
  <div class="header-inner">
    <a href="https://www.ase.ro" target="_blank" rel="noopener"><img class="header-logo" src="../assets/logos/ase_logo.png" alt="ASE"></a>
    <div class="header-title"><h1><a href="{T['site'].split('#')[0]}">{esc(T['course'])}</a></h1><p>{esc(head_t)}</p></div>
    <a href="https://ipe.ro/new/" target="_blank" rel="noopener"><img class="header-logo" src="../assets/logos/acad_logo.png" alt="{esc(T['inst'][1])}"></a>
  </div>
</header>
<nav><ul>
  {nav}
</ul></nav>
<div class="wlay">
<aside class="wtoc"><details open id="wtoc"><summary>{T['toc']}</summary><ol>{''.join(toc)}</ol></details></aside>
<main class="wmain">
{''.join(body)}
</main>
</div>
</body>
</html>
'''
    with open(os.path.join(OUT, f'L{k}{sfx}.html'), 'w', encoding='utf-8') as fh:
        fh.write(page)
    return cx, secs, eqs, stale, raw


def frames_report(k):
    deck = read_deck(k, 'en')
    ro = read_deck(k, 'ro')
    sec = None
    for f, g in zip(deck['frames'], ro['frames']):
        if f['section'] != sec:
            sec = f['section']
            print(f'\n# {sec}  |  {g["section"]}')
        ch = ' '.join(c['name'] for c in f['charts'])
        print(f'p{f["page"] or "?":>3} {{h={f["hash"]}}} {f["title"]}' + (f'   [charts: {ch}]' if ch else '') + f'\n         RO {{h={g["hash"]}}} {g["title"]}')


def main(k):
    os.makedirs(OUT, exist_ok=True)
    log = [f'NN web lecture L{k}']
    allowed = deck_numbers([numbers_json(k)])
    decks = {lang: read_deck(k, lang) for lang in ('en', 'ro')}
    allowed |= deck_numbers([decks['en']['src'], decks['ro']['src']])
    res = {}
    for lang in ('en', 'ro'):
        if not os.path.exists(os.path.join(lecture_dir(k), f'web_{lang}.md')):
            log.append(f'[{lang}] no web_{lang}.md: page not built')
            continue
        cx, secs, eqs, stale, raw = build_page(k, lang, decks[lang])
        res[lang] = (secs, eqs, cx.used_charts)
        all_charts = [c['name'] for f in decks[lang]['frames'] for c in f['charts']]
        miss = [c for c in all_charts if c not in cx.used_charts]
        log.append(f'[{lang}] {len(secs)} sections, {sum(len(s["blocks"]) for s in secs)} blocks, {len(eqs)} numbered equations, '
                   f'{len(cx.used_charts)} charts of {len(all_charts)}')
        if miss:
            log.append(f'[{lang}] charts of the deck not on the page: {", ".join(miss)}')
        for s in stale:
            log.append(f'[{lang}] text out of date (frame changed or hash wrong): {s}')
        bad = check_numbers(raw, allowed)
        if bad:
            log.append(f'[{lang}] numbers not found in the deck or L{k}_numbers.json: {", ".join(bad)}')
        if lang == 'ro':
            for x in check_ro(raw):
                log.append(f'[ro] {x}')
        log += [f'[{lang}] {x}' for x in cx.log]
    if len(res) == 2:
        (se, ee, ce), (sr, er, cr) = res['en'], res['ro']
        if len(se) != len(sr):
            log.append(f'EN/RO differ: {len(se)} vs {len(sr)} sections')
        if list(ee) != list(er):
            log.append('EN/RO differ: equation labels')
        if ce != cr:
            log.append('EN/RO differ: chart order')
    txt = '\n'.join(log) + '\n'
    with open(os.path.join(OUT, f'L{k}_log.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(txt)


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(1)
    ks = range(1, 15) if a[0] == 'all' else [int(a[0])]
    try:
        for k in ks:
            if '--frames' in a:
                frames_report(k)
            elif os.path.exists(os.path.join(lecture_dir(k), 'web_en.md')) or os.path.exists(os.path.join(lecture_dir(k), 'web_ro.md')):
                main(k)
    except Stop as e:
        print('STOP:', e)
        sys.exit(2)
