r"""
nn_build.py -- shared builder for NN lecture decks (Beamer, English + Romanian), used by lectures/Lk/build_Lk.py.

Pipeline per lecture k:
    analysis_Lk.py  -> Lk_numbers.json + charts/en/*.pdf + charts/ro/*.pdf      (all computation)
    build_Lk.py     -> lecturek.tex / lecturek_ro.tex -> PDFs                     (text only; numbers via @{...})

Number placeholders in slide text (same placeholder in both languages):
    @{a.b.c}          value of N["a"]["b"]["c"]
    @{a.b.c|.1f}      Python format spec; "," gives thin-space thousands; RO uses a decimal comma
    @{a.b.c|+.1f}     explicit sign (true minus / plus signs)
    @{a.b|date}       ISO date "2025-06-13"        -> "13 Jun 2025" / "13 iunie 2025"
    @{a.b|dt}         "2026-05-01 13:15"            -> "1 May 2026, 13:15" / "1 mai 2026, ora 13:15"
    @{a.b|ym}         "2020-04"                     -> "Apr 2020" / "aprilie 2020"
A missing key raises KeyError: no slide can show a number that the analysis did not produce.
"""

import json
import re
import subprocess
import os

MONTHS = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          "ro": ["ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie", "august", "septembrie",
                 "octombrie", "noiembrie", "decembrie"]}


# ---------------------------------------------------------------- numbers
def fmt(v, spec=None, lang="en"):
    if v is None:
        return "n/a"
    if spec in ("date", "dt", "ym"):
        s = str(v)
        if spec == "ym":
            y, m = s[:4], int(s[5:7])
            return f"{MONTHS[lang][m - 1]} {y}"
        y, m, dd = s[:4], int(s[5:7]), int(s[8:10])
        out = f"{dd} {MONTHS[lang][m - 1]} {y}"
        if spec == "dt":
            out += (", ora " if lang == "ro" else ", ") + s[11:16]
        return out
    if isinstance(v, str):
        return v.replace("&", r"\&")
    s = str(v) if spec is None else format(v, spec)
    if lang == "ro":
        s = s.replace(",", "\x00").replace(".", "{,}").replace("\x00", r"\,")
    else:
        s = s.replace(",", r"\,")
    if s.startswith("-"):
        s = r"\ensuremath{-}" + s[1:]
    elif s.startswith("+"):
        s = r"\ensuremath{+}" + s[1:]
    return s


class Numbers:
    def __init__(self, path, lang="en"):
        with open(path) as f:
            self.N = json.load(f)
        self.lang = lang

    def get(self, key):
        v = self.N
        for part in key.split("."):
            v = v[int(part)] if isinstance(v, list) else v[part]
        return v

    def fmt(self, key, spec=None):
        return fmt(self.get(key), spec, self.lang)

    PH = re.compile(r"@\{([A-Za-z][A-Za-z0-9_.]*(?:@\{[A-Za-z]+\}[A-Za-z0-9_.]*)*)(?:\|([^{}]*))?\}")

    def render(self, tex):
        # nested year placeholder, e.g. @{neg.hours.RO.@{Y}|.2f}
        for _ in range(3):
            tex = re.sub(r"@\{(Y|YP)\}(?=[^{}\s]*\})", lambda m: str(self.N[m.group(1)]), tex)
        return re.sub(r"@\{([A-Za-z][A-Za-z0-9_.\-]*)(?:\|([^{}]*))?\}",
                      lambda m: self.fmt(m.group(1), (m.group(2) or "").strip() or None), tex)


# ---------------------------------------------------------------- LaTeX snippets
_ABBR_DOT = {"e.g", "i.e", "vs", "p", "pp", "Fig", "fig", "cf", "al", "nr", "Nr", "dr", "Dr", "Prof", "etc", "approx", "ec", "eq", "Tab", "No"}
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZĂÂÎȘȚŞŢ"


def split_clauses(x):
    """Split a bullet at top level (outside braces and math) on '; ' and on sentence ends '. X'."""
    parts, buf, depth, math, i = [], [], 0, False, 0
    while i < len(x):
        c = x[i]
        if c == "\\" and i + 1 < len(x):
            nxt = x[i + 1]
            if nxt in "[(":
                math = True
            elif nxt in "])":
                math = False
            buf.append(x[i:i + 2]); i += 2; continue
        if c == "$":
            math = not math
        elif c in "([" and not math:
            depth += 1
        elif c in ")]" and not math:
            depth -= 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0 and not math and x[i:i + 2] == "; ":
            parts.append("".join(buf)); buf = []; i += 2; continue
        elif depth == 0 and not math and c == "." and x[i + 1:i + 2] == " " and x[i + 2:i + 3] in _UPPER:
            word = re.split(r"[\s(~]", "".join(buf))[-1]
            if word not in _ABBR_DOT and not re.fullmatch(r"[A-Z]", word):
                buf.append("."); parts.append("".join(buf)); buf = []; i += 2; continue
        buf.append(c); i += 1
    parts.append("".join(buf))
    parts = [q.strip() for q in parts if q.strip()]
    return parts


def _cap(q):
    return q[0].upper() + q[1:] if q and q[0].islower() else q


def hier(x, sub_opt="\\setlength{\\itemsep}{0pt}"):
    """Level-1 bullet = first clause; the remaining clauses become level-2 bullets (course format: no text without bullets)."""
    parts = split_clauses(x)
    if len(parts) <= 1:
        return x
    head, rest = parts[0], parts[1:]
    return (head + "\n    \\begin{itemize}" + sub_opt + "\n" +
            "\n".join("      \\item " + _cap(q) for q in rest) + "\n    \\end{itemize}")


def _one(x, nest):
    if isinstance(x, (tuple, list)):   # explicit level 1 heading + level 2 items
        return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                "\n".join("      \\item " + _cap(q) for q in x[1:]) + "\n    \\end{itemize}")
    return hier(x) if nest else x


def items(*xs, opt="", nest=True):
    return ("\\begin{itemize}" + opt + "\n" + "\n".join(r"  \item " + _one(x, nest) for x in xs if x) +
            "\n\\end{itemize}")


def _balanced(tex, start):
    """Index just after the brace group that opens at tex[start] == '{'."""
    d = 0
    for j in range(start, len(tex)):
        if tex[j] == "{":
            d += 1
        elif tex[j] == "}":
            d -= 1
            if d == 0:
                return j + 1
    raise ValueError("unbalanced braces")


def bulletize_where(tex):
    """\\where{a; b; c} -> level-1 'where' bullet with one level-2 bullet per definition."""
    out, i = [], 0
    while True:
        k = tex.find("\\where{", i)
        if k < 0:
            out.append(tex[i:]); break
        e = _balanced(tex, k + 6)
        body = tex[k + 7:e - 1]
        parts = split_clauses(body)
        lst = ("{\\footnotesize\\begin{itemize}\\setlength{\\itemsep}{0pt}\n  \\item \\textbf{\\txtwhere}\n"
               "    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
               "\n".join("      \\item " + _cap(q) for q in parts) + "\n    \\end{itemize}\n\\end{itemize}\\par}")
        out.append(tex[i:k]); out.append(lst); i = e
    return "".join(out)


def block(title, content, kind="block"):
    return f"\\begin{{{kind}}}{{{title}}}\n{content}\n\\end{{{kind}}}"


def cols(left, right, wl="0.49", wr="0.49"):
    return (f"\\begin{{columns}}[T,onlytextwidth]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n"
            f"\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}")


def table(spec, header, rows, size="footnotesize", stretch="1.05"):
    head = " & ".join(header) + r" \\ \midrule"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"{{\\{size}\\renewcommand{{\\arraystretch}}{{{stretch}}}\\setlength{{\\tabcolsep}}{{4pt}}\n"
            f"\\begin{{tabular}}{{{spec}}}\\toprule\n{head}\n{body}\n\\bottomrule\\end{{tabular}}}}")


def heat(v, lo, hi, good_low=True, shown=None, max_pct=55):
    """Cell coloured from green (best) to red (worst), Weron style; the scale (lo, hi) is set per row or column."""
    if v is None:
        return "--"
    x = 0.5 if hi == lo else min(max((v - lo) / (hi - lo), 0.0), 1.0)
    if not good_low:
        x = 1 - x
    col, pct = ("Forest", int(round((0.5 - x) * 2 * max_pct))) if x < 0.5 else ("Crimson", int(round((x - 0.5) * 2 * max_pct)))
    txt = shown if shown is not None else str(v)
    return f"\\cellcolor{{{col}!{pct}}}{txt}" if pct > 0 else txt


def seq(v, hi, shown=None, max_pct=60, col="MainBlue"):
    """Single-hue sequential cell colour for counts and shares."""
    if v is None:
        return "--"
    pct = 0 if not hi else int(round(min(v / hi, 1.0) * max_pct))
    txt = shown if shown is not None else str(v)
    if pct == 0:
        return txt
    return f"\\cellcolor{{{col}!{pct}}}" + (f"\\textcolor{{white}}{{{txt}}}" if pct > 40 else txt)


# ---------------------------------------------------------------- deck
QL_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Quantlets"))


def quantlet_name(k, chart):
    """NN_Lk_<chart> when the Quantlet folder Quantlets/Lk/NN_Lk_<chart>/ exists, else None."""
    q = f"NN_L{k}_{chart}"
    return q if os.path.isfile(os.path.join(QL_ROOT, f"L{k}", q, "Metainfo.txt")) else None


class Deck:
    """One deck in one language. Frame helpers follow the course rules (see lectures/README.md)."""

    def __init__(self, lecture, title, numbers, lang="en", heights=None):
        self.k, self.title, self.V, self.lang = lecture, title, numbers, lang
        self.H = heights if heights is not None else {}      # figure-height overrides from the auto-fit loop
        self.used_h = {}                                        # figure name -> height actually used (fraction of textheight)
        self.frame_figs = {}                                    # rendered frame title -> figures in that frame
        self._figs_now = []
        self.parts = []
        self.n_frames = 1
        self.n_main = None
        self.quiz = 0

    def t(self, en, ro):
        return en if self.lang == "en" else ro

    def section(self, name):
        self.parts.append(f"\n\\section{{{name}}}\n")
        self.n_frames += 1                       # outline frame inserted by \AtBeginSection

    def subsection(self, name):
        self.parts.append(f"\\subsection{{{name}}}\n")

    def frame(self, title, body, size="small", cite=None, tryit=None):
        """Citation line under the title; the 'Experiment yourself' pointer sits at its right end (Weron style)."""
        tr = self._pending_try or tryit
        self._pending_try = None
        # standard course layout (as MFM, SFM): no subtitle line; the source/citation sits in small grey type at the
        # bottom left and the notebook pointer at the bottom right, like the Quantlet link of the other courses
        # Quantlet of every chart on the frame (Quantlets/Lk/NN_Lk_<chart>/), linked at the bottom right as in MFM, SFM
        qls = [q for q in (quantlet_name(self.k, f) for f in self._figs_now) if q]
        qtxt = "\\hspace{0.8em}".join(f"\\qlink{{{q.replace('_', chr(92) + '_')}}}{{Quantlets/L{self.k}/{q}}}" for q in dict.fromkeys(qls))
        # as in MFM: a chart frame points to its Quantlet; the notebook pointer is kept only on frames without one
        right = qtxt or (f"\\tryit{{{tr}}}" if tr else "")
        tail = ""
        if cite or right:
            tail = ("\n\\par\\vfill\\noindent{\\scriptsize\\color{FooterGray}" + (cite or "") + "}"
                    + (f"\\hfill{right}" if right else "") + "\\par\\vspace{0.5mm}")
        self.parts.append(f"\\begin{{frame}}{{{title}}}\n\\{size}\n{body}{tail}\n\\end{{frame}}\n")
        self.frame_figs[self.V.render(title)] = list(self._figs_now)
        self._figs_now = []
        self.n_frames += 1

    _pending_try = None

    def bottom(self, job=None, tryit=None):
        """Job line pinned to the bottom (full width); the notebook pointer goes to the citation line of this frame."""
        if tryit:
            self._pending_try = tryit
        if not job:
            return ""
        return f"\n\\par\\vfill\\manager{{{hier(job)}}}\\par\\vspace{{0.5mm}}"

    def concept(self, title, intuition, left, right, job, tryit=None, cite=None, wl="0.53", wr="0.44", size="footnotesize"):
        """Master rule: one-sentence intuition, formula with every symbol defined (left), real-data example (right),
        and one line tying the concept to a decision at work."""
        self.frame(title, f"\\intuition{{{intuition}}}\n" + cols(left, right, wl, wr) + self.bottom(job, tryit), size, cite)

    def figframe(self, title, intuition, equation, name, bullets=(), job=None, tryit=None, cite=None, height="0.50\\textheight",
                 size="small"):
        """Weron-style frame: one message = intuition line, one equation, one large figure, at most three bullets."""
        body = (f"\\intuition{{{intuition}}}\n" if intuition else "") + (f"\\vspace{{-1mm}}{equation}\\vspace{{-2mm}}\n" if equation else "")
        body += self.fig(name, height)
        if bullets:
            body += "\n\\vspace{0.5mm}\n" + items(*bullets, opt="\\setlength{\\itemsep}{0pt}")
        self.frame(title, body + self.bottom(job, tryit), size, cite)

    def fig(self, name, height="0.62\\textheight", width="\\textwidth"):
        h = self.H.get(name, float(height.replace("\\textheight", "")))
        self.used_h[name] = h
        self._figs_now.append(name)
        height = f"{h:.3f}\\textheight"
        m = self.V.get(f"charts.{name}")
        lab = self.t("Source", "Sursa")
        dat = self.t("Data", "Date")
        return (f"\\begin{{center}}\\includegraphics[width={width},height={height},keepaspectratio]"
                f"{{charts/{self.lang}/{name}.pdf}}\\end{{center}}\\vspace{{-3mm}}\n"
                f"\\chartsource{{{lab}: {m['source_' + self.lang]}. {dat}: {m['period_' + self.lang]}.}}")

    def chart(self, title, name, bullets=None, height="0.60\\textheight", size="small", cite=None, job=None, tryit=None):
        body = self.fig(name, height)
        if bullets:
            body += "\n\\vspace{0.5mm}\n" + items(*bullets, opt="\\setlength{\\itemsep}{0pt}")
        self.frame(title, body + self.bottom(job, tryit), size, cite)

    def quiz_frame(self, title, question, answer, size="small"):
        self.quiz += 1
        self.frame(self.t("Quiz", "Test") + f" {self.quiz}: {title}",
                   items(question) + "\n\\vspace{2mm}\n" + block(self.t("Answer", "R\\u{a}spuns"), items(answer), "exampleblock"), size)

    def mark_main_end(self):
        self.n_main = self.n_frames
        self.parts.append("\\appendix\n")

    def tex(self):
        head = (f"% lecture{self.k}{'' if self.lang == 'en' else '_ro'}.tex -- GENERATED by build_L{self.k}.py from "
                f"L{self.k}_numbers.json; do not edit by hand.\n"
                "\\documentclass[10pt,aspectratio=169,t]{beamer}\n"
                "\\input{../common/nn_preamble}\n" +
                ("\\usepackage[romanian]{babel}\n\\nnromanian\n" if self.lang == "ro" else "") +
                f"\\nnlecture{{{self.k}}}{{{self.title}}}\n\\begin{{document}}\n"
                "{\\setbeamertemplate{headline}{}\\setbeamertemplate{footline}{}\n\\begin{frame}\\titlepage\\end{frame}}\n")
        tex = self.V.render(head + "\n".join(self.parts) + "\n\\end{document}\n")
        tex = bulletize_where(tex)
        # a line break right after a paragraph end (\where{...} or a table) is an error in LaTeX: turn it into space
        tex = re.sub(r"(\\par\})\\\\(\[[^\]]*\])?", r"\1\\vspace{1mm}", tex)
        tex = re.sub(r"(\\end\{tabular\}\})\\\\(\[[^\]]*\])?", r"\1\\par\\vspace{0.5mm}", tex)
        tex = re.sub(r"(\\where\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\})\\\\(\[[^\]]*\])?", r"\1\\vspace{1mm}", tex)
        return tex

    def write(self, path):
        tex = self.tex()
        left = re.findall(r"@\{[A-Za-z][^}]*\}", tex)
        assert not left, left[:5]
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        return tex


# ---------------------------------------------------------------- compile and checks
def compile_tex(path, runs=2):
    wd, name = os.path.split(os.path.abspath(path))
    r = None
    for _ in range(runs):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name], cwd=wd,
                           capture_output=True, text=True, errors="replace")
    log = open(os.path.join(wd, name[:-4] + ".log"), encoding="latin-1").read()
    errors = re.findall(r"^! .*", log, re.M)
    over = [m.group(0).strip() for m in re.finditer(r"Overfull \\[hv]box \(([\d.]+)pt too (wide|high)\)[^\n]*\n?", log)
            if float(m.group(1)) > 1.0]
    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    # map each overflow to the frame it belongs to (title of the last \begin{frame} before that line)
    src = open(os.path.join(wd, name), encoding="utf-8").read().split("\n")
    where_ = []
    for m in re.finditer(r"Overfull \\([hv])box \(([\d.]+)pt too (?:wide|high)\)[^\n]*?lines? (\d+)", log):
        if float(m.group(2)) <= 1.0:
            continue
        ln = int(m.group(3))
        title = next((re.sub(r"\\begin\{frame\}(\[[^\]]*\])?", "", src[i])[:70] for i in range(min(ln, len(src)) - 1, -1, -1)
                      if src[i].startswith("\\begin{frame}")), "?")
        where_.append((m.group(1), float(m.group(2)), title))
    return {"returncode": r.returncode, "errors": errors, "overfull": over, "pages": int(pages.group(1)) if pages else 0,
            "where": where_}


ACR_RE = re.compile(r"(?<![\\A-Za-z])([A-Z][A-Za-z0-9]*[A-Z](?:-[A-Z]+)?)(?![a-z])")


def _strip(tex):
    body = re.sub(r"(?<!\\)%.*", "", tex)
    body = re.sub(r"\\(?:qlink|quantlet)\{[^}]*\}\{[^}]*\}", " ", body)     # Quantlet links: names, not text
    body = re.sub(r"\\(?:includegraphics|href|url|input|graphicspath|definecolor|begin|end|cellcolor|textcolor|qlink|quantlet)(\[[^\]]*\])?\{[^}]*\}", " ", body)
    return body


def acronym_audit(tex, defined, ignore=()):
    """Uppercase tokens (>= 2 capitals) in the deck text that are not in the abbreviation list."""
    body = re.sub(r"\\[A-Za-z]+", " ", _strip(tex))
    return sorted(t for t in set(ACR_RE.findall(body)) if t not in defined and t not in ignore)


def first_use_audit(tex, acronyms, start_marker="% END-FRONT-MATTER"):
    """Each acronym must be spelled out at its first use in the body: first occurrence written as '(ACR'."""
    i0 = tex.find(start_marker)
    body = tex[i0:] if i0 >= 0 else tex
    body = re.sub(r"% BEGIN-ABBR.*?% END-ABBR", " ", body, flags=re.S)
    body = re.sub(r"\\framesubtitle\{[^\n]*\}", " ", body)          # citation lines are not first uses
    body = re.sub(r"\\noindent\{\\scriptsize\\color\{FooterGray\}[^\n]*", " ", body)   # bottom citation line (new layout)
    body = _strip(body)
    body = re.sub(r"(?<!\\)\\\[.*?(?<!\\)\\\]", " ", body, flags=re.S)                  # display math: symbols, defined in the text
    body = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", body, flags=re.S)           # inline math
    bad = []
    for a in acronyms:
        m = re.search(r"(?<![A-Za-z\\])" + re.escape(a) + r"(?![A-Za-z])", body)
        if m and body[max(0, m.start() - 1)] != "(":
            bad.append((a, body[max(0, m.start() - 60):m.end() + 10].replace("\n", " ")))
    return bad


def contact_sheet(pdf, out_png, cols_=5, dpi=36):
    import pymupdf
    from PIL import Image, ImageDraw
    doc = pymupdf.open(pdf)
    ims = []
    for i, pg in enumerate(doc):
        pix = pg.get_pixmap(dpi=dpi, alpha=False)
        im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ImageDraw.Draw(im).text((4, 4), str(i + 1), fill=(200, 0, 0))
        ims.append(im)
    w, h = ims[0].size
    rows = (len(ims) + cols_ - 1) // cols_
    sheet = Image.new("RGB", (cols_ * w, rows * h), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols_) * w, (i // cols_) * h))
    sheet.save(out_png)
    return len(ims)


def render_pages(pdf, out_dir, pages, dpi=80):
    """Render selected pages (1-based) to PNG for a close visual check."""
    import pymupdf
    os.makedirs(out_dir, exist_ok=True)
    doc = pymupdf.open(pdf)
    paths = []
    for p in pages:
        fn = os.path.join(out_dir, f"p{p:02d}.png")
        doc[p - 1].get_pixmap(dpi=dpi, alpha=False).save(fn)
        paths.append(fn)
    return paths


def autofit(build_fn, out_path, lang, max_iter=8, textheight_pt=230.0, min_h=0.18):
    """Build, compile and shrink the figures of every frame that overflows vertically, until the deck fits.
    Figures start at the heights requested in the builder (generous), so each ends as large as its frame allows."""
    H = {}
    res = D = tex = None
    for _ in range(max_iter):
        D = build_fn(lang, H)
        tex = D.write(out_path)
        res = compile_tex(out_path)
        changed = False
        for kind, amt, title in res["where"]:
            if kind != "v":
                continue
            tt = title.strip().lstrip("{").rstrip("}")[:60]
            key = next((k for k in D.frame_figs if k.startswith(tt) or tt.startswith(k[:60])), None)
            figs = D.frame_figs.get(key) if key else None
            if not figs:
                continue
            cut = (amt + 3.0) / textheight_pt / len(figs)
            for n in figs:
                new = max(min_h, D.used_h[n] - cut)
                if new < D.used_h[n] - 1e-6:
                    H[n] = round(new, 3)
                    changed = True
        if not changed:
            break
    return D, tex, res, H


# ---------------------------------------------------------------- standard front/back matter and the full check
def abbreviation_frames(D, abbr, per_frame=36):
    """Abbreviations frame(s): two columns, sorted; abbr = {ACR: (english or None, romanian or None)}.
    The markers let first_use_audit skip these frames."""
    i = 0 if D.lang == "en" else 1
    ab = sorted([(k, v[i]) for k, v in abbr.items() if v[i] is not None], key=lambda kv: kv[0].lower())
    chunks = [ab[j:j + per_frame] for j in range(0, len(ab), per_frame)] or [[]]

    def col(lst):
        return ("{\\scriptsize\\renewcommand{\\arraystretch}{0.96}\\begin{tabular}{@{}>{\\bfseries}l@{\\ \\ }"
                "p{0.74\\linewidth}@{}}" + "\n".join(f"{k} & {v}\\\\" for k, v in lst) + "\\end{tabular}}")

    D.parts.append("% BEGIN-ABBR\n")
    for n, lst in enumerate(chunks):
        hh = (len(lst) + 1) // 2
        sfx = f" ({n + 1}/{len(chunks)})" if len(chunks) > 1 else ""
        D.frame(D.t("Abbreviations", "Abrevieri") + sfx, "\\vspace{-1mm}\\begin{columns}[T,onlytextwidth]\n" + "".join(
            f"\\begin{{column}}{{0.49\\textwidth}}{col(lst[c * hh:(c + 1) * hh])}\\end{{column}}\n" for c in range(2)) +
            "\\end{columns}")
    D.parts.append("% END-ABBR\n")


def reference_frames(D, refs, per_frame=11):
    """References as an itemised list (no bibliography package), split over as many frames as needed."""
    chunks = [refs[j:j + per_frame] for j in range(0, len(refs), per_frame)]
    for n, lst in enumerate(chunks):
        sfx = f" ({n + 1}/{len(chunks)})" if len(chunks) > 1 else ""
        D.frame(D.t("References", "Bibliografie") + sfx, "\\begin{itemize}\\setlength{\\itemsep}{0pt}\\scriptsize\n" +
                "\n".join("\\item " + r for r in lst) + "\n\\end{itemize}", "scriptsize")


def run_build(build_fn, here, k, abbr, ignore=(), units=(), langs=("en", "ro"), max_main=60, max_quiz=5, exempt_first_use=()):
    """Autofit + compile both decks and apply every check of lectures/STYLE.md. Returns True only if
    0 LaTeX errors, no overfull box > 1 pt, no undefined acronym, every acronym spelled out at first use,
    at most `max_main` main frames and `max_quiz` quiz frames. Writes lecturek[_ro]_contact_sheet.png."""
    ok = True
    for lang in langs:
        name = f"lecture{k}" if lang == "en" else f"lecture{k}_ro"
        out = os.path.join(here, name + ".tex")
        D, tex, res, H = autofit(build_fn, out, lang)
        n_main = D.n_main if D.n_main is not None else D.n_frames
        print(f"[{lang}] frames: main {n_main} (incl. title and section outlines), total {D.n_frames}; quiz frames {D.quiz}; "
              f"pages {res['pages']}; errors {len(res['errors'])}; overfull>1pt {len(res['overfull'])}")
        for e in res["errors"][:10]:
            print("     ", e)
        for kind, amt, title in res["where"]:
            print(f"     {kind} {amt:6.1f}pt  {title}")
        defined = {a for a, v in abbr.items() if v[0 if lang == "en" else 1] is not None}
        miss = acronym_audit(tex, defined, ignore)
        print(f"[{lang}] acronyms missing from Abbreviations:", miss)
        bad = first_use_audit(tex, [a for a in defined if a not in units and a not in exempt_first_use])
        print(f"[{lang}] acronyms not spelled out at first use:", [b[0] for b in bad])
        for a, ctx in bad:
            print(f"      {a}: ...{ctx}")
        if res["pages"]:
            contact_sheet(os.path.join(here, name + ".pdf"), os.path.join(here, name + "_contact_sheet.png"))
        too_many = n_main > max_main or D.quiz > max_quiz
        if too_many:
            print(f"[{lang}] too many frames: main {n_main} > {max_main} or quiz {D.quiz} > {max_quiz}")
        ok &= (res["returncode"] == 0 and res["pages"] > 0 and not res["errors"] and not res["overfull"] and not miss
               and not bad and not too_many)
    print("BUILD", "OK" if ok else "FAILED")
    return ok
