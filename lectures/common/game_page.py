"""
game_page.py -- writes the two pages of a lecture game, interactive/Lk_game.html (EN) and Lk_game_ro.html (RO).

    from game_page import write_game
    write_game(k, en=dict(title=..., lead=..., how=..., lesson=[...li html...], src=...), ro=dict(...), plotly=False)

title: name of the game (also the page title); lead, how: the two intro paragraphs; lesson: <li> lines of "What the game
shows" (end screen); src: data sources line. The page loads ../assets/config.js, Lk_game_data.js, game_common.js and
Lk_game.js (the lecture's game logic, written by hand on top of NNGame.choice in game_common.js). plotly=True also loads
Plotly for games with live charts. Course name, header, navigation, link to the class board: shared here.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "interactive")
V = "20261109g"
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
S = {"en": dict(idx="../index.html", course="Neural Networks and Deep Learning", lec="Lecture", game="Game", back="← Course site",
                interl="Interactive lecture", pdfl="Lecture PDF", pdf="lecture{k}.pdf", inter="L{k}.html", board="Class board for the projector",
                shows="What the game shows", alt="Romanian Academy, Institute for Economic Forecasting"),
     "ro": dict(idx="../index_ro.html", course="Rețele neuronale și deep learning", lec="Cursul", game="Joc", back="← Site-ul cursului",
                interl="Cursul interactiv", pdfl="PDF-ul cursului", pdf="lecture{k}_ro.pdf", inter="L{k}_ro.html", board="Clasamentul sălii, pentru proiector",
                shows="Ce arată jocul", alt="Academia Română, Institutul de Prognoză Economică")}


def write_game(k, en, ro, plotly=False):
    for L, t in (("en", en), ("ro", ro)):
        s, suf = S[L], ("" if L == "en" else "_ro")
        sub = f"{s['lec']} {k} · {s['game']}: {t['title']}"
        lesson = f"<h4>{s['shows']}</h4><ul class=\"g-res\">" + "".join(t["lesson"]) + "</ul>"
        scripts = ([PLOTLY] if plotly else []) + [f"../assets/config.js?v={V}", f"L{k}_game_data.js?v={V}", f"game_common.js?v={V}", f"L{k}_game.js?v={V}"]
        html = f"""<!DOCTYPE html>
<html lang="{L}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{t['title']}</title>
<meta name="description" content="{s['course']}: {sub}">
<link rel="icon" href="../assets/logos/ase_logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css?v={V}">
<link rel="stylesheet" href="interactive.css?v={V}">
<link rel="stylesheet" href="game.css?v={V}">
<script>window.NN_LANG = '{L}';</script>
</head>
<body>
<header>
  <div class="header-inner">
    <a href="https://www.ase.ro" target="_blank" rel="noopener"><img class="header-logo" src="../assets/logos/ase_logo.png" alt="ASE"></a>
    <div class="header-title"><h1><a href="{s['idx']}">{s['course']}</a></h1><p>{sub}</p></div>
    <a href="https://ipe.ro/new/" target="_blank" rel="noopener"><img class="header-logo" src="../assets/logos/acad_logo.png" alt="{s['alt']}"></a>
  </div>
</header>
<nav><ul>
  <li><a href="{s['idx']}#chapters">{s['back']}</a></li>
  <li><a href="{s['inter'].format(k=k)}">{s['interl']}</a></li>
  <li><a href="../lectures/L{k}/{s['pdf'].format(k=k)}" target="_blank" rel="noopener">{s['pdfl']}</a></li>
  <li class="lang-switcher"><a href="L{k}_game.html" class="lang-btn{' active' if L == 'en' else ''}">EN</a><a href="L{k}_game_ro.html" class="lang-btn{' active' if L == 'ro' else ''}">RO</a></li>
</ul></nav>
<main class="imain g-wrap">
<h2 class="sec">{t['title']}</h2>
<section id="g-intro">
  <p class="lead">{t['lead']}</p>
  <p>{t['how']}</p>
  <button type="button" class="btn play" id="g-s"></button>
</section>
<section id="g-play" hidden>
  <p class="g-score" id="g-score" aria-live="polite"></p>
  <div class="blk" id="g-card"></div>
</section>
<template id="g-lesson">{lesson}</template>
<p class="src"><a href="game_board{suf}.html?game=l{k}">{s['board']}</a></p>
<p class="src foot">{t['src']}</p>
</main>
""" + "".join(f'<script src="{x}"></script>\n' for x in scripts) + "</body>\n</html>\n"
        with open(os.path.join(OUT, f"L{k}_game{suf}.html"), "w", encoding="utf-8") as f:
            f.write(html)
