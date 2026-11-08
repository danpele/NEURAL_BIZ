# NN lecture pipeline

Every lecture of *Neural Networks and Deep Learning* (NN) / *Rețele neuronale și deep learning* is built the same way,
in English and Romanian, from one analysis script and one builder, exactly as in the EMQA course
(`~/Documents/Teaching/Energy MBA - EMQA/repo/lectures/L1` is the most complete worked example of a full lecture).
**Read `STYLE.md` first**: it lists the house rules that every file must follow.

```
data/clean/*.csv                 real downloaded datasets (data/download_all.py, documented in data/SOURCES.md)
        │
lectures/Lk/analysis_Lk.py       ALL computation and training ──► Lk_numbers.json  (every number shown anywhere)
        │                                                     ──► charts/en/*.pdf, charts/ro/*.pdf
lectures/Lk/build_Lk.py          slide text only (EN+RO), numbers as @{key}
        │                                     ──► lecturek.tex, lecturek_ro.tex ──► lecturek.pdf, lecturek_ro.pdf
lectures/Lk/make_notebook_Lk.py  ──► notebooks/Lk_notebook[_ro].ipynb (executed; § = "Experiment yourself" pointers)
                                 ──► notebooks/Lk_seminar[_ro].ipynb (student) + Lk_seminar_solutions[_ro].ipynb (git-ignored)
lectures/Lk/make_quantlets_Lk.py ──► Quantlets/Lk/NN_Lk_<topic>/ (Metainfo.txt, executed notebook, <topic>.png)
lectures/Lk/web_en.md, web_ro.md ──► web/Lk.html, web/Lk_ro.html (lectures/common/build_web_lecture.py)
```

Lecture ids are `l1` … `l14`, folders `lectures/L1` … `lectures/L14`, decks `lecture1.tex` … `lecture14.tex`.

## Commands (from the repository root)

```bash
python3 data/download_all.py [name ...]        # (re)download data; then python3 data/make_site_data.py
python3 lectures/Lk/analysis_Lk.py             # numbers JSON + both chart sets
python3 lectures/Lk/make_quantlets_Lk.py       # Quantlets (before the deck, so frames link them)
python3 lectures/Lk/build_Lk.py                # both decks; "en" or "ro" builds one; exit 0 only if every check passes
python3 lectures/Lk/make_notebook_Lk.py        # lecture + seminar notebooks, EN + RO, executed
python3 lectures/common/build_web_lecture.py k --frames   # frame hashes for web_en.md / web_ro.md
python3 lectures/common/build_web_lecture.py k            # web/Lk.html, web/Lk_ro.html + web/Lk_log.txt
python3 assets/make_showcase.py                # chart carousels of the lecture cards (from the Quantlets)
python3 tools/check_site.py                    # headless Chrome: console errors, failed requests, broken links
```

`build_Lk.py` ends with `nn_build.run_build(...)`, which prints, per language, the frame count, LaTeX errors, overfull
boxes (with the frame title), the acronym audits, writes `lecturek[_ro]_contact_sheet.png`, and **fails** unless there
are 0 LaTeX errors, no overfull box > 1 pt, no undefined acronym, every acronym spelled out at first use, ≤ 60 main
frames and ≤ 5 quiz frames. Figures are requested large and `autofit` recompiles, shrinking only the figures of frames
that overflow, until the deck fits. Check both contact sheets frame by frame.

## Shared files (`lectures/common/`)

| File | Role |
|---|---|
| `nn_preamble.tex` | Beamer theme (EMQA look), colours `Ser1`–`Ser8` = chart palette, title page with ASE + Romanian Academy (IPE) affiliations and logos, header/footer, outline at each section, teaching macros (`\intuition`, `\where`, `\manager`, `\tryit`, `\chartsource`, `\simlabel`, `\qlink`), deep-learning notation (`\vx`, `\mW`, `\btheta`, `\loss`, `\softmax`, `\ReLU`, `\argmin`). `\nnromanian` switches the labels to Romanian. Loaded with `\input{../common/nn_preamble}`. |
| `nn_build.py` | `Numbers` (`@{key|fmt}` placeholders, decimal comma in RO; missing key = error), `Deck` (`frame`, `section`, `concept`, `figframe`, `chart`, `quiz_frame`, `mark_main_end`), `items`/`hier` (L1/L2 bullets), `table`, `heat`, `seq`, `cols`, `block`, `abbreviation_frames`, `reference_frames`, `compile_tex`, `autofit`, `acronym_audit`, `first_use_audit`, `contact_sheet`, `render_pages`, **`run_build`** (all checks). |
| `nn_plot.py` | Chart style and rules (`style`, `legend_bottom`, `year_ticks`, `date_axis`, `comma_axes`, `audit_figure`, `out`, `chart_meta`, `period`), palette by role (`PAL`, `SERIES`), source lines (`SRC`), `seed_all`, `device` (MPS/CUDA/CPU), and loaders for every dataset (`fraud`, `german_credit`, `hmeq`, `california`, `markets`, `bvb`, `prices_hourly`, `prices_daily`, `load_generation_ro`, `yield_curve`, `phrasebank`, `tweets_sentiment`, `tweets_topic`, `ecb_speeches`, `vision_root`, `elliptic_root`). |
| `nn_notebook.py` | `Notebook` (lecture and seminar notebooks, EN + RO, Colab cell, setup cell with seed and device, `task`/`solution` for seminars) and `Notebook.build_all`. |
| `nn_quantlet.py` | `Quantlet` (folder `Quantlets/Lk/NN_Lk_<topic>/`, Metainfo.txt, executed notebook with inlined style and `show()`), `check_numbers`. |
| `build_web_lecture.py` | Web page of a lecture from `web_en.md` / `web_ro.md` + the deck (format in its docstring). |
| `colab_cell.py` | First code cell of the notebooks: on Colab, clone the repository and move into the notebook's folder. |
| `interactive_common.py` | Optional interactive companion pages (`interactive/Lk.html`), copied from EMQA and renamed; not yet used. |
| `logos/` | Logos of the title page. |

## Starting lecture k

1. Copy the four scripts of `L0_smoke/` (`analysis_L0.py`, `build_L0.py`, `make_notebook_L0.py`, `make_quantlets_L0.py`)
   to `Lk/` and rename them; for a full-size example of the structure (objectives, sections, concept frames, wrap-up,
   "Research from our group", "Want to learn more?", appendix), read EMQA's `lectures/L1/build_L1.py`.
2. **Analysis** (`analysis_Lk.py`): load data only through `nn_plot` loaders; cut-off and splits at the top; call
   `npl.seed_all()`; put every number shown anywhere in `N` (rounded as shown); draw each chart for `lang in ("en", "ro")`
   with `npl.out(fig, HERE, lang, name)` and register it with `npl.chart_meta(N, name, src_key, period_en, period_ro)`.
   Long training runs: cache results in `lectures/Lk/cache/` (git-ignored) and state the hardware and time.
3. **Builder** (`build_Lk.py`): one `build(lang, H)` returning a `Deck(k, title, V, lang, heights=H)`; every string
   `t(en, ro)`; `ABBR = {ACR: (english, romanian)}`; front matter, `% END-FRONT-MATTER`, `abbreviation_frames`,
   sections, `reference_frames`, `D.mark_main_end()`, appendix; end with `run_build(build, HERE, k, ABBR, IGNORE, UNITS)`.
4. **Notebooks**: same code as the analysis, simplified, section numbers § matching the `tryit` pointers; a seminar with
   4–8 tasks on real data (solutions with `nb.solution`).
5. **Quantlets**: one per chart shown in the deck, named `NN_Lk_<chart>`; print the slide numbers with `show(...)`.
6. **Web page**: `web_en.md`, `web_ro.md` (format in `build_web_lecture.py`), then build it and read `web/Lk_log.txt`.
7. **Site**: in `assets/course-data.js` set the lecture's `status: 'ready'` (or add k to `READY` in `assets/site.js`)
   when the PDF and notebook are published, add its key formulas to `formulas`, its references to `bibliography`
   and new acronyms to `abbreviations`; run `python3 assets/make_showcase.py` and `python3 tools/check_site.py`.
8. **Quiz** (optional, ≤ 20 drawn per attempt): `assets/quizzes/lk.js` registering
   `NN_DATA.quizzes['lk'] = { draw: 20, questions: [{ correct: 0-3, en: {title, text, options: [4],
   correctExplanation, incorrectExplanation}, ro: {...} }] }` (incorrectExplanation never names a letter), and a
   `<script src="assets/quizzes/lk.js">` line after `data-table.js` in both `index.html` and `index_ro.html`.

## Smoke test

`L0_smoke/` proves the chain end to end (one real chart: S&P 500 drawdown; title + chart + Abbreviations +
References frames). It is not linked from the site; its generated outputs (PDFs, notebooks, Quantlet, web page) are
deleted after the test and can be regenerated with the commands above using `0` as the lecture number.
