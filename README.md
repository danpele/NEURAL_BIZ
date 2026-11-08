# Neural Networks and Deep Learning (NN) / Rețele neuronale și deep learning

Master's course, Bucharest University of Economic Studies (ASE). Instructor: Prof. dr. Daniel Traian Pele
(ASE, Department of Statistics and Econometrics; Romanian Academy, Institute for Economic Forecasting, IPE).

Course website: https://danpele.github.io/NEURAL_BIZ/ (EN) · https://danpele.github.io/NEURAL_BIZ/index_ro.html (RO)

Fourteen lectures, each with a seminar notebook, in English and Romanian; real public data only.

## Structure

| Path | Content |
|---|---|
| `index.html`, `index_ro.html` | Course website (EN / RO), rendered by `assets/site.js` from `assets/course-data.js` |
| `assets/` | Site shell (`site.js`, `site.css`, `nn.css`), course data, `config.js` (quiz login), `data-table.js` (generated), `showcase.js` (generated), `quizzes/` (quiz banks, one per lecture) |
| `lectures/` | One folder per lecture `L1`…`L14` (analysis, deck builder, notebooks maker, Quantlets maker, web text); `common/` = shared build library. Pipeline: `lectures/README.md`; house rules: `lectures/STYLE.md` |
| `notebooks/` | Lecture and seminar notebooks (EN + RO, Colab-ready) |
| `Quantlets/Lk/NN_Lk_<topic>/` | One Quantlet per chart: `Metainfo.txt`, executed notebook, PNG |
| `web/` | Web version of each lecture (`Lk.html`, `Lk_ro.html`) and of the case studies (`web/cases/`) |
| `cases/` | Research case studies (shared tools in `cases/common/`) |
| `data/` | `download_all.py` (downloads every dataset), `SOURCES.md` (documentation), `clean/` (files used by the course), `raw/` (not committed) |
| `project/EN`, `project/RO` | Templates for the team project: `AI_USE.md`, `AI_ERRORS.md` |
| `tools/check_site.py` | Local check of the site in headless Chrome (console errors, failed requests, broken links) |
| `docs/` | Internal notes (inventory of earlier course material); not linked from the site |
| `legacy/` | Material of earlier editions of the course (introductory chapters, Bishop chapter notebooks, old data), kept for reference and not linked from the site |

## Quick start

```bash
python3 data/download_all.py && python3 data/make_site_data.py   # data (EODHD key read from .env, never printed)
python3 tools/check_site.py                                     # site check
```

GitHub Pages serves the `main` branch as is (`.nojekyll`).
