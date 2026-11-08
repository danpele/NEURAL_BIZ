# House rules of the NN course (Neural Networks and Deep Learning)

These rules apply to every deck, chart, notebook, Quantlet, web page and site text of the course. The build scripts
check the ones that can be checked; the others are checked by reading the contact sheets frame by frame.

## 1. Data

1. **Real data only.** Every number, table and chart comes from a real public dataset of `data/clean/` (or an on-demand
   dataset of `data/SOURCES.md`). No invented numbers, no fictional companies or customers, no `make_classification`,
   `make_moons`, random-walk "prices" or simulated "returns" presented as data.
2. **The one exception:** a simulated or toy picture is allowed only when the frame is explicitly about a *mechanism*
   (the shape of an activation function, a loss surface, how gradient descent moves on a quadratic, a schematic of an
   architecture, the noising steps of a diffusion model). It is labelled `\simlabel` on the slide, at most a few per
   lecture, and never shows a "result".
3. **No hand-typed results.** Slide text uses placeholders `@{key|fmt}` filled from `Lk_numbers.json`; a missing key
   stops the build. Hand-typed numbers are limited to definitions and constants (0.5 threshold, 252 trading days,
   1.96, a learning rate chosen *a priori*), each stated as such.
4. **Leakage-free evaluation.** Scalers, imputers, feature selection and early stopping are fitted on the training
   (or validation) part only; time series are split in time order (never shuffled); the test set is used once.
   Imbalanced targets (fraud, default, illicit) are reported with precision, recall, PR-AUC and the cost of errors,
   not with accuracy alone.
5. **Seeds and uncertainty.** Every trained result states its seed; a claim that model A beats model B is backed by
   several seeds or a test (e.g. Diebold–Mariano, bootstrap), not by one run.
6. **No "not obtained" lists.** Never write in any public file (slides, site, SOURCES.md, notebooks) which data could not
   be downloaded, needed a key, or failed. Describe only what is used; failures go to the console.

## 2. Master's rigour

7. **Full formulas.** Every method is written with its formula and every symbol defined (`\where{...}`): the forward
   pass, the loss, the gradient/update rule, the regulariser. Derivations too long for a frame go to the appendix
   after the references.
8. **Own estimates on real data.** Each formula is followed by our own estimate on a real dataset, with the benchmark
   (linear/logistic regression, naive forecast, gradient boosting) next to the network.
9. **Research from our group.** Papers of the professor and of the group are presented accurately and positively;
   never criticise or downplay them (state scope conditions neutrally, as "applies to ...").

## 3. Slides

10. **Bullets only.** All slide text is in level-1 / level-2 bullets (⊡ / ▶): a string given to `items()`/`hier()` is
    split into a level-1 head and level-2 details. No free paragraphs.
11. **One message per frame.** The title is a short claim; the frame carries one idea: intuition line → formula →
    large figure or real-data example → at most three bullets → one "In practice" line. If a figure ends small after
    autofit, cut text, do not shrink the font.
12. **Acronyms.** Every acronym is spelled out at its first use as `full name (ACR)` and listed on the Abbreviations
    frame; `run_build` fails otherwise (math symbols and Quantlet names are exempt; units and journal codes go in
    `UNITS` / `IGNORE`).
13. **Title page** shows ASE (Bucharest University of Economic Studies) and the Romanian Academy, Institute for
    Economic Forecasting (IPE), with their logos (built into `lectures/common/nn_preamble.tex`; do not change it).
14. **Build quality.** 0 LaTeX errors and no overfull box > 1 pt in both languages; never `\hfuzz`, `\vfuzz`,
    `\hbadness`. At most 60 main frames and 5 quiz frames.
15. **Structure of a deck.** Title → learning objectives → Abbreviations → sections (outline frame at each) →
    wrap-up → "Research from our group" → "Want to learn more?" → References → appendix.
16. **References** are verified on Crossref/DOI or arXiv pages; every reference cited on a slide is in the References
    frames and in the site bibliography (`assets/course-data.js`).

## 4. Charts

17. **Transparent background, no gridlines, no in-chart title.** `nn_plot.style()`; `nn_plot.out()` refuses a chart that
    breaks the rules it can check.
18. **Legend at the bottom, outside the plot** (`nn_plot.legend_bottom(fig)`; never `ax.legend()`).
19. **Four-digit years** on every date axis (`nn_plot.date_axis`, `nn_plot.year_ticks`); never '15 or 15.
20. **Two chart sets.** `charts/en/<name>.pdf` and `charts/ro/<name>.pdf`; the Romanian chart is fully Romanian: axis
    labels, legend, category names, month names, decimal comma and true minus (`out(..., lang='ro')` formats the axes).
21. **Consistent palette.** Colours by role (`nn_plot.PAL`): data blue, the network orange, the benchmark green,
    train/validation/test blue/orange/purple, event class (fraud, default) red. One y-axis per panel, no twin axes.
22. **Display size.** 6.4 in wide, about 2–2.6 in tall, fonts 10–11 pt. Source and data period are not drawn in the
    chart: register them with `nn_plot.chart_meta`; the builder prints them under the figure.

## 5. Language

23. **EN and RO everywhere** from one source: `t(english, romanian)` in builders, `nb.T(en, ro)` in notebooks.
24. **Romanian** is natural academic Romanian, as on the EMQA pages: î-spelling inside words (*sînt, cînd, rămîne,
    pînă*; â only in *român-*, *România*), decimal comma, „de” after numbers ending in 20–99 or 00, the Romanian term
    with the English one in brackets at first use (*supraînvățare (overfitting)*), and **„studiu de caz”** (never
    „caz de studiu”). Technical terms that Romanian practice keeps in English (deep learning, dropout, transformer,
    embedding, batch) are written in italics or explained at first use.
