# NN research case studies / Studii de caz din cercetarea grupului

Neural Networks and Deep Learning (NN), Prof. Daniel Traian Pele.
Bucharest University of Economic Studies (ASE) and Romanian Academy, Institute for Economic Forecasting (IPE).

No case study has been written yet. The tools are the same as in the EMQA course
(`~/Documents/Teaching/Energy MBA - EMQA/repo/cases/` has seven complete examples).

## Layout of a case folder `cases/Ck_<slug>/`

| File | Content |
|---|---|
| `Ck_brief.tex` | One LaTeX source for both languages (`\T{EN}{RO}`); `Ck_case_EN.tex`, `Ck_case_RO.tex` are generated wrappers |
| `Ck_case_EN.pdf`, `Ck_case_RO.pdf` | The brief: question, paper, method, replication on the course data with figures and tables, discussion questions, scope of the classroom replication |
| `nb_spec.py` | Source of the two notebooks (cells in EN and RO); edit this, not the notebooks |
| `Ck_notebook_EN.ipynb`, `Ck_notebook_RO.ipynb` | The replication, executed; they write `numbers.json`, `numbers_<lang>.tex`, `tab_*.tex`, `fig/` |

## Commands (from `cases/`)

```bash
python3 common/make_notebooks.py Ck_<slug>     # write + execute the notebooks from nb_spec.py
python3 common/build_briefs.py Ck_<slug>       # compile the EN/RO briefs
python3 common/build_web_case.py k             # web/cases/Ck.html, Ck_ro.html (register the case in CASE first)
```

Register each case in `cases/common/build_web_case.py` (`CASE`) and in `assets/course-data.js` (`studies`).
Same rules as the lectures (`lectures/STYLE.md`): real data only, every number computed by the notebook, EN + RO,
„studiu de caz” in Romanian, the group's papers presented accurately and never criticised.
