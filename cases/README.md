# NN research case studies / Studii de caz din cercetarea grupului

Neural Networks and Deep Learning (NN), Prof. Daniel Traian Pele.
Bucharest University of Economic Studies (ASE) and Romanian Academy, Institute for Economic Forecasting (IPE).

There is one case per relevant lecture. Each case is built on a paper of our group and replicated in simplified form on the real data of the
course (`../data/clean/`, see `../data/SOURCES.md`). Every number in the briefs labelled as ours is computed by the notebooks; the tools are
the same as in the EMQA course (`~/Documents/Teaching/Energy MBA - EMQA/repo/cases/`).

## Mapping: case, lectures, paper, data

| Case | Lectures | Paper | 
|---|---|---|
| C1 `C1_fraud_xai/` | L3, L11, L1 | Diaconu, D. & Pele, D. T. (2026). Explainable AI in financial fraud detection: Evidence from the IEEE-CIS fraud dataset. *Proceedings of the International Conference on Business Excellence*, 20(1), 118–132. [doi:10.2478/picbe-2026-0015](https://doi.org/10.2478/picbe-2026-0015) |
| C2 `C2_spike_recurrent/` | L5, L11 | Andrei, A.-V., Pele, D.-T., Cramer, A.-A., Amza, A. & Diaconu, D. (2026). Explainable AI-based early warning system for electricity price spikes: Evidence from the Romanian market. *Proceedings of the International Conference on Business Excellence*, 20(1), 4820–4836. [doi:10.2478/picbe-2026-0352](https://doi.org/10.2478/picbe-2026-0352) |
| C3 `C3_shap_stability/` | L9, L11 | Diaconu, D. & Pele, D. T. (2026). SHAP stability across random and forward-time validation in graph-augmented fraud detection. submitted to Expert Systems with Applications. [code: github.com/QuantLet/GNN_XAI_Stability](https://github.com/QuantLet/GNN_XAI_Stability) |
| C4 `C4_battery_arbitrage/` | L10, L13 | Andrei, A.-V., Cepoi, C. O., Dumitrescu, B. A., Lessmann, S., Chiriac (Luchian), A.-C., Pele, D. T. & Petukhina, A. (2026). Joint probabilistic forecasting for battery arbitrage in day-ahead power markets. SSRN 6872381. [doi:10.2139/ssrn.6872381](https://doi.org/10.2139/ssrn.6872381) |
| C5 `C5_llm_var/` | L12 | Pele, D. T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A. T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S. & Härdle, W. K. (2026). In the beginning was the Word: LLM-VaR and LLM-ES. *Expert Systems with Applications*, 295, 128676. [doi:10.1016/j.eswa.2025.128676](https://doi.org/10.1016/j.eswa.2025.128676) |
| C6 `C6_foundation_risk/` | L13 | Jheng, S.-L., Tak, R., Găman, Ș., Mazurencu-Marinescu-Pele, M. & Pele, D. T. (2026). Can foundation models manage risk? Zero-shot VaR and ES forecasting with conformal calibration in CEE markets. *Economic Computation and Economic Cybernetics Studies and Research*, 60(2), 267–290. [doi:10.24818/18423264/60.2.26.14](https://doi.org/10.24818/18423264/60.2.26.14) |
| C7 `C7_vlm_anomaly/` | L14 | Jheng, S.-L., Pele, D. T., Tak, R. & Găman, Ș. (2026). A multimodal vision-language framework for financial anomaly detection. *Proceedings of the International Conference on Business Excellence*, 20(1), 191–201. [doi:10.2478/picbe-2026-0020](https://doi.org/10.2478/picbe-2026-0020) |

| Case | Data used |
|---|---|
| C1 | ULB card fraud (`creditcard_fraud_sample.csv`) |
| C2 | Romanian hourly day-ahead prices, load and generation |
| C3 | Elliptic Bitcoin graph (on demand, `data/raw/elliptic`) |
| C4 | Romanian, Bulgarian and Hungarian hourly day-ahead prices |
| C5 | S&P 500 daily returns (`markets_daily.csv`) |
| C6 | Eight BVB shares (`bvb_equities_daily.csv`) |
| C7 | S&P 500 daily prices (`markets_daily.csv`) |

Heavy model outputs are cached in `Ck_*/cache/` (LLM and VLM answers, foundation-model quantiles, GARCH fits, refits); the notebooks read the
cache when present and recompute otherwise. Run times with an empty cache on an Apple M5 Max: C1 under 1 minute, C2 about 6 minutes per notebook,
C3 about 11 minutes, C4 about 6 minutes, C6 about 5.5 minutes, C7 about 15 minutes (C5: see its notebook). Set `OMP_NUM_THREADS=1` when PyTorch
and XGBoost or scikit-learn gradient boosting run in the same process, and import torch before sklearn/shap.

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
