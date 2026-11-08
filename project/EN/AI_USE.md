# AI Use Declaration

Team: ...
Project title: ...

Declare every AI (artificial intelligence) tool you used. Undeclared use of AI counts as plagiarism.
You remain responsible for every line of code and text in this repository.
Never paste confidential data, contracts or prices of your employer into an AI tool: the project uses public data only.

## 1. Tools

| Tool (name, version) | Used by | Used for |
|---|---|---|
| e.g. ChatGPT, Claude, Copilot | team member | code / debugging / literature / writing |

## 2. What the AI produced

List the files or sections where AI-generated code or text was kept (edited or not).

| File or section | What the AI wrote | What we changed |
|---|---|---|
| `NN_team_fraud/mlp.ipynb` | first version of the fraud-detection MLP (multilayer perceptron) | moved the scaler inside the training fold, replaced accuracy with the area under the precision-recall curve |

## 3. What we checked ourselves

For each main result, say how you verified it without relying on the AI.

- **Reproduced number:** which number of the course (lecture, Quantlet or case study) we reproduced, its source, our value vs. the course value.
- **Data:** how we checked dates, time zones (CET, Central European Time, vs. UTC, Coordinated Universal Time), missing values and the 15-minute market time units since 1 October 2025.
- **Information set:** how we checked that every decision for day d uses only what is known at 12:00 CET on day d-1 (no look-ahead).
- **Formulas:** which formulas we checked against the lecture or the original paper.
- **References:** every reference was checked to exist and has a working DOI (Digital Object Identifier) or link.

## 4. What we did without AI

Which parts (the question, the design of the tests, the interpretation and the recommendation) are entirely the team's own work.
