"""
make_game_L12.py -- data of the Lecture 12 game "You or the language model? When the annotators disagree"
(interactive/L12_game.html, L12_game_ro.html).

The student labels real Financial PhraseBank sentences (negative, neutral, positive) and plays against the prompt classifier
of the lecture (section "A prompt as a classifier"): Qwen2.5-3B-Instruct, zero-shot, instruction 1, label probabilities from
the next-token distribution. After each answer the card shows the probability the model gave its label, the label of the
same model with 12 examples in the prompt (few-shot, first seed), the label of the annotators and how many of them agreed.

Nothing is recomputed with a language model: the probabilities are the cached outputs of analysis_L12.py
(cache/zs_pb_3B.json and fs_pb_3B_k4_s0.json on the 678 unanimous TEST sentences; cache/amb_zs_3B.json and amb_fs_3B.json
on the 2,579 sentences with 50-75 % agreement, never used for training or as examples). Splits rebuilt exactly as in
analysis_L12.py; the script checks that the cached outputs reproduce the accuracies and calibration error of L12_numbers.json.

Pool: 10 sentences per agreement level (100 % from the test set; 75, 66 and 50 % from the other sentences), drawn at random
(seed of the course) among sentences in plain ASCII with at most 40 words; every game draws 3 + 3 + 2 + 2 of them.

Run:  python3 lectures/L12/make_game_L12.py      ->  interactive/L12_game_data.js
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = npl.seed_all()
OUT = os.path.join(HERE, "..", "..", "interactive", "L12_game_data.js")
NUM = json.load(open(os.path.join(HERE, "L12_numbers.json"), encoding="utf-8"))
CACHE = os.path.join(HERE, "cache")
LAB = ["positive", "negative", "neutral"]          # column order of the cached probabilities (analysis_L12.py)
BTN = {"negative": 0, "neutral": 1, "positive": 2}   # buttons
LEVELS = {100: 3, 75: 3, 66: 2, 50: 2}               # cards per game by agreement level
N_LEV = 10                                           # pool per level

# ---------------------------------------------------------------- splits of analysis_L12.py
PB = npl.phrasebank(50)
PB100 = PB[PB.min_agreement_pct == 100].reset_index(drop=True)
pb_tr, pb_te = train_test_split(PB100, test_size=0.30, stratify=PB100.label, random_state=SEED)
pb_te = pb_te.reset_index(drop=True)
amb = PB[PB.min_agreement_pct < 100].reset_index(drop=True)
assert len(pb_te) == NUM["data"]["pb_te"]
assert (pd.read_csv(os.path.join(HERE, "results", "predictions_pb.csv")).text.values == pb_te.sentence.values).all()
assert (pd.read_csv(os.path.join(HERE, "results", "agreement_labels.csv")).sentence.values == amb.sentence.values).all()

ld = lambda f: np.array(json.load(open(os.path.join(CACHE, f)))["P"])  # noqa: E731
P = {"te": {"zs": ld("zs_pb_3B.json"), "fs": ld("fs_pb_3B_k4_s0.json")}, "amb": {"zs": ld("amb_zs_3B.json"), "fs": ld("amb_fs_3B.json")}}
lab_of = lambda p: np.array(LAB)[p.argmax(1)]  # noqa: E731

# ---------------------------------------------------------------- checks against L12_numbers.json
for m, key in (("zs", "zs_3B"), ("fs", "fs_3B")):
    assert round(100 * float(np.mean(lab_of(P["te"][m]) == pb_te.label.values)), 1) == NUM["agree"][key]["100"]
    for lev in (75, 66, 50):
        msk = (amb.min_agreement_pct == lev).values
        assert round(100 * float(np.mean(lab_of(P["amb"][m][msk]) == amb.label.values[msk])), 1) == NUM["agree"][key][str(lev)]


def ece(Pm, y, bins=10):
    """Expected calibration error as in reliability() of analysis_L12.py."""
    conf, ok = Pm.max(1), lab_of(Pm) == y
    edges, e = np.linspace(1 / 3, 1, bins + 1), 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        msk = (conf >= a) & (conf <= b if b == 1 else conf < b)
        if msk.sum():
            e += msk.mean() * abs(conf[msk].mean() - ok[msk].mean())
    return 100 * e


assert round(ece(P["te"]["zs"], pb_te.label.values), 1) == NUM["calib"]["zs_3B"]

# ---------------------------------------------------------------- pool
rng = np.random.default_rng(SEED)
pool = []
for lev in LEVELS:
    df, src = (pb_te, "te") if lev == 100 else (amb, "amb")
    ok = df.sentence.map(lambda s: s.isascii() and len(s.split()) <= 40).values & (df.min_agreement_pct.values == lev)
    for k in rng.choice(np.where(ok)[0], N_LEV, replace=False):
        pz, pf = P[src]["zs"][k], P[src]["fs"][k]
        pool.append({"s": df.sentence.iloc[k], "a": lev, "y": BTN[df.label.iloc[k]], "m": BTN[LAB[int(pz.argmax())]],
                     "p": round(float(pz.max()), 3), "f": BTN[LAB[int(pf.argmax())]]})

info = {"seed": SEED, "levels": {str(k): v for k, v in LEVELS.items()}, "per_level": N_LEV, "per_game": sum(LEVELS.values()),
        "n_te": int(len(pb_te)), "n_amb": int(len(amb))}
with open(OUT, "w", encoding="utf-8") as f:
    f.write("// generated by lectures/L12/make_game_L12.py -- do not edit\n")
    f.write("window.NN_GAME_L12 = " + json.dumps({"info": info, "pool": pool}, ensure_ascii=False) + ";\n")
print(OUT, info, "pool accuracy", np.mean([c["m"] == c["y"] for c in pool]))


# ---------------------------------------------------------------- the two pages (interactive/L12_game.html, L12_game_ro.html)
def ro(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


from game_page import write_game  # noqa: E402

A, C, SN, D = NUM["agree"]["zs_3B"], NUM["calib"], NUM["sens"], NUM["data"]
write_game(12, en=dict(
    title="You or the language model? When the annotators disagree",
    lead="You read 10 real sentences from financial news and label each one negative, neutral or positive for an investor in the "
         "company. You play against a large language model (LLM) used as a classifier with a prompt, as in Lecture 12. Some sentences "
         "were labelled the same way by all annotators of the data set; on others the annotators were split. After each answer you "
         "see which case it was.",
    how=f"The model is Qwen2.5-3B-Instruct, an open LLM with {NUM['models']['3B']['params_b']:.2f} billion parameters run on a laptop, asked without examples (zero-shot): "
        f"“Classify the sentiment of this sentence from financial news, from the point of view of an investor in the company. Answer "
        f"with one word: positive, negative or neutral.” Its answer is the label with the highest probability for the next word; the "
        f"card also shows that probability. The right answer is the label chosen by the majority of the 5–8 annotators.",
    lesson=[f"<li>The model is right on {A['100']:.1f}% of the sentences on which all annotators agree, on {A['75']:.1f}% at 75% "
            f"agreement, {A['66']:.1f}% at 66% and {A['50']:.1f}% when only half of them agree. A model cannot be more certain than "
            f"the people who labelled the data.</li>",
            f"<li>On the unanimous test sentences its probabilities are close to calibrated: the expected calibration error (ECE) is {C['zs_3B']:.1f}%, against "
            f"{C['tfidf']:.1f}% for a TF-IDF (term frequency–inverse document frequency) classifier trained on "
            f"{D['pb_tr']:,} sentences.</li>",
            f"<li>The prompt is a hyper-parameter: five wordings of the same instruction move the macro-F1 of this model by "
            f"{SN['pb_3B']['range']:.1f} points on these sentences and by {SN['tw_3B']['range']:.1f} points on finance tweets.</li>"],
    src=f"Data: Financial PhraseBank (Malo et al., 2014; 4,838 sentences, 5–8 annotators each). Unanimous cards: test set of the "
        f"lecture ({D['pb_te']} sentences); split cards: the {info['n_amb']:,} sentences with 50–75% agreement, never used for "
        f"training or as examples. The game pool has 10 sentences per agreement level (at most 40 words), drawn at random "
        f"(seed {SEED}); every game draws 3, 3, 2 and 2 of them. Model outputs: cached outputs of analysis_L12.py. "
        f"Code: lectures/L12/make_game_L12.py."),
  ro=dict(
    title="Dumneavoastră sau modelul de limbaj? Cînd adnotatorii nu sînt de acord",
    lead="Citiți 10 propoziții reale din știri financiare și le etichetați ca negative, neutre sau pozitive pentru un investitor în "
         "companie. Jucați împotriva unui model mare de limbaj (LLM, large language model) folosit drept clasificator printr-un "
         "prompt, ca în cursul 12. Unele propoziții au fost etichetate la fel de toți adnotatorii setului de date; la altele, "
         "adnotatorii au fost împărțiți. După fiecare răspuns vedeți despre care caz este vorba. Propozițiile sînt în limba engleză, "
         "ca în setul de date.",
    how=f"Modelul este Qwen2.5-3B-Instruct, un LLM deschis cu {ro(NUM['models']['3B']['params_b'], 2)} miliarde de parametri, rulat pe un laptop și întrebat fără "
        f"exemple (zero-shot): „Classify the sentiment of this sentence from financial news, from the point of view of an investor "
        f"in the company. Answer with one word: positive, negative or neutral.” Răspunsul său este eticheta cu cea mai mare "
        f"probabilitate pentru cuvîntul următor; cartonașul arată și această probabilitate. Răspunsul corect este eticheta aleasă "
        f"de majoritatea celor 5–8 adnotatori.",
    lesson=[f"<li>Modelul are dreptate la {ro(A['100'])} % dintre propozițiile asupra cărora toți adnotatorii sînt de acord, la "
            f"{ro(A['75'])} % cînd acordul este de 75 %, la {ro(A['66'])} % cînd este de 66 % și la {ro(A['50'])} % cînd doar jumătate "
            f"dintre ei sînt de acord. Un model nu poate fi mai sigur decît oamenii care au etichetat datele.</li>",
            f"<li>Pe propozițiile de test unanime, probabilitățile sale sînt aproape calibrate: eroarea de calibrare așteptată (ECE) este de {ro(C['zs_3B'])} %, față de "
            f"{ro(C['tfidf'])} % pentru un clasificator TF-IDF (frecvența termenilor – inversul frecvenței în documente) antrenat pe "
            f"{ro(D['pb_tr'], 0)} de propoziții.</li>",
            f"<li>Promptul este un hiperparametru: cinci formulări ale aceleiași instrucțiuni schimbă macro-F1 al acestui model cu "
            f"{ro(SN['pb_3B']['range'])} puncte pe aceste propoziții și cu {ro(SN['tw_3B']['range'])} puncte pe tweet-uri "
            f"financiare.</li>"],
    src=f"Date: Financial PhraseBank (Malo et al., 2014; 4.838 de propoziții, 5–8 adnotatori pentru fiecare). Cartonașele unanime: "
        f"setul de test din curs ({D['pb_te']} de propoziții); cartonașele cu adnotatori împărțiți: cele {ro(info['n_amb'], 0)} de "
        f"propoziții cu acord de 50–75 %, nefolosite nici la antrenare, nici ca exemple. Jocul are 10 propoziții pentru fiecare "
        f"nivel de acord (cel mult 40 de cuvinte), extrase aleator (sămînța {SEED}); fiecare joc extrage 3, 3, 2 și 2 dintre ele. "
        f"Rezultatele modelului: ieșirile salvate ale analysis_L12.py. Cod: lectures/L12/make_game_L12.py."))
print("pages written")
