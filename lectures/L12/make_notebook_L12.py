"""
make_notebook_L12.py -- Lecture 12 notebooks (EN + RO): lecture notebook (sections = the 'Experiment yourself' pointers of the
deck) and seminar notebook (student + solutions), all executed.
The full runs of the language models (about 40 minutes on a laptop graphics processor) are done by analysis_L12.py; the notebook
runs the smallest model (Qwen2.5-0.5B-Instruct) live on small samples and reads the full results from lectures/L12/results/.
Run:  python3 lectures/L12/make_notebook_L12.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Large language models in finance and prompt engineering", "Modele lingvistice mari în finanțe și formularea instrucțiunilor")
SETUP = ("RES = '../lectures/L12/results/'\nimport json, re, torch\nfrom sklearn.metrics import f1_score, accuracy_score\n"
         "from transformers import AutoTokenizer, AutoModelForCausalLM\n"
         "PB = pd.read_csv(DATA + 'financial_phrasebank.csv')\nSP = pd.read_csv(DATA + 'ecb_speeches_2022plus.csv', parse_dates=['date'])\n"
         "LAB = ['positive', 'negative', 'neutral']\nprint(len(PB), 'sentences;', len(SP), 'ECB speeches')")
MODEL = ("name = 'Qwen/Qwen2.5-0.5B-Instruct'\ntok = AutoTokenizer.from_pretrained(name)\n"
         "lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if DEVICE != 'cpu' else torch.float32).to(DEVICE).eval()")


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 Tokens and perplexity", "## §1 Tokeni și perplexitate"))
    nb.code(SETUP)
    nb.code(MODEL + "\nwords = PB.sentence.str.split().str.len().sum()\ntoks = sum(len(tok.encode(s)) for s in PB.sentence)\n"
            "print('Qwen tokens per word:', round(toks / words, 2))\nprint(tok.tokenize(PB.sentence[0]))")
    nb.code("txt = SP.sort_values('date').contents.iloc[-1][:4000]\nids = tok(txt, return_tensors='pt').input_ids[:, :512].to(DEVICE)\n"
            "with torch.no_grad():\n    nll = torch.nn.functional.cross_entropy(lm(ids).logits[0, :-1].float(), ids[0, 1:])\n"
            "print('perplexity of Qwen2.5-0.5B on the latest ECB speech (512 tokens):', round(float(torch.exp(nll)), 2))\n"
            "J = json.load(open(RES + 'lm_outputs.json'))\nprint({s: round(float(np.exp(v['new'])), 2) for s, v in J['ppl_nll'].items()})")
    nb.md(T("## §2 Next-token probabilities and temperature", "## §2 Probabilitățile următorului token și temperatura"))
    nb.code("prompt = 'Inflation in the euro area is expected to'\nids = tok(prompt, return_tensors='pt').input_ids.to(DEVICE)\n"
            "with torch.no_grad():\n    z = lm(ids).logits[0, -1].float().cpu()\nfor Tm in (0.5, 1.0, 2.0):\n    p = torch.softmax(z / Tm, -1)\n"
            "    top = torch.topk(p, 5)\n    print(Tm, [(tok.decode([i]), round(float(v), 3)) for i, v in zip(top.indices, top.values)])")
    nb.md(T("§ *Experiment yourself*: change the prompt to a sentence of your own and watch how the entropy moves with the temperature.",
            "§ *Exercițiu practic*: schimbați promptul cu o propoziție proprie și urmăriți cum se modifică entropia cu temperatura."))
    nb.md(T("## §3 A prompt as a classifier (live on 60 sentences, full results from the analysis)", "## §3 Un prompt ca un clasificator (live pe 60 de propoziții, rezultatele complete din analiză)"))
    nb.code("ids_lab = {l: sorted({tok.encode(v)[0] for v in (l, ' ' + l, l.capitalize(), ' ' + l.capitalize())}) for l in LAB}\n"
            "INSTR = ('Classify the sentiment of this sentence from financial news, from the point of view of an investor in the company. '\n"
            "         'Answer with one word: positive, negative or neutral.')\n"
            "def zero_shot(sentences):\n    out = []\n    for s in sentences:\n"
            "        msgs = [{'role': 'system', 'content': 'You are a careful financial analyst.'}, {'role': 'user', 'content': INSTR + '\\n\\nSentence: ' + s + '\\nAnswer:'}]\n"
            "        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors='pt').input_ids.to(DEVICE)\n"
            "        with torch.no_grad():\n            p = torch.softmax(lm(ids).logits[0, -1].float(), -1)\n"
            "        out.append(LAB[int(np.argmax([float(p[ids_lab[l]].sum()) for l in LAB]))])\n    return out\n"
            "P = pd.read_csv(RES + 'predictions_pb.csv')\nsmall = P.sample(60, random_state=SEED)\n"
            "pred = zero_shot(small.text.tolist())\nprint('live zero-shot 0.5B, 60 sentences, accuracy:', round(accuracy_score(small.label, pred), 3))")
    nb.code("def f1(df, m):\n    return f1_score(df.label, [LAB[i] for i in df[[f'{m}_{l}' for l in LAB]].values.argmax(1)], labels=LAB, average='macro')\n"
            "for m in ['tfidf', 'zs_0.5B', 'zs_1.5B', 'zs_3B', 'fs_1.5B', 'fs_3B', 'cot_3B', 'probe']:\n    print(m, round(100 * f1(P, m), 1))")
    nb.md(T("## §4 Few-shot examples and prompt sensitivity", "## §4 Exemple few-shot și sensibilitatea la prompt"))
    nb.code("L = pd.read_csv(RES + 'fewshot_labels_pb.csv')\nfor s in ('1.5B', '3B'):\n"
            "    print(s, [round(100 * np.mean([f1_score(L.label, L[f'{s}_k{k}_s{sd}'], labels=LAB, average='macro') for sd in (0, 1, 2)]), 1) for k in (0, 1, 2, 4, 8)])\n"
            "S = pd.read_csv(RES + 'sensitivity_labels_pb.csv')\nprint('3B, five wordings:', [round(100 * f1_score(S.label, S[f'3B_i{i}'], labels=LAB, average='macro'), 1) for i in range(1, 6)])")
    nb.md(T("## §5 Chain-of-thought", "## §5 Chain-of-thought"))
    nb.code("msgs = [{'role': 'user', 'content': INSTR.split(' Answer')[0] + ' First explain your reasoning in at most two short sentences, then give the final answer as Label: <positive|negative|neutral>.\\n\\nSentence: ' + small.text.iloc[0]}]\n"
            "ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors='pt').input_ids.to(DEVICE)\n"
            "with torch.no_grad():\n    g = lm.generate(ids, max_new_tokens=80, do_sample=False)\nprint(small.text.iloc[0])\nprint(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))")
    nb.md(T("## §6 Calibration, annotator agreement, cost", "## §6 Calibrare, acordul adnotatorilor, cost"))
    nb.code("def ece(m, bins=10):\n    Pm = P[[f'{m}_{l}' for l in LAB]].values\n    conf = Pm.max(1)\n    ok = np.array([LAB[i] == t for i, t in zip(Pm.argmax(1), P.label)])\n"
            "    edges = np.linspace(1 / 3, 1, bins + 1)\n    e = 0.0\n    for a, b in zip(edges[:-1], edges[1:]):\n"
            "        msk = (conf >= a) & (conf <= b if b == 1 else conf < b)\n        e += msk.mean() * abs(conf[msk].mean() - ok[msk].mean()) if msk.sum() else 0\n    return round(100 * e, 1)\n"
            "print({m: ece(m) for m in ('zs_3B', 'fs_3B', 'tfidf')})\nA = pd.read_csv(RES + 'agreement_labels.csv')\n"
            "print(A.groupby('agreement').apply(lambda g: round(100 * accuracy_score(g.label, g.zs_3B), 1)))\nprint(J['ms_per_item']['pb'])")
    nb.md(T("## §7 Retrieval on ECB speeches: TF-IDF and BM25 live", "## §7 Regăsirea în discursurile BCE: TF-IDF și BM25, live"))
    nb.code("from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "def body(r):\n    c = r.contents\n    h = f'SPEECH {r.title} {r.subtitle}'\n    c = c[len(h):] if c.startswith(h) else c\n"
            "    m = re.search(r'\\b(19|20)\\d{2}\\b', c[:120])\n    return c[m.end():].strip() if m else c\n"
            "SP = SP.sort_values('date').reset_index(drop=True)\nchunks, owner = [], []\nfor i, r in SP.iterrows():\n    w = body(r).split()\n"
            "    for j in range(0, max(1, len(w) - 50), 150):\n        chunks.append(' '.join(w[j:j + 200])); owner.append(i)\n        if j + 200 >= len(w):\n            break\n"
            "owner = np.array(owner)\nvec = TfidfVectorizer(stop_words='english', sublinear_tf=True)\nC = vec.fit_transform(chunks)\nS = (vec.transform(SP.title) @ C.T).toarray()\n"
            "rank = [int(np.where(pd.Series(S[q]).groupby(owner).max().sort_values(ascending=False).index.values == q)[0][0]) + 1 for q in range(len(SP))]\n"
            "print(len(chunks), 'chunks; TF-IDF recall@1 =', round(100 * np.mean(np.array(rank) <= 1), 1), '%')\n"
            "R = pd.read_csv(RES + 'rag_ranks.csv')\nprint({m: round(100 * float((R[f'rank_{m}'] <= 1).mean()), 1) for m in ('tfidf', 'bm25', 'dense', 'hybrid')})")
    nb.md(T("## §8 RAG against closed-book answers", "## §8 RAG față de răspunsurile fără documente"))
    nb.code("Q = pd.read_csv(RES + 'rag_qa.csv')\nprint(pd.DataFrame({m: Q[m].value_counts(normalize=True).mul(100).round(1) for m in ('closed', 'rag', 'oracle')}))")
    nb.md(T("## §9 LLM-VaR backtest", "## §9 Testarea retrospectivă a LLM-VaR"))
    nb.code("from scipy import stats\nV = pd.read_csv(RES + 'llmvar_sp500.csv', parse_dates=['date']).set_index('date')\n"
            "def kupiec(h, a):\n    n, k = len(h), int(h.sum()); p = k / n\n"
            "    lr = -2 * ((n - k) * np.log(1 - a) + k * np.log(a) - (n - k) * np.log(1 - p) - k * np.log(p)) if k else -2 * n * np.log(1 - a)\n"
            "    return round(float(1 - stats.chi2.cdf(lr, 1)), 3)\n"
            "for m in ('llm', 'hs', 'garch'):\n    for a, s in ((0.05, '05'), (0.01, '01')):\n        h = (V.r < V[f'{m}_{s}']).values\n"
            "        print(m, s, 'violations', int(h.sum()), 'expected', round(a * len(h), 1), 'Kupiec p', kupiec(h, a))")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(SETUP)
    nb.task(T("Count the Qwen2.5 tokens per word of the 2026 ECB speeches. Is it higher or lower than for the PhraseBank sentences?",
              "Numărați tokenii Qwen2.5 pe cuvînt pentru discursurile BCE din 2026. Este valoarea mai mare sau mai mică decît pentru propozițiile PhraseBank?"))
    nb.solution("tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')\ns26 = SP[SP.date.dt.year == 2026].contents\n"
                "print(round(sum(len(tok.encode(c)) for c in s26) / s26.str.split().str.len().sum(), 2))\n"
                "print(round(sum(len(tok.encode(c)) for c in PB.sentence) / PB.sentence.str.split().str.len().sum(), 2))")
    nb.task(T("Train TF-IDF + logistic regression on the Twitter training split and report the macro-F1 on the published validation split.",
              "Antrenați TF-IDF + regresie logistică pe partea de antrenare Twitter și raportați macro-F1 pe partea de validare publicată."))
    nb.solution("from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\n"
                "TW = pd.read_csv(DATA + 'twitter_fin_sentiment.csv')\ntr, va = TW[TW.split == 'train'], TW[TW.split == 'valid']\n"
                "vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)\nclf = LogisticRegression(max_iter=3000, class_weight='balanced', C=3).fit(vec.fit_transform(tr.text), tr.label_name)\n"
                "print(round(100 * f1_score(va.label_name, clf.predict(vec.transform(va.text)), average='macro'), 1))")
    nb.task(T("Using `predictions_pb.csv`, compute a 95 % bootstrap interval (1,000 resamples) for the macro-F1 of zero-shot 3B. Does it overlap with the interval of TF-IDF?",
              "Folosind `predictions_pb.csv`, calculați un interval bootstrap de 95 % (1.000 de reeșantionări) pentru macro-F1 al modelului zero-shot 3B. Se suprapune cu intervalul TF-IDF?"))
    nb.solution("P = pd.read_csv(RES + 'predictions_pb.csv')\nrng = np.random.default_rng(SEED)\n"
                "def ci(m):\n    pred = np.array([LAB[i] for i in P[[f'{m}_{l}' for l in LAB]].values.argmax(1)]); y = P.label.values\n"
                "    v = [f1_score(y[i], pred[i], labels=LAB, average='macro') for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]\n"
                "    return np.round(100 * np.percentile(v, [2.5, 97.5]), 1)\nprint('zero-shot 3B', ci('zs_3B'), ' TF-IDF', ci('tfidf'))")
    nb.task(T("Write your own instruction for zero-shot classification and test it with Qwen2.5-0.5B on 50 PhraseBank sentences with 100 % agreement. Compare with the instruction of the lecture.",
              "Scrieți propria instrucțiune de clasificare zero-shot și testați-o cu Qwen2.5-0.5B pe 50 de propoziții PhraseBank cu acord de 100 %. Comparați cu instrucțiunea din curs."))
    nb.solution(MODEL + "\nids_lab = {l: sorted({tok.encode(v)[0] for v in (l, ' ' + l, l.capitalize(), ' ' + l.capitalize())}) for l in LAB}\n"
                "def run(instr, sents):\n    out = []\n    for s in sents:\n"
                "        ids = tok(tok.apply_chat_template([{'role': 'user', 'content': instr + '\\n\\nSentence: ' + s + '\\nAnswer:'}], tokenize=False, add_generation_prompt=True), return_tensors='pt').input_ids.to(DEVICE)\n"
                "        with torch.no_grad():\n            p = torch.softmax(lm(ids).logits[0, -1].float(), -1)\n        out.append(LAB[int(np.argmax([float(p[ids_lab[l]].sum()) for l in LAB]))])\n    return out\n"
                "S50 = PB[PB.min_agreement_pct == 100].sample(50, random_state=SEED)\n"
                "for instr in ['Classify the sentiment of this sentence from financial news for an investor. Answer with one word: positive, negative or neutral.',\n"
                "              'Does this news make the company look better, worse or unchanged? Answer positive, negative or neutral.']:\n"
                "    print(round(accuracy_score(S50.label, run(instr, S50.sentence)), 3), instr[:60])")
    nb.task(T("Implement BM25 (k1 = 1.5, b = 0.75) on 200-word chunks of the ECB speeches and report recall@5 when the query is the speech title.",
              "Implementați BM25 (k1 = 1,5, b = 0,75) pe fragmente de 200 de cuvinte din discursurile BCE și raportați recall@5 cînd interogarea este titlul discursului."))
    nb.solution("from sklearn.feature_extraction.text import CountVectorizer\nSP = SP.sort_values('date').reset_index(drop=True)\nchunks, owner = [], []\n"
                "for i, c in enumerate(SP.contents):\n    w = c.split()\n    for j in range(0, max(1, len(w) - 50), 150):\n        chunks.append(' '.join(w[j:j + 200])); owner.append(i)\n"
                "owner = np.array(owner)\ncv = CountVectorizer(stop_words='english')\nDm = cv.fit_transform(chunks).tocsc().astype(float)\ndl = np.asarray(Dm.sum(1)).ravel()\n"
                "df = np.asarray((Dm > 0).sum(0)).ravel()\nidf = np.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))\nQm = cv.transform(SP.title)\nhits = []\n"
                "for q in range(len(SP)):\n    s = np.zeros(len(chunks))\n    for t in Qm[q].indices:\n        f = Dm[:, t].toarray().ravel()\n"
                "        s += idf[t] * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * dl / dl.mean()))\n"
                "    top = pd.Series(s).groupby(owner).max().sort_values(ascending=False).index[:5]\n    hits.append(q in top)\n"
                "print('recall@5:', round(100 * np.mean(hits), 1), '%  (here the chunks keep the title line, so the task is easier than in the lecture)')")
    nb.task(T("Run the Kupiec test for the 1 % LLM-VaR in `llmvar_sp500.csv`. Then compute the share of days on which the LLM-VaR was below the GARCH VaR.",
              "Aplicați testul Kupiec pentru LLM-VaR de 1 % din `llmvar_sp500.csv`. Apoi calculați ponderea zilelor în care LLM-VaR a fost sub VaR GARCH."))
    nb.solution("from scipy import stats\nV = pd.read_csv(RES + 'llmvar_sp500.csv')\nh = (V.r < V.llm_01).values\nn, k = len(h), int(h.sum()); p = max(k / n, 1e-12)\n"
                "lr = -2 * ((n - k) * np.log(0.99) + k * np.log(0.01) - (n - k) * np.log(1 - p) - k * np.log(p)) if k else -2 * n * np.log(0.99)\n"
                "print('violations', k, 'Kupiec p', round(float(1 - stats.chi2.cdf(lr, 1)), 3))\nprint('LLM below GARCH on', round(100 * float((V.llm_01 < V.garch_01).mean()), 1), '% of days')")


if __name__ == "__main__":
    Notebook.build_all(12, TITLE, lecture, kind="lecture")
    Notebook.build_all(12, TITLE, seminar, kind="seminar")
