"""
make_notebook_L7.py -- Lecture 7 notebooks (attention and transformers), EN + RO, executed:
    notebooks/L7_notebook[_ro].ipynb        lecture notebook, sections = the "Experiment yourself" pointers of the slides
    notebooks/L7_seminar[_ro].ipynb         seminar tasks (student); L7_seminar_solutions[_ro].ipynb (instructor, git-ignored)
The notebook trains each model once (seed 1); FinBERT is fine-tuned for one epoch only. The slides report the means over
3 seeds from lectures/L7/analysis_L7.py, so the last digits can differ.
Run:  python3 lectures/L7/make_notebook_L7.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Attention and transformers", "Atenția și transformerele")

LOAD = """import math, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tw = pd.read_csv(DATA + 'twitter_fin_sentiment.csv')            # 0 bearish, 1 bullish, 2 neutral
tr_all, te = tw[tw.split == 'train'].reset_index(drop=True), tw[tw.split == 'valid'].reset_index(drop=True)
tr, va = train_test_split(tr_all, test_size=0.10, stratify=tr_all.label, random_state=SEED)
tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
ytr, yva, yte = tr.label.values, va.label.values, te.label.values
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
def enc(texts, n=64):
    e = tok(list(texts), padding='max_length', truncation=True, max_length=n, return_tensors='pt')
    return e['input_ids'], e['attention_mask']
Itr, Mtr = enc(tr.text); Iva, Mva = enc(va.text); Ite, Mte = enc(te.text)
print('train', len(tr), 'validation', len(va), 'test', len(te), ' vocabulary', tok.vocab_size)"""


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 Data and tokens\nFinance tweets (bearish, bullish, neutral); the published validation split is our test set. Text lengths in WordPiece tokens.",
            "## §1 Datele și tokenii\nTweet-uri financiare (negative, pozitive, neutre); setul de validare publicat este setul nostru de test. Lungimea textelor în tokeni WordPiece."))
    nb.code(LOAD)
    nb.code("print(tok.tokenize(te.text[0]))\nsp = pd.read_csv(DATA + 'ecb_speeches_2022plus.csv')\n"
            "len_tw = np.array([len(x) for x in tok(list(tw.text))['input_ids']])\nlen_sp = np.array([len(x) for x in tok(list(sp.contents.astype(str)))['input_ids']])\n"
            "print('median tokens: tweets', int(np.median(len_tw)), ' ECB speeches', int(np.median(len_sp)), ' speeches > 512:', round(100 * np.mean(len_sp > 512), 1), '%')")
    nb.md(T("## §2 Scaled dot-product attention by hand\n$\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}(QK^\\top/\\sqrt{d_k})V$, checked against the first layer of FinBERT on a real tweet.",
            "## §2 Atenția cu produs scalar scalat, calculată manual\n$\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}(QK^\\top/\\sqrt{d_k})V$, verificată pe primul strat al FinBERT, pentru un tweet real."))
    nb.code("fb = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', attn_implementation='eager').eval()\n"
            "e1 = tok(te.text[0], return_tensors='pt')\nwith torch.no_grad():\n    out = fb(**e1, output_attentions=True, output_hidden_states=True)\n"
            "X0 = out.hidden_states[0][0]                       # input of layer 1: n x 768\nat = fb.bert.encoder.layer[0].attention.self\n"
            "Q, K = at.query(X0), at.key(X0)                    # all 12 heads, 64 dimensions each\nh = 0; q, k = Q[:, 64*h:64*(h+1)], K[:, 64*h:64*(h+1)]\n"
            "A = torch.softmax(q @ k.T / math.sqrt(64), dim=-1)\nprint('max difference with the model:', float((A - out.attentions[0][0, h]).abs().max()))\n"
            "print('row sums:', A.sum(1)[:3].tolist())\nraw = (q @ k.T).flatten(); print('sd of raw scores', round(float(raw.std()), 2), ' scaled', round(float((raw / 8).std()), 2))")
    nb.md(T("## §3 A small transformer from scratch\nToken + sinusoidal position embeddings, 2 encoder blocks (4 heads, $d=64$), mean pooling, 3 classes. **Experiment yourself:** set `pos=False`.",
            "## §3 Un transformer mic, de la zero\nReprezentările tokenilor și pozițiile sinusoidale, 2 blocuri (4 capete, $d=64$), media pe tokeni, 3 clase. **Exercițiu practic:** puneți `pos=False`."))
    nb.code("""class TinyTransformer(nn.Module):
    def __init__(self, vocab, d=64, heads=4, layers=2, ff=128, drop=0.1, pos=True, n=64):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d, padding_idx=0), pos
        p = torch.arange(n)[:, None]; div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(n, d); pe[:, 0::2], pe[:, 1::2] = torch.sin(p * div), torch.cos(p * div); self.register_buffer('pe', pe)
        blk = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(blk, layers, enable_nested_tensor=False); self.drop = nn.Dropout(drop); self.head = nn.Linear(d, 3)
    def forward(self, ids, mask):
        h = self.emb(ids) * math.sqrt(self.emb.embedding_dim) + (self.pe[:ids.shape[1]] if self.pos else 0)
        h = self.enc(self.drop(h), src_key_padding_mask=(mask == 0)); m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1))

def train_tiny(pos=True, seed=1, epochs=20, patience=4):
    torch.manual_seed(seed); model = TinyTransformer(tok.vocab_size, pos=pos).to(DEVICE)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr)), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01); best, state, wait = -1, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(ytr))
        for b in range(0, len(perm), 64):
            j = perm[b:b + 64]; loss = F.cross_entropy(model(Itr[j].to(DEVICE), Mtr[j].to(DEVICE)), torch.tensor(ytr)[j].to(DEVICE), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            fv = f1_score(yva, model(Iva.to(DEVICE), Mva.to(DEVICE)).argmax(1).cpu(), average='macro')
        print(f'epoch {ep + 1}: validation macro-F1 {fv:.3f}')
        if fv > best: best, wait, state = fv, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return model(Ite.to(DEVICE), Mte.to(DEVICE)).argmax(1).cpu().numpy()

pred_tiny = train_tiny()
print('small transformer, test macro-F1:', round(100 * f1_score(yte, pred_tiny, average='macro'), 1))""")
    nb.md(T("## §4 Attention inside FinBERT\nLast-layer attention averaged over the 12 heads for one test tweet; the entropy of attention by layer.",
            "## §4 Atenția din interiorul FinBERT\nAtenția din ultimul strat, mediată pe cele 12 capete, pentru un tweet de test; entropia atenției pe straturi."))
    nb.code("toks = tok.convert_ids_to_tokens(e1['input_ids'][0])\nA12 = out.attentions[-1][0].mean(0).numpy()\n"
            "fig, ax = plt.subplots(figsize=(5, 4.5))\nax.imshow(A12, cmap='Oranges')\nax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=90, fontsize=7)\n"
            "ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=7); plt.show()\n"
            "ent = []\nwith torch.no_grad():\n    o = fb(input_ids=Ite[:100], attention_mask=Mte[:100], output_attentions=True)\n"
            "for A in o.attentions:\n    vals = []\n    for r in range(100):\n        n = int(Mte[r].sum()); a = A[r, :, :n, :n].clamp_min(1e-12)\n"
            "        vals.append(float((-(a * a.log()).sum(-1) / math.log(n)).mean()))\n    ent.append(np.mean(vals))\nprint('normalised entropy by layer:', np.round(ent, 2))")
    nb.md(T("## §5 Benchmarks, zero-shot FinBERT, frozen FinBERT and fine-tuning\nFinBERT labels (positive, negative, neutral) are mapped to (bullish, bearish, neutral). Fine-tuning: one epoch here, three in the slides.",
            "## §5 Modele de referință, FinBERT fără antrenare, FinBERT înghețat și ajustarea fină\nEtichetele FinBERT (positive, negative, neutral) devin (pozitiv, negativ, neutru). Ajustarea fină: o epocă aici, trei în slide-uri."))
    nb.code("vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)\nXtr = vec.fit_transform(tr.text)\n"
            "p_tfidf = LogisticRegression(C=10, max_iter=3000, class_weight='balanced').fit(Xtr, ytr).predict(vec.transform(te.text))\n"
            "fb = fb.to(DEVICE); fin2our = {0: 1, 1: 0, 2: 2}\n"
            "def run(model, I, M, hidden=False):\n    L, H = [], []\n    with torch.no_grad():\n        for b in range(0, len(I), 256):\n"
            "            o = model(input_ids=I[b:b+256].to(DEVICE), attention_mask=M[b:b+256].to(DEVICE), output_hidden_states=hidden)\n"
            "            L.append(o.logits.float().cpu())\n            if hidden:\n                m = M[b:b+256].to(DEVICE).unsqueeze(-1).float(); H.append(((o.hidden_states[-1] * m).sum(1) / m.sum(1)).float().cpu())\n"
            "    return torch.cat(L), (torch.cat(H).numpy() if hidden else None)\n"
            "lg, hte = run(fb, Ite, Mte, hidden=True); _, htr = run(fb, Itr, Mtr, hidden=True)\np_zs = np.array([fin2our[int(i)] for i in lg.argmax(1)])\n"
            "p_probe = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(htr, ytr).predict(hte)\n"
            "for name, p in [('TF-IDF', p_tfidf), ('small transformer', pred_tiny), ('FinBERT zero-shot', p_zs), ('FinBERT frozen + logistic', p_probe)]:\n"
            "    print(f'{name:26s} macro-F1 {100 * f1_score(yte, p, average=\"macro\"):.1f}  accuracy {100 * accuracy_score(yte, p):.1f}')")
    nb.code("# fine-tuning: all weights, AdamW 2e-5, one epoch, batch 32\nour2fin = {0: 1, 1: 0, 2: 2}; yfin = torch.tensor([our2fin[int(v)] for v in ytr])\n"
            "torch.manual_seed(1); fb.train(); opt = torch.optim.AdamW(fb.parameters(), lr=2e-5, weight_decay=0.01)\nperm = torch.randperm(len(ytr))\n"
            "for b in range(0, len(perm), 32):\n    j = perm[b:b+32]; loss = F.cross_entropy(fb(input_ids=Itr[j].to(DEVICE), attention_mask=Mtr[j].to(DEVICE)).logits, yfin[j].to(DEVICE))\n"
            "    opt.zero_grad(); loss.backward(); opt.step()\nfb.eval(); lg, _ = run(fb, Ite, Mte)\np_ft = np.array([fin2our[int(i)] for i in lg.argmax(1)])\n"
            "print('FinBERT fine-tuned (1 epoch) macro-F1', round(100 * f1_score(yte, p_ft, average='macro'), 1))\nprint(confusion_matrix(yte, p_ft))")
    nb.md(T("## §6 How much labelled data do you need?\nTF-IDF and frozen FinBERT trained on 5 %, 25 % and 100 % of the training tweets.",
            "## §6 De cîte date etichetate este nevoie?\nTF-IDF și FinBERT înghețat, antrenate pe 5 %, 25 % și 100 % dintre tweet-urile de antrenare."))
    nb.code("for fr in [0.05, 0.25, 1.0]:\n    idx = np.arange(len(ytr)) if fr == 1 else train_test_split(np.arange(len(ytr)), train_size=fr, stratify=ytr, random_state=1)[0]\n"
            "    v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); xa = v2.fit_transform(tr.text.iloc[idx])\n"
            "    f_t = f1_score(yte, LogisticRegression(C=10, max_iter=3000, class_weight='balanced').fit(xa, ytr[idx]).predict(v2.transform(te.text)), average='macro')\n"
            "    f_p = f1_score(yte, LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(htr[idx], ytr[idx]).predict(hte), average='macro')\n"
            "    print(f'{len(idx):5d} tweets: TF-IDF {100 * f_t:.1f}   frozen FinBERT {100 * f_p:.1f}')")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(LOAD)
    nb.task(T("TF-IDF benchmark: compare unigrams with unigrams + bigrams (logistic regression, $C=10$, balanced class weights). Report macro-F1 and the recall of the bearish class on the test tweets.",
              "Modelul de referință TF-IDF: comparați unigramele cu unigramele + bigramele (regresie logistică, $C=10$, ponderi de clasă echilibrate). Raportați macro-F1 și recall-ul clasei negative pe tweet-urile de test."))
    nb.solution("from sklearn.metrics import recall_score\nfor ng in [(1, 1), (1, 2)]:\n    v = TfidfVectorizer(ngram_range=ng, min_df=2, sublinear_tf=True); x = v.fit_transform(tr.text)\n"
                "    p = LogisticRegression(C=10, max_iter=3000, class_weight='balanced').fit(x, ytr).predict(v.transform(te.text))\n"
                "    print(ng, round(100 * f1_score(yte, p, average='macro'), 1), round(100 * recall_score(yte, p, labels=[0], average='macro'), 1))")
    nb.task(T("Attention by hand: for the first test tweet, take the input of FinBERT's first layer and the query and key projections of head 3; compute the attention matrix and check that it equals the one returned by the model.",
              "Atenția calculată manual: pentru primul tweet de test, luați intrarea primului strat FinBERT și proiecțiile query și key ale capului 3; calculați matricea de atenție și verificați că este egală cu cea dată de model."))
    nb.solution("fb = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', attn_implementation='eager').eval()\ne1 = tok(te.text[0], return_tensors='pt')\n"
                "with torch.no_grad():\n    out = fb(**e1, output_attentions=True, output_hidden_states=True)\n    X0 = out.hidden_states[0][0]; at = fb.bert.encoder.layer[0].attention.self\n"
                "    h = 3; q = at.query(X0)[:, 64*h:64*(h+1)]; k = at.key(X0)[:, 64*h:64*(h+1)]\n    A = torch.softmax(q @ k.T / 8, -1)\nprint(float((A - out.attentions[0][0, h]).abs().max()))")
    nb.task(T("Why $\\sqrt{d_k}$? With the same projections, compute the standard deviation of the raw scores $\\mathbf q^\\top\\mathbf k$ and of the scaled scores, and the largest attention weight in each row with and without scaling.",
              "De ce $\\sqrt{d_k}$? Cu aceleași proiecții, calculați abaterea standard a scorurilor brute $\\mathbf q^\\top\\mathbf k$ și a celor scalate, precum și cea mai mare pondere de atenție din fiecare rînd, cu și fără scalare."))
    nb.solution("with torch.no_grad():\n    s = q @ k.T\n    print('sd raw', round(float(s.std()), 2), ' sd scaled', round(float((s / 8).std()), 2))\n"
                "    print('max weight per row, scaled  :', torch.softmax(s / 8, -1).max(1).values.mean().item())\n    print('max weight per row, unscaled:', torch.softmax(s, -1).max(1).values.mean().item())")
    nb.task(T("Zero-shot FinBERT: map its labels (positive, negative, neutral) to (bullish, bearish, neutral) and compute the confusion matrix on the test tweets. Which error is the most frequent?",
              "FinBERT fără antrenare: asociați etichetele lui (positive, negative, neutral) cu (pozitiv, negativ, neutru) și calculați matricea de confuzie pe tweet-urile de test. Care eroare este cea mai frecventă?"))
    nb.solution("fin2our = {0: 1, 1: 0, 2: 2}; fb = fb.to(DEVICE)\nwith torch.no_grad():\n"
                "    lg = torch.cat([fb(input_ids=Ite[b:b+256].to(DEVICE), attention_mask=Mte[b:b+256].to(DEVICE)).logits.cpu() for b in range(0, len(yte), 256)])\n"
                "p = np.array([fin2our[int(i)] for i in lg.argmax(1)])\nprint(confusion_matrix(yte, p)); print(round(100 * f1_score(yte, p, average='macro'), 1))")
    nb.task(T("Long documents: split the three longest European Central Bank speeches into chunks of 510 tokens, score each chunk with zero-shot FinBERT, and report the share of negative chunks per speech.",
              "Documente lungi: împărțiți cele trei cele mai lungi discursuri ale Băncii Centrale Europene în bucăți de 510 tokeni, dați un scor fiecărei bucăți cu FinBERT fără antrenare și raportați ponderea bucăților negative în fiecare discurs."))
    nb.solution("sp = pd.read_csv(DATA + 'ecb_speeches_2022plus.csv').nlargest(3, 'n_words')\nfor _, r in sp.iterrows():\n"
                "    ids = tok(str(r.contents), add_special_tokens=False)['input_ids']\n    chunks = [ids[i:i + 510] for i in range(0, len(ids), 510)]\n"
                "    neg = 0\n    for c in chunks:\n        x = torch.tensor([[tok.cls_token_id] + c + [tok.sep_token_id]]).to(DEVICE)\n"
                "        with torch.no_grad():\n            neg += int(fb(input_ids=x).logits.argmax(1).item() == 1)\n"
                "    print(r.date[:10], len(chunks), 'chunks,', round(100 * neg / len(chunks), 1), '% negative')")


if __name__ == "__main__":
    Notebook.build_all(7, TITLE, lecture, kind="lecture")
    Notebook.build_all(7, TITLE, seminar, kind="seminar")
