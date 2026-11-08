"""
make_notebook_L9.py -- Lecture 9 notebooks (graph neural networks on the Elliptic Bitcoin graph), EN + RO, executed:
    notebooks/L9_notebook[_ro].ipynb        lecture notebook, sections = the "Experiment yourself" pointers of the slides
    notebooks/L9_seminar[_ro].ipynb         seminar tasks (student); L9_seminar_solutions[_ro].ipynb (instructor, git-ignored)
The notebook trains each model once (seed 1, fewer epochs); the slides report the mean over 3 seeds from
lectures/L9/analysis_L9.py, so the last digits can differ.
Run:  python3 lectures/L9/make_notebook_L9.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Graph neural networks", "Rețele neuronale pe grafuri")

LOAD = """import subprocess, sys, urllib.request
try:
    import torch_geometric
except ImportError:                                   # Google Colab: install PyTorch Geometric
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric'], check=True)
import torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_undirected
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
RAW = '../data/raw/elliptic/'
os.makedirs(RAW, exist_ok=True)
for fn in ['elliptic_txs_features.csv.zip', 'elliptic_txs_edgelist.csv.zip', 'elliptic_txs_classes.csv.zip']:
    if not os.path.exists(RAW + fn):                  # about 146 MB, served by PyTorch Geometric
        urllib.request.urlretrieve('https://data.pyg.org/datasets/elliptic/' + fn, RAW + fn)
feat = pd.read_csv(RAW + 'elliptic_txs_features.csv.zip', header=None)
lab = pd.read_csv(RAW + 'elliptic_txs_classes.csv.zip')
edg = pd.read_csv(RAW + 'elliptic_txs_edgelist.csv.zip')
txid, step = feat[0].values, feat[1].values.astype(int)
X = feat.iloc[:, 2:].values.astype(np.float32)               # 93 local + 72 aggregated features
cls_map = dict(zip(lab.txId, lab['class'].astype(str)))
y = np.array([{'1': 1, '2': 0}.get(cls_map[t], -1) for t in txid])   # 1 illicit, 0 licit, -1 unknown
pos = {t: i for i, t in enumerate(txid)}
E = np.array([[pos[a] for a in edg.txId1], [pos[b] for b in edg.txId2]])
lab_m = y >= 0
train_m, test_m = lab_m & (step <= 34), lab_m & (step > 34)
fit_m, val_m = lab_m & (step < 30), lab_m & (step >= 30) & (step <= 34)
print('nodes', len(y), 'edges', E.shape[1], 'illicit', (y == 1).sum(), 'licit', (y == 0).sum(), 'unknown', (y == -1).sum())
print('labelled training', train_m.sum(), 'labelled test', test_m.sum(), 'illicit in test', (y[test_m] == 1).sum())"""

PREP = """mu, sd = X[step <= 34].mean(0), X[step <= 34].std(0) + 1e-6      # training steps only: no leakage
Xs = np.clip((X - mu) / sd, -10, 10).astype(np.float32)
ei = to_undirected(torch.tensor(E, dtype=torch.long))
xt, yt = torch.tensor(Xs), torch.tensor(np.where(y < 0, 0, y))
W = torch.tensor([0.3, 0.7])                                       # class weights of Weber et al. (2019)

def metrics(p, mask=test_m, thr=0.5):
    yy, pp = y[mask], p[mask]
    return dict(f1=round(f1_score(yy, pp >= thr), 3), prec=round(precision_score(yy, pp >= thr), 3),
                rec=round(recall_score(yy, pp >= thr), 3), prauc=round(average_precision_score(yy, pp), 3))

class GNN(torch.nn.Module):
    def __init__(self, kind, d_in, h=100, layers=2, heads=4, drop=0.2):
        super().__init__()
        self.kind, self.drop = kind, drop
        mk = {'gcn': lambda a: GCNConv(a, h), 'sage': lambda a: SAGEConv(a, h), 'gat': lambda a: GATConv(a, h // heads, heads=heads),
              'mlp': lambda a: torch.nn.Linear(a, h)}[kind]
        self.convs = torch.nn.ModuleList([mk(d_in if i == 0 else h) for i in range(layers)])
        self.out = torch.nn.Linear(h, 2)
    def embed(self, x, ei):
        for c in self.convs:
            x = F.dropout(F.relu(c(x) if self.kind == 'mlp' else c(x, ei)), self.drop, self.training)
        return x
    def forward(self, x, ei):
        return self.out(self.embed(x, ei))

def train(kind, layers=2, epochs=200, patience=30, seed=1):
    torch.manual_seed(seed)
    m = GNN(kind, xt.shape[1], layers=layers); opt = torch.optim.Adam(m.parameters(), lr=0.003, weight_decay=5e-4)
    best, state, wait = -1, None, 0
    fm = torch.tensor(fit_m)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(xt, ei)[fm], yt[fm], weight=W); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(xt, ei), 1)[:, 1].numpy()
        ap = average_precision_score(y[val_m], p[val_m])
        if ap > best:
            best, wait, state = ap, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return torch.softmax(m(xt, ei), 1)[:, 1].numpy(), m.embed(xt, ei).numpy()"""


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 The Elliptic graph\nNodes = Bitcoin transactions, edges = payment flows, 49 time steps; train on steps 1–34, test on 35–49 (Weber et al., 2019).",
            "## §1 Graful Elliptic\nNoduri = tranzacții Bitcoin, muchii = fluxuri de plată, 49 de pași de timp; antrenare pe pașii 1–34, test pe 35–49 (Weber et al., 2019)."))
    nb.code(LOAD)
    nb.code("deg = np.bincount(E[0], minlength=len(y)) + np.bincount(E[1], minlength=len(y))\n"
            "print('mean degree', round(deg.mean(), 2), ' median', int(np.median(deg)), ' max', deg.max())\n"
            "c = pd.DataFrame({'step': step, 'y': y})\ncnt = c[c.y >= 0].groupby(['step', 'y']).size().unstack(fill_value=0)\n"
            "fig, ax = plt.subplots(figsize=(6.4, 2.5))\nax.bar(cnt.index, cnt[0], label='licit'); ax.bar(cnt.index, cnt[1], bottom=cnt[0], label='illicit')\n"
            "ax.axvline(34.5, ls='--', color='grey'); ax.set_xlabel('time step'); fig.legend(loc='lower center', ncol=2)\n"
            "fig.tight_layout(rect=(0, 0.12, 1, 1)); plt.show()")
    nb.code("# label homophily: share of illicit among the labelled neighbours\n"
            "src, dst = np.concatenate([E[0], E[1]]), np.concatenate([E[1], E[0]])\n"
            "ill = np.bincount(src, weights=(y[dst] == 1), minlength=len(y)); labn = np.bincount(src, weights=(y[dst] >= 0), minlength=len(y))\n"
            "for c_, name in [(1, 'illicit'), (0, 'licit')]:\n    m = (y == c_) & (labn > 0)\n"
            "    print(f'illicit share among neighbours of {name} nodes: {100 * np.mean(ill[m] / labn[m]):.1f} %')")
    nb.md(T("## §2 Baselines without the graph\nLogistic regression and a random forest on the 165 features (72 of which are already aggregated over the neighbours).",
            "## §2 Modele de referință fără graf\nRegresie logistică și pădure aleatoare pe cele 165 de variabile (72 dintre ele sînt deja agregate pe vecini)."))
    nb.code(PREP)
    nb.code("lr = LogisticRegression(max_iter=3000, class_weight={0: 0.3, 1: 0.7}).fit(Xs[train_m], y[train_m])\n"
            "p_lr = lr.predict_proba(Xs)[:, 1]\nprint('logistic, all features', metrics(p_lr))\n"
            "rf = RandomForestClassifier(n_estimators=100, max_features=50, n_jobs=-1, random_state=1).fit(X[train_m], y[train_m])\n"
            "p_rf = rf.predict_proba(X)[:, 1]\nprint('random forest         ', metrics(p_rf))")
    nb.md(T("## §3 A graph convolutional network\n$H^{(l+1)}=\\mathrm{ReLU}(\\hat A H^{(l)} W^{(l)})$, $\\hat A=\\tilde D^{-1/2}(A+I)\\tilde D^{-1/2}$; full-batch training, "
            "loss on labelled steps 1–29, early stopping on steps 30–34. The MLP has the same layers without the edges.",
            "## §3 O rețea convoluțională pe graf\n$H^{(l+1)}=\\mathrm{ReLU}(\\hat A H^{(l)} W^{(l)})$, $\\hat A=\\tilde D^{-1/2}(A+I)\\tilde D^{-1/2}$; antrenare pe tot graful, "
            "pierderea pe pașii etichetați 1–29, oprire timpurie pe pașii 30–34. MLP-ul are aceleași straturi, fără muchii."))
    nb.code("p_gcn, emb = train('gcn')\nprint('GCN', metrics(p_gcn))\np_mlp, _ = train('mlp')\nprint('MLP', metrics(p_mlp))")
    nb.code("from sklearn.decomposition import PCA\nZ = PCA(2).fit_transform(emb[test_m])\nyy = y[test_m]\n"
            "fig, ax = plt.subplots(figsize=(6.4, 2.6))\nax.scatter(Z[yy == 0, 0], Z[yy == 0, 1], s=3, alpha=0.3, label='licit')\n"
            "ax.scatter(Z[yy == 1, 0], Z[yy == 1, 1], s=4, alpha=0.6, label='illicit')\nfig.legend(loc='lower center', ncol=2); fig.tight_layout(rect=(0, 0.12, 1, 1)); plt.show()")
    nb.md(T("## §4 GraphSAGE and GAT\nGraphSAGE separates the node from the mean of its neighbours; GAT learns attention weights over the neighbours (4 heads).",
            "## §4 GraphSAGE și GAT\nGraphSAGE separă nodul de media vecinilor; GAT învață ponderi de atenție pe vecini (4 capete)."))
    nb.code("p_sage, _ = train('sage')\nprint('GraphSAGE', metrics(p_sage))\np_gat, _ = train('gat', epochs=120)\nprint('GAT', metrics(p_gat))")
    nb.md(T("## §5 Depth and over-smoothing\n**Experiment yourself:** add 5 and 6 layers to the loop.",
            "## §5 Adîncimea și over-smoothing\n**Exercițiu practic:** adăugați 5 și 6 straturi în buclă."))
    nb.code("for L in [1, 2, 4]:\n    p, e = train('gcn', layers=L, epochs=150)\n    e = e[test_m] / (np.linalg.norm(e[test_m], axis=1, keepdims=True) + 1e-9)\n"
            "    ii = np.random.default_rng(1).choice(len(e), (5000, 2))\n"
            "    print(L, 'layers: PR-AUC', metrics(p)['prauc'], ' mean cosine similarity', round(float(np.mean(np.sum(e[ii[:, 0]] * e[ii[:, 1]], 1))), 2))")
    nb.md(T("## §6 Drift over time and the alert budget",
            "## §6 Schimbarea în timp și bugetul de alerte"))
    nb.code("rows = []\nfor s in range(35, 50):\n    m = test_m & (step == s)\n"
            "    rows.append({'step': s, 'rf': f1_score(y[m], p_rf[m] >= 0.5, zero_division=0), 'gcn': f1_score(y[m], p_gcn[m] >= 0.5, zero_division=0)})\n"
            "f = pd.DataFrame(rows)\nprint(f.round(3).to_string(index=False))\n"
            "for name, p in [('random forest', p_rf), ('GCN', p_gcn)]:\n    o = np.argsort(-p[test_m])[:1000]\n"
            "    print(f'{name}: top 1,000 alerts contain {100 * y[test_m][o].sum() / y[test_m].sum():.1f} % of the illicit, hit rate {100 * y[test_m][o].mean():.1f} %')")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup: the Elliptic graph", "## Pregătire: graful Elliptic"))
    nb.code(LOAD)
    nb.code(PREP)
    nb.task(T("Train a logistic regression on the 93 local features only, then on all 165 features. How much do the 72 neighbourhood aggregates add to the illicit-class F1 and the PR-AUC?",
              "Antrenați o regresie logistică numai pe cele 93 de variabile locale, apoi pe toate cele 165. Cît adaugă cele 72 de agregate pe vecini la F1 pe clasa ilicită și la PR-AUC?"))
    nb.solution("for cols, name in [(slice(0, 93), 'local'), (slice(0, 165), 'all')]:\n"
                "    m = LogisticRegression(max_iter=3000, class_weight={0: 0.3, 1: 0.7}).fit(Xs[train_m][:, cols], y[train_m])\n"
                "    print(name, metrics(m.predict_proba(Xs[:, cols])[:, 1]))")
    nb.task(T("Train the GCN and an MLP with the same layers (no edges). Which one has the larger PR-AUC on the test steps? What does the difference measure?",
              "Antrenați rețeaua GCN și un MLP cu aceleași straturi (fără muchii). Care are PR-AUC mai mare pe pașii de test? Ce măsoară diferența?"))
    nb.solution("p_gcn, emb = train('gcn'); p_mlp, _ = train('mlp')\nprint('GCN', metrics(p_gcn)); print('MLP', metrics(p_mlp))\n# the difference = value of the messages along the edges")
    nb.task(T("Random forest + embeddings: add the 100 hidden GCN outputs to the 165 features and train a random forest (100 trees, 50 features per split). Compare its F1 with the plain forest.",
              "Pădure aleatoare + reprezentări: adăugați cele 100 de ieșiri ascunse ale GCN la cele 165 de variabile și antrenați o pădure aleatoare (100 de arbori, 50 de variabile la fiecare împărțire). Comparați F1 cu pădurea simplă."))
    nb.solution("rf = RandomForestClassifier(n_estimators=100, max_features=50, n_jobs=-1, random_state=1).fit(X[train_m], y[train_m])\n"
                "Xe = np.hstack([X, emb])\nrfe = RandomForestClassifier(n_estimators=100, max_features=50, n_jobs=-1, random_state=1).fit(Xe[train_m], y[train_m])\n"
                "print('forest', metrics(rf.predict_proba(X)[:, 1])); print('forest + GCN', metrics(rfe.predict_proba(Xe)[:, 1]))")
    nb.task(T("Threshold choice: for the GCN scores, find the threshold that maximises the illicit F1 on the validation steps 30–34, then apply it to the test steps. Does F1 improve over 0.5?",
              "Alegerea pragului: pentru scorurile GCN, găsiți pragul care maximizează F1 pe clasa ilicită pe pașii de validare 30–34, apoi aplicați-l pe pașii de test. Se îmbunătățește F1 față de 0,5?"))
    nb.solution("ths = np.linspace(0.05, 0.95, 91)\nf_val = [f1_score(y[val_m], p_gcn[val_m] >= t) for t in ths]\nt_best = ths[int(np.argmax(f_val))]\n"
                "print('threshold', round(t_best, 2), 'test F1', round(f1_score(y[test_m], p_gcn[test_m] >= t_best), 3), 'vs 0.5:', metrics(p_gcn)['f1'])")
    nb.task(T("Drift: compute the illicit F1 of the random forest separately for test steps 35–42 and 43–49. What happened at step 43, and what would you change in production?",
              "Schimbarea în timp: calculați F1 pe clasa ilicită al pădurii aleatoare separat pentru pașii de test 35–42 și 43–49. Ce s-a întîmplat la pasul 43 și ce ați schimba în producție?"))
    nb.solution("p_rf = rf.predict_proba(X)[:, 1]\nfor a, b in [(35, 42), (43, 49)]:\n    m = test_m & (step >= a) & (step <= b)\n"
                "    print(a, b, round(f1_score(y[m], p_rf[m] >= 0.5), 3))\n# a dark market was shut down: new behaviour; monitor F1 per period and retrain on recent labels")


if __name__ == "__main__":
    Notebook.build_all(9, TITLE, lecture, kind="lecture")
    Notebook.build_all(9, TITLE, seminar, kind="seminar")
