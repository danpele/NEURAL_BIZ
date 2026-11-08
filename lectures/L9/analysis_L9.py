"""
analysis_L9.py -- Lecture 9 "Graph neural networks": all computation on the Elliptic Bitcoin transaction graph.
Writes lectures/L9/L9_numbers.json, charts/en|ro/*.pdf and lectures/L9/results/*.csv (small files read by the Quantlets).

Data: Elliptic (Weber et al., 2019), on demand: data/raw/elliptic/*.csv.zip (python3 data/download_all.py elliptic).
Design (as in Weber et al., 2019): temporal split, training = time steps 1-34, test = time steps 35-49 (used once);
the last 5 training steps (30-34) are the validation part for early stopping of the neural networks.
Features are standardised with the mean and standard deviation of the training steps only.
Neural models: 3 seeds each (mean and standard deviation reported); trees: 3 seeds.
Run:  python3 lectures/L9/analysis_L9.py      (about 6-8 minutes on a laptop CPU; trained results cached in cache/)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import nn_plot as npl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.nn import GCNConv, SAGEConv, GATConv  # noqa: E402
from torch_geometric.utils import to_undirected  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

npl.style()
SEED = npl.seed_all()
N = {"charts": {}, "seed": SEED}
CACHE = os.path.join(HERE, "cache")
RES = os.path.join(HERE, "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RES, exist_ok=True)
RAWD = os.path.join(npl.RAW, "elliptic")
T_TRAIN, T_VAL = 34, 30          # train = steps 1..34 (validation for early stopping = 30..34); test = 35..49
SEEDS = [1, 2, 3]
torch.set_num_threads(8)
t0 = time.time()

# ---------------------------------------------------------------- 0. data
fcache = os.path.join(CACHE, "elliptic.npz")
if os.path.exists(fcache):
    z = np.load(fcache, allow_pickle=True)
    txid, X, step, cls, E = z["txid"], z["X"], z["step"], z["cls"], z["E"]
else:
    feat = pd.read_csv(os.path.join(RAWD, "elliptic_txs_features.csv.zip"), header=None)
    lab = pd.read_csv(os.path.join(RAWD, "elliptic_txs_classes.csv.zip"))
    edg = pd.read_csv(os.path.join(RAWD, "elliptic_txs_edgelist.csv.zip"))
    txid = feat[0].values
    step = feat[1].values.astype(int)
    X = feat.iloc[:, 2:].values.astype(np.float32)          # 165 features: 93 local + 72 aggregated
    m = dict(zip(lab.txId, lab["class"].astype(str)))
    cls = np.array([{"1": 1, "2": 0}.get(m[t], -1) for t in txid])   # 1 = illicit, 0 = licit, -1 = unknown
    pos = {t: i for i, t in enumerate(txid)}
    E = np.array([[pos[a] for a in edg.txId1], [pos[b] for b in edg.txId2]])
    np.savez_compressed(fcache, txid=txid, X=X, step=step, cls=cls, E=E)
n, d = X.shape
LOCAL = list(range(0, 93))
deg_dir_out = np.bincount(E[0], minlength=n)
deg_dir_in = np.bincount(E[1], minlength=n)
deg = deg_dir_out + deg_dir_in
same = float(np.mean(step[E[0]] == step[E[1]]))
pd.DataFrame({"degree": deg, "illicit": (cls == 1).astype(int)}).groupby(["degree", "illicit"]).size().rename("n").reset_index().to_csv(
    os.path.join(RES, "degree_counts.csv"), index=False)
lab_mask = cls >= 0
train_m = lab_mask & (step <= T_TRAIN)
fit_m = lab_mask & (step < T_VAL)
val_m = lab_mask & (step >= T_VAL) & (step <= T_TRAIN)
test_m = lab_mask & (step > T_TRAIN)
N["data"] = {
    "n_nodes": int(n), "n_edges": int(E.shape[1]), "n_features": int(d), "n_local": 93, "n_agg": int(d - 93),
    "n_steps": int(step.max()), "n_illicit": int((cls == 1).sum()), "n_licit": int((cls == 0).sum()),
    "n_unknown": int((cls == -1).sum()),
    "pct_labelled": round(100 * lab_mask.mean(), 1),
    "pct_illicit_lab": round(100 * (cls == 1).sum() / lab_mask.sum(), 2),
    "n_train": int(train_m.sum()), "n_train_illicit": int((train_m & (cls == 1)).sum()),
    "n_test": int(test_m.sum()), "n_test_illicit": int((test_m & (cls == 1)).sum()),
    "pct_test_illicit": round(100 * (test_m & (cls == 1)).sum() / test_m.sum(), 2),
    "mean_degree": round(float(deg.mean()), 2), "median_degree": int(np.median(deg)), "max_degree": int(deg.max()),
    "pct_deg1": round(100 * float(np.mean(deg == 1)), 1),
    "pct_same_step": round(100 * same, 1),
    "nodes_per_step_mean": int(round(n / step.max())),
    "train_steps": "1-34", "test_steps": "35-49",
}
print("data loaded", round(time.time() - t0, 1), "s")

# standardise with the training steps only (leakage-free)
mu = X[step <= T_TRAIN].mean(0)
sd = X[step <= T_TRAIN].std(0) + 1e-6
Xs = ((X - mu) / sd).astype(np.float32)
Xs = np.clip(Xs, -10, 10)
y = cls.copy()

# ---------------------------------------------------------------- 1. graph statistics: steps, degree, homophily, receptive field
steps = np.arange(1, step.max() + 1)
cnt = pd.DataFrame({"step": steps,
                    "illicit": [int(((step == s) & (cls == 1)).sum()) for s in steps],
                    "licit": [int(((step == s) & (cls == 0)).sum()) for s in steps],
                    "unknown": [int(((step == s) & (cls == -1)).sum()) for s in steps]})
cnt.to_csv(os.path.join(RES, "timesteps.csv"), index=False)
N["data"]["max_illicit_step"] = int(cnt.loc[cnt.illicit.idxmax(), "step"])
N["data"]["max_illicit_n"] = int(cnt.illicit.max())

# undirected neighbour lists
Eu = np.concatenate([E, E[::-1]], axis=1)
order = np.argsort(Eu[0], kind="stable")
src, dst = Eu[0][order], Eu[1][order]
ptr = np.searchsorted(src, np.arange(n + 1))
# label homophily: among the labelled neighbours of a labelled node, share that is illicit
nb_ill = np.zeros(n)
nb_lab = np.zeros(n)
lab_dst = cls[dst]
np.add.at(nb_ill, src, (lab_dst == 1).astype(float))
np.add.at(nb_lab, src, (lab_dst >= 0).astype(float))
has = nb_lab > 0
h_ill = float(np.mean(nb_ill[(cls == 1) & has] / nb_lab[(cls == 1) & has]))
h_lic = float(np.mean(nb_ill[(cls == 0) & has] / nb_lab[(cls == 0) & has]))
# edge homophily among labelled-labelled edges
ee = (cls[E[0]] >= 0) & (cls[E[1]] >= 0)
edge_h = float(np.mean(cls[E[0]][ee] == cls[E[1]][ee]))
# expected edge homophily if labels were random (class shares)
p1 = (cls == 1).sum() / lab_mask.sum()
edge_h_rand = float(p1 ** 2 + (1 - p1) ** 2)
N["homophily"] = {"illicit_nb_of_illicit_pct": round(100 * h_ill, 1), "illicit_nb_of_licit_pct": round(100 * h_lic, 1),
                  "base_pct": round(100 * p1, 1), "edge_pct": round(100 * edge_h, 1), "edge_random_pct": round(100 * edge_h_rand, 1),
                  "n_illicit_with_lab_nb": int(((cls == 1) & has).sum()), "n_licit_with_lab_nb": int(((cls == 0) & has).sum())}
pd.DataFrame({"group": ["illicit", "licit"], "share_illicit_neighbours": [h_ill, h_lic], "base_rate": [p1, p1]}).to_csv(
    os.path.join(RES, "homophily.csv"), index=False)

# receptive field: number of distinct nodes within k hops (sample of 2,000 labelled test nodes)
rng = np.random.default_rng(SEED)
smp = rng.choice(np.where(test_m)[0], 2000, replace=False)
hops = []
for v in smp:
    seen = {v}
    frontier = {v}
    row = []
    for k in range(1, 5):
        nxt = set()
        for u in frontier:
            nxt.update(dst[ptr[u]:ptr[u + 1]].tolist())
        nxt -= seen
        seen |= nxt
        frontier = nxt
        row.append(len(seen) - 1)
    hops.append(row)
hops = np.array(hops)
rf = pd.DataFrame({"k": [1, 2, 3, 4], "median": np.median(hops, 0), "p90": np.percentile(hops, 90, axis=0),
                   "mean": hops.mean(0)})
rf.to_csv(os.path.join(RES, "receptive_field.csv"), index=False)
N["receptive"] = {f"k{k}_median": int(np.median(hops[:, k - 1])) for k in range(1, 5)}
N["receptive"].update({f"k{k}_p90": int(np.percentile(hops[:, k - 1], 90)) for k in range(1, 5)})
print("graph statistics", round(time.time() - t0, 1), "s")

# ---------------------------------------------------------------- 2. models
dev = "cpu"   # PyG scatter on 200k nodes is fast on the CPU; MPS lacks some sparse kernels
ei = to_undirected(torch.tensor(E, dtype=torch.long))
xt = torch.tensor(Xs)
yt = torch.tensor(np.where(y < 0, 0, y), dtype=torch.long)
m_fit, m_val, m_test = (torch.tensor(a) for a in (fit_m, val_m, test_m))
W = torch.tensor([0.3, 0.7])     # class weights of Weber et al. (2019): licit 0.3, illicit 0.7


class GNN(torch.nn.Module):
    def __init__(self, kind, d_in, h=100, layers=2, heads=4, drop=0.2):
        super().__init__()
        self.kind, self.drop = kind, drop
        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            a = d_in if i == 0 else h
            if kind == "gcn":
                self.convs.append(GCNConv(a, h))
            elif kind == "sage":
                self.convs.append(SAGEConv(a, h))
            elif kind == "gat":
                self.convs.append(GATConv(a, h // heads, heads=heads))
            elif kind == "mlp":
                self.convs.append(torch.nn.Linear(a, h))
        self.out = torch.nn.Linear(h, 2)

    def embed(self, x, ei):
        for c in self.convs:
            x = c(x) if self.kind == "mlp" else c(x, ei)
            x = F.relu(x)
            x = F.dropout(x, self.drop, self.training)
        return x

    def forward(self, x, ei):
        return self.out(self.embed(x, ei))


def train_gnn(kind, seed, layers=2, epochs=300, patience=40, lr=0.003, feats=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = xt if feats is None else xt[:, feats]
    model = GNN(kind, x.shape[1], layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best, best_state, best_ep, wait, hist = -1, None, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[m_fit], yt[m_fit], weight=W)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            o = model(x, ei)
            p = torch.softmax(o, 1)[:, 1].numpy()
            vl = float(F.cross_entropy(o[m_val], yt[m_val], weight=W))
        ap = average_precision_score(y[val_m], p[val_m])
        hist.append((ep, float(loss), vl, ap))
        if ap > best:
            best, best_ep, wait = ap, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x, ei), 1)[:, 1].numpy()
        emb = model.embed(x, ei).numpy()
    return p, emb, best_ep, hist, model


def metrics(p, thr=0.5, mask=test_m):
    yy, pp = y[mask], p[mask]
    yh = (pp >= thr).astype(int)
    return {"f1": f1_score(yy, yh), "prec": precision_score(yy, yh, zero_division=0), "rec": recall_score(yy, yh),
            "ap": average_precision_score(yy, pp)}


rcache = os.path.join(CACHE, "models.npz")
MODELS = ["lr_local", "lr_all", "rf", "mlp", "gcn", "sage", "gat", "rf_emb"]
if os.path.exists(rcache):
    zc = np.load(rcache, allow_pickle=True)
    P = zc["P"].item()
    EMB = zc["EMB"]
    HIST = zc["HIST"].item()
    BEST_EP = zc["BEST_EP"].item()
    TIME = zc["TIME"].item()
    IMP = zc["IMP"].item()
else:
    P, HIST, BEST_EP, TIME, IMP = {}, {}, {}, {}, {}
    trn = train_m
    for name, cols in [("lr_local", LOCAL), ("lr_all", list(range(d)))]:
        s0 = time.time()
        lr_ = LogisticRegression(max_iter=3000, C=1.0, class_weight={0: 0.3, 1: 0.7})
        lr_.fit(Xs[trn][:, cols], y[trn])
        P[name] = [lr_.predict_proba(Xs[:, cols])[:, 1]]
        TIME[name] = time.time() - s0
    P["rf"], IMP["rf"] = [], []
    s0 = time.time()
    for s in SEEDS:
        rfm = RandomForestClassifier(n_estimators=100, max_features=50, n_jobs=-1, random_state=s)
        rfm.fit(X[trn], y[trn])
        P["rf"].append(rfm.predict_proba(X)[:, 1])
        IMP["rf"].append(rfm.feature_importances_)
    TIME["rf"] = (time.time() - s0) / len(SEEDS)
    EMB = None
    for kind in ["mlp", "gcn", "sage", "gat"]:
        P[kind], HIST[kind], BEST_EP[kind] = [], [], []
        s0 = time.time()
        for s in SEEDS:
            p, emb, be, hist, _ = train_gnn(kind, s)
            P[kind].append(p)
            HIST[kind].append(hist)
            BEST_EP[kind].append(be)
            if kind == "gcn" and s == SEEDS[0]:
                EMB = emb
            print(kind, s, "best epoch", be, metrics(p))
        TIME[kind] = (time.time() - s0) / len(SEEDS)
    # random forest on the features + the GCN embeddings (Weber et al.: RF + embeddings); the GCN of each seed
    P["rf_emb"], IMP["rf_emb"] = [], []
    s0 = time.time()
    for i, s in enumerate(SEEDS):
        _, emb, _, _, _ = train_gnn("gcn", s)
        Xe = np.hstack([X, emb])
        rfm = RandomForestClassifier(n_estimators=100, max_features=50, n_jobs=-1, random_state=s)
        rfm.fit(Xe[trn], y[trn])
        P["rf_emb"].append(rfm.predict_proba(Xe)[:, 1])
        IMP["rf_emb"].append(rfm.feature_importances_)
    TIME["rf_emb"] = (time.time() - s0) / len(SEEDS)
    np.savez_compressed(rcache, P=np.array(P, dtype=object), EMB=EMB, HIST=np.array(HIST, dtype=object),
                        BEST_EP=np.array(BEST_EP, dtype=object), TIME=np.array(TIME, dtype=object), IMP=np.array(IMP, dtype=object))
print("models", round(time.time() - t0, 1), "s")

LABEL = {"en": {"lr_local": "Logistic, local features", "lr_all": "Logistic, all features", "rf": "Random forest",
                "mlp": "MLP (no graph)", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "rf_emb": "Random forest + GCN embeddings"},
         "ro": {"lr_local": "Logistic, variabile locale", "lr_all": "Logistic, toate variabilele", "rf": "Pădure aleatoare",
                "mlp": "MLP (fără graf)", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "rf_emb": "Pădure aleatoare + reprezentări GCN"}}
N["models"] = {}
rows = []
for mname in MODELS:
    ms = [metrics(p) for p in P[mname]]
    agg = {k: (float(np.mean([m_[k] for m_ in ms])), float(np.std([m_[k] for m_ in ms], ddof=1)) if len(ms) > 1 else 0.0)
           for k in ms[0]}
    N["models"][mname] = {"f1": round(agg["f1"][0], 3), "f1_sd": round(agg["f1"][1], 3), "prec": round(agg["prec"][0], 3),
                          "rec": round(agg["rec"][0], 3), "ap": round(agg["ap"][0], 3), "ap_sd": round(agg["ap"][1], 3),
                          "n_runs": len(ms), "sec": round(float(TIME[mname]), 1)}
    if mname in BEST_EP:
        N["models"][mname]["best_epoch_median"] = int(np.median(BEST_EP[mname]))
    for i, m_ in enumerate(ms):
        rows.append({"model": mname, "run": i + 1, **{k: round(float(v), 6) for k, v in m_.items()}})
pd.DataFrame(rows).to_csv(os.path.join(RES, "model_metrics.csv"), index=False)
# mean predicted score per model (first run) on the test nodes, for the Quantlets
pt = pd.DataFrame({"txId": txid[test_m], "step": step[test_m], "y": y[test_m]})
for mname in MODELS:
    pt[mname] = np.round(np.mean(P[mname], 0)[test_m], 6)
pt.to_csv(os.path.join(RES, "test_scores.csv"), index=False)
best_name = max(MODELS, key=lambda k: N["models"][k]["f1"])
N["best_f1_model"] = best_name
N["gcn_vs_mlp_ap_pts"] = round(100 * (N["models"]["gcn"]["ap"] - N["models"]["mlp"]["ap"]), 1)
N["rfemb_vs_rf_f1_pts"] = round(100 * (N["models"]["rf_emb"]["f1"] - N["models"]["rf"]["f1"]), 1)
N["gcn_params"] = int(sum(p_.numel() for p_ in GNN("gcn", d).parameters()))
N["gat_params"] = int(sum(p_.numel() for p_ in GNN("gat", d).parameters()))

# bootstrap (test nodes resampled, 1,000 draws): PR-AUC of the random forest minus PR-AUC of the GCN (seed means)
rng = np.random.default_rng(SEED)
yy = y[test_m]
pr, pg, pre = (np.mean(P[k], 0)[test_m] for k in ("rf", "gcn", "rf_emb"))
dif, dif2 = [], []
for _ in range(1000):
    i = rng.integers(0, len(yy), len(yy))
    if yy[i].sum() == 0:
        continue
    dif.append(average_precision_score(yy[i], pr[i]) - average_precision_score(yy[i], pg[i]))
    dif2.append(average_precision_score(yy[i], pre[i]) - average_precision_score(yy[i], pr[i]))
N["boot"] = {"rf_minus_gcn_ap": round(float(np.mean(dif)), 3), "lo": round(float(np.percentile(dif, 2.5)), 3),
             "hi": round(float(np.percentile(dif, 97.5)), 3),
             "rfemb_minus_rf_ap": round(float(np.mean(dif2)), 3), "lo2": round(float(np.percentile(dif2, 2.5)), 3),
             "hi2": round(float(np.percentile(dif2, 97.5)), 3), "n_boot": 1000}

# F1 by test time step (the dark-market shutdown at step 43, Weber et al., 2019)
fts = []
for s in range(T_TRAIN + 1, step.max() + 1):
    mm = test_m & (step == s)
    r = {"step": s, "n_illicit": int((y[mm] == 1).sum())}
    for mname in ["lr_all", "rf", "gcn", "rf_emb"]:
        p = np.mean(P[mname], 0)
        r[mname] = f1_score(y[mm], (p[mm] >= 0.5).astype(int), zero_division=0)
    fts.append(r)
fts = pd.DataFrame(fts)
fts.to_csv(os.path.join(RES, "f1_by_step.csv"), index=False)
pre43 = fts[fts.step < 43]
post43 = fts[fts.step >= 43]
N["shutdown"] = {"rf_f1_pre": round(float(pre43.rf.mean()), 3), "rf_f1_post": round(float(post43.rf.mean()), 3),
                 "gcn_f1_pre": round(float(pre43.gcn.mean()), 3), "gcn_f1_post": round(float(post43.gcn.mean()), 3),
                 "ill_pre": int(pre43.n_illicit.sum()), "ill_post": int(post43.n_illicit.sum())}

# alert budget: share of the illicit test transactions found among the k highest scores
kk = np.unique(np.linspace(50, 3000, 60).astype(int))
ab = {"k": kk}
for mname in ["lr_all", "rf", "gcn", "rf_emb"]:
    s_ = np.mean(P[mname], 0)[test_m]
    o = np.argsort(-s_)
    cum = np.cumsum(yy[o])
    ab[mname] = cum[kk - 1] / yy.sum()
ab = pd.DataFrame(ab)
ab.to_csv(os.path.join(RES, "alert_budget.csv"), index=False)
K0 = 1000
N["alerts"] = {"k": K0, **{m_: round(100 * float(ab.loc[ab.k == min(ab.k, key=lambda v: abs(v - K0)), m_].iloc[0]), 1)
                           for m_ in ["lr_all", "rf", "gcn", "rf_emb"]},
               "n_test_illicit": int(yy.sum())}
for m_ in ["rf", "gcn", "rf_emb"]:
    s_ = np.mean(P[m_], 0)[test_m]
    o = np.argsort(-s_)[:K0]
    N["alerts"][m_ + "_prec"] = round(100 * float(yy[o].mean()), 1)

# ---------------------------------------------------------------- 3. depth and over-smoothing (GCN, 1-6 layers, 2 seeds)
dcache = os.path.join(CACHE, "depth.csv")
if os.path.exists(dcache):
    depth = pd.read_csv(dcache)
else:
    rows = []
    for L_ in range(1, 7):
        for s in [1, 2]:
            p, emb, be, _, _ = train_gnn("gcn", s, layers=L_)
            e_ = emb[test_m]
            e_ = e_ / (np.linalg.norm(e_, axis=1, keepdims=True) + 1e-9)
            ii = np.random.default_rng(s).choice(len(e_), (5000, 2))
            cos = float(np.mean(np.sum(e_[ii[:, 0]] * e_[ii[:, 1]], 1)))
            m_ = metrics(p)
            rows.append({"layers": L_, "seed": s, "ap": m_["ap"], "f1": m_["f1"], "cos": cos})
            print("depth", L_, s, m_, cos)
    depth = pd.DataFrame(rows)
    depth.to_csv(dcache, index=False)
dg = depth.groupby("layers").mean(numeric_only=True).reset_index()
dg.to_csv(os.path.join(RES, "depth.csv"), index=False)
N["depth"] = {"ap_1": round(float(dg.ap[0]), 3), "ap_2": round(float(dg.ap[1]), 3), "ap_6": round(float(dg.ap[5]), 3),
              "cos_1": round(float(dg.cos[0]), 2), "cos_6": round(float(dg.cos[5]), 2),
              "best_layers": int(dg.loc[dg.ap.idxmax(), "layers"])}

# ---------------------------------------------------------------- 4. explanation stability across refits (Spearman of importances)
def stab(imps, k=None):
    k = k or 165
    v = [im[:k] / im[:k].sum() for im in imps]
    r = [spearmanr(v[i], v[j]).correlation for i in range(len(v)) for j in range(i + 1, len(v))]
    return float(np.mean(r))


N["stability"] = {"rf": round(stab(IMP["rf"]), 3), "rf_emb": round(stab(IMP["rf_emb"]), 3),
                  "emb_share_pct": round(100 * float(np.mean([im[165:].sum() for im in IMP["rf_emb"]])), 1),
                  "n_refits": len(SEEDS)}
top_rf = np.argsort(-np.mean(IMP["rf"], 0))[:10]
imp_df = pd.DataFrame({"feature": [f"f{j + 1}" for j in top_rf],
                       "rf": np.mean(IMP["rf"], 0)[top_rf], "rf_emb": np.mean(IMP["rf_emb"], 0)[top_rf]})
imp_df.to_csv(os.path.join(RES, "importance.csv"), index=False)
N["stability"]["top_feature"] = imp_df.feature.iloc[0]
N["stability"]["top_local_count"] = int(sum(j < 93 for j in top_rf))
N["stability"]["top10_share_rf_pct"] = round(100 * float(imp_df.rf.sum()), 1)
N["stability"]["top10_share_rfemb_pct"] = round(100 * float(imp_df.rf_emb.sum()), 1)

# ---------------------------------------------------------------- 5. charts
PER = {"en": "Elliptic Bitcoin transaction graph, 49 time steps (about two weeks each)",
       "ro": "graful tranzacțiilor Bitcoin Elliptic, 49 de pași de timp (circa două săptămîni fiecare)"}
TEST_EN = "test time steps 35-49"
TEST_RO = "pașii de timp de test 35-49"


def meta(name, en=None, ro=None):
    npl.chart_meta(N, name, "elliptic", en or PER["en"], ro or PER["ro"])


for lang in ("en", "ro"):
    ro = lang == "ro"
    L = LABEL[lang]
    # 5.1 nodes per time step
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.bar(cnt.step, cnt.licit, color=npl.PAL["negative"], width=0.8, label="licite" if ro else "licit")
    ax.bar(cnt.step, cnt.illicit, bottom=cnt.licit, color=npl.PAL["positive"], width=0.8, label="ilicite" if ro else "illicit")
    ax.axvline(T_TRAIN + 0.5, color=npl.MUTED, lw=1, ls="--", label="sfîrșitul antrenării" if ro else "end of training")
    ax.set_xlabel("pasul de timp" if ro else "time step")
    ax.set_ylabel("tranzacții etichetate" if ro else "labelled transactions")
    ax.set_xlim(0.3, 49.7)
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "elliptic_timesteps")

    # 5.2 degree distribution (complementary CDF, log-log)
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ds = np.sort(deg)
    ccdf = 1 - np.arange(len(ds)) / len(ds)
    ax.loglog(ds, ccdf, color=npl.PAL["data"], lw=1.4, label="toate tranzacțiile" if ro else "all transactions")
    di = np.sort(deg[cls == 1])
    ax.loglog(di, 1 - np.arange(len(di)) / len(di), color=npl.PAL["positive"], lw=1.4,
              label="tranzacții ilicite" if ro else "illicit transactions")
    ax.set_xlabel("gradul nodului (intrări + ieșiri)" if ro else "node degree (in + out)")
    ax.set_ylabel("P(grad ≥ x)" if ro else "P(degree ≥ x)")
    ax._keep_y = True
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "degree_dist")

    # 5.3 homophily
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    labs = ["vecinii tranzacțiilor ilicite", "vecinii tranzacțiilor licite"] if ro else ["neighbours of illicit nodes", "neighbours of licit nodes"]
    vals = [100 * h_ill, 100 * h_lic]
    ax.barh(labs, vals, color=[npl.PAL["positive"], npl.PAL["negative"]], height=0.55)
    ax.axvline(100 * p1, color=npl.MUTED, ls="--", lw=1.2, label=("ponderea ilicitelor printre tranzacțiile etichetate" if ro
                                                                  else "share of illicit among labelled transactions"))
    for i, v in enumerate(vals):
        ax.text(v + 1, i, (f"{v:.1f}".replace(".", ",") if ro else f"{v:.1f}") + " %", va="center", fontsize=10)
    ax.set_xlabel("vecini etichetați care sînt ilicite (%)" if ro else "labelled neighbours that are illicit (%)")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.invert_yaxis()
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "homophily", also_x=[ax])

    # 5.4 receptive field
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    ax.semilogy(rf.k, rf["median"], "o-", color=npl.PAL["data"], label="mediana" if ro else "median")
    ax.semilogy(rf.k, rf["p90"], "s--", color=npl.PAL["alt"], label="percentila 90" if ro else "90th percentile")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("numărul de straturi (salturi în graf)" if ro else "number of layers (hops in the graph)")
    ax.set_ylabel("noduri care influențează\nun nod" if ro else "nodes that reach\none node")
    ax._keep_y = True
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "receptive_field")

    # 5.5 model comparison (illicit F1 and PR-AUC, mean +- sd over seeds)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    order_ = ["lr_local", "lr_all", "mlp", "gcn", "sage", "gat", "rf", "rf_emb"]
    col = {"lr_local": npl.PAL["benchmark"], "lr_all": npl.PAL["benchmark"], "rf": npl.PAL["alt"], "rf_emb": npl.PAL["c7"],
           "mlp": npl.MUTED, "gcn": npl.PAL["model"], "sage": npl.PAL["model"], "gat": npl.PAL["model"]}
    names = [L[k].replace("Random forest + GCN embeddings", "RF + GCN emb.").replace("Pădure aleatoare + reprezentări GCN", "Pădure + repr. GCN")
             for k in order_]
    for ax, key, xl in [(axs[0], "f1", "F1, clasa ilicită" if ro else "F1, illicit class"),
                        (axs[1], "ap", "PR-AUC")]:
        v = [N["models"][k][key] for k in order_]
        e = [N["models"][k][key + "_sd"] for k in order_]
        ax.barh(range(len(order_)), v, xerr=e, color=[col[k] for k in order_], height=0.6, error_kw={"lw": 1, "capsize": 2})
        ax.set_xlabel(xl)
        ax.set_xlim(0, 1)
    axs[0].set_yticks(range(len(order_)))
    axs[0].set_yticklabels(names, fontsize=9.5)
    axs[0].invert_yaxis()
    from matplotlib.patches import Patch
    h = [Patch(color=npl.PAL["benchmark"], label="logistic" if not ro else "logistic"),
         Patch(color=npl.MUTED, label="MLP"), Patch(color=npl.PAL["model"], label="GNN"),
         Patch(color=npl.PAL["alt"], label="pădure aleatoare" if ro else "random forest"),
         Patch(color=npl.PAL["c7"], label="pădure + GCN" if ro else "forest + GCN")]
    npl.legend_bottom(fig, h, [x.get_label() for x in h], ncol=5)
    npl.out(fig, HERE, lang, "model_comparison", also_x=list(axs))

    # 5.6 F1 by test step
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for mname, c, ls in [("rf", npl.PAL["alt"], "-"), ("gcn", npl.PAL["model"], "-"), ("rf_emb", npl.PAL["c7"], "--"),
                         ("lr_all", npl.PAL["benchmark"], ":")]:
        ax.plot(fts.step, fts[mname], ls, color=c, marker="o", ms=3, label=L[mname])
    ax.axvline(42.5, color=npl.MUTED, lw=1, ls="--")
    ax.text(42.7, 0.95, "închiderea unei piețe\ndin dark web" if ro else "dark-market\nshutdown", fontsize=9, color=npl.MUTED, va="top")
    ax.set_ylim(0, 1)
    ax.set_xlabel("pasul de timp (test)" if ro else "time step (test)")
    ax.set_ylabel("F1, clasa ilicită" if ro else "F1, illicit class")
    ax.set_xticks(range(35, 50, 2))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "f1_by_timestep")

    # 5.7 alert budget
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for mname, c, ls in [("rf", npl.PAL["alt"], "-"), ("gcn", npl.PAL["model"], "-"), ("rf_emb", npl.PAL["c7"], "--"),
                         ("lr_all", npl.PAL["benchmark"], ":")]:
        ax.plot(ab.k, 100 * ab[mname], ls, color=c, label=L[mname])
    ax.axvline(K0, color=npl.MUTED, lw=0.8, ls=":")
    ax.set_xlabel("alerte verificate (scorurile cele mai mari)" if ro else "alerts reviewed (highest scores first)")
    ax.set_ylabel("ilicite găsite (%)" if ro else "illicit found (%)")
    ax.set_ylim(0, 100)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "alert_budget", also_x=[ax])

    # 5.8 depth / over-smoothing
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    axs[0].plot(dg.layers, dg.ap, "o-", color=npl.PAL["model"], label="PR-AUC (test)")
    axs[0].set_xlabel("straturi GCN" if ro else "GCN layers")
    axs[0].set_ylabel("PR-AUC")
    axs[1].plot(dg.layers, dg.cos, "s-", color=npl.PAL["data"], label="similaritatea cosinus medie" if ro else "mean cosine similarity")
    axs[1].set_xlabel("straturi GCN" if ro else "GCN layers")
    axs[1].set_ylabel("similaritate cosinus" if ro else "cosine similarity")
    for a_ in axs:
        a_.set_xticks(range(1, 7))
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "oversmoothing")

    # 5.9 embeddings of the GCN (PCA of the hidden layer, test nodes)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    et = EMB[test_m]
    ec = et - et[:].mean(0)
    U, S_, Vt = np.linalg.svd(ec[np.random.default_rng(1).choice(len(ec), 5000, replace=False)], full_matrices=False)
    Z = ec @ Vt[:2].T
    rs = np.random.default_rng(2)
    il = np.where(yy == 1)[0]
    li = rs.choice(np.where(yy == 0)[0], 3000, replace=False)
    ax.scatter(Z[li, 0], Z[li, 1], s=4, color=npl.PAL["negative"], alpha=0.35, lw=0, label="licite" if ro else "licit")
    ax.scatter(Z[il, 0], Z[il, 1], s=5, color=npl.PAL["positive"], alpha=0.6, lw=0, label="ilicite" if ro else "illicit")
    ax.set_xlabel("componenta principală 1" if ro else "principal component 1")
    ax.set_ylabel("componenta 2" if ro else "component 2")
    npl.legend_bottom(fig)
    npl.out(fig, HERE, lang, "gcn_embeddings", also_x=[ax])
    if lang == "en":
        var2 = float((S_[:2] ** 2).sum() / (S_ ** 2).sum())
        N["emb"] = {"var2_pct": round(100 * var2, 1), "dim": int(EMB.shape[1]), "n_licit_shown": 3000, "n_illicit_shown": int(len(il))}
        pd.DataFrame({"pc1": np.round(Z[:, 0], 5), "pc2": np.round(Z[:, 1], 5), "y": yy}).to_csv(os.path.join(RES, "embeddings_pca.csv"), index=False)

    # 5.10 PR curves
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    for mname, c, ls in [("rf", npl.PAL["alt"], "-"), ("gcn", npl.PAL["model"], "-"), ("rf_emb", npl.PAL["c7"], "--"),
                         ("lr_all", npl.PAL["benchmark"], ":")]:
        pr_, rc_, _ = precision_recall_curve(yy, np.mean(P[mname], 0)[test_m])
        ax.plot(rc_, pr_, ls, color=c, label=L[mname])
    ax.axhline(yy.mean(), color=npl.MUTED, lw=0.8, ls=":")
    ax.set_xlabel("recall (rata de detectare)" if ro else "recall")
    ax.set_ylabel("precizie" if ro else "precision")
    ax.set_ylim(0, 1.02)
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "pr_curves", also_x=[ax])

    # 5.11 importance and stability
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    xx = np.arange(len(imp_df))
    ax.bar(xx - 0.2, 100 * imp_df.rf, 0.4, color=npl.PAL["alt"], label=L["rf"])
    ax.bar(xx + 0.2, 100 * imp_df.rf_emb, 0.4, color=npl.PAL["c7"], label=L["rf_emb"])
    ax.set_xticks(xx)
    ax.set_xticklabels(imp_df.feature, fontsize=9)
    ax.set_ylabel("importanța (%)" if ro else "importance (%)")
    ax.set_xlabel("cele 10 variabile cele mai importante în pădurea aleatoare" if ro else "the 10 most important features of the random forest")
    npl.legend_bottom(fig, ncol=2)
    npl.out(fig, HERE, lang, "importance_shift")

for nm in ["elliptic_timesteps", "degree_dist", "homophily", "receptive_field"]:
    meta(nm)
for nm in ["model_comparison", "f1_by_timestep", "alert_budget", "oversmoothing", "gcn_embeddings", "pr_curves", "importance_shift"]:
    meta(nm, "own estimates; training steps 1-34, " + TEST_EN, "estimări proprii; antrenare pe pașii 1-34, " + TEST_RO)
N["runtime_min"] = round((time.time() - t0) / 60, 1)
N["hardware"] = "Apple silicon laptop, CPU"
with open(os.path.join(HERE, "L9_numbers.json"), "w") as fh:
    json.dump(N, fh, indent=1, ensure_ascii=False)
print(json.dumps({k: N[k] for k in ["models", "boot", "homophily", "depth", "stability", "alerts", "shutdown"]}, indent=1))
print("wrote L9_numbers.json and", len(N["charts"]), "charts x 2 languages in", N["runtime_min"], "min")
