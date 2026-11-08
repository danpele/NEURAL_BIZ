"""
make_quantlets_L9.py -- one Quantlet per chart of Lecture 9: Quantlets/L9/NN_L9_<chart>/ (Metainfo.txt, executed notebook, PNG).
The Elliptic data are not redistributed (downloaded on demand by data/download_all.py elliptic or by PyTorch Geometric);
the notebooks therefore read the small result files written by analysis_L9.py (lectures/L9/results/*.csv: counts per time
step, degree counts, metrics and test scores of the trained models) and redraw each chart. Every number printed with
show() is checked against L9_numbers.json.
Run:  python3 lectures/L9/make_quantlets_L9.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

RES = "REPO + 'lectures/L9/results/'"
DES = "Lecture 9 of Neural Networks and Deep Learning (NN), Elliptic Bitcoin transaction graph (Weber et al., 2019). "
DATA = "Elliptic data set (on demand: python3 data/download_all.py elliptic); lectures/L9/results/"
QS = []
LAB = ("LAB = {'lr_all': 'Logistic, all features', 'rf': 'Random forest', 'gcn': 'GCN', 'rf_emb': 'Random forest + GCN embeddings'}\n"
       "COL = {'lr_all': PAL['benchmark'], 'rf': PAL['alt'], 'gcn': PAL['model'], 'rf_emb': '#6250d6'}\n"
       "LS = {'lr_all': ':', 'rf': '-', 'gcn': '-', 'rf_emb': '--'}\n")
SAVE = "fig.tight_layout(rect=(0, 0.12, 1, 1))\nfig.savefig('{c}.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()"


def Q(chart, desc, kw, files, md, code):
    q = Quantlet(k=9, chart=chart, description=DES + desc, keywords=kw, datafile=DATA + files)
    q.md(md).code(code + "\n" + SAVE.format(c=chart))
    QS.append(q)


Q("elliptic_timesteps", "Labelled transactions per time step (licit and illicit) and the temporal split: training steps 1-34, test steps 35-49.",
  "Elliptic, Bitcoin, anti-money laundering, temporal split, class imbalance", "timesteps.csv",
  "Counts of labelled transactions in each of the 49 time steps.",
  "c = pd.read_csv(" + RES + " + 'timesteps.csv')\n"
  "show('data.max_illicit_n', 'largest number of illicit transactions in one step', int(c.illicit.max()))\n"
  "show('data.max_illicit_step', 'step with the most illicit transactions', int(c.step[c.illicit.idxmax()]))\n"
  "show('data.n_illicit', 'illicit transactions', int(c.illicit.sum()))\nshow('data.n_licit', 'licit transactions', int(c.licit.sum()))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.5))\nax.bar(c.step, c.licit, color=PAL['negative'], width=0.8, label='licit')\n"
  "ax.bar(c.step, c.illicit, bottom=c.licit, color=PAL['positive'], width=0.8, label='illicit')\n"
  "ax.axvline(34.5, color='#8a8a8a', lw=1, ls='--', label='end of training')\nax.set_xlabel('time step'); ax.set_ylabel('labelled transactions')\nlegend_bottom(fig)")

Q("degree_dist", "Complementary cumulative distribution of the node degree (in + out), all transactions and illicit transactions, log-log scale.",
  "graph, degree distribution, heavy tail, hubs, Elliptic", "degree_counts.csv",
  "$P(\\deg\\ge x)$ on log-log axes.",
  "d = pd.read_csv(" + RES + " + 'degree_counts.csv')\na = d.groupby('degree').n.sum()\n"
  "show('data.max_degree', 'largest degree', int(a.index.max()))\nshow('data.median_degree', 'median degree', int(np.repeat(a.index, a.values)[a.sum() // 2]))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nfor sub, col, lab in [(a, PAL['data'], 'all transactions'), (d[d.illicit == 1].set_index('degree').n, PAL['positive'], 'illicit transactions')]:\n"
  "    ccdf = sub[::-1].cumsum()[::-1] / sub.sum()\n    ax.loglog(sub.index, ccdf.values, color=col, label=lab)\n"
  "ax.set_xlabel('node degree (in + out)'); ax.set_ylabel('P(degree ≥ x)')\nlegend_bottom(fig)")

Q("homophily", "Label homophily: share of illicit transactions among the labelled neighbours of illicit and of licit transactions.",
  "homophily, graph neural networks, money laundering, neighbourhood", "homophily.csv",
  "Mean over nodes of (illicit labelled neighbours) / (labelled neighbours).",
  "h = pd.read_csv(" + RES + " + 'homophily.csv')\n"
  "show('homophily.illicit_nb_of_illicit_pct', 'illicit neighbours of illicit nodes (%)', round(100 * float(h.share_illicit_neighbours[0]), 1))\n"
  "show('homophily.illicit_nb_of_licit_pct', 'illicit neighbours of licit nodes (%)', round(100 * float(h.share_illicit_neighbours[1]), 1))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.2))\nax.barh(['neighbours of illicit nodes', 'neighbours of licit nodes'], 100 * h.share_illicit_neighbours, color=[PAL['positive'], PAL['negative']], height=0.55)\n"
  "ax.axvline(100 * h.base_rate[0], color='#8a8a8a', ls='--', label='share of illicit among labelled transactions')\nax.invert_yaxis()\n"
  "ax.set_xlabel('labelled neighbours that are illicit (%)')\nlegend_bottom(fig)")

Q("receptive_field", "Number of distinct transactions within k hops of a test transaction (2,000 sampled test nodes): median and 90th percentile.",
  "receptive field, message passing, GNN depth, k-hop neighbourhood", "receptive_field.csv",
  "Receptive field of an $L$-layer message-passing network = the $L$-hop neighbourhood.",
  "r = pd.read_csv(" + RES + " + 'receptive_field.csv')\n"
  "show('receptive.k2_median', 'median nodes within 2 hops', int(r['median'][1]))\nshow('receptive.k4_p90', '90th percentile within 4 hops', int(r['p90'][3]))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.semilogy(r.k, r['median'], 'o-', color=PAL['data'], label='median')\n"
  "ax.semilogy(r.k, r['p90'], 's--', color=PAL['alt'], label='90th percentile')\nax.set_xticks([1, 2, 3, 4])\n"
  "ax.set_xlabel('number of layers (hops in the graph)'); ax.set_ylabel('nodes that reach one node')\nlegend_bottom(fig)")

Q("model_comparison", "Illicit-class F1 and PR-AUC on test steps 35-49 for logistic regression, MLP, GCN, GraphSAGE, GAT, random forest and random forest + GCN embeddings (mean and sd over seeds).",
  "graph convolutional network, GraphSAGE, graph attention network, random forest, Elliptic, anti-money laundering", "model_metrics.csv",
  "Mean and standard deviation over the runs of each model.",
  "m = pd.read_csv(" + RES + " + 'model_metrics.csv')\ng = m.groupby('model').agg(['mean', 'std'])\n"
  "show('models.rf.f1', 'F1, random forest', round(float(g.loc['rf', ('f1', 'mean')]), 3))\nshow('models.gcn.f1', 'F1, GCN', round(float(g.loc['gcn', ('f1', 'mean')]), 3))\n"
  "show('models.gcn.ap', 'PR-AUC, GCN', round(float(g.loc['gcn', ('ap', 'mean')]), 3))\nshow('models.mlp.ap', 'PR-AUC, MLP', round(float(g.loc['mlp', ('ap', 'mean')]), 3))\n"
  "order = ['lr_local', 'lr_all', 'mlp', 'gcn', 'sage', 'gat', 'rf', 'rf_emb']\n"
  "names = ['Logistic, local', 'Logistic, all', 'MLP (no graph)', 'GCN', 'GraphSAGE', 'GAT', 'Random forest', 'RF + GCN emb.']\n"
  "fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)\nfor ax, k, xl in [(axs[0], 'f1', 'F1, illicit class'), (axs[1], 'ap', 'PR-AUC')]:\n"
  "    ax.barh(range(8), [g.loc[o, (k, 'mean')] for o in order], xerr=[np.nan_to_num(g.loc[o, (k, 'std')]) for o in order], color=PAL['model'], height=0.6)\n"
  "    ax.set_xlabel(xl); ax.set_xlim(0, 1)\naxs[0].set_yticks(range(8)); axs[0].set_yticklabels(names, fontsize=9.5); axs[0].invert_yaxis()")

Q("f1_by_timestep", "Illicit-class F1 per test time step for logistic regression, random forest, GCN and random forest + GCN; the dark-market shutdown at step 43 breaks all models.",
  "concept drift, temporal evaluation, dark market, Elliptic, F1", "f1_by_step.csv",
  "F1 at threshold 0.5 on the labelled transactions of each test step.",
  LAB + "f = pd.read_csv(" + RES + " + 'f1_by_step.csv')\n"
  "show('shutdown.rf_f1_pre', 'mean F1 of the random forest before step 43', round(float(f[f.step < 43].rf.mean()), 3))\n"
  "show('shutdown.rf_f1_post', 'mean F1 of the random forest from step 43', round(float(f[f.step >= 43].rf.mean()), 3))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.5))\nfor k in ['rf', 'gcn', 'rf_emb', 'lr_all']:\n    ax.plot(f.step, f[k], LS[k], color=COL[k], marker='o', ms=3, label=LAB[k])\n"
  "ax.axvline(42.5, color='#8a8a8a', lw=1, ls='--')\nax.set_ylim(0, 1); ax.set_xlabel('time step (test)'); ax.set_ylabel('F1, illicit class')\nlegend_bottom(fig, ncol=2)")

Q("alert_budget", "Share of all illicit test transactions found among the k highest scores (alerts reviewed), for four models.",
  "alert budget, recall at k, anti-money laundering, investigation capacity", "alert_budget.csv",
  "Recall among the top-$k$ scores of steps 35-49.",
  LAB + "a = pd.read_csv(" + RES + " + 'alert_budget.csv')\nk0 = a.k[(a.k - 1000).abs().idxmin()]\n"
  "show('alerts.rf', 'illicit found in the top 1,000, random forest (%)', round(100 * float(a.rf[a.k == k0].iloc[0]), 1))\n"
  "show('alerts.gcn', 'illicit found in the top 1,000, GCN (%)', round(100 * float(a.gcn[a.k == k0].iloc[0]), 1))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.5))\nfor k in ['rf', 'gcn', 'rf_emb', 'lr_all']:\n    ax.plot(a.k, 100 * a[k], LS[k], color=COL[k], label=LAB[k])\n"
  "ax.axvline(1000, color='#8a8a8a', lw=0.8, ls=':')\nax.set_ylim(0, 100); ax.set_xlabel('alerts reviewed (highest scores first)'); ax.set_ylabel('illicit found (%)')\nlegend_bottom(fig, ncol=2)")

Q("oversmoothing", "Test PR-AUC of a GCN and mean cosine similarity of its node representations as a function of the number of layers (1-6, mean of 2 seeds): over-smoothing.",
  "over-smoothing, GCN depth, cosine similarity, message passing", "depth.csv",
  "Mean cosine similarity of the last hidden representations of 5,000 random pairs of test nodes.",
  "d = pd.read_csv(" + RES + " + 'depth.csv')\n"
  "show('depth.ap_2', 'PR-AUC with 2 layers', round(float(d.ap[1]), 3))\nshow('depth.ap_6', 'PR-AUC with 6 layers', round(float(d.ap[5]), 3))\n"
  "show('depth.cos_6', 'cosine similarity with 6 layers', round(float(d.cos[5]), 2))\n"
  "fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))\naxs[0].plot(d.layers, d.ap, 'o-', color=PAL['model'], label='PR-AUC (test)')\n"
  "axs[1].plot(d.layers, d.cos, 's-', color=PAL['data'], label='mean cosine similarity')\nfor a in axs: a.set_xlabel('GCN layers'); a.set_xticks(range(1, 7))\n"
  "axs[0].set_ylabel('PR-AUC'); axs[1].set_ylabel('cosine similarity')\nlegend_bottom(fig, ncol=2)")

Q("gcn_embeddings", "Hidden representations of the GCN for the test transactions, projected on their first two principal components; illicit and licit.",
  "node embeddings, GCN, PCA, representation learning, Elliptic", "embeddings_pca.csv",
  "PCA of the 100-dimensional hidden layer (seed 1).",
  "e = pd.read_csv(" + RES + " + 'embeddings_pca.csv')\nshow('emb.n_illicit_shown', 'illicit test transactions', int((e.y == 1).sum()))\n"
  "li = e[e.y == 0].sample(3000, random_state=2)\nil = e[e.y == 1]\nfig, ax = plt.subplots(figsize=(6.4, 2.6))\n"
  "ax.scatter(li.pc1, li.pc2, s=4, color=PAL['negative'], alpha=0.35, lw=0, label='licit')\nax.scatter(il.pc1, il.pc2, s=5, color=PAL['positive'], alpha=0.6, lw=0, label='illicit')\n"
  "ax.set_xlabel('principal component 1'); ax.set_ylabel('component 2')\nlegend_bottom(fig)")

Q("pr_curves", "Precision-recall curves on the test steps for logistic regression, random forest, GCN and random forest + GCN (scores averaged over seeds).",
  "precision-recall curve, PR-AUC, imbalanced classification, anti-money laundering", "test_scores.csv",
  "Base rate = share of illicit among the labelled test transactions.",
  LAB + "from sklearn.metrics import precision_recall_curve, average_precision_score\ns = pd.read_csv(" + RES + " + 'test_scores.csv')\n"
  "show('data.pct_test_illicit', 'illicit among labelled test transactions (%)', round(100 * float(s.y.mean()), 2))\n"
  "show('models.lr_all.ap', 'PR-AUC, logistic regression', round(float(average_precision_score(s.y, s.lr_all)), 3))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.5))\nfor k in ['rf', 'gcn', 'rf_emb', 'lr_all']:\n    p, r, _ = precision_recall_curve(s.y, s[k])\n    ax.plot(r, p, LS[k], color=COL[k], label=LAB[k])\n"
  "ax.axhline(s.y.mean(), color='#8a8a8a', lw=0.8, ls=':')\nax.set_ylim(0, 1.02); ax.set_xlabel('recall'); ax.set_ylabel('precision')\nlegend_bottom(fig, ncol=2)")

Q("importance_shift", "Impurity importance of the 10 most important features of the random forest, without and with the GCN embeddings as extra inputs (mean of 3 refits).",
  "feature importance, explanation stability, graph embeddings, random forest, explainable AI", "importance.csv",
  "Share of the total importance carried by each feature.",
  "im = pd.read_csv(" + RES + " + 'importance.csv')\n"
  "show('stability.top10_share_rf_pct', 'importance of the top 10 features, forest (%)', round(100 * float(im.rf.sum()), 1))\n"
  "show('stability.top10_share_rfemb_pct', 'same features, forest + GCN (%)', round(100 * float(im.rf_emb.sum()), 1))\n"
  "x = np.arange(len(im))\nfig, ax = plt.subplots(figsize=(6.4, 2.5))\nax.bar(x - 0.2, 100 * im.rf, 0.4, color=PAL['alt'], label='Random forest')\n"
  "ax.bar(x + 0.2, 100 * im.rf_emb, 0.4, color='#6250d6', label='Random forest + GCN embeddings')\nax.set_xticks(x); ax.set_xticklabels(im.feature, fontsize=9)\n"
  "ax.set_ylabel('importance (%)'); ax.set_xlabel('the 10 most important features of the random forest')\nlegend_bottom(fig, ncol=2)")

if __name__ == "__main__":
    only = sys.argv[1:]
    bad_all, n_all = [], 0
    for q in QS:
        if only and q.chart not in only:
            continue
        q.write(execute=True)
        n, bad = check_numbers(q, os.path.join(HERE, "L9_numbers.json"))
        n_all += n
        bad_all += bad
        print(f"  {q.name}: {n} numbers checked", "OK" if not bad and n else (bad or "NO NUMBERS"))
    print(f"{n_all} numbers checked;", "OK" if not bad_all else bad_all)
    sys.exit(1 if bad_all else 0)
