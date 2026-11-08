# Figure: Basic ROC curve with AUC shading
# Reference: Bishop & Bishop (2024), Fig. 5.5b
# Chapter: 05
# Quantlet: https://github.com/danpele/NEURAL_BIZ/tree/master/Bishop_Ch_05

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
import os

# Course colors
BLUE    = '#1f4e79'
CRIMSON = '#c00000'
GREEN   = '#2e7d32'
GRAY    = '#333333'

plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor']   = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['legend.framealpha'] = 0.0
plt.rcParams['font.size'] = 11

np.random.seed(42)

# Generate synthetic scores for a good classifier
N = 500
y_true = np.concatenate([np.zeros(N), np.ones(N)])
scores_good = np.concatenate([
    np.random.normal(-0.8, 1.0, N),
    np.random.normal(1.2, 1.0, N)
])
scores_random = np.random.randn(2 * N)

# Compute ROC
def compute_roc(y, s, n_t=1000):
    thresholds = np.linspace(s.max() + 1, s.min() - 1, n_t)
    tpr, fpr = [], []
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    return np.array(fpr), np.array(tpr)

fpr_good, tpr_good = compute_roc(y_true, scores_good)
fpr_rand, tpr_rand = compute_roc(y_true, scores_random)
auc_good = np.trapezoid(tpr_good, fpr_good)
auc_rand = np.trapezoid(tpr_rand, fpr_rand)

fig, ax = plt.subplots(figsize=(5, 4))

# AUC shading
ax.fill_between(fpr_good, tpr_good, alpha=0.15, color=BLUE)

# ROC curves
ax.plot(fpr_good, tpr_good, color=BLUE, lw=2.5,
        label=f'Good classifier (AUC = {auc_good:.2f})')
ax.plot(fpr_rand, tpr_rand, color=CRIMSON, lw=1.5, linestyle='--',
        label=f'Random (AUC = {auc_rand:.2f})')
ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle=':', alpha=0.5)

# Annotations
ax.annotate('AUC', xy=(0.35, 0.55), fontsize=14, color=BLUE,
            fontweight='bold', alpha=0.6)

ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center', fontsize=9, ncol=1)

fig.tight_layout()

# Save
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'fig5_5b_roc_curve.pdf'), bbox_inches='tight', transparent=True)
fig.savefig(os.path.join(out_dir, 'fig5_5b_roc_curve.png'), bbox_inches='tight', transparent=True, dpi=300)

main_charts = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'charts')
if os.path.exists(main_charts):
    fig.savefig(os.path.join(main_charts, 'fig5_5b_roc_curve.pdf'), bbox_inches='tight', transparent=True)

print(f'Saved fig5_5b_roc_curve (AUC good={auc_good:.2f}, random={auc_rand:.2f})')
plt.close()
