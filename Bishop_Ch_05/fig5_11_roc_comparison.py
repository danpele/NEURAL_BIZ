# Figure: ROC curve comparison — Logistic Regression vs Naive Bayes
# Reference: Bishop & Bishop (2024), Fig. 5.11
# Chapter: 05
# Quantlet: https://github.com/danpele/NEURAL_BIZ/tree/master/Bishop_Ch_05

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
import os

# ── Course-wide colors ──
BLUE    = '#1f4e79'   # primary / training
CRIMSON = '#c00000'   # secondary / test
GREEN   = '#2e7d32'   # reference
GRAY    = '#333333'   # data points
GOLD    = '#d4a017'   # alternative

# ── Plot styling ──
plt.rc('text', usetex=False)
plt.rcParams['figure.facecolor'] = '#EBEBEB'
plt.rcParams['axes.facecolor']   = '#EBEBEB'
plt.rcParams['font.size'] = 11

np.random.seed(42)

# ── Generate synthetic two-class data ──
N = 200
# Class 0: centered at (-1, -1)
X0 = np.random.randn(N, 2) * 1.2 + np.array([-1, -1])
# Class 1: centered at (1, 1)
X1 = np.random.randn(N, 2) * 1.2 + np.array([1, 1])

X = np.vstack([X0, X1])
y = np.hstack([np.zeros(N), np.ones(N)])

# ── Fit logistic regression (closed-form approximation via IRLS) ──
# Add bias column
X_aug = np.column_stack([np.ones(2 * N), X])

# Simple gradient descent for logistic regression
w = np.zeros(3)
lr = 0.01
for _ in range(1000):
    z = X_aug @ w
    p = expit(z)
    grad = X_aug.T @ (p - y) / (2 * N)
    w -= lr * grad

# Logistic regression scores
scores_lr = X_aug @ w

# ── Naive Bayes (using only first feature, weaker model) ──
# Simple 1D model using only x1
mu0 = X0[:, 0].mean()
mu1 = X1[:, 0].mean()
sigma = X[:, 0].std()
# Log-likelihood ratio as score
scores_nb = (X[:, 0] - mu0)**2 / (2 * sigma**2) - (X[:, 0] - mu1)**2 / (2 * sigma**2)

# ── Compute ROC curves ──
def compute_roc(y_true, scores, n_thresholds=500):
    thresholds = np.linspace(scores.min() - 1, scores.max() + 1, n_thresholds)
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return np.array(fpr_list), np.array(tpr_list)

fpr_lr, tpr_lr = compute_roc(y, scores_lr)
fpr_nb, tpr_nb = compute_roc(y, scores_nb)

# ── AUC (trapezoidal) ──
auc_lr = -np.trapz(tpr_lr, fpr_lr)
auc_nb = -np.trapz(tpr_nb, fpr_nb)

# ── Plot ──
fig, ax = plt.subplots(figsize=(5, 3.5))

ax.plot(fpr_lr, tpr_lr, color=BLUE, lw=2.2,
        label=f'Logistic Regression (AUC={auc_lr:.3f})')
ax.plot(fpr_nb, tpr_nb, color=CRIMSON, lw=2.2, linestyle='--',
        label=f'Naive Bayes 1D (AUC={auc_nb:.3f})')
ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle=':', alpha=0.6,
        label='Random classifier')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='lower right', fontsize=8, framealpha=0.8)
ax.set_aspect('equal')

fig.tight_layout()

# ── Save ──
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'fig5_11_roc_comparison.pdf'),
            bbox_inches='tight', facecolor='#EBEBEB')
fig.savefig(os.path.join(out_dir, 'fig5_11_roc_comparison.png'),
            bbox_inches='tight', facecolor='#EBEBEB', dpi=300)

# Also save to the main charts directory used by slides
main_charts = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'charts')
if os.path.exists(main_charts):
    fig.savefig(os.path.join(main_charts, 'fig5_11_roc_comparison.pdf'),
                bbox_inches='tight', facecolor='#EBEBEB')
    fig.savefig(os.path.join(main_charts, 'fig5_11_roc_comparison.png'),
                bbox_inches='tight', facecolor='#EBEBEB', dpi=300)

print(f'Saved fig5_11_roc_comparison (LR AUC={auc_lr:.3f}, NB AUC={auc_nb:.3f})')
plt.close()
