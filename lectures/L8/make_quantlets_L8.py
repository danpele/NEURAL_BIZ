"""
make_quantlets_L8.py -- one Quantlet per chart of Lecture 8: Quantlets/L8/NN_L8_<chart>/ (Metainfo.txt, executed notebook, PNG).
Each notebook reads the public data (data/clean) and, for trained models, the small result files written by
analysis_L8.py (lectures/L8/results/*.csv: scores, samples and losses of the trained networks, seeds fixed); the training
code itself is in analysis_L8.py and in the lecture notebook. Every number printed with show() is checked against
L8_numbers.json.
Run:  python3 lectures/L8/make_quantlets_L8.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_quantlet import Quantlet, check_numbers  # noqa: E402

RES = "REPO + 'lectures/L8/results/'"
FRAUD = ("fr = pd.read_csv(" + RES + " + 'fraud_test_scores.csv')\n"
         "print(len(fr), 'test transactions in the sample,', int(fr.y.sum()), 'frauds')\n")
SPX = ("x = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])\n"
       "px = x[x.ticker == 'SP500'].set_index('date')['adjclose'].dropna()\n"
       "end = (px.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)\n"
       "r = np.log(px.loc[:end]).diff().dropna()\nrtr = r.loc[:'2019-12-31']\ns_tr = float(rtr.std())\nztr = rtr.values / s_tr\n")
YC = ("yc = pd.read_csv(DATA + 'ecb_yield_curve_daily.csv', parse_dates=['date']).set_index('date')\n"
      "end = (yc.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)\n"
      "MAT = ['y_3M', 'y_6M', 'y_1Y', 'y_2Y', 'y_3Y', 'y_5Y', 'y_7Y', 'y_10Y', 'y_15Y', 'y_20Y', 'y_30Y']\n"
      "yc = yc.loc[:end].dropna()\nY = yc[MAT].values\ny10, slope = Y[:, 7], Y[:, 7] - Y[:, 0]\n")
DES = "Lecture 8 of Neural Networks and Deep Learning (NN). "
QS = []


def Q(chart, desc, kw, data, md, code):
    q = Quantlet(k=8, chart=chart, description=DES + desc, keywords=kw, datafile=data)
    q.md(md).code(code)
    QS.append(q)


Q("ae_error_dist", "Reconstruction error of an autoencoder trained on legitimate card transactions only, for the legitimate and the "
  "fraudulent transactions of the test part (last 20 % of the time span); weights restore the full data.",
  "autoencoder, anomaly detection, credit card fraud, reconstruction error, unsupervised learning",
  "data/clean/creditcard_fraud_sample.csv (ULB, OpenML 1597); lectures/L8/results/fraud_test_scores.csv (scores of the trained models)",
  "Anomaly score $s(\\mathbf x)=\\frac1d\\|\\mathbf x-g(f(\\mathbf x))\\|^2$ of a 29-16-8-16-29 autoencoder (mean of 3 seeds).",
  FRAUD + "le = np.log10(fr.err_ae)\nleg, fra = fr.y == 0, fr.y == 1\n"
  "show('ae.median_ratio', 'median error fraud / legitimate', round(float(np.median(fr.err_ae[fra]) / np.median(fr.err_ae[leg])), 1))\n"
  "bins = np.linspace(np.percentile(le, 0.1), le.max(), 60)\nfig, ax = plt.subplots(figsize=(6.4, 2.4))\n"
  "ax.hist(le[leg], bins=bins, weights=fr.weight[leg], density=True, alpha=0.5, color=PAL['negative'], label='legitimate transactions')\n"
  "ax.hist(le[fra], bins=bins, density=True, alpha=0.6, color=PAL['positive'], label='frauds')\n"
  "ax.set_xlabel('log$_{10}$ reconstruction error'); ax.set_ylabel('density')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
  "fig.savefig('ae_error_dist.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("ae_pr", "Precision-recall curves of three fraud detectors on the test part: autoencoder (no labels), PCA reconstruction with 8 "
  "components (linear autoencoder, no labels), logistic regression trained with the labels; weighted to the full data.",
  "precision-recall curve, PR-AUC, autoencoder, PCA, logistic regression, credit card fraud",
  "lectures/L8/results/fraud_test_scores.csv (from data/clean/creditcard_fraud_sample.csv)",
  "PR-AUC (average precision) with sample weights $w_i$ (10 for the sampled legitimate transactions, 1 for frauds).",
  FRAUD + "from sklearn.metrics import average_precision_score, precision_recall_curve\n"
  "show('ae.ap_pca', 'PR-AUC of PCA (%)', round(100 * average_precision_score(fr.y, fr.err_pca, sample_weight=fr.weight), 1))\n"
  "show('ae.ap_logit', 'PR-AUC of the logistic regression (%)', round(100 * average_precision_score(fr.y, fr.p_logit, sample_weight=fr.weight), 1))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.5))\n"
  "for c, col, ls, lab in [('err_ae', PAL['model'], '-', 'autoencoder'), ('err_pca', PAL['benchmark'], '--', 'PCA, 8 components'),\n"
  "                        ('p_logit', PAL['alt'], ':', 'logistic regression (uses labels)')]:\n"
  "    p, r, _ = precision_recall_curve(fr.y, fr[c], sample_weight=fr.weight)\n    ax.plot(r, p, ls, color=col, label=lab)\n"
  "ax.set_xlabel('recall (frauds detected)'); ax.set_ylabel('precision'); ax.set_ylim(0, 1.02)\nlegend_bottom(fig)\n"
  "fig.tight_layout(rect=(0, 0.12, 1, 1))\nfig.savefig('ae_pr.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("ae_bottleneck", "Test PR-AUC of the fraud autoencoder as a function of the bottleneck size k (mean of 2 seeds), against PCA with 8 components.",
  "autoencoder, bottleneck, latent dimension, anomaly detection, PR-AUC",
  "lectures/L8/results/ae_bottleneck.csv, fraud_test_scores.csv",
  "Choice of the bottleneck size $k$ of the autoencoder.",
  "b = pd.read_csv(" + RES + " + 'ae_bottleneck.csv')\nfr = pd.read_csv(" + RES + " + 'fraud_test_scores.csv')\n"
  "from sklearn.metrics import average_precision_score\nap_pca = 100 * average_precision_score(fr.y, fr.err_pca, sample_weight=fr.weight)\n"
  "show('ae.best_k', 'best bottleneck size', int(b.k[b.ap.idxmax()]))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.2))\nax.plot(b.k, 100 * b.ap, 'o-', color=PAL['model'], label='autoencoder')\n"
  "ax.axhline(ap_pca, color=PAL['benchmark'], ls='--', label='PCA, 8 components')\nax.set_xticks(b.k)\n"
  "ax.set_xlabel('bottleneck size $k$'); ax.set_ylabel('PR-AUC, test (%)')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
  "fig.savefig('ae_bottleneck.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

CLN = "CL = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']\n"
Q("vae_latent", "Two-dimensional latent means of a variational autoencoder (VAE) for the 10,000 Fashion-MNIST test images, coloured by class "
  "(classes were not used in training); k-nearest-neighbour accuracy on the latent means.",
  "variational autoencoder, VAE, latent space, Fashion-MNIST, representation learning",
  "lectures/L8/results/vae_latent.csv (latent means of the trained VAE; images: Fashion-MNIST, Zalando Research)",
  "Latent means $\\boldsymbol\\mu_\\phi(\\mathbf x)$ of a VAE with $k=2$, 20 epochs, seed 1.",
  CLN + "z = pd.read_csv(" + RES + " + 'vae_latent.csv')\nfrom sklearn.neighbors import KNeighborsClassifier\n"
  "from sklearn.model_selection import cross_val_score\n"
  "show('vae.knn_acc_2d', '15-NN accuracy on the 2-D latent means (%)', round(100 * float(np.mean(cross_val_score(KNeighborsClassifier(15), z[['z1', 'z2']].values, z.y.values, cv=5))), 1))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 3.0))\nfor c in range(10):\n    m = z.y == c\n"
  "    ax.scatter(z.z1[m], z.z2[m], s=2, alpha=0.5, color=plt.cm.tab10(c), lw=0, label=CL[c])\n"
  "ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')\nlegend_bottom(fig, ncol=5)\nfig.tight_layout(rect=(0, 0.2, 1, 1))\n"
  "fig.savefig('vae_latent.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("vae_samples", "Fashion-MNIST test images, their reconstructions through a 16-dimensional VAE code, and new images decoded from random codes.",
  "variational autoencoder, VAE, image generation, reconstruction, Fashion-MNIST",
  "lectures/L8/results/vae_samples.csv (images and VAE outputs, 784 grey levels per row)",
  "Rows: real images, reconstructions $g(\\boldsymbol\\mu(\\mathbf x))$, samples $g(\\mathbf z)$, $\\mathbf z\\sim\\mathcal N(0,I)$.",
  "v = pd.read_csv(" + RES + " + 'vae_samples.csv')\nimg = v.drop(columns='row').values\n"
  "show('vae.n_test', 'test images of Fashion-MNIST', 10000)\n"
  "fig, axs = plt.subplots(3, 10, figsize=(6.4, 2.2))\nfor i in range(3):\n    for j in range(10):\n"
  "        axs[i, j].imshow(img[10 * i + j].reshape(28, 28), cmap='Greys'); axs[i, j].axis('off')\n"
  "fig.savefig('vae_samples.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("vae_elbo", "Reconstruction error and Kullback-Leibler term of the VAE (k = 16, beta = 1) on the test images, by epoch.",
  "ELBO, KL divergence, variational autoencoder, training curve",
  "lectures/L8/results/vae_elbo.csv", "ELBO = reconstruction term $-$ KL term, in nats per image.",
  "h = pd.read_csv(" + RES + " + 'vae_elbo.csv')\nshow('vae.beta.b1.rec', 'reconstruction, last epoch (nats)', round(float(h.rec.iloc[-1]), 1))\n"
  "show('vae.beta.b1.kl', 'KL, last epoch (nats)', round(float(h.kl.iloc[-1]), 1))\n"
  "fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.3))\naxs[0].plot(h.epoch, h.rec, 'o-', ms=3, color=PAL['data'], label='reconstruction error')\n"
  "axs[1].plot(h.epoch, h.kl, 's-', ms=3, color=PAL['model'], label='KL divergence')\n"
  "for a in axs: a.set_xlabel('epoch')\naxs[0].set_ylabel('nats per image')\nlegend_bottom(fig, ncol=2)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
  "fig.savefig('vae_elbo.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("gan_density", "Density (log scale) of standardised S&P 500 daily log returns 2005-2019, of 200,000 GAN samples (seed 1) and of the Gaussian.",
  "GAN, generative adversarial network, fat tails, kurtosis, S&P 500, synthetic returns",
  "data/clean/markets_daily.csv (Yahoo Finance, SP500); lectures/L8/results/gan_samples.csv (50,000 GAN samples)",
  "Excess kurtosis and 1 % quantile of real returns; GAN samples from the trained generator.",
  SPX + "from scipy import stats\ng = pd.read_csv(" + RES + " + 'gan_samples.csv').gan_seed1.values / s_tr\n"
  "show('gan.kurt_real', 'excess kurtosis, real', round(float(stats.kurtosis(ztr)), 2))\n"
  "show('gan.var_real_pct', '1 % quantile of daily returns (%)', round(100 * float(np.quantile(ztr, 0.01)) * s_tr, 2))\n"
  "show('gan.n_train', 'training days', len(rtr))\n"
  "b = np.linspace(-8, 8, 121)\nfig, ax = plt.subplots(figsize=(6.4, 2.5))\n"
  "ax.hist(ztr, bins=b, density=True, histtype='stepfilled', alpha=0.35, color=PAL['data'], label='S&P 500, 2005-2019')\n"
  "ax.hist(g, bins=b, density=True, histtype='step', lw=1.5, color=PAL['model'], label='GAN')\n"
  "xx = np.linspace(-8, 8, 400)\nax.plot(xx, stats.norm.pdf(xx), '--', color=PAL['benchmark'], label='Gaussian')\n"
  "ax.set_yscale('log'); ax.set_ylim(1e-5, 1)\nax.set_xlabel('standardised daily return'); ax.set_ylabel('density (log)')\n"
  "legend_bottom(fig, ncol=3)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\nfig.savefig('gan_density.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("gan_losses", "Discriminator and generator losses of the GAN on S&P 500 returns (seed 1, every 50 iterations), with the equilibrium values.",
  "GAN training, discriminator loss, generator loss, non-saturating loss, equilibrium",
  "lectures/L8/results/gan_losses.csv", "At equilibrium $D^*=1/2$: discriminator loss $2\\log 2$, generator loss $\\log 2$.",
  "h = pd.read_csv(" + RES + " + 'gan_losses.csv')\nshow('gan.steps', 'training iterations', int(h.iteration.max()))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.plot(h.iteration, h.loss_d, color=PAL['data'], lw=1, label='discriminator')\n"
  "ax.plot(h.iteration, h.loss_g, color=PAL['model'], lw=1, label='generator')\n"
  "ax.axhline(2 * np.log(2), color='#8a8a8a', ls='--', lw=1, label='2 log 2 (equilibrium)')\nax.axhline(np.log(2), color='#8a8a8a', ls=':', lw=1, label='log 2')\n"
  "ax.set_xlabel('iteration'); ax.set_ylabel('cross-entropy')\nlegend_bottom(fig, ncol=4)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
  "fig.savefig('gan_losses.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("gan_acf", "Autocorrelation of absolute daily returns, S&P 500 2005-2019 against GAN samples: the GAN draws days independently.",
  "volatility clustering, autocorrelation, stylized facts, GAN, synthetic data",
  "data/clean/markets_daily.csv; lectures/L8/results/gan_acf.csv",
  "$\\rho_\\ell(|r|)$ for lags 1-20.",
  SPX + "def acf_abs(x, lags=20):\n    a = np.abs(x) - np.abs(x).mean()\n    return np.array([np.sum(a[l:] * a[:-l]) / np.sum(a * a) for l in range(1, lags + 1)])\n"
  "ar = acf_abs(ztr)\nag = pd.read_csv(" + RES + " + 'gan_acf.csv').gan.values\n"
  "show('gan.acf1_real', 'ACF of |r| at lag 1, real', round(float(ar[0]), 2))\nshow('gan.acf20_real', 'ACF of |r| at lag 20, real', round(float(ar[-1]), 2))\n"
  "show('gan.acf1_gan', 'ACF of |r| at lag 1, GAN', round(float(ag[0]), 2))\n"
  "lags = np.arange(1, 21)\nfig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.bar(lags - 0.2, ar, 0.4, color=PAL['data'], label='S&P 500, 2005-2019')\n"
  "ax.bar(lags + 0.2, ag, 0.4, color=PAL['model'], label='GAN')\nax.axhline(0, color='#222222', lw=0.6)\n"
  "ax.set_xlabel('lag (days)'); ax.set_ylabel('autocorrelation of |r|')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
  "fig.savefig('gan_acf.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("diffusion_forward", "Forward diffusion process applied to the last euro-area AAA yield curve of the sample: the curve at noise steps 0, 20, 60, 120 and 199.",
  "diffusion model, DDPM, forward process, noise schedule, yield curve",
  "data/clean/ecb_yield_curve_daily.csv (European Central Bank, YC)",
  "$\\mathbf x_t=\\sqrt{\\bar\\alpha_t}\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\boldsymbol\\varepsilon$, linear $\\beta_t$ from $10^{-4}$ to 0.02, $T=200$ (standardised maturities).",
  YC + "ym, ys = Y.mean(0), Y.std(0)\nT = 200\nbeta = np.linspace(1e-4, 0.02, T)\nabar = np.cumprod(1 - beta)\n"
  "show('yc.n_days', 'daily curves', len(Y))\nshow('yc.T', 'noise steps', T)\n"
  "x0 = (Y[-1] - ym) / ys\neps = np.random.default_rng(7).standard_normal(11)\nmat = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nfor t, c in zip([0, 20, 60, 120, 199], SERIES):\n"
  "    xt = np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps\n    ax.plot(mat, xt * ys + ym, 'o-', ms=3, color=c, label=f't = {t}')\n"
  "ax.set_xscale('log'); ax.set_xticks([0.25, 1, 2, 5, 10, 30]); ax.set_xticklabels(['3M', '1Y', '2Y', '5Y', '10Y', '30Y']); ax.minorticks_off()\n"
  "ax.set_xlabel('maturity'); ax.set_ylabel('yield (%)')\nlegend_bottom(fig, ncol=5)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
  "fig.savefig('diffusion_forward.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("yc_density", "Density of the euro-area 10-year AAA yield: real daily curves against curves generated by a diffusion model (DDPM), a "
  "multivariate Gaussian and a VAE (5,000 samples each, seed 1).",
  "diffusion model, DDPM, VAE, yield curve, scenario generation, Wasserstein distance",
  "data/clean/ecb_yield_curve_daily.csv; lectures/L8/results/yc_generated_y10.csv (generated 10-year yields)",
  "Regimes of euro rates: share of negative 10-year yields, minimum and maximum.",
  YC + "g = pd.read_csv(" + RES + " + 'yc_generated_y10.csv')\n"
  "show('yc.pct_neg10_real', 'days with a negative 10-year yield (%)', round(100 * float(np.mean(y10 < 0)), 1))\n"
  "show('yc.y10_min', 'minimum 10-year yield (%)', round(float(y10.min()), 2))\nshow('yc.y10_max', 'maximum 10-year yield (%)', round(float(y10.max()), 2))\n"
  "b = np.linspace(-2, 6, 81)\nfig, ax = plt.subplots(figsize=(6.4, 2.5))\n"
  "ax.hist(y10, bins=b, density=True, histtype='stepfilled', alpha=0.35, color=PAL['data'], label='real')\n"
  "ax.hist(g.ddpm, bins=b, density=True, histtype='step', lw=1.6, color=PAL['model'], label='diffusion (DDPM)')\n"
  "ax.hist(g.gauss, bins=b, density=True, histtype='step', lw=1.2, ls='--', color=PAL['benchmark'], label='multivariate Gaussian')\n"
  "ax.hist(g.vae, bins=b, density=True, histtype='step', lw=1.2, ls=':', color=PAL['alt'], label='VAE')\n"
  "ax.set_xlabel('10-year yield (%)'); ax.set_ylabel('density')\nlegend_bottom(fig, ncol=4)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
  "fig.savefig('yc_density.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("yc_scatter", "Level (10-year yield) against slope (10 years minus 3 months) of euro-area AAA curves: real days and the share of inverted curves.",
  "yield curve, slope, inversion, diffusion model, synthetic scenarios",
  "data/clean/ecb_yield_curve_daily.csv (European Central Bank, YC)",
  "Inverted curve: 10-year yield below the 3-month yield.",
  YC + "show('yc.pct_inverted_real', 'inverted curves, real (%)', round(100 * float(np.mean(slope < 0)), 1))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.4))\nii = np.random.default_rng(3).choice(len(Y), 2500, replace=False)\n"
  "ax.scatter(y10[ii], slope[ii], s=2, alpha=0.4, color=PAL['data'], lw=0, label='real')\nax.axhline(0, color='#8a8a8a', lw=0.6)\n"
  "ax.set_xlabel('10Y (%)'); ax.set_ylabel('slope 10Y-3M (pp)')\nlegend_bottom(fig)\nfig.tight_layout(rect=(0, 0.12, 1, 1))\n"
  "fig.savefig('yc_scatter.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

Q("yc_latent_time", "Latent means of a two-factor VAE of the euro-area AAA yield curve over time, and their correlation with level and slope.",
  "variational autoencoder, factor model, level, slope, yield curve",
  "lectures/L8/results/yc_vae_latent.csv (latent means of the trained VAE, with level and slope from data/clean/ecb_yield_curve_daily.csv)",
  "Correlation of the latent means with the level (mean of 11 yields) and the slope (10Y - 3M).",
  "z = pd.read_csv(" + RES + " + 'yc_vae_latent.csv', parse_dates=['date'])\n"
  "show('yc.vae_corr.z2_level', 'corr(z2, level)', round(float(np.corrcoef(z.z2, z.level)[0, 1]), 2))\n"
  "show('yc.vae_corr.z1_slope', 'corr(z1, slope)', round(float(np.corrcoef(z.z1, z.slope)[0, 1]), 2))\n"
  "fig, ax = plt.subplots(figsize=(6.4, 2.3))\nax.plot(z.date, z.z1, color=PAL['model'], lw=0.9, label='$z_1$')\n"
  "ax.plot(z.date, z.z2, color=PAL['data'], lw=0.9, label='$z_2$')\nax.set_ylabel('latent mean')\nlegend_bottom(fig)\n"
  "fig.tight_layout(rect=(0, 0.12, 1, 1))\nfig.savefig('yc_latent_time.png', dpi=200, transparent=True, bbox_inches='tight')\nplt.show()")

if __name__ == "__main__":
    only = sys.argv[1:]
    bad_all, n_all = [], 0
    for q in QS:
        if only and q.chart not in only:
            continue
        q.write(execute=True)
        n, bad = check_numbers(q, os.path.join(HERE, "L8_numbers.json"))
        n_all += n
        bad_all += bad
        print(f"  {q.name}: {n} numbers checked", "OK" if not bad and n else (bad or "NO NUMBERS"))
    print(f"{n_all} numbers checked;", "OK" if not bad_all else bad_all)
    sys.exit(1 if bad_all else 0)
