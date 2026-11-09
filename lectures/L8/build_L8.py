r"""
build_L8.py -- Lecture 8 "Generative models: autoencoders, VAE, GAN, diffusion" (NN), English and Romanian from one source.
Writes lecture8.tex / lecture8_ro.tex from the bilingual slide text below and L8_numbers.json (analysis_L8.py),
compiles both and applies every check of lectures/STYLE.md (nn_build.run_build).
Run (from the repository root):
    python3 lectures/L8/analysis_L8.py
    python3 lectures/L8/make_quantlets_L8.py
    python3 lectures/L8/build_L8.py            # both languages; "en" or "ro" builds one
Hand-typed numbers are definitions only: 60/20/20 % time split, bottleneck 8, beta values, 1 % VaR level, 200 noise steps,
the linear noise schedule 1e-4 to 0.02, 0.1 % alert budget.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, cols, fmt, heat, hier, items, reference_frames,  # noqa: E402
                      run_build, table)

ABBR = {
    "AE": ("autoencoder", "autoencoder (rețea de autocodificare)"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "DDPM": ("denoising diffusion probabilistic model", "model probabilistic de difuzie cu eliminarea zgomotului (denoising diffusion)"),
    "ELBO": ("evidence lower bound", "marginea inferioară a verosimilității (evidence lower bound)"),
    "JS": ("Jensen--Shannon divergence", "divergența Jensen--Shannon"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată (rectified linear unit)"),
    "ES": ("expected shortfall", "pierderea așteptată în coadă (expected shortfall)"),
    "GAN": ("generative adversarial network", "rețea generativă adversarială"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "KL": ("Kullback--Leibler divergence", "divergența Kullback--Leibler"),
    "LLM": ("large language model", "model mare de limbaj"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "MNIST": ("Modified National Institute of Standards and Technology handwritten digits", "cifrele scrise de mînă ale Modified National Institute of Standards and Technology"),
    "PCA": ("principal component analysis", "analiza componentelor principale"),
    "PR-AUC": ("area under the precision-recall curve", "aria de sub curba precizie-recall"),
    "ROC": ("receiver operating characteristic", "caracteristica de operare a receptorului (receiver operating characteristic)"),
    "ROC-AUC": ("area under the ROC curve", "aria de sub curba ROC"),
    "TSTR": ("train on synthetic, test on real", "antrenare pe date sintetice, test pe date reale (train on synthetic, test on real)"),
    "VAE": ("variational autoencoder", "autoencoder variațional"),
    "VaR": ("Value-at-Risk", "valoarea la risc (Value-at-Risk)"),
    "WGAN": ("Wasserstein GAN", "GAN Wasserstein"),
}
IGNORE = {"ULB", "OpenML", "AAA", "ICLR", "NeurIPS", "NIPS", "SSRN", "ECECSR", "IEEE", "ACM", "SSCI", "JASIST", "II", "MIT",
          "PMLR", "ICML", "CEE", "JMLR", "PDF", "DOI", "AI", "IDA", "LLM-VaR", "LLM-ES", "S&P", "EU", "GA", "VA", "LeakyReLU", "TimeGAN", "NEURAL", "BIZ"}
UNITS = set()
REFS = [
    "Arjovsky, M., Chintala, S.\\ \\& Bottou, L.\\ (2017). Wasserstein generative adversarial networks. \\textit{Proceedings of ICML}, PMLR 70, 214--223. "
    "\\href{https://arxiv.org/abs/1701.07875}{arXiv:1701.07875}",
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}, ch.\\ 16--20. Springer. "
    "\\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Cont, R.\\ (2001). Empirical properties of asset returns: stylized facts and statistical issues. \\textit{Quantitative Finance}, 1(2), 223--236. "
    "\\href{https://doi.org/10.1080/713665670}{doi:10.1080/713665670}",
    "Dal Pozzolo, A., Caelen, O., Johnson, R.\\,A.\\ \\& Bontempi, G.\\ (2015). Calibrating probability with undersampling for unbalanced classification. "
    "\\textit{IEEE Symposium Series on Computational Intelligence}, 159--166. \\href{https://doi.org/10.1109/SSCI.2015.33}{doi:10.1109/SSCI.2015.33}",
    "Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A.\\ \\& Bengio, Y.\\ (2014). Generative adversarial nets. "
    "\\textit{Advances in Neural Information Processing Systems}, 27. \\href{https://arxiv.org/abs/1406.2661}{arXiv:1406.2661}",
    "Higgins, I., Matthey, L., Pal, A.\\ et al.\\ (2017). beta-VAE: Learning basic visual concepts with a constrained variational framework. "
    "\\textit{International Conference on Learning Representations}. \\href{https://openreview.net/forum?id=Sy2fzU9gl}{openreview.net}",
    "Ho, J., Jain, A.\\ \\& Abbeel, P.\\ (2020). Denoising diffusion probabilistic models. \\textit{Advances in Neural Information Processing Systems}, 33. "
    "\\href{https://arxiv.org/abs/2006.11239}{arXiv:2006.11239}",
    "Jheng, S.-L., Tak, R., Găman, Ș., Mazurencu-Marinescu-Pele, M.\\ \\& Pele, D.\\,T.\\ (2026). Can foundation models manage risk? Zero-shot VaR and ES "
    "forecasting with conformal calibration in CEE markets. \\textit{Economic Computation and Economic Cybernetics Studies and Research}, 60(2), 267--290. "
    "\\href{https://doi.org/10.24818/18423264/60.2.26.14}{doi:10.24818/18423264/60.2.26.14}",
    "Kingma, D.\\,P.\\ \\& Welling, M.\\ (2014). Auto-encoding variational Bayes. \\textit{International Conference on Learning Representations}. "
    "\\href{https://arxiv.org/abs/1312.6114}{arXiv:1312.6114}",
    "Kupiec, P.\\,H.\\ (1995). Techniques for verifying the accuracy of risk measurement models. \\textit{Journal of Derivatives}, 3(2), 73--84. "
    "\\href{https://doi.org/10.3905/jod.1995.407942}{doi:10.3905/jod.1995.407942}",
    "Pele, D.\\,T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A.\\,T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S.\\ \\& Härdle, W.\\,K.\\ (2026). "
    "In the beginning was the word: LLM-VaR and LLM-ES. SSRN Working Paper 5104383. \\href{https://doi.org/10.2139/ssrn.5104383}{doi:10.2139/ssrn.5104383}",
    "Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N.\\ \\& Ganguli, S.\\ (2015). Deep unsupervised learning using nonequilibrium thermodynamics. "
    "\\textit{Proceedings of ICML}, PMLR 37, 2256--2265. \\href{https://arxiv.org/abs/1503.03585}{arXiv:1503.03585}",
    "Svensson, L.\\,E.\\,O.\\ (1994). Estimating and interpreting forward interest rates: Sweden 1992--1994. NBER Working Paper 4871. "
    "\\href{https://doi.org/10.3386/w4871}{doi:10.3386/w4871}",
    "Wiese, M., Knobloch, R., Korn, R.\\ \\& Kretschmer, P.\\ (2020). Quant GANs: Deep generation of financial time series. \\textit{Quantitative Finance}, "
    "20(9), 1419--1440. \\href{https://doi.org/10.1080/14697688.2020.1730426}{doi:10.1080/14697688.2020.1730426}",
    "Xiao, H., Rasul, K.\\ \\& Vollgraf, R.\\ (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. "
    "\\href{https://arxiv.org/abs/1708.07747}{arXiv:1708.07747}",
    "Yoon, J., Jarrett, D.\\ \\& van der Schaar, M.\\ (2019). Time-series generative adversarial networks. \\textit{Advances in Neural Information "
    "Processing Systems}, 32. \\href{https://papers.nips.cc/paper/8789}{papers.nips.cc/paper/8789}",
]
IGNORE |= {"NBER"}


def it(*xs, sep="2pt"):
    def one(x):
        if isinstance(x, tuple):
            return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                    "\n".join("      \\item " + (s[0].upper() + s[1:] if s[0].islower() else s) for s in x[1:] if s) +
                    "\n    \\end{itemize}")
        return hier(x)
    return (f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{{sep}}}\n" + "\n".join("  \\item " + one(x) for x in xs if x) +
            "\n\\end{itemize}")


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L8_numbers.json"), lang)
    N = V.N
    ro = lang == "ro"
    D = Deck(8, "Generative models: autoencoders, VAE, GAN, diffusion" if not ro else "Modele generative: autoencodere, VAE, GAN, difuzie",
             V, lang, heights=H)
    t = D.t
    NB = t("notebook \\S", "notebook, secțiunea ")
    _c = D.concept
    D.concept = lambda *a, **k: _c(*a, **{**{"wl": "0.58", "wr": "0.39"}, **k})
    Y = N["yc"]

    def de(key):
        v = int(round(float(V.get(key))))
        return "de " if ro and (v >= 20 and (v % 100 >= 20 or v % 100 == 0)) else ""

    def n_(x, spec):
        return fmt(x, spec, lang)

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Explain the difference between a discriminative model $p(y\\mid\\mathbf x)$ and a generative model $p(\\mathbf x)$, and why finance needs both.",
          "Diferența dintre un model discriminativ $p(y\\mid\\mathbf x)$ și un model generativ $p(\\mathbf x)$; de ce finanțele au nevoie de amîndouă"),
        t("Derive and train four families: the autoencoder (AE), the variational autoencoder (VAE), the generative adversarial network (GAN) and the denoising diffusion probabilistic model (DDPM).",
          "Derivarea și antrenarea a patru familii de modele; autoencoderul (AE), autoencoderul variațional (VAE), rețeaua generativă adversarială (GAN) și modelul probabilistic de difuzie (DDPM)"),
        t("Use them on real data: card-fraud detection, clothing images in the format of the Modified National Institute of Standards and Technology (MNIST) digits, S\\&P 500 returns and the euro-area yield curve, each against a classical benchmark.",
          "Aplicarea lor pe date reale; detectarea fraudei cu carduri, imagini de îmbrăcăminte în formatul cifrelor Modified National Institute of Standards and Technology (MNIST), randamentele S\\&P 500 și curba randamentelor din zona euro, fiecare comparat cu un model clasic"),
        t("Judge synthetic data: tails, volatility clustering, risk measures such as Value-at-Risk (VaR), and what a generator cannot reproduce.",
          "Evaluarea datelor sintetice; cozile distribuției, gruparea volatilității, măsuri de risc precum valoarea la risc (VaR) și ce nu poate reproduce un generator"),
        sep="5pt"))
    abbreviation_frames(D, ABBR)

    # ------------------------------------------------------------------ 1 why generative
    D.section(t("Principles of generative modelling", "Principiile modelării generative"))
    D.concept(t("A generative model learns how the data themselves arise", "Un model generativ învață cum apar datele"),
              t("A classifier learns the boundary between classes; a generative model learns the whole distribution, so it can score, simulate and fill in data.",
                "Un clasificator învață granița dintre clase; un model generativ învață toată distribuția datelor."),
              "\\[ \\underbrace{p(y\\mid\\mathbf x)}_{\\text{" + t("discriminative", "discriminativ") + "}}\\qquad\\text{vs.}\\qquad"
              "\\underbrace{p(\\mathbf x)\\ \\text{" + t("or", "sau") + "}\\ p(\\mathbf x,y)=p(\\mathbf x\\mid y)\\,p(y)}_{\\text{" + t("generative", "generativ") + "}} \\]"
              "\\[ \\hat\\btheta=\\argmax_{\\btheta}\\ \\sum_{i=1}^{n}\\log p_{\\btheta}(\\mathbf x_i) \\]"
              "\\where{" + t("$\\mathbf x$ = an observation (transaction, image, day of returns, yield curve); $y$ = a label; "
                             "$p_{\\btheta}$ = model density, fitted by maximum likelihood.",
                             "$\\mathbf x$ = o observație (tranzacție, imagine, zi de randamente, curbă); $y$ = o etichetă; "
                             "$p_{\\btheta}$ = densitatea modelului, estimată prin verosimilitate maximă.") + "}",
              it((t("\\textbf{Four uses in finance}", "\\textbf{Patru utilizări în finanțe}"),
                  t("anomaly detection: low $p(\\mathbf x)$ = suspicious (fraud)", "detectarea anomaliilor: $p(\\mathbf x)$ mic = suspect (fraudă)"),
                  t("scenarios for stress tests and risk measures", "scenarii pentru teste de stres și măsuri de risc"),
                  t("synthetic data shared without client records", "date sintetice partajate fără înregistrări despre clienți")), sep="3pt"),
              t("a classifier ranks clients, a generative model also shows what could happen.",
                "un clasificator ordonează clienții, un model generativ arată și ce s-ar putea întîmpla."),
              cite="(Bishop \\& Bishop, 2024, ch.\\ 16)")

    D.concept(t("Latent-variable models", "Modele cu variabile latente"),
              t("Eleven yields or 784 pixels are driven by a handful of hidden factors; a latent-variable model writes this down.",
                "Unsprezece randamente sau 784 de pixeli sînt determinate de cîțiva factori ascunși; un model cu variabile latente formalizează acest lucru."),
              "\\[ p_{\\btheta}(\\mathbf x)=\\int p_{\\btheta}(\\mathbf x\\mid\\mathbf z)\\,p(\\mathbf z)\\,d\\mathbf z,\\qquad \\mathbf z\\in\\R^k,\\ k\\ll d \\]"
              "\\[ \\text{" + t("linear-Gaussian", "liniar-gaussian") + ":}\\ \\ \\mathbf x=W\\mathbf z+\\boldsymbol\\mu+\\boldsymbol\\varepsilon,\\ "
              "\\mathbf z\\sim\\mathcal N(0,I),\\ \\boldsymbol\\varepsilon\\sim\\mathcal N(0,\\sigma^2I) \\]"
              "\\where{" + t("$\\mathbf z$ = latent factors, $k$ of them; $d$ = dimension of the data; the linear-Gaussian case is probabilistic principal component analysis (PCA), "
                             "with a closed-form likelihood; with a neural decoder the integral has no closed form.",
                             "$\\mathbf z$ = factorii latenți, în număr de $k$; $d$ = dimensiunea datelor; cazul liniar-gaussian este analiza probabilistă a componentelor principale (PCA), "
                             "cu verosimilitate în formă închisă; cu un decodor neuronal integrala nu mai are formă închisă.") + "}",
              it((t("\\textbf{Yield curve}", "\\textbf{Curba randamentelor}"),
                  t("level, slope, curvature: three factors explain most daily moves", "nivel, pantă, curbură: trei factori explică majoritatea mișcărilor zilnice")),
                 (t("\\textbf{The price of non-linearity}", "\\textbf{Prețul neliniarității}"),
                  t("VAE: a lower bound on $\\log p$; GAN: no likelihood at all; diffusion: a bound built step by step",
                    "VAE: o margine inferioară pentru $\\log p$; GAN: nicio verosimilitate; difuzia: o margine construită pas cu pas")), sep="3pt"),
              t("in a model review, ask which part is the latent story and which part is fitted noise.",
                "la evaluarea unui model, întrebați care parte este povestea latentă și care este zgomot estimat."),
              cite="(Bishop \\& Bishop, 2024, sec.\\ 16.2)")

    D.frame(t("Four families of generative models", "Patru familii de modele generative"), table(
        "@{}p{0.15\\textwidth}p{0.20\\textwidth}p{0.17\\textwidth}p{0.17\\textwidth}p{0.23\\textwidth}@{}",
        [t("Family", "Familia"), t("Training signal", "Ce se optimizează"), t("Likelihood", "Verosimilitate"), t("Sampling", "Generarea"), t("In this lecture", "În acest curs")],
        [["AE", t("reconstruction error", "eroarea de reconstrucție"), t("none", "nu"), t("not a generator", "nu este generator"), t("card fraud as anomaly", "frauda cu carduri ca anomalie")],
         ["VAE", t("evidence lower bound", "marginea inferioară a verosimilității"), t("lower bound", "margine inferioară"), t("one pass, blurry", "o trecere, imagini neclare"),
          t("clothing images; yield curves", "imagini de îmbrăcăminte; curbe ale randamentelor")],
         ["GAN", t("game vs a discriminator", "joc contra unui discriminator"), t("none", "nu"), t("one pass, sharp", "o trecere, imagini clare"), t("S\\&P 500 daily returns", "randamentele zilnice S\\&P 500")],
         ["DDPM", t("denoising error", "eroarea de eliminare a zgomotului"), t("lower bound", "margine inferioară"), t("many steps, faithful", "mulți pași, fidel"),
          t("euro-area yield curves", "curbele randamentelor din zona euro")]],
        size="scriptsize", stretch="1.25") + "\n\\vspace{2mm}\n" + it(
        t("\\textbf{No free lunch}: likelihood, sample quality and speed rarely come together; the choice depends on the use.",
          "\\textbf{Nimic nu este gratuit}: verosimilitatea, calitatea eșantioanelor și viteza rareori vin împreună; alegerea depinde de utilizare"),
        t("\\textbf{Normalising flows} (exact likelihood, invertible networks) complete the picture; see Bishop \\& Bishop (2024), ch.\\ 18.",
          "\\textbf{Fluxurile de normalizare} (verosimilitate exactă, rețele inversabile) completează tabloul; vezi Bishop \\& Bishop (2024), cap.\\ 18")),
        "footnotesize", cite="(Bishop \\& Bishop, 2024, ch.\\ 17--20)")

    # ------------------------------------------------------------------ 2 autoencoders
    D.section(t("Autoencoders: anomaly detection for card fraud", "Autoencodere: detectarea anomaliilor la fraude cu carduri"))
    D.concept(t("An autoencoder compresses the data through a bottleneck and reconstructs them", "Un autoencoder comprimă datele printr-un strat îngust și le reconstruiește"),
              t("If the network rebuilds normal transactions from a few numbers, one it cannot rebuild is unusual.",
                "Dacă rețeaua reconstruiește tranzacțiile normale din cîteva numere, una pe care nu o poate reconstrui este neobișnuită."),
              "\\[ \\mathbf z=f_{\\phi}(\\mathbf x)\\in\\R^{k},\\qquad \\hat{\\mathbf x}=g_{\\psi}(\\mathbf z) \\]"
              "\\[ \\min_{\\phi,\\psi}\\ \\frac1n\\sum_{i=1}^{n}\\big\\|\\mathbf x_i-g_{\\psi}(f_{\\phi}(\\mathbf x_i))\\big\\|^2,\\qquad s(\\mathbf x)=\\tfrac1d\\|\\mathbf x-\\hat{\\mathbf x}\\|^2 \\]"
              "\\where{" + t("$f_\\phi$, $g_\\psi$ = encoder and decoder, small multilayer perceptrons (MLP) with rectified linear units (ReLU); $k$ = bottleneck size; "
                             "$s(\\mathbf x)$ = anomaly score; linear $f,g$ span the PCA subspace.",
                             "$f_\\phi$, $g_\\psi$ = codificatorul și decodorul, perceptroni multistrat mici (MLP) cu unități liniare rectificate (ReLU); $k$ = dimensiunea stratului îngust; "
                             "$s(\\mathbf x)$ = scorul de anomalie; cu $f,g$ liniare se obține subspațiul PCA.") + "}",
              it((t("\\textbf{Our autoencoder}", "\\textbf{Autoencoderul nostru}"),
                  t("@{fraud.d_in} inputs (28 anonymised components + log amount) $\\to$ 16 $\\to$ 8 $\\to$ 16 $\\to$ @{fraud.d_in}",
                    "@{fraud.d_in} de intrări (28 de componente anonimizate + logaritmul sumei) $\\to$ 16 $\\to$ 8 $\\to$ 16 $\\to$ @{fraud.d_in}"),
                  t("@{ae.params|,d} parameters, trained on legitimate transactions only", "@{ae.params|,d} " + de("ae.params") + "parametri, antrenat numai pe tranzacții legitime")),
                 t("\\textbf{No labels needed}: useful for new or unconfirmed frauds", "\\textbf{Nu are nevoie de etichete}: util pentru fraude noi sau neconfirmate"), sep="3pt"),
              t("an autoencoder models ``normal'' and flags what differs, not necessarily what is criminal.",
                "un autoencoder modelează ``normalul'' și semnalează ce diferă, nu neapărat ce este infracțional."),
              tryit=NB + "1", cite="(Bishop \\& Bishop, 2024, ch.\\ 19)")

    D.frame(t("The fraud experiment: two days of card payments, split in time", "Experimentul de fraudă: două zile de plăți cu cardul, împărțite în timp"), table(
        "@{}p{0.27\\textwidth}p{0.66\\textwidth}@{}",
        [t("Element", "Element"), t("Choice", "Alegerea")],
        [[t("Data", "Date"), t("ULB credit-card transactions (Dal Pozzolo et al., 2015): @{fraud.n_rows|,d} rows = all @{fraud.n_fraud} frauds + 10\\,\\% of the legitimate ones, re-weighted to the full data",
                              "tranzacțiile cu carduri ULB (Dal Pozzolo et al., 2015): @{fraud.n_rows|,d} de rînduri = toate cele @{fraud.n_fraud} de fraude și 10\\,\\% dintre cele legitime, reponderate la datele complete")],
         [t("Split in time", "Împărțire în timp"), t("first 60\\,\\% of the time span = training (legitimate only, @{fraud.n_train_legit|,d} rows), next 20\\,\\% = validation, last 20\\,\\% = test",
                                                     "primele 60\\,\\% din interval = antrenare (numai legitime, @{fraud.n_train_legit|,d} de rînduri), următoarele 20\\,\\% = validare, ultimele 20\\,\\% = test")],
         [t("Test", "Test"), t("@{fraud.n_full_test|,d} transactions of the full data, @{fraud.n_test_fraud} frauds (@{fraud.test_fraud_rate_pct|.3f}\\,\\%)",
                               "@{fraud.n_full_test|,d} de tranzacții din datele complete, @{fraud.n_test_fraud} de fraude (@{fraud.test_fraud_rate_pct|.3f}\\,\\%)")],
         [t("Benchmarks", "Modele de referință"), t("PCA with 8 components (linear autoencoder); logistic regression trained \\textit{with} the fraud labels",
                                                    "PCA cu 8 componente (autoencoder liniar); regresie logistică antrenată \\textit{cu} etichetele de fraudă")],
         [t("Metrics", "Măsuri"), t("area under the precision-recall curve (PR-AUC) and the area under the receiver operating characteristic curve (ROC-AUC), weighted to the full data; frauds among the top 0.1\\,\\% scores",
                                    "aria de sub curba precizie-recall (PR-AUC) și aria de sub curba caracteristicii de operare a receptorului (ROC-AUC), ponderate la datele complete; fraudele printre primele 0,1\\,\\% scoruri")]],
        size="scriptsize") + "\n\\vspace{1mm}\n" + it(
        t("\\textbf{Leakage-free}: means and standard deviations from the legitimate training transactions; early stopping on the validation part; seeds 1--3.",
          "\\textbf{Fără scurgeri de informație}: mediile și abaterile standard din tranzacțiile legitime de antrenare; oprire timpurie pe partea de validare; semințele 1--3")) +
        D.bottom(None, NB + "1"), "footnotesize", cite="(Dal Pozzolo et al., 2015)")

    D.chart(t("Reconstruction error of fraudulent and legitimate payments", "Eroarea de reconstrucție a plăților frauduloase și a celor legitime"), "ae_error_dist", [
        t("\\textbf{Separation}: the median error of a fraud is @{ae.median_ratio|.1f} times that of a legitimate transaction (test part).",
          "\\textbf{Separarea}: eroarea mediană a unei fraude este de @{ae.median_ratio|.1f} ori mai mare decît a unei tranzacții legitime (partea de test)"),
        t("\\textbf{Overlap}: many frauds still look normal to the autoencoder; they sit in the bulk of the legitimate distribution.",
          "\\textbf{Suprapunerea}: multe fraude arată totuși normal pentru autoencoder; se află în grosul distribuției legitime")],
        height="0.55\\textheight", tryit=NB + "1")

    D.chart(t("Autoencoder without labels compared with a supervised classifier",
              "Autoencoderul fără etichete comparat cu un clasificator supervizat"), "ae_pr", [
        t("\\textbf{PR-AUC on the test part}: autoencoder @{ae.ap|.1f}\\,\\% (sd @{ae.ap_sd|.1f} over 3 seeds), PCA @{ae.ap_pca|.1f}\\,\\%, logistic regression with labels @{ae.ap_logit|.1f}\\,\\%; a random score: @{ae.base_ap|.3f}\\,\\%.",
          "\\textbf{PR-AUC pe partea de test}: autoencoder @{ae.ap|.1f}\\,\\% (abatere standard @{ae.ap_sd|.1f} pe 3 semințe), PCA @{ae.ap_pca|.1f}\\,\\%, regresie logistică cu etichete @{ae.ap_logit|.1f}\\,\\%; un scor aleator: @{ae.base_ap|.3f}\\,\\%"),
        t("\\textbf{ROC-AUC} is high for all three (@{ae.auc|.1f}, @{ae.auc_pca|.1f}, @{ae.auc_logit|.1f}\\,\\%): with 1 fraud in 700 it hides the difference that PR-AUC shows.",
          "\\textbf{ROC-AUC} este mare pentru toate trei (@{ae.auc|.1f}, @{ae.auc_pca|.1f}, @{ae.auc_logit|.1f}\\,\\%): la o fraudă din 700 ascunde diferența pe care o arată PR-AUC")],
        height="0.52\\textheight", tryit=NB + "1")

    D.chart(t("Bottleneck size and anomaly detection", "Dimensiunea stratului îngust și detectarea anomaliilor"), "ae_bottleneck", [
        t("\\textbf{PR-AUC by bottleneck size}: @{ae.bottleneck.2|.1f}\\,\\% with $k=2$, @{ae.bottleneck.8|.1f}\\,\\% with $k=8$, @{ae.bottleneck.24|.1f}\\,\\% with $k=24$ (mean of 2 seeds); best $k$ = @{ae.best_k}.",
          "\\textbf{PR-AUC după dimensiunea stratului îngust}: @{ae.bottleneck.2|.1f}\\,\\% cu $k=2$, @{ae.bottleneck.8|.1f}\\,\\% cu $k=8$, @{ae.bottleneck.24|.1f}\\,\\% cu $k=24$ (media pe 2 semințe); cel mai bun $k$ = @{ae.best_k}"),
        t("\\textbf{Too narrow}: even normal payments are rebuilt badly; \\textbf{too wide}: the network learns to copy frauds as well.",
          "\\textbf{Prea îngust}: chiar și plățile normale sînt reconstruite prost; \\textbf{prea larg}: rețeaua învață să copieze și fraudele")],
        height="0.48\\textheight", tryit=NB + "1",
        job=t("choose $k$ on a validation period, never on the frauds you will report on.", "alegeți $k$ pe o perioadă de validare, niciodată pe fraudele despre care veți raporta."))

    D.frame(t("Frauds found among the top 0.1\\,\\% of scores", "Fraudele identificate în primele 0,1\\,\\% dintre scoruri"), cols(
        table("@{}lrr@{}", [t("Detector", "Detector"), t("Frauds found", "Fraude găsite"), t("Share of frauds", "Pondere")], [
            [t("Autoencoder (no labels)", "Autoencoder (fără etichete)"), "@{ae.top01_frauds_ae}", "@{ae.top01_recall_ae|.1f}\\,\\%"],
            [t("PCA, 8 components (no labels)", "PCA, 8 comp.\\ (fără etichete)"), "@{ae.top01_frauds_pca}", "@{ae.top01_recall_pca|.1f}\\,\\%"],
            [t("Logistic regression (labels)", "Regresie logistică (cu etichete)"), "@{ae.top01_frauds_logit}", "@{ae.top01_recall_logit|.1f}\\,\\%"]],
            size="scriptsize", stretch="1.3"),
        it((t("\\textbf{Alert budget}", "\\textbf{Bugetul de alerte}"),
            t("0.1\\,\\% of the test transactions = @{ae.n_alerts_top01} alerts to review", "0,1\\,\\% dintre tranzacțiile de test = @{ae.n_alerts_top01} de alerte de verificat")),
           (t("\\textbf{Reading}", "\\textbf{Interpretare}"),
            t("labels are worth a lot: supervised learning wins whenever confirmed frauds exist", "etichetele valorează mult: învățarea supervizată cîștigă ori de cîte ori există fraude confirmate"),
            t("the autoencoder is the detector of the unknown, a complement, not a replacement", "autoencoderul este detectorul necunoscutului, un complement, nu un înlocuitor")), sep="4pt"),
        "0.52", "0.45") + D.bottom(t("run both: the classifier for known fraud patterns, the autoencoder to surface new ones for investigators to label.",
                                     "folosiți-le pe amîndouă: clasificatorul pentru tiparele cunoscute, autoencoderul pentru a scoate la iveală tipare noi, pe care investigatorii le etichetează."), NB + "1"),
        "footnotesize", cite=t("(own computation, test part, weighted to the full data)", "(calcule proprii, partea de test, ponderat la datele complete)"))

    # ------------------------------------------------------------------ 3 VAE
    D.section(t("Variational autoencoders", "Autoencodere variaționale"))
    D.concept(t("The VAE makes the autoencoder a probabilistic generator", "VAE transformă autoencoderul într-un generator probabilist"),
              t("Instead of one code per input, the encoder outputs a distribution; the decoder learns to turn random codes into realistic data.",
                "În locul unui cod pentru fiecare intrare, codificatorul produce o distribuție; decodorul învață să transforme coduri aleatoare în date realiste."),
              "\\[ \\log p_{\\btheta}(\\mathbf x)\\ \\ge\\ \\underbrace{\\E_{q_{\\phi}(\\mathbf z\\mid\\mathbf x)}\\big[\\log p_{\\btheta}(\\mathbf x\\mid\\mathbf z)\\big]}_{\\text{"
              + t("reconstruction", "reconstrucție") + "}}-\\underbrace{\\mathrm{KL}\\big(q_{\\phi}(\\mathbf z\\mid\\mathbf x)\\,\\|\\,p(\\mathbf z)\\big)}_{\\text{" + t("regulariser", "regularizare") + "}} \\]"
              "\\where{" + t("right-hand side = evidence lower bound (ELBO); $q_\\phi(\\mathbf z\\mid\\mathbf x)=\\mathcal N(\\boldsymbol\\mu_\\phi(\\mathbf x),\\operatorname{diag}\\boldsymbol\\sigma^2_\\phi(\\mathbf x))$ = encoder; "
                             "$p_{\\btheta}(\\mathbf x\\mid\\mathbf z)$ = decoder; $p(\\mathbf z)=\\mathcal N(0,I)$ = prior; the second term is the Kullback--Leibler divergence (KL).",
                             "membrul drept = marginea inferioară a verosimilității (ELBO); $q_\\phi(\\mathbf z\\mid\\mathbf x)=\\mathcal N(\\boldsymbol\\mu_\\phi(\\mathbf x),\\operatorname{diag}\\boldsymbol\\sigma^2_\\phi(\\mathbf x))$ = codificatorul; "
                             "$p_{\\btheta}(\\mathbf x\\mid\\mathbf z)$ = decodorul; $p(\\mathbf z)=\\mathcal N(0,I)$ = distribuția a priori; al doilea termen este divergența Kullback--Leibler (KL).") + "}",
              it((t("\\textbf{Why a bound}", "\\textbf{De ce o margine}"),
                  "$\\log p(\\mathbf x)=\\text{ELBO}+\\mathrm{KL}\\big(q(\\mathbf z\\mid\\mathbf x)\\|p(\\mathbf z\\mid\\mathbf x)\\big)$",
                  t("the gap is the error of the encoder; derivation in Appendix A", "diferența este eroarea codificatorului; derivarea, în Anexa A")),
                 (t("\\textbf{Closed-form KL}", "\\textbf{KL în formă închisă}"),
                  "$\\mathrm{KL}=\\tfrac12\\sum_{j=1}^{k}\\big(\\mu_j^2+\\sigma_j^2-\\log\\sigma_j^2-1\\big)$"), sep="3pt"),
              t("the KL term is what makes random codes meaningful: it packs all inputs around the origin of the latent space.",
                "termenul KL face codurile aleatoare să aibă sens: aduce toate intrările în jurul originii spațiului latent."),
              tryit=NB + "2", cite="(Kingma \\& Welling, 2014; Bishop \\& Bishop, 2024, sec.\\ 19.2)", wl="0.58", wr="0.39")

    D.concept(t("The reparameterisation trick", "Trucul reparametrizării"),
              t("Sampling is not differentiable, but a shifted and scaled standard normal is.",
                "Extragerea aleatoare nu este derivabilă, dar o normală standard translatată și scalată este."),
              "\\[ \\mathbf z=\\boldsymbol\\mu_\\phi(\\mathbf x)+\\boldsymbol\\sigma_\\phi(\\mathbf x)\\odot\\boldsymbol\\varepsilon,\\qquad \\boldsymbol\\varepsilon\\sim\\mathcal N(0,I) \\]"
              "\\[ \\nabla_\\phi\\,\\E_{q_\\phi}[h(\\mathbf z)]=\\E_{\\boldsymbol\\varepsilon}\\big[\\nabla_\\phi\\,h(\\boldsymbol\\mu_\\phi+\\boldsymbol\\sigma_\\phi\\odot\\boldsymbol\\varepsilon)\\big] \\]"
              "\\where{" + t("$\\odot$ = element-wise product; the randomness sits in $\\boldsymbol\\varepsilon$, which does not depend on $\\phi$; one draw of $\\boldsymbol\\varepsilon$ per image and step is enough in practice.",
                             "$\\odot$ = produs element cu element; hazardul este în $\\boldsymbol\\varepsilon$, care nu depinde de $\\phi$; în practică ajunge o extragere a lui $\\boldsymbol\\varepsilon$ pentru fiecare imagine și pas.") + "}",
              it((t("\\textbf{Our VAE on Fashion-MNIST}", "\\textbf{VAE-ul nostru pe Fashion-MNIST}"),
                  t("28 $\\times$ 28 grey images of 10 kinds of clothing, a drop-in for the MNIST digits",
                    "imagini gri 28 $\\times$ 28 cu 10 tipuri de îmbrăcăminte, înlocuitorul cifrelor MNIST"),
                  t("@{vae.n_train|,d} training, @{vae.n_test|,d} test images; Bernoulli decoder (binary cross-entropy)",
                    "@{vae.n_train|,d} de imagini de antrenare, @{vae.n_test|,d} de test; decodor Bernoulli (entropie încrucișată binară)")),
                 (t("\\textbf{Size}", "\\textbf{Dimensiunea}"),
                  t("784 $\\to$ 400 $\\to$ 200 $\\to$ $k$; @{vae.params_16|,d} parameters for $k=16$; @{vae.epochs} epochs",
                    "784 $\\to$ 400 $\\to$ 200 $\\to$ $k$; @{vae.params_16|,d} de parametri pentru $k=16$; @{vae.epochs} de epoci")), sep="3pt"),
              t("the same trick drives every model that samples inside the network: VAEs, dropout as inference, stochastic policies.",
                "același truc stă la baza oricărui model care extrage aleator în interiorul rețelei: VAE, dropout ca inferență, politici stocastice."),
              tryit=NB + "2", cite="(Kingma \\& Welling, 2014; Xiao et al., 2017)")

    D.chart(t("A two-dimensional latent space separates the product types", "Un spațiu latent cu două dimensiuni separă tipurile de produse"), "vae_latent", [
        t("\\textbf{No labels used}: the VAE never saw the classes, yet shoes, bags and garments occupy separate regions.",
          "\\textbf{Fără etichete}: VAE nu a văzut clasele, dar încălțămintea, gențile și hainele ocupă regiuni separate"),
        t("\\textbf{Measured}: a 15-nearest-neighbour classifier on the two latent means reaches @{vae.knn_acc_2d|.1f}\\,\\% accuracy (5-fold, test images).",
          "\\textbf{Măsurat}: un clasificator cu 15 vecini pe cele două medii latente atinge o acuratețe de @{vae.knn_acc_2d|.1f}\\,\\% (5 partiții, imaginile de test)")],
        height="0.58\\textheight", tryit=NB + "2")

    D.chart(t("Quality of VAE reconstructions and generated samples", "Calitatea reconstrucțiilor VAE și a eșantioanelor generate"), "vae_samples", [
        t("\\textbf{Rows}: real test images; their reconstructions through a 16-dimensional code; new images decoded from random codes $\\mathbf z\\sim\\mathcal N(0,I)$.",
          "\\textbf{Rîndurile}: imagini reale de test; reconstrucția lor printr-un cod cu 16 dimensiuni; imagini noi decodate din coduri aleatoare $\\mathbf z\\sim\\mathcal N(0,I)$"),
        t("\\textbf{Blur} comes from the Gaussian posterior and the pixel-wise loss: the decoder averages over plausible details.",
          "\\textbf{Neclaritatea} vine din distribuția a posteriori gaussiană și din pierderea calculată pixel cu pixel: decodorul face media detaliilor plauzibile")],
        height="0.48\\textheight", tryit=NB + "2")

    D.chart(t("Trade-off between reconstruction and the KL penalty during training", "Compromisul dintre reconstrucție și penalizarea KL în timpul antrenării"), "vae_elbo", [
        t("\\textbf{Test, $k=16$, last epoch}: reconstruction @{vae.beta.b1.rec|.1f} nats per image, KL @{vae.beta.b1.kl|.1f} nats; ELBO @{vae.elbo_16|.1f} nats.",
          "\\textbf{Test, $k=16$, ultima epocă}: reconstrucția @{vae.beta.b1.rec|.1f} nats pe imagine, KL @{vae.beta.b1.kl|.1f} nats; ELBO @{vae.elbo_16|.1f} nats"),
        t("\\textbf{The KL rises} while the error falls: the encoder spends more information on each image as it learns.",
          "\\textbf{KL crește} cînd eroarea scade: codificatorul folosește mai multă informație pentru fiecare imagine pe măsură ce învață")],
        height="0.48\\textheight", tryit=NB + "2", cite="(own computation, seed 1)" if not ro else "(calcule proprii, sămînța 1)")

    B = N["vae"]["beta"]
    D.frame(t("$\\beta$-VAE: reconstruction quality and latent-space structure", "$\\beta$-VAE: calitatea reconstrucției și structura spațiului latent"), cols(
        "\\[ \\loss_\\beta=-\\E_{q}[\\log p(\\mathbf x\\mid\\mathbf z)]+\\beta\\,\\mathrm{KL}\\big(q(\\mathbf z\\mid\\mathbf x)\\|p(\\mathbf z)\\big) \\]" +
        table("@{}lrrr@{}", ["$\\beta$", t("Reconstruction", "Reconstrucție"), "KL", t("Active dims", "Dim.\\ active")],
              [[n_(b, ".1f"), n_(B[k]["rec"], ".1f"), n_(B[k]["kl"], ".1f"), str(B[k]["active"])] for b, k in [(0.5, "b05"), (1.0, "b1"), (4.0, "b4")]],
              size="scriptsize", stretch="1.3"),
        it((t("\\textbf{Larger $\\beta$}", "\\textbf{$\\beta$ mai mare}"),
            t("fewer active latent dimensions (KL per dimension $>0.05$): from @{vae.beta.b05.active} to @{vae.beta.b4.active} of 16",
              "mai puține dimensiuni latente active (KL pe dimensiune $>0{,}05$): de la @{vae.beta.b05.active} la @{vae.beta.b4.active} din 16"),
            t("worse reconstruction, smoother and more disentangled codes", "reconstrucție mai slabă, coduri mai netede și mai separate")),
           (t("\\textbf{Smaller $\\beta$}", "\\textbf{$\\beta$ mai mic}"),
            t("closer to a plain autoencoder: good copies, poor samples", "mai aproape de un autoencoder simplu: copii bune, eșantioane slabe")), sep="4pt"),
        "0.50", "0.47") + D.bottom(t("for scenario generation prefer a tidy latent space ($\\beta\\ge1$); for compression prefer reconstruction ($\\beta<1$).",
                                     "pentru generarea de scenarii preferați un spațiu latent ordonat ($\\beta\\ge1$); pentru comprimare, reconstrucția ($\\beta<1$)."), NB + "2"),
        "footnotesize", cite=t("(Higgins et al., 2017; own computation, test images, 20 epochs)", "(Higgins et al., 2017; calcule proprii, imaginile de test, 20 de epoci)"))

    # ------------------------------------------------------------------ 4 GAN
    D.section(t("Generative adversarial networks", "Rețele generative adversariale"))
    D.concept(t("Generative adversarial networks: generator and discriminator", "Rețelele generative adversariale: generatorul și discriminatorul"),
              t("The generator turns noise into fake data; the discriminator tries to tell fake from real; each improves against the other.",
                "Generatorul face date false din zgomot, discriminatorul caută falsul; fiecare progresează împotriva celuilalt."),
              "\\[ \\min_{G}\\max_{D}\\ \\E_{\\mathbf x\\sim p_{data}}\\big[\\log D(\\mathbf x)\\big]+\\E_{\\mathbf z\\sim p(\\mathbf z)}\\big[\\log\\big(1-D(G(\\mathbf z))\\big)\\big] \\]"
              "\\[ D^{*}(\\mathbf x)=\\frac{p_{data}(\\mathbf x)}{p_{data}(\\mathbf x)+p_G(\\mathbf x)}\\ \\Rightarrow\\ "
              "\\text{" + t("value", "valoarea") + "}=-\\log4+2\\,\\mathrm{JS}(p_{data}\\|p_G) \\]"
              "\\where{" + t("$G$ = generator network, $D$ = discriminator (probability that $\\mathbf x$ is real); $p_G$ = distribution of $G(\\mathbf z)$; "
                             "Jensen--Shannon divergence (JS), zero only when $p_G=p_{data}$.",
                             "$G$ = rețeaua generator, $D$ = discriminatorul (probabilitatea ca $\\mathbf x$ să fie real); $p_G$ = distribuția lui $G(\\mathbf z)$; "
                             "divergența Jensen--Shannon (JS), nulă numai cînd $p_G=p_{data}$.") + "}",
              it((t("\\textbf{At equilibrium}", "\\textbf{La echilibru}"),
                  t("$D^{*}=1/2$ everywhere: the detective can only guess; discriminator loss $2\\log 2$", "$D^{*}=1/2$ peste tot: detectivul doar ghicește; pierderea discriminatorului $2\\log 2$")),
                 (t("\\textbf{No likelihood}", "\\textbf{Fără verosimilitate}"),
                  t("a GAN never writes down $p_G$; it only learns to sample from it", "un GAN nu scrie niciodată $p_G$; doar învață să extragă din ea")), sep="3pt"),
              t("GANs are judged by what their samples get right (tails, dependence), not by a likelihood number.",
                "un GAN se judecă după eșantioane (cozi, dependență), nu după verosimilitate."),
              tryit=NB + "3", cite="(Goodfellow et al., 2014; Bishop \\& Bishop, 2024, ch.\\ 17)", wl="0.58", wr="0.39")

    D.concept(t("Training a GAN: alternate steps and the non-saturating loss", "Antrenarea unui GAN: pași alternativi și pierderea nesaturată"),
              t("Each step updates the detective on a batch of real and fake data, then the forger on fresh noise.",
                "La fiecare pas se actualizează întîi detectivul, apoi falsificatorul."),
              "\\[ \\loss_D=-\\tfrac1m\\sum_{i}\\big[\\log D(\\mathbf x_i)+\\log(1-D(G(\\mathbf z_i)))\\big] \\]"
              "\\[ \\loss_G=-\\tfrac1m\\sum_{i}\\log D(G(\\mathbf z_i))\\quad\\text{(" + t("non-saturating", "nesaturată") + ")} \\]"
              "\\where{" + t("$m=256$ days per batch; $\\mathbf z\\in\\R^8$ standard normal; $\\loss_G$ replaces $\\log(1-D(G))$, whose gradient vanishes when $D$ wins early; "
                             "Adam with learning rate $2\\cdot10^{-4}$, $\\beta_1=0.5$.",
                             "$m=256$ de zile pe lot; $\\mathbf z\\in\\R^8$ normal standard; $\\loss_G$ evită gradientul nul al lui $\\log(1-D(G))$ cînd $D$ cîștigă devreme; "
                             "Adam, rata $2\\cdot10^{-4}$, $\\beta_1=0{,}5$.") + "}",
              it((t("\\textbf{Our GAN}", "\\textbf{GAN-ul nostru}"),
                  t("S\\&P 500 daily log returns @{gan.train_first|date} to 2019, @{gan.n_train|,d} days, standardised",
                    "randamentele logaritmice zilnice ale indicelui S\\&P 500, @{gan.train_first|date} -- 2019, @{gan.n_train|,d} de zile, standardizate"),
                  t("generator @{gan.params_g|,d} and discriminator @{gan.params_d|,d} parameters; @{gan.steps|,d} steps, 3 seeds",
                    "generator cu @{gan.params_g|,d} de parametri, discriminator cu @{gan.params_d|,d}; @{gan.steps|,d} de pași, 3 semințe")), sep="3pt"),
              t("watch both losses: a discriminator loss stuck at zero means the generator has stopped learning.",
                "o pierdere a discriminatorului blocată la zero arată că generatorul nu mai învață."),
              tryit=NB + "3", cite="(Goodfellow et al., 2014)")

    D.chart(t("The two losses settle near the equilibrium values", "Cele două pierderi se stabilizează lîngă valorile de echilibru"), "gan_losses", [
        t("\\textbf{Equilibrium}: the discriminator loss hovers around $2\\log2$ and the generator loss around $\\log 2$, the values of $D^{*}=1/2$.",
          "\\textbf{Echilibrul}: pierderea discriminatorului oscilează în jurul lui $2\\log2$, iar a generatorului în jurul lui $\\log 2$, valorile lui $D^{*}=1/2$"),
        t("\\textbf{No convergence proof}: gradient play can cycle; the loss curves do not measure sample quality.",
          "\\textbf{Nicio garanție de convergență}: jocul pe gradienți poate oscila; curbele de pierdere nu măsoară calitatea eșantioanelor")],
        height="0.50\\textheight", tryit=NB + "3")

    D.chart(t("The GAN reproduces the fat tails missed by the Gaussian model", "GAN-ul reproduce cozile groase pe care modelul gaussian nu le surprinde"), "gan_density", [
        t("\\textbf{Excess kurtosis}: real @{gan.kurt_real|.2f}, GAN @{gan.kurt_gan|.2f} (sd @{gan.kurt_gan_sd|.2f} over 3 seeds), Gaussian 0.",
          "\\textbf{Excesul de aplatizare}: real @{gan.kurt_real|.2f}, GAN @{gan.kurt_gan|.2f} (abatere standard @{gan.kurt_gan_sd|.2f} pe 3 semințe), normala 0"),
        t("\\textbf{Kolmogorov--Smirnov distance} to the real returns: GAN @{gan.ks_gan|.3f}, fitted Gaussian @{gan.ks_norm|.3f}.",
          "\\textbf{Distanța Kolmogorov--Smirnov} față de randamentele reale: GAN @{gan.ks_gan|.3f}, normala estimată @{gan.ks_norm|.3f}")],
        height="0.52\\textheight", tryit=NB + "3", cite="(Cont, 2001)")

    D.frame(t("A risk check: 1\\,\\% Value-at-Risk on the years after training", "Un test de risc: valoarea la risc de 1\\,\\% în anii de după antrenare"), cols(
        "\\[ \\mathrm{VaR}_{0.01}=-q_{0.01}(r),\\qquad \\hat\\pi=\\frac{1}{T}\\sum_{t=1}^{T}\\mathbf 1\\{r_t<q_{0.01}\\} \\]" +
        table("@{}lrr@{}", [t("Method (fitted 2005--2019)", "Metoda (estimată 2005--2019)"), t("1\\,\\% quantile", "Cuantila de 1\\,\\%"), t("Exceedances", "Depășiri")], [
            [t("Historical simulation", "Simulare istorică"), "@{gan.var_real_pct|.2f}\\,\\%", "@{gan.viol_hs_pct|.2f}\\,\\%"],
            [t("GAN samples (3 seeds)", "Eșantioane GAN (3 semințe)"), "@{gan.var_gan_pct|.2f}\\,\\%", "@{gan.viol_gan_pct|.2f}\\,\\%"],
            [t("Gaussian", "Normală"), "@{gan.var_norm_pct|.2f}\\,\\%", "@{gan.viol_norm_pct|.2f}\\,\\%"]], size="scriptsize", stretch="1.3"),
        it((t("\\textbf{Test period}", "\\textbf{Perioada de test}"),
            t("@{gan.test_first|date} to @{gan.test_last|date}, @{gan.n_test|,d} days; a correct 1\\,\\% VaR is exceeded on about @{gan.n_viol_expected} days",
              "@{gan.test_first|date} -- @{gan.test_last|date}, @{gan.n_test|,d} de zile; o valoare VaR corectă de 1\\,\\% este depășită în circa @{gan.n_viol_expected} zile")),
           (t("\\textbf{Reading}", "\\textbf{Interpretare}"),
            t("the GAN quantile is as good as historical simulation; the Gaussian is exceeded @{gan.viol_norm_pct|.2f}\\,\\% of the days",
              "cuantila GAN este la fel de bună ca simularea istorică; normala este depășită în @{gan.viol_norm_pct|.2f}\\,\\% dintre zile"),
            t("unconditional quantiles only: no model here reacts to volatility", "numai cuantile necondiționate: niciun model de aici nu reacționează la volatilitate"),
            t("the expected shortfall (ES), the mean loss beyond the VaR, depends on the same tail", "pierderea așteptată în coadă (ES), media pierderilor dincolo de VaR, depinde de aceeași coadă")), sep="4pt"),
        "0.52", "0.45") + D.bottom(t("a generator is only as good as the risk number it reproduces: backtest it like any VaR model (Kupiec, 1995).",
                                     "un generator este atît de bun cît măsura de risc pe care o reproduce: testați-l retrospectiv ca pe orice model VaR (Kupiec, 1995)."), NB + "3"),
        "footnotesize", cite=t("(own computation)", "(calcule proprii)"))

    D.chart(t("A GAN on single days cannot learn volatility clustering", "Un GAN pe zile izolate nu poate învăța gruparea volatilității"), "gan_acf", [
        t("\\textbf{Real returns}: the autocorrelation of $|r_t|$ is @{gan.acf1_real|.2f} at lag 1 and still @{gan.acf20_real|.2f} at lag 20: calm and turbulent periods cluster.",
          "\\textbf{Randamentele reale}: autocorelația lui $|r_t|$ este @{gan.acf1_real|.2f} la decalajul 1 și încă @{gan.acf20_real|.2f} la decalajul 20: perioadele calme și cele agitate se grupează"),
        t("\\textbf{GAN samples} are independent draws (@{gan.acf1_gan|.2f} at lag 1): the right marginal, the wrong dynamics; sequence models (TimeGAN, Quant GANs) generate whole paths.",
          "\\textbf{Eșantioanele GAN} sînt extrageri independente (@{gan.acf1_gan|.2f} la decalajul 1): distribuția marginală corectă, dinamica greșită; modelele de secvențe (TimeGAN, Quant GANs) generează traiectorii întregi")],
        height="0.48\\textheight", tryit=NB + "3", cite="(Cont, 2001; Yoon et al., 2019; Wiese et al., 2020)")

    D.concept(t("Mode collapse and the Wasserstein GAN", "Colapsul modurilor și GAN-ul Wasserstein"),
              t("A generator can win by producing a few convincing samples; the Wasserstein distance gives useful gradients even when the supports do not overlap.",
                "Un generator poate cîștiga producînd cîteva eșantioane convingătoare; distanța Wasserstein dă gradienți utili chiar cînd suporturile nu se suprapun."),
              "\\[ W_1(p_{data},p_G)=\\sup_{\\|f\\|_{L}\\le1}\\ \\E_{p_{data}}[f(\\mathbf x)]-\\E_{p_G}[f(\\mathbf x)] \\]"
              "\\where{" + t("$f$ = critic network, constrained to be 1-Lipschitz (weight clipping or a gradient penalty); the generator minimises $W_1$: the Wasserstein GAN (WGAN).",
                             "$f$ = rețeaua critic, constrînsă să fie 1-Lipschitz (tăierea ponderilor sau o penalizare a gradientului); generatorul minimizează $W_1$: GAN-ul Wasserstein (WGAN).") + "}",
              it((t("\\textbf{Mode collapse}", "\\textbf{Colapsul modurilor}"),
                  t("the generator maps all $\\mathbf z$ to one region: e.g.\\ only calm days, never crashes", "generatorul duce toți $\\mathbf z$ în aceeași regiune: de exemplu, numai zile calme, niciodată prăbușiri")),
                 (t("\\textbf{Checks}", "\\textbf{Verificări}"),
                  t("compare tail quantiles, the share of extreme days and the spread of samples with the real data", "comparați cuantilele din coadă, ponderea zilelor extreme și dispersia eșantioanelor cu datele reale")), sep="3pt"),
              t("a stress-test generator that has collapsed onto calm days is worse than no generator.",
                "un generator pentru teste de stres care a colapsat pe zilele calme este mai rău decît lipsa unui generator."),
              cite="(Arjovsky, Chintala \\& Bottou, 2017)")

    # ------------------------------------------------------------------ 5 diffusion
    D.section(t("Diffusion models", "Modele de difuzie"))
    D.concept(t("Diffusion models: forward noising and reverse denoising", "Modelele de difuzie: adăugarea și eliminarea treptată a zgomotului"),
              t("Adding a little Gaussian noise many times turns any data into pure noise; a network learns each small reverse step.",
                "Cu zgomot gaussian adăugat treptat, orice date devin zgomot pur; o rețea învață fiecare pas înapoi."),
              "\\[ q(\\mathbf x_t\\mid\\mathbf x_{t-1})=\\mathcal N\\big(\\sqrt{1-\\beta_t}\\,\\mathbf x_{t-1},\\ \\beta_tI\\big) \\]"
              "\\[ q(\\mathbf x_t\\mid\\mathbf x_0)=\\mathcal N\\big(\\sqrt{\\bar\\alpha_t}\\,\\mathbf x_0,\\ (1-\\bar\\alpha_t)I\\big),\\quad \\bar\\alpha_t=\\prod_{s=1}^{t}(1-\\beta_s) \\]"
              "\\where{" + t("$\\mathbf x_0$ = a real observation (here a standardised yield curve); $\\beta_t$ = noise schedule, linear from $10^{-4}$ to 0.02 over $T$ = @{yc.T} steps; "
                             "$\\bar\\alpha_T\\approx0$, so $\\mathbf x_T$ is almost standard normal.",
                             "$\\mathbf x_0$ = o observație reală (aici o curbă a randamentelor standardizată); $\\beta_t$ = programul zgomotului, liniar de la $10^{-4}$ la 0,02 în $T$ = @{yc.T} de pași; "
                             "$\\bar\\alpha_T\\approx0$, deci $\\mathbf x_T$ este aproape normal standard.") + "}",
              it((t("\\textbf{Closed form}", "\\textbf{Formă închisă}"),
                  t("any step $t$ is reached in one draw: $\\mathbf x_t=\\sqrt{\\bar\\alpha_t}\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol\\varepsilon$",
                    "orice pas $t$ se obține dintr-o singură extragere: $\\mathbf x_t=\\sqrt{\\bar\\alpha_t}\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol\\varepsilon$")),
                 (t("\\textbf{No parameters}", "\\textbf{Fără parametri}"),
                  t("the forward process is fixed; only the reverse process is learnt", "procesul înainte este fix; numai procesul invers se învață")), sep="3pt"),
              t("diffusion turns generation into many small regression problems, which is why it trains stably.",
                "difuzia transformă generarea în multe probleme mici de regresie; de aceea se antrenează stabil."),
              tryit=NB + "4", cite="(Sohl-Dickstein et al., 2015; Ho, Jain \\& Abbeel, 2020)")

    D.chart(t("The forward process on a real yield curve", "Procesul înainte pe o curbă reală a randamentelor"), "diffusion_forward", [
        t("\\textbf{Curve of @{yc.forward_date|date}}, 11 maturities from 3 months to 30 years; by $t=199$ the shape is gone and only noise remains.",
          "\\textbf{Curba din @{yc.forward_date|date}}, 11 scadențe, de la 3 luni la 30 de ani; la $t=199$ forma a dispărut și a rămas doar zgomot"),
        t("\\textbf{The network} must learn to walk back from right to left, one small step at a time.",
          "\\textbf{Rețeaua} trebuie să învețe drumul înapoi, de la dreapta la stînga, cîte un pas mic")],
        height="0.52\\textheight", tryit=NB + "4", cite=t("(mechanism shown on real data; one draw of the noise)", "(mecanism ilustrat pe date reale; o singură extragere a zgomotului)"))

    D.concept(t("Learning to denoise: predict the noise that was added", "Învățarea eliminării zgomotului: prezicem zgomotul adăugat"),
              t("The network sees a noisy curve and the step $t$, and guesses the noise; subtracting the guess, step by step, gives a new curve.",
                "Rețeaua vede curba cu zgomot și pasul $t$ și ghicește zgomotul; scăzîndu-l pas cu pas, obținem o curbă nouă."),
              "\\[ \\loss(\\btheta)=\\E_{t,\\mathbf x_0,\\boldsymbol\\varepsilon}\\Big\\|\\boldsymbol\\varepsilon-\\boldsymbol\\varepsilon_{\\btheta}\\big(\\sqrt{\\bar\\alpha_t}\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol\\varepsilon,\\ t\\big)\\Big\\|^2 \\]"
              "\\[ \\mathbf x_{t-1}=\\frac{1}{\\sqrt{\\alpha_t}}\\Big(\\mathbf x_t-\\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\boldsymbol\\varepsilon_{\\btheta}(\\mathbf x_t,t)\\Big)+\\sqrt{\\beta_t}\\,\\mathbf u,\\ \\ \\mathbf u\\sim\\mathcal N(0,I) \\]"
              "\\where{" + t("$\\boldsymbol\\varepsilon_{\\btheta}$ = denoiser network; $\\alpha_t=1-\\beta_t$; $t$ uniform on $\\{1,\\dots,T\\}$; the loss is a simplified, reweighted ELBO.",
                             "$\\boldsymbol\\varepsilon_{\\btheta}$ = rețeaua care elimină zgomotul; $\\alpha_t=1-\\beta_t$; $t$ uniform pe $\\{1,\\dots,T\\}$; pierderea este un ELBO simplificat și reponderat.") + "}",
              it((t("\\textbf{Our denoiser}", "\\textbf{Rețeaua noastră}"),
                  t("MLP with 3 hidden layers of 256 and a sinusoidal embedding of $t$: @{yc.params|,d} parameters", "MLP cu 3 straturi ascunse de 256 și o reprezentare sinusoidală a lui $t$: @{yc.params|,d} de parametri"),
                  t("@{yc.steps|,d} steps of 512 curves; 5,000 curves sampled per seed", "@{yc.steps|,d} de pași cu loturi de 512 curbe; 5.000 de curbe generate pe sămînță")),
                 (t("\\textbf{Data}", "\\textbf{Date}"),
                  t("@{yc.n_days|,d} daily curves of the European Central Bank, @{yc.first|date} to @{yc.last|date}", "@{yc.n_days|,d} de curbe zilnice ale Băncii Centrale Europene, @{yc.first|date} -- @{yc.last|date}")), sep="3pt"),
              t("the same loss, with images and a text condition, runs the image generators you know; here it generates interest-rate scenarios.",
                "aceeași pierdere stă la baza generatoarelor de imagini, iar aici generează scenarii de dobîndă."),
              tryit=NB + "4", cite="(Ho, Jain \\& Abbeel, 2020; Bishop \\& Bishop, 2024, ch.\\ 20)", wl="0.58", wr="0.39")

    D.chart(t("The two regimes of euro rates in diffusion and Gaussian samples", "Cele două regimuri ale dobînzilor euro în eșantioanele generate prin difuzie și de modelul gaussian"), "yc_density", [
        t("\\textbf{Real 10-year yield}: from @{yc.y10_min|.2f}\\,\\% to @{yc.y10_max|.2f}\\,\\%, negative on @{yc.pct_neg10_real|.1f}\\,\\% of the days: a low-rate and a normal-rate regime.",
          "\\textbf{Randamentul real la 10 ani}: de la @{yc.y10_min|.2f}\\,\\% la @{yc.y10_max|.2f}\\,\\%, negativ în @{yc.pct_neg10_real|.1f}\\,\\% dintre zile: un regim de dobînzi scăzute și unul de dobînzi normale"),
        t("\\textbf{Wasserstein distance} of the generated 10-year yield to the real one: DDPM @{yc.ddpm.w_y10|.3f}, VAE @{yc.vae.w_y10|.3f}, Gaussian @{yc.gauss.w_y10|.3f} percentage points.",
          "\\textbf{Distanța Wasserstein} dintre randamentul generat la 10 ani și cel real: DDPM @{yc.ddpm.w_y10|.3f}, VAE @{yc.vae.w_y10|.3f}, normala @{yc.gauss.w_y10|.3f} puncte procentuale")],
        height="0.52\\textheight", tryit=NB + "4")

    D.chart(t("Diffusion preserves the nonlinear co-movement of level and slope", "Difuzia păstrează dependența neliniară dintre nivel și pantă"), "yc_scatter", [
        t("\\textbf{Inverted curves} (10 years below 3 months): @{yc.pct_inverted_real|.1f}\\,\\% of real days, @{yc.ddpm.pct_inverted|.1f}\\,\\% of diffusion samples, @{yc.gauss.pct_inverted|.1f}\\,\\% of Gaussian samples.",
          "\\textbf{Curbe inversate} (10 ani sub 3 luni): @{yc.pct_inverted_real|.1f}\\,\\% dintre zilele reale, @{yc.ddpm.pct_inverted|.1f}\\,\\% dintre eșantioanele difuziei, @{yc.gauss.pct_inverted|.1f}\\,\\% dintre cele normale"),
        t("\\textbf{Slope distance} (Wasserstein): DDPM @{yc.ddpm.w_slope|.3f}, VAE @{yc.vae.w_slope|.3f}, Gaussian @{yc.gauss.w_slope|.3f}.",
          "\\textbf{Distanța pantei} (Wasserstein): DDPM @{yc.ddpm.w_slope|.3f}, VAE @{yc.vae.w_slope|.3f}, normala @{yc.gauss.w_slope|.3f}")],
        height="0.50\\textheight", tryit=NB + "4")

    D.chart(t("A two-factor VAE of the curve rediscovers level and slope", "Un VAE cu doi factori ai curbei redescoperă nivelul și panta"), "yc_latent_time", [
        t("\\textbf{Correlations} of the latent means with the level (mean of the 11 yields) and the slope (10 years minus 3 months): up to @{yc.vae_corr.max_level|.2f} and @{yc.vae_corr.max_slope|.2f} in absolute value.",
          "\\textbf{Corelațiile} mediilor latente cu nivelul (media celor 11 randamente) și cu panta (10 ani minus 3 luni): pînă la @{yc.vae_corr.max_level|.2f} și @{yc.vae_corr.max_slope|.2f} în valoare absolută"),
        t("\\textbf{Interpretable compression}: the latent path tracks the 2014--2021 low-rate era and the 2022 tightening without being told.",
          "\\textbf{Comprimare interpretabilă}: traiectoria latentă urmărește perioada de dobînzi scăzute 2014--2021 și înăsprirea din 2022, fără să i se spună")],
        height="0.48\\textheight", tryit=NB + "4", cite="(Svensson, 1994)")

    # ------------------------------------------------------------------ 6 practice
    D.section(t("Generative models in practice", "Modelele generative în practică"))
    D.frame(t("Evaluation criteria for synthetic financial data", "Criterii de evaluare a datelor financiare sintetice"), cols(
        it((t("\\textbf{Fidelity}", "\\textbf{Fidelitatea}"),
            t("marginals and tails (kurtosis, 1\\,\\% quantile), dependence across assets and over time (autocorrelation of $|r_t|$)",
              "distribuțiile marginale și cozile (aplatizarea, cuantila de 1\\,\\%), dependența între active și în timp (autocorelația lui $|r_t|$)")),
           (t("\\textbf{Diversity}", "\\textbf{Diversitatea}"),
            t("no mode collapse: the share of extreme days and regimes as in the data", "fără colaps al modurilor: ponderea zilelor extreme și a regimurilor ca în date")),
           (t("\\textbf{Utility}", "\\textbf{Utilitatea}"),
            t("train on synthetic, test on real (TSTR): a model fitted on synthetic data must work on real data", "antrenare pe date sintetice, test pe date reale (TSTR): un model estimat pe date sintetice trebuie să funcționeze pe date reale")), sep="5pt"),
        it((t("\\textbf{Privacy}", "\\textbf{Confidențialitatea}"),
            t("generators can memorise records: check the distance of each sample to its nearest real record", "generatoarele pot memora înregistrări: verificați distanța fiecărui eșantion pînă la cea mai apropiată înregistrare reală")),
           (t("\\textbf{Our scorecard}", "\\textbf{Fișa noastră de evaluare}"),
            t("GAN returns: tails right, clustering wrong", "randamentele GAN: cozile corecte, gruparea volatilității greșită"),
            t("DDPM curves: regimes and inversions right, Gaussian wrong", "curbele DDPM: regimurile și inversările corecte, normala greșită")), sep="5pt"), "0.49", "0.48") +
        D.bottom(t("never use synthetic data for a decision before it passes the same backtests as the model it feeds.",
                   "nu folosiți date sintetice într-o decizie înainte ca ele să treacă aceleași teste retrospective ca modelul pe care îl alimentează."), None),
        "footnotesize", cite="(Cont, 2001; Yoon et al., 2019)")

    D.frame(t("Choosing a generative model", "Alegerea unui model generativ"), table(
        "@{}p{0.22\\textwidth}p{0.17\\textwidth}p{0.17\\textwidth}p{0.17\\textwidth}p{0.17\\textwidth}@{}",
        [t("Need", "Nevoia"), "AE", "VAE", "GAN", "DDPM"],
        [[t("Anomaly score", "Scor de anomalie"), t("yes (error)", "da (eroarea)"), t("yes (ELBO)", "da (ELBO)"), t("indirect", "indirect"), t("yes (bound)", "da (marginea)")],
         [t("Sample quality", "Calitatea eșantioanelor"), "--", t("blurry", "neclare"), t("sharp", "clare"), t("best", "cea mai bună")],
         [t("Sampling speed", "Viteza generării"), "--", t("one pass", "o trecere"), t("one pass", "o trecere"), t("@{yc.T} passes", "@{yc.T} de treceri")],
         [t("Training stability", "Stabilitatea antrenării"), t("high", "mare"), t("high", "mare"), t("low", "mică"), t("high", "mare")],
         [t("Latent factors", "Factori latenți"), t("yes", "da"), t("yes, smooth", "da, netezi"), t("no inverse", "fără invers"), t("no", "nu")]],
        size="scriptsize", stretch="1.25") + "\n\\vspace{2mm}\n" + it(
        t("\\textbf{Rule of thumb}: autoencoders to monitor, VAEs to compress and explore, diffusion to simulate faithfully, GAN models when sampling must be fast.",
          "\\textbf{Regulă practică}: autoencodere pentru monitorizare, VAE pentru comprimare și explorare, difuzie pentru simulări fidele, GAN cînd generarea trebuie să fie rapidă")),
        "footnotesize", cite="(Bishop \\& Bishop, 2024, ch.\\ 16--20)")

    # ------------------------------------------------------------------ wrap-up
    D.section(t("Wrap-up", "Concluzii"))
    D.frame(t("Key takeaways", "Idei principale"), it(
        t("\\textbf{Autoencoders} model ``normal'': on card fraud PR-AUC @{ae.ap|.1f}\\,\\% without labels, against @{ae.ap_logit|.1f}\\,\\% for a classifier that uses labels.",
          "\\textbf{Autoencoderele} modelează ``normalul'': la fraudele cu carduri, PR-AUC @{ae.ap|.1f}\\,\\% fără etichete, față de @{ae.ap_logit|.1f}\\,\\% pentru un clasificator cu etichete"),
        t("\\textbf{VAEs} maximise the ELBO = reconstruction $-$ KL; the reparameterisation trick makes them trainable; $\\beta$ trades detail for a tidy latent space.",
          "\\textbf{VAE} maximizează ELBO = reconstrucție $-$ KL; trucul reparametrizării le face antrenabile; $\\beta$ schimbă detaliul pe un spațiu latent ordonat"),
        t("\\textbf{GANs} reproduce fat tails (kurtosis @{gan.kurt_gan|.2f} vs @{gan.kurt_real|.2f} real) and a usable 1\\,\\% VaR, but not volatility clustering when trained on single days.",
          "\\textbf{GAN} reproduc cozile groase (aplatizare @{gan.kurt_gan|.2f}, față de @{gan.kurt_real|.2f} real) și o valoare VaR de 1\\,\\% utilizabilă, dar nu și gruparea volatilității cînd sînt antrenate pe zile izolate"),
        t("\\textbf{Diffusion} learns to undo noise; on euro yield curves it keeps both rate regimes and the inversions that a Gaussian misses.",
          "\\textbf{Difuzia} învață să elimine zgomotul; pe curbele euro păstrează ambele regimuri de dobîndă și inversările pe care distribuția normală nu le vede"),
        sep="6pt"))

    D.frame(t("Research from our group: generative models for tail risk", "Cercetările grupului nostru: modele generative pentru riscul extrem"), it(
        t("\\textbf{Large language models (LLM) as risk forecasters: LLM-VaR and LLM-ES} (Pele, Bolovăneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann \\& Härdle, 2026): uses generative language models, "
          "prompted with return histories, to forecast Value-at-Risk and expected shortfall, evaluated with standard risk backtests.",
          "\\textbf{Modele mari de limbaj (LLM) ca instrumente de prognoză a riscului: LLM-VaR și LLM-ES} (Pele, Bolovăneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann \\& Härdle, 2026); lucrarea folosește modele de limbaj generative, "
          "cărora li se dau istoricele randamentelor, pentru prognoza valorii la risc și a pierderii așteptate în coadă, evaluate cu testele retrospective standard"),
        t("\\textbf{Foundation models for tail risk} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026): uses pretrained probabilistic forecasting models, which generate sample paths, "
          "without retraining and with conformal calibration, to forecast VaR and ES in Central and Eastern European markets.",
          "\\textbf{Modele fundaționale pentru riscul extrem} (Jheng, Tak, Găman, Mazurencu-Marinescu-Pele \\& Pele, 2026); lucrarea folosește modele probabiliste de prognoză preantrenate, care generează traiectorii, "
          "fără reantrenare și cu calibrare conformală, pentru prognoza VaR și ES pe piețele din Europa Centrală și de Est"),
        t("\\textbf{Link with this lecture}: both treat a generator of scenarios as a risk model, and judge it by the backtests used on this lecture's GAN.",
          "\\textbf{Legătura cu acest curs}: ambele lucrări tratează un generator de scenarii ca pe un model de risc și îl judecă prin testele retrospective folosite aici pentru GAN"),
        sep="5pt") + "\n\\vspace{1mm}{\\scriptsize " + t("Findings as stated in the papers; the numbers on the other slides come from our own computations.",
                                                     "Rezultatele de pe acest slide sînt cele din lucrări; cifrele de pe celelalte slide-uri provin din calculele noastre.") + "}",
        "footnotesize", cite=t("(Bucharest University of Economic Studies; Romanian Academy, Institute for Economic Forecasting)",
                               "(Academia de Studii Economice din București; Academia Română, Institutul de Prognoză Economică)"))

    D.frame(t("Further reading and resources", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Bishop \\& Bishop (2024)}, chapters 16--20: latent variables, GANs, flows, autoencoders, diffusion.",
          "\\textbf{Bishop \\& Bishop (2024)}, capitolele 16--20: variabile latente, GAN, fluxuri, autoencodere, difuzie"),
        t("\\textbf{Lilian Weng's notes} on diffusion models: ", "\\textbf{Notele lui Lilian Weng} despre modelele de difuzie: ") +
        "\\href{https://lilianweng.github.io/posts/2021-07-11-diffusion-models/}{lilianweng.github.io}.",
        t("\\textbf{Ho et al.\\ (2020)}, the DDPM paper, short and readable.", "\\textbf{Ho et al.\\ (2020)}, lucrarea DDPM, scurtă și ușor de citit"), sep="4pt"),
        it(t("\\textbf{This course}: code, data, notebooks --- ", "\\textbf{Acest curs}: cod, date, notebook-uri --- ") + "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Experiment yourself}: the notebook reproduces every chart and number of this lecture.",
             "\\textbf{Exercițiu practic}: notebook-ul reproduce toate graficele și cifrele acestui curs"), sep="4pt"), "0.52", "0.45"))

    # ------------------------------------------------------------------ quiz
    D.quiz_frame(t("autoencoder", "autoencoder"),
                 t("Why is the fraud autoencoder trained on legitimate transactions only?", "De ce se antrenează autoencoderul de fraudă numai pe tranzacții legitime?"),
                 t("It must learn what ``normal'' looks like; frauds are then rebuilt badly and get a high score (median error @{ae.median_ratio|.1f} times larger).",
                   "Trebuie să învețe cum arată ``normalul''; fraudele sînt apoi reconstruite prost și primesc un scor mare (eroarea mediană de @{ae.median_ratio|.1f} ori mai mare)"))
    D.quiz_frame("ELBO", t("Which two terms make up the ELBO, and what does each do?", "Care sînt cei doi termeni ai ELBO și ce face fiecare?"),
                 t("Expected log-likelihood of the data given the code (reconstruction) minus the KL divergence of the encoder from the prior (keeps codes near $\\mathcal N(0,I)$, so random codes decode to realistic data).",
                   "Log-verosimilitatea așteptată a datelor dat fiind codul (reconstrucția) minus divergența KL a codificatorului față de distribuția a priori (ține codurile lîngă $\\mathcal N(0,I)$, ca un cod aleator să producă date realiste)"))
    D.quiz_frame(t("GAN equilibrium", "echilibrul GAN"),
                 t("What does the optimal discriminator output once the generator matches the data?", "Ce produce discriminatorul optim cînd generatorul reproduce datele?"),
                 t("$D^{*}(\\mathbf x)=1/2$ everywhere, because $p_{data}=p_G$; its loss is then $2\\log2$.", "$D^{*}(\\mathbf x)=1/2$ peste tot, pentru că $p_{data}=p_G$; pierderea lui este atunci $2\\log2$"))
    D.quiz_frame(t("synthetic returns", "randamente sintetice"),
                 t("Our GAN gets the tails of S\\&P 500 returns right. Why is it still unfit for a volatility model?", "GAN-ul nostru reproduce corect cozile randamentelor S\\&P 500. De ce nu este bun pentru un model de volatilitate?"),
                 t("It draws days independently: the autocorrelation of $|r_t|$ is @{gan.acf1_gan|.2f} instead of @{gan.acf1_real|.2f}; volatility clustering needs a sequence generator.",
                   "Extrage zilele independent: autocorelația lui $|r_t|$ este @{gan.acf1_gan|.2f} în loc de @{gan.acf1_real|.2f}; gruparea volatilității cere un generator de secvențe"))
    D.quiz_frame(t("diffusion", "difuzia"),
                 t("What does the diffusion network predict during training?", "Ce prezice rețeaua de difuzie în timpul antrenării?"),
                 t("The Gaussian noise $\\boldsymbol\\varepsilon$ that was added to $\\mathbf x_0$ to obtain $\\mathbf x_t$, given $\\mathbf x_t$ and $t$; generation subtracts the predicted noise step by step.",
                   "Zgomotul gaussian $\\boldsymbol\\varepsilon$ adăugat lui $\\mathbf x_0$ pentru a obține $\\mathbf x_t$, dați fiind $\\mathbf x_t$ și $t$; generarea scade pas cu pas zgomotul prezis"))

    reference_frames(D, REFS, per_frame=9)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.parts.append("\\section*{" + t("Appendix", "Anexă") + "}\n")
    D.frame(t("Appendix A: derivation of the ELBO", "Anexa A: derivarea ELBO"), cols(
        "\\[ \\log p(\\mathbf x)=\\E_{q(\\mathbf z\\mid\\mathbf x)}\\Big[\\log\\frac{p(\\mathbf x,\\mathbf z)}{q(\\mathbf z\\mid\\mathbf x)}\\Big]"
        "+\\E_{q}\\Big[\\log\\frac{q(\\mathbf z\\mid\\mathbf x)}{p(\\mathbf z\\mid\\mathbf x)}\\Big] \\]"
        "\\[ =\\underbrace{\\E_{q}[\\log p(\\mathbf x\\mid\\mathbf z)]-\\mathrm{KL}(q\\|p(\\mathbf z))}_{\\text{ELBO}}+\\underbrace{\\mathrm{KL}(q\\|p(\\mathbf z\\mid\\mathbf x))}_{\\ge0} \\]",
        it(t("Multiply and divide by $q(\\mathbf z\\mid\\mathbf x)$ inside $\\log p(\\mathbf x)=\\log p(\\mathbf x,\\mathbf z)-\\log p(\\mathbf z\\mid\\mathbf x)$ and take the expectation under $q$.",
             "Se înmulțește și se împarte cu $q(\\mathbf z\\mid\\mathbf x)$ în $\\log p(\\mathbf x)=\\log p(\\mathbf x,\\mathbf z)-\\log p(\\mathbf z\\mid\\mathbf x)$ și se ia media după $q$"),
           t("The second KL is non-negative, so the ELBO is a lower bound; it is tight when $q$ equals the true posterior.",
             "A doua divergență KL este nenegativă, deci ELBO este o margine inferioară; este exactă cînd $q$ este egală cu distribuția a posteriori adevărată"),
           t("For Gaussians $\\mathcal N(\\boldsymbol\\mu,\\operatorname{diag}\\boldsymbol\\sigma^2)$ and $\\mathcal N(0,I)$ the KL is $\\tfrac12\\sum_j(\\mu_j^2+\\sigma_j^2-\\log\\sigma_j^2-1)$.",
             "Pentru gaussienele $\\mathcal N(\\boldsymbol\\mu,\\operatorname{diag}\\boldsymbol\\sigma^2)$ și $\\mathcal N(0,I)$, divergența KL este $\\tfrac12\\sum_j(\\mu_j^2+\\sigma_j^2-\\log\\sigma_j^2-1)$"), sep="3pt"), "0.55", "0.42"),
        "small", cite="(Kingma \\& Welling, 2014)")
    D.frame(t("Appendix B: the optimal discriminator", "Anexa B: discriminatorul optim"), cols(
        "\\[ V(D)=\\int\\big[p_{data}(\\mathbf x)\\log D(\\mathbf x)+p_G(\\mathbf x)\\log(1-D(\\mathbf x))\\big]d\\mathbf x \\]"
        "\\[ \\frac{\\partial}{\\partial D}:\\ \\frac{p_{data}}{D}-\\frac{p_G}{1-D}=0\\ \\Rightarrow\\ D^{*}=\\frac{p_{data}}{p_{data}+p_G} \\]",
        it(t("Maximise the integrand pointwise: $a\\log y+b\\log(1-y)$ peaks at $y=a/(a+b)$.",
             "Se maximizează integrandul punct cu punct: $a\\log y+b\\log(1-y)$ are maximul în $y=a/(a+b)$"),
           t("Plugging $D^{*}$ back gives $V=-\\log4+\\mathrm{KL}(p_{data}\\|m)+\\mathrm{KL}(p_G\\|m)$, $m=(p_{data}+p_G)/2$, i.e.\\ $-\\log 4+2\\,\\mathrm{JS}$.",
             "Înlocuind $D^{*}$ se obține $V=-\\log4+\\mathrm{KL}(p_{data}\\|m)+\\mathrm{KL}(p_G\\|m)$, $m=(p_{data}+p_G)/2$, adică $-\\log 4+2\\,\\mathrm{JS}$"), sep="3pt"), "0.55", "0.42"),
        "small", cite="(Goodfellow et al., 2014)")
    D.frame(t("Appendix C: settings of the four experiments", "Anexa C: setările celor patru experimente"), table(
        "@{}p{0.18\\textwidth}p{0.74\\textwidth}@{}",
        [t("Model", "Model"), t("Settings", "Setări")],
        [["AE", t("29--16--8--16--29, ReLU, Adam $10^{-3}$, batch 256, at most 60 epochs, patience 6 on validation error; seeds 1--3",
                  "29--16--8--16--29, ReLU, Adam $10^{-3}$, lot 256, cel mult 60 de epoci, răbdare 6 pe eroarea de validare; semințele 1--3")],
         ["VAE", t("784--400--200--$k$, $k\\in\\{2,16\\}$, Adam $10^{-3}$, batch 128, 20 epochs, Bernoulli decoder", "784--400--200--$k$, $k\\in\\{2,16\\}$, Adam $10^{-3}$, lot 128, 20 de epoci, decodor Bernoulli")],
         ["GAN", t("noise 8; two hidden layers of 64, LeakyReLU 0.2; Adam $2\\cdot10^{-4}$, $\\beta_1=0.5$; 6,000 steps of 256 days", "zgomot 8; două straturi ascunse de 64, LeakyReLU 0,2; Adam $2\\cdot10^{-4}$, $\\beta_1=0{,}5$; 6.000 de pași cu loturi de 256 de zile")],
         ["DDPM", t("$T=200$, linear $\\beta_t$; MLP 3$\\times$256 with time embedding; Adam $10^{-3}$, cosine decay; 12,000 steps of 512", "$T=200$, $\\beta_t$ liniar; MLP 3$\\times$256 cu reprezentarea timpului; Adam $10^{-3}$, scădere cosinus; 12.000 de pași cu loturi de 512")],
         [t("Hardware", "Echipament"), t("Apple-silicon laptop; total run @{runtime_min|.1f} minutes", "laptop cu procesor Apple; rularea completă @{runtime_min|.1f} minute")]],
        size="scriptsize"), "small")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 8, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
