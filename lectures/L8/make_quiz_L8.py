"""
make_quiz_L8.py -- writes assets/quizzes/l8.js, the quiz bank of Lecture 8 (EN + RO), with every number read from
L8_numbers.json. Format of lectures/README.md: correct = index 0-3 of the right option; the explanation of a wrong
answer never names a letter (the engine shuffles the options).
Run:  python3 lectures/L8/make_quiz_L8.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = json.load(open(os.path.join(HERE, "L8_numbers.json")))


def g(key):
    v = N
    for p in key.split("."):
        v = v[p]
    return v


def f(key, spec="", lang="en"):
    s = format(g(key), spec)
    if lang == "ro":
        s = s.replace(",", " ").replace(".", ",")
    return s.replace("-", "−") if s.startswith("-") else s


Q = []


def q(correct, en, ro):
    """en/ro = (title, text, [4 options], correctExplanation, incorrectExplanation)."""
    keys = ["title", "text", "options", "correctExplanation", "incorrectExplanation"]
    Q.append({"correct": correct, "en": dict(zip(keys, en)), "ro": dict(zip(keys, ro))})


E, R = (lambda k, s="": f(k, s, "en")), (lambda k, s="": f(k, s, "ro"))

q(1, ("Discriminative or generative",
      "A bank wants a model that can both score transactions and simulate plausible new ones for a stress test. Which object must the model estimate?",
      ["$p(y\\mid\\mathbf x)$, the probability of fraud given the transaction", "$p(\\mathbf x)$ or $p(\\mathbf x, y)$, the distribution of the transactions themselves",
       "Only the decision boundary between fraud and non-fraud", "The confusion matrix of the classifier"],
      "Simulation needs the distribution of the data, $p(\\mathbf x)$ or the joint $p(\\mathbf x,y)$; a discriminative model only gives $p(y\\mid\\mathbf x)$.",
      "Scoring alone needs only $p(y\\mid\\mathbf x)$, but simulating new transactions requires a model of how the transactions themselves arise: a generative model."),
  ("Discriminativ sau generativ",
   "O bancă vrea un model care să dea scoruri tranzacțiilor și să simuleze tranzacții noi, plauzibile, pentru un test de stres. Ce trebuie să estimeze modelul?",
   ["$p(y\\mid\\mathbf x)$, probabilitatea de fraudă dată fiind tranzacția", "$p(\\mathbf x)$ sau $p(\\mathbf x, y)$, distribuția tranzacțiilor înseși",
    "Doar granița de decizie dintre fraudă și non-fraudă", "Matricea de confuzie a clasificatorului"],
   "Simularea are nevoie de distribuția datelor, $p(\\mathbf x)$ sau distribuția comună $p(\\mathbf x,y)$; un model discriminativ dă doar $p(y\\mid\\mathbf x)$.",
   "Pentru scoruri ajunge $p(y\\mid\\mathbf x)$, dar simularea unor tranzacții noi cere un model al felului în care apar tranzacțiile: un model generativ."))
q(2, ("Linear autoencoder",
      "An autoencoder with linear encoder and decoder and a bottleneck of size $k$ is trained with squared error. What does it learn?",
      ["A non-linear manifold of the data", "The class labels of the inputs", "The same subspace as principal component analysis with $k$ components", "A density $p(\\mathbf x)$ with a closed form"],
      "With linear maps and squared error the optimal reconstruction projects on the span of the first $k$ principal components, so PCA is the natural benchmark.",
      "Without non-linear activations the autoencoder can only learn a linear projection, and the best one for squared error is the principal subspace of dimension $k$."),
  ("Autoencoder liniar",
   "Un autoencoder cu codificator și decodor liniare și un strat îngust de dimensiune $k$ este antrenat cu eroarea pătratică. Ce învață?",
   ["O varietate neliniară a datelor", "Etichetele claselor", "Același subspațiu ca analiza componentelor principale cu $k$ componente", "O densitate $p(\\mathbf x)$ în formă închisă"],
   "Cu funcții liniare și eroare pătratică, reconstrucția optimă proiectează pe subspațiul primelor $k$ componente principale; de aceea PCA este modelul de referință firesc.",
   "Fără activări neliniare, autoencoderul poate învăța doar o proiecție liniară, iar cea mai bună pentru eroarea pătratică este subspațiul principal de dimensiune $k$."))
q(0, ("Training data of a fraud autoencoder",
      "Why was the fraud autoencoder of the lecture trained on legitimate transactions only?",
      ["So that it learns normal behaviour and rebuilds frauds badly, giving them a high score", "Because frauds have missing values", "To make training faster", "Because the labels of frauds are confidential"],
      "The reconstruction error is the anomaly score: a model of normal behaviour rebuilds unusual transactions poorly. On the test part the median error of a fraud is " + E("ae.median_ratio", ".1f") + " times that of a legitimate one.",
      "The idea is to model normal behaviour only; frauds then stand out by a large reconstruction error, which is used as the anomaly score."),
  ("Datele de antrenare ale unui autoencoder de fraudă",
   "De ce a fost antrenat autoencoderul de fraudă din curs numai pe tranzacții legitime?",
   ["Ca să învețe comportamentul normal și să reconstruiască prost fraudele, care primesc un scor mare", "Pentru că fraudele au valori lipsă", "Ca antrenarea să fie mai rapidă", "Pentru că etichetele fraudelor sînt confidențiale"],
   "Eroarea de reconstrucție este scorul de anomalie: un model al comportamentului normal reconstruiește prost tranzacțiile neobișnuite. Pe partea de test, eroarea mediană a unei fraude este de " + R("ae.median_ratio", ".1f") + " ori mai mare decît a uneia legitime.",
   "Ideea este să modelăm doar comportamentul normal; fraudele ies apoi în evidență printr-o eroare de reconstrucție mare, folosită ca scor de anomalie."))
q(3, ("Autoencoder versus labels",
      "On the test part of the card data, which detector reached the highest PR-AUC?",
      ["The autoencoder, " + E("ae.ap", ".1f") + " %", "PCA with 8 components, " + E("ae.ap_pca", ".1f") + " %", "A random score", "The logistic regression trained with the labels, " + E("ae.ap_logit", ".1f") + " %"],
      "With confirmed frauds available, supervised learning wins clearly: " + E("ae.ap_logit", ".1f") + " % against " + E("ae.ap", ".1f") + " % for the autoencoder.",
      "The supervised logistic regression uses the fraud labels and reaches " + E("ae.ap_logit", ".1f") + " %; the autoencoder, without labels, reaches " + E("ae.ap", ".1f") + " %. It is a complement for unknown patterns."),
  ("Autoencoder față de etichete",
   "Pe partea de test a datelor cu carduri, care detector a atins cel mai mare PR-AUC?",
   ["Autoencoderul, " + R("ae.ap", ".1f") + " %", "PCA cu 8 componente, " + R("ae.ap_pca", ".1f") + " %", "Un scor aleator", "Regresia logistică antrenată cu etichete, " + R("ae.ap_logit", ".1f") + " %"],
   "Cînd există fraude confirmate, învățarea supervizată cîștigă clar: " + R("ae.ap_logit", ".1f") + " % față de " + R("ae.ap", ".1f") + " % pentru autoencoder.",
   "Regresia logistică folosește etichetele de fraudă și atinge " + R("ae.ap_logit", ".1f") + " %; autoencoderul, fără etichete, atinge " + R("ae.ap", ".1f") + " %. El este un complement pentru tiparele necunoscute."))
q(1, ("ROC-AUC on rare events",
      "All three fraud detectors have a ROC-AUC above 90 %, yet their PR-AUC differs a lot. Why?",
      ["ROC-AUC is computed on the training data", "With one fraud in several hundred transactions, the false-positive rate stays small even when most alerts are false",
       "PR-AUC ignores the frauds", "ROC-AUC is not defined for weighted data"],
      "The false-positive rate divides by the huge number of legitimate transactions, so many false alerts barely move it; precision counts them directly.",
      "On rare events the ROC curve is dominated by the many legitimate transactions; precision, and hence PR-AUC, shows how many alerts are really frauds."),
  ("ROC-AUC la evenimente rare",
   "Toți cei trei detectori de fraudă au ROC-AUC peste 90 %, dar PR-AUC diferă mult. De ce?",
   ["ROC-AUC se calculează pe datele de antrenare", "La o fraudă la cîteva sute de tranzacții, rata alarmelor false rămîne mică chiar cînd majoritatea alertelor sînt false",
    "PR-AUC ignoră fraudele", "ROC-AUC nu este definit pentru date ponderate"],
   "Rata alarmelor false se împarte la numărul uriaș de tranzacții legitime, deci multe alerte false abia o mișcă; precizia le numără direct.",
   "La evenimente rare, curba ROC este dominată de multele tranzacții legitime; precizia, deci PR-AUC, arată cîte alerte sînt cu adevărat fraude."))
q(2, ("Bottleneck size",
      "What happened to the test PR-AUC of the fraud autoencoder when the bottleneck grew from 8 to 24?",
      ["It rose, because a wider code keeps more information", "It stayed exactly the same", "It fell, from " + E("ae.bottleneck.8", ".1f") + " % to " + E("ae.bottleneck.24", ".1f") + " %", "Training failed to converge"],
      "A wider bottleneck lets the network copy unusual inputs too, so frauds are rebuilt better and stand out less.",
      "The best size was " + E("ae.best_k") + ": a code that is too wide reconstructs frauds as well, which lowers their anomaly score."),
  ("Dimensiunea stratului îngust",
   "Ce s-a întîmplat cu PR-AUC de test al autoencoderului cînd stratul îngust a crescut de la 8 la 24?",
   ["A crescut, pentru că un cod mai larg păstrează mai multă informație", "A rămas exact la fel", "A scăzut, de la " + R("ae.bottleneck.8", ".1f") + " % la " + R("ae.bottleneck.24", ".1f") + " %", "Antrenarea nu a convers"],
   "Un strat îngust mai larg lasă rețeaua să copieze și intrările neobișnuite, deci fraudele sînt reconstruite mai bine și ies mai puțin în evidență.",
   "Dimensiunea cea mai bună a fost " + R("ae.best_k") + ": un cod prea larg reconstruiește și fraudele, ceea ce le scade scorul de anomalie."))
q(0, ("Alert budget",
      "The fraud team reviews the 0.1 % highest scores of the test part (" + E("ae.n_alerts_top01") + " alerts). How many frauds does the autoencoder put in that list?",
      [E("ae.top01_frauds_ae"), E("ae.top01_frauds_logit"), E("ae.n_alerts_top01"), "0"],
      "The autoencoder places " + E("ae.top01_frauds_ae") + " frauds among the top alerts, " + E("ae.top01_recall_ae", ".1f") + " % of all test frauds.",
      "The autoencoder finds " + E("ae.top01_frauds_ae") + " frauds in that list; the supervised logistic regression finds " + E("ae.top01_frauds_logit") + "."),
  ("Bugetul de alerte",
   "Echipa antifraudă verifică primele 0,1 % scoruri din partea de test (" + R("ae.n_alerts_top01") + " de alerte). Cîte fraude pune autoencoderul în această listă?",
   [R("ae.top01_frauds_ae"), R("ae.top01_frauds_logit"), R("ae.n_alerts_top01"), "0"],
   "Autoencoderul plasează " + R("ae.top01_frauds_ae") + " fraude printre primele alerte, " + R("ae.top01_recall_ae", ".1f") + " % din toate fraudele de test.",
   "Autoencoderul găsește " + R("ae.top01_frauds_ae") + " fraude în această listă; regresia logistică supervizată găsește " + R("ae.top01_frauds_logit") + "."))
q(1, ("The ELBO",
      "Which expression is the evidence lower bound (ELBO) of a VAE?",
      ["$\\log p(\\mathbf x)+\\mathrm{KL}(q\\|p(\\mathbf z))$", "$\\mathbb E_{q(\\mathbf z\\mid\\mathbf x)}[\\log p(\\mathbf x\\mid\\mathbf z)]-\\mathrm{KL}(q(\\mathbf z\\mid\\mathbf x)\\|p(\\mathbf z))$",
       "$\\|\\mathbf x-\\hat{\\mathbf x}\\|^2$", "$\\mathbb E[\\log D(\\mathbf x)]+\\mathbb E[\\log(1-D(G(\\mathbf z)))]$"],
      "The ELBO is the expected log-likelihood of the data given the code minus the KL divergence of the encoder from the prior; it is below $\\log p(\\mathbf x)$ by the KL from the true posterior.",
      "The ELBO combines a reconstruction term with a KL penalty that keeps the codes near the prior; the squared error is the plain autoencoder loss and the min-max value belongs to the GAN."),
  ("ELBO",
   "Care expresie este marginea inferioară a verosimilității (ELBO) a unui VAE?",
   ["$\\log p(\\mathbf x)+\\mathrm{KL}(q\\|p(\\mathbf z))$", "$\\mathbb E_{q(\\mathbf z\\mid\\mathbf x)}[\\log p(\\mathbf x\\mid\\mathbf z)]-\\mathrm{KL}(q(\\mathbf z\\mid\\mathbf x)\\|p(\\mathbf z))$",
    "$\\|\\mathbf x-\\hat{\\mathbf x}\\|^2$", "$\\mathbb E[\\log D(\\mathbf x)]+\\mathbb E[\\log(1-D(G(\\mathbf z)))]$"],
   "ELBO este log-verosimilitatea așteptată a datelor dat fiind codul minus divergența KL a codificatorului față de distribuția a priori; este sub $\\log p(\\mathbf x)$ cu divergența KL față de distribuția a posteriori adevărată.",
   "ELBO combină un termen de reconstrucție cu o penalizare KL care ține codurile lîngă distribuția a priori; eroarea pătratică este pierderea autoencoderului simplu, iar valoarea min-max aparține GAN-ului."))
q(3, ("The ELBO as a lower bound",
      "Why is the ELBO never above $\\log p(\\mathbf x)$?",
      ["Because the reconstruction term is negative", "Because the prior is standard normal", "Because the decoder is a neural network",
       "Because the gap equals $\\mathrm{KL}(q(\\mathbf z\\mid\\mathbf x)\\|p(\\mathbf z\\mid\\mathbf x))$, which is never negative"],
      "$\\log p(\\mathbf x)=\\text{ELBO}+\\mathrm{KL}(q\\|p(\\mathbf z\\mid\\mathbf x))$ and a KL divergence is non-negative; the bound is tight when the encoder equals the true posterior.",
      "The decomposition $\\log p(\\mathbf x)=\\text{ELBO}+\\mathrm{KL}(q\\|p(\\mathbf z\\mid\\mathbf x))$ shows the gap: the error of the encoder, measured by a KL divergence, which cannot be negative."),
  ("ELBO ca margine inferioară",
   "De ce ELBO nu este niciodată peste $\\log p(\\mathbf x)$?",
   ["Pentru că termenul de reconstrucție este negativ", "Pentru că distribuția a priori este normală standard", "Pentru că decodorul este o rețea neuronală",
    "Pentru că diferența este $\\mathrm{KL}(q(\\mathbf z\\mid\\mathbf x)\\|p(\\mathbf z\\mid\\mathbf x))$, care nu este niciodată negativă"],
   "$\\log p(\\mathbf x)=\\text{ELBO}+\\mathrm{KL}(q\\|p(\\mathbf z\\mid\\mathbf x))$, iar o divergență KL este nenegativă; marginea este exactă cînd codificatorul coincide cu distribuția a posteriori adevărată.",
   "Descompunerea $\\log p(\\mathbf x)=\\text{ELBO}+\\mathrm{KL}(q\\|p(\\mathbf z\\mid\\mathbf x))$ arată diferența: eroarea codificatorului, măsurată printr-o divergență KL, care nu poate fi negativă."))
q(0, ("Reparameterisation",
      "What does the reparameterisation trick $\\mathbf z=\\boldsymbol\\mu+\\boldsymbol\\sigma\\odot\\boldsymbol\\varepsilon$ achieve?",
      ["It moves the randomness into $\\boldsymbol\\varepsilon$, so gradients can flow to $\\boldsymbol\\mu$ and $\\boldsymbol\\sigma$", "It removes the need for a prior",
       "It makes the KL term zero", "It turns the VAE into a GAN"],
      "Because $\\boldsymbol\\varepsilon\\sim\\mathcal N(0,I)$ does not depend on the parameters, the sample is a differentiable function of $\\boldsymbol\\mu$ and $\\boldsymbol\\sigma$.",
      "Sampling $\\mathbf z$ directly is not differentiable; writing it as a shift and scale of a fixed standard normal draw lets back-propagation reach the encoder."),
  ("Reparametrizarea",
   "Ce realizează trucul reparametrizării $\\mathbf z=\\boldsymbol\\mu+\\boldsymbol\\sigma\\odot\\boldsymbol\\varepsilon$?",
   ["Mută hazardul în $\\boldsymbol\\varepsilon$, astfel încît gradientul să ajungă la $\\boldsymbol\\mu$ și $\\boldsymbol\\sigma$", "Elimină nevoia unei distribuții a priori",
    "Face termenul KL nul", "Transformă VAE într-un GAN"],
   "Pentru că $\\boldsymbol\\varepsilon\\sim\\mathcal N(0,I)$ nu depinde de parametri, eșantionul este o funcție derivabilă de $\\boldsymbol\\mu$ și $\\boldsymbol\\sigma$.",
   "Extragerea directă a lui $\\mathbf z$ nu este derivabilă; scrisă ca translatare și scalare a unei extrageri normale standard fixe, permite propagarea înapoi pînă la codificator."))
q(2, ("KL of two Gaussians",
      "For an encoder $\\mathcal N(\\mu,\\sigma^2)$ in one dimension and the prior $\\mathcal N(0,1)$, which KL formula is correct?",
      ["$\\mu^2+\\sigma^2$", "$\\tfrac12(\\mu-\\sigma)^2$", "$\\tfrac12(\\mu^2+\\sigma^2-\\log\\sigma^2-1)$", "$\\log\\sigma-\\mu$"],
      "The closed form is $\\tfrac12(\\mu^2+\\sigma^2-\\log\\sigma^2-1)$, zero when $\\mu=0$ and $\\sigma=1$.",
      "The KL between $\\mathcal N(\\mu,\\sigma^2)$ and $\\mathcal N(0,1)$ is $\\tfrac12(\\mu^2+\\sigma^2-\\log\\sigma^2-1)$; it vanishes only when the encoder equals the prior."),
  ("KL între două gaussiene",
   "Pentru un codificator $\\mathcal N(\\mu,\\sigma^2)$ într-o dimensiune și distribuția a priori $\\mathcal N(0,1)$, care formulă KL este corectă?",
   ["$\\mu^2+\\sigma^2$", "$\\tfrac12(\\mu-\\sigma)^2$", "$\\tfrac12(\\mu^2+\\sigma^2-\\log\\sigma^2-1)$", "$\\log\\sigma-\\mu$"],
   "Forma închisă este $\\tfrac12(\\mu^2+\\sigma^2-\\log\\sigma^2-1)$, nulă cînd $\\mu=0$ și $\\sigma=1$.",
   "Divergența KL dintre $\\mathcal N(\\mu,\\sigma^2)$ și $\\mathcal N(0,1)$ este $\\tfrac12(\\mu^2+\\sigma^2-\\log\\sigma^2-1)$; se anulează numai cînd codificatorul coincide cu distribuția a priori."))
q(1, ("Two latent dimensions",
      "A VAE with two latent dimensions was trained on Fashion-MNIST without labels. How well do the two latent means separate the classes?",
      ["Not at all: accuracy close to 10 %", "A 15-nearest-neighbour classifier on them reaches " + E("vae.knn_acc_2d", ".1f") + " % accuracy", "Perfectly: 100 % accuracy", "They cannot be computed without labels"],
      "Without any label, the 2-D codes already sort the products: " + E("vae.knn_acc_2d", ".1f") + " % accuracy for a 15-nearest-neighbour classifier on the test images.",
      "The latent space organises the images by similarity even without labels; a simple neighbour classifier on the two coordinates reaches " + E("vae.knn_acc_2d", ".1f") + " %."),
  ("Două dimensiuni latente",
   "Un VAE cu două dimensiuni latente a fost antrenat pe Fashion-MNIST fără etichete. Cît de bine separă clasele cele două medii latente?",
   ["Deloc: acuratețe aproape de 10 %", "Un clasificator cu 15 vecini atinge pe ele o acuratețe de " + R("vae.knn_acc_2d", ".1f") + " %", "Perfect: acuratețe 100 %", "Nu se pot calcula fără etichete"],
   "Fără nicio etichetă, codurile în 2 dimensiuni sortează deja produsele: " + R("vae.knn_acc_2d", ".1f") + " % acuratețe pentru un clasificator cu 15 vecini pe imaginile de test.",
   "Spațiul latent organizează imaginile după asemănare chiar fără etichete; un clasificator simplu cu vecini pe cele două coordonate atinge " + R("vae.knn_acc_2d", ".1f") + " %."))
q(3, ("Blurry samples",
      "Why are VAE samples of clothing blurry?",
      ["Because the images were compressed before training", "Because the KL term is negative", "Because the latent space has too many dimensions",
       "Because the Gaussian posterior and the pixel-wise loss make the decoder average over plausible details"],
      "The decoder minimises an expected pixel-wise loss under uncertainty about the code, so it outputs an average of the plausible images.",
      "Blur is the price of the pixel-wise likelihood and of a Gaussian posterior: the optimal output under uncertainty is an average, and averages of sharp edges are soft."),
  ("Eșantioane neclare",
   "De ce sînt neclare eșantioanele de îmbrăcăminte generate de VAE?",
   ["Pentru că imaginile au fost comprimate înainte de antrenare", "Pentru că termenul KL este negativ", "Pentru că spațiul latent are prea multe dimensiuni",
    "Pentru că distribuția a posteriori gaussiană și pierderea pixel cu pixel fac decodorul să facă media detaliilor plauzibile"],
   "Decodorul minimizează o pierdere pixel cu pixel așteptată, sub incertitudinea codului, deci produce o medie a imaginilor plauzibile.",
   "Neclaritatea este prețul verosimilității pixel cu pixel și al distribuției a posteriori gaussiene: rezultatul optim sub incertitudine este o medie, iar media unor margini clare este estompată."))
q(0, ("beta-VAE",
      "Raising $\\beta$ from 0.5 to 4 in the VAE of the lecture changed the number of active latent dimensions how?",
      ["It fell from " + E("vae.beta.b05.active") + " to " + E("vae.beta.b4.active") + " of 16", "It rose from 4 to 16", "It stayed at 16", "It fell to 0"],
      "A stronger KL weight pushes more dimensions to the prior: " + E("vae.beta.b05.active") + " active dimensions with $\\beta=0.5$, " + E("vae.beta.b4.active") + " with $\\beta=4$.",
      "Larger $\\beta$ penalises information in the code, so fewer dimensions stay active (" + E("vae.beta.b4.active") + " of 16 with $\\beta=4$) and the reconstruction worsens."),
  ("beta-VAE",
   "Cum s-a schimbat numărul dimensiunilor latente active cînd $\\beta$ a crescut de la 0,5 la 4 în VAE-ul din curs?",
   ["A scăzut de la " + R("vae.beta.b05.active") + " la " + R("vae.beta.b4.active") + " din 16", "A crescut de la 4 la 16", "A rămas 16", "A scăzut la 0"],
   "O pondere KL mai mare împinge mai multe dimensiuni spre distribuția a priori: " + R("vae.beta.b05.active") + " dimensiuni active cu $\\beta=0{,}5$, " + R("vae.beta.b4.active") + " cu $\\beta=4$.",
   "Un $\\beta$ mai mare penalizează informația din cod, deci rămîn active mai puține dimensiuni (" + R("vae.beta.b4.active") + " din 16 cu $\\beta=4$), iar reconstrucția se înrăutățește."))
q(2, ("Optimal discriminator",
      "For a fixed generator, what is the optimal discriminator of a GAN?",
      ["$D^*(\\mathbf x)=p_G(\\mathbf x)/p_{data}(\\mathbf x)$", "$D^*(\\mathbf x)=1$ for every real point", "$D^*(\\mathbf x)=p_{data}(\\mathbf x)/(p_{data}(\\mathbf x)+p_G(\\mathbf x))$", "$D^*(\\mathbf x)=\\log p_{data}(\\mathbf x)$"],
      "Maximising $p_{data}\\log D+p_G\\log(1-D)$ pointwise gives $D^*=p_{data}/(p_{data}+p_G)$, which equals 1/2 when the generator matches the data.",
      "The integrand $a\\log y+b\\log(1-y)$ peaks at $y=a/(a+b)$; with $a=p_{data}$ and $b=p_G$ this gives the optimal discriminator."),
  ("Discriminatorul optim",
   "Pentru un generator fixat, care este discriminatorul optim al unui GAN?",
   ["$D^*(\\mathbf x)=p_G(\\mathbf x)/p_{data}(\\mathbf x)$", "$D^*(\\mathbf x)=1$ pentru orice punct real", "$D^*(\\mathbf x)=p_{data}(\\mathbf x)/(p_{data}(\\mathbf x)+p_G(\\mathbf x))$", "$D^*(\\mathbf x)=\\log p_{data}(\\mathbf x)$"],
   "Maximizînd punct cu punct $p_{data}\\log D+p_G\\log(1-D)$ se obține $D^*=p_{data}/(p_{data}+p_G)$, egal cu 1/2 cînd generatorul reproduce datele.",
   "Integrandul $a\\log y+b\\log(1-y)$ are maximul în $y=a/(a+b)$; cu $a=p_{data}$ și $b=p_G$ se obține discriminatorul optim."))
q(1, ("Equilibrium loss",
      "At the GAN equilibrium, around which value does the discriminator loss (real + fake cross-entropy) settle?",
      ["0", "$2\\log 2\\approx 1.39$", "1", "$\\log 4\\cdot 2\\approx 2.77$"],
      "With $D^*=1/2$ each of the two cross-entropy terms equals $\\log 2$, so the discriminator loss is $2\\log 2$.",
      "When the detective can only guess, $D=1/2$ on both real and fake data, and each cross-entropy term is $-\\log(1/2)=\\log 2$; their sum is $2\\log 2$."),
  ("Pierderea la echilibru",
   "La echilibrul GAN, în jurul cărei valori se stabilizează pierderea discriminatorului (entropia încrucișată pentru real + fals)?",
   ["0", "$2\\log 2\\approx 1{,}39$", "1", "$\\log 4\\cdot 2\\approx 2{,}77$"],
   "Cu $D^*=1/2$, fiecare dintre cei doi termeni de entropie încrucișată este $\\log 2$, deci pierderea discriminatorului este $2\\log 2$.",
   "Cînd detectivul doar ghicește, $D=1/2$ atît pe datele reale, cît și pe cele false, iar fiecare termen este $-\\log(1/2)=\\log 2$; suma lor este $2\\log 2$."))
q(3, ("Non-saturating loss",
      "Why does the generator minimise $-\\log D(G(\\mathbf z))$ instead of $\\log(1-D(G(\\mathbf z)))$?",
      ["It gives a likelihood for the generator", "It removes the need for a discriminator", "It makes the samples independent",
       "The original loss has a vanishing gradient when the discriminator rejects the fakes early in training"],
      "Early on $D(G(\\mathbf z))\\approx0$, where $\\log(1-D)$ is flat; $-\\log D$ is steep there and gives the generator a useful signal.",
      "Both losses have the same fixed point, but $\\log(1-D)$ saturates when the discriminator wins; the non-saturating version keeps the gradient large."),
  ("Pierderea nesaturată",
   "De ce minimizează generatorul $-\\log D(G(\\mathbf z))$ în loc de $\\log(1-D(G(\\mathbf z)))$?",
   ["Dă o verosimilitate pentru generator", "Elimină nevoia unui discriminator", "Face eșantioanele independente",
    "Pierderea originală are gradientul aproape nul cînd discriminatorul respinge falsurile la începutul antrenării"],
   "La început $D(G(\\mathbf z))\\approx0$, unde $\\log(1-D)$ este plat; $-\\log D$ este abrupt acolo și dă generatorului un semnal util.",
   "Ambele pierderi au același punct fix, dar $\\log(1-D)$ se saturează cînd discriminatorul cîștigă; varianta nesaturată păstrează gradientul mare."))
q(0, ("Fat tails",
      "The excess kurtosis of standardised S&P 500 daily returns 2005–2019 is " + E("gan.kurt_real", ".2f") + ". What did the GAN samples reach?",
      [E("gan.kurt_gan", ".2f") + " on average over 3 seeds", "0, like the Gaussian", E("gan.kurt_real", ".2f") + " exactly", "A negative value"],
      "The GAN learns fat tails: excess kurtosis " + E("gan.kurt_gan", ".2f") + ", far above the Gaussian 0, though below the real " + E("gan.kurt_real", ".2f") + ".",
      "The GAN samples have an excess kurtosis of " + E("gan.kurt_gan", ".2f") + ": much fatter tails than a Gaussian, without matching the extremes of the real data exactly."),
  ("Cozi groase",
   "Excesul de aplatizare al randamentelor zilnice standardizate S&P 500 2005–2019 este " + R("gan.kurt_real", ".2f") + ". Cît au atins eșantioanele GAN?",
   [R("gan.kurt_gan", ".2f") + " în medie pe 3 semințe", "0, ca normala", R("gan.kurt_real", ".2f") + " exact", "O valoare negativă"],
   "GAN-ul învață cozi groase: exces de aplatizare " + R("gan.kurt_gan", ".2f") + ", mult peste 0 al normalei, deși sub valoarea reală de " + R("gan.kurt_real", ".2f") + ".",
   "Eșantioanele GAN au un exces de aplatizare de " + R("gan.kurt_gan", ".2f") + ": cozi mult mai groase decît ale normalei, fără să reproducă exact extremele datelor reale."))
q(2, ("VaR backtest",
      "Fitted on 2005–2019, which 1 % VaR was exceeded most often on the S&P 500 from 2020 onwards?",
      ["Historical simulation", "The GAN quantile", "The Gaussian, exceeded on " + E("gan.viol_norm_pct", ".2f") + " % of the days", "None was ever exceeded"],
      "The Gaussian underestimates the tail (" + E("gan.var_norm_pct", ".2f") + " % against " + E("gan.var_real_pct", ".2f") + " % for historical simulation), so it is exceeded " + E("gan.viol_norm_pct", ".2f") + " % of the days instead of about 1 %.",
      "Historical simulation (" + E("gan.viol_hs_pct", ".2f") + " %) and the GAN (" + E("gan.viol_gan_pct", ".2f") + " %) are close to 1 %; the Gaussian is exceeded " + E("gan.viol_norm_pct", ".2f") + " % of the days."),
  ("Testul retrospectiv al VaR",
   "Estimată pe 2005–2019, care valoare VaR de 1 % a fost depășită cel mai des pe S&P 500 din 2020 încoace?",
   ["Simularea istorică", "Cuantila GAN", "Normala, depășită în " + R("gan.viol_norm_pct", ".2f") + " % dintre zile", "Niciuna nu a fost depășită"],
   "Normala subestimează coada (" + R("gan.var_norm_pct", ".2f") + " % față de " + R("gan.var_real_pct", ".2f") + " % pentru simularea istorică), deci este depășită în " + R("gan.viol_norm_pct", ".2f") + " % dintre zile în loc de circa 1 %.",
   "Simularea istorică (" + R("gan.viol_hs_pct", ".2f") + " %) și GAN-ul (" + R("gan.viol_gan_pct", ".2f") + " %) sînt aproape de 1 %; normala este depășită în " + R("gan.viol_norm_pct", ".2f") + " % dintre zile."))
q(1, ("Volatility clustering",
      "The autocorrelation of $|r_t|$ at lag 1 is " + E("gan.acf1_real", ".2f") + " for real returns and " + E("gan.acf1_gan", ".2f") + " for GAN samples. What explains the gap?",
      ["The GAN was trained on too few days", "The GAN generates single days independently, so it cannot reproduce volatility clustering", "The real returns are standardised",
       "The discriminator was too weak"],
      "A generator of independent days can match the marginal distribution but not the dependence over time; sequence models generate whole paths.",
      "Each GAN sample is an independent day, so its absolute values are uncorrelated over time; clustering needs a generator of sequences, such as TimeGAN."),
  ("Gruparea volatilității",
   "Autocorelația lui $|r_t|$ la decalajul 1 este " + R("gan.acf1_real", ".2f") + " pentru randamentele reale și " + R("gan.acf1_gan", ".2f") + " pentru eșantioanele GAN. Ce explică diferența?",
   ["GAN-ul a fost antrenat pe prea puține zile", "GAN-ul generează zile izolate, independent, deci nu poate reproduce gruparea volatilității", "Randamentele reale sînt standardizate",
    "Discriminatorul a fost prea slab"],
   "Un generator de zile independente poate reproduce distribuția marginală, dar nu și dependența în timp; modelele de secvențe generează traiectorii întregi.",
   "Fiecare eșantion GAN este o zi independentă, deci valorile absolute nu sînt corelate în timp; gruparea volatilității cere un generator de secvențe, precum TimeGAN."))
q(3, ("Mode collapse",
      "A GAN for stress tests produces only calm days and never a crash. What is this failure called and what is a standard remedy?",
      ["Overfitting; more dropout", "Vanishing gradients; a smaller learning rate", "Label leakage; a temporal split", "Mode collapse; a Wasserstein GAN with a Lipschitz critic"],
      "Mode collapse means the generator covers only part of the distribution; the Wasserstein distance gives useful gradients even when the supports do not overlap.",
      "When the generator concentrates on a few convincing modes, the usual remedy is to train with the Wasserstein distance, estimated by a 1-Lipschitz critic."),
  ("Colapsul modurilor",
   "Un GAN pentru teste de stres produce numai zile calme și niciodată o prăbușire. Cum se numește această problemă și care este un remediu standard?",
   ["Supraînvățare; mai mult dropout", "Gradienți care dispar; o rată de învățare mai mică", "Scurgerea etichetelor; o împărțire în timp", "Colapsul modurilor; un GAN Wasserstein cu un critic Lipschitz"],
   "Colapsul modurilor înseamnă că generatorul acoperă doar o parte a distribuției; distanța Wasserstein dă gradienți utili chiar cînd suporturile nu se suprapun.",
   "Cînd generatorul se concentrează pe cîteva moduri convingătoare, remediul obișnuit este antrenarea cu distanța Wasserstein, estimată de un critic 1-Lipschitz."))
q(0, ("Forward diffusion",
      "In a DDPM, how is $\\mathbf x_t$ obtained from the clean observation $\\mathbf x_0$?",
      ["$\\mathbf x_t=\\sqrt{\\bar\\alpha_t}\\,\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol\\varepsilon$ in one draw", "Only by running all $t$ steps one after the other",
       "By a learned encoder", "By adding uniform noise of width $t$"],
      "The Gaussian steps compose in closed form, $q(\\mathbf x_t\\mid\\mathbf x_0)=\\mathcal N(\\sqrt{\\bar\\alpha_t}\\mathbf x_0,(1-\\bar\\alpha_t)I)$, so training can sample any $t$ directly.",
      "The forward process has no parameters and its Gaussian steps combine: $\\mathbf x_t=\\sqrt{\\bar\\alpha_t}\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\boldsymbol\\varepsilon$ with $\\bar\\alpha_t=\\prod_s(1-\\beta_s)$."),
  ("Difuzia înainte",
   "Într-un DDPM, cum se obține $\\mathbf x_t$ din observația curată $\\mathbf x_0$?",
   ["$\\mathbf x_t=\\sqrt{\\bar\\alpha_t}\\,\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol\\varepsilon$, dintr-o singură extragere", "Numai parcurgînd toți cei $t$ pași, unul după altul",
    "Printr-un codificator învățat", "Adăugînd un zgomot uniform de lățime $t$"],
   "Pașii gaussieni se compun în formă închisă, $q(\\mathbf x_t\\mid\\mathbf x_0)=\\mathcal N(\\sqrt{\\bar\\alpha_t}\\mathbf x_0,(1-\\bar\\alpha_t)I)$, deci la antrenare se poate extrage direct orice $t$.",
   "Procesul înainte nu are parametri, iar pașii lui gaussieni se combină: $\\mathbf x_t=\\sqrt{\\bar\\alpha_t}\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\boldsymbol\\varepsilon$, cu $\\bar\\alpha_t=\\prod_s(1-\\beta_s)$."))
q(2, ("Target of the denoising network",
      "What is the training target of the DDPM network $\\boldsymbol\\varepsilon_\\theta(\\mathbf x_t,t)$?",
      ["The clean curve $\\mathbf x_0$ only", "The label of the curve", "The Gaussian noise $\\boldsymbol\\varepsilon$ added to obtain $\\mathbf x_t$", "The step $t$"],
      "The loss is $\\|\\boldsymbol\\varepsilon-\\boldsymbol\\varepsilon_\\theta(\\mathbf x_t,t)\\|^2$, a simplified reweighted ELBO; generation subtracts the predicted noise step by step.",
      "The network sees the noisy input and the step and predicts the noise; knowing the noise is equivalent to knowing the clean input, and it gives a stable regression target."),
  ("Ținta rețelei de eliminare a zgomotului",
   "Care este ținta de antrenare a rețelei DDPM $\\boldsymbol\\varepsilon_\\theta(\\mathbf x_t,t)$?",
   ["Doar curba curată $\\mathbf x_0$", "Eticheta curbei", "Zgomotul gaussian $\\boldsymbol\\varepsilon$ adăugat pentru a obține $\\mathbf x_t$", "Pasul $t$"],
   "Pierderea este $\\|\\boldsymbol\\varepsilon-\\boldsymbol\\varepsilon_\\theta(\\mathbf x_t,t)\\|^2$, un ELBO simplificat și reponderat; generarea scade pas cu pas zgomotul prezis.",
   "Rețeaua vede intrarea cu zgomot și pasul și prezice zgomotul; a cunoaște zgomotul echivalează cu a cunoaște intrarea curată și dă o țintă de regresie stabilă."))
q(1, ("Two regimes of euro rates",
      "On the real euro-area curves the 10-year yield was negative on " + E("yc.pct_neg10_real", ".1f") + " % of the days. Which generator came closest to the real distribution of the 10-year yield (Wasserstein distance)?",
      ["The multivariate Gaussian, " + E("yc.gauss.w_y10", ".3f"), "The diffusion model, " + E("yc.ddpm.w_y10", ".3f"), "The two-factor VAE, " + E("yc.vae.w_y10", ".3f"), "All three were identical"],
      "The diffusion model reproduces both the low-rate and the normal-rate regime; its Wasserstein distance is " + E("yc.ddpm.w_y10", ".3f") + " against " + E("yc.gauss.w_y10", ".3f") + " for the Gaussian.",
      "A single Gaussian cannot be bimodal; the diffusion model has the smallest distance (" + E("yc.ddpm.w_y10", ".3f") + "), the VAE comes second (" + E("yc.vae.w_y10", ".3f") + ")."),
  ("Cele două regimuri ale dobînzilor euro",
   "Pe curbele reale din zona euro, randamentul la 10 ani a fost negativ în " + R("yc.pct_neg10_real", ".1f") + " % dintre zile. Care generator s-a apropiat cel mai mult de distribuția reală a randamentului la 10 ani (distanța Wasserstein)?",
   ["Normala multivariată, " + R("yc.gauss.w_y10", ".3f"), "Modelul de difuzie, " + R("yc.ddpm.w_y10", ".3f"), "VAE-ul cu doi factori, " + R("yc.vae.w_y10", ".3f"), "Toate trei au fost identice"],
   "Modelul de difuzie reproduce atît regimul de dobînzi scăzute, cît și pe cel de dobînzi normale; distanța lui Wasserstein este " + R("yc.ddpm.w_y10", ".3f") + ", față de " + R("yc.gauss.w_y10", ".3f") + " pentru normală.",
   "O singură gaussiană nu poate fi bimodală; modelul de difuzie are distanța cea mai mică (" + R("yc.ddpm.w_y10", ".3f") + "), iar VAE-ul este al doilea (" + R("yc.vae.w_y10", ".3f") + ")."))
q(0, ("Inverted curves",
      "Inverted curves (10 years below 3 months) occurred on " + E("yc.pct_inverted_real", ".1f") + " % of the real days. Which share of the VAE samples was inverted?",
      [E("yc.vae.pct_inverted", ".1f") + " %", E("yc.pct_inverted_real", ".1f") + " %", E("yc.gauss.pct_inverted", ".1f") + " %", "50 %"],
      "The two-factor VAE almost never produces inversions (" + E("yc.vae.pct_inverted", ".1f") + " %): its smooth decoder misses a rare but important shape, which the diffusion model keeps (" + E("yc.ddpm.pct_inverted", ".1f") + " %).",
      "Only " + E("yc.vae.pct_inverted", ".1f") + " % of the VAE curves are inverted, against " + E("yc.ddpm.pct_inverted", ".1f") + " % for diffusion and " + E("yc.pct_inverted_real", ".1f") + " % in the data."),
  ("Curbe inversate",
   "Curbele inversate (10 ani sub 3 luni) au apărut în " + R("yc.pct_inverted_real", ".1f") + " % dintre zilele reale. Ce pondere din eșantioanele VAE a fost inversată?",
   [R("yc.vae.pct_inverted", ".1f") + " %", R("yc.pct_inverted_real", ".1f") + " %", R("yc.gauss.pct_inverted", ".1f") + " %", "50 %"],
   "VAE-ul cu doi factori aproape nu produce inversări (" + R("yc.vae.pct_inverted", ".1f") + " %): decodorul lui neted ratează o formă rară, dar importantă, pe care modelul de difuzie o păstrează (" + R("yc.ddpm.pct_inverted", ".1f") + " %).",
   "Doar " + R("yc.vae.pct_inverted", ".1f") + " % dintre curbele VAE sînt inversate, față de " + R("yc.ddpm.pct_inverted", ".1f") + " % pentru difuzie și " + R("yc.pct_inverted_real", ".1f") + " % în date."))
q(3, ("Latent factors of the curve",
      "The two latent means of the yield-curve VAE were compared with the level and the slope of the curve. What did they show?",
      ["No relation with either", "They are both equal to the 3-month yield", "They only follow calendar time", "Correlations up to " + E("yc.vae_corr.max_level", ".2f") + " with the level and " + E("yc.vae_corr.max_slope", ".2f") + " with the slope, in absolute value"],
      "Without being told, the VAE rediscovers the classic factors: one latent mean follows the level and the other is related to the slope.",
      "The latent coordinates are strongly correlated with level (" + E("yc.vae_corr.max_level", ".2f") + ") and slope (" + E("yc.vae_corr.max_slope", ".2f") + "): interpretable compression of 11 yields into 2 numbers."),
  ("Factorii latenți ai curbei",
   "Cele două medii latente ale VAE-ului pentru curba randamentelor au fost comparate cu nivelul și panta curbei. Ce au arătat?",
   ["Nicio legătură cu niciunul", "Ambele sînt egale cu randamentul la 3 luni", "Urmăresc doar timpul calendaristic", "Corelații de pînă la " + R("yc.vae_corr.max_level", ".2f") + " cu nivelul și " + R("yc.vae_corr.max_slope", ".2f") + " cu panta, în valoare absolută"],
   "Fără să i se spună, VAE-ul redescoperă factorii clasici: o medie latentă urmărește nivelul, iar cealaltă este legată de pantă.",
   "Coordonatele latente sînt puternic corelate cu nivelul (" + R("yc.vae_corr.max_level", ".2f") + ") și cu panta (" + R("yc.vae_corr.max_slope", ".2f") + "): o comprimare interpretabilă a 11 randamente în 2 numere."))
q(1, ("Sampling cost",
      "Our diffusion model uses " + E("yc.T") + " noise steps. How many network evaluations does it need to generate one curve?",
      ["One", E("yc.T") + ", one per reverse step", "Two: encoder and decoder", "As many as the training steps"],
      "Generation runs the reverse chain from $t=T$ to $t=1$, one network evaluation per step; this is the price of the faithful samples.",
      "Unlike a GAN or a VAE, which generate in one pass, a DDPM walks back through all " + E("yc.T") + " steps, one network evaluation each."),
  ("Costul generării",
   "Modelul nostru de difuzie folosește " + R("yc.T") + " de pași de zgomot. Cîte evaluări ale rețelei sînt necesare pentru a genera o curbă?",
   ["Una", R("yc.T") + ", cîte una pentru fiecare pas înapoi", "Două: codificatorul și decodorul", "Cîți pași de antrenare au fost"],
   "Generarea parcurge lanțul invers de la $t=T$ la $t=1$, cu o evaluare a rețelei la fiecare pas; acesta este prețul eșantioanelor fidele.",
   "Spre deosebire de un GAN sau un VAE, care generează într-o singură trecere, un DDPM parcurge înapoi toți cei " + R("yc.T") + " de pași, cu cîte o evaluare a rețelei."))
q(2, ("Judging synthetic data",
      "A bank wants to share synthetic transaction data with a fintech partner. Which check addresses privacy?",
      ["The kurtosis of the amounts", "The accuracy of a classifier trained on the synthetic data", "The distance of each synthetic record to its nearest real record", "The number of training epochs"],
      "Generators can memorise records; a synthetic row that is almost identical to a real one leaks client information.",
      "Kurtosis checks fidelity and train-on-synthetic, test-on-real checks utility; privacy is checked by how close the synthetic records are to real ones."),
  ("Evaluarea datelor sintetice",
   "O bancă vrea să partajeze date sintetice despre tranzacții cu un partener fintech. Ce verificare privește confidențialitatea?",
   ["Aplatizarea sumelor", "Acuratețea unui clasificator antrenat pe datele sintetice", "Distanța fiecărei înregistrări sintetice pînă la cea mai apropiată înregistrare reală", "Numărul de epoci de antrenare"],
   "Generatoarele pot memora înregistrări; un rînd sintetic aproape identic cu unul real dezvăluie informații despre client.",
   "Aplatizarea verifică fidelitatea, iar antrenarea pe date sintetice cu test pe date reale verifică utilitatea; confidențialitatea se verifică prin cît de aproape sînt înregistrările sintetice de cele reale."))
q(0, ("TSTR",
      "What does the train-on-synthetic, test-on-real (TSTR) check measure?",
      ["Whether a model fitted on synthetic data works on real data", "Whether the generator trains faster than the discriminator", "Whether the latent space is two-dimensional", "Whether the data are stationary"],
      "TSTR measures utility: synthetic data are useful if models trained on them perform well on real data.",
      "TSTR fits a downstream model on synthetic data and evaluates it on held-out real data; a large drop means the generator missed what matters for the task."),
  ("TSTR",
   "Ce măsoară verificarea antrenare pe date sintetice, test pe date reale (TSTR)?",
   ["Dacă un model estimat pe date sintetice funcționează pe date reale", "Dacă generatorul se antrenează mai repede decît discriminatorul", "Dacă spațiul latent are două dimensiuni", "Dacă datele sînt staționare"],
   "TSTR măsoară utilitatea: datele sintetice sînt utile dacă modelele antrenate pe ele funcționează bine pe date reale.",
   "TSTR estimează un model pe date sintetice și îl evaluează pe date reale păstrate deoparte; o scădere mare arată că generatorul a ratat ce contează pentru sarcină."))
q(3, ("Choosing a family",
      "A risk team needs thousands of yield-curve scenarios overnight, faithful to both rate regimes; speed is secondary. Which family fits best?",
      ["A plain autoencoder", "A GAN on single days", "A Gaussian copula", "A diffusion model"],
      "Diffusion gave the most faithful curves (both regimes, realistic inversions); its many sampling steps are affordable for an overnight batch.",
      "The plain autoencoder is not a generator and a single Gaussian cannot be bimodal; the diffusion model reproduced both regimes on our data, at the cost of slower sampling."),
  ("Alegerea familiei",
   "O echipă de risc are nevoie peste noapte de mii de scenarii ale curbei randamentelor, fidele ambelor regimuri de dobîndă; viteza este secundară. Ce familie se potrivește cel mai bine?",
   ["Un autoencoder simplu", "Un GAN pe zile izolate", "O copulă gaussiană", "Un model de difuzie"],
   "Difuzia a dat cele mai fidele curbe (ambele regimuri, inversări realiste); pașii ei numeroși de generare sînt acceptabili pentru o rulare peste noapte.",
   "Autoencoderul simplu nu este generator, iar o singură gaussiană nu poate fi bimodală; modelul de difuzie a reprodus ambele regimuri pe datele noastre, cu prețul unei generări mai lente."))
q(1, ("Kolmogorov–Smirnov distance",
      "The Kolmogorov–Smirnov distance between generated and real standardised S&P 500 returns was " + E("gan.ks_gan", ".3f") + " for the GAN. What was it for the fitted Gaussian?",
      [E("gan.ks_gan", ".3f"), E("gan.ks_norm", ".3f"), "0", "1"],
      "The Gaussian is about " + E("gan.ks_norm", ".3f") + " away: its peak is too low and its tails too thin for daily returns.",
      "The fitted Gaussian has a distance of " + E("gan.ks_norm", ".3f") + ", several times that of the GAN, because it misses both the sharp centre and the fat tails."),
  ("Distanța Kolmogorov–Smirnov",
   "Distanța Kolmogorov–Smirnov dintre randamentele S&P 500 standardizate generate și cele reale a fost " + R("gan.ks_gan", ".3f") + " pentru GAN. Cît a fost pentru normala estimată?",
   [R("gan.ks_gan", ".3f"), R("gan.ks_norm", ".3f"), "0", "1"],
   "Normala este la circa " + R("gan.ks_norm", ".3f") + ": vîrful ei este prea jos, iar cozile prea subțiri pentru randamentele zilnice.",
   "Normala estimată are o distanță de " + R("gan.ks_norm", ".3f") + ", de cîteva ori mai mare decît a GAN-ului, pentru că ratează atît centrul ascuțit, cît și cozile groase."))
q(2, ("Latent-variable integral",
      "Why can a VAE not maximise $\\log p(\\mathbf x)=\\log\\int p(\\mathbf x\\mid\\mathbf z)p(\\mathbf z)\\,d\\mathbf z$ directly?",
      ["Because $p(\\mathbf z)$ is unknown", "Because the data are discrete", "Because with a neural decoder the integral has no closed form", "Because the encoder has too many parameters"],
      "Only in the linear-Gaussian case (probabilistic PCA) is the integral analytic; with a neural decoder one maximises the ELBO instead.",
      "The prior is chosen ($\\mathcal N(0,I)$); the difficulty is the integral over all codes through a non-linear decoder, which is why a lower bound is used."),
  ("Integrala modelului latent",
   "De ce nu poate un VAE să maximizeze direct $\\log p(\\mathbf x)=\\log\\int p(\\mathbf x\\mid\\mathbf z)p(\\mathbf z)\\,d\\mathbf z$?",
   ["Pentru că $p(\\mathbf z)$ este necunoscută", "Pentru că datele sînt discrete", "Pentru că, cu un decodor neuronal, integrala nu are formă închisă", "Pentru că codificatorul are prea mulți parametri"],
   "Doar în cazul liniar-gaussian (PCA probabilistă) integrala este analitică; cu un decodor neuronal se maximizează ELBO.",
   "Distribuția a priori este aleasă ($\\mathcal N(0,I)$); dificultatea este integrala pe toate codurile printr-un decodor neliniar, de aceea se folosește o margine inferioară."))
q(0, ("Time split for fraud",
      "Why were the card transactions split in time (first 60 % training, last 20 % test) rather than at random?",
      ["Because the model will be used on future transactions, so the test must come after the training data", "Because random splits are not allowed for weighted data",
       "To have more frauds in the training part", "Because the autoencoder needs sorted inputs"],
      "A temporal split mimics deployment: the detector is judged on transactions that come after everything it learnt from.",
      "Fraud patterns change over time; a random split would let the test resemble the training period too closely and overstate the performance."),
  ("Împărțirea în timp pentru fraudă",
   "De ce au fost împărțite tranzacțiile cu carduri în timp (primele 60 % antrenare, ultimele 20 % test) și nu aleator?",
   ["Pentru că modelul va fi folosit pe tranzacții viitoare, deci testul trebuie să vină după datele de antrenare", "Pentru că împărțirea aleatoare nu este permisă pentru date ponderate",
    "Ca să fie mai multe fraude în partea de antrenare", "Pentru că autoencoderul are nevoie de intrări sortate"],
   "O împărțire în timp imită utilizarea reală: detectorul este judecat pe tranzacții care vin după tot ce a învățat.",
   "Tiparele de fraudă se schimbă în timp; o împărțire aleatoare ar face testul prea asemănător cu perioada de antrenare și ar supraestima performanța."))
q(3, ("Generators as risk models",
      "The group papers on LLM-VaR and on foundation models for tail risk share an idea with this lecture. Which one?",
      ["Both train a GAN on daily returns", "Both use autoencoders for fraud", "Both replace backtests by the likelihood", "A generator of scenarios is treated as a risk model and judged by risk backtests"],
      "Language models and pretrained probabilistic forecasters produce scenarios or quantiles of returns; like our GAN, they are evaluated by VaR and ES backtests.",
      "The common thread is evaluation: whatever generates the scenarios, it is judged by the backtests of the risk measures it implies."),
  ("Generatoarele ca modele de risc",
   "Lucrările grupului despre LLM-VaR și despre modelele fundaționale pentru riscul extrem au o idee comună cu acest curs. Care?",
   ["Ambele antrenează un GAN pe randamente zilnice", "Ambele folosesc autoencodere pentru fraudă", "Ambele înlocuiesc testele retrospective cu verosimilitatea", "Un generator de scenarii este tratat ca model de risc și judecat prin teste retrospective"],
   "Modelele de limbaj și modelele probabiliste preantrenate produc scenarii sau cuantile ale randamentelor; ca GAN-ul nostru, sînt evaluate prin testele retrospective VaR și ES.",
   "Firul comun este evaluarea: oricine generează scenariile, este judecat prin testele retrospective ale măsurilor de risc pe care le implică."))

if __name__ == "__main__":
    assert len(Q) >= 30, len(Q)
    for x in Q:
        assert x["correct"] in (0, 1, 2, 3) and len(x["en"]["options"]) == 4 and len(x["ro"]["options"]) == 4
    head = ("// ============================================================\n"
            "// Quiz bank for lecture id 'l8': Generative models: autoencoders, VAE, GAN, diffusion (EN + RO)\n"
            "// GENERATED by lectures/L8/make_quiz_L8.py from L8_numbers.json; do not edit by hand.\n"
            "// correct = index (0-3) of the right option in the original order; incorrectExplanation never names a letter.\n"
            "// ============================================================\n")
    js = head + "window.NN_DATA.quizzes['l8'] = " + json.dumps({"draw": 20, "questions": Q}, ensure_ascii=False, indent=1) + ";\n"
    os.makedirs(os.path.join(ROOT, "assets", "quizzes"), exist_ok=True)
    with open(os.path.join(ROOT, "assets", "quizzes", "l8.js"), "w", encoding="utf-8") as fh:
        fh.write(js)
    print("assets/quizzes/l8.js:", len(Q), "questions")
