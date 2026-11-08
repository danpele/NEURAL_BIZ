"""Build the interactive page of Lecture 8 (Generative models: autoencoders, VAE, GAN, diffusion): interactive/L8.html (EN) and
interactive/L8_ro.html (RO).

Inputs (interactive/L8_data.js) come from the lecture's own results (lectures/L8/results, lectures/L8/cache, written by
analysis_L8.py) and from data/clean (S&P 500 returns, euro-area AAA yield curve); the page logic is in interactive/L8.js.
Run: OMP_NUM_THREADS=1 python3 lectures/L8/build_interactive_L8.py [--no-data]
"""
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, play, charts, rnd, DATA  # noqa: E402

K = 8

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('fraud-alerts', 'ae_error_dist', 'ae', 'fa'),
          ('vae-latent', 'vae_latent', 'vae', 'vl'),
          ('gan-tails', 'gan_density', 'gan', 'gt'),
          ('gan-acf', 'gan_acf', 'gan', 'ga'),
          ('diffusion-forward', 'diffusion_forward', 'diff', 'df'),
          ('yc-scenarios', 'yc_density', 'diff', 'yc')]

SECTIONS = {
 'en': dict(ae='Autoencoders: anomaly detection for card fraud', vae='Variational autoencoders',
            gan='Generative adversarial networks', diff='Diffusion models'),
 'ro': dict(ae='Autoencodere: detectarea anomaliilor la fraudele cu carduri', vae='Autoencodere variaționale',
            gan='Rețele generative adversariale', diff='Modele de difuzie'),
}

CLASSES = {'en': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
           'ro': ['Tricou', 'Pantaloni', 'Pulover', 'Rochie', 'Palton', 'Sandală', 'Cămașă', 'Pantof sport', 'Geantă', 'Gheată']}

BUDGETS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

L = {
 'en': dict(h1='Lecture 8 · Generative models: autoencoders, VAE, GAN, diffusion',
   note='Four families of generative models, each on real data: the autoencoder (AE), the variational autoencoder (VAE), the generative '
        'adversarial network (GAN) and the denoising diffusion probabilistic model (DDPM). The slide of the lecture is on the left, the same '
        'figure on the right, recomputed in your browser from the results of the lecture when you move a control. The ▶ button moves the main '
        'parameter step by step. Under each figure: a readout, a question for the room and the Quantlet with the code of the chart.',
   src='Card fraud: ULB Machine Learning Group and Worldline (OpenML 1597), two days of September 2013, test = last 20% of the time span, '
       'legitimate transactions weighted to the full data. Images: Fashion-MNIST, Zalando Research (Xiao et al., 2017), 10,000 test images. '
       'Returns: S&P 500 daily closes, Yahoo Finance, training 4 January 2005 – 31 December 2019, test 2 January 2020 – 30 October 2026. '
       'Yield curves: European Central Bank (ECB), euro-area AAA government bonds, 6 September 2004 – 30 October 2026, business days. '
       'Model outputs: own estimates of the lecture (analysis_L8.py).'),
 'ro': dict(h1='Cursul 8 · Modele generative: autoencodere, VAE, GAN, difuzie',
   note='Patru familii de modele generative, fiecare pe date reale: autoencoderul (AE), autoencoderul variațional (VAE), rețeaua generativă '
        'adversarială (GAN) și modelul probabilistic de difuzie cu eliminarea zgomotului (DDPM). În stînga este slide-ul din curs, în dreapta '
        'aceeași figură, recalculată în browser din rezultatele cursului cînd modificați un control. Butonul ▶ modifică parametrul principal pas '
        'cu pas. Sub fiecare figură: o sinteză numerică, o întrebare pentru sală și Quantlet-ul cu codul figurii.',
   src='Fraude cu carduri: ULB Machine Learning Group și Worldline (OpenML 1597), două zile din septembrie 2013, test = ultimele 20% din '
       'interval, tranzacțiile legitime ponderate la totalul datelor. Imagini: Fashion-MNIST, Zalando Research (Xiao et al., 2017), 10.000 de '
       'imagini de test. Randamente: prețurile zilnice de închidere ale S&P 500, Yahoo Finance, antrenare 4 ianuarie 2005 – 31 decembrie 2019, '
       'test 2 ianuarie 2020 – 30 octombrie 2026. Curbele randamentelor: Banca Centrală Europeană (BCE), obligațiuni guvernamentale AAA din zona '
       'euro, 6 septembrie 2004 – 30 octombrie 2026, zile lucrătoare. Rezultatele modelelor: estimări proprii din curs (analysis_L8.py).'),
}


def widgets(lang):
    ro = lang == 'ro'
    cls = [(i, c) for i, c in enumerate(CLASSES[lang])]
    return {
     'fraud-alerts': (select('fa-s', 'Scorul de anomalie' if ro else 'Anomaly score',
                             [('ae', 'Autoencoder (fără etichete)' if ro else 'Autoencoder (no labels)'),
                              ('pca', 'Analiza componentelor principale, PCA (fără etichete)' if ro else 'Principal component analysis, PCA (no labels)'),
                              ('logit', 'Regresie logistică (cu etichete)' if ro else 'Logistic regression (with labels)')], 'ae')
                      + slider('fa-b', 'Bugetul de alerte: ponderea tranzacțiilor verificate' if ro else 'Alert budget: share of transactions checked',
                               0, len(BUDGETS) - 1, 1, 2) + play('fa'),
                      charts('fa-hist', 'fa-pr')),
     'vae-latent': (slider('vl-a', 'Coordonata latentă z₁' if ro else 'Latent coordinate z₁', -2.1, 2.4, 0.3, 0)
                    + slider('vl-b', 'Coordonata latentă z₂' if ro else 'Latent coordinate z₂', -3.0, 2.7, 0.3, 0) + play('vl')
                    + toggles('vl-c', 'Clasele afișate pe hartă' if ro else 'Classes shown on the map', cls),
                    charts('vl-map', 'vl-img')),
     'gan-tails': (slider('gt-q', 'Probabilitatea din coadă a VaR (%)' if ro else 'Tail probability of the VaR (%)', 0.5, 5, 0.5, 1)
                   + select('gt-v', 'Vederea' if ro else 'View', [('all', 'Toată distribuția' if ro else 'Whole distribution'),
                                                                  ('tail', 'Coada stîngă (pierderi mari)' if ro else 'Left tail (large losses)')], 'all')
                   + toggles('gt-s', 'Distribuțiile' if ro else 'Distributions',
                             [('tr', 'Reale 2005–2019' if ro else 'Real 2005–2019'), ('g', 'GAN'), ('n', 'Normală' if ro else 'Gaussian')])
                   + play('gt'), charts('gt-hist', 'gt-bt')),
     'gan-acf': (select('ga-f', 'Transformarea randamentului' if ro else 'Transform of the return',
                        [('abs', '|r| (mărimea mișcării)' if ro else '|r| (size of the move)'), ('sq', 'r²'),
                         ('raw', 'r (cu semn)' if ro else 'r (signed)')], 'abs')
                 + slider('ga-L', 'Decalaje afișate (zile)' if ro else 'Lags shown (days)', 5, 60, 5, 20)
                 + toggles('ga-s', 'Seriile' if ro else 'Series',
                           [('tr', 'Reale 2005–2019' if ro else 'Real 2005–2019'), ('g', 'GAN'),
                            ('te', 'Reale 2020–2026' if ro else 'Real 2020–2026')], ['tr', 'g']),
                 charts('ga-chart', cls='one')),
     'diffusion-forward': (select('df-c', 'Curba reală (zona euro, AAA)' if ro else 'Real curve (euro area, AAA)',
                                  [('last', '30.10.2026, ultima curbă din eșantion' if ro else '30 Oct 2026, last curve of the sample'),
                                   ('max10', '16.06.2008, randamentul maxim la 10 ani' if ro else '16 Jun 2008, highest 10-year yield'),
                                   ('min10', '09.03.2020, randamentul minim la 10 ani' if ro else '9 Mar 2020, lowest 10-year yield'),
                                   ('inv', '28.12.2023, cea mai inversată curbă' if ro else '28 Dec 2023, most inverted curve')], 'last')
                           + select('df-e', 'Extragerea zgomotului ε' if ro else 'Draw of the noise ε',
                                    [('7', 'ε din slide (sămînța 7)' if ro else 'ε of the slide (seed 7)'),
                                     ('1', 'altă extragere (sămînța 1)' if ro else 'another draw (seed 1)'),
                                     ('2', 'altă extragere (sămînța 2)' if ro else 'another draw (seed 2)')], '7')
                           + slider('df-t', 'Pasul de difuzie t' if ro else 'Diffusion step t', 0, 199, 1, 60) + play('df'),
                           charts('df-curve', 'df-w')),
     'yc-scenarios': (select('yc-v', 'Variabila' if ro else 'Variable',
                             [('sl', 'Panta 10 ani − 3 luni (puncte procentuale)' if ro else 'Slope 10 years − 3 months (percentage points)'),
                              ('y10', 'Randamentul la 10 ani (%)' if ro else '10-year yield (%)')], 'sl')
                      + slider('yc-x', 'Pragul (%, respectiv puncte procentuale)' if ro else 'Threshold (%, or percentage points)', -1.5, 5, 0.1, 0)
                      + toggles('yc-m', 'Seriile' if ro else 'Series',
                                [('real', 'Curbe reale' if ro else 'Real curves'), ('ddpm', 'Difuzie (DDPM)' if ro else 'Diffusion (DDPM)'),
                                 ('gauss', 'Normală multivariată' if ro else 'Multivariate Gaussian'), ('vae', 'VAE')], ['real', 'ddpm', 'gauss']),
                      charts('yc-hist', 'yc-sc')),
    }


QA = {
 'en': {
  'fraud-alerts': ('The fraud team can check only the 0.1% highest scores of the test period, 57 alerts. How many of the 75 test frauds does '
                   'each score find, and what happens if the budget grows to 1%?',
                   'At 0.1% the autoencoder finds 8 frauds (recall 10.7%), principal component analysis (PCA) finds none and the logistic '
                   'regression, which has seen labelled frauds, finds 48 (64.0%). At 1% (572 alerts) the three find 56, 50 and 65 frauds: with a wider budget the gap between the '
                   'scores narrows, but precision falls (9.7% for the autoencoder), because legitimate transactions outnumber frauds by several hundred to one. The '
                   'autoencoder needs no labels, so it is a detector of the unknown that complements, not replaces, the supervised model.'),
  'vae-latent': ('Each point of the map is the code of a test image; the image on the right is what the decoder draws from the chosen point. '
                 'Where do the footwear classes sit, and does the drawing jump or change smoothly when you move between them?',
                 'Footwear occupies the right of the map: around (1.5, 1.5) 94% of the 50 nearest test images are sneakers, around (1.5, −1.5) '
                 '98% are ankle boots; around (0, 2.1) the images are trousers and around (0.9, −2.1) bags. Move z₂ from 1.5 to −1.5 at '
                 'z₁ = 1.5: the drawing turns gradually from a sneaker into a boot. The Kullback–Leibler (KL) term of the evidence lower bound '
                 '(ELBO) keeps the codes close to a standard normal, so nearby points decode to similar images and every point decodes to '
                 'something plausible. Near (0, 0) shirts and T-shirts overlap, which is why the drawing there is blurry.'),
  'gan-tails': ('The 1% Value-at-Risk (VaR) is the daily loss exceeded on 1% of days. Estimated on 2005–2019, how many violations do the three '
                'estimates give on the 1,717 trading days from January 2020 to October 2026, where about 17 are expected?',
                'At 1% the historical quantile is −3.48%, the GAN (seed 1) gives −3.52%, the Gaussian only −2.67%. Over 2020–2026 the first two '
                'are breached on 19 days each (1.11%), the Gaussian on 41 days (2.39%): it underestimates the tail because daily returns have '
                'fat tails (excess kurtosis 11.86 in 2005–2019). At 5% the three quantiles come close (−1.78%, −1.69%, −1.88%): the differences '
                'are in the far tail. These are unconditional quantiles: none of them reacts to the level of volatility.'),
  'gan-acf': ('Large moves of the S&P 500 come in clusters. Does the GAN, trained on single days, reproduce this?',
              'No. The autocorrelation of |r| is 0.28 at lag 1 in 2005–2019 and stays around 0.3 for 20 days (mean over lags 1–20: 0.31); '
              'for the GAN samples it is −0.01 at lag 1 and close to zero at every lag. The generator draws each day independently, so it '
              'can learn the distribution of one day, with its fat tails, but not the dependence between days. The signed returns r have almost '
              'no autocorrelation in either series: volatility, not direction, is predictable. A sequence model (a recurrent or convolutional '
              'GAN on windows of days) is needed for volatility clustering.'),
  'diffusion-forward': ('The forward process mixes the curve with noise: x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε. From which step does the noise weigh '
                        'more than the curve?',
                        'From t = 117, where ᾱ_t falls below 0.5 (√ᾱ_t = √(1 − ᾱ_t) ≈ 0.71). At t = 20 the curve is almost intact '
                        '(ᾱ = 0.977), at t = 60 ᾱ = 0.827, and at the last step, t = 199, ᾱ = 0.132: with the linear schedule of the lecture '
                        '(β from 0.0001 to 0.02 over 200 steps) about 36% of the signal survives in amplitude. The denoising network learns '
                        'to undo these steps, predicting the noise ε that was added; generation starts from pure noise and runs them backwards.'),
  'yc-scenarios': ('In the real data 8.7% of the days have an inverted curve (10-year yield below the 3-month yield). Which generator '
                   'reproduces this share, and which one misses it?',
                   'With the threshold at 0 on the slope: DDPM (seed 1) gives 10.5% inverted curves, the multivariate Gaussian 14.2%, the '
                   'two-factor VAE only 0.3%. The Gaussian also places scenarios where real curves never were (see the scatter: it fills '
                   'the space between the high-rate regime before 2012 and the negative-rate regime of 2015–2021). Switch to the 10-year yield '
                   'at 0: 13.8% of real days are negative, against 9.9% for DDPM, 9.3% for the Gaussian and 11.2% for the VAE. Diffusion keeps '
                   'both regimes and their curved level–slope relation; the Gaussian smooths them into one.'),
 },
 'ro': {
  'fraud-alerts': ('Echipa antifraudă poate verifica doar primele 0,1% scoruri din perioada de test, 57 de alerte. Cîte dintre cele 75 de '
                   'fraude de test găsește fiecare scor și ce se întîmplă dacă bugetul crește la 1%?',
                   'La 0,1%, autoencoderul găsește 8 fraude (recall 10,7%), analiza componentelor principale (PCA) nu găsește niciuna, iar '
                   'regresia logistică, care a văzut fraude etichetate, găsește 48 (64,0%). La 1% (572 de alerte), cele trei găsesc 56, 50 și 65 de fraude: cu un buget mai larg, diferența '
                   'dintre scoruri se reduce, dar precizia scade (9,7% pentru autoencoder), pentru că tranzacțiile legitime sînt de cîteva sute de ori mai numeroase '
                   'decît fraudele. Autoencoderul nu are nevoie de etichete, deci este un detector al necunoscutului, care completează modelul '
                   'supervizat, nu îl înlocuiește.'),
  'vae-latent': ('Fiecare punct de pe hartă este codul unei imagini de test; imaginea din dreapta este ceea ce desenează decodorul din punctul '
                 'ales. Unde se află încălțămintea și se schimbă desenul brusc sau treptat cînd treceți de la o clasă la alta?',
                 'Încălțămintea ocupă partea dreaptă a hărții: în jurul punctului (1,5; 1,5), 94% dintre cele mai apropiate 50 de imagini de '
                 'test sînt pantofi sport, în jurul punctului (1,5; −1,5) 98% sînt ghete; în jurul punctului (0; 2,1) sînt pantaloni, iar în '
                 'jurul punctului (0,9; −2,1) genți. Mutați z₂ de la 1,5 la −1,5 cu z₁ = 1,5: desenul trece treptat de la pantof sport la '
                 'gheată. Termenul Kullback–Leibler (KL) din marginea inferioară a evidenței (ELBO) ține codurile aproape de o normală '
                 'standard, deci punctele apropiate se decodează în imagini asemănătoare și orice punct se decodează în ceva plauzibil. '
                 'În jurul punctului (0; 0) cămășile și tricourile se suprapun, de aceea desenul de acolo este neclar.'),
  'gan-tails': ('Valoarea la risc (VaR) de 1% este pierderea zilnică depășită în 1% din zile. Estimată pe 2005–2019, cîte depășiri dau cele '
                'trei estimări în cele 1.717 zile de tranzacționare din ianuarie 2020 – octombrie 2026, cînd ne așteptăm la circa 17?',
                'La 1%, cuantila istorică este −3,48%, GAN-ul (sămînța 1) dă −3,52%, iar normala doar −2,67%. În 2020–2026, primele două sînt '
                'depășite în cîte 19 zile (1,11%), normala în 41 de zile (2,39%): subestimează coada, pentru că randamentele zilnice au cozi '
                'groase (exces de kurtosis 11,86 în 2005–2019). La 5%, cele trei cuantile se apropie (−1,78%, −1,69%, −1,88%): diferențele sînt '
                'în coada îndepărtată. Sînt cuantile necondiționate: niciuna nu reacționează la nivelul volatilității.'),
  'gan-acf': ('Mișcările mari ale S&P 500 vin în grupuri. Reproduce acest lucru GAN-ul antrenat pe zile izolate?',
              'Nu. Autocorelația lui |r| este 0,28 la decalajul 1 în 2005–2019 și rămîne în jur de 0,3 timp de 20 de zile (media pe decalajele '
              '1–20: 0,31); pentru eșantioanele GAN este −0,01 la decalajul 1 și aproape zero la orice decalaj. Generatorul extrage fiecare zi '
              'independent, deci poate învăța distribuția unei zile, cu cozile ei groase, dar nu dependența dintre zile. Randamentele cu semn r '
              'nu au aproape deloc autocorelație în niciuna dintre serii: volatilitatea, nu direcția, este previzibilă. Pentru gruparea '
              'volatilității este nevoie de un model de secvențe (un GAN recurent sau convoluțional pe ferestre de zile).'),
  'diffusion-forward': ('Procesul înainte amestecă curba cu zgomot: x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε. De la ce pas cîntărește zgomotul mai '
                        'mult decît curba?',
                        'De la t = 117, unde ᾱ_t scade sub 0,5 (√ᾱ_t = √(1 − ᾱ_t) ≈ 0,71). La t = 20 curba este aproape intactă (ᾱ = 0,977), '
                        'la t = 60 ᾱ = 0,827, iar la ultimul pas, t = 199, ᾱ = 0,132: cu programul liniar din curs (β de la 0,0001 la 0,02 '
                        'în 200 de pași) supraviețuiesc circa 36% din amplitudinea semnalului. Rețeaua de eliminare a zgomotului învață să '
                        'refacă acești pași, prezicînd zgomotul ε adăugat; generarea pornește de la zgomot pur și îi parcurge invers.'),
  'yc-scenarios': ('În datele reale, 8,7% dintre zile au o curbă inversată (randamentul la 10 ani sub cel la 3 luni). Care generator reproduce '
                   'această pondere și care o ratează?',
                   'Cu pragul la 0 pentru pantă: DDPM (sămînța 1) dă 10,5% curbe inversate, normala multivariată 14,2%, iar VAE-ul cu doi '
                   'factori doar 0,3%. Normala pune scenarii și acolo unde curbele reale nu au fost niciodată (vedeți diagrama de dispersie: '
                   'umple spațiul dintre regimul dobînzilor mari de dinainte de 2012 și regimul dobînzilor negative din 2015–2021). Treceți '
                   'la randamentul la 10 ani, cu pragul 0: 13,8% dintre zilele reale au randament negativ, față de 9,9% pentru DDPM, 9,3% pentru '
                   'normală și 11,2% pentru VAE. Difuzia păstrează ambele regimuri și relația curbă dintre nivel și pantă; normala le netezește '
                   'într-unul singur.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 budgets: %BUDGETS%,
 score: { ae: 'autoencoder', pca: 'PCA', logit: 'logistic regression' },
 scoreN: { ae: 'The autoencoder', pca: 'PCA', logit: 'The logistic regression' },
 legit: 'Legitimate (weighted)', fraud: 'Fraud', thr: 'Alert threshold',
 faHist: s => `Distribution of the ${s} score, test period`, faX: s => s === 'logit' ? 'log₁₀ of the predicted probability' : 'log₁₀ of the reconstruction error',
 density: 'Density', recall: 'Recall (share of frauds found)', precision: 'Precision (share of alerts that are frauds)', prTitle: 'Precision–recall curves, test period',
 point: 'Chosen budget',
 fa: (b, n, s, k, tot, rec, prec) => `Budget ${b} of the test transactions: ${window.NNI.cnt(n, 'alert', 'alerts')}. ${s} finds ${k} of the ${tot} frauds (recall ${rec}); precision ${prec}.`,
 fa2: (a, p, l) => `Frauds found at this budget: autoencoder ${a}, PCA ${p}, logistic regression ${l}.`,
 classes: %CLASSES%,
 z1: 'Latent coordinate z₁', z2: 'Latent coordinate z₂', mapTitle: 'Codes (posterior means) of 5,000 of the 10,000 test images', chosen: 'Chosen point',
 imgTitle: (a, b) => `Decoder output at z = (${a}, ${b})`,
 vl: (a, b, c1, p1, c2, p2, r) => `Point (${a}, ${b}): of the 50 nearest of all 10,000 test images, ${p1} are ${c1} and ${p2} ${c2} (all within a distance of ${r}).`,
 vl2: 'The image is the decoder output (sigmoid of the logits) of the two-dimensional VAE trained on the 60,000 training images.',
 ret: 'Daily log return (%)', real: 'Real 2005–2019', gan: 'GAN (seed 1)', norm: 'Gaussian', test: 'Real 2020–2026',
 gtHist: q => `Density of daily returns and the ${q} VaR`, btTitle: 'Test returns 2020–2026 against the VaR estimated on 2005–2019',
 hist: 'Historical', viol: 'Violations',
 gt: (q, h, g, n) => `VaR at ${q}: historical ${h}, GAN ${g}, Gaussian ${n} (daily log return).`,
 gt2: (e, h, g, n, N) => `Violations on ${window.NNI.cnt(N, 'test day', 'test days')} (expected ${e}): historical ${h}, GAN ${g}, Gaussian ${n}.`,
 gt3: (kr, kg) => `Excess kurtosis: real 2005–2019 ${kr}, GAN ${kg}, Gaussian 0.`,
 lag: 'Lag (days)', acf: 'Autocorrelation', f: { abs: '|r|', sq: 'r²', raw: 'r' },
 gaTitle: f => `Autocorrelation function (ACF) of ${f}, S&P 500 daily returns`,
 ga: (s, f, a1, m, L) => `${s}: ACF of ${f} at lag 1 = ${a1}; mean over lags 1–${L} = ${m}.`,
 band: b => `Dashed lines: 95% band for independent days (±${b} for 3,774 days).`,
 mat: 'Maturity (years, log scale)', yield: 'Yield (%)', x0: 'Real curve (t = 0)', xt: t => `Noised curve, t = ${t}`,
 dfTitle: d => `Forward process on the curve of ${d}`, wTitle: 'Weights of the curve and of the noise',
 step: 'Diffusion step t', sig: 'Weight of the curve √ᾱ_t', noi: 'Weight of the noise √(1 − ᾱ_t)',
 df: (t, ab, s, n, rm) => `t = ${t}: ᾱ_t = ${ab}, weight of the curve ${s}, weight of the noise ${n}; root mean square distance from the real curve ${rm} percentage points.`,
 df2: 'Noise schedule of the lecture: β_t linear from 0.0001 to 0.02 over 200 steps; the noise is added to the standardised yields (each maturity: mean and standard deviation over 2004–2026).',
 v: { y10: 'the 10-year yield', sl: 'the slope 10Y − 3M' }, vax: { y10: '10-year yield (%)', sl: 'Slope 10 years − 3 months (pp)' },
 m: { real: 'Real curves', ddpm: 'Diffusion (DDPM)', gauss: 'Multivariate Gaussian', vae: 'VAE' },
 ycHist: v => `Density: ${v}`, scTitle: '10-year yield against the slope', scX: '10-year yield (%)', scY: 'Slope 10Y − 3M (pp)',
 yc: (m, v, x, p, n) => `${m}: ${p} of ${window.NNI.cnt(n, 'curve', 'curves')} have ${v} below ${x}.`,
 yc2: 'Generated: 5,000 scenarios of each model (seed 1); real: 5,664 business days, 6 September 2004 – 30 October 2026.'
};
""",
 'ro': r"""window.LK_TEXT = {
 budgets: %BUDGETS%,
 score: { ae: 'autoencoderului', pca: 'PCA', logit: 'regresiei logistice' },
 scoreN: { ae: 'Autoencoderul', pca: 'PCA', logit: 'Regresia logistică' },
 legit: 'Legitime (ponderate)', fraud: 'Fraude', thr: 'Pragul de alertă',
 faHist: s => `Distribuția scorului ${s}, perioada de test`, faX: s => s === 'logit' ? 'log₁₀ al probabilității prezise' : 'log₁₀ al erorii de reconstrucție',
 density: 'Densitate', recall: 'Recall (ponderea fraudelor găsite)', precision: 'Precizie (ponderea fraudelor în alerte)', prTitle: 'Curbele precizie–recall, perioada de test',
 point: 'Bugetul ales',
 fa: (b, n, s, k, tot, rec, prec) => `Buget de ${b} din tranzacțiile de test: ${window.NNI.cnt(n, 'alertă', 'alerte')}. ${s} găsește ${k} din cele ${tot} de fraude (recall ${rec}); precizia ${prec}.`,
 fa2: (a, p, l) => `Fraude găsite la acest buget: autoencoder ${a}, PCA ${p}, regresie logistică ${l}.`,
 classes: %CLASSES%,
 z1: 'Coordonata latentă z₁', z2: 'Coordonata latentă z₂', mapTitle: 'Codurile (mediile a posteriori) a 5.000 din cele 10.000 de imagini de test', chosen: 'Punctul ales',
 imgTitle: (a, b) => `Ieșirea decodorului în z = (${a}; ${b})`,
 vl: (a, b, c1, p1, c2, p2, r) => `Punctul (${a}; ${b}): dintre cele mai apropiate 50 din toate cele 10.000 de imagini de test, ${p1} sînt din clasa ${c1}, iar ${p2} din clasa ${c2} (toate la o distanță de cel mult ${r}).`,
 vl2: 'Imaginea este ieșirea decodorului (sigmoida logiturilor) a VAE-ului bidimensional antrenat pe cele 60.000 de imagini de antrenare.',
 ret: 'Randamentul logaritmic zilnic (%)', real: 'Reale 2005–2019', gan: 'GAN (sămînța 1)', norm: 'Normală', test: 'Reale 2020–2026',
 gtHist: q => `Densitatea randamentelor zilnice și VaR de ${q}`, btTitle: 'Randamentele de test 2020–2026 față de VaR estimată pe 2005–2019',
 hist: 'Istorică', viol: 'Depășiri',
 gt: (q, h, g, n) => `VaR de ${q}: istorică ${h}, GAN ${g}, normală ${n} (randament logaritmic zilnic).`,
 gt2: (e, h, g, n, N) => `Depășiri în ${window.NNI.cnt(N, 'zi de test', 'zile de test')} (așteptate ${e}): istorică ${h}, GAN ${g}, normală ${n}.`,
 gt3: (kr, kg) => `Excesul de kurtosis: reale 2005–2019 ${kr}, GAN ${kg}, normală 0.`,
 lag: 'Decalajul (zile)', acf: 'Autocorelația', f: { abs: '|r|', sq: 'r²', raw: 'r' },
 gaTitle: f => `Funcția de autocorelație (ACF) a lui ${f}, randamentele zilnice S&P 500`,
 ga: (s, f, a1, m, L) => `${s}: ACF a lui ${f} la decalajul 1 = ${a1}; media pe decalajele 1–${L} = ${m}.`,
 band: b => `Liniile punctate: banda de 95% pentru zile independente (±${b} pentru 3.774 de zile).`,
 mat: 'Scadența (ani, scară logaritmică)', yield: 'Randamentul (%)', x0: 'Curba reală (t = 0)', xt: t => `Curba cu zgomot, t = ${t}`,
 dfTitle: d => `Procesul înainte pe curba din ${d}`, wTitle: 'Ponderile curbei și ale zgomotului',
 step: 'Pasul de difuzie t', sig: 'Ponderea curbei √ᾱ_t', noi: 'Ponderea zgomotului √(1 − ᾱ_t)',
 df: (t, ab, s, n, rm) => `t = ${t}: ᾱ_t = ${ab}, ponderea curbei ${s}, ponderea zgomotului ${n}; distanța pătratică medie față de curba reală ${rm} puncte procentuale.`,
 df2: 'Programul de zgomot din curs: β_t liniar de la 0,0001 la 0,02 în 200 de pași; zgomotul se adaugă randamentelor standardizate (fiecare scadență: media și abaterea standard pe 2004–2026).',
 v: { y10: 'randamentul la 10 ani', sl: 'panta 10A − 3L' }, vax: { y10: 'Randamentul la 10 ani (%)', sl: 'Panta 10 ani − 3 luni (pp)' },
 m: { real: 'Curbe reale', ddpm: 'Difuzie (DDPM)', gauss: 'Normală multivariată', vae: 'VAE' },
 ycHist: v => `Densitatea: ${v}`, scTitle: 'Randamentul la 10 ani față de pantă', scX: 'Randamentul la 10 ani (%)', scY: 'Panta 10A − 3L (pp)',
 yc: (m, v, x, p, n) => `${m}: ${p} din ${window.NNI.cnt(n, 'curbă', 'curbe')} au ${v} sub ${x}.`,
 yc2: 'Generate: 5.000 de scenarii pentru fiecare model (sămînța 1); reale: 5.664 de zile lucrătoare, 6 septembrie 2004 – 30 octombrie 2026.'
};
""",
}
for _lg in ('en', 'ro'):
    TEXT_JS[_lg] = TEXT_JS[_lg].replace('%BUDGETS%', json.dumps(BUDGETS)).replace('%CLASSES%', json.dumps(CLASSES[_lg], ensure_ascii=False))


def data():
    import base64
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from scipy import stats
    out = {}
    R = os.path.join(HERE, 'results')
    C = os.path.join(HERE, 'cache')
    # ---- 1. fraud test scores (ULB card fraud, last 20 % of the time span); legitimate rows carry the weight of the 10 % sample
    f = pd.read_csv(os.path.join(R, 'fraud_test_scores.csv'))
    out['fr'] = {'y': f['y'].astype(int).tolist(), 'w': rnd(float(f.loc[f.y == 0, 'weight'].iloc[0]), 6),
                 'ae': [float(f'{v:.4g}') for v in f['err_ae']], 'pca': [float(f'{v:.4g}') for v in f['err_pca']],
                 'logit': [max(round(float(v), 6), 1e-6) for v in f['p_logit']]}
    # ---- 2. VAE: 2-D latent means of the Fashion-MNIST test images + decoder outputs on a grid of latent points (uint8, base64)
    v = pd.read_csv(os.path.join(R, 'vae_latent.csv'))
    out['vl'] = {'z1': rnd(v.z1.values, 2), 'z2': rnd(v.z2.values, 2), 'y': v.y.astype(int).tolist()}
    VC = torch.load(os.path.join(C, 'vae.pt'), weights_only=False)
    dec = nn.Sequential(nn.Linear(2, 200), nn.ReLU(), nn.Linear(200, 400), nn.ReLU(), nn.Linear(400, 784))
    dec.load_state_dict({k[2:]: t for k, t in VC['m2'].items() if k.startswith('d.')})
    g1, g2 = np.round(np.arange(-2.1, 2.41, 0.3), 2), np.round(np.arange(-3.0, 2.71, 0.3), 2)
    Z = np.array([[a, b] for a in g1 for b in g2], dtype=np.float32)
    with torch.no_grad():
        img = torch.sigmoid(dec(torch.tensor(Z))).numpy()
    out['vd'] = {'g1': g1.tolist(), 'g2': g2.tolist(),
                 'img': base64.b64encode(np.round(img * 255).astype(np.uint8).tobytes()).decode()}
    # ---- 3. S&P 500 daily log returns (train 2005-2019, test 2020 to the cut-off) and the GAN samples of seed 1 (analysis_L8.py)
    m = pd.read_csv(os.path.join(DATA, 'markets_daily.csv'), parse_dates=['date'])
    px = m[m.ticker == 'SP500'].set_index('date')['adjclose'].dropna().sort_index()
    end = (px.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)
    px = px.loc[:end]
    r = np.log(px).diff().dropna()
    rtr, rte = r.loc[:'2019-12-31'], r.loc['2020-01-01':]
    gs = pd.read_csv(os.path.join(R, 'gan_samples.csv'))['gan_seed1'].values[:20000]
    out['gan'] = {'tr': rnd(100 * rtr.values, 3), 'te': rnd(100 * rte.values, 3), 'g': rnd(100 * gs, 3),
                  'ted': [d.strftime('%Y-%m-%d') for d in rte.index],
                  'mu': rnd(100 * float(rtr.mean()), 5), 'sd': rnd(100 * float(rtr.std()), 5),
                  'kr': rnd(float(stats.kurtosis(rtr.values)), 2), 'kg': rnd(float(stats.kurtosis(gs)), 2)}
    # ---- 4. euro-area AAA yield curve (ECB): four real curves for the forward process, standardisation of the lecture
    yc = pd.read_csv(os.path.join(DATA, 'ecb_yield_curve_daily.csv'), parse_dates=['date']).set_index('date')
    endy = (yc.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)
    yc = yc.loc[:endy].dropna()
    MAT = ['y_3M', 'y_6M', 'y_1Y', 'y_2Y', 'y_3Y', 'y_5Y', 'y_7Y', 'y_10Y', 'y_15Y', 'y_20Y', 'y_30Y']
    Y = yc[MAT].values.astype(np.float32)
    ym, ys = Y.mean(0), Y.std(0)
    sl = Y[:, 7] - Y[:, 0]
    pick = {'last': len(Y) - 1, 'max10': int(np.argmax(Y[:, 7])), 'min10': int(np.argmin(Y[:, 7])), 'inv': int(np.argmin(sl))}
    # noise draws: seed 7 is the draw of the slide (analysis_L8.py, chart diffusion_forward)
    eps = {str(s): rnd(np.random.default_rng(s).standard_normal(11), 4) for s in (7, 1, 2)}
    out['df'] = {'mat': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], 'ym': rnd(ym, 4), 'ys': rnd(ys, 4), 'eps': eps,
                 'curves': {k: {'date': str(yc.index[i].date()), 'y': rnd(Y[i], 4)} for k, i in pick.items()}}
    # ---- 5. yield-curve scenarios: real curves against DDPM, multivariate Gaussian and VAE samples (seed 1 of each, as analysis_L8.py)
    Yz = (Y - ym) / ys
    zd = np.load(os.path.join(C, 'ddpm.npz'), allow_pickle=True)
    rng = np.random.default_rng(20261108)
    GA = [rng.multivariate_normal(np.zeros(11), np.cov(Yz.T), 5000) for _ in range(3)]
    sc = {}
    for nm, Zz in [('real', Yz), ('ddpm', zd['DS'][0]), ('gauss', GA[0]), ('vae', zd['VS'][0])]:
        Yy = Zz * ys + ym
        sc[nm] = {'y10': rnd(Yy[:, 7], 3), 'sl': rnd(Yy[:, 7] - Yy[:, 0], 3)}
    out['yc'] = sc
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
