"""Build the interactive page of Lecture 7 (Attention and transformers): interactive/L7.html (EN) and interactive/L7_ro.html (RO).

Inputs (interactive/L7_data.js) come from the lecture's own results (lectures/L7/results, cache) and from the pretrained FinBERT
(ProsusAI/finbert, local Hugging Face cache) run on seven real test tweets: head-averaged attention of the 12 layers and, for the
lecture's example tweet, the raw query-key scores of every layer and head (cache/interactive_attention.json, computed once).
The page logic is in interactive/L7.js. Run: python3 lectures/L7/build_interactive_L7.py [--no-data]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, play, charts, rnd, ROOT  # noqa: E402

K = 7
RES = os.path.join(HERE, 'results')
CACHE = os.path.join(HERE, 'cache')
ATT_CACHE = os.path.join(CACHE, 'interactive_attention.json')
# test tweets (row of the published validation split): the lecture's example first, then two bearish, two bullish, two neutral
TWEETS = [9, 19, 195, 70, 63, 143, 97]
MODELS = ['tfidf', 'embbag', 'tiny_nopos', 'tiny', 'finbert_zs', 'finbert_probe', 'finbert_ft']

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('context-lengths', 'context_lengths', 'why', 'cl'),
          ('attention-map', 'attention_example', 'mech', 'am'),
          ('softmax-temperature', 'attention_entropy', 'mech', 'st'),
          ('positional-encoding', 'positional_encoding', 'arch', 'pe'),
          ('model-comparison', 'model_comparison', 'case', 'mc'),
          ('data-scaling', 'data_scaling', 'case', 'ds')]

SECTIONS = {
 'en': dict(why='Why attention', mech='The attention mechanism', arch='The transformer architecture',
            case='Case study: sentiment of finance tweets'),
 'ro': dict(why='De ce atenția', mech='Mecanismul atenției', arch='Arhitectura transformer',
            case='Studiu de caz: sentimentul tweet-urilor financiare'),
}

MNAME = {
 'en': dict(tfidf='TF-IDF + logistic regression', embbag='Embedding bag', tiny_nopos='Small transformer, no positions',
            tiny='Small transformer', finbert_zs='FinBERT zero-shot', finbert_probe='FinBERT frozen + logistic',
            finbert_ft='FinBERT fine-tuned'),
 'ro': dict(tfidf='TF-IDF + regresie logistică', embbag='Medie de embedding-uri', tiny_nopos='Transformer mic, fără poziții',
            tiny='Transformer mic', finbert_zs='FinBERT fără antrenare', finbert_probe='FinBERT înghețat + logistic',
            finbert_ft='FinBERT ajustat'),
}

L = {
 'en': dict(h1='Lecture 7 · Attention and transformers',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser when you move a control. '
        'The attention maps are those of the pretrained FinBERT (BERT, Bidirectional Encoder Representations from Transformers, further '
        'trained on financial text) on real test tweets; the model results are those of the case study of the lecture (2,388 test tweets, '
        'three classes: bearish, bullish, neutral). The ▶ button moves the main parameter step by step. Under each figure: a readout, a '
        'question for the room and the Quantlet with the code of the chart.',
   src='Tweets: Twitter financial news sentiment (Hugging Face, zeroshot), published validation split used as test set. Financial '
       'PhraseBank: Malo et al. (2014). Speeches: European Central Bank (ECB), 8 January 2022 to 30 October 2026. Tokens: WordPiece tokenizer '
       'of ProsusAI/finbert. Attention: ProsusAI/finbert, 12 layers × 12 heads, d_k = 64. Model results, test predictions (seed 1) and '
       'learning with less data: own estimates of the lecture (lectures/L7/results). Positional encoding: formula of Vaswani et al. (2017).'),
 'ro': dict(h1='Cursul 7 · Atenția și transformerele',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser cînd modificați un control. Hărțile de '
        'atenție sînt cele ale modelului preantrenat FinBERT (BERT, Bidirectional Encoder Representations from Transformers, antrenat '
        'suplimentar pe texte financiare) pe tweet-uri reale de test; rezultatele modelelor sînt cele din studiul de caz al cursului '
        '(2.388 de tweet-uri de test, trei clase: pesimist, optimist, neutru). Butonul ▶ modifică parametrul principal pas cu pas. Sub '
        'fiecare figură: o sinteză numerică, o întrebare pentru sală și Quantlet-ul cu codul figurii.',
   src='Tweet-uri: Twitter financial news sentiment (Hugging Face, zeroshot), setul de validare publicat folosit ca set de test. '
       'Financial PhraseBank: Malo et al. (2014). Discursuri: Banca Centrală Europeană (BCE), 8 ianuarie 2022 – 30 octombrie 2026. Tokeni: '
       'tokenizatorul WordPiece al modelului ProsusAI/finbert. Atenția: ProsusAI/finbert, 12 straturi × 12 capete, d_k = 64. Rezultatele '
       'modelelor, predicțiile de test (sămînța 1) și învățarea cu mai puține date: estimări proprii din curs (lectures/L7/results). '
       'Codificarea poziției: formula din Vaswani et al. (2017).'),
}

LABEL = {'en': ['bearish', 'bullish', 'neutral'], 'ro': ['pesimist', 'optimist', 'neutru']}


def widgets(lang):
    ro = lang == 'ro'
    att = json.load(open(ATT_CACHE, encoding='utf-8'))
    lab = LABEL[lang]
    tw_opts = [(i, f'{a["text"]} ({"observat" if ro else "label"}: {lab[a["label"]]})') for i, a in enumerate(att)]
    corp = [('tweets', 'Tweet-uri financiare' if ro else 'Finance tweets'), ('phrasebank', 'Financial PhraseBank'),
            ('speeches', 'Discursuri BCE' if ro else 'ECB speeches')]
    mopts = [(m, MNAME[lang][m]) for m in MODELS]
    return {
     'context-lengths': (select('cl-c', 'Corpusul' if ro else 'Corpus', corp, 'speeches')
                         + slider('cl-w', 'Fereastra de context (tokeni, 2^k)' if ro else 'Context window (tokens, 2^k)', 5, 15, 1, 9)
                         + play('cl'), charts('cl-hist', 'cl-cdf')),
     'attention-map': (select('am-t', 'Tweet-ul (setul de test)' if ro else 'Tweet (test set)', tw_opts, 0)
                       + slider('am-l', 'Stratul FinBERT' if ro else 'FinBERT layer', 1, 12, 1, 12)
                       + slider('am-q', 'Tokenul care întreabă (query)' if ro else 'Querying token (query)', 0, 12, 1, 0)
                       + play('am'), charts('am-map', 'am-row')),
     'softmax-temperature': (slider('st-c', 'Împărțitorul scorurilor: c · √d_k, c =' if ro else 'Divisor of the scores: c · √d_k, c =', -3, 3, 0.25, 0)
                             + slider('st-l', 'Stratul' if ro else 'Layer', 1, 12, 1, 6)
                             + select('st-h', 'Capul de atenție' if ro else 'Attention head', [(h, f'{"capul" if ro else "head"} {h + 1}') for h in range(12)], 0)
                             + slider('st-q', 'Tokenul care întreabă (query)' if ro else 'Querying token (query)', 0, 12, 1, 0)
                             + play('st'), charts('st-row', 'st-ent')),
     'positional-encoding': (slider('pe-t', 'Poziția tokenului t' if ro else 'Token position t', 0, 63, 1, 10)
                             + slider('pe-i', 'Perechea de dimensiuni i (dimensiunile 2i și 2i+1)' if ro else 'Dimension pair i (dimensions 2i and 2i+1)', 0, 31, 1, 2)
                             + play('pe'), charts('pe-map', 'pe-sim')),
     'model-comparison': (select('mc-a', 'Modelul A' if ro else 'Model A', mopts, 'finbert_ft')
                          + select('mc-b', 'Modelul B' if ro else 'Model B', mopts, 'tfidf')
                          + slider('mc-r', 'Replicări bootstrap' if ro else 'Bootstrap replications', 200, 2000, 200, 1000)
                          + play('mc'), charts('mc-bar', 'mc-boot')),
     'data-scaling': (slider('ds-n', 'Tweet-uri de antrenare (pasul din curs)' if ro else 'Training tweets (step of the lecture)', 0, 4, 1, 0)
                      + toggles('ds-m', 'Modele' if ro else 'Models',
                                [('tfidf', MNAME[lang]['tfidf']), ('tiny', MNAME[lang]['tiny']), ('probe', MNAME[lang]['finbert_probe'])])
                      + play('ds'), charts('ds-line', 'ds-bar')),
    }


QA = {
 'en': {
  'context-lengths': ('BERT reads at most 512 tokens. Which of the three corpora fits, and what does the quadratic cost of attention mean for an ECB speech?',
                      'Tweets (median 25 tokens, maximum 82) and Financial PhraseBank sentences (median 28) fit easily; 98.8% of the 400 ECB '
                      'speeches are longer than 512 tokens (median 3,338). Self-attention compares every token with every other one, so its cost '
                      'grows with n²: a median speech costs about (3,338/25)² ≈ 17,800 times a median tweet. Long documents are therefore cut into '
                      'windows or read by models built for long contexts.'),
  'attention-map': ('In the last layer, which word does the [CLS] token (the summary token used for classification) look at in “BTIG points to '
                    'breakfast pressure for Dunkin\' Brands”? Is that the explanation of the prediction?',
                    'In layer 12, [CLS] spreads its attention over “points”, “to”, “pressure” and “breakfast” (about 14–17% each); in layer 6 '
                    'it puts 66% on [SEP]. Many heads park attention on [SEP] when they have nothing useful to do. FinBERT calls this tweet '
                    'neutral (75%), while its label is bearish: attention weights show where information flows, not why the model decides, so '
                    'they are not an explanation.'),
  'softmax-temperature': ('Why divide the query-key scores by √d_k = 8 before the softmax? Move c below and above 1.',
                          'The dot product of two 64-dimensional vectors has a spread that grows with √d_k; without the division, the softmax '
                          'saturates: one token gets almost all the weight and the gradients of the others vanish. With c = 1/8 (2^−3) most rows '
                          'put nearly all their weight on one token; with c = 8 they become close to uniform (entropy / log n near 1). On the 500 '
                          'test tweets of the lecture, the normalised entropy falls from 0.76 in layer 1 to 0.36 in layer 8.'),
  'positional-encoding': ('Attention by itself does not see the order of the words. How does the sinusoidal encoding tell position 10 from position 11?',
                          'Each position gets a vector of 64 sines and cosines with wavelengths from 2π to about 2π·10,000^(62/64): the first pairs '
                          'change fast, the last ones slowly, like the hands of a clock. The dot product of two position vectors is largest when '
                          'they are equal and falls with the distance, so nearby positions look similar. In the case study, adding positions to the '
                          'small transformer raises macro-F1 by 0.9 points (95% interval 0.1 to 1.7).'),
  'model-comparison': ('FinBERT fine-tuned beats TF-IDF + logistic regression by about 9 points of macro-F1. Is that more than the noise of a test set of 2,388 tweets?',
                       'Yes. Resampling the test tweets (paired bootstrap on the seed-1 predictions) gives a difference of about 9 points with a '
                       '95% interval of roughly 7 to 11 points that excludes zero (lecture, averaged over three seeds: 9.3, interval 7.3 to 11.3). '
                       'FinBERT frozen + logistic against TF-IDF gives an interval around zero (lecture: −0.2, from −2.5 to 2.0): a large '
                       'pretrained model used without fine-tuning is not automatically better.'),
  'data-scaling': ('With only 429 labelled tweets, which model would you use?',
                   'The frozen FinBERT with a logistic regression on top: 70.4% macro-F1 with 429 tweets, against 52.1% for TF-IDF and 43.3% '
                   'for the small transformer trained from scratch. Pretraining brings knowledge of the language that the small data cannot '
                   'supply. With all 8,588 tweets TF-IDF catches up (75.5% against 75.3%), and the small transformer is still behind (69.7%).'),
 },
 'ro': {
  'context-lengths': ('BERT citește cel mult 512 tokeni. Care dintre cele trei corpusuri încape și ce înseamnă costul pătratic al atenției pentru un discurs BCE?',
                      'Tweet-urile (mediana 25 de tokeni, maximum 82) și frazele din Financial PhraseBank (mediana 28) încap ușor; 98,8% dintre '
                      'cele 400 de discursuri BCE au mai mult de 512 tokeni (mediana 3.338). Atenția compară fiecare token cu toți ceilalți, deci '
                      'costul crește cu n²: un discurs median costă aproximativ (3.338/25)² ≈ 17.800 de tweet-uri mediane. Documentele lungi sînt '
                      'de aceea tăiate în ferestre sau citite cu modele construite pentru contexte lungi.'),
  'attention-map': ('În ultimul strat, la ce cuvînt se uită tokenul [CLS] (tokenul-rezumat folosit la clasificare) în „BTIG points to breakfast '
                    'pressure for Dunkin\' Brands”? Este aceasta explicația predicției?',
                    'În stratul 12, [CLS] își împarte atenția între „points”, „to”, „pressure” și „breakfast” (circa 14–17% fiecare); în '
                    'stratul 6 pune 66% pe [SEP]. Multe capete își „parchează” atenția pe [SEP] cînd nu au nimic util de făcut. FinBERT '
                    'clasifică tweet-ul ca neutru (75%), deși eticheta lui este pesimist: ponderile atenției arată pe unde circulă informația, nu '
                    'de ce decide modelul, deci nu sînt o explicație.'),
  'softmax-temperature': ('De ce împărțim scorurile query-key la √d_k = 8 înainte de softmax? Mutați c sub 1 și peste 1.',
                          'Produsul scalar a doi vectori cu 64 de dimensiuni are o împrăștiere care crește cu √d_k; fără împărțire, softmax se '
                          'saturează: un singur token primește aproape toată ponderea, iar gradienții celorlalți dispar. Cu c = 1/8 (2^−3), cele mai '
                          'multe rînduri pun aproape toată ponderea pe un token; cu c = 8 devin aproape uniforme (entropie / log n aproape de 1). Pe '
                          'cele 500 de tweet-uri de test din curs, entropia normalizată scade de la 0,76 în stratul 1 la 0,36 în stratul 8.'),
  'positional-encoding': ('Atenția singură nu vede ordinea cuvintelor. Cum deosebește codificarea sinusoidală poziția 10 de poziția 11?',
                          'Fiecare poziție primește un vector de 64 de sinusuri și cosinusuri, cu lungimi de undă de la 2π pînă la circa '
                          '2π·10.000^(62/64): primele perechi se schimbă repede, ultimele încet, ca acele unui ceas. Produsul scalar a doi vectori de '
                          'poziție este maxim cînd pozițiile coincid și scade cu distanța, deci pozițiile apropiate seamănă. În studiul de caz, '
                          'adăugarea pozițiilor în transformerul mic crește macro-F1 cu 0,9 puncte (intervalul de 95%: 0,1 – 1,7).'),
  'model-comparison': ('FinBERT ajustat depășește TF-IDF + regresie logistică cu aproximativ 9 puncte de macro-F1. Este mai mult decît zgomotul unui set de test de 2.388 de tweet-uri?',
                       'Da. Reeșantionarea tweet-urilor de test (bootstrap pereche pe predicțiile cu sămînța 1) dă o diferență de circa 9 puncte, '
                       'cu un interval de 95% de aproximativ 7 – 11 puncte, care nu conține zero (în curs, media pe trei semințe: 9,3, intervalul '
                       '7,3 – 11,3). FinBERT înghețat + logistic față de TF-IDF dă un interval în jurul lui zero (în curs: −0,2, de la −2,5 la 2,0): '
                       'un model mare preantrenat, folosit fără ajustare, nu este automat mai bun.'),
  'data-scaling': ('Cu numai 429 de tweet-uri etichetate, ce model ați folosi?',
                   'FinBERT înghețat cu o regresie logistică deasupra: 70,4% macro-F1 cu 429 de tweet-uri, față de 52,1% pentru TF-IDF și '
                   '43,3% pentru transformerul mic antrenat de la zero. Preantrenarea aduce cunoașterea limbii pe care datele puține nu o pot '
                   'oferi. Cu toate cele 8.588 de tweet-uri, TF-IDF ajunge din urmă (75,5% față de 75,3%), iar transformerul mic rămîne în urmă (69,7%).'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 labels: ['bearish', 'bullish', 'neutral'],
 corpus: { tweets: 'Finance tweets', phrasebank: 'Financial PhraseBank', speeches: 'ECB speeches' },
 models: %(MEN)s,
 tokLog: 'Text length in tokens (log scale)', pctTexts: '% of texts', fits: '% of texts that fit in the window',
 window: w => `window ${w} tokens`, clHist: c => `Length of the texts: ${c}`, clCdf: 'Share of texts that fit in the window',
 cl: (c, n, w, f, med, mx) => `${c}: ${window.NNI.cnt(n, 'text', 'texts')}, median ${med} tokens, maximum ${mx}; a window of ${w} tokens holds ${f} of them in full.`,
 cl2: (w, k, r) => `The median ECB speech needs ${k} windows of ${w} tokens; one window of ${w} tokens costs ${r} times the attention of a median tweet (cost ∝ n²).`,
 keys: 'Keys (tokens looked at)', queries: 'Queries (tokens that ask)', weight: 'Attention weight',
 amMap: l => `Attention of layer ${l}, mean of the 12 heads`, amRow: q => `Row of the query “${q}”: where it looks`,
 am: (q, l, top, tp, sep, cls) => `Layer ${l}, query “${q}”: largest weight on a word “${top}” (${tp}); on [SEP] ${sep}, on [CLS] ${cls}.`,
 am2: (lab, pb, pu, pn) => `Label of the tweet: ${lab}. FinBERT zero-shot: bearish ${pb}, bullish ${pu}, neutral ${pn}.`,
 stRow: (q, h, l) => `Weights of the query “${q}”, head ${h}, layer ${l}`, now: 'with the chosen divisor', ref: 'with √d_k (c = 1, FinBERT)',
 stEnt: 'Normalised entropy of the attention rows, by layer', layer: 'Layer', ent: 'Entropy / log n',
 ent500: '500 test tweets, c = 1 (lecture)', entThis: 'this tweet, chosen c (12 heads)',
 st: (c, h, mx, k) => `c = ${c}: entropy of the row / log n = ${h}, largest weight ${mx}, effective number of tokens looked at ${k}.`,
 st2: (l, e1, ec) => `Layer ${l}, this tweet, mean over the 12 heads: entropy / log n = ${e1} with c = 1, ${ec} with the chosen c.`,
 position: 'Position t', dim: 'Dimension', peMap: 'Sinusoidal positional encoding, d = 64', peSim: t => `Similarity of position ${t} with every position`,
 dot: 'Dot product PE(t)·PE(s) / 32',
 pe: (t, i, lam, s, c) => `Dimensions ${2 * i} and ${2 * i + 1}: wavelength ${lam} positions; at t = ${t}: sin = ${s}, cos = ${c}.`,
 pe2: (t, s1, s5, s20) => `Similarity of t = ${t} with t+1: ${s1}, with t+5: ${s5}, with t+20: ${s20} (1 = identical).`,
 f1: 'Macro-F1, test (%)', diff: 'Difference of macro-F1, A − B (points)', freq: 'Replications',
 mcBar: 'Macro-F1 of the seven models (bar: mean of runs; dots: runs)', mcBoot: r => `Paired bootstrap, ${r} replications, seed-1 predictions`,
 mc: (a, b, fa, fb, d) => `${a}: ${fa}%; ${b}: ${fb}% (seed 1); difference ${d} points.`,
 mc2: (lo, hi, p) => `95% bootstrap interval ${lo} to ${hi} points; A better than B in ${p} of the replications.`,
 same: 'Choose two different models.',
 trainN: 'Training tweets (log scale)', dsLine: 'Macro-F1 on the 2,388 test tweets by size of the training set', dsBar: n => `Macro-F1 with ${n} training tweets`,
 ds: (n, s) => `${n} training tweets: ${s}.`,
 tf: 'TF-IDF + logistic regression', tiny: 'Small transformer', probe: 'FinBERT frozen + logistic'
};
""",
 'ro': r"""window.LK_TEXT = {
 labels: ['pesimist', 'optimist', 'neutru'],
 corpus: { tweets: 'Tweet-uri financiare', phrasebank: 'Financial PhraseBank', speeches: 'Discursuri BCE' },
 models: %(MRO)s,
 tokLog: 'Lungimea textului în tokeni (scară logaritmică)', pctTexts: '% din texte', fits: '% din texte care încap în fereastră',
 window: w => `fereastra de ${w} tokeni`, clHist: c => `Lungimea textelor: ${c}`, clCdf: 'Ponderea textelor care încap în fereastră',
 cl: (c, n, w, f, med, mx) => `${c}: ${window.NNI.cnt(n, 'text', 'texte')}, mediana ${med} tokeni, maximum ${mx}; o fereastră de ${w} tokeni cuprinde complet ${f} dintre ele.`,
 cl2: (w, k, r) => `Discursul BCE median are nevoie de ${k} ferestre de ${w} tokeni; o fereastră de ${w} tokeni costă de ${r} ori atenția unui tweet median (costul ∝ n²).`,
 keys: 'Chei (tokenii priviți)', queries: 'Interogări (tokenii care întreabă)', weight: 'Ponderea atenției',
 amMap: l => `Atenția stratului ${l}, media celor 12 capete`, amRow: q => `Rîndul interogării „${q}”: unde se uită`,
 am: (q, l, top, tp, sep, cls) => `Stratul ${l}, interogarea „${q}”: ponderea cea mai mare pe un cuvînt „${top}” (${tp}); pe [SEP] ${sep}, pe [CLS] ${cls}.`,
 am2: (lab, pb, pu, pn) => `Eticheta tweet-ului: ${lab}. FinBERT fără antrenare: pesimist ${pb}, optimist ${pu}, neutru ${pn}.`,
 stRow: (q, h, l) => `Ponderile interogării „${q}”, capul ${h}, stratul ${l}`, now: 'cu împărțitorul ales', ref: 'cu √d_k (c = 1, FinBERT)',
 stEnt: 'Entropia normalizată a rîndurilor de atenție, pe straturi', layer: 'Stratul', ent: 'Entropie / log n',
 ent500: '500 de tweet-uri de test, c = 1 (curs)', entThis: 'acest tweet, c ales (12 capete)',
 st: (c, h, mx, k) => `c = ${c}: entropia rîndului / log n = ${h}, ponderea maximă ${mx}, numărul efectiv de tokeni priviți ${k}.`,
 st2: (l, e1, ec) => `Stratul ${l}, acest tweet, media celor 12 capete: entropie / log n = ${e1} cu c = 1, ${ec} cu c ales.`,
 position: 'Poziția t', dim: 'Dimensiunea', peMap: 'Codificarea sinusoidală a poziției, d = 64', peSim: t => `Similaritatea poziției ${t} cu fiecare poziție`,
 dot: 'Produsul scalar PE(t)·PE(s) / 32',
 pe: (t, i, lam, s, c) => `Dimensiunile ${2 * i} și ${2 * i + 1}: lungimea de undă ${lam} poziții; la t = ${t}: sin = ${s}, cos = ${c}.`,
 pe2: (t, s1, s5, s20) => `Similaritatea lui t = ${t} cu t+1: ${s1}, cu t+5: ${s5}, cu t+20: ${s20} (1 = identic).`,
 f1: 'Macro-F1, test (%)', diff: 'Diferența de macro-F1, A − B (puncte)', freq: 'Replicări',
 mcBar: 'Macro-F1 al celor șapte modele (bara: media rulărilor; punctele: rulările)', mcBoot: r => `Bootstrap pereche, ${window.NNI.cnt(r, 'replicare', 'replicări')}, predicțiile cu sămînța 1`,
 mc: (a, b, fa, fb, d) => `${a}: ${fa}%; ${b}: ${fb}% (sămînța 1); diferența ${d} puncte.`,
 mc2: (lo, hi, p) => `Intervalul bootstrap de 95%: de la ${lo} la ${hi} puncte; A este mai bun decît B în ${p} dintre replicări.`,
 same: 'Alegeți două modele diferite.',
 trainN: 'Tweet-uri de antrenare (scară logaritmică)', dsLine: 'Macro-F1 pe cele 2.388 de tweet-uri de test, după mărimea setului de antrenare', dsBar: n => `Macro-F1 cu ${n} tweet-uri de antrenare`,
 ds: (n, s) => `${n} tweet-uri de antrenare: ${s}.`,
 tf: 'TF-IDF + regresie logistică', tiny: 'Transformer mic', probe: 'FinBERT înghețat + logistic'
};
""",
}
TEXT_JS = {k: v.replace('%(MEN)s', json.dumps(MNAME['en'], ensure_ascii=False)).replace('%(MRO)s', json.dumps(MNAME['ro'], ensure_ascii=False))
           for k, v in TEXT_JS.items()}


def attention_cache():
    """FinBERT (pretrained, eager attention) on the tweets TWEETS of the test set: tokens, zero-shot probabilities, head-averaged
    attention of the 12 layers; for the first tweet also the raw query-key scores q·k (before the division by √d_k = 8)."""
    if os.path.exists(ATT_CACHE):
        return json.load(open(ATT_CACHE, encoding='utf-8'))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tw = pd.read_csv(os.path.join(ROOT, 'data', 'clean', 'twitter_fin_sentiment.csv'))
    te = tw[tw.split == 'valid'].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained('ProsusAI/finbert', local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', local_files_only=True, attn_implementation='eager').eval()
    fin_map = {0: 1, 1: 0, 2: 2}  # FinBERT (positive, negative, neutral) -> ours (bearish 0, bullish 1, neutral 2)
    out = []
    for k, i in enumerate(TWEETS):
        e = tok(te.text[i], return_tensors='pt')
        with torch.no_grad():
            o = m(**e, output_attentions=True, output_hidden_states=True)
        p = torch.softmax(o.logits, 1)[0].numpy()
        pr = [0.0] * 3
        for f, ours in fin_map.items():
            pr[ours] = float(p[f])
        rec = dict(text=te.text[i], label=int(te.label[i]), toks=tok.convert_ids_to_tokens(e['input_ids'][0]), p=pr,
                   avg=[o.attentions[l][0].mean(0).numpy().round(3).tolist() for l in range(12)])
        if k == 0:
            sc = []
            for l in range(12):
                sa = m.bert.encoder.layer[l].attention.self
                h = o.hidden_states[l][0]
                q = sa.query(h).view(-1, 12, 64).transpose(0, 1)
                kk = sa.key(h).view(-1, 12, 64).transpose(0, 1)
                S = (q @ kk.transpose(1, 2)).detach()
                assert float((torch.softmax(S / 8.0, -1) - o.attentions[l][0]).abs().max()) < 1e-4
                sc.append(S.numpy().round(1).tolist())
            rec['qk'] = sc
        out.append(rec)
    json.dump(out, open(ATT_CACHE, 'w', encoding='utf-8'))
    return out


def data():
    import pandas as pd
    out = {}
    tl = pd.read_csv(os.path.join(RES, 'token_lengths.csv'))
    out['cl'] = {c: sorted(int(x) for x in tl.tokens[tl.corpus == c]) for c in ('tweets', 'phrasebank', 'speeches')}
    att = attention_cache()
    out['att'] = [dict(text=a['text'], label=a['label'], toks=a['toks'], p=rnd(a['p'], 3), avg=a['avg']) for a in att]
    out['qk'] = att[0]['qk']
    ae = pd.read_csv(os.path.join(RES, 'attention_entropy.csv'))
    out['ent'] = rnd(ae.entropy.tolist(), 4)
    mm = pd.read_csv(os.path.join(RES, 'model_metrics.csv'))
    out['runs'] = {m: rnd((100 * mm.macro_f1[mm.model == m]).tolist(), 2) for m in MODELS}
    tp = pd.read_csv(os.path.join(RES, 'test_predictions.csv'))
    out['pred'] = {c: ''.join(str(int(v)) for v in tp[c]) for c in ['y'] + MODELS}
    ds = pd.read_csv(os.path.join(RES, 'data_scaling.csv'))
    out['ds'] = {'n': ds.n.tolist(), 'tfidf': rnd((100 * ds.tfidf).tolist(), 2), 'tiny': rnd((100 * ds.tiny).tolist(), 2),
                 'probe': rnd((100 * ds.probe).tolist(), 2)}
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
