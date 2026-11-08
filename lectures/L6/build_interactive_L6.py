"""Build the interactive page of Lecture 6 (NLP foundations: text representation and word embeddings):
interactive/L6.html (EN) and interactive/L6_ro.html (RO).

Inputs (interactive/L6_data.js) are computed here from data/clean (ECB speeches, Financial PhraseBank, finance tweets) and from
the models trained by analysis_L6.py (lectures/L6/cache/*.pkl: our word2vec and GloVe vectors, the predictions of the sentence and
topic classifiers); the page logic is in interactive/L6.js. Run: python3 lectures/L6/build_interactive_L6.py [--no-data]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, charts, rnd  # noqa: E402

K = 6

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('zipf', 'zipf', 'text', 'zf'),
          ('word-weights', 'top_words', 'sparse', 'sc'),
          ('neighbours', 'w2v_pca', 'dense', 'nb'),
          ('word-map', 'w2v_pca', 'dense', 'pc'),
          ('agreement', 'agreement', 'clf', 'ag'),
          ('topics', 'topic_f1', 'case', 'tp')]

SECTIONS = {
 'en': dict(text='Text as data', sparse='Sparse representations: bag of words', dense='Dense representations: word embeddings',
            clf='Sentence classifiers', case='Case study: topics of finance tweets'),
 'ro': dict(text='Textul ca date', sparse='Reprezentări rare: sacul de cuvinte', dense='Reprezentări dense: vectorii de cuvinte',
            clf='Clasificatoare de fraze', case='Studiu de caz: temele tweet-urilor financiare'),
}

# the two embedding blocks share the slide of the 2-D map; their own titles
TITLES = {
 'en': {'neighbours': 'Nearest neighbours in our word2vec and GloVe spaces', 'word-map': 'A map of the embedding space'},
 'ro': {'neighbours': 'Vecinii cei mai apropiați în spațiile word2vec și GloVe', 'word-map': 'O hartă a spațiului vectorilor'},
}

L = {
 'en': dict(h1='Lecture 6 · NLP foundations: text representation and word embeddings',
   note='Natural language processing (NLP) turns text into numbers. The slide of the lecture is on the left, the same idea on the '
        'right, recomputed in your browser from the course data and from the models trained in the lecture when you move a control. '
        'Under each figure: a readout, a question for the room and the Quantlet with the code of the chart.',
   src='Data: speeches of the European Central Bank (ECB), 8 January 2022 to 30 October 2026, 400 speeches; Financial PhraseBank '
       '(Malo et al., 2014), 4,838 sentences of financial news, 5–8 annotators each; Twitter financial news, sentiment and topic '
       '(Hugging Face, zeroshot). The word vectors (word2vec, GloVe) and the classifiers were trained by lectures/L6/analysis_L6.py; '
       'the bag-of-words logistic regression of the second block is the model of the lecture (PhraseBank split 1), refitted here with '
       'the same settings.'),
 'ro': dict(h1='Cursul 6 · Bazele NLP: reprezentarea textului și vectorii de cuvinte',
   note='Prelucrarea limbajului natural (NLP, natural language processing) transformă textul în numere. În stînga este slide-ul din '
        'curs, în dreapta aceeași idee, recalculată în browser din datele cursului și din modelele antrenate în curs cînd modificați un '
        'control. Sub fiecare figură: o sinteză numerică, o întrebare pentru sală și Quantlet-ul cu codul figurii.',
   src='Date: discursurile Băncii Centrale Europene (BCE), 8 ianuarie 2022 – 30 octombrie 2026, 400 de discursuri; Financial '
       'PhraseBank (Malo et al., 2014), 4.838 de fraze din știri financiare, 5–8 adnotatori pentru fiecare; Twitter financial news, '
       'sentiment și teme (Hugging Face, zeroshot). Vectorii de cuvinte (word2vec, GloVe) și clasificatoarele au fost antrenate de '
       'lectures/L6/analysis_L6.py; regresia logistică pe sacul de cuvinte din al doilea bloc este modelul din curs (împărțirea 1 a '
       'PhraseBank), reestimat aici cu aceleași setări.'),
}

MODELS = ['majority', 'bow_lr', 'tfidf_lr', 'tfidf_mlp', 'w2v_lr', 'embbag', 'embbag_w2v']
MNAME = {'en': {'majority': 'Majority class', 'bow_lr': 'BoW + logistic', 'tfidf_lr': 'TF-IDF + logistic', 'tfidf_mlp': 'TF-IDF + MLP',
                'w2v_lr': 'Mean word2vec + logistic', 'embbag': 'Embedding bag (random start)', 'embbag_w2v': 'Embedding bag (word2vec start)'},
         'ro': {'majority': 'Clasa majoritară', 'bow_lr': 'BoW + logistic', 'tfidf_lr': 'TF-IDF + logistic', 'tfidf_mlp': 'TF-IDF + MLP',
                'w2v_lr': 'Media word2vec + logistic', 'embbag': 'Embedding bag (start aleator)', 'embbag_w2v': 'Embedding bag (start word2vec)'}}
TOPIC_RO = {"Analyst Update": "Actualizări ale analiștilor", "Fed | Central Banks": "Fed și băncile centrale",
            "Company | Product News": "Știri despre companii și produse", "Treasuries | Corporate Debt": "Titluri de stat și obligațiuni corporative",
            "Dividend": "Dividende", "Earnings": "Rezultate financiare", "Energy | Oil": "Energie și petrol",
            "Financials": "Sectorul financiar", "Currencies": "Valute", "General News | Opinion": "Știri generale și opinii",
            "Gold | Metals | Materials": "Aur, metale și materiale", "IPO": "Oferte publice inițiale", "Legal | Regulation": "Juridic și reglementare",
            "M&A | Investments": "Fuziuni, achiziții și investiții", "Macro": "Macroeconomie", "Markets": "Piețe",
            "Politics": "Politică", "Personnel Change": "Schimbări de personal", "Stock Commentary": "Comentarii despre acțiuni",
            "Stock Movement": "Mișcări ale acțiunilor"}
# label ids 0..19 of data/clean/twitter_fin_topic.csv
TOPICS = ['Analyst Update', 'Fed | Central Banks', 'Company | Product News', 'Treasuries | Corporate Debt', 'Dividend', 'Earnings',
          'Energy | Oil', 'Financials', 'Currencies', 'General News | Opinion', 'Gold | Metals | Materials', 'IPO', 'Legal | Regulation',
          'M&A | Investments', 'Macro', 'Markets', 'Politics', 'Personnel Change', 'Stock Commentary', 'Stock Movement']
# real PhraseBank sentences of the second block: row numbers of the PhraseBank (min_agreement >= 50, as analysis_L6.py), all test
# sentences of split 1 (the first is the default: it contains "down")
EX = [153, 1131, 1108, 322, 1056, 2037, 1373, 2137, 2096, 1296]
PROBES = ['inflation', 'energy', 'banks', 'climate', 'digital', 'wages', 'ukraine', 'rates']
GROUPS = {"monetary": ["inflation", "prices", "rates", "interest", "monetary", "policy", "tightening", "price"],
          "energy": ["energy", "gas", "oil", "electricity", "fossil", "renewable", "fuel", "commodity"],
          "banking": ["banks", "lending", "credit", "loans", "deposits", "capital", "liquidity", "supervisory"],
          "digital": ["digital", "payments", "payment", "cash", "crypto", "innovation", "technology", "data"]}
GNAME = {'en': {"monetary": "monetary policy", "energy": "energy", "banking": "banking", "digital": "digital and payments"},
         'ro': {"monetary": "politică monetară", "energy": "energie", "banking": "sistem bancar", "digital": "digitalizare și plăți"}}


def _stored():
    """L6_data.js as written by data() (build() writes the data before the pages)."""
    s = open(os.path.join(interactive_common.OUT, 'L6_data.js'), encoding='utf-8').read()
    return json.loads(s[s.index('{'):s.rstrip().rstrip(';').rindex('}') + 1])


def widgets(lang):
    ro = lang == 'ro'
    D = _stored()
    lab = ['negativă', 'neutră', 'pozitivă'] if ro else ['negative', 'neutral', 'positive']
    sopts = [(i, f'{e["s"][:72]}{"…" if len(e["s"]) > 72 else ""} ({lab[e["y"]]})') for i, e in enumerate(D['sc']['ex'])]
    txt = (f'<div class="ctl" style="flex-basis:100%"><label for="sc-txt">{"Sau scrieți propria frază (în engleză) și apăsați Enter" if ro else "Or type your own sentence (in English) and press Enter"}</label>'
           f'<input type="text" id="sc-txt" value="" style="padding:0.35rem 0.5rem;border:1px solid var(--border);border-radius:6px;font:inherit"></div>')
    words = D['nb']['words']
    return {
     'zipf': (slider('zf-m', 'Frecvența minimă pentru a intra în vocabular' if ro else 'Minimum count to enter the vocabulary', 1, 100, 1, 5)
              + radios('zf-s', 'Scara axelor' if ro else 'Axis scale', [('log', 'dublu logaritmică' if ro else 'log–log'), ('lin', 'liniară' if ro else 'linear')], 'log')
              + toggles('zf-f', 'Linia' if ro else 'Line', [('fit', 'legea putere estimată' if ro else 'fitted power law')], ['fit']),
              charts('zf-chart', 'zf-cov')),
     'word-weights': (select('sc-ex', 'Frază reală din Financial PhraseBank (eticheta adnotatorilor)' if ro else 'Real Financial PhraseBank sentence (label of the annotators)', sopts, 0)
                      + txt
                      + toggles('sc-sw', 'Preprocesarea' if ro else 'Preprocessing', [('stop', 'elimină cuvintele de legătură din engleză (lista scikit-learn)' if ro else 'remove English stop words (scikit-learn list)')], []),
                      charts('sc-chart', 'sc-prob')),
     'neighbours': (select('nb-w', 'Cuvîntul (din discursurile BCE)' if ro else 'Word (from the ECB speeches)', [(w, w) for w in words], 'inflation')
                    + slider('nb-k', 'Numărul de vecini' if ro else 'Number of neighbours', 3, 12, 1, 8),
                    charts('nb-w2v', 'nb-glove')),
     'word-map': (toggles('pc-g', 'Grupurile de cuvinte' if ro else 'Word groups', [(g, GNAME[lang][g]) for g in GROUPS], list(GROUPS))
                  + radios('pc-m', 'Vectorii' if ro else 'Vectors', [('w2v', 'word2vec (SGNS)'), ('glove', 'GloVe')], 'w2v'),
                  charts('pc-chart', cls='one')),
     'agreement': (select('ag-m', 'Modelul' if ro else 'Model', [(m, MNAME[lang][m]) for m in MODELS], 'tfidf_lr')
                   + radios('ag-a', 'Acordul adnotatorilor' if ro else 'Agreement of the annotators',
                            [('all', 'toate frazele' if ro else 'all sentences'), ('50', '50%'), ('66', '66%'), ('75', '75%'), ('100', '100%')], 'all'),
                   charts('ag-cm', 'ag-acc')),
     'topics': (select('tp-t', 'Tema' if ro else 'Topic', [(i, TOPIC_RO[t] if ro else t) for i, t in enumerate(TOPICS)], 10)
                + radios('tp-m', 'Modelul' if ro else 'Model', [('lr', 'TF-IDF + logistic'), ('eb', MNAME[lang]['embbag_w2v'])], 'lr'),
                charts('tp-row', 'tp-f1')),
    }


QA = {
 'en': {
  'zipf': ('Our word2vec keeps only the words that occur at least 5 times in the ECB speeches. How many distinct words does that remove, '
           'and how much of the running text do we lose?',
           'The speeches have 20,130 distinct words; a minimum count of 5 keeps 8,539 of them (42%) but still covers 98.4% of the '
           '1,289,896 tokens. Rare words are many but each is rare, so the lost text is only 1.6%. Raise the threshold to 100: 1,545 '
           'words still cover 86.1% of the text. This is Zipf\'s law at work (slope −1.498 on the log–log scale): a small vocabulary '
           'covers most of the text, and the long tail is where out-of-vocabulary (OOV) words come from.'),
  'word-weights': ('The first sentence, "… operating profit amounted to EUR 11.9 mn, down from EUR 15.0 mn a year earlier", is negative '
                   'with probability 96.3% for the bag-of-words (BoW) logistic regression. What happens if we remove the English stop '
                   'words before scoring it?',
                   'The model loses the word "down", which the scikit-learn stop-word list removes (together with "up", "not", "below" and '
                   '"above"), and "down" is the strongest negative word of the sentence (β_positive − β_negative = −4.41). Without it the '
                   'sentence is predicted neutral (41.0%, against 29.7% negative). The same happens to "Operating profit was EUR 139.7 mn, '
                   'up 23% …", which falls from 83.2% positive to a negative prediction (44.8%). Generic preprocessing removes exactly the '
                   'words that carry financial sentiment; type your own sentence to test other words.'),
  'neighbours': ('Which words are closest to "interest" in the ECB speeches? Does the vector of "interest" also know the sense '
                 '"attention, curiosity"?',
                 'Both models put "rates" and "rate" first (cosine 0.83 and 0.80 in word2vec, 0.91 and 0.83 in GloVe): in central-bank '
                 'speeches "interest" almost always means the interest rate. A static embedding has one vector per word, built from all '
                 'its contexts, so the rarer sense disappears. Words shared by the two top lists are coloured: for "inflation" only 2 of the '
                 'top 5 are common (headline, underlying); the lecture reports 1.88 shared words on average over its eight probe words.'),
  'word-map': ('Projected on two principal components (principal component analysis, PCA), do the four groups of words form separate '
               'regions? How much of the variation of the 100-dimensional vectors does the map keep?',
               'Yes: monetary-policy, energy, banking and payment words occupy different regions, although the map keeps only about 32% of '
               'the variance of the 32 vectors (31.8% for word2vec and 31.5% for GloVe as computed here from vectors rounded to four '
               'decimals; the lecture reports 31.9%). Keep only monetary policy and energy: the first two components now keep 45.9% '
               '(word2vec), because the map has fewer directions to show. A 2-D map is a picture, not a measure of similarity; use the '
               'cosine of the full vectors for that.'),
  'agreement': ('TF-IDF (term frequency–inverse document frequency) with logistic regression is right on 77.7% of the PhraseBank test '
                'sentences. Is it equally good on the sentences on which the annotators disagreed?',
                'No. On the sentences on which all annotators agreed it is right 88.4% of the time, on those with only 50% agreement 58.1% '
                'of the time, hardly better than always answering "neutral" (56.0%). The confusion matrix shows where the errors go: '
                'positive and negative sentences are mostly pushed to neutral (263 positive and 107 negative sentences over the five test '
                'sets). The other models, including the embedding bag, have the same profile: label noise caps the accuracy, so check the '
                'agreement of your own annotators before buying a bigger model.'),
  'topics': ('The topic "Gold | Metals | Materials" has the lowest F1 score of TF-IDF + logistic regression (45.2%). Why?',
             'It has only 13 test tweets, and its words overlap with macroeconomic and market news: 7 of the 13 are classified correctly, '
             '2 go to "Macro". With so few examples one error moves the F1 by several points; the embedding bag reaches 58.3% on the same '
             'topic while being worse on average (macro-F1 80.45% against 82.62%). Choose "Dividend" to see the easiest topic (F1 96.4%): '
             'its vocabulary is specific.'),
 },
 'ro': {
  'zipf': ('Modelul nostru word2vec păstrează doar cuvintele care apar de cel puțin 5 ori în discursurile BCE. Cîte cuvinte distincte '
           'elimină această regulă și cît din textul propriu-zis pierdem?',
           'Discursurile au 20.130 de cuvinte distincte; frecvența minimă 5 păstrează 8.539 dintre ele (42%), dar acoperă încă 98,4% '
           'din cele 1.289.896 de unități lexicale. Cuvintele rare sînt multe, dar fiecare apare rar, deci textul pierdut este doar '
           '1,6%. Ridicați pragul la 100: 1.545 de cuvinte acoperă încă 86,1% din text. Aceasta este legea lui Zipf (panta −1,498 în '
           'scară dublu logaritmică): un vocabular mic acoperă cea mai mare parte a textului, iar coada lungă produce cuvintele din '
           'afara vocabularului (OOV, out of vocabulary).'),
  'word-weights': ('Prima frază, „… operating profit amounted to EUR 11.9 mn, down from EUR 15.0 mn a year earlier”, este negativă cu '
                   'probabilitatea 96,3% după regresia logistică pe sacul de cuvinte (BoW, bag of words). Ce se întîmplă dacă eliminăm '
                   'cuvintele de legătură din engleză înainte de a o evalua?',
                   'Modelul pierde cuvîntul „down”, pe care lista scikit-learn îl elimină (împreună cu „up”, „not”, „below” și „above”), '
                   'iar „down” este cel mai puternic cuvînt negativ al frazei (β_pozitiv − β_negativ = −4,41). Fără el, fraza este '
                   'clasificată neutră (41,0%, față de 29,7% negativă). La fel se întîmplă cu „Operating profit was EUR 139.7 mn, up 23% '
                   '…”, care trece de la 83,2% pozitivă la o predicție negativă (44,8%). O preprocesare generică elimină exact cuvintele '
                   'care poartă sentimentul financiar; scrieți propria frază pentru a testa alte cuvinte.'),
  'neighbours': ('Care sînt cuvintele cele mai apropiate de „interest” în discursurile BCE? Cunoaște vectorul lui „interest” și sensul '
                 '„atenție, curiozitate”?',
                 'Ambele modele pun pe primele locuri „rates” și „rate” (cosinus 0,83 și 0,80 în word2vec, 0,91 și 0,83 în GloVe): în '
                 'discursurile unei bănci centrale, „interest” înseamnă aproape întotdeauna dobîndă. O reprezentare statică are un singur '
                 'vector pentru fiecare cuvînt, construit din toate contextele sale, deci sensul rar dispare. Cuvintele comune celor două '
                 'liste sînt colorate: pentru „inflation”, doar 2 dintre primii 5 vecini sînt comuni (headline, underlying); cursul '
                 'raportează în medie 1,88 cuvinte comune pentru cele opt cuvinte-test.'),
  'word-map': ('Proiectate pe două componente principale (PCA, analiza componentelor principale), formează cele patru grupuri de cuvinte '
               'regiuni separate? Cît din variația vectorilor cu 100 de dimensiuni păstrează harta?',
               'Da: cuvintele despre politica monetară, energie, sistemul bancar și plăți ocupă regiuni diferite, deși harta păstrează doar '
               'circa 32% din varianța celor 32 de vectori (31,8% pentru word2vec și 31,5% pentru GloVe, calculat aici pe vectori rotunjiți '
               'la patru zecimale; cursul raportează 31,9%). Păstrați doar politica monetară și energia: primele două componente păstrează '
               'acum 45,9% (word2vec), pentru că harta are mai puține direcții de arătat. O hartă 2-D este o imagine, nu o măsură a '
               'asemănării; pentru aceasta folosiți cosinusul vectorilor compleți.'),
  'agreement': ('TF-IDF (term frequency–inverse document frequency) cu regresie logistică clasifică corect 77,7% dintre frazele de test '
                'din PhraseBank. Este la fel de bun pe frazele asupra cărora adnotatorii nu au căzut de acord?',
                'Nu. Pe frazele asupra cărora toți adnotatorii au fost de acord are dreptate în 88,4% din cazuri, pe cele cu acord de '
                'doar 50% în 58,1% din cazuri, abia peste răspunsul „neutră” dat mereu (56,0%). Matricea de confuzie arată unde merg '
                'erorile: frazele pozitive și negative sînt împinse mai ales spre neutru (263 de fraze pozitive și 107 negative în cele '
                'cinci seturi de test). Celelalte modele, inclusiv embedding bag, au același profil: zgomotul din etichete limitează '
                'acuratețea, deci verificați acordul propriilor adnotatori înainte de a cumpăra un model mai mare.'),
  'topics': ('Tema „Aur, metale și materiale” are cel mai mic scor F1 pentru TF-IDF + regresie logistică (45,2%). De ce?',
             'Are doar 13 tweet-uri de test, iar cuvintele ei se suprapun cu știrile macroeconomice și de piață: 7 din cele 13 sînt '
             'clasificate corect, 2 ajung la „Macroeconomie”. Cu atît de puține exemple, o singură eroare mută F1 cu cîteva puncte; '
             'embedding bag atinge 58,3% pe aceeași temă, deși este mai slab în medie (macro-F1 80,45% față de 82,62%). Alegeți '
             '„Dividende” pentru tema cea mai ușoară (F1 96,4%): vocabularul ei este specific.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 rank: 'Rank of the word', freq: 'Frequency (occurrences)', inVoc: 'in the vocabulary', outVoc: 'below the minimum count', fit: s => `fitted power law, slope ${s}`,
 vocSize: 'Vocabulary size (words)', cover: 'Share of the text covered (%)', zfTitle: 'Rank–frequency of the words of the ECB speeches', covTitle: 'How much of the text a vocabulary covers',
 zf: (m, v, n, c, oov) => `Minimum count ${m}: ${window.NNI.cnt(v, 'word', 'words')} in the vocabulary out of ${n}; they cover ${c} of the tokens, so ${oov} of the running text is out of vocabulary (OOV).`,
 zf2: (t, s) => `Ten most frequent words: ${t}; together ${s} of all tokens.`,
 cls: ['negative', 'neutral', 'positive'], contrib: 'β positive − β negative (log-odds per occurrence)', prob: 'Probability (%)', probTitle: 'Predicted class probabilities',
 scTitle: 'Weight of each word of the sentence', notVoc: 'not in the vocabulary', empty: 'Type a sentence.',
 sc: (c, p, lab) => `Prediction of the bag-of-words (BoW) logistic regression: ${c} (${p})` + (lab ? `; label of the annotators: ${lab}.` : '.'),
 sc2: (n, k, oov) => `${window.NNI.cnt(n, 'token', 'tokens')}, ${k} of them in the vocabulary of the model (3,561 words seen at least twice in training)` + (oov ? `; not in the vocabulary: ${oov}.` : '.'),
 sc3: r => `Removed as stop words: ${r}.`,
 cos: 'Cosine similarity', shared: 'also in the other model’s list', own: 'only in this model',
 nb: (w, f, k, s, list) => `“${w}” occurs ${window.NNI.cnt(f, 'time', 'times')} in the ECB speeches. Of the ${k} nearest neighbours, ${s} are shared by word2vec and GloVe: ${list}.`,
 nbNone: 'none',
 pc1: 'First principal component', pc2: 'Second principal component',
 pc: (n, v, m) => `${window.NNI.cnt(n, 'word', 'words')}, ${m}: the two components keep ${v} of the variance of the vectors.`,
 pcFew: 'Choose at least one group.',
 trueC: 'Label of the annotators', predC: 'Predicted class', acc: 'Test accuracy (%)', lvl: { all: 'all', 50: '50%', 66: '66%', 75: '75%', 100: '100%' },
 agTitle: 'Confusion matrix, 5 test sets pooled', accTitle: 'Accuracy by agreement of the annotators', majority: 'majority class', agX: 'Agreement of the annotators',
 ag: (m, a, n, acc, maj) => `${m}, ${a === 'all' ? 'all test sentences' : 'sentences with ' + a + '% agreement'} (${n} over the 5 test sets): accuracy ${acc}; majority class ${maj}.`,
 ag2: (r0, r1, r2) => `Recall: negative ${r0}, neutral ${r1}, positive ${r2}.`,
 share: 'Share of the topic’s test tweets (%)', rowTitle: t => `Where the test tweets of “${t}” go`, f1: 'F1 on the test tweets (%)', f1Title: 'F1 by topic',
 tp: (t, n, p, r, f) => `${t}: ${window.NNI.cnt(n, 'test tweet', 'test tweets')}; precision ${p}, recall ${r}, F1 ${f}.`,
 tp2: (f, a) => `All 20 topics: macro-F1 ${f}, accuracy ${a}.`
};
""",
 'ro': r"""window.LK_TEXT = {
 rank: 'Rangul cuvîntului', freq: 'Frecvența (apariții)', inVoc: 'în vocabular', outVoc: 'sub frecvența minimă', fit: s => `legea putere estimată, panta ${s}`,
 vocSize: 'Dimensiunea vocabularului (cuvinte)', cover: 'Ponderea textului acoperit (%)', zfTitle: 'Rang–frecvență pentru cuvintele din discursurile BCE', covTitle: 'Cît din text acoperă un vocabular',
 zf: (m, v, n, c, oov) => `Frecvența minimă ${m}: ${window.NNI.cnt(v, 'cuvînt', 'cuvinte')} în vocabular din ${n}; ele acoperă ${c} din unitățile lexicale, deci ${oov} din text rămîne în afara vocabularului (OOV).`,
 zf2: (t, s) => `Cele mai frecvente zece cuvinte: ${t}; împreună, ${s} din toate unitățile lexicale.`,
 cls: ['negativă', 'neutră', 'pozitivă'], contrib: 'β pozitiv − β negativ (log-șanse la o apariție)', prob: 'Probabilitatea (%)', probTitle: 'Probabilitățile prezise ale claselor',
 scTitle: 'Ponderea fiecărui cuvînt din frază', notVoc: 'nu este în vocabular', empty: 'Scrieți o frază.',
 sc: (c, p, lab) => `Predicția regresiei logistice pe sacul de cuvinte (BoW): ${c} (${p})` + (lab ? `; eticheta adnotatorilor: ${lab}.` : '.'),
 sc2: (n, k, oov) => `${window.NNI.cnt(n, 'unitate lexicală', 'unități lexicale')}, dintre care ${k} în vocabularul modelului (3.561 de cuvinte văzute de cel puțin două ori la antrenare)` + (oov ? `; în afara vocabularului: ${oov}.` : '.'),
 sc3: r => `Eliminate drept cuvinte de legătură: ${r}.`,
 cos: 'Asemănarea cosinus', shared: 'și în lista celuilalt model', own: 'numai în acest model',
 nb: (w, f, k, s, list) => `„${w}” apare de ${window.NNI.cnt(f, 'dată', 'ori')} în discursurile BCE. Dintre cei ${k} vecini cei mai apropiați, ${s} sînt comuni pentru word2vec și GloVe: ${list}.`,
 nbNone: 'niciunul',
 pc1: 'Prima componentă principală', pc2: 'A doua componentă principală',
 pc: (n, v, m) => `${window.NNI.cnt(n, 'cuvînt', 'cuvinte')}, ${m}: cele două componente păstrează ${v} din varianța vectorilor.`,
 pcFew: 'Alegeți cel puțin un grup.',
 trueC: 'Eticheta adnotatorilor', predC: 'Clasa prezisă', acc: 'Acuratețea pe test (%)', lvl: { all: 'toate', 50: '50%', 66: '66%', 75: '75%', 100: '100%' },
 agTitle: 'Matricea de confuzie, cele 5 seturi de test cumulate', accTitle: 'Acuratețea după acordul adnotatorilor', majority: 'clasa majoritară', agX: 'Acordul adnotatorilor',
 ag: (m, a, n, acc, maj) => `${m}, ${a === 'all' ? 'toate frazele de test' : 'frazele cu acord de ' + a + '%'} (${n} în cele 5 seturi de test): acuratețea ${acc}; clasa majoritară ${maj}.`,
 ag2: (r0, r1, r2) => `Recall (sensibilitatea): negativă ${r0}, neutră ${r1}, pozitivă ${r2}.`,
 share: 'Ponderea din tweet-urile temei (%)', rowTitle: t => `Unde ajung tweet-urile de test din „${t}”`, f1: 'F1 pe tweet-urile de test (%)', f1Title: 'F1 pe teme',
 tp: (t, n, p, r, f) => `${t}: ${window.NNI.cnt(n, 'tweet de test', 'tweet-uri de test')}; precizia ${p}, recall ${r}, F1 ${f}.`,
 tp2: (f, a) => `Toate cele 20 de teme: macro-F1 ${f}, acuratețea ${a}.`
};
""",
}


def data():
    import torch  # noqa: F401  (torch before sklearn, as in analysis_L6.py)
    import pickle
    import re
    from collections import Counter
    import numpy as np
    import nn_plot as npl
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, confusion_matrix

    def ld(n):
        with open(os.path.join(HERE, 'cache', n + '.pkl'), 'rb') as f:
            return pickle.load(f)
    TOK = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*")

    def tok(s):                                   # tokeniser of analysis_L6.py
        s = re.sub(r"https?://\S+|@\w+", " ", str(s).lower())
        return ["<num>" if t[0].isdigit() else t for t in TOK.findall(s)]
    out = {}

    # 1. Zipf: frequencies of the words of the ECB speeches, by rank
    sp = npl.ecb_speeches()
    cnt = Counter(w for t in sp.contents for w in tok(t))
    mc = cnt.most_common()
    freq = np.array([c for _, c in mc])
    rk = np.arange(1, len(freq) + 1)
    sel = (rk >= 10) & (rk <= 10000)
    slope, icpt = np.polyfit(np.log(rk[sel]), np.log(freq[sel]), 1)
    out['zf'] = dict(freq=freq.tolist(), top=[w for w, _ in mc[:10]], slope=round(float(slope), 3), icpt=round(float(icpt), 4))

    # 2. Bag of words + logistic regression on PhraseBank, split 1 (the model of the top_words chart): coefficients, intercepts
    pb = npl.phrasebank(50)
    LAB = ['negative', 'neutral', 'positive']
    y = pb.label.map({k: i for i, k in enumerate(LAB)}).values
    toks = [tok(s) for s in pb.sentence]
    SEED = 20261108

    def pb_split(seed):
        rs = np.random.RandomState(SEED + seed)
        tr, va, te = [], [], []
        for c in range(3):
            ids = rs.permutation(np.where(y == c)[0])
            n = len(ids)
            tr += list(ids[:int(0.7 * n)]); va += list(ids[int(0.7 * n):int(0.85 * n)]); te += list(ids[int(0.85 * n):])
        return np.array(tr), np.array(va), np.array(te)
    tr, va, te = pb_split(0)
    jt = lambda ids: [' '.join(toks[i]) for i in ids]  # noqa: E731
    bow = CountVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, min_df=2).fit(jt(tr))
    best = None
    for C in [0.1, 1.0, 10.0]:
        m = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(bow.transform(jt(tr)), y[tr])
        f = f1_score(y[va], m.predict(bow.transform(jt(va))), average='macro')
        if best is None or f > best[0]:
            best = (f, C, m)
    m = best[2]
    voc = bow.get_feature_names_out().tolist()
    SE = ld('sentiment')
    assert np.abs(m.coef_ - SE['pb'][0]['coef']).max() < 0.01, 'refitted BoW model differs from the cached one'
    assert set(EX) <= set(te.tolist())
    out['sc'] = dict(voc=voc, b=[rnd(m.coef_[k], 3) for k in range(3)], b0=rnd(m.intercept_, 3),
                     stop=sorted(w for w in ENGLISH_STOP_WORDS if w in set(voc)),
                     ex=[dict(s=pb.sentence[i], y=int(y[i])) for i in EX])

    # 3. Nearest neighbours in our word2vec and GloVe spaces (ECB vocabulary, min count 5)
    vocab = [w for w, c in mc if c >= 5]
    w2i = {w: i for i, w in enumerate(vocab)}
    unit = lambda E: E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)  # noqa: E731
    Ew, Eg = unit(ld('w2v')['E']), unit(ld('glove')['E'])
    gw = [w for g in GROUPS.values() for w in g]
    extra = [w for w in vocab[60:2000] if w.isalpha() and len(w) >= 4 and w not in ENGLISH_STOP_WORDS and w not in PROBES + gw]
    words = PROBES + [w for w in gw if w not in PROBES] + sorted(extra[:150])
    nb = {}
    for name, E in [('w2v', Ew), ('glove', Eg)]:
        nb[name] = {}
        for w in words:
            s = E @ E[w2i[w]]
            s[w2i[w]] = -1
            top = np.argsort(-s)[:12]
            nb[name][w] = [[vocab[i] for i in top], rnd(s[top], 3)]
    out['nb'] = dict(words=words, nb=nb, freq={w: cnt[w] for w in words})

    # 4. Vectors of the 32 words of the PCA chart (unit length), both models
    out['pc'] = dict(groups=GROUPS, w2v={w: rnd(Ew[w2i[w]], 4) for w in gw}, glove={w: rnd(Eg[w2i[w]], 4) for w in gw})

    # 5. Confusion matrices by annotator agreement, the 5 PhraseBank test sets pooled, every model
    ag = pb.min_agreement_pct.values
    cm = {}
    for k in MODELS:
        cm[k] = {}
        for a in ['all', 50, 66, 75, 100]:
            M = np.zeros((3, 3), int)
            accs = []
            for r in SE['pb']:
                t = r['te']
                msk = np.ones(len(t), bool) if a == 'all' else ag[t] == a
                M += confusion_matrix(y[t][msk], r['preds'][k][msk], labels=[0, 1, 2])
                accs.append(float((r['preds'][k][msk] == y[t][msk]).mean()))
            cm[k][str(a)] = dict(M=M.tolist(), acc=round(100 * float(np.mean(accs)), 1))  # mean over splits, as the chart
    out['ag'] = dict(cm=cm, mname=MNAME)

    # 6. Topics of finance tweets: confusion matrices of TF-IDF + logistic and of the embedding bag on the 4,117 test tweets
    tp = npl.tweets_topic()
    TP = ld('topic')
    yt = tp.label.values[TP['te']]
    names = tp.drop_duplicates('label').set_index('label').label_name.sort_index().tolist()
    assert names == TOPICS
    out['tp'] = dict(names=names, ro=[TOPIC_RO[t] for t in TOPICS],
                     lr=confusion_matrix(yt, TP['lr'], labels=list(range(20))).tolist(),
                     eb=confusion_matrix(yt, TP['eb'], labels=list(range(20))).tolist())
    out['gname'] = GNAME
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
