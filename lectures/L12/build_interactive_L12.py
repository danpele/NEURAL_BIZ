"""Build the interactive page of Lecture 12 (Large language models in finance and prompt engineering):
interactive/L12.html (EN) and interactive/L12_ro.html (RO).

Inputs (interactive/L12_data.js) are taken from the results computed by lectures/L12/analysis_L12.py (lectures/L12/cache and
lectures/L12/results: next-token probabilities of Qwen2.5-1.5B-Instruct, label probabilities on Financial PhraseBank and Twitter
financial news, few-shot and prompt-sensitivity labels, RAG ranks on ECB speeches, LLM-VaR on the S&P 500); no model is re-run.
The page logic is in interactive/L12.js. Run: python3 lectures/L12/build_interactive_L12.py [--no-data]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, play, charts, rnd  # noqa: E402

K = 12
RES = os.path.join(HERE, 'results')
CACHE = os.path.join(HERE, 'cache')

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('next-token', 'next_token', 'lm', 'nt'),
          ('few-shot', 'fewshot_curve', 'prompt', 'fs'),
          ('prompt-sensitivity', 'prompt_sensitivity', 'prompt', 'ps'),
          ('calibration', 'calibration', 'prompt', 'cb'),
          ('rag-recall', 'rag_recall', 'rag', 'rg'),
          ('llm-var', 'llm_var', 'var', 'lv')]

SECTIONS = {
 'en': dict(lm='Next-token prediction in a language model', prompt='Evaluation of prompt engineering',
            rag='Retrieval-augmented generation (RAG)', var='A language model as a risk estimator (LLM-VaR)'),
 'ro': dict(lm='Prezicerea următorului token într-un model de limbaj', prompt='Evaluarea formulării instrucțiunilor (prompt engineering)',
            rag='Generarea augmentată prin regăsire (RAG)', var='Un model de limbaj ca estimator al riscului (LLM-VaR)'),
}

L = {
 'en': dict(h1='Lecture 12 · Large language models (LLM) in finance and prompt engineering',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the outputs of the '
        'open models run for the lecture (Qwen2.5 with 0.5, 1.5 and 3 billion parameters) when you move a control. Nothing is invented: '
        'every probability, label and Value-at-Risk (VaR) estimate below was produced by the models on real data. The ▶ button moves the main '
        'parameter step by step. Under each figure: a readout, a question for the room and the Quantlet with the code of the chart.',
   src='Next token and RAG: speeches of the European Central Bank (ECB), 2022–2026. Sentiment: Financial PhraseBank (Malo et al., 2014; '
       'sentences with 100% annotator agreement, test set of 678) and Twitter financial news (Zeroshot, test sample of 900 tweets). '
       'Models: Qwen2.5-Instruct (Alibaba), run locally; label probabilities = probability mass of the first token of each label. '
       'LLM-VaR: daily log returns of the S&P 500 index, 2 January 2025 to 30 October 2026, window of 250 days, after Pele et al. (2026).'),
 'ro': dict(h1='Cursul 12 · Modele lingvistice mari (LLM) în finanțe și formularea instrucțiunilor',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser din ieșirile modelelor deschise rulate pentru '
        'curs (Qwen2.5 cu 0,5, 1,5 și 3 miliarde de parametri) cînd modificați un control. Nimic nu este inventat: fiecare probabilitate, '
        'etichetă și estimare a valorii expuse la risc (VaR) de mai jos a fost produsă de modele pe date reale. Butonul ▶ modifică parametrul '
        'principal pas cu pas. Sub fiecare figură: o sinteză numerică, o întrebare pentru sală și Quantlet-ul cu codul figurii.',
   src='Următorul token și RAG: discursurile Băncii Centrale Europene (BCE), 2022–2026. Sentimentul: Financial PhraseBank (Malo et al., 2014; '
       'propozițiile cu acord de 100% între adnotatori, setul de test de 678) și Twitter financial news (Zeroshot, eșantion de test de 900 '
       'de tweet-uri). Modelele: Qwen2.5-Instruct (Alibaba), rulate local; probabilitatea unei etichete = masa de probabilitate a primului '
       'său token. LLM-VaR: randamentele logaritmice zilnice ale indicelui S&P 500, 2 ianuarie 2025 – 30 octombrie 2026, fereastră de 250 '
       'de zile, după Pele et al. (2026).'),
}

MODELS = ['tfidf', 'zs_0.5B', 'zs_1.5B', 'zs_3B', 'fs_1.5B', 'fs_3B', 'cot_3B', 'probe']


def model_names(ro):
    return {'tfidf': 'TF-IDF + regresie logistică' if ro else 'TF-IDF + logistic regression',
            'zs_0.5B': 'Zero-shot, Qwen2.5-0.5B', 'zs_1.5B': 'Zero-shot, Qwen2.5-1.5B', 'zs_3B': 'Zero-shot, Qwen2.5-3B',
            'fs_1.5B': 'Few-shot (k = 4), Qwen2.5-1.5B', 'fs_3B': 'Few-shot (k = 4), Qwen2.5-3B',
            'cot_3B': 'Chain-of-thought, Qwen2.5-3B' if not ro else 'Raționament pas cu pas (chain-of-thought), Qwen2.5-3B',
            'probe': 'Embedding Qwen2.5-1.5B + regresie logistică' if ro else 'Qwen2.5-1.5B embedding + logistic regression'}


def widgets(lang):
    ro = lang == 'ro'
    ds = [('pb', 'Financial PhraseBank (știri despre companii)' if ro else 'Financial PhraseBank (company news)'),
          ('tw', 'Tweet-uri financiare' if ro else 'Finance tweets')]
    mn = model_names(ro)
    years = [('all', 'Toți anii (2022–2026)' if ro else 'All years (2022–2026)')] + [(str(y), str(y)) for y in range(2022, 2027)]
    return {
     'next-token': (slider('nt-T', 'Temperatura T' if ro else 'Temperature T', 0.1, 2.0, 0.05, 1.0)
                    + slider('nt-k', 'Top-k: cîți tokeni candidați păstrăm' if ro else 'Top-k: how many candidate tokens are kept', 1, 8, 1, 8)
                    + play('nt'), charts('nt-chart', cls='one')),
     'few-shot': (select('fs-m', 'Modelul' if ro else 'Model', [('1.5B', 'Qwen2.5-1.5B'), ('3B', 'Qwen2.5-3B')], '3B')
                  + slider('fs-k', 'Numărul de exemple în prompt, k' if ro else 'Examples in the prompt, k', 0, 4, 1, 2)
                  + radios('fs-s', 'Setul de exemple (extragerea aleatoare)' if ro else 'Set of examples (random draw)',
                           [(0, '1'), (1, '2'), (2, '3')], 0) + play('fs'),
                  charts('fs-curve', 'fs-conf')),
     'prompt-sensitivity': (select('ps-d', 'Setul de date' if ro else 'Data set', ds, 'pb')
                            + select('ps-m', 'Modelul (zero-shot)' if ro else 'Model (zero-shot)', [('1.5B', 'Qwen2.5-1.5B'), ('3B', 'Qwen2.5-3B')], '1.5B')
                            + radios('ps-i', 'Formularea instrucțiunii' if ro else 'Wording of the instruction', [(i, str(i + 1)) for i in range(5)], 0)
                            + play('ps'), charts('ps-bars', 'ps-conf')),
     'calibration': (select('cb-d', 'Setul de date' if ro else 'Data set', ds, 'pb')
                     + select('cb-m', 'Modelul' if ro else 'Model', [(m, mn[m]) for m in MODELS], 'zs_3B')
                     + slider('cb-t', 'Răspundem doar dacă probabilitatea etichetei alese este cel puțin' if ro
                              else 'Answer only if the probability of the chosen label is at least', 0.33, 0.99, 0.01, 0.33)
                     + play('cb'), charts('cb-rel', 'cb-cov')),
     'rag-recall': (toggles('rg-r', 'Metoda de regăsire' if ro else 'Retriever',
                            [('tfidf', 'TF-IDF'), ('bm25', 'BM25'), ('dense', 'Embedding dense (bge-small)' if ro else 'Dense embeddings (bge-small)'),
                             ('hybrid', 'Hibrid (BM25 + dense)' if ro else 'Hybrid (BM25 + dense)')])
                    + slider('rg-k', 'k: cîte discursuri returnează regăsirea' if ro else 'k: speeches returned by the retriever', 1, 20, 1, 5)
                    + select('rg-y', 'Anul discursurilor' if ro else 'Year of the speeches', years, 'all') + play('rg'),
                    charts('rg-chart', cls='one')),
     'llm-var': (radios('lv-a', 'Nivelul VaR (probabilitatea cozii)' if ro else 'VaR level (tail probability)', [('05', '5%'), ('01', '1%')], '05')
                 + toggles('lv-m', 'Metode' if ro else 'Methods',
                           [('llm', 'LLM-VaR (Qwen2.5-3B)'), ('hs', 'Simulare istorică' if ro else 'Historical simulation'),
                            ('garch', 'GARCH(1,1), t')], ['llm', 'garch'])
                 + slider('lv-s', 'Începutul perioadei de testare' if ro else 'Start of the backtest period', 0, 20, 1, 0) + play('lv'),
                 charts('lv-chart', cls='one')),
    }


def _n():
    return json.load(open(os.path.join(HERE, 'L12_numbers.json'), encoding='utf-8'))


QA = {
 'en': {
  'next-token': ('The sentence of an ECB (European Central Bank) speech of 30 October 2026 starts “The inflation impact is limited:”. At '
                 'temperature 1, the model gives 10.1% to “the”. What happens to this probability at temperature 0.5 and at temperature 2?',
                 'Dividing the logits by T sharpens the distribution when T < 1 and flattens it when T > 1. Over the whole vocabulary of '
                 '151,665 tokens the lecture measures 49.9% for “the” at T = 0.5 and only 0.4% at T = 2; the entropy rises from 1.74 to 6.20 '
                 'and 10.29 nats. The chart renormalises over the k most likely tokens, as top-k sampling does: with k = 1 the model always '
                 'writes “the” (greedy decoding). The word actually spoken, “prices”, is not among the eight most likely tokens. For '
                 'classification and extraction use T = 0; high temperatures are for brainstorming.'),
  'few-shot': ('Adding labelled examples to the prompt (few-shot) is supposed to help. Does the 3-billion-parameter model classify the '
               'Financial PhraseBank sentences better with 4 examples than with none?',
               'No. Zero-shot, Qwen2.5-3B reaches a macro-F1 of 92.2%; with k = 1, 2, 4 and 8 examples the mean over three random draws of '
               'examples is 89.5%, 90.4%, 92.1% and 91.3%, and with k = 1 the draws range from 87.5% to 91.1%. The smaller 1.5B model gains a '
               'little at k = 1 (85.1% → 87.3%) and loses at k = 8 (84.1%). Which examples you draw matters about as much as how many. '
               'Macro-F1 averages the F1 score (harmonic mean of precision and recall) of the three classes, so the rare negative class counts as much as the neutral one.'),
  'prompt-sensitivity': ('Five paraphrases of the same instruction, same model, same sentences. How much can the wording alone move the macro-F1 score?',
                         'A lot for the smaller model: on Financial PhraseBank, Qwen2.5-1.5B ranges from 72.8% to 87.0% (14.2 points) across the five '
                         'wordings, Qwen2.5-3B only from 88.5% to 92.2% (3.6 points). On finance tweets both are fragile: 34.3–47.6% for 1.5B and '
                         '65.5–78.4% for 3B. The confusion matrix shows where the errors go; a reported LLM score is a score for one prompt, '
                         'so test several wordings and report the range.'),
  'calibration': ('A classifier that abstains when unsure can hand the hard cases to a human analyst. If the zero-shot 3B model answers only '
                  'when its probability is high, does its accuracy on the answered sentences rise?',
                  'Yes: move the threshold and watch coverage fall while accuracy on the answered sentences rises. This works only if the '
                  'probabilities are calibrated, that is, if “90% sure” is right about 90% of the time. The lecture measures the expected '
                  'calibration error (ECE) on Financial PhraseBank: 4.7% for zero-shot 3B, 3.6% for few-shot 3B and 5.4% for TF-IDF (term '
                  'frequency–inverse document frequency) with logistic regression. Compare the 0.5B model: its reliability curve is far from '
                  'the diagonal, and a threshold buys much less.'),
  'rag-recall': ('The query is the title of an ECB speech; the retriever must find that speech among 400 (8,400 chunks of 200 words). Which '
                 'retriever puts the right speech first most often?',
                 'The hybrid one. Recall@1 is 35.2% for TF-IDF, 37.0% for BM25 (Best Matching 25, a keyword score), 44.0% for dense '
                 'embeddings and 50.7% for the hybrid that fuses BM25 and dense ranks; at k = 10 the four reach 75.0%, 78.0%, 80.0% and 83.2%. '
                 'Keywords and meaning fail on different queries, so combining them helps. With retrieval, the 3B model answers “who gave '
                 'this speech?” correctly for 81.2% of 160 recent speeches, against 13.8% from its own memory.'),
  'llm-var': ('A language model reads the last 250 daily returns of the S&P 500 as text and writes a number: tomorrow’s 5% Value-at-Risk. '
              'Over 459 trading days we expect about 23 violations. How does it compare with GARCH and historical simulation?',
              'At 5% the LLM-VaR has 28 violations (6.1%, Kupiec p = 0.295), historical simulation 23 (5.01%, p = 0.991) and GARCH(1,1) with '
              'Student-t errors 35 (7.63%, p = 0.016, rejected at 5%). At 1% the counts are 4, 6 and 9 for an expected 4.6. The Kupiec test '
              'asks whether the share of violations equals the VaR level; a p-value below 0.05 rejects the model. The LLM passes the '
              'count test, but its VaR barely reacts to volatility (correlation with GARCH 0.05): move the start of the period and see '
              'where its violations cluster.'),
 },
 'ro': {
  'next-token': ('Fraza unui discurs al BCE (Banca Centrală Europeană) din 30 octombrie 2026 începe cu „The inflation impact is limited:”. La '
                 'temperatura 1, modelul dă 10,1% cuvîntului „the”. Ce se întîmplă cu această probabilitate la temperatura 0,5 și la temperatura 2?',
                 'Împărțirea logiților la T ascute distribuția cînd T < 1 și o aplatizează cînd T > 1. Pe întregul vocabular de 151.665 de '
                 'tokeni, cursul măsoară 49,9% pentru „the” la T = 0,5 și doar 0,4% la T = 2; entropia crește de la 1,74 la 6,20 și 10,29 '
                 'nats. Figura renormalizează pe cei mai probabili k tokeni, ca eșantionarea top-k: cu k = 1 modelul scrie mereu „the” '
                 '(decodare lacomă). Cuvîntul rostit efectiv, „prices”, nu este printre cei opt tokeni cei mai probabili. Pentru clasificare '
                 'și extragere folosiți T = 0; temperaturile mari sînt pentru generarea de idei.'),
  'few-shot': ('Adăugarea unor exemple etichetate în prompt (few-shot) ar trebui să ajute. Clasifică modelul cu 3 miliarde de parametri mai '
               'bine propozițiile din Financial PhraseBank cu 4 exemple decît fără niciunul?',
               'Nu. Fără exemple (zero-shot), Qwen2.5-3B atinge un macro-F1 de 92,2%; cu k = 1, 2, 4 și 8 exemple, media pe trei extrageri '
               'aleatoare ale exemplelor este 89,5%, 90,4%, 92,1% și 91,3%, iar cu k = 1 extragerile variază între 87,5% și 91,1%. Modelul '
               'mai mic, de 1,5 miliarde, cîștigă puțin la k = 1 (85,1% → 87,3%) și pierde la k = 8 (84,1%). Contează ce exemple extrageți '
               'aproape cît contează cîte. Macro-F1 mediază scorul F1 (media armonică a preciziei și a sensibilității) celor trei clase, deci '
               'clasa negativă, rară, contează cît cea neutră.'),
  'prompt-sensitivity': ('Cinci reformulări ale aceleiași instrucțiuni, același model, aceleași propoziții. Cît poate muta formularea singură scorul macro-F1?',
                         'Mult pentru modelul mic: pe Financial PhraseBank, Qwen2.5-1.5B variază între 72,8% și 87,0% (14,2 puncte) de la o '
                         'formulare la alta, Qwen2.5-3B doar între 88,5% și 92,2% (3,6 puncte). Pe tweet-urile financiare ambele sînt '
                         'fragile: 34,3–47,6% pentru 1,5B și 65,5–78,4% pentru 3B. Matricea de confuzie arată unde merg erorile; un scor '
                         'raportat pentru un LLM este scorul unui singur prompt, deci testați mai multe formulări și raportați intervalul.'),
  'calibration': ('Un clasificator care se abține cînd nu este sigur poate trimite cazurile dificile unui analist. Dacă modelul zero-shot 3B '
                  'răspunde doar cînd probabilitatea este mare, crește acuratețea pe propozițiile la care răspunde?',
                  'Da: mutați pragul și urmăriți cum scade acoperirea, în timp ce acuratețea pe propozițiile cu răspuns crește. Aceasta '
                  'funcționează numai dacă probabilitățile sînt calibrate, adică dacă „90% sigur” are dreptate în circa 90% din cazuri. '
                  'Cursul măsoară eroarea de calibrare așteptată (ECE) pe Financial PhraseBank: 4,7% pentru zero-shot 3B, 3,6% pentru few-shot '
                  '3B și 5,4% pentru TF-IDF (frecvența termenului – frecvența inversă în documente) cu regresie logistică. Comparați modelul '
                  '0,5B: curba sa de fiabilitate este departe de diagonală, iar un prag aduce mult mai puțin.'),
  'rag-recall': ('Interogarea este titlul unui discurs al BCE; regăsirea trebuie să găsească acel discurs printre 400 (8.400 de fragmente de '
                 '200 de cuvinte). Ce metodă pune cel mai des discursul corect pe primul loc?',
                 'Cea hibridă. Recall@1 este 35,2% pentru TF-IDF, 37,0% pentru BM25 (Best Matching 25, un scor pe cuvinte-cheie), 44,0% '
                 'pentru embedding-urile dense și 50,7% pentru metoda hibridă, care combină rangurile BM25 și dense; la k = 10 cele patru '
                 'ajung la 75,0%, 78,0%, 80,0% și 83,2%. Cuvintele-cheie și sensul greșesc la interogări diferite, de aceea combinarea lor '
                 'ajută. Cu regăsire, modelul 3B răspunde corect la „cine a ținut acest discurs?” pentru 81,2% din 160 de discursuri '
                 'recente, față de 13,8% din propria memorie.'),
  'llm-var': ('Un model de limbaj citește ca text ultimele 250 de randamente zilnice ale S&P 500 și scrie un număr: valoarea expusă la risc '
              '(VaR) de 5% pentru ziua următoare. Pe 459 de zile de tranzacționare ne așteptăm la circa 23 de depășiri. Cum se compară cu '
              'GARCH și cu simularea istorică?',
              'La 5%, LLM-VaR are 28 de depășiri (6,1%, p Kupiec = 0,295), simularea istorică 23 (5,01%, p = 0,991), iar GARCH(1,1) cu erori '
              'Student-t 35 (7,63%, p = 0,016, respins la 5%). La 1%, numerele sînt 4, 6 și 9, pentru 4,6 așteptate. Testul Kupiec verifică '
              'dacă ponderea depășirilor este egală cu nivelul VaR; o valoare p sub 0,05 respinge modelul. LLM-ul trece testul numărului '
              'de depășiri, dar VaR-ul său aproape nu reacționează la volatilitate (corelația cu GARCH este 0,05): mutați începutul '
              'perioadei și vedeți unde se grupează depășirile.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 prob: 'Probability (%)', token: 'Candidate next token', ntTitle: (T, k) => `Next-token probabilities after “The inflation impact is limited:”, T = ${T}, top-${k}`,
 nt: (T, k, top, p, h) => `T = ${T}, top-${k}: the model writes “${top}” with probability ${p}; entropy of the sampled distribution ${h} nats.`,
 nt2: 'Whole vocabulary (151,665 tokens), measured in the lecture: “the” has 49.9% at T = 0.5, 10.1% at T = 1 and 0.4% at T = 2. The word actually spoken: “prices”.',
 nt3: 'The bars renormalise the k most likely tokens, as top-k sampling does; at T = 1 and k = 8 the eight tokens hold together ',
 f1: 'Macro-F1 (%)', kAxis: 'Examples in the prompt, k', seed: s => `draw ${s}`, mean: 'Mean of the three draws',
 truth: 'True label', pred: 'Predicted label', confT: (t) => `Confusion matrix, ${t}`,
 lab: { pb: ['positive', 'negative', 'neutral'], tw: ['bullish', 'bearish', 'neutral'] },
 fs: (m, k, s, f1, acc, lo, hi) => `Qwen2.5-${m}, k = ${k}, draw ${s}: macro-F1 ${f1}, accuracy ${acc}. Over the three draws: ${lo} to ${hi}.`,
 fs0: f1 => `Zero-shot (k = 0) for comparison: macro-F1 ${f1}.`,
 instr: 'Instruction', psBars: (d, m) => `Macro-F1 by wording of the instruction, ${d}, Qwen2.5-${m}`,
 ps: (i, f1, lo, hi) => `Wording ${i}: macro-F1 ${f1}; the five wordings range from ${lo} to ${hi}.`,
 psText: t => `Instruction given to the model: “${t}”`,
 ds: { pb: 'Financial PhraseBank', tw: 'finance tweets' },
 conf: 'Probability of the chosen label', acc: 'Accuracy (%)', cov: 'Coverage: share of items answered (%)',
 relT: 'Reliability diagram (10 bins)', covT: 'Accuracy of the answered items vs coverage', diag: 'Perfect calibration', model: 'Model',
 cb: (ece, f1, acc) => `Expected calibration error (ECE) ${ece}; on all items macro-F1 ${f1}, accuracy ${acc}.`,
 cb2: (t, n, cov, acc) => `Threshold ${t}: the model answers ${n} items (coverage ${cov}) with accuracy ${acc}; the rest go to an analyst.`,
 recall: 'Recall@k (%)', kRet: 'k (speeches returned)', rgT: (y, n) => `Share of queries whose speech is in the top k, ${y} (${n} speeches)`,
 allYears: 'all years', rg: (nm, r, mrr) => `${nm}: recall@k ${r}, mean reciprocal rank (MRR) ${mrr}.`,
 ret: { tfidf: 'TF-IDF', bm25: 'BM25', dense: 'Dense', hybrid: 'Hybrid' },
 ret0: 'Choose at least one retriever.',
 ret1: 'Answers to “who gave this speech?” (160 speeches, 2025–2026, Qwen2.5-3B): 13.8% correct from memory, 81.2% with retrieval (hybrid, top 3), 98.8% with the right speech given.',
 ret_: 'Return (%)', retS: 'S&P 500 daily log return', viol: 'Violation',
 vm: { llm: 'LLM-VaR', hs: 'Historical simulation', garch: 'GARCH(1,1), t' },
 lvT: (a, d0, d1) => `S&P 500 daily returns and ${a} VaR, ${d0} to ${d1}`,
 lv: (nm, v, e, rt, p, ok) => `${nm}: ${window.NNI.cnt(v, 'violation', 'violations')} (expected ${e}), rate ${rt}; Kupiec p = ${p} ${ok ? '(not rejected at 5%)' : '(rejected at 5%)'}.`,
 lv0: 'Choose at least one method.'
};
""",
 'ro': r"""window.LK_TEXT = {
 prob: 'Probabilitatea (%)', token: 'Tokenul candidat următor', ntTitle: (T, k) => `Probabilitățile următorului token după „The inflation impact is limited:”, T = ${T}, top-${k}`,
 nt: (T, k, top, p, h) => `T = ${T}, top-${k}: modelul scrie „${top}” cu probabilitatea ${p}; entropia distribuției eșantionate ${h} nats.`,
 nt2: 'Pe întregul vocabular (151.665 de tokeni), măsurat în curs: „the” are 49,9% la T = 0,5, 10,1% la T = 1 și 0,4% la T = 2. Cuvîntul rostit efectiv: „prices”.',
 nt3: 'Barele renormalizează cei mai probabili k tokeni, ca eșantionarea top-k; la T = 1 și k = 8, cei opt tokeni au împreună ',
 f1: 'Macro-F1 (%)', kAxis: 'Exemple în prompt, k', seed: s => `extragerea ${s}`, mean: 'Media celor trei extrageri',
 truth: 'Eticheta reală', pred: 'Eticheta prezisă', confT: (t) => `Matricea de confuzie, ${t}`,
 lab: { pb: ['pozitiv', 'negativ', 'neutru'], tw: ['optimist (bullish)', 'pesimist (bearish)', 'neutru'] },
 fs: (m, k, s, f1, acc, lo, hi) => `Qwen2.5-${m}, k = ${k}, extragerea ${s}: macro-F1 ${f1}, acuratețea ${acc}. Pe cele trei extrageri: între ${lo} și ${hi}.`,
 fs0: f1 => `Fără exemple (k = 0), pentru comparație: macro-F1 ${f1}.`,
 instr: 'Instrucțiunea', psBars: (d, m) => `Macro-F1 pe formulări ale instrucțiunii, ${d}, Qwen2.5-${m}`,
 ps: (i, f1, lo, hi) => `Formularea ${i}: macro-F1 ${f1}; cele cinci formulări variază între ${lo} și ${hi}.`,
 psText: t => `Instrucțiunea dată modelului (în engleză, ca în experiment): „${t}”`,
 ds: { pb: 'Financial PhraseBank', tw: 'tweet-uri financiare' },
 conf: 'Probabilitatea etichetei alese', acc: 'Acuratețea (%)', cov: 'Acoperirea: ponderea elementelor cu răspuns (%)',
 relT: 'Diagrama de fiabilitate (10 intervale)', covT: 'Acuratețea pe elementele cu răspuns față de acoperire', diag: 'Calibrare perfectă', model: 'Modelul',
 cb: (ece, f1, acc) => `Eroarea de calibrare așteptată (ECE) ${ece}; pe toate elementele, macro-F1 ${f1}, acuratețea ${acc}.`,
 cb2: (t, n, cov, acc) => `Pragul ${t}: modelul răspunde la ${n} (acoperire ${cov}) cu acuratețea ${acc}; restul merg la un analist.`,
 recall: 'Recall@k (%)', kRet: 'k (discursuri returnate)', rgT: (y, n) => `Ponderea interogărilor al căror discurs este între primele k, ${y} (${n})`,
 allYears: 'toți anii', rg: (nm, r, mrr) => `${nm}: recall@k ${r}, rangul reciproc mediu (MRR) ${mrr}.`,
 ret: { tfidf: 'TF-IDF', bm25: 'BM25', dense: 'Dense', hybrid: 'Hibrid' },
 ret0: 'Alegeți cel puțin o metodă de regăsire.',
 ret1: 'Răspunsuri la „cine a ținut acest discurs?” (160 de discursuri, 2025–2026, Qwen2.5-3B): 13,8% corecte din memorie, 81,2% cu regăsire (hibrid, primele 3), 98,8% cu discursul corect furnizat.',
 ret_: 'Randamentul (%)', retS: 'Randamentul logaritmic zilnic S&P 500', viol: 'Depășire',
 vm: { llm: 'LLM-VaR', hs: 'Simulare istorică', garch: 'GARCH(1,1), t' },
 lvT: (a, d0, d1) => `Randamentele zilnice S&P 500 și VaR de ${a}, ${d0} – ${d1}`,
 lv: (nm, v, e, rt, p, ok) => `${nm}: ${window.NNI.cnt(v, 'depășire', 'depășiri')} (așteptate ${e}), rata ${rt}; p Kupiec = ${p} ${ok ? '(nu este respins la 5%)' : '(respins la 5%)'}.`,
 lv0: 'Alegeți cel puțin o metodă.'
};
""",
}


# ---------------------------------------------------------------- data (outputs of the models run in analysis_L12.py)
def data():
    import numpy as np
    import pandas as pd
    out = {}
    # 1. next token, Qwen2.5-1.5B-Instruct, top-8 probabilities at T = 1 (exact ratios: p_i(T) ∝ p_i(1)^(1/T))
    nt = json.load(open(os.path.join(CACHE, 'next_token.json'), encoding='utf-8'))
    out['nt'] = {'tok': nt['1.0']['tok'], 'p1': rnd(nt['1.0']['p'], 6),
                 'ent': {T: rnd(nt[T]['entropy'], 2) for T in ('0.5', '1.0', '2.0')}}
    # 2.-4. sentiment: labels as index strings (0, 1, 2 in the order of LAB), confidence of the chosen label
    LAB = {'pb': ['positive', 'negative', 'neutral'], 'tw': ['bullish', 'bearish', 'neutral']}
    enc = lambda ds, s: ''.join(str(LAB[ds].index(v)) for v in s)  # noqa: E731
    fs = pd.read_csv(os.path.join(RES, 'fewshot_labels_pb.csv'))
    out['fs'] = {'y': enc('pb', fs['label']), **{c: enc('pb', fs[c]) for c in fs.columns if c != 'label'}}
    out['ps'] = {}
    for ds in ('pb', 'tw'):
        s = pd.read_csv(os.path.join(RES, f'sensitivity_labels_{ds}.csv'))
        out['ps'][ds] = {'y': enc(ds, s['label']), **{c: enc(ds, s[c]) for c in s.columns if c != 'label'}}
    # instructions, as given to the model (analysis_L12.py, INSTR)
    src = open(os.path.join(HERE, 'analysis_L12.py'), encoding='utf-8').read()
    i0 = src.index('INSTR = {')
    ns = {}
    exec(src[i0:src.index('}\n', i0) + 2], ns)
    out['ps']['instr'] = ns['INSTR']
    out['cb'] = {}
    for ds in ('pb', 'tw'):
        p = pd.read_csv(os.path.join(RES, f'predictions_{ds}.csv'))
        d = {'y': enc(ds, p['label'])}
        for m in MODELS:
            P = p[[f'{m}_{l}' for l in LAB[ds]]].values
            d[m] = {'pred': ''.join(map(str, P.argmax(1))), 'conf': rnd(P.max(1), 3)}
        out['cb'][ds] = d
    # 5. RAG: rank of the correct speech for each retriever
    rk = pd.read_csv(os.path.join(RES, 'rag_ranks.csv'))
    out['rg'] = {'year': [int(str(x)[:4]) for x in rk['date']], **{m: rk[f'rank_{m}'].astype(int).tolist() for m in ('tfidf', 'bm25', 'dense', 'hybrid')}}
    # 6. LLM-VaR on the S&P 500
    v = pd.read_csv(os.path.join(RES, 'llmvar_sp500.csv'))
    out['lv'] = {'date': v['date'].tolist(), 'r': rnd(v['r'].values, 3),
                 **{c: rnd(v[c].values, 3) for c in ('llm_05', 'hs_05', 'garch_05', 'llm_01', 'hs_01', 'garch_01')}}
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
