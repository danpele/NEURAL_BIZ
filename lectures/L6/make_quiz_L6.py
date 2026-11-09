"""
make_quiz_L6.py -- writes assets/quizzes/l6.js (quiz bank of Lecture 6, EN + RO) with every number from L6_numbers.json.
Format: NN_DATA.quizzes['l6'] = {draw: 20, questions: [{correct: 0-3, en: {...}, ro: {...}}]}; explanations never name a letter.
Run:  python3 lectures/L6/make_quiz_L6.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = json.load(open(os.path.join(HERE, "L6_numbers.json")))


def get(key):
    v = N
    for p in key.split("."):
        v = v[int(p)] if isinstance(v, list) else v[p]
    return v


def fmt(v, spec, lang):
    if isinstance(v, str):
        return v
    s = format(v, spec) if spec else str(v)
    if lang == "ro":
        s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s.replace("-", "−") if s.startswith("-") else s


def R(text, lang):
    return re.sub(r"\{([A-Za-z][\w.]*)(?:\|([^{}]*))?\}", lambda m: fmt(get(m.group(1)), m.group(2), lang), text)


Q = []


def q(correct, en, ro):
    """en/ro = (title, text, [4 options], correctExplanation, incorrectExplanation)"""
    Q.append({"correct": correct,
              "en": dict(zip(["title", "text", "options", "correctExplanation", "incorrectExplanation"], en)),
              "ro": dict(zip(["title", "text", "options", "correctExplanation", "incorrectExplanation"], ro))})


q(0, ("Tokens and numbers", "Our tokeniser maps every number to the token <num>. What is lost?",
      ["Comparisons such as ‘down from 12 to 10’ that carry sentiment in finance", "Nothing at all", "The ability to use logistic regression", "The vocabulary of the ECB speeches"],
      "Numbers often carry the sentiment of a financial sentence; one token for all of them keeps the vocabulary small but hides the direction of a change.",
      "Collapsing numbers is a trade-off: it reduces the vocabulary and the noise, while the information in the size and direction of the numbers is lost."),
  ("Unități lexicale și numere", "Tokenizatorul nostru transformă fiecare număr în unitatea <num>. Ce se pierde?",
   ["Comparațiile de tipul „down from 12 to 10”, care poartă sentiment în finanțe", "Nimic", "Posibilitatea de a folosi regresia logistică", "Vocabularul discursurilor BCE"],
   "Numerele poartă adesea sentimentul unei fraze financiare; o singură unitate pentru toate menține vocabularul mic, dar ascunde direcția unei schimbări.",
   "Comprimarea numerelor este un compromis: reduce vocabularul și zgomotul, dar se pierde informația din mărimea și direcția numerelor."))
q(1, ("Stop words", "Why can removing English stop words hurt a financial sentiment model?",
      ["Stop words are rare", "Lists of stop words contain ‘up’, ‘down’, ‘not’, which carry sentiment", "Logistic regression needs stop words to converge", "Stop words are numbers"],
      "Generic stop-word lists were built for retrieval, not for sentiment; in finance the direction words are among them.",
      "Stop words are the most frequent words, not rare ones; the issue is that some of them change the meaning of the sentence."),
  ("Cuvintele de legătură", "De ce poate dăuna eliminarea cuvintelor de legătură din engleză unui model de sentiment financiar?",
   ["Cuvintele de legătură sînt rare", "Listele de cuvinte de legătură conțin „up”, „down”, „not”, care poartă sentiment", "Regresia logistică are nevoie de ele pentru a converge", "Cuvintele de legătură sînt numere"],
   "Listele generale de cuvinte de legătură au fost construite pentru căutare, nu pentru sentiment; în finanțe, printre ele se află cuvintele de direcție.",
   "Cuvintele de legătură sînt cele mai frecvente, nu rare; problema este că unele dintre ele schimbă sensul frazei."))
q(2, ("Zipf's law", "In the ECB speeches the rank–frequency slope is {corpus.zipf_slope|.3f}. What does Zipf's law imply for a text model?",
      ["All words are equally frequent", "The vocabulary is small", "A few words are very frequent and most are rare: {corpus.hapax_pct|.1f}% of the distinct words occur once", "Frequency grows with rank"],
      "With f(r) ≈ C r^(−a), the ten most frequent words make {corpus.top10_share_pct|.1f}% of the tokens, while the long tail cannot be learned from one corpus.",
      "Frequency falls with rank as a power law; the vocabulary of {corpus.n_types|,} distinct words is large, and most of it is rare."),
  ("Legea lui Zipf", "În discursurile BCE, panta rang–frecvență este {corpus.zipf_slope|.3f}. Ce implică legea lui Zipf pentru un model de text?",
   ["Toate cuvintele sînt la fel de frecvente", "Vocabularul este mic", "Puține cuvinte sînt foarte frecvente, iar cele mai multe sînt rare: {corpus.hapax_pct|.1f}% dintre cuvintele distincte apar o singură dată", "Frecvența crește cu rangul"],
   "Cu f(r) ≈ C r^(−a), cele mai frecvente zece cuvinte reprezintă {corpus.top10_share_pct|.1f}% din unitățile lexicale, iar coada lungă nu poate fi învățată dintr-un singur corpus.",
   "Frecvența scade cu rangul după o lege putere; vocabularul de {corpus.n_types|,} de cuvinte distincte este mare și în cea mai mare parte rar."))
q(3, ("Bag of words", "What does a bag-of-words representation ignore?",
      ["The vocabulary", "The number of occurrences of a word", "The words themselves", "The order of the words"],
      "x_i = (c_i1, …, c_iV) keeps counts only; ‘profit rose, costs fell’ and ‘costs rose, profit fell’ have the same vector. Bigrams restore some local order.",
      "BoW is built from the vocabulary and the counts; the order is exactly what it discards."),
  ("Sacul de cuvinte", "Ce ignoră reprezentarea de tip sac de cuvinte?",
   ["Vocabularul", "Numărul de apariții ale unui cuvînt", "Cuvintele înseși", "Ordinea cuvintelor"],
   "x_i = (c_i1, …, c_iV) păstrează doar frecvențele; „profit rose, costs fell” și „costs rose, profit fell” au același vector. Bigramele recuperează o parte din ordinea locală.",
   "BoW este construit din vocabular și frecvențe; ordinea este exact ceea ce elimină."))
q(0, ("Smoothed idf", "A word appears in every one of the n training sentences. What is its smoothed idf, ln((1 + n)/(1 + n_j)) + 1?",
      ["1, the minimum", "0", "ln n", "n"],
      "With n_j = n the logarithm is zero, so idf = 1: the word gets the smallest weight and barely changes the normalised vector.",
      "The +1 keeps the idf positive, so it cannot be 0; ln n would be the idf of a word that appears in about one document."),
  ("Idf netezit", "Un cuvînt apare în fiecare dintre cele n fraze de antrenare. Cît este idf-ul netezit, ln((1 + n)/(1 + n_j)) + 1?",
   ["1, minimul", "0", "ln n", "n"],
   "Cu n_j = n, logaritmul este zero, deci idf = 1: cuvîntul primește ponderea cea mai mică și aproape nu modifică vectorul normalizat.",
   "Termenul +1 menține idf pozitiv, deci nu poate fi 0; ln n ar fi idf-ul unui cuvînt care apare în aproximativ un document."))
q(1, ("Sublinear tf", "Why does our TF-IDF use 1 + ln c instead of the raw count c?",
      ["To make the vectors sparse", "So that a word repeated ten times does not weigh ten times more than once", "Because logistic regression requires logarithms", "To remove stop words"],
      "The logarithm dampens repetitions: the tenth occurrence adds much less information than the first.",
      "Sparsity comes from the vocabulary, not from the logarithm; the transformation is about diminishing returns of repetitions."),
  ("Tf sublinear", "De ce folosește TF-IDF-ul nostru 1 + ln c în loc de frecvența brută c?",
   ["Pentru a face vectorii rari", "Pentru ca un cuvînt repetat de zece ori să nu cîntărească de zece ori mai mult decît o dată", "Pentru că regresia logistică cere logaritmi", "Pentru a elimina cuvintele de legătură"],
   "Logaritmul atenuează repetițiile: a zecea apariție aduce mult mai puțină informație decît prima.",
   "Raritatea vine din vocabular, nu din logaritm; transformarea exprimă scăderea utilității repetițiilor."))
q(2, ("A learned lexicon", "Which words pushed most towards a positive label in our bag-of-words logistic regression on PhraseBank?",
      ["the, of, and", "{words.neg}", "{words.pos}", "bank, euro, inflation"],
      "The difference β_positive − β_negative ranks the words; the top ones are {words.pos}, and the most negative {words.neg}.",
      "Function words have small coefficients; the negative list is the other end of the same ranking."),
  ("Un dicționar învățat", "Ce cuvinte au împins cel mai mult spre eticheta pozitivă în regresia logistică pe sacul de cuvinte, pe PhraseBank?",
   ["the, of, and", "{words.neg}", "{words.pos}", "bank, euro, inflation"],
   "Diferența β_pozitiv − β_negativ ordonează cuvintele; primele sînt {words.pos}, iar cele mai negative {words.neg}.",
   "Cuvintele funcționale au coeficienți mici; lista negativă este celălalt capăt al aceleiași ordonări."))
q(3, ("One-hot vectors", "What is the cosine similarity of the one-hot vectors of ‘inflation’ and ‘prices’?",
      ["1", "0.5", "It depends on the corpus", "0: one-hot vectors are orthogonal"],
      "One-hot vectors share no non-zero coordinate; dense embeddings give cos(inflation, prices) = {cos.inflation_prices|.2f} in our ECB vectors.",
      "One-hot coding puts each word on its own axis, so any two different words are at 90°; only learned dense vectors can express similarity."),
  ("Vectorii one-hot", "Cît este similaritatea cosinus a vectorilor one-hot ai cuvintelor „inflation” și „prices”?",
   ["1", "0,5", "Depinde de corpus", "0: vectorii one-hot sînt ortogonali"],
   "Vectorii one-hot nu au nicio coordonată nenulă comună; vectorii denși dau cos(inflation, prices) = {cos.inflation_prices|.2f} în vectorii noștri BCE.",
   "Codificarea one-hot pune fiecare cuvînt pe propria axă, deci oricare două cuvinte diferite sînt la 90°; doar vectorii denși învățați pot exprima asemănarea."))
q(0, ("Embedding layer", "In code, how is the embedding e_w = Eᵀ o_w computed?",
      ["As a lookup of row w of E", "As a full matrix product with a one-hot vector", "By a convolution", "By solving a linear system"],
      "Multiplying by a one-hot vector selects one row; libraries implement it as an index lookup, which is far cheaper.",
      "The product and the lookup give the same result; the lookup avoids multiplying by thousands of zeros."),
  ("Stratul de embedding", "În cod, cum se calculează vectorul e_w = Eᵀ o_w?",
   ["Ca o căutare a rîndului w din E", "Ca un produs complet cu un vector one-hot", "Printr-o convoluție", "Prin rezolvarea unui sistem liniar"],
   "Înmulțirea cu un vector one-hot selectează un rînd; bibliotecile o implementează ca o indexare, mult mai ieftină.",
   "Produsul și căutarea dau același rezultat; căutarea evită înmulțirea cu mii de zerouri."))
q(1, ("Skip-gram", "What does the skip-gram model predict?",
      ["The centre word from the average of its context", "Each context word in a window from the centre word", "The next sentence", "The sentiment of the sentence"],
      "max Σ_t Σ_j log p(w_(t+j) | w_t); the mirror variant, CBOW, predicts the centre from the context.",
      "Predicting the centre from the averaged context is CBOW; skip-gram goes from the word to its neighbours."),
  ("Skip-gram", "Ce prezice modelul skip-gram?",
   ["Cuvîntul central din media contextului său", "Fiecare cuvînt de context dintr-o fereastră, pornind de la cuvîntul central", "Fraza următoare", "Sentimentul frazei"],
   "max Σ_t Σ_j log p(w_(t+j) | w_t); varianta în oglindă, CBOW, prezice cuvîntul central din context.",
   "Prezicerea cuvîntului central din media contextului este CBOW; skip-gram merge de la cuvînt la vecinii lui."))
q(2, ("Negative sampling", "Why does word2vec use negative sampling instead of the full softmax?",
      ["The softmax is not differentiable", "Negative sampling gives exact probabilities", "The softmax denominator sums over all {w2v.V|,} words for every pair; K = {w2v.neg} binary contrasts are far cheaper", "To avoid using a window"],
      "SGNS turns each pair into one logistic regression for the real context word and K for noise words: cost O(K) instead of O(V).",
      "The full softmax is differentiable and exact but expensive; negative sampling trades exact probabilities for speed."),
  ("Eșantionarea negativă", "De ce folosește word2vec eșantionarea negativă în locul softmax-ului complet?",
   ["Softmax-ul nu este derivabil", "Eșantionarea negativă dă probabilități exacte", "Numitorul softmax adună peste toate cele {w2v.V|,} de cuvinte pentru fiecare pereche; K = {w2v.neg} contraste binare sînt mult mai ieftine", "Pentru a evita folosirea unei ferestre"],
   "SGNS transformă fiecare pereche într-o regresie logistică pentru cuvîntul real de context și K pentru cuvintele de zgomot: cost O(K) în loc de O(V).",
   "Softmax-ul complet este derivabil și exact, dar scump; eșantionarea negativă schimbă probabilitățile exacte pe viteză."))
q(3, ("Noise distribution", "Why are the negative words drawn in proportion to f(w)^(3/4)?",
      ["To draw only rare words", "To match the test set", "Because 3/4 makes the loss convex", "To flatten the distribution: rare words are drawn more often than their frequency, very frequent ones less"],
      "The power 3/4 was found to give more useful contrasts than the raw frequency, which would mostly sample ‘the’ and ‘of’.",
      "The exponent does not make the problem convex and is unrelated to the test set; it moderates, without removing, the advantage of frequent words."),
  ("Distribuția de zgomot", "De ce sînt extrase cuvintele negative proporțional cu f(w)^(3/4)?",
   ["Pentru a extrage doar cuvinte rare", "Pentru a se potrivi cu setul de test", "Pentru că 3/4 face pierderea convexă", "Pentru a aplatiza distribuția: cuvintele rare sînt extrase mai des decît frecvența lor, iar cele foarte frecvente mai rar"],
   "Puterea 3/4 s-a dovedit că dă contraste mai utile decît frecvența brută, care ar extrage mai ales „the” și „of”.",
   "Exponentul nu face problema convexă și nu are legătură cu setul de test; moderează, fără să elimine, avantajul cuvintelor frecvente."))
q(0, ("SGNS and PMI", "What do skip-gram with negative sampling vectors approximate at the optimum?",
      ["u_oᵀ v_w = PMI(w, o) − ln K, a shifted pointwise mutual information", "The raw co-occurrence counts", "The tf-idf weights", "Random projections"],
      "Levy and Goldberg (2014) show SGNS implicitly factorises the PMI matrix shifted by ln K: word2vec and count-based methods are close relatives.",
      "The result links the predictive objective to co-occurrence statistics through the log ratio P(w,o)/(P(w)P(o)), not to raw counts or tf-idf."),
  ("SGNS și PMI", "Ce aproximează la optim vectorii skip-gram cu eșantionare negativă?",
   ["u_oᵀ v_w = PMI(w, o) − ln K, o informație mutuală punctuală deplasată", "Frecvențele brute de co-apariție", "Ponderile tf-idf", "Proiecții aleatoare"],
   "Levy și Goldberg (2014) arată că SGNS factorizează implicit matricea PMI deplasată cu ln K: word2vec și metodele bazate pe frecvențe sînt rude apropiate.",
   "Rezultatul leagă obiectivul predictiv de statisticile de co-apariție prin raportul logaritmic P(w,o)/(P(w)P(o)), nu de frecvențele brute sau de tf-idf."))
q(1, ("GloVe weighting", "In GloVe, what does the weight f(x) = min{1, (x/x_max)^α} do?",
      ["It removes all rare pairs", "It down-weights rare, noisy co-occurrences and caps the weight of very frequent ones at 1", "It normalises the vectors", "It selects the window size"],
      "Pairs with few co-occurrences are noisy and get a small weight; above x_max = {glove.xmax} the weight is 1, so ‘the’-type pairs do not dominate. {glove.share_below_xmax|.2f}% of our pairs are below x_max.",
      "Rare pairs are kept with small weights, not removed; the function acts on the loss, not on the vectors or the window."),
  ("Ponderarea GloVe", "În GloVe, ce face ponderea f(x) = min{1, (x/x_max)^α}?",
   ["Elimină toate perechile rare", "Reduce ponderea co-aparițiilor rare și zgomotoase și plafonează la 1 ponderea celor foarte frecvente", "Normalizează vectorii", "Alege dimensiunea ferestrei"],
   "Perechile cu puține co-apariții sînt zgomotoase și primesc o pondere mică; peste x_max = {glove.xmax} ponderea este 1, deci perechile de tipul „the” nu domină. {glove.share_below_xmax|.2f}% dintre perechile noastre sînt sub x_max.",
   "Perechile rare sînt păstrate cu ponderi mici, nu eliminate; funcția acționează asupra pierderii, nu asupra vectorilor sau a ferestrei."))
q(2, ("GloVe target", "What does GloVe fit with w_iᵀw̃_j + b_i + b̃_j?",
      ["The probability of a context word", "The tf-idf of the pair", "The logarithm of the co-occurrence count, ln X_ij", "The sentiment label"],
      "J = Σ f(X_ij)(w_iᵀw̃_j + b_i + b̃_j − ln X_ij)²: a weighted least-squares fit of the log counts, over non-zero pairs only.",
      "GloVe is a regression on log counts, computed once from the corpus; probabilities are the objective of word2vec."),
  ("Ținta GloVe", "Ce aproximează GloVe cu w_iᵀw̃_j + b_i + b̃_j?",
   ["Probabilitatea unui cuvînt de context", "Tf-idf-ul perechii", "Logaritmul frecvenței de co-apariție, ln X_ij", "Eticheta de sentiment"],
   "J = Σ f(X_ij)(w_iᵀw̃_j + b_i + b̃_j − ln X_ij)²: o aproximare prin cele mai mici pătrate ponderate a logaritmilor frecvențelor, doar pe perechile nenule.",
   "GloVe este o regresie pe logaritmii frecvențelor, calculate o singură dată din corpus; probabilitățile sînt obiectivul lui word2vec."))
q(3, ("Static embeddings", "Why can a static embedding not separate ‘interest rate’ from ‘of interest to investors’?",
      ["The corpus is too small", "Cosine similarity is undefined", "‘interest’ is out of vocabulary", "It has one vector per word type, an average over all its uses"],
      "Polysemy is collapsed into one vector; contextual models (Lecture 7) produce a different vector in each sentence.",
      "A larger corpus would still give one vector per word; ‘interest’ is in the ECB vocabulary, and cosine similarity is well defined."),
  ("Vectori statici", "De ce nu poate un vector static să deosebească „interest rate” de „of interest to investors”?",
   ["Corpusul este prea mic", "Similaritatea cosinus nu este definită", "„interest” este în afara vocabularului", "Are un singur vector pentru fiecare cuvînt, o medie a tuturor utilizărilor"],
   "Polisemia este comprimată într-un singur vector; modelele contextuale (cursul 7) produc un vector diferit în fiecare frază.",
   "Un corpus mai mare ar da tot un singur vector pentru fiecare cuvînt; „interest” se află în vocabularul BCE, iar similaritatea cosinus este bine definită."))
q(0, ("Coverage", "Our ECB vocabulary covers {oov.pb_cover|.1f}% of PhraseBank tokens and {oov.tw_cover|.1f}% of tweet tokens. What does the gap say?",
      ["Tweets use more words outside the central-bank vocabulary (tickers, slang), so pre-trained ECB vectors transfer less", "Tweets are longer", "The tokeniser fails on tweets", "PhraseBank is a subset of the speeches"],
      "Out-of-vocabulary tokens get no pre-trained vector; the further the target text is from the training corpus, the less the embeddings help.",
      "Coverage measures the overlap of vocabularies, not length or tokenisation errors; PhraseBank comes from company news, not from the speeches."),
  ("Acoperirea", "Vocabularul nostru BCE acoperă {oov.pb_cover|.1f}% din unitățile lexicale din PhraseBank și {oov.tw_cover|.1f}% din cele din tweet-uri. Ce arată diferența?",
   ["Tweet-urile folosesc mai multe cuvinte din afara vocabularului băncii centrale (simboluri bursiere, argou), deci vectorii BCE preantrenați se transferă mai greu", "Tweet-urile sînt mai lungi", "Tokenizatorul eșuează pe tweet-uri", "PhraseBank este o parte a discursurilor"],
   "Unitățile din afara vocabularului nu primesc vector preantrenat; cu cît textul țintă este mai departe de corpusul de antrenare, cu atît vectorii ajută mai puțin.",
   "Acoperirea măsoară suprapunerea vocabularelor, nu lungimea sau erorile de tokenizare; PhraseBank provine din știri despre companii, nu din discursuri."))
q(1, ("Macro-F1", "Why do we report macro-F1 and not accuracy on PhraseBank?",
      ["Accuracy cannot be computed for three classes", "‘Neutral’ is {data.pb_neu_pct|.1f}% of the data, so always saying neutral gets a high accuracy; macro-F1 weights each class equally", "Macro-F1 is always higher", "Accuracy needs probabilities"],
      "The majority class reaches an accuracy of {sent.pb.majority.acc|.2f}% but a macro-F1 of only {sent.pb.majority.f1|.2f}%.",
      "Accuracy is easy to compute but rewards the dominant class; macro-F1 averages the F1 of the three classes."),
  ("Macro-F1", "De ce raportăm macro-F1 și nu acuratețea pe PhraseBank?",
   ["Acuratețea nu poate fi calculată pentru trei clase", "Clasa „neutru” reprezintă {data.pb_neu_pct|.1f}% din date, deci răspunsul „mereu neutru” are o acuratețe mare; macro-F1 ponderează egal fiecare clasă", "Macro-F1 este întotdeauna mai mare", "Acuratețea are nevoie de probabilități"],
   "Clasa majoritară atinge o acuratețe de {sent.pb.majority.acc|.2f}%, dar un macro-F1 de doar {sent.pb.majority.f1|.2f}%.",
   "Acuratețea este ușor de calculat, dar răsplătește clasa dominantă; macro-F1 mediază scorurile F1 ale celor trei clase."))
q(2, ("Best sentiment model", "On PhraseBank, what macro-F1 did TF-IDF with logistic regression reach (mean of 5 splits)?",
      ["{sent.pb.majority.f1|.2f}%", "99.00%", "{sent.pb.tfidf_lr.f1|.2f}%", "{sent.pb.w2v_lr.f1|.2f}%"],
      "TF-IDF + logistic reached {sent.pb.tfidf_lr.f1|.2f}% (sd {sent.pb.tfidf_lr.f1_sd|.2f}); the embedding bag with word2vec start {sent.pb.embbag_w2v.f1|.2f}%.",
      "The other values belong to the majority class and to the mean word2vec model; no model is near 99% on this task."),
  ("Cel mai bun model de sentiment", "Pe PhraseBank, ce macro-F1 a atins TF-IDF cu regresie logistică (media pe 5 împărțiri)?",
   ["{sent.pb.majority.f1|.2f}%", "99,00%", "{sent.pb.tfidf_lr.f1|.2f}%", "{sent.pb.w2v_lr.f1|.2f}%"],
   "TF-IDF + logistic a atins {sent.pb.tfidf_lr.f1|.2f}% (abatere standard {sent.pb.tfidf_lr.f1_sd|.2f}); embedding bag cu start word2vec {sent.pb.embbag_w2v.f1|.2f}%.",
   "Celelalte valori aparțin clasei majoritare și modelului cu media word2vec; niciun model nu se apropie de 99% pe această sarcină."))
q(3, ("Paired comparison", "Over the 5 PhraseBank splits, TF-IDF + logistic minus the embedding bag (word2vec start) is {sent.pb_diff_tfidf_embw2v|+.2f} points with t = {sent.pb_diff_t|.2f}. Why a paired test?",
      ["Because the two models have the same number of parameters", "Because t-tests need five observations", "To make the difference larger", "Both models are scored on the same test sentences of each split, so the split-to-split variation cancels in the differences"],
      "Pairing removes the variation due to easy or hard splits; the t statistic has 4 degrees of freedom.",
      "Pairing is about the shared test sets, not about parameters or sample size; it does not change the mean difference, only its standard error."),
  ("Comparația pereche", "Pe cele 5 împărțiri PhraseBank, TF-IDF + logistic minus embedding bag (start word2vec) este {sent.pb_diff_tfidf_embw2v|+.2f} puncte, cu t = {sent.pb_diff_t|.2f}. De ce un test pereche?",
   ["Pentru că cele două modele au același număr de parametri", "Pentru că testele t au nevoie de cinci observații", "Pentru a mări diferența", "Ambele modele sînt evaluate pe aceleași fraze de test din fiecare împărțire, deci variația de la o împărțire la alta se anulează în diferențe"],
   "Împerecherea elimină variația datorată împărțirilor ușoare sau dificile; statistica t are 4 grade de libertate.",
   "Împerecherea ține de seturile de test comune, nu de parametri sau de mărimea eșantionului; nu schimbă diferența medie, doar eroarea ei standard."))
q(0, ("Label noise", "TF-IDF + logistic is right on {agree.tfidf_lr.100|.1f}% of unanimous sentences and {agree.tfidf_lr.50|.1f}% of those with 50% agreement. What does this suggest?",
      ["Disagreement among annotators caps what any model can reach on those sentences", "The model is overfitted to unanimous sentences", "Unanimous sentences are in the training set", "The test split is wrong"],
      "When the annotators themselves disagree, the label is partly noise; a better model cannot learn noise.",
      "All splits were random and stratified; the gap follows the difficulty of the sentences, measured by the annotators' agreement."),
  ("Zgomotul din etichete", "TF-IDF + logistic clasifică bine {agree.tfidf_lr.100|.1f}% din frazele cu acord unanim și {agree.tfidf_lr.50|.1f}% din cele cu acord de 50%. Ce sugerează aceasta?",
   ["Dezacordul dintre adnotatori limitează ce poate atinge orice model pe acele fraze", "Modelul este supraînvățat pe frazele cu acord unanim", "Frazele cu acord unanim se află în setul de antrenare", "Împărțirea de test este greșită"],
   "Cînd înșiși adnotatorii nu sînt de acord, eticheta este parțial zgomot; un model mai bun nu poate învăța zgomotul.",
   "Toate împărțirile au fost aleatoare și stratificate; diferența urmează dificultatea frazelor, măsurată prin acordul adnotatorilor."))
q(1, ("Few labels", "With only {curve.n_small} labelled sentences, which numbers did we measure?",
      ["All models were at the majority-class level", "TF-IDF {curve.tfidf_lr.p5|.1f}, mean word2vec {curve.w2v_lr.p5|.1f}, embedding bag random {curve.embbag.p5|.1f}, word2vec start {curve.embbag_w2v.p5|.1f} (macro-F1)", "Only the embedding bag could be trained", "The results were identical to those with all labels"],
      "The learning curve shows how each representation behaves when labels are scarce; compare it with the values at {curve.n_full|,} sentences.",
      "All four models were trained at every size; the numbers differ from the full-data results, which is the point of the curve."),
  ("Puține etichete", "Cu doar {curve.n_small} de fraze etichetate, ce valori am măsurat?",
   ["Toate modelele au fost la nivelul clasei majoritare", "TF-IDF {curve.tfidf_lr.p5|.1f}, media word2vec {curve.w2v_lr.p5|.1f}, embedding bag aleator {curve.embbag.p5|.1f}, start word2vec {curve.embbag_w2v.p5|.1f} (macro-F1)", "Doar embedding bag a putut fi antrenat", "Rezultatele au fost identice cu cele obținute cu toate etichetele"],
   "Curba de învățare arată cum se comportă fiecare reprezentare cînd etichetele sînt rare; comparați-o cu valorile pentru {curve.n_full|,} de fraze.",
   "Toate cele patru modele au fost antrenate la fiecare dimensiune; valorile diferă de cele cu toate datele, iar aceasta este ideea curbei."))
q(2, ("Classification errors", "In the pooled PhraseBank test sets, most errors of TF-IDF + logistic are of which kind?",
      ["Negative sentences read as positive", "Positive read as negative", "Negative or positive sentences read as neutral", "Neutral sentences read as positive, always"],
      "Opposite-sign errors are rare ({cm.neg_as_pos} negative read as positive); the recall is {cm.neg_recall|.1f}% for negative and {cm.pos_recall|.1f}% for positive sentences.",
      "The confusion matrix shows that the model hesitates towards neutral much more often than it flips the sign."),
  ("Erorile de clasificare", "În seturile de test PhraseBank reunite, ce tip de erori face cel mai des TF-IDF + logistic?",
   ["Fraze negative citite ca pozitive", "Fraze pozitive citite ca negative", "Fraze negative sau pozitive citite ca neutre", "Fraze neutre citite mereu ca pozitive"],
   "Erorile de semn opus sînt rare ({cm.neg_as_pos} fraze negative citite ca pozitive); sensibilitatea este {cm.neg_recall|.1f}% pentru frazele negative și {cm.pos_recall|.1f}% pentru cele pozitive.",
   "Matricea de confuzie arată că modelul ezită spre neutru mult mai des decît inversează semnul."))
q(3, ("Embedding bag", "What does the embedding-bag classifier compute?",
      ["A recurrent pass over the words", "A convolution over the characters", "Attention weights between words", "The mean of the vectors of the words and bigrams of the sentence, then dropout and a softmax layer"],
      "z = (1/|S|) Σ e_s, p̂ = softmax(W dropout(z) + b), trained with class weights: the architecture of fastText-type classifiers.",
      "The model has no recurrence, convolution or attention; its only structure is the average of learned vectors."),
  ("Embedding bag", "Ce calculează clasificatorul embedding bag?",
   ["O parcurgere recurentă a cuvintelor", "O convoluție pe caractere", "Ponderi de atenție între cuvinte", "Media vectorilor cuvintelor și bigramelor din frază, apoi dropout și un strat softmax"],
   "z = (1/|S|) Σ e_s, p̂ = softmax(W dropout(z) + b), antrenat cu ponderi pe clase: arhitectura clasificatoarelor de tip fastText.",
   "Modelul nu are recurență, convoluție sau atenție; singura lui structură este media unor vectori învățați."))
q(0, ("Class weights", "Why does the loss use weights ω_k = n/(3n_k)?",
      ["To give the rare negative class as much total weight as the dominant neutral class", "To speed up training", "To normalise the vectors", "To select the vocabulary"],
      "Without weights the gradient is dominated by neutral sentences, and the model learns to say ‘neutral’; the weights balance the three classes.",
      "Class weights change the objective, not the speed, the vectors or the vocabulary."),
  ("Ponderile claselor", "De ce folosește funcția de pierdere ponderile ω_k = n/(3n_k)?",
   ["Pentru a da clasei negative, rare, aceeași pondere totală ca clasei neutre, dominante", "Pentru a accelera antrenarea", "Pentru a normaliza vectorii", "Pentru a alege vocabularul"],
   "Fără ponderi, gradientul este dominat de frazele neutre, iar modelul învață să spună „neutru”; ponderile echilibrează cele trei clase.",
   "Ponderile claselor schimbă obiectivul, nu viteza, vectorii sau vocabularul."))
q(1, ("Topics", "On the 20-topic tweet data, which statement matches our results?",
      ["All topics are equally easy", "TF-IDF + logistic reached a macro-F1 of {topic.lr_f1|.2f}%; easiest topic {topic.best_class}, hardest {topic.worst_class}", "The majority topic beat every model", "Topic classification cannot be done with TF-IDF"],
      "Topics with specific vocabulary (earnings, central banks) are easy; broad topics are hard, F1 from {topic.worst_f1|.1f}% to {topic.best_f1|.1f}%.",
      "The per-topic chart shows a wide range of difficulty, and both models are far above the majority topic ({topic.majority_acc|.2f}% accuracy)."),
  ("Temele", "Pe datele cu 20 de teme ale tweet-urilor, ce afirmație corespunde rezultatelor noastre?",
   ["Toate temele sînt la fel de ușoare", "TF-IDF + logistic a atins un macro-F1 de {topic.lr_f1|.2f}%; tema cea mai ușoară {topic.best_class_ro}, cea mai dificilă {topic.worst_class_ro}", "Tema majoritară a depășit orice model", "Clasificarea pe teme nu se poate face cu TF-IDF"],
   "Temele cu vocabular specific (rezultate, bănci centrale) sînt ușoare; temele largi sînt dificile, cu F1 între {topic.worst_f1|.1f}% și {topic.best_f1|.1f}%.",
   "Graficul pe teme arată o gamă largă de dificultate, iar ambele modele sînt mult peste tema majoritară ({topic.majority_acc|.2f}% acuratețe)."))
q(2, ("Distributional hypothesis", "What is the idea behind all word embeddings of this lecture?",
      ["Words are defined by their spelling", "Words are defined by their frequency", "A word is characterised by the contexts in which it appears", "Words are independent of each other"],
      "‘You shall know a word by the company it keeps’ (Firth, 1957): word2vec predicts contexts and GloVe fits co-occurrence counts.",
      "Spelling is used only by subword models, and frequency alone does not separate meanings; the contexts do."),
  ("Ipoteza distribuțională", "Care este ideea din spatele tuturor vectorilor de cuvinte din acest curs?",
   ["Cuvintele sînt definite de felul în care se scriu", "Cuvintele sînt definite de frecvența lor", "Un cuvînt este caracterizat de contextele în care apare", "Cuvintele sînt independente unele de altele"],
   "„Un cuvînt se cunoaște după vecinii lui” (Firth, 1957): word2vec prezice contextele, iar GloVe aproximează frecvențele de co-apariție.",
   "Scrierea este folosită doar de modelele cu subcuvinte, iar frecvența singură nu separă sensurile; contextele o fac."))
q(3, ("Analogies", "How is an analogy a : b :: c : ? answered with embeddings?",
      ["By the most frequent word", "By a dictionary lookup", "By training a new classifier", "By the word whose vector is closest (cosine) to e_b − e_a + e_c, excluding a, b and c"],
      "Vector offsets capture regular relations when the corpus is large enough; on one small corpus the answers are noisier.",
      "The analogy uses only vector arithmetic on the learned embeddings, with no new training and no external dictionary."),
  ("Analogiile", "Cum se răspunde la o analogie a : b :: c : ? cu vectori de cuvinte?",
   ["Cu cel mai frecvent cuvînt", "Cu o căutare în dicționar", "Prin antrenarea unui nou clasificator", "Cu cuvîntul al cărui vector este cel mai apropiat (cosinus) de e_b − e_a + e_c, excluzînd a, b și c"],
   "Diferențele de vectori surprind relațiile regulate cînd corpusul este suficient de mare; pe un singur corpus mic, răspunsurile sînt mai zgomotoase.",
   "Analogia folosește doar aritmetica vectorilor învățați, fără o nouă antrenare și fără un dicționar extern."))
q(0, ("First model of a text project", "You start a project to classify customer complaints of a bank. What is the first model to build?",
      ["TF-IDF + logistic regression, as the benchmark any more complex model must beat", "A large language model", "A word2vec model trained on Wikipedia", "A recurrent network"],
      "It is fast, cheap and explainable word by word; on our financial sentences it was among the best models.",
      "More complex models may help, but only a measured gain over the simple benchmark justifies their cost."),
  ("Primul model al unui proiect de text", "Începeți un proiect de clasificare a reclamațiilor clienților unei bănci. Care este primul model pe care îl construiți?",
   ["TF-IDF + regresie logistică, ca reper pe care orice model mai complex trebuie să-l depășească", "Un model lingvistic mare", "Un model word2vec antrenat pe Wikipedia", "O rețea recurentă"],
   "Este rapid, ieftin și explicabil cuvînt cu cuvînt; pe frazele noastre financiare a fost printre cele mai bune modele.",
   "Modelele mai complexe pot ajuta, dar doar un cîștig măsurat față de reperul simplu le justifică costul."))
q(1, ("Group research: LLM-VaR", "What do Pele et al. (2026) show in ‘In the beginning was the Word: LLM-VaR and LLM-ES’?",
      ["That word2vec predicts stock returns", "That large language models can forecast value-at-risk and expected shortfall, benchmarked against classical risk models with formal backtests", "That sentiment dictionaries fail on tweets", "That GloVe beats word2vec"],
      "The paper moves language models from text scores to the risk numbers themselves, with formal backtests (Lecture 12).",
      "The study is about tail-risk forecasting with large language models, not about static embeddings or dictionaries."),
  ("Cercetarea grupului: LLM-VaR", "Ce arată Pele et al. (2026) în „In the beginning was the Word: LLM-VaR and LLM-ES”?",
   ["Că word2vec prognozează randamentele acțiunilor", "Că modelele lingvistice mari pot prognoza valoarea expusă la risc și expected shortfall, comparate cu modelele clasice de risc prin teste retrospective formale", "Că dicționarele de sentiment eșuează pe tweet-uri", "Că GloVe depășește word2vec"],
   "Lucrarea mută modelele de limbaj de la scoruri ale textului la cifrele de risc propriu-zise, cu teste retrospective formale (cursul 12).",
   "Studiul privește prognoza riscului de coadă cu modele lingvistice mari, nu vectorii statici sau dicționarele."))
q(2, ("Group research: BitMood", "Which NLP model did BitMood (Conda et al., 2026) use to score 120,000 Facebook posts?",
      ["A bag of words", "word2vec", "FinBERT, combined with the reactions of readers into compound sentiment scores", "GloVe"],
      "FinBERT, a BERT model adapted to financial text, gives sentiment per post; the paper combines it with Facebook reactions and finds predictive power for Bitcoin volume.",
      "The study used a transformer model of financial text, the topic of Lecture 7, not static embeddings or counts."),
  ("Cercetarea grupului: BitMood", "Ce model NLP a folosit BitMood (Conda et al., 2026) pentru a evalua 120.000 de postări de pe Facebook?",
   ["Un sac de cuvinte", "word2vec", "FinBERT, combinat cu reacțiile cititorilor în scoruri compuse de sentiment", "GloVe"],
   "FinBERT, un model BERT adaptat textelor financiare, dă sentimentul fiecărei postări; lucrarea îl combină cu reacțiile de pe Facebook și găsește putere predictivă pentru volumul Bitcoin.",
   "Studiul a folosit un model transformer pentru text financiar, subiectul cursului 7, nu vectori statici sau frecvențe."))
q(3, ("Vocabulary and leakage", "Where must the vocabulary of the TF-IDF model be built?",
      ["On all sentences, including the test set", "On the test set only", "On the ECB speeches", "On the training sentences of each split only"],
      "Building the vocabulary and the idf on test sentences leaks information about them into the features.",
      "Like any preprocessing step, the vocabulary is part of the model and must be estimated without the test data."),
  ("Vocabularul și scurgerea de informație", "Pe ce date trebuie construit vocabularul modelului TF-IDF?",
   ["Pe toate frazele, inclusiv pe cele de test", "Doar pe setul de test", "Pe discursurile BCE", "Doar pe frazele de antrenare ale fiecărei împărțiri"],
   "Construirea vocabularului și a idf-ului pe frazele de test scurge informație despre ele în trăsături.",
   "Ca orice etapă de preprocesare, vocabularul face parte din model și trebuie estimat fără datele de test."))


if __name__ == "__main__":
    out = []
    for x in Q:
        y_ = {"correct": x["correct"]}
        for lang in ("en", "ro"):
            d = dict(x[lang])
            y_[lang] = {k: ([R(o, lang) for o in v] if isinstance(v, list) else R(v, lang)) for k, v in d.items()}
        out.append(y_)
    assert len(out) >= 30 and all(len(o["en"]["options"]) == 4 and len(o["ro"]["options"]) == 4 for o in out)
    js = ("// ============================================================\n"
          "// Quiz bank for lecture id 'l6': NLP foundations: text representation and word embeddings (EN + RO)\n"
          "// Generated by lectures/L6/make_quiz_L6.py from L6_numbers.json; correct = index (0-3) in the original order.\n"
          "// incorrectExplanation never names a letter: the engine prepends the correct answer after shuffling.\n"
          "// ============================================================\n"
          "window.NN_DATA.quizzes['l6'] = " + json.dumps({"draw": 20, "questions": out}, ensure_ascii=False, indent=4) + ";\n")
    p = os.path.join(ROOT, "assets", "quizzes", "l6.js")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(js)
    print(f"wrote {p}: {len(out)} questions")
