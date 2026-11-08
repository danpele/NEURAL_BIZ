"""
make_quiz_L7.py -- writes assets/quizzes/l7.js, the quiz bank of Lecture 7 (EN + RO), with every number read from
L7_numbers.json. Format of lectures/README.md: correct = index 0-3 of the right option; the explanation of a wrong
answer never names a letter (the engine shuffles the options).
Run:  python3 lectures/L7/make_quiz_L7.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = json.load(open(os.path.join(HERE, "L7_numbers.json")))


def g(key):
    v = N
    for p in key.split("."):
        v = v[p]
    return v


def f(key, spec="", lang="en"):
    s = format(g(key), spec)
    if lang == "ro":
        s = s.replace(",", " ").replace(".", ",")
    return ("−" + s[1:]) if s.startswith("-") else s


E, R = (lambda k, s="": f(k, s, "en")), (lambda k, s="": f(k, s, "ro"))
Q = []


def q(correct, en, ro):
    keys = ["title", "text", "options", "correctExplanation", "incorrectExplanation"]
    Q.append({"correct": correct, "en": dict(zip(keys, en)), "ro": dict(zip(keys, ro))})


q(1, ("Attention formula", "Which expression is scaled dot-product attention?",
      ["$\\operatorname{softmax}(QK^\\top)V/d_k$", "$\\operatorname{softmax}(QK^\\top/\\sqrt{d_k})V$", "$\\sigma(QV^\\top)K$", "$QK^\\top V/\\sqrt{n}$"],
      "Scores $QK^\\top$ are divided by $\\sqrt{d_k}$, turned into weights by a row-wise softmax and used to average the values $V$.",
      "The scaling by $\\sqrt{d_k}$ is applied to the scores before the softmax, and the weights multiply the values, not the keys."),
  ("Formula atenției", "Care expresie este atenția cu produs scalar scalat?",
   ["$\\operatorname{softmax}(QK^\\top)V/d_k$", "$\\operatorname{softmax}(QK^\\top/\\sqrt{d_k})V$", "$\\sigma(QV^\\top)K$", "$QK^\\top V/\\sqrt{n}$"],
   "Scorurile $QK^\\top$ se împart la $\\sqrt{d_k}$, devin ponderi printr-un softmax pe rînduri și se folosesc pentru media valorilor $V$.",
   "Scalarea cu $\\sqrt{d_k}$ se aplică scorurilor înainte de softmax, iar ponderile înmulțesc valorile, nu cheile."))
q(3, ("Why scale", "With unit-variance entries, what is the variance of the dot product $\\mathbf q^\\top\\mathbf k$ of two $d_k$-dimensional vectors?",
      ["1", "$\\sqrt{d_k}$", "$1/d_k$", "$d_k$"],
      "It is the sum of $d_k$ products with variance 1, so the variance is $d_k$; dividing by $\\sqrt{d_k}$ brings it back to 1.",
      "Each product $q_lk_l$ has variance 1 and there are $d_k$ of them; without scaling the softmax saturates for large $d_k$."),
  ("De ce scalăm", "Cu elemente de varianță unu, care este varianța produsului scalar $\\mathbf q^\\top\\mathbf k$ a doi vectori cu $d_k$ dimensiuni?",
   ["1", "$\\sqrt{d_k}$", "$1/d_k$", "$d_k$"],
   "Este suma a $d_k$ produse cu varianța 1, deci varianța este $d_k$; împărțirea la $\\sqrt{d_k}$ o readuce la 1.",
   "Fiecare produs $q_lk_l$ are varianța 1 și sînt $d_k$ astfel de produse; fără scalare, softmax-ul se saturează pentru $d_k$ mare."))
q(0, ("Saturated softmax", "What goes wrong when attention scores have a very large spread?",
      ["The softmax becomes almost one-hot and its gradient almost vanishes", "The softmax outputs negative weights", "The model becomes linear", "The keys and values swap roles"],
      "With a dominant score, $\\operatorname{diag}(\\mathbf a)-\\mathbf a\\mathbf a^\\top$ is near zero, so queries and keys stop learning.",
      "Softmax weights are always positive and sum to one; the problem is saturation, which kills the gradient."),
  ("Softmax saturat", "Ce nu merge bine cînd scorurile de atenție au o dispersie foarte mare?",
   ["Softmax-ul devine aproape one-hot, iar gradientul aproape dispare", "Softmax-ul produce ponderi negative", "Modelul devine liniar", "Cheile și valorile își schimbă rolurile"],
   "Cu un scor dominant, $\\operatorname{diag}(\\mathbf a)-\\mathbf a\\mathbf a^\\top$ este aproape zero, deci întrebările și cheile nu mai învață.",
   "Ponderile softmax sînt mereu pozitive și însumează unu; problema este saturația, care anulează gradientul."))
q(2, ("Path length", "In self-attention, how many steps separate the first and the last word of an $n$-token sentence?",
      ["$n-1$", "$\\log n$", "One", "$n^2$"],
      "Every token attends to every other token directly, so the path is constant, against $n-1$ steps in a recurrent network.",
      "Recurrence needs $n-1$ steps and a convolution stack about $\\log_k n$ layers; attention links any pair in one layer."),
  ("Lungimea drumului", "În auto-atenție, cîți pași separă primul și ultimul cuvînt dintr-o propoziție de $n$ tokeni?",
   ["$n-1$", "$\\log n$", "Unul", "$n^2$"],
   "Fiecare token acordă atenție direct oricărui alt token, deci drumul este constant, față de $n-1$ pași într-o rețea recurentă.",
   "Recurența are nevoie de $n-1$ pași, iar o stivă de convoluții de circa $\\log_k n$ straturi; atenția leagă orice pereche într-un singur strat."))
q(1, ("Quadratic cost", "A median ECB speech has " + E("len.sp_median", ",") + " tokens and a median tweet " + E("len.tw_median") + ". Roughly how much more does one attention layer cost on the speech?",
      ["About " + E("len.sp_median", ",") + " / " + E("len.tw_median") + " times", "About " + E("len.cost_ratio", ",") + " times", "The same", "About twice"],
      "Attention cost grows with $n^2$: $(" + E("len.sp_median", ",") + "/" + E("len.tw_median") + ")^2\\approx" + E("len.cost_ratio", ",") + "$.",
      "Cost and memory are quadratic in the length, so the ratio of lengths must be squared."),
  ("Costul pătratic", "Un discurs median al BCE are " + R("len.sp_median", ",") + " de tokeni, iar un tweet median " + R("len.tw_median") + ". De cîte ori costă mai mult, aproximativ, un strat de atenție pe discurs?",
   ["De circa " + R("len.sp_median", ",") + " / " + R("len.tw_median") + " ori", "De circa " + R("len.cost_ratio", ",") + " de ori", "La fel", "De circa două ori"],
   "Costul atenției crește cu $n^2$: $(" + R("len.sp_median", ",") + "/" + R("len.tw_median") + ")^2\\approx" + R("len.cost_ratio", ",") + "$.",
   "Costul și memoria sînt pătratice în lungime, deci raportul lungimilor trebuie ridicat la pătrat."))
q(3, ("Context window", "What share of the " + E("len.n_speeches") + " ECB speeches exceed the 512-token window of BERT?",
      [E("len.tw_pct_over64", ".2f") + " %", "50 %", "0 %", E("len.sp_pct_over512", ".1f") + " %"],
      "Almost every speech (" + E("len.sp_pct_over512", ".1f") + " %) is longer than 512 tokens, so it must be chunked or read by a long-context model.",
      "With a median of " + E("len.sp_median", ",") + " tokens, nearly all speeches exceed the window; " + E("len.tw_pct_over64", ".2f") + " % is the share of tweets above 64 tokens."),
  ("Fereastra de context", "Ce pondere dintre cele " + R("len.n_speeches") + " de discursuri ale BCE depășesc fereastra de 512 tokeni a BERT?",
   [R("len.tw_pct_over64", ".2f") + " %", "50 %", "0 %", R("len.sp_pct_over512", ".1f") + " %"],
   "Aproape fiecare discurs (" + R("len.sp_pct_over512", ".1f") + " %) are peste 512 tokeni, deci trebuie împărțit sau citit de un model cu context lung.",
   "Cu o mediană de " + R("len.sp_median", ",") + " de tokeni, aproape toate discursurile depășesc fereastra; " + R("len.tw_pct_over64", ".2f") + " % este ponderea tweet-urilor de peste 64 de tokeni."))
q(0, ("Positional encoding", "Why does a transformer add positional encodings to the token embeddings?",
      ["Self-attention is permutation-equivariant and would otherwise ignore word order", "To reduce the vocabulary size", "To normalise the activations", "To mask future tokens"],
      "Without positions, “rates cut, stocks up” and “stocks cut, rates up” would give the same representations.",
      "Normalisation is done by layer norm and masking by the attention mask; positions are needed because attention treats the input as a set."),
  ("Codificarea poziției", "De ce adaugă un transformer codificări ale poziției la reprezentările tokenilor?",
   ["Auto-atenția este echivariantă la permutări și altfel ar ignora ordinea cuvintelor", "Ca să micșoreze vocabularul", "Ca să normalizeze activările", "Ca să mascheze tokenii viitori"],
   "Fără poziții, „dobînzile scad, acțiunile cresc” și „acțiunile scad, dobînzile cresc” ar avea aceleași reprezentări.",
   "Normalizarea se face prin normalizarea pe strat, iar mascarea prin masca atenției; pozițiile sînt necesare pentru că atenția tratează intrarea ca pe o mulțime."))
q(2, ("Positions on tweets", "Removing the positional encoding from our small transformer changed its macro-F1 by how much?",
      ["About −20 points", "Nothing at all", E("gain_pos", "+.1f") + " points (" + E("models.tiny.macro_f1", ".1f") + " % against " + E("models.tiny_nopos.macro_f1", ".1f") + " %)", "+15 points"],
      "On short, formulaic tweets order matters little: the gain from positions is " + E("gain_pos", "+.1f") + " points, with a bootstrap interval [" + E("boot.tiny_nopos.lo", "+.1f") + "; " + E("boot.tiny_nopos.hi", "+.1f") + "].",
      "Tweets are mostly bags of signal words, so the model without positions is only slightly worse; order matters more for negation and long texts."),
  ("Pozițiile pe tweet-uri", "Cu cît s-a schimbat macro-F1 al transformerului mic după eliminarea codificării poziției?",
   ["Cu circa −20 de puncte", "Deloc", R("gain_pos", "+.1f") + " puncte (" + R("models.tiny.macro_f1", ".1f") + " % față de " + R("models.tiny_nopos.macro_f1", ".1f") + " %)", "+15 puncte"],
   "Pe tweet-uri scurte și formulaice ordinea contează puțin: cîștigul adus de poziții este " + R("gain_pos", "+.1f") + " puncte, cu intervalul bootstrap [" + R("boot.tiny_nopos.lo", "+.1f") + "; " + R("boot.tiny_nopos.hi", "+.1f") + "].",
   "Tweet-urile sînt mai ales colecții de cuvinte-semnal, deci modelul fără poziții este doar puțin mai slab; ordinea contează mai mult pentru negație și texte lungi."))
q(1, ("Multi-head attention", "Compared with one head of dimension $d$, how many parameters do $h$ heads of dimension $d/h$ (plus the output projection) have?",
      ["$h$ times more", "About the same, $4d^2$", "$h^2$ times more", "Half as many"],
      "The $h$ projections of size $d\\times d/h$ add up to $d\\times d$ for each of $Q$, $K$, $V$, plus $W_O$: $4d^2$ in total.",
      "Splitting $d$ into $h$ heads keeps the total size; the gain is several kinds of relations per layer at the same cost."),
  ("Atenția cu mai multe capete", "Față de un singur cap de dimensiune $d$, cîți parametri au $h$ capete de dimensiune $d/h$ (plus proiecția de ieșire)?",
   ["De $h$ ori mai mulți", "Aproximativ la fel, $4d^2$", "De $h^2$ ori mai mulți", "Jumătate"],
   "Cele $h$ proiecții de dimensiune $d\\times d/h$ însumează $d\\times d$ pentru fiecare dintre $Q$, $K$, $V$, plus $W_O$: în total $4d^2$.",
   "Împărțirea lui $d$ în $h$ capete păstrează dimensiunea totală; cîștigul este reprezentat de mai multe tipuri de relații pe strat, la același cost."))
q(3, ("Transformer block", "Which sequence describes a pre-norm transformer block?",
      ["Attention, then softmax over the vocabulary", "Convolution, pooling, dense layer", "Recurrent update, then attention", "$Z=X+\\mathrm{MHA}(\\mathrm{LN}(X))$, then $X'=Z+\\mathrm{FFN}(\\mathrm{LN}(Z))$"],
      "Each sub-layer is normalised, applied and added back through a residual connection; the feed-forward network acts on each position separately.",
      "A transformer block has two residual sub-layers, multi-head attention and a position-wise feed-forward network, each with layer normalisation."),
  ("Blocul transformer", "Care succesiune descrie un bloc transformer cu normalizare înainte?",
   ["Atenție, apoi softmax pe vocabular", "Convoluție, pooling, strat dens", "Actualizare recurentă, apoi atenție", "$Z=X+\\mathrm{MHA}(\\mathrm{LN}(X))$, apoi $X'=Z+\\mathrm{FFN}(\\mathrm{LN}(Z))$"],
   "Fiecare substrat este normalizat, aplicat și adăugat înapoi printr-o conexiune reziduală; rețeaua directă acționează separat pe fiecare poziție.",
   "Un bloc transformer are două substraturi reziduale, atenția cu mai multe capete și rețeaua directă aplicată fiecărei poziții, fiecare cu normalizare pe strat."))
q(0, ("Where the parameters are", "In a transformer block with model width $d$ and feed-forward width $4d$, where are most of the parameters?",
      ["In the feed-forward network, about $8d^2$ of $12d^2$", "In the attention softmax", "In the layer normalisation", "In the positional encoding"],
      "Attention has about $4d^2$ parameters and the feed-forward network about $8d^2$; layer norms and sinusoidal positions add almost nothing.",
      "Softmax and sinusoidal positions have no parameters and layer norm only $2d$ per norm; the two feed-forward matrices dominate."),
  ("Unde sînt parametrii", "Într-un bloc transformer cu lățimea $d$ și rețeaua directă de lățime $4d$, unde se află cei mai mulți parametri?",
   ["În rețeaua directă, circa $8d^2$ din $12d^2$", "În softmax-ul atenției", "În normalizarea pe strat", "În codificarea poziției"],
   "Atenția are circa $4d^2$ parametri, iar rețeaua directă circa $8d^2$; normalizările și pozițiile sinusoidale aproape nu adaugă nimic.",
   "Softmax-ul și pozițiile sinusoidale nu au parametri, iar normalizarea pe strat doar $2d$; cele două matrice ale rețelei directe domină."))
q(2, ("BERT pretraining", "What is the pretraining task of BERT?",
      ["Predicting the next token from the left context only", "Translating between two languages", "Predicting randomly masked tokens from both sides (masked language modelling)", "Classifying sentiment"],
      "About 15 % of the tokens are masked and predicted from the full left and right context, which makes BERT a bidirectional encoder.",
      "Next-token prediction with a causal mask is the GPT objective; sentiment is a downstream task learned by fine-tuning."),
  ("Preantrenarea BERT", "Care este sarcina de preantrenare a BERT?",
   ["Prezicerea tokenului următor doar din contextul din stînga", "Traducerea între două limbi", "Prezicerea unor tokeni mascați aleator din ambele părți (modelarea limbajului cu cuvinte mascate)", "Clasificarea sentimentului"],
   "Circa 15 % dintre tokeni sînt mascați și preziși din tot contextul, din stînga și din dreapta, ceea ce face din BERT un codificator bidirecțional.",
   "Prezicerea tokenului următor, cu mască cauzală, este obiectivul GPT; sentimentul este o sarcină ulterioară, învățată prin ajustare fină."))
q(1, ("Causal mask", "In a decoder such as GPT, which attention weights are forced to zero?",
      ["Those on padding tokens only", "Those from position $i$ to any later position $j>i$", "Those on the first token", "None"],
      "The causal mask sets $M_{ij}=-\\infty$ for $j>i$, so each token sees only the past and the model can generate text left to right.",
      "Generation must not peek at future tokens; the causal mask removes them from every softmax."),
  ("Masca cauzală", "Într-un decodor precum GPT, ce ponderi de atenție sînt forțate la zero?",
   ["Doar cele pe tokenii de completare", "Cele de la poziția $i$ spre orice poziție ulterioară $j>i$", "Cele pe primul token", "Niciuna"],
   "Masca cauzală pune $M_{ij}=-\\infty$ pentru $j>i$, deci fiecare token vede doar trecutul, iar modelul poate genera text de la stînga la dreapta.",
   "Generarea nu are voie să vadă tokenii viitori; masca cauzală îi elimină din fiecare softmax."))
q(3, ("Encoder or decoder", "A bank wants to label 10,000 complaints a day by topic. Which transformer family fits best?",
      ["A decoder-only model generating free text", "An encoder-decoder for translation", "A recurrent network", "An encoder (BERT-type), fine-tuned on labelled complaints"],
      "Encoders read the whole text in both directions and output class probabilities cheaply and consistently: the natural tool for labelling.",
      "Decoders are built to write text; for classification at scale an encoder with a classification head is simpler, cheaper and more stable."),
  ("Codificator sau decodor", "O bancă vrea să eticheteze după subiect 10 000 de reclamații pe zi. Ce familie de transformere se potrivește cel mai bine?",
   ["Un model doar decodor, care generează text liber", "Un codificator-decodor pentru traducere", "O rețea recurentă", "Un codificator (de tip BERT), ajustat pe reclamații etichetate"],
   "Codificatoarele citesc tot textul în ambele direcții și produc ieftin și consecvent probabilitățile claselor: instrumentul firesc pentru etichetare.",
   "Decodoarele sînt construite să scrie text; pentru clasificare la scară mare, un codificator cu un strat de clasificare este mai simplu, mai ieftin și mai stabil."))
q(0, ("Subword tokens", "Why does FinBERT use a WordPiece vocabulary of " + E("tok.vocab", ",") + " pieces instead of whole words?",
      ["Any word, including tickers and new names, can be split into known pieces", "Whole words cannot be embedded", "It removes the need for positions", "It makes attention linear in $n$"],
      "Rare words are split into frequent subwords, so the vocabulary stays fixed and no word is ever out of vocabulary.",
      "Subword tokenization solves the open-vocabulary problem; it does not change the attention cost or remove the need for positions."),
  ("Tokeni de subcuvinte", "De ce folosește FinBERT un vocabular WordPiece de " + R("tok.vocab", ",") + " de bucăți în loc de cuvinte întregi?",
   ["Orice cuvînt, inclusiv simbolurile bursiere și numele noi, se poate împărți în bucăți cunoscute", "Cuvintele întregi nu pot avea reprezentări", "Elimină nevoia pozițiilor", "Face atenția liniară în $n$"],
   "Cuvintele rare se împart în subcuvinte frecvente, deci vocabularul rămîne fix și niciun cuvînt nu este necunoscut.",
   "Tokenizarea pe subcuvinte rezolvă problema vocabularului deschis; nu schimbă costul atenției și nu elimină nevoia pozițiilor."))
q(2, ("Best model on tweets", "Which model reached the highest macro-F1 on the " + E("data.n_test", ",") + " test tweets?",
      ["TF-IDF + logistic regression, " + E("models.tfidf.macro_f1", ".1f") + " %", "The small transformer, " + E("models.tiny.macro_f1", ".1f") + " %", "Fine-tuned FinBERT, " + E("models.finbert_ft.macro_f1", ".1f") + " %", "Zero-shot FinBERT, " + E("models.finbert_zs.macro_f1", ".1f") + " %"],
      "Fine-tuning a pretrained financial encoder gives " + E("models.finbert_ft.macro_f1", ".1f") + " %, " + E("gain_ft_vs_tfidf", "+.1f") + " points above TF-IDF.",
      "Pretraining plus fine-tuning wins clearly; TF-IDF is a strong benchmark, and the transformer trained from scratch is the weakest of the three."),
  ("Cel mai bun model pe tweet-uri", "Care model a atins cel mai mare macro-F1 pe cele " + R("data.n_test", ",") + " de tweet-uri de test?",
   ["TF-IDF + regresie logistică, " + R("models.tfidf.macro_f1", ".1f") + " %", "Transformerul mic, " + R("models.tiny.macro_f1", ".1f") + " %", "FinBERT ajustat, " + R("models.finbert_ft.macro_f1", ".1f") + " %", "FinBERT fără antrenare, " + R("models.finbert_zs.macro_f1", ".1f") + " %"],
   "Ajustarea fină a unui codificator financiar preantrenat dă " + R("models.finbert_ft.macro_f1", ".1f") + " %, cu " + R("gain_ft_vs_tfidf", "+.1f") + " puncte peste TF-IDF.",
   "Preantrenarea plus ajustarea fină cîștigă clar; TF-IDF este un model de referință puternic, iar transformerul antrenat de la zero este cel mai slab dintre cele trei."))
q(1, ("Transformer from scratch", "Why did the small transformer trained from scratch (" + E("models.tiny.macro_f1", ".1f") + " %) lose to TF-IDF (" + E("models.tfidf.macro_f1", ".1f") + " %)?",
      ["Attention cannot classify text", "It must learn language and task from " + E("data.n_train", ",") + " tweets with about " + E("models.tiny.params", ",") + " parameters", "It used the test set for training", "TF-IDF uses a transformer internally"],
      "Most parameters sit in the embedding table and must be learned from few examples; the bootstrap difference is " + E("boot.tiny_tfidf.mean", "+.1f") + " points [" + E("boot.tiny_tfidf.lo", "+.1f") + "; " + E("boot.tiny_tfidf.hi", "+.1f") + "].",
      "Transformers are data-hungry; without pretraining a few thousand labelled tweets are not enough to beat a well-tuned sparse model."),
  ("Transformerul de la zero", "De ce a pierdut transformerul mic antrenat de la zero (" + R("models.tiny.macro_f1", ".1f") + " %) în fața TF-IDF (" + R("models.tfidf.macro_f1", ".1f") + " %)?",
   ["Atenția nu poate clasifica texte", "Trebuie să învețe limba și sarcina din " + R("data.n_train", ",") + " de tweet-uri, cu circa " + R("models.tiny.params", ",") + " de parametri", "A folosit setul de test la antrenare", "TF-IDF folosește intern un transformer"],
   "Cei mai mulți parametri sînt în tabelul reprezentărilor și trebuie învățați din puține exemple; diferența bootstrap este " + R("boot.tiny_tfidf.mean", "+.1f") + " puncte [" + R("boot.tiny_tfidf.lo", "+.1f") + "; " + R("boot.tiny_tfidf.hi", "+.1f") + "].",
   "Transformerele au nevoie de multe date; fără preantrenare, cîteva mii de tweet-uri etichetate nu ajung pentru a bate un model rar bine calibrat."))
q(3, ("Zero-shot domain shift", "Zero-shot FinBERT reached only " + E("models.finbert_zs.macro_f1", ".1f") + " % macro-F1 on tweets. What is the main reason?",
      ["FinBERT has too few parameters", "Tweets are longer than 512 tokens", "The labels of the tweets are random", "FinBERT was trained on news sentences, a different domain and label definition"],
      "Domain shift: news sentences and finance tweets differ in style and in what annotators call positive or negative; fine-tuning closes the gap (" + E("gain_ft_vs_zs", "+.1f") + " points).",
      "Tweets are short and FinBERT is large; the problem is the mismatch between the training domain and the target domain."),
  ("Schimbarea domeniului", "FinBERT fără antrenare a atins doar " + R("models.finbert_zs.macro_f1", ".1f") + " % macro-F1 pe tweet-uri. Care este motivul principal?",
   ["FinBERT are prea puțini parametri", "Tweet-urile au peste 512 tokeni", "Etichetele tweet-urilor sînt aleatoare", "FinBERT a fost antrenat pe propoziții din știri, un domeniu și o definiție a etichetelor diferite"],
   "Schimbarea domeniului: propozițiile din știri și tweet-urile financiare diferă ca stil și ca înțeles al etichetelor; ajustarea fină închide diferența (" + R("gain_ft_vs_zs", "+.1f") + " puncte).",
   "Tweet-urile sînt scurte, iar FinBERT este mare; problema este nepotrivirea dintre domeniul de antrenare și domeniul-țintă."))
q(0, ("Frozen features", "Frozen FinBERT vectors with a logistic regression reached " + E("models.finbert_probe.macro_f1", ".1f") + " % on all training tweets. Where do they help most?",
      ["With few labels: " + E("scaling.probe_small", ".1f") + " % with " + E("scaling.n_small") + " tweets, against " + E("scaling.tfidf_small", ".1f") + " % for TF-IDF", "Only with all tweets", "Never: they are always worse than TF-IDF", "Only on long documents"],
      "Pretrained representations carry language knowledge, so they need far fewer labels; with all tweets TF-IDF catches up (difference " + E("boot.probe_tfidf.mean", "+.1f") + " points, interval including zero).",
      "The data-scaling chart shows the largest advantage of frozen FinBERT at the smallest training size."),
  ("Variabile înghețate", "Vectorii FinBERT înghețați cu o regresie logistică au atins " + R("models.finbert_probe.macro_f1", ".1f") + " % pe toate tweet-urile de antrenare. Unde ajută cel mai mult?",
   ["Cu puține etichete: " + R("scaling.probe_small", ".1f") + " % cu " + R("scaling.n_small") + " de tweet-uri, față de " + R("scaling.tfidf_small", ".1f") + " % pentru TF-IDF", "Doar cu toate tweet-urile", "Niciodată: sînt mereu mai slabi decît TF-IDF", "Doar pe documente lungi"],
   "Reprezentările preantrenate conțin cunoștințe de limbă, deci au nevoie de mult mai puține etichete; cu toate tweet-urile TF-IDF ajunge din urmă (diferența " + R("boot.probe_tfidf.mean", "+.1f") + " puncte, interval care conține zero).",
   "Graficul cu volumul datelor arată cel mai mare avantaj al FinBERT înghețat la cel mai mic număr de exemple de antrenare."))
q(2, ("Bootstrap interval", "The paired bootstrap gives FinBERT fine-tuned minus TF-IDF = " + E("boot.ft_tfidf.mean", "+.1f") + " points, 95 % interval [" + E("boot.ft_tfidf.lo", "+.1f") + "; " + E("boot.ft_tfidf.hi", "+.1f") + "]. What does this mean?",
      ["The difference is due to chance", "TF-IDF is better on average", "The advantage of fine-tuned FinBERT is unlikely to be a sampling accident on this test set", "The test set was too small to say anything"],
      "The interval lies well above zero, so the gain survives resampling of the test tweets.",
      "An interval that excludes zero supports a real difference; here even its lower end is " + E("boot.ft_tfidf.lo", "+.1f") + " points."),
  ("Intervalul bootstrap", "Bootstrap-ul pereche dă FinBERT ajustat minus TF-IDF = " + R("boot.ft_tfidf.mean", "+.1f") + " puncte, intervalul de 95 % [" + R("boot.ft_tfidf.lo", "+.1f") + "; " + R("boot.ft_tfidf.hi", "+.1f") + "]. Ce înseamnă?",
   ["Diferența este întîmplătoare", "TF-IDF este mai bun în medie", "Avantajul FinBERT ajustat este puțin probabil să fie un accident de eșantionare pe acest set de test", "Setul de test a fost prea mic pentru a spune ceva"],
   "Intervalul este mult deasupra lui zero, deci cîștigul rezistă la reeșantionarea tweet-urilor de test.",
   "Un interval care nu conține zero susține o diferență reală; aici chiar și capătul inferior este " + R("boot.ft_tfidf.lo", "+.1f") + " puncte."))
q(1, ("Macro-F1", "Why is macro-F1, not accuracy, the headline metric for the tweets?",
      ["Accuracy cannot be computed for three classes", "It averages the per-class F1, so the rare bearish class counts as much as the frequent neutral class", "It ignores precision", "It is always higher than accuracy"],
      "With " + E("data.pct_neutral_test", ".1f") + " % neutral tweets, accuracy is dominated by the neutral class; macro-F1 weighs each class equally.",
      "The costly errors are on bearish and bullish news, which are the minority; macro-F1 gives them equal weight with the neutral majority."),
  ("Macro-F1", "De ce este macro-F1, și nu acuratețea, măsura principală pentru tweet-uri?",
   ["Acuratețea nu se poate calcula pentru trei clase", "Face media F1 pe clase, deci clasa negativă rară contează cît clasa neutră frecventă", "Ignoră precizia", "Este mereu mai mare decît acuratețea"],
   "Cu " + R("data.pct_neutral_test", ".1f") + " % tweet-uri neutre, acuratețea este dominată de clasa neutră; macro-F1 dă fiecărei clase aceeași pondere.",
   "Erorile costisitoare apar la știrile negative și pozitive, care sînt minoritare; macro-F1 le dă aceeași pondere ca majorității neutre."))
q(3, ("Remaining errors", "After fine-tuning, which confusion is most frequent for bearish tweets?",
      ["Bearish read as bullish", "No errors remain", "Neutral read as bearish", "Bearish read as neutral (" + E("cm.bear_as_neutral") + " tweets)"],
      "Most remaining errors are with the neutral class: " + E("cm.bear_as_neutral") + " bearish tweets read as neutral, only " + E("cm.bear_as_bull") + " read as bullish.",
      "The model rarely flips the sign of sentiment; it mostly underestimates the strength of a mildly negative tweet."),
  ("Erorile rămase", "După ajustarea fină, care confuzie este cea mai frecventă pentru tweet-urile negative?",
   ["Negative citite ca pozitive", "Nu mai rămîne nicio eroare", "Neutre citite ca negative", "Negative citite ca neutre (" + R("cm.bear_as_neutral") + " de tweet-uri)"],
   "Cele mai multe erori rămase sînt cu clasa neutră: " + R("cm.bear_as_neutral") + " de tweet-uri negative citite ca neutre, doar " + R("cm.bear_as_bull") + " citite ca pozitive.",
   "Modelul rareori inversează semnul sentimentului; de cele mai multe ori subestimează intensitatea unui tweet ușor negativ."))
q(0, ("Fine-tuning learning rate", "Why is FinBERT fine-tuned with a small learning rate ($2\\cdot10^{-5}$) and only 3 epochs?",
      ["To adapt to tweets without erasing what pretraining learned (catastrophic forgetting)", "Because larger rates are not allowed by AdamW", "To make the model smaller", "Because the tokenizer changes during training"],
      "Small steps keep the weights near the pretrained solution, so the language knowledge stays and the decision layer adapts.",
      "A large learning rate would overwrite the pretrained weights; the small rate is a deliberate choice, not a software limit."),
  ("Rata de învățare la ajustare", "De ce se ajustează FinBERT cu o rată de învățare mică ($2\\cdot10^{-5}$) și doar 3 epoci?",
   ["Ca să se adapteze la tweet-uri fără să șteargă ce a învățat la preantrenare (uitare catastrofală)", "Pentru că AdamW nu permite rate mai mari", "Ca să micșoreze modelul", "Pentru că tokenizatorul se schimbă la antrenare"],
   "Pașii mici țin ponderile aproape de soluția preantrenată, deci cunoștințele de limbă rămîn, iar stratul de decizie se adaptează.",
   "O rată de învățare mare ar suprascrie ponderile preantrenate; rata mică este o alegere deliberată, nu o limită a programului."))
q(2, ("Cost of fine-tuning", "How large is FinBERT and how long did one fine-tuning run take on a laptop graphics processor?",
      ["2 million weights, a few seconds", "1 billion weights, a day", E("models.finbert_ft.params_m", ".1f") + " million weights, about " + E("models.finbert_ft.min_per_seed", ".1f") + " minutes per seed", "109 weights, instantly"],
      "BERT-base has " + E("models.finbert_ft.params_m", ".1f") + " million weights; three epochs on " + E("data.n_train", ",") + " tweets took about " + E("models.finbert_ft.min_per_seed", ".1f") + " minutes.",
      "Fine-tuning a base-size encoder on a few thousand short texts is cheap today: minutes, not days."),
  ("Costul ajustării", "Cît de mare este FinBERT și cît a durat o ajustare pe procesorul grafic al unui laptop?",
   ["2 milioane de ponderi, cîteva secunde", "1 miliard de ponderi, o zi", R("models.finbert_ft.params_m", ".1f") + " milioane de ponderi, circa " + R("models.finbert_ft.min_per_seed", ".1f") + " minute pe sămînță", "109 ponderi, instantaneu"],
   "BERT-base are " + R("models.finbert_ft.params_m", ".1f") + " milioane de ponderi; trei epoci pe " + R("data.n_train", ",") + " de tweet-uri au durat circa " + R("models.finbert_ft.min_per_seed", ".1f") + " minute.",
   "Ajustarea unui codificator de dimensiune medie pe cîteva mii de texte scurte este azi ieftină: minute, nu zile."))
q(1, ("Attention as explanation", "Can an attention map justify a credit or trading decision to a regulator?",
      ["Yes, attention weights are the reasons of the model", "No: different maps can give the same prediction, and much attention goes to separator tokens", "Yes, if the heads are averaged", "Only for decoder models"],
      "Up to " + E("attn.sep_max_pct", ".1f") + " % of FinBERT's attention goes to [SEP], and attention ignores the size of the values and the residual stream; attribution methods are needed.",
      "Attention shows where a head looked, not why the model decided; Jain and Wallace (2019) found different maps with the same predictions."),
  ("Atenția ca explicație", "Poate o hartă de atenție să justifice în fața unui supraveghetor o decizie de credit sau de tranzacționare?",
   ["Da, ponderile de atenție sînt motivele modelului", "Nu: hărți diferite pot da aceeași predicție, iar multă atenție merge spre tokenii separatori", "Da, dacă se face media pe capete", "Doar pentru modelele decodor"],
   "Pînă la " + R("attn.sep_max_pct", ".1f") + " % din atenția FinBERT merge spre [SEP], iar atenția ignoră mărimea valorilor și fluxul rezidual; sînt necesare metode de atribuire.",
   "Atenția arată unde s-a uitat un cap, nu de ce a decis modelul; Jain și Wallace (2019) au găsit hărți diferite cu aceleași predicții."))
q(3, ("Entropy of attention", "How does the normalised entropy of FinBERT's attention change from layer 1 to the middle layers?",
      ["It stays at 1 (uniform attention) in every layer", "It increases to 1 in the last layer", "It is always 0", "It falls from " + E("attn.ent_l1", ".2f") + " in layer 1 to " + E("attn.ent_min", ".2f") + " in layer " + E("attn.ent_min_layer")],
      "Early layers spread attention broadly; middle layers concentrate it on few tokens, often the separators.",
      "Entropy divided by $\\log n$ equals 1 for uniform attention; FinBERT's middle layers are much sharper than its first layer."),
  ("Entropia atenției", "Cum se schimbă entropia normalizată a atenției FinBERT de la stratul 1 la straturile din mijloc?",
   ["Rămîne 1 (atenție uniformă) în toate straturile", "Crește la 1 în ultimul strat", "Este mereu 0", "Scade de la " + R("attn.ent_l1", ".2f") + " în stratul 1 la " + R("attn.ent_min", ".2f") + " în stratul " + R("attn.ent_min_layer")],
   "Primele straturi împrăștie atenția; straturile din mijloc o concentrează pe puțini tokeni, adesea pe separatori.",
   "Entropia împărțită la $\\log n$ este 1 pentru atenție uniformă; straturile din mijloc ale FinBERT sînt mult mai concentrate decît primul strat."))
q(0, ("Leakage from the benchmark data", "Why was FinBERT not evaluated on Financial PhraseBank in this lecture?",
      ["FinBERT was fine-tuned on Financial PhraseBank, so a test there would be contaminated", "PhraseBank has no labels", "PhraseBank sentences are too long", "PhraseBank is in Finnish"],
      "A model must be tested on data it has not seen in any training stage; tweets are new to every model compared.",
      "PhraseBank is labelled, short and in English; the issue is that FinBERT already learned from it."),
  ("Contaminarea datelor de test", "De ce nu a fost evaluat FinBERT pe Financial PhraseBank în acest curs?",
   ["FinBERT a fost ajustat pe Financial PhraseBank, deci un test acolo ar fi contaminat", "PhraseBank nu are etichete", "Propozițiile PhraseBank sînt prea lungi", "PhraseBank este în finlandeză"],
   "Un model trebuie testat pe date pe care nu le-a văzut în nicio etapă de antrenare; tweet-urile sînt noi pentru toate modelele comparate.",
   "PhraseBank este etichetat, scurt și în engleză; problema este că FinBERT a învățat deja din el."))
q(2, ("Validation set", "How were the hyperparameters and the stopping epoch chosen without touching the test tweets?",
      ["On the test tweets, then reported on the same tweets", "On Financial PhraseBank", "On a 10 % stratified validation part of the published training tweets", "They were not chosen"],
      "The published training split was divided into " + E("data.n_train", ",") + " training and " + E("data.n_val", ",") + " validation tweets; the test tweets were used once.",
      "Choosing settings on the test set inflates the reported score; the validation part plays that role instead."),
  ("Setul de validare", "Cum au fost alese hiperparametrii și epoca de oprire fără a atinge tweet-urile de test?",
   ["Pe tweet-urile de test, apoi raportate pe aceleași tweet-uri", "Pe Financial PhraseBank", "Pe o parte de validare de 10 %, stratificată, din tweet-urile de antrenare publicate", "Nu au fost aleși"],
   "Setul de antrenare publicat a fost împărțit în " + R("data.n_train", ",") + " de tweet-uri de antrenare și " + R("data.n_val", ",") + " de validare; tweet-urile de test au fost folosite o singură dată.",
   "Alegerea setărilor pe setul de test umflă scorul raportat; partea de validare are acest rol."))
q(1, ("Overfitting signal", "For the small transformer (seed 1) the validation loss is lowest at epoch " + E("models.tiny.minloss_epoch_seed1") + " and then rises. What does this show?",
      ["The model is underfitting", "The model becomes overconfident on the training data: it overfits", "The learning rate is too small", "The validation set is leaking"],
      "Training loss keeps falling while validation loss rises: the classic sign of overfitting, even though the macro-F1 still creeps up.",
      "A rising validation loss with a falling training loss means memorisation; the model's probabilities become too extreme."),
  ("Semnalul supraînvățării", "Pentru transformerul mic (sămînța 1), pierderea pe validare este minimă la epoca " + R("models.tiny.minloss_epoch_seed1") + " și apoi crește. Ce arată acest lucru?",
   ["Modelul nu învață suficient", "Modelul devine prea sigur pe datele de antrenare: supraînvață", "Rata de învățare este prea mică", "Setul de validare scurge informație"],
   "Pierderea pe antrenare continuă să scadă, iar cea pe validare crește: semnul clasic al supraînvățării, chiar dacă macro-F1 mai urcă puțin.",
   "O pierdere de validare în creștere, cu o pierdere de antrenare în scădere, înseamnă memorare; probabilitățile modelului devin prea extreme."))
q(3, ("Long documents", "What is a practical way to classify a 3,000-token central-bank speech with a 512-token encoder?",
      ["Truncate to the first 50 tokens", "Increase the learning rate", "Remove the positional encoding", "Split it into chunks of at most 512 tokens and pool the chunk scores"],
      "Chunking keeps each piece inside the window; pooling (mean, max or attention over chunks) gives a document-level score.",
      "Truncation discards most of the text; chunking or hierarchical models use all of it within the window limit."),
  ("Documente lungi", "Care este o soluție practică pentru clasificarea unui discurs de bancă centrală de 3 000 de tokeni cu un codificator de 512 tokeni?",
   ["Tăierea la primii 50 de tokeni", "Creșterea ratei de învățare", "Eliminarea codificării poziției", "Împărțirea în bucăți de cel mult 512 tokeni și agregarea scorurilor pe bucăți"],
   "Împărțirea ține fiecare bucată în fereastră; agregarea (medie, maxim sau atenție pe bucăți) dă un scor la nivel de document.",
   "Tăierea aruncă cea mai mare parte a textului; împărțirea în bucăți sau modelele ierarhice îl folosesc integral, în limita ferestrei."))
q(0, ("From score to signal", "A desk aggregates tweet sentiment by firm and day. Which rule prevents look-ahead bias?",
      ["Use only texts time-stamped before the decision time", "Use the closing price of the same day as a feature", "Average sentiment over the whole sample", "Fine-tune on the test period"],
      "A signal used at time $t$ may contain only information available at $t$; texts published later would leak the future.",
      "Any input known only after the decision, including later tweets or same-day closing prices, inflates backtests."),
  ("De la scor la semnal", "O mesă de tranzacționare agregă sentimentul tweet-urilor pe firmă și zi. Ce regulă previne folosirea informației din viitor?",
   ["Folosirea numai a textelor cu marcaj de timp anterior momentului deciziei", "Folosirea prețului de închidere din aceeași zi ca variabilă", "Media sentimentului pe întregul eșantion", "Ajustarea pe perioada de test"],
   "Un semnal folosit la momentul $t$ poate conține doar informația disponibilă la $t$; textele publicate mai tîrziu ar scurge viitorul.",
   "Orice intrare cunoscută abia după decizie, inclusiv tweet-urile ulterioare sau prețul de închidere din aceeași zi, umflă testele retrospective."))
q(2, ("GAT link", "Which model of Lecture 9 is attention restricted to the edges of a graph?",
      ["The graph convolutional network", "GraphSAGE", "The graph attention network (GAT)", "The random forest"],
      "A GAT computes softmax attention weights over a node's neighbours only; a transformer does the same over all pairs of tokens.",
      "The GCN uses fixed degree weights and GraphSAGE a mean of sampled neighbours; only the GAT learns attention weights."),
  ("Legătura cu GAT", "Care model din cursul 9 este atenție restrînsă la muchiile unui graf?",
   ["Rețeaua convoluțională pe graf", "GraphSAGE", "Rețeaua cu atenție pe graf (GAT)", "Pădurea aleatoare"],
   "O rețea GAT calculează ponderi de atenție softmax numai pe vecinii unui nod; un transformer face același lucru pe toate perechile de tokeni.",
   "GCN folosește ponderi fixe după grad, iar GraphSAGE o medie a vecinilor eșantionați; doar GAT învață ponderi de atenție."))
q(1, ("Group research", "What do LLM-VaR/LLM-ES (Pele et al., 2026) and the foundation-model study of Jheng et al. (2026) share with FinBERT?",
      ["They are all trained only on tweets", "They use models pretrained on large data, applied with little or no task-specific training, and judge them by backtests", "They replace backtests by attention maps", "They all classify sentiment"],
      "Pretraining once and reusing the model is the common idea; for risk numbers the evaluation is by VaR and ES backtests.",
      "The papers forecast Value-at-Risk and expected shortfall, not sentiment; what they share with FinBERT is the pretrain-then-reuse logic."),
  ("Cercetarea grupului", "Ce au în comun LLM-VaR/LLM-ES (Pele et al., 2026) și studiul despre modelele fundaționale al lui Jheng et al. (2026) cu FinBERT?",
   ["Toate sînt antrenate numai pe tweet-uri", "Folosesc modele preantrenate pe date multe, aplicate cu puțină sau fără antrenare specifică, și le judecă prin teste retrospective", "Înlocuiesc testele retrospective cu hărți de atenție", "Toate clasifică sentimentul"],
   "Preantrenarea o singură dată și refolosirea modelului este ideea comună; pentru măsurile de risc evaluarea se face prin testele retrospective VaR și ES.",
   "Lucrările prognozează valoarea la risc și pierderea așteptată în coadă, nu sentimentul; ce au în comun cu FinBERT este logica preantrenare-apoi-refolosire."))
q(0, ("Embedding bag", "A model that only averages learned token embeddings, without attention, reached " + E("models.embbag.macro_f1", ".1f") + " %. What does this suggest about finance tweets?",
      ["They are short and formulaic, so a bag of signal words already carries much of the sentiment", "Attention is useless for all texts", "Embeddings cannot be learned", "The test set is mislabelled"],
      "The gap to the small transformer is small (" + E("models.tiny.macro_f1", ".1f") + " %): on short texts most of the signal sits in individual words.",
      "This holds for short tweets; for longer texts with negation and contrast, attention and order matter much more."),
  ("Media reprezentărilor", "Un model care doar face media reprezentărilor învățate ale tokenilor, fără atenție, a atins " + R("models.embbag.macro_f1", ".1f") + " %. Ce sugerează acest lucru despre tweet-urile financiare?",
   ["Sînt scurte și formulaice, deci o colecție de cuvinte-semnal conține deja mare parte din sentiment", "Atenția este inutilă pentru orice text", "Reprezentările nu pot fi învățate", "Setul de test este etichetat greșit"],
   "Diferența față de transformerul mic este mică (" + R("models.tiny.macro_f1", ".1f") + " %): pe texte scurte, cea mai mare parte a semnalului stă în cuvintele izolate.",
   "Acest lucru este valabil pentru tweet-uri scurte; pentru texte mai lungi, cu negație și contrast, atenția și ordinea contează mult mai mult."))
q(3, ("Bearish recall", "Which share of the bearish test tweets does fine-tuned FinBERT find, and with what precision?",
      [E("models.tfidf.rec_bearish", ".1f") + " % with " + E("models.tfidf.prec_bearish", ".1f") + " % precision", "100 % with 100 % precision", E("models.tiny.rec_bearish", ".1f") + " % with " + E("models.tiny.prec_bearish", ".1f") + " % precision",
       E("models.finbert_ft.rec_bearish", ".1f") + " % with " + E("models.finbert_ft.prec_bearish", ".1f") + " % precision"],
      "Fine-tuned FinBERT finds " + E("models.finbert_ft.rec_bearish", ".1f") + " % of the bearish tweets and " + E("models.finbert_ft.prec_bearish", ".1f") + " % of its bearish calls are right.",
      "The other pairs belong to TF-IDF and to the small transformer; fine-tuned FinBERT is better on both recall and precision of the bearish class."),
  ("Recall-ul negativ", "Ce pondere dintre tweet-urile negative de test găsește FinBERT ajustat și cu ce precizie?",
   [R("models.tfidf.rec_bearish", ".1f") + " %, cu precizia " + R("models.tfidf.prec_bearish", ".1f") + " %", "100 %, cu precizia 100 %", R("models.tiny.rec_bearish", ".1f") + " %, cu precizia " + R("models.tiny.prec_bearish", ".1f") + " %",
    R("models.finbert_ft.rec_bearish", ".1f") + " %, cu precizia " + R("models.finbert_ft.prec_bearish", ".1f") + " %"],
   "FinBERT ajustat găsește " + R("models.finbert_ft.rec_bearish", ".1f") + " % dintre tweet-urile negative, iar " + R("models.finbert_ft.prec_bearish", ".1f") + " % dintre semnalările lui negative sînt corecte.",
   "Celelalte perechi aparțin TF-IDF și transformerului mic; FinBERT ajustat este mai bun atît la recall, cît și la precizia clasei negative."))

if __name__ == "__main__":
    assert len(Q) >= 30, len(Q)
    for x in Q:
        assert x["correct"] in (0, 1, 2, 3) and len(x["en"]["options"]) == 4 and len(x["ro"]["options"]) == 4
    head = ("// ============================================================\n"
            "// Quiz bank for lecture id 'l7': Attention and transformers (EN + RO)\n"
            "// GENERATED by lectures/L7/make_quiz_L7.py from L7_numbers.json; do not edit by hand.\n"
            "// correct = index (0-3) of the right option in the original order; incorrectExplanation never names a letter.\n"
            "// ============================================================\n")
    js = head + "window.NN_DATA.quizzes['l7'] = " + json.dumps({"draw": 20, "questions": Q}, ensure_ascii=False, indent=1) + ";\n"
    with open(os.path.join(ROOT, "assets", "quizzes", "l7.js"), "w", encoding="utf-8") as fh:
        fh.write(js)
    print("assets/quizzes/l7.js:", len(Q), "questions")
