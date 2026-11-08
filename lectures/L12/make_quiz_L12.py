"""
make_quiz_L12.py -- quiz bank of Lecture 12 (EN + RO) -> assets/quizzes/l12.js.
Every number in a question comes from L12_numbers.json; correct = index (0-3) of the right option in the original order.
Run:  python3 lectures/L12/make_quiz_L12.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "L12_numbers.json")))
OUT = os.path.join(HERE, "..", "..", "assets", "quizzes", "l12.js")
C, R, V = N["cls"], N["rag"], N["llmvar"]


def en(x, d=1):
    return f"{x:,.{d}f}"


def ro(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


Q = []


def q(correct, e, r):
    k = ("title", "text", "options", "correctExplanation", "incorrectExplanation")
    Q.append({"correct": correct, "en": dict(zip(k, e)), "ro": dict(zip(k, r))})


q(1, ("Language-model objective", "What does a large language model learn during pre-training?",
      ["To classify sentences into topics", "The probability of the next token given the previous tokens", "To translate between languages only", "A reward for helpful answers"],
      "Pre-training minimises the cross-entropy of the next token; every later use (answers, code, classification) is built on this probability.",
      "Classification, translation and rewards come later or are special uses; the pre-training objective is next-token prediction."),
  ("Obiectivul modelului de limbaj", "Ce învață un model lingvistic mare la preantrenare?",
   ["Să clasifice propoziții pe teme", "Probabilitatea următorului token, dată fiind secvența de tokeni anteriori", "Doar să traducă între limbi", "O recompensă pentru răspunsuri utile"],
   "Preantrenarea minimizează entropia încrucișată a următorului token; toate utilizările ulterioare (răspunsuri, cod, clasificare) se bazează pe această probabilitate.",
   "Clasificarea, traducerea și recompensele vin mai tîrziu sau sînt utilizări speciale; obiectivul preantrenării este prezicerea următorului token."))
P3 = N["ppl"]["3B"]
q(2, ("Perplexity", f"On ECB speeches after July 2025, Qwen2.5-3B had an average next-token loss of {en(P3['nll_new'], 3)} nats. What is its perplexity?",
      [en(P3['nll_new'], 3), en(1 / P3['nll_new'], 2), en(P3['new'], 2), en(100 * P3['nll_new'], 0)],
      f"Perplexity = exp(loss) = e^{en(P3['nll_new'], 3)} ≈ {en(P3['new'], 2)}: the model is as uncertain as if it chose among about {P3['new']:.0f} equally likely tokens.",
      "Perplexity is the exponential of the average cross-entropy, not the loss itself, its inverse or a percentage."),
  ("Perplexitatea", f"Pe discursurile BCE de după iulie 2025, Qwen2.5-3B a avut o pierdere medie a următorului token de {ro(P3['nll_new'], 3)} nats. Cît este perplexitatea?",
   [ro(P3['nll_new'], 3), ro(1 / P3['nll_new'], 2), ro(P3['new'], 2), ro(100 * P3['nll_new'], 0)],
   f"Perplexitatea = exp(pierderea) = e^{ro(P3['nll_new'], 3)} ≈ {ro(P3['new'], 2)}: modelul este la fel de nesigur ca și cum ar alege între circa {P3['new']:.0f} tokeni la fel de probabili.",
   "Perplexitatea este exponențiala entropiei încrucișate medii, nu pierderea însăși, inversa ei sau un procent."))
q(0, ("Model size", f"On ECB speeches after July 2025 the perplexity was {en(N['ppl']['05B']['new'], 2)} for Qwen2.5-0.5B and {en(N['ppl']['3B']['new'], 2)} for 3B. What does this show?",
      ["The larger model predicts this unseen text better", "The smaller model is better calibrated", "The larger model memorised these speeches", "Perplexity cannot compare models"],
      "Both models use the same tokenizer, so lower perplexity on the same unseen text means better next-token prediction.",
      "The speeches were published after the training data, so memorisation cannot explain the gap; same-tokenizer perplexities are comparable."),
  ("Dimensiunea modelului", f"Pe discursurile BCE de după iulie 2025, perplexitatea a fost {ro(N['ppl']['05B']['new'], 2)} pentru Qwen2.5-0.5B și {ro(N['ppl']['3B']['new'], 2)} pentru 3B. Ce arată aceasta?",
   ["Modelul mai mare prezice mai bine acest text nevăzut", "Modelul mai mic este mai bine calibrat", "Modelul mai mare a memorat aceste discursuri", "Perplexitatea nu poate compara modele"],
   "Ambele modele folosesc același tokenizator, deci o perplexitate mai mică pe același text nevăzut înseamnă o prezicere mai bună a următorului token.",
   "Discursurile au fost publicate după datele de antrenare, deci memorarea nu explică diferența; perplexitățile cu același tokenizator sînt comparabile."))
q(3, ("Contamination", f"The 3B model had perplexity {en(N['ppl']['3B']['old'], 2)} on 2022–2023 speeches and {en(N['ppl']['3B']['new'], 2)} on speeches after July 2025. What is the careful reading?",
      ["Proof that the model memorised the old speeches", "Proof that the language of the ECB became more complex", "No information at all",
       "A possible sign of contamination: evaluate on text written after the training cut-off"],
      "Lower perplexity on text the model may have seen is a warning sign, but topics also changed; it is a symptom, not a proof.",
      "The gap alone cannot separate memorisation from topic change; it is still informative and motivates post-cut-off test sets."),
  ("Contaminarea", f"Modelul de 3B a avut perplexitatea {ro(N['ppl']['3B']['old'], 2)} pe discursurile din 2022–2023 și {ro(N['ppl']['3B']['new'], 2)} pe cele de după iulie 2025. Care este interpretarea prudentă?",
   ["Dovada că modelul a memorat discursurile vechi", "Dovada că limbajul BCE a devenit mai complex", "Nicio informație",
    "Un posibil semn de contaminare: evaluați pe texte scrise după limita datelor de antrenare"],
   "O perplexitate mai mică pe text pe care modelul l-ar fi putut vedea este un semnal de avertizare, dar și temele s-au schimbat; este un simptom, nu o dovadă.",
   "Diferența singură nu poate separa memorarea de schimbarea temelor; rămîne totuși informativă și justifică seturi de test ulterioare limitei."))
q(1, ("Tokens per word", f"On the Financial PhraseBank there are {en(N['tok']['qwen_per_word'], 2)} Qwen tokens per word. Why does this matter for a bank?",
      ["It does not matter", "Costs and context limits are counted in tokens, not words", "It determines the accuracy of the model", "It fixes the temperature"],
      "Prices of commercial models and the maximum context are expressed in tokens, so budgets must convert words to tokens.",
      "Tokenisation does not set accuracy or temperature; it sets how much text fits in a context and how much it costs."),
  ("Tokeni pe cuvînt", f"Pe Financial PhraseBank revin {ro(N['tok']['qwen_per_word'], 2)} tokeni Qwen pe cuvînt. De ce contează pentru o bancă?",
   ["Nu contează", "Costurile și limitele contextului se numără în tokeni, nu în cuvinte", "Determină acuratețea modelului", "Fixează temperatura"],
   "Prețurile modelelor comerciale și contextul maxim se exprimă în tokeni, deci bugetele trebuie să transforme cuvintele în tokeni.",
   "Tokenizarea nu stabilește acuratețea sau temperatura; stabilește cît text încape într-un context și cît costă."))
q(2, ("Temperature", "What happens to the next-token distribution when the temperature T increases from 0.5 to 2?",
      ["It becomes more peaked", "It does not change", "It becomes flatter: entropy rises and unlikely tokens become plausible", "The most likely token changes"],
      f"Dividing logits by a larger T flattens the softmax; in our example the entropy rose from {en(N['next']['entropy']['t05'], 2)} to {en(N['next']['entropy']['t20'], 2)} nats.",
      "Temperature rescales logits without changing their order, so the top token stays the same while the distribution spreads out."),
  ("Temperatura", "Ce se întîmplă cu distribuția următorului token cînd temperatura T crește de la 0,5 la 2?",
   ["Devine mai concentrată", "Nu se schimbă", "Devine mai plată: entropia crește, iar tokenii improbabili devin plauzibili", "Se schimbă tokenul cel mai probabil"],
   f"Împărțirea scorurilor la un T mai mare aplatizează softmax; în exemplul nostru entropia a crescut de la {ro(N['next']['entropy']['t05'], 2)} la {ro(N['next']['entropy']['t20'], 2)} nats.",
   "Temperatura rescalează scorurile fără să le schimbe ordinea, deci tokenul cel mai probabil rămîne același, iar distribuția se împrăștie."))
q(0, ("Classification temperature", "Which temperature should be used to classify 10,000 news items with an LLM?",
      ["Zero (or read the label probabilities directly)", "1.5, for diversity", "2, to avoid overconfidence", "A random temperature per item"],
      "Classification needs a deterministic answer; at T = 0 or with label probabilities, the same item always gets the same label.",
      "Sampling at a positive temperature makes labels random by construction; diversity is a goal of generation, not of classification."),
  ("Temperatura pentru clasificare", "Ce temperatură trebuie folosită pentru a clasifica 10.000 de știri cu un LLM?",
   ["Zero (sau citirea directă a probabilităților etichetelor)", "1,5, pentru diversitate", "2, pentru a evita încrederea excesivă", "O temperatură aleatoare pentru fiecare știre"],
   "Clasificarea are nevoie de un răspuns determinist; la T = 0 sau cu probabilitățile etichetelor, aceeași știre primește mereu aceeași etichetă.",
   "Eșantionarea la o temperatură pozitivă face etichetele aleatoare prin construcție; diversitatea este un scop al generării, nu al clasificării."))
q(3, ("SFT loss", "In supervised fine-tuning on (instruction, answer) pairs, on which tokens is the loss computed?",
      ["On the instruction tokens only", "On all tokens of the conversation equally", "On the system prompt", "On the answer tokens only"],
      "The model reads the instruction as context and is graded on producing the reference answer, token by token.",
      "Grading the instruction would teach the model to write prompts; the loss is restricted to the answer."),
  ("Pierderea SFT", "La ajustarea fină supervizată pe perechi (instrucțiune, răspuns), pe ce tokeni se calculează pierderea?",
   ["Doar pe tokenii instrucțiunii", "Pe toți tokenii conversației, în mod egal", "Pe promptul de sistem", "Doar pe tokenii răspunsului"],
   "Modelul citește instrucțiunea drept context și este evaluat pe producerea răspunsului de referință, token cu token.",
   "Evaluarea instrucțiunii l-ar învăța pe model să scrie prompturi; pierderea se limitează la răspuns."))
q(1, ("LoRA", "A 4096 × 4096 weight matrix is adapted with LoRA of rank 8. How many trainable weights does the adapter have?",
      ["16,777,216", "65,536", "4,096", "8"],
      "B is 4096 × 8 and A is 8 × 4096: 2 × 4096 × 8 = 65,536 weights instead of 16.8 million.",
      "Only the two low-rank factors are trained; the full matrix stays frozen."),
  ("LoRA", "O matrice de ponderi 4096 × 4096 este adaptată cu LoRA de rang 8. Cîte ponderi antrenabile are adaptorul?",
   ["16.777.216", "65.536", "4.096", "8"],
   "B este 4096 × 8, iar A este 8 × 4096: 2 × 4096 × 8 = 65.536 de ponderi în loc de 16,8 milioane.",
   "Se antrenează doar cei doi factori de rang mic; matricea completă rămîne fixă."))
q(2, ("DPO", "What does direct preference optimisation (DPO) remove from the RLHF pipeline?",
      ["The pre-training stage", "The human preference data", "The separate reward model and the reinforcement-learning loop", "The supervised fine-tuning model used as reference"],
      "DPO optimises the policy directly on preference pairs with a logistic loss, using the SFT model as reference; no reward model is fitted.",
      "DPO still needs preference data and a reference model; it skips the explicit reward model and the policy-gradient optimisation."),
  ("DPO", "Ce elimină optimizarea directă a preferințelor (DPO) din fluxul RLHF?",
   ["Etapa de preantrenare", "Datele de preferințe umane", "Modelul de recompensă separat și bucla de învățare prin recompensă", "Modelul SFT folosit ca referință"],
   "DPO optimizează politica direct pe perechile de preferințe, cu o pierdere logistică, folosind modelul SFT ca referință; nu se estimează un model de recompensă.",
   "DPO are în continuare nevoie de date de preferințe și de un model de referință; renunță la modelul explicit de recompensă și la optimizarea prin gradientul politicii."))
q(0, ("Label probabilities", "How does the lecture turn a prompt into a classifier without generating text?",
      ["It reads the probability of the first token of each label word at the answer position and normalises over labels", "It counts how often each label appears in the prompt",
       "It asks the model to output a JSON list", "It trains a new output layer"],
      "One forward pass gives the next-token distribution; summing the probabilities of each label's first tokens and normalising gives class probabilities.",
      "No training and no parsing are needed: the label probabilities are read directly from the model's next-token distribution."),
  ("Probabilitățile etichetelor", "Cum transformă cursul un prompt în clasificator fără a genera text?",
   ["Citește probabilitatea primului token al fiecărui cuvînt-etichetă în poziția răspunsului și normalizează pe etichete", "Numără de cîte ori apare fiecare etichetă în prompt",
    "Cere modelului o listă JSON", "Antrenează un nou strat de ieșire"],
   "O singură trecere înainte dă distribuția următorului token; însumarea probabilităților primilor tokeni ai fiecărei etichete și normalizarea dau probabilitățile claselor.",
   "Nu este nevoie de antrenare sau de interpretarea textului: probabilitățile etichetelor se citesc direct din distribuția următorului token."))
q(1, ("Macro-F1", "Why does the lecture report macro-F1 rather than accuracy on the Financial PhraseBank?",
      ["Macro-F1 is always higher", f"Neutral sentences are {en(N['data']['pb_share_neutral'])} % of the data, so accuracy rewards predicting 'neutral'; macro-F1 weighs each class equally",
       "Accuracy cannot be computed with three classes", "Macro-F1 does not need a test set"],
      "Macro-F1 averages the F1 of the three classes, so the small negative class counts as much as the large neutral class.",
      "With an imbalanced label distribution, accuracy is dominated by the majority class; macro-F1 is the balanced alternative."),
  ("Macro-F1", "De ce raportează cursul macro-F1, și nu acuratețea, pe Financial PhraseBank?",
   ["Macro-F1 este întotdeauna mai mare", f"Propozițiile neutre reprezintă {ro(N['data']['pb_share_neutral'])} % din date, deci acuratețea răsplătește răspunsul „neutru”; macro-F1 ponderează egal fiecare clasă",
    "Acuratețea nu se poate calcula cu trei clase", "Macro-F1 nu are nevoie de un set de test"],
   "Macro-F1 face media F1 pe cele trei clase, deci clasa mică negativă contează cît clasa mare neutră.",
   "Cînd etichetele sînt dezechilibrate, acuratețea este dominată de clasa majoritară; macro-F1 este alternativa echilibrată."))
q(2, ("Zero-shot versus trained", f"On PhraseBank, zero-shot Qwen2.5-3B reached {en(C['pb']['zs_3B']['f1'])} % macro-F1 and TF-IDF with logistic regression {en(C['pb']['tfidf']['f1'])} %. What is the main practical difference?",
      ["TF-IDF used a larger language model", "The LLM was fine-tuned on PhraseBank", f"TF-IDF needed {N['data']['pb_tr']:,} labelled sentences; the prompt needed none", "Both used the same labelled data"],
      "The zero-shot prompt uses no labelled example at all, while the classical model is trained on the labelled training split.",
      "No fine-tuning took place: the LLM reads only the instruction and the sentence; the classical model learns from the labels."),
  ("Zero-shot față de antrenat", f"Pe PhraseBank, Qwen2.5-3B zero-shot a ajuns la {ro(C['pb']['zs_3B']['f1'])} % macro-F1, iar TF-IDF cu regresie logistică la {ro(C['pb']['tfidf']['f1'])} %. Care este diferența practică principală?",
   ["TF-IDF a folosit un model de limbaj mai mare", "LLM a fost ajustat fin pe PhraseBank", f"TF-IDF a avut nevoie de {N['data']['pb_tr']:,} de propoziții etichetate; promptul, de niciuna".replace(",", "."), "Ambele au folosit aceleași date etichetate"],
   "Promptul zero-shot nu folosește niciun exemplu etichetat, în timp ce modelul clasic este antrenat pe partea etichetată de antrenare.",
   "Nu a existat nicio ajustare fină: LLM citește doar instrucțiunea și propoziția; modelul clasic învață din etichete."))
q(0, ("Small model", f"Zero-shot Qwen2.5-0.5B reached only {en(C['pb']['zs_05B']['f1'])} % macro-F1 on PhraseBank. What does this show?",
      ["Instruction following on a nuanced task needs a sufficiently large model", "Prompting never works", "The data set is mislabelled", "Macro-F1 is wrong for this task"],
      f"The same prompt gives {en(C['pb']['zs_15B']['f1'])} % with 1.5B and {en(C['pb']['zs_3B']['f1'])} % with 3B parameters: capacity matters.",
      "The larger models of the same family succeed with the same prompt and data, so the problem is the size of the smallest model."),
  ("Modelul mic", f"Qwen2.5-0.5B zero-shot a ajuns doar la {ro(C['pb']['zs_05B']['f1'])} % macro-F1 pe PhraseBank. Ce arată aceasta?",
   ["Urmarea instrucțiunilor pe o sarcină nuanțată cere un model suficient de mare", "Prompturile nu funcționează niciodată", "Setul de date este etichetat greșit", "Macro-F1 nu este potrivit pentru această sarcină"],
   f"Același prompt dă {ro(C['pb']['zs_15B']['f1'])} % cu 1,5B și {ro(C['pb']['zs_3B']['f1'])} % cu 3B parametri: capacitatea contează.",
   "Modelele mai mari din aceeași familie reușesc cu același prompt și aceleași date, deci problema este dimensiunea celui mai mic model."))
q(3, ("Bootstrap interval", "How are the 95 % intervals of macro-F1 in the lecture computed?",
      ["From the normal approximation of accuracy", "From the three few-shot seeds", "From cross-validation folds", "By resampling the test set with replacement 1,000 times and taking the 2.5 and 97.5 percentiles"],
      "The bootstrap redraws the test sentences with replacement, recomputes macro-F1 each time and reads the percentiles of the 1,000 values.",
      "Seeds measure the randomness of the example choice; the bootstrap interval measures the sampling uncertainty of the test set."),
  ("Intervalul bootstrap", "Cum se calculează intervalele de 95 % pentru macro-F1 din curs?",
   ["Din aproximarea normală a acurateței", "Din cele trei rulări few-shot", "Din părțile validării încrucișate", "Prin reeșantionarea setului de test cu revenire de 1.000 de ori și luarea percentilelor 2,5 și 97,5"],
   "Bootstrap-ul reextrage propozițiile de test cu revenire, recalculează macro-F1 de fiecare dată și citește percentilele celor 1.000 de valori.",
   "Rulările măsoară caracterul aleator al alegerii exemplelor; intervalul bootstrap măsoară incertitudinea de eșantionare a setului de test."))
q(1, ("Few-shot", f"For Qwen2.5-1.5B, macro-F1 was {en(N['fewshot']['15B']['0']['mean'])} % with no example and {en(N['fewshot']['15B']['8']['mean'])} % with 24 examples. What is the lesson?",
      ["More examples always help", "More examples do not always help: test the number of examples like any other setting", "Few-shot prompting is forbidden", "Examples change the model weights"],
      "In-context examples can help, hurt or change little; their number and choice are settings to validate, not free improvements.",
      "Examples in the prompt never change the weights, and their effect depends on the task and the model."),
  ("Few-shot", f"Pentru Qwen2.5-1.5B, macro-F1 a fost {ro(N['fewshot']['15B']['0']['mean'])} % fără exemple și {ro(N['fewshot']['15B']['8']['mean'])} % cu 24 de exemple. Care este lecția?",
   ["Mai multe exemple ajută întotdeauna", "Mai multe exemple nu ajută întotdeauna: testați numărul de exemple ca pe orice altă setare", "Prompturile few-shot sînt interzise", "Exemplele schimbă ponderile modelului"],
   "Exemplele din context pot ajuta, pot strica sau pot schimba puțin; numărul și alegerea lor sînt setări de validat, nu îmbunătățiri gratuite.",
   "Exemplele din prompt nu schimbă niciodată ponderile, iar efectul lor depinde de sarcină și de model."))
q(2, ("Prompt sensitivity", f"Five paraphrases of the same instruction gave a macro-F1 range of {en(N['sens']['tw_3B']['range'])} points on Twitter for the 3B model. What should a bank do?",
      ["Pick the best paraphrase on the test set", "Average all five answers on the test set", "Choose the wording on a validation set, freeze it and version it", "Ignore the wording"],
      "The wording is a hyper-parameter; selecting it on the test set would inflate the reported score.",
      "Any choice made on the test set becomes optimistic; the wording must be fixed before the final test."),
  ("Sensibilitatea la prompt", f"Cinci reformulări ale aceleiași instrucțiuni au dat o amplitudine a macro-F1 de {ro(N['sens']['tw_3B']['range'])} puncte pe Twitter pentru modelul de 3B. Ce trebuie să facă o bancă?",
   ["Să aleagă cea mai bună reformulare pe setul de test", "Să facă media celor cinci răspunsuri pe setul de test", "Să aleagă formularea pe un set de validare, să o fixeze și să o versioneze", "Să ignore formularea"],
   "Formularea este un hiperparametru; alegerea ei pe setul de test ar umfla scorul raportat.",
   "Orice alegere făcută pe setul de test devine optimistă; formularea trebuie fixată înainte de testul final."))
q(0, ("Chain-of-thought", f"Chain-of-thought gave {en(C['pb']['cot_3B']['f1'])} % on PhraseBank against {en(C['pb']['zs_3B']['f1'])} % for the direct zero-shot answer, at about {en(C['pb']['cot_3B']['ms'], 0)} ms per sentence. What is the conclusion?",
      ["For one-word labels on short sentences, chain-of-thought costs more and did not help here", "Chain-of-thought always improves accuracy", "The direct answer is slower", "Chain-of-thought removes hallucinations"],
      "Reasoning text helps multi-step problems; for short classification it added cost without a gain on our data.",
      "Generated reasoning is not free and not always useful; it must be tested like any other prompt."),
  ("Chain-of-thought", f"Chain-of-thought a dat {ro(C['pb']['cot_3B']['f1'])} % pe PhraseBank, față de {ro(C['pb']['zs_3B']['f1'])} % pentru răspunsul direct zero-shot, cu circa {ro(C['pb']['cot_3B']['ms'], 0)} ms pe propoziție. Care este concluzia?",
   ["Pentru etichete de un cuvînt pe propoziții scurte, chain-of-thought costă mai mult și nu a ajutat aici", "Chain-of-thought îmbunătățește întotdeauna acuratețea", "Răspunsul direct este mai lent", "Chain-of-thought elimină halucinațiile"],
   "Textul de raționament ajută problemele în mai mulți pași; pentru clasificarea scurtă a adăugat cost fără cîștig, pe datele noastre.",
   "Raționamentul generat nu este gratuit și nu este întotdeauna util; trebuie testat ca orice alt prompt."))
q(3, ("Calibration", f"What does an expected calibration error (ECE) of {en(N['calib']['zs_3B'])} % mean?",
      [f"The model is wrong {en(N['calib']['zs_3B'])} % of the time", f"The model abstains in {en(N['calib']['zs_3B'])} % of cases", f"{en(N['calib']['zs_3B'])} % of labels are noisy", f"On average, the stated confidence differs from the observed accuracy by about {en(N['calib']['zs_3B'])} points"],
      "ECE averages, over confidence bins, the gap between mean confidence and accuracy, weighted by the share of items in each bin.",
      "ECE measures the reliability of probabilities, not the error rate or label noise."),
  ("Calibrarea", f"Ce înseamnă o eroare de calibrare așteptată (ECE) de {ro(N['calib']['zs_3B'])} %?",
   [f"Modelul greșește în {ro(N['calib']['zs_3B'])} % din cazuri", f"Modelul se abține în {ro(N['calib']['zs_3B'])} % din cazuri", f"{ro(N['calib']['zs_3B'])} % dintre etichete sînt eronate", f"În medie, încrederea declarată diferă de acuratețea observată cu circa {ro(N['calib']['zs_3B'])} puncte"],
   "ECE face media, pe intervale de încredere, a diferenței dintre încrederea medie și acuratețe, ponderată cu ponderea elementelor din fiecare interval.",
   "ECE măsoară fiabilitatea probabilităților, nu rata erorii sau zgomotul din etichete."))
ag = N["agree"]["zs_3B"]
q(1, ("Annotator agreement", f"Zero-shot 3B was right on {en(ag['100'])} % of unanimous sentences and on {en(ag['50'])} % of sentences with 50 % annotator agreement. Why?",
      ["The model is biased against long sentences", "Sentences that split human annotators are genuinely ambiguous; no model can be more certain than its labels", "The 50 % sentences are in another language", "The prompt was changed"],
      "When half of the annotators chose another label, the 'true' label is itself uncertain; accuracy falls for every model.",
      "Same model, same prompt, same language: what changes is the ambiguity of the sentences."),
  ("Acordul adnotatorilor", f"Modelul zero-shot de 3B a avut dreptate pentru {ro(ag['100'])} % din propozițiile unanime și pentru {ro(ag['50'])} % din cele cu acord de 50 %. De ce?",
   ["Modelul este părtinitor față de propozițiile lungi", "Propozițiile care îi împart pe adnotatori sînt cu adevărat ambigue; niciun model nu poate fi mai sigur decît etichetele sale", "Propozițiile de 50 % sînt în altă limbă", "Promptul a fost schimbat"],
   "Cînd jumătate dintre adnotatori au ales altă etichetă, eticheta „adevărată” este ea însăși nesigură; acuratețea scade pentru orice model.",
   "Același model, același prompt, aceeași limbă: se schimbă doar ambiguitatea propozițiilor."))
q(2, ("Embedding probe", "What is an 'embedding probe' in this lecture?",
      ["A prompt with many examples", "A test of the tokenizer", "A logistic regression trained on the averaged hidden states of a frozen LLM", "Fine-tuning all weights of the LLM"],
      f"The LLM is frozen; only a linear classifier is trained on its sentence representations ({en(C['tw']['probe']['f1'])} % macro-F1 on Twitter).",
      "No weight of the LLM changes and no prompt is used; the labels train a small classifier on top of the representations."),
  ("Clasificatorul pe embedding", "Ce este un „clasificator pe embedding” în acest curs?",
   ["Un prompt cu multe exemple", "Un test al tokenizatorului", "O regresie logistică antrenată pe stările ascunse mediate ale unui LLM fix", "Ajustarea fină a tuturor ponderilor LLM"],
   f"LLM rămîne fix; se antrenează doar un clasificator liniar pe reprezentările propozițiilor ({ro(C['tw']['probe']['f1'])} % macro-F1 pe Twitter).",
   "Nicio pondere a LLM nu se schimbă și nu se folosește niciun prompt; etichetele antrenează un mic clasificator peste reprezentări."))
q(0, ("Cost", f"TF-IDF needed {en(C['pb']['tfidf']['ms'], 2)} ms per sentence and zero-shot 3B {en(C['pb']['zs_3B']['ms'])} ms. When is the LLM still the better choice?",
      ["When there are no labelled examples to train the small model", "When there are millions of labelled documents", "Always", "Never"],
      "Without labels a classical model cannot be trained, while a prompt works immediately; with many labels and many documents, the cheap model wins.",
      "Cost and the availability of labels decide; neither option dominates in every situation."),
  ("Costul", f"TF-IDF a avut nevoie de {ro(C['pb']['tfidf']['ms'], 2)} ms pe propoziție, iar zero-shot 3B de {ro(C['pb']['zs_3B']['ms'])} ms. Cînd este totuși LLM alegerea mai bună?",
   ["Cînd nu există exemple etichetate pentru a antrena modelul mic", "Cînd există milioane de documente etichetate", "Întotdeauna", "Niciodată"],
   "Fără etichete nu se poate antrena un model clasic, în timp ce un prompt funcționează imediat; cu multe etichete și multe documente, modelul ieftin cîștigă.",
   "Costul și disponibilitatea etichetelor decid; nicio variantă nu este cea mai bună în orice situație."))
q(3, ("RAG", "In retrieval-augmented generation, what is p_η(z | q)?",
      ["The probability of the answer", "The temperature of the generator", "The language-model loss", "The retriever's relevance of document z for question q"],
      "RAG factorises the answer into retrieving documents z for the question q and generating the answer from q and z.",
      "The generator gives the answer probability; the retriever scores the relevance of each document."),
  ("RAG", "În generarea augmentată prin regăsire, ce este p_η(z | q)?",
   ["Probabilitatea răspunsului", "Temperatura generatorului", "Pierderea modelului de limbaj", "Relevanța documentului z pentru întrebarea q, dată de modulul de regăsire"],
   "RAG descompune răspunsul în regăsirea documentelor z pentru întrebarea q și generarea răspunsului din q și z.",
   "Generatorul dă probabilitatea răspunsului; modulul de regăsire evaluează relevanța fiecărui document."))
q(1, ("BM25", "Which property distinguishes BM25 from raw word counts?",
      ["It uses neural embeddings", "It weights words by rarity and saturates repeated occurrences, with a length normalisation", "It ignores stop words only", "It needs labelled data"],
      "BM25 multiplies an inverse document frequency by a saturating term-frequency factor that also corrects for chunk length.",
      "BM25 is a lexical, unsupervised score; it needs no embeddings and no labels."),
  ("BM25", "Ce proprietate deosebește BM25 de simpla numărare a cuvintelor?",
   ["Folosește embedding neuronale", "Ponderează cuvintele după raritate și saturează aparițiile repetate, cu o normalizare după lungime", "Ignoră doar cuvintele de legătură", "Are nevoie de date etichetate"],
   "BM25 înmulțește frecvența inversă în documente cu un factor de frecvență care se saturează și corectează și lungimea fragmentului.",
   "BM25 este un scor lexical, nesupervizat; nu are nevoie de embedding sau de etichete."))
q(2, ("Hybrid retrieval", f"Recall@1 was {en(R['bm25']['r1'])} % for BM25, {en(R['dense']['r1'])} % for dense retrieval and {en(R['hybrid']['r1'])} % for the hybrid. Why can the hybrid beat both?",
      ["It uses a larger language model", "It is trained on the test titles", "Lexical and semantic matches fail on different queries, so fusing the rankings recovers more", "It retrieves more chunks"],
      "Reciprocal rank fusion rewards documents ranked high by either method; their errors are partly independent.",
      "The hybrid uses the same chunks and no training; its gain comes from combining two different notions of similarity."),
  ("Regăsirea hibridă", f"Recall@1 a fost {ro(R['bm25']['r1'])} % pentru BM25, {ro(R['dense']['r1'])} % pentru regăsirea densă și {ro(R['hybrid']['r1'])} % pentru metoda hibridă. De ce poate metoda hibridă să le depășească pe amîndouă?",
   ["Folosește un model de limbaj mai mare", "Este antrenată pe titlurile de test", "Potrivirile lexicale și cele semantice eșuează pe interogări diferite, deci fuziunea clasamentelor recuperează mai mult", "Regăsește mai multe fragmente"],
   "Fuziunea rangurilor reciproce răsplătește documentele clasate sus de oricare dintre metode; erorile lor sînt parțial independente.",
   "Metoda hibridă folosește aceleași fragmente și nicio antrenare; cîștigul vine din combinarea a două noțiuni diferite de similaritate."))
q(0, ("Hallucination", f"Asked who gave an ECB speech from 2025–2026 without documents and forced to answer, the model was wrong in {en(R['qa_closed']['wrong'])} % of cases. With 'I don't know' allowed it abstained in {en(R['qa_closed_abstain']['abstain'])} %. What follows?",
      ["Allowing abstention turns confident errors into honest abstentions, but only retrieval gives correct answers", "The model knows the answers but hides them", "Abstention makes the model more accurate", "The questions were ambiguous"],
      f"With retrieved documents the model answered correctly in {en(R['qa_rag']['correct'])} % of cases: grounding, not instructions, supplies the missing knowledge.",
      "The speeches are after the training data, so the model cannot know; abstaining avoids wrong names but does not create correct ones."),
  ("Halucinațiile", f"Întrebat cine a ținut un discurs BCE din 2025–2026, fără documente și obligat să răspundă, modelul a greșit în {ro(R['qa_closed']['wrong'])} % din cazuri. Cînd „nu știu” a fost permis, s-a abținut în {ro(R['qa_closed_abstain']['abstain'])} %. Ce rezultă?",
   ["Permiterea abținerii transformă erorile sigure în abțineri oneste, dar doar regăsirea aduce răspunsuri corecte", "Modelul știe răspunsurile, dar le ascunde", "Abținerea face modelul mai precis", "Întrebările erau ambigue"],
   f"Cu documentele regăsite, modelul a răspuns corect în {ro(R['qa_rag']['correct'])} % din cazuri: ancorarea în documente, nu instrucțiunile, aduce cunoștințele lipsă.",
   "Discursurile sînt ulterioare datelor de antrenare, deci modelul nu le poate cunoaște; abținerea evită numele greșite, dar nu creează răspunsuri corecte."))
q(3, ("Where RAG fails", f"RAG answered correctly in {en(R['qa_rag']['correct'])} % of cases and the oracle (the right speech given) in {en(R['qa_oracle']['correct'])} %. Where is the remaining gap?",
      ["In the language model's reading ability", "In the temperature", "In the chat template", "Mostly in retrieval: the right speech was not always among the top 3"],
      f"With the right document the model is almost always correct; recall@3 of the hybrid retriever on these questions was {en(R['qa_r3_hybrid'])} %.",
      "The oracle shows that reading is not the bottleneck; improving retrieval closes most of the gap."),
  ("Unde greșește RAG", f"RAG a răspuns corect în {ro(R['qa_rag']['correct'])} % din cazuri, iar oracolul (cu discursul corect furnizat) în {ro(R['qa_oracle']['correct'])} %. Unde este diferența rămasă?",
   ["În capacitatea de citire a modelului de limbaj", "În temperatură", "În șablonul de conversație", "Mai ales în regăsire: discursul corect nu a fost întotdeauna printre primele 3"],
   f"Cu documentul corect, modelul are aproape întotdeauna dreptate; recall@3 al metodei hibride pe aceste întrebări a fost {ro(R['qa_r3_hybrid'])} %.",
   "Oracolul arată că citirea nu este blocajul; îmbunătățirea regăsirii închide cea mai mare parte a diferenței."))
q(1, ("LLM-VaR idea", "In the LLM-VaR framework of Pele et al. (2026), what does the model receive and return?",
      ["Prices and news; a buy or sell signal", "A window of past returns in a structured prompt; a quantile estimate (VaR) without any fine-tuning", "A trained GARCH model; its parameters", "Balance sheets; a credit score"],
      "The general-purpose model reads past returns as text and answers with the quantile, zero-shot.",
      "LLM-VaR is a zero-shot risk estimator from returns; it is not a trading or credit model and involves no estimation."),
  ("Ideea LLM-VaR", "În cadrul LLM-VaR al lui Pele et al. (2026), ce primește și ce întoarce modelul?",
   ["Prețuri și știri; un semnal de cumpărare sau vînzare", "O fereastră de randamente trecute într-un prompt structurat; o estimare a cuantilei (VaR), fără ajustare fină", "Un model GARCH estimat; parametrii lui", "Bilanțuri; un scor de credit"],
   "Modelul general citește randamentele trecute ca text și răspunde cu cuantila, fără reantrenare.",
   "LLM-VaR este un estimator de risc zero-shot pe baza randamentelor; nu este un model de tranzacționare sau de credit și nu presupune estimare."))
q(2, ("LLM-VaR result", f"With a 3B open model, the 5 % LLM-VaR had {V['llm_05']['viol']} violations (expected {en(V['exp_05'])}) and a mean of {en(V['llm_05']['mean'], 2)} %, while GARCH had a mean of {en(V['garch_05']['mean'], 2)} %. Their correlation was {en(V['corr_llm_garch_05'], 2)}. What is the right reading?",
      ["The small model tracks volatility better than GARCH", "The small model is useless for any purpose", "Coverage is acceptable because the small model is cautious, but its VaR hardly reacts to volatility", "GARCH was not estimated"],
      "A deeper average VaR passes the count test, but a near-zero correlation with GARCH shows it does not adapt to changing risk; the published study used much larger models.",
      "Passing Kupiec is necessary, not sufficient: a constant cautious level can pass while ignoring volatility clusters."),
  ("Rezultatul LLM-VaR", f"Cu un model deschis de 3B, LLM-VaR de 5 % a avut {V['llm_05']['viol']} de depășiri (așteptate {ro(V['exp_05'])}) și o medie de {ro(V['llm_05']['mean'], 2)} %, iar GARCH o medie de {ro(V['garch_05']['mean'], 2)} %. Corelația lor a fost {ro(V['corr_llm_garch_05'], 2)}. Care este interpretarea corectă?",
   ["Modelul mic urmărește volatilitatea mai bine decît GARCH", "Modelul mic nu este util pentru nimic", "Acoperirea este acceptabilă pentru că modelul mic este prudent, dar VaR sa abia reacționează la volatilitate", "GARCH nu a fost estimat"],
   "O VaR medie mai jos trece testul numărării, dar o corelație aproape nulă cu GARCH arată că nu se adaptează la riscul variabil; studiul publicat a folosit modele mult mai mari.",
   "Trecerea testului Kupiec este necesară, nu suficientă: un nivel prudent constant poate trece testul ignorînd grupările de volatilitate."))
q(0, ("Kupiec test", "What does the Kupiec unconditional-coverage test check?",
      ["Whether the share of violations equals the nominal level α", "Whether violations are independent", "Whether returns are normal", "Whether two VaR models are equal"],
      "It is a likelihood-ratio test of the observed violation rate against α; independence is tested separately.",
      "Kupiec only counts violations; clustering is the subject of the Christoffersen test."),
  ("Testul Kupiec", "Ce verifică testul Kupiec al acoperirii necondiționate?",
   ["Dacă ponderea depășirilor este egală cu nivelul nominal α", "Dacă depășirile sînt independente", "Dacă randamentele sînt normale", "Dacă două modele VaR sînt egale"],
   "Este un test al raportului de verosimilitate pentru rata observată a depășirilor față de α; independența se testează separat.",
   "Testul Kupiec doar numără depășirile; gruparea lor este subiectul testului Christoffersen."))
q(3, ("Answer prefix", "Why did the LLM-VaR prompt end with the beginning of the answer ('The 0.05 quantile of tomorrow's return is ')?",
      ["To raise the temperature", "To give the model the right answer", "To shorten the context", "To force the model to write a number instead of an explanation, so that the answer can be parsed"],
      "Without a prefix, a small model often starts explaining; prefilling the answer constrains the output format.",
      "The prefix contains no number and does not change the temperature or context length; it fixes the output format."),
  ("Prefixul răspunsului", "De ce s-a încheiat promptul LLM-VaR cu începutul răspunsului („The 0.05 quantile of tomorrow's return is ”)?",
   ["Pentru a crește temperatura", "Pentru a-i da modelului răspunsul corect", "Pentru a scurta contextul", "Pentru a-l obliga pe model să scrie un număr în loc de o explicație, astfel încît răspunsul să poată fi citit"],
   "Fără prefix, un model mic începe adesea să explice; precompletarea răspunsului restrînge formatul ieșirii.",
   "Prefixul nu conține niciun număr și nu schimbă temperatura sau lungimea contextului; fixează formatul ieșirii."))
q(1, ("Data protection", "A bank wants to classify client complaints with an LLM. Which option keeps client data in-house?",
      ["Sending the texts to a public chatbot", "An open model run on the bank's own hardware", "Posting the texts in a shared spreadsheet", "Using a larger temperature"],
      "Local open models process the text without it leaving the institution, which simplifies data-protection obligations.",
      "Any external service receives the text; only local processing keeps it inside the bank."),
  ("Protecția datelor", "O bancă vrea să clasifice reclamațiile clienților cu un LLM. Ce variantă păstrează datele clienților în instituție?",
   ["Trimiterea textelor unui chatbot public", "Un model deschis rulat pe infrastructura proprie a băncii", "Publicarea textelor într-o foaie de calcul partajată", "Folosirea unei temperaturi mai mari"],
   "Modelele deschise rulate local prelucrează textul fără ca acesta să părăsească instituția, ceea ce simplifică obligațiile de protecție a datelor.",
   "Orice serviciu extern primește textul; doar prelucrarea locală îl păstrează în bancă."))
q(2, ("Retrieval evaluation", "Your RAG assistant gives a wrong answer although the right document is in the corpus. What do you check first?",
      ["The temperature", "The length of the answer", "Whether the right document is among the top-k retrieved chunks (recall@k)", "The font of the documents"],
      "If the document is not retrieved, no prompt can fix the answer; the retriever is evaluated first.",
      "Generation can only use what retrieval provides; recall@k diagnoses the most frequent failure."),
  ("Evaluarea regăsirii", "Asistentul RAG dă un răspuns greșit, deși documentul corect se află în corpus. Ce verificați mai întîi?",
   ["Temperatura", "Lungimea răspunsului", "Dacă documentul corect este printre primele k fragmente regăsite (recall@k)", "Fontul documentelor"],
   "Dacă documentul nu este regăsit, niciun prompt nu poate corecta răspunsul; se evaluează întîi modulul de regăsire.",
   "Generarea poate folosi doar ce oferă regăsirea; recall@k diagnostichează eșecul cel mai frecvent."))
q(0, ("Confusion", f"The best prompt confused {N['confusion']['m'][2][0]} neutral sentences with positive ones but only {N['confusion']['m'][1][0] + N['confusion']['m'][0][1]} positive and negative ones with each other. What does this suggest?",
      ["Errors sit on the neutral border, e.g. 'profit rose, but less than expected'", "The model cannot read negative words", "The labels are random", "The model only predicts neutral"],
      "Mixed or mildly positive news is the hard case; opposite polarities are rarely confused.",
      "The model distinguishes good from bad news well; its errors concentrate where the sentiment is weak."),
  ("Confuzia", f"Cel mai bun prompt a confundat {N['confusion']['m'][2][0]} de propoziții neutre cu cele pozitive, dar doar {N['confusion']['m'][1][0] + N['confusion']['m'][0][1]} propoziții pozitive și negative între ele. Ce sugerează aceasta?",
   ["Erorile stau la granița cu neutrul, de exemplu „profitul a crescut, dar mai puțin decît se aștepta”", "Modelul nu poate citi cuvintele negative", "Etichetele sînt aleatoare", "Modelul prezice doar neutru"],
   "Știrile mixte sau ușor pozitive sînt cazul dificil; polaritățile opuse se confundă rar.",
   "Modelul distinge bine știrile bune de cele rele; erorile se concentrează acolo unde sentimentul este slab."))

assert len(Q) >= 30, len(Q)
js = ("// ============================================================\n"
      "// Quiz bank for lecture id 'l12': Large language models in finance and prompt engineering (EN + RO)\n"
      "// GENERATED by lectures/L12/make_quiz_L12.py from L12_numbers.json; do not edit by hand.\n"
      "// correct = index (0-3) of the right option in the original order; incorrectExplanation never names a letter.\n"
      "// ============================================================\n"
      "window.NN_DATA.quizzes['l12'] = " + json.dumps({"draw": 20, "questions": Q}, ensure_ascii=False, indent=1) + ";\n")
with open(OUT, "w", encoding="utf-8") as f:
    f.write(js)
print(f"{len(Q)} questions written to assets/quizzes/l12.js")
