"""
make_quiz_L14.py -- quiz bank of Lecture 14 (EN + RO) -> assets/quizzes/l14.js.
Every number in a question comes from L14_numbers.json; correct = index (0-3) of the right option in the original order.
Run:  python3 lectures/L14/make_quiz_L14.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "L14_numbers.json")))
OUT = os.path.join(HERE, "..", "..", "assets", "quizzes", "l14.js")
Z, F, RG, VL, TB = N["zs"], N["fewshot"], N["regime"], N["vl"], N["table"]


def en(x, d=1):
    return f"{x:,.{d}f}"


def ro(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


Q = []


def q(correct, e, r):
    k = ("title", "text", "options", "correctExplanation", "incorrectExplanation")
    Q.append({"correct": correct, "en": dict(zip(k, e)), "ro": dict(zip(k, r))})


q(1, ("Patches", "A 224 × 224 image is cut into 16 × 16 patches by a vision transformer. How many tokens enter the transformer, including the class token?",
      ["196", "197", "256", "50,176"],
      "14 × 14 = 196 patch tokens plus one class token = 197.",
      "The image is divided into 224/16 = 14 patches per side; the class token is added to the sequence."),
  ("Peticele", "O imagine de 224 × 224 este tăiată în petice de 16 × 16 de un transformer pentru imagini. Cîți tokeni intră în transformer, inclusiv tokenul de clasă?",
   ["196", "197", "256", "50.176"],
   "14 × 14 = 196 de tokeni-petic plus un token de clasă = 197.",
   "Imaginea se împarte în 224/16 = 14 petice pe latură; tokenul de clasă se adaugă secvenței."))
q(3, ("Why ViTs need pre-training", "Why does a vision transformer need much more training data than a CNN of similar size?",
      ["It has no attention", "It cannot process colour", "It uses a smaller image", "It has no built-in locality or translation invariance, so it must learn them from data"],
      "Convolutions build locality into the architecture; a ViT treats patches as a set with positions and learns spatial structure from examples.",
      "A ViT is built on attention and handles colour and full-size images; what it lacks is the inductive bias of convolutions."),
  ("De ce are nevoie ViT de preantrenare", "De ce are nevoie un transformer pentru imagini de mult mai multe date de antrenare decît o CNN de dimensiune similară?",
   ["Nu are atenție", "Nu poate prelucra culoarea", "Folosește o imagine mai mică", "Nu are localitatea și invarianța la translație încorporate, deci trebuie să le învețe din date"],
   "Convoluțiile încorporează localitatea în arhitectură; un ViT tratează peticele ca o mulțime cu poziții și învață structura spațială din exemple.",
   "Un ViT se bazează pe atenție și prelucrează culoarea și imaginile întregi; îi lipsește prejudecata inductivă a convoluțiilor."))
q(0, ("Contrastive loss", "In CLIP's contrastive loss, what plays the role of the 'class labels' for image i in a batch of n pairs?",
      ["The n captions of the batch; the correct class is caption i", "Hand-labelled ImageNet classes", "The pixel values", "The temperature τ"],
      "Each row of the similarity matrix is a softmax classifier over the batch captions, with its own caption as the target.",
      "CLIP uses no class labels: the matching caption in the batch is the target, the others are negatives."),
  ("Pierderea contrastivă", "În pierderea contrastivă a CLIP, ce joacă rolul „etichetelor de clasă” pentru imaginea i dintr-un lot de n perechi?",
   ["Cele n descrieri din lot; clasa corectă este descrierea i", "Clasele ImageNet etichetate manual", "Valorile pixelilor", "Temperatura τ"],
   "Fiecare rînd al matricei de similaritate este un clasificator softmax peste descrierile din lot, cu propria descriere drept țintă.",
   "CLIP nu folosește etichete de clasă: descrierea potrivită din lot este ținta, celelalte sînt exemple negative."))
q(2, ("Zero-shot classifier", "How does CLIP classify an image zero-shot?",
      ["It retrains its last layer on the classes", "It counts pixels of each colour", "It embeds one sentence per class and picks the class whose text embedding is closest to the image embedding", "It asks a person"],
      "The class prompts become the weight vectors; the cosine similarity with the image embedding decides the class.",
      "No training happens in zero-shot use; the text encoder turns the class names into the classifier."),
  ("Clasificatorul zero-shot", "Cum clasifică CLIP o imagine fără reantrenare (zero-shot)?",
   ["Își reantrenează ultimul strat pe clase", "Numără pixelii fiecărei culori", "Calculează embedding-ul unei propoziții pentru fiecare clasă și alege clasa al cărei embedding de text este cel mai apropiat de cel al imaginii", "Întreabă un om"],
   "Prompturile claselor devin vectorii de ponderi; similaritatea cosinus cu embedding-ul imaginii decide clasa.",
   "În utilizarea zero-shot nu are loc nicio antrenare; codificatorul de text transformă numele claselor în clasificator."))
q(1, ("Prompt wording", f"ViT-B/16 classified {en(Z['B16_name'])} % of EuroSAT test images with bare folder names and {en(Z['B16_sat'])} % with 'a satellite photo of {{class}}.'. Why?",
      ["The second prompt is longer", "Telling the model the kind of image matches the captions it saw in pre-training", "Folder names contain typos", "The second prompt uses a larger model"],
      "Web captions describe photos in context; naming the image type moves the text embedding towards the right region of the joint space.",
      "Same model, same images: only the wording changed, and the domain cue explains the gain."),
  ("Formularea promptului", f"ViT-B/16 a clasificat corect {ro(Z['B16_name'])} % din imaginile de test EuroSAT cu numele simple ale dosarelor și {ro(Z['B16_sat'])} % cu „a satellite photo of {{class}}.”. De ce?",
   ["Al doilea prompt este mai lung", "Precizarea tipului de imagine se potrivește cu descrierile văzute la preantrenare", "Numele dosarelor conțin greșeli", "Al doilea prompt folosește un model mai mare"],
   "Descrierile de pe internet prezintă fotografiile în context; numirea tipului de imagine mută embedding-ul de text spre regiunea potrivită a spațiului comun.",
   "Același model, aceleași imagini: s-a schimbat doar formularea, iar indiciul de domeniu explică cîștigul."))
q(3, ("Prompt ensemble", f"Averaging five satellite templates gave {en(Z['B16_ens'])} % with ViT-B/16. What is a prompt ensemble?",
      ["Five models voting", "Five images per class", "Five retrainings", "The mean of the text embeddings of several templates for the same class, used as one weight vector"],
      "Averaging embeddings of several phrasings reduces the noise of any single wording.",
      "The ensemble is built in text-embedding space with one model; no image or training is added."),
  ("Ansamblul de prompturi", f"Media a cinci șabloane satelitare a dat {ro(Z['B16_ens'])} % cu ViT-B/16. Ce este un ansamblu de prompturi?",
   ["Cinci modele care votează", "Cinci imagini pe clasă", "Cinci reantrenări", "Media embedding-urilor de text ale mai multor șabloane pentru aceeași clasă, folosită ca un singur vector de ponderi"],
   "Media embedding-urilor mai multor formulări reduce zgomotul unei formulări unice.",
   "Ansamblul se construiește în spațiul embedding-urilor de text, cu un singur model; nu se adaugă imagini sau antrenare."))
tc = N["zs_top_conf"]
q(0, ("Confusion", f"{tc['n']} of the {tc['n_true']} sea-or-lake test images were labelled 'highway' by zero-shot CLIP. What is the best fix?",
      ["Label a few images per class and train a probe, or rewrite the two prompts", "Use a larger image", "Lower the temperature", "Remove the class"],
      "Look-alike classes need either sharper descriptions or a few labelled examples; a probe with one image per class already beats zero-shot.",
      "The problem is the class description in the joint space; resolution or temperature do not separate the classes."),
  ("Confuzia", f"{tc['n']} dintre cele {tc['n_true']} de imagini de test cu mare sau lac au fost etichetate „autostradă” de CLIP zero-shot. Care este cea mai bună soluție?",
   ["Etichetarea cîtorva imagini pe clasă și antrenarea unui clasificator pe embedding sau rescrierea celor două prompturi", "O imagine mai mare", "Scăderea temperaturii", "Eliminarea clasei"],
   "Clasele care arată la fel au nevoie fie de descrieri mai precise, fie de cîteva exemple etichetate; un clasificator cu o imagine pe clasă depășește deja zero-shot.",
   "Problema este descrierea clasei în spațiul comun; rezoluția sau temperatura nu separă clasele."))
q(2, ("Linear probe", f"With one labelled image per class, a logistic regression on CLIP embeddings reached {en(F['probe']['1']['mean'])} %, above zero-shot ({en(Z[N['zs_best']])} %). Why is this cheap?",
      ["It retrains the whole CLIP model", "It uses the test images", "Only a linear classifier on frozen embeddings is trained, in seconds", "It needs a graphics card for hours"],
      "The embeddings are computed once; fitting a logistic regression on 10 to 1,280 vectors takes seconds on a laptop.",
      "CLIP stays frozen and the test set is not used for training; the cost is that of a small logistic regression."),
  ("Clasificatorul liniar", f"Cu o singură imagine etichetată pe clasă, o regresie logistică pe embedding-urile CLIP a ajuns la {ro(F['probe']['1']['mean'])} %, peste zero-shot ({ro(Z[N['zs_best']])} %). De ce este ieftin?",
   ["Reantrenează întregul model CLIP", "Folosește imaginile de test", "Se antrenează doar un clasificator liniar pe embedding-uri fixe, în cîteva secunde", "Are nevoie de o placă grafică ore întregi"],
   "Embedding-urile se calculează o singură dată; estimarea unei regresii logistice pe 10 pînă la 1.280 de vectori durează cîteva secunde pe un laptop.",
   "CLIP rămîne fix, iar setul de test nu se folosește la antrenare; costul este cel al unei regresii logistice mici."))
q(1, ("Labels and models", f"With 16 images per class the CLIP probe reached {en(F['probe']['16']['mean'])} % and a CNN from scratch {en(F['cnn']['16']['mean'])} %; with 128 per class {en(F['probe']['128']['mean'])} % against {en(F['cnn']['128']['mean'])} %. What is the lesson?",
      ["Pre-trained features always win", "Pre-trained features win with few labels; with many labels a small CNN can catch up and pass them", "CNNs are always better", "The number of labels does not matter"],
      "Transfer helps most when labels are scarce; with enough labels a model trained on the target data adapts better to its specifics.",
      "The ranking reverses as labels grow: the choice depends on how many labels you have."),
  ("Etichete și modele", f"Cu 16 imagini pe clasă, clasificatorul pe CLIP a ajuns la {ro(F['probe']['16']['mean'])} %, iar o CNN de la zero la {ro(F['cnn']['16']['mean'])} %; cu 128 pe clasă, {ro(F['probe']['128']['mean'])} % față de {ro(F['cnn']['128']['mean'])} %. Care este lecția?",
   ["Trăsăturile preantrenate cîștigă întotdeauna", "Trăsăturile preantrenate cîștigă cu puține etichete; cu multe etichete, o CNN mică le poate ajunge și depăși", "CNN este întotdeauna mai bună", "Numărul etichetelor nu contează"],
   "Transferul ajută cel mai mult cînd etichetele sînt puține; cu suficiente etichete, un model antrenat pe datele-țintă se adaptează mai bine la particularitățile lor.",
   "Ordinea se inversează pe măsură ce etichetele cresc: alegerea depinde de numărul de etichete disponibile."))
q(3, ("Seeds", "Why is the few-shot curve shown with a shaded band over 5 seeds?",
      ["To make the chart colourful", "Because the test set changes", "Because the model changes", "Because with few labels the result depends strongly on which images are drawn"],
      f"With one image per class the probe ranged from {en(F['probe']['1']['min'])} % to {en(F['probe']['1']['max'])} % across seeds.",
      "The test set and the model are fixed; the randomness comes from the choice of the few labelled images."),
  ("Rulări repetate", "De ce este curba cu puține exemple arătată cu o bandă colorată pe 5 rulări?",
   ["Pentru ca graficul să fie colorat", "Pentru că setul de test se schimbă", "Pentru că modelul se schimbă", "Pentru că, cu puține etichete, rezultatul depinde puternic de imaginile extrase"],
   f"Cu o imagine pe clasă, clasificatorul a variat între {ro(F['probe']['1']['min'])} % și {ro(F['probe']['1']['max'])} % de la o rulare la alta.",
   "Setul de test și modelul sînt fixe; caracterul aleator vine din alegerea celor cîteva imagini etichetate."))
S = N["search"]
q(0, ("Text-to-image search", f"Ranking test images by similarity to each class prompt, {en(S['mean'])} % of the top 100 belonged to the class on average. How can a bank use this?",
      ["Search first, label the hits, then train a probe: a cheap way to build a labelled set", "Replace all human labelling forever", "Measure the size of the images", "Compute the temperature"],
      "Search concentrates relevant images; a person confirms them faster than labelling random images.",
      f"Precision varies by class (from {en(S['per_class'][S['worst']])} % to {en(S['per_class'][S['best']])} %), so a human check remains necessary."),
  ("Căutarea imaginilor după text", f"Ordonînd imaginile de test după similaritatea cu promptul fiecărei clase, în medie {ro(S['mean'])} % dintre primele 100 aparțineau clasei. Cum poate folosi o bancă acest lucru?",
   ["Întîi caută, apoi etichetează rezultatele, apoi antrenează un clasificator: o cale ieftină spre un set etichetat", "Înlocuiește pentru totdeauna etichetarea umană", "Măsoară dimensiunea imaginilor", "Calculează temperatura"],
   "Căutarea concentrează imaginile relevante; un om le confirmă mai repede decît dacă ar eticheta imagini alese aleator.",
   f"Precizia variază de la o clasă la alta (de la {ro(S['per_class'][S['worst']])} % la {ro(S['per_class'][S['best']])} %), deci verificarea umană rămîne necesară."))
q(2, ("Attention rollout", "What does attention rollout compute?",
      ["The gradient of the loss with respect to the pixels", "A random mask", "The product of the head-averaged attention matrices of all layers, with the identity added for the residual connection", "The output of the last layer only"],
      "Rollout propagates attention through the layers, giving how much each patch contributes to the class token.",
      "Rollout is attention-based, not gradient-based, and uses every layer, not only the last."),
  ("Propagarea atenției", "Ce calculează propagarea atenției (attention rollout)?",
   ["Gradientul pierderii în raport cu pixelii", "O mască aleatoare", "Produsul matricelor de atenție mediate pe capete din toate straturile, cu identitatea adăugată pentru conexiunea reziduală", "Doar ieșirea ultimului strat"],
   "Propagarea transmite atenția prin straturi și arată cît contribuie fiecare petic la tokenul de clasă.",
   "Metoda se bazează pe atenție, nu pe gradienți, și folosește toate straturile, nu doar pe ultimul."))
q(1, ("VLM architecture", "How does a vision-language model such as Qwen2-VL let a language model 'see'?",
      ["It converts the image into a text description first", "A vision encoder turns patches into vectors that a projector maps into the word-embedding space; the LLM reads them as tokens", "It trains the LLM on pixels directly", "It uses OCR only"],
      "Image tokens and text tokens enter the same transformer, so the model can answer questions about the image.",
      "No intermediate caption and no pixel-level training of the LLM are needed; OCR is one ability, not the mechanism."),
  ("Arhitectura VLM", "Cum îi permite un model vizual-lingvistic precum Qwen2-VL unui model de limbaj să „vadă”?",
   ["Transformă întîi imaginea într-o descriere text", "Un codificator vizual transformă peticele în vectori pe care un proiector îi aduce în spațiul embedding-urilor de cuvinte; LLM îi citește ca tokeni", "Antrenează LLM direct pe pixeli", "Folosește doar OCR"],
   "Tokenii de imagine și cei de text intră în același transformer, astfel încît modelul poate răspunde la întrebări despre imagine.",
   "Nu este nevoie de o descriere intermediară sau de antrenarea LLM pe pixeli; OCR este o abilitate, nu mecanismul."))
q(3, ("Rolling-quantile label", "How were the 'stress' labels of the S&P 500 windows defined?",
      ["By an analyst looking at the charts", "By the VLM itself", "By the sign of the next month's return", "20-day volatility above its rolling 90th percentile of the previous three years, using past data only"],
      "A rule based on past quantiles is objective, reproducible and free of look-ahead, as in the group's framework.",
      "Human labelling is subjective and future returns would leak information; the label uses only data available at the window's end."),
  ("Eticheta cu cuantile glisante", "Cum au fost definite etichetele de „criză” pentru ferestrele S&P 500?",
   ["De un analist care s-a uitat la grafice", "De modelul VLM însuși", "După semnul randamentului din luna următoare", "Volatilitatea pe 20 de zile peste percentila 90 glisantă din ultimii trei ani, folosind doar date din trecut"],
   "O regulă bazată pe cuantile din trecut este obiectivă, reproductibilă și nu folosește viitorul, ca în cadrul grupului.",
   "Etichetarea umană este subiectivă, iar randamentele viitoare ar introduce informație din viitor; eticheta folosește doar date disponibile la sfîrșitul ferestrei."))
q(0, ("Pictures or numbers", f"On the test years the AUC was {en(RG['numeric']['auc'], 3)} for logistic regression on numbers, {en(RG['clip_zs']['auc'], 3)} for CLIP zero-shot and {en(RG['vlm_zs']['auc'], 3)} for Qwen2-VL zero-shot. What is the lesson?",
      ["When the numbers behind a chart exist, they are the better input; images help when numbers are missing", "Images always beat numbers", "VLMs are better than CLIP for charts", "AUC cannot compare these models"],
      "The chart is drawn from the same numbers with loss of resolution; the numeric model reads volatility directly.",
      "All four scores are on the same windows and labels, so the AUCs are comparable; the numbers came first here."),
  ("Imagini sau cifre", f"Pe anii de test, AUC a fost {ro(RG['numeric']['auc'], 3)} pentru regresia logistică pe cifre, {ro(RG['clip_zs']['auc'], 3)} pentru CLIP zero-shot și {ro(RG['vlm_zs']['auc'], 3)} pentru Qwen2-VL zero-shot. Care este lecția?",
   ["Cînd cifrele din spatele unui grafic există, ele sînt intrarea mai bună; imaginile ajută cînd cifrele lipsesc", "Imaginile bat întotdeauna cifrele", "Modelele VLM sînt mai bune decît CLIP pentru grafice", "AUC nu poate compara aceste modele"],
   "Graficul este desenat din aceleași cifre, cu pierdere de rezoluție; modelul numeric citește direct volatilitatea.",
   "Toate cele patru scoruri sînt pe aceleași ferestre și etichete, deci AUC-urile sînt comparabile; aici cifrele au fost pe primul loc."))
q(2, ("Average precision", f"The stress share in the test windows is {en(N['charts_task']['base_test'])} %. Why does the lecture report average precision next to AUC?",
      ["Average precision is always higher", "AUC is not defined for images", "For a rare event, average precision shows how clean the top-ranked alarms are; a random score has average precision equal to the event share", "To have more numbers"],
      f"Average precision was {en(RG['numeric']['ap'], 3)} for the numbers and {en(RG['vlm_zs']['ap'], 3)} for the VLM, against {en(N['charts_task']['base_test'] / 100, 3)} for a random score.",
      "AUC can look good when negatives dominate; average precision focuses on the positive class, which is what an alarm system needs."),
  ("Precizia medie", f"Ponderea crizelor în ferestrele de test este {ro(N['charts_task']['base_test'])} %. De ce raportează cursul precizia medie lîngă AUC?",
   ["Precizia medie este întotdeauna mai mare", "AUC nu este definit pentru imagini", "Pentru un eveniment rar, precizia medie arată cît de curate sînt alarmele de pe primele locuri; un scor aleator are o precizie medie egală cu ponderea evenimentului", "Pentru a avea mai multe cifre"],
   f"Precizia medie a fost {ro(RG['numeric']['ap'], 3)} pentru cifre și {ro(RG['vlm_zs']['ap'], 3)} pentru VLM, față de {ro(N['charts_task']['base_test'] / 100, 3)} pentru un scor aleator.",
   "AUC poate arăta bine cînd exemplele negative domină; precizia medie se concentrează pe clasa pozitivă, de care are nevoie un sistem de alarmă."))
q(1, ("Threshold bias", f"The VLM answered the trend question correctly in {en(VL['trend_acc'])} % of charts (majority answer: {en(VL['trend_base'])} %), but its AUC was {en(VL['trend_auc'], 3)}. What explains this?",
      ["The model cannot see the charts", "Its probabilities rank the charts well but the 0.5 threshold is biased towards one answer; recalibrating the threshold fixes most of it", "The labels are wrong", "The images are too small"],
      "A high AUC means 'higher' charts get higher scores than 'lower' ones; only the cut-off is misplaced.",
      "Good ranking with poor accuracy is a calibration problem, solved on validation charts, not a vision failure."),
  ("Pragul deplasat", f"Modelul VLM a răspuns corect la întrebarea despre tendință pentru {ro(VL['trend_acc'])} % din grafice (răspunsul majoritar: {ro(VL['trend_base'])} %), dar AUC a fost {ro(VL['trend_auc'], 3)}. Ce explică aceasta?",
   ["Modelul nu vede graficele", "Probabilitățile lui ordonează bine graficele, dar pragul de 0,5 este deplasat spre un răspuns; recalibrarea pragului rezolvă cea mai mare parte", "Etichetele sînt greșite", "Imaginile sînt prea mici"],
   "Un AUC mare înseamnă că graficele „mai sus” primesc scoruri mai mari decît cele „mai jos”; doar pragul este plasat greșit.",
   "O ordonare bună cu acuratețe slabă este o problemă de calibrare, rezolvată pe grafice de validare, nu un eșec al vederii."))
q(0, ("Reading tables", f"Reading the 10-year yield from {TB['n']} rendered ECB tables, the VLM gave the exact value in {en(TB['exact'])} % of cases. What should a bank do with documents of known layout?",
      ["Extract numbers with a rule-based parser and use the VLM only where the layout varies, with checks and human review", "Trust the VLM for all numbers", "Never digitise documents", "Use a larger temperature"],
      "Silent, plausible reading errors are the main risk; deterministic extraction plus validation checks is safer for fixed templates.",
      f"A mean absolute error of {en(TB['mae'], 3)} percentage points on a yield is not acceptable for reporting; controls are needed."),
  ("Citirea tabelelor", f"Citind randamentul la 10 ani din {TB['n']} de tabele BCE desenate, modelul VLM a dat valoarea exactă în {ro(TB['exact'])} % din cazuri. Ce trebuie să facă o bancă cu documentele cu structură cunoscută?",
   ["Să extragă cifrele cu un program bazat pe reguli și să folosească VLM doar unde structura variază, cu verificări și control uman", "Să aibă încredere în VLM pentru toate cifrele", "Să nu digitalizeze niciodată documentele", "Să folosească o temperatură mai mare"],
   "Erorile de citire tăcute și plauzibile sînt riscul principal; extragerea deterministă plus verificările sînt mai sigure pentru șabloane fixe.",
   f"O eroare absolută medie de {ro(TB['mae'], 3)} puncte procentuale la un randament nu este acceptabilă pentru raportare; sînt necesare controale."))
q(3, ("Cost", f"CLIP ViT-B/32 embedded an image in {en(N['clip']['B32']['ms_per_image'], 2)} ms and the VLM needed about {en(VL['ms_per_image'], 0)} ms per image and question. What follows for a bank screening a very large archive of images?",
      ["The VLM is always the right tool", "Speed does not matter", "Both are equally expensive", "Use embeddings for screening and the VLM only on the few images that need a question answered"],
      "A cascade keeps the cheap model on the bulk and the expensive model on the hard cases.",
      "The cost ratio is large, so the expensive model should see only the cases where it adds information."),
  ("Costul", f"CLIP ViT-B/32 a calculat embedding-ul unei imagini în {ro(N['clip']['B32']['ms_per_image'], 2)} ms, iar VLM a avut nevoie de circa {ro(VL['ms_per_image'], 0)} ms pentru o imagine și o întrebare. Ce rezultă pentru o bancă ce analizează o arhivă foarte mare de imagini?",
   ["VLM este întotdeauna instrumentul potrivit", "Viteza nu contează", "Ambele sînt la fel de costisitoare", "Embedding-urile pentru triere și VLM doar pentru puținele imagini care au nevoie de un răspuns la o întrebare"],
   "O cascadă păstrează modelul ieftin pentru volumul mare și modelul scump pentru cazurile dificile.",
   "Raportul costurilor este mare, deci modelul scump trebuie să vadă doar cazurile în care aduce informație."))
q(2, ("Choosing a model", "A team has 16 labelled images per class and needs a land-use classifier next week. What is the sensible first model?",
      ["A ViT trained from scratch", "A CNN trained from scratch on these images only", "A logistic regression on frozen CLIP embeddings, compared with zero-shot CLIP", "No model until 100,000 labels exist"],
      f"With 16 images per class the CLIP probe reached {en(F['probe']['16']['mean'])} % against {en(F['cnn']['16']['mean'])} % for the CNN.",
      "Training from scratch needs many more labels; a foundation model with a small head is the right start."),
  ("Alegerea modelului", "O echipă are 16 imagini etichetate pe clasă și are nevoie de un clasificator al utilizării terenului săptămîna viitoare. Care este primul model rezonabil?",
   ["Un ViT antrenat de la zero", "O CNN antrenată de la zero doar pe aceste imagini", "O regresie logistică pe embedding-uri CLIP fixe, comparată cu CLIP zero-shot", "Niciun model pînă nu există 100.000 de etichete"],
   f"Cu 16 imagini pe clasă, clasificatorul pe CLIP a ajuns la {ro(F['probe']['16']['mean'])} %, față de {ro(F['cnn']['16']['mean'])} % pentru CNN.",
   "Antrenarea de la zero are nevoie de mult mai multe etichete; un model fundație cu un mic clasificator este începutul potrivit."))
q(1, ("Benchmark first", "Which principle held in every lecture of the course?",
      ["Deep networks always beat linear models", "Every network is reported next to a simple benchmark on the same data and test period", "Only accuracy is reported", "The test set can be reused for tuning"],
      "A network is useful only if it beats the benchmark that a practitioner would otherwise use, on the same test data.",
      "Networks often do not beat good benchmarks, metrics depend on the problem, and the test set is used once."),
  ("Întîi reperul", "Ce principiu a fost valabil în fiecare curs?",
   ["Rețelele profunde bat întotdeauna modelele liniare", "Fiecare rețea este raportată lîngă un reper simplu, pe aceleași date și pe aceeași perioadă de test", "Se raportează doar acuratețea", "Setul de test poate fi refolosit pentru ajustare"],
   "O rețea este utilă doar dacă depășește reperul pe care un practician l-ar folosi altfel, pe aceleași date de test.",
   "Rețelele adesea nu depășesc reperele bune, metricile depind de problemă, iar setul de test se folosește o singură dată."))
q(3, ("Leakage", "Which of these is data leakage?",
      ["Fixing the seed", "Using a validation set for early stopping", "Reporting bootstrap intervals", "Choosing the prompt wording on the test set"],
      "Any decision taken on the test set makes the reported test score optimistic.",
      "Seeds, validation-based early stopping and intervals are good practice; selection on the test set is leakage."),
  ("Scurgerea de informație", "Care dintre acestea este o scurgere de informație?",
   ["Fixarea valorii generatorului aleator", "Folosirea unui set de validare pentru oprirea timpurie", "Raportarea intervalelor bootstrap", "Alegerea formulării promptului pe setul de test"],
   "Orice decizie luată pe setul de test face ca scorul raportat pe test să fie optimist.",
   "Valorile fixe ale generatorului aleator, oprirea timpurie pe validare și intervalele sînt bune practici; selecția pe setul de test este scurgere de informație."))
q(0, ("Foundation-model caveat", "What is the common caveat of foundation models for text, time series and images?",
      ["Their pre-training data may include your test data, so evaluate on data they cannot have seen", "They cannot be used without fine-tuning", "They only work in English", "They need labelled data"],
      "Web-scale pre-training may contain public benchmarks; post-cut-off or private test data give honest estimates.",
      "Foundation models work zero-shot, in many languages, without labels; contamination of evaluations is the shared risk."),
  ("Avertismentul modelelor fundație", "Care este avertismentul comun al modelelor fundație pentru text, serii de timp și imagini?",
   ["Datele lor de preantrenare pot include datele voastre de test, deci evaluați pe date pe care nu le-au putut vedea", "Nu pot fi folosite fără ajustare fină", "Funcționează doar în engleză", "Au nevoie de date etichetate"],
   "Preantrenarea la scara internetului poate conține seturi publice de evaluare; datele de test ulterioare limitei sau private dau estimări oneste.",
   "Modelele fundație funcționează zero-shot, în multe limbi, fără etichete; contaminarea evaluărilor este riscul comun."))
q(2, ("Regulation", "Which use of AI does the European Union Artificial Intelligence Act list as high-risk, relevant to banks?",
      ["Spam filtering", "Weather forecasting", "Credit scoring of natural persons", "Image compression"],
      "Creditworthiness assessment of individuals is a high-risk use: documentation, data governance and human oversight are required.",
      "The high-risk list targets decisions with large effects on people, such as access to credit."),
  ("Reglementarea", "Ce utilizare a inteligenței artificiale este considerată cu risc ridicat de Regulamentul Uniunii Europene privind inteligența artificială, relevantă pentru bănci?",
   ["Filtrarea mesajelor nedorite", "Prognoza meteo", "Evaluarea bonității persoanelor fizice (scoringul de credit)", "Comprimarea imaginilor"],
   "Evaluarea bonității persoanelor fizice este o utilizare cu risc ridicat: sînt cerute documentarea, guvernanța datelor și supravegherea umană.",
   "Lista utilizărilor cu risc ridicat vizează deciziile cu efecte mari asupra oamenilor, precum accesul la credit."))
q(1, ("Model card", "What belongs in a model card according to the lecture?",
      ["Only the accuracy", "Data, period, metric, benchmark, seeds and known failure cases", "The source code only", "The names of the developers only"],
      "A model card lets a reviewer reproduce and judge the model; known failures (such as sea read as highway) are part of it.",
      "A single score hides the conditions under which it holds; the card documents them."),
  ("Fișa modelului", "Ce trebuie să conțină o fișă a modelului, potrivit cursului?",
   ["Doar acuratețea", "Datele, perioada, metrica, reperul, rulările și cazurile cunoscute de eșec", "Doar codul sursă", "Doar numele dezvoltatorilor"],
   "O fișă a modelului permite unui evaluator să reproducă și să judece modelul; eșecurile cunoscute (de exemplu marea citită drept autostradă) fac parte din ea.",
   "Un singur scor ascunde condițiile în care este valabil; fișa le documentează."))
q(3, ("ESG use", "How can zero-shot image models support ESG reporting of a portfolio?",
      ["By predicting share prices", "By replacing the annual report", "By computing interest rates", "By measuring forest cover or industrial land around assets from satellite images without labelled data"],
      "Zero-shot labels and text search over satellite images give environmental indicators where no labels exist.",
      "The models classify land use; they do not forecast prices or rates."),
  ("Utilizarea pentru ESG", "Cum pot sprijini modelele de imagini zero-shot raportarea ESG a unui portofoliu?",
   ["Prin prognoza prețurilor acțiunilor", "Prin înlocuirea raportului anual", "Prin calculul ratelor dobînzii", "Prin măsurarea suprafeței împădurite sau a terenului industrial din jurul activelor, din imagini satelitare, fără date etichetate"],
   "Etichetele zero-shot și căutarea după text în imagini satelitare oferă indicatori de mediu acolo unde nu există etichete.",
   "Modelele clasifică utilizarea terenului; nu prognozează prețuri sau dobînzi."))
q(0, ("Group research", "What did the multimodal framework of Jheng, Pele, Tak and Găman (2026) do?",
      ["It projected market indicators onto multi-panel images and used a VLM zero-shot to classify regimes and explain its choice", "It trained a CNN on satellite images", "It forecast electricity prices", "It replaced the S&P 500 with Bitcoin"],
      "The paper reframes regime classification as visual recognition, with rolling-quantile labels and natural-language rationales, validated out of sample on the S&P 500.",
      "The study is about market regimes read from chart images by a vision-language model."),
  ("Cercetarea grupului", "Ce a făcut cadrul multimodal al lui Jheng, Pele, Tak și Găman (2026)?",
   ["A proiectat indicatori de piață pe imagini cu mai multe panouri și a folosit un VLM zero-shot pentru a clasifica regimurile și a-și explica alegerea", "A antrenat o CNN pe imagini satelitare", "A prognozat prețurile energiei electrice", "A înlocuit S&P 500 cu Bitcoin"],
   "Lucrarea reformulează clasificarea regimurilor ca recunoaștere vizuală, cu etichete bazate pe cuantile glisante și justificări în limbaj natural, validată pe date noi pentru S&P 500.",
   "Studiul se referă la regimuri de piață citite din imagini ale graficelor de un model vizual-lingvistic."))
q(2, ("Probe or full training", f"On all {N['data']['n_train']:,} training images the CNN reached {en(N['cnn_full']['mean'])} % and the CLIP probe {en(N['probe_full'])} %. Which statement is correct?",
      ["Foundation models are always best", "The CNN cannot be trained on so many images", "With abundant in-domain labels, a small task-specific network can beat a general foundation model", "Both are below zero-shot"],
      "Specialised training on many labelled images captures details of Sentinel-2 imagery that web-trained features miss.",
      f"Both are far above zero-shot ({en(Z[N['zs_best']])} %); with many labels, the trained CNN is ahead here."),
  ("Clasificator sau antrenare completă", f"Pe toate cele {N['data']['n_train']:,} de imagini de antrenare, CNN a ajuns la {ro(N['cnn_full']['mean'])} %, iar clasificatorul pe CLIP la {ro(N['probe_full'])} %. Ce afirmație este corectă?".replace(",", ".", 1),
   ["Modelele fundație sînt întotdeauna cele mai bune", "CNN nu poate fi antrenată pe atîtea imagini", "Cu multe etichete din domeniu, o rețea mică specifică sarcinii poate depăși un model fundație general", "Ambele sînt sub zero-shot"],
   "Antrenarea specializată pe multe imagini etichetate surprinde detalii ale imaginilor Sentinel-2 pe care trăsăturile învățate de pe internet le ratează.",
   f"Ambele sînt mult peste zero-shot ({ro(Z[N['zs_best']])} %); cu multe etichete, CNN antrenată este înainte aici."))
q(1, ("Document pipeline", "In the bank document pipeline of the lecture, what happens when a total does not add up?",
      ["The VLM answer is accepted anyway", "The case is sent to a person for review", "The document is deleted", "The temperature is raised"],
      "Validation checks catch silent extraction errors; failed checks go to human review.",
      "Checks exist precisely to stop plausible but wrong numbers from entering reports."),
  ("Fluxul documentelor", "În fluxul documentelor dintr-o bancă descris în curs, ce se întîmplă cînd un total nu se adună?",
   ["Răspunsul VLM este acceptat oricum", "Cazul este trimis spre verificare unei persoane", "Documentul este șters", "Se crește temperatura"],
   "Verificările prind erorile tăcute de extragere; verificările eșuate merg la control uman.",
   "Verificările există tocmai pentru a opri intrarea în rapoarte a cifrelor plauzibile, dar greșite."))

q(3, ("Temperature in CLIP", "What is the role of the learned temperature τ in CLIP's similarity s_ij = u_iᵀv_j / τ?",
      ["It sets the image resolution", "It chooses the number of classes", "It is the learning rate", "It scales the cosine similarities before the softmax, controlling how peaked the matching distribution is"],
      "Cosine similarities lie in [−1, 1]; dividing by a small τ sharpens the softmax over captions so that training signals are strong.",
      "τ acts like the temperature of a language model: it rescales scores before the softmax; it is not a resolution or learning rate."),
  ("Temperatura în CLIP", "Care este rolul temperaturii învățate τ în similaritatea CLIP s_ij = u_iᵀv_j / τ?",
   ["Stabilește rezoluția imaginii", "Alege numărul de clase", "Este rata de învățare", "Scalează similaritățile cosinus înainte de softmax și controlează cît de concentrată este distribuția potrivirilor"],
   "Similaritățile cosinus sînt în [−1, 1]; împărțirea la un τ mic accentuează softmax pe descrieri, astfel încît semnalul de antrenare este puternic.",
   "τ funcționează ca temperatura unui model de limbaj: rescalează scorurile înainte de softmax; nu este o rezoluție sau o rată de învățare."))
q(0, ("Patch size and cost", f"CLIP ViT-B/32 needed {en(N['clip']['B32']['ms_per_image'], 2)} ms per image and ViT-B/16 {en(N['clip']['B16']['ms_per_image'], 2)} ms. Why is the B/16 model slower although both have a similar number of parameters?",
      ["Patches of 16 pixels give four times more tokens than patches of 32, and attention cost grows with the number of tokens", "B/16 has more layers", "B/16 uses colour and B/32 does not", "B/16 is trained on more data"],
      "A 224-pixel image gives 7 × 7 = 49 patches of 32 pixels but 14 × 14 = 196 patches of 16 pixels.",
      "The two models have the same depth and inputs; the difference is the number of tokens per image."),
  ("Mărimea peticului și costul", f"CLIP ViT-B/32 a avut nevoie de {ro(N['clip']['B32']['ms_per_image'], 2)} ms pe imagine, iar ViT-B/16 de {ro(N['clip']['B16']['ms_per_image'], 2)} ms. De ce este modelul B/16 mai lent, deși ambele au un număr similar de parametri?",
   ["Peticele de 16 pixeli dau de patru ori mai mulți tokeni decît cele de 32, iar costul atenției crește cu numărul de tokeni", "B/16 are mai multe straturi", "B/16 folosește culoarea, iar B/32 nu", "B/16 este antrenat pe mai multe date"],
   "O imagine de 224 de pixeli dă 7 × 7 = 49 de petice de 32 de pixeli, dar 14 × 14 = 196 de petice de 16 pixeli.",
   "Cele două modele au aceeași adîncime și aceleași intrări; diferența este numărul de tokeni pe imagine."))
assert len(Q) >= 30, len(Q)
js = ("// ============================================================\n"
      "// Quiz bank for lecture id 'l14': Multimodal and vision-language models; synthesis of the course (EN + RO)\n"
      "// GENERATED by lectures/L14/make_quiz_L14.py from L14_numbers.json; do not edit by hand.\n"
      "// correct = index (0-3) of the right option in the original order; incorrectExplanation never names a letter.\n"
      "// ============================================================\n"
      "window.NN_DATA.quizzes['l14'] = " + json.dumps({"draw": 20, "questions": Q}, ensure_ascii=False, indent=1) + ";\n")
with open(OUT, "w", encoding="utf-8") as f:
    f.write(js)
print(f"{len(Q)} questions written to assets/quizzes/l14.js")
