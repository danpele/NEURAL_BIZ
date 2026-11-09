"""
make_quiz_L9.py -- writes assets/quizzes/l9.js, the quiz bank of Lecture 9 (EN + RO), with every number read from
L9_numbers.json. Format of lectures/README.md: correct = index 0-3 of the right option; the explanation of a wrong
answer never names a letter (the engine shuffles the options).
Run:  python3 lectures/L9/make_quiz_L9.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = json.load(open(os.path.join(HERE, "L9_numbers.json")))


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


q(0, ("Why a graph", "Why can a single transaction look ordinary while still being part of money laundering?",
      ["Because laundering shows in the chain of payments around it, not in the row of the transaction", "Because transaction amounts are always rounded",
       "Because illicit transactions have no features", "Because Bitcoin transactions are anonymous only after 49 time steps"],
      "Layering splits and merges funds across many transactions; the pattern lives in the links, which is what a graph model reads.",
      "The features of one transaction may be normal; what betrays laundering is how it connects to other transactions, so the graph matters."),
  ("De ce un graf", "De ce poate o singură tranzacție să pară obișnuită, deși face parte dintr-o operațiune de spălare a banilor?",
   ["Pentru că spălarea se vede în lanțul de plăți din jurul ei, nu în rîndul tranzacției", "Pentru că sumele tranzacțiilor sînt mereu rotunjite",
    "Pentru că tranzacțiile ilicite nu au variabile", "Pentru că tranzacțiile Bitcoin sînt anonime doar după 49 de pași de timp"],
   "Stratificarea împarte și reunește fondurile prin multe tranzacții; tiparul stă în legături, adică exact ce citește un model pe graf.",
   "Variabilele unei tranzacții pot fi normale; ceea ce trădează spălarea este felul în care se leagă de alte tranzacții, deci graful contează."))
q(2, ("Size of the Elliptic graph", "How many transactions (nodes) and payment flows (edges) does the Elliptic graph have?",
      [E("data.n_nodes", ",") + " nodes and " + E("data.n_nodes", ",") + " edges", E("data.n_illicit", ",") + " nodes and " + E("data.n_licit", ",") + " edges",
       E("data.n_nodes", ",") + " nodes and " + E("data.n_edges", ",") + " edges", E("data.n_edges", ",") + " nodes and " + E("data.n_nodes", ",") + " edges"],
      "The graph has " + E("data.n_nodes", ",") + " Bitcoin transactions and " + E("data.n_edges", ",") + " payment flows, with " + E("data.n_features") + " features per transaction.",
      "There are " + E("data.n_nodes", ",") + " nodes and " + E("data.n_edges", ",") + " edges; " + E("data.n_illicit", ",") + " and " + E("data.n_licit", ",") + " are the numbers of illicit and licit labels."),
  ("Dimensiunea grafului Elliptic", "Cîte tranzacții (noduri) și cîte fluxuri de plată (muchii) are graful Elliptic?",
   [R("data.n_nodes", ",") + " de noduri și " + R("data.n_nodes", ",") + " de muchii", R("data.n_illicit", ",") + " de noduri și " + R("data.n_licit", ",") + " de muchii",
    R("data.n_nodes", ",") + " de noduri și " + R("data.n_edges", ",") + " de muchii", R("data.n_edges", ",") + " de noduri și " + R("data.n_nodes", ",") + " de muchii"],
   "Graful are " + R("data.n_nodes", ",") + " de tranzacții Bitcoin și " + R("data.n_edges", ",") + " de fluxuri de plată, cu " + R("data.n_features") + " de variabile pe tranzacție.",
   "Sînt " + R("data.n_nodes", ",") + " de noduri și " + R("data.n_edges", ",") + " de muchii; " + R("data.n_illicit", ",") + " și " + R("data.n_licit", ",") + " sînt numerele de etichete ilicite și licite."))
q(1, ("Class imbalance", "Only " + E("data.pct_illicit_lab", ".2f") + " % of the labelled transactions are illicit. Which metric is least useful on its own?",
      ["PR-AUC", "Accuracy", "Recall of the illicit class", "Precision of the alerts"],
      "Labelling everything licit is right most of the time and finds no criminal, so accuracy alone is misleading on such data.",
      "Precision, recall and PR-AUC focus on the rare illicit class; accuracy is dominated by the many licit transactions."),
  ("Dezechilibrul claselor", "Doar " + R("data.pct_illicit_lab", ".2f") + " % dintre tranzacțiile etichetate sînt ilicite. Care măsură este cea mai puțin utilă luată singură?",
   ["PR-AUC", "Acuratețea", "Recall-ul clasei ilicite", "Precizia alertelor"],
   "Dacă etichetăm totul ca licit, avem dreptate de cele mai multe ori și nu găsim niciun infractor, deci acuratețea singură induce în eroare pe astfel de date.",
   "Precizia, recall-ul și PR-AUC se concentrează pe clasa ilicită rară; acuratețea este dominată de multele tranzacții licite."))
q(3, ("Temporal split", "Why are the models trained on time steps 1–34 and tested on 35–49 instead of a random split of nodes?",
      ["Because time steps 35–49 have no labels", "Because random splits are impossible on graphs", "To have more illicit transactions in training",
       "Because the model will score future transactions, and illicit behaviour changes over time"],
      "A temporal split mimics deployment and exposes drift; a random split would mix future patterns into training.",
      "Both periods are labelled and random node splits are technically possible, but they would leak future behaviour; the honest test comes after the training period."),
  ("Împărțirea în timp", "De ce se antrenează modelele pe pașii de timp 1–34 și se testează pe 35–49, în loc de o împărțire aleatoare a nodurilor?",
   ["Pentru că pașii 35–49 nu au etichete", "Pentru că împărțirea aleatoare este imposibilă pe grafuri", "Ca să fie mai multe tranzacții ilicite la antrenare",
    "Pentru că modelul va da scoruri tranzacțiilor viitoare, iar comportamentul ilicit se schimbă în timp"],
   "O împărțire în timp imită utilizarea reală și scoate la iveală schimbările; o împărțire aleatoare ar amesteca tipare viitoare în antrenare.",
   "Ambele perioade au etichete, iar împărțirea aleatoare a nodurilor este posibilă tehnic, dar ar scurge informație din viitor; testul corect vine după perioada de antrenare."))
q(0, ("Homophily", "Among the labelled neighbours of an illicit transaction, " + E("homophily.illicit_nb_of_illicit_pct", ".1f") + " % are illicit. For a licit transaction the share is:",
      [E("homophily.illicit_nb_of_licit_pct", ".1f") + " %", E("homophily.illicit_nb_of_illicit_pct", ".1f") + " %", "50 %", "100 %"],
      "Licit transactions have only " + E("homophily.illicit_nb_of_licit_pct", ".1f") + " % illicit neighbours: illicit transactions cluster together, which is the homophily a GNN exploits.",
      "The gap between " + E("homophily.illicit_nb_of_illicit_pct", ".1f") + " % and " + E("homophily.illicit_nb_of_licit_pct", ".1f") + " % shows that the neighbours carry information about the label."),
  ("Homofilia", "Dintre vecinii etichetați ai unei tranzacții ilicite, " + R("homophily.illicit_nb_of_illicit_pct", ".1f") + " % sînt ilicite. Pentru o tranzacție licită ponderea este:",
   [R("homophily.illicit_nb_of_licit_pct", ".1f") + " %", R("homophily.illicit_nb_of_illicit_pct", ".1f") + " %", "50 %", "100 %"],
   "Tranzacțiile licite au doar " + R("homophily.illicit_nb_of_licit_pct", ".1f") + " % vecini ilicite: tranzacțiile ilicite se grupează, iar aceasta este homofilia exploatată de o rețea GNN.",
   "Diferența dintre " + R("homophily.illicit_nb_of_illicit_pct", ".1f") + " % și " + R("homophily.illicit_nb_of_licit_pct", ".1f") + " % arată că vecinii poartă informație despre etichetă."))
q(2, ("Normalised adjacency", "Which matrix does a GCN layer multiply the node representations with?",
      ["$A$, the raw adjacency", "$D-A$, the Laplacian", "$\\tilde D^{-1/2}(A+I)\\tilde D^{-1/2}$", "$A^\\top A$"],
      "Self-loops keep each node's own features and the symmetric degree normalisation stops hubs from dominating: entries $1/\\sqrt{\\tilde d_u\\tilde d_v}$.",
      "The GCN uses the normalised adjacency with self-loops; the raw adjacency would let hubs explode the scale and would drop the node's own features."),
  ("Adiacența normalizată", "Cu ce matrice înmulțește un strat GCN reprezentările nodurilor?",
   ["$A$, adiacența brută", "$D-A$, laplacianul", "$\\tilde D^{-1/2}(A+I)\\tilde D^{-1/2}$", "$A^\\top A$"],
   "Buclele păstrează variabilele proprii ale nodului, iar normalizarea simetrică după grad împiedică nodurile centrale să domine: elementele sînt $1/\\sqrt{\\tilde d_u\\tilde d_v}$.",
   "GCN folosește adiacența normalizată cu bucle; adiacența brută ar lăsa nodurile centrale să explodeze scala și ar pierde variabilele proprii ale nodului."))
q(1, ("Permutation equivariance", "We renumber the transactions of the graph. What must happen to the GNN's node scores?",
      ["They change, because the model has learned positions", "They are permuted in the same way and are otherwise unchanged", "They all become equal", "They must be recomputed after retraining"],
      "A GNN layer is permutation-equivariant: $f(P\\mathbf X,PAP^\\top)=Pf(\\mathbf X,A)$, because aggregation over neighbours does not depend on their order.",
      "Node numbers are names, not positions; message passing aggregates over sets of neighbours, so relabelling only reorders the outputs."),
  ("Echivarianța la permutări", "Renumerotăm tranzacțiile grafului. Ce trebuie să se întîmple cu scorurile rețelei GNN?",
   ["Se schimbă, pentru că modelul a învățat poziții", "Se permută la fel și altfel rămîn neschimbate", "Devin toate egale", "Trebuie recalculate după reantrenare"],
   "Un strat GNN este echivariant la permutări: $f(P\\mathbf X,PAP^\\top)=Pf(\\mathbf X,A)$, pentru că agregarea pe vecini nu depinde de ordinea lor.",
   "Numerele nodurilor sînt nume, nu poziții; transmiterea mesajelor agregă pe mulțimi de vecini, deci renumerotarea doar reordonează ieșirile."))
q(3, ("Aggregators", "Which aggregator can tell apart a node with two identical neighbours $\\{a,a\\}$ from a node with one neighbour $\\{a\\}$?",
      ["Mean", "Maximum", "Minimum", "Sum"],
      "The sum keeps the count of messages; with an injective update it makes the network as discriminative as the Weisfeiler–Leman test (GIN).",
      "Mean, maximum and minimum of $\\{a,a\\}$ and $\\{a\\}$ are all $a$; only the sum ($2a$ against $a$) separates them."),
  ("Agregatoarele", "Ce agregator poate deosebi un nod cu doi vecini identici $\\{a,a\\}$ de un nod cu un singur vecin $\\{a\\}$?",
   ["Media", "Maximul", "Minimul", "Suma"],
   "Suma păstrează numărul de mesaje; cu o actualizare injectivă face rețeaua la fel de discriminativă ca testul Weisfeiler–Leman (GIN).",
   "Media, maximul și minimul lui $\\{a,a\\}$ și $\\{a\\}$ sînt toate $a$; doar suma ($2a$ față de $a$) le separă."))
q(0, ("Receptive field", "How many nodes can influence a test transaction after two message-passing layers (median over 2,000 test nodes)?",
      [E("receptive.k2_median"), E("receptive.k4_median"), E("receptive.k4_p90", ","), "All " + E("data.n_nodes", ",") + " nodes"],
      "Two layers reach the 2-hop neighbourhood: a median of " + E("receptive.k2_median") + " nodes; four layers reach " + E("receptive.k4_median") + ".",
      "The receptive field of an $L$-layer GNN is the $L$-hop neighbourhood; on this sparse graph two hops cover a median of " + E("receptive.k2_median") + " transactions."),
  ("Cîmpul receptiv", "Cîte noduri pot influența o tranzacție de test după două straturi de transmitere a mesajelor (mediana pe 2 000 de noduri de test)?",
   [R("receptive.k2_median"), R("receptive.k4_median"), R("receptive.k4_p90", ","), "Toate cele " + R("data.n_nodes", ",") + " de noduri"],
   "Două straturi ajung la vecinătatea de 2 salturi: o mediană de " + R("receptive.k2_median") + " noduri; patru straturi ajung la " + R("receptive.k4_median") + ".",
   "Cîmpul receptiv al unei rețele GNN cu $L$ straturi este vecinătatea de $L$ salturi; pe acest graf rar, două salturi acoperă o mediană de " + R("receptive.k2_median") + " tranzacții."))
q(2, ("Over-smoothing", "What happened when the GCN went from 2 to 6 layers?",
      ["PR-AUC rose, because each node saw more of the graph", "Nothing changed", "PR-AUC fell from " + E("depth.ap_2", ".3f") + " to " + E("depth.ap_6", ".3f") + " and node representations became more similar",
       "Training became impossible"],
      "Repeated degree-weighted averaging pushes representations towards a common vector (cosine similarity " + E("depth.cos_6", ".2f") + " with 6 layers), so classes blur.",
      "Deeper is not better for GNNs: the mean cosine similarity rose to " + E("depth.cos_6", ".2f") + " and PR-AUC fell to " + E("depth.ap_6", ".3f") + "."),
  ("Over-smoothing", "Ce s-a întîmplat cînd rețeaua GCN a trecut de la 2 la 6 straturi?",
   ["PR-AUC a crescut, pentru că fiecare nod a văzut mai mult din graf", "Nimic nu s-a schimbat", "PR-AUC a scăzut de la " + R("depth.ap_2", ".3f") + " la " + R("depth.ap_6", ".3f") + ", iar reprezentările nodurilor au devenit mai asemănătoare",
    "Antrenarea a devenit imposibilă"],
   "Medierea repetată, ponderată după grad, împinge reprezentările spre un vector comun (similaritate cosinus " + R("depth.cos_6", ".2f") + " cu 6 straturi), deci clasele se estompează.",
   "La rețelele GNN mai adînc nu înseamnă mai bine: similaritatea cosinus medie a urcat la " + R("depth.cos_6", ".2f") + ", iar PR-AUC a scăzut la " + R("depth.ap_6", ".3f") + "."))
q(1, ("GraphSAGE", "What makes GraphSAGE suitable for scoring new transactions in real time?",
      ["It uses the full Laplacian spectrum", "Its weights do not depend on the graph, and it aggregates a sample of neighbours: it is inductive", "It needs no node features", "It trains only on unknown nodes"],
      "GraphSAGE learns functions of a node and its sampled neighbourhood, so a new node gets a score without retraining, at bounded cost.",
      "Spectral methods are tied to one fixed graph; GraphSAGE separates self and neighbour weights and samples neighbours, so it generalises to unseen nodes."),
  ("GraphSAGE", "Ce face GraphSAGE potrivit pentru scorurile tranzacțiilor noi, în timp real?",
   ["Folosește întregul spectru al laplacianului", "Ponderile nu depind de graf, iar agregarea se face pe un eșantion de vecini: este inductiv", "Nu are nevoie de variabilele nodurilor", "Se antrenează numai pe nodurile necunoscute"],
   "GraphSAGE învață funcții ale nodului și ale vecinătății eșantionate, deci un nod nou primește un scor fără reantrenare, cu un cost limitat.",
   "Metodele spectrale sînt legate de un graf fix; GraphSAGE separă ponderile nodului de ale vecinilor și eșantionează vecinii, deci se generalizează la noduri nevăzute."))
q(3, ("GAT and transformers", "What is the relation between a graph attention network (GAT) and the transformer of Lecture 7?",
      ["They are unrelated", "A GAT has no softmax", "A transformer uses the Laplacian of the token graph", "A transformer is attention on the complete graph of tokens; a GAT restricts it to the edges"],
      "Both weight neighbours with a softmax of learned scores; the difference is which pairs may attend to each other.",
      "The softmax over learned compatibility scores is common to both; a GAT only lets a node attend to its graph neighbours."),
  ("GAT și transformerele", "Care este legătura dintre o rețea cu atenție pe graf (GAT) și transformerul din cursul 7?",
   ["Nu au nicio legătură", "GAT nu are softmax", "Transformerul folosește laplacianul grafului tokenilor", "Transformerul aplică atenția pe graful complet al tokenilor; GAT o restrînge la muchii"],
   "Ambele ponderează vecinii cu un softmax al unor scoruri învățate; diferența este care perechi au voie să-și acorde atenție.",
   "Softmax-ul pe scoruri de compatibilitate învățate este comun; GAT lasă un nod să acorde atenție doar vecinilor din graf."))
q(0, ("Unknown nodes", "About " + E("data.pct_labelled", ".1f") + " % of the Elliptic nodes are labelled. What role do the unknown nodes play in GCN training?",
      ["They pass messages between labelled nodes but do not enter the loss", "They are deleted before training", "They are treated as licit", "They are used as the test set"],
      "Training is semi-supervised: all nodes take part in message passing, only labelled training nodes enter the loss.",
      "Unknown nodes are neither deleted nor relabelled; they carry information along the edges, which is part of the value of the graph."),
  ("Nodurile necunoscute", "Circa " + R("data.pct_labelled", ".1f") + " % dintre nodurile Elliptic sînt etichetate. Ce rol au nodurile necunoscute la antrenarea GCN?",
   ["Transmit mesaje între nodurile etichetate, dar nu intră în funcția de pierdere", "Sînt șterse înainte de antrenare", "Sînt tratate ca licite", "Sînt folosite ca set de test"],
   "Antrenarea este semi-supervizată: toate nodurile participă la transmiterea mesajelor, dar în funcția de pierdere intră numai nodurile etichetate de antrenare.",
   "Nodurile necunoscute nu sînt nici șterse, nici reetichetate; ele duc informație de-a lungul muchiilor, ceea ce face parte din valoarea grafului."))
q(2, ("Leakage-free scaling", "Feature means and standard deviations for standardisation are computed on:",
      ["All 49 time steps", "The test steps only", "The training steps 1–34 only", "Each time step separately, including the test"],
      "Statistics from the test period would leak information about the future into the model; only the training steps may be used.",
      "Any statistic computed with test data is a leak; the scaler is fitted on steps 1–34 and applied unchanged to the test steps."),
  ("Standardizare fără scurgeri", "Mediile și abaterile standard pentru standardizarea variabilelor se calculează pe:",
   ["Toți cei 49 de pași", "Numai pașii de test", "Numai pașii de antrenare 1–34", "Fiecare pas separat, inclusiv cei de test"],
   "Statisticile din perioada de test ar scurge în model informație despre viitor; se pot folosi numai pașii de antrenare.",
   "Orice statistică calculată cu date de test este o scurgere; standardizarea se estimează pe pașii 1–34 și se aplică neschimbată pe pașii de test."))
q(1, ("Best model on Elliptic", "Which model had the highest illicit-class F1 on the test steps?",
      ["The GCN, " + E("models.gcn.f1", ".3f"),
       {"rf": "The random forest", "rf_emb": "The random forest with GCN embeddings"}.get(N["best_f1_model"], N["best_f1_model"]) + ", " + E("models." + N["best_f1_model"] + ".f1", ".3f"),
       "The logistic regression, " + E("models.lr_all.f1", ".3f"), "The MLP, " + E("models.mlp.f1", ".3f")],
      "Tree ensembles on Elliptic's pre-aggregated features lead; the GCN beats the same network without edges but not the forest.",
      "The forest-based model leads (F1 " + E("models." + N["best_f1_model"] + ".f1", ".3f") + "), the GCN reaches " + E("models.gcn.f1", ".3f") + " and the logistic regression " + E("models.lr_all.f1", ".3f") + "."),
  ("Cel mai bun model pe Elliptic", "Care model a avut cel mai mare F1 pe clasa ilicită pe pașii de test?",
   ["Rețeaua GCN, " + R("models.gcn.f1", ".3f"),
    {"rf": "Pădurea aleatoare", "rf_emb": "Pădurea aleatoare cu reprezentări GCN"}.get(N["best_f1_model"], N["best_f1_model"]) + ", " + R("models." + N["best_f1_model"] + ".f1", ".3f"),
    "Regresia logistică, " + R("models.lr_all.f1", ".3f"), "MLP-ul, " + R("models.mlp.f1", ".3f")],
   "Ansamblurile de arbori pe variabilele deja agregate ale Elliptic conduc; GCN bate aceeași rețea fără muchii, dar nu și pădurea.",
   "Modelul bazat pe pădure conduce (F1 " + R("models." + N["best_f1_model"] + ".f1", ".3f") + "), GCN ajunge la " + R("models.gcn.f1", ".3f") + ", iar regresia logistică la " + R("models.lr_all.f1", ".3f") + "."))
q(3, ("Value of the edges", "How do we isolate the value of the graph edges for a neural model?",
      ["Compare the GCN with the random forest", "Compare the GCN with and without dropout", "Compare GraphSAGE with GAT", "Compare the GCN with an MLP that has the same layers but no messages"],
      "Same layers, same features, same training: the only difference is the message passing. On Elliptic the PR-AUC difference is " + E("gcn_vs_mlp_ap_pts", "+.1f") + " points.",
      "A fair ablation changes one thing only; the MLP with identical layers removes just the edges, so the gap measures their value."),
  ("Valoarea muchiilor", "Cum izolăm valoarea muchiilor grafului pentru un model neuronal?",
   ["Comparăm GCN cu pădurea aleatoare", "Comparăm GCN cu și fără dropout", "Comparăm GraphSAGE cu GAT", "Comparăm GCN cu un MLP cu aceleași straturi, dar fără mesaje"],
   "Aceleași straturi, aceleași variabile, aceeași antrenare: singura diferență este transmiterea mesajelor. Pe Elliptic diferența de PR-AUC este " + R("gcn_vs_mlp_ap_pts", "+.1f") + " puncte.",
   "O comparație corectă schimbă un singur lucru; MLP-ul cu straturi identice elimină doar muchiile, deci diferența le măsoară valoarea."))
q(0, ("Why the forest is strong", "Why is a random forest without the graph hard to beat on Elliptic?",
      [E("data.n_agg") + " of its " + E("data.n_features") + " features are already aggregates over the neighbours", "Random forests use the graph internally",
       "The test set contains no illicit transactions", "The forest was trained on the test steps"],
      "The aggregated features are a hand-made message-passing layer computed by Elliptic; trees also fit rule-like fraud thresholds.",
      "The forest never sees the edges directly, but " + E("data.n_agg") + " features summarise each node's neighbourhood: the graph information is already in the table."),
  ("De ce pădurea este puternică", "De ce este greu de bătut pe Elliptic o pădure aleatoare fără graf?",
   [R("data.n_agg") + " dintre cele " + R("data.n_features") + " de variabile ale ei sînt deja agregate pe vecini", "Pădurile aleatoare folosesc graful intern",
    "Setul de test nu conține tranzacții ilicite", "Pădurea a fost antrenată pe pașii de test"],
   "Variabilele agregate sînt un strat de transmitere a mesajelor construit manual de Elliptic; în plus, arborii prind bine pragurile de tip regulă ale fraudei.",
   "Pădurea nu vede direct muchiile, dar " + R("data.n_agg") + " de variabile rezumă vecinătatea fiecărui nod: informația din graf este deja în tabel."))
q(2, ("Local features only", "A logistic regression on the 93 local features only reaches an illicit F1 of " + E("models.lr_local.f1", ".3f") + ". With all features it reaches:",
      [E("models.lr_local.f1", ".3f"), "0", E("models.lr_all.f1", ".3f"), "1"],
      "The neighbourhood aggregates change the F1 of the same model from " + E("models.lr_local.f1", ".3f") + " to " + E("models.lr_all.f1", ".3f") + ".",
      "Adding the " + E("data.n_agg") + " aggregated features gives " + E("models.lr_all.f1", ".3f") + ": a measure of how much the neighbours tell about a transaction."),
  ("Doar variabilele locale", "O regresie logistică numai pe cele 93 de variabile locale atinge F1 " + R("models.lr_local.f1", ".3f") + " pe clasa ilicită. Cu toate variabilele atinge:",
   [R("models.lr_local.f1", ".3f"), "0", R("models.lr_all.f1", ".3f"), "1"],
   "Agregatele pe vecini schimbă F1 al aceluiași model de la " + R("models.lr_local.f1", ".3f") + " la " + R("models.lr_all.f1", ".3f") + ".",
   "Adăugarea celor " + R("data.n_agg") + " de variabile agregate dă " + R("models.lr_all.f1", ".3f") + ": o măsură a cît spun vecinii despre o tranzacție."))
q(1, ("Bootstrap", "How was the uncertainty of the PR-AUC difference between the random forest and the GCN assessed?",
      ["By training on the test set", "By a paired bootstrap of the test transactions, 1,000 draws", "By a t-test on the training loss", "It was not assessed"],
      "Resampling the test nodes with the same draw for both models gave a difference of " + E("boot.rf_minus_gcn_ap", ".3f") + " with a 95 % interval [" + E("boot.lo", ".3f") + "; " + E("boot.hi", ".3f") + "].",
      "Uncertainty comes from the finite test set; the paired bootstrap resamples it 1,000 times and reports a 95 % interval of the difference."),
  ("Bootstrap", "Cum a fost evaluată incertitudinea diferenței de PR-AUC dintre pădurea aleatoare și GCN?",
   ["Prin antrenare pe setul de test", "Printr-un bootstrap pereche al tranzacțiilor de test, 1 000 de extrageri", "Printr-un test t pe pierderea de antrenare", "Nu a fost evaluată"],
   "Reeșantionarea nodurilor de test, cu aceeași extragere pentru ambele modele, a dat o diferență de " + R("boot.rf_minus_gcn_ap", ".3f") + ", cu intervalul de 95 % [" + R("boot.lo", ".3f") + "; " + R("boot.hi", ".3f") + "].",
   "Incertitudinea vine din setul de test finit; bootstrap-ul pereche îl reeșantionează de 1 000 de ori și raportează un interval de 95 % al diferenței."))
q(3, ("Concept drift", "What happened to the models' F1 from time step 43 onwards?",
      ["It rose for all models", "It stayed constant", "Only the GCN improved", "It collapsed for all models after a dark-market shutdown changed illicit behaviour"],
      "The random forest falls from a mean F1 of " + E("shutdown.rf_f1_pre", ".3f") + " before step 43 to " + E("shutdown.rf_f1_post", ".3f") + " after it; no model had seen the new behaviour.",
      "After the shutdown of a major dark market, illicit flows changed form; every model trained on steps 1–34 lost most of its F1."),
  ("Schimbarea conceptului", "Ce s-a întîmplat cu F1 al modelelor de la pasul de timp 43 încolo?",
   ["A crescut pentru toate modelele", "A rămas constant", "Doar GCN s-a îmbunătățit", "S-a prăbușit pentru toate modelele, după ce închiderea unei piețe ilegale a schimbat comportamentul ilicit"],
   "Pădurea aleatoare scade de la un F1 mediu de " + R("shutdown.rf_f1_pre", ".3f") + " înainte de pasul 43 la " + R("shutdown.rf_f1_post", ".3f") + " după el; niciun model nu văzuse noul comportament.",
   "După închiderea unei piețe ilegale importante, fluxurile ilicite și-au schimbat forma; toate modelele antrenate pe pașii 1–34 și-au pierdut cea mai mare parte din F1."))
q(0, ("Alert budget", "An investigation team can review " + E("alerts.k", ",") + " alerts from the test period. What share of all illicit transactions does the random forest put in that list?",
      [E("alerts.rf", ".1f") + " %", E("alerts.rf_prec", ".1f") + " %", "100 %", E("data.pct_test_illicit", ".2f") + " %"],
      "The top " + E("alerts.k", ",") + " forest scores contain " + E("alerts.rf", ".1f") + " % of the " + E("alerts.n_test_illicit", ",") + " illicit test transactions; " + E("alerts.rf_prec", ".1f") + " % of those alerts are illicit.",
      "Recall at " + E("alerts.k", ",") + " alerts is " + E("alerts.rf", ".1f") + " %; " + E("alerts.rf_prec", ".1f") + " % is the hit rate of the alerts, a different quantity."),
  ("Bugetul de alerte", "O echipă de investigații poate verifica " + R("alerts.k", ",") + " de alerte din perioada de test. Ce pondere din toate tranzacțiile ilicite pune pădurea aleatoare în această listă?",
   [R("alerts.rf", ".1f") + " %", R("alerts.rf_prec", ".1f") + " %", "100 %", R("data.pct_test_illicit", ".2f") + " %"],
   "Primele " + R("alerts.k", ",") + " de scoruri ale pădurii conțin " + R("alerts.rf", ".1f") + " % dintre cele " + R("alerts.n_test_illicit", ",") + " de tranzacții ilicite de test; " + R("alerts.rf_prec", ".1f") + " % dintre aceste alerte sînt ilicite.",
   "Recall-ul la " + R("alerts.k", ",") + " de alerte este " + R("alerts.rf", ".1f") + " %; " + R("alerts.rf_prec", ".1f") + " % este rata de reușită a alertelor, o mărime diferită."))
q(2, ("Embeddings for other models", "What are the GCN embeddings used for in the forest + GCN model?",
      ["They replace all original features", "They are the labels", "They are added as " + E("emb.dim") + " extra inputs to the random forest", "They decide the train-test split"],
      "The hidden layer of the GCN summarises each node's neighbourhood; the forest uses these vectors alongside the original features.",
      "The original features stay; the GCN's " + E("emb.dim") + " hidden outputs are appended as new columns, a common way to reuse graph knowledge in tabular models."),
  ("Reprezentări pentru alte modele", "La ce servesc reprezentările GCN în modelul pădure + GCN?",
   ["Înlocuiesc toate variabilele originale", "Sînt etichetele", "Sînt adăugate ca " + R("emb.dim") + " de intrări suplimentare ale pădurii aleatoare", "Decid împărțirea în antrenare și test"],
   "Stratul ascuns al GCN rezumă vecinătatea fiecărui nod; pădurea folosește acești vectori alături de variabilele originale.",
   "Variabilele originale rămîn; cele " + R("emb.dim") + " de ieșiri ascunse ale GCN se adaugă drept coloane noi, un mod obișnuit de a refolosi cunoașterea din graf în modelele tabelare."))
q(1, ("Explanation stability", "After adding GCN embeddings, the 10 top features of the forest carry " + E("stability.top10_share_rfemb_pct", ".1f") + " % of the importance instead of " + E("stability.top10_share_rf_pct", ".1f") + " %. What is the right reading?",
      ["The original features became useless", "Part of the shift is mechanical, because new inputs absorb importance; retraining variability must also be accounted for",
       "The forest is broken", "The embeddings leaked the labels"],
      "Added inputs absorb attribution mass (" + E("stability.emb_share_pct", ".1f") + " % here) and refits differ by chance; both must be accounted for before reading a new explanation, as in Diaconu and Pele (2026).",
      "A change of explanation after a change of model may be mechanical; comparing against retraining variability and matched controls is the rigorous test."),
  ("Stabilitatea explicațiilor", "După adăugarea reprezentărilor GCN, primele 10 variabile ale pădurii au " + R("stability.top10_share_rfemb_pct", ".1f") + " % din importanță în loc de " + R("stability.top10_share_rf_pct", ".1f") + " %. Care este interpretarea corectă?",
   ["Variabilele originale au devenit inutile", "O parte a schimbării este mecanică, pentru că intrările noi preiau importanță; trebuie luată în calcul și variabilitatea reantrenării",
    "Pădurea este defectă", "Reprezentările au scurs etichetele"],
   "Intrările adăugate preiau o parte din atribuire (" + R("stability.emb_share_pct", ".1f") + " % aici), iar reantrenările diferă întîmplător; ambele trebuie luate în calcul înainte de a citi o explicație nouă, ca în Diaconu și Pele (2026).",
   "O schimbare a explicației după o schimbare de model poate fi mecanică; comparația cu variabilitatea reantrenării și cu controale potrivite este testul riguros."))
q(3, ("Edges and time", "In Elliptic, " + E("data.pct_same_step", ".1f") + " % of the edges join transactions of the same time step. Why does this matter?",
      ["It makes the graph undirected", "It means there are no hubs", "It doubles the number of features", "Test nodes cannot receive messages from training-period nodes, so the temporal split does not leak through edges"],
      "If edges crossed periods, a test node could receive information from the future (or the past labelled period) via messages; here the steps are separate graphs.",
      "Leakage through edges is a graph-specific pitfall; with edges inside one step, messages never cross the train-test boundary."),
  ("Muchii și timp", "În Elliptic, " + R("data.pct_same_step", ".1f") + " % dintre muchii leagă tranzacții din același pas de timp. De ce contează acest lucru?",
   ["Face graful neorientat", "Înseamnă că nu există noduri centrale", "Dublează numărul de variabile", "Nodurile de test nu pot primi mesaje de la nodurile din perioada de antrenare, deci împărțirea în timp nu scurge informație prin muchii"],
   "Dacă muchiile ar traversa perioadele, un nod de test ar putea primi prin mesaje informație din viitor (sau din perioada etichetată); aici pașii sînt grafuri separate.",
   "Scurgerea prin muchii este o capcană specifică grafurilor; cu muchii în interiorul unui pas, mesajele nu trec niciodată granița dintre antrenare și test."))
q(0, ("Class weights", "The logistic regression uses class weights 0.7 (illicit) and 0.3 (licit) in the loss. What is the effect visible in the results?",
      ["Higher recall at the cost of precision: recall " + E("models.lr_all.rec", ".3f") + ", precision " + E("models.lr_all.prec", ".3f"), "Higher precision and lower recall", "No effect on the predictions", "All transactions are predicted illicit"],
      "Up-weighting the rare class pushes the model to flag more illicit transactions: recall rises, precision falls.",
      "Class weights trade precision for recall; the recall of the logistic regression (" + E("models.lr_all.rec", ".3f") + ") far exceeds its precision (" + E("models.lr_all.prec", ".3f") + ")."),
  ("Ponderile claselor", "GCN folosește în funcția de pierdere ponderile 0,7 (ilicit) și 0,3 (licit). Ce efect se vede în rezultate?",
   ["Recall mai mare cu prețul preciziei: recall " + R("models.lr_all.rec", ".3f") + ", precizie " + R("models.lr_all.prec", ".3f"), "Precizie mai mare și recall mai mic", "Niciun efect asupra predicțiilor", "Toate tranzacțiile sînt prezise ilicite"],
   "Creșterea ponderii clasei rare împinge modelul să semnaleze mai multe tranzacții ilicite: recall-ul crește, precizia scade.",
   "Ponderile de clasă schimbă precizie pe recall; recall-ul regresiei logistice (" + R("models.lr_all.rec", ".3f") + ") depășește mult precizia ei (" + R("models.lr_all.prec", ".3f") + ")."))
q(2, ("Graph tasks", "A bank wants to know whether a cluster of accounts forms a laundering ring. Which graph task is this?",
      ["Node classification", "Edge prediction", "Graph classification", "Graph generation"],
      "The unit of decision is a whole subgraph, so its node representations are pooled (e.g. summed) into one graph-level prediction.",
      "Node classification labels single transactions and edge prediction single links; a ring is a property of the whole cluster."),
  ("Sarcini pe grafuri", "O bancă vrea să știe dacă un grup de conturi formează o rețea de spălare a banilor. Ce sarcină pe graf este aceasta?",
   ["Clasificarea nodurilor", "Predicția muchiilor", "Clasificarea grafurilor", "Generarea de grafuri"],
   "Unitatea de decizie este un subgraf întreg, deci reprezentările nodurilor lui se agregă (de exemplu prin sumă) într-o singură predicție la nivel de graf.",
   "Clasificarea nodurilor etichetează tranzacții izolate, iar predicția muchiilor legături izolate; o rețea de spălare este o proprietate a întregului grup."))
q(1, ("Degree distribution", "The mean degree is " + E("data.mean_degree", ".2f") + " but the largest hub has " + E("data.max_degree") + " neighbours. What does this imply for a GNN?",
      ["Nothing, degrees do not matter", "Messages must be normalised by degree, or hubs dominate the sums", "Hubs must be deleted", "Only hubs should be classified"],
      "A heavy-tailed degree distribution makes raw sums explode at hubs; symmetric degree normalisation (GCN) or mean aggregation keeps scales comparable.",
      "Without normalisation a node with hundreds of neighbours would receive hundreds of messages; this is why the GCN uses $\\tilde D^{-1/2}\\tilde A\\tilde D^{-1/2}$."),
  ("Distribuția gradelor", "Gradul mediu este " + R("data.mean_degree", ".2f") + ", dar cel mai mare nod central are " + R("data.max_degree") + " de vecini. Ce implică acest lucru pentru o rețea GNN?",
   ["Nimic, gradele nu contează", "Mesajele trebuie normalizate după grad, altfel nodurile centrale domină sumele", "Nodurile centrale trebuie șterse", "Doar nodurile centrale trebuie clasificate"],
   "O distribuție a gradelor cu coadă grea face ca sumele brute să explodeze la nodurile centrale; normalizarea simetrică după grad (GCN) sau agregarea prin medie păstrează scalele comparabile.",
   "Fără normalizare, un nod cu sute de vecini ar primi sute de mesaje; de aceea GCN folosește $\\tilde D^{-1/2}\\tilde A\\tilde D^{-1/2}$."))
q(3, ("Laplacian smoothness", "For a node signal $\\mathbf x$, what does a small value of $\\mathbf x^\\top L\\mathbf x$ mean?",
      ["The graph has few nodes", "The signal is large", "The graph is disconnected", "Neighbouring nodes have similar values: the signal is smooth on the graph"],
      "$\\mathbf x^\\top L\\mathbf x=\\sum_{(u,v)\\in\\mathcal E}(x_u-x_v)^2$ sums squared differences across edges, so it is small when neighbours look alike.",
      "The quadratic form of the Laplacian measures variation along edges; homophilous labels are smooth signals, which neighbour averaging preserves."),
  ("Netezimea laplaciană", "Pentru un semnal $\\mathbf x$ pe noduri, ce înseamnă o valoare mică a lui $\\mathbf x^\\top L\\mathbf x$?",
   ["Graful are puține noduri", "Semnalul este mare", "Graful este neconex", "Nodurile vecine au valori asemănătoare: semnalul este neted pe graf"],
   "$\\mathbf x^\\top L\\mathbf x=\\sum_{(u,v)\\in\\mathcal E}(x_u-x_v)^2$ adună diferențele pătratice de-a lungul muchiilor, deci este mică atunci cînd vecinii seamănă.",
   "Forma pătratică a laplacianului măsoară variația de-a lungul muchiilor; etichetele homofile sînt semnale netede, pe care media pe vecini le păstrează."))
q(0, ("PR-AUC baseline", "What PR-AUC would a random score achieve on the Elliptic test steps?",
      ["About " + E("data.pct_test_illicit", ".2f") + " %, the share of illicit transactions", "50 %", "0 %", "100 %"],
      "A random ranking has precision equal to the base rate at every recall, so its PR-AUC equals the share of illicit transactions, " + E("data.pct_test_illicit", ".2f") + " %.",
      "Unlike ROC-AUC, whose random baseline is 0.5, the baseline of PR-AUC is the prevalence of the positive class."),
  ("Nivelul de bază al PR-AUC", "Ce PR-AUC ar obține un scor aleator pe pașii de test Elliptic?",
   ["Circa " + R("data.pct_test_illicit", ".2f") + " %, ponderea tranzacțiilor ilicite", "50 %", "0 %", "100 %"],
   "O ordonare aleatoare are precizia egală cu rata de bază la orice recall, deci PR-AUC este egal cu ponderea tranzacțiilor ilicite, " + R("data.pct_test_illicit", ".2f") + " %.",
   "Spre deosebire de ROC-AUC, al cărui nivel aleator este 0,5, nivelul de bază al PR-AUC este frecvența clasei pozitive."))
q(2, ("Embedding space", "What does the PCA projection of the GCN hidden layer show for the test transactions?",
      ["All transactions on a single point", "Random noise without structure", "Illicit transactions gathering in a region of their own", "Only the time steps"],
      "The GCN learns a representation in which illicit transactions cluster; its first two principal components explain " + E("emb.var2_pct", ".1f") + " % of the variance.",
      "The projection is not noise: the learned space separates much of the illicit activity, which is why the embeddings help other models."),
  ("Spațiul reprezentărilor", "Ce arată proiecția PCA a stratului ascuns GCN pentru tranzacțiile de test?",
   ["Toate tranzacțiile într-un singur punct", "Zgomot aleator, fără structură", "Tranzacțiile ilicite adunîndu-se într-o regiune proprie", "Doar pașii de timp"],
   "Rețeaua GCN învață o reprezentare în care tranzacțiile ilicite se grupează; primele două componente principale explică " + R("emb.var2_pct", ".1f") + " % din varianță.",
   "Proiecția nu este zgomot: spațiul învățat separă o mare parte din activitatea ilicită; de aceea reprezentările ajută alte modele."))
q(1, ("Group research", "What is the inferential unit in the explanation-stability framework of Diaconu and Pele (2026)?",
      ["A single feature", "The complete model refit", "A single transaction", "One SHAP value"],
      "The paper treats the complete refit as the unit of inference and compares graph-augmented models with graph-free controls matched in dimension and covariance.",
      "Comparing single features before and after an intervention can mislead, because retraining alone moves attributions; the paper therefore reasons across repeated refits."),
  ("Cercetarea grupului", "Care este unitatea de inferență în cadrul de analiză a stabilității explicațiilor din Diaconu și Pele (2026)?",
   ["O singură variabilă", "Reantrenarea completă a modelului", "O singură tranzacție", "O singură valoare SHAP"],
   "Lucrarea tratează reantrenarea completă ca unitate de inferență și compară modelele cu variabile de graf cu controale fără graf, cu aceeași dimensiune și covarianță.",
   "Compararea variabilelor izolate înainte și după o intervenție poate induce în eroare, pentru că reantrenarea însăși mișcă atribuirile; de aceea lucrarea raționează pe reantrenări repetate."))
q(3, ("Number of layers in a GNN", "A practitioner wants a GNN to see 10 hops of money flow. What is the recommended design?",
      ["Stack 10 plain GCN layers", "Use a single layer with a larger hidden size", "Remove the degree normalisation", "Keep 2–3 layers and add residual links or skip connections if more depth is needed"],
      "Plain deep GCNs over-smooth; residual or jumping-knowledge connections let information from early layers survive.",
      "Ten averaging layers would blur all nodes together; shallow models with residual links are the standard remedy."),
  ("Numărul de straturi într-o rețea GNN", "Un practician vrea ca o rețea GNN să vadă 10 salturi ale fluxului de bani. Care este arhitectura recomandată?",
   ["10 straturi GCN simple", "Un singur strat cu o dimensiune ascunsă mai mare", "Eliminarea normalizării după grad", "2–3 straturi și conexiuni reziduale dacă este nevoie de mai multă adîncime"],
   "Rețelele GCN simple și adînci netezesc excesiv; conexiunile reziduale lasă informația din primele straturi să supraviețuiască.",
   "Zece straturi de mediere ar estompa toate nodurile; modelele puțin adînci, cu conexiuni reziduale, sînt remediul obișnuit."))
q(0, ("Inductive vs transductive", "Tomorrow a new block of transactions arrives. Which property lets a model score them without retraining?",
      ["Inductive learning: the layer is a function of node features and neighbourhood, not of a fixed node set", "Transductive learning on today's graph only",
       "A larger number of epochs", "Using only the unknown nodes"],
      "Message-passing layers with shared weights (GraphSAGE, GCN applied to the new graph) are functions of local structure, so they apply to new nodes.",
      "Models tied to one fixed node set (such as learned node-ID embeddings) cannot score unseen nodes; shared-weight message passing can."),
  ("Inductiv și transductiv", "Mîine sosește un nou bloc de tranzacții. Ce proprietate permite unui model să le dea scoruri fără reantrenare?",
   ["Învățarea inductivă: stratul este o funcție de variabilele nodului și de vecinătate, nu de o mulțime fixă de noduri", "Învățarea transductivă numai pe graful de azi",
    "Un număr mai mare de epoci", "Folosirea numai a nodurilor necunoscute"],
   "Straturile de transmitere a mesajelor cu ponderi comune (GraphSAGE, GCN aplicat noului graf) sînt funcții ale structurii locale, deci se aplică nodurilor noi.",
   "Modelele legate de o mulțime fixă de noduri (de exemplu reprezentări învățate pentru fiecare identificator) nu pot da scoruri nodurilor nevăzute; transmiterea mesajelor cu ponderi comune poate."))

if __name__ == "__main__":
    assert len(Q) >= 30, len(Q)
    for x in Q:
        assert x["correct"] in (0, 1, 2, 3) and len(x["en"]["options"]) == 4 and len(x["ro"]["options"]) == 4
    head = ("// ============================================================\n"
            "// Quiz bank for lecture id 'l9': Graph neural networks (EN + RO)\n"
            "// GENERATED by lectures/L9/make_quiz_L9.py from L9_numbers.json; do not edit by hand.\n"
            "// correct = index (0-3) of the right option in the original order; incorrectExplanation never names a letter.\n"
            "// ============================================================\n")
    js = head + "window.NN_DATA.quizzes['l9'] = " + json.dumps({"draw": 20, "questions": Q}, ensure_ascii=False, indent=1) + ";\n"
    with open(os.path.join(ROOT, "assets", "quizzes", "l9.js"), "w", encoding="utf-8") as fh:
        fh.write(js)
    print("assets/quizzes/l9.js:", len(Q), "questions")
