r"""
build_L9.py -- Lecture 9 "Graph neural networks" (NN), English and Romanian from one source.
Writes lecture9.tex / lecture9_ro.tex from the bilingual slide text below and L9_numbers.json (analysis_L9.py),
compiles both and applies every check of lectures/STYLE.md (nn_build.run_build).
Run (from the repository root):
    python3 lectures/L9/analysis_L9.py
    python3 lectures/L9/make_quantlets_L9.py
    python3 lectures/L9/build_L9.py            # both languages; "en" or "ro" builds one
Hand-typed numbers are definitions only: time steps 34/35 (split of Weber et al., 2019), class weights 0.3/0.7,
threshold 0.5, 100 hidden units, learning rate 0.003, 1,000 alerts, 1,000 bootstrap draws.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from nn_build import (Deck, Numbers, abbreviation_frames, cols, fmt, heat, hier, items, reference_frames,  # noqa: E402
                      run_build, table)

ABBR = {
    "AML": ("anti-money laundering", "combaterea spălării banilor (anti-money laundering)"),
    "ASE": ("Bucharest University of Economic Studies", "Academia de Studii Economice din București"),
    "BTC": ("bitcoin (currency unit)", "bitcoin (unitate monetară)"),
    "CNN": ("convolutional neural network", "rețea neuronală convoluțională"),
    "CPU": ("central processing unit", "unitate centrală de procesare"),
    "GAT": ("graph attention network", "rețea cu atenție pe graf (graph attention network)"),
    "GCN": ("graph convolutional network", "rețea convoluțională pe graf (graph convolutional network)"),
    "GIN": ("graph isomorphism network", "rețea de izomorfism pe graf (graph isomorphism network)"),
    "GNN": ("graph neural network", "rețea neuronală pe graf (graph neural network)"),
    "GraphSAGE": ("graph sample and aggregate", "eșantionare și agregare pe graf (graph sample and aggregate)"),
    "IPE": ("Institute for Economic Forecasting, Romanian Academy", "Institutul de Prognoză Economică, Academia Română"),
    "MLP": ("multilayer perceptron", "perceptron multistrat"),
    "MPNN": ("message-passing neural network", "rețea neuronală cu transmitere de mesaje (message passing)"),
    "PCA": ("principal component analysis", "analiza componentelor principale"),
    "PR-AUC": ("area under the precision-recall curve", "aria de sub curba precizie-recall"),
    "PyG": ("PyTorch Geometric (library)", "PyTorch Geometric (bibliotecă de programe)"),
    "ReLU": ("rectified linear unit", "unitate liniară rectificată (rectified linear unit)"),
    "SHAP": ("SHapley Additive exPlanations", "explicații aditive bazate pe valorile Shapley (SHapley Additive exPlanations)"),
    "WL": ("Weisfeiler--Leman graph isomorphism test", "testul de izomorfism Weisfeiler--Leman"),
    "XAI": ("explainable artificial intelligence", "inteligență artificială explicabilă"),
}
IGNORE = {"TP", "FP", "FN", "KDD", "ICLR", "NeurIPS", "NIPS", "SSRN", "ECECSR", "ESWA", "AAAI", "IEEE", "ACM", "SIGKDD",
          "OpenML", "ULB", "AD", "CI", "EvolveGCN", "GN", "LeakyReLU", "NEURAL", "BIZ", "PAP", "S01045ED1V01Y202009AIM", "ISE", "II", "MIT", "CS", "AMLA", "EU", "UE", "PDF", "DOI", "URL", "JMLR", "PMLR", "ICML", "AI", "IDA"}
UNITS = {"BTC"}
REFS = [
    "Bishop, C.\\,M.\\ \\& Bishop, H.\\ (2024). \\textit{Deep Learning: Foundations and Concepts}, ch.\\ 13. Springer. "
    "\\href{https://doi.org/10.1007/978-3-031-45468-4}{doi:10.1007/978-3-031-45468-4}",
    "Breiman, L.\\ (2001). Random forests. \\textit{Machine Learning}, 45(1), 5--32. \\href{https://doi.org/10.1023/A:1010933404324}{doi:10.1023/A:1010933404324}",
    "Diaconu, D.\\ \\& Pele, D.\\,T.\\ (2026). Testing SHAP stability under distribution shift in graph-enhanced fraud detection. "
    "Submitted to \\textit{Expert Systems with Applications}. Code: "
    "\\href{https://github.com/QuantLet/GNN_XAI_Stability}{github.com/QuantLet/GNN\\_XAI\\_Stability}",
    "Efron, B.\\ \\& Tibshirani, R.\\,J.\\ (1993). \\textit{An Introduction to the Bootstrap}. Chapman \\& Hall. "
    "\\href{https://doi.org/10.1201/9780429246593}{doi:10.1201/9780429246593}",
    "Fey, M.\\ \\& Lenssen, J.\\,E.\\ (2019). Fast graph representation learning with PyTorch Geometric. ICLR Workshop on Representation "
    "Learning on Graphs and Manifolds. \\href{https://arxiv.org/abs/1903.02428}{arXiv:1903.02428}",
    "Gilmer, J., Schoenholz, S.\\,S., Riley, P.\\,F., Vinyals, O.\\ \\& Dahl, G.\\,E.\\ (2017). Neural message passing for quantum chemistry. "
    "\\textit{Proceedings of ICML}, PMLR 70, 1263--1272. \\href{https://arxiv.org/abs/1704.01212}{arXiv:1704.01212}",
    "Hamilton, W.\\,L., Ying, R.\\ \\& Leskovec, J.\\ (2017). Inductive representation learning on large graphs. \\textit{Advances in "
    "Neural Information Processing Systems}, 30. \\href{https://arxiv.org/abs/1706.02216}{arXiv:1706.02216}",
    "Hamilton, W.\\,L.\\ (2020). \\textit{Graph Representation Learning}. Morgan \\& Claypool. "
    "\\href{https://doi.org/10.2200/S01045ED1V01Y202009AIM046}{doi:10.2200/S01045ED1V01Y202009AIM046}",
    "Kipf, T.\\,N.\\ \\& Welling, M.\\ (2017). Semi-supervised classification with graph convolutional networks. \\textit{International "
    "Conference on Learning Representations}. \\href{https://arxiv.org/abs/1609.02907}{arXiv:1609.02907}",
    "Li, Q., Han, Z.\\ \\& Wu, X.-M.\\ (2018). Deeper insights into graph convolutional networks for semi-supervised learning. "
    "\\textit{Proceedings of the AAAI Conference on Artificial Intelligence}, 32(1). \\href{https://doi.org/10.1609/aaai.v32i1.11604}{doi:10.1609/aaai.v32i1.11604}",
    "Lundberg, S.\\,M.\\ \\& Lee, S.-I.\\ (2017). A unified approach to interpreting model predictions. \\textit{Advances in Neural "
    "Information Processing Systems}, 30. \\href{https://arxiv.org/abs/1705.07874}{arXiv:1705.07874}",
    "Pareja, A., Domeniconi, G., Chen, J.\\ et al.\\ (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. "
    "\\textit{Proceedings of the AAAI Conference on Artificial Intelligence}, 34(4), 5363--5370. \\href{https://doi.org/10.1609/aaai.v34i04.5984}{doi:10.1609/aaai.v34i04.5984}",
    "Toma, F.-M., Dima, D.-M., Grecu, R., Pele, D.\\,T.\\ \\& Lessmann, S.\\ (2026). Energy Financial Risk Meter: Systemic risk in the EU "
    "energy sector. SSRN Working Paper 6985301. \\href{https://doi.org/10.2139/ssrn.6985301}{doi:10.2139/ssrn.6985301}",
    "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P.\\ \\& Bengio, Y.\\ (2018). Graph attention networks. "
    "\\textit{International Conference on Learning Representations}. \\href{https://arxiv.org/abs/1710.10903}{arXiv:1710.10903}",
    "Weber, M., Domeniconi, G., Chen, J., Weidele, D.\\,K.\\,I., Bellei, C., Robinson, T.\\ \\& Leiserson, C.\\,E.\\ (2019). Anti-money "
    "laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. KDD Workshop on Anomaly "
    "Detection in Finance. \\href{https://arxiv.org/abs/1908.02591}{arXiv:1908.02591}",
    "Xu, K., Hu, W., Leskovec, J.\\ \\& Jegelka, S.\\ (2019). How powerful are graph neural networks? \\textit{International Conference "
    "on Learning Representations}. \\href{https://arxiv.org/abs/1810.00826}{arXiv:1810.00826}",
    "Ying, R., Bourgeois, D., You, J., Zitnik, M.\\ \\& Leskovec, J.\\ (2019). GNNExplainer: Generating explanations for graph neural "
    "networks. \\textit{Advances in Neural Information Processing Systems}, 32. \\href{https://arxiv.org/abs/1903.03894}{arXiv:1903.03894}",
]


def it(*xs, sep="2pt"):
    """L1/L2 bullets: a string is split by hier(); a tuple (head, sub1, ...) gives an explicit L1 head with L2 items."""
    def one(x):
        if isinstance(x, tuple):
            return (x[0] + "\n    \\begin{itemize}\\setlength{\\itemsep}{0pt}\n" +
                    "\n".join("      \\item " + (s[0].upper() + s[1:] if s[0].islower() else s) for s in x[1:] if s) +
                    "\n    \\end{itemize}")
        return hier(x)
    return (f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{{sep}}}\n" + "\n".join("  \\item " + one(x) for x in xs if x) +
            "\n\\end{itemize}")


def build(lang, H=None):
    V = Numbers(os.path.join(HERE, "L9_numbers.json"), lang)
    N = V.N
    ro = lang == "ro"
    D = Deck(9, "Graph neural networks" if not ro else "Rețele neuronale pe grafuri", V, lang, heights=H)
    t = D.t
    NB = t("notebook \\S", "notebook, secțiunea ")
    M = N["models"]

    def de(key):
        v = int(round(float(V.get(key))))
        return "de " if ro and (v >= 20 and (v % 100 >= 20 or v % 100 == 0)) else ""

    def n_(x, spec):
        return fmt(x, spec, lang)

    OWN = t("(own computation, Elliptic data)", "(calcule proprii, datele Elliptic)")

    # ------------------------------------------------------------------ front matter
    D.parts.append("% END-FRONT-MATTER\n")
    D.frame(t("Learning objectives", "Obiectivele cursului"), it(
        t("Represent financial data as a graph: nodes, edges, node features; adjacency, degree and the normalised adjacency matrix.",
          "Reprezentarea datelor financiare ca graf; noduri, muchii și variabilele nodurilor; matricea de adiacență, matricea gradelor și adiacența normalizată"),
        t("Write the message-passing scheme of a graph neural network (GNN) and its three classic layers: graph convolutional network (GCN), "
          "graph sample and aggregate (GraphSAGE), graph attention network (GAT).",
          "Schema de transmitere a mesajelor a unei rețele neuronale pe graf (GNN); cele trei straturi clasice: rețeaua convoluțională pe graf (GCN), "
          "eșantionarea și agregarea pe graf (GraphSAGE), rețeaua cu atenție pe graf (GAT)"),
        t("Train them on the Elliptic Bitcoin graph for anti-money laundering (AML), in time order, against a logistic regression and a random forest.",
          "Antrenarea lor pe graful Bitcoin Elliptic pentru combaterea spălării banilor (AML), în ordine cronologică; comparația cu regresia logistică și cu pădurea aleatoare"),
        t("Judge the results like an investigator: precision, recall, the area under the precision-recall curve (PR-AUC), alerts per analyst, drift and stability of explanations.",
          "Evaluarea rezultatelor ca un investigator; precizia, recall-ul și aria de sub curba precizie-recall (PR-AUC); alertele pe analist, schimbarea în timp și stabilitatea explicațiilor"),
        sep="5pt"))
    abbreviation_frames(D, ABBR)

    # ------------------------------------------------------------------ 1 why graphs
    D.section(t("Why graphs for financial crime", "De ce grafuri pentru infracțiunile financiare"))
    D.concept(t("Money laundering hides in the links, not in the rows", "Spălarea banilor se ascunde în legături, nu în rînduri"),
              t("A single payment can look ordinary; the chain of payments around it often does not.",
                "O plată izolată poate părea obișnuită; lanțul de plăți din jurul ei, de cele mai multe ori, nu."),
              "\\[ G=(\\mathcal V,\\mathcal E,\\mathbf X),\\qquad \\mathbf X\\in\\R^{n\\times d},\\qquad y_v\\in\\{0,1\\} \\]"
              "\\where{" + t("$\\mathcal V$ = nodes (transactions), $n=|\\mathcal V|$; $\\mathcal E$ = edges (a payment flow from one transaction to the next); "
                             "$\\mathbf X$ = node features, $d$ per node; $y_v=1$ if transaction $v$ is illicit.",
                             "$\\mathcal V$ = nodurile (tranzacțiile), $n=|\\mathcal V|$; $\\mathcal E$ = muchiile (un flux de plată de la o tranzacție la următoarea); "
                             "$\\mathbf X$ = variabilele nodurilor, cîte $d$ pentru fiecare nod; $y_v=1$ dacă tranzacția $v$ este ilicită.") + "}",
              it((t("\\textbf{Elliptic graph} (Weber et al., 2019)", "\\textbf{Graful Elliptic} (Weber et al., 2019)"),
                  t("@{data.n_nodes|,d} Bitcoin transactions, @{data.n_edges|,d} payment flows",
                    "@{data.n_nodes|,d} " + de("data.n_nodes") + "tranzacții Bitcoin, @{data.n_edges|,d} " + de("data.n_edges") + "fluxuri de plată"),
                  t("@{data.n_features} features per transaction, @{data.n_steps} time steps",
                    "@{data.n_features} " + de("data.n_features") + "variabile pe tranzacție, @{data.n_steps} " + de("data.n_steps") + "pași de timp")),
                 (t("\\textbf{The task}", "\\textbf{Sarcina}"),
                  t("node classification: illicit (scams, ransomware, dark markets) or licit",
                    "clasificarea nodurilor: ilicit (fraude, ransomware, piețe ilegale) sau licit")), sep="3pt"),
              t("a bank's anti-money-laundering team already thinks in chains of payments; a GNN learns from the same chains.",
                "echipa de conformitate a unei bănci gîndește deja în lanțuri de plăți; o rețea GNN învață din aceleași lanțuri."),
              tryit=NB + "1", cite="(Weber et al., 2019; Bishop \\& Bishop, 2024, ch.\\ 13)")

    D.frame(t("The Elliptic data in one table", "Datele Elliptic într-un singur tabel"), table(
        "@{}p{0.30\\textwidth}p{0.62\\textwidth}@{}",
        [t("Element", "Element"), t("Value in our copy of the data", "Valoarea în copia noastră a datelor")],
        [[t("Nodes, edges", "Noduri, muchii"), t("@{data.n_nodes|,d} transactions; @{data.n_edges|,d} directed payment flows",
                                               "@{data.n_nodes|,d} " + de("data.n_nodes") + "tranzacții; @{data.n_edges|,d} " + de("data.n_edges") + "fluxuri de plată orientate")],
         [t("Labels", "Etichete"), t("@{data.n_illicit|,d} illicit, @{data.n_licit|,d} licit, @{data.n_unknown|,d} unknown (@{data.pct_labelled|.1f}\\,\\% labelled)",
                                     "@{data.n_illicit|,d} ilicite, @{data.n_licit|,d} licite, @{data.n_unknown|,d} necunoscute (@{data.pct_labelled|.1f}\\,\\% etichetate)")],
         [t("Features", "Variabile"), t("@{data.n_local} local (amounts, fees, inputs, outputs) + @{data.n_agg} aggregated over the neighbours",
                                        "@{data.n_local} locale (sume, comisioane, intrări, ieșiri) și @{data.n_agg} agregate pe vecini")],
         [t("Time", "Timpul"), t("@{data.n_steps} steps of about two weeks; edges join transactions of the same step (@{data.pct_same_step|.1f}\\,\\%)",
                                 "@{data.n_steps} de pași de circa două săptămîni; muchiile leagă tranzacții din același pas (@{data.pct_same_step|.1f}\\,\\%)")],
         [t("Training / test", "Antrenare / test"), t("steps 1--34 (@{data.n_train|,d} labelled) / steps 35--49 (@{data.n_test|,d} labelled, @{data.n_test_illicit|,d} illicit)",
                                                       "pașii 1--34 (@{data.n_train|,d} etichetate) / pașii 35--49 (@{data.n_test|,d} etichetate, @{data.n_test_illicit|,d} ilicite)")]],
        size="scriptsize") + "\n\\vspace{1mm}\n" + it(
        t("\\textbf{Imbalance}: only @{data.pct_illicit_lab|.2f}\\,\\% of the labelled transactions are illicit; accuracy alone would be meaningless.",
          "\\textbf{Dezechilibrul claselor}: doar @{data.pct_illicit_lab|.2f}\\,\\% dintre tranzacțiile etichetate sînt ilicite; acuratețea singură nu spune nimic"),
        t("\\textbf{Anonymised}: the feature names are hidden by Elliptic; we read them as $f_1,\\dots,f_{165}$.",
          "\\textbf{Date anonimizate}: Elliptic nu publică denumirile variabilelor; le notăm $f_1,\\dots,f_{165}$")) +
        D.bottom(None, NB + "1"), "footnotesize", cite="(Weber et al., 2019)")

    D.chart(t("Labels arrive in waves: illicit activity is not stationary", "Etichetele vin în valuri: activitatea ilicită nu este staționară"),
            "elliptic_timesteps", [
                t("\\textbf{Peak}: @{data.max_illicit_n} illicit transactions in step @{data.max_illicit_step}; about @{data.nodes_per_step_mean|,d} transactions per step in total.",
                  "\\textbf{Vîrful}: @{data.max_illicit_n} " + de("data.max_illicit_n") + "tranzacții ilicite în pasul @{data.max_illicit_step}; în medie, circa @{data.nodes_per_step_mean|,d} " + de("data.nodes_per_step_mean") + "tranzacții pe pas"),
                t("\\textbf{Split in time}: train on steps 1--34, test once on steps 35--49, as in Weber et al.\\ (2019); never shuffle the steps.",
                  "\\textbf{Împărțire în timp}: antrenare pe pașii 1--34, test o singură dată pe pașii 35--49, ca în Weber et al.\\ (2019); pașii nu se amestecă")],
            height="0.55\\textheight", tryit=NB + "1",
            job=t("a model is judged on the months after its training data, exactly as it will be used.",
                  "un model este judecat pe lunile de după datele de antrenare, exact cum va fi folosit."))

    D.chart(t("Most transactions touch one or two others; a few are hubs", "Cele mai multe tranzacții ating una sau două altele; puține sînt noduri centrale"),
            "degree_dist", [
                t("\\textbf{Sparse graph}: mean degree @{data.mean_degree|.2f}, median @{data.median_degree}; @{data.pct_deg1|.1f}\\,\\% of the nodes have one neighbour only.",
                  "\\textbf{Graf rar}: gradul mediu @{data.mean_degree|.2f}, mediana @{data.median_degree}; @{data.pct_deg1|.1f}\\,\\% dintre noduri au un singur vecin"),
                t("\\textbf{Heavy tail}: the largest hub has @{data.max_degree} neighbours; a GNN must not let hubs drown everyone else (hence the normalisation).",
                  "\\textbf{Coadă grea}: cel mai mare nod central are @{data.max_degree} " + de("data.max_degree") + "vecini; o rețea GNN nu trebuie să lase nodurile centrale să le acopere pe celelalte (de aici normalizarea)")],
            height="0.55\\textheight", tryit=NB + "1", cite="$\\deg(v)=|\\mathcal N(v)|$")

    D.chart(t("Illicit transactions sit next to illicit transactions", "Tranzacțiile ilicite stau lîngă alte tranzacții ilicite"), "homophily", [
        t("\\textbf{Homophily}: @{homophily.illicit_nb_of_illicit_pct|.1f}\\,\\% of the labelled neighbours of an illicit node are illicit, against @{homophily.illicit_nb_of_licit_pct|.1f}\\,\\% for a licit node.",
          "\\textbf{Homofilie}: @{homophily.illicit_nb_of_illicit_pct|.1f}\\,\\% dintre vecinii etichetați ai unui nod ilicit sînt ilicite, față de @{homophily.illicit_nb_of_licit_pct|.1f}\\,\\% pentru un nod licit"),
        t("\\textbf{Edge homophily} @{homophily.edge_pct|.1f}\\,\\% of the labelled edges join the same class; with random labels: @{homophily.edge_random_pct|.1f}\\,\\%.",
          "\\textbf{Homofilia muchiilor}: @{homophily.edge_pct|.1f}\\,\\% dintre muchiile etichetate leagă noduri din aceeași clasă; cu etichete aleatoare: @{homophily.edge_random_pct|.1f}\\,\\%")],
        height="0.48\\textheight", tryit=NB + "1",
        job=t("homophily is the reason to look at the neighbours: who you trade with says something about you.",
              "homofilia este motivul pentru care ne uităm la vecini: cu cine tranzacționezi spune ceva despre tine."))

    # ------------------------------------------------------------------ 2 graphs as matrices
    D.section(t("Graphs as matrices", "Grafurile ca matrice"))
    D.concept(t("The adjacency matrix turns a graph into linear algebra", "Matricea de adiacență transformă graful în algebră liniară"),
              t("Every operation of a GNN is a product of the feature matrix with a (normalised) adjacency matrix and a weight matrix.",
                "Orice operație a unei rețele GNN este un produs între matricea variabilelor, o matrice de adiacență (normalizată) și o matrice de ponderi."),
              "\\[ A_{uv}=\\begin{cases}1,&(u,v)\\in\\mathcal E\\\\0,&\\text{" + t("otherwise", "altfel") + "}\\end{cases},\\qquad "
              "D=\\operatorname{diag}\\Big(\\sum_v A_{uv}\\Big) \\]"
              "\\[ \\tilde A=A+I,\\qquad \\hat A=\\tilde D^{-1/2}\\tilde A\\,\\tilde D^{-1/2} \\]"
              "\\where{" + t("$A$ = adjacency, symmetrised; $D$ = degree matrix; $\\tilde A$ adds self-loops (a node keeps its own features); "
                             "$\\hat A$ has entries $1/\\sqrt{\\tilde d_u\\tilde d_v}$.",
                             "$A$ = matricea de adiacență, simetrizată; $D$ = matricea gradelor; $\\tilde A$ adaugă bucle (nodul își păstrează variabilele); "
                             "$\\hat A$ are elementele $1/\\sqrt{\\tilde d_u\\tilde d_v}$.") + "}",
              it((t("\\textbf{Sparse storage}", "\\textbf{Stocare rară}"),
                  t("$A$ has $n^2\\approx 4\\cdot 10^{10}$ cells but only @{data.n_edges|,d} edges", "$A$ are $n^2\\approx 4\\cdot 10^{10}$ celule, dar numai @{data.n_edges|,d} " + de("data.n_edges") + "muchii"),
                  t("libraries keep an edge list $2\\times|\\mathcal E|$, as in PyTorch Geometric (PyG)", "bibliotecile păstrează o listă de muchii $2\\times|\\mathcal E|$, ca PyTorch Geometric (PyG)")),
                 (t("\\textbf{Why normalise}", "\\textbf{De ce normalizăm}"),
                  t("without it a hub of degree @{data.max_degree} would add @{data.max_degree} messages and explode the scale",
                    "fără normalizare, un nod central de grad @{data.max_degree} ar aduna @{data.max_degree} " + de("data.max_degree") + "mesaje, iar scala ar exploda")), sep="3pt"),
              t("a bank's payment network is stored as an edge list, never as a dense matrix.",
                "rețeaua de plăți a unei bănci se stochează ca listă de muchii, niciodată ca matrice densă."),
              tryit=NB + "2", cite="(Kipf \\& Welling, 2017)")

    D.concept(t("A graph model must not depend on how we number the nodes", "Un model pe graf nu trebuie să depindă de numerotarea nodurilor"),
              t("Transaction 17 and transaction 4,203 are names, not positions; relabelling them must not change any prediction.",
                "Tranzacția 17 și tranzacția 4.203 sînt nume, nu poziții; renumerotarea lor nu trebuie să schimbe nicio predicție."),
              "\\[ f\\big(P\\mathbf X,\\;PAP^{\\top}\\big)=P\\,f(\\mathbf X,A)\\quad\\forall P \\]"
              "\\where{" + t("$P$ = any $n\\times n$ permutation matrix; $f$ returns one output row per node: \\textit{equivariance}; "
                             "a graph-level output must be \\textit{invariant}: $g(P\\mathbf X,PAP^{\\top})=g(\\mathbf X,A)$.",
                             "$P$ = orice matrice de permutare $n\\times n$; $f$ dă cîte un rînd de rezultate pentru fiecare nod: \\textit{echivarianță}; "
                             "un rezultat la nivelul întregului graf trebuie să fie \\textit{invariant}: $g(P\\mathbf X,PAP^{\\top})=g(\\mathbf X,A)$.") + "}",
              it((t("\\textbf{Why not a multilayer perceptron (MLP)?}", "\\textbf{De ce nu un perceptron multistrat (MLP)?}"),
                  t("on the rows of $A$ it would need $n$ inputs and change with the numbering", "pe rîndurile lui $A$ ar avea nevoie de $n$ intrări și s-ar schimba odată cu numerotarea")),
                 (t("\\textbf{Why not a convolutional neural network (CNN)?}", "\\textbf{De ce nu o rețea neuronală convoluțională (CNN)?}"),
                  t("it assumes a grid with a fixed number of ordered neighbours", "presupune o grilă cu un număr fix de vecini ordonați"),
                  t("a node has between 1 and @{data.max_degree} neighbours, in no order", "un nod are între 1 și @{data.max_degree} " + de("data.max_degree") + "vecini, fără nicio ordine")), sep="3pt"),
              t("ask of any vendor model: does the score change if the input file is sorted differently? It must not.",
                "întrebați despre orice model cumpărat: se schimbă scorul dacă fișierul de intrare este sortat altfel? Nu are voie."),
              cite="(Bishop \\& Bishop, 2024, sec.\\ 13.1)")

    D.concept(t("Smooth signals on a graph: the Laplacian", "Semnale netede pe graf: laplacianul"),
              t("The graph Laplacian measures how much a node signal differs between neighbours; GCN layers are low-pass filters of it.",
                "Laplacianul grafului măsoară cît diferă un semnal de la un nod la vecinii lui; straturile GCN sînt filtre trece-jos ale acestui semnal."),
              "\\[ L=D-A,\\qquad \\mathbf x^{\\top}L\\,\\mathbf x=\\sum_{(u,v)\\in\\mathcal E}\\big(x_u-x_v\\big)^2\\ \\ge 0 \\]"
              "\\[ L=U\\Lambda U^{\\top},\\qquad \\hat{\\mathbf x}=U^{\\top}\\mathbf x\\ \\ (\\text{" + t("graph Fourier transform", "transformata Fourier pe graf") + "}) \\]"
              "\\where{" + t("$\\mathbf x\\in\\R^n$ = one feature on all nodes; small $\\mathbf x^{\\top}L\\mathbf x$ = neighbours look alike (homophily); "
                             "$U$, $\\Lambda$ = eigenvectors and eigenvalues of $L$ (frequencies).",
                             "$\\mathbf x\\in\\R^n$ = o variabilă pe toate nodurile; $\\mathbf x^{\\top}L\\mathbf x$ mic = vecinii seamănă (homofilie); "
                             "$U$, $\\Lambda$ = vectorii și valorile proprii ale lui $L$ (frecvențele).") + "}",
              it((t("\\textbf{Spectral filter}", "\\textbf{Filtru spectral}"),
                  "$g_\\theta\\star\\mathbf x=U\\,g_\\theta(\\Lambda)\\,U^{\\top}\\mathbf x$",
                  t("costly: $U$ is dense, $O(n^2)$", "costisitor: $U$ este densă, $O(n^2)$")),
                 (t("\\textbf{GCN}", "\\textbf{GCN}"),
                  t("first-order polynomial in $L$ gives $\\hat A$ of the previous slide", "un polinom de gradul întîi în $L$ duce la $\\hat A$ de pe slide-ul anterior"),
                  t("derivation in Appendix A", "derivarea, în Anexa A")), sep="3pt"),
              t("homophilous fraud graphs are smooth signals: averaging over neighbours removes noise, not information.",
                "grafurile de fraudă cu homofilie sînt semnale netede: media pe vecini elimină zgomot, nu informație."),
              cite="(Kipf \\& Welling, 2017; Hamilton, 2020, ch.\\ 7)")

    D.concept(t("Node, edge or graph: three prediction tasks", "Nod, muchie sau graf: trei sarcini de predicție"),
              t("The same message-passing layers serve three kinds of business questions; only the read-out changes.",
                "Aceleași straturi de transmitere a mesajelor răspund la trei tipuri de întrebări; se schimbă doar ieșirea."),
              "\\[ \\hat y_v=\\sigma(\\mathbf w^{\\top}\\mathbf h_v),\\quad \\hat y_{uv}=\\sigma(\\mathbf h_u^{\\top}\\mathbf h_v),\\quad "
              "\\hat y_G=\\sigma\\Big(\\mathbf w^{\\top}\\sum_{v}\\mathbf h_v\\Big) \\]"
              "\\where{" + t("$\\mathbf h_v$ = representation of node $v$ after the last layer; $\\sigma$ = logistic function; the sum makes the graph output permutation-invariant.",
                             "$\\mathbf h_v$ = reprezentarea nodului $v$ după ultimul strat; $\\sigma$ = funcția logistică; suma face ca ieșirea pe graf să nu depindă de numerotare.") + "}",
              it((t("\\textbf{Node}", "\\textbf{Nod}"), t("is this transaction illicit? (this lecture)", "este această tranzacție ilicită? (acest curs)")),
                 (t("\\textbf{Edge}", "\\textbf{Muchie}"), t("will these two firms trade; is this payment a mule transfer?", "vor face aceste două firme comerț; este această plată un transfer prin intermediari?")),
                 (t("\\textbf{Graph}", "\\textbf{Graf}"), t("is this cluster of accounts a laundering ring; is this interbank network fragile?",
                                                            "este acest grup de conturi o rețea de spălare; este această rețea interbancară fragilă?")), sep="3pt"),
              t("define the unit of decision first (a payment, a client, a ring), then the graph task.",
                "stabiliți întîi unitatea de decizie (o plată, un client, o rețea), apoi sarcina pe graf."),
              cite="(Hamilton, 2020, ch.\\ 1)")

    # ------------------------------------------------------------------ 3 message passing
    D.section(t("Message passing", "Transmiterea mesajelor"))
    tik = ("\\begin{center}\\begin{tikzpicture}[scale=0.9, every node/.style={circle, draw=MainBlue, thick, minimum size=6mm, inner sep=0pt, font=\\scriptsize}]\n"
           "\\node[fill=Ser2!35] (v) at (0,0) {$v$};\n\\node (a) at (-1.6,0.9) {$u_1$};\n\\node[fill=Ser8!30] (b) at (-1.7,-0.8) {$u_2$};\n"
           "\\node (c) at (1.6,0.9) {$u_3$};\n\\node (d) at (2.9,0.2) {$w$};\n"
           "\\draw[-{Stealth}, thick, Ser1] (a) -- node[draw=none, above, font=\\tiny] {$\\mathbf m_{u_1v}$} (v);\n"
           "\\draw[-{Stealth}, thick, Ser1] (b) -- (v);\n\\draw[-{Stealth}, thick, Ser1] (c) -- (v);\n"
           "\\draw[thick, FooterGray] (c) -- (d);\n\\end{tikzpicture}\\end{center}")
    D.concept(t("A GNN layer: collect messages, aggregate, update", "Un strat GNN: colectează mesajele, le agregă, actualizează"),
              t("Each node sends a message to its neighbours; each node sums what it receives and updates its own state.",
                "Fiecare nod trimite un mesaj vecinilor; fiecare nod adună ce primește și își actualizează starea."),
              "\\[ \\mathbf m_v^{(l)}=\\bigoplus_{u\\in\\mathcal N(v)}\\phi^{(l)}\\big(\\mathbf h_u^{(l)},\\mathbf h_v^{(l)},\\mathbf e_{uv}\\big) \\]"
              "\\[ \\mathbf h_v^{(l+1)}=\\psi^{(l)}\\big(\\mathbf h_v^{(l)},\\mathbf m_v^{(l)}\\big),\\qquad \\mathbf h_v^{(0)}=\\mathbf x_v \\]"
              "\\where{" + t("$\\mathcal N(v)$ = neighbours of $v$; $\\phi$ = message function; $\\bigoplus$ = permutation-invariant aggregator (sum, mean, max); "
                             "$\\psi$ = update (a small network); $\\mathbf e_{uv}$ = edge features, if any.",
                             "$\\mathcal N(v)$ = vecinii lui $v$; $\\phi$ = funcția mesajului; $\\bigoplus$ = agregator care nu depinde de ordine (sumă, medie, maxim); "
                             "$\\psi$ = actualizarea (o rețea mică); $\\mathbf e_{uv}$ = variabilele muchiei, dacă există.") + "}",
              tik + it(t("\\textbf{Message-passing neural network (MPNN)}: every layer of this lecture is a choice of $\\phi$, $\\bigoplus$, $\\psi$.",
                         "\\textbf{Rețeaua neuronală cu transmitere de mesaje (MPNN)}: fiecare strat din acest curs este o alegere a lui $\\phi$, $\\bigoplus$, $\\psi$"), sep="2pt"),
              t("the weights $\\phi,\\psi$ are shared by all nodes: the model size does not grow with the graph.",
                "ponderile $\\phi,\\psi$ sînt comune tuturor nodurilor: dimensiunea modelului nu crește odată cu graful."),
              tryit=NB + "3", cite="(Gilmer et al., 2017; Bishop \\& Bishop, 2024, sec.\\ 13.2)", wl="0.56", wr="0.41")

    D.concept(t("The aggregator decides what a GNN can tell apart", "Agregatorul decide ce poate deosebi o rețea GNN"),
              t("Mean and max forget how many neighbours sent a message; the sum keeps it, which makes it as powerful as the Weisfeiler--Leman test.",
                "Media și maximul uită cîți vecini au trimis mesaje; suma păstrează această informație și devine la fel de puternică precum testul Weisfeiler--Leman."),
              "\\[ \\mathbf h_v^{(l+1)}=\\text{MLP}^{(l)}\\Big((1+\\varepsilon)\\,\\mathbf h_v^{(l)}+\\sum_{u\\in\\mathcal N(v)}\\mathbf h_u^{(l)}\\Big) \\]"
              "\\where{" + t("graph isomorphism network (GIN) layer; $\\varepsilon$ = learnable scalar; a sum over a multiset is injective if the MLP is rich enough, "
                             "so GIN separates any two neighbourhoods that the Weisfeiler--Leman (WL) test separates.",
                             "stratul rețelei de izomorfism pe graf (GIN); $\\varepsilon$ = scalar estimat; suma pe un multiset este injectivă dacă MLP-ul este suficient de bogat, "
                             "deci GIN deosebește orice vecinătăți pe care le deosebește testul Weisfeiler--Leman (WL).") + "}",
              it((t("\\textbf{Example}", "\\textbf{Exemplu}"),
                  t("neighbours $\\{a,a\\}$ and $\\{a\\}$: same mean, same max, different sum", "vecinii $\\{a,a\\}$ și $\\{a\\}$: aceeași medie, același maxim, sumă diferită")),
                 (t("\\textbf{In fraud}", "\\textbf{În fraudă}"),
                  t("``one payment from a mixer'' and ``ten payments from mixers'' should not look the same",
                    "``o plată de la un serviciu de amestecare'' și ``zece astfel de plăți'' nu trebuie să arate la fel"),
                  t("but sums of hubs need normalisation, which is why GCN uses $\\hat A$", "dar sumele nodurilor centrale cer normalizare; de aceea GCN folosește $\\hat A$")), sep="3pt"),
              t("choose the aggregator for the question: counts matter (sum), proportions matter (mean), one red flag matters (max).",
                "alegeți agregatorul după întrebare: contează numărul (suma), contează proporția (media), contează un singur semnal de alarmă (maximul)."),
              cite="(Xu et al., 2019)")

    D.chart(t("Each layer widens the neighbourhood that reaches a node", "Fiecare strat lărgește vecinătatea care ajunge la un nod"), "receptive_field", [
        t("\\textbf{Receptive field} of a test transaction: median @{receptive.k1_median} nodes after 1 layer, @{receptive.k2_median} after 2, @{receptive.k4_median} after 4.",
          "\\textbf{Cîmpul receptiv} al unei tranzacții de test: mediana este @{receptive.k1_median} noduri după 1 strat, @{receptive.k2_median} după 2, @{receptive.k4_median} după 4"),
        t("\\textbf{Tail}: the 90th percentile reaches @{receptive.k4_p90|,d} nodes after 4 layers; hubs connect distant parts of the graph.",
          "\\textbf{Coada}: percentila 90 ajunge la @{receptive.k4_p90|,d} " + de("receptive.k4_p90") + "noduri după 4 straturi; nodurile centrale leagă părți îndepărtate ale grafului")],
        height="0.52\\textheight", tryit=NB + "5", cite="(own computation, 2,000 test nodes)" if not ro else "(calcule proprii, 2.000 de noduri de test)",
        job=t("$L$ layers = $L$ hops of money flow; two hops already cover most of what an investigator would check by hand.",
              "$L$ straturi = $L$ salturi ale fluxului de bani; două salturi acoperă deja mare parte din ce ar verifica manual un investigator."))

    D.chart(t("Too many layers make all nodes look alike: over-smoothing", "Prea multe straturi fac toate nodurile să semene: suprasmoothing-ul (over-smoothing)"), "oversmoothing", [
        t("\\textbf{Accuracy}: PR-AUC @{depth.ap_1|.3f} with 1 GCN layer, @{depth.ap_2|.3f} with 2, @{depth.ap_6|.3f} with 6; best depth @{depth.best_layers}.",
          "\\textbf{Precizia}: PR-AUC @{depth.ap_1|.3f} cu 1 strat GCN, @{depth.ap_2|.3f} cu 2, @{depth.ap_6|.3f} cu 6; adîncimea cea mai bună: @{depth.best_layers}"),
        t("\\textbf{Similarity}: mean cosine similarity of node representations rises from @{depth.cos_1|.2f} (1 layer) to @{depth.cos_6|.2f} (6 layers); repeated averaging drives them to a common vector.",
          "\\textbf{Similaritatea}: similaritatea cosinus medie a reprezentărilor crește de la @{depth.cos_1|.2f} (1 strat) la @{depth.cos_6|.2f} (6 straturi); medierea repetată le aduce spre un vector comun")],
        height="0.52\\textheight", tryit=NB + "5", cite=t("(own computation, 2 seeds; Li, Han \\& Wu, 2018)", "(calcule proprii, 2 semințe; Li, Han \\& Wu, 2018)"),
        job=t("GNNs are shallow by design: two or three layers, then residual links or jumping knowledge if more is needed.",
              "rețelele GNN sînt intenționat puțin adînci: două-trei straturi, apoi conexiuni reziduale dacă este nevoie de mai mult."))

    # ------------------------------------------------------------------ 4 three classic layers
    D.section(t("GCN, GraphSAGE and GAT", "GCN, GraphSAGE și GAT"))
    D.concept(t("GCN: average the neighbours with degree weights, then regress", "GCN: media vecinilor ponderată cu gradele, apoi o regresie"),
              t("A GCN layer is a linear regression on the normalised average of a node and its neighbours, followed by a rectified linear unit (ReLU).",
                "Un strat GCN este o regresie liniară pe media normalizată a nodului și a vecinilor lui, urmată de o unitate liniară rectificată (ReLU)."),
              "\\[ H^{(l+1)}=\\ReLU\\big(\\hat A\\,H^{(l)}W^{(l)}\\big) \\]\\[ "
              "\\mathbf h_v^{(l+1)}=\\ReLU\\Big(\\sum_{u\\in\\mathcal N(v)\\cup\\{v\\}}\\frac{W^{(l)\\top}\\mathbf h_u^{(l)}}{\\sqrt{\\tilde d_u\\tilde d_v}}\\Big) \\]"
              "\\where{" + t("$H^{(l)}\\in\\R^{n\\times d_l}$ = node representations, $H^{(0)}=\\mathbf X$; $W^{(l)}$ = weights shared by all nodes; "
                             "$\\ReLU(x)=\\max(0,x)$; output: $\\softmax(\\hat A H^{(1)}W^{(2)})$.",
                             "$H^{(l)}\\in\\R^{n\\times d_l}$ = reprezentările nodurilor, $H^{(0)}=\\mathbf X$; $W^{(l)}$ = ponderi comune tuturor nodurilor; "
                             "$\\ReLU(x)=\\max(0,x)$; ieșirea: $\\softmax(\\hat A H^{(1)}W^{(2)})$.") + "}",
              it((t("\\textbf{Our GCN}", "\\textbf{Rețeaua GCN folosită}"),
                  t("2 layers, 100 hidden units, dropout 0.2: @{gcn_params|,d} parameters", "2 straturi, 100 de unități ascunse, dropout 0,2: @{gcn_params|,d} " + de("gcn_params") + "parametri"),
                  t("full-batch training on the whole graph, @{models.gcn.sec|.0f} s per seed on the processor (CPU)",
                    "antrenare pe tot graful odată, @{models.gcn.sec|.0f} s pe sămînță, pe unitatea centrală de procesare (CPU)")),
                 (t("\\textbf{Cost}", "\\textbf{Costul}"), "$O(|\\mathcal E|\\,d_l+n\\,d_l d_{l+1})$ " + t("per layer: linear in the edges", "pe strat: liniar în numărul de muchii")), sep="3pt"),
              t("a GCN is the graph analogue of a moving average: robust, cheap, and blind to which neighbour matters most.",
                "o rețea GCN este analogul pe graf al unei medii mobile: robustă, ieftină și oarbă la care vecin contează cel mai mult."),
              tryit=NB + "3", cite="(Kipf \\& Welling, 2017)")

    D.concept(t("GraphSAGE: keep yourself apart from your neighbours, and sample them", "GraphSAGE: separă nodul de vecini și eșantionează vecinii"),
              t("GraphSAGE concatenates a node's own state with the mean of a sample of its neighbours, so it also works on nodes never seen in training.",
                "GraphSAGE alătură starea proprie a nodului și media unui eșantion de vecini, deci funcționează și pe noduri nevăzute la antrenare."),
              "\\[ \\mathbf h_v^{(l+1)}=\\ReLU\\Big(W_1^{(l)}\\mathbf h_v^{(l)}+W_2^{(l)}\\frac{1}{|\\mathcal S(v)|}\\sum_{u\\in\\mathcal S(v)}\\mathbf h_u^{(l)}\\Big) \\]"
              "\\where{" + t("$\\mathcal S(v)\\subseteq\\mathcal N(v)$ = a random sample of at most $k$ neighbours (all of them in our full-batch run); "
                             "$W_1$, $W_2$ = separate weights for the node and for its neighbourhood.",
                             "$\\mathcal S(v)\\subseteq\\mathcal N(v)$ = un eșantion aleator de cel mult $k$ vecini (toți, în rularea noastră pe tot graful); "
                             "$W_1$, $W_2$ = ponderi separate pentru nod și pentru vecinătatea lui.") + "}",
              it((t("\\textbf{Inductive}", "\\textbf{Inductiv}"),
                  t("the weights do not depend on the graph: tomorrow's transactions get a score without retraining",
                    "ponderile nu depind de graf: tranzacțiile de mîine primesc un scor fără reantrenare")),
                 (t("\\textbf{Scalable}", "\\textbf{Scalabil}"),
                  t("mini-batches of $k$-hop samples: cost bounded by $k^L$ per node, not by the hub degree",
                    "mini-loturi de eșantioane pe $k$ salturi: costul este limitat la $k^L$ pe nod, nu de gradul nodurilor centrale")),
                 (t("\\textbf{On Elliptic}", "\\textbf{Pe Elliptic}"),
                  t("illicit F1 @{models.sage.f1|.3f} (sd @{models.sage.f1_sd|.3f}), PR-AUC @{models.sage.ap|.3f}",
                    "F1 pe clasa ilicită @{models.sage.f1|.3f} (abatere standard @{models.sage.f1_sd|.3f}), PR-AUC @{models.sage.ap|.3f}")), sep="3pt"),
              t("payment systems that score transactions in real time need inductive layers such as GraphSAGE.",
                "sistemele de plăți care dau scoruri în timp real au nevoie de straturi inductive precum GraphSAGE."),
              tryit=NB + "4", cite="(Hamilton, Ying \\& Leskovec, 2017)")

    D.concept(t("GAT: let the network decide which neighbour matters", "GAT: rețeaua decide care vecin contează"),
              t("A graph attention layer weights each neighbour by a learned score: the attention of Lecture 7, restricted to the edges of the graph.",
                "Un strat GAT ponderează fiecare vecin cu un scor învățat: atenția din cursul 7, restrînsă la muchii."),
              "\\[ \\alpha_{vu}=\\frac{\\exp\\big(\\mathrm{LeakyReLU}(\\mathbf a^{\\top}[W\\mathbf h_v\\,\\|\\,W\\mathbf h_u])\\big)}"
              "{\\sum_{k\\in\\mathcal N(v)\\cup\\{v\\}}\\exp\\big(\\mathrm{LeakyReLU}(\\mathbf a^{\\top}[W\\mathbf h_v\\,\\|\\,W\\mathbf h_k])\\big)} \\]"
              "\\[ \\mathbf h_v^{(l+1)}=\\big\\|_{k=1}^{K}\\ \\ReLU\\Big(\\sum_{u}\\alpha_{vu}^{k}W^{k}\\mathbf h_u\\Big) \\]"
              "\\where{" + t("$\\mathbf a$ = attention vector; $\\|$ = concatenation; $K$ = number of heads (4 here); weights $\\alpha_{vu}$ sum to one over the neighbours of $v$.",
                             "$\\mathbf a$ = vectorul de atenție; $\\|$ = concatenare; $K$ = numărul de capete (aici 4); ponderile $\\alpha_{vu}$ însumează unu pe vecinii lui $v$.") + "}",
              it((t("\\textbf{Link with transformers}", "\\textbf{Legătura cu transformerele}"),
                  t("a transformer is a GAT on the complete graph of the tokens", "un transformer este o rețea GAT pe graful complet al tokenilor")),
                 (t("\\textbf{On Elliptic}", "\\textbf{Pe Elliptic}"),
                  t("@{gat_params|,d} parameters; illicit F1 @{models.gat.f1|.3f}, PR-AUC @{models.gat.ap|.3f}",
                    "@{gat_params|,d} " + de("gat_params") + "parametri; F1 pe clasa ilicită @{models.gat.f1|.3f}, PR-AUC @{models.gat.ap|.3f}")), sep="3pt"),
              t("attention weights are a first, not a final, explanation: they show where the model looked, not why.",
                "ponderile de atenție sînt o primă explicație, nu una finală: arată unde s-a uitat modelul, nu de ce."),
              tryit=NB + "4", cite="(Veličković et al., 2018)", wl="0.56", wr="0.41")

    D.concept(t("Training on one big graph: masks, class weights, early stopping", "Antrenarea pe un singur graf mare: măști, ponderi de clasă, oprire timpurie"),
              t("All nodes pass messages, but only labelled training nodes enter the loss; validation steps decide when to stop.",
                "Toate nodurile transmit mesaje, dar în funcția de pierdere intră doar nodurile etichetate de antrenare; pașii de validare decid oprirea."),
              "\\[ \\loss(\\btheta)=-\\frac{1}{|\\mathcal V_{tr}|}\\sum_{v\\in\\mathcal V_{tr}}\\big[w_1\\,y_v\\log\\hat p_v+w_0(1-y_v)\\log(1-\\hat p_v)\\big] \\]"
              "\\where{" + t("$\\mathcal V_{tr}$ = labelled nodes of steps 1--29; $\\hat p_v$ = predicted probability of illicit; "
                             "$w_1=0.7$, $w_0=0.3$ = class weights of Weber et al.\\ (2019); Adam, learning rate 0.003, weight decay $5\\cdot10^{-4}$.",
                             "$\\mathcal V_{tr}$ = nodurile etichetate din pașii 1--29; $\\hat p_v$ = probabilitatea estimată de ilicit; "
                             "$w_1=0{,}7$, $w_0=0{,}3$ = ponderile de clasă din Weber et al.\\ (2019); Adam, rata de învățare 0,003, penalizarea ponderilor $5\\cdot10^{-4}$.") + "}",
              it((t("\\textbf{Validation}", "\\textbf{Validarea}"),
                  t("labelled nodes of steps 30--34: best epoch by PR-AUC, patience 40", "nodurile etichetate din pașii 30--34: epoca cea mai bună după PR-AUC, răbdare 40 de epoci"),
                  t("median best epoch of the GCN: @{models.gcn.best_epoch_median}", "epoca mediană reținută pentru GCN: @{models.gcn.best_epoch_median}")),
                 (t("\\textbf{Leakage-free scaling}", "\\textbf{Standardizare fără scurgeri de informație}"),
                  t("feature means and standard deviations from steps 1--34 only", "mediile și abaterile standard ale variabilelor numai din pașii 1--34")), sep="3pt"),
              t("the unknown nodes are not wasted: they carry messages between labelled ones (semi-supervised learning).",
                "nodurile necunoscute nu se pierd: ele transmit mesaje între nodurile etichetate (învățare semi-supervizată)."),
              tryit=NB + "3", cite="(Kipf \\& Welling, 2017; Weber et al., 2019)")

    # ------------------------------------------------------------------ 5 case study
    D.section(t("Case study: illicit Bitcoin transactions", "Studiu de caz: tranzacții Bitcoin ilicite"))
    D.frame(t("Eight models, one temporal test", "Opt modele, un singur test în timp"), table(
        "@{}p{0.27\\textwidth}p{0.66\\textwidth}@{}",
        [t("Model", "Model"), t("What it sees", "Ce vede")],
        [[t("Logistic regression", "Regresie logistică"), t("local features only (@{data.n_local}), then all @{data.n_features}", "doar variabilele locale (@{data.n_local}), apoi toate cele @{data.n_features}")],
         [t("Random forest", "Pădure aleatoare"), t("all features; 100 trees, 50 features per split (Weber et al., 2019)", "toate variabilele; 100 de arbori, 50 de variabile la fiecare împărțire (Weber et al., 2019)")],
         [t("MLP (no graph)", "MLP (fără graf)"), t("all features, same layers as the GCN but no messages: isolates the value of the edges", "toate variabilele, aceleași straturi ca GCN, dar fără mesaje: izolează valoarea muchiilor")],
         ["GCN, GraphSAGE, GAT", t("all features + the graph; 2 layers, 100 units", "toate variabilele și graful; 2 straturi, 100 de unități")],
         [t("Random forest + GCN", "Pădure + GCN"), t("all features + the 100 hidden GCN outputs of each node", "toate variabilele și cele 100 de ieșiri ascunse GCN ale fiecărui nod")]],
        size="scriptsize") + "\n\\vspace{1mm}\n" + it(
        t("\\textbf{Metrics on steps 35--49}: illicit-class precision, recall and F1 at threshold 0.5; PR-AUC (threshold-free).",
          "\\textbf{Măsuri pe pașii 35--49}: precizia, recall-ul și F1 pe clasa ilicită, la pragul 0,5; PR-AUC (fără prag)"),
        t("\\textbf{Uncertainty}: @{models.gcn.n_runs} seeds per network and per forest; bootstrap of the test nodes for the differences.",
          "\\textbf{Incertitudinea}: @{models.gcn.n_runs} semințe pentru fiecare rețea și pădure; bootstrap pe nodurile de test pentru diferențe")) +
        D.bottom(None, NB + "2"), "footnotesize", cite="(Weber et al., 2019)")

    D.chart(t("On Elliptic the forest leads; graph layers mainly add ranking power to a network",
              "Pe Elliptic pădurea conduce; straturile pe graf adaugă mai ales putere de ordonare unei rețele"), "model_comparison", [
        t("\\textbf{Illicit F1}: random forest @{models.rf.f1|.3f}, forest + GCN @{models.rf_emb.f1|.3f}, GCN @{models.gcn.f1|.3f}, MLP @{models.mlp.f1|.3f}, logistic @{models.lr_all.f1|.3f}.",
          "\\textbf{F1 pe clasa ilicită}: pădurea @{models.rf.f1|.3f}, pădurea + GCN @{models.rf_emb.f1|.3f}, GCN @{models.gcn.f1|.3f}, MLP @{models.mlp.f1|.3f}, logistic @{models.lr_all.f1|.3f}"),
        t("\\textbf{Value of the edges}: PR-AUC of the GCN minus the MLP, same layers: @{gcn_vs_mlp_ap_pts|+.1f} points.",
          "\\textbf{Valoarea muchiilor}: PR-AUC GCN minus MLP, cu aceleași straturi: @{gcn_vs_mlp_ap_pts|+.1f} puncte")],
        height="0.56\\textheight", tryit=NB + "2", cite=t("(own computation; mean $\\pm$ sd over 3 seeds)", "(calcule proprii; media $\\pm$ abaterea standard pe 3 semințe)"))

    rows = []
    names = {"lr_local": t("Logistic, local", "Logistic, locale"), "lr_all": t("Logistic, all", "Logistic, toate"), "mlp": "MLP",
             "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "rf": t("Random forest", "Pădure aleatoare"), "rf_emb": t("Forest + GCN", "Pădure + GCN")}
    keys = ["lr_local", "lr_all", "mlp", "gcn", "sage", "gat", "rf", "rf_emb"]
    lo_f, hi_f = min(M[k]["f1"] for k in keys), max(M[k]["f1"] for k in keys)
    lo_a, hi_a = min(M[k]["ap"] for k in keys), max(M[k]["ap"] for k in keys)
    for k in keys:
        rows.append([names[k], n_(M[k]["prec"], ".3f"), n_(M[k]["rec"], ".3f"),
                     heat(M[k]["f1"], lo_f, hi_f, good_low=False, shown=n_(M[k]["f1"], ".3f")),
                     heat(M[k]["ap"], lo_a, hi_a, good_low=False, shown=n_(M[k]["ap"], ".3f")),
                     n_(M[k]["sec"], ".1f")])
    D.frame(t("Precision, recall and the cost of training", "Precizie, recall și costul antrenării"), cols(
        table("@{}lrrrrr@{}", [t("Model", "Model"), t("Prec.", "Prec."), t("Recall", "Recall"), "F1", "PR-AUC", t("Sec.", "Sec.")], rows, size="scriptsize"),
        it((t("\\textbf{Reading the table}", "\\textbf{Cum citim tabelul}"),
            t("precision = share of alerts that are illicit; recall = share of illicit found", "precizia = ponderea alertelor care sînt ilicite; recall = ponderea ilicitelor găsite"),
            t("green = best, red = worst in the column", "verde = cel mai bun, roșu = cel mai slab din coloană")),
           (t("\\textbf{Forest}", "\\textbf{Pădurea}"),
            t("precision @{models.rf.prec|.3f}, recall @{models.rf.rec|.3f}", "precizie @{models.rf.prec|.3f}, recall @{models.rf.rec|.3f}")),
           (t("\\textbf{Logistic, class weights 0.3/0.7}", "\\textbf{Logistic, ponderi de clasă 0,3/0,7}"),
            t("precision @{models.lr_all.prec|.3f}, recall @{models.lr_all.rec|.3f}: the weights buy recall with false alarms", "precizie @{models.lr_all.prec|.3f}, recall @{models.lr_all.rec|.3f}: ponderile cumpără recall cu alarme false")), sep="3pt"),
        "0.56", "0.41") + D.bottom(t("the right model depends on what an error costs: a missed laundering case or an analyst's wasted hour.",
                                     "modelul potrivit depinde de costul unei erori: un caz de spălare ratat sau o oră pierdută de un analist."), NB + "2"),
        "footnotesize", cite=t("(own computation; sec.\\ = training time per seed, CPU)", "(calcule proprii; sec.\\ = timpul de antrenare pe sămînță, CPU)"))

    D.chart(t("Precision-recall curves: the ranking behind the F1", "Curbele precizie-recall: ordonarea din spatele lui F1"), "pr_curves", [
        t("\\textbf{PR-AUC}: random forest @{models.rf.ap|.3f}, forest + GCN @{models.rf_emb.ap|.3f}, GCN @{models.gcn.ap|.3f}, logistic @{models.lr_all.ap|.3f}; a random score would give @{data.pct_test_illicit|.2f}\\,\\% (dotted line).",
          "\\textbf{PR-AUC}: pădurea @{models.rf.ap|.3f}, pădurea + GCN @{models.rf_emb.ap|.3f}, GCN @{models.gcn.ap|.3f}, logistic @{models.lr_all.ap|.3f}; un scor aleator ar da @{data.pct_test_illicit|.2f}\\,\\% (linia punctată)")],
        height="0.58\\textheight", tryit=NB + "2", cite="$\\text{PR-AUC}=\\sum_k (R_k-R_{k-1})P_k$")

    D.concept(t("Is the gap real? A bootstrap of the test transactions", "Este diferența reală? Un bootstrap al tranzacțiilor de test"),
              t("One test set gives one number; resampling the test nodes shows how much that number could move.",
                "Un singur set de test dă un singur număr; reeșantionarea nodurilor de test arată cît s-ar putea mișca acest număr."),
              "\\[ \\Delta^{*b}=\\text{PR-AUC}_A\\big(\\mathcal T^{*b}\\big)-\\text{PR-AUC}_B\\big(\\mathcal T^{*b}\\big),\\quad b=1,\\dots,B \\]"
              "\\[ \\text{CI}_{95\\%}=\\big[q_{0.025}(\\Delta^{*}),\\ q_{0.975}(\\Delta^{*})\\big] \\]"
              "\\where{" + t("$\\mathcal T^{*b}$ = test nodes drawn with replacement; $B=1{,}000$; scores = mean over 3 seeds; the same draw is used for both models (paired).",
                             "$\\mathcal T^{*b}$ = noduri de test extrase cu întoarcere; $B=1.000$; scorurile = media pe 3 semințe; aceeași extragere pentru ambele modele (comparație pereche).") + "}",
              it((t("\\textbf{Forest minus GCN}", "\\textbf{Pădurea minus GCN}"),
                  t("$\\Delta$ = @{boot.rf_minus_gcn_ap|.3f}, 95\\,\\% interval [@{boot.lo|.3f}; @{boot.hi|.3f}]", "$\\Delta$ = @{boot.rf_minus_gcn_ap|.3f}, intervalul de 95\\,\\% [@{boot.lo|.3f}; @{boot.hi|.3f}]")),
                 (t("\\textbf{Forest + GCN minus forest}", "\\textbf{Pădurea + GCN minus pădurea}"),
                  t("$\\Delta$ = @{boot.rfemb_minus_rf_ap|.3f}, 95\\,\\% interval [@{boot.lo2|.3f}; @{boot.hi2|.3f}]", "$\\Delta$ = @{boot.rfemb_minus_rf_ap|.3f}, intervalul de 95\\,\\% [@{boot.lo2|.3f}; @{boot.hi2|.3f}]")),
                 t("an interval that excludes 0 supports a real difference on this test period", "un interval care nu conține 0 susține o diferență reală pe această perioadă de test"), sep="3pt"),
              t("report the interval, not only the winner: a fraud committee decides on ranges, not on the third decimal.",
                "raportați intervalul, nu doar cîștigătorul: un comitet antifraudă decide pe intervale, nu pe a treia zecimală."),
              tryit=NB + "2", cite="(Efron \\& Tibshirani, 1993)")

    D.frame(t("Why the forest is hard to beat here: its features already pass messages", "De ce pădurea este greu de bătut aici: variabilele ei transmit deja mesaje"), cols(
        it((t("\\textbf{Hand-made message passing}", "\\textbf{Transmitere de mesaje construită manual}"),
            t("@{data.n_agg} of the @{data.n_features} features are aggregates over the one-hop neighbours (max, min, sd, correlation)",
              "@{data.n_agg} dintre cele @{data.n_features} de variabile sînt agregate pe vecinii de la un salt (maxim, minim, abatere standard, corelație)"),
            t("this is one fixed GNN layer, computed by Elliptic before release", "este un strat GNN fix, calculat de Elliptic înainte de publicarea datelor")),
           (t("\\textbf{Local features only}", "\\textbf{Doar variabilele locale}"),
            t("logistic regression F1 @{models.lr_local.f1|.3f} vs @{models.lr_all.f1|.3f} with the aggregates", "regresia logistică are F1 @{models.lr_local.f1|.3f}, față de @{models.lr_all.f1|.3f} cu agregatele")),
           (t("\\textbf{Trees}", "\\textbf{Arborii}"),
            t("thresholds on amounts and counts fit fraud patterns that are rules (``more than $k$ inputs'')", "pragurile pe sume și numere se potrivesc tiparelor de fraudă care sînt reguli (``mai mult de $k$ intrări'')")), sep="4pt"),
        it((t("\\textbf{Where GNNs earn their keep}", "\\textbf{Unde își justifică rețelele GNN costul}"),
            t("raw graphs without engineered aggregates", "grafuri brute, fără agregate construite manual"),
            t("multi-hop patterns (layering through many accounts)", "tipare pe mai multe salturi (stratificarea prin multe conturi)"),
            t("inductive scoring of new nodes, and embeddings for other models", "scoruri inductive pentru noduri noi și reprezentări pentru alte modele")),
           (t("\\textbf{Combine}", "\\textbf{Combinați}"),
            t("forest + GCN representations: F1 change @{rfemb_vs_rf_f1_pts|+.1f} points", "pădure + reprezentări GCN: variația lui F1 este @{rfemb_vs_rf_f1_pts|+.1f} puncte")), sep="4pt"), "0.49", "0.48") +
        D.bottom(t("before training a GNN, check whether the data provider has already aggregated the neighbourhood for you.",
                   "înainte de a antrena o rețea GNN, verificați dacă furnizorul datelor nu a agregat deja vecinătatea."), NB + "2"),
        "footnotesize", cite="(Weber et al., 2019)")

    D.chart(t("A dark-market shutdown breaks every model: concept drift", "Închiderea unei piețe ilegale strică toate modelele: schimbarea conceptului"), "f1_by_timestep", [
        t("\\textbf{Before step 43}: mean F1 random forest @{shutdown.rf_f1_pre|.3f}, GCN @{shutdown.gcn_f1_pre|.3f}; \\textbf{from step 43}: @{shutdown.rf_f1_post|.3f} and @{shutdown.gcn_f1_post|.3f}.",
          "\\textbf{Înainte de pasul 43}: F1 mediu pădure @{shutdown.rf_f1_pre|.3f}, GCN @{shutdown.gcn_f1_pre|.3f}; \\textbf{de la pasul 43}: @{shutdown.rf_f1_post|.3f} și @{shutdown.gcn_f1_post|.3f}"),
        t("\\textbf{New behaviour}: after a major dark market closed, illicit flows changed form; no model trained on steps 1--34 had seen it (Weber et al., 2019).",
          "\\textbf{Comportament nou}: după închiderea unei piețe ilegale importante, fluxurile ilicite și-au schimbat forma; niciun model antrenat pe pașii 1--34 nu o văzuse (Weber et al., 2019)")],
        height="0.52\\textheight", tryit=NB + "6",
        job=t("monitor F1 per period and retrain on recent labels; a static AML model ages within weeks.",
              "monitorizați F1 pe fiecare perioadă și reantrenați pe etichete recente; un model AML static îmbătrînește în cîteva săptămîni."))

    D.chart(t("What an investigation team gets from 1,000 alerts", "Ce obține o echipă de investigații din 1\\,000 de alerte"), "alert_budget", [
        t("\\textbf{Top @{alerts.k|,d} scores} of the test period contain @{alerts.rf|.1f}\\,\\% of all illicit transactions for the random forest, @{alerts.gcn|.1f}\\,\\% for the GCN, @{alerts.rf_emb|.1f}\\,\\% for forest + GCN.",
          "\\textbf{Primele @{alerts.k|,d} de scoruri} din perioada de test conțin @{alerts.rf|.1f}\\,\\% dintre toate tranzacțiile ilicite pentru pădure, @{alerts.gcn|.1f}\\,\\% pentru GCN și @{alerts.rf_emb|.1f}\\,\\% pentru pădure + GCN"),
        t("\\textbf{Hit rate} of those alerts: @{alerts.rf_prec|.1f}\\,\\% (forest), @{alerts.gcn_prec|.1f}\\,\\% (GCN), out of @{alerts.n_test_illicit|,d} illicit test transactions.",
          "\\textbf{Rata de reușită} a acestor alerte: @{alerts.rf_prec|.1f}\\,\\% (pădurea), @{alerts.gcn_prec|.1f}\\,\\% (GCN), din @{alerts.n_test_illicit|,d} de tranzacții ilicite de test")],
        height="0.52\\textheight", tryit=NB + "6",
        job=t("size the team from this curve: the alert budget, not the F1, is the constraint in a compliance department.",
              "dimensionați echipa după această curbă: bugetul de alerte, nu F1, este constrîngerea unui departament de conformitate."))

    D.chart(t("The GCN learns a space where illicit transactions gather", "Rețeaua GCN învață un spațiu în care tranzacțiile ilicite se adună"), "gcn_embeddings", [
        t("\\textbf{Hidden layer} of the GCN (@{emb.dim} dimensions) projected by principal component analysis (PCA) on its first two components: @{emb.var2_pct|.1f}\\,\\% of the variance.",
          "\\textbf{Stratul ascuns} al rețelei GCN (@{emb.dim} de dimensiuni), proiectat prin analiza componentelor principale (PCA) pe primele două componente: @{emb.var2_pct|.1f}\\,\\% din varianță"),
        t("\\textbf{Reuse}: these vectors are the inputs of the forest + GCN model and can feed any other risk model.",
          "\\textbf{Refolosire}: acești vectori sînt intrările modelului pădure + GCN și pot alimenta orice alt model de risc")],
        height="0.55\\textheight", tryit=NB + "3", cite=t("(own computation; @{emb.n_illicit_shown|,d} illicit and 3,000 licit test nodes)",
                                                         "(calcule proprii; @{emb.n_illicit_shown|,d} de noduri ilicite și 3.000 de noduri licite de test)"))

    D.chart(t("Adding graph features reshuffles the explanations", "Adăugarea variabilelor de graf reordonează explicațiile"), "importance_shift", [
        t("\\textbf{Importance mass}: the 10 top features of the forest carry @{stability.top10_share_rf_pct|.1f}\\,\\% of the importance; with GCN embeddings they keep @{stability.top10_share_rfemb_pct|.1f}\\,\\%, the embeddings take @{stability.emb_share_pct|.1f}\\,\\%.",
          "\\textbf{Masa importanței}: primele 10 variabile ale pădurii au @{stability.top10_share_rf_pct|.1f}\\,\\% din importanță; cu reprezentările GCN păstrează @{stability.top10_share_rfemb_pct|.1f}\\,\\%, reprezentările iau @{stability.emb_share_pct|.1f}\\,\\%"),
        t("\\textbf{Stability across @{stability.n_refits} refits}: mean Spearman correlation of the importances @{stability.rf|.3f} (forest) and @{stability.rf_emb|.3f} (forest + GCN).",
          "\\textbf{Stabilitatea pe @{stability.n_refits} reantrenări}: corelația Spearman medie a importanțelor @{stability.rf|.3f} (pădurea) și @{stability.rf_emb|.3f} (pădure + GCN)")],
        height="0.50\\textheight", tryit=NB + "6",
        job=t("compare explanations across refits before reading a change of model as a change of explanation (next: research from our group).",
              "comparați explicațiile între reantrenări înainte de a interpreta o schimbare de model ca o schimbare de explicație."))

    # ------------------------------------------------------------------ 6 practice
    D.section(t("GNNs in production", "Rețelele GNN în producție"))
    D.frame(t("Four pitfalls of graph models in finance", "Patru capcane ale modelelor pe graf în finanțe"), cols(
        it((t("\\textbf{Leakage through edges}", "\\textbf{Scurgeri de informație prin muchii}"),
            t("a test node must not receive messages from the future", "un nod de test nu trebuie să primească mesaje din viitor"),
            t("in Elliptic @{data.pct_same_step|.1f}\\,\\% of the edges stay inside one time step, so the temporal split is clean",
              "în Elliptic @{data.pct_same_step|.1f}\\,\\% dintre muchii rămîn în același pas de timp, deci împărțirea în timp este curată")),
           (t("\\textbf{Few labels}", "\\textbf{Puține etichete}"),
            t("only @{data.pct_labelled|.1f}\\,\\% of the nodes are labelled; labels come from past investigations and are biased towards cases already found",
              "doar @{data.pct_labelled|.1f}\\,\\% dintre noduri sînt etichetate; etichetele vin din investigații anterioare și favorizează cazurile deja găsite")), sep="5pt"),
        it((t("\\textbf{Drift}", "\\textbf{Schimbarea în timp}"),
            t("criminals adapt: the shutdown at step 43 changed the patterns", "infractorii se adaptează: închiderea din pasul 43 a schimbat tiparele"),
            t("temporal GNNs (EvolveGCN) let the weights evolve with time", "rețelele GNN temporale (EvolveGCN) lasă ponderile să evolueze în timp")),
           (t("\\textbf{Explainability}", "\\textbf{Explicabilitatea}"),
            t("a suspicious-activity report needs a reason a regulator can read", "un raport de activitate suspectă are nevoie de o justificare pe care un supraveghetor o poate citi"),
            t("GNNExplainer finds the subgraph and features behind one prediction", "GNNExplainer găsește subgraful și variabilele din spatele unei predicții")), sep="5pt"), "0.49", "0.48") +
        D.bottom(t("test a graph model the way it will run: on later periods, with the labels available at that time.",
                   "testați un model pe graf așa cum va funcționa: pe perioade ulterioare, cu etichetele disponibile atunci."), None),
        "footnotesize", cite="(Pareja et al., 2020; Ying et al., 2019)")

    D.frame(t("Graphs everywhere in finance", "Grafuri peste tot în finanțe"), cols(
        it((t("\\textbf{Payments and cards}", "\\textbf{Plăți și carduri}"),
            t("cardholder--merchant graphs: a compromised terminal links many fraud victims", "grafuri deținător--comerciant: un terminal compromis leagă multe victime")),
           (t("\\textbf{Credit}", "\\textbf{Credit}"),
            t("guarantor and supplier networks: default spreads along trade credit", "rețele de garanți și furnizori: neplata se propagă prin creditul comercial")),
           (t("\\textbf{Crypto}", "\\textbf{Cripto}"),
            t("address and transaction graphs (this lecture), mixer detection, wallet clustering", "grafuri de adrese și tranzacții (acest curs), detectarea serviciilor de amestecare, gruparea portofelelor")), sep="5pt"),
        it((t("\\textbf{Systemic risk}", "\\textbf{Risc sistemic}"),
            t("interbank exposures and tail-dependence networks: who falls if one bank fails", "expuneri interbancare și rețele de dependență în cozi: cine cade dacă o bancă cade")),
           (t("\\textbf{Markets}", "\\textbf{Piețe}"),
            t("graphs of stocks linked by correlation, supply chain or ownership, for return forecasts", "grafuri de acțiuni legate prin corelație, lanț de aprovizionare sau acționariat, pentru prognoza randamentelor")),
           (t("\\textbf{Energy}", "\\textbf{Energie}"),
            t("power grids and interconnected price zones are graphs by construction", "rețelele electrice și zonele de preț interconectate sînt grafuri prin construcție")), sep="5pt"), "0.49", "0.48") +
        D.bottom(t("whenever the risk of one unit depends on its counterparties, a graph model is worth a try.",
                   "ori de cîte ori riscul unei unități depinde de contrapărțile ei, un model pe graf merită încercat."), None),
        "footnotesize", cite="(Hamilton, 2020; Bishop \\& Bishop, 2024, ch.\\ 13)")

    # ------------------------------------------------------------------ wrap-up
    D.section(t("Wrap-up", "Concluzii"))
    D.frame(t("What to remember", "Ce reținem"), it(
        t("\\textbf{A GNN layer} = message, permutation-invariant aggregation, update; GCN averages with degree weights, GraphSAGE separates self and neighbours, GAT learns the weights.",
          "\\textbf{Un strat GNN} = mesaj, agregare care nu depinde de ordine, actualizare; GCN face media cu ponderi după grad, GraphSAGE separă nodul de vecini, GAT învață ponderile"),
        t("\\textbf{Depth} = hops: two layers reach a median of @{receptive.k2_median} transactions; more layers over-smooth (PR-AUC @{depth.ap_6|.3f} with 6 layers).",
          "\\textbf{Adîncimea} = numărul de salturi: două straturi ajung la o mediană de @{receptive.k2_median} tranzacții; mai multe straturi netezesc excesiv (PR-AUC @{depth.ap_6|.3f} cu 6 straturi)"),
        t("\\textbf{On Elliptic} the edges help a network (GCN vs MLP: @{gcn_vs_mlp_ap_pts|+.1f} PR-AUC points), but a random forest on pre-aggregated features leads (F1 @{models.rf.f1|.3f}).",
          "\\textbf{Pe Elliptic} muchiile ajută o rețea (GCN față de MLP: @{gcn_vs_mlp_ap_pts|+.1f} puncte PR-AUC), dar o pădure aleatoare pe variabile deja agregate conduce (F1 @{models.rf.f1|.3f})"),
        t("\\textbf{Judge AML models} on later periods, with PR-AUC, alert budgets and bootstrap intervals; watch drift and the stability of explanations.",
          "\\textbf{Judecați modelele AML} pe perioade ulterioare, cu PR-AUC, bugete de alerte și intervale bootstrap; urmăriți schimbarea în timp și stabilitatea explicațiilor"),
        sep="6pt"))

    D.frame(t("Research from our group: graphs, fraud and explanations", "Cercetările grupului nostru: grafuri, fraudă și explicații"), it(
        t("\\textbf{Explanation stability in graph-augmented fraud detection} (Diaconu \\& Pele, 2026): treats the complete model refit as the unit of inference "
          "and compares SHapley Additive exPlanations (SHAP) of fraud models with and without graph embeddings against graph-free controls matched in dimension and covariance.",
          "\\textbf{Stabilitatea explicațiilor în detectarea fraudei cu variabile de graf} (Diaconu \\& Pele, 2026); lucrarea tratează reantrenarea completă a modelului drept unitate de inferență "
          "și compară explicațiile SHapley Additive exPlanations (SHAP) ale modelelor de fraudă cu și fără reprezentări de graf, față de controale fără graf, cu aceeași dimensiune și covarianță"),
        t("\\textbf{Key results}: augmentation can shrink the attributions of the original features mechanically; after accounting for retraining variability, graph-free controls reproduce most of the movement in marginal SHAP stability, while local, row-level rankings change.",
          "\\textbf{Rezultate}: adăugarea de variabile poate micșora mecanic atribuirile variabilelor originale; după ce se ține cont de variabilitatea reantrenării, controalele fără graf reproduc cea mai mare parte a variației stabilității SHAP marginale, iar ordonările locale, la nivel de rînd, se schimbă"),
        t("\\textbf{Energy Financial Risk Meter} (Toma, Dima, Grecu, Pele \\& Lessmann, 2026): measures systemic risk in the European Union energy sector from the network of tail dependencies between firms.",
          "\\textbf{Energy Financial Risk Meter} (Toma, Dima, Grecu, Pele \\& Lessmann, 2026): măsoară riscul sistemic din sectorul energetic al Uniunii Europene pe baza rețelei de dependențe în cozi dintre firme"),
        sep="5pt") + "\n\\vspace{1mm}{\\scriptsize " + t("Findings as stated in the papers; the numbers on the other slides come from our own computations.",
                                                     "Rezultatele de pe acest slide sînt cele din lucrări; cifrele de pe celelalte slide-uri provin din calculele noastre.") + "}",
        "footnotesize", cite=t("(Bucharest University of Economic Studies; Romanian Academy, Institute for Economic Forecasting)",
                               "(Academia de Studii Economice din București; Academia Română, Institutul de Prognoză Economică)"))

    D.frame(t("Want to learn more?", "Lecturi și resurse suplimentare"), cols(it(
        t("\\textbf{Bishop \\& Bishop (2024)}, chapter 13: graph neural networks, from message passing to graph attention.",
          "\\textbf{Bishop \\& Bishop (2024)}, capitolul 13: rețele neuronale pe grafuri, de la transmiterea mesajelor la atenția pe graf"),
        t("\\textbf{Hamilton (2020)}, \\textit{Graph Representation Learning}, free draft: ", "\\textbf{Hamilton (2020)}, \\textit{Graph Representation Learning}, versiune gratuită: ") +
        "\\href{https://www.cs.mcgill.ca/~wlh/grl_book/}{cs.mcgill.ca/\\textasciitilde wlh/grl\\_book}.",
        t("\\textbf{PyTorch Geometric} tutorials: ", "\\textbf{PyTorch Geometric}, tutoriale: ") + "\\href{https://pytorch-geometric.readthedocs.io}{pytorch-geometric.readthedocs.io}.", sep="4pt"),
        it(t("\\textbf{Elliptic data set} and the original paper: ", "\\textbf{Setul Elliptic} și lucrarea originală: ") + "\\href{https://arxiv.org/abs/1908.02591}{arXiv:1908.02591}.",
           t("\\textbf{This course}: code, data, notebooks --- ", "\\textbf{Acest curs}: cod, date, notebook-uri --- ") + "\\href{https://github.com/danpele/NEURAL_BIZ}{github.com/danpele/NEURAL\\_BIZ}.",
           t("\\textbf{Experiment yourself}: the notebook reproduces every chart and number of this lecture.",
             "\\textbf{Exercițiu practic}: notebook-ul reproduce toate graficele și cifrele acestui curs"), sep="4pt"), "0.52", "0.45"))

    # ------------------------------------------------------------------ quiz
    D.quiz_frame(t("permutations", "permutări"),
                 t("We renumber the transactions of the graph. What must happen to the GCN scores?", "Renumerotăm tranzacțiile grafului. Ce trebuie să se întîmple cu scorurile GCN?"),
                 t("They are permuted in the same way and are otherwise unchanged: a GNN layer is permutation-equivariant, because the aggregation over neighbours does not depend on order.",
                   "Se permută la fel și altfel nu se schimbă: un strat GNN este echivariant la permutări, pentru că agregarea pe vecini nu depinde de ordine"))
    D.quiz_frame(t("depth", "adîncimea"),
                 t("Why not stack ten GCN layers to see ten hops of money flow?", "De ce nu folosim zece straturi GCN, ca să vedem zece salturi ale fluxului de bani?"),
                 t("Repeated degree-weighted averaging makes all representations similar (over-smoothing): on Elliptic PR-AUC falls from @{depth.ap_2|.3f} (2 layers) to @{depth.ap_6|.3f} (6 layers).",
                   "Medierea repetată face toate reprezentările asemănătoare (over-smoothing): pe Elliptic, PR-AUC scade de la @{depth.ap_2|.3f} (2 straturi) la @{depth.ap_6|.3f} (6 straturi)"))
    D.quiz_frame(t("the strong benchmark", "modelul de referință puternic"),
                 t("Why does a random forest without the graph beat the GCN on Elliptic?", "De ce pădurea aleatoare fără graf bate rețeaua GCN pe Elliptic?"),
                 t("@{data.n_agg} of its features are already aggregates over the neighbours, a hand-made message-passing layer; trees also fit rule-like fraud thresholds well.",
                   "@{data.n_agg} dintre variabile sînt deja agregate pe vecini, un strat de transmitere a mesajelor construit manual; în plus, arborii prind bine pragurile de tip regulă ale fraudei"))
    D.quiz_frame(t("evaluation", "evaluarea"),
                 t("Only @{data.pct_test_illicit|.2f}\\,\\% of the test transactions are illicit. Which metric would mislead a compliance officer?",
                   "Doar @{data.pct_test_illicit|.2f}\\,\\% dintre tranzacțiile de test sînt ilicite. Ce măsură l-ar induce în eroare pe un ofițer de conformitate?"),
                 t("Accuracy: labelling everything licit is right most of the time and finds no criminal; use precision, recall, PR-AUC and the alert budget.",
                   "Acuratețea: dacă etichetăm totul ca licit avem dreptate în majoritatea cazurilor și nu găsim niciun infractor; folosim precizia, recall-ul, PR-AUC și bugetul de alerte"))
    D.quiz_frame(t("attention on graphs", "atenția pe grafuri"),
                 t("What is the link between a GAT layer and the transformer of Lecture 7?", "Care este legătura dintre un strat GAT și transformerul din cursul 7?"),
                 t("Both weight neighbours by a softmax of learned scores; a transformer is attention on the complete graph of the tokens, a GAT restricts it to the edges of the graph.",
                   "Ambele ponderează vecinii cu un softmax al unor scoruri învățate; transformerul aplică atenția pe graful complet al tokenilor, GAT o restrînge la muchiile grafului"))

    reference_frames(D, REFS, per_frame=9)
    D.mark_main_end()

    # ------------------------------------------------------------------ appendix
    D.parts.append("\\section*{" + t("Appendix", "Anexă") + "}\n")
    D.frame(t("Appendix A: from spectral filters to the GCN layer", "Anexa A: de la filtrele spectrale la stratul GCN"), cols(
        "\\[ g_\\theta\\star\\mathbf x\\approx\\sum_{k=0}^{K}\\theta_k T_k(\\tilde L)\\,\\mathbf x,\\quad \\tilde L=\\tfrac{2}{\\lambda_{\\max}}L_{sym}-I \\]"
        "\\[ K=1,\\ \\lambda_{\\max}\\approx2:\\quad g_\\theta\\star\\mathbf x\\approx\\theta_0\\mathbf x-\\theta_1D^{-1/2}AD^{-1/2}\\mathbf x \\]"
        "\\[ \\theta=\\theta_0=-\\theta_1:\\quad \\theta\\,(I+D^{-1/2}AD^{-1/2})\\,\\mathbf x\\ \\to\\ \\theta\\,\\hat A\\,\\mathbf x \\]",
        it(t("$T_k$ = Chebyshev polynomials; $L_{sym}=I-D^{-1/2}AD^{-1/2}$; a polynomial of degree $K$ only mixes nodes within $K$ hops.",
             "$T_k$ = polinoamele Cebîșev; $L_{sym}=I-D^{-1/2}AD^{-1/2}$; un polinom de grad $K$ amestecă doar noduri aflate la cel mult $K$ salturi"),
           t("One shared parameter per feature pair avoids overfitting; the renormalisation trick $I+D^{-1/2}AD^{-1/2}\\to\\tilde D^{-1/2}\\tilde A\\tilde D^{-1/2}$ keeps eigenvalues in $(-1,1]$.",
             "Un singur parametru comun evită supraînvățarea; trucul de renormalizare $I+D^{-1/2}AD^{-1/2}\\to\\tilde D^{-1/2}\\tilde A\\tilde D^{-1/2}$ ține valorile proprii în $(-1,1]$"),
           t("With $d$ input features and $d'$ outputs, $\\theta$ becomes the matrix $W\\in\\R^{d\\times d'}$.",
             "Cu $d$ variabile de intrare și $d'$ ieșiri, $\\theta$ devine matricea $W\\in\\R^{d\\times d'}$"), sep="3pt"), "0.52", "0.45"),
        "small", cite="(Kipf \\& Welling, 2017)")
    D.frame(t("Appendix B: the Weisfeiler--Leman test", "Anexa B: testul Weisfeiler--Leman"), cols(
        "\\[ c_v^{(l+1)}=\\operatorname{hash}\\Big(c_v^{(l)},\\ \\{\\!\\{c_u^{(l)}:u\\in\\mathcal N(v)\\}\\!\\}\\Big) \\]",
        it(t("Start with the same colour for all nodes; refine by hashing each node's colour with the multiset of its neighbours' colours.",
             "Se pornește cu aceeași culoare pentru toate nodurile; se rafinează prin aplicarea unei funcții hash culorii nodului și multisetului de culori ale vecinilor"),
           t("Two graphs with different colour histograms are not isomorphic; equal histograms are inconclusive.",
             "Două grafuri cu histograme de culori diferite nu sînt izomorfe; histogramele egale nu decid"),
           t("A message-passing GNN is at most as discriminative as WL; GIN reaches this bound with sum aggregation and an injective update.",
             "O rețea GNN cu transmitere de mesaje deosebește cel mult cît testul WL; GIN atinge această limită cu agregare prin sumă și o actualizare injectivă"), sep="3pt"), "0.50", "0.47"),
        "small", cite="(Xu et al., 2019)")
    D.frame(t("Appendix C: settings of the experiment", "Anexa C: setările experimentului"), table(
        "@{}p{0.30\\textwidth}p{0.62\\textwidth}@{}",
        [t("Setting", "Setare"), t("Value", "Valoare")],
        [[t("Split", "Împărțirea"), t("train steps 1--29, validation 30--34, test 35--49 (labelled nodes only in the loss)", "antrenare pașii 1--29, validare 30--34, test 35--49 (numai nodurile etichetate intră în funcția de pierdere)")],
         [t("Networks", "Rețele"), t("2 layers, 100 units, ReLU, dropout 0.2; GAT: 4 heads of 25; Adam, rate 0.003, decay $5\\cdot10^{-4}$; at most 300 epochs, patience 40",
                                     "2 straturi, 100 de unități, ReLU, dropout 0,2; GAT: 4 capete a cîte 25; Adam, rata 0,003, penalizare $5\\cdot10^{-4}$; cel mult 300 de epoci, răbdare 40")],
         [t("Class weights", "Ponderi de clasă"), "0.3 / 0.7" if not ro else "0,3 / 0,7"],
         [t("Random forest", "Pădure aleatoare"), t("100 trees, 50 features per split, raw features", "100 de arbori, 50 de variabile la fiecare împărțire, variabile nestandardizate")],
         [t("Seeds", "Semințe"), "1, 2, 3"],
         [t("Software, hardware", "Programe, echipament"), t("PyTorch, PyTorch Geometric, scikit-learn; laptop CPU, total @{runtime_min|.1f} minutes",
                                                            "PyTorch, PyTorch Geometric, scikit-learn; procesorul unui laptop, în total @{runtime_min|.1f} minute")]],
        size="scriptsize"), "small", cite="(Weber et al., 2019; Fey \\& Lenssen, 2019)")
    return D


if __name__ == "__main__":
    langs = [a for a in sys.argv[1:] if a in ("en", "ro")] or ["en", "ro"]
    sys.exit(0 if run_build(build, HERE, 9, ABBR, IGNORE, UNITS, langs=langs, exempt_first_use={"ASE", "IPE"}) else 1)
