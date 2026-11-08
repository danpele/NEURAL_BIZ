<!--
Versiunea web a cursului 12 (RO). Construită în web/L12_ro.html de lectures/common/build_web_lecture.py (formatul: vezi docstring-ul).
Codurile cadrelor: python3 build_web_lecture.py 12 --frames
-->

## @intro

Textul este cel mai mare set de date al unei instituții financiare care este încă citit mai ales de oameni: știri, discursuri ale băncilor centrale, rapoarte anuale, dosare de credit. Modelele lingvistice mari (LLM) citesc text la scară mare, dar sînt și sigure pe ele, costisitoare și ușor de păcălit. Cursul tratează un LLM ca ceea ce este, un model de probabilitate a următorului token, și verifică fiecare afirmație pe date financiare etichetate, cu modele deschise care rulează pe un laptop.

Pornim de la obiectivul unui model de limbaj, de la tokeni, perplexitate și temperatură, măsurate pe 400 de discursuri ale Băncii Centrale Europene (BCE). Explicăm apoi preantrenarea, ajustarea pe instrucțiuni și alinierea. Miezul cursului este formularea instrucțiunilor (prompt engineering) ca activitate măsurabilă: prompturi zero-shot, few-shot și chain-of-thought pe Financial PhraseBank și pe tweet-uri financiare, cu intervale bootstrap, rulări repetate, reformulări, calibrare și cost. Construim generarea augmentată prin regăsire (RAG) pe discursurile BCE și măsurăm halucinațiile, iar la final testăm retrospectiv un model de limbaj folosit pentru valoarea expusă la risc (VaR), după cadrul LLM-VaR al grupului nostru.

@goals
- **Să scrieți** obiectivul unui model de limbaj, perplexitatea și temperatura de eșantionare și să le măsurați pe text real.
- **Să explicați** preantrenarea, ajustarea pe instrucțiuni și alinierea la preferințele umane, cu funcțiile lor de pierdere.
- **Să transformați** un prompt într-un clasificator și să evaluați prompturile zero-shot, few-shot și chain-of-thought, cu intervale de încredere.
- **Să construiți și să evaluați** generarea augmentată prin regăsire pe discursuri reale și să măsurați halucinațiile.
- **Să testați retrospectiv** un model de limbaj folosit pentru valoarea expusă la risc și să cunoașteți riscurile: scurgerea de date, costul, calibrarea.

# De la prezicerea următorului token la modelele lingvistice mari

Un LLM este antrenat să prezică următorul token. Tot restul, de la răspunsuri la întrebări pînă la redactarea unui raport de risc, înseamnă folosirea repetată a acestei singure probabilități.

## De ce contează în finanțe {h=61ee4a4c}

Prețurile reacționează la știri în cîteva minute, băncile centrale ghidează așteptările prin cuvinte, iar dosarele de credit și rapoartele anuale costă ore de analiză. Toate modelele din acest curs sînt deschise și rulează local (Qwen2.5-Instruct cu 0,5, 1,5 și 3 miliarde de parametri), deci niciun text nu părăsește calculatorul și nu se plătește nicio interfață de programare a aplicațiilor (API).

## Obiectivul modelului de limbaj {h=338971ad}

$$ p_{\boldsymbol\theta}(x_1,\dots,x_T)=\prod_{t=1}^{T}p_{\boldsymbol\theta}(x_t\mid x_{<t}),\qquad \mathcal L=-\frac{1}{T}\sum_{t=1}^{T}\log p_{\boldsymbol\theta}(x_t\mid x_{<t}),\qquad \mathrm{PPL}=e^{\mathcal L} $$

Perplexitatea este numărul efectiv de tokeni următori echiprobabili. Pe discursurile BCE publicate după iulie 2025 (primii 1 024 de tokeni din 40 de discursuri), perplexitatea este 14,71 pentru modelul de 0,5B, 10,63 pentru 1,5B și 9,43 pentru 3B.

## Tokenii sînt fragmente de cuvinte {h=1c7c0d8b}

Codificarea byte-pair (BPE) pornește de la octeți și unește repetat perechea vecină cea mai frecventă, pînă cînd vocabularul este complet: Qwen2.5 are 151 665 de tokeni, iar vocabularul WordPiece al FinBERT, o versiune financiară a modelului de reprezentări bidirecționale ale codificatorului transformer (BERT), are 30 522. Pe Financial PhraseBank revin 1,35 tokeni Qwen pe cuvînt.

## Lungimea propozițiilor în cuvinte și în tokeni {h=c7e730ad}

@chart tokens_per_sentence
Lungimea propozițiilor din Financial PhraseBank, în cuvinte și în tokeni.

O propoziție are în medie 23,1 cuvinte și 31,1 tokeni Qwen; un discurs BCE are în medie 4 143 de tokeni. Costurile și limitele contextului se numără în tokeni.

## Temperatura {h=77abff68}

$$ p_T(v\mid x_{<t})=\frac{\exp(z_v/T)}{\sum_u\exp(z_u/T)},\qquad H_T=-\sum_v p_T(v)\log p_T(v) $$

Cu promptul „The inflation impact is limited:”, luat din discursul BCE din 30 octombrie 2026, cel mai probabil token următor, „the”, are probabilitatea 49,9 % la temperatura 0,5, 10,1 % la 1 și 0,4 % la 2; entropia crește de la 1,74 la 10,29 nats.

## Probabilitățile următorului token pentru un model real {h=d38c918e}

@chart next_token
Cei mai probabili tokeni următori ai Qwen2.5-1.5B-Instruct, la trei temperaturi.

@practice pentru clasificare și extragere folosiți temperatura 0; temperaturile mari sînt pentru generarea de idei.

## Scalarea {h=902b0620}

Pierderea scade ca o putere a dimensiunii modelului și a datelor, $\mathcal L(P,D)=E+A/P^{\alpha}+B/D^{\beta}$ (Kaplan et al., 2020; Hoffmann et al., 2022). Pe cele trei dimensiuni ale modelului nostru, panta logaritmului pierderii în raport cu logaritmul numărului de parametri este −0,100: trei puncte descriu eșantionul, nu o lege.

## Discursurile mai vechi par mai ușoare {h=8eab6207}

@chart perplexity_size
Perplexitatea pe discursurile BCE din 2022–2023 și din iulie 2025.

Pentru modelul de 3B, perplexitatea este 8,90 pe discursurile din 2022–2023 și 9,43 pe cele din iulie 2025, ulterioare datelor de antrenare. Și temele s-au schimbat, așa că diferența este un simptom, nu o dovadă, a memorării; este un motiv să evaluați LLM-urile pe documente scrise după limita datelor lor de antrenare.

# Preantrenare, ajustare pe instrucțiuni și aliniere

## Trei etape de antrenare {h=e335ad17}

Un transformer generativ preantrenat (GPT) este antrenat întîi pe trilioane de tokeni, cu pierderea next-token; ajustarea fină supervizată (SFT) îl învață apoi formatul răspunsurilor, iar învățarea prin recompensă din feedback uman (RLHF) sau optimizarea directă a preferințelor (DPO) îl învață ce răspunsuri preferă oamenii (Ouyang et al., 2022).

## Ajustarea pe instrucțiuni {h=d3c78a72}

$$ \mathcal L_{\text{SFT}}(\boldsymbol\theta)=-\sum_{(\mathbf x,\mathbf y)}\sum_{t=1}^{|\mathbf y|}\log p_{\boldsymbol\theta}(y_t\mid\mathbf x,\mathbf y_{<t}) $$

Pierderea se calculează doar pe tokenii răspunsului. Adaptarea de rang mic (LoRA) antrenează $\mathbf W=\mathbf W_0+\mathbf B\mathbf A$, cu un rang $r$ mic: un strat de 4096 × 4096 cu $r=8$ are 65 536 de ponderi antrenabile în loc de 16,8 milioane (Hu et al., 2022).

## Alinierea {h=dad65eb1}

Oamenii aleg răspunsul mai bun dintr-o pereche; un model de recompensă $r_\phi$ se estimează cu $\Pr(y_w\succ y_l)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$, iar politica maximizează recompensa cu o penalizare Kullback–Leibler care o ține aproape de modelul SFT. DPO optimizează același obiectiv direct din perechile de preferințe (Rafailov et al., 2023). Alinierea stabilește tonul și siguranța; nu adaugă cunoștințe financiare.

## Modelele deschise folosite {h=14e34a5b}

Qwen2.5-Instruct cu 0,49, 1,54 și 3,09 miliarde de parametri (Qwen Team, 2024), modelul de embedding bge-small pentru regăsire și reprezentarea frecvența termenului–frecvența inversă în documente (TF-IDF), cu regresie logistică, drept reper clasic.

# Formularea instrucțiunilor, evaluată pe date etichetate

Un prompt este un model. Se alege pe date de validare și se evaluează o singură dată pe un set de test separat, cu aceleași statistici ca orice clasificator.

## Un prompt devine clasificator {h=65740da7}

$$ \hat p(c\mid x)=\frac{\sum_{v\in\mathcal V_c}p_{\boldsymbol\theta}(v\mid\mathrm{prompt}(x))}{\sum_{c'}\sum_{v\in\mathcal V_{c'}}p_{\boldsymbol\theta}(v\mid\mathrm{prompt}(x))},\qquad \hat c=\arg\max_c\hat p(c\mid x) $$

unde $\mathcal V_c$ sînt primii tokeni ai cuvîntului-etichetă $c$ în toate formele lui. O singură trecere înainte pe propoziție este suficientă, iar instrucțiunea și exemplele comune se codifică o singură dată.

## Două seturi de date etichetate {h=0528e960}

Financial PhraseBank (Malo et al., 2014) conține 4 838 de propoziții din știri despre companii listate în Finlanda; folosim cele 2 259 cu etichete unanime, împărțite în 1 581 de propoziții de antrenare și 678 de test. Setul Twitter financial news oferă 9 543 tweet-uri de antrenare și un eșantion de test stratificat de 900 de tweet-uri.

## LLM-uri cu prompt comparate cu un reper antrenat {h=c481d5ec}

@chart prompt_results
Macro-F1 cu intervale bootstrap de 95 % pe ambele seturi de date.

| Metoda | Macro-F1 PhraseBank (%) | Macro-F1 Twitter (%) |
|---|---|---|
| TF-IDF + regresie logistică | 86,9 | 75,5 |
| Zero-shot 0,5B | 31,0 | 18,7 |
| Zero-shot 1,5B | 85,1 | 40,9 |
| Zero-shot 3B | 92,2 | 77,4 |
| Few-shot 3B (12 exemple) | 92,1 | 79,8 |
| Chain-of-thought 3B | 88,4 | 65,1 |
| Embedding LLM + logistică | 87,5 | 79,0 |

Pe PhraseBank, modelul de 3B ajunge la 92,2 % macro-F1 fără exemple, peste clasificatorul TF-IDF antrenat pe 1 581 de propoziții etichetate (86,9 %). Pe tweet-uri, mai scurte și mai zgomotoase, cele mai bune rezultate le dau modelul de 3B cu exemple (79,8 %) și embedding-urile LLM cu un clasificator antrenat (79,0 %). Modelul de 0,5B este aproape de clasa majoritară.

## Acuratețea și macro-F1, cu intervale bootstrap {h=2f2d315d}

Macro-F1 face media celor trei scoruri F1 pe clase, astfel încît clasa mică negativă contează cît cea neutră. Intervalele de 95 % provin din 1 000 de reeșantionări bootstrap ale setului de test: pentru zero-shot 3B pe PhraseBank intervalul este [89,8; 94,4] %, pentru TF-IDF [83,8; 90,0] %.

## Exemple în prompt {h=0a6142ee}

@chart fewshot_curve
Macro-F1 în funcție de numărul de exemple din prompt, trei rulări.

Învățarea în context schimbă răspunsul fără să modifice nicio pondere. Pentru modelul de 1,5B, macro-F1 trece de la 85,1 % fără exemple la 84,1 % cu 24 de exemple; pentru modelul de 3B, de la 92,2 % la 91,3 %. Pe această sarcină instrucțiunea fără exemple este deja puternică, iar alegerea exemplelor mută scorul cu cîteva puncte.

## Sensibilitatea la prompt {h=27a1723f}

@chart prompt_sensitivity
Macro-F1 pentru cinci reformulări ale aceleiași instrucțiuni.

Cinci instrucțiuni cu același sens dau o amplitudine de 3,6 puncte pe PhraseBank și de 12,9 puncte pe Twitter pentru modelul de 3B și de 14,2 puncte pe PhraseBank pentru modelul de 1,5B. Formularea promptului este un hiperparametru: alegeți-o pe date de validare și fixați-o.

## Chain-of-thought {h=168e5876}

Dacă îl lăsăm pe model să scrie două propoziții de raționament înainte de etichetă (Wei et al., 2022), obținem 88,4 % macro-F1 pe PhraseBank și 65,1 % pe Twitter, sub răspunsul direct zero-shot, cu circa 389 ms pe propoziție în loc de 48,5 ms. Chain-of-thought ajută la raționamentele în mai mulți pași, nu la etichete de un cuvînt pe propoziții scurte.

## Calibrarea {h=5ece6711}

@chart calibration
Diagrama de fiabilitate a probabilităților prezise ale etichetelor.

Eroarea de calibrare așteptată (ECE) este 4,7 % pentru zero-shot 3B, 3,6 % pentru few-shot 3B și 5,4 % pentru TF-IDF cu regresie logistică: pe această sarcină, probabilitățile etichetelor date de LLM sînt aproximativ la fel de fiabile ca ale clasificatorului antrenat. Calibrarea trebuie totuși verificată pe date de validare înainte ca scorurile să fie folosite ca probabilități de risc (Guo et al., 2017).

## Acolo unde oamenii nu sînt de acord, modelele greșesc {h=5a367788}

@chart agreement
Acuratețea în funcție de acordul adnotatorilor.

Modelul zero-shot de 3B are dreptate pentru 93,1 % din propozițiile de test unanime, dar doar pentru 53,7 % din propozițiile asupra cărora jumătate dintre adnotatori nu au fost de acord. Un model nu poate fi mai sigur decît oamenii care au etichetat datele.

## Matricea de confuzie {h=bd4798f3}

Pentru cel mai bun model cu prompt pe PhraseBank, 20 de propoziții neutre au fost citite ca pozitive, iar 13 propoziții pozitive ca neutre, în timp ce pozitivul și negativul s-au confundat rar. Erorile stau la granița cu neutrul.

## Costul față de precizie {h=968e329a}

@chart cost_accuracy
Macro-F1 în funcție de milisecundele pe propoziție, pe un laptop.

TF-IDF are nevoie de 2,50 ms pe propoziție, zero-shot 3B de 48,5 ms, iar chain-of-thought de circa 389 ms. Cu etichete, un model mic antrenat este greu de depășit; fără etichete, un prompt este un început solid; pentru milioane de documente, costul decide.

## Embedding LLM plus un clasificator mic {h=d429340a}

Media ultimelor stări ascunse ale Qwen2.5-1.5B, cu o regresie logistică deasupra, dă 87,5 % pe PhraseBank și 79,0 % pe Twitter: cea mai ieftină formă de ajustare fină.

# Generare augmentată prin regăsire pe discursurile BCE

## Întîi regăsim, apoi răspundem {h=641c2912}

$$ p(y\mid q)\approx\sum_{z\in\mathrm{top}_k(q)}p_\eta(z\mid q)\,p_{\boldsymbol\theta}(y\mid q,z) $$

Corpusul are 400 de discursuri BCE, tăiate în 8 400 de fragmente de 200 de cuvinte (Lewis et al., 2020).

## Trei metode de regăsire {h=616f64a4}

Scorul Best Matching 25 (BM25) ponderează cuvintele comune după raritatea lor și saturează aparițiile repetate; regăsirea densă compară embedding-uri prin similaritatea cosinus; metoda hibridă combină cele două clasamente prin fuziunea rangurilor reciproce. Rangul reciproc mediu (MRR): BM25 0,515, dense 0,567, hibrid 0,622.

## Găsirea discursului corect după titlu {h=3d1eaa1a}

@chart rag_recall
Recall@k pentru cele patru metode de regăsire.

Metoda hibridă pune discursul corect pe primul loc pentru 50,7 % din titluri și între primele zece pentru 83,2 %.

## Fără documente modelul inventează vorbitorul {h=3020d297}

@chart rag_qa
Cine a ținut discursul? Răspunsuri fără documente, cu RAG și cu discursul corect.

Pe 160 de discursuri ținute după datele de antrenare ale modelului, un răspuns obligatoriu fără documente este greșit în 86,2 % din cazuri (modelul a răspuns „Philip Lane” de 137 de ori). Cînd „nu știu” este permis, se abține la 100,0 % din întrebări. Cu primele 3 discursuri regăsite răspunde corect în 81,2 % din cazuri, iar cu discursul corect în 98,8 %.

@practice un nume greșit spus cu siguranță este eșecul costisitor: cereți surse în fiecare răspuns și verificați-le.

## Proiectarea unui sistem RAG pentru o bancă {h=e52e4321}

Dimensiunea fragmentelor, metadatele care însoțesc fiecare fragment, o metodă de regăsire hibridă (recall@3 de 73,1 % pe setul de întrebări), permisiunea de a se abține și un set fix de evaluare, rulat din nou după fiecare schimbare, sînt deciziile care contează.

# Modelele de limbaj ca modele de risc: LLM-VaR

## LLM-VaR {h=a19b5e69}

În cadrul LLM-VaR (Pele et al., 2026), un model general citește ultimele 250 de randamente zilnice într-un prompt și scrie valoarea expusă la risc, fără estimare și fără ajustare fină. Testăm ideea cu un model deschis de 3B pe S&P 500, din 2 ianuarie 2025 pînă la 30 octombrie 2026 (459 de zile), față de simularea istorică (HS) și un model GARCH(1,1) cu erori Student-t, cu testul Kupiec al acoperirii.

## LLM-VaR cu un model deschis de 3B {h=cd1dbdce}

@chart llm_var
Randamentele S&P 500 cu LLM-VaR de 5 % și VaR GARCH.

La 5 %, LLM-VaR a fost depășită de 28 de ori, GARCH de 35 de ori, iar simularea istorică de 23 de ori, față de 23,0 depășiri așteptate. VaR medie de 5 % a modelului mic este −1,99 %, față de −1,42 % pentru GARCH, iar cele două serii sînt aproape necorelate (0,05): modelul de 3B obține acoperirea fiind prudent, nu urmărind volatilitatea.

## Testarea retrospectivă {h=974591b4}

| Model | Depășiri 1 % | p Kupiec (1 %) | Depășiri 5 % | p Kupiec (5 %) |
|---|---|---|---|---|
| LLM-VaR (Qwen2.5-3B) | 4 | 0,777 | 28 | 0,295 |
| Simularea istorică | 6 | 0,528 | 23 | 0,991 |
| GARCH(1,1)-t | 9 | 0,067 | 35 | 0,016 |

Studiul publicat a evaluat modele generale mult mai mari; rezultatele de aici descriu doar un model de 3B rulat pe laptop.

# Riscuri și guvernanță

## Șase riscuri și testul pentru fiecare {h=70eac6fa}

Halucinațiile (86,2 % nume greșite fără documente) se controlează prin RAG cu surse; scurgerea datelor, prin testare pe texte scrise după limita datelor de antrenare; sensibilitatea la prompt (12,9 puncte), prin alegerea și fixarea prompturilor pe date de validare; calibrarea (ECE 4,7 % zero-shot), prin verificare și, la nevoie, recalibrare pe date de validare; costul, prin cel mai mic model care trece testul; protecția datelor, prin modele locale și prin obligațiile din Regulamentul Uniunii Europene (UE) privind inteligența artificială (AI), pentru utilizările cu risc ridicat precum scoringul de credit.

## @end

Un prompt este un model: măsurați-l pe date etichetate, cu intervale, rulări repetate și reformulări; ancorați răspunsurile în documente; testați retrospectiv orice cifră de risc.
