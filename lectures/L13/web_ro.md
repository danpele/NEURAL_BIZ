<!--
Versiunea web a cursului 13 (RO). Construită în web/L13_ro.html de lectures/common/build_web_lecture.py (formatul: vezi docstring-ul).
Codurile cadrelor: python3 build_web_lecture.py 13 --frames
-->

## @intro

În fiecare zi, înainte de prînz, un furnizor de energie electrică, un producător hidro sau operatorul unei baterii trimite oferte pentru fiecare oră a zilei următoare, iar o bancă raportează pierderea pe care portofoliul său ar trebui să o depășească doar o dată la o sută de zile. Ambele decizii au nevoie de o prognoză: un preț pentru fiecare dintre cele 24 de ore de mîine sau o cuantilă a randamentului de mîine. Cursul pune o întrebare simplă, pe date reale: sînt rețelele neuronale antrenate pe seriile noastre sau noile modele fundație pentru serii de timp, preantrenate pe milioane de alte serii, mai bune decît reperele simple pe care practicienii le folosesc deja?

Prognozăm prețurile pentru ziua următoare din România pe un an de test întreg (1 noiembrie 2025 -- 31 octombrie 2026, 364 de zile complete), cu nouă strategii: o regulă naivă, modelul autoregresiv estimat prin Lasso (LEAR), o rețea neuronală profundă (DNN) cu 24 de ieșiri, analiza prin expansiune în baze neuronale (N-BEATS), trei modele fundație folosite fără reantrenare (Chronos-2, Chronos-Bolt și modelul fundație al Google Research, TimesFM), Chronos-2 cu covariabile și un ansamblu simplu. Apoi transformăm cuantilele modelului fundație în intervale calibrate prin predicție conformală și folosim aceeași idee pentru valoarea expusă la risc (VaR) a indicelui S&P 500 și a Bitcoin.

@goals
- **Să formulați** o problemă de prognoză pentru ziua următoare fără scurgeri de informație: setul de informații la închiderea licitației, originea glisantă, reperele.
- **Să scrieți și să estimați** LEAR, o rețea DNN cu 24 de ieșiri și N-BEATS pe prețurile din România.
- **Să explicați** cum prognozează modelele fundație fără nicio antrenare pe datele noastre și să le testați.
- **Să comparați** prognozele cu testul Diebold–Mariano și să calibrați prognozele probabilistice prin predicție conformală.
- **Să folosiți** un model fundație pentru valoarea expusă la risc și să o testați retrospectiv.

# Problema de prognoză și datele

O prognoză nu poate fi mai bună decît informația pe care are voie să o folosească. Secțiunea descrie datele, faptele stilizate ale prețurilor energiei electrice din România și regulile jocului: ce se știe la închiderea licitației, cum rămîne anul de test neatins și ce reper trebuie să depășească orice model.

## Două decizii de afaceri bazate pe prognoze pentru ziua următoare {h=636fbf17}

Pe piața pentru ziua următoare, prețul fiecărei ore de mîine este stabilit de o singură licitație, cuplată la nivel european și organizată în România de operatorul pieței de energie electrică și gaze naturale (OPCOM). Ofertele se închid la ora 12:00, ora Europei Centrale (CET), în ziua dinaintea livrării. O eroare de 1 EUR/MWh pe 100 MW timp de 24 de ore costă 2 400 de euro pe zi, așa că și îmbunătățirile mici contează. În finanțe, valoarea expusă la risc este cuantila randamentului de mîine care ar trebui depășită în 1 % din zile; după regulile Basel, numărul depășirilor dintr-un an modifică cerința de capital.

## Numai date reale {h=15ce4a81}

Prețurile pentru România (RO), Germania–Luxemburg (DE-LU) și Ungaria (HU) și consumul României provin de pe energy-charts.info, portalul de date al Institutului Fraunhofer pentru Sisteme de Energie Solară (ISE), care republică datele burselor și ale Rețelei europene a operatorilor de transport și de sistem pentru energie electrică (ENTSO-E). Fișierul de prețuri acoperă perioada 1 ianuarie 2015 -- 31 octombrie 2026, cu 103 595 de prețuri orare. Prețurile zilnice ale S&P 500 și ale Bitcoin provin de la Yahoo Finance. Orele lipsă rămîn lipsă: cele două zile pe an cu schimbarea orei intră în analiză doar dacă există toate cele 24 de ore.

## Media zilnică a prețurilor pentru ziua următoare în România {h=56e29095}

@chart price_history
Prețurile medii zilnice pentru ziua următoare în România și în Germania–Luxemburg.

Prețul mediu în România a fost de 50,41 EUR/MWh în 2019 și de 265,31 EUR/MWh în 2022; media zilnică maximă, 738,28 EUR/MWh, a fost atinsă pe 30 august 2022. Mediile zilnice din România și din Germania evoluează împreună (corelație 0,936), de aceea prețurile din Germania sînt variabile de intrare utile. Un model antrenat doar înainte de 2021 nu a văzut niciodată prețuri ca în 2022, iar unul antrenat mai ales pe 2022 supraestimează prețurile de azi: fereastra de antrenare este o decizie de modelare. Folosim trei ani, ca Lago et al. (2021).

## Producția solară a transformat vîrful de la prînz într-un minim {h=340c24ac}

@chart hourly_profile
Prețul mediu pe ore de livrare, în cîțiva ani.

În 2019 ora cea mai ieftină era noaptea; în 2026 (ianuarie–octombrie) cea mai ieftină este ora 13:00, cu 67,90 EUR/MWh, iar ora de seară 19:00 este cea mai scumpă, cu 217,05 EUR/MWh. Prognoza trebuie să acopere 24 de ore diferite, cu dinamici diferite; de aceea toate modelele de mai jos au 24 de ieșiri.

## Setul de informații la închiderea licitației {h=99e5dd4c}

Pentru ziua de livrare $d$, un model poate folosi doar ce se știe înainte de ora 12:00 CET din ziua $d-1$:

$$ \hat{\mathbf y}_d=f_{\boldsymbol\theta}(\mathbf x_d),\qquad \mathbf x_d=(\mathbf y_{d-1},\mathbf y_{d-2},\mathbf y_{d-3},\mathbf y_{d-7},\mathbf y^{\text{DE}}_{d-1},\boldsymbol\ell_{d-2},\mathbf c_d) $$

unde $\mathbf y_d\in\mathbb R^{24}$ sînt prețurile din România în ziua $d$, $\mathbf y^{\text{DE}}$ prețurile din Germania, $\boldsymbol\ell$ consumul României (complet cu două zile în urmă), iar $\mathbf c_d$ variabilele pentru ziua săptămînii. Rezultă 151 de variabile de intrare și 3 819 eșantioane zilnice din 2016. Modelele se reestimează pe o fereastră glisantă de 1 092 de zile, iar anul de test se folosește o singură dată.

@practice consumul zilei $d-1$ nu este complet la prînz; folosirea lui ar introduce viitorul în prognoză.

## Măsurile erorii și modelul de referință {h=f23332fb}

Raportăm eroarea absolută medie (MAE) în EUR/MWh și MAE relativă, $\mathrm{rMAE}=\mathrm{MAE}/\mathrm{MAE}^{\text{naiv}}$, unde prognoza naivă este prețul aceleiași ore de acum o săptămînă pentru luni, sîmbătă și duminică și prețul din ziua precedentă în celelalte zile. Erorile procentuale nu au sens cînd prețurile sînt aproape de zero sau negative; în anul de test au existat 195 de ore cu preț negativ, prețul mediu a fost de 134,59 EUR/MWh, iar prognoza naivă a avut o MAE de 27,74 EUR/MWh.

## @end

Regulile jocului: doar informația disponibilă la închiderea licitației, o fereastră glisantă, un an de test neatins și prognoza naivă ca prag minim.

# Un reper liniar puternic și o rețea profundă

Înaintea unei rețele profunde, un reper liniar puternic: LEAR este modelul de referință deschis al prognozei prețurilor energiei electrice. Rețeaua DNN folosește exact aceleași intrări, astfel încît comparația izolează modelul.

## Transformarea care stabilizează varianța {h=939866bb}

Prețurile au vîrfuri și pot fi negative. Sinusul hiperbolic invers se comportă ca un logaritm pentru valori mari și ca identitatea în jurul lui zero:

$$ z=\operatorname{asinh}\Big(\frac{y-\tilde m}{\hat\sigma}\Big),\qquad \hat\sigma=\frac{\mathrm{MAD}}{0.6745} $$

unde $\tilde m$ este mediana, iar MAD abaterea absolută mediană a ferestrei de antrenare (Uniejewski, Weron și Ziel, 2018). Prognoza se face pe scara transformată și se readuce în EUR/MWh cu $y=\tilde m+\hat\sigma\sinh(\hat z)$.

## LEAR: 24 de regresii Lasso, cîte una pentru fiecare oră {h=2485a17e}

$$ \hat{\boldsymbol\beta}_h=\arg\min_{\boldsymbol\beta}\sum_{d\in W}\big(z_{d,h}-\mathbf x_d^{\top}\boldsymbol\beta\big)^2+\lambda_h\lVert\boldsymbol\beta\rVert_1 $$

cu $\lambda_h$ ales prin criteriul informațional Akaike (AIC) pe traiectoria Lasso și reestimare zilnică. Pe anul de test, LEAR are o MAE de 22,79 EUR/MWh (rMAE 0,822).

## Rețeaua DNN: o singură rețea, 24 de ieșiri {h=67ce7984}

Un perceptron multistrat (MLP) transformă intrările în prognoza celor 24 de ore deodată:

$$ \hat{\mathbf z}_d=\mathbf W_3\,\mathrm{ReLU}\big(\mathbf W_2\,\mathrm{ReLU}(\mathbf W_1\mathbf x_d+\mathbf b_1)+\mathbf b_2\big)+\mathbf b_3,\qquad z_{d,h}=\operatorname{asinh}\frac{y_{d,h}-m_d}{s_d} $$

unde $m_d$ este mediana prețurilor din ziua $d-1$, iar $s_d$ o scară robustă a prețurilor decalate, astfel încît fiecare zi este adusă la propriul nivel (normalizare pe eșantion). Rețeaua are 256 și 128 de neuroni ascunși (74 904 parametri), dropout 0,1, pierderea L1 și oprire timpurie pe ultimele 20 % din fereastră; este reantrenată lunar, cu 4 rulări. MAE pe test este de 23,06 EUR/MWh; rulările individuale se situează între 23,57 și 24,03.

## Oprirea timpurie {h=20204952}

@chart training_curves
Pierderea pe antrenare și pe validare pentru DNN și N-BEATS.

Pierderea pe validare a rețelei DNN este minimă la epoca 14 din 44; N-BEATS, antrenată pe mult mai multe ferestre orare, nu se mai îmbunătățește după epoca 2.

## @end

LEAR și DNN folosesc aceeași informație; pe acest an, DNN nu depășește LEAR în mod semnificativ, un rezultat frecvent în literatură.

# Arhitecturi profunde de prognoză

Trei idei definesc rețelele moderne de prognoză: o rețea recurentă care produce o distribuție (DeepAR), stive de blocuri complet conectate (N-BEATS) și arhitecturi transformer aplicate pe petice ale seriei.

## O rețea recurentă probabilistă {h=7bade3c4}

Rețeaua recurentă autoregresivă profundă (DeepAR) a lui Salinas et al. (2020) parcurge seria cu o rețea cu memorie pe termen scurt și lung (LSTM) și produce parametrii distribuției de mîine; se antrenează prin verosimilitate maximă, simultan pe multe serii înrudite. Designul ei global și probabilistic este strămoșul modelelor fundație de azi.

## N-BEATS: blocuri care explică intrarea și prognozează restul {h=9345d75d}

Fiecare bloc citește reziduul intrării, elimină ce poate explica (reconstrucția) și adaugă contribuția sa la prognoză:

$$ \mathbf r_\ell=\mathbf r_{\ell-1}-\hat{\mathbf r}_\ell,\qquad \hat{\mathbf y}=\sum_{\ell=1}^{L}\hat{\mathbf y}_\ell $$

Modelul nostru N-BEATS citește doar ultimele 168 de ore de prețuri, are 6 blocuri și 1 739 904 parametri și a fost antrenat o singură dată, înainte de anul de test: MAE pe test 23,23 EUR/MWh (trei rulări, între 23,69 și 24,12).

## Arhitecturi transformer pentru serii de timp: peticele devin tokeni {h=afead399}

Arhitectura transformer pe petice (PatchTST) taie seria în petice de, de exemplu, 16 ore, reprezintă fiecare petic ca pe un cuvînt și aplică mecanismul de atenție între petice. Un context de 2 048 de ore devine 128 de tokeni, iar costul atenției scade de 256 de ori. Modelele fundație de mai jos se bazează pe această idee.

# Modele fundație pentru serii de timp

Un model fundație este preantrenat o singură dată pe o colecție foarte mare de serii și apoi folosit ca atare: citește contextul nostru și întoarce cuantile, fără niciun pas de gradient pe datele noastre.

## Un model fundație prognozează o serie pe care nu a văzut-o niciodată {h=96087dc3}

$$ \hat q^{\tau}_{T+k}=g_{\boldsymbol\theta^{*}}\big(y_{T-C+1:T}\big)_{k,\tau} $$

unde $C$ este lungimea contextului (aici 2 048 de ore), $\tau$ nivelul cuantilei, iar $\boldsymbol\theta^{*}$ ponderile preantrenate, fixe, obținute prin minimizarea pierderii cuantile (pinball) pe corpusul de preantrenare. Testăm Chronos-2 și Chronos-Bolt (Amazon) și TimesFM (Google Research).

## Chronos: tokenizarea seriilor de timp {h=c8562437}

Chronos original (Ansari et al., 2024) împarte seria la valoarea ei absolută medie, taie valorile scalate în intervale și antrenează un model de limbaj să prezică următorul interval, cu entropia încrucișată. Chronos-Bolt și Chronos-2 înlocuiesc intervalele cu petice și ieșiri de cuantile. Modelul cunoaște forme (sezonalitate, revenire la medie), nu piața românească.

## Precizia în anul de test {h=84a7decb}

| Model | MAE (EUR/MWh) | rMAE |
|---|---|---|
| Naiv | 27,74 | 1,000 |
| LEAR | 22,79 | 0,822 |
| DNN | 23,06 | 0,831 |
| N-BEATS | 23,23 | 0,837 |
| Chronos-Bolt | 21,67 | 0,781 |
| TimesFM | 22,37 | 0,806 |
| Chronos-2 | 20,72 | 0,747 |
| Chronos-2 + covariabile | 20,41 | 0,736 |
| Ansamblul LEAR + DNN + Chronos-2 | 20,77 | 0,749 |

## Rețele antrenate față de modele fundație fără reantrenare {h=be58f614}

@chart mae_models
Eroarea absolută medie a celor nouă prognoze în anul de test.

Prognoza Chronos-2 fără reantrenare, fără nicio antrenare pe date românești, are o MAE mai mică (20,72 EUR/MWh) decît LEAR (22,79) și decît cele două rețele antrenate aici. Adăugarea prețurilor din Germania și Ungaria și a consumului României drept covariabile din trecut o reduce în continuare, la 20,41 EUR/MWh.

@practice începeți cu reperul și cu un model fundație: amîndouă costă minute, nu săptămîni.

## Erorile pe ore {h=34b639ea}

@chart mae_by_hour
MAE pe ore de livrare.

Toate modelele greșesc mai mult seara: ora cea mai grea pentru LEAR este 19:00 (MAE 35,78 EUR/MWh), la fel și pentru Chronos-2 (34,05), în timp ce orele de noapte se prognozează cu erori de circa 10,75 EUR/MWh.

## O săptămînă de prognoze {h=ec67ef0a}

@chart forecast_week
Prețurile realizate și trei prognoze, 19–25 octombrie 2026.

În săptămîna 19 octombrie 2026 -- 25 octombrie 2026, MAE a fost de 43,34 pentru LEAR, 32,33 pentru DNN și 25,41 EUR/MWh pentru Chronos-2. Toate modelele urmăresc forma zilnică și netezesc vîrfurile de seară.

## Un context mai lung ajută, pînă la un punct {h=6a96a329}

@chart context_length
MAE Chronos-2 în funcție de lungimea contextului.

Cu o săptămînă de context (168 de ore), MAE a lui Chronos-2 este de 24,45 EUR/MWh; cu 2 048 de ore scade la 20,72, iar 8 192 de ore nu mai ajută (20,95). Lungimea contextului este principalul parametru de ajustare al unui model folosit fără reantrenare.

## Testul Diebold–Mariano {h=8fba3822}

O eroare medie mai mică poate fi noroc. Cu diferența zilnică a pierderilor $\Delta_d=\lVert\mathbf e^{A}_d\rVert_1-\lVert\mathbf e^{B}_d\rVert_1$ pe cele 24 de ore ale fiecărei zile,

$$ \mathrm{DM}=\frac{\bar\Delta}{\hat\sigma_\Delta/\sqrt{|\mathcal T|}}\ \overset{a}{\sim}\ \mathcal N(0,1) $$

testează dacă modelul B este mai precis decît modelul A (Diebold și Mariano, 1995; Ziel și Weron, 2018). Lucrul cu pierderi zilnice, nu orare, ține cont de faptul că cele 24 de ore ale unei zile folosesc aceeași informație.

## Testele Diebold–Mariano pe perechi de modele {h=0e1d4f86}

@chart dm_pvalues
Valorile p unilaterale ale testului Diebold–Mariano multivariat.

Chronos-2 este semnificativ mai precis decît LEAR (p < 0,001), la fel și ansamblul, în timp ce DNN nu este semnificativ mai bună decît LEAR (p = 0,761). Adăugarea covariabilelor la Chronos-2 aduce o îmbunătățire semnificativă (p = 0,007).

## @end

Modelele fundație prognozează prețurile din România fără reantrenare la fel de bine sau mai bine decît modelele antrenate aici; semnificația se testează pe pierderile zilnice.

# Prognoze probabilistice și calibrare conformală

Operatorul unei baterii sau un departament de risc are nevoie de distribuția prețului de mîine, nu doar de centrul ei. Modelele fundație produc direct cuantile; predicția conformală le face fiabile.

## Prognozele de cuantile și pierderea pinball {h=062373fe}

O prognoză de cuantilă $\hat q^{\tau}$ se evaluează cu pierderea pinball $\rho_\tau(u)=u(\tau-\mathbb 1\{u<0\})$, $u=y-\hat q^{\tau}$, mediată pe cele nouă decile. În anul de test: Chronos-2 8,25, cu covariabile 8,11, Chronos-Bolt 8,62, TimesFM 8,97 și LEAR cu erori empirice 9,12 EUR/MWh.

## O săptămînă de cuantile Chronos-2 {h=4677bb1c}

@chart chronos_fan
Benzile de cuantile Chronos-2 și prețurile realizate, 19–25 octombrie 2026.

## Corecția conformală {h=27732c19}

Scorul de conformitate al fiecărei ore trecute măsoară cît de departe de banda brută a căzut valoarea realizată, $s_i=\max(\hat q^{\alpha/2}_i-y_i,\ y_i-\hat q^{1-\alpha/2}_i)$. Luăm ca $\hat Q$ al $\lceil (n+1)(1-\alpha)\rceil$-lea cel mai mic scor din ultimele 60 de zile și lărgim banda cu $\hat Q$ pe ambele părți; pentru date interschimbabile, acoperirea este garantat cel puțin $1-\alpha$ (Romano, Patterson și Candès, 2019), iar pentru serii de timp garanția este aproximativă. Pentru banda de 80 %, acoperirea brută a lui Chronos-2 a fost de 76,5 %, iar cea conformală de 79,3 %, cu lățimea medie trecînd de la 60,03 la 63,89 EUR/MWh.

## Acoperirea față de nivelul nominal {h=0b99428b}

@chart coverage
Acoperirea empirică a intervalelor de 50, 80 și 90 %.

Pentru banda de 90 %: brut 88,3 %, conformal 89,4 %, LEAR cu erori empirice 85,0 %.

## Consumul este mai ușor de prognozat decît prețul {h=65f4bb45}

@chart load_mape
Eroarea procentuală absolută medie a prognozelor consumului orar al României.

Pentru consumul României (în medie 5 932 MW), erorile procentuale au sens: aceeași oră din săptămîna trecută dă o eroare procentuală absolută medie (MAPE) de 7,15 %, TimesFM 4,47 %, iar Chronos-2 3,80 %.

# Modele fundație pentru riscul financiar

Același mecanism al cuantilelor dă o valoare expusă la risc.

## Valoarea expusă la risc și testarea ei retrospectivă {h=0d85c6d7}

VaR pe o zi, la nivelul $\alpha$, este cuantila condiționată de nivel $\alpha$ a randamentului de mîine. Testul Kupiec compară ponderea depășirilor cu $\alpha$,

$$ \mathrm{LR}_{uc}=-2\ln\frac{(1-\alpha)^{n_0}\alpha^{n_1}}{(1-\hat\pi)^{n_0}\hat\pi^{n_1}}\sim\chi^2_1 $$

iar testul Christoffersen verifică dacă depășirile nu vin grupat. Comparăm simularea istorică (HS), un model GARCH(1,1) cu erori Student-$t$ și cuantilele Chronos-2 calculate din ultimele 1 024 de randamente.

## O deplasare conformală unilaterală {h=e366f336}

Cu $s_i=\hat q^{\alpha}_i-r_i$ pe ultimele 250 de zile și $c_t$ a $\lceil (n+1)(1-\alpha)\rceil$-a cea mai mică valoare a lor, VaR calibrată este $\hat q^{\alpha}_{t+1}-c_t$. Pentru S&P 500, la nivelul de 1 %, cuantila brută Chronos-2 a fost depășită de 22 de ori, cea conformală de 16 ori, față de 17,2 depășiri așteptate.

## VaR calibrată urmărește regimurile de volatilitate {h=8a7bd9c1}

@chart var_sp500
Randamentele S&P 500 cu VaR de 1 % a modelului GARCH(1,1)-t și a lui Chronos-2 cu deplasare conformală.

Pe 1 717 zile de tranzacționare, VaR conformală Chronos-2 a trecut testul Kupiec la ambele niveluri (p = 0,774 la 1 % și 0,434 la 5 %), în timp ce GARCH a fost depășită de 32 de ori la 1 % (p = 0,001). Pentru Bitcoin, toate metodele dau o acoperire acceptabilă la 1 %; VaR conformală Chronos-2 a avut 21 de depășiri, față de 25,0 așteptate.

@practice recalibrarea este o schimbare de model: documentați-o și testați-o pe date noi.

## Prognozele cuantilelor cu modele de limbaj și cu modele de serii de timp {h=8c748764}

În cursul 12, randamentele trecute au fost scrise în promptul unui model de limbaj (LLM-VaR, Pele et al., 2026); aici, un model fundație pentru serii de timp le citește ca numere. Ambele căi sînt zero-shot și ambele trebuie testate retrospectiv, pe date noi, față de GARCH și simularea istorică.

## @end

Calea modelelor fundație dă cuantile în cîteva secunde; calibrarea conformală și testele retrospective standard le fac utilizabile pentru risc.
