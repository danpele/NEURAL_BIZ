<!--
Versiunea web a testului de funcționare (RO). Construită în web/L0_ro.html de lectures/common/build_web_lecture.py.
-->

## @intro

Această pagină doar testează fluxul de lucru: un grafic calculat pe date reale, sursa lui, Quantlet-ul și legătura către slide.

@goals
- **Verificați** că slide-urile, notebook-ul, Quantlet-ul și pagina web sînt construite din aceleași cifre.

# Un grafic pe date reale

Graficul de mai jos este calculat din valorile zilnice de închidere ale indicelui S&P 500.

## Pierderile pe acțiuni vin în episoade lungi {h=1c7282cf}

Scăderea $D_t = P_t / \max_{s \le t} P_s - 1$ arată cît de departe se află indicele sub maximul anterior. Scăderea maximă este de −56,78 %, la 9 martie 2009.

@chart sp500_drawdown
Scăderea indicelui S&P 500 față de maximul anterior.

@practice Un model antrenat numai pe o perioadă liniștită nu a văzut niciodată o scădere de această amploare.

## @end

Aceleași cifre apar pe slide, în notebook și în Quantlet.
