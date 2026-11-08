#!/usr/bin/env python3
"""
make_site_data.py -- writes assets/data-table.js (the Data section of the course site, EN + RO) from
data/clean/MANIFEST.json, which data/download_all.py maintains. Rerun after every download:
    python3 data/download_all.py && python3 data/make_site_data.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
M = json.load(open(os.path.join(ROOT, "clean", "MANIFEST.json"), encoding="utf-8"))

ORDER = ["fraud", "german", "hmeq", "california", "markets", "bvb", "power", "yield", "phrasebank", "tweets", "speeches",
         "elliptic", "fashion_mnist", "cifar10", "eurosat"]

# short notes for the site (the full notes are in data/SOURCES.md)
NOTES = {
    "fraud": ("All 492 frauds and a 10 % random sample of the legitimate transactions; column weight restores the full data (284,807 transactions). Licence: DbCL v1.0 (Database Contents License).",
              "Toate cele 492 de fraude și un eșantion aleator de 10 % din tranzacțiile legitime; coloana weight reface datele complete (284.807 tranzacții). Licență: DbCL v1.0 (Database Contents License)."),
    "german": ("default = 1 for a bad credit risk (300 of 1,000). Licence: CC BY 4.0 (Creative Commons Attribution).",
               "default = 1 pentru un risc de credit ridicat (300 din 1.000). Licență: CC BY 4.0 (Creative Commons Attribution)."),
    "hmeq": ("BAD = 1 for a default or serious delinquency; missing values kept. Companion data of Baesens, Roesch and Scheule (2016).",
             "BAD = 1 pentru neplată sau întîrziere gravă la plată; valorile lipsă sînt păstrate. Date însoțitoare ale cărții Baesens, Roesch și Scheule (2016)."),
    "california": ("Target: median house value in 100,000 USD, capped by the source at 5.00001.",
                   "Variabila-țintă: valoarea mediană a locuinței, în sute de mii de USD, plafonată de sursă la 5,00001."),
    "markets": ("Long format (date, ticker); adjclose adjusted for splits and dividends. Yahoo Finance terms: personal and educational use.",
                "Format lung (dată, simbol); adjclose este ajustat pentru divizări și dividende. Condițiile Yahoo Finance: uz personal și educațional."),
    "bvb": ("Prices in RON (Romanian leu), adjusted for splits and dividends; listing dates differ by stock. EODHD (End of Day Historical Data): personal and educational use.",
            "Prețuri în RON (leu românesc), ajustate pentru divizări și dividende; datele de listare diferă de la o acțiune la alta. EODHD (End of Day Historical Data): uz personal și educațional."),
    "power": ("Timestamps in UTC (Coordinated Universal Time); 15-minute prices averaged to the hour from October 2025. Prices: teaching use; generation and load: CC BY 4.0.",
              "Momente de timp în UTC (timpul universal coordonat); din octombrie 2025, prețurile pe 15 minute sînt mediate la nivel orar. Prețuri: uz didactic; producție și consum: CC BY 4.0."),
    "yield": ("Spot rates fitted by the European Central Bank (ECB) with the Svensson model on AAA-rated euro area government bonds; reuse with the source acknowledged.",
              "Rate spot estimate de Banca Centrală Europeană (BCE) cu modelul Svensson pe obligațiunile guvernamentale cu rating AAA din zona euro; reutilizare cu menționarea sursei."),
    "phrasebank": ("Labels positive, neutral, negative, with the share of annotators who agree. Licence: CC BY-NC-SA 3.0 (non-commercial).",
                   "Etichete pozitiv, neutru, negativ, cu ponderea adnotatorilor care sînt de acord. Licență: CC BY-NC-SA 3.0 (necomercial)."),
    "tweets": ("Sentiment (bearish, bullish, neutral) and 20 topics; train and validation splits as published. Licence: MIT.",
               "Sentiment (pesimist, optimist, neutru) și 20 de teme; împărțirea în antrenare și validare este cea publicată. Licență: MIT."),
    "speeches": ("Speeches from 1 January 2022 (full archive from 1997: data/raw, not committed); reuse with the source acknowledged.",
                 "Discursurile de la 1 ianuarie 2022 (arhiva completă, din 1997, se află în data/raw și nu este publicată); reutilizare cu menționarea sursei."),
    "elliptic": ("203,769 transactions, 234,355 payment flows, 49 time steps; labels illicit, licit, unknown. Downloaded by the notebook (torch_geometric) or by data/download_all.py; not redistributed here.",
                 "203.769 de tranzacții, 234.355 de fluxuri de plăți, 49 de intervale de timp; etichete ilicit, licit, necunoscut. Descărcat de notebook (torch_geometric) sau de data/download_all.py; nu este redistribuit aici."),
    "fashion_mnist": ("70,000 product images, 28 x 28 pixels, from the Zalando catalogue. Licence: MIT.",
                      "70.000 de imagini de produse, 28 x 28 pixeli, din catalogul Zalando. Licență: MIT."),
    "cifar10": ("60,000 colour images, 32 x 32 pixels, 10 classes.", "60.000 de imagini color, 32 x 32 pixeli, 10 clase."),
    "eurosat": ("27,000 Sentinel-2 satellite images, 64 x 64 pixels, 10 land-use classes. Licence: MIT.",
                "27.000 de imagini satelitare Sentinel-2, 64 x 64 pixeli, 10 clase de utilizare a terenului. Licență: MIT."),
}
UNITS = {"markets": ("index points, USD, USD per EUR", "puncte de indice, USD, USD pentru 1 EUR")}


def size(n):
    return f"{n / 1024:.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"


COV = {   # cross-sections: what the rows are; time series: computed from the files
    "fraud": ("September 2013, 2 days", "septembrie 2013, 2 zile"),
    "german": ("1,000 applicants", "1.000 de solicitanți"),
    "hmeq": ("5,960 loans", "5.960 de credite"),
    "california": ("1990 census; 20,640 block groups", "recensămîntul din 1990; 20.640 de grupuri de blocuri"),
    "phrasebank": ("4,838 sentences", "4.838 de fraze"),
    "tweets": ("11,931 + 21,107 tweets", "11.931 + 21.107 tweet-uri"),
    "elliptic": ("49 time steps (about two weeks each)", "49 de intervale de timp (de circa două săptămîni fiecare)"),
    "fashion_mnist": ("70,000 images", "70.000 de imagini"),
    "cifar10": ("60,000 images", "60.000 de imagini"),
    "eurosat": ("27,000 images", "27.000 de imagini"),
}


def coverage(k, e):
    if k in COV:
        return {"en": COV[k][0], "ro": COV[k][1]}
    import pandas as pd
    lo, hi = [], []
    for f in e["files"]:
        d = pd.read_csv(os.path.join(ROOT, "clean", f["file"]), usecols=[f["columns"][0]])[f["columns"][0]].astype(str).str[:10]
        lo.append(d.min()); hi.append(d.max())
    c = f"{min(lo)} – {max(hi)}"
    return {"en": c, "ro": c}


series, files = [], []
for k in ORDER:
    if k not in M:
        continue
    e = M[k]
    cov = coverage(k, e)
    lec_en = "Lectures " + ", ".join(x[1:] for x in e["lectures"])
    lec_ro = "Cursurile " + ", ".join(x[1:] for x in e["lectures"])
    note = NOTES.get(k, (e["note"], e["note"]))
    series.append({
        "id": k, "name": e["name"], "file": [f["file"] for f in e["files"]] or None,
        "freq": e["freq"],
        "coverage": cov,
        "units": {"en": lec_en, "ro": lec_ro},
        "source": e["source"], "href": e["url"],
        "note": {"en": note[0], "ro": note[1]},
        "lectures": e["lectures"],
    })
    for f in e["files"]:
        files.append({"file": f["file"], "size": size(f["size"]), "rows": f["rows"]})
files.sort(key=lambda f: f["file"])

js = ("// ============================================================\n"
      "// NN course website - Data section. GENERATED by data/make_site_data.py from data/clean/MANIFEST.json\n"
      "// (written by data/download_all.py); do not edit by hand. Loaded after course-data.js.\n"
      "// ============================================================\n"
      "(function () {\n    'use strict';\n    window.NN_DATA.data.series = " + json.dumps(series, ensure_ascii=False, indent=1) +
      ";\n    window.NN_DATA.data.files = " + json.dumps(files, ensure_ascii=False, indent=1) + ";\n})();\n")
open(os.path.join(REPO, "assets", "data-table.js"), "w", encoding="utf-8").write(js)
print(f"wrote assets/data-table.js: {len(series)} datasets, {len(files)} files")
