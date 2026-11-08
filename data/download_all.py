#!/usr/bin/env python3
"""
download_all.py -- real-data layer of the course "Neural Networks and Deep Learning" (NN)
=====================================================================================
Downloads every dataset used in the lectures from its public source, keeps the raw files under
data/raw/<dataset>/ (git-ignored) and writes the cleaned files used by the lectures, notebooks and Quantlets
under data/clean/. It then regenerates data/clean/MANIFEST.json and data/SOURCES.md (rows, columns, size,
coverage, source, licence, download date, lectures that use each file).

Rules
-----
* Real data only: no synthetic rows, no interpolation, no gap filling. Missing values stay missing.
* A dataset larger than about 50 MB is not committed: the repository keeps a documented subset (stated in
  SOURCES.md) or the dataset is downloaded on demand into data/raw/ by this script or by the notebook.
* Secrets (the EODHD key) are read from a .env file inside the script and are never printed or written.
* Idempotent: raw files already present are reused; --refresh downloads everything again.

Usage (from the repository root)
--------------------------------
    python3 data/download_all.py                 # everything
    python3 data/download_all.py fraud phrasebank # only some datasets (names: see DATASETS at the bottom)
    python3 data/download_all.py --refresh        # force new downloads
    python3 data/download_all.py --sources-only   # rewrite SOURCES.md from MANIFEST.json
"""
import io
import json
import os
import shutil
import sys
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
RAW = os.path.join(ROOT, "raw")
CLEAN = os.path.join(ROOT, "clean")
MANIFEST = os.path.join(CLEAN, "MANIFEST.json")
for _d in (RAW, CLEAN):
    os.makedirs(_d, exist_ok=True)

REFRESH = "--refresh" in sys.argv
TODAY = date.today().isoformat()
UA = {"User-Agent": "Mozilla/5.0 (NN course data script; academic use)"}
SEED = 20261108
# energy series are shared with the course Energy Markets Quantitative Analysis (same download pipeline)
EMQA_CLEAN = os.path.expanduser("~/Documents/Teaching/Energy MBA - EMQA/repo/data/clean")
ENV_CANDIDATES = [os.path.join(REPO, ".env"), os.path.expanduser("~/Documents/2026 MFM/MFM/.env")]


# --------------------------------------------------------------------------------------------- helpers
def log(*a):
    print("[data]", *a, flush=True)


def read_secret(name):
    """Value of NAME from the environment or from the first .env file that defines it (never printed)."""
    if os.environ.get(name):
        return os.environ[name]
    for p in ENV_CANDIDATES:
        if os.path.isfile(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return None


def fetch(url, rel, params=None, timeout=300):
    """Download url to data/raw/rel (reused if present unless --refresh); returns the local path."""
    path = os.path.join(RAW, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path) and os.path.getsize(path) > 0 and not REFRESH:
        return path
    log("download", url.split("?")[0])
    with requests.get(url, params=params, headers=UA, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, path)
    return path


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u in ("B", "KB") else f"{n:.1f} {u}"
        n /= 1024


def write_clean(df, name, index=False, float_format=None):
    path = os.path.join(CLEAN, name)
    df.to_csv(path, index=index, float_format=float_format)
    log("  ->", name, f"{len(df):,} rows", human(os.path.getsize(path)))
    return path


def manifest_load():
    return json.load(open(MANIFEST)) if os.path.isfile(MANIFEST) else {}


def manifest_save(M):
    json.dump(M, open(MANIFEST, "w"), indent=1, ensure_ascii=False)


def entry(key, files, **kw):
    """Manifest entry: metadata of one dataset plus the facts of each clean file (rows, columns, size)."""
    out = dict(kw)
    out["downloaded"] = TODAY
    out["files"] = []
    for fn, cov in files:
        p = os.path.join(CLEAN, fn)
        df = pd.read_csv(p, nrows=5)
        n = sum(1 for _ in open(p, encoding="utf-8")) - 1
        out["files"].append({"file": fn, "rows": n, "columns": list(df.columns), "size": os.path.getsize(p),
                             "coverage": cov})
    return key, out


# --------------------------------------------------------------------------------------------- 1. credit-card fraud
def ds_fraud():
    """ULB credit-card fraud (Dal Pozzolo et al., 2015), OpenML id 1597: 284,807 card transactions of September 2013."""
    p = fetch("https://openml.org/data/v1/download/1673544/creditcard.arff", "fraud/creditcard.arff")
    cols, rows_start = [], None
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.lower().startswith("@attribute"):
                cols.append(s.split()[1].strip("'\""))
            if s.lower().startswith("@data"):
                rows_start = i + 1
                break
    df = pd.read_csv(p, skiprows=rows_start, header=None, names=cols, quotechar="'")
    df["Class"] = df["Class"].astype(str).str.strip("'\"").astype(int)
    n_all, n_fraud = len(df), int(df["Class"].sum())
    rng = np.random.default_rng(SEED)
    legit = df.index[df["Class"] == 0]
    keep_legit = rng.choice(legit, size=int(round(0.10 * len(legit))), replace=False)
    sub = df.loc[np.sort(np.concatenate([df.index[df["Class"] == 1], keep_legit]))].copy()
    sub["weight"] = np.where(sub["Class"] == 1, 1.0, len(legit) / len(keep_legit))
    sub.insert(0, "row_id", sub.index.astype(int))
    sub[[c for c in sub.columns if c.startswith("V")]] = sub[[c for c in sub.columns if c.startswith("V")]].round(6)
    write_clean(sub, "creditcard_fraud_sample.csv")
    return entry("fraud", [("creditcard_fraud_sample.csv", "September 2013 (2 days); all frauds + 10 % random sample of legitimate transactions")],
                 name={"en": "Credit-card fraud (ULB Machine Learning Group, European cardholders)",
                       "ro": "Fraude cu carduri bancare (ULB Machine Learning Group, deținători de carduri din Europa)"},
                 source="OpenML dataset 1597 (Worldline and ULB Machine Learning Group)",
                 url="https://www.openml.org/d/1597",
                 licence="Database Contents License (DbCL) v1.0 (Kaggle release); OpenML: public",
                 cite="Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE SSCI, 159-166. doi:10.1109/SSCI.2015.33",
                 lectures=["l1", "l2", "l3", "l8", "l11"],
                 note=(f"Full data: {n_all:,} transactions, {n_fraud} frauds ({100 * n_fraud / n_all:.3f} %), {human(os.path.getsize(p))} raw "
                       "(kept in data/raw/, not committed). The committed file keeps all frauds and a 10 % random sample of the "
                       f"legitimate transactions (seed {SEED}); column weight = inverse sampling rate, so weighted statistics "
                       "reproduce the full data. V1-V28 are PCA (principal component analysis) components released by the "
                       "data owner; Time = seconds since the first transaction; Amount in EUR; row_id = row in the full file."),
                 freq={"en": "transaction", "ro": "tranzacție"})


# --------------------------------------------------------------------------------------------- 2. German credit
GERMAN_COLS = ["checking_status", "duration_months", "credit_history", "purpose", "credit_amount_dm", "savings_status",
               "employment_since", "installment_rate_pct", "personal_status_sex", "other_debtors", "residence_since",
               "property", "age_years", "other_installment_plans", "housing", "existing_credits", "job",
               "num_dependents", "telephone", "foreign_worker", "default"]


def ds_german():
    p = fetch("https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip", "german/statlog_german.zip")
    with zipfile.ZipFile(p) as z:
        txt = z.read("german.data").decode("latin-1")
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+", header=None, names=GERMAN_COLS)
    df["default"] = (df["default"] == 2).astype(int)        # 1 = bad credit risk, 0 = good
    write_clean(df, "german_credit.csv")
    return entry("german", [("german_credit.csv", "1,000 loan applicants of a German bank")],
                 name={"en": "German credit (Statlog): credit scoring of 1,000 loan applicants",
                       "ro": "German credit (Statlog): scoringul de credit pentru 1.000 de solicitanți de împrumut"},
                 source="UCI Machine Learning Repository, dataset 144 (H. Hofmann, University of Hamburg)",
                 url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
                 licence="CC BY 4.0 (Creative Commons Attribution 4.0)",
                 cite="Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. doi:10.24432/C5NC77",
                 lectures=["l1", "l3", "l11"],
                 note=("Categorical attributes keep the UCI codes (A11, A12, ...; code book in the UCI page); "
                       "default = 1 for a bad credit risk (300 of 1,000). The UCI cost matrix weights a missed bad risk 5 times a rejected good one."),
                 freq={"en": "applicant", "ro": "solicitant"})


# --------------------------------------------------------------------------------------------- 3. HMEQ
def ds_hmeq():
    src = os.path.join(REPO, "legacy", "data", "hmeq.csv")
    raw = os.path.join(RAW, "hmeq", "hmeq.csv")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    if not os.path.isfile(raw) or REFRESH:
        try:
            fetch("http://www.creditriskanalytics.net/uploads/1/9/5/1/19511601/hmeq.csv", "hmeq/hmeq.csv")
        except Exception as e:  # the author site is sometimes down: the copy kept in legacy/data is the same file
            log("  hmeq: source site unavailable, using legacy/data/hmeq.csv", type(e).__name__)
            shutil.copy(src, raw)
    df = pd.read_csv(raw)
    write_clean(df, "hmeq.csv")
    return entry("hmeq", [("hmeq.csv", "5,960 home-equity loans (United States)")],
                 name={"en": "HMEQ: home-equity loan defaults", "ro": "HMEQ: neplata creditelor ipotecare de tip home equity"},
                 source="Baesens, Roesch & Scheule (2016), Credit Risk Analytics (companion data)",
                 url="http://www.creditriskanalytics.net/datasets-private2.html",
                 licence="Distributed by the book authors for teaching and research",
                 cite="Baesens, B., Roesch, D. & Scheule, H. (2016). Credit Risk Analytics: Measurement Techniques, Applications, and Examples in SAS. Wiley.",
                 lectures=["l1", "l3", "l11"],
                 note="BAD = 1 if the applicant defaulted or was seriously delinquent; missing values kept as missing (they are part of the exercise).",
                 freq={"en": "loan", "ro": "credit"})


# --------------------------------------------------------------------------------------------- 4. California housing
def ds_california():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(data_home=os.path.join(RAW, "california"), as_frame=True)
    df = d.frame.rename(columns={"MedHouseVal": "MedHouseVal_100k_usd"})
    write_clean(df, "california_housing.csv")
    return entry("california", [("california_housing.csv", "20,640 census block groups, California, 1990 census")],
                 name={"en": "California housing (1990 census block groups)", "ro": "Locuințe din California (grupuri de blocuri, recensămîntul din 1990)"},
                 source="StatLib repository (Pace & Barry, 1997), via scikit-learn",
                 url="https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html",
                 licence="Public research dataset (StatLib); redistributed by scikit-learn (BSD-3-Clause)",
                 cite="Pace, R. K. & Barry, R. (1997). Sparse spatial autoregressions. Statistics & Probability Letters, 33(3), 291-297. doi:10.1016/S0167-7152(96)00140-X",
                 lectures=["l2", "l3", "l11"],
                 note="Target MedHouseVal_100k_usd = median house value in units of 100,000 USD, capped at 5.00001 by the data source (keep the cap in mind).",
                 freq={"en": "block group", "ro": "grup de blocuri"})


# --------------------------------------------------------------------------------------------- 5. market prices (Yahoo)
YAHOO = {"^GSPC": "SP500", "^STOXX50E": "EUROSTOXX50", "^GDAXI": "DAX", "^VIX": "VIX", "EURUSD=X": "EURUSD",
         "BTC-USD": "BTCUSD", "AAPL": "AAPL", "MSFT": "MSFT", "JPM": "JPM", "XOM": "XOM", "NVDA": "NVDA"}


def ds_markets():
    import yfinance as yf
    frames = {}
    for t, n in YAHOO.items():
        raw = os.path.join(RAW, "yahoo", n + ".csv")
        if not os.path.isfile(raw) or REFRESH:
            os.makedirs(os.path.dirname(raw), exist_ok=True)
            x = yf.download(t, start="2005-01-01", progress=False, auto_adjust=False, threads=False)
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = x.columns.get_level_values(0)
            x.to_csv(raw)
        x = pd.read_csv(raw, index_col=0, parse_dates=True)
        frames[n] = x
    rows = []
    for n, x in frames.items():
        y = x[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        y.columns = ["open", "high", "low", "close", "adjclose", "volume"]
        y["ticker"] = n
        rows.append(y)
    df = pd.concat(rows).reset_index().rename(columns={"index": "date", "Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "ticker", "open", "high", "low", "close", "adjclose", "volume"]].dropna(subset=["close"])
    write_clean(df.sort_values(["ticker", "date"]), "markets_daily.csv", float_format="%.6f")
    cov = f"{df['date'].min()} to {df['date'].max()} (BTCUSD from {df.loc[df.ticker == 'BTCUSD', 'date'].min()})"
    return entry("markets", [("markets_daily.csv", cov)],
                 name={"en": "Stock indices, stocks, volatility, FX and Bitcoin: S&P 500, EURO STOXX 50, DAX, VIX, EUR/USD, BTC/USD, Apple, Microsoft, JPMorgan, ExxonMobil, NVIDIA",
                       "ro": "Indici bursieri, acțiuni, volatilitate, curs valutar și Bitcoin: S&P 500, EURO STOXX 50, DAX, VIX, EUR/USD, BTC/USD, Apple, Microsoft, JPMorgan, ExxonMobil, NVIDIA"},
                 source="Yahoo Finance (via yfinance)", url="https://finance.yahoo.com/quote/%5EGSPC",
                 licence="Yahoo Finance terms: personal and educational use",
                 cite="Yahoo Finance, daily quotes downloaded with yfinance",
                 lectures=["l5", "l10", "l13"],
                 note="Long format (date, ticker); adjclose = close adjusted for splits and dividends. Exchange trading days; no gap filling.",
                 freq={"en": "daily (trading days)", "ro": "zilnic (zile de tranzacționare)"})


# --------------------------------------------------------------------------------------------- 6. Bucharest Stock Exchange (EODHD)
BVB = {"TLV": "Banca Transilvania", "BRD": "BRD - Groupe Société Générale", "SNP": "OMV Petrom", "SNG": "Romgaz",
       "SNN": "Nuclearelectrica", "H2O": "Hidroelectrica", "TGN": "Transgaz", "EL": "Electrica", "TEL": "Transelectrica",
       "DIGI": "Digi Communications", "FP": "Fondul Proprietatea", "M": "MedLife", "ONE": "One United Properties"}


def ds_bvb():
    key = read_secret("EODHD_API_KEY")
    frames = []
    for t in BVB:
        raw = os.path.join(RAW, "eodhd", t + ".json")
        if not os.path.isfile(raw) or REFRESH:
            if not key:
                raise RuntimeError("EODHD key not found")
            os.makedirs(os.path.dirname(raw), exist_ok=True)
            r = requests.get(f"https://eodhd.com/api/eod/{t}.RO", params={"api_token": key, "fmt": "json", "from": "2005-01-01"},
                             timeout=120)
            r.raise_for_status()
            open(raw, "w").write(r.text)
        x = pd.DataFrame(json.load(open(raw)))
        if x.empty:
            continue
        x["ticker"] = t
        frames.append(x)
    df = pd.concat(frames).rename(columns={"adjusted_close": "adjclose"})
    df = df[["date", "ticker", "open", "high", "low", "close", "adjclose", "volume"]].sort_values(["ticker", "date"])
    write_clean(df, "bvb_equities_daily.csv")
    first = df.groupby("ticker")["date"].min()
    return entry("bvb", [("bvb_equities_daily.csv", f"{df['date'].min()} to {df['date'].max()}; listing dates differ by stock")],
                 name={"en": "Bucharest Stock Exchange (BVB): 13 most traded shares (" + ", ".join(BVB) + ")",
                       "ro": "Bursa de Valori București (BVB): 13 dintre cele mai tranzacționate acțiuni (" + ", ".join(BVB) + ")"},
                 source="EODHD (End of Day Historical Data) API", url="https://eodhd.com/",
                 licence="EODHD terms: personal and educational use",
                 cite="EODHD, end-of-day prices of the Bucharest Stock Exchange",
                 lectures=["l5", "l9", "l10", "l13"],
                 note="Prices in RON; adjclose adjusted for splits and dividends. First date per ticker: " +
                      "; ".join(f"{k} {v}" for k, v in first.items()) + ".",
                 freq={"en": "daily (trading days)", "ro": "zilnic (zile de tranzacționare)"})


# --------------------------------------------------------------------------------------------- 7. electricity (shared with EMQA)
def ds_power():
    if not os.path.isdir(EMQA_CLEAN):
        raise RuntimeError("EMQA data folder not found")
    for fn in ("power_da_prices_hourly_utc.csv", "power_da_prices_daily_cet.csv"):
        shutil.copy(os.path.join(EMQA_CLEAN, fn), os.path.join(CLEAN, fn))
        log("  ->", fn, human(os.path.getsize(os.path.join(CLEAN, fn))))
    g = pd.read_csv(os.path.join(EMQA_CLEAN, "generation_ro_mw_hourly_utc.csv"))
    keep = ["timestamp_utc", "load", "nuclear", "hydro_run_of_river", "hydro_water_reservoir", "brown_coal_lignite",
            "hard_coal", "gas", "wind_onshore", "solar", "cross_border_electricity_trading"]
    g = g[[c for c in keep if c in g.columns]]
    g = g.dropna(subset=[c for c in g.columns if c != "timestamp_utc"], how="all")
    path = os.path.join(CLEAN, "ro_load_generation_hourly_utc.csv")
    g.to_csv(path, index=False, float_format="%.1f")
    log("  -> ro_load_generation_hourly_utc.csv", human(os.path.getsize(path)))
    P = pd.read_csv(os.path.join(CLEAN, "power_da_prices_hourly_utc.csv"), usecols=["timestamp_utc"])
    return entry("power", [("power_da_prices_hourly_utc.csv", f"{P.timestamp_utc.iloc[0][:10]} to {P.timestamp_utc.iloc[-1][:10]} (BG from 2016-10-31)"),
                           ("power_da_prices_daily_cet.csv", "same period, CET/CEST calendar days"),
                           ("ro_load_generation_hourly_utc.csv", f"{g.timestamp_utc.iloc[0][:10]} to {g.timestamp_utc.iloc[-1][:10]}")],
                 name={"en": "Day-ahead electricity prices (Romania, Bulgaria, Hungary, Germany-Luxembourg) and Romanian load and generation by source",
                       "ro": "Prețurile energiei electrice pentru ziua următoare (România, Bulgaria, Ungaria, Germania-Luxemburg), consumul și producția României pe surse"},
                 source="energy-charts.info (Fraunhofer ISE), from exchange and ENTSO-E data; same files as the course Energy Markets Quantitative Analysis",
                 url="https://api.energy-charts.info/",
                 licence="Prices: private and internal (teaching) use; generation and load: CC BY 4.0",
                 cite="Fraunhofer ISE, Energy-Charts (energy-charts.info)",
                 lectures=["l5", "l10", "l13"],
                 note=("Timestamps in UTC (interval start). Prices in EUR/MWh: hourly to 30 Sep 2025, then the hourly mean of the "
                       "15-minute prices (mtu_minutes gives the native resolution). Load and generation in mean MW per hour; "
                       "cross_border_electricity_trading > 0 = net import. Missing hours stay missing."),
                 freq={"en": "hourly (UTC) and daily (CET)", "ro": "orar (UTC) și zilnic (CET)"})


# --------------------------------------------------------------------------------------------- 8. ECB yield curve
YC_MAT = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]


def ds_yield():
    cols = {}
    for m in YC_MAT:
        u = f"https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{m}"
        p = fetch(u, f"ecb_yc/SR_{m}.csv", params={"format": "csvdata"})
        x = pd.read_csv(p, usecols=["TIME_PERIOD", "OBS_VALUE"]).set_index("TIME_PERIOD")["OBS_VALUE"]
        cols["y_" + m] = x
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    write_clean(df.reset_index(), "ecb_yield_curve_daily.csv")
    return entry("yield", [("ecb_yield_curve_daily.csv", f"{df.index.min()} to {df.index.max()}")],
                 name={"en": "Euro area AAA government bond yield curve (spot rates, 11 maturities from 3 months to 30 years)",
                       "ro": "Curba randamentelor obligațiunilor guvernamentale AAA din zona euro (rate spot, 11 scadențe, de la 3 luni la 30 de ani)"},
                 source="European Central Bank (ECB) Data Portal, dataset YC (Svensson model, AAA-rated issuers)",
                 url="https://data.ecb.europa.eu/data/datasets/YC",
                 licence="ECB statistics: reuse free of charge with the source acknowledged",
                 cite="European Central Bank, Euro area yield curves, ECB Data Portal (YC)",
                 lectures=["l8", "l13"],
                 note="Percent per year; TARGET business days. Curves are fitted by the ECB with the Svensson (1994) model on AAA-rated euro area central government bonds.",
                 freq={"en": "daily (TARGET days)", "ro": "zilnic (zile TARGET)"})


# --------------------------------------------------------------------------------------------- 9. Financial PhraseBank
def ds_phrasebank():
    p = fetch("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip",
              "phrasebank/FinancialPhraseBank-v1.0.zip")
    rows = []
    with zipfile.ZipFile(p) as z:
        for agree in ("50", "66", "75", "All"):
            name = next(n for n in z.namelist() if n.endswith(f"Sentences_{agree}Agree.txt"))
            for line in z.read(name).decode("latin-1").splitlines():
                if "@" in line:
                    s, lab = line.rsplit("@", 1)
                    rows.append((s.strip(), lab.strip(), agree))
    df = pd.DataFrame(rows, columns=["sentence", "label", "agreement"])
    # one row per sentence, with the highest agreement level it reaches (All > 75 > 66 > 50)
    rank = {"50": 50, "66": 66, "75": 75, "All": 100}
    df["min_agreement_pct"] = df["agreement"].map(rank)
    df = df.sort_values("min_agreement_pct", ascending=False).drop_duplicates("sentence").drop(columns="agreement")
    df = df.sort_values(["min_agreement_pct", "sentence"], ascending=[False, True]).reset_index(drop=True)
    write_clean(df, "financial_phrasebank.csv")
    return entry("phrasebank", [("financial_phrasebank.csv", f"{len(df):,} sentences from English financial news about companies listed in Finland")],
                 name={"en": "Financial PhraseBank: sentiment of sentences from financial news",
                       "ro": "Financial PhraseBank: sentimentul frazelor din știri financiare"},
                 source="Malo et al. (2014), Aalto University; copy on Hugging Face (takala/financial_phrasebank)",
                 url="https://huggingface.co/datasets/takala/financial_phrasebank",
                 licence="CC BY-NC-SA 3.0 (Creative Commons Attribution-NonCommercial-ShareAlike)",
                 cite="Malo, P., Sinha, A., Korhonen, P., Wallenius, J. & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. JASIST, 65(4), 782-796. doi:10.1002/asi.23062",
                 lectures=["l6", "l7", "l12"],
                 note="label in {positive, neutral, negative}, annotated by 5-8 finance and business students; min_agreement_pct = highest share of annotators that agree (50, 66, 75 or 100).",
                 freq={"en": "sentence", "ro": "frază"})


# --------------------------------------------------------------------------------------------- 10. Twitter financial news
def ds_tweets():
    base = "https://huggingface.co/datasets/zeroshot/"
    parts = []
    for split, fn in (("train", "sent_train.csv"), ("valid", "sent_valid.csv")):
        p = fetch(base + "twitter-financial-news-sentiment/resolve/main/" + fn, "tweets/" + fn)
        x = pd.read_csv(p)
        x["split"] = split
        parts.append(x)
    s = pd.concat(parts, ignore_index=True)
    s["label_name"] = s["label"].map({0: "bearish", 1: "bullish", 2: "neutral"})
    write_clean(s, "twitter_fin_sentiment.csv")
    parts = []
    for split, fn in (("train", "topic_train.csv"), ("valid", "topic_valid.csv")):
        p = fetch(base + "twitter-financial-news-topic/resolve/main/" + fn, "tweets/" + fn)
        x = pd.read_csv(p)
        x["split"] = split
        parts.append(x)
    t = pd.concat(parts, ignore_index=True)
    TOP = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt", "Dividend",
           "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion", "Gold | Metals | Materials",
           "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets", "Politics", "Personnel Change",
           "Stock Commentary", "Stock Movement"]
    t["label_name"] = t["label"].map(dict(enumerate(TOP)))
    write_clean(t, "twitter_fin_topic.csv")
    return entry("tweets", [("twitter_fin_sentiment.csv", "11,932 finance tweets, English; split train/valid as published"),
                            ("twitter_fin_topic.csv", "21,107 finance tweets, 20 topics; split train/valid as published")],
                 name={"en": "Twitter financial news: sentiment (bearish, bullish, neutral) and topic (20 classes)",
                       "ro": "Știri financiare de pe Twitter: sentiment (pesimist, optimist, neutru) și temă (20 de clase)"},
                 source="Hugging Face datasets zeroshot/twitter-financial-news-sentiment and -topic",
                 url="https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment",
                 licence="MIT",
                 cite="Zeroshot (2022). Twitter financial news sentiment and topic datasets, Hugging Face",
                 lectures=["l6", "l7", "l12"],
                 note="Text as published (tweets contain links and tickers); labels assigned by the dataset authors.",
                 freq={"en": "tweet", "ro": "tweet"})


# --------------------------------------------------------------------------------------------- 11. ECB speeches
def ds_speeches():
    p = fetch("https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv", "ecb_speeches/all_ECB_speeches.csv")
    df = pd.read_csv(p, sep="|", quoting=3, on_bad_lines="skip", engine="python")
    df = df.dropna(subset=["date", "contents"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_all, d0 = len(df), df["date"].min()
    sub = df[df["date"] >= "2022-01-01"].copy()
    sub["contents"] = sub["contents"].str.replace(r"\s+", " ", regex=True).str.strip()
    sub["n_words"] = sub["contents"].str.split().str.len()
    sub["date"] = sub["date"].dt.strftime("%Y-%m-%d")
    write_clean(sub.sort_values("date"), "ecb_speeches_2022plus.csv")
    return entry("speeches", [("ecb_speeches_2022plus.csv", f"{sub['date'].min()} to {sub['date'].max()}")],
                 name={"en": "Speeches of the European Central Bank (ECB) Executive Board and Supervisory Board, full text",
                       "ro": "Discursurile membrilor Comitetului executiv și ai Consiliului de supraveghere ai Băncii Centrale Europene (BCE), text integral"},
                 source="European Central Bank, speeches dataset", url="https://www.ecb.europa.eu/press/key/html/downloads.en.html",
                 licence="ECB website: reproduction permitted with the source acknowledged",
                 cite="European Central Bank, Speeches dataset (all_ECB_speeches.csv)",
                 lectures=["l12", "l7"],
                 note=(f"The full file has {n_all:,} speeches from {d0:%Y-%m-%d} ({human(os.path.getsize(p))}, kept in data/raw/); "
                       "the committed file keeps the speeches from 1 January 2022 (whitespace normalised, n_words added)."),
                 freq={"en": "speech", "ro": "discurs"})


# --------------------------------------------------------------------------------------------- 12. Elliptic (on demand)
def ds_elliptic():
    base = "https://data.pyg.org/datasets/elliptic/"
    info = {}
    for fn in ("elliptic_txs_features.csv.zip", "elliptic_txs_edgelist.csv.zip", "elliptic_txs_classes.csv.zip"):
        p = fetch(base + fn, "elliptic/" + fn, timeout=900)
        info[fn] = os.path.getsize(p)
    with zipfile.ZipFile(os.path.join(RAW, "elliptic", "elliptic_txs_classes.csv.zip")) as z:
        cl = pd.read_csv(z.open(z.namelist()[0]))
    with zipfile.ZipFile(os.path.join(RAW, "elliptic", "elliptic_txs_edgelist.csv.zip")) as z:
        ed = pd.read_csv(z.open(z.namelist()[0]))
    vc = cl["class"].astype(str).value_counts()
    key, e = "elliptic", dict(
        name={"en": "Elliptic Bitcoin transaction graph (illicit vs licit transactions)",
              "ro": "Graful tranzacțiilor Bitcoin Elliptic (tranzacții ilicite și licite)"},
        source="Elliptic, released with Weber et al. (2019); files served by PyTorch Geometric (EllipticBitcoinDataset)",
        url="https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
        licence="Released by Elliptic for research and teaching (see the Kaggle page); not redistributed here",
        cite="Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T. & Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. KDD Workshop on Anomaly Detection in Finance. arXiv:1908.02591",
        lectures=["l9"],
        note=(f"Download on demand: {len(cl):,} transactions (nodes), {len(ed):,} payment flows (edges), 49 time steps, "
              f"166 features; labels: illicit {vc.get('1', 0):,}, licit {vc.get('2', 0):,}, unknown {vc.get('unknown', 0):,}. "
              f"About {human(sum(info.values()))} zipped: not committed. python3 data/download_all.py elliptic puts the files in "
              "data/raw/elliptic/; torch_geometric.datasets.EllipticBitcoinDataset downloads the same files."),
        freq={"en": "transaction (node)", "ro": "tranzacție (nod)"}, downloaded=TODAY, files=[], on_demand=True)
    return key, e


# --------------------------------------------------------------------------------------------- 13. images (on demand)
def ds_vision():
    import torchvision
    root = os.path.join(RAW, "vision")
    os.makedirs(root, exist_ok=True)
    tr = torchvision.datasets.FashionMNIST(root, train=True, download=True)
    te = torchvision.datasets.FashionMNIST(root, train=False, download=True)
    c10 = torchvision.datasets.CIFAR10(root, train=True, download=True)
    c10t = torchvision.datasets.CIFAR10(root, train=False, download=True)
    eu = torchvision.datasets.EuroSAT(root, download=True)
    common = dict(downloaded=TODAY, files=[], on_demand=True, freq={"en": "image", "ro": "imagine"})
    out = [
        ("fashion_mnist", dict(common,
            name={"en": "Fashion-MNIST: Zalando product images (10 clothing categories)", "ro": "Fashion-MNIST: imagini de produse Zalando (10 categorii de îmbrăcăminte)"},
            source="Zalando Research", url="https://github.com/zalandoresearch/fashion-mnist", licence="MIT",
            cite="Xiao, H., Rasul, K. & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747",
            lectures=["l4", "l8"],
            note=f"Download on demand (torchvision.datasets.FashionMNIST): {len(tr):,} training and {len(te):,} test images, 28 x 28 grey levels, from the Zalando online catalogue.")),
        ("cifar10", dict(common,
            name={"en": "CIFAR-10: 60,000 colour photographs in 10 classes", "ro": "CIFAR-10: 60.000 de fotografii color în 10 clase"},
            source="Canadian Institute for Advanced Research; University of Toronto", url="https://www.cs.toronto.edu/~kriz/cifar.html",
            licence="Free for research and teaching (as stated by the authors)",
            cite="Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto",
            lectures=["l4"],
            note=f"Download on demand (torchvision.datasets.CIFAR10): {len(c10):,} training and {len(c10t):,} test images, 32 x 32 RGB (red, green, blue).")),
        ("eurosat", dict(common,
            name={"en": "EuroSAT: Sentinel-2 satellite images of land use in Europe (10 classes)", "ro": "EuroSAT: imagini satelitare Sentinel-2 ale utilizării terenurilor din Europa (10 clase)"},
            source="Helber et al. (2019), from Copernicus Sentinel-2 data", url="https://github.com/phelber/EuroSAT", licence="MIT",
            cite="Helber, P., Bischke, B., Dengel, A. & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE JSTARS, 12(7), 2217-2226. doi:10.1109/JSTARS.2019.2918242",
            lectures=["l4", "l14"],
            note=f"Download on demand (torchvision.datasets.EuroSAT): {len(eu):,} images of 64 x 64 pixels (RGB bands), classes such as annual crop, industrial, residential, river; used for transfer learning and zero-shot vision-language classification.")),
    ]
    return out


# --------------------------------------------------------------------------------------------- SOURCES.md
ORDER = ["fraud", "german", "hmeq", "california", "markets", "bvb", "power", "yield", "phrasebank", "tweets", "speeches",
         "elliptic", "fashion_mnist", "cifar10", "eurosat"]


def write_sources(M):
    L = ["# Data sources of the course *Neural Networks and Deep Learning*", "",
         "Generated by `data/download_all.py` from `data/clean/MANIFEST.json`; do not edit by hand. Every dataset is real and "
         "public; nothing is simulated, interpolated or gap-filled. Files in `data/clean/` are committed and linked from the "
         "Data page of the course site; datasets marked *on demand* are downloaded by the script (or by the notebook) into "
         "`data/raw/`, which is not committed.", "",
         "| Dataset | File(s) | Rows | Size | Coverage | Source | Licence | Downloaded | Lectures |", "|---|---|---|---|---|---|---|---|---|"]
    for k in [k for k in ORDER if k in M] + [k for k in M if k not in ORDER]:
        e = M[k]
        lec = ", ".join(x.upper() for x in e["lectures"])
        if e.get("files"):
            for i, f in enumerate(e["files"]):
                L.append(f"| {e['name']['en'] if i == 0 else '〃'} | `{f['file']}` | {f['rows']:,} | {human(f['size'])} | {f['coverage']} | "
                         f"[{e['source']}]({e['url']}) | {e['licence']} | {e['downloaded']} | {lec} |")
        else:
            L.append(f"| {e['name']['en']} | on demand | | | | [{e['source']}]({e['url']}) | {e['licence']} | {e['downloaded']} | {lec} |")
    L += ["", "## Details", ""]
    for k in [k for k in ORDER if k in M] + [k for k in M if k not in ORDER]:
        e = M[k]
        L += [f"### {e['name']['en']}", "",
              f"- **Source:** [{e['source']}]({e['url']})",
              f"- **Licence:** {e['licence']}",
              f"- **Cite as:** {e['cite']}",
              f"- **Unit of observation:** {e['freq']['en']}",
              f"- **Lectures:** {', '.join(x.upper() for x in e['lectures'])}",
              f"- **Notes:** {e['note']}"]
        for f in e.get("files", []):
            L.append(f"- **`data/clean/{f['file']}`** ({f['rows']:,} rows, {human(f['size'])}; {f['coverage']}): columns "
                     + ", ".join(f"`{c}`" for c in f["columns"]))
        L.append("")
    open(os.path.join(ROOT, "SOURCES.md"), "w", encoding="utf-8").write("\n".join(L))
    log("wrote data/SOURCES.md")


DATASETS = {"fraud": ds_fraud, "german": ds_german, "hmeq": ds_hmeq, "california": ds_california, "markets": ds_markets,
            "bvb": ds_bvb, "power": ds_power, "yield": ds_yield, "phrasebank": ds_phrasebank, "tweets": ds_tweets,
            "speeches": ds_speeches, "elliptic": ds_elliptic, "vision": ds_vision}

if __name__ == "__main__":
    M = manifest_load()
    if "--sources-only" not in sys.argv:
        names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(DATASETS)
        for n in names:
            log("==", n)
            try:
                res = DATASETS[n]()
                for k, e in (res if isinstance(res, list) else [res]):
                    M[k] = e
                manifest_save(M)
            except Exception as ex:          # reported on the console only; nothing is written for a failed dataset
                log(f"  {n}: FAILED ({type(ex).__name__}: {str(ex)[:200]})")
    write_sources(M)
