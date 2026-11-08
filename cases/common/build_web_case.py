"""Web version of an NN research case study: one page per language, built from the case brief.

usage:  cd cases && python3 common/build_web_case.py            # all cases C1-C8
        python3 common/build_web_case.py C3                     # one case
writes  web/cases/C{k}.html (EN), web/cases/C{k}_ro.html (RO), web/cases/img/C{k}/<figure>_<lang>.png
        and a log web/cases/C{k}_log.txt (LaTeX commands left unconverted, acronyms, Romanian spelling checks).

The text is the case brief itself: cases/<folder>/C{k}_brief.tex (one source, \\T{EN}{RO}), with the numbers of the notebook
(numbers_<lang>.tex, written from numbers.json) and its tables (tab_<name>_<lang>.tex). Nothing is typed twice: when the brief
or the notebook changes, rebuild the pages. The figures are the notebook's PDF charts (fig/<lang>/*.pdf) rendered to PNG on a
white background, so they read the same in light and dark browser themes.
The page layout is that of the web lectures (web/LK.html, lectures/common/build_web_lecture.py): web/web.css + web/cases/case.css.
"""
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.normpath(os.path.join(HERE, '..'))
ROOT = os.path.normpath(os.path.join(CASES, '..'))
OUT = os.path.join(ROOT, 'web', 'cases')
REPO = 'https://github.com/danpele/NEURAL_BIZ'
COLAB = 'https://colab.research.google.com/github/danpele/NEURAL_BIZ/blob/main/'
MATHJAX = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js'
VER = '20261108a'

# ------------------------------------------------------------------ the cases
LECT = {1: ('Deep learning for business: from linear models to the multilayer perceptron', 'Deep learning pentru afaceri: de la modelele liniare la perceptronul multistrat'),
        2: ('Training: backpropagation and optimisation', 'Antrenarea: propagarea înapoi a erorii și optimizarea'),
        3: ('Regularisation and generalisation', 'Regularizare și generalizare'),
        4: ('Convolutional neural networks and transfer learning', 'Rețele neuronale convoluționale și transferul învățării'),
        5: ('Recurrent networks for sequences and financial time series', 'Rețele recurente pentru secvențe și serii de timp financiare'),
        6: ('NLP foundations: text representation and word embeddings', 'Bazele NLP: reprezentarea textului și scufundările cuvintelor'),
        7: ('Attention and transformers', 'Atenția și arhitectura transformer'),
        8: ('Generative models', 'Modele generative'),
        9: ('Graph neural networks', 'Rețele neuronale pe grafuri'),
        10: ('Reinforcement learning', 'Învățarea prin recompensă'),
        11: ('Explainability (XAI)', 'Explicabilitate (XAI)'),
        12: ('Large language models in finance', 'Modele lingvistice mari în finanțe'),
        13: ('Deep learning for energy and financial time series', 'Deep learning pentru serii de timp din energie și finanțe'),
        14: ('Multimodal models and course synthesis', 'Modele multimodale și sinteza cursului')}

PICBE = 'Proceedings of the International Conference on Business Excellence'
SSRN = {'en': 'SSRN (Social Science Research Network) Working Paper', 'ro': 'SSRN (Social Science Research Network), document de lucru'}
# Research case studies: one entry per case folder cases/C<k>_<slug>/ (filled when the case is written). Example:
#   1: dict(folder='C1_fraud_mlp', lectures=[1, 3],
#           authors='Surname, A. & Pele, D. T.', year=2026, title='...', venue='Journal, vol(issue), pages',
#           link='https://doi.org/...', linktext='doi:...'),
ESWA = 'Expert Systems with Applications'
CASE = {
    1: dict(folder='C1_fraud_xai', lectures=[3, 11, 1],
            authors='Diaconu, D. & Pele, D. T.', year=2026,
            title='Explainable AI in financial fraud detection: Evidence from the IEEE-CIS fraud dataset',
            venue=PICBE + ', 20(1), 118–132', link='https://doi.org/10.2478/picbe-2026-0015', linktext='doi:10.2478/picbe-2026-0015'),
    2: dict(folder='C2_spike_recurrent', lectures=[5, 11],
            authors='Andrei, A.-V., Pele, D.-T., Cramer, A.-A., Amza, A. & Diaconu, D.', year=2026,
            title='Explainable AI-based early warning system for electricity price spikes: Evidence from the Romanian market',
            venue=PICBE + ', 20(1), 4820–4836', link='https://doi.org/10.2478/picbe-2026-0352', linktext='doi:10.2478/picbe-2026-0352'),
    3: dict(folder='C3_shap_stability', lectures=[9, 11],
            authors='Diaconu, D. & Pele, D. T.', year=2026,
            title='SHAP stability across random and forward-time validation in graph-augmented fraud detection',
            venue='submitted to ' + ESWA, link='https://github.com/QuantLet/GNN_XAI_Stability', linktext='code: github.com/QuantLet/GNN_XAI_Stability'),
    4: dict(folder='C4_battery_arbitrage', lectures=[10, 13],
            authors='Andrei, A.-V., Cepoi, C. O., Dumitrescu, B. A., Lessmann, S., Chiriac (Luchian), A.-C., Pele, D. T. & Petukhina, A.', year=2026,
            title='Joint probabilistic forecasting for battery arbitrage in day-ahead power markets',
            venue='SSRN 6872381', link='https://doi.org/10.2139/ssrn.6872381', linktext='doi:10.2139/ssrn.6872381'),
    5: dict(folder='C5_llm_var', lectures=[12],
            authors='Pele, D. T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A. T., Spilak, B., Andrei, A.-V., Toma, F.-M., Lessmann, S. & Härdle, W. K.', year=2026,
            title='In the beginning was the Word: LLM-VaR and LLM-ES',
            venue=ESWA + ', 295, 128676', link='https://doi.org/10.1016/j.eswa.2025.128676', linktext='doi:10.1016/j.eswa.2025.128676'),
    6: dict(folder='C6_foundation_risk', lectures=[13],
            authors='Jheng, S.-L., Tak, R., Găman, Ș., Mazurencu-Marinescu-Pele, M. & Pele, D. T.', year=2026,
            title='Can foundation models manage risk? Zero-shot VaR and ES forecasting with conformal calibration in CEE markets',
            venue='Economic Computation and Economic Cybernetics Studies and Research, 60(2), 267–290',
            link='https://doi.org/10.24818/18423264/60.2.26.14', linktext='doi:10.24818/18423264/60.2.26.14'),
    7: dict(folder='C7_vlm_anomaly', lectures=[14],
            authors='Jheng, S.-L., Pele, D. T., Tak, R. & Găman, Ș.', year=2026,
            title='A multimodal vision-language framework for financial anomaly detection',
            venue=PICBE + ', 20(1), 191–201', link='https://doi.org/10.2478/picbe-2026-0020', linktext='doi:10.2478/picbe-2026-0020'),
}

L = {
    'en': dict(course='Neural Networks and Deep Learning', back='Course site', site='../../index.html#cases', case='Case study',
               lecture='Lecture', brief='Case brief (PDF)', nb='Notebook', colab='Open in Colab', toc='Contents',
               question='The business question', intuition='Intuition first', decision='In practice', figure='Figure', table='Table',
               src='Source', glance=['Question', 'Data', 'Method', 'Result', 'Decision'], author='Prof. dr. Daniel Traian PELE',
               inst=['Bucharest University of Economic Studies, Department of Statistics and Econometrics',
                     'Romanian Academy, Institute for Economic Forecasting'],
               paper='The paper', paper_none='Case study on the course data; no single paper is replicated.',
               materials='Materials', mat_brief='Case brief', mat_nb='Notebook on GitHub', mat_colab='Open in Colab',
               lang_en='English', lang_ro='Romanian', related='Related lecture', related_pl='Related lectures',
               abbr='Abbreviations', enlarge='Open the chart at full size',
               foot='The text, numbers, charts and tables of this page are those of the case brief; every number is computed by the '
                    'case notebook on the course data.',
               all_cases='All case studies', note_c8='This case is not built on a paper of the group; it is a descriptive analysis on real data.'),
    'ro': dict(course='Rețele neuronale și deep learning', back='Site-ul cursului', site='../../index_ro.html#cases', case='Studiul de caz',
               lecture='Cursul', brief='Studiul de caz (PDF)', nb='Notebook', colab='Deschide în Colab', toc='Cuprins',
               question='Problema decizională', intuition='Intuiția metodei', decision='În practică', figure='Figura', table='Tabelul',
               src='Sursa', glance=['Întrebare', 'Date', 'Metodă', 'Rezultat', 'Decizie'], author='Prof. dr. Daniel Traian PELE',
               inst=['Academia de Studii Economice din București, Departamentul de Statistică și Econometrie',
                     'Academia Română, Institutul de Prognoză Economică'],
               paper='Lucrarea de referință', paper_none='Studiu de caz pe datele cursului; nu replică o anumită lucrare.',
               materials='Materiale', mat_brief='Studiul de caz', mat_nb='Notebook pe GitHub', mat_colab='Deschide în Colab',
               lang_en='engleză', lang_ro='română', related='Cursul asociat', related_pl='Cursurile asociate',
               abbr='Abrevieri', enlarge='Deschide graficul la dimensiunea completă',
               foot='Textul, cifrele, graficele și tabelele acestei pagini sînt cele ale studiului de caz; toate cifrele sînt calculate de '
                    'notebook-ul studiului pe datele cursului.',
               all_cases='Toate studiile de caz', note_c8='Acest studiu de caz nu se bazează pe o lucrare a grupului; este o analiză descriptivă pe date reale.'),
}

# acronyms that the brief may use without spelling them out on the page: the first use gets the expansion in brackets
# and all acronyms found on the page are listed under "Abbreviations"
ACR = {
    'ADF': ('Augmented Dickey–Fuller', 'Dickey–Fuller augmentat'),
    'AI': ('artificial intelligence', 'inteligență artificială'),
    'AR': ('autoregressive', 'autoregresiv'),
    'ARIMA': ('autoregressive integrated moving average', 'model autoregresiv integrat cu medie mobilă'),
    'ARMA': ('autoregressive moving average', 'model autoregresiv cu medie mobilă'),
    'AUC': ('area under the ROC curve, ROC = receiver operating characteristic', 'aria de sub curba ROC, ROC = receiver operating characteristic'),
    'ROC': ('receiver operating characteristic', 'receiver operating characteristic'),
    'BCE': (None, 'Banca Centrală Europeană'),
    'BG': ('Bulgaria', 'Bulgaria'),
    'CCC': ('constant conditional correlation', 'corelație condiționată constantă'),
    'CDS': ('credit default swap', 'credit default swap'),
    'CET': ('Central European Time', 'ora Europei Centrale'),
    'CISS': ('Composite Indicator of Systemic Stress', 'indicatorul compozit de stres sistemic'),
    'CoVaR': ('conditional Value-at-Risk', 'Value-at-Risk condiționat'),
    'DCC': ('dynamic conditional correlation', 'corelație condiționată dinamică'),
    'DE': ('Germany', 'Germania'),
    'DE-LU': ('Germany-Luxembourg', 'Germania-Luxemburg'),
    'DM': ('Diebold–Mariano', 'Diebold–Mariano'),
    'DR': ('demand response', 'răspunsul cererii'),
    'ECB': ('European Central Bank', None),
    'EEX': ('European Energy Exchange', 'European Energy Exchange'),
    'EFRM': ('Energy Financial Risk Meter', 'Energy Financial Risk Meter'),
    'ENTSO-E': ('European Network of Transmission System Operators for Electricity',
                'Rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică'),
    'ES': ('Expected Shortfall', 'Expected Shortfall'),
    'EU': ('European Union', None),
    'EUA': ('EU emission allowance', 'certificat de emisii al UE'),
    'EUR': (None, None),
    'GARCH': ('generalised autoregressive conditional heteroskedasticity', 'heteroscedasticitate condiționată autoregresivă generalizată'),
    'GPRD': ('daily geopolitical risk index', 'indicele zilnic de risc geopolitic'),
    'HAC': ('heteroskedasticity- and autocorrelation-consistent', 'robust la heteroscedasticitate și autocorelare'),
    'HU': ('Hungary', 'Ungaria'),
    'LSEG': ('London Stock Exchange Group', 'London Stock Exchange Group'),
    'MAE': ('mean absolute error', 'eroarea absolută medie'),
    'MAPE': ('mean absolute percentage error', 'eroarea procentuală absolută medie'),
    'MCMMP': (None, 'metoda celor mai mici pătrate'),
    'ML': ('machine learning', 'învățare automată'),
    'MTU': ('market time unit', 'interval de tranzacționare al pieței'),
    'OLS': ('ordinary least squares', None),
    'OTS': (None, 'operatorul de transport și de sistem'),
    'PPA': ('power purchase agreement', 'contract de achiziție de energie pe termen lung'),
    'PPS': ('purchasing power standards', 'standarde ale puterii de cumpărare'),
    'PZU': ('Piața pentru Ziua Următoare, the Romanian day-ahead market', 'piața pentru ziua următoare'),
    'RF': ('Random Forest', 'Random Forest'),
    'RMSE': ('root mean squared error', 'rădăcina erorii pătratice medii'),
    'RO': ('Romania', 'România'),
    'SARIMA': ('seasonal ARIMA', 'ARIMA sezonier'),
    'SHAP': ('SHapley Additive exPlanations', 'SHapley Additive exPlanations'),
    'SRMC': ('short-run marginal cost', 'costul marginal pe termen scurt'),
    'TSO': ('transmission system operator', None),
    'TTF': ('Title Transfer Facility', 'Title Transfer Facility'),
    'TVA': (None, 'taxa pe valoarea adăugată'),
    'UE': (None, 'Uniunea Europeană'),
    'VaR': ('Value-at-Risk', 'Value-at-Risk'),
    'VAT': ('value added tax', None),
    'XGBoost': ('extreme gradient boosting', 'extreme gradient boosting'),
    'MSE': ('mean squared error', 'eroarea pătratică medie'),
    'NHITS': ('Neural Hierarchical Interpolation for Time Series', 'Neural Hierarchical Interpolation for Time Series'),
    'NHiTS': ('Neural Hierarchical Interpolation for Time Series', 'Neural Hierarchical Interpolation for Time Series'),
    'TimesFM': ('Time Series Foundation Model', 'Time Series Foundation Model'),
    'MLP': ('multilayer perceptron', 'perceptron multistrat'),
    'LSTM': ('Long Short-Term Memory', 'Long Short-Term Memory'),
    'CEE': ('Central and Eastern Europe', None),
    'ECE': (None, 'Europa Centrală și de Est'),
    'BET': ('Bucharest Exchange Trading, the main index of the Bucharest Stock Exchange', 'Bucharest Exchange Trading, indicele principal al Bursei de Valori București'),
    'DCC-GARCH': ('dynamic conditional correlation GARCH model', 'model GARCH cu corelație condiționată dinamică'),
    'TVP-VAR': ('time-varying parameter vector autoregression', 'model vectorial autoregresiv cu parametri variabili în timp'),
    'CC': ('Creative Commons', 'Creative Commons'),
    'SNP': ('stock-exchange ticker of OMV Petrom', 'simbolul bursier al OMV Petrom'),
    'SNG': ('stock-exchange ticker of Romgaz', 'simbolul bursier al Romgaz'),
    'SNN': ('stock-exchange ticker of Nuclearelectrica', 'simbolul bursier al Nuclearelectrica'),
    'EODHD': ('EOD Historical Data, a market-data provider', 'EOD Historical Data, furnizor de date de piață'),
    'ICE': ('Intercontinental Exchange', 'Intercontinental Exchange'),
    'VSTOXX': ('volatility index of the EURO STOXX 50', 'indicele de volatilitate al EURO STOXX 50'),
    'COVID': ('coronavirus disease', 'boala cauzată de coronavirus'),
    'LASSO': ('Least Absolute Shrinkage and Selection Operator', 'Least Absolute Shrinkage and Selection Operator'),
    'GACV': ('generalised approximate cross-validation', 'validare încrucișată aproximativă generalizată'),
    'RO-ERM': ('energy risk meter for Romanian companies', 'indicator de risc energetic pentru companiile din România'),
    'SAR': ('seasonal autoregressive term', 'termen autoregresiv sezonier'),
    'MA': ('moving-average term', 'termen de medie mobilă'),
    'SMA': ('seasonal moving-average term', 'termen sezonier de medie mobilă'),
    'VAR': ('vector autoregression', 'model vectorial autoregresiv'),
    'EV': ('electric vehicle', 'vehicul electric'),
    'US': ('United States', None),
    'SUA': (None, 'Statele Unite ale Americii'),
    'MCO': (None, 'metoda celor mai mici pătrate'),
    'BESS': ('battery energy storage system', 'sistem de stocare a energiei în baterii'),
    'CAPEX': ('capital expenditure', 'cheltuieli de capital'),
    'DRF': ('Distributional Random Forests', 'Distributional Random Forests'),
    'QB': ('quantile-based', 'bazat pe cuantile'),
    'NH': ('naive fixed pair of hours', 'pereche fixă de ore'),
    'CRPS': ('continuous ranked probability score', 'scorul de probabilitate clasat continuu'),
    'TFT': ('Temporal Fusion Transformer', 'Temporal Fusion Transformer'),
    'NPV': ('net present value', None),
    'VAN': (None, 'valoarea actualizată netă'),
    'XAI': ('explainable artificial intelligence', 'inteligență artificială explicabilă'),
    'LIME': ('Local Interpretable Model-agnostic Explanations', 'Local Interpretable Model-agnostic Explanations'),
    'ECECSR': ('Economic Computation and Economic Cybernetics Studies and Research', 'Economic Computation and Economic Cybernetics Studies and Research'),
    'PCCB-LE': ('OPCOM centralised bilateral market, extended auction', 'piața centralizată a contractelor bilaterale OPCOM, licitație extinsă'),
    'PCCB-NC': ('OPCOM centralised bilateral market, continuous negotiation', 'piața centralizată a contractelor bilaterale OPCOM, negociere continuă'),
    'PC-OTC': ('OPCOM centralised market with continuous double negotiation', 'piața centralizată OPCOM cu negociere dublă continuă'),
    'RON': ('Romanian leu', 'leul românesc'),
    'DC': ('Eurostat household consumption band, 2,500–4,999 kWh per year', 'banda de consum casnic Eurostat, 2.500–4.999 kWh pe an'),
    'ID': ('Eurostat non-household consumption band, 2,000–19,999 MWh per year', 'banda de consum noncasnic Eurostat, 2.000–19.999 MWh pe an'),
    'SSRN': ('Social Science Research Network', 'Social Science Research Network'),
    'OTC': ('over the counter', 'extrabursier'),
    'OPCOM': ('the Romanian power and gas market operator', 'operatorul pieței de energie electrică și de gaze naturale din România'),
    'VIF': ('variance inflation factor', 'factorul de inflație a varianței'),
    'PICBE': ('Proceedings of the International Conference on Business Excellence', 'Proceedings of the International Conference on Business Excellence'),
    'JSES': ('Journal of Social and Economic Statistics', 'Journal of Social and Economic Statistics'),
    'ULB': ('Université Libre de Bruxelles', 'Université Libre de Bruxelles'),
    'IEEE-CIS': ('Institute of Electrical and Electronics Engineers, Computational Intelligence Society', 'Institute of Electrical and Electronics Engineers, Computational Intelligence Society'),
    'ROC-AUC': ('area under the receiver operating characteristic curve', 'aria de sub curba ROC (receiver operating characteristic)'),
    'PR-AUC': ('area under the precision–recall curve', 'aria de sub curba precizie–sensibilitate'),
    'GraphSAGE': ('graph sample and aggregate', 'eșantionare și agregare pe graf'),
    'ReLU': ('rectified linear unit', 'unitate liniară rectificată'),
    'AML': ('anti-money laundering', 'combaterea spălării banilor'),
    'GNN': ('graph neural network', 'rețea neuronală pe grafuri'),
    'JS': ('Jensen–Shannon', 'Jensen–Shannon'),
    'TreeSHAP': ('SHAP for tree ensembles', 'SHAP pentru ansambluri de arbori'),
    'TSFM': ('time series foundation model', 'model fundație pentru serii de timp'),
    'BVB': ('Bucharest Stock Exchange', 'Bursa de Valori București'),
    'HS': ('historical simulation', 'simulare istorică'),
    'LLM': ('large language model', 'model lingvistic mare'),
    'EWMA': ('exponentially weighted moving average', 'medie mobilă ponderată exponențial'),
    'DCS': ('Dynamic Conditional Score', 'Dynamic Conditional Score'),
    'SI': (None, 'simulare istorică'),
    'GJR-GARCH': ('Glosten–Jagannathan–Runkle GARCH model', 'modelul GARCH Glosten–Jagannathan–Runkle'),
    'FZ0': ('Fissler–Ziegel loss of Patton, Ziegel and Chen (2019)', 'funcția de pierdere Fissler–Ziegel a lui Patton, Ziegel și Chen (2019)'),
    'TLV': ('stock-exchange ticker of Banca Transilvania', 'simbolul bursier al Banca Transilvania'),
    'BRD': ('stock-exchange ticker of BRD – Groupe Société Générale', 'simbolul bursier al BRD – Groupe Société Générale'),
    'TGN': ('stock-exchange ticker of Transgaz', 'simbolul bursier al Transgaz'),
    'EL': ('stock-exchange ticker of Electrica', 'simbolul bursier al Electrica'),
    'TEL': ('stock-exchange ticker of Transelectrica', 'simbolul bursier al Transelectrica'),
    'GRU': ('Gated Recurrent Unit', 'Gated Recurrent Unit'),
    'VLM': ('vision-language model', 'model vizual-lingvistic'),
    'CLIP': ('Contrastive Language-Image Pre-training', 'Contrastive Language-Image Pre-training'),
    'MACD': ('moving average convergence divergence', 'moving average convergence divergence'),
    'RSI': ('relative strength index', 'relative strength index'),
    'GPT': ('generative pre-trained transformer', 'transformer generativ preantrenat'),
    'ICBE': ('International Conference on Business Excellence', 'Conferința internațională Business Excellence'),
    'EMA': ('exponential moving average', 'medie mobilă exponențială'),
    'ViT': ('vision transformer', 'vision transformer'),
    'FINDER-LLM': ('FINancial Anomaly DEtection and Recognition using Large Language Models', 'FINancial Anomaly DEtection and Recognition using Large Language Models'),
    'CI': ('confidence interval', None),
}
# described in words next to their first use (or band definitions given with their numbers): listed, never expanded in the text
NOINS = {'PCCB-LE', 'PCCB-NC', 'PC-OTC', 'DC', 'ID', 'SNP', 'SNG', 'SNN', 'NH', 'QB', 'TLV', 'BRD', 'TGN', 'EL', 'TEL', 'FINDER-LLM', 'ROC-AUC', 'PR-AUC', 'ReLU', 'TreeSHAP', 'CI', 'EMA'}


class Stop(Exception):
    pass


# ------------------------------------------------------------------ LaTeX helpers
def strip_comments(s):
    out = []
    for line in s.split('\n'):
        m = re.search(r'(?<!\\)%', line)
        out.append(line if not m else line[:m.start()])
    return '\n'.join(out)


def group(s, i, open_='{', close='}'):
    """Content of the {...} group starting at s[i] (after whitespace) and the index after it."""
    while i < len(s) and s[i] in ' \t\n':
        i += 1
    if i >= len(s) or s[i] != open_:
        return None, i
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise Stop(f'unclosed group at: {s[i:i + 80]!r}')


def opt(s, i):
    """Skip an optional [..] argument."""
    j = i
    while j < len(s) and s[j] in ' \t\n':
        j += 1
    if j < len(s) and s[j] == '[':
        return s.index(']', j) + 1
    return i


def args(s, i, n):
    out = []
    for _ in range(n):
        g, i = group(s, i)
        if g is None:
            raise Stop(f'missing argument at: {s[i:i + 80]!r}')
        out.append(g)
    return out, i


def pick_lang(s, lang):
    """Resolve every \\T{en}{ro}."""
    pat = re.compile(r'\\T(?![a-zA-Z])')
    while True:
        m = pat.search(s)
        if not m:
            return s
        (a, b), j = args(s, m.end(), 2)
        s = s[:m.start()] + pick_lang(a if lang == 'en' else b, lang) + s[j:]


def read_macros(path):
    """\\newcommand{\\Name}{value} lines (zero-argument macros)."""
    out = {}
    src = strip_comments(open(path, encoding='utf-8').read())
    for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}', src):
        if src[m.end():m.end() + 1] == '[':
            continue
        g, _ = group(src, m.end())
        out[m.group(1)] = g
    return out


def expand(s, macros):
    for _ in range(4):
        for name in sorted(macros, key=len, reverse=True):
            s = re.sub(r'\\' + name + r'(?![a-zA-Z])(\{\})?', lambda m: macros[name], s)
    return s


# ------------------------------------------------------------------ conversion
class Conv:
    def __init__(self, k, lang):
        self.k, self.lang, self.T = k, lang, L[lang]
        c = CASE[k]
        self.dir = os.path.join(CASES, c['folder'])
        self.folder = c['folder']
        self.math, self.blocks, self.links = [], [], []
        self.labels, self.nfig, self.ntab = {}, 0, 0
        self.log = []

    # placeholders
    def blk(self, h):
        self.blocks.append(h)
        return f'\n\n\x01{len(self.blocks) - 1}\x01\n\n'

    def protect_math(self, s):
        s = re.sub(r'\\\\\[[^\]]*\]', r'\\\\', s)                 # \\[4pt] -> \\
        s = re.sub(r'(?<!\\)\\\[(.*?)\\\]', lambda m: self.blk('<div class="dm">\\[' + html.escape(m.group(1).strip(), quote=False) + '\\]</div>'),
                   s, flags=re.S)
        s = re.sub(r'(?<!\\)\$(.+?)(?<!\\)\$', lambda m: self.keep_math(m.group(1)), s, flags=re.S)
        return s

    def keep_math(self, t):
        self.math.append(t)
        return f'\x02{len(self.math) - 1}\x02'

    def restore(self, h):
        for _ in range(6):
            h2 = re.sub(r'\x01(\d+)\x01', lambda m: self.blocks[int(m.group(1))], h)
            h2 = re.sub(r'\x03(\d+)\x03', lambda m: self.links[int(m.group(1))], h2)
            if h2 == h:
                break
            h = h2
        h = re.sub(r'\x02(\d+)\x02', lambda m: '<span class="im">\\(' + html.escape(re.sub(r'\s+', ' ', self.math[int(m.group(1))]).strip(),
                                                                                     quote=False) + '\\)</span>', h)
        h = re.sub(r'\x04REF:([^\x04]+)\x04', lambda m: str(self.labels.get(m.group(1), '??')), h)
        return h

    def link(self, h):
        self.links.append(h)
        return f'\x03{len(self.links) - 1}\x03'

    # inline text -> HTML (math already protected)
    def inline(self, s):
        s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        s = s.replace('\\&amp;', '&amp;')

        def url_clean(u):
            return u.replace('\\_', '_').replace('\\#', '#').replace('\\%', '%').replace('&amp;', '&').strip()

        def sub_cmd(name, n, fn):
            nonlocal s
            pat = re.compile(r'\\' + name + r'(?![a-zA-Z])')
            while True:
                m = pat.search(s)
                if not m:
                    return
                a, j = args(s, opt(s, m.end()), n)
                s = s[:m.start()] + fn(*a) + s[j:]

        sub_cmd('href', 2, lambda u, t: self.link(f'<a href="{html.escape(url_clean(u))}" target="_blank" rel="noopener">{self.inline_raw(t)}</a>'))
        sub_cmd('url', 1, lambda u: self.link(f'<a href="{html.escape(url_clean(u))}" target="_blank" rel="noopener">'
                                              f'{html.escape(url_clean(u))}</a>'))
        sub_cmd('ref', 1, lambda x: f'\x04REF:{x}\x04')
        sub_cmd('label', 1, lambda x: '')
        sub_cmd('color', 1, lambda x: '')
        sub_cmd('vspace', 1, lambda x: '')
        sub_cmd('hspace', 1, lambda x: '')

        def tt(x):
            t = x.replace('\\_', '_')
            m = re.fullmatch(r'C(\d)_notebook_(EN|RO)\.ipynb', t)
            if m:
                return self.link(f'<a href="{REPO}/blob/main/cases/{self.folder}/{t}" target="_blank" rel="noopener"><code>{t}</code></a>')
            if t in ('data/SOURCES.md', 'download_c8_data.py'):
                p = 'data/SOURCES.md' if t == 'data/SOURCES.md' else f'cases/{self.folder}/{t}'
                return self.link(f'<a href="{REPO}/blob/main/{p}" target="_blank" rel="noopener"><code>{t}</code></a>')
            return f'<code>{self.inline_raw(x)}</code>'
        sub_cmd('texttt', 1, tt)
        for name, tag in (('textbf', 'strong'), ('emph', 'em'), ('textit', 'em'), ('mathit', 'em')):
            sub_cmd(name, 1, lambda x, tag=tag: f'<{tag}>{x}</{tag}>')
        sub_cmd('text', 1, lambda x: x)
        s = re.sub(r'\\\\(\[[^\]]*\])?', '<br>', s)
        s = s.replace('---', '—').replace('--', '–').replace('``', '“').replace("''", '”')
        s = re.sub(r'\\,', '\u202f', s)
        s = s.replace('~', '\u00a0')
        s = re.sub(r'\\ ', ' ', s)
        for a, b in (('\\%', '%'), ('\\_', '_'), ('\\#', '#'), ('\\$', '$'), ('\\textbullet', '•'), ('\\dots', '…'), ('\\ldots', '…'),
                     ('\\@', ''), ('\\-', ''), ('\\{', '{'), ('\\}', '}')):
            s = s.replace(a, b)
        s = re.sub(r'\\(small|footnotesize|scriptsize|normalsize|large|Large|LARGE|Huge|huge|bfseries|centering|hfill|noindent|par|newpage|'
                   r'raggedright|raggedleft|sloppy|medskip|smallskip|bigskip)(?![a-zA-Z])\s*', '', s)
        left = set(re.findall(r'\\[A-Za-z]+', s))
        if left:
            self.log.append('unconverted: ' + ' '.join(sorted(left)))
            s = re.sub(r'\\[A-Za-z]+', '', s)
        s = s.replace('{', '').replace('}', '')
        return s

    def inline_raw(self, t):
        return self.inline(t)

    # display units ---------------------------------------------------------
    def png(self, name):
        cand = [os.path.join(self.dir, 'fig', self.lang, f'{name}.pdf'), os.path.join(self.dir, f'{name}.pdf')]
        pdf = next((p for p in cand if os.path.exists(p)), None)
        if not pdf:
            raise Stop(f'figure {name}: no PDF in fig/{self.lang}/ or in the case folder')
        outd = os.path.join(OUT, 'img', f'C{self.k}')
        os.makedirs(outd, exist_ok=True)
        fn = name if name.endswith(('_en', '_ro')) else f'{name}_{self.lang}'
        dst = os.path.join(outd, fn + '.png')
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(pdf):
            import pymupdf
            page = pymupdf.open(pdf)[0]
            zoom = 1500 / max(page.rect.width, 1)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            pix.save(dst)
        return f'img/C{self.k}/{fn}.png'

    def table_html(self, name):
        p = os.path.join(self.dir, f'tab_{name}_{self.lang}.tex')
        src = strip_comments(open(p, encoding='utf-8').read())
        m = re.search(r'\\begin\{tabular\}', src)
        spec, i = group(src, m.end())
        body = src[i:src.index('\\end{tabular}')]
        align = [c for c in re.sub(r'[^lrc]', '', spec)]
        parts = re.split(r'\\(toprule|midrule|bottomrule)', body)
        # parts: text, rulename, text, ...
        segs, cur = [], None
        for x in parts:
            if x in ('toprule', 'midrule', 'bottomrule'):
                if cur is not None:
                    segs.append(cur)
                cur = ''
            elif cur is not None:
                cur += x
        rows = lambda t: [r for r in (r.strip() for r in re.split(r'\\\\', t)) if r]
        head = rows(segs[0]) if len(segs) > 1 else []
        bodies = [rows(t) for t in segs[1:]] if len(segs) > 1 else [rows(segs[0])]

        def cells(r, tag):
            cs = re.split(r'(?<!\\)&', r)
            out = []
            for j, c in enumerate(cs):
                c = self.protect_math(c.strip())
                cls = ' class="r"' if j < len(align) and align[j] == 'r' else ''
                out.append(f'<{tag}{cls}>{self.inline(c)}</{tag}>')
            return '<tr>' + ''.join(out) + '</tr>'
        h = ''.join(cells(r, 'th') for r in head)
        b = ''.join(f'<tbody>{"".join(cells(r, "td") for r in rs)}</tbody>' for rs in bodies if rs)
        return f'<div class="tbl"><table>{"<thead>" + h + "</thead>" if h else ""}{b}</table></div>'

    def caption(self, kind, cap, label):
        if kind == 'figure':
            self.nfig += 1
            n = self.nfig
        else:
            self.ntab += 1
            n = self.ntab
        if label:
            self.labels[label] = n
        word = self.T['figure'] if kind == 'figure' else self.T['table']
        return f'<p class="fcap"><span class="fnum">{word} {n}.</span> {self.inline(cap)}</p>'

    def srcline(self, s):
        return f'<p class="fsrc">{self.T["src"]}: {self.inline(s)}</p>' if s else ''

    def units(self, body):
        """Sequential scan of a figure/table/center body -> list of units {kind, name, caption, src, label}."""
        tok = re.compile(r'\\(includegraphics|tabin|captionof|caption|src|label|fig)(?![a-zA-Z])')
        us, i = [], 0
        while True:
            m = tok.search(body, i)
            if not m:
                return us
            t = m.group(1)
            j = opt(body, m.end())
            if t == 'includegraphics':
                (name,), i = args(body, j, 1)
                us.append(dict(kind='img', name=name.strip(), caption=None, src=None, label=None))
            elif t == 'tabin':
                (name,), i = args(body, j, 1)
                if us and us[-1]['kind'] == 'pending':
                    us[-1].update(kind='tab', name=name.strip())
                else:
                    us.append(dict(kind='tab', name=name.strip(), caption=None, src=None, label=None))
            elif t == 'fig':
                (name, cap, src, _), i = args(body, j, 4)
                us.append(dict(kind='img', name=name.strip(), caption=cap, src=src, label=f'fig:{name.strip()}'))
            elif t in ('caption', 'captionof'):
                if t == 'captionof':
                    _, j = args(body, j, 1)
                (cap,), i = args(body, j, 1)
                if not us:
                    us.append(dict(kind='pending', name=None, caption=None, src=None, label=None))
                us[-1]['caption'] = cap
            elif t == 'src':
                (s,), i = args(body, j, 1)
                if us:
                    us[-1]['src'] = s
            elif t == 'label':
                (s,), i = args(body, j, 1)
                if us:
                    us[-1]['label'] = s.strip()

    def unit_html(self, u, numbered=True):
        if u['kind'] == 'tab':
            cap = self.caption('table', u['caption'], u['label']) if u['caption'] else ''
            return f'<figure class="ctab">{cap}{self.table_html(u["name"])}{self.srcline(u["src"])}</figure>'
        src = self.png(u['name'])
        cap = self.caption('figure', u['caption'], u['label']) if u['caption'] else ''
        alt = re.sub(r'<[^>]+>', '', self.restore(self.inline(u['caption']))) if u['caption'] else self.T['figure']
        return (f'<figure class="cfig"><a href="{src}" target="_blank" rel="noopener" title="{html.escape(self.T["enlarge"])}">'
                f'<img src="{src}" alt="{html.escape(alt)}" loading="lazy"></a><figcaption>{cap}{self.srcline(u["src"])}</figcaption></figure>')

    def float_env(self, body, kind):
        us = self.units(body)
        if kind == 'table' or all(u['kind'] == 'tab' for u in us):
            return self.blk(''.join(self.unit_html(u) for u in us))
        h = ''.join(self.unit_html(u) for u in us)
        return self.blk(f'<div class="cfig-row{" one" if len(us) == 1 else ""}">{h}</div>')

    # environments and blocks ---------------------------------------------
    def env(self, s, name, fn, innermost=False):
        pat = re.compile(r'\\begin\{' + name + r'\}')
        while True:
            ms = list(pat.finditer(s))
            if not ms:
                return s
            # innermost: the last \begin whose body has no other \begin{name}
            m = ms[-1] if innermost else ms[0]
            end = s.index('\\end{' + name + '}', m.end())
            j = opt(s, m.end())
            s = s[:m.start()] + fn(s[j:end]) + s[end + len('\\end{' + name + '}'):]

    def lst(self, body, tag):
        items = re.split(r'\\item(?![a-zA-Z])', body)[1:]
        return self.blk(f'<{tag}>' + ''.join(f'<li>{self.flow(it, wrap=False)}</li>' for it in items) + f'</{tag}>')

    def kpis(self, s):
        run = re.compile(r'(?:\\begin\{minipage\}(?:(?!\\end\{minipage\}).)*?\\Huge(?:(?!\\end\{minipage\}).)*?\\end\{minipage\}\s*(?:\\hfill)?\s*)+', re.S)

        def tiles(m):
            out = []
            for mp in re.finditer(r'\\begin\{minipage\}(.*?)\\end\{minipage\}', m.group(0), re.S):
                b = mp.group(1)
                b = b[opt(b, 0):]
                _, i = group(b, b.index('{'))            # width
                b = b[i:]
                k = b.index('{\\color')
                v, i = group(b, k)
                red = 'IDAred' in v
                v = re.sub(r'\\color\{[^}]*\}|\\Huge|\\bfseries', '', v)
                lab = b[i:].replace('\\\\', '', 1)
                out.append(f'<div class="kpi{" kpi-red" if red else ""}"><span class="kpi-v">{self.inline(v).strip()}</span>'
                           f'<span class="kpi-l">{self.inline(lab).strip()}</span></div>')
            return self.blk('<div class="kpis">' + ''.join(out) + '</div>')
        return run.sub(tiles, s)

    def glance(self, s):
        pat = re.compile(r'\\glance(?![a-zA-Z])')
        while True:
            m = pat.search(s)
            if not m:
                return s
            a, j = args(s, m.end(), 5)
            steps = ''.join(f'<li class="gl{" gl-dec" if i == 4 else ""}"><span class="gl-h">{self.T["glance"][i]}</span>'
                            f'<span class="gl-t">{self.inline(x)}</span></li>' for i, x in enumerate(a))
            s = s[:m.start()] + self.blk(f'<ol class="glance">{steps}</ol>') + s[j:]

    def size_groups(self, s):
        pat = re.compile(r'\{\\(small|footnotesize|scriptsize)(?![a-zA-Z])')
        while True:
            m = pat.search(s)
            if not m:
                return s
            try:
                g, j = group(s, m.start())
            except Stop:                                 # group closed in a later section of the brief: runs to the end of this one
                g, j = s[m.start() + 1:], len(s)
            inner = g[len(m.group(1)) + 1:]
            inner = re.sub(r'^\s*\\color\{[^}]*\}', '', inner)
            h = self.flow(inner)
            if m.group(1) != 'small':
                h = f'<div class="cnote">{h}</div>'
            s = s[:m.start()] + self.blk(h) + s[j:]

    def flow(self, s, wrap=True):
        """Block-level conversion: boxes, floats, lists, paragraphs."""
        s = self.size_groups(s)
        s = self.env(s, 'figure', lambda b: self.float_env(b, 'figure'))
        s = self.env(s, 'table', lambda b: self.float_env(b, 'table'))
        s = self.env(s, 'center', lambda b: self.float_env(b, 'figure'))
        s = self.kpis(s)
        s = self.glance(s)
        for name in ('scenario', 'intuition', 'decision'):
            title = {'scenario': self.T['question'], 'intuition': self.T['intuition'], 'decision': self.T['decision']}[name]
            s = self.env(s, name, lambda b, name=name, title=title: self.blk(
                f'<div class="cbox cbox-{name}"><p class="cbox-t">{title}</p>{self.flow(b)}</div>'))
        while re.search(r'\\begin\{(itemize|enumerate)\}', s):
            m = None
            for mm in re.finditer(r'\\begin\{(itemize|enumerate)\}((?:(?!\\begin\{(?:itemize|enumerate)\}).)*?)\\end\{\1\}', s, re.S):
                m = mm
                break
            if not m:
                raise Stop('unbalanced lists')
            body = m.group(2)
            body = body[opt(body, 0):] if body.lstrip().startswith('[') else body
            s = s[:m.start()] + self.lst(body, 'ul' if m.group(1) == 'itemize' else 'ol') + s[m.end():]
        left = re.findall(r'\\begin\{(\w+)\}', s)
        if left:
            self.log.append('unconverted environments: ' + ', '.join(sorted(set(left))))
        out = []
        for para in re.split(r'\n\s*\n', s):
            for piece in re.split(r'(\x01\d+\x01)', para):
                if re.fullmatch(r'\x01\d+\x01', piece):
                    out.append(piece)
                    continue
                t = self.inline(piece).strip()
                t = re.sub(r'^(<br>\s*)+|(\s*<br>)+$', '', t).strip()
                if t:
                    out.append(f'<p>{t}</p>')
        if not wrap and len(out) == 1 and not out[0].startswith('\x01'):
            return out[0].removeprefix('<p>').removesuffix('</p>')
        return ''.join(out)


# ------------------------------------------------------------------ acronyms
FUNC = {'of', 'and', 'the', 'for', 'a', 'an', 'to', 'in', 'on', 'de', 'și', 'al', 'ale', 'ai', 'la', 'cu', 'în', 'pe', 'pentru', 'din', 'a'}


def explained(a, exp, before, after):
    """Does the brief already spell out acronym a next to its first use?"""
    if re.match(r'\s*\(', after):                               # ABBR (expansion)
        return True
    near = (before[-140:] + ' ' + after[:140]).lower()
    words = [w for w in re.findall(r'[\w-]+', exp.split(',')[0].lower()) if w not in FUNC]
    if words and words[0][:6] in near and (len(words) == 1 or words[1][:4] in near):
        return True
    letters = re.sub(r'[^a-z]', '', a.lower())

    def initials(ws):
        ws = [w for w in ws if w.lower() not in FUNC]
        return ''.join(w[0].lower() for w in ws[-len(letters):]) == letters if len(ws) >= len(letters) else False
    if re.search(r'\(\s*$', before):                             # expansion (ABBR ...
        if initials(re.findall(r'[\w]+', before.rstrip()[:-1])[-len(letters) - 4:]):
            return True
    m = re.match(r'\s*,\s*([^,;)]*)', after)                    # (ABBR, expansion)
    if m and re.search(r'\(\s*$', before):
        ws = [w for w in re.findall(r'\w+', m.group(1)) if w.lower() not in FUNC][:len(letters)]
        if len(ws) == len(letters) and ''.join(w[0].lower() for w in ws) == letters:
            return True
    return False


def acronyms(h, lang, log):
    """First use of each known acronym in the running text: add the expansion if the brief does not give one nearby."""
    found = []
    parts = re.split(r'(<[^>]+>|\\\(.*?\\\)|\\\[.*?\\\])', h, flags=re.S)
    plain = [p if i % 2 == 0 else ('' if p.startswith('<') else ' ') for i, p in enumerate(parts)]
    seen = set()
    idx = 1 if lang == 'ro' else 0
    for i, p in enumerate(parts):
        if i % 2 or not p:
            continue

        def rep(m):
            a = m.group(0)
            if a not in ACR or ACR[a][idx] is None or a in seen:
                return a
            seen.add(a)
            found.append(a)
            exp = ACR[a][idx]
            for b in ACR:                                    # acronyms used inside the expansion count as explained
                if b not in seen and re.search(r'(?<![\w-])' + re.escape(b) + r'(?![\w-])', exp):
                    seen.add(b)
                    found.append(b)
            if a in NOINS or exp.lower() == a.lower():
                return a
            before = (''.join(plain[max(0, i - 40):i]) + m.string[:m.start()])[-220:]
            after = (m.string[m.end():] + ''.join(plain[i + 1:i + 40]))[:220]
            if explained(a, exp, before, after):
                return a
            log.append(f'expanded at first use: {a} -> {exp}  [...{before[-50:]}|{after[:30]}...]')
            if before.count('“') + before.count('„') > before.count('”'):   # inside a quotation: editorial square brackets
                return f'{a} [{exp}]'
            if before.count('(') > before.count(')'):          # already inside brackets: no nested brackets
                return f'{a}, {exp}'
            return f'{a} ({exp})'
        parts[i] = re.sub(r'(?<![\w-])[A-Z][A-Za-z]*[A-Z](?:-[A-Z]+)?(?![\w-])', rep, p)
    return ''.join(parts), found


def check_ro(text):
    issues = []
    plain = re.sub(r'<[^>]+>|\\\(.*?\\\)|\\\[.*?\\\]', ' ', text, flags=re.S)
    plain = re.sub(r'https?://\S+', ' ', plain)
    for w in set(re.findall(r'\w*[âÂ]\w*', plain)):
        if not re.match(r'^(?:ne)?r[oO]m[âÂ]n', w, re.I):
            issues.append(f'"â" outside the român- family: {w}')
    for ch in 'şţŞŢ':
        if ch in plain:
            issues.append(f'cedilla letter {ch}')
    for w in set(re.findall(r'\b(sunt|suntem|sunteți|Sunt)\b', plain)):
        issues.append(f'"{w}" -> sînt')
    return issues


# ------------------------------------------------------------------ page
def build(k, lang):
    c = CASE[k]
    T = L[lang]
    cv = Conv(k, lang)
    src = strip_comments(open(os.path.join(cv.dir, f'C{k}_brief.tex'), encoding='utf-8').read())
    nums = read_macros(os.path.join(cv.dir, f'numbers_{lang}.tex'))
    local = {n: v for n, v in read_macros(os.path.join(cv.dir, f'C{k}_brief.tex')).items() if n not in ('emqacase', 'fig')}
    body = src[src.index('\\begin{document}') + len('\\begin{document}'):src.index('\\end{document}')]
    body = pick_lang(body, lang)
    body = expand(body, local)
    body = expand(body, nums)
    body = body.replace('\\lang', lang).replace('\\ds', ',' if lang == 'ro' else '.')
    body = re.sub(r'\\(E|1|asinh|where)(?![a-zA-Z])', lambda m: {'E': '\\mathbb{E}', '1': '\\mathbf{1}', 'asinh': '\\operatorname{asinh}',
                                                                  'where': 'where' if lang == 'en' else 'unde'}[m.group(1)], body)
    m = re.search(r'\\emqatitle', body)
    (title, sub), j = args(body, m.end(), 2)
    body = body[:m.start()] + body[j:]
    body = cv.protect_math(body)
    title_h = cv.restore(cv.inline(cv.protect_math(title)))
    title_h = re.sub(r'^(Case|Studiul de caz) \d+:\s*', '', title_h)
    sub_h = cv.restore(cv.inline(sub))
    # sections
    pieces = re.split(r'\\section(\*?)(?![a-zA-Z])', body)
    pre = pieces[0]
    secs = []
    for star, rest in zip(pieces[1::2], pieces[2::2]):
        t, j = group(rest, 0)
        secs.append(dict(title=cv.restore(cv.inline(t)), star=bool(star), body=rest[j:]))
    pre_h = cv.restore(cv.flow(pre))
    toc = [f'<li><a href="#intro">{T["question"]}</a></li>']
    out = []
    for j, s in enumerate(secs, 1):
        sid = 'glance' if s['star'] else f'sec-{j}'
        s['h'] = cv.flow(s['body'])
        s['id'] = sid
    # references are resolved only after all floats are numbered
    sec_html = []
    for s in secs:
        toc.append(f'<li><a href="#{s["id"]}">{s["title"]}</a></li>')
        sec_html.append(f'<section class="csec" id="{s["id"]}"><h2 class="sec">{s["title"]}</h2><div class="ctext">{cv.restore(s["h"])}</div></section>')
    main_h = pre_h + ''.join(sec_html)
    main_h, found = acronyms(main_h, lang, cv.log)
    idx = 1 if lang == 'ro' else 0
    abbr = sorted({a for a in found if ACR[a][idx] and ACR[a][idx].lower() != a.lower()}, key=str.lower)

    # header block: paper, materials
    sfx = 'EN' if lang == 'en' else 'RO'
    other = 'RO' if lang == 'en' else 'EN'
    f = c['folder']
    lec_links = ' '.join(f'<a class="btn btn-outline" href="../L{n}{"_ro" if lang == "ro" else ""}.html">{T["lecture"]} {n}: '
                         f'{LECT[n][idx]}</a>' for n in c['lectures'])
    if c.get('paper', True):
        venue = c['venue'].replace('SSRN ', SSRN[lang] + ' ') if c['venue'].startswith('SSRN') else c['venue']
        authors = c['authors'] if lang == 'en' else c['authors'].replace(' & ', ' și ')
        paper = (f'<div class="cpaper"><p class="cpaper-h">{T["paper"]}</p><p class="cpaper-c">{html.escape(authors)} ({c["year"]}). '
                 f'{html.escape(c["title"])}. <em>{html.escape(venue)}</em>. <a href="{c["link"]}" target="_blank" rel="noopener">'
                 f'{c["linktext"]}</a></p></div>')
    else:
        paper = f'<div class="cpaper"><p class="cpaper-h">{T["paper"]}</p><p class="cpaper-c">{T["note_c8"]}</p></div>'

    def mat(lg):
        return (f'<a class="btn btn-primary" href="../../cases/{f}/C{k}_case_{lg}.pdf" target="_blank" rel="noopener">{T["mat_brief"]} (PDF)</a>'
                f'<a class="btn btn-outline" href="{REPO}/blob/main/cases/{f}/C{k}_notebook_{lg}.ipynb" target="_blank" rel="noopener">{T["mat_nb"]}</a>'
                f'<a class="btn btn-colab" href="{COLAB}cases/{f}/C{k}_notebook_{lg}.ipynb" target="_blank" rel="noopener">{T["mat_colab"]}</a>')
    lname = {'EN': T['lang_en'], 'RO': T['lang_ro']}
    materials = (f'<div class="cmat"><p class="cpaper-h">{T["materials"]}</p>'
                 f'<div class="link-row"><span class="link-row-h">{lname[sfx]}</span><div class="link-row-b">{mat(sfx)}</div></div>'
                 f'<div class="link-row"><span class="link-row-h">{lname[other]}</span><div class="link-row-b">{mat(other)}</div></div>'
                 f'<div class="link-row"><span class="link-row-h">{T["related"] if len(c["lectures"]) == 1 else T["related_pl"]}</span>'
                 f'<div class="link-row-b">{lec_links}</div></div></div>')
    sub_block = '' if c.get('paper', True) else f'<p class="tp-subt">{sub_h}</p>'
    lk = c['lectures'][0]
    kicker = f'{T["case"]} {k}' + (f' · {T["lecture"]} {lk}' if c.get('paper', True) else '')
    intro = (f'<section class="wintro" id="intro"><div class="tp"><p class="tp-k">{kicker}</p><h2 class="tp-sub">{title_h}</h2>{sub_block}'
             f'<p class="tp-au">{T["author"]}</p><p class="tp-in">{"<br>".join(T["inst"])}</p></div>{paper}{materials}</section>')
    if abbr:
        toc.append(f'<li><a href="#abbr">{T["abbr"]}</a></li>')
        main_h += (f'<section class="wbib" id="abbr"><h2 class="sec">{T["abbr"]}</h2><div class="tbl"><table class="abbr"><tbody>' +
                   ''.join(f'<tr><th>{a}</th><td>{ACR[a][idx]}</td></tr>' for a in abbr) + '</tbody></table></div></section>')
    main_h += f'<p class="src foot">{T["foot"]}</p>'
    head_t = f'{T["case"]} {k}: {re.sub(r"<[^>]+>", "", title_h)}'
    page_en, page_ro = f'C{k}.html', f'C{k}_ro.html'
    en_a, ro_a = ('active', '') if lang == 'en' else ('', 'active')
    lec_nav = ''.join(f'<li><a href="../L{n}{"_ro" if lang == "ro" else ""}.html">{T["lecture"]} {n}</a></li>' for n in c['lectures'])
    nav = (f'<li><a href="{T["site"]}">← {T["back"]}</a></li>'
           f'<li><a href="../../cases/{f}/C{k}_case_{sfx}.pdf" target="_blank" rel="noopener">{T["brief"]}</a></li>'
           f'<li><a href="{REPO}/blob/main/cases/{f}/C{k}_notebook_{sfx}.ipynb" target="_blank" rel="noopener">{T["nb"]}</a></li>'
           f'<li><a href="{COLAB}cases/{f}/C{k}_notebook_{sfx}.ipynb" target="_blank" rel="noopener">{T["colab"]}</a></li>'
           f'{lec_nav}'
           f'<li class="lang-switcher"><a href="{page_en}" class="lang-btn {en_a}">EN</a><a href="{page_ro}" class="lang-btn {ro_a}">RO</a></li>')
    desc = re.sub(r'<[^>]+>', '', title_h)
    page = f'''<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(head_t)} | NN</title>
<meta name="description" content="{html.escape(T['course'])}: {html.escape(T['case'])} {k}, {html.escape(desc)}">
<link rel="icon" href="../../assets/logos/ase_logo.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../../assets/site.css?v={VER}">
<link rel="stylesheet" href="../web.css?v={VER}">
<link rel="stylesheet" href="case.css?v={VER}">
<script>
window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']], processEscapes: false,
  macros: {{ E: '\\\\mathbb{{E}}', Var: '\\\\operatorname{{Var}}', Cov: '\\\\operatorname{{Cov}}', Corr: '\\\\operatorname{{Corr}}',
            asinh: '\\\\operatorname{{asinh}}', EUR: '\\\\text{{EUR}}' }} }},
  options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }},
  startup: {{ pageReady: () => MathJax.startup.defaultPageReady().then(() => {{ document.body.setAttribute('data-mj', 'done'); window.dispatchEvent(new Event('nn-mathjax')); }}) }} }};
</script>
<script id="MathJax-script" async src="{MATHJAX}"></script>
<script defer src="../web.js?v={VER}"></script>
</head>
<body class="web case">
<header>
  <div class="header-inner">
    <a href="https://www.ase.ro" target="_blank" rel="noopener"><img class="header-logo" src="../../assets/logos/ase_logo.png" alt="ASE"></a>
    <div class="header-title"><h1><a href="{T['site'].split('#')[0]}">{html.escape(T['course'])}</a></h1><p>{html.escape(head_t)}</p></div>
    <a href="https://ipe.ro/new/" target="_blank" rel="noopener"><img class="header-logo" src="../../assets/logos/acad_logo.png" alt="{html.escape(T['inst'][1])}"></a>
  </div>
</header>
<nav><ul>
  {nav}
</ul></nav>
<div class="wlay">
<aside class="wtoc"><details open id="wtoc"><summary>{T['toc']}</summary><ol>{''.join(toc)}</ol>
<p class="toc-all"><a href="{T['site']}">{T['all_cases']}</a></p></details></aside>
<main class="wmain">
{intro}
{main_h}
</main>
</div>
</body>
</html>
'''
    page = cv.restore(page)
    page = re.sub(r'(?<![\d.,])((?:19|20)\d\d)([–-])(\d\d)(?![\d%]|[.,]\d)',
                  lambda m: m.group(1) + m.group(2) + (m.group(1)[:2] + m.group(3) if int(m.group(3)) > int(m.group(1)[2:]) else m.group(3)), page)
    left = re.findall(r'\\(?!\(|\)|\[|\])[A-Za-z]+', re.sub(r'\\\(.*?\\\)|\\\[.*?\\\]|<script.*?</script>', '', page, flags=re.S))
    if left:
        cv.log.append('LaTeX left in text: ' + ' '.join(sorted(set(left))))
    if '??' in re.sub(r'<script.*?</script>', '', page, flags=re.S):
        cv.log.append('unresolved \\ref')
    if lang == 'ro':
        cv.log += check_ro(main_h)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, page_en if lang == 'en' else page_ro), 'w', encoding='utf-8') as fh:
        fh.write(page)
    return cv, len(secs), cv.nfig, cv.ntab, found


def main(k):
    log = [f'NN web case C{k}']
    res = {}
    for lang in ('en', 'ro'):
        cv, ns, nf, nt, found = build(k, lang)
        res[lang] = (ns, nf, nt)
        log.append(f'[{lang}] {ns} sections, {nf} numbered figures, {nt} tables; acronyms: {", ".join(found)}')
        log += [f'[{lang}] {x}' for x in dict.fromkeys(cv.log)]
    if res['en'] != res['ro']:
        log.append(f'EN/RO differ: {res["en"]} vs {res["ro"]}')
    txt = '\n'.join(log) + '\n'
    with open(os.path.join(OUT, f'C{k}_log.txt'), 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(txt)


if __name__ == '__main__':
    a = sys.argv[1:]
    ks = [int(x.lstrip('Cc')) for x in a] if a else sorted(CASE)
    try:
        for k in ks:
            main(k)
    except Stop as e:
        print('STOP:', e)
        sys.exit(2)
