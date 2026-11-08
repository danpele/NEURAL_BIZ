"""
make_notebook_L5.py -- Lecture 5 notebooks (RNN, LSTM, GRU on financial and energy series), EN + RO, executed:
notebooks/L5_notebook[_ro].ipynb and the seminar notebooks L5_seminar[_ro].ipynb (+ _solutions, git-ignored).
One seed per network here (the slides average 5 or 3 seeds), so numbers are close to, not identical with, the slides.
Run:  python3 lectures/L5/make_notebook_L5.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nn_notebook import Notebook  # noqa: E402

TITLE = ("Recurrent networks: RNN, LSTM and GRU for sequences and financial time series",
         "Rețele recurente: RNN, LSTM și GRU pentru secvențe și serii de timp financiare")

LOAD = r"""import torch, torch.nn as nn
torch.set_num_threads(4)
x = pd.read_csv(DATA + 'markets_daily.csv', parse_dates=['date'])
def ohlc(tk):
    d = x[x.ticker == tk].set_index('date')[['open', 'high', 'low', 'close', 'adjclose']].dropna()
    end = (d.index.max() + pd.Timedelta(days=1)).to_period('M').to_timestamp() - pd.Timedelta(days=1)
    d = d.loc[:end]; d = d[d.high > d.low]
    gk = 0.5 * np.log(d.high / d.low) ** 2 - (2 * np.log(2) - 1) * np.log(d.close / d.open) ** 2
    d['y'] = np.log(gk.clip(lower=1e-8)); d['r'] = 100 * np.log(d.adjclose).diff()
    return d.dropna()
spx = ohlc('SP500')
TR_END, VA_END = pd.Timestamp('2016-12-31'), pd.Timestamp('2019-12-31')
print(spx.index.min().date(), spx.index.max().date(), len(spx))"""

NET = r"""class RecNet(nn.Module):
    def __init__(self, cell, n_in, hidden=32, n_out=1, n_static=0):
        super().__init__()
        self.rnn = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell](n_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden + n_static, n_out)
    def forward(self, x, s=None):
        h, _ = self.rnn(x)
        z = h[:, -1] if s is None else torch.cat([h[:, -1], s], 1)
        return self.out(z)

def train_rec(cell, Xtr, Ytr, Xva, Yva, hidden=32, epochs=40, lr=1e-3, bs=64, Str=None, Sva=None, patience=6, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    n_out = Ytr.shape[1] if Ytr.ndim > 1 else 1
    m = RecNet(cell, Xtr.shape[2], hidden, n_out, 0 if Str is None else Str.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr).view(len(Ytr), -1)
    Xv, Yv = torch.tensor(Xva), torch.tensor(Yva).view(len(Yva), -1)
    St = None if Str is None else torch.tensor(Str); Sv = None if Sva is None else torch.tensor(Sva)
    g = torch.Generator().manual_seed(seed); best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt), generator=g)
        for j in range(0, len(perm), bs):
            b = perm[j:j + bs]
            loss = ((m(Xt[b], None if St is None else St[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(((m(Xv, Sv) - Yv) ** 2).mean())
        if vl < best - 1e-5: best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); m.eval()
    return m

def predict(m, X, S=None):
    with torch.no_grad():
        return m(torch.tensor(X), None if S is None else torch.tensor(S)).numpy()

def dm_test(l1, l2):
    from scipy.stats import norm
    d = np.asarray(l1) - np.asarray(l2); n = len(d); dc = d - d.mean(); L = int(4 * (n / 100) ** (2 / 9))
    v = dc @ dc / n + sum(2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, L + 1))
    s = d.mean() / np.sqrt(v / n); return round(float(s), 2), round(float(2 * (1 - norm.cdf(abs(s)))), 3)"""

VOLDATA = r"""def vol_data(d, W=22):
    y = d.y.values.astype(np.float32); r = d.r.values.astype(np.float32); tr = d.index <= TR_END
    mu, sd, sr = y[tr].mean(), y[tr].std(), r[tr].std()
    f = np.stack([(y - mu) / sd, r / sr, np.abs(r) / sr], 1).astype(np.float32)
    T_ = np.arange(W - 1, len(y) - 1)
    X = np.stack([f[t - W + 1:t + 1] for t in T_]); Y = ((y[T_ + 1] - mu) / sd).astype(np.float32)
    td = d.index[T_ + 1]
    H = np.column_stack([np.ones(len(T_)), y[T_], pd.Series(y).rolling(5).mean().values[T_], pd.Series(y).rolling(22).mean().values[T_]])
    return X, Y, td, y[T_ + 1], y[T_], H, mu, sd"""


def lecture(nb):
    T = nb.T
    nb.md(T("## §1 Returns, volatility and memory", "## §1 Randamente, volatilitate și memorie"))
    nb.code(LOAD)
    nb.code("from statsmodels.tsa.stattools import acf\nar, aa = acf(spx.r, nlags=50, fft=True)[1:], acf(spx.r.abs(), nlags=50, fft=True)[1:]\n"
            "print('rho_r(1) =', round(ar[0], 3), ' rho_|r|(1) =', round(aa[0], 3), ' rho_|r|(50) =', round(aa[-1], 3))\n"
            "fig, ax = plt.subplots(figsize=(7, 2.4)); ax.vlines(np.arange(1, 51) - .15, 0, ar, label='r'); ax.vlines(np.arange(1, 51) + .15, 0, aa, color='C1', label='|r|')\n"
            "fig.legend(loc='lower center', ncol=2); fig.tight_layout(rect=(0, .1, 1, 1)); plt.show()")
    nb.md(T("## §2 Benchmarks first: naive, HAR, GARCH", "## §2 Întîi modelele de referință: naiv, HAR, GARCH"))
    nb.code(NET + "\n" + VOLDATA)
    nb.code("X, Y, td, ytrue, naive, H, mu, sd = vol_data(spx)\ntr, va, te = td <= TR_END, (td > TR_END) & (td <= VA_END), td > VA_END\n"
            "beta = np.linalg.lstsq(H[tr], ytrue[tr], rcond=None)[0]; har = H @ beta\nmse = lambda p: float(np.mean((p[te] - ytrue[te]) ** 2))\n"
            "print('HAR coefficients', np.round(beta, 3)); print('test MSE: naive', round(mse(naive), 4), ' HAR', round(mse(har), 4))")
    nb.md(T("## §3 RNN, LSTM and GRU for volatility", "## §3 RNN, LSTM și GRU pentru volatilitate"))
    nb.code("P = {}\nfor cell in ['rnn', 'lstm', 'gru']:\n    m = train_rec(cell, X[tr], Y[tr], X[va], Y[va])\n"
            "    P[cell] = predict(m, X)[:, 0] * sd + mu; print(cell, 'test MSE', round(mse(P[cell]), 4))\n"
            "print('DM LSTM vs HAR:', dm_test((P['lstm'][te] - ytrue[te]) ** 2, (har[te] - ytrue[te]) ** 2))")
    nb.md(T("## §4 Returns: is there anything to forecast?", "## §4 Randamentele: este ceva de prognozat?"))
    nb.code("r = spx.r.values.astype(np.float32); d = spx.index; Tn = np.arange(21, len(r) - 1); tdr = d[Tn + 1]\n"
            "a, b, c = tdr <= TR_END, (tdr > TR_END) & (tdr <= VA_END), tdr > VA_END; s = r[d <= TR_END].std()\n"
            "Xr = np.stack([r[t - 21:t + 1, None] / s for t in Tn]).astype(np.float32); Yr = (r[Tn + 1] / s).astype(np.float32)\n"
            "m = train_rec('lstm', Xr[a], Yr[a], Xr[b], Yr[b], hidden=16); p = predict(m, Xr)[:, 0] * s\n"
            "yv = r[Tn + 1]; print('out-of-sample R2 (%) of the LSTM against the zero forecast:', round(100 * (1 - np.sum((yv[c] - p[c]) ** 2) / np.sum(yv[c] ** 2)), 3))\n"
            "print('sign accuracy %', round(100 * np.mean(np.sign(p[c]) == np.sign(yv[c])), 1), ' share of up days %', round(100 * np.mean(yv[c] > 0), 1))")
    nb.md(T("## §5 Romanian load, day ahead", "## §5 Consumul din România, pentru ziua următoare"))
    nb.code("ld = pd.read_csv(DATA + 'ro_load_generation_hourly_utc.csv', parse_dates=['timestamp_utc']).set_index('timestamp_utc')['load']\n"
            "END = (spx.index.max() + pd.Timedelta(days=1)).tz_localize('UTC')\nld = ld[(ld.index >= '2016-01-01') & (ld.index < END)]\n"
            "days = pd.date_range(ld.index.min().normalize(), ld.index.max().normalize(), freq='D', tz='UTC')\n"
            "M = ld.reindex(pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq='h', tz='UTC')).values.reshape(-1, 24)\n"
            "dd = days[:len(M)]; LTR, LVA = pd.Timestamp('2022-12-31', tz='UTC'), pd.Timestamp('2023-12-31', tz='UTC')\n"
            "lmu, lsd = np.nanmean(M[dd <= LTR]), np.nanstd(M[dd <= LTR]); flat = M.reshape(-1)\nX_, Y_, S_, D_, NV = [], [], [], [], []\n"
            "for d in range(14, len(M) - 1):\n    xx = flat[(d + 1) * 24 - 168:(d + 1) * 24]\n"
            "    if np.isnan(xx).any() or np.isnan(M[d + 1]).any() or np.isnan(M[d - 6]).any(): continue\n"
            "    dows = np.zeros((7, 7)); dows[np.arange(7), [dd[d - 6 + j].dayofweek for j in range(7)]] = 1\n"
            "    X_.append(np.concatenate([((xx - lmu) / lsd).reshape(7, 24), dows], 1))   # one step per day: 24 hourly loads + weekday\n"
            "    tg = dd[d + 1]; dow = np.zeros(7); dow[tg.dayofweek] = 1; doy = tg.dayofyear / 365.25\n"
            "    S_.append(np.r_[dow, np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)]); Y_.append((M[d + 1] - M[d - 6]) / lsd); D_.append(tg); NV.append(M[d - 6])\n"
            "X_, Y_, S_ = [np.array(v, np.float32) for v in (X_, Y_, S_)]; D_ = pd.DatetimeIndex(D_); NV = np.array(NV)\n"
            "a, b, c = D_ <= LTR, (D_ > LTR) & (D_ <= LVA), D_ > LVA; Ytrue = Y_ * lsd + NV   # target = deviation from the seasonal naive\n"
            "mape = lambda p: round(float(np.mean(np.abs(p[c] - Ytrue[c]) / Ytrue[c]) * 100), 2)\n"
            "from sklearn.linear_model import Ridge\nZ = np.column_stack([X_[:, :, :24].reshape(len(X_), -1), S_]); rg = Ridge(alpha=1).fit(Z[a], Y_[a])\n"
            "lstm = train_rec('lstm', X_[a], Y_[a], X_[b], Y_[b], hidden=48, epochs=40, lr=3e-3, bs=64, Str=S_[a], Sva=S_[b], patience=5)\n"
            "pl = predict(lstm, X_, S_) * lsd + NV\nprint('MAPE %: seasonal naive', mape(NV), ' ridge ARX', mape(rg.predict(Z) * lsd + NV), ' LSTM', mape(pl))")
    nb.md(T("## §6 How far back does the network look?", "## §6 Cît de departe privește rețeaua în trecut?") + "\n\n" +
          T("$S(k)=\\frac1n\\sum_i|\\partial \\hat y_{i,12}/\\partial x_{i,T+1-k}|$, the sensitivity of the noon forecast to the input $k$ hours earlier.",
            "$S(k)=\\frac1n\\sum_i|\\partial \\hat y_{i,12}/\\partial x_{i,T+1-k}|$, sensibilitatea prognozei pentru ora 12 la intrarea de acum $k$ ore."))
    nb.code("rnn = train_rec('rnn', X_[a], Y_[a], X_[b], Y_[b], hidden=48, epochs=40, lr=3e-3, bs=64, Str=S_[a], Sva=S_[b], patience=5)\n"
            "fig, ax = plt.subplots(figsize=(7, 2.4)); ids = np.where(c)[0][:200]\nfor name, m in [('RNN', rnn), ('LSTM', lstm)]:\n"
            "    xx = torch.tensor(X_[ids], requires_grad=True); m(xx, torch.tensor(S_[ids]))[:, 12].sum().backward()\n"
            "    g = xx.grad[:, :, :24].abs().mean(0).numpy().reshape(-1)[::-1]; ax.semilogy(np.arange(1, 169), g / g.max(), label=name)\n"
            "fig.legend(loc='lower center', ncol=2); fig.tight_layout(rect=(0, .1, 1, 1)); plt.show()")


def seminar(nb):
    T = nb.T
    nb.md(T("## Setup", "## Pregătire"))
    nb.code(LOAD)
    nb.code(NET + "\n" + VOLDATA)
    nb.task(T("Count the parameters of an LSTM and of a GRU layer with $d=3$ and $n_h=32$, by the formulas $4n_h(n_h+d+1)$ and $3n_h(n_h+d+1)$, and compare with PyTorch (which keeps two bias vectors).",
              "Numărați parametrii unui strat LSTM și ai unui strat GRU cu $d=3$ și $n_h=32$, cu formulele $4n_h(n_h+d+1)$ și $3n_h(n_h+d+1)$, și comparați cu PyTorch (care păstrează doi vectori de termeni liberi)."))
    nb.solution("for cell, k in [(nn.LSTM, 4), (nn.GRU, 3)]:\n    m = cell(3, 32)\n    print(cell.__name__, k * 32 * (32 + 3 + 1), sum(p.numel() for p in m.parameters()))")
    nb.task(T("Build the Garman–Klass log variance of the DAX (`ohlc('DAX')`) and compare its first-lag autocorrelation with that of the S&P 500.",
              "Construiți log-varianța Garman–Klass a indicelui DAX (`ohlc('DAX')`) și comparați autocorelația sa de ordinul 1 cu cea a S&P 500."))
    nb.solution("dax = ohlc('DAX')\nprint('DAX', round(dax.y.autocorr(1), 3), ' S&P 500', round(spx.y.autocorr(1), 3))")
    nb.task(T("Fit the HAR model on the DAX training years (to 2016) and report its test MSE (2020 onwards) against the naive forecast.",
              "Estimați modelul HAR pe anii de antrenare ai DAX (pînă în 2016) și raportați MSE pe test (din 2020) față de prognoza naivă."))
    nb.solution("X, Y, td, ytrue, naive, H, mu, sd = vol_data(dax)\ntr, va, te = td <= TR_END, (td > TR_END) & (td <= VA_END), td > VA_END\n"
                "beta = np.linalg.lstsq(H[tr], ytrue[tr], rcond=None)[0]; har = H @ beta\nmse = lambda p: float(np.mean((p[te] - ytrue[te]) ** 2))\n"
                "print(np.round(beta, 3), round(mse(naive), 4), round(mse(har), 4))")
    nb.task(T("Train a GRU with 32 units on the same windows (early stopping on 2017–2019). Report its test MSE for three seeds.",
              "Antrenați un GRU cu 32 de unități pe aceleași ferestre (oprire timpurie pe 2017–2019). Raportați MSE pe test pentru trei semințe."))
    nb.solution("preds = []\nfor s in range(3):\n    m = train_rec('gru', X[tr], Y[tr], X[va], Y[va], seed=SEED + s)\n"
                "    preds.append(predict(m, X)[:, 0] * sd + mu); print('seed', s, round(mse(preds[-1]), 4))\ngru = np.mean(preds, 0)")
    nb.task(T("Test with Diebold–Mariano whether the seed-averaged GRU beats HAR on the DAX. What would you put into production?",
              "Testați cu Diebold–Mariano dacă GRU mediat pe semințe depășește HAR pe DAX. Ce model ați pune în producție?"))
    nb.solution("print('MSE GRU', round(mse(gru), 4), ' DM (GRU vs HAR):', dm_test((gru[te] - ytrue[te]) ** 2, (har[te] - ytrue[te]) ** 2))")
    nb.task(T("Romanian load: compute the MAPE of the seasonal naive forecast (same hour one week earlier) by day of the week on 2024–2026. Which days are hardest, and why?",
              "Consumul din România: calculați MAPE a prognozei naive sezoniere (aceeași oră cu o săptămînă înainte) pe zile ale săptămînii, în 2024–2026. Care zile sînt cele mai dificile și de ce?"))
    nb.solution("ld = pd.read_csv(DATA + 'ro_load_generation_hourly_utc.csv', parse_dates=['timestamp_utc']).set_index('timestamp_utc')['load']\n"
                "ld = ld[ld.index >= '2024-01-01']; e = (ld - ld.shift(168)).abs() / ld\n"
                "print((100 * e.groupby(e.index.dayofweek).mean()).round(2))")


if __name__ == "__main__":
    Notebook.build_all(5, TITLE, lecture, kind="lecture")
    Notebook.build_all(5, TITLE, seminar, kind="seminar")
