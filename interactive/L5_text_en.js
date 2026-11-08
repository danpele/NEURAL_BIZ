// Readouts of the Lecture 5 interactive page, written by lectures/L5/build_interactive_L5.py.
window.LK_TEXT = {
 mk: { SP500: 'S&P 500', TLV: 'Banca Transilvania (TLV)' }, ser: { r: 'returns rₜ', a: 'absolute returns |rₜ|', s: 'squared returns rₜ²' },
 lag: 'Lag (trading days)', acf: 'Autocorrelation', band: '±1.96/√T band (white noise)',
 acTitle: (m, s, a, b) => `${m}: autocorrelation of daily ${s}, ${a} to ${b}`,
 ac: (r1, r10, n, L, b, T) => `ρ(1) = <b>${r1}</b>, ρ(10) = ${r10}; ${n} of ${L} lags outside ±${b} (${T} trading days).`,
 mod: { obs: 'Observed (Garman–Klass range)', naive: 'Naive (yesterday)', har: 'HAR', garch: 'GARCH(1,1)', rnn: 'RNN (mean of 5 seeds)', lstm: 'LSTM (mean of 5 seeds)', gru: 'GRU (mean of 5 seeds)' },
 vol: 'Annualised volatility (%)', vfTitle: (a, b) => `S&P 500, one-day-ahead volatility forecasts, ${a} to ${b}`,
 vf: (m, e, r) => `${m}: MSE of the log variance <b>${e}</b> (rank ${r})`, vf0: (n, a, b) => `${n} trading days from ${a} to ${b}; whole test period: naive 0.966, GARCH 0.786, HAR 0.706, LSTM 0.625.`,
 rtMod: { mean: 'Training mean', ar: 'AR(1)', lstm: 'LSTM' }, cum: 'Cumulative Σ r² − Σ (r − forecast)²  (%²)',
 rtTitle: m => `${m}: gain over the zero forecast, cumulated over the test days (rising = better)`,
 rt: (m, r2, h) => `${m}: out-of-sample R² <b>${r2}</b>${h ? `; correct sign on ${h} of the days` : ''}.`, rt0: (n, up) => `${n} test days; ${up} of them are up days.`,
 lmod: { obs: 'Observed load', snaive: 'Seasonal naive', ridge: 'Ridge ARX', lstm: 'LSTM (mean of 3 seeds)', gru: 'GRU (mean of 3 seeds)' },
 mw: 'Load (MW)', hourUTC: 'Hour of day d+1 (UTC)', ldTitle: (a, b) => a === b ? `Day-ahead forecasts of the Romanian load, ${a}` : `Day-ahead forecasts of the Romanian load, ${a} to ${b}`,
 ld: (m, w, t) => `${m}: MAPE <b>${w}</b> on the days shown (whole test period ${t}).`,
 mae: 'MAE (MW)', leTitle: n => `Mean absolute error by hour, ${n} test days`,
 le: (m, a, p, h, hm) => `${m}: MAE <b>${a} MW</b>, MAPE ${p}; worst hour ${h}:00 UTC (${hm} MW).`
};
