// Shared helpers of the interactive lecture pages (NN), same design as the MFM interactive chapters. Exposed as window.NNI.
(function () {
'use strict';
const LANG = window.NN_LANG === 'ro' ? 'ro' : 'en';
// course palette (lectures/common/nn_plot.py PAL): fixed colour per zone / entity
const C = { RO: '#2a78d6', DE_LU: '#eb6834', HU: '#1baf7a', BG: '#eda100', c5: '#e87ba4', c6: '#008300', c7: '#6250d6', c8: '#e34948',
  navy: '#1F2A44', ink: '#222222', blue: '#1A3A6E', red: '#DC3545', green: '#2E7D32', amber: '#B5853F', grey: '#8A94A6', band: '#C5D2E8', grid: '#E3E9F4' };
const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const ZONE = LANG === 'ro'
  ? { RO: 'România', BG: 'Bulgaria', HU: 'Ungaria', DE_LU: 'Germania–Luxemburg' }
  : { RO: 'Romania', BG: 'Bulgaria', HU: 'Hungary', DE_LU: 'Germany–Luxembourg' };
const DASH = { RO: 'solid', BG: 'dash', HU: 'dashdot', DE_LU: 'dot' };
const PLAY = LANG === 'ro' ? ['Pornește', 'Pauză'] : ['Play', 'Pause'];
const MONTHS = LANG === 'ro' ? ['ian', 'feb', 'mar', 'apr', 'mai', 'iun', 'iul', 'aug', 'sep', 'oct', 'nov', 'dec']
  : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// ---------------------------------------------------------------- statistics
const sum = a => a.reduce((s, x) => s + x, 0);
const mean = a => (a.length ? sum(a) / a.length : NaN);
const sdev = a => { const m = mean(a); return Math.sqrt(sum(a.map(x => (x - m) ** 2)) / (a.length - 1)); };
const quantile = (a, q) => { const b = [...a].sort((x, y) => x - y), p = (b.length - 1) * q, i = Math.floor(p); return i + 1 < b.length ? b[i] + (p - i) * (b[i + 1] - b[i]) : b[i]; };
const median = a => quantile(a, 0.5);
const finite = a => a.filter(x => x !== null && Number.isFinite(x));
// standard normal: density, cdf (Abramowitz-Stegun 7.1.26) and quantile (Acklam)
const npdf = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
function ncdf(x) { const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2); return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y); }
function nqnt(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239],
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572],
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783],
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416], pl = 0.02425;
  if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - pl) return -nqnt(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// ---------------------------------------------------------------- formatting (decimal comma and thin minus in Romanian)
const nf = (x, d = 1) => {
  if (x === null || x === undefined || !Number.isFinite(x)) return '–';
  let s = Math.abs(x).toFixed(d);
  if (LANG === 'ro') s = s.replace('.', ',');
  const [i, f] = s.split(LANG === 'ro' ? ',' : '.');
  const g = i.length > 4 ? i.replace(/\B(?=(\d{3})+(?!\d))/g, LANG === 'ro' ? '.' : ',') : i;
  return (x < 0 && +Math.abs(x).toFixed(d) !== 0 ? '−' : '') + g + (f !== undefined ? (LANG === 'ro' ? ',' : '.') + f : '');
};
const pct = (x, d = 1) => nf(100 * x, d) + '%';
// a count with its noun: cnt(2861, 'day', 'days') -> "2861 days"; in Romanian cnt(2861, 'zi', 'zile') -> "2861 de zile", cnt(12, ..) -> "12 zile"
const cnt = (n, one, many, d = 0) => {
  const r = Math.round(Math.abs(n));
  if (r === 1 && d === 0) return `1 ${one}`;
  return `${nf(n, d)} ${LANG === 'ro' && (r % 100 === 0 || r % 100 >= 20) ? 'de ' : ''}${many}`;
};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const SEP = LANG === 'ro' ? ',.' : '.,';           // Plotly: decimal and thousands separators
const AX = (title, extra) => Object.assign({ title: { text: title, font: { size: 12, color: C.navy } }, gridcolor: C.grid, zerolinecolor: C.band,
  color: C.navy, tickfont: { size: 11, color: C.navy }, automargin: true }, extra || {});
function layout(extra) {
  const l = Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', separators: SEP,
    font: { family: 'Inter, system-ui, sans-serif', color: C.navy, size: 12 },
    margin: { l: 56, r: 12, t: 30, b: 40 },
    legend: { orientation: 'h', x: 0, xanchor: 'left', y: -0.2, yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
    hoverlabel: { bgcolor: '#fff', bordercolor: C.navy, font: { color: C.navy, family: 'Inter, sans-serif' } },
    hovermode: 'closest'
  }, extra || {});
  if (l.xaxis && l.xaxis.title && l.xaxis.title.text) l.legend = Object.assign({}, l.legend, { y: -0.28 });
  return l;
}
const ttl = s => ({ text: s, x: 0, xanchor: 'left', font: { size: 12, color: C.navy } });
const CFG = { displayModeBar: false, responsive: true };
// Romanian month and day names on date axes (Plotly's default locale is English); full four-digit years everywhere
const RO_LOCALE = { moduleType: 'locale', name: 'ro', dictionary: {}, format: {
  days: ['duminică', 'luni', 'marți', 'miercuri', 'joi', 'vineri', 'sîmbătă'], shortDays: ['dum', 'lun', 'mar', 'mie', 'joi', 'vin', 'sîm'],
  months: ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
  shortMonths: ['ian.', 'feb.', 'mar.', 'apr.', 'mai', 'iun.', 'iul.', 'aug.', 'sep.', 'oct.', 'nov.', 'dec.'], date: '%d.%m.%Y' } };
let _locale = false;
const plot = (id, tr, lay) => {
  if (LANG === 'ro' && !_locale) { Plotly.register(RO_LOCALE); CFG.locale = 'ro'; _locale = true; }
  return Plotly.react(id, tr, lay, CFG);
};
const out = (id, items) => { $(id).innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const vline = (x, color, dash, yref) => ({ type: 'line', xref: 'x', yref: yref || 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color, width: 1.5, dash: dash || 'solid' } });
const hline = (y, color, dash) => ({ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: y, y1: y, line: { color, width: 1.5, dash: dash || 'solid' } });

// ---------------------------------------------------------------- controls
// slider: shows its value next to the label with fmt(value); returns a getter of the numeric value
function bindSlider(id, fmt, onChange) {
  const el = $(id), lab = $(id + '-v');
  const show = () => { if (lab) lab.textContent = fmt ? fmt(+el.value) : el.value; };
  el.addEventListener('input', () => { show(); onChange(); });
  show();
  return () => +el.value;
}
function bindSelect(id, onChange) { const el = $(id); el.addEventListener('change', onChange); return () => el.value; }
// check-box group: getter returns the list of checked values (in page order)
function bindToggles(id, onChange) {
  const box = $(id);
  box.addEventListener('change', onChange);
  return () => [...box.querySelectorAll('input:checked')].map(i => i.value);
}
function bindRadios(id, onChange) {
  const box = $(id);
  box.addEventListener('change', onChange);
  return () => { const r = box.querySelector('input:checked'); return r ? r.value : null; };
}
class Player {
  constructor(id, input, values, ms) { this.b = $(id); this.input = input; this.values = values; this.ms = ms || 600; this.timer = null; this.b.addEventListener('click', () => this.toggle()); this.label(); }
  label() { this.b.textContent = this.timer ? '⏸ ' + PLAY[1] : '▶ ' + PLAY[0]; }
  toggle() { this.timer ? this.stop() : this.start(); }
  start() {
    const el = this.input, cur = this.values.findIndex(v => +v >= +el.value - 1e-9);
    let i = cur >= 0 && cur < this.values.length - 1 ? cur + 1 : 0;
    this.timer = setInterval(() => {
      el.value = this.values[i]; el.dispatchEvent(new Event('input'));
      if (++i >= this.values.length) this.stop();
    }, this.ms);
    this.label();
  }
  stop() { clearInterval(this.timer); this.timer = null; this.label(); }
}
const range = (a, b, st) => { const r = []; for (let x = a; st > 0 ? x <= b + 1e-9 : x >= b - 1e-9; x += st) r.push(+x.toFixed(6)); return r; };
// run a block's init and log (not throw) its errors, so one block cannot switch off the others
function safe(name, f) { try { f(); } catch (e) { console.error('NN block ' + name + ':', e); } }

window.NNI = { LANG, C, SERIES, ZONE, DASH, MONTHS, sum, mean, sdev, quantile, median, finite, npdf, ncdf, nqnt, nf, pct, cnt, $, esc, AX, layout, ttl,
  CFG, plot, out, vline, hline, bindSlider, bindSelect, bindToggles, bindRadios, Player, range, safe };
})();

// left-hand menu on every page (assets/sidemenu.js)
;(function () { var c = document.currentScript; if (!c || !c.src) return; var e = document.createElement('script'); e.src = new URL('../assets/sidemenu.js?v=20261108a', c.src).href; document.head.appendChild(e); })();
