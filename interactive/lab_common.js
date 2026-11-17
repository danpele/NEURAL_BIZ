// Shared mechanics of the lab pages (predict, test, prove): number format, the round flow, the score, range labels,
// a small SVG chart helper and an optional three.js stage. Texts come from window.LAB_T (L<k>_lab_text_{en,ro}.js).
const T = window.LAB_T;
export const RO = T.lang === 'ro';
export const $ = id => document.getElementById(id);
export const fmt = (x, d) => { const s = Number(x).toFixed(d).replace(/^-/, '−'); return RO ? s.replace('.', ',') : s; };
export const med = v => { const s = [...v].sort((a, b) => a - b), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

const score = {};
function setScore() {
    const n = Object.keys(score).length, ok = Object.values(score).filter(Boolean).length;
    $('lab-score').textContent = n ? T.score(ok, n) : '';
}

// one round: predict (pick and lock), test (run, then the controls open), prove (verdict and proof)
export function round(id, cfg, N) {
    const root = $(id), r = T.rounds[id];
    root.querySelector('.lab-q').innerHTML = r.q;
    const opts = root.querySelector('.lab-opts');
    opts.innerHTML = r.opts.map((o, i) => `<label class="lab-opt"><input type="radio" name="${id}-o" value="${i}"> <span>${o}</span></label>`).join('');
    const lock = root.querySelector('.lab-lock'), run = root.querySelector('.lab-run'), why = root.querySelector('.lab-why'),
        proof = root.querySelector('.lab-proof'), ctl = root.querySelector('.lab-ctls'), out = root.querySelector('.lab-out');
    lock.textContent = T.lock; run.textContent = T.run; why.textContent = T.why;
    let pick = null, ran = false;
    opts.addEventListener('change', e => { pick = +e.target.value; lock.disabled = false; });
    lock.disabled = true; run.disabled = true; why.disabled = true;
    ctl.querySelectorAll('input,select').forEach(x => (x.disabled = true));
    lock.addEventListener('click', () => {
        opts.querySelectorAll('input').forEach(x => (x.disabled = true));
        lock.disabled = true; root.classList.add('locked');
        root.querySelector('.lab-step-test').classList.add('on');
        run.disabled = false; cfg.activate && cfg.activate(root);
        out.innerHTML = T.afterLock;
    });
    run.addEventListener('click', () => {
        run.disabled = true;
        cfg.run(root, out, () => {
            run.disabled = false;
            if (!ran) {
                ran = true; why.disabled = false;
                ctl.querySelectorAll('input,select').forEach(x => (x.disabled = false));
                root.querySelector('.lab-step-prove').classList.add('on');
                cfg.after && cfg.after(root);
            }
        }, ran);
    });
    why.addEventListener('click', () => {
        const right = pick === r.answer;
        score[id] = right; setScore();
        proof.innerHTML = `<p class="lab-verdict ${right ? 'ok' : 'no'}">${right ? T.right : T.wrong(r.opts[r.answer])}</p>` + r.proof(N, fmt);
        proof.hidden = false; why.disabled = true;
    });
}

// frame-by-frame player: step(k) draws frame k (returns true to stop early); one timer for the whole page
let timer = null;
export function play(n, step, done, total = 3000) {
    clearInterval(timer);
    let k = 0;
    const delay = Math.max(12, Math.min(70, total / n));
    timer = setInterval(() => {
        const stop = step(k);
        if (stop || k >= n - 1) { clearInterval(timer); done(); }
        k++;
    }, delay);
}

export function bindRange(root, sel, show) {
    const el = root.querySelector(sel), lab = root.querySelector(sel + '-v');
    const f = () => { lab.innerHTML = show(+el.value); };
    el.addEventListener('input', f); f();
    return el;
}

// SVG chart: axes {lo, hi, log}, ticks, labels; returns the coordinate maps and a drawing helper
const SVGNS = 'http://www.w3.org/2000/svg';
export function chart(box, o) {
    box.innerHTML = '';
    const W = o.W || 640, H = o.H || 380, ml = 62, mr = 18, mt = 16, mb = 46;
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('class', 'lab-svg');
    const tf = (a, v) => (a.log ? Math.log10(v) : v);
    const cl = (a, v) => Math.min(Math.max(v, a.lo), a.hi);
    const X = v => ml + (tf(o.x, cl(o.x, v)) - tf(o.x, o.x.lo)) / (tf(o.x, o.x.hi) - tf(o.x, o.x.lo)) * (W - ml - mr);
    const Y = v => mt + (1 - (tf(o.y, cl(o.y, v)) - tf(o.y, o.y.lo)) / (tf(o.y, o.y.hi) - tf(o.y, o.y.lo))) * (H - mt - mb);
    const el = (n, a, txt) => { const e = document.createElementNS(SVGNS, n); for (const k in a) e.setAttribute(k, a[k]); if (txt !== undefined) e.textContent = txt; svg.appendChild(e); return e; };
    for (const [v, s] of o.yt) { el('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), class: 'grid' }); el('text', { x: ml - 8, y: Y(v) + 4, class: 'tick', 'text-anchor': 'end' }, s); }
    for (const [v, s] of o.xt) el('text', { x: X(v), y: H - mb + 18, class: 'tick', 'text-anchor': 'middle' }, s);
    el('text', { x: (ml + W - mr) / 2, y: H - 6, class: 'axl', 'text-anchor': 'middle' }, o.xlab);
    el('text', { x: 14, y: (mt + H - mb) / 2, class: 'axl', 'text-anchor': 'middle', transform: `rotate(-90 14 ${(mt + H - mb) / 2})` }, o.ylab);
    const line = (xs, ys, cls) => {
        let d = '';
        xs.forEach((x, i) => { if (isFinite(ys[i])) d += (d ? 'L' : 'M') + X(x).toFixed(1) + ',' + Y(ys[i]).toFixed(1); });
        if (d) el('path', { d, class: cls });
    };
    const hline = (v, cls, txt, below) => {
        el('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), class: cls });
        if (txt) el('text', { x: W - mr - 6, y: Y(v) + (below ? 15 : -6), class: 'tick', 'text-anchor': 'end' }, txt);
    };
    const vline = (v, cls, txt) => {
        el('line', { x1: X(v), x2: X(v), y1: mt, y2: H - mb, class: cls });
        if (txt) el('text', { x: X(v) + 5, y: mt + 13, class: 'tick' }, txt);
    };
    const dot = (x, y, cls, r = 3.5) => el('circle', { cx: X(x), cy: Y(y), r, class: cls });
    box.appendChild(svg);
    return { svg, X, Y, el, line, hline, vline, dot, ml, mr, mt, mb, W, H };
}
