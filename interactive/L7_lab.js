// Lecture 7 lab: predict, test, prove. The attention of FinBERT on real finance tweets, drawn as a 3D landscape (query
// tokens x key tokens, height = attention weight), the [SEP] sink by layer, positional encoding and pretraining with
// little data. Every number is computed here from window.LK_DATA (L7_data.js, built by lectures/L7/build_interactive_L7.py)
// and window.LAB_NUM (lecture numbers and attention_entropy.csv of analysis_L7.py, written into the page by build_lab_L7.py);
// the texts are in L7_lab_text_{en,ro}.js (window.LAB_T).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { $, fmt, round, play, bindRange, chart } from './lab_common.js?v=20261118a';

const D = window.LK_DATA, M = window.LAB_NUM, T = window.LAB_T;
const softmax = (row, t) => { const m = Math.max(...row.map(v => v / t)), e = row.map(v => Math.exp(v / t - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
const att = (l, h, t) => D.qk[l][h].map(r => softmax(r, t));
const rowMax = A => A.reduce((s, r) => s + Math.max(...r), 0) / A.length;
const nEnt = A => A.reduce((s, r) => s - r.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0) / Math.log(r.length), 0) / A.length;
const mean = v => v.reduce((a, b) => a + b, 0) / v.length;
const DIV = [1, 2, 4, 8, 16];

// ------------------------------------------------------------------ numbers shared by the texts
const N = {};
{
    let m8 = 0, m1 = 0, o8 = 0, o1 = 0, rows = 0;
    for (let l = 0; l < 12; l++) for (let h = 0; h < 12; h++) {
        const a8 = att(l, h, 8), a1 = att(l, h, 1);
        m8 += rowMax(a8); m1 += rowMax(a1);
        o8 += a8.filter(r => Math.max(...r) > 0.9).length; o1 += a1.filter(r => Math.max(...r) > 0.9).length; rows += a8.length;
    }
    N.max8 = m8 / 144; N.max1 = m1 / 144; N.one8 = o8 / rows; N.one1 = o1 / rows;
    const s = D.qk.flat(3); N.sLo = Math.min(...s); N.sHi = Math.max(...s);
    N.l1h1max8 = rowMax(att(0, 0, 8)); N.l1h1max1 = rowMax(att(0, 0, 1));
}
N.ent = D.ent; N.entL1 = D.ent[0]; N.entL12 = D.ent[11];
N.entMinL = D.ent.indexOf(Math.min(...D.ent)) + 1; N.entMin = Math.min(...D.ent);
N.sep = M.on_sep; N.cls = M.on_cls; N.nTw = M.n_tweets;
N.sepMaxL = M.on_sep.indexOf(Math.max(...M.on_sep)) + 1; N.sepMax = Math.max(...M.on_sep);
N.sepL1 = M.on_sep[0]; N.clsMaxL = M.on_cls.indexOf(Math.max(...M.on_cls)) + 1; N.clsMax = Math.max(...M.on_cls);
N.sep7 = l => mean(D.att.map(a => mean(a.avg[l].map(r => r[r.length - 1]))));
N.sep7max = N.sep7(N.sepMaxL - 1);
const R = D.runs, MOD = ['embbag', 'tiny_nopos', 'tiny', 'finbert_zs', 'finbert_probe', 'tfidf', 'finbert_ft'];
for (const m of MOD) N[m] = M.f1[m];                         // means of the runs, rounded as in L7_numbers.json
N.posGain = M.gain_pos; N.boot = M.boot_pos;
N.ds = D.ds; N.dsN0 = D.ds.n[0]; N.dsNfull = D.ds.n.at(-1);
N.ds0 = { tfidf: D.ds.tfidf[0], tiny: D.ds.tiny[0], probe: D.ds.probe[0] };
N.dsF = { tfidf: D.ds.tfidf.at(-1), tiny: D.ds.tiny.at(-1), probe: D.ds.probe.at(-1) };
N.pct0 = 100 * D.ds.n[0] / D.ds.n.at(-1);

// ------------------------------------------------------------------ the 3D attention landscape (one per round)
const COLS = [[0.86, 0.90, 0.96], [0.36, 0.55, 0.78], [0.10, 0.23, 0.43], [0.75, 0.22, 0.17]];
const ramp = t => { t = Math.min(Math.max(t, 0), 1) * 3; const i = Math.min(Math.floor(t), 2), f = t - i; return COLS[i].map((c, j) => c + f * (COLS[i + 1][j] - c)); };
const HMAX = 3.0, CELL = 0.62;
function label(text, size, vertical, color = '#1c2433') {
    // a text sprite; vertical labels read top to bottom and hang below their anchor, horizontal ones end at it
    const cv = document.createElement('canvas'), ctx = cv.getContext('2d'), px = 56;
    ctx.font = `600 ${px}px Inter, Arial, sans-serif`;
    const w = Math.ceil(ctx.measureText(text).width) + 12, h = px + 16;
    cv.width = vertical ? h : w; cv.height = vertical ? w : h;
    ctx.font = `600 ${px}px Inter, Arial, sans-serif`; ctx.fillStyle = color; ctx.textBaseline = 'middle';
    if (vertical) { ctx.translate(h / 2, 6); ctx.rotate(Math.PI / 2); ctx.fillText(text, 0, 0); }
    else ctx.fillText(text, 6, h / 2);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sp.scale.set(size * cv.width / h, size * cv.height / h, 1);
    sp.center.set(vertical ? 0.5 : 1, vertical ? 1 : 0.5);
    return sp;
}
function makeLand(root) {
    const box = root.querySelector('.lab-stage'), el = document.createElement('div');
    el.className = 'lab-canvas'; box.appendChild(el);
    const key = document.createElement('p'); key.className = 'lab-key'; key.innerHTML = T.key3d; box.appendChild(key);
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch (e) { el.innerHTML = `<p class="lab-nogl">${T.nogl}</p>`; return { set() {} }; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    camera.position.set(1.5, 11.5, 13.2);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.target.set(0, 0.4, 0.9); controls.maxPolarAngle = Math.PI * 0.49;
    controls.minDistance = 5; controls.maxDistance = 28;
    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9); sun.position.set(5, 12, 8); scene.add(sun);
    const grp = new THREE.Group(); scene.add(grp);
    let toks = null, bars = [], labs = [];
    const build = tk => {
        grp.clear(); labs.forEach(s => { s.material.map.dispose(); s.material.dispose(); }); labs = [];
        toks = tk; const n = tk.length, off = (n - 1) / 2;
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(n * CELL + 0.4, n * CELL + 0.4), new THREE.MeshStandardMaterial({ color: 0xeef1f6, roughness: 1 }));
        floor.rotation.x = -Math.PI / 2; grp.add(floor);
        const geo = new THREE.BoxGeometry(CELL * 0.82, 1, CELL * 0.82);
        bars = [];
        for (let i = 0; i < n; i++) {
            const row = [];
            for (let j = 0; j < n; j++) {
                const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.6 }));
                m.position.set((j - off) * CELL, 0, (i - off) * CELL); grp.add(m); row.push(m);
            }
            bars.push(row);
        }
        tk.forEach((t, i) => {
            const a = label(t, 0.36, false); a.position.set(-(off + 0.8) * CELL, 0.02, (i - off) * CELL); grp.add(a); labs.push(a);
            const b = label(t, 0.36, true); b.position.set((i - off) * CELL, 0.02, (off + 0.8) * CELL); grp.add(b); labs.push(b);
        });
    };
    let visible = false;
    const resize = () => { const w = el.clientWidth || 600, h = el.clientHeight || 450; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(el); resize();
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(el);
    (function loop() { if (visible) { controls.update(); renderer.render(scene, camera); } requestAnimationFrame(loop); })();
    return {
        set(A, tk) {
            if (tk !== toks) build(tk);
            A.forEach((r, i) => r.forEach((p, j) => {
                const m = bars[i][j], h = Math.max(p * HMAX, 0.01);
                m.scale.y = h; m.position.y = h / 2; m.material.color.setRGB(...ramp(Math.sqrt(p)));
            }));
        },
    };
}

// ------------------------------------------------------------------ round 1: why divide by the square root of d
const TK0 = D.att[0].toks;
const st1 = makeLand($('r1'));
const r1v = () => { const r = $('r1'); return { l: +r.querySelector('#r1-l').value - 1, h: +r.querySelector('#r1-h').value - 1, t: DIV[+r.querySelector('#r1-t').value] }; };
const r1read = (l, h, t) => { const A = att(l, h, t); return T.r1read(l + 1, h + 1, fmt(t, t < 2 ? 2 : t % 1 ? 1 : 0), fmt(rowMax(A), 2), A.filter(r => Math.max(...r) > 0.9).length, A.length); };
st1.set(att(0, 0, 8), TK0);
round('r1', {
    run: (root, out, done) => {
        const { l, h, t } = r1v(), n = 40, a = Math.log(8), b = Math.log(t);
        play(n, k => { const tt = Math.exp(a + (b - a) * k / (n - 1)); st1.set(att(l, h, tt), TK0); out.innerHTML = r1read(l, h, tt); }, done, 2600);
    },
}, N);
const r1show = () => { const { l, h, t } = r1v(); st1.set(att(l, h, t), TK0); $('r1').querySelector('.lab-out').innerHTML = r1read(l, h, t); };
bindRange($('r1'), '#r1-l', v => String(v)); bindRange($('r1'), '#r1-h', v => String(v));
bindRange($('r1'), '#r1-t', v => (DIV[v] === 8 ? '8 = √64' : String(DIV[v])));
['#r1-l', '#r1-h', '#r1-t'].forEach(s => $('r1').querySelector(s).addEventListener('input', r1show));

// ------------------------------------------------------------------ round 2: which layers focus
const st2 = makeLand($('r2'));
const r2tw = () => +$('r2').querySelector('#r2-w').value;
const r2read = (w, l) => T.r2read(l + 1, fmt(nEnt(D.att[w].avg[l]), 2), fmt(D.ent[l], 2));
st2.set(D.att[0].avg[0], D.att[0].toks);
round('r2', {
    run: (root, out, done) => {
        const w = r2tw();
        play(12, k => { st2.set(D.att[w].avg[k], D.att[w].toks); root.querySelector('#r2-l').value = k + 1; root.querySelector('#r2-l-v').textContent = k + 1; out.innerHTML = r2read(w, k); }, done, 4200);
    },
}, N);
const r2show = () => { const w = r2tw(), l = +$('r2').querySelector('#r2-l').value - 1; st2.set(D.att[w].avg[l], D.att[w].toks); $('r2').querySelector('.lab-out').innerHTML = r2read(w, l); };
bindRange($('r2'), '#r2-l', v => String(v));
$('r2').querySelector('#r2-l').addEventListener('input', r2show);
$('r2').querySelector('#r2-w').addEventListener('change', r2show);

// ------------------------------------------------------------------ round 3: the [SEP] sink (500 test tweets)
function sepChart(root, upto) {
    const c = chart(root.querySelector('.lab-stage'), {
        x: { lo: 0.4, hi: 12.6 }, y: { lo: 0, hi: 0.7 }, xlab: T.layer, ylab: T.share,
        xt: Array.from({ length: 12 }, (_, i) => [i + 1, String(i + 1)]), yt: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].map(v => [v, fmt(100 * v, 0) + '%']),
    });
    const w = (c.X(2) - c.X(1)) * 0.36;
    for (let l = 0; l < upto; l++) {
        c.el('rect', { x: c.X(l + 1) - w, y: c.Y(N.sep[l]), width: w, height: c.Y(0) - c.Y(N.sep[l]), class: 'bar-sep' });
        c.el('rect', { x: c.X(l + 1), y: c.Y(N.cls[l]), width: w, height: c.Y(0) - c.Y(N.cls[l]), class: 'bar-cls' });
    }
    c.el('rect', { x: c.W - c.mr - 150, y: c.mt + 4, width: 11, height: 11, class: 'bar-sep' });
    c.el('text', { x: c.W - c.mr - 134, y: c.mt + 14, class: 'tick' }, T.onSep);
    c.el('rect', { x: c.W - c.mr - 150, y: c.mt + 22, width: 11, height: 11, class: 'bar-cls' });
    c.el('text', { x: c.W - c.mr - 134, y: c.mt + 32, class: 'tick' }, T.onCls);
}
const r3read = l => T.r3read(l + 1, fmt(100 * N.sep[l], 1), fmt(100 * N.cls[l], 1), fmt(100 * N.sep7(l), 1));
round('r3', {
    activate: root => sepChart(root, 0),
    run: (root, out, done) => { play(12, k => { sepChart(root, k + 1); out.innerHTML = r3read(k); }, done, 3000); },
}, N);

// ------------------------------------------------------------------ round 4: positional encoding (macro-F1 of each run)
function runChart(root, upto) {
    const box = root.querySelector('.lab-stage');
    const c = chart(box, {
        H: 400, x: { lo: 60, hi: 90 }, y: { lo: -0.5, hi: MOD.length - 0.5 }, xlab: T.f1, ylab: '',
        xt: [60, 65, 70, 75, 80, 85, 90].map(v => [v, String(v)]), yt: [],
    });
    MOD.forEach((m, i) => {
        const y = MOD.length - 1 - i;
        c.el('line', { x1: c.ml, x2: c.W - c.mr, y1: c.Y(y), y2: c.Y(y), class: 'grid' });
        c.el('text', { x: c.ml + 6, y: c.Y(y) - 13, class: 'tick' + (m.startsWith('tiny') ? ' hl' : '') }, T.model[m]);
        if (i < upto) {
            R[m].forEach(v => c.dot(v, y, m.startsWith('tiny') ? 'main-dot' : 'seed', 4.5));
            c.el('line', { x1: c.X(N[m]), x2: c.X(N[m]), y1: c.Y(y) - 9, y2: c.Y(y) + 9, class: 'meanbar' });
        }
    });
}
round('r4', {
    activate: root => runChart(root, 0),
    run: (root, out, done) => {
        play(MOD.length, k => { runChart(root, k + 1); const m = MOD[k]; out.innerHTML = T.r4read(T.model[m], fmt(N[m], 1), R[m].map(v => fmt(v, 1)).join(RO_SEP())); }, done, 3000);
    },
}, N);
function RO_SEP() { return T.lang === 'ro' ? '; ' : ', '; }

// ------------------------------------------------------------------ round 5: little labelled data
function dsChart(root, upto) {
    const c = chart(root.querySelector('.lab-stage'), {
        x: { lo: 380, hi: 9800, log: true }, y: { lo: 40, hi: 90 }, xlab: T.nTrain, ylab: T.f1,
        xt: D.ds.n.map(v => [v, v.toLocaleString(T.lang)]), yt: [40, 50, 60, 70, 80, 90].map(v => [v, String(v)]),
    });
    c.hline(N.finbert_ft, 'lin', T.ftLine(fmt(N.finbert_ft, 1)), true);
    const xs = D.ds.n.slice(0, upto);
    for (const [k, cls] of [['probe', 'ap'], ['tfidf', 'trn'], ['tiny', 'main']]) {
        c.line(xs, D.ds[k].slice(0, upto), cls);
        if (upto) { c.dot(xs.at(-1), D.ds[k][upto - 1], cls + '-dot'); c.el('text', { x: c.X(xs[0]) + 4, y: c.Y(D.ds[k][0]) - 14, class: 'leg-' + cls }, T.ds[k]); }
    }
}
const r5read = i => T.r5read(D.ds.n[i].toLocaleString(T.lang), fmt(100 * D.ds.n[i] / N.dsNfull, 0), fmt(D.ds.probe[i], 1), fmt(D.ds.tfidf[i], 1), fmt(D.ds.tiny[i], 1));
round('r5', {
    activate: root => dsChart(root, 0),
    run: (root, out, done) => { const n = +root.querySelector('#r5-n').value + 1; play(n, k => { dsChart(root, k + 1); out.innerHTML = r5read(k); }, done, 2500); },
}, N);
bindRange($('r5'), '#r5-n', v => D.ds.n[v].toLocaleString(T.lang));
$('r5').querySelector('#r5-n').addEventListener('input', e => { const i = +e.target.value; dsChart($('r5'), i + 1); $('r5').querySelector('.lab-out').innerHTML = r5read(i); });

// start
sepChart($('r3'), 0); runChart($('r4'), 0); dsChart($('r5'), 0);
window.LAB_N = N;   // for checks
