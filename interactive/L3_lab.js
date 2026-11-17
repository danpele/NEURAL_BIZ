// Lecture 3 lab: predict, test, prove. Overfitting of the 256-256 network on German credit (600 applicants), the
// dropout x weight-decay grid in 3D, early stopping on card fraud and double descent on California housing.
// Every number is computed here from window.LK_DATA (L3_data.js, built by lectures/L3/build_interactive_L3.py from the
// caches of analysis_L3.py); the texts are in L3_lab_text_{en,ro}.js (window.LAB_T).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { $, fmt, med, round, play, bindRange, chart } from './lab_common.js?v=20261117b';

const D = window.LK_DATA, T = window.LAB_T;
const num = v => (v % 1 ? fmt(v, 1) : String(v));
const key = (p, wd) => `${p.toFixed(1)}|${wd.toFixed(1)}`;
const C = D.ov.c, C0 = C[key(0, 0)], EP = C0.va.length;
const argmin = v => v.reduce((b, x, i) => (x < v[b] ? i : b), 0), argmax = v => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);

// ------------------------------------------------------------------ numbers shared by the texts (as in L3_numbers.json)
const N = { epochs: EP, logit: D.ov.logit_va, logitAuc: D.ov.logit_auc };
N.best = argmin(C0.va) + 1; N.vmin = C0.va[N.best - 1]; N.v1 = C0.va[0]; N.vend = C0.va[EP - 1]; N.tend = C0.tr[EP - 1];
N.ratio = N.vend / N.vmin;
N.aucBest = C0.auc[N.best - 1]; N.aucEnd = C0.auc[EP - 1]; N.aucMaxEp = argmax(C0.auc) + 1; N.aucMax = C0.auc[N.aucMaxEp - 1];
N.grid = {};
for (const p of D.ov.dps) for (const wd of D.ov.wds) N.grid[key(p, wd)] = C[key(p, wd)].va[EP - 1];
N.dp = N.grid[key(0.5, 0)]; N.wd = N.grid[key(0, 1)]; N.both = N.grid[key(0.5, 1)]; N.none = N.grid[key(0, 0)];
N.gridBest = Math.min(...Object.values(N.grid));
const earlyMins = Object.keys(N.grid).map(k => Math.min(...C[k].va));
N.gridMinEarly = Math.min(...earlyMins); N.gridMaxEarly = Math.max(...earlyMins);
N.aucWd = C[key(0, 1)].auc[EP - 1];
N.steps = Math.ceil(600 / 64) * EP; N.shrink = Math.pow(1 - 0.001 * 1, N.steps);   // AdamW: lr 0.001, batch 64, 600 applicants
const HO = D.ho['0.2'], hoM = HO.reduce((a, b) => a + b, 0) / HO.length;
N.hoN = HO.length; N.hoMin = Math.min(...HO); N.hoMax = Math.max(...HO);
N.hoSd = Math.sqrt(HO.reduce((a, b) => a + (b - hoM) ** 2, 0) / (HO.length - 1));
const E = D.es;
N.esBest = argmin(E.va) + 1; N.esApAt = E.ap[N.esBest - 1]; N.esApMaxEp = argmax(E.ap) + 1; N.esApMax = E.ap[N.esApMaxEp - 1];
N.esApEnd = E.ap.at(-1); N.esEp = E.va.length; N.esStop = N.esBest + 20; N.esVmin = E.va[N.esBest - 1]; N.esVend = E.va.at(-1);
const DD = D.dd, ddTe = DD.te.map(med), ddTr = DD.tr.map(med);
N.n = DD.n; N.ddPeakI = argmax(ddTe); N.ddPeakP = DD.p[N.ddPeakI]; N.ddPeak = ddTe[N.ddPeakI];
const under = DD.p.map((p, i) => i).filter(i => DD.p[i] < DD.n);
N.ddBestI = under.reduce((b, i) => (ddTe[i] < ddTe[b] ? i : b), under[0]);
N.ddBestP = DD.p[N.ddBestI]; N.ddBest = ddTe[N.ddBestI];
N.ddBigP = DD.p.at(-1); N.ddBig = ddTe.at(-1); N.ddSmallP = DD.p[0]; N.ddSmall = ddTe[0];
N.ddTrPeak = ddTr[N.ddPeakI];
N.ddRiseI = ddTe.findIndex(v => v > 10); N.ddRiseP = DD.p[N.ddRiseI]; N.ddRise = ddTe[N.ddRiseI]; N.ddSeeds = DD.te[0].length;

// ------------------------------------------------------------------ rounds 1-2: one network, 150 epochs (2D curves)
const EPT = [1, 25, 50, 75, 100, 125, 150].map(e => [e, String(e)]);
function lossChart(root, upto) {
    const c = chart(root.querySelector('.lab-stage'), {
        x: { lo: 1, hi: EP }, y: { lo: 1e-4, hi: 4, log: true }, xlab: T.ep, ylab: T.lossY, xt: EPT,
        yt: [[1e-4, fmt(0.0001, 4)], [1e-3, fmt(0.001, 3)], [0.01, fmt(0.01, 2)], [0.1, fmt(0.1, 1)], [0.5, fmt(0.5, 1)], [1, '1'], [3, '3']],
    });
    c.hline(N.logit, 'lin', T.logitLine(fmt(N.logit, 3)), true);
    const xs = C0.va.slice(0, upto).map((_, i) => i + 1);
    c.line(xs, C0.tr.slice(0, upto), 'trn');
    c.line(xs, C0.va.slice(0, upto), 'main');
    if (upto) { c.dot(upto, C0.tr[upto - 1], 'trn-dot'); c.dot(upto, C0.va[upto - 1], 'main-dot'); }
    c.el('text', { x: c.W - c.mr - 6, y: c.Y(0.05), class: 'leg-main', 'text-anchor': 'end' }, T.legVal);
    c.el('text', { x: c.W - c.mr - 6, y: c.Y(0.05) + 16, class: 'leg-trn', 'text-anchor': 'end' }, T.legTr);
}
function aucChart(root, upto) {
    const c = chart(root.querySelector('.lab-stage'), {
        x: { lo: 1, hi: EP }, y: { lo: 0.6, hi: 0.82 }, xlab: T.ep, ylab: T.aucY, xt: EPT,
        yt: [0.6, 0.65, 0.7, 0.75, 0.8].map(v => [v, fmt(v, 2)]),
    });
    c.hline(N.logitAuc, 'lin', T.logitLine(fmt(N.logitAuc, 3)), true);
    const xs = C0.auc.slice(0, upto).map((_, i) => i + 1);
    c.line(xs, C0.auc.slice(0, upto), 'main');
    if (upto) c.dot(upto, C0.auc[upto - 1], 'main-dot');
    if (upto >= N.best) c.vline(N.best, 'mark', T.bestLoss(N.best));
}
round('r1', {
    activate: root => lossChart(root, 0),
    run: (root, out, done) => {
        const n = +root.querySelector('#r1-k').value;
        play(n, k => {
            lossChart(root, k + 1);
            out.innerHTML = T.r1read(k + 1, fmt(C0.tr[k], 4), fmt(C0.va[k], 3), fmt(N.logit, 3));
        }, done);
    },
}, N);
bindRange($('r1'), '#r1-k', v => String(v));
$('r1').querySelector('#r1-k').addEventListener('input', e => lossChart($('r1'), +e.target.value));

round('r2', {
    activate: root => aucChart(root, 0),
    run: (root, out, done) => {
        const n = +root.querySelector('#r2-k').value;
        play(n, k => {
            aucChart(root, k + 1);
            out.innerHTML = T.r2read(k + 1, fmt(C0.va[k], 3), fmt(C0.auc[k], 3), fmt(N.logitAuc, 3));
        }, done);
    },
}, N);
bindRange($('r2'), '#r2-k', v => String(v));
$('r2').querySelector('#r2-k').addEventListener('input', e => aucChart($('r2'), +e.target.value));

// ------------------------------------------------------------------ round 3: the dropout x weight-decay grid in 3D
const COLS = [[0.18, 0.55, 0.34], [0.88, 0.69, 0.25], [0.75, 0.22, 0.17]];
const ramp = t => { t = Math.min(Math.max(t, 0), 1) * 2; const i = Math.min(Math.floor(t), 1), f = t - i; return COLS[i].map((c, j) => c + f * (COLS[i + 1][j] - c)); };
const HS = 2.0;                                                     // scene units per unit of validation loss
function label(text, size = 0.62, color = '#1c2433') {
    const cv = document.createElement('canvas'), ctx = cv.getContext('2d'), px = 64;
    ctx.font = `600 ${px}px Inter, Arial, sans-serif`;
    cv.width = Math.ceil(ctx.measureText(text).width) + 16; cv.height = px + 20;
    ctx.font = `600 ${px}px Inter, Arial, sans-serif`; ctx.fillStyle = color; ctx.textBaseline = 'middle';
    ctx.fillText(text, 8, cv.height / 2);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sp.scale.set(size * cv.width / cv.height, size, 1);
    return sp;
}
function makeGrid(root) {
    const box = root.querySelector('.lab-stage'), el = document.createElement('div');
    el.className = 'lab-canvas'; box.appendChild(el);
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch (e) { el.innerHTML = `<p class="lab-nogl">${T.nogl}</p>`; return { set() {} }; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    camera.position.set(10.5, 9.0, 12.5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.target.set(0, 1.2, 0); controls.maxPolarAngle = Math.PI * 0.49;
    controls.minDistance = 5; controls.maxDistance = 26;
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(6, 12, 9); scene.add(sun);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(7.6, 7.6), new THREE.MeshStandardMaterial({ color: 0xeef1f6, roughness: 1 }));
    floor.rotation.x = -Math.PI / 2; scene.add(floor);
    const pos = i => (i - 1) * 2.2, bars = {};
    D.ov.dps.forEach((p, i) => D.ov.wds.forEach((wd, j) => {
        const m = new THREE.Mesh(new THREE.BoxGeometry(1.3, 1, 1.3), new THREE.MeshStandardMaterial({ roughness: 0.6 }));
        m.position.set(pos(j), 0.5, pos(i)); scene.add(m);
        const lab = label('', 0.5); scene.add(lab);
        bars[key(p, wd)] = { m, lab, x: pos(j), z: pos(i) };
    }));
    // the logistic-regression benchmark as a translucent plane
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(7.0, 7.0), new THREE.MeshBasicMaterial({ color: 0xd4a017, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
    plane.rotation.x = -Math.PI / 2; plane.position.y = N.logit * HS; scene.add(plane);
    const key3 = document.createElement('p'); key3.className = 'lab-key'; key3.innerHTML = T.planeKey(fmt(N.logit, 3)); box.appendChild(key3);
    D.ov.wds.forEach((wd, j) => { const s = label(T.wdTick(num(wd)), 0.42); s.position.set(pos(j), 0.05, 4.3); scene.add(s); });
    D.ov.dps.forEach((p, i) => { const s = label(T.dpTick(num(p)), 0.42); s.position.set(4.6, 0.05, pos(i)); scene.add(s); });
    let visible = false;
    const resize = () => { const w = el.clientWidth || 600, h = el.clientHeight || 450; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(el); resize();
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(el);
    (function loop() { if (visible) { controls.update(); renderer.render(scene, camera); } requestAnimationFrame(loop); })();
    return {
        set(ep) {
            for (const k in bars) {
                const v = C[k].va[ep - 1], b = bars[k], h = Math.max(v * HS, 0.02);
                b.m.scale.y = h; b.m.position.y = h / 2;
                b.m.material.color.setRGB(...ramp((v - 0.5) / 1.7));
                scene.remove(b.lab); b.lab.material.map.dispose(); b.lab.material.dispose();
                b.lab = label(fmt(v, 3), 0.48); b.lab.position.set(b.x, h + 0.42, b.z); scene.add(b.lab);
            }
        },
    };
}
const g3 = makeGrid($('r3'));
g3.set(1);
const gridRead = ep => {
    const ks = Object.keys(C), v = ks.map(k => C[k].va[ep - 1]), lo = argmin(v), hi = argmax(v);
    const nm = k => { const [p, wd] = k.split('|').map(Number); return T.cell(num(p), num(wd)); };
    return T.r3read(ep, fmt(v[lo], 3), nm(ks[lo]), fmt(v[hi], 3), nm(ks[hi]));
};
round('r3', {
    run: (root, out, done) => {
        const n = +root.querySelector('#r3-k').value;
        play(n, k => { g3.set(k + 1); out.innerHTML = gridRead(k + 1); }, done, 3500);
    },
}, N);
bindRange($('r3'), '#r3-k', v => String(v));
$('r3').querySelector('#r3-k').addEventListener('input', e => { if (!e.target.disabled) { g3.set(+e.target.value); $('r3').querySelector('.lab-out').innerHTML = gridRead(+e.target.value); } });

// ------------------------------------------------------------------ round 4: early stopping on card fraud (two panels)
function esChart(root, upto, stop) {
    const box = root.querySelector('.lab-stage');
    box.innerHTML = '<div class="lab-panel"></div><div class="lab-panel"></div>';
    const [p1, p2] = box.querySelectorAll('.lab-panel'), ET = [1, 20, 40, 60, 80, 100, 120].map(e => [e, String(e)]);
    const xs = E.va.slice(0, upto).map((_, i) => i + 1);
    const a = chart(p1, { H: 250, x: { lo: 1, hi: N.esEp }, y: { lo: 0.012, hi: 0.08, log: true }, xlab: '', ylab: T.esLossY, xt: [],
        yt: [0.015, 0.02, 0.03, 0.05, 0.08].map(v => [v, fmt(v, 3)]) });
    a.line(xs, E.va.slice(0, upto), 'main');
    const b = chart(p2, { H: 270, x: { lo: 1, hi: N.esEp }, y: { lo: 0.2, hi: 0.8 }, xlab: T.ep, ylab: T.esApY, xt: ET,
        yt: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].map(v => [v, fmt(v, 1)]) });
    b.line(xs, E.ap.slice(0, upto), 'ap');
    if (stop) {
        a.vline(stop, 'mark'); b.vline(stop, 'mark');
        a.el('text', { x: a.X(stop) + (stop > 90 ? -5 : 5), y: a.mt + 13, class: 'tick', 'text-anchor': stop > 90 ? 'end' : 'start' }, T.stopAt(stop));
        a.dot(stop, E.va[stop - 1], 'main-dot'); b.dot(stop, E.ap[stop - 1], 'ap-dot');
    }
}
const esRead = e => T.r4read(e, fmt(E.va[e - 1], 4), fmt(E.ap[e - 1], 3));
round('r4', {
    activate: root => esChart(root, 0, 0),
    run: (root, out, done) => {
        play(N.esEp, k => { esChart(root, k + 1, 0); out.innerHTML = esRead(k + 1); }, done, 3600);
    },
    after: root => { const e = +root.querySelector('#r4-s').value; esChart(root, N.esEp, e); root.querySelector('.lab-out').innerHTML = esRead(e); },
}, N);
bindRange($('r4'), '#r4-s', v => String(v));
$('r4').querySelector('#r4-s').addEventListener('input', e => { esChart($('r4'), N.esEp, +e.target.value); $('r4').querySelector('.lab-out').innerHTML = esRead(+e.target.value); });

// ------------------------------------------------------------------ round 5: double descent (random-feature regression)
function ddChart(root, upto) {
    const c = chart(root.querySelector('.lab-stage'), {
        x: { lo: 8, hi: 10000, log: true }, y: { lo: 0.01, hi: 1e9, log: true }, xlab: T.ddX, ylab: T.ddY,
        xt: [10, 25, 50, 100, 200, 400, 1000, 2000, 4000, 8000].map(v => [v, v.toLocaleString(T.lang)]),
        yt: [[0.01, fmt(0.01, 2)], [0.1, fmt(0.1, 1)], [1, '1'], [100, '100'], [1e4, '10⁴'], [1e6, '10⁶'], [1e8, '10⁸']],
    });
    c.vline(N.n, 'mark', T.pn(N.n));
    c.hline(N.ddBest, 'lin', T.ddBestLine(fmt(N.ddBest, 3), N.ddBestP), true);
    for (let i = 0; i < upto; i++) DD.te[i].forEach(v => c.dot(DD.p[i], v, 'seed', 2.6));
    c.line(DD.p.slice(0, upto), ddTr.slice(0, upto).map(v => Math.max(v, 0.0101)), 'trn');
    c.line(DD.p.slice(0, upto), ddTe.slice(0, upto), 'main');
    c.el('text', { x: c.W - c.mr - 6, y: c.mt + 14, class: 'leg-main', 'text-anchor': 'end' }, T.legTe);
    c.el('text', { x: c.W - c.mr - 6, y: c.mt + 30, class: 'leg-trn', 'text-anchor': 'end' }, T.legTrDd);
}
const big = v => (v >= 1e4 ? T.big(fmt(v / 1e6, v >= 1e5 ? 1 : 2)) : fmt(v, v >= 100 ? 0 : 3));
const ddRead = i => T.r5read(DD.p[i].toLocaleString(T.lang), big(ddTe[i]), big(Math.min(...DD.te[i])), big(Math.max(...DD.te[i])), fmt(ddTr[i], 3));
round('r5', {
    activate: root => ddChart(root, 0),
    run: (root, out, done) => {
        const n = +root.querySelector('#r5-p').value + 1;
        play(n, k => { ddChart(root, k + 1); out.innerHTML = ddRead(k); }, done, 3600);
    },
}, N);
bindRange($('r5'), '#r5-p', v => DD.p[v].toLocaleString(T.lang));
$('r5').querySelector('#r5-p').addEventListener('input', e => { const i = +e.target.value; ddChart($('r5'), i + 1); $('r5').querySelector('.lab-out').innerHTML = ddRead(i); });

// start: empty charts until a prediction is locked
lossChart($('r1'), 0); aucChart($('r2'), 0); esChart($('r4'), 0, 0); ddChart($('r5'), 0);
window.LAB_N = N;   // for checks
