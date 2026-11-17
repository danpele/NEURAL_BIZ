// Lecture 12 lab: predict, test, prove. Temperature on the real next-token distribution of Qwen2.5-1.5B (drawn in 3D),
// zero-shot classification by model size, few-shot examples, the wording of the prompt and retrieval for RAG on the ECB
// speeches. Every number is computed here from window.LK_DATA (L12_data.js, built by lectures/L12/build_interactive_L12.py)
// and window.LAB_NUM (L12_numbers.json of analysis_L12.py, written into the page by build_lab_L12.py); the texts are in
// L12_lab_text_{en,ro}.js (window.LAB_T).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { $, fmt, round, play, bindRange, chart } from './lab_common.js?v=20261118c';

const D = window.LK_DATA, M = window.LAB_NUM, T = window.LAB_T;

// macro-F1 of a prediction string against a label string (classes '0', '1', '2')
function macroF1(y, p) {
    let s = 0;
    for (const c of '012') {
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < y.length; i++) { const a = y[i] === c, b = p[i] === c; if (a && b) tp++; else if (b) fp++; else if (a) fn++; }
        s += tp ? 2 * tp / (2 * tp + fp + fn) : 0;
    }
    return 100 * s / 3;
}
const mean = v => v.reduce((a, b) => a + b, 0) / v.length;

// ------------------------------------------------------------------ numbers shared by the texts
const N = { next: M.next, cls: M.cls.pb, rag: M.rag };
const P1 = D.nt.p1, TK = D.nt.tok;
N.ratio2 = t => Math.pow(P1[1] / P1[0], 1 / t);
N.entMax = Math.log(M.vocab);
const Y = D.cb.pb.y, CB = D.cb.pb;
N.f1 = {}; for (const m in CB) if (m !== 'y') N.f1[m] = macroF1(Y, CB[m].pred);
N.zsNeu = [...CB['zs_0.5B'].pred].filter(c => c === '2').length;          // classes: 0 positive, 1 negative, 2 neutral
const FS = D.fs, KS = [0, 1, 2, 4, 8];
N.fs = {};
for (const sz of ['1.5B', '3B']) N.fs[sz] = KS.map(k => [0, 1, 2].map(s => macroF1(FS.y, FS[`${sz}_k${k}_s${s}`])));
N.fsMean = sz => N.fs[sz].map(mean);
const PS = D.ps;
N.ps = {}; for (const ds of ['pb', 'tw']) for (const sz of ['1.5B', '3B']) N.ps[`${ds}_${sz}`] = [1, 2, 3, 4, 5].map(i => macroF1(PS[ds].y, PS[ds][`${sz}_i${i}`]));
N.range = k => Math.max(...N.ps[k]) - Math.min(...N.ps[k]);
const RG = D.rg, RMS = ['tfidf', 'bm25', 'dense', 'hybrid'];
const half = x => { const y = x * 10, f = Math.floor(y), d = y - f; return (Math.abs(d - 0.5) < 1e-9 ? (f % 2 ? f + 1 : f) : Math.round(y)) / 10; };   // round half to even, as Python
N.rec = (m, k) => half(100 * RG[m].filter(r => r !== null && r <= k).length / RG[m].length);

// ------------------------------------------------------------------ round 1: temperature, drawn as a 3D ridge
const TS = [0.25, 0.5, 0.75, 1, 1.5, 2];
const TGRID = Array.from({ length: 36 }, (_, i) => 0.25 + 1.75 * i / 35);
function label(text, size, color = '#1c2433', cx = 0.5) {
    const cv = document.createElement('canvas'), ctx = cv.getContext('2d'), px = 56;
    ctx.font = `600 ${px}px Inter, Arial, sans-serif`;
    cv.width = Math.ceil(ctx.measureText(text).width) + 12; cv.height = px + 16;
    ctx.font = `600 ${px}px Inter, Arial, sans-serif`; ctx.fillStyle = color; ctx.textBaseline = 'middle';
    ctx.fillText(text, 6, cv.height / 2);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sp.scale.set(size * cv.width / cv.height, size, 1); sp.center.set(cx, 0.5);
    return sp;
}
const COLS = [[0.86, 0.90, 0.96], [0.36, 0.55, 0.78], [0.10, 0.23, 0.43], [0.75, 0.22, 0.17]];
const ramp = t => { t = Math.min(Math.max(t, 0), 1) * 3; const i = Math.min(Math.floor(t), 2), f = t - i; return COLS[i].map((c, j) => c + f * (COLS[i + 1][j] - c)); };
function makeRidge(root) {
    const box = root.querySelector('.lab-stage'), el = document.createElement('div');
    el.className = 'lab-canvas'; box.appendChild(el);
    const key = document.createElement('p'); key.className = 'lab-key'; key.innerHTML = T.key3d; box.appendChild(key);
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch (e) { el.innerHTML = `<p class="lab-nogl">${T.nogl}</p>`; return { set() {} }; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    camera.position.set(-10.5, 6.5, 6.0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.target.set(0, 0.9, 0); controls.maxPolarAngle = Math.PI * 0.49;
    controls.minDistance = 5; controls.maxDistance = 26;
    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9); sun.position.set(5, 12, 8); scene.add(sun);
    // surface: x = token (rank 1 to 8), z = temperature, height = probability relative to the most likely token
    const nx = TK.length, nz = TGRID.length, X = i => (i - (nx - 1) / 2) * 0.9, Z = j => (j / (nz - 1) - 0.5) * 7, H = 3.2;
    const g = new THREE.BufferGeometry(), pos = [], col = [], idx = [];
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
        const r = Math.pow(P1[i] / P1[0], 1 / TGRID[j]); pos.push(X(i), r * H, Z(j)); col.push(...ramp(r));
    }
    for (let j = 0; j < nz - 1; j++) for (let i = 0; i < nx - 1; i++) { const p = j * nx + i; idx.push(p, p + 1, p + nx, p + 1, p + nx + 1, p + nx); }
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals();
    scene.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, side: THREE.DoubleSide, transparent: true, opacity: 0.55 })));
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(nx * 0.9 + 0.6, 7.6), new THREE.MeshStandardMaterial({ color: 0xeef1f6, roughness: 1 }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.01; scene.add(floor);
    TK.forEach((t, i) => { const s = label(t.trim() || '␣', 0.34); s.position.set(X(i), 0.02, Z(0) - 0.6); scene.add(s); });
    [0.25, 0.5, 1, 1.5, 2].forEach(t => { const j = Math.round((t - 0.25) / 1.75 * (nz - 1)); const s = label(`T = ${fmt(t, t % 1 ? (t * 4 % 2 ? 2 : 1) : 0)}`, 0.34, '#555', 1); s.position.set(X(nx - 1) + 1.4, 0.02, Z(j)); scene.add(s); });
    // the bars of the current temperature
    const bars = TK.map((_, i) => { const m = new THREE.Mesh(new THREE.BoxGeometry(0.55, 1, 0.22), new THREE.MeshStandardMaterial({ color: 0xc0392b, roughness: 0.5 })); m.position.x = X(i); scene.add(m); return m; });
    let visible = false;
    const resize = () => { const w = el.clientWidth || 600, h = el.clientHeight || 450; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(el); resize();
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(el);
    (function loop() { if (visible) { controls.update(); renderer.render(scene, camera); } requestAnimationFrame(loop); })();
    return {
        set(t) {
            const z = Z((t - 0.25) / 1.75 * (nz - 1));
            bars.forEach((m, i) => { const h = Math.max(Math.pow(P1[i] / P1[0], 1 / t) * H, 0.02); m.scale.y = h; m.position.set(X(i), h / 2, z); });
        },
    };
}
const st1 = makeRidge($('r1'));
st1.set(1);
const r1read = t => {
    const known = { 0.5: ['0.5', M.next.top1_p05], 1: ['1.0', M.next.top1_p], 2: ['2.0', M.next.top1_p2] }[t];
    const ent = known ? M.next.entropy['t' + known[0].replace('.', '')] : null;
    return T.r1read(fmt(t, 2), fmt(N.ratio2(t), 2), known ? fmt(known[1], 1) : null, ent !== null && ent !== undefined ? fmt(ent, 2) : null);
};
round('r1', {
    run: (root, out, done) => {
        const t1 = TS[+root.querySelector('#r1-t').value], n = 30;
        play(n, k => { const t = 1 + (t1 - 1) * k / (n - 1); st1.set(t); out.innerHTML = k === n - 1 ? r1read(t1) : r1read(Math.round(t * 100) / 100); }, done, 2400);
    },
}, N);
bindRange($('r1'), '#r1-t', v => fmt(TS[v], 2));
$('r1').querySelector('#r1-t').addEventListener('input', e => { const t = TS[+e.target.value]; st1.set(t); $('r1').querySelector('.lab-out').innerHTML = r1read(t); });

// ------------------------------------------------------------------ round 2: zero-shot by model size (PhraseBank, 678 sentences)
const R2 = ['majority', 'tfidf', 'zs_0.5B', 'zs_1.5B', 'zs_3B', 'probe'];
const key2 = m => m.replace('.', '').replace('zs_05B', 'zs_05B');
function dotChart(root, rows, upto, val, ci, xlo, xhi, xlab, hl) {
    const c = chart(root.querySelector('.lab-stage'), {
        H: 360, x: { lo: xlo, hi: xhi }, y: { lo: -0.5, hi: rows.length - 0.5 }, xlab, ylab: '',
        xt: Array.from({ length: 6 }, (_, i) => xlo + i * (xhi - xlo) / 5).map(v => [v, String(Math.round(v))]), yt: [],
    });
    rows.forEach((m, i) => {
        const y = rows.length - 1 - i;
        c.el('line', { x1: c.ml, x2: c.W - c.mr, y1: c.Y(y), y2: c.Y(y), class: 'grid' });
        c.el('text', { x: c.ml + 6, y: c.Y(y) - 13, class: 'tick' + (hl(m) ? ' hl' : '') }, T.model[m] || m);
        if (i < upto) {
            const r = ci(m);
            if (r) c.el('line', { x1: c.X(r[0]), x2: c.X(r[1]), y1: c.Y(y), y2: c.Y(y), class: 'meanbar' });
            c.dot(val(m), y, hl(m) ? 'main-dot' : 'seed', 5.5);
        }
    });
    return c;
}
const ci2 = m => { const r = M.cls.pb[key2(m)]; return r && r.lo !== undefined ? [r.lo, r.hi] : null; };
const draw2 = (root, n) => dotChart(root, R2, n, m => (m === 'majority' ? M.cls.pb.majority.f1 : N.f1[m]), ci2, 20, 100, T.f1pb, m => m.startsWith('zs'));
round('r2', {
    activate: root => draw2(root, 0),
    run: (root, out, done) => { play(R2.length, k => { draw2(root, k + 1); const m = R2[k], r = ci2(m); out.innerHTML = T.r2read(T.model[m], fmt(m === 'majority' ? M.cls.pb.majority.f1 : N.f1[m], 1), r ? fmt(r[0], 1) : '', r ? fmt(r[1], 1) : ''); }, done, 3000); },
}, N);

// ------------------------------------------------------------------ round 3: few-shot examples
function fsChart(root, upto) {
    const c = chart(root.querySelector('.lab-stage'), {
        x: { lo: -0.5, hi: 4.5 }, y: { lo: 80, hi: 95 }, xlab: T.shots, ylab: T.f1short,
        xt: KS.map((k, i) => [i, String(k)]), yt: [80, 85, 90, 95].map(v => [v, String(v)]),
    });
    for (const [sz, cls] of [['1.5B', 'trn'], ['3B', 'main']]) {
        const mu = N.fsMean(sz).slice(0, upto);
        c.line(mu.map((_, i) => i), mu, cls);
        N.fs[sz].slice(0, upto).forEach((v, i) => v.forEach(x => c.dot(i, x, cls + '-dot', 3)));
        if (upto) c.el('text', { x: c.X(0) + 8, y: c.Y(N.fsMean(sz)[0]) - 10, class: 'leg-' + cls }, `Qwen2.5-${sz}`);
    }
}
round('r3', {
    activate: root => fsChart(root, 0),
    run: (root, out, done) => { play(KS.length, k => { fsChart(root, k + 1); out.innerHTML = T.r3read(KS[k], fmt(N.fsMean('3B')[k], 1), N.fs['3B'][k].map(x => fmt(x, 1)).join(T.lang === 'ro' ? '; ' : ', '), fmt(N.fsMean('1.5B')[k], 1)); }, done, 3000); },
}, N);

// ------------------------------------------------------------------ round 4: five wordings of the same instruction
function psView(root, ds, upto) {
    const box = root.querySelector('.lab-stage'), ins = D.ps.instr[ds];
    box.innerHTML = `<table class="lab-table"><thead><tr><th>#</th><th>${T.instr}</th><th>1.5B</th><th>3B</th></tr></thead><tbody>` +
        ins.map((s, i) => `<tr><td>${i + 1}</td><td class="ins">${s}</td><td class="${i < upto ? 'got' : ''}">${i < upto ? fmt(N.ps[`${ds}_1.5B`][i], 1) : '…'}</td><td>${i < upto ? fmt(N.ps[`${ds}_3B`][i], 1) : '…'}</td></tr>`).join('') +
        `</tbody></table><p class="lab-cmnote">${T.psNote(ds === 'pb' ? T.pb : T.tw)}</p>`;
}
const dsSel = () => $('r4').querySelector('input[name="r4-d"]:checked').value;
round('r4', {
    activate: root => psView(root, dsSel(), 0),
    run: (root, out, done) => { const ds = dsSel(); play(5, k => { psView(root, ds, k + 1); out.innerHTML = T.r4read(k + 1, fmt(N.ps[`${ds}_1.5B`][k], 1)); }, () => { out.innerHTML = T.r4done(fmt(N.range(`${ds}_1.5B`), 1), fmt(N.range(`${ds}_3B`), 1)); done(); }, 3000); },
}, N);
$('r4').querySelectorAll('input[name="r4-d"]').forEach(x => x.addEventListener('change', () => { const ds = dsSel(); psView($('r4'), ds, 5); $('r4').querySelector('.lab-out').innerHTML = T.r4done(fmt(N.range(`${ds}_1.5B`), 1), fmt(N.range(`${ds}_3B`), 1)); }));

// ------------------------------------------------------------------ round 5: retrieval for RAG (recall at k)
function ragChart(root, kmax) {
    const c = chart(root.querySelector('.lab-stage'), {
        x: { lo: 1, hi: 10 }, y: { lo: 30, hi: 90 }, xlab: T.k, ylab: T.recall,
        xt: Array.from({ length: 10 }, (_, i) => [i + 1, String(i + 1)]), yt: [30, 40, 50, 60, 70, 80, 90].map(v => [v, v + '%']),
    });
    const cls = { tfidf: 'other', bm25: 'trn', dense: 'ap', hybrid: 'main' };
    for (const m of RMS) {
        const ks = Array.from({ length: kmax }, (_, i) => i + 1);
        c.line(ks, ks.map(k => N.rec(m, k)), cls[m]);
        if (kmax) c.el('text', { x: c.X(kmax) + (kmax > 6 ? -6 : 6), y: c.Y(N.rec(m, kmax)) + (m === 'bm25' ? 14 : -6), class: 'tick', 'text-anchor': kmax > 6 ? 'end' : 'start' }, T.ret[m]);
    }
}
const ragRead = k => T.r5read(k, RMS.map(m => `${T.ret[m]} ${fmt(N.rec(m, k), 1)}%`).join(', '));
round('r5', {
    activate: root => ragChart(root, 0),
    run: (root, out, done) => { const n = +root.querySelector('#r5-k').value; play(n, k => { ragChart(root, k + 1); out.innerHTML = ragRead(k + 1); }, done, 2500); },
}, N);
bindRange($('r5'), '#r5-k', v => String(v));
$('r5').querySelector('#r5-k').addEventListener('input', e => { const k = +e.target.value; ragChart($('r5'), k); $('r5').querySelector('.lab-out').innerHTML = ragRead(k); });

// start
draw2($('r2'), 0); fsChart($('r3'), 0); ragChart($('r5'), 0);
window.LAB_N = N;   // for checks
