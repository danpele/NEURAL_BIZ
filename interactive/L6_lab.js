// Lecture 6 lab: predict, test, prove. Zipf's law in the speeches of the European Central Bank (ECB), our own word2vec
// drawn as a 3D word space, analogies, mean embeddings against TF-IDF, and the agreement of the PhraseBank annotators.
// Every number is computed here from window.LK_DATA (L6_data.js, built by lectures/L6/build_interactive_L6.py) and
// window.LAB_NUM (L6_numbers.json of analysis_L6.py, written into the page by build_lab_L6.py); the texts are in
// L6_lab_text_{en,ro}.js (window.LAB_T).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { $, fmt, round, play, bindRange, chart } from './lab_common.js?v=20261118b';

const D = window.LK_DATA, M = window.LAB_NUM, T = window.LAB_T;
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const cos = (a, b) => dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));

// ------------------------------------------------------------------ numbers shared by the texts
const N = {};
const F = D.zf.freq, TOT = F.reduce((a, b) => a + b, 0), CUM = [];
F.reduce((a, b, i) => (CUM[i] = a + b), 0);
N.tot = TOT; N.types = F.length; N.top10 = CUM[9] / TOT; N.top100 = CUM[99] / TOT;
N.hapax = F.filter(v => v === 1).length; N.hapaxPct = N.hapax / F.length; N.slope = D.zf.slope;
N.r50 = CUM.findIndex(c => c >= TOT / 2) + 1; N.r90 = CUM.findIndex(c => c >= 0.9 * TOT) + 1;
N.top = D.zf.top.map(w => w.replace('<', '&lt;').replace('>', '&gt;'));
const W = D.pc.w2v, GR = D.pc.groups, WORDS = Object.values(GR).flat();
const grp = {}; for (const g in GR) for (const w of GR[g]) grp[w] = g;
N.nb = D.nb.nb; N.cosIP = cos(W.inflation, W.prices); N.cosIC = M.cos.inflation_climate; N.cosII = cos(W.inflation, W.interest);
N.nbInf = D.nb.nb.w2v.inflation; N.nbInfG = D.nb.nb.glove.inflation;
N.rankPrices = D.nb.nb.w2v.inflation[0].indexOf('prices');
N.analogy = M.analogy; N.corpusTok = M.corpus.n_tokens; N.V = M.w2v.V;
const P = M.sent.pb, Q = M.sent.tw;
N.pb = P; N.tw = Q; N.gapPb = P.tfidf_lr.f1 - P.w2v_lr.f1; N.gapTw = Q.tfidf_lr.f1 - Q.w2v_lr.f1; N.w2vGain = M.sent.pb_w2v_gain;
N.oov = M.oov;
const CM = D.ag.cm.tfidf_lr, MAJ = D.ag.cm.majority;
N.acc = k => CM[k].acc; N.maj = k => MAJ[k].acc;
N.n = k => CM[k].M.flat().reduce((a, b) => a + b, 0);

// ------------------------------------------------------------------ round 1: Zipf's law (log-log rank-frequency chart)
function zipfChart(root, upto) {
    const c = chart(root.querySelector('.lab-stage'), {
        x: { lo: 1, hi: 30000, log: true }, y: { lo: 1, hi: 1e5, log: true }, xlab: T.rank, ylab: T.freq,
        xt: [1, 10, 100, 1000, 10000].map(v => [v, v.toLocaleString(T.lang)]), yt: [1, 10, 100, 1000, 1e4, 1e5].map(v => [v, v.toLocaleString(T.lang)]),
    });
    const xs = [], ys = [];
    for (let r = 1; r <= upto; r = r < 100 ? r + 1 : Math.ceil(r * 1.04)) { xs.push(r); ys.push(F[r - 1]); }
    if (upto && xs.at(-1) !== upto) { xs.push(upto); ys.push(F[upto - 1]); }
    // the fitted line of the lecture: ln f = icpt + slope ln r, least squares over ranks 10 to 10,000
    const fit = r => Math.exp(D.zf.icpt) * Math.pow(r, D.zf.slope);
    c.line([10, 10000], [fit(10), fit(10000)], 'lin');
    c.el('text', { x: c.X(10000) + 4, y: c.Y(fit(10000)) - 6, class: 'tick' }, T.fitLab(fmt(D.zf.slope, 2)));
    c.line(xs, ys, 'main');
    if (upto) c.dot(upto, F[upto - 1], 'main-dot');
    for (let i = 0; i < Math.min(upto, 3); i++) c.el('text', { x: c.X(i + 1) + 6, y: c.Y(F[i]) - 4 + 13 * (i % 2), class: 'tick' }, D.zf.top[i]);
}
const RANKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, F.length];
const zipfRead = r => T.r1read(r.toLocaleString(T.lang), F[r - 1].toLocaleString(T.lang), fmt(100 * CUM[r - 1] / TOT, 1));
round('r1', {
    activate: root => zipfChart(root, 0),
    run: (root, out, done) => {
        const n = +root.querySelector('#r1-r').value + 1;
        play(n, k => { zipfChart(root, RANKS[k]); out.innerHTML = zipfRead(RANKS[k]); }, done, 2800);
    },
}, N);
bindRange($('r1'), '#r1-r', v => RANKS[v].toLocaleString(T.lang));
$('r1').querySelector('#r1-r').addEventListener('input', e => { const r = RANKS[+e.target.value]; zipfChart($('r1'), r); $('r1').querySelector('.lab-out').innerHTML = zipfRead(r); });

// ------------------------------------------------------------------ round 2: the word space in 3D (PCA of our word2vec)
function pca3(vecs) {
    const n = vecs.length, d = vecs[0].length, mu = Array(d).fill(0);
    vecs.forEach(v => v.forEach((x, j) => (mu[j] += x / n)));
    const X = vecs.map(v => v.map((x, j) => x - mu[j]));
    let G = X.map(a => X.map(b => dot(a, b)));                     // Gram matrix: its eigenvectors give the PCA scores
    const tr = G.reduce((s, r, i) => s + r[i], 0), out = [], lam = [];
    for (let k = 0; k < 3; k++) {
        let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1 + k));
        for (let it = 0; it < 300; it++) {
            const w = G.map(r => dot(r, v)), s = Math.sqrt(dot(w, w));
            v = w.map(x => x / s);
        }
        const l = dot(v, G.map(r => dot(r, v)));
        lam.push(l); out.push(v.map(x => x * Math.sqrt(l)));
        G = G.map((r, i) => r.map((x, j) => x - l * v[i] * v[j]));
    }
    return { xyz: vecs.map((_, i) => [out[0][i], out[1][i], out[2][i]]), share: lam.map(l => l / tr) };
}
const PC = pca3(WORDS.map(w => W[w]));
N.pc3 = PC.share.reduce((a, b) => a + b, 0); N.pc2 = PC.share[0] + PC.share[1];
const GCOL = { monetary: 0xc0392b, energy: 0xd4a017, banking: 0x2e5c9a, digital: 0x1e7b3c };
const GCSS = { monetary: '#c0392b', energy: '#a07400', banking: '#2e5c9a', digital: '#1e7b3c' };
function label(text, size, color = '#1c2433') {
    const cv = document.createElement('canvas'), ctx = cv.getContext('2d'), px = 56;
    ctx.font = `600 ${px}px Inter, Arial, sans-serif`;
    cv.width = Math.ceil(ctx.measureText(text).width) + 12; cv.height = px + 16;
    ctx.font = `600 ${px}px Inter, Arial, sans-serif`; ctx.fillStyle = color; ctx.textBaseline = 'middle';
    ctx.fillText(text, 6, cv.height / 2);
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sp.scale.set(size * cv.width / cv.height, size, 1); sp.center.set(0, 0.5);
    return sp;
}
function makeSpace(root) {
    const box = root.querySelector('.lab-stage'), el = document.createElement('div');
    el.className = 'lab-canvas'; box.appendChild(el);
    const key = document.createElement('p'); key.className = 'lab-key';
    key.innerHTML = Object.keys(GR).map(g => `<span class="sw" style="background:${GCSS[g]}"></span>${D.gname[T.lang][g]}`).join(' · ');
    box.appendChild(key);
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch (e) { el.innerHTML = `<p class="lab-nogl">${T.nogl}</p>`; return { probe() {} }; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    camera.position.set(7, 5, 9);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.autoRotate = true; controls.autoRotateSpeed = 0.6;
    controls.minDistance = 3; controls.maxDistance = 25;
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9); sun.position.set(5, 10, 7); scene.add(sun);
    const ext = Math.max(...PC.xyz.flat().map(Math.abs)), S = 3.2 / ext;
    const pos = {}, balls = {};
    WORDS.forEach((w, i) => {
        const p = PC.xyz[i].map(x => x * S); pos[w] = new THREE.Vector3(...p);
        const m = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 14), new THREE.MeshStandardMaterial({ color: GCOL[grp[w]] }));
        m.position.copy(pos[w]); scene.add(m); balls[w] = m;
        const s = label(w, 0.3, GCSS[grp[w]]); s.position.copy(pos[w]).add(new THREE.Vector3(0.14, 0.08, 0)); scene.add(s);
    });
    const axes = new THREE.AxesHelper(0.8); axes.material.transparent = true; axes.material.opacity = 0.35; scene.add(axes);
    const lines = new THREE.Group(); scene.add(lines);
    root.addEventListener('pointerdown', () => { controls.autoRotate = false; });
    let visible = false;
    const resize = () => { const w = el.clientWidth || 600, h = el.clientHeight || 450; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(el); resize();
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(el);
    (function loop() { if (visible) { controls.update(); renderer.render(scene, camera); } requestAnimationFrame(loop); })();
    return {
        probe(w, frac = 1) {
            lines.clear();
            for (const b of Object.values(balls)) b.scale.setScalar(1);
            if (!W[w]) return;
            balls[w].scale.setScalar(1.8);
            WORDS.filter(v => v !== w).map(v => [v, cos(W[w], W[v])]).sort((a, b) => b[1] - a[1]).slice(0, Math.round(7 * frac))
                .forEach(([v, c]) => {
                    const g = new THREE.BufferGeometry().setFromPoints([pos[w], pos[v]]);
                    lines.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x1c2433, transparent: true, opacity: Math.max(0.15, Math.min(1, c * 1.6)) })));
                });
        },
    };
}
const sp2 = makeSpace($('r2'));
const probeW = () => $('r2').querySelector('#r2-w').value;
const r2read = (w, k) => {
    const nb = D.nb.nb.w2v[w], gl = D.nb.nb.glove[w];
    const list = nb[0].slice(0, k).map((v, i) => `${v} ${fmt(nb[1][i], 2)}`).join(', ');
    const inCloud = W[w] ? WORDS.filter(v => v !== w).map(v => [v, cos(W[w], W[v])]).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([v, c]) => `${v} ${fmt(c, 2)}`).join(', ') : '';
    return T.r2read(w, list, gl[0].slice(0, 5).join(', '), inCloud);
};
round('r2', {
    run: (root, out, done) => {
        const w = probeW();
        play(12, k => { sp2.probe(w, (k + 1) / 12); out.innerHTML = r2read(w, k + 1); }, done, 2600);
    },
}, N);
$('r2').querySelector('#r2-w').addEventListener('change', () => { const w = probeW(); sp2.probe(w); $('r2').querySelector('.lab-out').innerHTML = r2read(w, 12); });

// ------------------------------------------------------------------ round 3: analogies (b - a + c, nearest word)
const AN = Object.entries(M.analogy);
function anList(root, upto) {
    const box = root.querySelector('.lab-stage');
    box.innerHTML = `<table class="lab-table"><thead><tr><th>${T.anQ}</th><th>${T.anExp}</th><th>${T.anGot}</th></tr></thead><tbody>` +
        AN.map(([q, got], i) => {
            const [ab, c] = q.split('::'), [a, b] = ab.split(':');
            return `<tr><td>${a} : ${b} :: ${c} : ?</td><td>${T.expect[q] || ''}</td><td class="${i < upto ? 'got' : ''}">${i < upto ? got : '…'}</td></tr>`;
        }).join('') + '</tbody></table>';
}
round('r3', {
    activate: root => anList(root, 0),
    run: (root, out, done) => { play(AN.length, k => { anList(root, k + 1); out.innerHTML = T.r3read(k + 1, AN.length); }, done, 2500); },
}, N);

// ------------------------------------------------------------------ round 4: TF-IDF against mean word2vec (macro-F1)
const MODS = ['majority', 'bow_lr', 'tfidf_lr', 'tfidf_mlp', 'w2v_lr', 'embbag', 'embbag_w2v'];
function f1Chart(root, upto) {
    const c = chart(root.querySelector('.lab-stage'), {
        H: 400, x: { lo: 20, hi: 80 }, y: { lo: -0.5, hi: MODS.length - 0.5 }, xlab: T.f1, ylab: '',
        xt: [20, 30, 40, 50, 60, 70, 80].map(v => [v, String(v)]), yt: [],
    });
    MODS.forEach((m, i) => {
        const y = MODS.length - 1 - i, hl = m === 'tfidf_lr' || m === 'w2v_lr';
        c.el('line', { x1: c.ml, x2: c.W - c.mr, y1: c.Y(y), y2: c.Y(y), class: 'grid' });
        c.el('text', { x: c.ml + 6, y: c.Y(y) - 13, class: 'tick' + (hl ? ' hl' : '') }, D.ag.mname[T.lang][m]);
        if (i < upto) {
            const r = P[m];
            c.el('line', { x1: c.X(r.f1 - r.f1_sd), x2: c.X(r.f1 + r.f1_sd), y1: c.Y(y), y2: c.Y(y), class: 'meanbar' });
            c.dot(r.f1, y, hl ? 'main-dot' : 'seed', 5);
            c.dot(Q[m].f1, y, 'trn-dot', 3.5);
        }
    });
    c.el('circle', { cx: c.W - c.mr - 190, cy: c.H - c.mb - 26, r: 5, class: 'main-dot' });
    c.el('text', { x: c.W - c.mr - 180, y: c.H - c.mb - 22, class: 'tick' }, T.legPb);
    c.el('circle', { cx: c.W - c.mr - 190, cy: c.H - c.mb - 10, r: 3.5, class: 'trn-dot' });
    c.el('text', { x: c.W - c.mr - 180, y: c.H - c.mb - 6, class: 'tick' }, T.legTw);
}
round('r4', {
    activate: root => f1Chart(root, 0),
    run: (root, out, done) => {
        play(MODS.length, k => { f1Chart(root, k + 1); const m = MODS[k]; out.innerHTML = T.r4read(D.ag.mname[T.lang][m], fmt(P[m].f1, 1), fmt(P[m].f1_sd, 1), fmt(Q[m].f1, 1)); }, done, 3000);
    },
}, N);

// ------------------------------------------------------------------ round 5: annotator agreement (confusion matrix)
const AG = ['50', '66', '75', '100', 'all'];
function cmChart(root, k) {
    const box = root.querySelector('.lab-stage'), Mx = CM[k].M, n = Mx.flat().reduce((a, b) => a + b, 0);
    const rows = Mx.map(r => r.reduce((a, b) => a + b, 0));
    box.innerHTML = `<table class="lab-cm"><thead><tr><th>${T.trueLab} \\ ${T.predLab}</th>${T.cls.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>` +
        Mx.map((r, i) => `<tr><th>${T.cls[i]}</th>` + r.map((v, j) => {
            const p = v / rows[i], bg = i === j ? `rgba(30,123,60,${0.15 + 0.75 * p})` : `rgba(192,57,43,${Math.min(0.9, 0.08 + 1.6 * p)})`;
            return `<td style="background:${bg};color:${p > 0.55 ? '#fff' : '#1c2433'}"><b>${v.toLocaleString(T.lang)}</b><br><span>${fmt(100 * p, 0)}%</span></td>`;
        }).join('') + '</tr>').join('') + `</tbody></table><p class="lab-cmnote">${T.cmNote(T.agName(k), n.toLocaleString(T.lang), fmt(CM[k].acc, 1), fmt(MAJ[k].acc, 1))}</p>`;
}
const cmRead = k => T.r5read(T.agName(k), fmt(CM[k].acc, 1), fmt(MAJ[k].acc, 1), fmt(CM[k].acc - MAJ[k].acc, 1));
round('r5', {
    activate: root => { root.querySelector('.lab-stage').innerHTML = ''; },
    run: (root, out, done) => { play(4, k => { cmChart(root, AG[k]); root.querySelector('#r5-a').value = k; root.querySelector('#r5-a-v').textContent = T.agName(AG[k]); out.innerHTML = cmRead(AG[k]); }, done, 3200); },
}, N);
bindRange($('r5'), '#r5-a', v => T.agName(AG[v]));
$('r5').querySelector('#r5-a').addEventListener('input', e => { const k = AG[+e.target.value]; cmChart($('r5'), k); $('r5').querySelector('.lab-out').innerHTML = cmRead(k); });

// start
zipfChart($('r1'), 0); f1Chart($('r4'), 0);
window.LAB_N = N;   // for checks
