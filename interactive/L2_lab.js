// Lecture 2 lab: predict, test, prove. Gradient descent on the real loss surface of the lecture (California housing,
// median house value on median income, 20,640 block groups) and the learning-rate sweep of the 64-64 network.
// Every number is computed here from window.LK_DATA (L2_data.js, built by lectures/L2/build_interactive_L2.py), with
// the formulas of analysis_L2.py; the texts are in L2_lab_text_{en,ro}.js (window.LAB_T).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const D = window.LK_DATA, T = window.LAB_T, RO = T.lang === 'ro';
const $ = id => document.getElementById(id);
const fmt = (x, d) => { const s = Number(x).toFixed(d).replace(/^-/, '−'); return RO ? s.replace('.', ',') : s; };

// ------------------------------------------------------------------ the quadratic loss of value = a + b * income
function surface(kind) {
    const m = D.gd[kind];
    const H = [[2, 2 * m.mx], [2 * m.mx, 2 * m.mxx]], c = [2 * m.my, 2 * m.mxy];
    const tr = H[0][0] + H[1][1], det = H[0][0] * H[1][1] - H[0][1] * H[0][1];
    const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
    const lmax = tr / 2 + disc, lmin = tr / 2 - disc;
    const wstar = [(c[0] * H[1][1] - c[1] * H[0][1]) / det, (H[0][0] * c[1] - H[0][1] * c[0]) / det];
    const loss = (a, b) => m.myy - 2 * (a * m.my + b * m.mxy) + a * a + 2 * a * b * m.mx + b * b * m.mxx;
    const grad = (a, b) => [H[0][0] * a + H[0][1] * b - c[0], H[0][1] * a + H[1][1] * b - c[1]];
    return { kind, lmax, lmin, cond: lmax / lmin, wstar, loss, grad, mstar: loss(wstar[0], wstar[1]) };
}
const S = { raw: surface('raw'), std: surface('std') };
const W0 = { raw: [4.0, -0.5], std: [4.0, -1.0] };          // starting points of analysis_L2.py

function descend(s, w0, eta, steps, beta) {
    let a = w0[0], b = w0[1], va = 0, vb = 0;
    const out = [[a, b, s.loss(a, b)]];
    for (let k = 0; k < steps; k++) {
        const g = s.grad(a, b);
        va = beta * va + g[0]; vb = beta * vb + g[1];
        a -= eta * va; b -= eta * vb;
        const L = s.loss(a, b);
        out.push([a, b, L]);
        if (!isFinite(L) || L > 1e12) break;
    }
    return out;
}

// numbers shared by the texts (the same as L2_numbers.json, "surface")
const R = S.raw, Z = S.std;
const N = {
    lmax: R.lmax, lmin: R.lmin, cond: R.cond, eta: 1 / R.lmax, etaMax: 2 / R.lmax, mstar: R.mstar,
    start: R.loss(...W0.raw), slope: R.wstar[1], intercept: R.wstar[0],
    gd60: descend(R, W0.raw, 1 / R.lmax, 60, 0).at(-1)[2],
    mom60: descend(R, W0.raw, 0.5 / R.lmax, 60, 0.9).at(-1)[2],
    std1: descend(Z, W0.std, 1 / Z.lmax, 1, 0).at(-1)[2], condStd: Z.cond, lStd: Z.lmax,
    flat: 1 - R.lmin / R.lmax,
};
N.flat60 = Math.pow(N.flat, 60);

// learning-rate sweep of the 64-64 network (plain SGD, batch 128, 30 epochs, 3 seeds; median, diverged above 5)
const LRS = D.lr.lrs, key = lr => String(lr).replace(/^1$/, '1.0');
const med = v => { const s = [...v].sort((x, y) => x - y); return s[1]; };
const SW = LRS.map(lr => { const f = D.lr.final[key(lr)]; return { lr, f, div: f.filter(v => v > 5).length, med: Math.max(...f) < 5 ? med(f) : NaN }; });
const okSW = SW.filter(r => isFinite(r.med));
N.best = okSW.reduce((p, r) => (r.med < p.med ? r : p));
N.firstDiv = SW.find(r => !isFinite(r.med));
N.lin = D.mse_lin_val;
N.slow = SW[0];

// ------------------------------------------------------------------ the 3D stages (one per round, each with its own scene)
const A0 = -1, A1 = 5, B0 = -1.2, B1 = 1.4, HALF = 5, HMAX = 4.2, OUT = 7.5;
const toX = a => (a - (A0 + A1) / 2) / ((A1 - A0) / 2) * HALF;
const toZ = b => -(b - (B0 + B1) / 2) / ((B1 - B0) / 2) * HALF;
const TOP = { raw: S.raw.loss(A1, B1), std: S.std.loss(A1, B1) };
const toY = (L, kind) => HMAX * Math.log1p(Math.max(L, 0)) / Math.log1p(TOP[kind]);

const COLS = [[0.10, 0.23, 0.43], [0.36, 0.55, 0.78], [0.86, 0.91, 0.96], [0.93, 0.85, 0.66]];
function ramp(t) {
    t = Math.min(Math.max(t, 0), 1) * (COLS.length - 1);
    const i = Math.min(Math.floor(t), COLS.length - 2), f = t - i;
    return COLS[i].map((c, j) => c + f * (COLS[i + 1][j] - c));
}
function buildSurface(s) {
    const n = 110, kind = s.kind, g = new THREE.BufferGeometry(), pos = [], col = [], idx = [];
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
        const a = A0 + (A1 - A0) * i / n, b = B0 + (B1 - B0) * j / n, y = toY(s.loss(a, b), kind);
        pos.push(toX(a), y, toZ(b)); col.push(...ramp(y / HMAX));
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const p = i * (n + 1) + j;
        idx.push(p, p + n + 1, p + 1, p + 1, p + n + 1, p + n + 2);
    }
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals();
    const grp = new THREE.Group();
    grp.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.0, side: THREE.DoubleSide })));
    // level curves of the loss, drawn in ink on the surface (marching squares)
    const lv = [0.02, 0.06, 0.15, 0.3, 0.5].map(q => s.mstar + q * (TOP[kind] - s.mstar));
    const segs = [], P = (i, j) => [A0 + (A1 - A0) * i / n, B0 + (B1 - B0) * j / n];
    for (const L of lv) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const c = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], v = c.map(q => s.loss(...P(...q)) - L), pts = [];
        for (let k = 0; k < 4; k++) {
            const k2 = (k + 1) % 4;
            if ((v[k] < 0) !== (v[k2] < 0)) {
                const f = v[k] / (v[k] - v[k2]), p1 = P(...c[k]), p2 = P(...c[k2]);
                pts.push([p1[0] + f * (p2[0] - p1[0]), p1[1] + f * (p2[1] - p1[1])]);
            }
        }
        if (pts.length >= 2) for (const q of pts.slice(0, 2)) segs.push(toX(q[0]), toY(L, kind) + 0.02, toZ(q[1]));
    }
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(segs, 3));
    grp.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x1c2433, transparent: true, opacity: 0.5 })));
    const star = new THREE.Mesh(new THREE.OctahedronGeometry(0.2), new THREE.MeshStandardMaterial({ color: 0xd4a017, emissive: 0x6b4f00 }));
    star.position.set(toX(s.wstar[0]), toY(s.mstar, kind) + 0.25, toZ(s.wstar[1]));
    grp.add(star);
    return grp;
}

function makeStage(root, kinds) {
    const box = root.querySelector('.lab-stage'), el = document.createElement('div');
    el.className = 'lab-canvas'; box.appendChild(el);
    let renderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch (e) { el.innerHTML = `<p class="lab-nogl">${T.nogl}</p>`; return { show() {}, place() { return false; } }; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    camera.position.set(12.5, 10.5, 13.5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.target.set(0, 1.0, 0); controls.maxPolarAngle = Math.PI * 0.49;
    controls.minDistance = 6; controls.maxDistance = 30;
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(6, 14, 8); scene.add(sun);
    const meshes = {};
    for (const k of kinds) meshes[k] = buildSurface(S[k]);
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.2, 24, 16), new THREE.MeshStandardMaterial({ color: 0xc0392b, emissive: 0x400000 }));
    const trailGeo = new THREE.BufferGeometry(), trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xc0392b }));
    scene.add(ball, trail);
    let shown = null, visible = false;
    const resize = () => {
        const w = el.clientWidth || 600, h = el.clientHeight || 450;
        renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(el); resize();
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(el);
    (function loop() { if (visible) { controls.update(); renderer.render(scene, camera); } requestAnimationFrame(loop); })();
    const st = {
        kind: kinds[0],
        show(kind) { if (shown) scene.remove(meshes[shown]); shown = kind; st.kind = kind; scene.add(meshes[kind]); },
        place(path, k) {
            const pts = [];
            let off = false;
            for (let i = 0; i <= k && i < path.length; i++) {
                const [a, b, L] = path[i], x = toX(a), z = toZ(b), y = toY(L, st.kind) + 0.2;
                if (Math.abs(x) > OUT || Math.abs(z) > OUT || y > HMAX * 2.2 || !isFinite(y)) { off = true; break; }
                pts.push(x, y, z);
            }
            trailGeo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
            trailGeo.computeBoundingSphere();
            if (pts.length) ball.position.set(pts.at(-3), pts.at(-2), pts.at(-1));
            ball.visible = !off;
            return off;
        },
    };
    st.show(kinds[0]);
    const s0 = S[kinds[0]], w0 = W0[kinds[0]];
    st.place([[w0[0], w0[1], s0.loss(...w0)]], 0);
    return st;
}

// ------------------------------------------------------------------ rounds
const score = {};
function setScore() {
    const n = Object.keys(score).length, ok = Object.values(score).filter(Boolean).length;
    $('lab-score').textContent = n ? T.score(ok, n) : '';
}

function round(id, cfg) {
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
            if (!ran) { ran = true; why.disabled = false; ctl.querySelectorAll('input,select').forEach(x => (x.disabled = false)); root.querySelector('.lab-step-prove').classList.add('on'); }
        }, ran);
    });
    why.addEventListener('click', () => {
        const right = pick === r.answer;
        score[id] = right; setScore();
        proof.innerHTML = `<p class="lab-verdict ${right ? 'ok' : 'no'}">${right ? T.right : T.wrong(r.opts[r.answer])}</p>` + r.proof(N, fmt);
        proof.hidden = false; why.disabled = true;
        if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([proof]);
    });
}

let timer = null;
function animate(path, step, out, done, describe) {
    clearInterval(timer);
    let k = 0;
    const delay = Math.max(12, Math.min(60, 3000 / path.length));
    timer = setInterval(() => {
        const off = step(k);
        out.innerHTML = describe(path, Math.min(k, path.length - 1), off);
        if (off || k >= path.length - 1) { clearInterval(timer); done(); }
        k++;
    }, delay);
}
const readout = (s, path, k, off) => {
    const [a, b, L] = path[k];
    return T.readout(k, fmt(a, 3), fmt(b, 3), off ? '∞' : fmt(L, 4), fmt(s.mstar, 4), off);
};
function bindRange(root, sel, show) {
    const el = root.querySelector(sel), lab = root.querySelector(sel + '-v');
    const f = () => { lab.innerHTML = show(+el.value); };
    el.addEventListener('input', f); f();
    return el;
}

// round 1: the narrow valley
const st1 = makeStage($('r1'), ['raw']), st2 = makeStage($('r2'), ['raw']), st3 = makeStage($('r3'), ['std', 'raw']);
const start = (st, kind) => { st.show(kind); st.place([[...W0[kind], S[kind].loss(...W0[kind])]], 0); };
function play(st, path, out, done) {
    animate(path, k => st.place(path, k), out, done, (p, i, off) => readout(S[st.kind], p, i, off));
}
round('r1', {
    run: (root, out, done) => {
        const m = +root.querySelector('#r1-e').value, k = +root.querySelector('#r1-k').value;
        play(st1, descend(S.raw, W0.raw, m / S.raw.lmax, k, 0), out, done);
    },
});
bindRange($('r1'), '#r1-e', v => `${fmt(v, 2)} × 1/λ<sub>max</sub> = ${fmt(v / R.lmax, 4)}`);
bindRange($('r1'), '#r1-k', v => String(v));

// round 2: just above the edge
round('r2', {
    run: (root, out, done) => {
        const m = +root.querySelector('#r2-e').value;
        play(st2, descend(S.raw, W0.raw, m / S.raw.lmax, 60, 0), out, done);
    },
});
bindRange($('r2'), '#r2-e', v => `${fmt(v, 2)} × 1/λ<sub>max</sub> = ${fmt(v / R.lmax, 4)}`);

// round 3: two fixes
round('r3', {
    run: (root, out, done) => {
        const mode = root.querySelector('input[name="r3-m"]:checked').value;
        if (mode === 'std') { start(st3, 'std'); play(st3, descend(S.std, W0.std, 1 / S.std.lmax, 60, 0), out, done); }
        else { start(st3, 'raw'); play(st3, descend(S.raw, W0.raw, 0.5 / S.raw.lmax, 60, 0.9), out, done); }
    },
});
$('r3').querySelectorAll('input[name="r3-m"]').forEach(x => x.addEventListener('change', () => start(st3, x.value === 'std' ? 'std' : 'raw')));

// round 4: a real network (2D chart of the validation curves)
const SVGNS = 'http://www.w3.org/2000/svg';
function chart4(root, idx, upto, all) {
    const box = root.querySelector('.lab-stage');
    box.innerHTML = '';
    const W = 640, Hh = 380, ml = 58, mr = 16, mt = 16, mb = 46;
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`); svg.setAttribute('class', 'lab-svg');
    const y0 = Math.log10(0.2), y1 = Math.log10(6);
    const X = e => ml + (e - 1) / 29 * (W - ml - mr), Y = v => mt + (1 - (Math.log10(Math.min(Math.max(v, 0.2), 6)) - y0) / (y1 - y0)) * (Hh - mt - mb);
    const el = (n, a, txt) => { const e = document.createElementNS(SVGNS, n); for (const k in a) e.setAttribute(k, a[k]); if (txt) e.textContent = txt; svg.appendChild(e); return e; };
    for (const v of [0.2, 0.3, 0.5, 1, 2, 5]) {
        el('line', { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), class: 'grid' });
        el('text', { x: ml - 8, y: Y(v) + 4, class: 'tick', 'text-anchor': 'end' }, fmt(v, v < 1 ? 1 : 0));
    }
    for (const e of [1, 5, 10, 15, 20, 25, 30]) el('text', { x: X(e), y: Hh - mb + 18, class: 'tick', 'text-anchor': 'middle' }, String(e));
    el('text', { x: (ml + W - mr) / 2, y: Hh - 6, class: 'axl', 'text-anchor': 'middle' }, T.r4x);
    el('text', { x: 14, y: (mt + Hh - mb) / 2, class: 'axl', 'text-anchor': 'middle', transform: `rotate(-90 14 ${(mt + Hh - mb) / 2})` }, T.r4y);
    el('line', { x1: ml, x2: W - mr, y1: Y(N.lin), y2: Y(N.lin), class: 'lin' });
    el('text', { x: ml + 8, y: Y(N.lin) + 16, class: 'tick' }, T.r4lin(fmt(N.lin, 3)));
    const line = (v, n, cls) => {
        let d = '';
        for (let e = 0; e < n; e++) { const val = v[e]; if (val > 5) break; d += (d ? 'L' : 'M') + X(e + 1).toFixed(1) + ',' + Y(val).toFixed(1); }
        if (d) el('path', { d, class: cls });
        if (n && v[0] <= 5 && v.slice(0, n).some(x => x > 5)) { const e = v.findIndex(x => x > 5); el('text', { x: X(e + 1), y: mt + 12, class: 'div', 'text-anchor': 'middle' }, T.diverged); }
        if (n && v[0] > 5) el('text', { x: X(1) + 4, y: mt + 12, class: 'div' }, T.diverged);
    };
    if (all) LRS.forEach((lr, i) => i !== idx && line(D.lr.curve[key(lr)], 30, 'other'));
    line(D.lr.curve[key(LRS[idx])], upto, 'main');
    box.appendChild(svg);
}
round('r4', {
    activate: root => chart4(root, +root.querySelector('#r4-e').value, 0, false),
    run: (root, out, done, ran) => {
        const i = +root.querySelector('#r4-e').value, v = D.lr.curve[key(LRS[i])], r = SW[i];
        let e = 0;
        clearInterval(timer);
        timer = setInterval(() => {
            e++;
            chart4(root, i, e, ran);
            const cur = v[e - 1];
            out.innerHTML = T.r4read(fmt(LRS[i], LRS[i] < 0.001 ? 4 : 3), e, cur > 5 ? null : fmt(cur, 3), r.f.map(x => (x > 5 ? '∞' : fmt(x, 3))), r.div);
            if (e >= 30) { clearInterval(timer); done(); }
        }, 70);
    },
});
bindRange($('r4'), '#r4-e', v => fmt(LRS[v], LRS[v] < 0.001 ? 4 : LRS[v] < 0.01 ? 3 : LRS[v] < 0.1 ? 2 : 1));
$('r4').querySelector('#r4-e').addEventListener('input', () => chart4($('r4'), +$('r4').querySelector('#r4-e').value, 0, true));

// start
chart4($('r4'), +$('r4').querySelector('#r4-e').value, 0, false);
window.LAB_N = N;   // for checks
