// ============================================================
// NN course website (shell shared with the SFM, TSA and BA sites) - rendering + quiz engine
// Reads window.NN_DATA (course-data.js), window.NN_CONFIG (config.js)
// and quiz banks registered in NN_DATA.quizzes by lecture id (assets/quizzes/<id>.js).
// Lectures are referenced by their stable `id`; `num` is only the display order.
// Language is set by the page shell: window.NN_LANG = 'en' | 'ro'.
// ============================================================
(function () {
    'use strict';

    const LANG = window.NN_LANG === 'ro' ? 'ro' : 'en';
    const D = window.NN_DATA;
    const T = D.ui[LANG];
    const CFG = window.NN_CONFIG || {};
    const LETTERS = ['A', 'B', 'C', 'D', 'E', 'F'];

    const isConfigured = v => typeof v === 'string' && v.length > 0 && !v.startsWith('YOUR_');
    const $ = id => document.getElementById(id);
    const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

    // values still to be supplied by the instructor start with 'TODO' in course-data.js;
    // on the page they read "To be announced"
    const tbd = s => (typeof s === 'string' && /^TODO/.test(s)) ? `<span class="tbd">${T.tbd}</span>` : s;

    function typeset(el) {
        if (window.MathJax && MathJax.typesetPromise) {
            MathJax.typesetPromise(el ? [el] : undefined).then(() => fitMath()).catch(() => {});
        }
    }

    // shrink display formulas that are wider than their box, so no sideways scrolling is needed (as on the MFM site)
    function fitMath(root) {
        (root || document).querySelectorAll('.cf-tex').forEach(box => {
            const m = box.querySelector('mjx-container');
            if (!m || !box.clientWidth) return;
            m.style.fontSize = '';
            const w = m.scrollWidth, avail = box.clientWidth - 4;
            if (w > avail) m.style.fontSize = Math.max(55, Math.floor(100 * avail / w)) + '%';
        });
    }
    let fitTimer;
    window.addEventListener('resize', () => { clearTimeout(fitTimer); fitTimer = setTimeout(() => fitMath(), 150); });
    document.addEventListener('toggle', e => {
        if (e.target.classList && e.target.classList.contains('chapter-formulas') && e.target.open) fitMath(e.target);
    }, true);

    // ------------------------------------------------------------
    // STATIC TEXT (header, nav, section titles)
    // ------------------------------------------------------------
    function renderStatic() {
        document.title = T.pageTitle;
        document.documentElement.lang = LANG;
        document.querySelectorAll('[data-t]').forEach(el => {
            const path = el.getAttribute('data-t').split('.');
            let v = T;
            for (const p of path) v = v && v[p];
            if (typeof v === 'string') el.innerHTML = v;
        });
    }

    const cards = list => list.map(c =>
        `<div class="info-card"><h3>${c.h}</h3>${c.p.map(p => `<p>${tbd(p)}</p>`).join('')}</div>`
    ).join('');

    // ------------------------------------------------------------
    // OVERVIEW, OBJECTIVES
    // ------------------------------------------------------------
    function renderOverview() {
        $('overview-cards').innerHTML = cards(D.overview[LANG]);
        $('objectives-list').innerHTML = D.objectives[LANG].map(o => `<li>${o}</li>`).join('');
    }

    // ------------------------------------------------------------
    // LECTURE CARDS
    // Each lecture lists its files (lecture PDF, notebook, code folder). A file is linked when
    // the lecture number is in READY or when the file is found on the server (HEAD request),
    // so a card switches from "in preparation" to links as soon as the PDF is published.
    // ------------------------------------------------------------
    const READY = [];   // lectures whose PDF and notebook are published (add the number when lecture k is released); others are probed
    // lecture buttons in labelled rows, as on the MFM site: lecture (slides, web page, interactive), code (notebook + Colab, Quantlets)
    const LINK_ROWS = [['course', ['slides', 'web', 'interactive']], ['code', ['notebook', 'quantlets']]];
    const LINK_CLASS = { slides: 'btn-primary', web: 'btn-outline', interactive: 'btn-outline', notebook: 'btn-outline', quantlets: 'btn-outline' };
    const avail = {};   // avail[id] = { slides, notebook, quantlets, web, interactive }

    // web lecture (web/LK.html) and interactive companion (interactive/LK.html), _ro on the Romanian page;
    // their buttons appear only once the file is found on the server (probePages), never as "in preparation"
    const pageOf = (type, l) => `${type}/L${l.num}${LANG === 'ro' ? '_ro' : ''}.html`;
    // web page of a research case study (web/cases/CK.html, CK_ro.html; built by cases/common/build_web_case.py)
    const studyPage = s => `web/cases/C${s.num}${LANG === 'ro' ? '_ro' : ''}.html`;

    function linkButton(type, l, ok) {
        const label = T.links[type];
        if (type === 'web' || type === 'interactive') return ok ? `<a href="${pageOf(type, l)}" class="btn ${LINK_CLASS[type]}">${label}</a>` : '';
        if (!ok) return `<span class="btn disabled" aria-disabled="true" title="${T.comingSoon}">${label} · ${T.comingSoonShort}</span>`;
        if (type === 'slides') return `<a href="${(LANG === 'ro' && l.files.pdf_ro) || l.files.pdf}" class="btn ${LINK_CLASS[type]}" target="_blank" rel="noopener">${label}</a>`;
        if (type === 'notebook') {
            const nb = (LANG === 'ro' && l.files.notebook_ro) || l.files.notebook;
            return `<span class="link-group"><a href="${D.repo}/blob/main/${nb}" class="btn btn-outline" target="_blank" rel="noopener">${label}</a>` +
                `<a href="${D.colab}${nb}" class="btn btn-colab" target="_blank" rel="noopener" title="${T.links.colab}">Colab</a></span>`;
        }
        return `<a href="${D.repo}/tree/main/${l.files.quantlets}" class="btn btn-outline" target="_blank" rel="noopener" title="${T.links.quantletsTitle}"><img class="btn-ql" src="assets/logos/ql_logo.png" alt="">${label}</a>`;
    }

    // ------------------------------------------------------------
    // CHART CAROUSEL (assets/showcase.js, generated from the Quantlets by assets/make_showcase.py)
    // the image opens the lightbox, the caption opens the Quantlet that draws the chart
    // ------------------------------------------------------------
    const chartsOf = id => (D.lectureCharts || {})[id] || [];
    const slideIdx = {};   // current chart of each lecture card

    function showcase(l) {
        const cs = chartsOf(l.id);
        if (!cs.length) return '';
        const i = Math.min(slideIdx[l.id] || 0, cs.length - 1), c = cs[i];
        const nav = cs.length > 1
            ? `<button type="button" class="sc-arrow" data-step="-1" aria-label="${T.prevChart}">&#8249;</button>` +
              `<span class="sc-count">${i + 1} / ${cs.length}</span>` +
              `<button type="button" class="sc-arrow" data-step="1" aria-label="${T.nextChart}">&#8250;</button>`
            : '';
        return `<div class="chapter-showcase" data-id="${l.id}">
            <button type="button" class="chapter-chart" data-src="${c.src}" data-cap="${esc(c[LANG])}" data-url="${c.url}" title="${T.enlarge}">
                <img src="${c.src}" alt="${esc(c[LANG])}" loading="lazy"></button>
            <div class="sc-bar">
                <a class="sc-cap" href="${c.url}" target="_blank" rel="noopener" title="${T.openQuantlet}: ${c.q}">${esc(c[LANG])}</a>
                <div class="sc-nav">${nav}</div>
            </div>
        </div>`;
    }

    function stepChart(id, step) {
        const n = chartsOf(id).length;
        if (!n) return;
        slideIdx[id] = ((slideIdx[id] || 0) + step + n) % n;
        const box = document.querySelector(`.chapter-showcase[data-id="${id}"]`);
        if (box) box.outerHTML = showcase(D.lectures.find(x => x.id === id));
        const next = chartsOf(id)[(slideIdx[id] + 1) % n];   // preload the next chart
        if (next) { const im = new Image(); im.src = next.src; }
    }

    // one delegated handler for every card (cards are re-rendered when the probe finds their files)
    function initShowcase() {
        $('chapters-grid').addEventListener('click', e => {
            const arrow = e.target.closest('.sc-arrow');
            if (arrow) { stepChart(arrow.closest('.chapter-showcase').dataset.id, parseInt(arrow.dataset.step, 10)); return; }
            const fig = e.target.closest('.chapter-chart');
            if (fig) openLightbox(fig.dataset.src, fig.dataset.cap, fig.dataset.url);
        });
        $('chapters-grid').addEventListener('keydown', e => {
            const box = e.target.closest && e.target.closest('.chapter-showcase');
            if (!box || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
            e.preventDefault();
            stepChart(box.dataset.id, e.key === 'ArrowLeft' ? -1 : 1);
            const nb = document.querySelector(`.chapter-showcase[data-id="${box.dataset.id}"] .sc-arrow[data-step="${e.key === 'ArrowLeft' ? -1 : 1}"]`);
            if (nb) nb.focus();
        });
    }

    // ------------------------------------------------------------
    // LIGHTBOX (lecture charts)
    // ------------------------------------------------------------
    function openLightbox(src, caption, url) {
        $('lightbox-img').src = src;
        $('lightbox-img').alt = caption;
        $('lightbox-cap').innerHTML = esc(caption) +
            (url ? ` <a href="${url}" target="_blank" rel="noopener">${T.openQuantlet} &#8599;</a>` : '');
        $('lightbox').hidden = false;
        document.body.classList.add('no-scroll');
        $('lightbox-close').focus();
    }

    function initLightbox() {
        const box = $('lightbox');
        const close = () => { box.hidden = true; document.body.classList.remove('no-scroll'); };
        $('lightbox-close').setAttribute('aria-label', T.close);
        $('lightbox-close').onclick = close;
        box.onclick = e => { if (e.target === box) close(); };
        document.addEventListener('keydown', e => { if (e.key === 'Escape' && !box.hidden) close(); });
    }

    // on the narrow lecture cards, put the parts of a multi-part formula on separate lines (as on the MFM site)
    const stackTex = tex => /\\qquad/.test(tex) && !/\\begin\{/.test(tex)
        ? '$$\\begin{gathered}' + tex.slice(2, -2).replace(/,?\s*\\qquad\s*/g, () => ' \\\\ ') + '\\end{gathered}$$'
        : tex;

    function lectureCard(l) {
        const a = avail[l.id] || {};
        const ready = !!a.slides;
        const badge = ready ? '' : `<span class="badge badge-soon">${l.status === 'prep' ? T.comingSoon : T.coming}</span>`;
        const anyFile = a.slides || a.notebook || a.quantlets;
        const links = (l.status === 'prep' || anyFile)
            ? LINK_ROWS.map(([row, types]) =>
                `<div class="link-row"><span class="link-row-h">${T.rows[row]}</span><div class="link-row-b">${types.map(t => linkButton(t, l, a[t])).join('')}</div></div>`).join('')
            : `<p class="chapter-soon">${T.chapterSoon}</p>`;
        const fs = (D.formulas || []).filter(f => f.ch === l.id);
        const formulas = fs.length
            ? `<details class="chapter-formulas"><summary>${T.formulas} (${fs.length})</summary>` +
              fs.map(f => `<div class="cf"><h5>${f[LANG]}</h5><div class="cf-tex">${stackTex((LANG === 'ro' && f.tex_ro) || f.tex)}</div></div>`).join('') +
              '</details>'
            : '';
        // research case study of the lecture (lecture k <-> Ck; C8 is listed under the lectures it relates to)
        const studies = (D.studies || []).filter(s => s.lectures.includes(l.id));
        const study = studies.length
            ? `<p class="lecture-study"><strong>${studies.length > 1 ? T.studyPlural : T.studyShort}:</strong> ${studies.map(s =>
                `<a href="${studyPage(s)}">C${s.num}. ${s.title[LANG]}</a>`).join('; ')}</p>` : '';
        const cases = study + ((l.cases || []).length
            ? `<p class="lecture-cases"><strong>${T.casesShort}:</strong> ${l.cases.map(id => {
                const c = (D.cases || []).find(x => x.id === id);
                return c ? `<a href="#episode-${id}">${c[LANG].h}</a>` : '';
            }).join(', ')}</p>` : '');
        const vids = (l.videos || []).length
            ? `<div class="quantinar-box video-box"><h4>${T.videos}</h4><ul>` +
              l.videos.map(v => `<li><a href="https://www.youtube.com/watch?v=${v.id}" target="_blank" rel="noopener">${esc(v.title)}</a> <span class="muted-note">${esc(v.by)}</span></li>`).join('') + '</ul></div>'
            : '';
        const further = (l.further || []).length
            ? `<div class="quantinar-box"><h4>${T.further}</h4><ul>` +
              l.further.map(f => `<li><a href="${f.url}" target="_blank" rel="noopener">${f[LANG]}</a></li>`).join('') + '</ul></div>'
            : '';
        const qn = (l.quantinar || []).length
            ? `<div class="quantinar-box"><h4><img src="assets/logos/qr_logo.png" alt="">${T.quantinar}</h4><ul>` +
              l.quantinar.map(c => `<li><a href="${c.url}" target="_blank" rel="noopener">${esc(c.title)}</a></li>`).join('') + '</ul></div>'
            : '';
        return `<div class="chapter-card${ready ? '' : ' soon'}" id="lecture-${l.id}">
            <div class="chapter-header"><h3>${T.chapter} ${l.num}: ${l.title[LANG]}</h3><div class="badges">${badge}</div></div>
            ${showcase(l)}
            <div class="chapter-body">
                <ul>${l.topics[LANG].map(t => `<li>${t}</li>`).join('')}</ul>
                ${cases}
                ${formulas}
                <div class="chapter-links">${links}</div>
                ${vids}
                ${further}
                ${qn}
            </div>
        </div>`;
    }

    function renderLectures() {
        D.lectures.forEach(l => { if (READY.includes(l.num)) avail[l.id] = { slides: true, notebook: true, quantlets: true }; });
        $('chapters-grid').innerHTML = D.lectures.map(lectureCard).join('');
    }

    const exists = path => fetch(path, { method: 'HEAD', cache: 'no-store' }).then(r => r.ok).catch(() => false);

    function rerenderLecture(l) {
        const old = $('lecture-' + l.id);
        if (old) { old.outerHTML = lectureCard(l); typeset($('lecture-' + l.id)); }
    }

    // On GitHub Pages: every published file listed by ONE request to the GitHub API (recursive tree of main), so the
    // site never sends a request for a file that does not exist (no 404 in the console). Elsewhere (local server), or when
    // the API fails or is rate-limited: null, and only lectures with status 'ready' are probed with HEAD requests.
    let _published;
    function publishedFiles() {
        if (_published !== undefined) return _published;
        if (!/\.github\.io$/.test(location.hostname)) return (_published = Promise.resolve(null));
        const api = D.repo.replace('https://github.com/', 'https://api.github.com/repos/') + '/git/trees/main?recursive=1';
        return (_published = fetch(api).then(r => r.ok ? r.json() : Promise.reject(r.status))
            .then(x => new Set(x.tree.filter(e => e.type === 'blob' || e.type === 'tree').map(e => e.path)))
            .catch(() => null));
    }

    // web page and interactive page of every lecture: a button appears only when its file is published
    function probePages() {
        publishedFiles().then(pub => D.lectures.forEach(l => {
            if (!pub && l.status !== 'ready' && !READY.includes(l.num)) return;
            const has = p => pub ? Promise.resolve(pub.has(p)) : exists(p);
            // interactive companions are probed only for lectures that declare one (interactive: true in course-data.js),
            // so a local copy sends no request for a page that does not exist
            Promise.all([has(pageOf('web', l)), l.interactive ? has(pageOf('interactive', l)) : Promise.resolve(false)]).then(([w, i]) => {
                if (!w && !i) return;
                avail[l.id] = Object.assign(avail[l.id] || {}, { web: w, interactive: i });
                rerenderLecture(l);
            });
        }));
    }

    function probeLectures() {
        publishedFiles().then(pub => D.lectures.forEach(l => {
            if (READY.includes(l.num)) return;
            if (!pub && l.status !== 'ready') return;          // local copy: lectures in preparation are not probed
            const f = l.files || {};
            const has = p => pub ? Promise.resolve(!!p && pub.has(p)) : exists(p || '');
            // the notebook and the Quantlets are linked together with the PDF
            has(f.pdf).then(s => {
                if (!s) return;
                return has(f.notebook).then(n => {
                    avail[l.id] = Object.assign(avail[l.id] || {}, { slides: true, notebook: n, quantlets: chartsOf(l.id).length > 0 || (pub ? pub.has(f.quantlets) : false) });
                    rerenderLecture(l);
                });
            });
        }));
    }

    // ------------------------------------------------------------
    // CASE STUDIES, DATA
    // ------------------------------------------------------------
    // lectures that use a case study (from lectures[].cases), linked to their cards
    function usedIn(id) {
        const ls = D.lectures.filter(l => (l.cases || []).includes(id));
        return ls.length ? `<p class="case-lecture">${T.usedIn}: ${ls.map(l => `<a href="#lecture-${l.id}">${T.chapter} ${l.num}</a>`).join(', ')}</p>` : '';
    }

    // research case studies C1-C8: one card each, linking to the web page, the brief, the notebook (GitHub, Colab) and the lectures
    function studyCard(s) {
        const S = T.study, sfx = LANG === 'ro' ? 'RO' : 'EN';
        const nb = `cases/${s.folder}/C${s.num}_notebook_${sfx}.ipynb`;
        const lect = s.lectures.map(id => D.lectures.find(l => l.id === id)).filter(Boolean);
        const tag = s.extra ? S.extra : `${S.lecture} ${lect[0].num}`;
        const paper = s.paper
            ? `<p class="study-paper"><span class="study-k">${S.paper}:</span> ${s.paper.cite} <a href="${s.paper.url}" target="_blank" rel="noopener">${s.paper.link}</a></p>`
            : `<p class="study-paper"><span class="study-k">${S.paper}:</span> ${S.noPaper}</p>`;
        const lectures = `<p class="case-lecture">${s.extra ? S.related : S.lecture}: ${lect.map(l =>
            `<a href="#lecture-${l.id}">${T.chapter} ${l.num}: ${l.title[LANG]}</a>`).join(', ')}</p>`;
        return `<article class="study-card" id="case-c${s.num}">
            <div class="study-head"><span class="study-num">C${s.num}</span><span class="study-tag">${tag}</span></div>
            <h3><a href="${studyPage(s)}">${s.title[LANG]}</a></h3>
            <p class="study-sum">${s.summary[LANG]}</p>
            ${paper}
            <div class="study-links">
                <a class="btn btn-primary" href="${studyPage(s)}">${S.page}</a>
                <a class="btn btn-outline" href="cases/${s.folder}/C${s.num}_case_${sfx}.pdf" target="_blank" rel="noopener">${S.brief}</a>
                <span class="link-group"><a class="btn btn-outline" href="${D.repo}/blob/main/${nb}" target="_blank" rel="noopener">${S.notebook}</a><a class="btn btn-colab" href="${D.colab}${nb}" target="_blank" rel="noopener" title="${S.colab}">Colab</a></span>
            </div>
            ${lectures}
        </article>`;
    }

    function renderCases() {
        const studies = D.studies || [];
        $('cases-grid').innerHTML = studies.length ? studies.map(studyCard).join('') : `<p class="tbd">${T.casesSoon}</p>`;
        if (!$('episodes-grid')) return;
        $('episodes-grid').innerHTML = (D.cases || []).map(c =>
            `<div class="info-card case-card" id="episode-${c.id}"><h3>${c[LANG].h}</h3>${c[LANG].p.map(p => `<p>${p}</p>`).join('')}` +
            (c.quantlet ? `<p class="case-link"><a class="btn btn-outline" href="${c.quantlet}" target="_blank" rel="noopener"><img class="btn-ql" src="assets/logos/ql_logo.png" alt="">${T.caseQuantlet}</a></p>` : '') +
            usedIn(c.id) + `</div>`
        ).join('');
    }

    const loc = v => (v && typeof v === 'object') ? v[LANG] : v;

    const fileLink = f => `<a href="data/clean/${encodeURIComponent(f)}" download><code>${esc(f)}</code></a>`;
    function renderData() {
        const d = D.data;
        $('data-intro').innerHTML = d.intro[LANG];
        const head = T.dataCols.map(h => `<th>${h}</th>`).join('');
        $('data-table').innerHTML = `<table class="sessions data-table"><thead><tr>${head}</tr></thead><tbody>` +
            d.series.map(s => `<tr><td>${s.name[LANG]}<br>${s.file ? [].concat(s.file).map(fileLink).join('<br>') : `<span class="muted-note">${T.onDemand}</span>`}</td><td>${s.freq[LANG]}</td>` +
                `<td class="c">${loc(s.coverage)}</td><td>${loc(s.units)}</td>` +
                `<td><a href="${s.href}" target="_blank" rel="noopener">${s.source}</a>${s.note ? `<br><span class="muted-note">${s.note[LANG]}</span>` : ''}</td></tr>`).join('') +
            '</tbody></table>';
        $('data-files').innerHTML = `<h3 class="sub">${T.dataFiles}</h3><ul class="data-files">` +
            d.files.map(f => `<li>${fileLink(f.file)} <span class="muted-note">${LANG === 'ro' ? f.size.replace(".", ",") : f.size}</span></li>`).join('') +
            `</ul><p><a href="https://github.com/danpele/NEURAL_BIZ/tree/main/data/clean" target="_blank" rel="noopener">${T.dataRepo} ↗</a></p>`;
        $('data-licence').innerHTML = d.licence[LANG];
    }

    // ------------------------------------------------------------
    // TEAM PROJECT AND AI POLICY (section #project, same layout as the MFM site)
    // an item is a string or a [label, value] pair; a value that starts with 'TODO' reads "To be announced"
    // ------------------------------------------------------------
    const item = x => Array.isArray(x) ? `<strong>${x[0]}:</strong> ${tbd(x[1])}` : tbd(x);

    function renderProject() {
        const P = D.project;
        $('project-cards').innerHTML = P.cards[LANG].map(c =>
            `<div class="info-card project-card"><h3>${c.h}</h3>${c.p.map(p => `<p>${item(p)}</p>`).join('')}</div>`
        ).join('');
        $('ai-policy').innerHTML = P.ai[LANG].map(o => `<li>${item(o)}</li>`).join('');
    }

    // ------------------------------------------------------------
    // RESOURCES, CONTACT, FOOTER
    // ------------------------------------------------------------
    function renderResources() {
        $('resources-list').innerHTML = D.resources.map(r =>
            `<a href="${r.href}" class="resource-item" target="_blank" rel="noopener">
                ${r.img ? `<img class="resource-logo" src="${r.img}" alt="">` : `<div class="resource-icon">${r.icon}</div>`}
                <div><strong>${r[LANG][0]}</strong><p>${r[LANG][1]}</p></div>
            </a>`
        ).join('');
        $('bibliography').innerHTML = D.bibliography.map(g =>
            `<h4 class="biblio-group">${g.h[LANG]} <span class="muted-note">(${g.items.length})</span></h4>` +
            `<ol class="biblio">${g.items.map(b => `<li>${b}</li>`).join('')}</ol>`).join('');
        $('abbreviations').innerHTML = D.abbreviations.map(([a, en, ro]) =>
            `<div class="abbr-row"><dt>${a}</dt><dd>${LANG === 'ro' && ro ? ro : en}</dd></div>`).join('');
    }

    function renderContact() {
        const c = D.contact;
        $('contact-cards').innerHTML =
            `<div class="info-card"><h3>${T.instructor}</h3><p><strong>${c.name}</strong></p>` +
            c[LANG].map(p => `<p>${p}</p>`).join('') +
            `<p style="margin-top:0.6rem"><a href="mailto:${c.email}">${c.email}</a></p>` +
            `<p style="margin-top:0.8rem"><a class="btn btn-primary" href="https://danpele.github.io/${LANG === 'ro' ? 'index_ro.html' : ''}" target="_blank" rel="noopener">${LANG === 'ro' ? 'Site-ul personal' : 'Personal website'}</a></p></div>` +
            `<div class="info-card"><h3>${T.office}</h3><p>${tbd(c.office[LANG])}</p></div>`;
        $('footer-logos').innerHTML = D.footerLogos.map(([href, src, alt]) =>
            `<a href="${href}" target="_blank" rel="noopener"><img src="${src}" alt="${alt}"></a>`
        ).join('');
        $('footer-text').innerHTML = `&copy; 2026 ${T.footer}`;
        document.querySelector('.hero-inner').insertAdjacentHTML('afterbegin', `<figure class="motto"><blockquote lang="it">“Tristo è quel discepolo che non avanza il suo maestro.”</blockquote>` +
            `<p>${LANG === 'ro' ? 'Trist este discipolul care nu își depășește maestrul.' : 'Poor is the pupil who does not surpass his master.'}</p><figcaption>— Leonardo da Vinci</figcaption></figure>`);
    }

    // ------------------------------------------------------------
    // GOOGLE SIGN-IN (ASE accounts) - used by the quizzes only when configured in config.js
    // ------------------------------------------------------------
    const ALLOWED = ['ase.ro', 'stud.ase.ro'];
    const session = {
        get(k) { try { return sessionStorage.getItem('nn-' + k); } catch (e) { return null; } },
        set(k, v) { try { sessionStorage.setItem('nn-' + k, v); } catch (e) { /* ignore */ } },
        del(k) { try { sessionStorage.removeItem('nn-' + k); } catch (e) { /* ignore */ } }
    };

    function decodeJwt(t) {
        try {
            const b = t.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
            return JSON.parse(decodeURIComponent(atob(b).split('').map(c =>
                '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)).join('')));
        } catch (e) { return null; }
    }

    function getUser() {
        const cred = session.get('google-credential');
        const u = cred && decodeJwt(cred);
        if (!u || u.exp * 1000 < Date.now()) return null;
        return { name: u.name || u.email, email: u.email, credential: cred };
    }

    const loginEnabled = () => isConfigured(CFG.GOOGLE_CLIENT_ID);

    function onGoogleCredential(resp) {
        const u = decodeJwt(resp.credential);
        const domain = u && String(u.email).split('@')[1];
        if (!u || !ALLOWED.includes(domain)) {
            $('login-msg').textContent = T.loginWrongDomain;
            return;
        }
        session.set('google-credential', resp.credential);
        renderLogin();
        if (activeLecture) showQuiz(activeLecture);
    }

    function gisInit() {
        if (gisInit.done || !(window.google && google.accounts && google.accounts.id)) return !!gisInit.done;
        gisInit.done = true;
        google.accounts.id.initialize({ client_id: CFG.GOOGLE_CLIENT_ID, auto_select: true, callback: onGoogleCredential });
        return true;
    }

    function renderLogin() {
        const box = $('quiz-login');
        if (!loginEnabled()) { box.style.display = 'none'; return; }
        if (!document.querySelector('script[src*="accounts.google.com/gsi"]')) {
            const s = document.createElement('script');
            s.src = 'https://accounts.google.com/gsi/client'; s.async = true; s.onload = () => renderLogin();
            document.head.appendChild(s);
        }
        box.style.display = '';
        const user = getUser();
        if (user) {
            box.innerHTML = `<p>${T.loggedAs} <strong>${esc(user.name)}</strong> (${esc(user.email)})
                <button class="btn btn-outline" id="g-logout" style="margin-left:0.5rem;padding:0.2rem 0.6rem">${T.logout}</button></p>`;
            $('g-logout').onclick = () => {
                session.del('google-credential');
                if (window.google && google.accounts) google.accounts.id.disableAutoSelect();
                renderLogin();
                if (activeLecture) showQuiz(activeLecture);
            };
            return;
        }
        box.innerHTML = `<p>${T.loginPrompt}</p><div id="g-button"></div><p class="login-msg" id="login-msg" aria-live="polite"></p>`;
        const el = $('g-button');
        if (el && gisInit()) {
            google.accounts.id.renderButton(el, { theme: 'outline', size: 'large', text: 'signin_with', locale: LANG, width: 280 });
        }
    }

    function sendResults(l, score, total) {
        const user = getUser();
        const out = $('quiz-save-status');
        if (!isConfigured(CFG.QUIZ_SCORES_URL) || !user) return;
        out.textContent = T.saving;
        // text/plain avoids a CORS preflight; the backend parses the JSON body
        fetch(CFG.QUIZ_SCORES_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'text/plain;charset=utf-8' },
            body: JSON.stringify({ credential: user.credential, chapter: l.id, score, total, lang: LANG })
        }).then(r => r.json()).then(j => {
            out.textContent = j.ok ? T.saved : (j.error === 'token' ? T.saveExpired : T.saveFailed);
        }).catch(() => { out.textContent = T.saveFailed; });
    }

    // ------------------------------------------------------------
    // QUIZ ENGINE
    // Bank format (assets/quizzes/<id>.js), loaded by a <script> tag in index.html / index_ro.html:
    //   NN_DATA.quizzes['<id>'] = { draw: 20, questions: [
    //     { correct: <index 0-3>, en: {title, text, options:[4], correctExplanation, incorrectExplanation}, ro: {...} } ] }
    // ------------------------------------------------------------
    let quiz = null;
    let activeLecture = null;

    function shuffle(arr) {
        const a = arr.slice();
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    function renderQuizTabs() {
        $('quiz-tabs').innerHTML = D.lectures.map(l => {
            const has = !!D.quizzes[l.id];
            return `<button class="btn btn-outline${has ? '' : ' unavailable'}" data-ch="${l.id}" title="${esc(l.title[LANG])}">${T.chapterShort} ${l.num}</button>`;
        }).join('');
        $('quiz-tabs').querySelectorAll('button').forEach(b => {
            b.onclick = () => showQuiz(b.getAttribute('data-ch'));
        });
    }

    function showQuiz(id) {
        activeLecture = id;
        $('quiz-tabs').querySelectorAll('button').forEach(b =>
            b.classList.toggle('active', b.getAttribute('data-ch') === id));
        const l = D.lectures.find(c => c.id === id);
        const bank = D.quizzes[id];
        const box = $('quiz-container');
        const heading = `<h3>${T.chapter} ${l.num}: ${l.title[LANG]}</h3>`;
        if (!bank) {
            quiz = null;
            box.innerHTML = `${heading}<p class="quiz-intro">${T.quizSoon}</p>`;
            return;
        }
        if (loginEnabled() && !getUser()) {
            quiz = null;
            box.innerHTML = `${heading}<p class="quiz-intro">${T.loginRequired}</p>`;
            return;
        }
        box.innerHTML = `${heading}
            <p class="quiz-intro">${T.quizIntro}</p>
            <div id="quiz-questions"></div>
            <div class="quiz-actions">
                <button class="btn btn-primary" id="quiz-score">${T.calc}</button>
                <button class="btn btn-outline" id="quiz-reset">${T.reset}</button>
            </div>
            <p class="score-display" id="quiz-score-display" aria-live="polite"></p>
            <p class="save-status" id="quiz-save-status" aria-live="polite"></p>
            <div id="quiz-answer-key"></div>`;
        $('quiz-score').onclick = calculateScore;
        $('quiz-reset').onclick = () => showQuiz(id);
        newAttempt(l, bank);
    }

    function newAttempt(l, bank) {
        const n = Math.min(bank.draw || 20, bank.questions.length);
        const items = shuffle(bank.questions).slice(0, n).map(q => {
            const order = shuffle(q[LANG].options.map((_, i) => i));   // order[pos] = original option index
            return { q, order, correctPos: order.indexOf(q.correct) };
        });
        quiz = { lecture: l, items, answers: {} };

        $('quiz-questions').innerHTML = items.map((it, k) => {
            const L = it.q[LANG];
            return `<div class="quiz-question" id="qq${k}">
                <h4>${T.question} ${k + 1}: ${L.title}</h4>
                <p>${L.text}</p>
                <ul class="quiz-options" role="radiogroup">
                    ${it.order.map((orig, pos) =>
                        `<li tabindex="0" role="radio" aria-checked="false" data-k="${k}" data-pos="${pos}">
                            <input type="radio" name="qq${k}" tabindex="-1"> ${LETTERS[pos]}) ${L.options[orig]}</li>`).join('')}
                </ul>
                <div class="answer-reveal answer-correct" id="qq${k}-ok"><strong>${T.correct}</strong> ${L.correctExplanation}</div>
                <div class="answer-reveal answer-incorrect" id="qq${k}-ko"><strong>${T.incorrect}</strong>
                    ${T.correctIs} ${LETTERS[it.correctPos]}) ${L.options[it.q.correct]}. ${L.incorrectExplanation}</div>
            </div>`;
        }).join('');

        $('quiz-questions').querySelectorAll('.quiz-options li').forEach(li => {
            const pick = () => answer(parseInt(li.dataset.k, 10), parseInt(li.dataset.pos, 10));
            li.onclick = pick;
            li.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } };
        });
        typeset($('quiz-container'));
    }

    function answer(k, pos) {
        if (!quiz || quiz.answers[k] !== undefined) return;   // locked after the first answer
        quiz.answers[k] = pos;
        const qDiv = $('qq' + k);
        qDiv.querySelectorAll('.quiz-options li').forEach(li => {
            const mine = parseInt(li.dataset.pos, 10) === pos;
            li.classList.add('locked');
            li.classList.toggle('chosen', mine);
            li.setAttribute('aria-checked', mine ? 'true' : 'false');
            li.querySelector('input').checked = mine;
        });
        const ok = pos === quiz.items[k].correctPos;
        $('qq' + k + '-ok').style.display = ok ? 'block' : 'none';
        $('qq' + k + '-ko').style.display = ok ? 'none' : 'block';
    }

    function calculateScore() {
        if (!quiz) return;
        const total = quiz.items.length;
        let score = 0, answered = 0;
        quiz.items.forEach((it, k) => {
            if (quiz.answers[k] !== undefined) {
                answered++;
                if (quiz.answers[k] === it.correctPos) score++;
            }
        });
        const pct = Math.round((score / total) * 100);
        let msg = `${T.score}: ${score}/${total} (${pct}%)`;
        if (answered < total) msg += ` - ${total - answered} ${T.unanswered}`;
        msg += ' - ' + T.verdicts[pct >= 90 ? 3 : pct >= 70 ? 2 : pct >= 50 ? 1 : 0];
        $('quiz-score-display').textContent = msg;
        if (!quiz.sent) { quiz.sent = true; sendResults(quiz.lecture, score, total); }

        const rows = quiz.items.map((it, k) => {
            const L = it.q[LANG];
            const a = quiz.answers[k];
            const cls = a === undefined ? '' : (a === it.correctPos ? 'ok' : 'ko');
            const res = a === undefined ? '-' : (a === it.correctPos ? '<span class="res-ok">OK</span>' : '<span class="res-ko">X</span>');
            const yours = a === undefined ? '-' : `${LETTERS[a]}) ${L.options[it.order[a]]}`;
            return `<tr class="${cls}"><td class="c"><strong>${k + 1}</strong></td><td>${L.text}</td>
                <td>${LETTERS[it.correctPos]}) ${L.options[it.q.correct]}</td><td>${yours}</td><td class="c">${res}</td></tr>`;
        }).join('');
        $('quiz-answer-key').innerHTML = `<div class="answer-key"><h4>${T.detailed}</h4><table>
            <tr><th class="c">${T.colQ}</th><th>${T.colQuestion}</th><th>${T.colCorrect}</th><th>${T.colYours}</th><th class="c">${T.colResult}</th></tr>
            ${rows}</table></div>`;
        typeset($('quiz-answer-key'));
    }

    // ------------------------------------------------------------
    // INIT
    // ------------------------------------------------------------
    // left sidebar with the page sections (wide screens): built from the visible <section id> headings, current one highlighted
    function renderSideToc() {
        const main = document.querySelector('main');
        if (!main || main.querySelector('.side-toc')) return;
        const secs = [...main.querySelectorAll(':scope > section[id]')].filter(s => !s.hidden && s.querySelector('h2'));
        if (secs.length < 3) return;
        const aside = document.createElement('aside');
        aside.className = 'side-toc';
        aside.setAttribute('aria-label', LANG === 'ro' ? 'Secțiunile paginii' : 'Page sections');
        aside.innerHTML = `<p class="side-toc-h">${LANG === 'ro' ? 'Pe această pagină' : 'On this page'}</p><ol>` +
            secs.map(s => `<li><a href="#${s.id}">${esc(s.querySelector('h2').textContent.trim())}</a></li>`).join('') + '</ol>';
        main.prepend(aside);
        main.classList.add('with-side-toc');
        const links = [...aside.querySelectorAll('a')];
        const mark = () => {
            let cur = secs[0];
            for (const s of secs) if (!s.hidden && s.getBoundingClientRect().top < 200) cur = s;
            links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + cur.id));
        };
        window.addEventListener('scroll', mark, { passive: true });
        mark();
    }

    function init() {
        renderStatic();
        renderOverview();
        initLightbox();
        renderLectures();
        initShowcase();
        probeLectures();
        probePages();
        renderCases();
        renderData();
        renderProject();
        renderResources();
        renderContact();
        renderLogin();
        renderQuizTabs();
        const first = D.lectures.find(c => D.quizzes[c.id]) || D.lectures[0];
        showQuiz(first.id);
        renderSideToc();
        typeset();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

// left-hand menu on every page (assets/sidemenu.js)
;(function () { var c = document.currentScript; if (!c || !c.src) return; var e = document.createElement('script'); e.src = new URL('sidemenu.js?v=20261108a', c.src).href; document.head.appendChild(e); })();
