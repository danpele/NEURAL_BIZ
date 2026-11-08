// Left-hand menu on every page of the course site (home page, web lectures, case studies, interactive pages).
// The page keeps its own menu (aside.side-toc on the home page, aside.wtoc on the web pages, .toc on the interactive pages);
// this script makes sure a menu is always on the left:
//   wide screens (>= 1240 px) without a left column: the links are pinned in a fixed left panel;
//   home page from 900 px: its own left column is shown (it was shown only from 1180 px);
//   smaller screens: a "Menu" button on the left edge opens the same links in a drawer that slides in from the left.
(function () {
  'use strict';
  if (window.__sideMenu) return;
  window.__sideMenu = true;
  const RO = (document.documentElement.lang || '').toLowerCase().startsWith('ro') || /_ro\.html$/.test(location.pathname);
  const LABEL = RO ? 'Cuprins' : 'Contents';
  const CSS = `
@media (min-width: 900px) and (max-width: 1179.98px) {
  main.with-side-toc { max-width: 1440px; display: grid; grid-template-columns: 180px minmax(0, 1fr); column-gap: 1.5rem; }
  main.with-side-toc > section { grid-column: 2; min-width: 0; }
  main.with-side-toc > .side-toc { display: block; grid-column: 1; grid-row: 1 / span 40; align-self: start; position: sticky; top: 90px; max-height: calc(100vh - 110px); overflow-y: auto; }
}
.sm-panel { position: fixed; left: 0; top: 0; bottom: 0; width: 250px; background: #fff; border-right: 1px solid #e3e7ee; box-shadow: 2px 0 14px rgba(15,23,42,.12);
  overflow-y: auto; padding: 1rem 0.9rem 2rem; z-index: 2000; transform: translateX(-105%); transition: transform .22s ease; font: 0.9rem/1.35 system-ui, -apple-system, 'Segoe UI', sans-serif; }
.sm-panel.open { transform: none; }
.sm-panel.pinned { transform: none; box-shadow: none; z-index: 900; }
.sm-panel h4 { margin: 0 0 .6rem; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: #6b7280; display: flex; justify-content: space-between; align-items: center; }
.sm-panel h4 button { border: 0; background: none; font-size: 1.3rem; line-height: 1; cursor: pointer; color: #6b7280; padding: 0 .2rem; }
.sm-panel.pinned h4 button { display: none; }
.sm-panel ol { list-style: none; margin: 0; padding: 0; border-left: 2px solid #e3e7ee; }
.sm-panel li a { display: block; padding: .32rem 0 .32rem .75rem; margin-left: -2px; border-left: 2px solid transparent; color: #1A3A6E; text-decoration: none; }
.sm-panel li.sub a { padding-left: 1.5rem; font-size: .84rem; color: #374151; }
.sm-panel li a:hover { background: #EEF3FA; }
.sm-panel li a.on { border-left-color: #c8102e; font-weight: 600; }
.sm-btn { position: fixed; left: 0; top: 40%; z-index: 1999; writing-mode: vertical-rl; transform: rotate(180deg); background: #1F2A44; color: #fff; border: 0;
  border-radius: 0 6px 6px 0; padding: .8rem .45rem; cursor: pointer; font: 600 .82rem system-ui, -apple-system, sans-serif; letter-spacing: .04em; box-shadow: 0 2px 8px rgba(0,0,0,.2); }
.sm-btn::before { content: '\\2630  '; }
.sm-shade { position: fixed; inset: 0; background: rgba(15,23,42,.35); z-index: 1998; display: none; }
.sm-shade.open { display: block; }
body.sm-pinned { padding-left: 250px; }
@media print { .sm-panel, .sm-btn, .sm-shade { display: none !important; } body.sm-pinned { padding-left: 0; } }`;
  const st = document.createElement('style');
  st.textContent = CSS;
  document.head.appendChild(st);

  // the page's own menu, and whether it is already shown as a left column
  const own = () => document.querySelector('aside.side-toc') || document.querySelector('aside.wtoc') || document.querySelector('.toc');
  function ownIsLeft(m) {
    if (!m) return false;
    const r = m.getBoundingClientRect(), cs = getComputedStyle(m);
    return cs.display !== 'none' && r.width > 60 && r.width < 360 && r.left < 320 && (m.matches('aside') && (cs.position === 'sticky' || cs.position === 'fixed'));
  }
  function links() {
    const m = own();
    let a = [];
    if (m && m.matches('aside')) a = [...m.querySelectorAll('a[href^="#"]')].map(x => [x.getAttribute('href'), x.textContent.trim(), 1]);
    if (!a.length) {   // pages without a side menu (interactive pages): the section and block headings
      a = [...document.querySelectorAll('h2, h3')].map(h => {
        const box = h.id ? h : h.closest('[id]');
        if (!box || box === document.body || h.closest('header, footer, nav, .sm-panel')) return null;
        return ['#' + box.id, h.textContent.trim(), h.tagName === 'H2' ? 1 : 2];
      }).filter(Boolean);
    }
    const seen = new Set();
    return a.filter(([h, t]) => { if (!t || seen.has(h)) return false; seen.add(h); return true; });
  }

  let panel, btn, shade, built = 0;
  function build() {
    const L = links();
    if (!L.length) return false;
    if (panel && built === L.length) return true;
    if (!panel) {
      panel = document.createElement('nav'); panel.className = 'sm-panel'; panel.setAttribute('aria-label', LABEL);
      btn = document.createElement('button'); btn.className = 'sm-btn'; btn.type = 'button'; btn.textContent = LABEL;
      shade = document.createElement('div'); shade.className = 'sm-shade';
      document.body.append(shade, panel, btn);
      btn.addEventListener('click', () => toggle(true));
      shade.addEventListener('click', () => toggle(false));
      document.addEventListener('keydown', e => { if (e.key === 'Escape') toggle(false); });
      panel.addEventListener('click', e => {
        if (e.target.closest('h4 button')) toggle(false);
        else if (e.target.closest('a') && !panel.classList.contains('pinned')) toggle(false);
      });
    }
    panel.innerHTML = `<h4><span>${LABEL}</span><button type="button" aria-label="Close">&times;</button></h4><ol>` +
      L.map(([h, t, lv]) => `<li${lv === 2 ? ' class="sub"' : ''}><a href="${h}">${t.replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' })[c])}</a></li>`).join('') + '</ol>';
    built = L.length;
    return true;
  }
  function toggle(on) {
    if (!panel || panel.classList.contains('pinned')) return;
    panel.classList.toggle('open', on); shade.classList.toggle('open', on);
  }
  function layout() {
    const m = own();
    if (ownIsLeft(m)) {   // the page shows its own left column: nothing to add
      if (panel) { panel.style.display = 'none'; btn.style.display = 'none'; shade.classList.remove('open'); }
      document.body.classList.remove('sm-pinned');
      return;
    }
    if (!build()) return;
    panel.style.display = '';
    const wide = window.innerWidth >= 1240;
    panel.classList.toggle('pinned', wide);
    panel.classList.remove('open'); shade.classList.remove('open');
    document.body.classList.toggle('sm-pinned', wide);
    btn.style.display = wide ? 'none' : '';
  }
  // highlight the section in view
  function spy() {
    if (!panel || panel.style.display === 'none') return;
    const as = [...panel.querySelectorAll('a')];
    let cur = null;
    for (const a of as) { const t = document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1))); if (t && t.getBoundingClientRect().top < 140) cur = a; }
    as.forEach(a => a.classList.toggle('on', a === cur));
  }
  let tm;
  const later = () => { clearTimeout(tm); tm = setTimeout(() => { layout(); spy(); }, 120); };
  window.addEventListener('resize', later);
  window.addEventListener('scroll', () => requestAnimationFrame(spy), { passive: true });
  const start = () => { layout(); spy(); setTimeout(later, 600); setTimeout(later, 2000); };   // the home page builds its menu after loading the data
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', start) : start();
})();
