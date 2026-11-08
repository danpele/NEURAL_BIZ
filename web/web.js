// Web version of an NN lecture (web/LK.html): table of contents state and wide inline formulas.
(function () {
'use strict';
const toc = document.getElementById('wtoc');
if (toc && window.innerWidth < 1000) toc.open = false;
// inline formulas wider than their line get their own horizontal scroll instead of widening the page
function fitMath() {
  document.querySelectorAll('.ctext mjx-container:not([display])').forEach(m => {
    m.classList.remove('wide');
    const box = m.closest('li, p, td, th, .ctext');
    if (box && m.getBoundingClientRect().width > box.clientWidth - 4) m.classList.add('wide');
  });
}
window.addEventListener('nn-mathjax', fitMath);
if (document.body.getAttribute('data-mj') === 'done') fitMath();
let rt = null;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(fitMath, 200); });
// highlight the current section in the table of contents
const links = [...document.querySelectorAll('.wtoc a')].filter(a => (a.getAttribute('href') || '').startsWith('#'));   // in-page anchors only
const heads = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
function mark() {
  let cur = null;
  for (const h of heads) if (h.getBoundingClientRect().top < 140) cur = h;
  links.forEach(a => a.classList.toggle('on', !!cur && a.getAttribute('href') === '#' + cur.id));
}
let st = null;
window.addEventListener('scroll', () => { if (!st) st = requestAnimationFrame(() => { st = null; mark(); }); }, { passive: true });
mark();
})();

// left-hand menu on every page (assets/sidemenu.js)
;(function () { var c = document.currentScript; if (!c || !c.src) return; var e = document.createElement('script'); e.src = new URL('../assets/sidemenu.js?v=20261108a', c.src).href; document.head.appendChild(e); })();
