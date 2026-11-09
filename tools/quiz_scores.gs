// ============================================================
// NN 2026/2027 - scorurile quiz-urilor (Google Apps Script, Web App)
// Rețele neuronale și deep learning (Business Analytics, FABIZ) - Daniel Traian PELE
//
// Site-ul trimite, după fiecare quiz, tokenul "Sign in with Google" al studentului
// și scorul. Scriptul verifică tokenul la Google (audiență = clientul OAuth,
// e-mail verificat, token neexpirat, domeniu ASE) și scrie un rînd în foaia "Scoruri".
// Numele și e-mailul se iau DOAR din tokenul verificat, nu din ce trimite pagina.
//
// Instalare (o singură dată, din contul danpele@ase.ro):
//   1. https://script.google.com > New project; lipiți acest fișier în Code.gs; salvați
//      cu numele "NN Quiz Scores".
//   2. Selectați funcția setup() și apăsați Run (acceptați permisiunile). În Execution log
//      apare SHEET_URL: adresa foii "NN 2026/2027 - Scoruri quiz".
//   3. Deploy > New deployment > tip "Web app": Execute as "Me", Who has access "Anyone".
//      Copiați URL-ul care se termină în /exec și trimiteți-l: se pune în
//      repo/assets/config.js (QUIZ_SCORES_URL).
// ============================================================

// Același client OAuth ca la MFM (proiectul Google Cloud "MFM Quiz Login"),
// valabil pentru originea https://danpele.github.io
var CLIENT_ID = '1095360272769-rhjjncfor0gumhev6a0l6tnnrnmdrnna.apps.googleusercontent.com';
var ALLOWED_DOMAINS = ['ase.ro', 'stud.ase.ro'];
var HEADER = ['Data', 'Nume', 'E-mail', 'Curs', 'Scor', 'Total', 'Procent', 'Limba'];

function setup() {
  var ss = SpreadsheetApp.create('NN 2026/2027 - Scoruri quiz');
  var sh = ss.getSheets()[0];
  sh.setName('Scoruri');
  sh.getRange(1, 1, 1, HEADER.length).setValues([HEADER]).setFontWeight('bold');
  sh.setFrozenRows(1);
  PropertiesService.getScriptProperties().setProperty('SS_ID', ss.getId());
  Logger.log('SHEET_URL: ' + ss.getUrl());
}

function verify_(idToken) {
  var r = UrlFetchApp.fetch('https://oauth2.googleapis.com/tokeninfo?id_token=' + encodeURIComponent(idToken),
                            { muteHttpExceptions: true });
  if (r.getResponseCode() !== 200) return null;
  var t = JSON.parse(r.getContentText());
  if (t.aud !== CLIENT_ID) return null;
  if (String(t.email_verified) !== 'true') return null;
  if (Number(t.exp) * 1000 < Date.now()) return null;
  var domain = String(t.email).split('@')[1];
  if (ALLOWED_DOMAINS.indexOf(domain) < 0) return null;
  return { name: t.name || t.email, email: t.email };
}

// ------------------------------------------------------------
// Jocurile de la curs (de ex. cursul 1, "Dumneavoastră sau rețeaua?"): rezultate anonime, cu pseudonim, fără logare.
// Foaia "Joc" se creează la primul rezultat. Clasamentul sălii: GET ?board=l1&session=AAAALLZZ (ziua cursului).
// ------------------------------------------------------------
var GAME_HEADER = ['Data', 'Sesiune', 'Joc', 'Pseudonim', 'Corect', 'Rețeaua', 'Total', 'Neplăți oprite', 'Clienți buni respinși', 'Limba'];

function gameSheet_() {
  var ss = SpreadsheetApp.openById(PropertiesService.getScriptProperties().getProperty('SS_ID'));
  var sh = ss.getSheetByName('Joc');
  if (!sh) {
    sh = ss.insertSheet('Joc');
    sh.getRange(1, 1, 1, GAME_HEADER.length).setValues([GAME_HEADER]).setFontWeight('bold');
    sh.setFrozenRows(1);
  }
  return sh;
}

function saveGame_(p) {
  var n = Number(p.total), you = Number(p.you), net = Number(p.net);
  var ok = /^l([1-9]|1[0-4])$/.test(String(p.game)) && /^\d{8}$/.test(String(p.session)) &&
           n > 0 && n <= 50 && you >= 0 && you <= n && net >= 0 && net <= n;
  if (!ok) return { ok: false, error: 'data' };
  var nick = String(p.nick || '').replace(/[\u0000-\u001f<>]/g, '').trim().slice(0, 20) || 'Anonim';
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  gameSheet_().appendRow([new Date(), String(p.session), String(p.game), nick, you, net, n,
    Number(p.caught) || 0, Number(p.lost) || 0, String(p.lang || '')]);
  lock.releaseLock();
  return { ok: true };
}

function board_(game, session) {
  var rows = gameSheet_().getDataRange().getValues().slice(1)
    .filter(function (r) { return String(r[1]) === session && String(r[2]) === game; });
  var top = rows.map(function (r) { return { nick: String(r[3]), you: Number(r[4]), net: Number(r[5]), n: Number(r[6]), t: new Date(r[0]).getTime() }; })
    .sort(function (a, b) { return (b.you / b.n - a.you / a.n) || (a.t - b.t); });
  var mean = function (k) { return top.length ? top.reduce(function (s, r) { return s + r[k] / r.n; }, 0) / top.length : null; };
  return { ok: true, players: top.length, you: mean('you'), net: mean('net'),
           beat: top.filter(function (r) { return r.you > r.net; }).length,
           tie: top.filter(function (r) { return r.you === r.net; }).length,
           scores: top.map(function (r) { return r.you; }), n: top.length ? top[0].n : 10,
           top: top.slice(0, 10).map(function (r) { return { nick: r.nick, you: r.you, net: r.net, n: r.n }; }) };
}

function doPost(e) {
  var out = { ok: false };
  try {
    var p = JSON.parse(e.postData.contents);
    if (p.type === 'game') {
      return ContentService.createTextOutput(JSON.stringify(saveGame_(p))).setMimeType(ContentService.MimeType.JSON);
    }
    var who = verify_(p.credential);
    if (!who) {
      out.error = 'token';
    } else {
      var score = Number(p.score), total = Number(p.total);
      // lecture ids of the NN site: l1 ... l14
      if (!(total > 0 && score >= 0 && score <= total) || !/^l([1-9]|1[0-4])$/.test(String(p.chapter))) {
        out.error = 'data';
      } else {
        var lock = LockService.getScriptLock();
        lock.waitLock(20000);
        var ss = SpreadsheetApp.openById(PropertiesService.getScriptProperties().getProperty('SS_ID'));
        ss.getSheetByName('Scoruri').appendRow([new Date(), who.name, who.email,
          'Cursul ' + String(p.chapter).slice(1), score, total, Math.round(100 * score / total), String(p.lang || '')]);
        lock.releaseLock();
        out = { ok: true, name: who.name };
      }
    }
  } catch (err) {
    out.error = 'server';
  }
  return ContentService.createTextOutput(JSON.stringify(out)).setMimeType(ContentService.MimeType.JSON);
}

function doGet(e) {
  var q = (e && e.parameter) || {};
  if (q.board) {
    var out;
    try {
      out = /^l([1-9]|1[0-4])$/.test(String(q.board)) && /^\d{8}$/.test(String(q.session))
        ? board_(String(q.board), String(q.session)) : { ok: false, error: 'data' };
    } catch (err) {
      out = { ok: false, error: 'server' };
    }
    return ContentService.createTextOutput(JSON.stringify(out)).setMimeType(ContentService.MimeType.JSON);
  }
  return ContentService.createTextOutput('NN quiz scores endpoint');
}
