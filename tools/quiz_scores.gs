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

function doPost(e) {
  var out = { ok: false };
  try {
    var p = JSON.parse(e.postData.contents);
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

function doGet() {
  return ContentService.createTextOutput('NN quiz scores endpoint');
}
