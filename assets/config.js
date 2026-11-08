// ============================================================
// NN site configuration (shared by index.html and index_ro.html)
// GOOGLE_CLIENT_ID: OAuth client (Web) of the Google Cloud project "MFM Quiz Login" (same client as the MFM and EMQA
//   sites), valid for the origin https://danpele.github.io; the quizzes require "Sign in with Google" with an ASE account.
// QUIZ_SCORES_URL: Google Apps Script web app that verifies the Google token and stores the score in the instructor's
//   sheet. TODO (instructor): deploy the script for this course and paste its /exec URL here; until then the score is
//   shown to the student but not recorded.
// ============================================================
window.NN_CONFIG = {
    GOOGLE_CLIENT_ID: '1095360272769-rhjjncfor0gumhev6a0l6tnnrnmdrnna.apps.googleusercontent.com',
    QUIZ_SCORES_URL: 'YOUR_QUIZ_SCORES_URL'
};
