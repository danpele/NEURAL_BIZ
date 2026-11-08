"""Build the interactive page of Lecture 14 (Multimodal and vision-language models): interactive/L14.html (EN) and
interactive/L14_ro.html (RO).

Inputs (interactive/L14_data.js) come from the lecture's own results: EuroSAT pixels and CLIP image embeddings
(lectures/L14/cache), zero-shot predictions and similarity scores, few-shot accuracies and the S&P 500 regime scores
(lectures/L14/results) and L14_numbers.json. Light extra computations here: CLIP text embeddings of the four prompt templates
(as in analysis_L14.py) for 30 test images, and the attention rollout of CLIP ViT-B/16 for 10 test images.
The page logic is in interactive/L14.js. Run: OMP_NUM_THREADS=1 python3 lectures/L14/build_interactive_L14.py [--no-data]
"""
import base64
import io
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import interactive_common  # noqa: E402
from interactive_common import slider, select, toggles, radios, charts, rnd  # noqa: E402

K = 14
SEED = 20261108
CLS = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
NAME = {'en': {"AnnualCrop": "Annual crop", "Forest": "Forest", "HerbaceousVegetation": "Herbaceous vegetation", "Highway": "Highway",
               "Industrial": "Industrial", "Pasture": "Pasture", "PermanentCrop": "Permanent crop", "Residential": "Residential",
               "River": "River", "SeaLake": "Sea or lake"},
        'ro': {"AnnualCrop": "Culturi anuale", "Forest": "Pădure", "HerbaceousVegetation": "Vegetație ierboasă", "Highway": "Autostradă",
               "Industrial": "Zonă industrială", "Pasture": "Pășune", "PermanentCrop": "Culturi permanente", "Residential": "Zonă rezidențială",
               "River": "Rîu", "SeaLake": "Mare sau lac"}}
PROMPT_NAMES = {"AnnualCrop": "annual crop land", "Forest": "forest", "HerbaceousVegetation": "brushland or shrubland",
                "Highway": "highway or road", "Industrial": "industrial buildings or commercial buildings", "Pasture": "pasture land",
                "PermanentCrop": "permanent crop land", "Residential": "residential buildings or homes or apartments",
                "River": "river", "SeaLake": "lake or sea"}
TEMPL = {"name": ["{}"], "photo": ["a photo of {}."], "sat": ["a satellite photo of {}."],
         "ens": ["a centered satellite photo of {}.", "a centered satellite photo of a {}.", "a centered satellite photo of the {}.",
                 "a satellite photo of {}.", "an aerial photo of {}."]}
CLIPS = {"B32": "openai/clip-vit-base-patch32", "B16": "openai/clip-vit-base-patch16"}

# (anchor, chart file / Quantlet, section key, id prefix)
BLOCKS = [('vit-patches', 'vit_patches_attention', 'vit', 'vp'),
          ('zero-shot', 'zeroshot_examples', 'clip', 'zs'),
          ('confusion', 'zeroshot_confusion', 'clip', 'cm'),
          ('text-search', 'text_search', 'clip', 'ts'),
          ('few-shot', 'fewshot_eurosat', 'clip', 'fs'),
          ('chart-regime', 'chart_regime', 'vlm', 'cr')]

SECTIONS = {
 'en': dict(vit='Vision transformers', clip='CLIP: images and text in one space', vlm='Vision-language models: reading charts'),
 'ro': dict(vit='Transformere pentru imagini', clip='CLIP: imagini și text în același spațiu',
            vlm='Modele vizual-lingvistice: citirea graficelor'),
}

L = {
 'en': dict(h1='Lecture 14 · Multimodal and vision-language models',
   note='The slide of the lecture is on the left, the same figure on the right, recomputed in your browser from the results of the '
        'lecture when you move a control. Under each figure: a readout, a question for the room and the Quantlet with the code of the '
        'chart. Models: CLIP (contrastive language–image pre-training) ViT-B/32 and ViT-B/16, where ViT is a vision transformer; '
        'Qwen2-VL-2B, a vision-language model (VLM); a convolutional neural network (CNN) trained from scratch.',
   src='Images: EuroSAT, Sentinel-2 RGB images of land use in Europe, 64 × 64 pixels, 10 classes (Helber et al., 2019); 21,600 '
       'training and 5,400 test images, split fixed in advance. Market charts: S&P 500 daily closes (Yahoo Finance), windows of 120 '
       'trading days every 5 days, test windows from 2019 to October 2026. Open models from Hugging Face; all scores were computed '
       'in the lecture (lectures/L14/analysis_L14.py), except the CLIP text embeddings of the 30 example images and the attention '
       'rollout of the 10 images of the first block, recomputed with the same models for this page.'),
 'ro': dict(h1='Cursul 14 · Modele multimodale și vizual-lingvistice',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, recalculată în browser din rezultatele cursului cînd modificați '
        'un control. Sub fiecare figură: o sinteză numerică, o întrebare pentru sală și Quantlet-ul cu codul figurii. Modele: CLIP '
        '(preantrenare contrastivă limbaj–imagine) ViT-B/32 și ViT-B/16, unde ViT este un transformer pentru imagini; Qwen2-VL-2B, un '
        'model vizual-lingvistic (VLM); o rețea neuronală convoluțională (CNN) antrenată de la zero.',
   src='Imagini: EuroSAT, imagini RGB Sentinel-2 ale utilizării terenului în Europa, 64 × 64 de pixeli, 10 clase (Helber et al., '
       '2019); 21.600 de imagini de antrenare și 5.400 de test, împărțire fixată dinainte. Grafice de piață: prețurile zilnice de '
       'închidere ale indicelui S&P 500 (Yahoo Finance), ferestre de 120 de zile de tranzacționare la fiecare 5 zile, ferestrele de '
       'test din 2019 pînă în octombrie 2026. Modele deschise de pe Hugging Face; toate scorurile au fost calculate în curs '
       '(lectures/L14/analysis_L14.py), cu excepția embedding-urilor de text CLIP pentru cele 30 de imagini-exemplu și a propagării '
       'atenției pentru cele 10 imagini din primul bloc, recalculate cu aceleași modele pentru această pagină.'),
}


def widgets(lang):
    ro = lang == 'ro'
    nm = NAME[lang]
    img10 = [(i, nm[c]) for i, c in enumerate(CLS)]
    img30 = [(3 * i + j, f'{nm[c]} · {j + 1}') for i, c in enumerate(CLS) for j in range(3)]
    models = [('B32', 'CLIP ViT-B/32'), ('B16', 'CLIP ViT-B/16')]
    tpl = [('name', 'Numele dosarului („AnnualCrop”)' if ro else 'Folder name ("AnnualCrop")'),
           ('photo', '„a photo of {class}.”' if ro else '"a photo of {class}."'),
           ('sat', '„a satellite photo of {class}.”' if ro else '"a satellite photo of {class}."'),
           ('ens', 'Media a 5 șabloane satelitare' if ro else 'Average of 5 satellite templates')]
    return {
     'vit-patches': (select('vp-i', 'Imaginea de test' if ro else 'Test image', img10, 3)
                     + slider('vp-p', 'Mărimea peticului (pixeli din imaginea de 224 × 224)' if ro else 'Patch size (pixels of the 224 × 224 image)', 0, 5, 1, 2)
                     + radios('vp-v', 'Ce vede modelul' if ro else 'What the model sees',
                              [('grid', 'Imaginea cu grila de petice' if ro else 'Image with the patch grid'),
                               ('mean', 'Media de culoare a fiecărui petic' if ro else 'Mean colour of each patch')], 'grid')
                     + slider('vp-a', 'Opacitatea hărții de atenție (%)' if ro else 'Opacity of the attention map (%)', 0, 100, 5, 45),
                     charts('vp-img', 'vp-att')),
     'zero-shot': (select('zs-i', 'Imaginea de test' if ro else 'Test image', img30, 0)
                   + select('zs-m', 'Modelul' if ro else 'Model', models, 'B16')
                   + select('zs-t', 'Promptul (șablonul textului)' if ro else 'Prompt (text template)', tpl, 'ens')
                   + slider('zs-s', 'Scala logiților (inversul temperaturii)' if ro else 'Logit scale (inverse temperature)', 1, 200, 1, 100),
                   charts('zs-img', 'zs-bar')),
     'confusion': (select('cm-m', 'Modelul' if ro else 'Model', models, 'B16')
                   + select('cm-t', 'Promptul (șablonul textului)' if ro else 'Prompt (text template)', tpl, 'ens')
                   + radios('cm-n', 'Celulele arată' if ro else 'Cells show',
                            [('row', 'Procente pe rînd' if ro else 'Row percentages'), ('n', 'Numărul de imagini' if ro else 'Number of images')], 'row'),
                   charts('cm-chart', cls='one')),
     'text-search': (select('ts-c', 'Căutarea (clasa din prompt)' if ro else 'Query (class in the prompt)', img10, 0)
                     + slider('ts-k', 'Numărul de imagini returnate, k' if ro else 'Number of images returned, k', 10, 1000, 10, 100)
                     + toggles('ts-a', 'Comparație' if ro else 'Comparison', [('avg', 'Media celor 10 căutări' if ro else 'Mean of the 10 queries')], ['avg']),
                     charts('ts-chart', cls='one') + '<div class="thumbs" id="ts-thumbs" style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px"></div>'),
     'few-shot': (slider('fs-k', 'Imagini etichetate pe clasă' if ro else 'Labelled images per class', 0, 7, 1, 4)
                  + toggles('fs-s', 'Serii' if ro else 'Series',
                            [('probe', 'CLIP + regresie logistică' if ro else 'CLIP + logistic regression'),
                             ('cnn', 'CNN antrenată de la zero' if ro else 'CNN trained from scratch'),
                             ('zs', 'CLIP zero-shot (fără etichete)' if ro else 'CLIP zero-shot (no labels)')], ['probe', 'cnn', 'zs']),
                  charts('fs-chart', cls='one')),
     'chart-regime': (select('cr-m', 'Modelul' if ro else 'Model',
                             [('numeric', 'Cifre + regresie logistică' if ro else 'Numbers + logistic regression'),
                              ('clip_probe', 'Imagine CLIP + regresie logistică' if ro else 'CLIP image + logistic regression'),
                              ('clip_zs', 'CLIP zero-shot'), ('vlm_stress', 'Qwen2-VL zero-shot')], 'numeric')
                      + slider('cr-a', 'Ferestre semnalate ca stres (% din test)' if ro else 'Windows flagged as stress (% of the test set)', 2, 40, 1, 10),
                      charts('cr-roc', 'cr-time')),
    }


QA = {
 'en': {
  'vit-patches': ('CLIP ViT-B/16 cuts the 224 × 224 image into patches of 16 × 16 pixels. How many tokens does the transformer read, and what happens to the cost of attention if the patches are halved to 8 × 8?',
                  '14 × 14 = 196 patch tokens plus one class token, 197 in all (lecture numbers: 12 layers, 12 heads, width 768). With 8 × 8 patches the grid is 28 × 28 = 784 tokens, four times more, and self-attention compares every token with every other one, so its cost grows about 16 times. Switch to "mean colour of each patch": with large patches a road or a river disappears inside one token, which is why small details need small patches. The attention map shows where the class token looks after all 12 layers (attention rollout).'),
  'zero-shot': ('CLIP was never trained on EuroSAT. How can it classify these images, and what does the logit scale change?',
                'Each class becomes a sentence ("a centered satellite photo of forest."); the image is assigned to the class whose sentence embedding has the largest cosine similarity with the image embedding. The cosines are close to each other (between about 0.2 and 0.3), so the softmax multiplies them by a scale, 100 in CLIP. The scale does not change the predicted class, only how confident the probabilities look: at scale 1 every class gets about 10%, at 200 the top class takes almost everything. On all 5,400 test images the best set-up, ViT-B/16 with 5 templates, is right 56.4% of the time; 5 of the 8 images on the slide are correct.'),
  'confusion': ('Which land uses does zero-shot CLIP confuse, and does a better prompt help?',
                'With ViT-B/16 and 5 templates, 399 of the 600 sea or lake images are called highway, and permanent crops are recognised in only 18.6% of cases, while industrial areas (93.6%) and residential areas (92.3%) are almost always right. The prompt matters a lot: the bare folder name gives 34.1% (ViT-B/32) and 42.4% (ViT-B/16), the average of 5 satellite templates 49.1% and 56.4%. Words that describe the image as a satellite photo move the text embedding closer to this kind of picture.'),
  'text-search': ('The same embeddings can be used the other way round: type a description, retrieve images. Is a search for "annual crop land" as reliable as one for "industrial buildings"?',
                  'No. Among the 100 images most similar to the prompt, 99% are industrial for the industrial query, but only 20% are annual crops for the annual crop query, and 21% for sea or lake; the mean over the 10 queries is 58.9%. Move k: precision usually falls as more images are returned, and it cannot stay at 100% once k exceeds the number of test images of the class (400 to 600, given in the readout). A search engine on a bank\'s satellite archive would need checking class by class.'),
  'few-shot': ('How many labelled images per class do you need before a network trained from scratch beats a logistic regression on CLIP embeddings?',
               'With few labels the pre-trained features win: one image per class gives 59.0% with the CLIP probe, already above zero-shot (56.4%); with 4 per class the probe reaches 73.8% and the CNN 50.3%, with 16 per class 84.7% against 78.5%. At 64 per class the CNN catches up (90.9% against 90.1%) and at 128 it is ahead (94.1% against 91.6%); on all 21,600 training images the CNN reaches 97.4%, the probe 95.5%. Labels are expensive: pre-trained embeddings are the cheap start, a dedicated model pays off once the labels exist.'),
  'chart-regime': ('A model can look at the picture of the last 120 days of the S&P 500 or at four numbers computed from it (returns, volatility, drawdown). Which detects stress better?',
                   'The numbers: area under the ROC (receiver operating characteristic) curve (AUC) 0.933 for the logistic regression on the numbers, 0.907 for CLIP zero-shot, 0.880 for the CLIP image probe and 0.758 for Qwen2-VL zero-shot, on 394 test windows of which 8.4% are stress. Move the share of windows flagged: with 10% of the windows flagged (39 alarms), the numbers catch 21 of the 33 stress windows, CLIP zero-shot 20, the CLIP probe 14 and Qwen2-VL 11; to catch 25 of them the numbers need 59 alarms, Qwen2-VL 191. When the numbers behind a chart exist, give the model the numbers; the image route is for documents where only the picture exists.'),
 },
 'ro': {
  'vit-patches': ('CLIP ViT-B/16 împarte imaginea de 224 × 224 de pixeli în petice de 16 × 16. Cîți tokeni citește transformerul și ce se întîmplă cu costul atenției dacă peticele se înjumătățesc la 8 × 8?',
                  '14 × 14 = 196 de tokeni-petic plus un token de clasă, în total 197 (cifrele din curs: 12 straturi, 12 capete de atenție, lățimea 768). Cu petice de 8 × 8, grila are 28 × 28 = 784 de tokeni, de patru ori mai mulți, iar autoatenția compară fiecare token cu toți ceilalți, deci costul ei crește de circa 16 ori. Alegeți „media de culoare a fiecărui petic”: cu petice mari, un drum sau un rîu dispare într-un singur token; de aceea detaliile mici cer petice mici. Harta de atenție arată unde se uită tokenul de clasă după toate cele 12 straturi (propagarea atenției, attention rollout).'),
  'zero-shot': ('CLIP nu a fost antrenat niciodată pe EuroSAT. Cum poate clasifica aceste imagini și ce schimbă scala logiților?',
                'Fiecare clasă devine o propoziție („a centered satellite photo of forest.”); imaginea este atribuită clasei a cărei propoziție are embedding-ul cu cea mai mare similaritate cosinus față de embedding-ul imaginii. Cosinusurile sînt apropiate între ele (între circa 0,2 și 0,3), de aceea funcția softmax le înmulțește cu o scală, 100 în CLIP. Scala nu schimbă clasa prezisă, ci doar cît de sigure par probabilitățile: la scala 1, fiecare clasă primește circa 10%, la 200 clasa de pe primul loc ia aproape tot. Pe toate cele 5.400 de imagini de test, cea mai bună variantă, ViT-B/16 cu 5 șabloane, are dreptate în 56,4% din cazuri; 5 din cele 8 imagini de pe slide sînt corecte.'),
  'confusion': ('Ce utilizări ale terenului confundă CLIP în regim zero-shot și ajută un prompt mai bun?',
                'Cu ViT-B/16 și 5 șabloane, 399 din cele 600 de imagini cu mare sau lac sînt numite autostradă, iar culturile permanente sînt recunoscute doar în 18,6% din cazuri, în timp ce zonele industriale (93,6%) și cele rezidențiale (92,3%) sînt aproape mereu corecte. Promptul contează mult: simplul nume al dosarului dă 34,1% (ViT-B/32) și 42,4% (ViT-B/16), media a 5 șabloane satelitare 49,1% și 56,4%. Cuvintele care descriu imaginea ca fotografie din satelit apropie embedding-ul textului de acest tip de imagine.'),
  'text-search': ('Aceleași embedding-uri pot fi folosite invers: scrieți o descriere și regăsiți imagini. Este căutarea „annual crop land” la fel de sigură ca „industrial buildings”?',
                  'Nu. Dintre cele 100 de imagini cele mai apropiate de prompt, 99% sînt industriale pentru căutarea zonelor industriale, dar doar 20% sînt culturi anuale pentru căutarea culturilor anuale și 21% pentru mare sau lac; media celor 10 căutări este 58,9%. Modificați k: precizia scade de regulă pe măsură ce sînt returnate mai multe imagini și nu poate rămîne 100% cînd k depășește numărul imaginilor de test din clasă (între 400 și 600, dat în sinteză). Un motor de căutare pe arhiva de imagini satelitare a unei bănci ar trebui verificat clasă cu clasă.'),
  'few-shot': ('De cîte imagini etichetate pe clasă este nevoie pentru ca o rețea antrenată de la zero să întreacă o regresie logistică pe embedding-urile CLIP?',
               'Cu puține etichete cîștigă reprezentările preantrenate: o imagine pe clasă dă 59,0% cu regresia pe CLIP, deja peste zero-shot (56,4%); cu 4 imagini pe clasă, regresia ajunge la 73,8% și CNN la 50,3%, cu 16 pe clasă la 84,7% față de 78,5%. La 64 pe clasă, CNN o ajunge din urmă (90,9% față de 90,1%), iar la 128 o depășește (94,1% față de 91,6%); pe toate cele 21.600 de imagini de antrenare, CNN atinge 97,4%, regresia 95,5%. Etichetele sînt scumpe: embedding-urile preantrenate sînt începutul ieftin, un model dedicat se justifică atunci cînd etichetele există.'),
  'chart-regime': ('Un model poate privi imaginea ultimelor 120 de zile ale indicelui S&P 500 sau patru cifre calculate din ea (randamente, volatilitate, scăderea față de maxim). Care detectează mai bine stresul?',
                   'Cifrele: aria de sub curba ROC (caracteristica de operare a receptorului), AUC, este 0,933 pentru regresia logistică pe cifre, 0,907 pentru CLIP zero-shot, 0,880 pentru regresia pe imaginea CLIP și 0,758 pentru Qwen2-VL zero-shot, pe 394 de ferestre de test, dintre care 8,4% de stres. Modificați ponderea ferestrelor semnalate: cu 10% din ferestre semnalate (39 de alarme), cifrele prind 21 din cele 33 de ferestre de stres, CLIP zero-shot 20, regresia pe imaginea CLIP 14, iar Qwen2-VL 11; pentru a prinde 25 dintre ele, modelul pe cifre are nevoie de 59 de alarme, Qwen2-VL de 191. Cînd cifrele din spatele unui grafic există, dați modelului cifrele; calea imaginii este pentru documentele în care există doar imaginea.'),
 },
}

TEXT_JS = {
 'en': r"""window.LK_TEXT = {
 cls: %(en)s,
 px: 'Pixels (224 × 224 input)', img: p => `Input image cut into ${p} × ${p} patches`, att: o => `Attention of the class token (rollout, ViT-B/16), opacity ${o}%`,
 vp: (n, g, p, rel) => `Patch ${p} × ${p} pixels: grid ${g} × ${g} = ${n} patch tokens + 1 class token = ${n + 1} tokens; attention compares ${window.NNI.nf((n + 1) ** 2, 0)} pairs of tokens, ${rel} times the cost with 16 × 16 patches.`,
 vp2: (c, top, share) => `${c}: the most attended patches (top 10% of the map) take ${share} of the attention; peak at row ${top[0]}, column ${top[1]} of the 14 × 14 grid.`,
 prob: 'Probability (%)', simTitle: (s) => `Softmax of ${s} × cosine similarity`,
 zs: (t, p, pr, ok) => `True class: ${t}. Predicted: ${p} with probability ${pr} (${ok ? 'correct' : 'wrong'}).`,
 zs2: (cmax, cmin, s, h) => `Cosine similarities from ${cmin} to ${cmax}; at scale ${s} the entropy of the probabilities is ${h} bits (maximum 3.32 for 10 equal classes).`,
 pred: 'Predicted class (zero-shot)', truth: 'True class',
 cm: (m, acc, best, bestv, worst, worstv, t, p, n, nt) => `${m}: accuracy ${acc} on 5,400 test images. Best class ${best} (${bestv}), worst ${worst} (${worstv}). Largest confusion: ${n} of ${nt} images of ${t} predicted as ${p}.`,
 k: 'Images returned, k', prec: 'Precision of the top k (%)', avg: 'Mean of the 10 queries', query: q => `Query: "a centered satellite photo of ${q}." (5 templates averaged)`,
 ts: (q, k, p, n, a) => `Query ${q}: ${n} of the top ${k} images belong to the class, precision ${p}; mean over the 10 queries ${a}.`,
 ts2: 'Top 8 images (green border: right class, red: wrong class).',
 shots: 'Labelled images per class (log scale)', acc: 'Test accuracy (%)', probe: 'CLIP + logistic regression (5 seeds)', cnn: 'CNN from scratch (3 seeds)',
 zsl: 'CLIP zero-shot', full: 'CNN on all 21,600 training images', fullp: 'CLIP probe on all 21,600 training images',
 fs: (k, n, p, pl, ph, c) => `${window.NNI.cnt(k, 'image', 'images')} per class (${n} labelled images): CLIP probe ${p} (range over seeds ${pl}–${ph})${c}.`,
 fsc: (c, cl, ch) => `; CNN ${c} (range ${cl}–${ch})`, fsn: '; the CNN was not trained at this size',
 fs2: (z, f, fp) => `Zero-shot (no labels) ${z}; with all training labels: CNN ${f}, CLIP probe ${fp}.`,
 fpr: 'False alarm rate (share of calm windows flagged)', tpr: 'Hit rate (share of stress windows flagged)', chance: 'Random guess',
 score: 'Stress score', stress: 'Stress window (actual)', thr: 'Alarm threshold', date: 'End of the 120-day window',
 cr: (m, auc, a, tp, nt, fp, nf, pr) => `${m}: AUC ${auc}. Flagging ${a} of the test windows catches ${tp} of ${nt} stress windows, with ${fp} false alarms among ${nf} calm windows; precision of the alarms ${pr}.`,
 cr2: (n, s) => `${n} test windows (7 January 2019 – 30 October 2026), of which ${s} are stress (20-day volatility above the rolling 90th percentile of the previous 3 years).`
};
""",
 'ro': r"""window.LK_TEXT = {
 cls: %(ro)s,
 px: 'Pixeli (intrarea de 224 × 224)', img: p => `Imaginea de intrare tăiată în petice de ${p} × ${p}`, att: o => `Atenția tokenului de clasă (propagare, ViT-B/16), opacitate ${o}%`,
 vp: (n, g, p, rel) => `Petic de ${p} × ${p} pixeli: grila ${g} × ${g} = ${window.NNI.cnt(n, 'token-petic', 'tokeni-petic')} + 1 token de clasă = ${window.NNI.cnt(n + 1, 'token', 'tokeni')}; atenția compară ${window.NNI.cnt((n + 1) ** 2, 'pereche', 'perechi')} de tokeni, de ${rel} ori costul cu petice de 16 × 16.`,
 vp2: (c, top, share) => `${c}: peticele cele mai urmărite (primele 10% din hartă) primesc ${share} din atenție; maximul este pe rîndul ${top[0]}, coloana ${top[1]} a grilei de 14 × 14.`,
 prob: 'Probabilitatea (%)', simTitle: (s) => `Softmax de ${s} × similaritatea cosinus`,
 zs: (t, p, pr, ok) => `Clasa reală: ${t}. Prezisă: ${p}, cu probabilitatea ${pr} (${ok ? 'corect' : 'greșit'}).`,
 zs2: (cmax, cmin, s, h) => `Similaritățile cosinus sînt între ${cmin} și ${cmax}; la scala ${s}, entropia probabilităților este ${h} biți (maximul este 3,32 pentru 10 clase egale).`,
 pred: 'Clasa prezisă (zero-shot)', truth: 'Clasa reală',
 cm: (m, acc, best, bestv, worst, worstv, t, p, n, nt) => `${m}: acuratețea ${acc} pe 5.400 de imagini de test. Cea mai bună clasă: ${best} (${bestv}), cea mai slabă: ${worst} (${worstv}). Cea mai mare confuzie: ${n} din ${nt} imagini din clasa ${t} sînt prezise ca ${p}.`,
 k: 'Imagini returnate, k', prec: 'Precizia primelor k (%)', avg: 'Media celor 10 căutări', query: q => `Căutarea: „a centered satellite photo of ${q}.” (media a 5 șabloane)`,
 ts: (q, k, p, n, a) => `Căutarea ${q}: ${n} din primele ${window.NNI.cnt(k, 'imagine', 'imagini')} aparțin clasei, precizia ${p}; media celor 10 căutări ${a}.`,
 ts2: 'Primele 8 imagini (chenar verde: clasa corectă, roșu: altă clasă).',
 shots: 'Imagini etichetate pe clasă (scară logaritmică)', acc: 'Acuratețea pe test (%)', probe: 'CLIP + regresie logistică (5 valori aleatoare)', cnn: 'CNN de la zero (3 valori aleatoare)',
 zsl: 'CLIP zero-shot', full: 'CNN pe toate cele 21.600 de imagini de antrenare', fullp: 'Regresia pe CLIP cu toate cele 21.600 de imagini',
 fs: (k, n, p, pl, ph, c) => `${window.NNI.cnt(k, 'imagine', 'imagini')} pe clasă (${window.NNI.cnt(n, 'imagine etichetată', 'imagini etichetate')}): regresia pe CLIP ${p} (interval după valoarea aleatoare ${pl}–${ph})${c}.`,
 fsc: (c, cl, ch) => `; CNN ${c} (interval ${cl}–${ch})`, fsn: '; CNN nu a fost antrenată la această mărime',
 fs2: (z, f, fp) => `Zero-shot (fără etichete) ${z}; cu toate etichetele de antrenare: CNN ${f}, regresia pe CLIP ${fp}.`,
 fpr: 'Rata alarmelor false (ponderea ferestrelor calme semnalate)', tpr: 'Rata de detectare (ponderea ferestrelor de stres semnalate)', chance: 'Ghicire aleatoare',
 score: 'Scorul de stres', stress: 'Fereastră de stres (reală)', thr: 'Pragul de alarmă', date: 'Sfîrșitul ferestrei de 120 de zile',
 cr: (m, auc, a, tp, nt, fp, nf, pr) => `${m}: AUC ${auc}. Dacă semnalăm ${a} din ferestrele de test, prindem ${tp} din ${window.NNI.cnt(nt, 'fereastră', 'ferestre')} de stres, cu ${window.NNI.cnt(fp, 'alarmă falsă', 'alarme false')} din ${window.NNI.cnt(nf, 'fereastră calmă', 'ferestre calme')}; precizia alarmelor ${pr}.`,
 cr2: (n, s) => `${window.NNI.cnt(n, 'fereastră', 'ferestre')} de test (7 ianuarie 2019 – 30 octombrie 2026), dintre care ${s} de stres (volatilitatea pe 20 de zile peste percentila 90 glisantă a ultimilor 3 ani).`
};
""",
}
TEXT_JS = {k: v.replace('%(' + k + ')s', json.dumps([NAME[k][c] for c in CLS], ensure_ascii=False)) for k, v in TEXT_JS.items()}


# ---------------------------------------------------------------- data (real course data only)
def jpeg(a, q=82):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(a).save(b, format='JPEG', quality=q)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()


def data():
    import torch  # noqa: F401  (before scikit-learn)
    import numpy as np
    import pandas as pd
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(1)
    C, R = os.path.join(HERE, 'cache'), os.path.join(HERE, 'results')
    N = json.load(open(os.path.join(HERE, 'L14_numbers.json'), encoding='utf-8'))
    zs = pd.read_csv(os.path.join(R, 'eurosat_zeroshot.csv'))
    sb = pd.read_csv(os.path.join(R, 'eurosat_scores_best.csv'))
    idx_te = zs['index'].values
    yte = np.array([CLS.index(c) for c in zs['true']])
    PX = np.load(os.path.join(C, 'eurosat_px.npy'), mmap_mode='r')
    EMB = {k: np.load(os.path.join(C, f'eurosat_{k}.npz'))['E'] for k in CLIPS}
    out = {}

    # 30 example test images, 3 per class (seeded)
    rng = np.random.default_rng(SEED)
    ex = [int(p) for c in range(10) for p in rng.choice(np.where(yte == c)[0], 3, replace=False)]
    out['ex'] = {'y': [int(yte[p]) for p in ex], 'jpg': [jpeg(np.asarray(PX[idx_te[p]])) for p in ex], 'sim': {}}
    rollout = []
    for k in CLIPS:
        model = CLIPModel.from_pretrained(CLIPS[k], attn_implementation='eager').eval()
        proc = CLIPProcessor.from_pretrained(CLIPS[k])
        for tk, tl in TEMPL.items():
            W = []
            for c in CLS:
                names = [c] if tk == 'name' else [PROMPT_NAMES[c]]
                with torch.no_grad():
                    t = proc(text=[tpl.format(n) for tpl in tl for n in names], return_tensors='pt', padding=True)
                    e = model.get_text_features(**t)
                    e = e if isinstance(e, torch.Tensor) else e.pooler_output
                    e = torch.nn.functional.normalize(e.float(), dim=-1).numpy().mean(0)
                W.append(e / np.linalg.norm(e))
            S = EMB[k][idx_te[ex]] @ np.array(W).T
            agree = float((S.argmax(1) == np.array([CLS.index(c) for c in zs[f'{k}_{tk}'].values[ex]])).mean())
            print(f'  {k} {tk}: recomputed prediction = lecture prediction for {100 * agree:.0f}% of the 30 images')
            out['ex']['sim'][f'{k}_{tk}'] = rnd(S, 4)
        if k == 'B16':                       # attention rollout (Abnar and Zuidema, 2020), as in analysis_L14.py
            for c in range(10):
                img = Image.fromarray(np.asarray(PX[idx_te[ex[3 * c]]]))
                px = proc(images=[img], return_tensors='pt')['pixel_values']
                with torch.no_grad():
                    o = model.vision_model(pixel_values=px, output_attentions=True)
                Rm = torch.eye(o.attentions[0].shape[-1])
                for A in o.attentions:
                    A = A.mean(1)[0] + torch.eye(A.shape[-1])
                    Rm = (A / A.sum(-1, keepdim=True)) @ Rm
                m = Rm[0, 1:].reshape(14, 14).numpy()
                rollout.append(rnd(m / m.sum(), 5))
        del model
    # raw pixels (64 x 64 x 3, base64) of the first example of each class, for the patch view
    out['vit'] = {'px': [base64.b64encode(np.ascontiguousarray(PX[idx_te[ex[3 * c]]]).tobytes()).decode() for c in range(10)],
                  'att': rollout, **N['vit']}

    # confusion matrices of the 8 set-ups (model x template)
    out['cm'] = {}
    for col in [c for c in zs.columns if c.startswith('B')]:
        pr = np.array([CLS.index(c) for c in zs[col]])
        M = np.zeros((10, 10), int)
        np.add.at(M, (yte, pr), 1)
        out['cm'][col] = M.tolist()

    # text-to-image search with the best set-up (ViT-B/16, 5 templates): labels of the 1,000 most similar test images per query
    S = sb[CLS].values
    assert (S.argmax(1) == np.array([CLS.index(c) for c in zs['B16_ens']])).all()
    out['ts'] = {'lab': [], 'jpg': [], 'n': [int((yte == c).sum()) for c in range(10)]}
    for c in range(10):
        o = np.argsort(-S[:, c])
        out['ts']['lab'].append(''.join(str(int(v)) for v in yte[o[:1000]]))
        out['ts']['jpg'].append([[jpeg(np.asarray(PX[idx_te[p]]), 75), int(yte[p])] for p in o[:8]])

    # few-shot accuracies (per seed)
    fs = pd.read_csv(os.path.join(R, 'fewshot_eurosat.csv'))
    out['fs'] = {m: {str(k): rnd(100 * g['acc'].values, 2) for k, g in fs[fs.method == m].groupby('k')} for m in ('probe', 'cnn')}
    out['fs'].update(zs=N['zs'][N['zs_best']], full=N['cnn_full']['mean'], fullp=N['probe_full'])

    # S&P 500 regime scores on the test windows
    cr = pd.read_csv(os.path.join(R, 'chart_regime.csv'))
    cr = cr[cr.split == 'test']
    out['cr'] = {'end': cr['end'].tolist(), 'y': cr['stress'].astype(int).tolist(),
                 **{m: rnd(cr[m].values, 5) for m in ('numeric', 'clip_probe', 'clip_zs', 'vlm_stress')},
                 'auc': {m: N['regime'][k]['auc'] for m, k in (('numeric', 'numeric'), ('clip_probe', 'clip_probe'),
                                                                ('clip_zs', 'clip_zs'), ('vlm_stress', 'vlm_zs'))}}
    return out


if __name__ == '__main__':
    interactive_common.build(sys.modules[__name__])
