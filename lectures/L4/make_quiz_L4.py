"""
make_quiz_L4.py -- writes assets/quizzes/l4.js (quiz bank of Lecture 4, EN + RO) with every number from L4_numbers.json.
Format: NN_DATA.quizzes['l4'] = {draw: 20, questions: [{correct: 0-3, en: {...}, ro: {...}}]}; explanations never name a letter.
Run:  python3 lectures/L4/make_quiz_L4.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
N = json.load(open(os.path.join(HERE, "L4_numbers.json")))


def get(key):
    v = N
    for p in key.split("."):
        v = v[int(p)] if isinstance(v, list) else v[p]
    return v


def fmt(v, spec, lang):
    if isinstance(v, str):
        return v
    s = format(v, spec) if spec else str(v)
    if lang == "ro":
        s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s.replace("-", "−") if s.startswith("-") else s


def R(text, lang):
    return re.sub(r"\{([A-Za-z][\w.]*)(?:\|([^{}]*))?\}", lambda m: fmt(get(m.group(1)), m.group(2), lang), text)


Q = []


def q(correct, en, ro):
    """en/ro = (title, text, [4 options], correctExplanation, incorrectExplanation)"""
    Q.append({"correct": correct,
              "en": dict(zip(["title", "text", "options", "correctExplanation", "incorrectExplanation"], en)),
              "ro": dict(zip(["title", "text", "options", "correctExplanation", "incorrectExplanation"], ro))})


q(1, ("Output size", "A 64 × 64 tile passes a 3 × 3 convolution with padding 1 and stride 2. What is the height of the output map?",
      ["64", "{conv.out_s2}", "{conv.out_valid}", "16"],
      "⌊(64 + 2·1 − 3)/2⌋ + 1 = {conv.out_s2}: stride 2 halves the map.",
      "Apply H_out = ⌊(H_in + 2p − k)/s⌋ + 1 with H_in = 64, k = 3, p = 1, s = 2; padding 1 keeps the size only at stride 1."),
  ("Dimensiunea ieșirii", "O imagine de 64 × 64 trece printr-o convoluție 3 × 3 cu completare 1 și pasul 2. Care este înălțimea hărții de ieșire?",
   ["64", "{conv.out_s2}", "{conv.out_valid}", "16"],
   "⌊(64 + 2·1 − 3)/2⌋ + 1 = {conv.out_s2}: pasul 2 înjumătățește harta.",
   "Aplicați H_out = ⌊(H_in + 2p − k)/s⌋ + 1 cu H_in = 64, k = 3, p = 1, s = 2; completarea 1 păstrează dimensiunea doar la pasul 1."))
q(2, ("Valid convolution", "Same 64 × 64 input, 3 × 3 kernel, no padding, stride 1. Output height?",
      ["64", "32", "{conv.out_valid}", "61"],
      "⌊(64 − 3)/1⌋ + 1 = {conv.out_valid}: without padding, each 3 × 3 convolution loses one pixel on each side.",
      "Without padding the kernel cannot be centred on the border pixels, so the map shrinks by k − 1 = 2 pixels."),
  ("Convoluția fără completare", "Aceeași intrare de 64 × 64, nucleu 3 × 3, fără completare, pasul 1. Înălțimea ieșirii?",
   ["64", "32", "{conv.out_valid}", "61"],
   "⌊(64 − 3)/1⌋ + 1 = {conv.out_valid}: fără completare, fiecare convoluție 3 × 3 pierde cîte un pixel pe fiecare margine.",
   "Fără completare, nucleul nu poate fi centrat pe pixelii de margine, deci harta scade cu k − 1 = 2 pixeli."))
q(0, ("Parameters of a layer", "How many parameters has a convolutional layer with 3 × 3 kernels, 32 input channels and 64 output channels (with biases)?",
      ["(9·32 + 1)·64 = 18,496", "9·64 = 576", "32·64 = 2,048", "It depends on the image size"],
      "(k²·C_in + 1)·C_out = (9·32 + 1)·64 = 18,496, whatever the size of the image.",
      "Each of the 64 filters has a 3 × 3 kernel for each of the 32 input channels plus one bias; the count does not depend on the image size, because the weights are shared across positions."),
  ("Parametrii unui strat", "Cîți parametri are un strat convoluțional cu nuclee 3 × 3, 32 de canale de intrare și 64 de canale de ieșire (cu termeni liberi)?",
   ["(9·32 + 1)·64 = 18.496", "9·64 = 576", "32·64 = 2.048", "Depinde de dimensiunea imaginii"],
   "(k²·C_in + 1)·C_out = (9·32 + 1)·64 = 18.496, oricare ar fi dimensiunea imaginii.",
   "Fiecare dintre cele 64 de filtre are cîte un nucleu 3 × 3 pentru fiecare dintre cele 32 de canale de intrare, plus un termen liber; numărul nu depinde de dimensiunea imaginii, pentru că ponderile sînt partajate între poziții."))
q(3, ("Why fewer parameters", "On Fashion-MNIST our CNN has {fmnist.cnn.params|,} parameters and the MLP {fmnist.mlp.params|,}. What is the main reason the CNN needs fewer?",
      ["It uses ReLU instead of sigmoid", "It is trained for fewer epochs", "It has fewer layers than the MLP", "Weight sharing and local connectivity: one small kernel is reused at every position"],
      "A kernel of k²·C_in weights is applied at every position; the MLP needs a separate weight for each pixel and unit.",
      "The activation function and the number of epochs do not change the parameter count, and the CNN is in fact deeper; the saving comes from reusing the same kernel over the whole image."),
  ("De ce mai puțini parametri", "Pe Fashion-MNIST, CNN-ul nostru are {fmnist.cnn.params|,} de parametri, iar MLP {fmnist.mlp.params|,}. Care este motivul principal pentru care CNN are nevoie de mai puțini?",
   ["Folosește ReLU în loc de sigmoidă", "Este antrenată mai puține epoci", "Are mai puține straturi decît MLP", "Partajarea ponderilor și conectivitatea locală: un nucleu mic este refolosit în fiecare poziție"],
   "Un nucleu de k²·C_in ponderi este aplicat în fiecare poziție; MLP are nevoie de o pondere separată pentru fiecare pixel și unitate.",
   "Funcția de activare și numărul de epoci nu schimbă numărul de parametri, iar CNN este de fapt mai adîncă; economia vine din refolosirea aceluiași nucleu pe întreaga imagine."))
q(1, ("Accuracy of the CNN", "What test accuracy did the CNN reach on Fashion-MNIST (mean over three seeds)?",
      ["{fmnist.mlp.acc_mean|.2f}%", "{fmnist.cnn.acc_mean|.2f}%", "{fmnist.softmax.acc_mean|.2f}%", "99.50%"],
      "The CNN reached {fmnist.cnn.acc_mean|.2f}% (standard deviation {fmnist.cnn.acc_sd|.2f}), above the MLP ({fmnist.mlp.acc_mean|.2f}%) and the softmax regression ({fmnist.softmax.acc_mean|.2f}%).",
      "The other values are those of the MLP and of the softmax regression; accuracies above 99% are typical of handwritten digits, not of clothing photos."),
  ("Acuratețea CNN", "Ce acuratețe pe test a atins CNN pe Fashion-MNIST (media pe trei semințe)?",
   ["{fmnist.mlp.acc_mean|.2f}%", "{fmnist.cnn.acc_mean|.2f}%", "{fmnist.softmax.acc_mean|.2f}%", "99,50%"],
   "CNN a atins {fmnist.cnn.acc_mean|.2f}% (abaterea standard {fmnist.cnn.acc_sd|.2f}), peste MLP ({fmnist.mlp.acc_mean|.2f}%) și regresia softmax ({fmnist.softmax.acc_mean|.2f}%).",
   "Celelalte valori sînt cele ale MLP și ale regresiei softmax; acuratețile de peste 99% sînt tipice pentru cifrele scrise de mînă, nu pentru fotografii de îmbrăcăminte."))
q(2, ("McNemar's test", "On the same test images, {fmnist.mcnemar.cnn_only} images are right only for the CNN and {fmnist.mcnemar.mlp_only} only for the MLP. What does McNemar's test use?",
      ["All 10,000 images, right or wrong", "Only the images both models get wrong", "Only the discordant images, b and c: χ² = (|b − c| − 1)²/(b + c)", "The difference of the two accuracies divided by the number of seeds"],
      "Images on which both models agree carry no information about which is better; with b = {fmnist.mcnemar.cnn_only} and c = {fmnist.mcnemar.mlp_only}, χ² = {fmnist.mcnemar.chi2|.1f}.",
      "The test conditions on the discordant pairs; concordant images cancel out, and the seeds measure a different source of noise (training), not the test-set sampling."),
  ("Testul McNemar", "Pe aceleași imagini de test, {fmnist.mcnemar.cnn_only} imagini sînt corecte doar pentru CNN și {fmnist.mcnemar.mlp_only} doar pentru MLP. Ce folosește testul McNemar?",
   ["Toate cele 10.000 de imagini, corecte sau greșite", "Doar imaginile pe care ambele modele le greșesc", "Doar imaginile discordante, b și c: χ² = (|b − c| − 1)²/(b + c)", "Diferența celor două acurateți împărțită la numărul de semințe"],
   "Imaginile pe care modelele sînt de acord nu spun nimic despre care este mai bun; cu b = {fmnist.mcnemar.cnn_only} și c = {fmnist.mcnemar.mlp_only}, χ² = {fmnist.mcnemar.chi2|.1f}.",
   "Testul se bazează pe perechile discordante; imaginile concordante se anulează, iar semințele măsoară altă sursă de zgomot (antrenarea), nu eșantionarea setului de test."))
q(0, ("Max pooling gradient", "During backpropagation, where does the gradient of a 2 × 2 max-pooling output go?",
      ["Only to the input position that was the maximum of the window", "Equally to the four inputs of the window", "To no input: pooling has no gradient", "To the kernel weights of the pooling layer"],
      "Max pooling is the identity on the arg-max element and constant on the others, so only that position receives the gradient.",
      "Average pooling would split the gradient equally; max pooling has no weights, but it still passes the gradient, to the winning position only."),
  ("Gradientul agregării prin maxim", "În propagarea înapoi, unde ajunge gradientul unei ieșiri de agregare prin maxim 2 × 2?",
   ["Numai la poziția de intrare care a fost maximul ferestrei", "În mod egal la cele patru intrări ale ferestrei", "La nicio intrare: agregarea nu are gradient", "La ponderile nucleului stratului de agregare"],
   "Agregarea prin maxim este identitatea pentru elementul maxim și constantă pentru celelalte, deci doar acea poziție primește gradientul.",
   "Agregarea prin medie ar împărți gradientul în mod egal; agregarea prin maxim nu are ponderi, dar transmite gradientul, doar către poziția cîștigătoare."))
q(3, ("Global average pooling", "What does global average pooling (GAP) do at the end of our CNN?",
      ["It doubles the number of channels", "It adds 128 × 10 weights", "It applies a 7 × 7 kernel with learned weights", "It averages each feature map to one number, with no weights"],
      "GAP turns 128 maps of 7 × 7 into 128 numbers; only the final linear layer has weights.",
      "GAP has no learned weights and does not change the number of channels; it is what keeps the head of the network small."),
  ("Agregarea prin medie globală", "Ce face agregarea prin medie globală (GAP) la sfîrșitul CNN-ului nostru?",
   ["Dublează numărul de canale", "Adaugă 128 × 10 ponderi", "Aplică un nucleu 7 × 7 cu ponderi învățate", "Mediază fiecare hartă de trăsături la un singur număr, fără ponderi"],
   "GAP transformă 128 de hărți de 7 × 7 în 128 de numere; doar stratul liniar final are ponderi.",
   "GAP nu are ponderi învățate și nu schimbă numărul de canale; ea menține mică partea finală a rețelei."))
q(1, ("Receptive field", "Twelve stacked 3 × 3 convolutions with stride 1 and no pooling: what receptive field does a unit of the last layer have?",
      ["12 × 12", "{rf.plain_12} × {rf.plain_12}", "{rf.pooled_8conv} × {rf.pooled_8conv}", "The whole image"],
      "r_l = r_(l−1) + (k − 1) with stride 1, so r = 1 + 12·2 = {rf.plain_12}.",
      "Each 3 × 3 layer adds k − 1 = 2 pixels at stride 1; only pooling or strides make the field grow faster ({rf.pooled_8conv} pixels with pooling after every pair of eight layers)."),
  ("Cîmpul receptiv", "Douăsprezece convoluții 3 × 3 suprapuse, cu pasul 1 și fără agregare: ce cîmp receptiv are o unitate din ultimul strat?",
   ["12 × 12", "{rf.plain_12} × {rf.plain_12}", "{rf.pooled_8conv} × {rf.pooled_8conv}", "Întreaga imagine"],
   "r_l = r_(l−1) + (k − 1) la pasul 1, deci r = 1 + 12·2 = {rf.plain_12}.",
   "Fiecare strat 3 × 3 adaugă k − 1 = 2 pixeli la pasul 1; doar agregarea sau pașii mai mari fac cîmpul să crească mai repede ({rf.pooled_8conv} pixeli cu agregare după fiecare pereche, la opt straturi)."))
q(2, ("Two 3 × 3 versus one 5 × 5", "Why did VGG replace a 5 × 5 kernel by two stacked 3 × 3 kernels with C channels?",
      ["Two 3 × 3 kernels see a smaller region", "It removes the need for padding", "Same 5 × 5 receptive field with 18C² instead of 25C² weights, and one more nonlinearity", "It makes the network shallower"],
      "Two 3 × 3 layers cover 5 × 5 pixels and cost 2·9C² = 18C² weights instead of 25C², with a ReLU in between.",
      "The receptive field of two stacked 3 × 3 kernels is exactly 5 × 5; the gain is in parameters and in an extra nonlinearity, while the network gets deeper, not shallower."),
  ("Două nuclee 3 × 3 sau unul 5 × 5", "De ce a înlocuit VGG un nucleu 5 × 5 cu două nuclee 3 × 3 suprapuse, cu C canale?",
   ["Două nuclee 3 × 3 văd o zonă mai mică", "Elimină nevoia de completare", "Același cîmp receptiv de 5 × 5 cu 18C² în loc de 25C² ponderi și o neliniaritate în plus", "Face rețeaua mai puțin adîncă"],
   "Două straturi 3 × 3 acoperă 5 × 5 pixeli și costă 2·9C² = 18C² ponderi în loc de 25C², cu un ReLU între ele.",
   "Cîmpul receptiv al două nuclee 3 × 3 suprapuse este exact 5 × 5; cîștigul este în parametri și într-o neliniaritate suplimentară, iar rețeaua devine mai adîncă, nu mai puțin adîncă."))
q(0, ("Degradation", "A plain 32-layer network ends with a higher training loss than a residual network of the same depth ({cifar.plain32.train_loss|.3f} vs {cifar.res32.train_loss|.3f}). What is this?",
      ["The degradation problem: an optimisation difficulty of deep plain networks", "Overfitting of the plain network", "A sign that the residual network uses a larger learning rate", "Data leakage from the test set"],
      "Overfitting would lower the training loss; a higher training loss at equal depth means the plain network is harder to optimise, which the identity shortcut fixes.",
      "Both networks used the same optimiser, schedule and data; the difference in training loss points to optimisation, not to overfitting or leakage."),
  ("Degradarea", "O rețea simplă de 32 de straturi termină cu o pierdere la antrenare mai mare decît o rețea reziduală de aceeași adîncime ({cifar.plain32.train_loss|.3f} față de {cifar.res32.train_loss|.3f}). Ce este acest fenomen?",
   ["Problema degradării: o dificultate de optimizare a rețelelor adînci simple", "Supraînvățarea rețelei simple", "Un semn că rețeaua reziduală folosește o rată de învățare mai mare", "O scurgere de informație din setul de test"],
   "Supraînvățarea ar scădea pierderea la antrenare; o pierdere la antrenare mai mare la aceeași adîncime înseamnă că rețeaua simplă este mai greu de optimizat, iar scurtătura identitate rezolvă acest lucru.",
   "Ambele rețele au folosit același optimizator, aceeași programare și aceleași date; diferența de pierdere la antrenare indică optimizarea, nu supraînvățarea sau scurgerile de informație."))
q(3, ("Residual gradient", "For y = x + F(x), what is ∂L/∂x?",
      ["∂L/∂y · ∂F/∂x", "∂L/∂y − ∂F/∂x", "Zero, because the shortcut has no weights", "∂L/∂y · (I + ∂F/∂x)"],
      "The identity path adds I to the Jacobian, so the gradient reaches x even when ∂F/∂x is small.",
      "Differentiating x + F(x) gives I + ∂F/∂x; dropping the identity term would remove exactly the direct path that makes deep residual networks trainable."),
  ("Gradientul rezidual", "Pentru y = x + F(x), cît este ∂L/∂x?",
   ["∂L/∂y · ∂F/∂x", "∂L/∂y − ∂F/∂x", "Zero, deoarece scurtătura nu are ponderi", "∂L/∂y · (I + ∂F/∂x)"],
   "Drumul identitate adaugă I la jacobian, deci gradientul ajunge la x chiar cînd ∂F/∂x este mic.",
   "Derivata lui x + F(x) este I + ∂F/∂x; omiterea termenului identitate ar elimina exact drumul direct care face antrenabile rețelele reziduale adînci."))
q(1, ("Augmentation", "On CIFAR-10 the same small CNN reached {cifar.cnn.test_acc|.2f}% without and {cifar.cnn_aug.test_acc|.2f}% with augmentation. What did augmentation change most?",
      ["The number of parameters", "The gap between training and validation accuracy, from {cifar.cnn.gap|.1f} to {cifar.cnn_aug.gap|.1f} points", "The size of the test set", "The learning-rate schedule"],
      "Random crops and flips act as extra data and a regulariser: the network fits the training images less and generalises better.",
      "Augmentation changes only the training inputs; the network, the schedule and the test set were identical in both runs."),
  ("Augmentarea", "Pe CIFAR-10, aceeași CNN mică a atins {cifar.cnn.test_acc|.2f}% fără și {cifar.cnn_aug.test_acc|.2f}% cu augmentare. Ce a schimbat cel mai mult augmentarea?",
   ["Numărul de parametri", "Decalajul dintre acuratețea de antrenare și cea de validare, de la {cifar.cnn.gap|.1f} la {cifar.cnn_aug.gap|.1f} puncte", "Dimensiunea setului de test", "Programarea ratei de învățare"],
   "Decupările și oglindirile aleatoare acționează ca date suplimentare și ca regularizare: rețeaua se potrivește mai puțin imaginilor de antrenare și generalizează mai bine.",
   "Augmentarea schimbă doar intrările de antrenare; rețeaua, programarea și setul de test au fost identice în ambele rulări."))
q(2, ("Augmentation by domain", "Which augmentation is safe for EuroSAT satellite tiles but not for handwritten digits?",
      ["Adding Gaussian noise", "Changing brightness slightly", "Rotations by 90° and vertical flips", "Random crops of a few pixels"],
      "A satellite tile has no ‘up’, so rotations and flips keep the land-use label; a rotated 6 becomes a 9.",
      "Noise, brightness and small crops preserve the label in both domains; the orientation changes are the ones that depend on the domain."),
  ("Augmentarea după domeniu", "Ce augmentare este sigură pentru imaginile satelitare EuroSAT, dar nu pentru cifrele scrise de mînă?",
   ["Adăugarea de zgomot gaussian", "O mică schimbare a luminozității", "Rotațiile cu 90° și oglindirile verticale", "Decupări aleatoare de cîțiva pixeli"],
   "O imagine satelitară nu are ‘sus’, deci rotațiile și oglindirile păstrează clasa de utilizare a terenului; un 6 rotit devine 9.",
   "Zgomotul, luminozitatea și decupările mici păstrează eticheta în ambele domenii; schimbările de orientare sînt cele care depind de domeniu."))
q(0, ("Batch normalisation at test time", "A CNN with batch normalisation scores well in validation but erratically when it scores one image at a time in production. Most likely cause?",
      ["The model was left in training mode, so BN used the statistics of each single-image batch", "The learning rate was too small", "The kernels are too large", "Global average pooling was used"],
      "In eval mode BN uses running averages from training; in training mode it standardises with the current batch, which for one image is meaningless.",
      "Learning rate, kernel size and pooling are fixed at prediction time; only the BN mode changes the computation between validation and production."),
  ("Normalizarea pe loturi la predicție", "O CNN cu normalizare pe loturi merge bine la validare, dar dă rezultate haotice în producție, cînd clasifică cîte o imagine. Care este cauza cea mai probabilă?",
   ["Modelul a rămas în modul de antrenare, deci BN a folosit statisticile fiecărui lot de o singură imagine", "Rata de învățare a fost prea mică", "Nucleele sînt prea mari", "S-a folosit agregarea prin medie globală"],
   "În modul eval, BN folosește mediile mobile din antrenare; în modul de antrenare standardizează cu lotul curent, ceea ce pentru o singură imagine nu are sens.",
   "Rata de învățare, dimensiunea nucleului și agregarea sînt fixe la predicție; doar modul BN schimbă calculul între validare și producție."))
q(3, ("Weakest product class", "Which Fashion-MNIST class had the lowest recall for our CNN?",
      ["Trouser", "Bag", "Sneaker", "{fmnist.worst_class}"],
      "{fmnist.worst_class} has a recall of {fmnist.worst_recall|.1f}%: it is confused with garments of similar shape.",
      "Trousers, bags and sneakers have distinctive shapes and are almost always recognised; the hard classes are upper-body garments."),
  ("Clasa de produse cea mai slabă", "Ce clasă Fashion-MNIST a avut cea mai mică sensibilitate pentru CNN-ul nostru?",
   ["Pantaloni", "Geantă", "Pantof sport", "{fmnist.worst_class_ro}"],
   "{fmnist.worst_class_ro} are o sensibilitate de {fmnist.worst_recall|.1f}%: este confundată cu articole de formă asemănătoare.",
   "Pantalonii, gențile și pantofii sport au forme distincte și sînt recunoscute aproape întotdeauna; clasele dificile sînt articolele pentru partea de sus a corpului."))
q(1, ("Learned filters", "What do most first-layer filters of the trained CNN look like?",
      ["Copies of whole training images", "Oriented edge and contrast detectors", "Random noise with no structure", "Colour detectors for red and green"],
      "Gradient descent finds small oriented differences, similar to Sobel kernels; deeper layers combine them into parts.",
      "A 3 × 3 kernel cannot store a 28 × 28 image, the inputs are grey levels, and the learned kernels show clear structure."),
  ("Filtrele învățate", "Cum arată majoritatea filtrelor din primul strat al CNN antrenate?",
   ["Copii ale unor imagini întregi de antrenare", "Detectori orientați de muchii și contraste", "Zgomot aleator fără structură", "Detectori de culoare pentru roșu și verde"],
   "Coborîrea pe gradient găsește diferențe orientate mici, asemănătoare nucleelor Sobel; straturile mai adînci le combină în părți.",
   "Un nucleu 3 × 3 nu poate memora o imagine de 28 × 28, intrările sînt niveluri de gri, iar nucleele învățate au o structură clară."))
q(2, ("Feature extraction", "In transfer learning by feature extraction on EuroSAT, which parameters are estimated?",
      ["All ResNet-18 weights", "Only the batch-normalisation statistics", "Only the new head (here a logistic regression on 512 features)", "None: the ImageNet classifier is used directly"],
      "The body is frozen; one forward pass gives 512 features per tile and only the head is fitted.",
      "Training all weights is fine-tuning; the ImageNet classifier predicts ImageNet classes, not land use, so a new head is always needed."),
  ("Extragerea trăsăturilor", "În transferul învățării prin extragerea trăsăturilor pe EuroSAT, ce parametri sînt estimați?",
   ["Toate ponderile ResNet-18", "Doar statisticile normalizării pe loturi", "Doar noul strat final (aici, o regresie logistică pe 512 trăsături)", "Niciunul: se folosește direct clasificatorul ImageNet"],
   "Corpul este înghețat; o trecere înainte dă 512 trăsături pentru fiecare imagine și se estimează doar stratul final.",
   "Antrenarea tuturor ponderilor înseamnă ajustare fină; clasificatorul ImageNet prezice clase ImageNet, nu utilizări ale terenului, deci este nevoie mereu de un nou strat final."))
q(0, ("Fine-tuning accuracy", "With all training labels, which EuroSAT model had the highest test accuracy?",
      ["Fine-tuned ResNet-18 ({eurosat.finetune_acc|.2f}%)", "CNN from scratch ({eurosat.scratch_acc|.2f}%)", "Frozen ResNet-18 + logistic ({eurosat.probe_acc|.2f}%)", "All three were equal"],
      "Fine-tuning adapts the ImageNet features to Sentinel-2 textures: {eurosat.finetune_acc|.2f}% against {eurosat.scratch_acc|.2f}% from scratch.",
      "The numbers on the slide put the fine-tuned network first; frozen features cannot adapt to the new domain, and the scratch CNN starts without any prior features."),
  ("Acuratețea ajustării fine", "Cu toate etichetele de antrenare, ce model EuroSAT a avut cea mai mare acuratețe pe test?",
   ["ResNet-18 ajustat fin ({eurosat.finetune_acc|.2f}%)", "CNN de la zero ({eurosat.scratch_acc|.2f}%)", "ResNet-18 înghețat + logistic ({eurosat.probe_acc|.2f}%)", "Toate trei au fost egale"],
   "Ajustarea fină adaptează trăsăturile ImageNet la texturile Sentinel-2: {eurosat.finetune_acc|.2f}%, față de {eurosat.scratch_acc|.2f}% de la zero.",
   "Cifrele de pe slide pun pe primul loc rețeaua ajustată fin; trăsăturile înghețate nu se pot adapta noului domeniu, iar CNN de la zero pornește fără nicio trăsătură învățată anterior."))
q(3, ("Label budget", "With only {budget.n_small} labelled tiles per class, which result did we measure?",
      ["The scratch CNN beat both pre-trained models", "All three models were close to 50%", "The frozen features were useless", "Fine-tuned {budget.finetune.10|.1f}%, frozen {budget.probe.10|.1f}%, scratch {budget.scratch.10|.1f}%"],
      "With few labels the pre-trained features carry most of the knowledge; the scratch CNN has too little data to learn its own.",
      "The chart of accuracy against the label budget shows the pre-trained models far ahead when labels are scarce, and the gap closing as labels grow."),
  ("Bugetul de etichete", "Cu doar {budget.n_small} imagini etichetate pe clasă, ce rezultat am măsurat?",
   ["CNN de la zero le-a depășit pe ambele modele preantrenate", "Toate trei modelele au fost în jur de 50%", "Trăsăturile înghețate au fost inutile", "Ajustat fin {budget.finetune.10|.1f}%, înghețat {budget.probe.10|.1f}%, de la zero {budget.scratch.10|.1f}%"],
   "Cu puține etichete, trăsăturile preantrenate poartă cea mai mare parte a cunoașterii; CNN de la zero are prea puține date pentru a-și învăța propriile trăsături.",
   "Graficul acurateței în funcție de bugetul de etichete arată modelele preantrenate mult înainte cînd etichetele sînt rare și diferența care se închide pe măsură ce etichetele cresc."))
q(1, ("Discriminative learning rates", "Why did we fine-tune the ResNet-18 body with a smaller learning rate (3·10⁻⁴) than the new head (3·10⁻³)?",
      ["The body has fewer parameters", "The body already holds useful features that large steps would destroy; the random head must move fast", "Batch normalisation requires it", "To make training slower on purpose"],
      "Large updates on pre-trained weights erase what was learned on ImageNet; the head starts from random weights and needs larger steps.",
      "The body has far more parameters than the head, and BN works with any learning rate; the choice protects the pre-trained features."),
  ("Rate de învățare diferențiate", "De ce am ajustat corpul ResNet-18 cu o rată de învățare mai mică (3·10⁻⁴) decît noul strat final (3·10⁻³)?",
   ["Corpul are mai puțini parametri", "Corpul conține deja trăsături utile, pe care pașii mari le-ar distruge; stratul final aleator trebuie să se miște repede", "Normalizarea pe loturi o cere", "Pentru a încetini antrenarea în mod intenționat"],
   "Actualizările mari ale ponderilor preantrenate șterg ce s-a învățat pe ImageNet; stratul final pornește de la ponderi aleatoare și are nevoie de pași mai mari.",
   "Corpul are mult mai mulți parametri decît stratul final, iar BN funcționează cu orice rată de învățare; alegerea protejează trăsăturile preantrenate."))
q(2, ("Grad-CAM", "In Grad-CAM, how is the weight α of a feature map computed?",
      ["As the largest activation of the map", "By a separate network trained on labelled heat maps", "As the mean, over positions, of the gradient of the class score with respect to the map", "As the weight of the final linear layer only"],
      "α_o^c = (1/HW) Σ ∂z_c/∂A_o(i,j); the map M^c = ReLU(Σ α_o^c A_o) then shows where the evidence for class c lies.",
      "Grad-CAM needs no extra training and uses gradients, which makes it work for any layer, not only before a single linear layer."),
  ("Grad-CAM", "În Grad-CAM, cum se calculează ponderea α a unei hărți de trăsături?",
   ["Ca cea mai mare activare a hărții", "Cu o rețea separată, antrenată pe hărți termice etichetate", "Ca media, pe poziții, a gradientului scorului clasei în raport cu harta", "Doar ca pondere a stratului liniar final"],
   "α_o^c = (1/HW) Σ ∂z_c/∂A_o(i,j); harta M^c = ReLU(Σ α_o^c A_o) arată apoi unde se află dovezile pentru clasa c.",
   "Grad-CAM nu are nevoie de antrenare suplimentară și folosește gradienți, ceea ce îl face aplicabil oricărui strat, nu doar celui dinaintea unui strat liniar."))
q(0, ("A spurious cue", "The Grad-CAM map of a tile classified as ‘industrial’ lights up a cloud. What do you do?",
      ["Treat it as a possible spurious cue: inspect similar tiles, add cloudy examples, re-test on the target region", "Ignore it: the prediction was correct", "Delete the cloud pixels from the test set", "Increase the learning rate and retrain"],
      "An explanation that points to an irrelevant object is a warning that the model may fail when that object changes; it calls for data and testing, not for ignoring.",
      "Removing test data or changing the optimiser does not address a model that may rely on the wrong evidence; the right response is targeted data and a new test."),
  ("Un indiciu fals", "Harta Grad-CAM a unei imagini clasificate drept ‘zonă industrială’ evidențiază un nor. Ce faceți?",
   ["O tratați ca pe un posibil indiciu fals: verificați imagini asemănătoare, adăugați exemple cu nori și testați din nou pe regiunea țintă", "O ignorați: predicția a fost corectă", "Ștergeți pixelii cu nori din setul de test", "Măriți rata de învățare și reantrenați"],
   "O explicație care indică un obiect irelevant avertizează că modelul poate greși cînd acel obiect se schimbă; cere date și teste, nu ignorare.",
   "Eliminarea datelor de test sau schimbarea optimizatorului nu rezolvă un model care se poate baza pe dovezi greșite; răspunsul corect înseamnă date țintite și un test nou."))
q(3, ("Shares, not tiles", "An insurer needs the share of crops in a region. On the test tiles the true share was {landuse.crop_true|.2f}% and the predicted {landuse.crop_pred|.2f}%. What explains why shares can be accurate even with errors?",
      ["The model never makes errors on crops", "Shares are computed on the training tiles", "The test tiles are all crops", "Errors in opposite directions cancel in the totals; the expected share is Σ_j π_j P(ŷ = k | y = j)"],
      "Tiles of other classes predicted as crops offset crop tiles predicted as other classes; the confusion matrix gives the expected bias of the share.",
      "The model does make errors and the shares are measured on the test tiles; what matters for the share is the balance of errors into and out of the class."),
  ("Ponderi, nu imagini", "Un asigurător are nevoie de ponderea culturilor într-o regiune. Pe imaginile de test, ponderea reală a fost {landuse.crop_true|.2f}%, iar cea prezisă {landuse.crop_pred|.2f}%. Ce explică de ce ponderile pot fi exacte chiar cu erori?",
   ["Modelul nu greșește niciodată la culturi", "Ponderile sînt calculate pe imaginile de antrenare", "Toate imaginile de test sînt culturi", "Erorile în sensuri opuse se compensează în totaluri; ponderea așteptată este Σ_j π_j P(ŷ = k | y = j)"],
   "Imaginile din alte clase prezise drept culturi compensează imaginile cu culturi prezise drept alte clase; matricea de confuzie dă deplasarea așteptată a ponderii.",
   "Modelul greșește, iar ponderile sînt măsurate pe imaginile de test; pentru pondere contează echilibrul dintre erorile care intră în clasă și cele care ies din ea."))
q(1, ("Domain shift", "The EuroSAT model will be used on tiles from another season and another country. What is the first step?",
      ["Deploy it: test accuracy is above 90%", "Label a small sample from the target region and season and re-measure", "Retrain from scratch with a deeper network", "Remove batch normalisation"],
      "Test accuracy holds for the population of the test set; a new region or season can change the error rates, so measure there first.",
      "A high accuracy on one population does not transfer automatically; a deeper network or removing BN does not answer whether the error rates changed."),
  ("Schimbarea domeniului", "Modelul EuroSAT va fi folosit pe imagini din alt anotimp și din altă țară. Care este primul pas?",
   ["Îl puneți în producție: acuratețea pe test depășește 90%", "Etichetați un mic eșantion din regiunea și anotimpul țintă și măsurați din nou", "Îl reantrenați de la zero cu o rețea mai adîncă", "Eliminați normalizarea pe loturi"],
   "Acuratețea pe test este valabilă pentru populația setului de test; o regiune sau un anotimp nou pot schimba ratele de eroare, deci măsurați întîi acolo.",
   "O acuratețe ridicată pe o populație nu se transferă automat; o rețea mai adîncă sau eliminarea BN nu răspund la întrebarea dacă s-au schimbat ratele de eroare."))
q(2, ("Equivariance", "What does translation equivariance of a convolutional layer mean?",
      ["The output is the same wherever the object is", "The layer has the same number of parameters for any image", "Shifting the input shifts the feature map by the same amount", "The layer is invariant to rotations"],
      "f(T_δ X) = T_δ f(X): the same filter fires at the shifted position; invariance comes later, from pooling.",
      "Invariance (same output anywhere) is obtained only after pooling; convolutions are not rotation-invariant, and the parameter count is a separate property."),
  ("Echivarianța", "Ce înseamnă echivarianța la translații a unui strat convoluțional?",
   ["Ieșirea este aceeași oriunde s-ar afla obiectul", "Stratul are același număr de parametri pentru orice imagine", "O deplasare a intrării deplasează harta de trăsături cu aceeași valoare", "Stratul este invariant la rotații"],
   "f(T_δ X) = T_δ f(X): același filtru se activează în poziția deplasată; invarianța apare mai tîrziu, prin agregare.",
   "Invarianța (aceeași ieșire oriunde) se obține abia după agregare; convoluțiile nu sînt invariante la rotații, iar numărul de parametri este o altă proprietate."))
q(0, ("1 × 1 convolution", "What is a 1 × 1 convolution with C_in input and C_out output channels?",
      ["A dense layer applied to the channel vector of every pixel, with (C_in + 1)·C_out parameters", "A layer that does nothing", "A pooling layer", "A convolution that only sees one channel"],
      "With k = 1 the formula reduces to a linear map of the C_in channels at each position: it mixes channels and changes their number.",
      "A 1 × 1 kernel still has weights across all input channels; it changes the channel dimension, which is why ResNet uses it on the shortcut when shapes change."),
  ("Convoluția 1 × 1", "Ce este o convoluție 1 × 1 cu C_in canale de intrare și C_out de ieșire?",
   ["Un strat dens aplicat vectorului de canale al fiecărui pixel, cu (C_in + 1)·C_out parametri", "Un strat care nu face nimic", "Un strat de agregare", "O convoluție care vede un singur canal"],
   "Cu k = 1, formula se reduce la o transformare liniară a celor C_in canale în fiecare poziție: amestecă și schimbă numărul canalelor.",
   "Un nucleu 1 × 1 are ponderi pe toate canalele de intrare; schimbă dimensiunea canalelor, motiv pentru care ResNet îl folosește pe scurtătură cînd formele diferă."))
q(3, ("Kernel gradient", "Why does the gradient of a convolution kernel sum over all positions (i, j)?",
      ["Because of batch normalisation", "Because the loss is a sum over classes", "Because of the padding", "Because the same kernel weight is used at every position, so each use contributes"],
      "∂L/∂K(u,v) = Σ_(i,j) δ(i,j) X(is + u − p, js + v − p): a shared parameter collects the gradients of all its uses.",
      "Padding and the class sum do not create the sum over positions; weight sharing does, exactly as for the time steps of a recurrent network."),
  ("Gradientul nucleului", "De ce adună gradientul unui nucleu de convoluție contribuțiile tuturor pozițiilor (i, j)?",
   ["Din cauza normalizării pe loturi", "Pentru că pierderea este o sumă pe clase", "Din cauza completării marginilor", "Pentru că aceeași pondere a nucleului este folosită în fiecare poziție, deci fiecare utilizare contribuie"],
   "∂L/∂K(u,v) = Σ_(i,j) δ(i,j) X(is + u − p, js + v − p): un parametru partajat adună gradienții tuturor utilizărilor sale.",
   "Completarea și suma pe clase nu creează suma pe poziții; o creează partajarea ponderilor, exact ca la pașii de timp ai unei rețele recurente."))
q(1, ("Test set discipline", "During the Fashion-MNIST case study, when were the 10,000 test images used?",
      ["To choose the number of epochs", "Once, after training and early stopping on the validation images", "At every epoch, to monitor learning", "To fit the pixel mean and standard deviation"],
      "Early stopping used the {fmnist.n_valid|,} validation images, scaling used the training part; the test set gave the final number only.",
      "Using test images for epochs, monitoring or scaling would leak information and make the reported accuracy optimistic."),
  ("Disciplina setului de test", "În studiul de caz Fashion-MNIST, cînd au fost folosite cele 10.000 de imagini de test?",
   ["Pentru alegerea numărului de epoci", "O singură dată, după antrenare și oprirea timpurie pe imaginile de validare", "La fiecare epocă, pentru monitorizarea învățării", "Pentru estimarea mediei și abaterii standard a pixelilor"],
   "Oprirea timpurie a folosit cele {fmnist.n_valid|,} de imagini de validare, scalarea a folosit partea de antrenare; setul de test a dat doar cifra finală.",
   "Folosirea imaginilor de test pentru epoci, monitorizare sau scalare ar produce o scurgere de informație și ar face optimistă acuratețea raportată."))
q(2, ("Group research", "What does the vision–language framework of Jheng, Pele, Tak and Găman (2026) do?",
      ["Trains a CNN from scratch on satellite images", "Detects fraud in card transactions from images of receipts", "Projects market indicators onto a multi-panel image and lets a vision–language model classify market regimes with rationales", "Generates synthetic stock charts with a diffusion model"],
      "The paper reframes market regime classification as visual recognition with a zero-shot vision–language model (GPT-4.1), with rolling-quantile labels and natural-language rationales, tested out of sample on the S&P 500.",
      "The framework works on images of market indicators and uses a pre-trained multimodal model without training a network from scratch; it was awarded the Best Paper Award at ICBE 2026."),
  ("Cercetarea grupului", "Ce face cadrul viziune–limbaj al lui Jheng, Pele, Tak și Găman (2026)?",
   ["Antrenează de la zero o CNN pe imagini satelitare", "Detectează frauda cu carduri din imaginile bonurilor", "Proiectează indicatorii de piață într-o imagine cu mai multe panouri și lasă un model viziune–limbaj să clasifice regimurile pieței, cu justificări", "Generează grafice sintetice de acțiuni cu un model de difuzie"],
   "Lucrarea reformulează clasificarea regimurilor de piață ca recunoaștere vizuală, cu un model viziune–limbaj folosit fără antrenare suplimentară (GPT-4.1), cu etichete din cuantile glisante și justificări în limbaj natural, testat în afara eșantionului pe S&P 500.",
   "Cadrul lucrează pe imagini ale indicatorilor de piață și folosește un model multimodal preantrenat, fără a antrena o rețea de la zero; a primit premiul Best Paper la ICBE 2026."))
q(0, ("Seeds and noise", "Why does the lecture report the CNN accuracy as a mean over three seeds?",
      ["Different random initialisations and batch orders give different networks; the spread measures training noise", "Because the test set changes with the seed", "Because McNemar's test needs three runs", "To make the accuracy higher"],
      "The standard deviation over seeds ({fmnist.cnn.acc_sd|.2f} points for the CNN) shows how much of a difference between models could be training luck.",
      "The test set was fixed; McNemar works on one pair of models; averaging over seeds does not raise the accuracy, it measures its stability."),
  ("Semințe și zgomot", "De ce raportează cursul acuratețea CNN ca medie pe trei semințe?",
   ["Inițializările aleatoare și ordinea loturilor diferite dau rețele diferite; dispersia măsoară zgomotul antrenării", "Pentru că setul de test se schimbă cu sămînța", "Pentru că testul McNemar are nevoie de trei rulări", "Pentru a crește acuratețea"],
   "Abaterea standard pe semințe ({fmnist.cnn.acc_sd|.2f} puncte pentru CNN) arată cît din diferența dintre modele ar putea fi noroc la antrenare.",
   "Setul de test a fost fix; testul McNemar lucrează pe o pereche de modele; media pe semințe nu crește acuratețea, ci îi măsoară stabilitatea."))
q(3, ("Conv1d in finance", "What is a one-dimensional causal convolution on a price series?",
      ["A recurrent network with gates", "A fixed moving average with equal weights", "A Fourier transform of the series", "A learned filter z_t = b + Σ_u K(u) x_(t−u) that uses only past values"],
      "It is a learned bank of moving-average-type filters over time, the time analogue of a 2-D convolution over space.",
      "The weights are learned, not fixed and equal; there is no recurrence or gating, and no transformation to frequencies."),
  ("Conv1d în finanțe", "Ce este o convoluție cauzală unidimensională pe o serie de prețuri?",
   ["O rețea recurentă cu porți", "O medie mobilă fixă cu ponderi egale", "O transformată Fourier a seriei", "Un filtru învățat z_t = b + Σ_u K(u) x_(t−u), care folosește doar valori trecute"],
   "Este un set învățat de filtre de tip medie mobilă în timp, analogul temporal al convoluției bidimensionale în spațiu.",
   "Ponderile sînt învățate, nu fixe și egale; nu există recurență sau porți și nici o transformare în frecvențe."))
q(2, ("Images in the confusion matrix", "In the EuroSAT confusion matrix of the fine-tuned network, what was the largest confusion?",
      ["Sea/lake taken for Forest", "Highway taken for River, always", "{eurosat.top_conf.true} taken for {eurosat.top_conf.pred} ({eurosat.top_conf.n} tiles)", "There were no errors"],
      "The network made {eurosat.n_errors_ft} errors on {eurosat.n_test|,} tiles; the largest cell is {eurosat.top_conf.true} → {eurosat.top_conf.pred}, two classes that look alike from space.",
      "The confusion matrix shows few, concentrated errors between visually similar vegetation and crop classes, not between water and forest."),
  ("Matricea de confuzie", "În matricea de confuzie EuroSAT a rețelei ajustate fin, care a fost cea mai mare confuzie?",
   ["Mare/lac clasificat drept Pădure", "Autostradă clasificată mereu drept Rîu", "{eurosat.top_conf.true_ro} clasificat drept {eurosat.top_conf.pred_ro} ({eurosat.top_conf.n} imagini)", "Nu a existat nicio eroare"],
   "Rețeaua a făcut {eurosat.n_errors_ft} erori la {eurosat.n_test|,} de imagini; cea mai mare celulă este {eurosat.top_conf.true_ro} → {eurosat.top_conf.pred_ro}, două clase care arată asemănător din spațiu.",
   "Matricea de confuzie arată erori puține și concentrate între clase de vegetație și culturi asemănătoare vizual, nu între apă și pădure."))
q(1, ("Cross-correlation", "Deep-learning libraries compute Σ K(u,v) X(i+u, j+v). How does this differ from the mathematical convolution?",
      ["It does not differ at all", "The kernel is not flipped; since K is learned, the flipped kernel would simply be learned instead", "It is slower", "It works only with odd kernel sizes"],
      "This is cross-correlation; a true convolution flips the kernel, but a learned kernel absorbs the flip, so nothing is lost.",
      "The two operations differ only by the flip of the kernel; speed and kernel size are unrelated to this distinction."),
  ("Corelația încrucișată", "Bibliotecile de deep learning calculează Σ K(u,v) X(i+u, j+v). Cum diferă aceasta de convoluția matematică?",
   ["Nu diferă deloc", "Nucleul nu este răsturnat; cum K este învățat, s-ar învăța pur și simplu nucleul răsturnat", "Este mai lentă", "Funcționează numai cu nuclee de dimensiune impară"],
   "Aceasta este corelația încrucișată; o convoluție propriu-zisă răstoarnă nucleul, dar un nucleu învățat absoarbe răsturnarea, deci nu se pierde nimic.",
   "Cele două operații diferă doar prin răsturnarea nucleului; viteza și dimensiunea nucleului nu au legătură cu această distincție."))

if __name__ == "__main__":
    out = []
    for x in Q:
        y_ = {"correct": x["correct"]}
        for lang in ("en", "ro"):
            d = dict(x[lang])
            y_[lang] = {k: ([R(o, lang) for o in v] if isinstance(v, list) else R(v, lang)) for k, v in d.items()}
        out.append(y_)
    assert len(out) >= 30 and all(len(o["en"]["options"]) == 4 and len(o["ro"]["options"]) == 4 for o in out)
    js = ("// ============================================================\n"
          "// Quiz bank for lecture id 'l4': Convolutional neural networks (CNN) and transfer learning (EN + RO)\n"
          "// Generated by lectures/L4/make_quiz_L4.py from L4_numbers.json; correct = index (0-3) in the original order.\n"
          "// incorrectExplanation never names a letter: the engine prepends the correct answer after shuffling.\n"
          "// ============================================================\n"
          "window.NN_DATA.quizzes['l4'] = " + json.dumps({"draw": 20, "questions": out}, ensure_ascii=False, indent=4) + ";\n")
    p = os.path.join(ROOT, "assets", "quizzes", "l4.js")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(js)
    print(f"wrote {p}: {len(out)} questions")
