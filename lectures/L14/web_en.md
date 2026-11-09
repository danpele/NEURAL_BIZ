<!--
Web version of Lecture 14 (EN). Built into web/L14.html by lectures/common/build_web_lecture.py (format: see its docstring).
Frame hashes: python3 build_web_lecture.py 14 --frames
-->

## @intro

Business data are not only tables. Banks and insurers look at satellite images of the land they finance, read scanned contracts and annual reports, and analysts read price charts. Multimodal models put images and text into one representation: contrastive language–image pre-training (CLIP) learns from image–caption pairs, and vision-language models (VLM) let a large language model read an image. This last lecture measures what these models can and cannot do on real data, and then draws the lessons of the whole course.

We classify 5,400 test images of the EuroSAT data set (Sentinel-2 satellite images of land use in Europe) with CLIP without any labelled example, compare it with a convolutional neural network (CNN) trained from scratch as the number of labels grows, search images with words, and ask a 2-billion-parameter VLM to read market charts and yield-curve tables. The market-stress experiment follows the multimodal anomaly-detection framework of our group, with open models.

@goals
- **Explain** how a vision transformer turns an image into tokens and where its attention goes.
- **Write** the contrastive loss of CLIP, use it as a zero-shot classifier and compare it with a CNN trained on labelled images.
- **Describe** how a VLM connects an image encoder to a language model, and test what it reads in charts and tables.
- **Synthesise** the course: choose a model for a business problem, from logistic regression to foundation models, with the right evaluation.

# Why multimodal models

## Images, documents and charts as business data {h=e94960b4}

Satellite images serve agricultural credit, insurance and environmental, social and governance (ESG) reporting; documents mix text, tables and layout; charts summarise markets for people. When the numbers behind an image exist, they are usually better inputs; when they do not, a multimodal model is the only way to use the image.

## Business application: land use {h=34251a69}

A bank financing farms checks that the plots are cropland; an insurer compares images before and after a flood or a drought; an asset manager measures forest cover and industrial sites around its assets. Images are free and frequent (Sentinel-2 revisits every few days), labels are expensive: hence zero-shot and few-shot models.

## Real data only {h=eb0d7783}

EuroSAT (Helber et al., 2019) contains 27,000 images of 64 × 64 pixels in 10 land-use classes, split 80/20 into 21,600 training and 5,400 test images. The market experiment uses 945 windows of 120 trading days of the S&P 500 (Yahoo Finance), and the table experiment 60 euro area AAA yield curves of the European Central Bank (ECB).

# Vision transformers

## A vision transformer reads patches {h=a2e53b3c}

A vision transformer (ViT) cuts a 224 × 224 image into 14 × 14 = 196 patches of 16 × 16 pixels, embeds each patch linearly, adds a class token and position embeddings, and applies 12 transformer layers with 12 attention heads (Dosovitskiy et al., 2021):

$$ \mathbf z_0=[\mathbf x_{\text{cls}};\ \mathbf E\mathbf p_1;\ \dots;\ \mathbf E\mathbf p_{196}]+\mathbf E_{\text{pos}} $$

Without convolutions, locality must be learned from data, which is why ViTs need very large pre-training.

## Patches and attention on real images {h=4b82ef03}

@chart vit_patches_attention
Patch grid and class-token attention (attention rollout) of CLIP ViT-B/16 on four EuroSAT test images.

The attention rollout multiplies the head-averaged attention matrices of all layers, with the identity added for the residual connection (Abnar & Zuidema, 2020).

# CLIP: images and text in one space

## CLIP learns by matching images with their captions {h=2578a31c}

$$ s_{ij}=\frac{\mathbf u_i^{\top}\mathbf v_j}{\tau},\qquad \mathcal L=-\frac{1}{2n}\sum_{i=1}^{n}\Big[\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}+\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}\Big] $$

where $\mathbf u_i$ and $\mathbf v_j$ are the normalised image and text embeddings and $\tau$ a learned temperature (Radford et al., 2021). The ViT-B/32 model has 151.3 million parameters and the ViT-B/16 model has 149.6 million.

## Zero-shot classification {h=04f85c95}

Each class is described by sentences such as "a satellite photo of {class}."; their averaged text embedding $\mathbf w_c$ becomes the weight vector of the class, and the image goes to the class with the largest cosine similarity. The best zero-shot model, CLIP ViT-B/16 with an ensemble of five templates, classifies 56.4 % of the test images correctly, against 10 % by chance.

## The words in the prompt change the accuracy {h=7af7001d}

@chart zeroshot_prompts
Zero-shot accuracy for four prompt designs and two CLIP models.

With the bare folder names, ViT-B/16 reaches 42.4 %; "a satellite photo of" lifts it to 55.4 % and the five-template ensemble to 56.4 %. Telling the model what kind of picture it sees matters more than the size of the model.

## Eight random test images {h=47e1b4b5}

@chart zeroshot_examples
Zero-shot predictions on eight random test images.

## Where zero-shot fails {h=f49c7339}

@chart zeroshot_confusion
Confusion matrix of the best zero-shot model, row percentages.

Industrial and residential areas are recognised in more than nine cases out of ten (93.6 % and 92.3 %), but permanent crops only in 18.6 % of cases, and 399 of the 600 sea-or-lake images were called highway. Look-alike classes need examples.

## Searching images with words {h=2cbf6133}

@chart text_search
Precision of the 100 images closest to each class prompt.

The same embeddings, used the other way round, turn a sentence into an image search: on average 58.9 % of the 100 images closest to a class prompt truly belong to the class, from 99.0 % for industrial areas to 20.0 % for annual crops. Searching first and labelling the hits is the cheapest way to build a labelled image set.

## A linear probe on CLIP features {h=6b6e22c5}

Keeping CLIP frozen and training only a logistic regression on its image embeddings already beats zero-shot with one labelled image per class (59.0 %), and reaches 95.5 % with all 21,600 training images.

## Pre-trained features win with few labels {h=aaab70d7}

@chart fewshot_eurosat
Test accuracy against labelled images per class: CLIP probe, CNN from scratch, zero-shot.

With 16 labelled images per class the probe reaches 84.7 % against 78.5 % for a CNN trained from scratch; from about 64 per class the CNN catches up, and with 128 it is ahead (94.1 % against 91.6 %). On all training images the CNN reaches 97.4 %. Foundation models pay off when labels are scarce.

## Size and speed {h=14232a95}

The CNN has 289.2 thousand parameters; CLIP ViT-B/32 embeds an image in 4.23 ms on a laptop and ViT-B/16 in 20.49 ms; the VLM needs about 262 ms per image and question.

# Vision-language models: reading charts and tables

## A VLM plugs an image encoder into a language model {h=618572b4}

$$ \mathbf H_v=\mathbf W_p\,f_{\text{ViT}}(I),\qquad p(y_t\mid y_{<t},I,q)=\mathrm{LLM}_{\boldsymbol\theta}\big([\mathbf H_v;\ \mathbf e(q);\ \mathbf e(y_{<t})]\big) $$

Image patches are projected into the word-embedding space of a large language model (LLM), which reads them as extra tokens (Liu et al., 2023). We use Qwen2-VL-2B-Instruct, with 2.21 billion parameters (Wang et al., 2024).

## Market regimes as a visual recognition task {h=869125eb}

Following Jheng, Pele, Tak and Găman (2026), each window of 120 trading days of the S&P 500 is drawn as a two-panel image (price and drawdown) and labelled "stress" when the 20-day volatility at its end exceeds its rolling 90th percentile of the previous three years. Models are trained on windows up to 2018 and tested on 394 windows from 2019, where the stress share is 8.4 %.

## From the picture or from the numbers? {h=bdf6f3e9}

@chart chart_regime
A calm and a stress window, and the ROC curves of four detectors on the test years.

The area under the receiver operating characteristic curve (AUC) is 0.933 for logistic regression on returns, volatility and drawdown, 0.907 for CLIP zero-shot, 0.880 for logistic regression on CLIP image embeddings and 0.758 for Qwen2-VL zero-shot.

## Discrimination and precision {h=3a953429}

Average precision, which matters for a rare event, is 0.578 for the numbers, 0.605 for CLIP zero-shot, 0.449 for the CLIP probe and 0.302 for the VLM. When the numbers exist, a picture adds a lossy step; images earn their place when numbers are missing.

## What a small VLM can read {h=06a15ea7}

Asked whether the price ended higher or lower over the window, the VLM is right in 57.1 % of 394 charts at the 0.5 threshold, below the 74.9 % of always answering the majority, yet its AUC is 0.974: the ranking is right and the threshold is biased, so recalibration on validation charts fixes most of the problem. Reading the 10-year yield from 60 rendered tables, it gives the exact value in 25.0 % of cases, with a mean absolute error of 0.192 percentage points.

## A document pipeline in a bank {h=d19b3980}

Known templates go to a rule-based parser; optical character recognition and layout analysis handle text and tables; a VLM reads free-form pages and charts; checks on totals, dates and ranges send failures to a person; and an exact-match rate per field is measured on a labelled sample.

# Synthesis of the course

## Methods of the course by data type {h=adf81613}

Tables (lectures 1–3), images (4, 14), sequences (5, 13), text (6, 7, 12), generation (8), graphs (9), decisions (10) and trust (11): each data type has its architecture, and every lecture compared it with a benchmark on real data.

## Choosing a model {h=d9a9e5d1}

Start with the decision and the cost of errors, fit the benchmark first, count the labels (few labels: a foundation model zero-shot or with a probe; many labels: a trained network), match the architecture to the data, and prove the result with a test, several seeds, calibration and an explanation.

## Five evaluation rules {h=78b78844}

No leakage; a benchmark next to every network; seeds and intervals; metrics for rare events; real data and reproducible scripts.

## Foundation models for images, text and time series {h=b5604d73}

The same pattern, pre-train once and use zero-shot or with a small head, worked for text (Lecture 12), time series (Lecture 13) and images (this lecture: 56.4 % zero-shot, 95.5 % with a probe on all labels). The common caveat is that pre-training data may contain the test set.

## Responsible use {h=917ec2d7}

The European Union Artificial Intelligence Act lists credit scoring among high-risk uses; a model card documents data, period, metric, benchmark, seeds and known failure cases; models are re-tested on new data; local open models keep client data in-house.

## @end

Multimodal models label and search images without labels and read charts and tables imperfectly; with numbers available, numbers win. Across the course, the same discipline applies: benchmark first, real data, no leakage, several seeds, a test, an explanation.
