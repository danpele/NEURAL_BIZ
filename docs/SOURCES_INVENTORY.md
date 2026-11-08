# Sources inventory: Neural Networks and Deep Learning (2026/27 rebuild)

- **Purpose:** this file maps the existing NEURAL_BIZ material to the 14 new lectures (L1-L14), records which charts and examples use real or simulated data, and lists reusable assets and gaps. It is internal and is not published on the course site.
- **Date:** 2026-11-08
- **Method:** I grepped every `.tex`, `.py` and `.ipynb` source for data loading (`read_csv`, `fetch_*`, `load_*`, `keras/torchvision.datasets`, `load_dataset`, `yf.download`, URLs) and for generators (`np.random.*`, `make_moons/classification/blobs/circles`, `np.sin`, `torch.randn`, "synthetic/simulated"). For deck figures without a generator script, I read the PDF text layer (titles, legends, axis labels).
- **Real-data rule:** simulated data is allowed only for mechanism illustrations: an activation curve, a loss surface, an architecture schematic, or a toy view of how gradient descent moves. Anything that presents itself as a result, a case study or a dataset view must use real data.

**Path abbreviations**

| Short | Full path (under `Teaching/NEURAL_BIZ - Retele neuronale/`) |
|---|---|
| `A/` | `2026 (Asus)/Deep Learning/` (2026 beamer decks, seminars, projects) |
| `A/C/` | `2026 (Asus)/Deep Learning/Courses/` (decks + figure PDFs at top level; generator scripts in `figs/`) |
| `AS/` | `2026 (Asus)/` (2025 PDFs, Bishop book, tf-keras tutorials, `NEURAL_BIZ/`, `charts/`, `Bishop_DL_Figs/`) |
| `D/` | `2026 (Drive)/` (published PDFs, MNIST notebook, student projects `Proiecte/`) |
| `legacy/` | `repo/legacy/` (old repo: `0. Intro` ... `4. NN Foundations`, `Bishop_Ch_01..20`, `data/`) |
| `QL/` | github.com/QuantLet/NN_DL (31 notebooks, ch01-ch13 + LLM workshop; cloned 2026-11-08) |

**New lecture codes:** L1 overview/linear models/perceptron/MLP · L2 backprop and optimisation · L3 regularisation and generalisation · L4 CNN and transfer learning · L5 RNN/LSTM/GRU · L6 NLP foundations and embeddings · L7 attention and transformers · L8 generative models (AE/VAE/GAN/diffusion) · L9 GNN · L10 RL · L11 XAI · L12 LLMs and prompt engineering · L13 DL for energy and financial time series, time-series foundation models · L14 multimodal/VLM and synthesis

---

## 1. Source mapping

### 1.1 2026 beamer decks: `A/C/` (`preamble_nn.tex`; each deck also has a `_handout` variant)

| Deck | Frames | New lecture(s) | Content | Figures |
|---|---|---|---|---|
| `ch00_CourseOverview.tex` | 63 | L1, L14 | Why DL now, roadmap, business domains, software stack (PyTorch), open-science workflow, grading, 8 "teaser" results, Quantlets | 14 incl. + 16 TikZ |
| `ch01_PerceptronsMLP.tex` | 104 | L1 | Biology to McCulloch-Pitts, perceptron, activations, MLP, XOR, universal approximation, German credit case study, ablations | 29 |
| `ch02_BackpropOptimization.tex` | 85 | L2 | Computational graphs, chain rule, backprop, GD variants, momentum/RMSProp/Adam, LR schedules, S&P 500 VaR case study | 19 |
| `ch03_RegularizationGeneralization.tex` | 70 | L3 | Bias-variance, L1/L2, dropout, batch norm, early stopping, credit-card fraud case study | 19 |
| `ch04_ConvolutionalNetworks.tex` | 86 | L4 | Convolution, pooling, receptive field, LeNet/VGG/ResNet, transfer learning, MNIST case study, candlestick/FINDER slides | 23 |
| `ch05_RecurrentNetworks.tex` | 79 | L5 | RNN, BPTT, vanishing gradient, LSTM/GRU, sequence patterns, crypto crash prediction case study | 18 |
| `ch06_NLPFoundations.tex` | 72 | L6 | Preprocessing, BoW/TF-IDF, Word2Vec/GloVe, financial embeddings, text mining case study | 18 |
| `ch06_AttentionTransformers.tex` | 77 | L7 | Attention, multi-head, positional encoding, transformer block, BERT vs GPT, FinBERT sentiment | 19 |
| `ch07_GenerativeModels.tex` | 75 | L8 | Latent variable models, VAE, GAN/WGAN, synthetic financial returns (text says S&P 500 constituents 2000-2023), TSTR. Diffusion is only mentioned (3 hits) | 17 (placeholders, see sec. 2) |
| `ch08_GraphNeuralNetworks.tex` | 72 | L9 | Graph representations, message passing, GCN/GAT/GraphSAGE, interbank systemic risk (text says "50 largest European banks"). No fraud or transaction-graph case | 6 (placeholders) |
| `ch09_ReinforcementLearning.tex` | 84 | L10 | MDP, policy gradient, PPO, actor-critic, DQN, portfolio optimisation. Hedging appears once; nothing on battery dispatch | 7 (placeholders) + 5 TikZ |
| `ch10_Explainability.tex` | 79 | L11 | SHAP, LIME, integrated gradients/saliency, CAV, stability, model risk (SR 11-7), German credit audit | 0 data figures |
| `ch11_LLMsFinance.tex` | 80 | L12 (L13) | Foundation models, scaling laws, zero-shot forecasting, LLM-VaR, risk report automation | 4 |
| `ch11b_PromptEngineering.tex` | 79 | L12 | Prompt anatomy, zero/few-shot, chain-of-thought, business prompt patterns, risks | 0 data figures + 4 TikZ |
| `ch12_EnergyFinance.tex` | 70 | L13 | Energy market structure, price forecasting, volatility, AI4EFin, Romanian OPCOM day-ahead (text only). No Chronos/TimesFM | 0 data figures |
| `ch13_MultimodalVLMs.tex` | 71 | L14 | Multimodal architectures, CLIP/LLaVA/GPT-4V, chart understanding, FINDER-VLM crash prediction (S&P 500, EURO STOXX 50, BET, text only) | 0 data figures |
| `ch14_Synthesis.tex` | 70 | L14 | Roadmap review, 11 open problems, research projects, IDA ecosystem | 0 |
| `llm_from_scratch.tex` (+ handout) | 70 | L7, L12 | NanoGPT "from scratch" workshop: tokenisation, embeddings, self-attention, training, sampling | 13 (`nn_llm_*`) |

**2025 English decks** (`A/C/chapterN_*.tex`, mostly TikZ, a few Bishop book figures):

| Deck | Frames | New lecture(s) | Content |
|---|---|---|---|
| `chapter1_introduction_foundations.tex` | 105 | L1 | AI/ML/DL landscape, linear algebra/calculus/probability refresher, perceptron, activations (28 TikZ) |
| `chapter2_feedforward_networks.tex` | 104 | L1, L2 | Notation, losses, GD, backprop, optimisers, initialisation, Keras, real-estate case, UAT (14 TikZ) |
| `chapter3_regularization_optimization.tex` | 99 | L3, L11 | Bias-variance, L1/L2, dropout, BN, augmentation, tuning, evaluation, fraud case, SHAP/LIME (40 TikZ) |
| `chapter4_convolutional_neural_networks.tex` | 103 | L4 | Convolution, pooling, architectures, transfer learning, YOLO, quality inspection, ViT (30 TikZ) |
| `chapter5_recurrent_networks_sequences.tex` | 104 | L5 | RNN, vanishing gradient, LSTM, GRU, bidirectional, seq2seq, demand forecasting (24 TikZ) |
| `chapter6_nlp_transformers.tex` | 106 | L6, L7, L12 | NLP pipeline, embeddings, attention, transformer, BERT, LLMs, sentiment, NER (20 TikZ) |
| `chapter7_generative_models_frontiers.tex` | 109 | L8, L9, L14 | AE, VAE, GAN, **diffusion**, recommenders, topic models, GNN intro, responsible AI, synthesis (17 TikZ) |

**Bishop-based decks** (`A/C/Bishop_lectures/bishop_chN_*.tex`, 20 decks; `A/Seminars/Bishop_seminars/bishop_semN_*.tex`, 20 seminar decks). Figures come from `AS/charts/` and `AS/Bishop_DL_Figs/`.

| Bishop ch. | Frames | New lecture(s) | Topic |
|---|---|---|---|
| 1 Deep learning revolution | 85 | L1 | Polynomial fitting, overfitting, history |
| 2 Probabilities | 77 | L1 (background) | Probability, Bayes, information |
| 3 Standard distributions | 74 | background | Gaussian, mixtures, exponential family |
| 4 Regression | 64 | L1 | Linear basis models, regularisation, bias-variance |
| 5 Classification | 70 | L1 | Discriminants, logistic regression, ROC |
| 6 Deep neural networks | 65 | L1 | MLP, UAT, representation learning |
| 7 Gradient descent | 57 | L2 | SGD, momentum, Adam, normalisation |
| 8 Backpropagation | 60 | L2 | Autodiff, Jacobians |
| 9 Regularisation | 65 | L3 | Weight decay, dropout, early stopping, invariances |
| 10 CNNs | 75 | L4 | Convolution, pooling, detection, segmentation |
| 11 Structured distributions | 62 | background (L9 message passing) | Graphical models |
| 12 Transformers | 68 | L7, L12 | Attention, NLP, LLMs, ViT |
| 13 Graph neural networks | 50 | L9 | Message passing, graph tasks |
| 14 Sampling | 53 | L8 (background) | MCMC, Gibbs, Langevin |
| 15 Discrete latent variables | 59 | L8 (background) | K-means, GMM, EM |
| 16 Continuous latent variables | 51 | L8 | PCA, probabilistic PCA, factor analysis |
| 17 GANs | 51 | L8 | GAN, CycleGAN |
| 18 Normalising flows | 50 | L8 | Coupling, autoregressive and continuous flows |
| 19 Autoencoders | 46 | L8 | AE, VAE |
| 20 Diffusion models | 50 | L8 | DDPM, score matching, guidance |

### 1.2 Seminars: `A/Seminars/` (7 decks of 51-57 frames, each with `notebooks/seminarN_exercises.ipynb`)

| Seminar | Frames | New lecture(s) | Content | Data in notebook / deck |
|---|---|---|---|---|
| `seminar1_python_tensorflow` | 51 | L1 | Python refresher, TF/Keras, first model | Synthetic regression (`np.random.uniform`), `make_moons`, `make_blobs` |
| `seminar2_ann_regression_classification` | 53 | L1, L2 | Regression and classification with ANNs | California housing, Iris, `make_classification` "churn" (deck: Telco churn CSV) |
| `seminar3_tuning_fraud_detection` | 53 | L3 | Class imbalance, regularisation, tuning, fraud pipeline | Notebook: `make_moons`, `make_classification` fraud. Deck refers to the Kaggle/ULB credit-card fraud data |
| `seminar4_cnn_image_classification` | 53 | L4 | Image handling, CNN, transfer learning, defect detection | MNIST, CIFAR-10 (deck mentions MVTec AD) |
| `seminar5_rnn_demand_forecasting` | 53 | L5, L13 | Time-series preparation, LSTM, multi-step forecasting | Airline passengers (real) + synthetic store demand (sin + promotions + noise). Deck: `yfinance ^GSPC`, household power |
| `seminar6_nlp_sentiment_ner` | 57 | L6, L7 | Preprocessing, sentiment, NER | IMDB, 20 Newsgroups. Deck also uses CoNLL-2003, Financial PhraseBank, yelp, ag_news |
| `seminar7_transformers_recommender` | 53 | L7, L8 | BERT fine-tuning, recommender, deployment | Synthetic transactions, synthetic ratings, hand-written sentiment texts (x15), GAN on MNIST. Deck: MovieLens 100k, Book-Crossing |

### 1.3 Lecture notebooks and local quantlets: `A/C/notebooks/`, `A/C/quantlets/`

| Notebook | New lecture | Content | Data |
|---|---|---|---|
| `ch00_CourseOverview.ipynb` / `quantlets/NN_DL_ch00_PytorchBasics.py` | L1 | Tensors, autograd, linear fit | Synthetic `y = 3x + 2 + noise` |
| `quantlets/NN_DL_ch00_DLTimeline.py` | L1 | DL timeline diagram | None (dates) |
| `ch01_PerceptronsMLP.ipynb` | L1 | Perceptron, MLP, UAT, ablation | Gaussian blobs, `sin*exp` target, random "credit" features |
| `quantlets/NN_DL_ch01_CreditMLP.py` | L1 | Credit MLP | **Synthetic** 1000x20 random features (unlike the QL version, which uses credit-g) |
| `quantlets/NN_DL_ch01_Activations.py`, `NN_DL_ch01_XORViz.py` | L1 | Activations, XOR | Analytic grids |
| `chapter1_lecture.ipynb` | L1 | Math refresher plots | Analytic (`linspace`) |
| `chapter2_lecture.ipynb` | L1, L2 | Losses, GD surfaces, MLP | Analytic + `make_classification` |
| `chapter3_lecture.ipynb` | L3 | Overfitting, bootstrap, regularisation | `sin` + noise, `make_moons`, `make_classification` |
| `chapter4_lecture.ipynb` | L4 | Convolution/pooling demos, CNN | CIFAR-10 + synthetic pattern image + random 8x8 inputs |
| `chapter5_lecture.ipynb` | L5 | NumPy RNN, vanishing gradients, LSTM | Random sequences, `sin(t)+0.3 sin(3t)` signal |
| `chapter6_lecture.ipynb` | L6 | Embeddings, analogies | gensim pretrained if available, else **synthetic embeddings** |
| `chapter7_lecture.ipynb` | L8 | VAE, GAN | MNIST + 1-D Gaussian mixture GAN |

### 1.4 QuantLet NN_DL: `QL/` (charts used in the 2026 decks)

| Quantlet | Lecture | Status | Data |
|---|---|---|---|
| `NN_DL_ch01_Activations` | L1 | Complete | Analytic curves |
| `NN_DL_ch01_CreditMLP` | L1, L11 | Complete, executed | **German credit (OpenML `credit-g`, n=1000)** |
| `NN_DL_ch01_XORViz` | L1 | Complete | XOR grid |
| `NN_DL_ch02_BackpropViz` | L2 | Complete | Random weights (gradient-flow demo) |
| `NN_DL_ch02_VaRForecasting` | L2 | Complete | **S&P 500 from stooq.com / GitHub `datasets/s-and-p-500`**. Falls back to a GARCH(1,1)-t simulation if the download fails |
| `NN_DL_ch03_FraudDetection` | L3 | Complete | **ULB credit-card fraud (OpenML `creditcard`)**. Falls back to `make_classification` |
| `NN_DL_ch03_RegularizationViz` | L3 | Complete | `sin` + noise samples, analytic L1/L2 contours |
| `NN_DL_ch04_ConvViz`, `NN_DL_ch04_MNIST` | L4 | Complete | **MNIST (torchvision)** |
| `NN_DL_ch04_FINDER` | L4 | **Stub** (imports only) | none |
| `NN_DL_ch05_CryptoPrediction` | L5 | Complete | **BTC (GitHub `datasets/bitcoin`)**. Falls back to synthetic returns with injected crashes; the deck figure shows this fallback |
| `NN_DL_ch05_LSTMViz` | L5 | Complete | Random input (gradient demo) |
| `NN_DL_ch06_AttentionViz` | L7 | Complete | Random Q/K/V, analytic positional encoding |
| `NN_DL_ch06_EmbeddingViz` | L6 | Complete | **Random Gaussian clusters** labelled as financial word groups |
| `NN_DL_ch06_FinBERT` | L7 | Complete | ProsusAI FinBERT on ~20 **hand-written** headlines |
| `NN_DL_ch06_TextMining` | L6 | Complete | 20 **hand-written** labelled headlines, duplicated "with variations"; random word vectors |
| `NN_DL_ch07_GANViz`, `_SyntheticData` | L8 | **Stubs** | none |
| `NN_DL_ch08_GNNViz`, `_SystemicRisk` | L9 | **Stubs** ("TODO: build interbank network") | none |
| `NN_DL_ch09_Portfolio`, `_RLViz` | L10 | **Stubs** | none |
| `NN_DL_ch10_ModelRisk`, `_SHAPViz` | L11 | **Stubs** | none |
| `NN_DL_ch11_LLMVaR`, `_PromptEngineering` | L12 | **Stubs** | none |
| `NN_DL_ch12_EnergyForecast`, `_VolatilityViz` | L13 | **Stubs** | none |
| `NN_DL_ch13_FINDERVLM`, `_MultimodalViz` | L14 | **Stubs** | none |
| `NN_DL_LLM_Workshop/NN_DL_LLM_NanoGPT` | L7, L12 | Complete | **Tiny Shakespeare (karpathy/char-rnn)** |

That is 31 notebooks, 15 of them 15-17-line stubs. Nothing from ch07 onward is implemented.

### 1.5 Student and instructor projects

| Item | Lecture | Content | Data (real unless marked) |
|---|---|---|---|
| `A/Projects/ANN_RealEstate.zip` | L1, L2 | ANN house-price regression | King County/Seattle-area house sales 2014 (`data.csv`, 4,600 rows) |
| `A/Projects/CNN.zip` (`CNN_Zipper.ipynb`) | L4 | Zipper defect classification, CLAHE | MVTec-AD-style zipper images (2,220 files) |
| `A/Projects/RNN.zip` | L5 | RNN demand forecast | `demand.csv`, 100 periods, provenance unknown (likely toy) |
| `A/Projects/NLP_NER.zip` | L6, L7 | NER for shipment IDs | **Template-generated** sentences (37 train items): synthetic |
| `A/Projects/NLP_RS.zip` | L6 | Embedding-based movie recommender | TMDB `Movies.csv` |
| `A/Projects/NLP_Sentiment.zip`, `NLP_SentimentANN.ipynb` | L6 | Sentence-embedding sentiment | Tweets `SentimentData.csv` |
| `A/Projects/NLP_TopicModeling.zip` | L6 | Top2Vec | Tweets 2021 (`Tweets.csv`) |
| `D/Proiecte/` Boboc-Darvas et al. | L1 | Movie success prediction | TMDB movies |
| `D/Proiecte/` Alexe et al. | L1 | MLP on cinema people | Kaggle "9000 popular people in cinema" |
| `D/Proiecte/` Anton et al. | L6, L7 | Fake-news detection | FakeNewsNet (BuzzFeed, PolitiFact) |
| `D/Proiecte/` Ostafe et al. (zip) | L1, L3 | London house prices ML | Kaggle `jakewright/house-price-data` |
| `D/Proiecte/` Tanase et al. | L1, L3 | Bank customer churn | Kaggle Bank Customer Churn (10k) |
| `D/Proiecte/` Teusdea, Uta | L1 | Single-layer perceptron | ISLR Advertising |
| `D/Proiecte/` Zamfirescu et al. | L1 | F1 podium prediction | Compiled F1 results (xlsx) |
| `D/Proiecte/` Guran et al. | L1 | Flight price prediction | Kaggle flight price (India) |
| `D/Proiecte/` Dutu, Ionescu, Mirea | L5, L13 | LSTM energy demand | UCI Household Electric Power Consumption 2006-2010 |
| `D/Proiecte/` Popescu, Chirhoc, Simion, Pene | L3 | Suspicious-transaction ANN | **Rule-generated synthetic** `transactions.csv` (10k) |
| `D/Proiecte/` Calinescu et al. | L6, L7 | Stress detection | Dreaddit (Reddit) |
| `AS/Exam/` | L14 | 7 student project folders (June 2025) | Not inventoried in detail |
| `AS/20250411 Fraud-Detection-Demo/` (= `D/`), QuantLet by S.-L. Jheng | L3 | MLP/RF/XGB fraud, 13 PNG charts | **"Synthetic Data Sampling.csv"** (PaySim-style simulated mobile-money data) |

### 1.6 Other PDFs, books and tutorials

| Item | Lecture | Content |
|---|---|---|
| `AS/Deep Learning Bishop 2024.pdf` (656 pp.), Springer copy, `Concepts_in_Deep_Learning_Solutions_v1.0.pdf` (110 pp., ch. 2-10) | All (core text) | Bishop and Bishop (2024), *Deep Learning: Foundations and Concepts*, with exercise solutions |
| `AS/2025*.pdf` (= `D/2025*`): Intro 61 pp., Linear Regression 121, Linear Methods 68, NN foundations 115, NLP foundations 24, Word Embeddings 77, RNN part 1 38, RNN part 2 73, LSTM 72 | L1, L5, L6 | 2025 lecture PDFs (no source `.tex` here) |
| `D/2026*.pdf`, `D/ch00..ch14*.pdf` | as in 1.1 | Published PDFs of the 2026 decks |
| `A/Slides/` CNN01, CNN02, NLP01-04, RNN01 (16-41 pp.) | L4-L7, L12 | Third-party slides (K. Bruessau, Univ. Hamburg). Reference only |
| `D/When Doctors Meet AI_20260324_eng.pdf` (50 pp.) | L14 | Guest talk (medical AI) |
| `AS/tf-keras-tutorial-*.ipynb` (4) | L4, L6 | Kaggle tf-keras series: CNN on MNIST (part 2), binary image classification (part 3), InceptionV3 transfer (part 4), Bi-LSTM/GloVe/GRU on the "NLP getting started" disaster-tweets data (part 6) |
| `D/MNIST.ipynb` | L4 | MNIST via `fetch_openml('mnist_784')` |
| `2026 (iCloud)/Introduction to Neural Networks.pdf` | L1 | Single PDF (not inspected) |

### 1.7 Chapter notebooks: `AS/NEURAL_BIZ/Ch_01..07` (one notebook per folder)

| Notebook | Lecture | Data |
|---|---|---|
| `Ch_01/NB_ch1_tensorflow_basics` | L1 | Synthetic (`np.random.normal/uniform`) |
| `Ch_02/NB_ch2_ann_real_estate` | L1, L2 | California housing |
| `Ch_02/NB_ch2_ann_classification` | L1 | Breast cancer (sklearn) |
| `Ch_03/NB_ch3_fraud_detection` | L3 | `make_classification` (synthetic) |
| `Ch_03/NB_ch3_regularization_demo` | L3 | `make_moons` |
| `Ch_04/NB_ch4_cnn_cifar10`, `NB_ch4_transfer_learning` | L4 | CIFAR-10 |
| `Ch_04/NB_ch4_cnn_quality_inspection` | L4 | Synthetic random images |
| `Ch_05/NB_ch5_rnn_demand_forecast` | L5 | Synthetic `sin` + noise demand |
| `Ch_06/NB_ch6_sentiment_analysis` | L6 | IMDB |
| `Ch_06/NB_ch6_topic_modeling` | L6 | 20 Newsgroups |
| `Ch_06/NB_ch6_ner_business` | L6 | spaCy on hand-written example sentences |
| `Ch_07/NB_ch7_bert_finetuning` | L7 | GLUE SST-2 |
| `Ch_07/NB_ch7_recommender_system` | L8 | Synthetic ratings |
| `Ch_07/NB_ch7_vae_anomaly` | L8 | MNIST |

`AS/NEURAL_BIZ/Bishop_Ch_01..20` is a near-duplicate of `legacy/Bishop_Ch_*`, adding `executed_*.ipynb` and the figure scripts behind `AS/charts/`.

### 1.8 Legacy repo: `legacy/`

| Folder | Lecture | Content | Data |
|---|---|---|---|
| `legacy/0. Intro` | L1 (prerequisites) | Python primer, pandas/EDA, Python ML | HMEQ (`data/hmeq.csv`); primer uses random arrays |
| `legacy/1. Simple_linear_regression` | L1 | Simple linear regression, R2 explanation PDFs | Diamonds (`datasets/diamonds.csv`) |
| `legacy/2. Linear regression practice` | L1 | Multiple regression | Sacramento real-estate transactions |
| `legacy/3. Logistic regression` | L1 | Logistic regression (5 notebooks) | UCI Bank Marketing, Titanic, ISLR Advertising, Coursera ex2data1/2 (exam scores and microchip tests; provenance unclear) |
| `legacy/4. NN Foundations` | L1, L2, L5 | Linear regression to DL, logistic regression to DL, Beijing PM2.5 forecasting | California housing, ULB credit-card fraud (TF mirror), UCI Beijing PM2.5 (id 381); the logistic notebook also uses `make_moons` and random toy sets |
| `legacy/data/` | L1, L3, L11 | Shared datasets | `hmeq.csv` (+ prepared/modeling), `loan_data.csv` (UK application credit data, 1,225 rows, Thomas et al.), wine quality red/white |
| `legacy/Bishop_Ch_01..20` | see 1.1 Bishop table | 3-6 notebooks per chapter: `*_figures`, core topic, `*_finance` | See sec. 2.4. 17 of the `*_finance`/`*_business` notebooks are 17-line stubs (ch04-ch11, ch13-ch20) |

---

## 2. Charts and examples built on synthetic / simulated data

**Keys:** **R** = real data · **S-OK** = synthetic, acceptable as a mechanism illustration · **S-REPLACE** = synthetic or fabricated, must be replaced with real data · **PH** = generic placeholder figure (titles such as "Ch07 ...", "Method A-D", "Asset 1-4"), must be rebuilt. "Fabricated" means the curve or number was typed in or shaped by a formula (for example, ROC curves from beta draws, loss curves from `exp` + noise, hard-coded confusion matrices) and is presented as an empirical result.

### 2.1 2026 deck figures: `A/C/nn_ch*.pdf` (generators in `A/C/figs/*.py`)

| Figure(s) | Generator | Data | Class | Real replacement |
|---|---|---|---|---|
| `nn_ch00_teaser_{mlp,lstm,cnn,transformer,gan,gnn,rl,llmvar}` | `nn_ch00_teaser_figures.py` | Beta-draw ROC curves, `sin` volatility, `sin` prices, t-distributed "real vs GAN" returns, random bank graph, random-walk market | S-REPLACE (all 8) | Regenerate from the real L-cases: German credit ROC, S&P 500/BET realised volatility, ULB fraud, Financial PhraseBank, S&P 500 returns vs GAN, Elliptic graph, ETF portfolio backtest, S&P 500 VaR |
| `nn_ch00_course_roadmap`, `nn_ch00_dl_timeline`, `nn_ch00_pytorch_basics` | `nn_ch00_course_roadmap.py`, `quantlets/NN_DL_ch00_*` | Schematic / dates / toy linear fit | S-OK (pytorch_basics: replace with one real feature, e.g. California housing `MedInc` vs price) | n/a |
| `nn_ch01_activations*`, `sigmoid`, `step_function`, `relu_variants`, `biological_neuron`, `mcculloch_pitts`, `perceptron_diagram`, `mlp_architecture`, `credit_mlp_arch` | `nn_ch01_activation_figures.py`, `neurons_and_activations.py`, `mlp*.py` | Analytic / schematic | S-OK | n/a |
| `nn_ch01_xor_problem`, `xor_mlp_solution`, `hidden_space(_xor)`, `uat_bumps`, `uat_approximation`, `cross_entropy_surface`, `gradient_descent_1d`, `learning_rate_effect` | `perceptron*.py`, `uat*.py`, `mlp_figures.py` | XOR truth table, `sin(x)` target, analytic losses | S-OK (mechanism) | n/a |
| `nn_ch01_perceptron_boundary`, `perceptron_decision_boundary`, `perceptron_convergence` | `nn_ch01_perceptron*.py` | Gaussian blobs (`np.random.randn`) | S-REPLACE | Two standardised features from German credit or HMEQ (e.g. `DEBTINC`, `CLAGE`), or UCI banknote authentication |
| `nn_ch01_depth_vs_width`, `logistic_limitation`, hook figure (moons) | `mlp.py`, `hook_figures.py` | Two-moons / spiral built from `sin`/`cos` + noise | S-REPLACE | 2-D real subset (HMEQ two ratios, or a 2-component PCA of German credit) |
| `nn_ch01_overfitting`, `overfit_analysis` | `mlp.py`, `ablation.py` | Fabricated `exp` + noise loss curves | S-REPLACE | Real train/validation curves of the German credit MLP |
| `nn_ch01_german_credit_eda` | `credit_figures.py` | `np.random.normal/exponential` ages and amounts, "synthetic" correlation heatmap, presented as German credit | S-REPLACE | Real `credit-g` columns (age, credit_amount, real correlations) |
| `nn_ch01_credit_roc`, `credit_confusion`, `credit_training(_curves)`, `credit_importance`, `credit_shap`, `calibration` | `credit_figures.py`, `credit_scoring.py` | `make_roc(0.76/0.79)` synthetic ROC, hard-coded confusion matrices and importances, simulated calibration | S-REPLACE | Rerun `QL/NN_DL_ch01_CreditMLP` (real credit-g, already executed: acc 0.77) and export its figures; real SHAP |
| `nn_ch01_ablation_{hidden,activation,lr}` | `ablation_figures.py`, `ablation.py` | Hard-coded AUCs, `exp` + noise loss curves | S-REPLACE | Real ablation grid on German credit |
| `nn_ch01_credit_default_rates` | `credit_scoring.py` | Hard-coded default rates by rating (AAA 0.01% ... CCC 15%), no source | S-REPLACE (or cite) | S&P Global / Moody's annual default study table, with citation |
| `nn_ch01_q1_answer`, `q2_answer` | `qa_figures.py` | Toy grids | S-OK | n/a |
| `nn_ch02_computational_graph`, `chain_rule`, `backprop_flow`, `vanishing_gradient` (0.25^n), `gd_variants`, `adam_momentum` (contours), `learning_rate_schedule`, `q1_answer` | `nn_ch02_all_figures.py` | Schematic / analytic / toy optimiser paths on a quadratic | S-OK | n/a |
| `nn_ch02_optimizer_comparison`, `lr_sensitivity`, `q2_answer` | `nn_ch02_all_figures.py` | Fabricated `exp` + noise loss curves | S-REPLACE | Real runs (SGD/momentum/RMSProp/Adam; 4 LRs) on the S&P 500 VaR MLP or California housing |
| `nn_ch02_var_returns`, `var_forecast`, `var_violations` | `QL/NN_DL_ch02_VaRForecasting` (deck PDFs say "Data: stooq.com") | **R**: S&P 500 daily log-returns | R | Keep. Remove the GARCH-simulation fallback (the local `nn_ch02_all_figures.py` versions are simulated and obsolete) |
| `nn_ch03_bias_variance`, `l2_contour`, `l1_sparsity`, `dropout_diagram` | `nn_ch03_all_figures.py` | Analytic / schematic | S-OK | n/a |
| `nn_ch03_batchnorm_effect` | same | Random normal "activations" | S-OK (better: real activations of the fraud MLP before/after BN) | ULB fraud MLP hidden activations |
| `nn_ch03_early_stopping`, `ablation_dropout`, `ablation_weightdecay`, `q2_answer` | same | Fabricated loss/accuracy curves | S-REPLACE | Real runs on ULB credit-card fraud |
| `nn_ch03_q1_answer` (under/good/over-fit boundaries) | same | Two-moons + random-weight MLPs | S-REPLACE (borderline) | 2-D real data (HMEQ two ratios) |
| `nn_ch03_fraud_class_balance`, `fraud_roc`, `fraud_precision_recall`, `fraud_confusion` | Deck PDFs = `QL/NN_DL_ch03_FraudDetection` outputs. The local script versions are formula-based (PR = 1-0.55 r^1.6, hard-coded CMs) | Probably R (ULB, 0.2% fraud), but the QL notebook has a `make_classification` fallback and no stored outputs | Verify / regenerate | Rerun on ULB credit-card fraud with the fallback removed. Discard the formula-based versions in `figs/` |
| `nn_ch04_convolution_1d/2d`, `pooling`, `receptive_field`, `lenet_arch`, `resnet_skip`, `transfer_learning`, `finder_pipeline`, `q1_answer` | `nn_ch04_all_figures.py` | Schematic | S-OK | n/a |
| `nn_ch04_feature_maps` | same | Synthetic edge image | S-OK (better: real MNIST digit / CIFAR image) | MNIST |
| `nn_ch04_q2_answer` (gradient with/without skip) | same | Formula + noise | S-OK (mechanism) | n/a |
| `nn_ch04_mnist_{samples,variation,training,confusion,filters,misclassified}` | `gen_mnist_real.py`, `QL/NN_DL_ch04_*` | **R**: MNIST | R | Keep |
| `nn_ch04_mnist_comparison` | `gen_mnist_real.py` | Own CNN (real) + hard-coded LR/MLP/ResNet accuracies | Partly fabricated | Train all four on MNIST/Fashion-MNIST |
| `nn_ch04_candlestick_examples` | `nn_ch04_all_figures.py` | Hand-shaped OHLC + `np.random.uniform` wicks | S-REPLACE | Real OHLC windows (S&P 500 / BET, EODHD or stooq) where the pattern occurs |
| `nn_ch04_finder_roc`, `finder_confusion`, `arch_comparison` | same | `make_roc` power curves, hard-coded matrix and accuracies | S-REPLACE | Real chart-image CNN on S&P 500/BET windows, or drop and use CIFAR-10/EuroSAT transfer-learning results |
| `nn_ch05_rnn_unrolled`, `bptt`, `vanishing_gradient`, `lstm_gates`, `gru_gates`, `sequence_patterns`, `q1_answer`, `q2_answer` | `nn_ch05_all_figures.py`, `QL/NN_DL_ch05_LSTMViz` | Schematic / random sequences | S-OK | n/a |
| `nn_ch05_btc_returns` (title "Simulated Bitcoin Daily Returns") | `nn_ch05_all_figures.py` / QL fallback | t-distributed returns + injected crashes | S-REPLACE | Real BTC-USD daily (EODHD/CoinGecko/stooq) |
| `nn_ch05_crypto_lstm_forecast`, `crypto_roc` (AUC 0.71/0.77/0.78), `crypto_comparison`, `ablation_seqlen`, `ablation_hidden` | same | Fabricated probabilities, ROC and bars | S-REPLACE | Real LSTM/GRU/MLP results on BTC (or S&P 500/BET drawdown events) |
| `nn_ch06_tokenization`, `bow_matrix`, `word2vec_cbow/skipgram`, `nlp_text_examples`, `(nlp_)text_classification_pipeline`, `finbert_pipeline`, `attention_mechanism`, `multihead`, `scaled_dotprod`, `positional_encoding`, `transformer_block`, `encoder_decoder`, `bert_vs_gpt`, `q1_answer`, `q2_answer` | `gen_nlp_real.py`, `nn_ch06_all_figures.py` | Schematic / worked toy example / analytic | S-OK | n/a |
| `nn_ch06_glove_cooccurrence`, `tfidf` | `gen_nlp_real.py` | Hard-coded co-occurrence counts and weights | S-REPLACE | Counts and TF-IDF computed on Financial PhraseBank |
| `nn_ch06_word_embedding_space`, `nlp_word_embedding_space`, `financial_embeddings` | `gen_nlp_real.py`, `QL/EmbeddingViz` | Hand-placed 2-D points | S-REPLACE | PCA of real GloVe-6B / word2vec vectors (king/queen, bull/bear) |
| `nn_ch06_embedding_tsne` | `gen_nlp_real.py`, `QL/EmbeddingViz` | Random Gaussian clusters labelled Positive/Negative/Monetary/Instruments | S-REPLACE | t-SNE of GloVe or FinBERT embeddings of PhraseBank words |
| `nn_ch06_text_confusion`, `text_roc` / `nlp_text_roc` (62/68/74/85%), `ablation_dim`, `ablation_epochs`, `sentiment_comparison` (62/71/85%), `sentiment_confusion` (142/128/150) | no script / `nn_ch06_all_figures.py` | Hard-coded results | S-REPLACE | Real BoW/TF-IDF/Word2Vec/FinBERT results on Financial PhraseBank (sentences_allagree) |
| `nn_ch06_attention_heatmap` | `nn_ch06_all_figures.py`, `QL/FinBERT` | Hand-designed weights | S-REPLACE | Real BERT/FinBERT attention on a PhraseBank headline |
| `nn_ch07_*` (15 files: acf_squared_returns, correlation_heatmap, gan_training_dynamics, latent_interpolation, marginal_comparison, mode_collapse, tstr_results, vae_latent_space, q1/q3_answer, plus 4 "[Architecture Diagram]" boxes) | none (auto-generated placeholders) | Generic | **PH** | Real S&P 500 constituent / index returns (EODHD), real VAE/GAN on them, MNIST/Fashion-MNIST for latent space; TikZ for architectures |
| `nn_ch08_attention_heatmap`, `message_passing`, `results_auc`, `tsne_embeddings` ("Cluster A/B") | none | Generic | **PH** | Elliptic Bitcoin transaction graph (illicit vs licit) for L9 fraud; TikZ for message passing |
| `nn_ch09_action_distribution`, `cumulative_returns`, `drawdown`, `training_curves`, `weight_dynamics` ("Asset 1-4", "Day 0-250") | none | Generic | **PH** | Real ETF/index prices (S&P 500 sector ETFs or BET constituents) for a portfolio agent; Romanian day-ahead prices (energy-charts) for battery dispatch |
| `nn_ch11_llmvar_results` | none | Hard-coded violation rates (GPT-3.5/4/4o vs GARCH/HS) | Paper result, not reproduced | Cite the source paper explicitly or rerun on real S&P 500/BET |
| `nn_ch11_scaling_laws` = `nn_llm_scaling_laws` | none | Kaplan et al. power law | S-OK (analytic law; cite) | n/a |
| `nn_llm_architecture`, `causal_mask`, `param_breakdown` | `gen_llm_figures.py`, `QL/NanoGPT` | Schematic / exact parameter count | S-OK | n/a |
| `nn_llm_attention_weights`, `attention_heads`, `finance_attention` | `gen_llm_figures.py` | `np.random.rand` x causal mask | S-REPLACE | Attention of the trained NanoGPT (Tiny Shakespeare) or GPT-2 on a real sentence |
| `nn_llm_training_curve` | `gen_llm_figures.py` | `4 exp(-0.0006 t) + noise` | S-REPLACE | Logged loss from the real NanoGPT run |
| `nn_llm_text_evolution`, `temperature`, `next_token_dist`, `embedding_viz` | NanoGPT notebook (partly) / hard-coded | Hard-coded text (real Shakespeare quotes shown as model output at step 4500) | S-REPLACE | Actual samples and probabilities from the trained model |
| Bishop book figures (`AS/Bishop_DL_Figs/Chapter-N/Figure_k.pdf`, `A/figures/bishop/`), `nn_ch01_cajal_neuron`, `perceptron_mark1`, ch00 external PNGs (Unsplash, Epoch AI) | external | Book figures / photos | Not data (cite) | Epoch AI compute chart is real data (CC BY) |

### 2.2 QuantLet NN_DL scripts (`QL/`)

| Quantlet | Data | Class | Action / real replacement |
|---|---|---|---|
| ch01_Activations, ch01_XORViz, ch02_BackpropViz, ch05_LSTMViz, ch06_AttentionViz | Analytic / random weights / random Q,K,V | S-OK | AttentionViz could use real BERT attention |
| ch01_CreditMLP | German credit (OpenML) | R | Keep. Make it the source of all L1 credit figures |
| ch02_VaRForecasting | S&P 500 (stooq / GitHub) + GARCH fallback | R (fallback S) | Remove the fallback; pin a data snapshot; add BET |
| ch03_FraudDetection | ULB credit-card (OpenML) + `make_classification` fallback | R (fallback S) | Remove the fallback; execute and store outputs |
| ch03_RegularizationViz | `sin` + noise, random dropout mask | S-OK for the L1/L2 contour and dropout schematic; S-REPLACE for the overfitting demo | Polynomial/MLP overfitting on a small real sample (e.g. 30 California housing districts) |
| ch04_ConvViz, ch04_MNIST | MNIST | R | Keep |
| ch04_FINDER | empty stub | n/a | Implement on real OHLC or drop |
| ch05_CryptoPrediction | BTC (GitHub datasets/bitcoin) + synthetic fallback (used for the deck figure) | R (fallback S) | Real BTC-USD from EODHD/stooq; remove the fallback |
| ch06_EmbeddingViz | Random clusters | S-REPLACE | GloVe-6B 50d/100d |
| ch06_FinBERT | ~20 hand-written headlines | S-REPLACE | Financial PhraseBank (4,846 sentences) |
| ch06_TextMining | 20 hand-written labelled headlines, duplicated; random word vectors | S-REPLACE | Financial PhraseBank + pretrained word2vec/GloVe |
| ch07-ch13 (14 notebooks) | stubs | n/a | Write from scratch on real data (see sec. 5) |
| LLM_Workshop/NanoGPT | Tiny Shakespeare | R | Keep; export the real figures (replaces `gen_llm_figures.py`) |

### 2.3 Seminar and lecture notebooks

| Notebook | Synthetic element | Class | Real replacement |
|---|---|---|---|
| `seminar1_exercises` | Random regression data; `make_moons`; deck also `make_blobs` | S-REPLACE (first Keras model) | California housing (regression); HMEQ default (classification) |
| `seminar2_exercises` | `make_classification` "churn" | S-REPLACE | Kaggle Bank Customer Churn (already in `D/Proiecte/` Tanase) or IBM Telco churn |
| `seminar3_exercises` | `make_moons` (regularisation), `make_classification` fraud | S-REPLACE | ULB credit-card fraud (the deck already names it) |
| `seminar5_exercises` | Synthetic store demand (sin + promotions + noise) | S-REPLACE | Romanian electricity load / day-ahead price (energy-charts, Transelectrica), or the Kaggle store-item demand set the deck already cites |
| `seminar7_exercises` | Synthetic transactions (anomaly AE), synthetic ratings (recommender), hand-written review texts x15 (BERT) | S-REPLACE | ULB fraud; MovieLens 100k (the deck already uses it); Financial PhraseBank |
| `seminar4_exercises`, `seminar6_exercises` | none (MNIST, CIFAR-10, IMDB, 20NG) | R | Keep |
| `A/C/notebooks/ch00_CourseOverview`, `quantlets/NN_DL_ch00_PytorchBasics.py` | `y = 3x + 2 + noise` | S-REPLACE (mild) | California housing single feature |
| `A/C/notebooks/ch01_PerceptronsMLP`, `quantlets/NN_DL_ch01_CreditMLP.py` | Gaussian blobs; random "credit" features | S-REPLACE | German credit (as in the QL version) |
| `chapter1_lecture` | Analytic plots | S-OK | n/a |
| `chapter2_lecture` | Loss surfaces (OK) + `make_classification` | S-REPLACE for the classifier part | HMEQ / German credit |
| `chapter3_lecture` | `sin` overfitting (OK), `make_moons`, `make_classification` | S-REPLACE for the classifier parts | ULB fraud, HMEQ |
| `chapter4_lecture` | Synthetic pattern image, random 8x8 inputs | S-OK (filter demo); CIFAR-10 is real | n/a |
| `chapter5_lecture` | `sin(t)+0.3 sin(3t)` forecasting target | S-REPLACE | S&P 500/BET returns or Romanian day-ahead prices |
| `chapter6_lecture` | Synthetic embeddings fallback | S-REPLACE | Require GloVe via gensim-data; no fallback |
| `chapter7_lecture` | 1-D Gaussian-mixture GAN | S-OK as a GAN mechanism toy; better: real S&P 500 daily returns | S&P 500 returns |
| `AS/NEURAL_BIZ/Ch_01 tensorflow_basics` | Random tensors/data | S-OK for tensor ops; S-REPLACE for model fitting | California housing |
| `Ch_03 fraud_detection`, `Ch_03 regularization_demo` | `make_classification`, `make_moons` | S-REPLACE | ULB fraud; HMEQ |
| `Ch_04 cnn_quality_inspection` | Random images | S-REPLACE | MVTec AD zipper/bottle (the `CNN.zip` project already has zipper images) |
| `Ch_05 rnn_demand_forecast` | `sin` + noise demand | S-REPLACE | Romanian load / UCI household power |
| `Ch_07 recommender_system` | Random ratings | S-REPLACE | MovieLens 100k |
| `AS/20250411 Fraud-Detection-Demo` | PaySim-style simulated transactions | S-REPLACE | ULB credit-card fraud or IEEE-CIS fraud |
| `A/Projects/NLP_NER.zip` | Template shipment-ID sentences | S-REPLACE | CoNLL-2003 or FiNER-139 (financial NER) |
| `D/Proiecte/ Popescu et al. fraud_nn_project` | Rule-generated transactions | Student work (flag only) | n/a |

### 2.4 Legacy Bishop chapter code (`legacy/Bishop_Ch_*`, mirrored in `AS/NEURAL_BIZ/Bishop_Ch_*`; produces `AS/charts/`)

| Chapter notebooks | Data | Class | Real replacement |
|---|---|---|---|
| ch01 `figures`, `polynomial_fitting` | `sin(2 pi x)` + noise (Bishop curve fitting) | S-OK (canonical overfitting mechanism) | `polynomial_fitting_finance` already uses S&P 500 (yfinance) |
| ch02 `figures`, `probability`, `bayesian_finance` | Normal/multivariate/beta samples | S-OK (probability mechanics) | n/a |
| ch03 `distributions`, `figures` | Random samples | S-OK; `distributions_finance` uses S&P 500 (R) | n/a |
| ch04 `figures`, `linear_regression` | `sin` + uniform x, synthetic regression | S-OK for basis functions; S-REPLACE for "fit a model" exercises | California housing |
| ch05 `classification`, `figures`, `fig5_5b_roc_curve.py`, `fig5_11_roc_comparison.py` | `make_blobs`, `make_moons`, synthetic ROC scores | S-REPLACE (ROC figures); boundary plots S-OK | HMEQ / German credit ROC |
| ch05 `credit_scoring` | `make_classification` | S-REPLACE | German credit / HMEQ / `legacy/data/loan_data.csv` |
| ch06 `deep_networks`, `figures` | `sin` targets, random data | S-OK (UAT); `mlp_demo` uses S&P 500 (R) | n/a |
| ch07 `gradient_descent`, `figures` | `make_moons`, random | S-OK for optimiser paths; S-REPLACE for training comparisons | ULB fraud or California housing |
| ch08 `backpropagation`, `figures` | `tf.random.normal` tensors | S-OK | n/a |
| ch09 `regularization`, `figures` | `sin` + noise; MNIST (R) | S-OK / R | n/a |
| ch10 `convolutional_networks`, `figures` | MNIST, CIFAR-10 (R); synthetic image | R / S-OK | n/a |
| ch11 `graphical_models`, `figures` | Iris (R) + simulated Bayesian-network samples | S-OK | n/a |
| ch12 `transformers` (IMDB, R), `tft` (S&P 500, R), `finbert` (hand-written earnings sentences), `figures` | mixed | finbert: S-REPLACE | Financial PhraseBank / real earnings-call transcripts |
| ch13 `contagion`, `fraud_graph` | Simulated networks and transactions | S-REPLACE | Elliptic Bitcoin graph; real bank network (EBA transparency exercise / ECB data) for contagion |
| ch13 `gnn_demo`, `graph_neural_networks`, `figures` | Zachary karate club (R) + random features | R / S-OK | Elliptic for the business case |
| ch14 `sampling`, `figures` | Random samples | S-OK (sampling mechanism) | n/a |
| ch15 `em_algorithm`, `figures` | Synthetic GMM | S-REPLACE (EM demo) | S&P 500 / BET return regimes (2-component GMM) |
| ch16 `pca_latent_variables` | Olivetti faces, MNIST, digits (R); `figures` synthetic | R / S-OK | n/a |
| ch17 `gans` | MNIST (R); `figures` toy | R / S-OK | n/a |
| ch18 `normalizing_flows`, `figures` | `make_moons`, `make_circles` | S-OK (density mechanism); better: bivariate S&P 500 / DAX returns | S&P 500 + BET daily returns |
| ch19 `autoencoders_vae` | MNIST (R); `figures` toy | R / S-OK | n/a |
| ch20 `diffusion_models` | MNIST (R); `figures` toy | R / S-OK | n/a |
| ch04-ch20 `*_finance` / `*_business` (17) | stubs | n/a | n/a |

### 2.5 Summary of must-replace items

- **Decks (`A/C/`):**
  - ch00: 8 teaser figures.
  - ch01: about 17 figures (perceptron boundary/convergence, depth vs width, overfitting x2, credit EDA, ROC, confusion, training, importance, SHAP, calibration, 3 ablations, default rates).
  - ch02: 3 figures (optimizer comparison, LR sensitivity, q2).
  - ch03: 5 figures (early stopping, 2 ablations, q1, q2), plus 4 fraud figures to verify.
  - ch04: 5 figures (candlesticks, FINDER ROC/confusion, arch comparison, MNIST comparison baselines).
  - ch05: 6 figures (BTC returns, forecast, ROC, comparison, 2 ablations).
  - ch06: about 15 figures (co-occurrence, TF-IDF, 3 embedding-space, t-SNE, text confusion/ROC, 2 ablations, sentiment comparison/confusion, attention heatmap).
  - ch07: 15 placeholders. ch08: 4 placeholders. ch09: 5 placeholders.
  - LLM workshop: 7 figures. ch11: LLM-VaR bar chart (paper numbers).
- **QL:** EmbeddingViz, FinBERT, TextMining. Remove the synthetic fallbacks in VaRForecasting, FraudDetection and CryptoPrediction.
- **Seminars:** notebooks 1, 2, 3, 5, 7.
- **Lecture notebooks:** ch00, ch01, chapter2, chapter3 (classifier parts), chapter5, chapter6.
- **NEURAL_BIZ/Ch_\*:** ch1, ch3 x2, ch4 quality, ch5, ch7 recommender.
- **Fraud demo:** PaySim-style CSV.
- **Bishop legacy:** ch05 ROC/credit, ch12 finbert, ch13 contagion/fraud graph, ch15 EM.

---

## 3. Real datasets already used in the existing material

| Domain | Dataset | Where |
|---|---|---|
| Credit risk | German credit (UCI / OpenML `credit-g`) | QL ch01_CreditMLP, ch10 deck text |
| Credit risk | HMEQ home-equity loans | `legacy/data/hmeq*.csv`, `legacy/0. Intro` |
| Credit risk | UK application credit data (Thomas et al.), 1,225 rows | `legacy/data/loan_data.csv` |
| Fraud | ULB credit-card fraud (284,807 tx, 0.17% fraud) | QL ch03_FraudDetection, `legacy/4. NN Foundations`, seminar 3 deck |
| Market risk | S&P 500 index (stooq.com, GitHub `datasets/s-and-p-500`, yfinance `^GSPC`) | QL ch02_VaR, deck ch02 VaR figures, legacy Bishop ch01/03/06/12 |
| Crypto | Bitcoin daily prices (GitHub `datasets/bitcoin`) | QL ch05 (fallback was used for the deck figure) |
| Real estate | California housing | seminar 2, `Ch_02`, `legacy/4. NN Foundations` |
| Real estate | King County/Seattle 2014 sales; Sacramento transactions; London house prices (Kaggle) | `A/Projects/ANN_RealEstate`, `legacy/2.`, student project |
| Marketing / churn | UCI Bank Marketing; ISLR Advertising; Kaggle Bank Customer Churn; Telco churn (deck only) | `legacy/3.`, student projects, seminar 2 deck |
| Tabular other | Titanic, Iris, Breast cancer, Wine quality, Diamonds, Coursera ex2data | `legacy/`, seminar 2, `Ch_02` |
| Images | MNIST, CIFAR-10, Olivetti faces, sklearn digits, zipper defect images (MVTec-style) | decks ch04, seminar 4, `Ch_04`, Bishop ch10/16/17/19/20, `A/Projects/CNN.zip` |
| Time series | Airline passengers; UCI household power consumption; UCI Beijing PM2.5 | seminar 5, student project (Dutu), `legacy/4.` |
| Text | IMDB reviews; 20 Newsgroups; GLUE SST-2; Tiny Shakespeare; Financial PhraseBank, CoNLL-2003, ag_news, yelp (seminar 6 deck code only); disaster tweets; Dreaddit; FakeNewsNet; Twitter sentiment and topic sets | seminar 6, `Ch_06/07`, QL NanoGPT, tf-keras part 6, projects |
| Recommender | MovieLens 100k, Book-Crossing (deck code only); TMDB movies | seminar 7 deck, `NLP_RS.zip`, student project |
| Graphs | Zachary karate club | legacy Bishop ch13 |
| Energy | Romanian OPCOM day-ahead **named in deck ch12 text only, no data file or figure exists** | ch12, ch14 |
| Indices | S&P 500, EURO STOXX 50, BET **named in deck ch13 text only** | ch13 |

---

## 4. Reusable assets

| Asset | Location | Reuse in |
|---|---|---|
| Beamer preamble and colour palette (MAIN_BLUE #1A3A6E, IDA_RED, FOREST, CRIMSON, AMBER), slide structure (Why, theory, business case, results, code, review, summary) | `A/C/preamble_nn.tex` | All lectures |
| TikZ diagrams (perceptron, MLP, backprop, dropout, CNN, LSTM/GRU, attention, transformer, VAE/GAN, diffusion), about 200 `tikzpicture` blocks | `A/C/chapter1..7_*.tex` (2025), `ch00`, `ch09`, `ch11`, `ch11b`, `ch12`, `ch13` | L1-L8, L10, L12-L14 |
| Clean mechanism figures (activations, XOR/hidden space, UAT, L1/L2 contours, GD paths, LR schedules, conv/pool/receptive field, LeNet/ResNet/transfer, RNN/LSTM/GRU cells, attention/multi-head/positional encoding/BERT-vs-GPT, causal mask) | `A/C/figs/*.py` (scripts kept) | L1-L7 |
| Real MNIST pipeline with figures | `A/C/figs/gen_mnist_real.py`, `QL/NN_DL_ch04_*` | L4 |
| German credit MLP (executed, real) | `QL/NN_DL_ch01_CreditMLP` | L1, L11 (SHAP base model) |
| S&P 500 VaR MLP with Kupiec backtest | `QL/NN_DL_ch02_VaRForecasting` | L2, L13 |
| ULB fraud MLP | `QL/NN_DL_ch03_FraudDetection`, `legacy/4. NN Foundations/From_Logistic_Regression_to_Deep_Learning.ipynb` | L3, L8 (AE anomaly), L11 |
| NanoGPT from scratch (real corpus) + 70-frame workshop deck | `QL/NN_DL_LLM_NanoGPT`, `A/C/llm_from_scratch.tex` | L7, L12 |
| Beijing PM2.5 LSTM, California housing linear-to-DL notebooks | `legacy/4. NN Foundations/` | L1, L5, L13 |
| Bishop exercise solutions and figure set | `AS/Concepts_in_Deep_Learning_Solutions_v1.0.pdf`, `AS/Bishop_DL_Figs/`, `AS/charts/` | Problem sets, theory slides |
| Bishop seminar decks (20) | `A/Seminars/Bishop_seminars/` | Exercise sessions |
| Seminar decks 4 and 6 (real data, HuggingFace pipelines, Grad-CAM code) | `A/Seminars/` | L4, L6, L7 |
| Prompt-engineering deck (text and TikZ, no data) | `A/C/ch11b_PromptEngineering.tex` | L12 |
| XAI deck (SHAP/LIME/IG/CAV/stability theory; needs real figures) | `A/C/ch10_Explainability.tex` | L11 |
| Student project exemplars with real data (fake news, Dreaddit, energy LSTM, churn) | `D/Proiecte/` | Project brief / examples |
| External figure source log | `A/C/figs/external/SOURCES*.md` | Licensing |

---

## 5. Gaps

| Lecture | Existing coverage | Gap |
|---|---|---|
| L8 Generative | ch07 deck (VAE/GAN theory, 15 placeholder figures); 2025 chapter7 and Bishop ch17-20 cover diffusion; MNIST VAE/GAN notebooks exist | **Diffusion barely in the 2026 deck.** No real financial generative case (QL ch07 stubs). Real S&P 500 / BET returns and Fashion-MNIST notebooks needed |
| L9 GNN | ch08 deck (interbank systemic risk, text only); Bishop ch13; karate club | **No fraud or transaction-network case and no real graph data.** QL ch08 stubs. Need Elliptic Bitcoin (PyG `EllipticBitcoinDataset`) GCN/GAT/GraphSAGE notebook and figures |
| L10 RL | ch09 deck (portfolio, PPO/DQN theory; 5 placeholder figures) | **No hedging and no battery dispatch; no runnable code** (QL ch09 stubs). Need: ETF portfolio env on real prices; deep hedging on S&P 500 options or index paths; battery arbitrage on Romanian day-ahead prices (energy-charts) |
| L11 XAI | ch10 deck theory (no figures); 2025 chapter3 SHAP/LIME section | **No figures or notebooks** (QL ch10 stubs). Need SHAP/LIME/integrated gradients plus a stability study on the real German credit / HMEQ MLP |
| L12 LLMs | ch11 + ch11b decks, NanoGPT workshop; seminar 6/7 HF pipelines | No executed LLM notebooks (QL ch11 stubs). LLM-VaR figure carries paper numbers. Need a reproducible prompt/LLM classification on Financial PhraseBank and a real VaR replication |
| L13 Energy and TS foundation models | ch12 deck (OPCOM text, no figures, no data); seminar 5 and student energy LSTM | **No real energy data in the material; no Chronos/TimesFM/Moirai content at all** (0 hits). Need Romanian day-ahead prices and load from energy-charts / ENTSO-E, BET / S&P 500, and Chronos zero-shot vs LSTM benchmarks |
| L14 Multimodal / VLM | ch13 deck (CLIP/LLaVA theory, FINDER-VLM text); ch14 synthesis deck | **No data, figures or code** (QL ch13 and ch04_FINDER stubs). Need a CLIP zero-shot demo on real chart images (S&P 500 / BET candlestick windows) or a document VQA example |
| L6/L7 NLP | Decks complete but most result figures fabricated; seminar 6 is solid | Rebuild all result figures on Financial PhraseBank with real embeddings (GloVe) and FinBERT |
| L4 CNN | MNIST real; transfer learning only schematic | Real transfer-learning results (CIFAR-10 / EuroSAT / MVTec AD); FINDER chart-pattern case is fabricated |
| L5 RNN | Theory complete; crypto case fabricated | Real BTC or BET/S&P 500 sequence case with honest results |
| L1-L3 | Strongest coverage | Replace the fabricated result figures with real runs (German credit, S&P 500 VaR, ULB fraud) |
| Cross-cutting | n/a | No Romanian datasets in use (BET index, Romanian electricity, BNR series); no acronym/abbreviation slides checked; executed QL notebooks lack stored outputs except CreditMLP |
