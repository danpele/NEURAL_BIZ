<!--
Web version of Lecture 12 (EN). Built into web/L12.html by lectures/common/build_web_lecture.py (format: see its docstring).
Frame hashes: python3 build_web_lecture.py 12 --frames
-->

## @intro

Text is the largest data set of a financial institution that is still read mostly by people: news, central-bank speeches, annual reports, loan files. Large language models (LLM) read text at scale, but they are also confident, expensive and easy to fool. This lecture treats an LLM as what it is, a next-token probability model, and measures every claim on labelled financial data with open models that run on a laptop.

We start from the language-model objective, tokens, perplexity and temperature, measured on 400 speeches of the European Central Bank (ECB). We then explain pre-training, instruction tuning and alignment. The core of the lecture is prompt engineering as a measurable activity: zero-shot, few-shot and chain-of-thought prompts on the Financial PhraseBank and on finance tweets, with bootstrap intervals, seeds, paraphrases, calibration and cost. We build retrieval-augmented generation (RAG) over the ECB speeches and measure hallucination, and we backtest a language model used for the Value-at-Risk (VaR), after the LLM-VaR framework of our group.

@goals
- **Write** the language-model objective, perplexity and the temperature of sampling, and measure them on real text.
- **Explain** pre-training, instruction tuning and alignment from human preferences, with their loss functions.
- **Turn** a prompt into a classifier and measure zero-shot, few-shot and chain-of-thought prompts with confidence intervals.
- **Build and evaluate** retrieval-augmented generation over real speeches, and measure hallucination.
- **Backtest** a language model used for the Value-at-Risk and know the risks: leakage, cost, calibration.

# Next-token prediction and large language models

An LLM is trained to predict the next token. Everything else, from answering questions to writing a risk report, is repeated use of that single probability.

## Text data in financial institutions {h=5506846a}

Prices react to news within minutes, central banks steer expectations with words, and loan files and annual reports cost analyst hours. All models in this lecture are open and run locally (Qwen2.5-Instruct with 0.5, 1.5 and 3 billion parameters), so no text leaves the machine and no application programming interface (API) is paid.

## The language-model objective {h=59ec94e3}

$$ p_{\boldsymbol\theta}(x_1,\dots,x_T)=\prod_{t=1}^{T}p_{\boldsymbol\theta}(x_t\mid x_{<t}),\qquad \mathcal L=-\frac{1}{T}\sum_{t=1}^{T}\log p_{\boldsymbol\theta}(x_t\mid x_{<t}),\qquad \mathrm{PPL}=e^{\mathcal L} $$

Perplexity is the effective number of equally likely next tokens. On ECB speeches published after July 2025 (first 1,024 tokens of 40 speeches) the perplexity is 14.71 for the 0.5B model, 10.63 for 1.5B and 9.43 for 3B.

## Subword tokens {h=de1a3534}

Byte-pair encoding (BPE) starts from bytes and repeatedly merges the most frequent adjacent pair until the vocabulary is full: Qwen2.5 has 151,665 tokens; the WordPiece vocabulary of FinBERT, a financial version of bidirectional encoder representations from transformers (BERT), has 30,522. On the Financial PhraseBank there are 1.35 Qwen tokens per word.

## Sentence length in words and in tokens {h=9293003a}

@chart tokens_per_sentence
Length of the Financial PhraseBank sentences in words and in tokens.

A sentence has on average 23.1 words and 31.1 Qwen tokens; an ECB speech has on average 4,143 tokens. Costs and context limits are counted in tokens.

## Temperature {h=b0811731}

$$ p_T(v\mid x_{<t})=\frac{\exp(z_v/T)}{\sum_u\exp(z_u/T)},\qquad H_T=-\sum_v p_T(v)\log p_T(v) $$

With the prompt "The inflation impact is limited:", taken from the ECB speech of 30 October 2026, the most likely next token "the" has probability 49.9 % at temperature 0.5, 10.1 % at 1 and 0.4 % at 2; the entropy grows from 1.74 to 10.29 nats.

## Next-token probabilities of a real model {h=434bb8ca}

@chart next_token
Top next tokens of Qwen2.5-1.5B-Instruct at three temperatures.

@practice for classification and extraction use temperature 0; reserve high temperatures for generating ideas.

## Scaling {h=fc09257c}

The loss falls as a power of model size and data, $\mathcal L(P,D)=E+A/P^{\alpha}+B/D^{\beta}$ (Kaplan et al., 2020; Hoffmann et al., 2022). On our three model sizes the slope of log loss on log parameters is −0.100: three points describe our sample, they are not a law.

## Training-data cut-off {h=3719104b}

@chart perplexity_size
Perplexity on ECB speeches of 2022–2023 and from July 2025.

For the 3B model, perplexity is 8.90 on speeches of 2022–2023 and 9.43 on speeches from July 2025, after the model's training data. Topics also changed, so the gap is a symptom, not a proof, of memorisation; it is a reason to evaluate LLMs on documents written after their training cut-off.

# Pre-training, instruction tuning and alignment

## Three training stages {h=76c9c0be}

A generative pre-trained transformer (GPT) is first trained on trillions of tokens with the next-token loss; supervised fine-tuning (SFT) then teaches the format of answers, and reinforcement learning from human feedback (RLHF) or direct preference optimisation (DPO) teaches which answers people prefer (Ouyang et al., 2022).

## Instruction tuning {h=6bb22115}

$$ \mathcal L_{\text{SFT}}(\boldsymbol\theta)=-\sum_{(\mathbf x,\mathbf y)}\sum_{t=1}^{|\mathbf y|}\log p_{\boldsymbol\theta}(y_t\mid\mathbf x,\mathbf y_{<t}) $$

The loss is computed on the answer tokens only. Low-rank adaptation (LoRA) trains $\mathbf W=\mathbf W_0+\mathbf B\mathbf A$ with a small rank $r$: a 4096 × 4096 layer with $r=8$ has 65,536 trainable weights instead of 16.8 million (Hu et al., 2022).

## Alignment {h=76c86071}

People choose the better of two answers; a reward model $r_\phi$ is fitted with $\Pr(y_w\succ y_l)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$ and the policy maximises the reward with a Kullback–Leibler penalty that keeps it close to the SFT model. DPO optimises the same objective directly from the preference pairs (Rafailov et al., 2023). Alignment sets tone and safety; it does not add financial knowledge.

## The open models used here {h=a55074bf}

Qwen2.5-Instruct with 0.49, 1.54 and 3.09 billion parameters (Qwen Team, 2024), the bge-small embedding model for retrieval, and a term frequency–inverse document frequency (TF-IDF) representation with logistic regression as the classical benchmark.

# Evaluation of prompt engineering on labelled data

A prompt is a model. It is chosen on validation data and scored once on a held-out test set, with the same statistics as any classifier.

## A prompt as a classifier {h=a78713ad}

$$ \hat p(c\mid x)=\frac{\sum_{v\in\mathcal V_c}p_{\boldsymbol\theta}(v\mid\mathrm{prompt}(x))}{\sum_{c'}\sum_{v\in\mathcal V_{c'}}p_{\boldsymbol\theta}(v\mid\mathrm{prompt}(x))},\qquad \hat c=\arg\max_c\hat p(c\mid x) $$

where $\mathcal V_c$ are the first tokens of the label word $c$ in all its spellings. One forward pass per sentence is enough, and the shared instruction and examples are encoded once.

## Two labelled data sets {h=4234b836}

The Financial PhraseBank (Malo et al., 2014) contains 4,838 sentences from news about Finnish listed companies; we use the 2,259 with unanimous labels, split into 1,581 training and 678 test sentences. The Twitter financial news data set gives 9,543 training tweets and a stratified test sample of 900 tweets.

## Prompted LLMs against a trained benchmark {h=c2e2adc7}

@chart prompt_results
Macro-F1 with 95 % bootstrap intervals on both data sets.

| Method | PhraseBank macro-F1 (%) | Twitter macro-F1 (%) |
|---|---|---|
| TF-IDF + logistic regression | 86.9 | 75.5 |
| Zero-shot 0.5B | 31.0 | 18.7 |
| Zero-shot 1.5B | 85.1 | 40.9 |
| Zero-shot 3B | 92.2 | 77.4 |
| Few-shot 3B (12 examples) | 92.1 | 79.8 |
| Chain-of-thought 3B | 88.4 | 65.1 |
| LLM embeddings + logistic | 87.5 | 79.0 |

On PhraseBank the 3B model reaches 92.2 % macro-F1 zero-shot, above the TF-IDF classifier trained on 1,581 labelled sentences (86.9 %). On tweets, which are shorter and noisier, the best results come from the 3B model with examples (79.8 %) and from LLM embeddings with a trained classifier (79.0 %). The 0.5B model is close to the majority class.

## Accuracy and macro-F1 with bootstrap intervals {h=7af539f0}

Macro-F1 averages the three per-class F1 scores, so the small negative class counts as much as the neutral one. The 95 % intervals come from 1,000 bootstrap resamples of the test set: for zero-shot 3B on PhraseBank the interval is [89.8; 94.4] %, for TF-IDF [83.8; 90.0] %.

## Examples in the prompt {h=2ff98acb}

@chart fewshot_curve
Macro-F1 against the number of examples in the prompt, three seeds.

In-context learning changes the output without changing any weight. For the 1.5B model, macro-F1 moves from 85.1 % with no example to 84.1 % with 24; for the 3B model from 92.2 % to 91.3 %. On this task the zero-shot instruction is already strong, and the choice of examples moves the score by several points.

## Prompt sensitivity {h=92b0744b}

@chart prompt_sensitivity
Macro-F1 under five paraphrases of the same instruction.

Five instructions with the same meaning give a range of 3.6 points on PhraseBank and 12.9 points on Twitter for the 3B model, and 14.2 points on PhraseBank for the 1.5B model. The wording of a prompt is a hyper-parameter: select it on validation data and freeze it.

## Chain-of-thought {h=a8a5a234}

Letting the model write two sentences of reasoning before the label (Wei et al., 2022) gives 88.4 % macro-F1 on PhraseBank and 65.1 % on Twitter, below the direct zero-shot answer, at about 389 ms per sentence instead of 48.5 ms. Chain-of-thought helps multi-step reasoning, not one-word labels on short sentences.

## Calibration {h=aad9db85}

@chart calibration
Reliability diagram of the predicted label probabilities.

The expected calibration error (ECE) is 4.7 % for zero-shot 3B, 3.6 % for few-shot 3B and 5.4 % for TF-IDF with logistic regression: on this task the label probabilities of the LLM are about as reliable as those of the trained classifier. Calibration must still be checked on validation data before scores are used as risk probabilities (Guo et al., 2017).

## Accuracy by level of annotator agreement {h=b2d8e599}

@chart agreement
Accuracy by annotator agreement.

Zero-shot 3B is right on 93.1 % of the unanimous test sentences but only on 53.7 % of the sentences on which half of the annotators disagreed. A model cannot be more certain than the people who labelled the data.

## The confusion matrix {h=f95eba35}

For the best prompted model on PhraseBank, 20 neutral sentences were read as positive and 13 positive sentences as neutral, while positive and negative were rarely swapped. Errors sit on the neutral border.

## Cost against accuracy {h=a2a30d55}

@chart cost_accuracy
Macro-F1 against milliseconds per sentence on a laptop.

TF-IDF needs 2.50 ms per sentence, zero-shot 3B 48.5 ms and chain-of-thought about 389 ms. With labels, a small trained model is hard to beat; without labels, a prompt is a strong start; for millions of documents, cost decides.

## LLM embeddings with a small classifier {h=5ee07f3a}

Averaging the last hidden states of Qwen2.5-1.5B and fitting a logistic regression on top gives 87.5 % on PhraseBank and 79.0 % on Twitter: the cheapest form of fine-tuning.

# Retrieval-augmented generation on ECB speeches

## Retrieval followed by generation {h=81fa0b52}

$$ p(y\mid q)\approx\sum_{z\in\mathrm{top}_k(q)}p_\eta(z\mid q)\,p_{\boldsymbol\theta}(y\mid q,z) $$

The corpus has 400 ECB speeches cut into 8,400 chunks of 200 words (Lewis et al., 2020).

## Three retrievers {h=5e7eb3b0}

The Best Matching 25 (BM25) score weights shared words by their rarity and saturates repeated counts; dense retrieval compares embeddings by cosine similarity; the hybrid combines both rankings by reciprocal rank fusion. Mean reciprocal rank (MRR): BM25 0.515, dense 0.567, hybrid 0.622.

## Retrieval of the correct speech from its title {h=7ebb218d}

@chart rag_recall
Recall@k of the four retrievers.

The hybrid retriever puts the right speech first for 50.7 % of the titles and within the first ten for 83.2 %.

## Answers with and without retrieved documents {h=63db9336}

@chart rag_qa
Who gave the speech? Answers without documents, with RAG and with the right speech.

On 160 speeches given after the model's training data, a forced answer without documents is wrong in 86.2 % of cases (the model answered "Philip Lane" 137 times). When "I don't know" is allowed, it abstains on 100.0 % of questions. With the top-3 retrieved speeches it answers correctly in 81.2 % of cases, and with the right speech in 98.8 %.

@practice a confident wrong name is the costly failure: require sources in every answer and check them.

## Designing a RAG system for a bank {h=956f9d44}

Chunk size, metadata that travel with each chunk, a hybrid retriever (recall@3 of 73.1 % on the question set), permission to abstain and a fixed evaluation set rerun after every change are the choices that matter.

# Language models as risk models: LLM-VaR

## LLM-VaR {h=80eeef3c}

In the LLM-VaR framework (Pele et al., 2026) a general-purpose model reads the last 250 daily returns in a prompt and writes the Value-at-Risk, with no estimation and no fine-tuning. We test the idea with an open 3B model on the S&P 500, from 2 January 2025 to 30 October 2026 (459 days), against historical simulation (HS) and a GARCH(1,1) model with Student-t errors, with the Kupiec test of coverage.

## LLM-VaR with a 3B open model {h=c01f39c0}

@chart llm_var
S&P 500 returns with the 5 % LLM-VaR and the GARCH VaR.

At 5 %, the LLM-VaR was breached 28 times, GARCH 35 times and historical simulation 23 times, against 23.0 expected. The mean 5 % VaR of the small model is −1.99 % against −1.42 % for GARCH, and the two series are almost uncorrelated (0.05): the 3B model reaches its coverage by being cautious, not by tracking volatility.

## Backtest {h=7c0ff035}

| Model | Violations 1 % | Kupiec p (1 %) | Violations 5 % | Kupiec p (5 %) |
|---|---|---|---|---|
| LLM-VaR (Qwen2.5-3B) | 4 | 0.777 | 28 | 0.295 |
| Historical simulation | 6 | 0.528 | 23 | 0.991 |
| GARCH(1,1)-t | 9 | 0.067 | 35 | 0.016 |

The published study evaluated much larger general-purpose models; the results here describe a 3B model on a laptop only.

# Risks and governance

## Six risks and the test for each {h=a84d6f38}

Hallucination (86.2 % wrong names without documents) is controlled by RAG with sources; leakage by testing on text written after the training cut-off; prompt sensitivity (12.9 points) by selecting and freezing prompts on validation data; miscalibration (ECE 4.7 % zero-shot) by checking and, if needed, recalibrating on validation data; cost by the smallest model that passes the test; data protection by local models and by the duties of the European Union (EU) Artificial Intelligence (AI) Act for high-risk uses such as credit scoring.

## @end

A prompt is a model: measure it on labelled data, with intervals, seeds and paraphrases; ground answers in documents; backtest any risk number.
