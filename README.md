# learnai

A collection of deep learning experiments, NLP hackathon pipelines, AI security attacks, and automata learning research.

---

## 🛠️ Core Libraries & Tech Stack

| Category | Libraries / Tools | Used For |
|---|---|---|
| **NLP & Transformers** | `transformers`, `datasets`, `evaluate`, `torchcrf` | Fine-tuning RoBERTa, XLM-R, RoGPT2, CRF token classification (NER), text tokenization |
| **Deep Learning** | `torch`, `torchvision` | Custom RNNs/GRUs, CNNs, MLPs, training loops, model inference |
| **Automata Learning** | `aalpy` | Active Automata Learning ($L^*$ algorithm, Random W-method, SUL oracle) |
| **Machine Learning** | `scikit-learn`, `scipy` | TF-IDF vectorization, Ridge, Logistic Regression, K-Fold CV, decision boundary estimation |
| **Data & Viz** | `pandas`, `numpy`, `matplotlib`, `Pillow` | Data manipulation, ensembling/blending predictions, metrics visualization |

---

## 🚀 Projects

### 1. Nitro NLP Hackathon (`nitronlp/`)
Competition pipelines and training experiments built for the **Nitro NLP Hackathons** (WAICC 2024 & 2025).

* **Libraries:** `transformers`, `datasets`, `evaluate`, `torchcrf`, `torch`, `scikit-learn`, `pandas`
* **Components:**
  * **General NLP Pipeline (`nlp_general_pipeline.ipynb`):** Reusable template for text classification — Hugging Face dataset loading, tokenization, transformer fine-tuning, evaluation, and CSV submission export.
  * **WAICC 2025 (`2025waicc/`):** Romanian NLP competition experiments featuring RoGPT-2 (zero-shot & calibrated), RoBERT, and weighted ensembling / score blending (`merge.py`).
  * **Nitro NLP 2024 (`2024waicc/`):** Multi-task classification using TF-IDF + Logistic Regression baselines combined with fine-tuned XLM-RoBERTa and model ensembling.
  * **Practice Tasks (`learning_tasks/`):**
    * `ronec-tokenclassification.ipynb`: Named Entity Recognition (NER) on Romanian RONEC dataset using RoBERT + linear CRF (`torchcrf`).
    * `commonlitreadabilityprize.ipynb`: Text readability regression blending RoBERTa-base and TF-IDF + Ridge with K-Fold cross-validation.
    * `disastertweets.ipynb`: Disaster tweet classification baseline.

---

### 2. DFA Extraction from RNNs (`dfa-extraction-rnn/`)
Extracting exact, minimal Deterministic Finite Automata (DFAs) from continuous PyTorch RNNs.

* **Libraries:** `aalpy`, `torch`
* **Approach:**
  * Train a PyTorch RNN on regular languages (Tomita Grammar 2: `(0*11)*0*`).
  * Wrap the RNN inside an `aalpy` System Under Learning (SUL) oracle.
  * Run Angluin's $L^*$ active learning algorithm with Random W-method equivalence queries to extract and visualize the learned DFA.
* **Notebook:** [`dfa-extraction-rnn/demo.ipynb`](dfa-extraction-rnn/demo.ipynb)

---

### 3. Trustworthy AI & Dataset Inference (`trustworthy-ai/`)
Membership inference and dataset provenance attacks, developed for the **CISPA European Championship in Trustworthy AI** (Stockholm).

* **Libraries:** `torch`, `torchvision`, `numpy`, `scipy`
* **Approach:**
  * **Blind Walk Attack (`01_dataset_inference.ipynb`):** Estimates distance to the decision boundary of a black-box ResNet-18 vision model by adding calibrated Gaussian noise along random vectors until prediction flips.
  * Generates membership scores across 1,000 subsets to verify whether data was part of the training set (`TPR@FPR=0.05`).
  * Includes literature notes and competition specs in `trustworthy-ai/papers/`.

---

### 4. Deep Learning Foundations (`cnn/`, `rnn/`, `mlp/`)
Fundamental PyTorch implementations from scratch:

* **Libraries:** `torch`, `torchvision`, `Pillow`
* **`cnn/`:** Dogs vs. Cats image classifier (`catvsdog.ipynb`) and MNIST ConvNet (`digit_mnist.ipynb`).
* **`rnn/`:** Character-level GRU lyric generator with temperature sampling (`eminem_rnn_generator.ipynb`) + regex lyrics preprocessor (`lyrics_cleaning.ipynb`).
* **`mlp/`:** Standard feedforward neural nets on MNIST and Fashion-MNIST.
* **`model_demo.ipynb`:** Quick interactive notebook loading saved checkpoints for both the character lyric generator and the Tomita-2 grammar classifier.

---

## 📁 Repository Structure

```text
.
├── nitronlp/                         # Nitro NLP Hackathon & practice pipelines
│   ├── nlp_general_pipeline.ipynb    # Fast HF prototyping template
│   ├── 2025waicc/                    # WAICC 2025: RoGPT2, RoBERT, ensemble scripts
│   ├── 2024waicc/                    # Nitro NLP 2024: XLM-R, TF-IDF + LR blending
│   └── learning_tasks/               # Romanian NER (RONEC + CRF), CommonLit, Tweets
│
├── dfa-extraction-rnn/               # Active Automata Learning from RNNs
│   └── demo.ipynb                    # PyTorch RNN + AALpy L* extraction
│
├── trustworthy-ai/                   # CISPA European Championship in Trustworthy AI
│   ├── 01_dataset_inference.ipynb    # Blind-walk decision boundary attack
│   └── papers/                       # Challenge papers & specs
│
├── cnn/                              # CNNs (Dogs vs. Cats, MNIST)
├── rnn/                              # Character-level GRU generator & cleaning
├── mlp/                              # MLPs (MNIST, Fashion-MNIST)
├── model_demo.ipynb                  # Checkpoint demo notebook
└── requirements.txt                  # Python dependencies
```

---

## ⚡ Quickstart

1. **Clone & enter repo:**
   ```bash
   git clone https://github.com/costitza/learnai.git
   cd learnai
   ```

2. **Environment & dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
