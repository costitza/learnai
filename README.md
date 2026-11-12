# learnai

A hands-on collection of deep learning implementations, AI security attacks, and research experiments.

This repository covers:
1. **Research Project:** Extracting Deterministic Finite Automata (DFAs) from Recurrent Neural Networks using Angluin's $L^*$ active learning algorithm (presented as research work at my faculty).
2. **AI Security & Trustworthy AI:** Dataset inference attacks against vision models (ResNet18) built for the **[CISPA European Championship in Trustworthy AI](https://cispa.de/en/cispa-hackathon-championship-stockholm)** (Stockholm).
3. **Deep Learning Foundations:** Core PyTorch implementations across Computer Vision, Sequence Modeling, and MLPs.

---

## Highlights

### 1. DFA Extraction from RNNs via Angluin's $L^*$ Algorithm (`dfa-extraction-rnn/`)
*Presented at the Faculty of Computer Science.*

Recurrent Neural Networks trained on formal languages internalize state-transition dynamics, but their internal representations are continuous and hard to interpret. This project extracts an exact, minimal Deterministic Finite Automaton (DFA) from a trained PyTorch RNN:
- **Language:** Tomita Grammar 2 — regular expression `(0*11)*0*`.
- **System Under Learning (SUL):** Wrapped the trained RNN in an active learning oracle using `aalpy`. The oracle runs inference over character sequences and outputs discrete accept/reject decisions.
- **Active Automata Learning ($L^*$):** Angluin's $L^*$ algorithm iteratively queries the RNN with membership and equivalence queries (using a Random W-Method equivalence oracle) until it constructs the minimal DFA that mirrors the network's classification boundaries.
- **Artifact:** The extracted automaton visualizes the exact states and transitions the RNN learned.

Check out [`dfa-extraction-rnn/demo.ipynb`](file:///home/raulababei/Desktop/learnai/dfa-extraction-rnn/demo.ipynb) for the full training, SUL wrapper, and extraction pipeline.

---

### 2. Trustworthy AI & Dataset Inference (`trustworthy-ai/`)
*Built for the CISPA Hackathon / European Championship in Trustworthy AI (Stockholm).*

Organized by CISPA (Prof. Dr. Adam Dziedzic and Prof. Dr. Franziska Boenisch), this competition focused on real-world AI security, privacy, and data provenance.

- **Task 1: Dataset Inference Attack on Vision APIs (`01_dataset_inference.ipynb`)**
  - **Goal:** Determine whether a specific dataset was used to train a target image classifier (ResNet-18) without direct access to the training pipeline.
  - **Method:** Implemented the **Blind Walk** boundary estimation technique. By progressively adding calibrated Gaussian noise along random vectors until prediction labels flip, we estimate the distance from an input sample to the model's decision boundary. Training samples typically sit deeper inside decision basins and require significantly more perturbation steps to cross the boundary than unseen samples.
  - Generates membership scores across 1,000 subsets (100 images each) to maximize attack success (`TPR@FPR=0.05`).
- **Research Papers & Literature (`trustworthy-ai/papers/`):**
  - Dataset Inference for Vision APIs & Text Attribution (Competition specifications)
  - *Data Provenance for Image Auto-Regressive Generation* (ICLR 2026, CISPA)
  - *SoK: Data Reconstruction Attacks Against Machine Learning Models* (USENIX Security 2025, CISPA)
  - *Adversarial Inputs for Linear Algebra Backends*

---

### 3. Deep Learning Foundations

Practical implementations of standard architectures from scratch in PyTorch:
- **`cnn/`:**
  - `catvsdog.ipynb`: Custom 3-stage convolutional network for binary classification on the Microsoft Dogs vs. Cats dataset. Includes corrupt image verification and evaluation loops.
  - `digit_mnist.ipynb`: 2-layer ConvNet with max pooling on MNIST.
  - `split_data.py`: Script to prepare and shuffle train/test directory splits.
- **`rnn/`:**
  - `eminem_rnn_generator.ipynb`: Character-level GRU text generator trained on lyrics, featuring temperature-scaled sampling.
  - `lyrics_cleaning.ipynb`: Regex preprocessing pipeline to strip metadata, section tags, and noisy headers.
- **`mlp/`:**
  - `mnist.ipynb`: 2-hidden-layer feedforward network on MNIST digits.
  - `fashion_mnist.ipynb`: Multilayer perceptron for Fashion-MNIST clothing classification.
- **`model_demo.ipynb`:**
  - Quick interactive notebook loading saved checkpoints for both the character lyric generator and the Tomita-2 grammar classifier.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── model_demo.ipynb                  # Interactive demo for trained models
│
├── dfa-extraction-rnn/               # Research: Automata extraction from RNNs
│   └── demo.ipynb                    # PyTorch RNN + AALpy L* extraction
│
├── trustworthy-ai/                   # CISPA European Championship in Trustworthy AI
│   ├── 01_dataset_inference.ipynb    # Blind-walk decision boundary attack
│   └── papers/                       # Challenge briefs & related security papers
│
├── cnn/                              # Convolutional Neural Networks
│   ├── catvsdog.ipynb                # Dogs vs. Cats classifier
│   ├── digit_mnist.ipynb             # MNIST CNN
│   └── split_data.py                 # Dataset train/test splitting utility
│
├── rnn/                              # Recurrent Neural Networks
│   ├── eminem_rnn_generator.ipynb    # Char-level GRU lyric generator
│   └── lyrics_cleaning.ipynb         # Lyrics dataset preprocessing
│
└── mlp/                              # Multilayer Perceptrons
    ├── mnist.ipynb                   # MNIST MLP
    └── fashion_mnist.ipynb           # Fashion-MNIST MLP
```

---

## Setup & Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/costitza/learnai.git
   cd learnai
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

---

## Tech Stack & Tools

- **Frameworks:** PyTorch, Torchvision
- **Automata & Formal Learning:** AALpy (Active Automata Learning for Python)
- **Data & Visualization:** NumPy, Pandas, Matplotlib, Pillow
- **Environment:** Jupyter Notebooks, Python 3.10+
