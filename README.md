# Sequential-to-Attention: Sentiment Classification

A comparative benchmark of five deep learning architectures — **RNN, LSTM, GRU, Bi-LSTM, and a custom Transformer encoder (built from scratch)** — for binary sentiment classification on the IMDB Movie Reviews dataset. The project traces the architectural evolution of sequence modeling in NLP, from simple recurrence to multi-head self-attention, and empirically demonstrates *why* each successive architecture outperforms the last.

## Overview

| | |
|---|---|
| **Task** | Binary sentiment classification (positive / negative) |
| **Dataset** | IMDB Movie Reviews — 50,000 total (25,000 train / 25,000 test) |
| **Framework** | PyTorch |
| **Hardware** | Google Colab, T4 GPU |
| **Architectures** | Vanilla RNN → LSTM → GRU → Bi-LSTM → Custom Transformer Encoder |

## Motivation

Sentiment in a review often depends on words that are far apart in the sequence (e.g. *"I wanted to love this movie... but the ending ruined it"*). This makes IMDB reviews a strong testbed for showing how architectures differ in handling **long-range dependencies**:

- **Vanilla RNNs** suffer from vanishing gradients, losing signal from early tokens by the time they reach the end of a long sequence.
- **LSTM / GRU** use gating mechanisms to preserve long-term information.
- **Bi-LSTM** reads sequences in both directions, capturing context from both past and future tokens.
- **Transformers** replace recurrence entirely with multi-head self-attention, letting every token directly attend to every other token in parallel.

## Pipeline

1. **Data loading** — IMDB dataset loaded via Hugging Face `datasets` (`stanfordnlp/imdb`).
2. **Preprocessing** — lowercasing, HTML tag removal, non-alphabetic character stripping.
3. **Vocabulary & encoding** — custom word-to-index vocabulary (20,002 tokens, min frequency 2) built only from the training set; reviews padded/truncated to a fixed length of 200 tokens.
4. **Modeling** — five architectures implemented from scratch in PyTorch, each with its own embedding layer, sequence encoder, and classification head.
5. **Training** — a shared, reusable training/evaluation loop (`Adam` optimizer, cross-entropy loss, gradient clipping) used across all models for a fair comparison.
6. **Evaluation** — accuracy and F1-score tracked per epoch on the held-out test set.
7. **Comparison** — summary table and visualizations (accuracy bar chart, F1 bar chart, per-epoch training curves) generated to compare all five models.

## Architectures

| Model | Key Mechanism |
|---|---|
| **Vanilla RNN** | Single-direction recurrence with `tanh` activation; prone to vanishing gradients on long sequences |
| **LSTM** | Input/forget/output gates + cell state to preserve long-term dependencies |
| **GRU** | Simplified gating (reset/update) — fewer parameters, faster convergence |
| **Bi-LSTM** | Two LSTMs (forward + backward) concatenated for bidirectional context |
| **Custom Transformer Encoder** | Multi-head self-attention + sinusoidal positional encoding + padding-aware masking + mean-pooling classification head — trained from scratch, no pretrained weights |

## Results

| Model | Best Test Accuracy | Best Test F1 | Best Epoch | Final Train Accuracy | Parameters |
|---|---|---|---|---|---|
| Vanilla RNN | 50.82% | 0.6273 | 5 | 55.40% | 2,029,898 |
| LSTM | 68.16% | 0.6429 | 5 | 71.67% | 2,118,218 |
| GRU | 81.44% | 0.8172 | 5 | 82.49% | 2,088,778 |
| Bi-LSTM | 83.49% | 0.8349 | 10 | 96.98% | 2,236,234 |
| **Custom Transformer** | **84.14%** | **0.8442** | 4 | 90.62% | 2,825,474 |

**Key observations:**
- The vanilla RNN barely beats random guessing (~50%), directly illustrating the vanishing gradient problem on 200-token sequences.
- LSTM and GRU both substantially improve over the RNN through gated recurrence, with GRU converging faster in this run.
- Bi-LSTM improves further via bidirectional context but shows clear overfitting (96.98% train vs. 83.49% test accuracy).
- The custom Transformer encoder — trained entirely from scratch with **no pretrained weights** — achieves the **highest accuracy of all five models**, validating that multi-head self-attention captures contextual dependencies more effectively than sequential recurrence, even without pretraining.

## Repository Structure

```
├── Untitled3.ipynb          # Full notebook: data pipeline, all 5 models, training, comparison
├── accuracy_comparison.png  # Bar chart of best test accuracy per model
├── f1_comparison.png        # Bar chart of best test F1 per model
├── training_curves.png      # Test accuracy & train loss curves across epochs
└── README.md
```

## How to Run

1. Open the notebook in Google Colab.
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type).
3. Run all cells sequentially — the notebook installs dependencies, downloads IMDB via Hugging Face `datasets`, builds the vocabulary, trains all five models, and generates comparison plots.

**Dependencies:** `torch`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`

## Tech Stack

`Python` · `PyTorch` · `Hugging Face Datasets` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Google Colab (T4 GPU)`
