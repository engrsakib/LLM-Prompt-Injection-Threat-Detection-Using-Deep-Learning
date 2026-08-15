# LLM Prompt Injection Threat Detection Using Deep Learning

**Capstone Thesis Project** — Comparative evaluation of deep learning classifiers for detecting and categorizing prompt injection and jailbreak attacks against large language model (LLM) applications.

---

## Overview

This repository supports a thesis on **LLM security**, specifically **prompt injection detection**. The goal is to train, compare, and benchmark multiple deep learning models on a professional-grade threat intelligence dataset, moving beyond simple binary classification toward multi-class intent recognition, severity scoring, and technique-aware defenses.

The work is grounded in recent research on prompt injection taxonomies, LLM agent security, and adversarial NLP — including papers on ARGUS, AgentSentry, CLAWGUARD, MetaSecAlign, and systematic reviews of LLM defense mechanisms (see `papers/`).

---

## Dataset

**Source:** [neuralchemy/prompt-injection-Threat-Matrix](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix)

| Property | Details |
|----------|---------|
| **Samples** | 32,320 curated prompts |
| **Splits** | 80% train / 10% validation / 10% test |
| **Configs** | `binary` (benign vs malicious) and `multiclass` (7-way intent) |
| **License** | CC BY-NC 4.0 (research use) |

### Intent Classes (Multiclass)

| Label | Intent | Description |
|-------|--------|-------------|
| 0 | `benign` | Normal user input |
| 1 | `direct_injection` | Explicit instruction override |
| 2 | `system_extraction` | Attempts to leak system prompt |
| 3 | `role_hijack` | Persona or role manipulation |
| 4 | `obfuscation` | Encoded or disguised attacks |
| 5 | `tool_abuse` | Malicious tool/function calls |
| 6 | `indirect_injection` | Context-based injection |

### Key Features

- **Severity score** (1–10): attack impact and likelihood
- **Technique label**: attack method (e.g., encoding, role-play)
- **Surface label**: attack surface targeted
- **Ambiguity flag**: borderline / hard-to-classify samples

### Quick Load

```python
from datasets import load_dataset

binary_ds = load_dataset("neuralchemy/prompt-injection-Threat-Matrix", "binary")
multi_ds  = load_dataset("neuralchemy/prompt-injection-Threat-Matrix", "multiclass")
```

**Baseline model (pre-trained on this dataset):** [neuralchemy/distilbert-base-threat-matrix](https://huggingface.co/neuralchemy/distilbert-base-threat-matrix)

---

## Reference Papers

Local copies are stored in `papers/` (downloaded from [Google Drive](https://drive.google.com/drive/folders/1K9nPcdnSYI6iGPgNNwDmbew5pM4voqwz)).

| # | Topic |
|---|-------|
| 1 | Effectiveness of existing prompt injection detection methods |
| 4–6 | Real-world LLM compromise, AgentSentry, indirect injection |
| 7, 10 | Network intrusion / malicious traffic classification (MLP, decision-making) |
| 8–9, 11 | ARGUS — defending LLM agents against prompt injection |
| 12 | CLAWGUARD — runtime security for LLM agents |
| 13–15 | Multi-model hybrid defense, threat taxonomy, prompt injection attacks |
| 16 | Federated learning poisoning detection |
| 17–18 | MetaSecAlign — secure foundation LLM against prompt injection |
| 19 | Systematic literature review on LLM prompt injection defenses |
| 20–21 | Retrieval barrier, TaintP2X taint-style injection detection |

---

## 12 Deep Learning Models to Implement

The following models are selected for **text classification** on the Threat Matrix dataset. They span transformer encoders, lightweight distillation variants, CNN/RNN hybrids, and security-domain pre-trained models — suitable for binary and multiclass experiments within a capstone timeline.

| # | Model | Type | Why Implement |
|---|-------|------|---------------|
| 1 | **DistilBERT** | Distilled Transformer | Fast baseline; official model already exists on this dataset |
| 2 | **BERT-base-uncased** | Transformer Encoder | Strong general-purpose text classifier; widely cited baseline |
| 3 | **RoBERTa-base** | Transformer Encoder | Improved pre-training over BERT; strong on adversarial text |
| 4 | **DeBERTa-v3-base** | Transformer Encoder | Disentangled attention; top performance on GLUE-style tasks |
| 5 | **SecureBERT** | Domain-specific BERT | Pre-trained on cybersecurity corpus; relevant for threat detection |
| 6 | **ALBERT-base-v2** | Lightweight Transformer | Parameter-efficient; good for ablation vs full BERT |
| 7 | **ELECTRA-small** | Discriminator-pretrained | Efficient alternative with competitive accuracy |
| 8 | **TextCNN (Kim CNN)** | Convolutional | Classic short-text classifier; fast to train, good comparison point |
| 9 | **BiLSTM + Attention** | Recurrent + Attention | Captures sequential patterns in obfuscated / encoded attacks |
| 10 | **CNN-BiLSTM Hybrid** | CNN + RNN | Combines local n-gram features with long-range dependencies |
| 11 | **DistilRoBERTa** | Distilled Transformer | Balance of RoBERTa accuracy and DistilBERT speed |
| 12 | **XLNet-base** | Permutation LM | Handles bidirectional context; useful for indirect injection |

### Suggested Experiment Matrix

| Task | Models to prioritize |
|------|---------------------|
| **Binary detection** (benign vs malicious) | DistilBERT, BERT, SecureBERT, TextCNN, DistilRoBERTa |
| **7-class intent classification** | RoBERTa, DeBERTa, BiLSTM+Attention, CNN-BiLSTM |
| **Severity regression / scoring** | DeBERTa, RoBERTa, XLNet |
| **Low-resource / fast inference** | DistilBERT, ALBERT, ELECTRA-small, TextCNN |

### Evaluation Metrics

- Accuracy, Precision, Recall, F1 (macro & weighted)
- ROC-AUC (binary config)
- Confusion matrix per intent class
- Severity MAE (if predicting severity as regression)
- Inference latency (ms/sample) for deployment comparison

---

## Project Structure (planned)

```
model/
??? papers/              # Reference PDFs
??? configs/             # Training hyperparameters
??? src/                 # Training, evaluation, and inference code
??? scripts/             # Data download and utility scripts
??? outputs/             # Checkpoints, logs, plots
??? README.md
```

---

## Getting Started

```bash
pip install torch transformers datasets scikit-learn pandas
```

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ds = load_dataset("neuralchemy/prompt-injection-Threat-Matrix", "multiclass")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=7
)
```

---

## Citation

If you use the dataset, cite:

```bibtex
@dataset{jajoo2026threatmatrix,
  author    = {Sanskar Jajoo},
  title     = {Neuralchemy Prompt Injection Threat Matrix},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix}
}
```

---

## License

Research use only. Dataset license: **CC BY-NC 4.0**. Commercial use requires permission from the dataset author.
