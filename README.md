<div align="center">

# LLM Prompt Injection Threat Detection Using Deep Learning

**Comparative evaluation of deep learning classifiers for detecting and categorizing prompt injection and jailbreak attacks against LLM-integrated applications.**

<br/>

**Author:** [Md. Nazmus Sakib](https://engrskib.com) ù **Co-Author:** Kazi Omar Faruq

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Dataset License: CC BY-NC 4.0](https://img.shields.io/badge/Dataset-CC%20BY--NC%204.0-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/)

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets%20%26%20Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

[![Thesis](https://img.shields.io/badge/Project-Capstone%20Thesis-blue?style=flat-square)]()
[![Institution](https://img.shields.io/badge/Institution-DIU-green?style=flat-square)]()
[![Domain](https://img.shields.io/badge/Domain-LLM%20Security-red?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=flat-square)]()

</div>

---

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Model Registry](#model-registry)
- [Reference Papers](#reference-papers)
- [Experiment Design](#experiment-design)
- [Data Engineering (Phase 1)](#data-engineering-phase-1)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository supports a **capstone thesis** on LLM security, focused on **prompt injection detection**. The objective is to train, compare, and benchmark multiple deep learning architectures on a professional-grade threat intelligence dataset ù extending beyond binary classification toward multi-class intent recognition, severity scoring, and technique-aware defenses.

The research is grounded in recent work on prompt injection taxonomies, LLM agent security, and adversarial NLP, including **ARGUS**, **AgentSentry**, **CLAWGUARD**, **MetaSecAlign**, and systematic reviews of LLM defense mechanisms. Full PDFs are available in [`papers/`](papers/).

---

## Technology Stack

### Core Framework

| Technology | Version | Role | Badge |
|------------|---------|------|-------|
| **Python** | `3.11+` | Primary language | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **PyTorch** | `2.5+` | Deep learning backend | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| **Transformers** | `4.46+` | Pre-trained model loading & fine-tuning | ![Transformers](https://img.shields.io/badge/-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black) |
| **Datasets** | `3.1+` | Hugging Face dataset pipeline | ![Datasets](https://img.shields.io/badge/-Datasets-FFD21E?style=flat-square&logo=huggingface&logoColor=black) |
| **Accelerate** | `1.1+` | Distributed / mixed-precision training | ![Accelerate](https://img.shields.io/badge/-Accelerate-FFD21E?style=flat-square&logo=huggingface&logoColor=black) |
| **Tokenizers** | `0.20+` | Fast text tokenization | ![Tokenizers](https://img.shields.io/badge/-Tokenizers-FFD21E?style=flat-square&logo=huggingface&logoColor=black) |

### ML & Evaluation

| Technology | Version | Role | Badge |
|------------|---------|------|-------|
| **scikit-learn** | `1.5+` | Metrics, splits, baselines | ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| **pandas** | `2.2+` | Data analysis & reporting | ![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white) |
| **NumPy** | `2.0+` | Numerical computation | ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Matplotlib** | `3.9+` | Result visualization | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) |
| **Seaborn** | `0.13+` | Confusion matrices & plots | ![Seaborn](https://img.shields.io/badge/-Seaborn-444876?style=flat-square) |

### DevOps & Environment

| Technology | Version | Role | Badge |
|------------|---------|------|-------|
| **CUDA** | `12.1+` | GPU acceleration (optional) | ![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white) |
| **Git** | `2.40+` | Version control | ![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white) |
| **Jupyter** | `1.1+` | Experiment notebooks | ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) |

### Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><b>requirements.txt</b> (recommended versions)</summary>

```txt
torch>=2.5.0
transformers>=4.46.0
datasets>=3.1.0
accelerate>=1.1.0
tokenizers>=0.20.0
scikit-learn>=1.5.0
pandas>=2.2.0
numpy>=2.0.0
matplotlib>=3.9.0
seaborn>=0.13.0
evaluate>=0.4.0
tqdm>=4.66.0
```

</details>

---

## Dataset

<div align="center">

<a href="https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix">
  <img src="https://huggingface.co/datasets/huggingface/brand/resolve/main/hf_dataset_with_text.svg" alt="Hugging Face Dataset" width="320"/>
</a>

<br/><br/>

[![Hugging Face Dataset](https://img.shields.io/badge/??%20Hugging%20Face-prompt--injection--Threat--Matrix-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix)
[![Dataset Size](https://img.shields.io/badge/Samples-32%2C320-blue?style=for-the-badge)](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=for-the-badge)](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix)

</div>

| Property | Value |
|----------|-------|
| **Repository** | [`neuralchemy/prompt-injection-Threat-Matrix`](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix) |
| **Samples** | 32,320 curated prompts |
| **Splits** | 80% train ù 10% validation ù 10% test |
| **Configs** | `binary` ù `multiclass` (default) |
| **Format** | Parquet |
| **Language** | English |
| **License** | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

### Intent Classes (Multiclass)

| Label | Intent | Description |
|:-----:|--------|-------------|
| `0` | `benign` | Normal user input |
| `1` | `direct_injection` | Explicit instruction override |
| `2` | `system_extraction` | Attempts to leak system prompt |
| `3` | `role_hijack` | Persona or role manipulation |
| `4` | `obfuscation` | Encoded or disguised attacks |
| `5` | `tool_abuse` | Malicious tool/function calls |
| `6` | `indirect_injection` | Context-based injection |

### Schema Highlights

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Input prompt |
| `label` / `intent_label` | int | Classification target |
| `severity` | int | Threat severity (1ù10) |
| `technique` | string | Attack technique category |
| `surface` | string | Targeted attack surface |
| `ambiguity` | bool | Borderline / hard-to-classify flag |

### Quick Load

```python
from datasets import load_dataset

binary_ds = load_dataset("neuralchemy/prompt-injection-Threat-Matrix", "binary")
multi_ds  = load_dataset("neuralchemy/prompt-injection-Threat-Matrix", "multiclass")
```

**Official baseline (pre-trained):** [`neuralchemy/distilbert-base-threat-matrix`](https://huggingface.co/neuralchemy/distilbert-base-threat-matrix)

---

## Model Registry

Twelve deep learning models selected for comparative evaluation. Each entry includes architecture type, Hugging Face checkpoint (where applicable), parameter scale, and thesis role.

| # | Model | Architecture | Hugging Face Checkpoint | Params | Thesis Role |
|:-:|-------|--------------|-------------------------|:------:|-------------|
| 1 | **DistilBERT** | Distilled Transformer | [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) | 66M | Primary baseline; dataset-aligned reference |
| 2 | **BERT** | Transformer Encoder | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | 110M | Canonical text classification baseline |
| 3 | **RoBERTa** | Transformer Encoder | [`roberta-base`](https://huggingface.co/roberta-base) | 125M | Robust pre-training; adversarial text |
| 4 | **DeBERTa-v3** | Disentangled Attention | [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) | 184M | State-of-the-art encoder candidate |
| 5 | **SecureBERT** | Cybersecurity BERT | [`ehsanaul/securebert-base`](https://huggingface.co/ehsanaul/securebert-base) | 110M | Domain-specific security embeddings |
| 6 | **ALBERT** | Lightweight Transformer | [`albert-base-v2`](https://huggingface.co/albert-base-v2) | 12M | Parameter-efficient comparison |
| 7 | **ELECTRA** | Discriminator-pretrained | [`google/electra-small-discriminator`](https://huggingface.co/google/electra-small-discriminator) | 14M | Fast, compute-efficient encoder |
| 8 | **TextCNN** | Convolutional (Kim) | *Custom implementation* | ~1M | Classical DL baseline |
| 9 | **BiLSTM + Attention** | Recurrent + Attention | *Custom implementation* | ~2M | Sequential / obfuscation patterns |
| 10 | **CNN-BiLSTM** | Hybrid CNN-RNN | *Custom implementation* | ~3M | Local + long-range feature fusion |
| 11 | **DistilRoBERTa** | Distilled Transformer | [`distilroberta-base`](https://huggingface.co/distilroberta-base) | 82M | Accuracyùspeed trade-off |
| 12 | **XLNet** | Permutation LM | [`xlnet-base-cased`](https://huggingface.co/xlnet-base-cased) | 110M | Bidirectional context modeling |

### Model Icons (Architecture Family)

<p align="center">

<img src="https://img.shields.io/badge/??%20DistilBERT-distilbert--base--uncased-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="DistilBERT"/>
<img src="https://img.shields.io/badge/??%20BERT-bert--base--uncased-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="BERT"/>
<img src="https://img.shields.io/badge/??%20RoBERTa-roberta--base-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="RoBERTa"/>
<img src="https://img.shields.io/badge/??%20DeBERTa-deberta--v3--base-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="DeBERTa"/>

<br/>

<img src="https://img.shields.io/badge/??%20SecureBERT-securebert--base-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="SecureBERT"/>
<img src="https://img.shields.io/badge/??%20ALBERT-albert--base--v2-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="ALBERT"/>
<img src="https://img.shields.io/badge/??%20ELECTRA-electra--small-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="ELECTRA"/>
<img src="https://img.shields.io/badge/??%20TextCNN-Custom-444876?style=flat-square" alt="TextCNN"/>

<br/>

<img src="https://img.shields.io/badge/??%20BiLSTM-Custom-444876?style=flat-square" alt="BiLSTM"/>
<img src="https://img.shields.io/badge/??%20CNN--BiLSTM-Custom-444876?style=flat-square" alt="CNN-BiLSTM"/>
<img src="https://img.shields.io/badge/??%20DistilRoBERTa-distilroberta--base-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="DistilRoBERTa"/>
<img src="https://img.shields.io/badge/??%20XLNet-xlnet--base--cased-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="XLNet"/>

</p>

> **Legend:** ?? = Hugging Face pre-trained checkpoint ù ?? = Custom PyTorch implementation

---

## Reference Papers

Local copies in [`papers/`](papers/) ù [Google Drive collection](https://drive.google.com/drive/folders/1K9nPcdnSYI6iGPgNNwDmbew5pM4voqwz)

| Paper | Focus Area |
|-------|------------|
| `1.pdf` | Effectiveness of existing detection methods |
| `4.pdf` ù `6.pdf` | Real-world LLM compromise, AgentSentry, indirect injection |
| `7.pdf`, `10.pdf` | Network intrusion & malicious traffic classification |
| `8.pdf` ù `9.pdf`, `11.pdf` | ARGUS ù LLM agent defense |
| `12.pdf` | CLAWGUARD ù runtime agent security |
| `13.pdf` ù `15.pdf` | Hybrid defense, threat taxonomy, injection attacks |
| `16.pdf` | Federated learning poisoning detection |
| `17.pdf` ù `18.pdf` | MetaSecAlign ù secure foundation LLM |
| `19.pdf` | Systematic literature review on LLM defenses |
| `20.pdf` ù `21.pdf` | Retrieval barrier, TaintP2X injection detection |

---

## Experiment Design

### TaskùModel Mapping

| Experiment | Recommended Models | Primary Metrics |
|------------|-------------------|-----------------|
| **Binary detection** | DistilBERT, BERT, SecureBERT, TextCNN, DistilRoBERTa | Accuracy, F1, ROC-AUC |
| **7-class intent classification** | RoBERTa, DeBERTa, BiLSTM+Attention, CNN-BiLSTM | Macro-F1, Confusion Matrix |
| **Severity scoring** | DeBERTa, RoBERTa, XLNet | MAE, RMSE |
| **Low-latency deployment** | DistilBERT, ALBERT, ELECTRA, TextCNN | Latency (ms/sample), F1 |

### Evaluation Protocol

```
Accuracy ù Precision ù Recall ù F1 (macro & weighted)
ROC-AUC (binary) ù Confusion matrix (multiclass)
Severity MAE ù Inference latency ù Model size (MB)
```

---

## Data Engineering (Phase 1)

Professional-grade preprocessing pipeline for IEEE reproducibility.

### Run locally

```bash
pip install -r requirements.txt
python -m src.data.prepare --config configs/data.yaml
```

### Run with DVC

```bash
dvc repro prepare
```

### What it does

| Step | Module | Description |
|------|--------|-------------|
| Ingestion | `src/data/ingest.py` | Hugging Face download + snapshot ID |
| Cleaning | `src/data/cleaning.py` | Unicode NFKC, control chars, zero-width removal |
| Dedup | `src/data/dedup.py` | Exact (SHA-256) + near (MinHash LSH) |
| Validation | `src/data/validation.py` | intent, binary_label, severity, technique checks |
| Reporting | `src/data/reporting.py` | Class distribution JSON + Markdown |
| Export | `src/data/pipeline.py` | Parquet to `data/processed/` |

### Output artifacts

```
data/processed/
??? train.parquet
??? validation.parquet
??? test.parquet
??? metadata.json
??? reports/
    ??? class_distribution.json
    ??? class_distribution.md
    ??? validation_report.json
    ??? dedup_report.json
```

Full documentation: [`docs/DATA_CARD.md`](docs/DATA_CARD.md)

---

## Project Structure

```
model/
??? papers/                  # Reference literature (PDF)
??? configs/                 # Experiment & data pipeline configs
??? docs/                    # DATA_CARD and research docs
??? src/
?   ??? data/                # Ingest, clean, dedup, validate, export
?   ??? models/              # Model definitions (transformers + custom)
?   ??? training/            # Training loops & checkpoints
?   ??? eval/                # Metrics, plots, benchmark reports
??? scripts/                 # Download & utility scripts
??? data/                    # Raw + processed artifacts (gitignored)
??? outputs/                 # Checkpoints, logs, result tables
??? dvc.yaml                 # DVC pipeline definition
??? requirements.txt
??? README.md
```

---

## Getting Started

### 1. Clone & set up environment

```bash
git clone <repository-url>
cd model
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Load dataset & baseline model

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ds = load_dataset("neuralchemy/prompt-injection-Threat-Matrix", "multiclass")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=7
)
```

### 3. Run training (planned)

```bash
python -m src.training.train_cli --model distilbert-base-uncased --config configs/default.yaml
```

---

## Citation

If you use the **Threat Matrix** dataset in this thesis, please cite:

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

See [`LICENSE`](LICENSE) for the full MIT license text.

| Component | License | Holder |
|-----------|---------|--------|
| **This repository** | [MIT License](LICENSE) | Md. Nazmus Sakib ù Kazi Omar Faruq |
| **Threat Matrix dataset** | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Sanskar Jajoo / Neuralchemy |
| **Pre-trained models** | Respective Hugging Face model licenses | Model authors |

### MIT License (This Repository)

You **may**:

- Use, copy, modify, merge, publish, distribute, sublicense, and sell copies of this software
- Use the software for personal, academic, research, and commercial projects

You **must**:

- Include the copyright notice and full MIT license text in all copies or substantial portions

The software is provided **"AS IS"**, without warranty of any kind. The authors are not liable for any damages arising from use of this software.

**Copyright ù 2026** Md. Nazmus Sakib ([engrskib.com](https://engrskib.com)) and Kazi Omar Faruq.

### CC BY-NC 4.0 (External Dataset Only)

The [Threat Matrix dataset](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix) is **not** covered by MIT. It uses **Creative Commons Attribution-NonCommercial 4.0**.

You **may** (non-commercial only):

- **Share** ù copy and redistribute the dataset in any medium or format
- **Adapt** ù remix, transform, and build upon the dataset

You **must**:

- **Attribute** ù give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** ù you may not use the dataset for commercial purposes without permission from the dataset author

You **may not**:

- Use the dataset commercially without explicit permission from Neuralchemy / Sanskar Jajoo
- Apply legal terms or technological measures that legally restrict others from doing anything the license permits

**Full CC BY-NC 4.0 legal code:** https://creativecommons.org/licenses/by-nc/4.0/legalcode

---

<div align="center">

**Daffodil International University ù 7th Capstone Design ù 2026**

**Md. Nazmus Sakib** ([engrskib.com](https://engrskib.com)) ù **Kazi Omar Faruq**

*LLM Security ù Prompt Injection Detection ù Deep Learning*

</div>

---

## Quickstart: Docker (CPU-only)

Build:

```
docker build -t prompt-injection-model -f docker/Dockerfile .
```

Run (mount outputs for checkpoints):

```
docker run --rm -v %cd%/outputs:/app/outputs prompt-injection-model --config configs/default.yaml
```

Use small batch sizes when training on CPU.

---

## Quickstart: Kaggle

1. Upload processed `data/processed/` to Kaggle as a dataset or use Hugging Face streaming.
2. In a Kaggle notebook cell:

```bash
!git clone <repo-url> repo && cd repo
!pip install -r requirements.txt
!python -m src.training.train_cli --config configs/kaggle_debug.yaml
```

A starter notebook is provided at `notebooks/kaggle_run.md`.
