<div align="center">

# LLM Prompt Injection Threat Detection Using Deep Learning

**Comparative evaluation of 16 deep-learning classifiers for detecting and categorizing prompt injection and jailbreak attacks against LLM-integrated applications.**

<br/>

**Author:** [Md. Nazmus Sakib](https://engrskib.com) · **Co-Author:** Kazi Omar Faruq

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
[![Status](https://img.shields.io/badge/Status-Pipeline%20implemented%20%7C%20training%20pending-orange?style=flat-square)]()

</div>

---

## Table of Contents

- [Overview](#overview)
- [Status at a glance](#status-at-a-glance)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Data Engineering (Phases 1-3)](#data-engineering-phases-1-3)
- [Phase 3.5: Leakage-aware re-split](#phase-35-leakage-aware-re-split)
- [Model Registry (16 backbones)](#model-registry-16-backbones)
- [Training](#training)
- [Evaluation, tables and figures](#evaluation-tables-and-figures)
- [Kaggle workflow](#kaggle-workflow)
- [Artifact map](#artifact-map)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Troubleshooting](#troubleshooting)
- [Limitations and honesty notes](#limitations-and-honesty-notes)
- [Reference Papers](#reference-papers)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository supports a **capstone thesis** on LLM security, focused on **prompt injection detection**. The objective is to train, compare and benchmark multiple deep learning architectures on the Threat Matrix dataset, extending beyond binary classification toward 7-class intent recognition and technique-aware analysis.

The research is grounded in recent work on prompt injection taxonomies, LLM agent security and adversarial NLP, including **ARGUS**, **AgentSentry**, **CLAWGUARD**, **MetaSecAlign** and systematic reviews of LLM defence mechanisms. Full PDFs are available in [`papers/`](papers/).

---

## Status at a glance

| Area | State | Evidence |
|------|-------|----------|
| Data Phases 1-3 (clean, dedup, enrich, augment, audit) | implemented, executed | `data/processed/` + reports |
| **Phase 3.5** group-aware leakage-free re-split | implemented, **executed** (`v1_leakfree`, audit passed) | `data/processed/splits/v1_leakfree/manifest.json` |
| 16-model registry + shared classifier interface | implemented | `src/models/`, `docs/MODEL_REGISTRY.md` |
| Trainer (resume, early stop, history), per-model configs | implemented | `src/training/` , `configs/models/*.yaml` |
| Metrics, per-example predictions, tables, 17 figure types | implemented | `src/eval/` |
| Resumable Kaggle sweep + thin notebook | implemented, **not yet executed on Kaggle** | `src/training/sweep.py`, `notebooks/kaggle_train.ipynb` |
| CPU smoke runs (TextCNN, DistilBERT, 300 train rows) | **executed locally** | `outputs/smoke/` (not publication results) |
| Full training of the 16 models (3 seeds) | **not executed** (requires GPU; see Kaggle workflow) | - |
| Llama-3.1-8B | blocked (gated checkpoint; needs approved `HF_TOKEN`) | registry |

Unit tests: `pytest` -> 42 passed (tiny synthetic fixtures, no network).

---

## Technology Stack

| Technology | Version used locally | Role |
|------------|----------------------|------|
| **Python** | 3.13.7 (CI: 3.11) | Primary language |
| **PyTorch** | 2.13 (CPU build locally) | Deep learning backend |
| **Transformers** | 5.22 (>= 4.51 required for Qwen3) | Pre-trained model loading & fine-tuning |
| **PEFT** | 0.21 | LoRA / QLoRA for 7B-8B decoders |
| **Datasets / Hub** | 5.1 / 2.4 | Dataset snapshot + checkpoint metadata |
| **scikit-learn** | 1.5+ | Metrics |
| **pandas / pyarrow** | 2.2+ | Data processing, Parquet |
| **datasketch** | 1.6+ | MinHash LSH near-duplicate detection |
| **Matplotlib** | 3.9+ | Figures (Okabe-Ito colour-blind-safe palette) |
| **DVC / MLflow** | 3.x / 2.x | Pipeline + experiment tracking |

Install:

```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # pytest, ruff, mypy
# Kaggle sessions: pip install -r requirements-kaggle.txt (pinned, no torch re-pin)
```

---

## Dataset

| Property | Value |
|----------|-------|
| **Upstream** | [`neuralchemy/prompt-injection-Threat-Matrix`](https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix), config `multiclass` |
| **Local snapshot** | `data/raw/{train,validation,test}.jsonl` + `snapshot_metadata.json` (id `tm-4e7fede7adba`) - **the pipeline never re-downloads** |
| **Raw rows (measured)** | 25,856 / 3,232 / 3,232 = 32,320 |
| **Licence** | CC BY-NC 4.0 **according to the local snapshot metadata**; the upstream page could not be re-verified at build time |
| **Language** | English |

### Intent classes (`intent_label`; logits column order everywhere)

| Label | Intent | Description |
|:-----:|--------|-------------|
| `0` | `benign` | Normal user input |
| `1` | `direct_injection` | Explicit instruction override |
| `2` | `system_extraction` | Attempts to leak system prompt |
| `3` | `role_hijack` | Persona or role manipulation |
| `4` | `obfuscation` | Encoded or disguised attacks |
| `5` | `tool_abuse` | Malicious tool/function calls |
| `6` | `indirect_injection` | Context-based injection |

### Schema (as measured on the local snapshot)

| Field | Type | Measured facts |
|-------|------|----------------|
| `text` | string | prompt |
| `label` / `intent_label` | int | identical for every row |
| `binary_label` | int | equals `intent_label != 0` for every row |
| `severity` | int | values **{1, 2, 3}** only (not 1-10 as documented upstream) |
| `technique` | string | attack technique |
| `surface` | string | constant `user_input` - carries no information |
| `source` | string | `hackaprompt`, `core`, `synthetic_threat`, `synthetic_benign`, `synthetic_short` |
| `ambiguity` | bool | borderline flag |

Synthetic share (`source` starts with `synthetic`): 37.9% / 37.5% / 38.6% per raw split. Synthetic rows are **kept**; results are additionally reported for synthetic vs non-synthetic subsets (`src/eval/tables.py::subgroup_table`).

Full data card: [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

---

## Data Engineering (Phases 1-3)

```bash
python -m src.data.prepare --config configs/data.yaml --phase all   # or: dvc repro prepare enrich finalize
```

| Phase | Modules | Outputs (`data/processed/`) |
|-------|---------|-----------------------------|
| 1 Ingest / clean / dedup / validate | `ingest.py`, `cleaning.py`, `dedup.py`, `validation.py` | `train|validation|test.parquet`, `metadata.json`, `reports/` |
| 2 Obfuscation features, balancing, augmentation, subsets | `obfuscation.py`, `analysis.py`, `balancing.py`, `augment.py` | `train_balanced.parquet`, `train_augmented.parquet`, `subsets/ambiguity_*` |
| 3 Technique subsets, adversarial paraphrase, leakage audit, Kaggle zip | `technique_subsets.py`, `adversarial_paraphrase.py`, `leakage.py`, `kaggle_package.py` | `train_adversarial.parquet`, `kaggle_package.zip`, `reports/leakage_report.json` |

**Important:** the Phase 3 audit of these *upstream-split* artifacts reports cross-split overlap (`reports/leakage_report.json`, `passed: false`: 127 exact + 125 near pairs after processing; on the raw files: 14 / 20 / 4 exact duplicates train-val / train-test / val-test and 198 benign `(example N)` template families spanning splits). These historical files are kept for traceability but **must not be used for model evaluation** - use Phase 3.5.

---

## Phase 3.5: Leakage-aware re-split

```bash
python -m src.data.phase35 --config configs/data.yaml      # ~90 s on CPU; or: dvc repro resplit
```

Policy (**group-aware re-split**, chosen with the supervisor's approval): all raw rows are pooled, grouped into *families*, and whole families are assigned to one split with a stratified greedy placement (~80/10/10) that uses the upstream split as tie-break.

| Step | Rule |
|------|------|
| Family links | exact normalised fingerprint **or** template stem (`(example N)` suffix removed) **or** MinHash-LSH word-3-gram Jaccard >= 0.85 (configurable). Rows without shingles are never linked. A `source` category alone is **never** a family key. |
| Within-split dedup | exact + near (same threshold) inside each split only |
| Audit | full (not sampled) raw / normalised / near / family checks across all split pairs; **the run fails unless it passes** |
| Train variants | `train_balanced`, `train_augmented`, `train_adversarial` regenerated **from the new train split only**; each is audited against the new validation/test (all passed, 0 overlaps) |
| Manifest | `manifest.json` with `frozen: true`, SHA-256 of every artifact, before/after counts, moved-row matrix, dropped-row log, thresholds, licence caveat |

Measured result (`data/processed/splits/v1_leakfree/manifest.json`, 2026-11-07):

| | train | validation | test |
|--|------:|-----------:|-----:|
| rows after group split (before dedup) | 25,853 | 3,232 | 3,235 |
| exact / near removed within split | 327 / 3,433 | 51 / 401 | 32 / 401 |
| **final rows** | **22,093** | **2,780** | **2,802** |
| families | 18,826 | 2,386 | 2,385 |
| synthetic rows | 7,388 (33.4%) | 909 (32.7%) | 910 (32.5%) |

32,320 rows -> 23,597 families (22,390 singletons, largest 664); 3,508 rows changed split; 4,645 rows dropped as within-split duplicates; cross-split raw/exact/near/family overlaps after re-split: **0 / 0 / 0 / 0**. All 7 classes present in every split. Train variants: balanced 22,984, augmented 26,116, adversarial 46,801 rows.

> Passing these heuristic checks does **not** prove zero statistical leakage (paraphrases below the Jaccard threshold, shared templates with different wording, etc.). Raw-file numbers (14/20/4) and processed-file numbers (127/125) are different measurements and must never be combined.

Every training / evaluation entrypoint loads `manifest.json`, refuses non-frozen or failed-audit splits and re-verifies artifact hashes (`src/data/phase35.py::load_frozen_manifest`).

---

## Model Registry (16 backbones)

Full table with checkpoints, revisions, licences, parameter counts and evidence: [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md) (regenerate with `python -m src.models.registry`; `--check --online` verifies Hub metadata without downloading weights).

| # | `model_id` | Model | Checkpoint | Tier | Status (2026-11-07) |
|:-:|-----------|-------|------------|------|---------------------|
| 1 | `distilbert` | DistilBERT | `distilbert-base-uncased` | cpu_ok | **smoke-tested** |
| 2 | `bert` | BERT | `bert-base-uncased` | gpu_small | implemented |
| 3 | `roberta` | RoBERTa | `roberta-base` | gpu_small | implemented |
| 4 | `deberta_v3` | DeBERTa-v3 | `microsoft/deberta-v3-base` | gpu_small | implemented |
| 5 | `securebert` | SecureBERT | `ehsanaghaei/SecureBERT` @ `3a47918d` (**corrected**: `ehsanaul/securebert-base` returns 401) | gpu_small | implemented |
| 6 | `albert` | ALBERT | `albert-base-v2` | cpu_ok | implemented |
| 7 | `electra_small` | ELECTRA-small | `google/electra-small-discriminator` | cpu_ok | implemented |
| 8 | `textcnn` | TextCNN | custom, train-only vocab | cpu_ok | **smoke-tested** |
| 9 | `bilstm_attention` | BiLSTM + Attention | custom, train-only vocab | cpu_ok | implemented |
| 10 | `cnn_bilstm` | CNN-BiLSTM | custom, train-only vocab | cpu_ok | implemented |
| 11 | `distilroberta` | DistilRoBERTa | `distilroberta-base` | cpu_ok | implemented |
| 12 | `xlnet` | XLNet | `xlnet-base-cased` | gpu_small | implemented |
| 13 | `modernbert_large` | ModernBERT-large (experimental) | `answerdotai/ModernBERT-large` @ `45bb4654` | gpu_medium | implemented |
| 14 | `mistral_7b` | Mistral-7B v0.3 base + QLoRA (experimental) | `mistralai/Mistral-7B-v0.3` @ `caa1feb0` | gpu_large_qlora | implemented |
| 15 | `qwen3_8b` | Qwen3-8B + QLoRA (experimental; post-trained release, base is `Qwen3-8B-Base`) | `Qwen/Qwen3-8B` @ `b968826d` | gpu_large_qlora | implemented |
| 16 | `llama31_8b` | Llama-3.1-8B + QLoRA (experimental) | `meta-llama/Llama-3.1-8B` @ `d04e592b` | gpu_large_qlora | **blocked** (gated) |

"implemented" means the code path exists and unit tests cover the interface; it does **not** mean the model has been trained. Decoder models were verified only with a random tiny Llama config (padding invariance + adapter/head save-reload); no 7B/8B weights were loaded locally.

Shared interface (`src/models/base.py`): every model consumes `input_ids`/`attention_mask`, returns 7 logits in `LABEL_ORDER`, saves/loads from a directory; custom models fit their vocabulary on the training split only; decoders use LoRA (`peft`, `TaskType.SEQ_CLS`) with the `score` head trained and saved, right padding and last-non-pad-token pooling, 4-bit QLoRA when `bitsandbytes` + CUDA exist (fallback recorded in `load_info`).

Excluded by design: GPT-4 Attack Validator, Lakera Guard, MetaSecAlign (external / paid / generative; no data leaves the machine).

---

## Training

One job = one model x one seed. Validation is used **only** for early stopping (macro-F1) and binary-threshold selection; test predictions are produced once from the best validation checkpoint.

```bash
# CPU smoke test (300 stratified train rows, 100 val/test rows, 1 epoch, max_length 64) -> outputs/smoke/
python -m src.training.train_cli --model textcnn    --seed 42 --smoke
python -m src.training.train_cli --model distilbert --seed 42 --smoke

# Full run (GPU) -> outputs/runs/<model>/seed_<seed>/
python -m src.training.train_cli --model roberta --seed 42 --config configs/experiment.yaml
python -m src.training.train_cli --model roberta --seed 42 --resume          # continue after interruption
python -m src.training.train_cli --model roberta --seed 42 --train-file train_adversarial.parquet   # ablation variant

# Resumable sweep (models x seeds, manifest in outputs/sweep/manifest.json)
python -m src.training.sweep --config configs/experiment.yaml --models textcnn distilbert --seeds 42 43 44
```

Configuration precedence: registry defaults -> `configs/models/<model_id>.yaml` -> `configs/experiment.yaml: training_overrides` -> CLI flags (`--epochs --batch-size --learning-rate --max-length --grad-accum --precision`). Common max-length policy: 256 tokens; truncation rates per split are logged in `run_summary.json`. Default seeds: 42, 43, 44.

Each run writes `config.json`, `history.jsonl` (step/epoch loss, lr, val metrics), `checkpoint/` (best), `last/` (resume state incl. optimizer/scheduler/RNG; deleted after success unless `--keep-last`), `threshold.json`, `metrics_{validation,test}.json`, `predictions_{validation,test}.parquet`, `run_summary.json` (wall time, hardware, parameter counts, truncation rates).

---

## Evaluation, tables and figures

`src/eval/evaluate.py` validates probabilities (shape N x 7, finite, rows sum to 1) and computes: accuracy, balanced accuracy, macro/micro/weighted precision/recall/F1, per-class P/R/F1/support, MCC, one-vs-rest AUROC and AP (per class + macro + weighted), raw and row-normalised confusion matrices, per-class TP/TN/FP/FN, calibration (top-label ECE, multiclass Brier, NLL, reliability bins), and binary attack metrics with `p_attack = sum(p[1:7])` and a threshold chosen on validation only (precision, recall, F1, specificity, FPR, FNR, MCC, AUROC, AP, TP/TN/FP/FN). Metrics that cannot be computed (e.g. AUROC for a class absent from the gold labels) are `null` with a reason under `unavailable`.

```bash
python -m src.eval.tables --runs outputs/runs --out outputs/tables            # runs_test.csv, seed_summary_test.csv (mean/std over seeds),
                                                                              # per_class_test.csv, subgroups_test.csv (source/technique/ambiguity/severity/synthetic),
                                                                              # bootstrap_ci_test.json (group-aware percentile CI over families)
python -m src.eval.plots  --runs outputs/runs --out outputs/figures            # 17 figure types, PNG 300 DPI + PDF + SVG, skipped.json for missing prerequisites
```

Seed variance (mean/std over seeds) and sampling uncertainty (family-level bootstrap CI of one run) are always reported separately. Figures: confusion counts, normalised confusion, binary TP/TN/FP/FN, per-class OvR counts, dataset distribution (labelled DATASET), loss curves, val macro-F1 curves, cross-model bars (+ separate bootstrap-CI figure), model x class F1 heatmap, OvR ROC + macro/binary ROC, PR curves with prevalence baselines, AUROC/AP comparison, subset performance with n, train-variant ablation, reliability diagram, efficiency vs F1, confidence/error histograms.

---

## Kaggle workflow

See [`notebooks/kaggle_run.md`](notebooks/kaggle_run.md) and the thin notebook [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) (cells only call `src/` modules).

1. Build the split package locally: `bash scripts/package_kaggle.sh` -> `data/processed/splits/v1_leakfree/kaggle_package_v1_leakfree.zip`.
2. Upload it **manually** as a *private* Kaggle dataset (only if you have verified permission; licence CC BY-NC 4.0 per snapshot metadata). Nothing in the code logs in, publishes or submits.
3. Add the repo (dataset or `git clone`), optionally an `HF_TOKEN` **Kaggle Secret** for gated checkpoints; tokens are only read from `os.environ`.
4. `python -m src.kaggle.env` prints accelerator / VRAM / disk and the compute tiers available; the sweep marks unavailable tiers `blocked` instead of crashing.
5. Start with `configs/kaggle_debug.yaml` (TextCNN + DistilBERT on 2,000 rows), then `configs/experiment.yaml`; `--budget-minutes` stops before the session limit; re-running resumes from `outputs/sweep/manifest.json`.
6. Download `/kaggle/working/outputs` and copy into local `outputs/` to regenerate docs tables and figures.

---

## Artifact map

```
data/raw/                                 # local snapshot (never modified)
data/processed/                           # Phases 1-3 on upstream splits (historical; audit FAILED - do not evaluate on these)
data/processed/splits/v1_leakfree/        # Phase 3.5 (frozen; audit PASSED)
  train|validation|test.parquet, train_balanced|augmented|adversarial.parquet
  manifest.json, dropped_rows.jsonl, family_assignments.parquet
  reports/leakage_report*.json, subsets/, kaggle_package/, kaggle_package_v1_leakfree.zip
outputs/smoke/<model>_seed<seed>/         # CPU smoke runs (not publication results)
outputs/runs/<model>/seed_<seed>/         # production runs (none yet)
outputs/sweep/manifest.json               # sweep state
outputs/tables/, outputs/figures/         # aggregated tables and figures
docs/registry.json, docs/MODEL_REGISTRY.md
```

---

## Project Structure

```
model/
├── papers/                  # Reference literature (PDF)
├── configs/                 # data.yaml, experiment.yaml, kaggle_debug.yaml, models/<model_id>.yaml (16)
├── docs/                    # DATA_CARD, MODEL_REGISTRY, REPRODUCIBILITY, registry.json
├── notebooks/               # kaggle_train.ipynb (thin), kaggle_run.md
├── src/
│   ├── data/                # Phases 1-3 + families.py, resplit.py, phase35.py, leakage.py
│   ├── models/              # base.py, vocab.py, textcnn.py, bilstm_attention.py, cnn_bilstm.py,
│   │                        # hf_encoder.py, decoder_classifier.py, registry.py
│   ├── training/            # data.py, trainer.py, train_cli.py, sweep.py
│   ├── eval/                # evaluate.py, predict.py, tables.py, plots.py
│   └── kaggle/              # env.py
├── scripts/                 # download_data.py, kaggle_run.sh, package_kaggle.sh
├── tests/                   # 42 tests (data pipeline, study modules, smoke)
├── data/, outputs/, mlruns/ # gitignored artifacts
├── dvc.yaml, params.yaml, pyproject.toml, requirements*.txt
└── README.md
```

---

## Getting Started

```bash
git clone <repository-url> && cd model
python -m venv .venv && .venv\Scripts\activate      # Windows; use source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt -r requirements-dev.txt

python -m compileall -q src tests && pytest -q          # 42 passed
python -m src.data.phase35 --config configs/data.yaml   # needs data/raw/*.jsonl (local snapshot)
python -m src.training.train_cli --model textcnn --seed 42 --smoke
python -m src.eval.tables --runs outputs/smoke --out outputs/smoke/tables
python -m src.eval.plots  --runs outputs/smoke --out outputs/smoke/figures --tag "SMOKE"
python -m src.models.registry --check
```

Docker (CPU): `docker build -t prompt-injection-model -f docker/Dockerfile .` then `docker run --rm -v %cd%/data:/app/data -v %cd%/outputs:/app/outputs prompt-injection-model --model textcnn --seed 42 --smoke`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `LeakageError: ... not frozen` or `audit failed` | You pointed at Phase 1-3 outputs or a modified split dir. Run `python -m src.data.phase35`; use `--allow-unfrozen` only for local debugging. |
| `Model 'llama31_8b' is registered as BLOCKED` | Gated checkpoint. Export `HF_TOKEN` with approved access (env var / Kaggle Secret) and pass `--force-blocked`. |
| `bitsandbytes` import error / QLoRA fallback recorded | `bitsandbytes` needs Linux + CUDA. The decoder loads without quantization (needs more VRAM) and records `fallback_reason` in `load_info`. |
| Qwen3 "model type not recognised" | transformers >= 4.51 required (`requirements.txt` pins >= 4.51). |
| Sweep job `blocked: compute tier ... not available` | `python -m src.kaggle.env` shows the detected VRAM; choose a GPU session or smaller models. |
| Kaggle session killed mid-run | Re-run the notebook: `outputs/sweep/manifest.json` re-queues stale jobs and the trainer resumes from `last/`. |
| Smoke metrics look poor | Expected: 300 train rows, 1 epoch. Smoke runs only verify the pipeline. |

---

## Limitations and honesty notes

- **No production training has been executed.** All 16 models are implemented and unit-tested; only TextCNN and DistilBERT were run as CPU smoke tests on 300-row subsets. No thesis-level performance numbers exist yet.
- Decoder (7B/8B) paths were validated with a tiny random Llama config; real weights were never loaded locally (no CUDA on the development machine).
- Leakage checks are heuristic (exact, template, Jaccard >= 0.85 word-3-gram, family). Semantic paraphrases below the threshold can still cross splits.
- The dataset licence/provenance relies on the local snapshot metadata; the upstream page was inaccessible. Do not redistribute the data without verified permission.
- `severity` has only three values in the snapshot; severity-regression experiments described in earlier drafts are not meaningful on this data and are not implemented.
- Historical Phase 1-3 outputs (`data/processed/*.parquet`, `kaggle_package.zip`) are kept but are not valid evaluation data.

---

## Reference Papers

Local copies in [`papers/`](papers/) · [Google Drive collection](https://drive.google.com/drive/folders/1K9nPcdnSYI6iGPgNNwDmbew5pM4voqwz)

| Paper | Focus Area |
|-------|------------|
| `1.pdf` | Effectiveness of existing detection methods |
| `4.pdf` - `6.pdf` | Real-world LLM compromise, AgentSentry, indirect injection |
| `7.pdf`, `10.pdf` | Network intrusion & malicious traffic classification |
| `8.pdf` - `9.pdf`, `11.pdf` | ARGUS - LLM agent defence |
| `12.pdf` | CLAWGUARD - runtime agent security |
| `13.pdf` - `15.pdf` | Hybrid defence, threat taxonomy, injection attacks |
| `16.pdf` | Federated learning poisoning detection |
| `17.pdf` - `18.pdf` | MetaSecAlign - secure foundation LLM |
| `19.pdf` | Systematic literature review on LLM defences |
| `20.pdf` - `21.pdf` | Retrieval barrier, TaintP2X injection detection |

---

## Citation

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
| **This repository** | [MIT License](LICENSE) | Md. Nazmus Sakib · Kazi Omar Faruq |
| **Threat Matrix dataset** | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (per local snapshot metadata) | Sanskar Jajoo / Neuralchemy |
| **Pre-trained models** | Respective Hugging Face model licences (see `docs/MODEL_REGISTRY.md`) | Model authors |

The dataset is **not** covered by MIT: it is Creative Commons Attribution-NonCommercial 4.0. Attribute the authors, do not use it commercially, and do not redistribute it without verified permission. Full legal code: https://creativecommons.org/licenses/by-nc/4.0/legalcode

**Copyright © 2026** Md. Nazmus Sakib ([engrskib.com](https://engrskib.com)) and Kazi Omar Faruq.

---

<div align="center">

**Daffodil International University · 7th Capstone Design · 2026**

*LLM Security · Prompt Injection Detection · Deep Learning*

</div>
