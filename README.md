# Neurological MRI XAI Pipeline

PyTorch pipeline for classifying neurological MRI scans with **Swin Transformer**, **LoRA** fine-tuning, **SAM** brain ROI extraction, and **Florence-2** natural-language reporting.

Refactored from the legacy Kaggle notebook ([`legacy/notebook8010ef336b.ipynb`](legacy/notebook8010ef336b.ipynb)) into a modular, pip-installable package.

## Dataset

[Kaggle: neurological-disorders-mri-dataset-for-xai](https://www.kaggle.com/datasets/engrsakib02/neurological-disorders-mri-dataset-for-xai)

| Class | Description |
|-------|-------------|
| `AD_MildDemented` / `AD_ModerateDemented` / `AD_VeryMildDemented` | Alzheimer's stages |
| `BT_glioma` / `BT_meningioma` / `BT_pituitary` | Brain tumors |
| `MS` | Multiple sclerosis |
| `Normal` | Healthy control |

~16,400 images · stratified **80/10/10** train/val/test split (no data leakage).

---

## Directory Structure

```
model/
├── Colab_Runner.ipynb                 # Master orchestrator (Colab / Kaggle)
├── configs/default.yaml               # Hyperparameters and feature toggles
├── scripts/
│   ├── download_data.py               # Kaggle API or Google Drive dataset fetch
│   └── download_weights.py            # SAM checkpoint download
├── src/neuro_mri_xai/
│   ├── config.py                      # YAML loader + env overrides
│   ├── report.py                      # Florence-2 HTML diagnostic reports
│   ├── data/                          # Data engineering & augmentations
│   │   ├── dataset.py                 # ImageFolder wrapper + stratified splits
│   │   └── transforms.py              # Train / Val / Test pipelines
│   ├── models/                        # Model architectures
│   │   ├── swin_classifier.py         # Swin Transformer (timm)
│   │   ├── lora.py                    # PEFT LoRA adapters
│   │   ├── sam_roi.py                 # SAM brain ROI segmentation
│   │   └── florence_reporter.py       # Florence-2 caption generator
│   ├── training/                      # Model training pipeline
│   │   ├── trainer.py                 # AMP, early stopping, checkpointing
│   │   └── train_cli.py               # CLI entry point
│   ├── evaluation/                    # Testing & evaluation
│   │   ├── metrics.py                 # Accuracy, P/R/R, F1, AUC-ROC
│   │   ├── checkpoint.py              # Saved model loader
│   │   └── test_eval.py               # Test-set evaluation CLI
│   ├── explainability/                # Feature extraction & XAI
│   │   ├── gradcam.py                 # Grad-CAM heatmaps
│   │   ├── attention_rollout.py       # Attention saliency maps
│   │   ├── sam_overlay.py             # Grad-CAM + SAM ROI overlay
│   │   ├── pipeline.py                # End-to-end XAI orchestrator
│   │   └── xai_cli.py                 # XAI CLI entry point
│   └── utils/                         # Paths, seed, plotting helpers
├── tests/                             # Smoke and unit tests
├── legacy/                            # Original monolithic notebook (reference)
└── outputs/                           # Checkpoints, figures, reports (gitignored)
```

---

## Step-by-Step Execution (Kaggle / Google Colab)

Run these shell commands sequentially in a notebook cell or terminal.

### 0. Setup environment

```bash
pip install -r requirements.txt
pip install -e .
```

### 1. Download dataset

```bash
# Option A — Kaggle API (set KAGGLE_USERNAME / KAGGLE_KEY first)
python scripts/download_data.py --source kaggle

# Option B — Google Drive (mount Drive, upload dataset to configured path)
python scripts/download_data.py --source gdrive
```

### 2. Download SAM weights

```bash
python scripts/download_weights.py
```

### 3. Train Swin + LoRA

```bash
python -m neuro_mri_xai.training.train_cli --config configs/default.yaml
```

### 4. Evaluate on held-out test set

```bash
python -m neuro_mri_xai.evaluation.test_eval \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_swin.pt
```

Outputs: `outputs/figures/confusion_matrix.png`, `roc_curves.png`, `metrics.json`.

### 5. Generate XAI visualizations (single image)

```bash
python -m neuro_mri_xai.explainability.xai_cli \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_swin.pt \
  --image /path/to/sample.jpg \
  --output-dir outputs/figures
```

### 6. Generate HTML diagnostic report (Florence-2 + XAI)

```bash
python -m neuro_mri_xai.report \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_swin.pt \
  --image /path/to/sample.jpg
```

---

## Quick Start (Colab)

Open [`Colab_Runner.ipynb`](Colab_Runner.ipynb) — it runs all steps above automatically.

Before running, set your GitHub repo URL in cell 2 and Kaggle credentials in Colab Secrets.

---

## Local Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -m pytest tests/ -v
```

## Configuration

Edit [`configs/default.yaml`](configs/default.yaml) or set environment variables:

| Variable | Purpose |
|----------|---------|
| `NEURO_MRI_DATA_DIR` | Override dataset path |
| `NEURO_MRI_PROJECT_ROOT` | Override project root |
| `NEURO_MRI_USE_LORA` | Enable/disable LoRA (`true`/`false`) |
| `NEURO_MRI_SAM_ENABLED` | Enable/disable SAM ROI (`true`/`false`) |

## License

Copyright (C) 2026 Md. Nazmus Sakib — [GNU GPL v3.0](LICENSE)
