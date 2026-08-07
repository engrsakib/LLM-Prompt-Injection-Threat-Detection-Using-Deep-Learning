# Neurological MRI XAI Pipeline

PyTorch pipeline for classifying neurological MRI scans with Swin Transformer, LoRA fine-tuning, SAM brain ROI extraction, and Florence-2 natural-language reporting.

## Dataset

[Kaggle: neurological-disorders-mri-dataset-for-xai](https://www.kaggle.com/datasets/engrsakib02/neurological-disorders-mri-dataset-for-xai)

8 classes: AD (mild/moderate/very mild), brain tumors (glioma/meningioma/pituitary), MS, Normal.

## Quick Start (Google Colab)

Open [`Colab_Runner.ipynb`](Colab_Runner.ipynb) and run all cells. The notebook will:

1. Clone this repo
2. Install dependencies
3. Download the dataset (Kaggle API or Google Drive)
4. Train Swin + LoRA
5. Evaluate, explain, and generate reports

## Local Development

```bash
pip install -r requirements.txt
pip install -e .

python scripts/download_data.py --source kaggle
python scripts/download_weights.py
python -m neuro_mri_xai.train --config configs/default.yaml
python -m neuro_mri_xai.evaluate --checkpoint outputs/checkpoints/best_swin.pt
python -m neuro_mri_xai.report --checkpoint outputs/checkpoints/best_swin.pt --image path/to/image.jpg
```

## Project Structure

```
configs/default.yaml       # Hyperparameters and toggles
scripts/                   # Data and weight download helpers
src/neuro_mri_xai/         # Core package
Colab_Runner.ipynb         # Master Colab orchestrator
legacy/                    # Original Kaggle notebook (reference)
```

## Configuration

Edit [`configs/default.yaml`](configs/default.yaml) or set environment variables:

- `NEURO_MRI_DATA_DIR` — override dataset path
- `NEURO_MRI_PROJECT_ROOT` — override project root
