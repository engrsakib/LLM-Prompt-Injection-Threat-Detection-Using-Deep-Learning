#!/usr/bin/env bash
# Upload helper: package is at data/processed/kaggle_package.zip
# Manual Kaggle upload steps:
# 1. Go to https://www.kaggle.com/datasets
# 2. Click "New Dataset"
# 3. Upload data/processed/kaggle_package.zip
# 4. Set title: Prompt Injection Threat Matrix (Processed)
# 5. Set license: CC BY-NC 4.0 (upstream dataset)

set -euo pipefail
python -m src.data.finalize --config configs/data.yaml
echo "Kaggle package ready: data/processed/kaggle_package.zip"
