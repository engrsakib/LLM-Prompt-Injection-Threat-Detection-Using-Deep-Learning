#!/bin/bash
# Kaggle-friendly runner (debug profile). Equivalent to notebooks/kaggle_train.ipynb cells 2-7.
# Assumes the repo is the current directory and the v1_leakfree split package is mounted under /kaggle/input.
set -euo pipefail
pip install -q -r requirements-kaggle.txt
python -m src.kaggle.env
SPLITS_DIR="$(python - <<'PY'
from src.kaggle.env import find_kaggle_splits_dir
p = find_kaggle_splits_dir()
print(p if p else "")
PY
)"
if [ -z "$SPLITS_DIR" ]; then
  echo "No mounted split package with manifest.json + train.parquet found under /kaggle/input" >&2
  exit 1
fi
python -m src.training.sweep --config configs/kaggle_debug.yaml --splits-dir "$SPLITS_DIR" \
  --runs-dir /kaggle/working/outputs/runs --manifest /kaggle/working/outputs/sweep/manifest.json
python -m src.eval.tables --runs /kaggle/working/outputs/runs --out /kaggle/working/outputs/tables --n-boot 500
python -m src.eval.plots --runs /kaggle/working/outputs/runs --out /kaggle/working/outputs/figures --splits-dir "$SPLITS_DIR"
