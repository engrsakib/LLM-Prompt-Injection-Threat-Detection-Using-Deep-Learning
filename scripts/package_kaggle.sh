#!/usr/bin/env bash
# Package the existing frozen Phase 3.5 splits for Kaggle.
#
# Output: data/processed/splits/v1_leakfree/kaggle_upload_v1_leakfree.zip
#
# Manual Kaggle upload (only if you have verified permission to redistribute; upstream licence is
# CC BY-NC 4.0 per local snapshot metadata and the upstream page could not be re-verified):
# 1. https://www.kaggle.com/datasets -> "New Dataset" (keep it PRIVATE)
# 2. Extract the ZIP above and upload its files together.
# 3. Title: Prompt Injection Threat Matrix - v1_leakfree (leakage-aware re-split)
# 4. Licence: CC BY-NC 4.0 (upstream dataset)
set -euo pipefail
python -m src.data.kaggle_package "$@"
