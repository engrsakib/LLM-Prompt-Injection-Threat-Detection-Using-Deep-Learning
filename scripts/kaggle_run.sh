#!/bin/bash
# Kaggle-friendly runner
pip install -r requirements.txt
python -m src.training.train_cli --config configs/kaggle_debug.yaml

