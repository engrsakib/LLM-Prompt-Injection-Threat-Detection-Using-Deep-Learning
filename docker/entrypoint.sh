#!/bin/bash
set -e
exec python -m src.training.train_cli --config "${CONFIG:-configs/default.yaml}"

