#!/bin/bash
# Default container entrypoint: train one model x seed. Override with CLI args, e.g.
#   docker run ... prompt-injection-model --model textcnn --seed 42 --smoke
set -e
if [ "$#" -eq 0 ]; then
  set -- --model "${MODEL:-textcnn}" --seed "${SEED:-42}" --config "${CONFIG:-configs/experiment.yaml}"
fi
exec python -m src.training.train_cli "$@"
