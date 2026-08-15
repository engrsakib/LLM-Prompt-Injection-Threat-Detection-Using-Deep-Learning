"""Training CLI with MLflow dataset-version logging."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml

from src.data.mlflow_tracking import get_dataset_version_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_loop(cfg: dict) -> None:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    mlflow_cfg = cfg.get("mlflow", {})
    experiment_name = mlflow_cfg.get("experiment_name", "prompt-injection")
    mlflow.set_experiment(experiment_name)

    processed_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    dataset_params = get_dataset_version_params(processed_dir)

    with mlflow.start_run():
        mlflow.log_params({k: str(v) for k, v in cfg.items() if not isinstance(v, dict)})
        mlflow.log_params({k: str(v) for k, v in dataset_params.items() if v is not None})

        metadata = processed_dir / "metadata.json"
        if metadata.exists():
            mlflow.log_artifact(str(metadata), artifact_path="dataset")

        print("Training with config:", cfg)
        print("Dataset version:", dataset_params)
        mlflow.log_metric("setup_complete", 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    train_loop(cfg)


if __name__ == "__main__":
    main()
