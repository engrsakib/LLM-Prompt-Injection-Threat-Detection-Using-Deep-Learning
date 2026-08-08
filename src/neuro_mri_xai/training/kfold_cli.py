# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Stratified patient-level k-fold cross-validation training."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data.dataset import ensure_dataset_available
from neuro_mri_xai.training.train_cli import run_training
from neuro_mri_xai.utils.cli import add_data_dir_argument
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.seed import set_seed


def run_kfold_training(
    config_path: str = "configs/default.yaml",
    data_dir: str | None = None,
    n_folds: int = 5,
    epochs: int | None = None,
) -> dict:
    config = load_config(config_path, data_dir=data_dir)
    set_seed(config.dataset.seed)
    ensure_dataset_available(config)

    config.dataset.split_strategy = "patient"
    fold_results: dict[str, dict] = {}

    for fold in range(n_folds):
        print(f"\n========== Fold {fold + 1}/{n_folds} ==========")
        cfg = deepcopy(config)
        cfg.dataset.n_folds = n_folds
        cfg.dataset.fold_index = fold
        cfg.dataset.test_split = 0.0

        ckpt_dir = ensure_dir(cfg.training.checkpoint_dir / f"fold_{fold}")
        cfg.training.checkpoint_dir = ckpt_dir
        cfg.training.log_dir = ensure_dir(cfg.training.log_dir / f"fold_{fold}")

        import tempfile
        from pathlib import Path

        fold_config_path = Path(tempfile.gettempdir()) / f"neuro_mri_fold_{fold}.yaml"
        import yaml

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw.setdefault("dataset", {})
        raw["dataset"]["split_strategy"] = "patient"
        raw["dataset"]["n_folds"] = n_folds
        raw["dataset"]["fold_index"] = fold
        raw["dataset"]["test_split"] = 0.0
        raw.setdefault("training", {})
        raw["training"]["checkpoint_dir"] = str(ckpt_dir)
        raw["training"]["log_dir"] = str(cfg.training.log_dir)
        with open(fold_config_path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f)

        ckpt = run_training(str(fold_config_path), epochs=epochs, data_dir=data_dir)
        fold_results[f"fold_{fold}"] = {"checkpoint": ckpt}

    summary_dir = ensure_dir(config.training.log_dir / "kfold")
    summary_path = summary_dir / "kfold_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2)
    print(f"\nK-fold training complete. Summary: {summary_path}")
    return fold_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Patient-level stratified k-fold training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    add_data_dir_argument(parser)
    args = parser.parse_args(argv)
    run_kfold_training(args.config, data_dir=args.data_dir, n_folds=args.folds, epochs=args.epochs)


if __name__ == "__main__":
    main()
