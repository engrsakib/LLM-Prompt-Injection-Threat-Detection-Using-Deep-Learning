# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Soft-voting ensemble evaluation across multiple checkpoints."""

from __future__ import annotations

import argparse
import json

import torch

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data import get_dataloaders
from neuro_mri_xai.data.dataset import ensure_dataset_available
from neuro_mri_xai.evaluation.checkpoint import load_checkpoint_model
from neuro_mri_xai.explainability.batch_export import export_xai_batch
from neuro_mri_xai.models.ensemble import evaluate_soft_voting_sequential
from neuro_mri_xai.models.sam_roi import unload_sam
from neuro_mri_xai.utils.cli import add_data_dir_argument
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.plotting import save_confusion_matrix
from neuro_mri_xai.utils.seed import set_seed
from neuro_mri_xai.utils.vram import empty_cuda_cache


def run_ensemble_evaluation(
    config_path: str,
    checkpoint_paths: list[str],
    data_dir: str | None = None,
    weights: list[float] | None = None,
    export_xai: bool = False,
    xai_max_samples: int = 16,
) -> dict:
    config = load_config(config_path, data_dir=data_dir)
    set_seed(config.dataset.seed)
    ensure_dataset_available(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_paths:
        checkpoint_paths = list(config.ensemble.checkpoint_paths)
    if not checkpoint_paths:
        raise ValueError("Provide --checkpoints or set ensemble.checkpoint_paths in config")

    class_names = config.get_class_names()
    _, _, test_loader, _ = get_dataloaders(config)
    if config.sam.enabled:
        unload_sam()

    results = evaluate_soft_voting_sequential(
        checkpoint_paths,
        config,
        test_loader,
        class_names,
        device,
        weights=weights,
    )
    figures_dir = ensure_dir(config.evaluation.figures_dir)

    per_class = results["metrics"].get("per_class", [])
    save_confusion_matrix(
        results["confusion_matrix"],
        class_names,
        figures_dir / "confusion_matrix_ensemble.png",
        title="Confusion Matrix — Soft Voting Ensemble",
        per_class_metrics=per_class,
    )

    payload = {
        k: v for k, v in results["metrics"].items() if k != "classification_report"
    }
    payload["checkpoints"] = checkpoint_paths
    payload["ensemble_weights"] = (
        weights if weights else [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
    )

    metrics_path = figures_dir / "ensemble_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(figures_dir / "ensemble_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(results["metrics"]["classification_report"])

    print(f"Ensemble test accuracy: {payload['accuracy']:.4f}")
    print(results["metrics"]["classification_report"])

    if export_xai or config.evaluation.export_batch_xai:
        xai_dir = figures_dir / "xai_batch"
        xai_model, _ = load_checkpoint_model(checkpoint_paths[0], config)
        try:
            export_xai_batch(
                xai_model,
                config,
                class_names,
                xai_dir,
                max_samples=xai_max_samples or config.evaluation.xai_max_samples,
            )
        finally:
            xai_model.cpu()
            del xai_model
            empty_cuda_cache()

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate soft-voting ensemble")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Checkpoint paths for each ensemble member",
    )
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--export-xai", action="store_true")
    parser.add_argument("--xai-max-samples", type=int, default=16)
    add_data_dir_argument(parser)
    args = parser.parse_args(argv)
    run_ensemble_evaluation(
        args.config,
        checkpoint_paths=args.checkpoints or [],
        data_dir=args.data_dir,
        weights=args.weights,
        export_xai=args.export_xai,
        xai_max_samples=args.xai_max_samples,
    )


if __name__ == "__main__":
    main()
