# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Evaluate saved checkpoints on the held-out test set."""

from __future__ import annotations

import argparse
import json

import torch

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data import get_dataloaders
from neuro_mri_xai.data.dataset import ensure_dataset_available
from neuro_mri_xai.evaluation.checkpoint import load_checkpoint_model
from neuro_mri_xai.evaluation.metrics import (
    evaluate_classifier,
    extract_embeddings,
    plot_roc_curves,
    run_sklearn_baselines,
)
from neuro_mri_xai.explainability.batch_export import export_xai_batch
from neuro_mri_xai.models.sam_roi import make_roi_fn, unload_sam
from neuro_mri_xai.utils.cli import add_data_dir_argument
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.plotting import save_confusion_matrix
from neuro_mri_xai.utils.seed import set_seed


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    data_dir: str | None = None,
    export_xai: bool = False,
    xai_max_samples: int | None = None,
) -> dict:
    config = load_config(config_path, data_dir=data_dir)
    set_seed(config.dataset.seed)
    ensure_dataset_available(config)
    print(f"Using dataset: {config.dataset.data_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roi_fn = make_roi_fn(config) if config.sam.enabled else None
    _, _, test_loader, _ = get_dataloaders(config, roi_fn=roi_fn)
    if config.sam.enabled:
        unload_sam()

    model, class_names = load_checkpoint_model(checkpoint_path, config)
    results = evaluate_classifier(model, test_loader, class_names, device)

    figures_dir = ensure_dir(config.evaluation.figures_dir)
    per_class = results["metrics"].get("per_class", [])
    save_confusion_matrix(
        results["confusion_matrix"],
        class_names,
        figures_dir / "confusion_matrix.png",
        per_class_metrics=per_class if config.evaluation.export_per_class_metrics else None,
    )
    plot_roc_curves(
        results["y_true"],
        results["y_prob"],
        class_names,
        figures_dir / "roc_curves.png",
    )

    payload = {k: v for k, v in results["metrics"].items() if k != "classification_report"}
    with open(figures_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if per_class:
        with open(figures_dir / "per_class_metrics.json", "w", encoding="utf-8") as f:
            json.dump(per_class, f, indent=2)
    with open(figures_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(results["metrics"]["classification_report"])

    print("Test metrics:", payload)
    print(results["metrics"]["classification_report"])

    if config.evaluation.run_sklearn_baselines:
        _, _, test_loader_emb, _ = get_dataloaders(config, roi_fn=roi_fn)
        X, y = extract_embeddings(model, test_loader_emb, device)
        baseline_results = run_sklearn_baselines(X, y, seed=config.dataset.seed)
        print("Sklearn baselines on Swin embeddings:", baseline_results)
        with open(figures_dir / "sklearn_baselines.json", "w", encoding="utf-8") as f:
            json.dump(baseline_results, f, indent=2)

    if export_xai or config.evaluation.export_batch_xai:
        max_n = xai_max_samples or config.evaluation.xai_max_samples
        xai_out = export_xai_batch(model, config, class_names, figures_dir / "xai_batch", max_n)
        print(f"Batch XAI exported: {xai_out['count']} samples -> {xai_out['output_dir']}")

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Swin MRI classifier on test set")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-xai", action="store_true", help="Export batch XAI visualizations")
    parser.add_argument("--xai-max-samples", type=int, default=None)
    add_data_dir_argument(parser)
    args = parser.parse_args(argv)
    run_evaluation(
        args.config,
        args.checkpoint,
        data_dir=args.data_dir,
        export_xai=args.export_xai,
        xai_max_samples=args.xai_max_samples,
    )


if __name__ == "__main__":
    main()
