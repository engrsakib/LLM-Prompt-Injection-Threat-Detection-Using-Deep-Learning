# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Benchmark Swin, ConvNeXt, and DenseNet on the held-out test set."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import yaml

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data import get_dataloaders
from neuro_mri_xai.data.dataset import ensure_dataset_available
from neuro_mri_xai.evaluation.checkpoint import load_checkpoint_model
from neuro_mri_xai.evaluation.metrics import evaluate_classifier
from neuro_mri_xai.models.classifier import BENCHMARK_BACKBONES
from neuro_mri_xai.training.train_cli import run_training
from neuro_mri_xai.utils.cli import add_data_dir_argument
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.plotting import save_confusion_matrix
from neuro_mri_xai.utils.seed import set_seed


def _safe_backbone_name(backbone: str) -> str:
    return backbone.replace(".", "_").replace("/", "_")


def _checkpoint_for_backbone(config, backbone: str) -> Path:
    return config.training.checkpoint_dir / f"best_{_safe_backbone_name(backbone)}.pt"


def _write_backbone_config(base_path: str, backbone: str, ckpt_dir: Path) -> Path:
    with open(base_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.setdefault("model", {})["backbone"] = backbone
    raw.setdefault("training", {})["checkpoint_dir"] = str(ckpt_dir)
    tmp = Path(tempfile.gettempdir()) / f"benchmark_{_safe_backbone_name(backbone)}.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(raw, f)
    return tmp


def run_benchmark(
    config_path: str = "configs/default.yaml",
    data_dir: str | None = None,
    train: bool = True,
    epochs: int | None = None,
    backbones: list[str] | None = None,
) -> dict:
    config = load_config(config_path, data_dir=data_dir)
    set_seed(config.dataset.seed)
    ensure_dataset_available(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_list = backbones or list(BENCHMARK_BACKBONES.values())
    results: dict[str, dict] = {}
    figures_dir = ensure_dir(config.evaluation.figures_dir / "benchmark")
    bench_ckpt_root = ensure_dir(config.training.checkpoint_dir / "benchmark")

    for backbone in backbone_list:
        print(f"\n=== Benchmark: {backbone} ===")
        ckpt_dir = ensure_dir(bench_ckpt_root / _safe_backbone_name(backbone))
        ckpt_path = ckpt_dir / "best_swin.pt"
        branded_ckpt = _checkpoint_for_backbone(config, backbone)

        if train or not ckpt_path.exists():
            fold_config = _write_backbone_config(config_path, backbone, ckpt_dir)
            run_training(str(fold_config), epochs=epochs, data_dir=data_dir)

        if not ckpt_path.exists():
            print(f"Skipping {backbone}: no checkpoint at {ckpt_path}")
            continue

        if branded_ckpt != ckpt_path:
            branded_ckpt.write_bytes(ckpt_path.read_bytes())

        cfg = load_config(config_path, data_dir=data_dir)
        cfg.model.backbone = backbone
        _, _, test_loader, class_names = get_dataloaders(cfg, roi_fn=None)
        model, class_names = load_checkpoint_model(ckpt_path, cfg)
        eval_result = evaluate_classifier(model, test_loader, class_names, device)

        per_class = eval_result["metrics"].get("per_class", [])
        cm_path = figures_dir / f"confusion_matrix_{_safe_backbone_name(backbone)}.png"
        save_confusion_matrix(
            eval_result["confusion_matrix"],
            class_names,
            cm_path,
            title=f"Confusion Matrix — {backbone}",
            per_class_metrics=per_class,
        )

        payload = {
            k: v
            for k, v in eval_result["metrics"].items()
            if k not in ("classification_report",)
        }
        results[backbone] = payload
        print(f"{backbone} test accuracy: {payload['accuracy']:.4f}")

    summary_path = figures_dir / "benchmark_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark summary saved to {summary_path}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark Swin / ConvNeXt / DenseNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-train", action="store_true", help="Evaluate existing checkpoints only")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=None,
        help="Override backbone list (timm model names)",
    )
    add_data_dir_argument(parser)
    args = parser.parse_args(argv)
    run_benchmark(
        args.config,
        data_dir=args.data_dir,
        train=not args.no_train,
        epochs=args.epochs,
        backbones=args.backbones,
    )


if __name__ == "__main__":
    main()
