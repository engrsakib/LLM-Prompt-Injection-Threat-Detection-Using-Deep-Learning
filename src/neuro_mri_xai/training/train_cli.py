# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""CLI entry point for Swin + LoRA training."""

from __future__ import annotations

import argparse

import torch

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data import get_dataloaders
from neuro_mri_xai.data.constants import NUM_CLASSES
from neuro_mri_xai.data.dataset import (
    compute_class_weights,
    ensure_dataset_available,
    get_train_labels,
)
from neuro_mri_xai.models import build_model
from neuro_mri_xai.models.swin_classifier import apply_swin_partial_freeze, log_trainable_params
from neuro_mri_xai.training.trainer import Trainer
from neuro_mri_xai.utils.cli import add_data_dir_argument
from neuro_mri_xai.utils.seed import set_seed


def run_training(
    config_path: str = "configs/default.yaml",
    resume: str | None = None,
    epochs: int | None = None,
    data_dir: str | None = None,
) -> str:
    config = load_config(config_path, data_dir=data_dir)
    set_seed(config.dataset.seed)
    ensure_dataset_available(config)
    print(f"Using dataset: {config.dataset.data_dir}")

    # SAM ROI is expensive during training; disable unless explicitly enabled via env.
    roi_fn = None
    train_loader, val_loader, _, class_names = get_dataloaders(config, roi_fn=roi_fn)

    if not class_names:
        class_names = config.get_class_names()
    config.model.num_classes = NUM_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, pretrained=True)

    if config.training.freeze_early_backbone:
        apply_swin_partial_freeze(model)
        log_trainable_params(model, label="Partial fine-tune")

    class_weights = None
    if config.training.use_class_weights:
        train_labels = get_train_labels(config)
        class_weights = compute_class_weights(train_labels, NUM_CLASSES, device=device)
        print(f"Using class-weighted CrossEntropyLoss ({NUM_CLASSES} classes)")

    trainer = Trainer(
        model,
        config,
        class_names,
        device=device,
        class_weights=class_weights,
    )
    ckpt_path = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        resume_path=resume,
    )
    print(f"Best checkpoint: {ckpt_path} (val_acc={trainer.best_val_acc:.4f})")
    print(f"Data dir: {config.dataset.data_dir}")
    return str(ckpt_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Swin + LoRA MRI classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    add_data_dir_argument(parser)
    args = parser.parse_args(argv)
    run_training(args.config, resume=args.resume, epochs=args.epochs, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
