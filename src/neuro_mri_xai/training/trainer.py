# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Training and validation loops with AMP, early stopping, and checkpointing."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from neuro_mri_xai.config import Config
from neuro_mri_xai.data import get_dataloaders
from neuro_mri_xai.models import build_model
from neuro_mri_xai.models.lora import save_lora_adapter
from neuro_mri_xai.models.sam_roi import make_roi_fn, unload_sam
from neuro_mri_xai.models.swin_classifier import get_backbone_and_head_params
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.plotting import save_training_curves
from neuro_mri_xai.utils.seed import set_seed


@dataclass
class TrainerState:
    best_val_acc: float = 0.0
    epochs_without_improvement: int = 0
    history: dict = field(default_factory=lambda: {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
    })


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.dataset.seed)

        roi_fn = make_roi_fn(config) if config.sam.enabled else None
        self.train_loader, self.val_loader, self.test_loader, self.class_names = get_dataloaders(
            config, roi_fn=roi_fn,
        )
        if config.sam.enabled:
            unload_sam()

        self.model = build_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        self.scaler = GradScaler(enabled=config.training.use_amp and self.device.type == "cuda")
        self.state = TrainerState()
        ensure_dir(config.training.checkpoint_dir)
        ensure_dir(config.training.log_dir)

    def _build_optimizer(self) -> AdamW:
        cfg = self.config.training
        if self.config.model.use_lora:
            params = [p for p in self.model.parameters() if p.requires_grad]
            return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        backbone_params, head_params = get_backbone_and_head_params(self.model)
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": cfg.backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": cfg.lr})
        if not groups:
            groups = [{"params": self.model.parameters(), "lr": cfg.lr}]
        return AdamW(groups, weight_decay=cfg.weight_decay)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float) -> None:
        payload = {
            "epoch": epoch,
            "val_acc": val_acc,
            "class_names": self.class_names,
            "config_backbone": self.config.model.backbone,
            "use_lora": self.config.model.use_lora,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)
        if self.config.model.use_lora and hasattr(self.model, "save_pretrained"):
            save_lora_adapter(self.model, path.parent / "lora_adapter")

    def train(self) -> Path:
        cfg = self.config.training
        best_path = cfg.checkpoint_dir / "best_swin.pt"
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_loss)
            self.state.history["train_loss"].append(train_loss)
            self.state.history["val_loss"].append(val_loss)
            self.state.history["train_acc"].append(train_acc)
            self.state.history["val_acc"].append(val_acc)

            log_file = cfg.log_dir / "training_log.csv"
            write_header = not log_file.exists()
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
                w.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

            print(
                f"Epoch {epoch}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )
            if val_acc > self.state.best_val_acc:
                self.state.best_val_acc = val_acc
                self.state.epochs_without_improvement = 0
                self._save_checkpoint(best_path, epoch, val_acc)
                print(f"  Saved best checkpoint (val_acc={val_acc:.4f})")
            else:
                self.state.epochs_without_improvement += 1
            if self.state.epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        save_training_curves(self.state.history, cfg.log_dir / "training_curves.png")
        return best_path
