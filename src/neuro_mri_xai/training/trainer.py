# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Training loop with AMP, early stopping, and checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuro_mri_xai.config import Config
from neuro_mri_xai.models.lora import save_lora_adapter
from neuro_mri_xai.models.swin_classifier import get_backbone_and_head_params
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.plotting import save_training_curves
from neuro_mri_xai.utils.vram import empty_cuda_cache, log_gpu_mem

CHECKPOINT_SCHEMA_VERSION = 1


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        class_names: list[str],
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.class_names = class_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        backbone_params, head_params = get_backbone_and_head_params(self.model)
        param_groups = []
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": config.training.backbone_lr},
            )
        if head_params:
            param_groups.append({"params": head_params, "lr": config.training.lr})

        self.optimizer = torch.optim.AdamW(
            param_groups or self.model.parameters(),
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = None
        if config.training.use_cosine_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.epochs,
            )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.training.use_amp and torch.cuda.is_available(),
        )
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(train):
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.config.training.use_amp and torch.cuda.is_available(),
                ):
                    logits = self.model(images)
                    loss = F.cross_entropy(logits, labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _save_checkpoint(self, epoch: int, val_acc: float, path: Path) -> None:
        ensure_dir(path.parent)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "epoch": epoch,
            "val_accuracy": val_acc,
            "use_lora": self.config.model.use_lora,
            "backbone": self.config.model.backbone,
        }
        torch.save(payload, path)

        if self.config.model.use_lora:
            adapter_dir = path.parent / "lora_adapter"
            save_lora_adapter(self.model, adapter_dir)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpt["lora_adapter_dir"] = str(adapter_dir)
            torch.save(ckpt, path)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int | None = None,
        resume_path: str | Path | None = None,
    ) -> Path:
        if resume_path and Path(resume_path).exists():
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.best_val_acc = float(ckpt.get("val_accuracy", 0.0))
            self.best_epoch = int(ckpt.get("epoch", 0))
            print(f"Resumed from {resume_path} (val_acc={self.best_val_acc:.4f})")

        n_epochs = epochs or self.config.training.epochs
        ckpt_path = self.config.training.checkpoint_dir / "best_swin.pt"
        log_gpu_mem("training start")

        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch}/{n_epochs} — "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc, ckpt_path)
                print(f"  Saved best checkpoint (val_acc={val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        curves_path = self.config.training.log_dir / "training_curves.png"
        save_training_curves(self.history, curves_path)
        empty_cuda_cache()
        log_gpu_mem("training end")
        return ckpt_path
