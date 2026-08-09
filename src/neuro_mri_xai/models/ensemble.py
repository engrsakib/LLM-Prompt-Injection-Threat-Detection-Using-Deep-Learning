# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Soft-voting ensemble over multiple timm classifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from neuro_mri_xai.config import Config


class SoftVotingEnsemble(nn.Module):
    """Average softmax probabilities from multiple classifiers."""

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None) -> None:
        super().__init__()
        if not models:
            raise ValueError("Ensemble requires at least one model")
        self.models = nn.ModuleList(models)
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("weights length must match number of models")
            total = sum(weights)
            self.weights = torch.tensor([w / total for w in weights], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged class probabilities (B, num_classes)."""
        device = x.device
        weights = self.weights.to(device)
        probs = torch.stack(
            [F.softmax(model(x), dim=1) for model in self.models],
            dim=0,
        )
        weighted = (probs * weights.view(-1, 1, 1)).sum(dim=0)
        return weighted

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return pseudo-logits (log-probabilities) for compatibility with argmax."""
        probs = self.forward(x).clamp(min=1e-8)
        return torch.log(probs)


def build_soft_voting_ensemble(
    models: list[nn.Module],
    weights: list[float] | None = None,
) -> SoftVotingEnsemble:
    ensemble = SoftVotingEnsemble(models, weights=weights)
    for model in ensemble.models:
        model.eval()
    return ensemble


@torch.no_grad()
def evaluate_soft_voting_sequential(
    checkpoint_paths: list[str],
    config: Config,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    weights: list[float] | None = None,
) -> dict:
    """Evaluate soft voting by loading one checkpoint on GPU at a time."""
    from neuro_mri_xai.evaluation.checkpoint import load_checkpoint_model
    from neuro_mri_xai.evaluation.metrics import compute_metrics, confusion_matrix
    from neuro_mri_xai.utils.vram import empty_cuda_cache

    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required")

    n_models = len(checkpoint_paths)
    if weights is None:
        weight_tensor = torch.full((n_models,), 1.0 / n_models)
    else:
        if len(weights) != n_models:
            raise ValueError("weights length must match number of checkpoints")
        total = sum(weights)
        weight_tensor = torch.tensor([w / total for w in weights], dtype=torch.float32)

    accumulated_probs: torch.Tensor | None = None
    all_labels: torch.Tensor | None = None

    for model_idx, ckpt_path in enumerate(checkpoint_paths):
        model, class_names = load_checkpoint_model(ckpt_path, config)
        model.eval()
        batch_probs: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        for images, labels in loader:
            images = images.to(device)
            batch_probs.append(F.softmax(model(images), dim=1).cpu())
            batch_labels.append(labels)

        model.cpu()
        del model
        empty_cuda_cache()

        model_probs = torch.cat(batch_probs, dim=0)
        weighted = weight_tensor[model_idx] * model_probs
        accumulated_probs = weighted if accumulated_probs is None else accumulated_probs + weighted
        all_labels = torch.cat(batch_labels) if all_labels is None else all_labels

    assert accumulated_probs is not None and all_labels is not None
    y_prob = accumulated_probs.numpy()
    y_true = all_labels.numpy()
    y_pred = y_prob.argmax(axis=1)
    return {
        "metrics": compute_metrics(y_true, y_pred, y_prob, class_names),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
