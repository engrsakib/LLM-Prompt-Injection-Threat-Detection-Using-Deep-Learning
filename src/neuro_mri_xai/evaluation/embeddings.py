# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Embedding extraction for sklearn baselines without mutating live models."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _unwrap_backbone(model: nn.Module) -> nn.Module:
    """Reach the timm backbone inside optional PEFT / wrapper modules."""
    inner = model
    for _ in range(3):
        if hasattr(inner, "base_model"):
            inner = inner.base_model  # type: ignore[assignment]
            continue
        if hasattr(inner, "model") and isinstance(inner.model, nn.Module):
            inner = inner.model
            continue
        break
    return inner


def _pool_features(feat: torch.Tensor) -> torch.Tensor:
    if feat.dim() == 4:
        return feat.mean(dim=[2, 3])
    if feat.dim() == 3:
        return feat.mean(dim=1)
    return feat


def _forward_embedding_batch(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return pooled embedding vectors without modifying ``backbone``."""
    if hasattr(backbone, "forward_features"):
        feat = backbone.forward_features(images)
        return _pool_features(feat)

    probe = copy.deepcopy(backbone)
    if hasattr(probe, "head"):
        probe.head = nn.Identity()
    feat = probe(images)
    return _pool_features(feat)


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pooled backbone embeddings; leaves the live model head untouched."""
    model.eval()
    backbone = _unwrap_backbone(model)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, lbls in loader:
        feat = _forward_embedding_batch(backbone, images.to(device))
        features.append(feat.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)
