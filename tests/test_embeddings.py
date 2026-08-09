"""Tests for embedding extraction without model mutation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuro_mri_xai.evaluation.embeddings import extract_embeddings


class _FeatureClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, channels: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.head = nn.Linear(channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        pooled = feat.mean(dim=[2, 3])
        return self.head(pooled)


class _HeadOnlyClassifier(nn.Module):
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.head = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x).mean(dim=[2, 3])
        return self.head(feat)


def test_extract_embeddings_preserves_classification_head() -> None:
    model = _FeatureClassifier()
    original_head = model.head
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])),
        batch_size=2,
    )
    X, y = extract_embeddings(model, loader, torch.device("cpu"))
    assert X.shape[0] == y.shape[0] == 4
    assert model.head is original_head
    logits = model(torch.randn(1, 3, 8, 8))
    assert logits.shape == (1, 8)


def test_extract_embeddings_deepcopy_fallback_without_forward_features() -> None:
    model = _HeadOnlyClassifier()
    original_head = model.head
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 8, 8), torch.tensor([0, 1])),
        batch_size=2,
    )
    X, y = extract_embeddings(model, loader, torch.device("cpu"))
    assert X.shape == (2, 16)
    assert y.shape == (2,)
    assert model.head is original_head
    logits = model(torch.randn(1, 3, 8, 8))
    assert logits.shape == (1, 8)
