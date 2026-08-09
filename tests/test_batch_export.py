"""Tests for batch XAI export helpers."""

from __future__ import annotations

import pytest
import torch

from neuro_mri_xai.explainability.prediction import class_logits, sample_prediction


def test_sample_prediction_from_batch_logits() -> None:
    logits = torch.tensor([[2.0, 0.5, 0.1]])
    pred_idx, confidence = sample_prediction(logits)
    assert pred_idx == 0
    assert 0.0 < confidence <= 1.0


def test_sample_prediction_from_1d_logits() -> None:
    logits = torch.tensor([0.1, 3.0, 0.2])
    pred_idx, confidence = sample_prediction(logits)
    assert pred_idx == 1
    assert confidence > 0.5


def test_sample_prediction_does_not_use_flat_argmax_index() -> None:
    logits = torch.tensor([[0.0, 5.0, 0.0]])
    pred_idx, confidence = sample_prediction(logits)
    assert pred_idx == 1
    assert abs(confidence - sample_prediction(logits)[1]) < 1e-6


def test_sample_prediction_from_3d_token_logits() -> None:
    """3D (batch, tokens, classes) must yield class index, not flat token index."""
    torch.manual_seed(0)
    output = torch.randn(1, 1698, 8)
    output[0, :, 3] = 100.0
    pred_idx, confidence = sample_prediction(output)
    assert 0 <= pred_idx < 8
    assert pred_idx == 3
    assert 0.0 < confidence <= 1.0


def test_class_logits_pools_token_dimension() -> None:
    output = torch.ones(1, 4, 8)
    output[0, 2, 5] = 10.0
    reduced = class_logits(output)
    assert reduced.shape == (1, 8)
    assert reduced[0, 5] > reduced[0, 0]


def test_compute_gradcam_rejects_invalid_target_class() -> None:
    pytest.importorskip("torch")
    import torch.nn as nn

    from neuro_mri_xai.explainability.gradcam import compute_gradcam

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = _TinyModel()
    tensor = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="out of bounds"):
        compute_gradcam(model, tensor, target_class=11874, target_layer=model.conv)
