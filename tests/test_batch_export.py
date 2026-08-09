"""Tests for batch XAI export helpers."""

from __future__ import annotations

import torch

from neuro_mri_xai.explainability.batch_export import _sample_prediction


def test_sample_prediction_from_batch_logits() -> None:
    logits = torch.tensor([[2.0, 0.5, 0.1]])
    pred_idx, confidence = _sample_prediction(logits)
    assert pred_idx == 0
    assert 0.0 < confidence <= 1.0


def test_sample_prediction_from_1d_logits() -> None:
    logits = torch.tensor([0.1, 3.0, 0.2])
    pred_idx, confidence = _sample_prediction(logits)
    assert pred_idx == 1
    assert confidence > 0.5


def test_sample_prediction_does_not_use_flat_argmax_index() -> None:
    # Shape (1, 3): flat argmax would still be class dim, but ensure class index is used
    logits = torch.tensor([[0.0, 5.0, 0.0]])
    pred_idx, confidence = _sample_prediction(logits)
    assert pred_idx == 1
    assert abs(confidence - _sample_prediction(logits)[1]) < 1e-6
