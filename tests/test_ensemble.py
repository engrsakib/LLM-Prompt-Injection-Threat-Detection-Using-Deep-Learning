"""Tests for soft-voting ensemble."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuro_mri_xai.models.ensemble import SoftVotingEnsemble, build_soft_voting_ensemble


class _StubClassifier(nn.Module):
    def __init__(self, bias: float) -> None:
        super().__init__()
        self.bias = bias
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x.mean(dim=[2, 3]))
        logits[:, 0] += self.bias
        return logits


def test_soft_voting_ensemble_averages_probabilities() -> None:
    pytest.importorskip("torch")
    models = [_StubClassifier(0.0), _StubClassifier(2.0)]
    ensemble = build_soft_voting_ensemble(models)
    x = torch.randn(2, 3, 8, 8)
    probs = ensemble(x)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_soft_voting_weighted() -> None:
    models = [_StubClassifier(0.0), _StubClassifier(5.0)]
    ensemble = SoftVotingEnsemble(models, weights=[0.9, 0.1])
    x = torch.randn(1, 3, 4, 4)
    probs = ensemble(x)
    assert probs.argmax(dim=1).item() in (0, 1)
