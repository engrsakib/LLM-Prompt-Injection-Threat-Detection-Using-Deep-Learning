# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Soft-voting ensemble over multiple timm classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
