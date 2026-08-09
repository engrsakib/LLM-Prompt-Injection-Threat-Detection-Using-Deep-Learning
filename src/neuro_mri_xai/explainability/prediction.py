# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Helpers to derive class indices from classifier logits."""

from __future__ import annotations

import torch


def class_logits(output: torch.Tensor) -> torch.Tensor:
    """Reduce model output to shape (batch, num_classes)."""
    if output.dim() == 1:
        return output.unsqueeze(0)
    if output.dim() == 2:
        return output
    if output.dim() == 3:
        # (batch, tokens/spatial, num_classes) — pool over middle axis
        return output.mean(dim=1)
    raise ValueError(
        f"Cannot derive class logits from output shape {tuple(output.shape)}; "
        "expected (batch, num_classes) or (batch, *, num_classes).",
    )


def sample_prediction(output: torch.Tensor) -> tuple[int, float]:
    """Return predicted class index and confidence for batch item 0."""
    logits = class_logits(output)
    probs = torch.softmax(logits, dim=-1)
    pred_idx = int(logits[0].argmax(dim=-1).item())
    confidence = float(probs[0, pred_idx].item())
    return pred_idx, confidence
