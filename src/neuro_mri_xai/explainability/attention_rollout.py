# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Attention rollout / input-gradient saliency for Swin classifiers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_attention_rollout(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Input-gradient saliency map (attention-style rollout fallback for timm Swin)."""
    model.eval()
    x = input_tensor.clone().requires_grad_(True)
    output = model(x)
    class_idx = int(output.argmax(dim=1).item())
    model.zero_grad(set_to_none=True)
    output[0, class_idx].backward()
    saliency = x.grad.abs().sum(dim=1).squeeze().detach().cpu().numpy()
    return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
