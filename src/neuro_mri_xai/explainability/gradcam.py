# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Grad-CAM heatmaps for Swin Transformer classifiers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_mri_xai.models.classifier import get_gradcam_target_layer
from neuro_mri_xai.models.swin_classifier import get_swin_target_layers


class _GradCAMHook:
    def __init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

    def forward_hook(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output

    def backward_hook(self, _module: nn.Module, _grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0]


def compute_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    target_class: int,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single preprocessed image tensor (1,C,H,W)."""
    model.eval()
    layer = target_layer
    if layer is None:
        try:
            layer, _ = get_swin_target_layers(model)
        except ValueError:
            layer = get_gradcam_target_layer(model)

    hook = _GradCAMHook()
    handle_f = layer.register_forward_hook(hook.forward_hook)
    handle_b = layer.register_full_backward_hook(hook.backward_hook)

    try:
        tensor = tensor.clone().requires_grad_(True)
        logits = model(tensor)
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward()

        if hook.activations is None or hook.gradients is None:
            return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

        acts = hook.activations.detach()
        grads = hook.gradients.detach()

        if acts.dim() == 3:
            # (B, tokens, C) — Swin feature map before pooling
            weights = grads.mean(dim=1, keepdim=True)
            cam = (weights * acts).sum(dim=-1)
            side = int(cam.shape[-1] ** 0.5)
            if side * side == cam.shape[-1]:
                cam = cam.view(1, 1, side, side)
            else:
                cam = cam.unsqueeze(1)
        elif acts.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
        else:
            return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)
    finally:
        handle_f.remove()
        handle_b.remove()
