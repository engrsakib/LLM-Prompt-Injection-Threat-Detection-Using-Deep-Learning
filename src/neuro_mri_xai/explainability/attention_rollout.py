# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Attention rollout saliency maps for Swin Transformer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_mri_xai.explainability.gradcam import compute_gradcam
from neuro_mri_xai.models.swin_classifier import unwrap_model


class _AttentionCapture:
    def __init__(self) -> None:
        self.attention_maps: list[torch.Tensor] = []

    def hook(self, _module: nn.Module, _inputs: tuple, _output: tuple) -> None:
        # timm WindowAttention returns (attn_output, attn_weights) when need_weights
        if isinstance(_output, tuple) and len(_output) >= 2 and _output[1] is not None:
            self.attention_maps.append(_output[1].detach())


def compute_attention_rollout(
    model: nn.Module,
    tensor: torch.Tensor,
    target_class: int | None = None,
    num_blocks: int = 2,
) -> np.ndarray:
    """Build attention saliency from the last N Swin blocks; fallback to Grad-CAM."""
    backbone = unwrap_model(model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        if target_class is not None:
            return compute_gradcam(model, tensor, target_class)
        return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

    capture = _AttentionCapture()
    handles: list = []
    hooked = 0
    for stage in reversed(layers):
        blocks = getattr(stage, "blocks", [])
        for block in reversed(blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            handles.append(attn.register_forward_hook(capture.hook))
            hooked += 1
            if hooked >= num_blocks:
                break
        if hooked >= num_blocks:
            break

    try:
        model.eval()
        with torch.no_grad():
            model(tensor)

        if not capture.attention_maps:
            if target_class is not None:
                return compute_gradcam(model, tensor, target_class)
            return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

        rollout: torch.Tensor | None = None
        for attn in reversed(capture.attention_maps):
            attn_mean = attn.mean(dim=1)
            if rollout is None:
                rollout = attn_mean[0] + torch.eye(attn_mean.shape[-1], device=attn.device)
            else:
                rollout = attn_mean[0] @ rollout

        if rollout is None:
            return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

        saliency = rollout.mean(dim=0)
        side = int(saliency.shape[0] ** 0.5)
        if side * side != saliency.shape[0]:
            if target_class is not None:
                return compute_gradcam(model, tensor, target_class)
            return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

        grid = saliency[: side * side].view(1, 1, side, side)
        grid = F.interpolate(grid, size=tensor.shape[2:], mode="bilinear", align_corners=False)
        arr = grid[0, 0].cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr.astype(np.float32)
    finally:
        for h in handles:
            h.remove()
