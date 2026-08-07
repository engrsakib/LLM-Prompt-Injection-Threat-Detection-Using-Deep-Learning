# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Grad-CAM visual explanation for Swin Transformer classifiers."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from neuro_mri_xai.utils.paths import ensure_dir


def find_gradcam_target_layer(model: nn.Module) -> nn.Module:
    inner = model.base_model if hasattr(model, "base_model") else model
    if hasattr(inner, "model"):
        inner = inner.model
    if hasattr(inner, "layers") and len(inner.layers) > 0:
        last_stage = inner.layers[-1]
        if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
            return last_stage.blocks[-1].norm1
    for _, module in reversed(list(inner.named_modules())):
        if "norm" in type(module).__name__.lower():
            return module
    raise RuntimeError("Could not find Grad-CAM target layer")


class GradCAM:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.target_layer = find_gradcam_target_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd(_m: nn.Module, _i: tuple, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def bwd(_m: nn.Module, _gi: tuple, grad_output: tuple) -> None:
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        output[0, class_idx].backward(retain_graph=True)

        grads, acts = self.gradients, self.activations
        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks failed")

        if acts.dim() == 3:
            b, tokens, c = acts.shape
            side = int(tokens**0.5)
            if side * side == tokens:
                acts = acts.view(b, side, side, c).permute(0, 3, 1, 2)
                grads = grads.view(b, side, side, c).permute(0, 3, 1, 2)

        weights = grads.mean(dim=(2, 3), keepdim=True) if grads.dim() == 4 else grads.mean(dim=1, keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)


def display_gradcam(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_path: str | Path | None = None,
    alpha: float = 0.4,
    title: str = "Grad-CAM Medical Explanation",
) -> Path | None:
    img = np.array(Image.open(image_path).convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    jet = cm.get_cmap("jet")
    colored = jet(heatmap_resized)[:, :, :3]
    overlay = (alpha * colored + (1 - alpha) * img / 255.0).clip(0, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    plt.show()
    plt.close(fig)
    return None
