# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""End-to-end XAI orchestrator for a single MRI sample."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from neuro_mri_xai.config import Config
from neuro_mri_xai.data.transforms import get_val_transforms
from neuro_mri_xai.explainability.attention_rollout import compute_attention_rollout
from neuro_mri_xai.explainability.gradcam import compute_gradcam
from neuro_mri_xai.explainability.sam_overlay import render_sam_constrained_overlay
from neuro_mri_xai.models.sam_roi import resolve_roi_fn, unload_sam
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.vram import empty_cuda_cache, log_gpu_mem


def _save_heatmap_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    title: str,
    alpha: float = 0.4,
) -> Path:
    import cv2

    heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = (alpha * colored + (1 - alpha) * image_rgb).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(blended)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def explain_sample(
    model: nn.Module,
    image_path: str | Path,
    config: Config,
    class_names: list[str],
    output_dir: str | Path,
) -> dict:
    """Run classification + Grad-CAM + attention + SAM overlay for one image."""
    output_dir = ensure_dir(output_dir)
    image_path = Path(image_path)
    device = next(model.parameters()).device

    pil_image = Image.open(image_path).convert("RGB")
    roi_fn = resolve_roi_fn(config)
    if roi_fn is not None:
        pil_image = roi_fn(pil_image)
    image_rgb = np.array(pil_image)

    transform = get_val_transforms(config.dataset.image_size)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
    prediction = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    log_gpu_mem("XAI classification")

    gradcam = compute_gradcam(model, tensor, pred_idx)
    gradcam_path = _save_heatmap_overlay(
        image_rgb,
        gradcam,
        output_dir / f"{image_path.stem}_gradcam.png",
        "Grad-CAM",
        alpha=config.explainability.alpha,
    )

    attention = compute_attention_rollout(model, tensor, target_class=pred_idx)
    attention_path = _save_heatmap_overlay(
        image_rgb,
        attention,
        output_dir / f"{image_path.stem}_attention.png",
        "Attention Saliency",
        alpha=config.explainability.alpha,
    )

    sam_overlay_path: str | None = None
    sam_mask_path: str | None = None
    mask_confidence = 0.0
    if config.sam.enabled:
        overlay, _mask, mask_confidence = render_sam_constrained_overlay(
            image_rgb,
            gradcam,
            config,
        )
        sam_out = output_dir / f"{image_path.stem}_sam_overlay.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(overlay)
        ax.set_title("SAM-Constrained Grad-CAM")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(sam_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        sam_overlay_path = str(sam_out)

        sam_mask_path = str(output_dir / f"{image_path.stem}_sam_mask.png")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(_mask, cmap="gray")
        ax.set_title("SAM ROI Mask")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(sam_mask_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if config.vram.sequential_models:
            unload_sam()
            empty_cuda_cache()
            log_gpu_mem("SAM unloaded")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "gradcam_path": str(gradcam_path),
        "attention_path": str(attention_path),
        "sam_overlay_path": sam_overlay_path,
        "sam_mask_path": sam_mask_path,
        "mask_confidence": mask_confidence,
    }
