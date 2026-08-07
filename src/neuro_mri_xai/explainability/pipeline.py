# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""End-to-end XAI pipeline orchestrating Grad-CAM, attention, and SAM overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from neuro_mri_xai.config import Config
from neuro_mri_xai.data.transforms import get_test_transforms
from neuro_mri_xai.explainability.attention_rollout import compute_attention_rollout
from neuro_mri_xai.explainability.gradcam import GradCAM, display_gradcam
from neuro_mri_xai.explainability.sam_overlay import save_sam_overlay
from neuro_mri_xai.utils.paths import ensure_dir


def explain_sample(
    model: torch.nn.Module,
    image_path: str | Path,
    config: Config,
    class_names: list[str],
    output_dir: str | Path | None = None,
) -> dict:
    device = next(model.parameters()).device
    image_path = Path(image_path)
    pil_image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_image)

    tensor = get_test_transforms(config.dataset.image_size)(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())

    gradcam = GradCAM(model)
    heatmap = gradcam(tensor, class_idx)
    gradcam.remove_hooks()
    attention = compute_attention_rollout(model, tensor)

    result: dict = {
        "prediction": class_names[class_idx],
        "confidence": confidence,
        "gradcam_path": None,
        "attention_path": None,
        "sam_overlay_path": None,
    }

    if output_dir:
        out = ensure_dir(output_dir)
        stem = image_path.stem
        gradcam_path = out / f"{stem}_gradcam.png"
        attention_path = out / f"{stem}_attention.png"
        display_gradcam(image_path, heatmap, gradcam_path, alpha=config.explainability.alpha)
        display_gradcam(
            image_path, attention, attention_path,
            alpha=config.explainability.alpha, title="Attention Saliency",
        )
        result["gradcam_path"] = str(gradcam_path)
        result["attention_path"] = str(attention_path)
        if config.sam.enabled:
            sam_path = out / f"{stem}_sam_overlay.png"
            save_sam_overlay(image_rgb, heatmap, sam_path, config)
            result["sam_overlay_path"] = str(sam_path)

    return result
