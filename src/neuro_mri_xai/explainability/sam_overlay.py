# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Overlay Grad-CAM heatmaps onto SAM brain ROI masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from neuro_mri_xai.config import Config
from neuro_mri_xai.models.sam_roi import extract_brain_mask, overlay_heatmap_on_mask, unload_sam


def create_sam_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    config: Config,
    alpha: float | None = None,
) -> np.ndarray:
    """Generate anatomically constrained XAI overlay using SAM brain mask."""
    mask = extract_brain_mask(image_rgb, config)
    overlay_alpha = alpha if alpha is not None else config.explainability.alpha
    return overlay_heatmap_on_mask(image_rgb, heatmap, mask, overlay_alpha)


def save_sam_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    config: Config,
    alpha: float | None = None,
) -> Path:
    overlay = create_sam_overlay(image_rgb, heatmap, config, alpha=alpha)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    if config.sam.enabled:
        unload_sam()
    return output_path
