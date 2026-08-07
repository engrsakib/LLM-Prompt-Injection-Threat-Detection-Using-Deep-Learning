# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Grad-CAM + SAM ROI constrained overlay rendering."""

from __future__ import annotations

import numpy as np

from neuro_mri_xai.config import Config
from neuro_mri_xai.models.sam_roi import extract_brain_mask, overlay_heatmap_on_mask


def render_sam_constrained_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (overlay_rgb, mask, mask_confidence)."""
    mask, confidence = extract_brain_mask(image_rgb, config)
    overlay = overlay_heatmap_on_mask(
        image_rgb,
        heatmap,
        mask,
        alpha=config.explainability.alpha,
    )
    return overlay, mask, confidence
