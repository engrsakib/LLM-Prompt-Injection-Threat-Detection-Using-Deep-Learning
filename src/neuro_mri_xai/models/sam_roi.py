# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""SAM-based brain ROI extraction and XAI overlay."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image

from neuro_mri_xai.config import Config

if TYPE_CHECKING:
    from segment_anything import SamPredictor

_sam_predictor: "SamPredictor | None" = None


def _otsu_bbox(image_rgb: np.ndarray, padding: float = 0.05) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    h, w = gray.shape
    if coords is None:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(coords)
    pad_x, pad_y = int(bw * padding), int(bh * padding)
    return max(0, x - pad_x), max(0, y - pad_y), min(w, x + bw + pad_x), min(h, y + bh + pad_y)


def _load_sam_predictor(config: Config) -> "SamPredictor":
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor

    from segment_anything import SamPredictor, sam_model_registry

    checkpoint = config.sam_checkpoint_path()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}. Run scripts/download_weights.py"
        )

    if config.vram.sam_on_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[config.sam.model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def unload_sam() -> None:
    global _sam_predictor
    _sam_predictor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_brain_mask(image_rgb: np.ndarray, config: Config) -> tuple[np.ndarray, float]:
    """Zero-shot brain ROI mask via SAM center-point prompt; Otsu bbox fallback."""
    h, w = image_rgb.shape[:2]
    try:
        predictor = _load_sam_predictor(config)
        predictor.set_image(image_rgb)
        cx, cy = w // 2, h // 2
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        confidence = float(scores[best_idx])
        return masks[best_idx].astype(np.uint8) * 255, confidence
    except Exception:
        x1, y1, x2, y2 = _otsu_bbox(image_rgb)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask, 0.0


def crop_roi_pil(image: Image.Image, config: Config) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    mask, _ = extract_brain_mask(arr, config)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, bw, bh = cv2.boundingRect(coords)
    return Image.fromarray(arr[y : y + bh, x : x + bw])


def make_roi_fn(config: Config):
    def _roi(image: Image.Image) -> Image.Image:
        return crop_roi_pil(image, config)

    return _roi


def overlay_heatmap_on_mask(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = np.clip(heatmap, 0, 1) * (mask.astype(np.float32) / 255.0)
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (alpha * colored + (1 - alpha) * image_rgb).astype(np.uint8)
