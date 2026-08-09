# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Batch XAI export for test-set samples."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from neuro_mri_xai.config import Config
from neuro_mri_xai.data.dataset import (
    MRIDataset,
    ensure_dataset_available,
    stratified_split_indices,
)
from neuro_mri_xai.data.splits import build_group_ids
from neuro_mri_xai.data.transforms import get_val_transforms
from neuro_mri_xai.explainability.attention_rollout import compute_attention_rollout
from neuro_mri_xai.explainability.gradcam import compute_gradcam
from neuro_mri_xai.explainability.pipeline import _save_heatmap_overlay
from neuro_mri_xai.explainability.sam_overlay import render_sam_constrained_overlay
from neuro_mri_xai.models.sam_roi import unload_sam
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.vram import empty_cuda_cache

logger = logging.getLogger(__name__)


def _sample_prediction(output: torch.Tensor) -> tuple[int, float]:
    """Return predicted class index and confidence for batch item 0."""
    if output.dim() == 1:
        logits = output.unsqueeze(0)
    elif output.dim() == 2:
        logits = output
    else:
        logits = output.reshape(output.size(0), -1)

    probs = torch.softmax(logits, dim=-1)
    sample_probs = probs[0]
    pred_idx = int(sample_probs.argmax(dim=-1).item())
    confidence = float(sample_probs[pred_idx].item())
    return pred_idx, confidence


def _get_test_sample_paths(config: Config, max_samples: int) -> list[tuple[Path, int]]:
    data_dir = ensure_dataset_available(config)
    expected = config.get_class_names()
    base = MRIDataset(data_dir, transform=None, roi_fn=None, expected_classes=expected)
    labels = [label for _, label in base.samples]
    class_names = base.classes
    groups = build_group_ids(base.samples, class_names)
    _, _, test_idx = stratified_split_indices(
        labels,
        val_split=config.dataset.val_split,
        test_split=max(config.dataset.test_split, 0.05),
        seed=config.dataset.seed,
        groups=groups,
        split_strategy=config.dataset.split_strategy,
        n_folds=1,
        fold_index=0,
    )

    selected = test_idx[:max_samples] if max_samples > 0 else test_idx
    return [(base.samples[i][0], base.samples[i][1]) for i in selected]


def export_xai_batch(
    model: torch.nn.Module,
    config: Config,
    class_names: list[str],
    output_dir: str | Path,
    max_samples: int = 16,
) -> dict:
    """Export Grad-CAM, attention, SAM ROI, and metadata for test samples."""
    output_dir = ensure_dir(output_dir)
    device = next(model.parameters()).device
    transform = get_val_transforms(config.dataset.image_size)
    samples = _get_test_sample_paths(config, max_samples)

    records: list[dict] = []
    model.eval()

    for path, label_idx in samples:
        stem = path.stem
        sample_dir = ensure_dir(output_dir / stem)
        pil_image = Image.open(path).convert("RGB")
        image_rgb = np.array(pil_image)
        tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred_idx, confidence = _sample_prediction(output)

        gradcam = compute_gradcam(model, tensor, pred_idx)
        gradcam_path = _save_heatmap_overlay(
            image_rgb,
            gradcam,
            sample_dir / f"{stem}_gradcam.png",
            "Grad-CAM",
            alpha=config.explainability.alpha,
        )

        attention = compute_attention_rollout(model, tensor, target_class=pred_idx)
        attention_path = _save_heatmap_overlay(
            image_rgb,
            attention,
            sample_dir / f"{stem}_attention.png",
            "Attention Saliency",
            alpha=config.explainability.alpha,
        )

        sam_overlay_path: str | None = None
        sam_mask_path: str | None = None
        mask_confidence = 0.0
        if config.sam.enabled:
            overlay, mask, mask_confidence = render_sam_constrained_overlay(
                image_rgb,
                gradcam,
                config,
            )
            sam_overlay_path = str(sample_dir / f"{stem}_sam_overlay.png")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay)
            ax.set_title("SAM-Constrained Grad-CAM")
            ax.axis("off")
            plt.tight_layout()
            fig.savefig(sam_overlay_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            sam_mask_path = str(sample_dir / f"{stem}_sam_mask.png")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(mask, cmap="gray")
            ax.set_title("SAM ROI Mask")
            ax.axis("off")
            plt.tight_layout()
            fig.savefig(sam_mask_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            np.save(sample_dir / f"{stem}_gradcam.npy", gradcam)
            if config.vram.sequential_models:
                unload_sam()
                empty_cuda_cache()

        true_class = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
        pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        records.append(
            {
                "image": str(path),
                "true_class": true_class,
                "predicted_class": pred_class,
                "confidence": confidence,
                "correct": pred_idx == label_idx,
                "gradcam_path": str(gradcam_path),
                "attention_path": str(attention_path),
                "sam_overlay_path": sam_overlay_path,
                "sam_mask_path": sam_mask_path,
                "mask_confidence": mask_confidence,
            },
        )

    summary_path = output_dir / "xai_batch_index.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"samples": records, "count": len(records)}, f, indent=2)

    logger.info("Exported XAI for %d samples to %s", len(records), output_dir)
    return {"output_dir": str(output_dir), "count": len(records), "index_path": str(summary_path)}
