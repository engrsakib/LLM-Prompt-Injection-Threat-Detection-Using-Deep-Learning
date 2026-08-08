# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""XAI CLI entry point."""

from __future__ import annotations

import argparse

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data.dataset import ensure_dataset_available
from neuro_mri_xai.evaluation.checkpoint import load_checkpoint_model
from neuro_mri_xai.explainability.batch_export import export_xai_batch
from neuro_mri_xai.explainability.pipeline import explain_sample
from neuro_mri_xai.models.sam_roi import unload_sam
from neuro_mri_xai.utils.cli import add_data_dir_argument
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.vram import empty_cuda_cache


def run_xai(
    config_path: str,
    checkpoint_path: str,
    image_path: str | None = None,
    output_dir: str | None = None,
    data_dir: str | None = None,
    batch: bool = False,
    max_samples: int = 16,
) -> dict:
    config = load_config(config_path, data_dir=data_dir)
    if data_dir is not None:
        ensure_dataset_available(config)
        print(f"Using dataset: {config.dataset.data_dir}")
    out = ensure_dir(output_dir or config.explainability.figures_dir)
    model, class_names = load_checkpoint_model(checkpoint_path, config)

    if batch:
        result = export_xai_batch(model, config, class_names, out, max_samples=max_samples)
        if config.sam.enabled:
            unload_sam()
        empty_cuda_cache()
        print(f"Batch XAI: {result['count']} samples -> {result['output_dir']}")
        return result

    if image_path is None:
        raise ValueError("--image is required unless --batch is set")

    result = explain_sample(model, image_path, config, class_names, out)
    if config.sam.enabled:
        unload_sam()
    empty_cuda_cache()
    print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")
    print(f"Grad-CAM: {result['gradcam_path']}")
    print(f"Attention: {result['attention_path']}")
    if result.get("sam_overlay_path"):
        print(f"SAM overlay: {result['sam_overlay_path']}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate XAI visualizations for one MRI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None, help="Single image path (omit with --batch)")
    parser.add_argument("--batch", action="store_true", help="Export XAI for test-set samples")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    add_data_dir_argument(parser)
    args = parser.parse_args(argv)
    run_xai(
        args.config,
        args.checkpoint,
        args.image,
        args.output_dir,
        data_dir=args.data_dir,
        batch=args.batch,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
