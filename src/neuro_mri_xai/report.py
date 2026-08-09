# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""HTML diagnostic reports combining classification, XAI, and Florence-2."""

from __future__ import annotations

import argparse
import base64
import html
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data.dataset import ensure_dataset_available
from neuro_mri_xai.evaluation.checkpoint import load_checkpoint_model
from neuro_mri_xai.explainability.pipeline import explain_sample
from neuro_mri_xai.models.florence_reporter import (
    generate_diagnostic_text,
    template_diagnostic_text,
    unload_florence,
)
from neuro_mri_xai.models.sam_roi import unload_sam
from neuro_mri_xai.utils.cli import add_data_dir_argument
from neuro_mri_xai.utils.paths import ensure_dir
from neuro_mri_xai.utils.vram import empty_cuda_cache, log_gpu_mem


def _img_to_base64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_html_report(
    image_path: Path,
    prediction: str,
    confidence: float,
    diagnostic_text: str,
    figure_paths: dict[str, str | None],
) -> str:
    figures_html = ""
    for label, path in figure_paths.items():
        if path and Path(path).exists():
            b64 = _img_to_base64(path)
            safe_label = html.escape(label.replace("_", " ").title())
            figures_html += f'<div class="figure"><h3>{safe_label}</h3>'
            figures_html += (
                f'<img src="data:image/png;base64,{b64}" alt="{safe_label}"/></div>'
            )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    safe_prediction = html.escape(prediction)
    safe_image_name = html.escape(image_path.name)
    safe_diagnostic = html.escape(diagnostic_text)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>MRI Diagnostic Report — {safe_prediction}</title>
<style>
body {{ font-family: Georgia, serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ color: #1a365d; }}
.meta {{ background: #edf2f7; padding: 1rem; border-radius: 8px; }}
.figure {{ margin: 1.5rem 0; text-align: center; }}
.figure img {{ max-width: 100%; border: 1px solid #cbd5e0; border-radius: 4px; }}
.disclaimer {{ color: #718096; font-size: 0.85rem; margin-top: 2rem; }}
pre {{ white-space: pre-wrap; line-height: 1.6; }}
</style></head><body>
<h1>Neurological MRI Diagnostic Report</h1>
<p>Generated: {ts}</p>
<div class="meta">
<p><strong>Source image:</strong> {safe_image_name}</p>
<p><strong>Predicted class:</strong> {safe_prediction}</p>
<p><strong>Confidence:</strong> {confidence:.1%}</p>
</div>
<h2>Clinical Summary</h2><pre>{safe_diagnostic}</pre>
<h2>Explainability Visualizations</h2>{figures_html}
<p class="disclaimer">This report is AI-generated for research and interpretability purposes only.
It is not a substitute for professional medical diagnosis.</p>
</body></html>"""


def generate_report(
    checkpoint: str | Path,
    image: str | Path,
    config_path: str = "configs/default.yaml",
    output_dir: str | Path | None = None,
    skip_florence: bool = False,
    data_dir: str | None = None,
) -> Path:
    config = load_config(config_path, data_dir=data_dir)
    if data_dir is not None:
        ensure_dataset_available(config)
        print(f"Using dataset: {config.dataset.data_dir}")
    output_dir = ensure_dir(output_dir or config.report.output_dir)
    image_path = Path(image)
    model, class_names = load_checkpoint_model(checkpoint, config)
    xai_dir = ensure_dir(output_dir / "xai_cache")
    xai = explain_sample(model, image_path, config, class_names, xai_dir)
    log_gpu_mem("post-XAI")

    pil_image = Image.open(image_path).convert("RGB")
    use_florence = config.florence.enabled and not skip_florence
    if use_florence:
        try:
            diagnostic_text = generate_diagnostic_text(
                pil_image,
                xai["prediction"],
                xai["confidence"],
                config,
            )
        except Exception as exc:
            diagnostic_text = template_diagnostic_text(
                xai["prediction"],
                xai["confidence"],
                config,
                florence_unavailable=True,
            )
            print(f"Warning: Florence report generation failed ({exc}); using template fallback.")
        finally:
            unload_florence()
            empty_cuda_cache()
            log_gpu_mem("Florence unloaded")
    else:
        diagnostic_text = f"Predicted: {xai['prediction']} ({xai['confidence']:.1%})"
    if config.sam.enabled:
        unload_sam()
        empty_cuda_cache()
    report_path = output_dir / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
    report_path.write_text(
        build_html_report(
            image_path,
            xai["prediction"],
            xai["confidence"],
            diagnostic_text,
            {
                "gradcam": xai.get("gradcam_path"),
                "attention_saliency": xai.get("attention_path"),
                "sam_constrained_overlay": xai.get("sam_overlay_path"),
            },
        ),
        encoding="utf-8",
    )
    print(f"Report saved to {report_path}")
    return report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate HTML diagnostic report")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-florence", action="store_true")
    add_data_dir_argument(parser)
    args = parser.parse_args(argv)
    generate_report(
        args.checkpoint,
        args.image,
        args.config,
        args.output_dir,
        args.skip_florence,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
