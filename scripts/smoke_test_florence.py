#!/usr/bin/env python3
"""Smoke test for Florence-2 integration (local or Kaggle).

Usage:
    python scripts/smoke_test_florence.py
    python scripts/smoke_test_florence.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _resolve_project_dir() -> Path:
    kaggle = Path("/kaggle/working/Neurological-MRI-XAI-Pipeline")
    if kaggle.exists():
        return kaggle
    return PROJECT_DIR


def vram_status() -> dict[str, float | str]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0, "device": "cpu"}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "peak_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "device": torch.cuda.get_device_name(0),
    }


def print_vram(label: str) -> dict[str, float | str]:
    stats = vram_status()
    print(
        f"{label}: allocated={stats['allocated_mb']:.1f} MB, "
        f"reserved={stats['reserved_mb']:.1f} MB "
        f"({stats['device']})"
    )
    return stats


def make_dummy_rgb_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    rng = np.random.default_rng(42)
    array = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def reload_florence_module():
    import neuro_mri_xai.models.florence_reporter as fr

    return importlib.reload(fr)


def run_smoke_test(config_path: Path | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    project_dir = _resolve_project_dir()
    fr = reload_florence_module()
    from neuro_mri_xai.config import Config, load_config

    fr.unload_florence()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("=" * 72)
    print("Florence-2 integration smoke test")
    print(f"Project root: {project_dir}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    print("=" * 72)

    vram_before = print_vram("VRAM before")

    if config_path is None:
        config_path = project_dir / "configs" / "default.yaml"
    config = load_config(config_path) if config_path.exists() else Config()

    image = make_dummy_rgb_image()
    class_names = config.get_class_names()
    predicted_class = class_names[0] if class_names else "Normal"
    confidence = 0.87

    t0 = time.perf_counter()
    outcome = "FAIL"
    detail = ""
    caption: str | None = None
    report = ""

    try:
        model, processor = fr._load_florence(config)
        raw_inputs = fr._build_florence_processor_inputs(
            processor,
            fr.DEFAULT_FLORENCE_CAPTION_TASK,
            image,
            model=model,
        )
        fr._validate_florence_inputs(raw_inputs, processor, model)
        token_count = fr._count_florence_image_tokens(
            processor, raw_inputs["input_ids"], model
        )
        print(
            f"\nInput validation OK: image_tokens={token_count}, "
            f"pixel_values={tuple(raw_inputs['pixel_values'].shape)}"
        )

        fr.unload_florence()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        caption = fr.generate_caption(image, config)
        print("\n--- generate_caption ---")
        if caption:
            preview = caption[:400] + ("..." if len(caption) > 400 else "")
            print(f"Caption ({len(caption)} chars): {preview}")
        else:
            print("Caption: None (graceful degradation)")

        fr.unload_florence()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        report = fr.generate_diagnostic_text(image, predicted_class, confidence, config)
        print("\n--- generate_diagnostic_text ---")
        print(report[:600] + ("..." if len(report) > 600 else ""))

        assert isinstance(report, str) and report.strip()
        assert "Predicted diagnosis:" in report
        assert "DISCLAIMER" in report

        fallback_used = "Florence-2 caption unavailable" in report
        if caption:
            outcome = "PASS"
            detail = "Florence-2 caption generated successfully"
        elif fallback_used:
            outcome = "PASS"
            detail = "Graceful template fallback (no unhandled exception)"
        else:
            outcome = "PASS"
            detail = "Diagnostic text generated"

    except ValueError as exc:
        msg = str(exc)
        if "image tokens" in msg or ("tokens" in msg and "features" in msg):
            detail = f"Image token / feature mismatch: {exc}"
        else:
            detail = f"ValueError: {exc}"
        raise AssertionError(detail) from exc
    except Exception as exc:
        msg = str(exc).lower()
        if "missing" in msg or "unexpected" in msg or "state dict" in msg:
            detail = f"State dict load mismatch: {exc}"
        else:
            detail = f"{type(exc).__name__}: {exc}"
        raise AssertionError(detail) from exc
    finally:
        fr.unload_florence()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.perf_counter() - t0
        vram_after = print_vram("VRAM after")
        if torch.cuda.is_available():
            print(f"VRAM peak:  {vram_after['peak_mb']:.1f} MB")
        print(f"Execution time: {elapsed:.2f}s")
        print(f"\nRESULT: {outcome} — {detail}")
        print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Florence-2 smoke test")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()
    run_smoke_test(config_path=args.config)


if __name__ == "__main__":
    main()
