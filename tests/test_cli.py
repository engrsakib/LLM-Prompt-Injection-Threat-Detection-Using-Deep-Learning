# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for CLI data-dir override helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from neuro_mri_xai.config import Config, DatasetConfig
from neuro_mri_xai.utils.cli import apply_data_dir_override


def _make_imagefolder(root: Path, classes: list[str]) -> Path:
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32)).save(cls_dir / "img.jpg")
    return root


def test_apply_data_dir_override_nested(tmp_path: Path) -> None:
    inner = _make_imagefolder(tmp_path / "bundle" / "data", ["A", "B"])
    cfg = Config(dataset=DatasetConfig(data_dir=Path("data")))
    apply_data_dir_override(cfg, tmp_path / "bundle")
    assert cfg.dataset.data_dir == inner.resolve()


def test_apply_data_dir_override_direct(tmp_path: Path) -> None:
    root = _make_imagefolder(tmp_path / "data", ["X"])
    cfg = Config(dataset=DatasetConfig(data_dir=Path("data")))
    apply_data_dir_override(cfg, root)
    assert cfg.dataset.data_dir == root.resolve()


def test_apply_data_dir_override_missing_defers_to_kagglehub(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    cfg = Config(dataset=DatasetConfig(data_dir=Path("data")))
    apply_data_dir_override(cfg, missing)
    assert cfg.dataset.data_dir == missing.resolve()
