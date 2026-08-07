# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for dataset download and ImageFolder path resolution."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from neuro_mri_xai.config import Config, DatasetConfig
from neuro_mri_xai.data.download import download_dataset
from neuro_mri_xai.utils.paths import find_kagglehub_cache, resolve_imagefolder_root


def _make_imagefolder(root: Path, classes: list[str], n_per_class: int = 2) -> Path:
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            Image.new("RGB", (32, 32), color=(100, 50, i * 30)).save(cls_dir / f"{i}.jpg")
    return root


def test_resolve_imagefolder_root_direct(tmp_path: Path) -> None:
    data_root = _make_imagefolder(tmp_path / "data", ["A", "B"])
    assert resolve_imagefolder_root(data_root) == data_root.resolve()


def test_resolve_imagefolder_root_nested_data(tmp_path: Path) -> None:
    inner = _make_imagefolder(tmp_path / "bundle" / "data", ["X", "Y"])
    assert resolve_imagefolder_root(tmp_path / "bundle") == inner.resolve()


def test_find_kagglehub_cache_missing() -> None:
    assert find_kagglehub_cache("nonexistent-user/nonexistent-dataset") is None


def test_download_dataset_falls_back_to_kaggle(tmp_path: Path) -> None:
    cfg = Config(
        dataset=DatasetConfig(
            source="kagglehub",
            kagglehub_handle="owner/fake-dataset",
            kaggle_dataset="owner/fallback-dataset",
        ),
    )

    fallback_root = _make_imagefolder(tmp_path / "kaggle_out" / "data", ["Normal", "MS"])

    with (
        mock.patch(
            "neuro_mri_xai.data.download.download_kagglehub",
            side_effect=RuntimeError("network error"),
        ),
        mock.patch(
            "neuro_mri_xai.data.download.download_kaggle",
            return_value=fallback_root,
        ) as mock_kaggle,
    ):
        path = download_dataset(cfg, output=tmp_path / "kaggle_out")

    mock_kaggle.assert_called_once()
    assert path == fallback_root.resolve()


def test_download_dataset_use_kagglehub_flag(tmp_path: Path) -> None:
    cfg = Config(
        dataset=DatasetConfig(
            source="kaggle",
            kagglehub_handle="owner/primary-dataset",
            kaggle_dataset="owner/fallback-dataset",
        ),
    )
    hub_root = _make_imagefolder(tmp_path / "hub", ["A", "B"])

    with mock.patch(
        "neuro_mri_xai.data.download.download_kagglehub",
        return_value=hub_root,
    ) as mock_hub:
        path = download_dataset(cfg, use_kagglehub=True)

    mock_hub.assert_called_once_with("owner/primary-dataset")
    assert path == hub_root.resolve()
