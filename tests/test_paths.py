"""Tests for path resolution utilities."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data.constants import EXPECTED_CLASS_NAMES
from neuro_mri_xai.utils.paths import (
    RuntimeEnv,
    detect_runtime_env,
    ensure_dir,
    get_data_root,
    get_project_root,
    resolve_imagefolder_root,
    resolve_kaggle_dataset_root,
    resolve_path,
)


def test_get_project_root_finds_configs():
    root = get_project_root()
    assert (root / "configs" / "default.yaml").exists()


def test_load_config_class_names() -> None:
    cfg = load_config()
    assert cfg.model.num_classes == 8
    assert cfg.get_class_names() == EXPECTED_CLASS_NAMES
    assert cfg.dataset.class_names == EXPECTED_CLASS_NAMES


def test_resolve_path_relative():
    root = get_project_root()
    resolved = resolve_path("outputs/checkpoints", root)
    assert resolved.is_absolute()
    assert resolved.parent.name == "outputs"


def test_ensure_dir(tmp_path):
    target = tmp_path / "nested" / "dir"
    result = ensure_dir(target)
    assert result.exists() and result.is_dir()


def test_detect_local_env():
    with mock.patch("neuro_mri_xai.utils.paths.os.path.exists", return_value=False):
        assert detect_runtime_env() == RuntimeEnv.LOCAL


def test_env_override_project_root(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("dataset:\n  source: kaggle\n")
    with mock.patch.dict(os.environ, {"NEURO_MRI_PROJECT_ROOT": str(tmp_path)}):
        assert get_project_root() == tmp_path.resolve()


def test_get_data_root_env_override(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for cls in EXPECTED_CLASS_NAMES:
        cls_dir = data_root / cls
        cls_dir.mkdir(parents=True)
        (cls_dir / "img.jpg").write_bytes(b"x")
    with mock.patch.dict(os.environ, {"NEURO_MRI_DATA_DIR": str(data_root)}):
        assert get_data_root() == data_root.resolve()


def test_resolve_imagefolder_root_nested(tmp_path: Path) -> None:
    inner = tmp_path / "bundle" / "data" / "Normal"
    inner.mkdir(parents=True)
    (inner / "img.jpg").write_bytes(b"x")
    resolved = resolve_imagefolder_root(tmp_path / "bundle")
    assert resolved == (tmp_path / "bundle" / "data").resolve()


def test_resolve_kaggle_dataset_root_from_fixture(tmp_path: Path) -> None:
    mount = tmp_path / "datasets" / "engrsakib02" / "neurological-disorders-mri-dataset-for-xai"
    data_dir = mount / "data"
    for cls in EXPECTED_CLASS_NAMES:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        (cls_dir / "img.jpg").write_bytes(b"x")
    resolved = resolve_kaggle_dataset_root(mount)
    assert resolved == data_dir.resolve()
