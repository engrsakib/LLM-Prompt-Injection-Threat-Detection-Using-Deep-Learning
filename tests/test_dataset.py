"""Tests for dataset splits and transforms."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image

from neuro_mri_xai.config import Config, DatasetConfig
from neuro_mri_xai.data.constants import EXPECTED_CLASS_NAMES, NUM_CLASSES
from neuro_mri_xai.data.dataset import (
    MRIDataset,
    compute_class_weights,
    ensure_dataset_available,
    resolve_data_dir,
    stratified_split_indices,
)
from neuro_mri_xai.data.transforms import get_transforms
from neuro_mri_xai.utils.paths import resolve_dataset_root, resolve_imagefolder_root


def _make_fake_dataset(root: Path, n_per_class: int = 10) -> None:
    for class_name in EXPECTED_CLASS_NAMES:
        cls_dir = root / class_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j in range(n_per_class):
            Image.new("RGB", (64, 64), color=(j * 10 % 255, 50, 100)).save(
                cls_dir / f"img_{j}.jpg",
            )


def test_mri_dataset_loads_samples(tmp_path: Path) -> None:
    _make_fake_dataset(tmp_path)
    ds = MRIDataset(tmp_path, transform=get_transforms(224, train=False))
    assert len(ds) == NUM_CLASSES * 10
    assert len(ds.classes) == NUM_CLASSES
    assert ds.classes == sorted(EXPECTED_CLASS_NAMES)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert 0 <= label < NUM_CLASSES


def test_mri_dataset_rejects_wrong_class_count(tmp_path: Path) -> None:
    (tmp_path / "OnlyOneClass").mkdir()
    (tmp_path / "OnlyOneClass" / "a.jpg").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="Expected 8 class"):
        MRIDataset(tmp_path)


def test_resolve_imagefolder_outer_root_to_data(tmp_path: Path) -> None:
    inner = tmp_path / "neurological-disorders-mri-dataset-for-xai" / "data"
    _make_fake_dataset(inner, n_per_class=1)
    outer = tmp_path / "neurological-disorders-mri-dataset-for-xai"
    resolved = resolve_imagefolder_root(outer)
    assert resolved == inner.resolve()


def test_resolve_data_dir_navigates_data_subfolder(tmp_path: Path) -> None:
    inner = tmp_path / "bundle" / "data"
    _make_fake_dataset(inner, n_per_class=1)
    cfg = Config(dataset=DatasetConfig(data_dir=tmp_path / "bundle"))
    assert resolve_data_dir(cfg) == inner.resolve()


def test_resolve_dataset_root_prefers_data_subfolder(tmp_path: Path) -> None:
    inner = tmp_path / "mount" / "data"
    _make_fake_dataset(inner, n_per_class=1)
    outer = tmp_path / "mount"
    assert resolve_dataset_root(outer) == inner.resolve()


def test_resolve_dataset_root_flat_class_layout(tmp_path: Path) -> None:
    outer = tmp_path / "mount"
    _make_fake_dataset(outer, n_per_class=1)
    assert resolve_dataset_root(outer) == outer.resolve()


def test_ensure_dataset_available_updates_config(tmp_path: Path) -> None:
    inner = tmp_path / "bundle" / "data"
    _make_fake_dataset(inner, n_per_class=1)
    cfg = Config(dataset=DatasetConfig(data_dir=tmp_path / "bundle"))
    resolved = ensure_dataset_available(cfg)
    assert resolved == inner.resolve()
    assert cfg.dataset.data_dir == inner.resolve()


def test_ensure_dataset_available_missing_raises(tmp_path: Path) -> None:
    cfg = Config(
        dataset=DatasetConfig(
            data_dir=tmp_path / "missing",
            kagglehub_fallback_handle="owner/fake-dataset",
        ),
    )
    hub_root = tmp_path / "hub" / "data"
    _make_fake_dataset(hub_root, n_per_class=1)

    with mock.patch(
        "neuro_mri_xai.data.download.resolve_kagglehub_dataset",
        return_value=hub_root,
    ):
        resolved = ensure_dataset_available(cfg)

    assert resolved == hub_root.resolve()
    assert cfg.dataset.data_dir == hub_root.resolve()


def test_stratified_split_sizes() -> None:
    labels = [i % 8 for i in range(800)]
    train_idx, val_idx, test_idx = stratified_split_indices(labels, 0.1, 0.1, 42)
    assert len(train_idx) == 640 and len(val_idx) == 80 and len(test_idx) == 80
    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))


def test_stratified_split_16400() -> None:
    labels = [i % 8 for i in range(16400)]
    train_idx, val_idx, test_idx = stratified_split_indices(labels, 0.1, 0.1, 42)
    total = len(train_idx) + len(val_idx) + len(test_idx)
    assert total == 16400
    assert abs(len(test_idx) - 1640) <= 1
    assert abs(len(val_idx) - 1640) <= 1
    assert abs(len(train_idx) - 13120) <= 1


def test_get_transforms_train_vs_eval() -> None:
    assert (
        len(get_transforms(224, train=True).transforms)
        == len(get_transforms(224, train=False).transforms) + 3
    )


def test_compute_class_weights_inverse_frequency() -> None:
    labels = [0, 0, 0, 0, 1, 1]
    weights = compute_class_weights(labels, num_classes=2)
    assert weights.shape == (2,)
    assert weights[0] < weights[1]
    assert torch.isclose(weights[0], torch.tensor(6.0 / (2.0 * 4.0)))
    assert torch.isclose(weights[1], torch.tensor(6.0 / (2.0 * 2.0)))


def test_compute_per_class_metrics() -> None:
    import numpy as np

    from neuro_mri_xai.evaluation.metrics import compute_per_class_metrics

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = compute_per_class_metrics(y_true, y_pred, ["A", "B"])
    assert len(metrics) == 2
    assert metrics[0]["class"] == "A"
    assert "precision" in metrics[0]
    assert metrics[1]["recall"] == 1.0


def test_compute_metrics_handles_missing_classes_in_report() -> None:
    import numpy as np

    from neuro_mri_xai.evaluation.metrics import compute_metrics

    class_names = ["A", "B", "C"]
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.4, 0.5, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
        ],
    )

    result = compute_metrics(y_true, y_pred, y_prob, class_names)
    assert "C" in result["classification_report"]
    assert result["per_class"][2]["support"] == 0
