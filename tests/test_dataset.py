"""Tests for dataset splits and transforms."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from neuro_mri_xai.data.dataset import MRIDataset, stratified_split_indices
from neuro_mri_xai.data.transforms import get_transforms


def _make_fake_dataset(root: Path, n_per_class: int = 10, n_classes: int = 8) -> None:
    for i in range(n_classes):
        cls_dir = root / f"class_{i}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j in range(n_per_class):
            Image.new("RGB", (64, 64), color=(j * 10 % 255, 50, 100)).save(cls_dir / f"img_{j}.jpg")


def test_mri_dataset_loads_samples(tmp_path: Path) -> None:
    _make_fake_dataset(tmp_path)
    ds = MRIDataset(tmp_path, transform=get_transforms(224, train=False))
    assert len(ds) == 80
    assert len(ds.classes) == 8
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert 0 <= label < 8


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
        == len(get_transforms(224, train=False).transforms) + 2
    )
