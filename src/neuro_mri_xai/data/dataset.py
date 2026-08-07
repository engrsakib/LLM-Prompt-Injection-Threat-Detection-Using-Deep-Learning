# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""PyTorch ImageFolder wrapper with stratified 80/10/10 splits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from neuro_mri_xai.config import Config
from neuro_mri_xai.data.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)
from neuro_mri_xai.utils.paths import resolve_imagefolder_root


def resolve_data_dir(config: Config) -> Path:
    """Resolve ImageFolder root from config, validating layout when present."""
    data_dir = Path(config.dataset.data_dir)
    if data_dir.exists():
        resolved = resolve_imagefolder_root(data_dir)
        if resolved is not None:
            return resolved
    return data_dir


class MRIDataset(Dataset):
    """ImageFolder-style dataset with optional SAM ROI preprocessing."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        roi_fn: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.roi_fn = roi_fn
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.classes: list[str] = []

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}. "
                "Run: python scripts/download_data.py --use-kagglehub "
                "or python scripts/download_data.py --source kaggle",
            )

        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = self.root / class_name
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                for path in class_dir.glob(ext):
                    self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.roi_fn is not None:
            image = self.roi_fn(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def stratified_split_indices(
    labels: list[int],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices into train/val/test with stratification (default 80/10/10)."""
    indices = list(range(len(labels)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, stratify=labels, random_state=seed,
    )
    val_ratio = val_split / (1.0 - test_split)
    train_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, stratify=train_labels, random_state=seed,
    )
    return train_idx, val_idx, test_idx


def get_dataloaders(
    config: Config,
    roi_fn: Callable[[Image.Image], Image.Image] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    data_dir = resolve_data_dir(config)
    image_size = config.dataset.image_size
    use_roi = roi_fn if config.sam.enabled else None

    base_dataset = MRIDataset(root=data_dir, transform=None, roi_fn=use_roi)
    class_names = base_dataset.classes
    labels = [label for _, label in base_dataset.samples]

    train_idx, val_idx, test_idx = stratified_split_indices(
        labels,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        seed=config.dataset.seed,
    )

    train_ds = Subset(
        MRIDataset(data_dir, get_train_transforms(image_size), roi_fn=use_roi),
        train_idx,
    )
    val_ds = Subset(
        MRIDataset(data_dir, get_val_transforms(image_size), roi_fn=use_roi),
        val_idx,
    )
    test_ds = Subset(
        MRIDataset(data_dir, get_test_transforms(image_size), roi_fn=use_roi),
        test_idx,
    )

    loader_kwargs: dict = {
        "batch_size": config.dataset.batch_size,
        "num_workers": config.dataset.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.dataset.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, class_names
