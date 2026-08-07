# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""PyTorch ImageFolder wrapper with stratified 80/10/10 splits."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from neuro_mri_xai.data.constants import (
    DEFAULT_KAGGLE_DATA_DIR,
    DEFAULT_KAGGLEHUB_FALLBACK_HANDLE,
    EXPECTED_CLASS_NAMES,
    NUM_CLASSES,
)
from neuro_mri_xai.data.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)
from neuro_mri_xai.utils.paths import (
    resolve_dataset_root,
    resolve_imagefolder_root,
    resolve_kaggle_dataset_root,
)

if TYPE_CHECKING:
    from neuro_mri_xai.config import Config

logger = logging.getLogger(__name__)


def _local_dataset_root(config: Config) -> Path | None:
    """Return a valid local ImageFolder root when one exists on disk."""
    data_dir = resolve_data_dir(config)
    if not data_dir.is_dir():
        return None
    return resolve_imagefolder_root(data_dir)


def _kagglehub_fallback_handle(config: Config) -> str:
    handle = config.dataset.kagglehub_fallback_handle or DEFAULT_KAGGLEHUB_FALLBACK_HANDLE
    return handle.strip()


def _resolve_via_kagglehub(config: Config) -> Path:
    """Download or locate dataset via kagglehub when local paths are unavailable."""
    from neuro_mri_xai.data.download import resolve_kagglehub_dataset

    handle = _kagglehub_fallback_handle(config)
    logger.info("Local dataset unavailable; resolving via kagglehub (%s)", handle)
    try:
        return resolve_kagglehub_dataset(handle)
    except ImportError as exc:
        raise FileNotFoundError(
            "Dataset not found locally and kagglehub is not installed. "
            f"Install kagglehub or pass --data-dir. ({exc})",
        ) from exc
    except Exception as exc:
        raise FileNotFoundError(
            f"Dataset not found locally and kagglehub download failed for '{handle}': {exc}",
        ) from exc


def resolve_data_dir(config: Config) -> Path:
    """Resolve ImageFolder root with Kaggle mount and data/ subfolder fallbacks."""
    candidate = Path(config.dataset.data_dir)

    if candidate.is_dir():
        resolved = resolve_dataset_root(candidate)
        if resolved is not None:
            return resolved

    kaggle = resolve_kaggle_dataset_root()
    if kaggle is not None:
        return kaggle

    if not candidate.is_absolute():
        project_path = config.project_root / candidate
        if project_path.is_dir():
            resolved = resolve_dataset_root(project_path)
            if resolved is not None:
                return resolved

    return candidate


def ensure_dataset_available(config: Config) -> Path:
    """Resolve dataset path locally, or auto-download via kagglehub fallback."""
    local = _local_dataset_root(config)
    if local is not None:
        config.dataset.data_dir = local
        return local

    resolved = _resolve_via_kagglehub(config)
    if resolve_imagefolder_root(resolved) is None:
        raise FileNotFoundError(
            f"kagglehub resolved to {resolved}, but no valid ImageFolder layout was found. "
            f"Expected 8 class subdirectories ({', '.join(EXPECTED_CLASS_NAMES[:3])}, ...).",
        )
    config.dataset.data_dir = resolved
    logger.info("Using dataset at %s", resolved)
    return resolved


def _validate_class_folders(class_names: list[str], root: Path) -> None:
    if len(class_names) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} class subdirectories under {root}, found {len(class_names)}: "
            f"{class_names}",
        )
    missing = [name for name in EXPECTED_CLASS_NAMES if name not in class_names]
    if missing:
        raise RuntimeError(
            f"Missing expected class folders under {root}: {missing}. Found: {class_names}",
        )


class MRIDataset(Dataset):
    """ImageFolder-style dataset with optional SAM ROI preprocessing.

    Expects class-labelled subdirectories directly under ``root`` (typically the
    ``data/`` folder):

        root/
        ├── AD_MildDemented/
        ├── ...
        └── Normal/
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        roi_fn: Callable[[Image.Image], Image.Image] | None = None,
        expected_classes: list[str] | None = None,
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
                "Pass --data-dir pointing to the folder containing class subdirectories "
                f"(e.g. {DEFAULT_KAGGLE_DATA_DIR}).",
            )

        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        _validate_class_folders(self.classes, self.root)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        if expected_classes:
            expected_sorted = sorted(expected_classes)
            if self.classes != expected_sorted:
                raise RuntimeError(
                    f"Class folder order mismatch under {self.root}. "
                    f"Expected (sorted): {expected_sorted}, found: {self.classes}",
                )

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
        indices,
        test_size=test_split,
        stratify=labels,
        random_state=seed,
    )
    val_ratio = val_split / (1.0 - test_split)
    train_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        stratify=train_labels,
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def get_dataloaders(
    config: Config,
    roi_fn: Callable[[Image.Image], Image.Image] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    data_dir = ensure_dataset_available(config)
    image_size = config.dataset.image_size
    use_roi = roi_fn if config.sam.enabled else None
    expected_classes = config.get_class_names()

    base_dataset = MRIDataset(
        root=data_dir,
        transform=None,
        roi_fn=use_roi,
        expected_classes=expected_classes,
    )
    class_names = base_dataset.classes
    labels = [label for _, label in base_dataset.samples]

    train_idx, val_idx, test_idx = stratified_split_indices(
        labels,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        seed=config.dataset.seed,
    )

    train_ds = Subset(
        MRIDataset(
            data_dir,
            get_train_transforms(image_size),
            roi_fn=use_roi,
            expected_classes=expected_classes,
        ),
        train_idx,
    )
    val_ds = Subset(
        MRIDataset(
            data_dir,
            get_val_transforms(image_size),
            roi_fn=use_roi,
            expected_classes=expected_classes,
        ),
        val_idx,
    )
    test_ds = Subset(
        MRIDataset(
            data_dir,
            get_test_transforms(image_size),
            roi_fn=use_roi,
            expected_classes=expected_classes,
        ),
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
