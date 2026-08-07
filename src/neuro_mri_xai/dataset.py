"""Dataset loading, transforms, and stratified splits."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from neuro_mri_xai.config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(image_size: int, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MRIDataset(Dataset):
    """ImageFolder-style dataset with optional SAM ROI preprocessing."""

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        roi_fn: Callable[[Image.Image], Image.Image] | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.roi_fn = roi_fn
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.classes: list[str] = []

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}. Run scripts/download_data.py first."
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


def _stratified_split_indices(
    labels: list[int],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
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
    data_dir = config.dataset.data_dir
    image_size = config.dataset.image_size
    batch_size = config.dataset.batch_size
    num_workers = config.dataset.num_workers
    use_roi = roi_fn if config.sam.enabled else None

    base_dataset = MRIDataset(root=data_dir, transform=None, roi_fn=use_roi)
    class_names = base_dataset.classes
    labels = [label for _, label in base_dataset.samples]

    train_idx, val_idx, test_idx = _stratified_split_indices(
        labels,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        seed=config.dataset.seed,
    )

    train_ds = Subset(
        MRIDataset(data_dir, get_transforms(image_size, train=True), roi_fn=use_roi),
        train_idx,
    )
    val_ds = Subset(
        MRIDataset(data_dir, get_transforms(image_size, train=False), roi_fn=use_roi),
        val_idx,
    )
    test_ds = Subset(
        MRIDataset(data_dir, get_transforms(image_size, train=False), roi_fn=use_roi),
        test_idx,
    )

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, class_names
