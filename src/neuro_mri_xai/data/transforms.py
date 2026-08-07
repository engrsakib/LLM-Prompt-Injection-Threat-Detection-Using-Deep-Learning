# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Train, validation, and test augmentation pipelines."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_test_transforms(image_size: int) -> transforms.Compose:
    """Test pipeline — no random augmentation (matches validation)."""
    return get_val_transforms(image_size)


def get_transforms(image_size: int, train: bool = False) -> transforms.Compose:
    """Backward-compatible helper."""
    return get_train_transforms(image_size) if train else get_val_transforms(image_size)
