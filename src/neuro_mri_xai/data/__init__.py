# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Data engineering: datasets, splits, and augmentation pipelines."""

from neuro_mri_xai.data.constants import EXPECTED_CLASS_NAMES, NUM_CLASSES
from neuro_mri_xai.data.dataset import (
    MRIDataset,
    ensure_dataset_available,
    get_dataloaders,
    resolve_data_dir,
    stratified_split_indices,
)
from neuro_mri_xai.data.transforms import (
    get_test_transforms,
    get_train_transforms,
    get_transforms,
    get_val_transforms,
)

__all__ = [
    "EXPECTED_CLASS_NAMES",
    "NUM_CLASSES",
    "MRIDataset",
    "ensure_dataset_available",
    "get_dataloaders",
    "resolve_data_dir",
    "stratified_split_indices",
    "get_train_transforms",
    "get_val_transforms",
    "get_test_transforms",
    "get_transforms",
]
