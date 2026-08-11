# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Patient/group-aware stratified splits and k-fold cross-validation."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

logger = logging.getLogger(__name__)


def extract_group_id(path: Path, class_name: str) -> str:
    """Derive a patient/case group ID from filename heuristics (class-aware).

    Groups slices from the same subject/case so they never span train and test.
    """
    stem = path.stem

    paren_match = re.match(r"^(.+?)\s*\((\d+)\)$", stem)
    if paren_match:
        return f"{class_name}/case_{paren_match.group(2)}"

    bt_match = re.match(r"^(.+?)_\d+$", stem)
    if bt_match:
        return f"{class_name}/{bt_match.group(1)}"

    prefix_match = re.sub(r"_?\d+$", "", stem)
    if prefix_match != stem:
        return f"{class_name}/{prefix_match}"

    return f"{class_name}/{stem}"


def build_group_ids(samples: list[tuple[Path, int]], class_names: list[str]) -> list[str]:
    """Return one group ID per sample."""
    groups: list[str] = []
    for path, label in samples:
        class_name = class_names[label] if label < len(class_names) else str(label)
        groups.append(extract_group_id(path, class_name))
    return groups


def _expand_group_indices(group_to_indices: dict[str, list[int]], group_names: list[str]) -> list[int]:
    indices: list[int] = []
    for group in group_names:
        indices.extend(group_to_indices[group])
    return indices


def _split_groups(
    groups: list[str],
    group_labels: list[int],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified group split with fallback when too few groups exist."""
    n_classes = len(set(group_labels))
    if len(groups) * test_size < n_classes:
        logger.warning(
            "Only %d groups for %d classes; using non-stratified group split (test_size=%.2f)",
            len(groups),
            n_classes,
            test_size,
        )
        return train_test_split(
            groups,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    return train_test_split(
        groups,
        test_size=test_size,
        stratify=group_labels,
        random_state=seed,
    )


def _split_class_counts(labels: list[int], indices: list[int], num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for idx in indices:
        label = labels[idx]
        if 0 <= label < num_classes:
            counts[label] += 1
    return counts


def _classes_missing_support(
    labels: list[int],
    indices: list[int],
    num_classes: int,
    min_support: int,
) -> list[int]:
    counts = _split_class_counts(labels, indices, num_classes)
    return [cls_idx for cls_idx, count in enumerate(counts) if count < min_support]


def image_stratified_holdout_split(
    labels: list[int],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Image-level stratified 80/10/10 split (may include slice leakage across groups)."""
    indices = list(range(len(labels)))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        stratify=labels,
        random_state=seed,
    )
    val_ratio = val_split / max(1.0 - test_split, 1e-6)
    train_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        stratify=train_labels,
        random_state=seed,
    )
    pseudo_groups = [str(i) for i in range(len(labels))]
    _log_split_stats(
        "image stratified holdout",
        pseudo_groups,
        labels,
        train_idx,
        val_idx,
        test_idx,
    )
    return train_idx, val_idx, test_idx


def _patient_split_needs_fallback(
    labels: list[int],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    min_class_support: int,
) -> list[int]:
    """Return class indices under-represented in val or test after patient split."""
    if min_class_support <= 0:
        return []
    num_classes = len(set(labels))
    missing: set[int] = set()
    for split_indices in (val_idx, test_idx):
        missing.update(
            _classes_missing_support(labels, split_indices, num_classes, min_class_support)
        )
    return sorted(missing)


def patient_stratified_holdout_split(
    labels: list[int],
    groups: list[str],
    val_split: float,
    test_split: float,
    seed: int,
    min_class_support: int = 1,
) -> tuple[list[int], list[int], list[int]]:
    """Split at group level with class stratification (zero patient leakage)."""
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    group_to_label: dict[str, int] = {}
    for idx, (group, label) in enumerate(zip(groups, labels, strict=True)):
        group_to_indices[group].append(idx)
        group_to_label[group] = label

    all_groups = list(group_to_indices.keys())
    group_labels = [group_to_label[g] for g in all_groups]

    if test_split <= 0:
        train_val_groups = all_groups
        test_groups: list[str] = []
    else:
        train_val_groups, test_groups = _split_groups(
            all_groups,
            group_labels,
            test_split,
            seed,
        )
    train_val_group_labels = [group_to_label[g] for g in train_val_groups]
    val_ratio = val_split / max(1.0 - test_split, 1e-6)
    if len(train_val_groups) * val_ratio >= len(set(train_val_group_labels)):
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=val_ratio,
            stratify=train_val_group_labels,
            random_state=seed,
        )
    else:
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )

    train_idx = _expand_group_indices(group_to_indices, train_groups)
    val_idx = _expand_group_indices(group_to_indices, val_groups)
    test_idx = _expand_group_indices(group_to_indices, test_groups)

    missing = _patient_split_needs_fallback(
        labels,
        train_idx,
        val_idx,
        test_idx,
        min_class_support,
    )
    if missing:
        logger.warning(
            "Patient split left classes with <%d val/test samples: %s; "
            "falling back to image-level stratified split",
            min_class_support,
            missing,
        )
        return image_stratified_holdout_split(labels, val_split, test_split, seed)

    _log_split_stats("patient holdout", groups, labels, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


def patient_stratified_kfold_split(
    labels: list[int],
    groups: list[str],
    n_folds: int,
    fold_index: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified group k-fold: train / val for fold; test indices empty."""
    if fold_index < 0 or fold_index >= n_folds:
        raise ValueError(f"fold_index must be in [0, {n_folds - 1}], got {fold_index}")

    labels_arr = np.array(labels)
    groups_arr = np.array(groups)
    indices = np.arange(len(labels))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (train_idx, val_idx) in enumerate(sgkf.split(indices, labels_arr, groups_arr)):
        if i == fold_index:
            train_list = train_idx.tolist()
            val_list = val_idx.tolist()
            _log_split_stats(f"k-fold {fold_index + 1}/{n_folds}", groups, labels, train_list, val_list, [])
            return train_list, val_list, []

    raise RuntimeError("StratifiedGroupKFold produced no splits")


def stratified_split_indices(
    labels: list[int],
    val_split: float,
    test_split: float,
    seed: int,
    groups: list[str] | None = None,
    split_strategy: str = "image",
    n_folds: int = 5,
    fold_index: int = 0,
    min_class_support: int = 1,
) -> tuple[list[int], list[int], list[int]]:
    """Dispatch image-level or patient-level stratified splitting."""
    strategy = split_strategy.lower()
    if strategy == "patient":
        if groups is None:
            raise ValueError("Patient-level split requires group IDs for each sample")
        if n_folds > 1:
            return patient_stratified_kfold_split(labels, groups, n_folds, fold_index, seed)
        return patient_stratified_holdout_split(
            labels,
            groups,
            val_split,
            test_split,
            seed,
            min_class_support=min_class_support,
        )

    if strategy == "stratified":
        strategy = "image"

    if strategy != "image":
        logger.warning("Unknown split_strategy %r; using image-level stratified split", strategy)

    return image_stratified_holdout_split(labels, val_split, test_split, seed)


def _log_split_stats(
    name: str,
    groups: list[str],
    labels: list[int],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> None:
    def _group_set(idxs: list[int]) -> set[str]:
        return {groups[i] for i in idxs}

    train_g = _group_set(train_idx)
    val_g = _group_set(val_idx)
    test_g = _group_set(test_idx)
    overlap_tv = train_g & val_g
    overlap_tt = train_g & test_g
    overlap_vt = val_g & test_g
    if overlap_tv or overlap_tt or overlap_vt:
        logger.warning(
            "%s split has group leakage: train∩val=%d train∩test=%d val∩test=%d",
            name,
            len(overlap_tv),
            len(overlap_tt),
            len(overlap_vt),
        )
    else:
        logger.info(
            "%s split: %d train / %d val / %d test images "
            "(%d / %d / %d unique groups, zero leakage)",
            name,
            len(train_idx),
            len(val_idx),
            len(test_idx),
            len(train_g),
            len(val_g),
            len(test_g),
        )
