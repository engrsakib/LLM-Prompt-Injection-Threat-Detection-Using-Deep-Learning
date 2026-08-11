"""Tests for patient-level splits and group ID extraction."""

from __future__ import annotations

from pathlib import Path

from neuro_mri_xai.data.splits import (
    build_group_ids,
    extract_group_id,
    patient_stratified_holdout_split,
    stratified_split_indices,
)


def test_extract_group_id_bt_pattern() -> None:
    path = Path("Te-glTr_0042.jpg")
    assert extract_group_id(path, "BT_glioma") == "BT_glioma/Te-glTr"


def test_extract_group_id_paren_pattern() -> None:
    path = Path("MildImpairment (1002).jpg")
    assert extract_group_id(path, "AD_MildDemented") == "AD_MildDemented/case_1002"


def test_patient_split_has_zero_group_leakage() -> None:
    samples = []
    labels = []
    for case in range(160):
        path = Path(f"Case ({case}).jpg")
        samples.append((path, case % 8))
        labels.append(case % 8)

    groups = build_group_ids(samples, [f"c{i}" for i in range(8)])
    train_idx, val_idx, test_idx = patient_stratified_holdout_split(
        labels, groups, val_split=0.1, test_split=0.1, seed=42
    )

    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    test_groups = {groups[i] for i in test_idx}
    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)


def test_patient_split_falls_back_when_minority_missing_from_test() -> None:
    """One group per class can leave test empty for some labels; fallback must cover all classes."""
    samples = []
    labels = []
    class_names = [f"class_{i}" for i in range(8)]
    for cls in range(8):
        for _ in range(20):
            path = Path("shared_stem.jpg")
            samples.append((path, cls))
            labels.append(cls)

    groups = build_group_ids(samples, class_names)
    train_idx, val_idx, test_idx = patient_stratified_holdout_split(
        labels,
        groups,
        val_split=0.1,
        test_split=0.1,
        seed=42,
        min_class_support=1,
    )

    for split_name, split_idx in ("val", val_idx), ("test", test_idx):
        counts = [0] * 8
        for idx in split_idx:
            counts[labels[idx]] += 1
        for cls, count in enumerate(counts):
            assert count >= 1, f"{split_name} split missing class index {cls} (count={count})"


def test_image_stratified_holdout_covers_all_classes() -> None:
    labels = [i % 8 for i in range(160)]
    train_idx, val_idx, test_idx = stratified_split_indices(
        labels,
        val_split=0.1,
        test_split=0.1,
        seed=42,
        split_strategy="stratified",
        min_class_support=1,
    )
    assert len(train_idx) + len(val_idx) + len(test_idx) == 160
    for split_idx in (val_idx, test_idx):
        present = {labels[i] for i in split_idx}
        assert present == set(range(8))


def test_stratified_kfold_covers_all_indices_once() -> None:
    labels = [i % 8 for i in range(400)]
    groups = [f"group_{i // 5}" for i in range(400)]
    covered: set[int] = set()
    for fold in range(5):
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels,
            val_split=0.1,
            test_split=0.0,
            seed=42,
            groups=groups,
            split_strategy="patient",
            n_folds=5,
            fold_index=fold,
        )
        assert test_idx == []
        overlap = set(train_idx) & set(val_idx)
        assert not overlap
        covered.update(train_idx)
        covered.update(val_idx)
    assert covered == set(range(len(labels)))
