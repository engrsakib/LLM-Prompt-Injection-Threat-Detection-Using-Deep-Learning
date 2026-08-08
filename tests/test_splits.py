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
