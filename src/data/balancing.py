"""Minority-class balancing strategies for training data."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def class_counts(df: pd.DataFrame, label_col: str = "intent_label") -> dict[int, int]:
    return {int(k): int(v) for k, v in Counter(df[label_col].astype(int)).items()}


def compute_balance_targets(
    counts: dict[int, int],
    strategy: str = "oversample_median",
    target_cap_multiplier: float = 2.0,
) -> dict[int, int]:
    """
    Compute per-class target counts.

    Strategies:
    - oversample_median: raise minority classes up to median count
    - oversample_max_minority: raise all non-majority classes to 80% of majority
    """
    if not counts:
        return {}

    values = list(counts.values())
    majority = max(values)
    median = int(np.median(values))

    targets: dict[int, int] = {}
    for label, count in counts.items():
        if strategy == "oversample_median":
            targets[label] = max(count, median)
        elif strategy == "oversample_max_minority":
            targets[label] = max(count, int(majority * 0.8))
        else:
            targets[label] = count

        cap = int(count * target_cap_multiplier)
        if targets[label] > cap and label != max(counts, key=counts.get):
            targets[label] = max(count, cap)

    return targets


def balance_minority_classes(
    df: pd.DataFrame,
    label_col: str = "intent_label",
    strategy: str = "oversample_median",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Oversample minority classes in the training split only."""
    rng = np.random.default_rng(seed)
    counts = class_counts(df, label_col=label_col)
    targets = compute_balance_targets(counts, strategy=strategy)

    balanced_parts: list[pd.DataFrame] = [df]
    oversample_report: dict[str, dict] = {
        "strategy": strategy,
        "before": {str(k): v for k, v in counts.items()},
        "targets": {str(k): v for k, v in targets.items()},
        "added_rows_by_class": {},
    }

    for label, target in targets.items():
        class_df = df[df[label_col].astype(int) == label]
        current = len(class_df)
        if current >= target:
            oversample_report["added_rows_by_class"][str(label)] = 0
            continue

        needed = target - current
        sampled = class_df.sample(n=needed, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        sampled = sampled.copy()
        sampled["is_balanced_duplicate"] = True
        balanced_parts.append(sampled)
        oversample_report["added_rows_by_class"][str(label)] = needed

    balanced = pd.concat(balanced_parts, ignore_index=True)
    if "is_balanced_duplicate" not in balanced.columns:
        balanced["is_balanced_duplicate"] = False
    else:
        balanced["is_balanced_duplicate"] = (
            balanced["is_balanced_duplicate"].fillna(False).astype(bool)
        )
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    oversample_report["after"] = {
        str(k): int(v) for k, v in class_counts(balanced, label_col=label_col).items()
    }
    oversample_report["output_rows"] = int(len(balanced))
    return balanced, oversample_report
