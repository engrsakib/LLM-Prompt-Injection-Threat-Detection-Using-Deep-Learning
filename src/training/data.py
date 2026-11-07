"""Split loading for training / evaluation.

Only versioned Phase 3.5 splits (``data/processed/splits/<version>/``) are accepted by default;
their ``manifest.json`` must be frozen and must have passed the leakage audit. The legacy
``data/processed/*.parquet`` files can be used explicitly with ``allow_unfrozen=True`` (for
local debugging only - production evaluation refuses them).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.constants import NUM_CLASSES, SPLITS

TEXT_COL = "text"
LABEL_COL = "intent_label"
OPTIONAL_COLS = ("row_id", "text_hash", "source", "technique", "ambiguity_bucket", "severity", "family_id", "parent_row_id")


@dataclass
class SplitBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    splits_dir: Path
    train_file: str
    manifest: dict

    def frame(self, split: str) -> pd.DataFrame:
        return getattr(self, split)


def _ensure_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if TEXT_COL not in df.columns:
        for alt in ("text_clean", "text_original"):
            if alt in df.columns:
                df[TEXT_COL] = df[alt]
                break
        else:
            raise ValueError(f"{name}: no text column")
    if LABEL_COL not in df.columns:
        if "label" in df.columns:
            df[LABEL_COL] = df["label"]
        else:
            raise ValueError(f"{name}: no intent_label column")
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    bad = df[~df[LABEL_COL].between(0, NUM_CLASSES - 1)]
    if len(bad):
        raise ValueError(f"{name}: {len(bad)} rows with labels outside [0,{NUM_CLASSES - 1}]")
    if "row_id" not in df.columns:
        df["row_id"] = [f"{name}:{i}" for i in range(len(df))]
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    return df


def load_splits(
    splits_dir: str | Path,
    train_file: str = "train.parquet",
    allow_unfrozen: bool = False,
) -> SplitBundle:
    splits_dir = Path(splits_dir)
    manifest: dict = {}
    if not allow_unfrozen:
        from src.data.phase35 import load_frozen_manifest

        manifest = load_frozen_manifest(splits_dir)
    frames = {}
    for split in SPLITS:
        path = splits_dir / (train_file if split == "train" else f"{split}.parquet")
        if not path.exists():
            raise FileNotFoundError(path)
        frames[split] = _ensure_columns(pd.read_parquet(path), split)
    return SplitBundle(frames["train"], frames["validation"], frames["test"], splits_dir, train_file, manifest)


def stratified_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic label-stratified subset with at least one row per present class (smoke runs)."""
    if n >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    per_class = max(1, n // NUM_CLASSES)
    for _, group in df.groupby(LABEL_COL):
        take = min(len(group), per_class)
        idx = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[np.sort(idx)])
    out = pd.concat(parts)
    if len(out) < n:
        remaining = df.drop(out.index)
        extra = rng.choice(len(remaining), size=min(n - len(out), len(remaining)), replace=False)
        out = pd.concat([out, remaining.iloc[np.sort(extra)]])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
