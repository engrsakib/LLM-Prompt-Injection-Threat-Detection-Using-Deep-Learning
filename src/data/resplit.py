"""Group-aware stratified re-splitting.

Whole families are assigned to exactly one of train / validation / test. Assignment is a
deterministic greedy procedure: families are processed largest-first and placed in the
split whose per-class deficit (relative to the target fractions) would be reduced most;
when several splits are within a small tolerance, the family's majority *upstream* split is
preferred so that most rows keep their original split.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


@dataclass
class ResplitReport:
    fractions: dict[str, float]
    tolerance: float
    rows: int
    families: int
    rows_kept_in_upstream_split: int
    rows_moved: int
    moved_matrix: dict[str, dict[str, int]]
    class_counts: dict[str, dict[str, int]]
    family_counts: dict[str, int]
    classes_missing: dict[str, list[int]]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def group_stratified_split(
    df: pd.DataFrame,
    family_col: str = "family_id",
    label_col: str = "intent_label",
    upstream_col: str = "upstream_split",
    fractions: dict[str, float] | None = None,
    tolerance: float = 0.02,
) -> tuple[pd.Series, ResplitReport]:
    """Return a Series of split names aligned with ``df.index`` plus a report."""
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    splits = [s for s in SPLIT_NAMES if s in fractions]

    labels = df[label_col].astype(int)
    total_by_label = Counter(labels.tolist())
    targets = {s: {lab: total_by_label[lab] * fractions[s] for lab in total_by_label} for s in splits}
    current = {s: defaultdict(int) for s in splits}

    fam_rows: dict[str, list[int]] = defaultdict(list)
    for pos, fid in enumerate(df[family_col].tolist()):
        fam_rows[fid].append(pos)
    label_list = labels.tolist()
    upstream_list = df[upstream_col].astype(str).tolist() if upstream_col in df.columns else [""] * len(df)

    # Largest families first; ties broken by id for determinism
    order = sorted(fam_rows.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    assignment: dict[str, str] = {}
    for fid, positions in order:
        lab_counts = Counter(label_list[p] for p in positions)
        upstream_major = Counter(upstream_list[p] for p in positions).most_common(1)[0][0]
        scores: dict[str, float] = {}
        for s in splits:
            score = 0.0
            for lab, cnt in lab_counts.items():
                tgt = max(targets[s][lab], 1e-9)
                score += cnt * (tgt - current[s][lab]) / tgt
            scores[s] = score / max(len(positions), 1)
        best_score = max(scores.values())
        candidates = [s for s in splits if scores[s] >= best_score - tolerance]
        chosen = upstream_major if upstream_major in candidates else max(candidates, key=lambda s: scores[s])
        assignment[fid] = chosen
        for lab, cnt in lab_counts.items():
            current[chosen][lab] += cnt

    assigned = pd.Series([assignment[f] for f in df[family_col].tolist()], index=df.index, name="split")

    moved_matrix: dict[str, dict[str, int]] = {s: {t: 0 for t in splits} for s in SPLIT_NAMES}
    kept = 0
    for up, new in zip(upstream_list, assigned.tolist()):
        if up in moved_matrix:
            moved_matrix[up][new] += 1
        if up == new:
            kept += 1
    class_counts = {
        s: {str(k): int(v) for k, v in sorted(Counter(labels[assigned == s].tolist()).items())} for s in splits
    }
    missing = {s: sorted(set(total_by_label) - set(labels[assigned == s].tolist())) for s in splits}
    report = ResplitReport(
        fractions=fractions,
        tolerance=tolerance,
        rows=int(len(df)),
        families=len(fam_rows),
        rows_kept_in_upstream_split=int(kept),
        rows_moved=int(len(df) - kept),
        moved_matrix=moved_matrix,
        class_counts=class_counts,
        family_counts={s: int(sum(1 for f, sp in assignment.items() if sp == s)) for s in splits},
        classes_missing=missing,
    )
    return assigned, report
