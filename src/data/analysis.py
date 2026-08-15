"""Analysis reports for severity buckets and evaluation subsets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SEVERITY_BUCKETS = {
    "low_1_2": (1, 2),
    "moderate_3_4": (3, 4),
    "high_5_6": (5, 6),
    "critical_7_10": (7, 10),
}


def assign_severity_bucket(severity: int) -> str:
    for bucket, (low, high) in SEVERITY_BUCKETS.items():
        if low <= severity <= high:
            return bucket
    return "unknown"


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add severity bucket and normalized ambiguity flag."""
    out = df.copy()
    out["severity_bucket"] = out["severity"].astype(int).map(assign_severity_bucket)
    if "ambiguity" in out.columns:
        out["ambiguity"] = out["ambiguity"].astype(bool)
    else:
        out["ambiguity"] = False
    return out


def build_severity_bucket_report(split_frames: dict[str, pd.DataFrame]) -> dict:
    report: dict[str, dict] = {}
    for split, frame in split_frames.items():
        enriched = add_derived_columns(frame)
        bucket_counts = enriched["severity_bucket"].value_counts().to_dict()
        by_intent = (
            enriched.groupby(["intent_label", "severity_bucket"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )
        report[split] = {
            "severity_bucket_counts": {str(k): int(v) for k, v in bucket_counts.items()},
            "intent_by_severity_bucket": by_intent,
        }
    return report


def export_ambiguity_subsets(
    split_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict:
    """Export hard-case subsets where ambiguity=true."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    for split, frame in split_frames.items():
        enriched = add_derived_columns(frame)
        subset = enriched[enriched["ambiguity"] == True]  # noqa: E712
        summary[split] = int(len(subset))
        if len(subset) > 0:
            subset.to_parquet(output_dir / f"ambiguity_{split}.parquet", index=False)

    return summary


def write_phase2_reports(
    split_frames: dict[str, pd.DataFrame],
    report_dir: Path,
    subsets_dir: Path,
) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)

    severity_report = build_severity_bucket_report(split_frames)
    severity_path = report_dir / "severity_buckets.json"
    severity_path.write_text(json.dumps(severity_report, indent=2), encoding="utf-8")

    md_lines = ["# Severity Bucket Analysis", ""]
    for split, data in severity_report.items():
        md_lines.append(f"## {split}")
        md_lines.append("")
        for bucket, count in sorted(data["severity_bucket_counts"].items()):
            md_lines.append(f"- `{bucket}`: {count}")
        md_lines.append("")
    (report_dir / "severity_buckets.md").write_text("\n".join(md_lines), encoding="utf-8")

    ambiguity_summary = export_ambiguity_subsets(split_frames, subsets_dir)
    ambiguity_path = report_dir / "ambiguity_subsets.json"
    ambiguity_path.write_text(json.dumps(ambiguity_summary, indent=2), encoding="utf-8")

    return {
        "severity_buckets": severity_report,
        "ambiguity_subsets": ambiguity_summary,
    }
