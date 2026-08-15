"""Reporting utilities for processed dataset splits."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.constants import INTENT_LABELS, SPLITS


def _distribution(series: pd.Series) -> dict[str, int]:
    counts = series.fillna("unknown").astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def build_class_distribution(df: pd.DataFrame) -> dict:
    """Build per-split class distribution summary."""
    summary: dict = {
        "row_count": int(len(df)),
        "intent_label": _distribution(df.get("intent_label", df.get("label", pd.Series()))),
        "intent": _distribution(df.get("intent", pd.Series())),
        "binary_label": _distribution(df.get("binary_label", pd.Series())),
        "technique": _distribution(df.get("technique", pd.Series())),
        "severity": _distribution(df.get("severity", pd.Series())),
        "ambiguity": _distribution(df.get("ambiguity", pd.Series(dtype=object))),
    }

    if "intent_label" in df.columns:
        summary["intent_label_named"] = {
            INTENT_LABELS.get(int(k), str(k)): int(v)
            for k, v in Counter(df["intent_label"].dropna().astype(int)).items()
        }

    return summary


def write_distribution_reports(
    split_frames: dict[str, pd.DataFrame],
    report_dir: Path,
) -> dict:
    """Write JSON and Markdown distribution reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict] = {}

    for split_name, frame in split_frames.items():
        payload[split_name] = build_class_distribution(frame)

    json_path = report_dir / "class_distribution.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = ["# Class Distribution Report", ""]
    for split_name in SPLITS:
        if split_name not in payload:
            continue
        split_data = payload[split_name]
        md_lines.append(f"## {split_name}")
        md_lines.append("")
        md_lines.append(f"- Rows: **{split_data['row_count']}**")
        md_lines.append("")
        md_lines.append("### Intent labels")
        md_lines.append("")
        for label, count in sorted(split_data.get("intent_label_named", {}).items()):
            md_lines.append(f"- `{label}`: {count}")
        md_lines.append("")
        md_lines.append("### Binary label")
        md_lines.append("")
        for label, count in sorted(split_data["binary_label"].items()):
            md_lines.append(f"- `{label}`: {count}")
        md_lines.append("")
        md_lines.append("### Severity")
        md_lines.append("")
        for label, count in sorted(split_data["severity"].items(), key=lambda x: int(x[0])):
            md_lines.append(f"- `{label}`: {count}")
        md_lines.append("")

    md_path = report_dir / "class_distribution.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return payload
