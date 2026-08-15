"""Cross-split leakage detection between train, validation, and test."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.cleaning import cleaning_fingerprint
from src.data.dedup import _minhash


@dataclass
class LeakageFinding:
    leak_type: str
    source_split: str
    target_split: str
    count: int
    examples: list[dict] = field(default_factory=list)


def _fingerprints(series: pd.Series) -> set[str]:
    return {cleaning_fingerprint(str(v)) for v in series if str(v).strip()}


def exact_leakage_check(split_frames: dict[str, pd.DataFrame]) -> list[LeakageFinding]:
    """Detect exact normalized-text overlap across splits."""
    findings: list[LeakageFinding] = []
    split_names = list(split_frames.keys())
    fps = {
        split: _fingerprints(
            frames.get("text_obfuscation_aware", frames.get("text_clean", frames["text"]))
        )
        for split, frames in split_frames.items()
        if split in ("train", "validation", "test")
    }

    for i, src in enumerate(split_names):
        for tgt in split_names[i + 1 :]:
            if src not in fps or tgt not in fps:
                continue
            overlap = fps[src] & fps[tgt]
            if overlap:
                examples = []
                for fp in list(overlap)[:5]:
                    examples.append({"fingerprint": fp, "source": src, "target": tgt})
                findings.append(
                    LeakageFinding(
                        leak_type="exact_text_overlap",
                        source_split=src,
                        target_split=tgt,
                        count=len(overlap),
                        examples=examples,
                    )
                )
    return findings


def near_leakage_check(
    split_frames: dict[str, pd.DataFrame],
    threshold: float = 0.85,
    sample_limit: int = 2000,
) -> list[LeakageFinding]:
    """
    Detect near-duplicate leakage using MinHash between split pairs.

    Uses a capped sample per split for scalability.
    """
    findings: list[LeakageFinding] = []
    splits = [s for s in ("train", "validation", "test") if s in split_frames]

    signatures: dict[str, list[tuple[str, object]]] = {}
    for split in splits:
        frame = split_frames[split]
        text_col = (
            "text_obfuscation_aware"
            if "text_obfuscation_aware" in frame.columns
            else "text_clean"
            if "text_clean" in frame.columns
            else "text"
        )
        sample = frame[text_col].dropna().astype(str).head(sample_limit)
        signatures[split] = [(text, _minhash(text)) for text in sample]

    for i, src in enumerate(splits):
        for tgt in splits[i + 1 :]:
            near_matches = 0
            examples: list[dict] = []
            for src_text, src_mh in signatures[src]:
                for tgt_text, tgt_mh in signatures[tgt]:
                    if src_mh.jaccard(tgt_mh) >= threshold:
                        near_matches += 1
                        if len(examples) < 5:
                            examples.append(
                                {
                                    "source_preview": src_text[:80],
                                    "target_preview": tgt_text[:80],
                                    "jaccard": round(src_mh.jaccard(tgt_mh), 4),
                                }
                            )
                        break
            if near_matches:
                findings.append(
                    LeakageFinding(
                        leak_type="near_duplicate_overlap",
                        source_split=src,
                        target_split=tgt,
                        count=near_matches,
                        examples=examples,
                    )
                )
    return findings


def run_leakage_audit(
    split_frames: dict[str, pd.DataFrame],
    near_threshold: float = 0.85,
) -> dict:
    """Run full leakage audit and return structured report."""
    exact = exact_leakage_check(split_frames)
    near = near_leakage_check(split_frames, threshold=near_threshold)

    def _serialize(items: list[LeakageFinding]) -> list[dict]:
        return [
            {
                "leak_type": item.leak_type,
                "source_split": item.source_split,
                "target_split": item.target_split,
                "count": item.count,
                "examples": item.examples,
            }
            for item in items
        ]

    passed = len(exact) == 0 and len(near) == 0
    return {
        "passed": passed,
        "exact_leakage": _serialize(exact),
        "near_leakage": _serialize(near),
        "summary": {
            "exact_leak_pairs": len(exact),
            "near_leak_pairs": len(near),
            "total_exact_overlaps": sum(item.count for item in exact),
            "total_near_overlaps": sum(item.count for item in near),
        },
    }


def write_leakage_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
