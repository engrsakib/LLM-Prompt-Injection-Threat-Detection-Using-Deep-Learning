"""Exact and near-duplicate removal utilities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

from src.data.cleaning import cleaning_fingerprint

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class DedupReport:
    input_count: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    output_count: int = 0
    near_duplicate_threshold: float = 0.0
    near_duplicate_examples: list[dict[str, str]] = field(default_factory=list)


def _text_shingles(text: str, n: int = 3) -> set[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if len(tokens) >= n:
        return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
    return {" ".join(tokens)} if tokens else set()


def _minhash(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in _text_shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def exact_deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicates using normalized text fingerprints."""
    seen: set[str] = set()
    kept: list[dict] = []
    removed = 0

    for record in records:
        text = record.get("text_clean", record.get("text", ""))
        fp = hashlib.sha256(cleaning_fingerprint(text).encode("utf-8")).hexdigest()
        if fp in seen:
            removed += 1
            continue
        seen.add(fp)
        record["text_fingerprint"] = fp
        kept.append(record)

    return kept, removed


def near_deduplicate(
    records: list[dict],
    threshold: float = 0.85,
    num_perm: int = 128,
    max_examples: int = 10,
) -> tuple[list[dict], DedupReport]:
    """Remove near-duplicates within a split using MinHash LSH."""
    report = DedupReport(
        input_count=len(records),
        near_duplicate_threshold=threshold,
    )

    if len(records) <= 1:
        report.output_count = len(records)
        return records, report

    try:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    except ValueError:
        # Threshold too extreme for MinHash band configuration; skip near-dedup.
        report.output_count = len(records)
        return records, report

    kept: list[dict] = []
    signatures: list[MinHash] = []

    for idx, record in enumerate(records):
        text = record.get("text_clean", record.get("text", ""))
        mh = _minhash(text, num_perm=num_perm)
        duplicates = lsh.query(mh)

        if duplicates:
            report.near_duplicates_removed += 1
            if len(report.near_duplicate_examples) < max_examples:
                report.near_duplicate_examples.append(
                    {
                        "duplicate_text_preview": text[:120],
                        "matched_index": str(duplicates[0]),
                    }
                )
            continue

        key = f"row-{idx}"
        lsh.insert(key, mh)
        signatures.append(mh)
        kept.append(record)

    report.output_count = len(kept)
    return kept, report


def deduplicate_records(
    records: list[dict],
    near_duplicate_threshold: float = 0.85,
) -> tuple[list[dict], DedupReport]:
    """Apply exact then near duplicate removal."""
    exact_kept, exact_removed = exact_deduplicate(records)
    near_kept, near_report = near_deduplicate(
        exact_kept,
        threshold=near_duplicate_threshold,
    )

    near_report.exact_duplicates_removed = exact_removed
    near_report.input_count = len(records)
    near_report.output_count = len(near_kept)
    return near_kept, near_report
