"""Cross-split leakage detection between train, validation, and test.

Three checks are performed:

* ``raw_text_overlap``      - identical raw text (``text_original`` when available),
* ``exact_text_overlap``    - identical normalised fingerprint (NFKC, lowercase, whitespace),
* ``near_duplicate_overlap`` - MinHash LSH Jaccard >= threshold on word 3-gram shingles.

``run_leakage_audit`` keeps the historical sampled behaviour by default (``full=False``)
so existing reports stay comparable; Phase 3.5 calls it with ``full=True`` which indexes
every row with LSH. Rows without word tokens are excluded from the near-duplicate check
(see ``src.data.families``).

Passing these heuristics does **not** prove zero statistical leakage: paraphrases,
translations or shared templates with low lexical overlap are not detected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from datasketch import MinHashLSH

from src.data.cleaning import cleaning_fingerprint
from src.data.families import NUM_PERM, minhash_for

EVAL_SPLITS = ("train", "validation", "test")


class LeakageError(RuntimeError):
    """Raised when a configured leakage gate fails."""


@dataclass
class LeakageFinding:
    leak_type: str
    source_split: str
    target_split: str
    count: int
    examples: list[dict] = field(default_factory=list)


def _text_column(frame: pd.DataFrame) -> str:
    for col in ("text_obfuscation_aware", "text_clean", "text"):
        if col in frame.columns:
            return col
    raise KeyError("No text column found")


def _fingerprints(series: pd.Series) -> set[str]:
    return {cleaning_fingerprint(str(v)) for v in series if str(v).strip()}


def _pairs(split_names: list[str]):
    for i, src in enumerate(split_names):
        for tgt in split_names[i + 1 :]:
            yield src, tgt


def raw_text_leakage_check(split_frames: dict[str, pd.DataFrame]) -> list[LeakageFinding]:
    """Identical raw strings across splits (no normalisation)."""
    findings: list[LeakageFinding] = []
    names = [s for s in split_frames if s in EVAL_SPLITS]
    sets = {}
    for s in names:
        frame = split_frames[s]
        col = "text_original" if "text_original" in frame.columns else _text_column(frame)
        sets[s] = {str(v) for v in frame[col] if str(v).strip()}
    for src, tgt in _pairs(names):
        overlap = sets[src] & sets[tgt]
        if overlap:
            findings.append(
                LeakageFinding(
                    "raw_text_overlap", src, tgt, len(overlap),
                    [{"text": t[:120], "source": src, "target": tgt} for t in sorted(overlap)[:5]],
                )
            )
    return findings


def exact_leakage_check(split_frames: dict[str, pd.DataFrame]) -> list[LeakageFinding]:
    """Detect exact normalized-text overlap across splits."""
    findings: list[LeakageFinding] = []
    split_names = [s for s in split_frames if s in EVAL_SPLITS]
    fps = {split: _fingerprints(split_frames[split][_text_column(split_frames[split])]) for split in split_names}

    for src, tgt in _pairs(split_names):
        overlap = fps[src] & fps[tgt]
        if overlap:
            examples = [{"fingerprint": fp[:160], "source": src, "target": tgt} for fp in sorted(overlap)[:5]]
            findings.append(LeakageFinding("exact_text_overlap", src, tgt, len(overlap), examples))
    return findings


def near_leakage_check(
    split_frames: dict[str, pd.DataFrame],
    threshold: float = 0.85,
    sample_limit: int | None = 2000,
    num_perm: int = NUM_PERM,
) -> list[LeakageFinding]:
    """Near-duplicate leakage with MinHash LSH.

    ``sample_limit=None`` indexes every row (used by Phase 3.5); the historical default
    caps each split at 2000 rows for speed and is kept for backward compatibility.
    """
    findings: list[LeakageFinding] = []
    splits = [s for s in EVAL_SPLITS if s in split_frames]
    try:
        _ = MinHashLSH(threshold=threshold, num_perm=num_perm)
    except ValueError:
        return findings

    signatures: dict[str, list[tuple[str, object]]] = {}
    for split in splits:
        frame = split_frames[split]
        series = frame[_text_column(frame)].dropna().astype(str)
        if sample_limit is not None:
            series = series.head(sample_limit)
        sigs = []
        for text in series:
            mh = minhash_for(text, num_perm=num_perm)
            if mh is not None:
                sigs.append((text, mh))
        signatures[split] = sigs

    for src, tgt in _pairs(splits):
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for idx, (_, mh) in enumerate(signatures[tgt]):
            lsh.insert(str(idx), mh)
        near_matches = 0
        examples: list[dict] = []
        for src_text, src_mh in signatures[src]:
            hits = lsh.query(src_mh)
            if not hits:
                continue
            best_idx = int(hits[0])
            tgt_text, tgt_mh = signatures[tgt][best_idx]
            jac = src_mh.jaccard(tgt_mh)
            if jac < threshold:
                # LSH is approximate; confirm with the estimated Jaccard
                confirmed = [(int(h), src_mh.jaccard(signatures[tgt][int(h)][1])) for h in hits]
                confirmed = [c for c in confirmed if c[1] >= threshold]
                if not confirmed:
                    continue
                best_idx, jac = max(confirmed, key=lambda c: c[1])
                tgt_text = signatures[tgt][best_idx][0]
            near_matches += 1
            if len(examples) < 5:
                examples.append(
                    {"source_preview": src_text[:80], "target_preview": tgt_text[:80], "jaccard": round(float(jac), 4)}
                )
        if near_matches:
            findings.append(LeakageFinding("near_duplicate_overlap", src, tgt, near_matches, examples))
    return findings


def family_leakage_check(split_frames: dict[str, pd.DataFrame], family_col: str = "family_id") -> list[LeakageFinding]:
    """Families (duplicate/template groups) present in more than one split."""
    findings: list[LeakageFinding] = []
    names = [s for s in split_frames if s in EVAL_SPLITS and family_col in split_frames[s].columns]
    fams = {s: set(split_frames[s][family_col].astype(str)) for s in names}
    for src, tgt in _pairs(names):
        overlap = fams[src] & fams[tgt]
        if overlap:
            findings.append(
                LeakageFinding("family_overlap", src, tgt, len(overlap), [{"family_id": f} for f in sorted(overlap)[:5]])
            )
    return findings


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


def run_leakage_audit(
    split_frames: dict[str, pd.DataFrame],
    near_threshold: float = 0.85,
    full: bool = False,
    include_raw: bool = True,
) -> dict:
    """Run the leakage audit and return a structured report.

    ``full=True`` disables the per-split sampling cap of the near-duplicate check and adds
    the family-overlap check when a ``family_id`` column is present.
    """
    raw = raw_text_leakage_check(split_frames) if include_raw else []
    exact = exact_leakage_check(split_frames)
    near = near_leakage_check(split_frames, threshold=near_threshold, sample_limit=None if full else 2000)
    family = family_leakage_check(split_frames) if full else []

    passed = not (raw or exact or near or family)
    return {
        "passed": passed,
        "mode": "full" if full else "sampled_2000",
        "near_threshold": near_threshold,
        "raw_leakage": _serialize(raw),
        "exact_leakage": _serialize(exact),
        "near_leakage": _serialize(near),
        "family_leakage": _serialize(family),
        "summary": {
            "raw_leak_pairs": len(raw),
            "exact_leak_pairs": len(exact),
            "near_leak_pairs": len(near),
            "family_leak_pairs": len(family),
            "total_raw_overlaps": sum(i.count for i in raw),
            "total_exact_overlaps": sum(i.count for i in exact),
            "total_near_overlaps": sum(i.count for i in near),
            "total_family_overlaps": sum(i.count for i in family),
        },
        "caveat": (
            "Heuristic checks (exact / normalised / MinHash near-duplicate / family). Passing does not "
            "prove zero statistical leakage (paraphrases, translations, shared low-overlap templates)."
        ),
    }


def assert_no_leakage(report: dict, context: str = "") -> None:
    if not report.get("passed", False):
        summary = report.get("summary", {})
        raise LeakageError(f"Leakage gate failed {context}: {json.dumps(summary)}")


def write_leakage_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
